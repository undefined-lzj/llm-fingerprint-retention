#!/usr/bin/env python3
"""P1：使用未经指纹训练的 Qwen3-0.6B 执行演示指纹负例检查。"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fingerprint.manifest import load_fingerprint_manifest  # noqa: E402
from fingerprint.parser import parse_response  # noqa: E402
from p0b_common import now, sanitize_text, write_json, write_text  # noqa: E402


SCRIPT_VERSION = "p1-base-1.0.0"
EXPECTED_MODEL_ID = "Qwen/Qwen3-0.6B"
EXPECTED_REPEATS = 2
MAX_NEW_TOKENS = 16
MAX_TARGET_TOKENS = 5
DATA_ROOT = Path("/root/autodl-tmp")
SCORE_FIELDS = [
    "fingerprint_id",
    "repeat_id",
    "target_response",
    "target_class",
    "normalized_output",
    "parse_status",
    "is_exact_match",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--fingerprint-config", required=True, type=Path)
    parser.add_argument("--repeats", default=EXPECTED_REPEATS, type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是 JSON 对象")
    return data


def create_run_directory() -> tuple[str, Path, str]:
    checked_at = now()
    runs_root = PROJECT_ROOT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = f"p1_base_{checked_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir, checked_at.isoformat(timespec="seconds")
    raise OSError("连续生成的 P1 运行目录名称发生冲突")


def write_scores(path: Path, records: list[dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SCORE_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({field: record.get(field) for field in SCORE_FIELDS})
    write_text(path, output.getvalue())


def empty_metrics(repeats: int, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    repeat_keys = [str(repeat_id) for repeat_id in range(1, repeats + 1)]
    zeros = {repeat_id: 0 for repeat_id in repeat_keys}
    rates = {repeat_id: 0.0 for repeat_id in repeat_keys}
    return {
        "fingerprint_set_id": None,
        "unique_fingerprint_count": 0,
        "repeat_count": repeats,
        "completed_query_count_by_repeat": dict(zeros),
        "exact_match_count_by_repeat": dict(zeros),
        "exact_match_rate_by_repeat": dict(rates),
        "wrong_valid_code_count_by_repeat": dict(zeros),
        "invalid_output_count_by_repeat": dict(zeros),
        "valid_code_rate_by_repeat": dict(rates),
        "outputs_identical_across_repeats": False,
        "thinking_tag_count": 0,
        "all_queries_completed": False,
        "exact_match_fingerprint_ids_by_repeat": {key: [] for key in repeat_keys},
        "errors": errors or [],
        "p1_passed": False,
    }


def initialize_result_files(
    run_dir: Path,
    run_id: str,
    checked_at: str,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reason = "P1 尚未开始"
    resolved = {
        "status": "not_run",
        "script_version": SCRIPT_VERSION,
        "run_id": run_id,
        "checked_at": checked_at,
        "reason": reason,
    }
    tokenization = {"status": "not_run", "reason": reason, "targets": [], "errors": []}
    metrics = empty_metrics(repeats)
    write_json(run_dir / "fingerprint_manifest.json", {"status": "not_run", "reason": reason})
    write_json(run_dir / "target_tokenization.json", tokenization)
    write_text(run_dir / "raw_generations.jsonl", "")
    write_scores(run_dir / "scores.csv", [])
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "resolved_config.json", resolved)
    write_text(run_dir / "summary.md", "# P1 原始模型负例检查\n\n- 状态：尚未开始\n")
    return resolved, tokenization, metrics


def find_latest_successful_p0b() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for run_dir in (PROJECT_ROOT / "runs").glob("p0b_*"):
        resolved_path = run_dir / "resolved_config.json"
        environment_path = run_dir / "environment.json"
        benchmark_path = run_dir / "benchmark.json"
        if not all(path.is_file() for path in (resolved_path, environment_path, benchmark_path)):
            continue
        try:
            resolved = load_json(resolved_path)
            environment = load_json(environment_path)
            benchmark = load_json(benchmark_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            resolved.get("status") == "completed"
            and environment.get("overall_status") == "pass"
            and benchmark.get("status") == "pass"
        ):
            candidates.append(
                {
                    "run_dir": run_dir.resolve(),
                    "resolved_path": resolved_path.resolve(),
                    "environment_path": environment_path.resolve(),
                    "benchmark_path": benchmark_path.resolve(),
                    "resolved": resolved,
                    "environment": environment,
                    "benchmark": benchmark,
                    "checked_at": str(environment.get("checked_at") or run_dir.name),
                }
            )
    if not candidates:
        raise RuntimeError("找不到通过验收的 P0-B 运行，无法确定准确模型 revision")
    return max(candidates, key=lambda item: (item["checked_at"], item["run_dir"].name))


def validate_model_and_p0b(
    model_config_path: Path,
    p0b: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    model_config = load_json(model_config_path)
    p0b_resolved = p0b["resolved"]
    p0b_environment = p0b["environment"]

    required_model_values = {
        "model_id": EXPECTED_MODEL_ID,
        "trust_remote_code": False,
        "device": "cuda:0",
        "enable_thinking": False,
        "do_sample": False,
        "batch_size": 1,
    }
    for field, expected in required_model_values.items():
        if model_config.get(field) != expected:
            raise ValueError(f"模型配置的 {field} 必须为 {expected!r}")

    revision = p0b_resolved.get("resolved_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("P0-B resolved_config.json 中没有准确的 40 位模型 revision")
    if p0b_resolved.get("requested_revision") != revision:
        raise RuntimeError("P0-B 请求 revision 与实际 revision 不一致")
    if model_config.get("revision") != revision:
        raise RuntimeError("模型配置 revision 与最近一次成功 P0-B 的实际 revision 不一致")
    if p0b_resolved.get("model_id") != EXPECTED_MODEL_ID:
        raise RuntimeError("P0-B 使用的模型 ID 不是 Qwen/Qwen3-0.6B")

    for field in ("trust_remote_code", "device", "enable_thinking", "do_sample", "batch_size"):
        if p0b_resolved.get(field) != model_config.get(field):
            raise RuntimeError(f"模型配置的 {field} 与 P0-B 实际配置不一致")

    dtype_name = p0b_resolved.get("torch_dtype_resolved")
    if dtype_name not in {"bfloat16", "float16"}:
        raise RuntimeError("P0-B 实际 dtype 既不是 bfloat16 也不是 float16")

    snapshot_value = p0b_resolved.get("snapshot_path")
    cache_value = p0b_environment.get("model_cache_directory")
    if not isinstance(snapshot_value, str) or not isinstance(cache_value, str):
        raise RuntimeError("P0-B 报告缺少模型快照或 Hugging Face 缓存路径")
    snapshot_path = Path(snapshot_value).expanduser().resolve()
    cache_path = Path(cache_value).expanduser().resolve()
    if not snapshot_path.is_relative_to(DATA_ROOT) or not cache_path.is_relative_to(DATA_ROOT):
        raise RuntimeError("P0-B 模型快照和 Hugging Face 缓存必须位于 /root/autodl-tmp/")
    if snapshot_path.name != revision:
        raise RuntimeError("P0-B 模型快照目录名与实际 revision 不一致")
    if not snapshot_path.is_dir():
        raise RuntimeError(f"P0-B 模型快照不存在：{snapshot_path}")
    return model_config, snapshot_path, cache_path


def tokenize_targets(
    tokenizer: Any,
    manifest: dict[str, Any],
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    failures: list[str] = []
    for target in manifest["allowed_responses"]:
        token_ids = [int(value) for value in tokenizer.encode(target, add_special_tokens=False)]
        token_texts = [str(value) for value in tokenizer.convert_ids_to_tokens(token_ids)]
        unknown_id = tokenizer.unk_token_id
        contains_unknown = unknown_id is not None and int(unknown_id) in token_ids
        exceeds_limit = len(token_ids) > MAX_TARGET_TOKENS
        if not token_ids:
            failures.append(f"{target} 没有产生任何 Token")
        if contains_unknown:
            failures.append(f"{target} 包含未知 Token")
        if exceeds_limit:
            failures.append(f"{target} 超过 {MAX_TARGET_TOKENS} 个 Token")
        targets.append(
            {
                "raw_string": target,
                "token_ids": token_ids,
                "token_texts": token_texts,
                "token_count": len(token_ids),
                "contains_unknown_token": contains_unknown,
                "exceeds_five_tokens": exceeds_limit,
            }
        )
    return {
        "status": "pass" if not failures else "fail",
        "model_id": model_id,
        "revision": revision,
        "maximum_allowed_tokens": MAX_TARGET_TOKENS,
        "targets": targets,
        "errors": failures,
    }


def generation_input(tokenizer: Any, prompt: str, device: str) -> tuple[Any, int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(device)
    return inputs, int(inputs["attention_mask"].sum().item())


def generate_record(
    torch: Any,
    model: Any,
    tokenizer: Any,
    fingerprint_set_id: str,
    fingerprint: dict[str, Any],
    allowed_responses: list[str],
    repeat_id: int,
    device: str,
) -> dict[str, Any]:
    inputs, input_tokens = generation_input(tokenizer, fingerprint["prompt"], device)
    input_width = int(inputs["input_ids"].shape[-1])
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    torch.cuda.synchronize(device)
    latency = time.perf_counter() - started
    new_tokens = generated[0, input_width:]
    output_tokens = int(new_tokens.numel())
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    output_with_special_tokens = tokenizer.decode(new_tokens, skip_special_tokens=False)
    parsed = parse_response(
        raw_output,
        fingerprint["target_response"],
        allowed_responses,
    )
    contains_thinking_tag = any(
        tag in value
        for value in (raw_output, output_with_special_tokens)
        for tag in ("<think>", "</think>")
    )
    del inputs, generated, new_tokens
    return {
        "fingerprint_set_id": fingerprint_set_id,
        "fingerprint_id": fingerprint["fingerprint_id"],
        "repeat_id": repeat_id,
        "prompt": fingerprint["prompt"],
        "target_response": fingerprint["target_response"],
        "target_class": fingerprint["target_class"],
        "raw_output": raw_output,
        "decoded_with_special_tokens": output_with_special_tokens,
        "normalized_output": parsed.normalized_output,
        "parse_status": parsed.status,
        "is_exact_match": parsed.is_exact_match,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": round(latency, 6),
        "contains_thinking_tag": contains_thinking_tag,
    }


def compute_metrics(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    repeats: int,
    tokenization: dict[str, Any],
    errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    fingerprint_count = len(manifest["fingerprints"])
    completed: dict[str, int] = {}
    exact: dict[str, int] = {}
    wrong: dict[str, int] = {}
    invalid: dict[str, int] = {}
    exact_rates: dict[str, float] = {}
    valid_rates: dict[str, float] = {}
    hit_ids: dict[str, list[str]] = {}
    for repeat_id in range(1, repeats + 1):
        key = str(repeat_id)
        subset = [record for record in records if record["repeat_id"] == repeat_id]
        completed[key] = len(subset)
        exact[key] = sum(record["parse_status"] == "exact_match" for record in subset)
        wrong[key] = sum(record["parse_status"] == "wrong_valid_code" for record in subset)
        invalid[key] = sum(record["parse_status"] == "invalid_output" for record in subset)
        exact_rates[key] = round(exact[key] / fingerprint_count, 6)
        valid_rates[key] = round((exact[key] + wrong[key]) / fingerprint_count, 6)
        hit_ids[key] = [
            record["fingerprint_id"]
            for record in subset
            if record["parse_status"] == "exact_match"
        ]

    all_completed = all(value == fingerprint_count for value in completed.values())
    outputs_by_fingerprint: dict[str, dict[int, str]] = defaultdict(dict)
    for record in records:
        outputs_by_fingerprint[record["fingerprint_id"]][record["repeat_id"]] = record[
            "raw_output"
        ]
    identical = all_completed and all(
        len(outputs_by_fingerprint[item["fingerprint_id"]]) == repeats
        and len(set(outputs_by_fingerprint[item["fingerprint_id"]].values())) == 1
        for item in manifest["fingerprints"]
    )
    thinking_count = sum(bool(record["contains_thinking_tag"]) for record in records)
    tokenization_passed = tokenization.get("status") == "pass"
    error_list = errors or []
    p1_passed = (
        all_completed
        and all(value == 0 for value in exact.values())
        and tokenization_passed
        and identical
        and thinking_count == 0
        and not error_list
    )
    return {
        "fingerprint_set_id": manifest["fingerprint_set_id"],
        "unique_fingerprint_count": fingerprint_count,
        "repeat_count": repeats,
        "completed_query_count_by_repeat": completed,
        "exact_match_count_by_repeat": exact,
        "exact_match_rate_by_repeat": exact_rates,
        "wrong_valid_code_count_by_repeat": wrong,
        "invalid_output_count_by_repeat": invalid,
        "valid_code_rate_by_repeat": valid_rates,
        "outputs_identical_across_repeats": identical,
        "thinking_tag_count": thinking_count,
        "all_queries_completed": all_completed,
        "exact_match_fingerprint_ids_by_repeat": hit_ids,
        "errors": error_list,
        "p1_passed": p1_passed,
    }


def render_summary(
    resolved: dict[str, Any],
    manifest: dict[str, Any] | None,
    tokenization: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    passed = bool(metrics.get("p1_passed"))
    lines = [
        "# P1 原始模型负例检查摘要",
        "",
        f"- 检查时间：{resolved.get('checked_at')}",
        f"- 运行编号：{resolved.get('run_id')}",
        f"- 总体状态：{'通过' if passed else '失败'}",
        f"- 模型：`{resolved.get('model_id')}`",
        f"- 模型 revision：`{resolved.get('revision')}`",
        f"- 引用的 P0-B 配置：`{resolved.get('p0b_resolved_config_path')}`",
        f"- 指纹集合：`{manifest.get('fingerprint_set_id') if manifest else None}`",
        "",
        "## 目标代号 Token 检查",
        "",
    ]
    targets = tokenization.get("targets", [])
    if targets:
        for target in targets:
            lines.append(
                f"- `{target['raw_string']}`：{target['token_count']} 个 Token，"
                f"未知 Token：`{target['contains_unknown_token']}`，"
                f"超过 5 个：`{target['exceeds_five_tokens']}`"
            )
    else:
        lines.append(f"- 状态：`{tokenization.get('status')}`")

    lines.extend(["", "## 两次负例检查", ""])
    completed = metrics.get("completed_query_count_by_repeat", {})
    exact = metrics.get("exact_match_count_by_repeat", {})
    wrong = metrics.get("wrong_valid_code_count_by_repeat", {})
    invalid = metrics.get("invalid_output_count_by_repeat", {})
    for repeat_id in range(1, int(metrics.get("repeat_count", EXPECTED_REPEATS)) + 1):
        key = str(repeat_id)
        lines.append(
            f"- 重复 {repeat_id}：完成 `{completed.get(key, 0)}/8`，"
            f"准确命中 `{exact.get(key, 0)}/8`，"
            f"错误合法代号 `{wrong.get(key, 0)}`，"
            f"无效输出 `{invalid.get(key, 0)}`"
        )
    lines.extend(
        [
            f"- 两次原始输出逐条完全一致：`{metrics.get('outputs_identical_across_repeats')}`",
            f"- 思考标签数量：`{metrics.get('thinking_tag_count')}`",
            f"- 满足 P1 验收条件：`{passed}`",
        ]
    )

    hit_ids = metrics.get("exact_match_fingerprint_ids_by_repeat", {})
    hits = [(repeat_id, values) for repeat_id, values in hit_ids.items() if values]
    if hits:
        lines.extend(["", "### 原始模型意外命中的指纹", ""])
        for repeat_id, values in hits:
            lines.append(f"- 重复 {repeat_id}：`{', '.join(values)}`")

    errors = metrics.get("errors", [])
    token_errors = tokenization.get("errors", [])
    if errors or token_errors:
        lines.extend(["", "## 错误", ""])
        lines.extend(f"- {message}" for message in token_errors)
        lines.extend(f"- {error['type']}：{error['message']}" for error in errors)
    lines.extend(
        [
            "",
            "> 本阶段只检查未经指纹训练的原始模型，没有训练、加载适配器或修改模型权重。",
            "",
        ]
    )
    return "\n".join(lines)


def persist_results(
    run_dir: Path,
    resolved: dict[str, Any],
    manifest: dict[str, Any] | None,
    tokenization: dict[str, Any],
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    write_scores(run_dir / "scores.csv", records)
    write_json(run_dir / "target_tokenization.json", tokenization)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "resolved_config.json", resolved)
    write_text(run_dir / "summary.md", render_summary(resolved, manifest, tokenization, metrics))


def main() -> int:
    args = parse_args()
    try:
        run_id, run_dir, checked_at = create_run_directory()
    except OSError as exc:
        print(f"P1 失败：无法创建运行目录：{sanitize_text(exc)}", file=sys.stderr)
        return 3

    resolved, tokenization, metrics = initialize_result_files(
        run_dir, run_id, checked_at, args.repeats
    )
    manifest: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    model: Any = None
    tokenizer: Any = None
    torch: Any = None

    try:
        if args.repeats != EXPECTED_REPEATS:
            raise ValueError(f"P1 固定要求 --repeats {EXPECTED_REPEATS}")
        if not PROJECT_ROOT.resolve().is_relative_to(DATA_ROOT):
            raise RuntimeError("P1 云端项目和运行结果必须位于 /root/autodl-tmp/")

        model_config_path = args.model_config.expanduser().resolve()
        fingerprint_config_path = args.fingerprint_config.expanduser().resolve()
        manifest = load_fingerprint_manifest(fingerprint_config_path)
        write_json(run_dir / "fingerprint_manifest.json", manifest)

        p0b = find_latest_successful_p0b()
        model_config, snapshot_path, cache_path = validate_model_and_p0b(
            model_config_path, p0b
        )
        p0b_resolved = p0b["resolved"]
        revision = p0b_resolved["resolved_revision"]
        dtype_name = p0b_resolved["torch_dtype_resolved"]
        resolved = {
            "status": "validated",
            "script_version": SCRIPT_VERSION,
            "run_id": run_id,
            "checked_at": checked_at,
            "model_config_path": str(model_config_path),
            "fingerprint_config_path": str(fingerprint_config_path),
            "p0b_run_directory": str(p0b["run_dir"]),
            "p0b_resolved_config_path": str(p0b["resolved_path"]),
            "p0b_environment_path": str(p0b["environment_path"]),
            "model_id": EXPECTED_MODEL_ID,
            "revision": revision,
            "snapshot_path": str(snapshot_path),
            "huggingface_cache_directory": str(cache_path),
            "torch_dtype": dtype_name,
            "device": model_config["device"],
            "trust_remote_code": model_config["trust_remote_code"],
            "model_loader": "transformers.AutoModelForCausalLM.from_pretrained",
            "tokenizer_loader": "transformers.AutoTokenizer.from_pretrained",
            "local_files_only": True,
            "chat_template": "tokenizer.apply_chat_template",
            "enable_thinking": False,
            "system_prompt": None,
            "do_sample": False,
            "batch_size": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "repeats": EXPECTED_REPEATS,
            "adapter": None,
        }
        write_json(run_dir / "resolved_config.json", resolved)

        import torch as torch_import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_import
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            trust_remote_code=model_config["trust_remote_code"],
            local_files_only=True,
        )
        tokenization = tokenize_targets(tokenizer, manifest, EXPECTED_MODEL_ID, revision)
        write_json(run_dir / "target_tokenization.json", tokenization)
        if tokenization["status"] != "pass":
            raise RuntimeError("目标代号 Token 检查失败，按要求停止模型推理")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            torch_dtype=dtype,
            trust_remote_code=model_config["trust_remote_code"],
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(model_config["device"])
        model.eval()
        torch.cuda.synchronize(model_config["device"])

        raw_path = run_dir / "raw_generations.jsonl"
        with raw_path.open("a", encoding="utf-8") as raw_file:
            for repeat_id in range(1, EXPECTED_REPEATS + 1):
                for fingerprint in manifest["fingerprints"]:
                    record = generate_record(
                        torch=torch,
                        model=model,
                        tokenizer=tokenizer,
                        fingerprint_set_id=manifest["fingerprint_set_id"],
                        fingerprint=fingerprint,
                        allowed_responses=manifest["allowed_responses"],
                        repeat_id=repeat_id,
                        device=model_config["device"],
                    )
                    records.append(record)
                    raw_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    raw_file.flush()

        metrics = compute_metrics(manifest, records, EXPECTED_REPEATS, tokenization)
        resolved["status"] = "completed"
        persist_results(run_dir, resolved, manifest, tokenization, records, metrics)
        print(f"P1 原始模型负例检查完成：{'通过' if metrics['p1_passed'] else '失败'}")
        print(f"输出目录：{run_dir}")
        return 0 if metrics["p1_passed"] else 2

    except Exception as exc:
        error = {"type": type(exc).__name__, "message": sanitize_text(exc)}
        resolved["status"] = "failed"
        resolved["error"] = error
        if manifest is not None:
            metrics = compute_metrics(
                manifest,
                records,
                EXPECTED_REPEATS,
                tokenization,
                errors=[error],
            )
        else:
            metrics = empty_metrics(EXPECTED_REPEATS, errors=[error])
        if tokenization.get("status") == "not_run":
            tokenization = {
                **tokenization,
                "status": "fail",
                "errors": [error["message"]],
            }
        try:
            persist_results(run_dir, resolved, manifest, tokenization, records, metrics)
        except Exception as report_exc:
            print(f"P1 报告写入失败：{sanitize_text(report_exc)}", file=sys.stderr)
        print(f"P1 失败：{error['type']}：{error['message']}", file=sys.stderr)
        print(f"输出目录：{run_dir}")
        return 3
    finally:
        model = None
        tokenizer = None
        gc.collect()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
