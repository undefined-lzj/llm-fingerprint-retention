#!/usr/bin/env python3
"""P3-1：原始模型32条负例筛查，以及B0均匀LoRA注入闭环。"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import io
import json
import math
import random
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (SCRIPT_DIR, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from fingerprint.p2 import (  # noqa: E402
    collate_training_examples,
    parameter_summary,
    resolve_target_modules,
    summarize_training_losses,
)
from fingerprint.p3 import (  # noqa: E402
    P1_RESPONSES,
    build_completion_example,
    build_training_order,
    build_training_records,
    compute_capability_metrics,
    compute_fingerprint_metrics,
    create_unique_p3_run_directory,
    load_jsonl,
    load_p3_fingerprint_manifest,
    outputs_identical_between_models,
    sha256_file,
    tokenize_fingerprint_targets,
    validate_b0_training_config,
    validate_dolly_split_manifest,
)
from fingerprint.parser import parse_response  # noqa: E402
from p0b_common import sanitize_text, write_json, write_text  # noqa: E402


SCRIPT_VERSION = "p3-1-b0-1.0.1"
DATA_ROOT = Path("/root/autodl-tmp")
EXPECTED_MODEL_ID = "Qwen/Qwen3-0.6B"
EXPECTED_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
EXPECTED_DTYPE = "bfloat16"
FORBIDDEN_WARNING_FRAGMENTS = (
    "`torch_dtype` is deprecated",
    "generation flags are not valid",
    "`generation_config` default values have been modified",
)
SCORE_FIELDS = [
    "fingerprint_id",
    "repeat_id",
    "target_response",
    "normalized_output",
    "parse_status",
    "is_exact_match",
]


class TeeStream:
    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file

    @property
    def encoding(self) -> str | None:
        return self.terminal.encoding

    def write(self, value: str) -> int:
        written = self.terminal.write(value)
        self.log_file.write(value)
        self.log_file.flush()
        return written

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("base-screen", "train-b0"))
    parser.add_argument(
        "--training-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "p3_1_qwen3_0_6b_b0.json",
    )
    parser.add_argument(
        "--fingerprint-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.json",
    )
    parser.add_argument(
        "--fingerprint-sha",
        type=Path,
        default=PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.sha256",
    )
    parser.add_argument(
        "--dolly-manifest",
        type=Path,
        default=PROJECT_ROOT / "data_manifests" / "p3_dolly_split_manifest.json",
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}顶层必须是JSON对象")
    return value


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        for record in records
    )
    write_text(path, content)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()


def write_scores(path: Path, records: list[dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SCORE_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({field: record.get(field) for field in SCORE_FIELDS})
    write_text(path, output.getvalue())


def write_capability_scores(
    path: Path,
    records: list[dict[str, Any]],
    known_codes: set[str],
) -> None:
    fields = [
        "original_index",
        "category",
        "content_sha256",
        "output_nonempty",
        "is_exact_known_code",
        "contains_thinking_tag",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        raw_output = str(record.get("raw_output") or "")
        writer.writerow(
            {
                "original_index": record["original_index"],
                "category": record["category"],
                "content_sha256": record["content_sha256"],
                "output_nonempty": bool(raw_output.strip()),
                "is_exact_known_code": raw_output.strip() in known_codes,
                "contains_thinking_tag": "<think>" in raw_output
                or "</think>" in raw_output,
            }
        )
    write_text(path, output.getvalue())


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def latest_successful_p2() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for run_dir in (PROJECT_ROOT / "runs").glob("p2_lora_*"):
        comparison_path = run_dir / "comparison.json"
        resolved_path = run_dir / "resolved_config.json"
        if not comparison_path.is_file() or not resolved_path.is_file():
            continue
        try:
            comparison = load_json(comparison_path)
            resolved = load_json(resolved_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if comparison.get("p2_passed") is True and resolved.get("status") == "completed":
            candidates.append((run_dir.resolve(), resolved, comparison))
    if not candidates:
        raise RuntimeError("找不到通过验收的P2运行")
    return max(candidates, key=lambda item: item[0].name)


def validate_p2_reference(
    p2_dir: Path,
    p2_resolved: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    for field in ("model_id", "revision", "dtype"):
        if p2_resolved.get(field) != config[field]:
            raise RuntimeError(f"P2的{field}与P3-1锁定配置不一致")
    if p2_resolved.get("local_files_only") is not True:
        raise RuntimeError("P2没有使用local_files_only=true")
    snapshot = Path(str(p2_resolved.get("snapshot_path"))).expanduser().resolve()
    if not snapshot.is_relative_to(DATA_ROOT) or not snapshot.is_dir():
        raise RuntimeError("P2本地模型快照不存在或不在/root/autodl-tmp/")
    if snapshot.name != EXPECTED_REVISION:
        raise RuntimeError("P2模型快照目录与锁定revision不一致")
    inherited = p2_resolved.get("training_config")
    if not isinstance(inherited, dict):
        raise RuntimeError("P2 resolved_config没有记录完整训练配置")
    fields = (
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "task_type",
        "target_modules",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "bf16",
        "gradient_checkpointing",
    )
    for field in fields:
        if inherited.get(field) != config.get(field):
            raise RuntimeError(f"B0字段{field}没有继承P2成功配置")
    if p2_dir.name != p2_resolved.get("run_id"):
        raise RuntimeError("P2运行目录名与resolved_config不一致")
    return snapshot


def clean_generation_config(model: Any, maximum_new_tokens: int) -> Any:
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.max_new_tokens = maximum_new_tokens
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.num_return_sequences = 1
    return generation_config


def generate_greedily(
    model: Any,
    inputs: Any,
    generation_config: Any,
    *,
    pad_token_id: int,
) -> Any:
    """以显式贪心设置生成，禁止Transformers回填模型的采样默认值。"""

    if generation_config.do_sample is not False:
        raise RuntimeError("生成配置未关闭采样")
    if any(
        getattr(generation_config, name) is not None
        for name in ("temperature", "top_p", "top_k")
    ):
        raise RuntimeError("贪心生成配置仍包含采样参数")
    return model.generate(
        **inputs,
        generation_config=generation_config,
        use_model_defaults=False,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
    )


def release_gpu(torch: Any) -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_model(
    AutoModelForCausalLM: Any,
    torch: Any,
    snapshot: Path,
    device: str,
) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)


def build_generation_inputs(tokenizer: Any, prompt: str, device: str) -> tuple[Any, int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(device)
    return inputs, int(inputs["attention_mask"].sum().item())


def evaluate_fingerprints(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    manifest: dict[str, Any],
    repeats: int,
    maximum_new_tokens: int,
    device: str,
    output_dir: Path,
    expected_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation_config = clean_generation_config(model, maximum_new_tokens)
    model.eval()
    records: list[dict[str, Any]] = []
    allowed = manifest["allowed_responses"]
    for repeat_id in range(1, repeats + 1):
        for fingerprint in manifest["fingerprints"]:
            inputs, input_tokens = build_generation_inputs(
                tokenizer, fingerprint["prompt"], device
            )
            input_width = int(inputs["input_ids"].shape[-1])
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = generate_greedily(
                    model,
                    inputs,
                    generation_config,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize(device)
            latency = time.perf_counter() - started
            new_tokens = generated[0, input_width:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            output_with_special = tokenizer.decode(new_tokens, skip_special_tokens=False)
            parsed = parse_response(
                raw_output,
                fingerprint["target_response"],
                allowed,
            )
            contains_thinking = any(
                tag in value
                for value in (raw_output, output_with_special)
                for tag in ("<think>", "</think>")
            )
            record = {
                "fingerprint_set_id": manifest["fingerprint_set_id"],
                "fingerprint_id": fingerprint["fingerprint_id"],
                "repeat_id": repeat_id,
                "prompt": fingerprint["prompt"],
                "record_id": fingerprint["record_id"],
                "target_response": fingerprint["target_response"],
                "raw_output": raw_output,
                "decoded_with_special_tokens": output_with_special,
                "normalized_output": parsed.normalized_output,
                "parse_status": parsed.status,
                "is_exact_match": parsed.is_exact_match,
                "input_tokens": input_tokens,
                "output_tokens": int(new_tokens.numel()),
                "latency_seconds": round(latency, 6),
                "contains_thinking_tag": contains_thinking,
            }
            records.append(record)
            print(
                f"{expected_mode} repeat={repeat_id} {fingerprint['fingerprint_id']} "
                f"status={parsed.status} output={raw_output!r}"
            )
            del inputs, generated, new_tokens
    metrics = compute_fingerprint_metrics(
        manifest,
        records,
        repeats=repeats,
        expected_mode=expected_mode,
    )
    write_jsonl(output_dir / "raw_generations.jsonl", records)
    write_scores(output_dir / "scores.csv", records)
    write_json(output_dir / "metrics.json", metrics)
    return records, metrics


def average_completion_loss(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    maximum_length: int,
    device: str,
) -> dict[str, Any]:
    model.eval()
    total_weighted_loss = 0.0
    total_supervised_tokens = 0
    per_example: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in records:
            example = build_completion_example(
                tokenizer,
                sample_id=f"capability:{row['original_index']}:{row['content_sha256'][:12]}",
                sample_type="capability",
                prompt=row["prompt"],
                response=row["response"],
                max_sequence_length=maximum_length,
            )
            batch = collate_training_examples(
                [example],
                pad_token_id=tokenizer.pad_token_id,
                tensor_factory=lambda value: torch.tensor(
                    value, dtype=torch.long, device=device
                ),
            )
            output = model(**batch)
            loss = float(output.loss.detach().float().item())
            shifted_labels = batch["labels"][:, 1:]
            token_count = int((shifted_labels != -100).sum().item())
            if not math.isfinite(loss) or token_count <= 0:
                raise FloatingPointError("能力评估出现非有限loss或零监督Token")
            total_weighted_loss += loss * token_count
            total_supervised_tokens += token_count
            per_example.append(
                {
                    "original_index": row["original_index"],
                    "content_sha256": row["content_sha256"],
                    "category": row["category"],
                    "completion_token_count": token_count,
                    "average_completion_token_loss": loss,
                }
            )
            del batch, output, shifted_labels
    return {
        "example_count": len(records),
        "completion_token_count": total_supervised_tokens,
        "average_completion_token_loss": total_weighted_loss
        / total_supervised_tokens,
        "per_example": per_example,
    }


def generate_capability_samples(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    maximum_new_tokens: int,
    device: str,
) -> list[dict[str, Any]]:
    selected = records[:10]
    if len(selected) != 10:
        raise ValueError("capability_eval不足10条生成样本")
    generation_config = clean_generation_config(model, maximum_new_tokens)
    model.eval()
    outputs: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in selected:
            inputs, input_tokens = build_generation_inputs(tokenizer, row["prompt"], device)
            input_width = int(inputs["input_ids"].shape[-1])
            generated = generate_greedily(
                model,
                inputs,
                generation_config,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[0, input_width:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(
                {
                    "original_index": row["original_index"],
                    "content_sha256": row["content_sha256"],
                    "category": row["category"],
                    "prompt": row["prompt"],
                    "reference_response": row["response"],
                    "raw_output": raw_output,
                    "input_tokens": input_tokens,
                    "output_tokens": int(new_tokens.numel()),
                }
            )
            del inputs, generated, new_tokens
    return outputs


def train_b0(
    *,
    torch: Any,
    get_linear_schedule_with_warmup: Any,
    model: Any,
    examples: list[dict[str, Any]],
    training_plan: list[dict[str, int | str]],
    tokenizer: Any,
    config: dict[str, Any],
    metrics_path: Path,
    device: str,
) -> tuple[list[float], list[dict[str, Any]]]:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    accumulation = config["gradient_accumulation_steps"]
    if len(training_plan) % accumulation:
        raise ValueError("B0训练记录不能被梯度累积步数整除")
    optimizer_steps = len(training_plan) // accumulation
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=optimizer_steps,
    )
    losses: list[float] = []
    metric_rows: list[dict[str, Any]] = []
    model.train()
    original_use_cache = bool(model.config.use_cache)
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats(device)
    write_text(metrics_path, "")
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    started = time.perf_counter()
    try:
        for plan_position, item in enumerate(training_plan, start=1):
            example = examples[int(item["record_index"])]
            batch = collate_training_examples(
                [example],
                pad_token_id=tokenizer.pad_token_id,
                tensor_factory=lambda value: torch.tensor(
                    value, dtype=torch.long, device=device
                ),
            )
            output = model(**batch)
            raw_loss = output.loss * float(example["sample_weight"])
            if not bool(torch.isfinite(raw_loss).item()):
                raise FloatingPointError(f"训练位置{plan_position}出现非有限loss")
            accumulated_loss += float(raw_loss.detach().float().item())
            (raw_loss / accumulation).backward()
            del batch, output, raw_loss
            if plan_position % accumulation:
                continue

            optimizer_step = plan_position // accumulation
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, config["max_grad_norm"]
            )
            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError(f"第{optimizer_step}步出现非有限梯度")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            average_loss = accumulated_loss / accumulation
            accumulated_loss = 0.0
            losses.append(average_loss)
            row = {
                "optimizer_step": optimizer_step,
                "epoch": int(item["epoch"]),
                "loss": round(average_loss, 8),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_gpu_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
            }
            started = time.perf_counter()
            metric_rows.append(row)
            append_jsonl(metrics_path, row)
            if optimizer_step % config["logging_steps"] == 0:
                print(
                    f"B0 step {optimizer_step}/{optimizer_steps} epoch={item['epoch']} "
                    f"loss={average_loss:.8f} lr={row['learning_rate']:.8g}"
                )
    finally:
        model.config.use_cache = original_use_cache
    return losses, metric_rows


def initial_evaluation_metrics(expected_mode: str) -> dict[str, Any]:
    return {
        "fingerprint_set_id": "p3_dev_key_v1",
        "expected_mode": expected_mode,
        "unique_fingerprint_count": 32,
        "repeat_count": 2,
        "completed_query_count_by_repeat": {"1": 0, "2": 0},
        "exact_match_count_by_repeat": {"1": 0, "2": 0},
        "exact_match_rate_by_repeat": {"1": 0.0, "2": 0.0},
        "wrong_valid_code_count_by_repeat": {"1": 0, "2": 0},
        "invalid_output_count_by_repeat": {"1": 0, "2": 0},
        "exact_match_fingerprint_ids_by_repeat": {"1": [], "2": []},
        "outputs_identical_across_repeats": False,
        "non_identical_fingerprint_ids": [],
        "thinking_tag_count": 0,
        "all_queries_completed": False,
        "evaluation_passed": False,
    }


def initial_comparison() -> dict[str, Any]:
    return {
        "base_screen_passed": False,
        "base_exact_match_rate_by_repeat": None,
        "training_completed": False,
        "only_lora_parameters_trainable": False,
        "b0_adapter_reload_completed": False,
        "b0_adapter_exact_match_rate_by_repeat": None,
        "b0_merged_reload_completed": False,
        "b0_merged_exact_match_rate_by_repeat": None,
        "adapter_and_merged_outputs_identical": False,
        "capability_check_passed": False,
        "capability_relative_loss_change": None,
        "thinking_tag_count": 0,
        "forbidden_warning_count": 0,
        "failure_reasons": ["P3-1尚未完成"],
        "p3_1_passed": False,
    }


def initialize_run(run_dir: Path, run_id: str, checked_at: str) -> None:
    for name in (
        "base_screen",
        "b0_adapter",
        "b0_adapter_evaluation",
        "b0_merged_model",
        "b0_merged_evaluation",
        "capability_evaluation",
    ):
        (run_dir / name).mkdir(exist_ok=False)
    for name, mode in (
        ("base_screen", "negative"),
        ("b0_adapter_evaluation", "positive"),
        ("b0_merged_evaluation", "positive"),
    ):
        directory = run_dir / name
        write_text(directory / "raw_generations.jsonl", "")
        write_scores(directory / "scores.csv", [])
        write_json(directory / "metrics.json", initial_evaluation_metrics(mode))
    write_json(
        run_dir / "resolved_config.json",
        {
            "status": "not_run",
            "script_version": SCRIPT_VERSION,
            "run_id": run_id,
            "checked_at": checked_at,
            "stage_id": "P3-1",
            "experiment_tier": "pilot",
            "paper_usage": "方法预实验，不进入论文主结果",
        },
    )
    write_text(run_dir / "fingerprint_manifest.json", "{}\n")
    write_json(run_dir / "target_tokenization.json", {"status": "not_run"})
    write_json(run_dir / "dolly_split_manifest.json", {"status": "not_run"})
    write_json(run_dir / "training_data_summary.json", {"status": "not_run"})
    write_json(run_dir / "training_order_sha256.json", {"status": "not_run"})
    write_text(run_dir / "training_metrics.jsonl", "")
    write_json(
        run_dir / "training_summary.json",
        {"training_completed": False, "oom_detected": False, "errors": []},
    )
    write_json(run_dir / "comparison.json", initial_comparison())
    write_text(run_dir / "capability_evaluation" / "raw_generations.jsonl", "")
    write_capability_scores(
        run_dir / "capability_evaluation" / "scores.csv", [], set()
    )
    write_json(
        run_dir / "capability_evaluation" / "metrics.json",
        {"status": "not_run", "capability_check_passed": False},
    )
    write_text(run_dir / "summary.md", "# P3-1 B0摘要\n\n- 状态：尚未开始\n")


def render_summary(
    resolved: dict[str, Any],
    base_metrics: dict[str, Any],
    training_summary: dict[str, Any],
    adapter_metrics: dict[str, Any],
    merged_metrics: dict[str, Any],
    capability_metrics: dict[str, Any],
    comparison: dict[str, Any],
    error: dict[str, str] | None = None,
) -> str:
    lines = [
        "# P3-1开发指纹与B0普通注入基线摘要",
        "",
        f"- 运行编号：`{resolved.get('run_id')}`",
        f"- 状态：`{resolved.get('status')}`",
        f"- 模型：`{resolved.get('model_id')}`",
        f"- revision：`{resolved.get('revision')}`",
        f"- P2父运行：`{resolved.get('parent_run_ids')}`",
        f"- 指纹集合：`{resolved.get('fingerprint_set_id')}`",
        f"- Dolly revision：`{resolved.get('dolly_dataset_revision')}`",
        "",
        "## 原始模型筛查",
        "",
        f"- 每轮命中：`{base_metrics.get('exact_match_count_by_repeat')}`",
        f"- 两轮输出一致：`{base_metrics.get('outputs_identical_across_repeats')}`",
        f"- 两轮不一致的指纹：`{base_metrics.get('non_identical_fingerprint_ids')}`",
        f"- 通过：`{base_metrics.get('evaluation_passed')}`",
        "",
        "## B0训练与重载",
        "",
        f"- 正常数据：`{training_summary.get('normal_example_count')}`",
        f"- 指纹数据：`{training_summary.get('fingerprint_repeated_count')}`",
        f"- optimizer step：`{training_summary.get('optimizer_step_count')}`",
        f"- 仅LoRA参数可训练：`{training_summary.get('only_lora_parameters_trainable')}`",
        f"- 适配器每轮命中：`{adapter_metrics.get('exact_match_count_by_repeat')}`",
        f"- 合并模型每轮命中：`{merged_metrics.get('exact_match_count_by_repeat')}`",
        f"- 适配器与合并模型输出一致：`{comparison.get('adapter_and_merged_outputs_identical')}`",
        "",
        "## 能力冒烟检查",
        "",
        f"- 基础模型平均completion loss：`{capability_metrics.get('base_average_completion_token_loss')}`",
        f"- B0平均completion loss：`{capability_metrics.get('b0_average_completion_token_loss')}`",
        f"- 相对变化：`{capability_metrics.get('relative_loss_change')}`",
        f"- 通过：`{capability_metrics.get('capability_check_passed')}`",
        "",
        f"- P3-1通过：`{comparison.get('p3_1_passed')}`",
    ]
    if comparison.get("failure_reasons"):
        lines.extend(["", "## 未通过原因", ""])
        lines.extend(f"- {reason}" for reason in comparison["failure_reasons"])
    if error:
        lines.extend(["", "## 执行错误", "", f"- {error['type']}：{error['message']}"])
    lines.extend(
        [
            "",
            "> 本阶段未执行代理微调、易遗忘加权、B1、P方法、量化或P3-2。",
            "",
        ]
    )
    return "\n".join(lines)


def load_and_validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.training_config.expanduser().resolve()
    fingerprint_path = args.fingerprint_config.expanduser().resolve()
    fingerprint_sha_path = args.fingerprint_sha.expanduser().resolve()
    dolly_manifest_path = args.dolly_manifest.expanduser().resolve()
    config = load_json(config_path)
    validate_b0_training_config(config)
    fingerprints = load_p3_fingerprint_manifest(fingerprint_path, fingerprint_sha_path)
    dolly_manifest = load_json(dolly_manifest_path)
    validate_dolly_split_manifest(dolly_manifest, require_files=True)
    if dolly_manifest.get("fingerprint_manifest_sha256") != sha256_file(fingerprint_path):
        raise RuntimeError("Dolly冻结清单引用的指纹SHA256与当前文件不一致")
    p2_dir, p2_resolved, _p2_comparison = latest_successful_p2()
    snapshot = validate_p2_reference(p2_dir, p2_resolved, config)
    return {
        "config_path": config_path,
        "fingerprint_path": fingerprint_path,
        "fingerprint_sha_path": fingerprint_sha_path,
        "dolly_manifest_path": dolly_manifest_path,
        "config": config,
        "fingerprints": fingerprints,
        "dolly_manifest": dolly_manifest,
        "p2_dir": p2_dir,
        "p2_resolved": p2_resolved,
        "snapshot": snapshot,
    }


def validate_cloud_environment(config: dict[str, Any]) -> None:
    if not PROJECT_ROOT.resolve().is_relative_to(DATA_ROOT):
        raise RuntimeError("P3-1正式运行必须位于/root/autodl-tmp/")
    minimum = int(config["minimum_free_disk_gib"] * 1024**3)
    if shutil.disk_usage(PROJECT_ROOT).free < minimum:
        raise RuntimeError("数据盘剩余空间不足8GiB")


def execute_base_screen(args: argparse.Namespace, run_id: str, run_dir: Path, checked_at: str) -> int:
    resolved: dict[str, Any] = {
        "status": "not_run",
        "script_version": SCRIPT_VERSION,
        "run_id": run_id,
        "checked_at": checked_at,
        "stage_id": "P3-1",
        "experiment_tier": "pilot",
        "paper_usage": "方法预实验，不进入论文主结果",
        "git_commit": git_commit(),
    }
    base_metrics = initial_evaluation_metrics("negative")
    training_summary: dict[str, Any] = {"training_completed": False}
    capability_metrics: dict[str, Any] = {}
    comparison = initial_comparison()
    torch: Any = None
    model: Any = None
    tokenizer: Any = None
    try:
        validate_cloud_environment(load_json(args.training_config.expanduser().resolve()))
        inputs = load_and_validate_inputs(args)
        config = inputs["config"]
        fingerprints = inputs["fingerprints"]
        dolly_manifest = inputs["dolly_manifest"]
        snapshot = inputs["snapshot"]
        p2_dir = inputs["p2_dir"]

        shutil.copy2(inputs["fingerprint_path"], run_dir / "fingerprint_manifest.json")
        shutil.copy2(inputs["dolly_manifest_path"], run_dir / "dolly_split_manifest.json")
        resolved.update(
            {
                "status": "validated",
                "model_id": config["model_id"],
                "revision": config["revision"],
                "dtype": config["dtype"],
                "device": config["device"],
                "local_files_only": True,
                "snapshot_path": str(snapshot),
                "fingerprint_set_id": fingerprints["fingerprint_set_id"],
                "fingerprint_manifest_sha256": sha256_file(inputs["fingerprint_path"]),
                "dolly_dataset_revision": dolly_manifest["dataset_revision"],
                "dolly_split_manifest_sha256": sha256_file(inputs["dolly_manifest_path"]),
                "parent_run_ids": [p2_dir.name],
                "p2_resolved_config_path": str(
                    (p2_dir / "resolved_config.json").relative_to(PROJECT_ROOT)
                ),
                "training_config_path": str(inputs["config_path"].relative_to(PROJECT_ROOT)),
                "training_config_sha256": sha256_file(inputs["config_path"]),
                "training_config": config,
                "generation": {
                    "enable_thinking": False,
                    "do_sample": False,
                    "batch_size": 1,
                    "max_new_tokens": 16,
                    "repeats": 2,
                    "system_prompt": None,
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                },
            }
        )
        write_json(run_dir / "resolved_config.json", resolved)

        import torch as torch_import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_import
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU不支持BF16")
        device = config["device"]
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            trust_remote_code=False,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer缺少PAD和EOS Token")
            tokenizer.pad_token = tokenizer.eos_token
        target_rows = tokenize_fingerprint_targets(tokenizer, fingerprints, maximum_tokens=5)
        write_json(
            run_dir / "target_tokenization.json",
            {
                "fingerprint_set_id": fingerprints["fingerprint_set_id"],
                "maximum_tokens": 5,
                "all_targets_passed": True,
                "targets": target_rows,
            },
        )

        model = load_base_model(AutoModelForCausalLM, torch, snapshot, device)
        _base_records, base_metrics = evaluate_fingerprints(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            manifest=fingerprints,
            repeats=config["evaluation_repeats"],
            maximum_new_tokens=config["evaluation_max_new_tokens"],
            device=device,
            output_dir=run_dir / "base_screen",
            expected_mode="negative",
        )
        if not base_metrics["evaluation_passed"]:
            reasons: list[str] = []
            matched = base_metrics["exact_match_fingerprint_ids_by_repeat"]
            if any(matched.values()):
                reasons.append(f"命中指纹={matched}")
            mismatched = base_metrics["non_identical_fingerprint_ids"]
            if mismatched:
                reasons.append(f"两轮输出不一致={mismatched}")
            if base_metrics["thinking_tag_count"]:
                reasons.append(
                    f"思考标签数量={base_metrics['thinking_tag_count']}"
                )
            if not base_metrics["all_queries_completed"]:
                reasons.append("查询未全部完成")
            detail = "；".join(reasons) or "未知验收项失败"
            raise RuntimeError(f"原始模型负例筛查未通过：{detail}")

        capability_rows = load_jsonl(
            Path(dolly_manifest["splits"]["capability_eval"]["file_path"])
        )
        base_loss = average_completion_loss(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            records=capability_rows,
            maximum_length=config["max_seq_length"],
            device=device,
        )
        write_json(run_dir / "capability_evaluation" / "base_loss.json", base_loss)
        resolved["status"] = "base_screen_passed"
        write_json(run_dir / "resolved_config.json", resolved)
        comparison.update(
            {
                "base_screen_passed": True,
                "base_exact_match_rate_by_repeat": base_metrics[
                    "exact_match_rate_by_repeat"
                ],
                "failure_reasons": ["B0训练尚未执行"],
            }
        )
        write_json(run_dir / "comparison.json", comparison)
        write_text(
            run_dir / "summary.md",
            render_summary(
                resolved,
                base_metrics,
                training_summary,
                initial_evaluation_metrics("positive"),
                initial_evaluation_metrics("positive"),
                capability_metrics,
                comparison,
            ),
        )
        print("P3-1原始模型负例筛查通过：两轮均0/32。")
        print(f"输出目录：{run_dir}")
        return 0
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": sanitize_text(exc)}
        resolved["status"] = "base_screen_failed"
        resolved["error"] = error
        comparison.update(
            {
                "base_exact_match_rate_by_repeat": base_metrics.get(
                    "exact_match_rate_by_repeat"
                ),
                "failure_reasons": [f"{error['type']}：{error['message']}"],
                "p3_1_passed": False,
            }
        )
        write_json(run_dir / "resolved_config.json", resolved)
        write_json(run_dir / "comparison.json", comparison)
        write_text(
            run_dir / "summary.md",
            render_summary(
                resolved,
                base_metrics,
                training_summary,
                initial_evaluation_metrics("positive"),
                initial_evaluation_metrics("positive"),
                capability_metrics,
                comparison,
                error,
            ),
        )
        print(f"P3-1原始筛查失败：{error['type']}：{error['message']}", file=sys.stderr)
        print(f"输出目录：{run_dir}")
        return 3
    finally:
        model = None
        tokenizer = None
        release_gpu(torch)


def execute_train_b0(args: argparse.Namespace, run_dir: Path) -> int:
    resolved = load_json(run_dir / "resolved_config.json")
    base_metrics = load_json(run_dir / "base_screen" / "metrics.json")
    training_summary: dict[str, Any] = {
        "training_completed": False,
        "oom_detected": False,
        "errors": [],
    }
    adapter_metrics = initial_evaluation_metrics("positive")
    merged_metrics = initial_evaluation_metrics("positive")
    capability_metrics: dict[str, Any] = {}
    comparison = load_json(run_dir / "comparison.json")
    adapter_records: list[dict[str, Any]] = []
    merged_records: list[dict[str, Any]] = []
    stages = {
        "adapter_reload_completed": False,
        "merge_completed": False,
        "merged_reload_completed": False,
    }
    torch: Any = None
    model: Any = None
    tokenizer: Any = None
    try:
        if resolved.get("status") != "base_screen_passed":
            raise RuntimeError("当前运行目录没有通过原始模型负例筛查")
        if base_metrics.get("evaluation_passed") is not True:
            raise RuntimeError("base_screen/metrics.json没有通过验收，禁止B0训练")
        inputs = load_and_validate_inputs(args)
        config = inputs["config"]
        fingerprints = inputs["fingerprints"]
        dolly_manifest = inputs["dolly_manifest"]
        snapshot = inputs["snapshot"]
        validate_cloud_environment(config)
        if resolved.get("training_config_sha256") != sha256_file(inputs["config_path"]):
            raise RuntimeError("B0训练配置在原始筛查后发生变化")
        if resolved.get("fingerprint_manifest_sha256") != sha256_file(
            inputs["fingerprint_path"]
        ):
            raise RuntimeError("P3指纹文件在原始筛查后发生变化")
        if resolved.get("dolly_split_manifest_sha256") != sha256_file(
            inputs["dolly_manifest_path"]
        ):
            raise RuntimeError("Dolly冻结清单在原始筛查后发生变化")
        if resolved.get("parent_run_ids") != [inputs["p2_dir"].name]:
            raise RuntimeError("P2父运行在原始筛查后发生变化")

        normal_records = load_jsonl(
            Path(dolly_manifest["splits"]["normal_train"]["file_path"])
        )
        capability_records = load_jsonl(
            Path(dolly_manifest["splits"]["capability_eval"]["file_path"])
        )
        training_records = build_training_records(
            normal_records,
            fingerprints,
            config["fingerprint_repeat"],
        )
        training_plan, order_sha256 = build_training_order(
            training_records,
            epochs=config["num_train_epochs"],
            seed=config["data_seed"],
        )
        effective_batch = (
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
        )
        optimizer_steps = len(training_plan) // effective_batch

        from peft import LoraConfig, PeftModel, get_peft_model
        import torch as torch_import
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )

        torch = torch_import
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU不支持BF16")
        device = config["device"]
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            trust_remote_code=False,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer缺少PAD和EOS Token")
            tokenizer.pad_token = tokenizer.eos_token

        examples: list[dict[str, Any]] = []
        for row in training_records:
            example = build_completion_example(
                tokenizer,
                sample_id=row["sample_id"],
                sample_type=row["sample_type"],
                prompt=row["prompt"],
                response=row["response"],
                max_sequence_length=config["max_seq_length"],
                sample_weight=row["sample_weight"],
            )
            examples.append(example)
        if len(examples) != 1256 or any(example["was_truncated"] for example in examples):
            raise RuntimeError("B0训练样本数量或截断检查失败")
        write_json(
            run_dir / "training_data_summary.json",
            {
                "status": "validated",
                "normal_example_count": 1000,
                "fingerprint_unique_count": 32,
                "fingerprint_repeat": 8,
                "fingerprint_repeated_count": 256,
                "total_training_record_count": 1256,
                "normal_example_weight": 1,
                "fingerprint_example_weight": 1,
                "maximum_sequence_length": config["max_seq_length"],
                "minimum_input_token_count": min(
                    example["input_token_count"] for example in examples
                ),
                "maximum_input_token_count": max(
                    example["input_token_count"] for example in examples
                ),
                "all_user_and_template_labels_masked": all(
                    all(label == -100 for label in example["labels"][: example["prompt_token_count"]])
                    for example in examples
                ),
                "all_assistant_regions_supervised": all(
                    example["supervised_token_count"] > 0 for example in examples
                ),
                "any_sequence_truncated": False,
            },
        )
        write_json(
            run_dir / "training_order_sha256.json",
            {
                "data_seed": config["data_seed"],
                "epoch_count": config["num_train_epochs"],
                "records_per_epoch": len(training_records),
                "total_plan_record_count": len(training_plan),
                "sha256": order_sha256,
                "first_ten_sample_ids": [
                    row["sample_id"] for row in training_plan[:10]
                ],
                "last_ten_sample_ids": [
                    row["sample_id"] for row in training_plan[-10:]
                ],
            },
        )

        model = load_base_model(AutoModelForCausalLM, torch, snapshot, device)
        module_matches = resolve_target_modules(model, config["target_modules"])
        print("实际找到的LoRA目标模块：")
        for target in config["target_modules"]:
            for full_name in module_matches[target]:
                print(f"- {full_name}")
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias=config["lora_bias"],
            task_type=config["task_type"],
            target_modules=config["target_modules"],
        )
        model = get_peft_model(model, lora_config)
        parameters = parameter_summary(model)
        training_summary = {
            **training_summary,
            **parameters,
            "normal_example_count": 1000,
            "fingerprint_unique_count": 32,
            "fingerprint_repeated_count": 256,
            "total_training_record_count": 1256,
            "num_train_epochs": config["num_train_epochs"],
            "effective_batch_size": effective_batch,
            "expected_optimizer_step_count": optimizer_steps,
            "training_order_sha256": order_sha256,
            "actual_target_modules": module_matches,
        }
        write_json(run_dir / "training_summary.json", training_summary)
        print(f"模型总参数量：{parameters['total_parameter_count']}")
        print(f"LoRA可训练参数量：{parameters['trainable_parameter_count']}")
        print(f"可训练参数比例：{parameters['trainable_parameter_ratio']:.8%}")
        print("正常数据数量：1000")
        print("指纹原始数量：32；重复后数量：256")
        print(f"有效batch size：{effective_batch}")
        print(f"optimizer step数量：{optimizer_steps}")
        print(f"数据顺序SHA256：{order_sha256}")

        losses, metric_rows = train_b0(
            torch=torch,
            get_linear_schedule_with_warmup=get_linear_schedule_with_warmup,
            model=model,
            examples=examples,
            training_plan=training_plan,
            tokenizer=tokenizer,
            config=config,
            metrics_path=run_dir / "training_metrics.jsonl",
            device=device,
        )
        loss_summary = summarize_training_losses(losses, optimizer_steps)
        training_summary.update(
            {
                **loss_summary,
                "training_metric_record_count": len(metric_rows),
                "oom_detected": False,
                "errors": [],
            }
        )
        write_json(run_dir / "training_summary.json", training_summary)

        adapter_dir = run_dir / "b0_adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        resolved["status"] = "b0_adapter_saved"
        write_json(run_dir / "resolved_config.json", resolved)
        model = None
        release_gpu(torch)

        reload_base = load_base_model(AutoModelForCausalLM, torch, snapshot, device)
        adapter_model = PeftModel.from_pretrained(
            reload_base,
            adapter_dir,
            is_trainable=False,
            local_files_only=True,
        )
        stages["adapter_reload_completed"] = True
        adapter_records, adapter_metrics = evaluate_fingerprints(
            torch=torch,
            model=adapter_model,
            tokenizer=tokenizer,
            manifest=fingerprints,
            repeats=config["evaluation_repeats"],
            maximum_new_tokens=config["evaluation_max_new_tokens"],
            device=device,
            output_dir=run_dir / "b0_adapter_evaluation",
            expected_mode="positive",
        )
        if not adapter_metrics["evaluation_passed"]:
            raise RuntimeError("B0适配器没有在两轮中全部达到32/32，停止后续步骤")

        merged_model = adapter_model.merge_and_unload(safe_merge=True)
        stages["merge_completed"] = True
        merged_model.generation_config = clean_generation_config(
            merged_model, config["evaluation_max_new_tokens"]
        )
        merged_dir = run_dir / "b0_merged_model"
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        adapter_model = None
        reload_base = None
        merged_model = None
        release_gpu(torch)

        merged_tokenizer = AutoTokenizer.from_pretrained(
            merged_dir,
            trust_remote_code=False,
            local_files_only=True,
        )
        reloaded_merged = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            dtype=torch.bfloat16,
            trust_remote_code=False,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device)
        stages["merged_reload_completed"] = True
        merged_records, merged_metrics = evaluate_fingerprints(
            torch=torch,
            model=reloaded_merged,
            tokenizer=merged_tokenizer,
            manifest=fingerprints,
            repeats=config["evaluation_repeats"],
            maximum_new_tokens=config["evaluation_max_new_tokens"],
            device=device,
            output_dir=run_dir / "b0_merged_evaluation",
            expected_mode="positive",
        )
        if not merged_metrics["evaluation_passed"]:
            raise RuntimeError("B0合并模型没有在两轮中全部达到32/32")

        b0_loss = average_completion_loss(
            torch=torch,
            model=reloaded_merged,
            tokenizer=merged_tokenizer,
            records=capability_records,
            maximum_length=config["max_seq_length"],
            device=device,
        )
        capability_generations = generate_capability_samples(
            torch=torch,
            model=reloaded_merged,
            tokenizer=merged_tokenizer,
            records=capability_records,
            maximum_new_tokens=config["evaluation_max_new_tokens"],
            device=device,
        )
        base_loss = load_json(run_dir / "capability_evaluation" / "base_loss.json")
        known_codes = set(P1_RESPONSES) | set(fingerprints["allowed_responses"])
        capability_metrics = compute_capability_metrics(
            base_average_loss=float(base_loss["average_completion_token_loss"]),
            b0_average_loss=float(b0_loss["average_completion_token_loss"]),
            generation_records=capability_generations,
            known_codes=known_codes,
            maximum_relative_increase=config[
                "capability_max_relative_loss_increase"
            ],
        )
        write_json(run_dir / "capability_evaluation" / "b0_loss.json", b0_loss)
        write_jsonl(
            run_dir / "capability_evaluation" / "raw_generations.jsonl",
            capability_generations,
        )
        write_capability_scores(
            run_dir / "capability_evaluation" / "scores.csv",
            capability_generations,
            known_codes,
        )
        write_json(
            run_dir / "capability_evaluation" / "metrics.json",
            capability_metrics,
        )
        reloaded_merged = None
        merged_tokenizer = None
        release_gpu(torch)

        between_identical = outputs_identical_between_models(
            adapter_records, merged_records
        )
        terminal_text = (run_dir / "terminal_output.log").read_text(encoding="utf-8")
        warning_count = sum(
            terminal_text.count(fragment) for fragment in FORBIDDEN_WARNING_FRAGMENTS
        )
        thinking_count = (
            int(base_metrics.get("thinking_tag_count", 0))
            + int(adapter_metrics.get("thinking_tag_count", 0))
            + int(merged_metrics.get("thinking_tag_count", 0))
            + int(capability_metrics.get("thinking_tag_count", 0))
        )
        checks = [
            (base_metrics.get("evaluation_passed") is True, "原始模型筛查未通过"),
            (training_summary.get("training_completed") is True, "B0训练未完成"),
            (
                training_summary.get("all_losses_finite") is True,
                "B0训练loss出现NaN或Inf",
            ),
            (
                training_summary.get("only_lora_parameters_trainable") is True,
                "存在非LoRA可训练参数",
            ),
            (stages["adapter_reload_completed"], "B0适配器重载失败"),
            (adapter_metrics.get("evaluation_passed") is True, "B0适配器评估未通过"),
            (stages["merge_completed"], "B0合并失败"),
            (stages["merged_reload_completed"], "B0合并模型重载失败"),
            (merged_metrics.get("evaluation_passed") is True, "B0合并模型评估未通过"),
            (between_identical, "适配器与合并模型指纹输出不一致"),
            (
                capability_metrics.get("capability_check_passed") is True,
                "能力冒烟检查未通过",
            ),
            (thinking_count == 0, "输出包含思考标签"),
            (warning_count == 0, "日志包含弃用或无效采样参数警告"),
        ]
        failure_reasons = [reason for passed, reason in checks if not passed]
        comparison = {
            "base_screen_passed": base_metrics["evaluation_passed"],
            "base_exact_match_rate_by_repeat": base_metrics[
                "exact_match_rate_by_repeat"
            ],
            "training_completed": training_summary.get("training_completed", False),
            "only_lora_parameters_trainable": training_summary.get(
                "only_lora_parameters_trainable", False
            ),
            "b0_adapter_reload_completed": stages["adapter_reload_completed"],
            "b0_adapter_exact_match_rate_by_repeat": adapter_metrics[
                "exact_match_rate_by_repeat"
            ],
            "b0_merged_reload_completed": stages["merged_reload_completed"],
            "b0_merged_exact_match_rate_by_repeat": merged_metrics[
                "exact_match_rate_by_repeat"
            ],
            "adapter_and_merged_outputs_identical": between_identical,
            "capability_check_passed": capability_metrics.get(
                "capability_check_passed", False
            ),
            "capability_relative_loss_change": capability_metrics.get(
                "relative_loss_change"
            ),
            "thinking_tag_count": thinking_count,
            "forbidden_warning_count": warning_count,
            "failure_reasons": failure_reasons,
            "p3_1_passed": not failure_reasons,
        }
        resolved["status"] = "completed"
        write_json(run_dir / "resolved_config.json", resolved)
        write_json(run_dir / "comparison.json", comparison)
        write_text(
            run_dir / "summary.md",
            render_summary(
                resolved,
                base_metrics,
                training_summary,
                adapter_metrics,
                merged_metrics,
                capability_metrics,
                comparison,
            ),
        )
        print(f"P3-1 B0完成：{'通过' if comparison['p3_1_passed'] else '失败'}")
        print(f"输出目录：{run_dir}")
        return 0 if comparison["p3_1_passed"] else 2
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": sanitize_text(exc)}
        is_oom = "outofmemory" in error["type"].lower() or "out of memory" in error[
            "message"
        ].lower()
        training_summary.update(
            {
                "oom_detected": is_oom,
                "errors": [error],
            }
        )
        resolved["status"] = "b0_failed"
        resolved["error"] = error
        comparison.update(
            {
                "base_screen_passed": base_metrics.get("evaluation_passed", False),
                "training_completed": training_summary.get("training_completed", False),
                "only_lora_parameters_trainable": training_summary.get(
                    "only_lora_parameters_trainable", False
                ),
                "b0_adapter_reload_completed": stages["adapter_reload_completed"],
                "b0_adapter_exact_match_rate_by_repeat": adapter_metrics.get(
                    "exact_match_rate_by_repeat"
                ),
                "b0_merged_reload_completed": stages["merged_reload_completed"],
                "b0_merged_exact_match_rate_by_repeat": merged_metrics.get(
                    "exact_match_rate_by_repeat"
                ),
                "failure_reasons": [f"{error['type']}：{error['message']}"],
                "p3_1_passed": False,
            }
        )
        write_json(run_dir / "resolved_config.json", resolved)
        write_json(run_dir / "training_summary.json", training_summary)
        write_json(run_dir / "comparison.json", comparison)
        write_text(
            run_dir / "summary.md",
            render_summary(
                resolved,
                base_metrics,
                training_summary,
                adapter_metrics,
                merged_metrics,
                capability_metrics,
                comparison,
                error,
            ),
        )
        print(f"P3-1 B0失败：{error['type']}：{error['message']}", file=sys.stderr)
        print(f"输出目录：{run_dir}")
        return 3
    finally:
        model = None
        tokenizer = None
        release_gpu(torch)


def resolve_existing_run(path: Path) -> Path:
    run_dir = path.expanduser().resolve()
    runs_root = (PROJECT_ROOT / "runs").resolve()
    if not run_dir.is_relative_to(runs_root) or not run_dir.name.startswith("p3_1_b0_"):
        raise ValueError("--run-dir必须是本项目runs/下的P3-1目录")
    if not run_dir.is_dir():
        raise ValueError(f"P3-1运行目录不存在：{run_dir}")
    return run_dir


def main() -> int:
    args = parse_args()
    if args.stage == "base-screen":
        if args.run_dir is not None:
            print("base-screen阶段不能传入--run-dir", file=sys.stderr)
            return 3
        checked_at_dt = datetime.now().astimezone()
        try:
            run_id, run_dir = create_unique_p3_run_directory(
                PROJECT_ROOT / "runs", checked_at_dt
            )
            initialize_run(
                run_dir,
                run_id,
                checked_at_dt.isoformat(timespec="seconds"),
            )
        except Exception as exc:
            print(f"无法创建P3-1运行目录：{sanitize_text(exc)}", file=sys.stderr)
            return 3
        log_mode = "w"
    else:
        if args.run_dir is None:
            print("train-b0阶段必须传入--run-dir", file=sys.stderr)
            return 3
        try:
            run_dir = resolve_existing_run(args.run_dir)
        except Exception as exc:
            print(f"无效P3-1运行目录：{sanitize_text(exc)}", file=sys.stderr)
            return 3
        run_id = run_dir.name
        log_mode = "a"

    terminal_path = run_dir / "terminal_output.log"
    try:
        log_file = terminal_path.open(log_mode, encoding="utf-8", buffering=1)
    except OSError as exc:
        print(f"无法写入terminal_output.log：{sanitize_text(exc)}", file=sys.stderr)
        return 3
    with log_file:
        with redirect_stdout(TeeStream(sys.stdout, log_file)), redirect_stderr(
            TeeStream(sys.stderr, log_file)
        ):
            print(f"P3-1阶段：{args.stage}")
            print(f"终端日志：{terminal_path}")
            if args.stage == "base-screen":
                return execute_base_screen(
                    args,
                    run_id,
                    run_dir,
                    checked_at_dt.isoformat(timespec="seconds"),
                )
            return execute_train_b0(args, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
