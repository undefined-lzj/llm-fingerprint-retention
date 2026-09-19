#!/usr/bin/env python3
"""P3-2：代理遗忘测量，以及从同一B0起点进行B1/P公平训练。"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    load_jsonl,
    load_p3_fingerprint_manifest,
    outputs_identical_between_models,
    sha256_file,
    validate_dolly_split_manifest,
)
from fingerprint.p32 import (  # noqa: E402
    EXPECTED_MODEL_ID,
    EXPECTED_REVISION,
    build_branch_training_records,
    build_continuation_training_plan,
    build_proxy_training_plan,
    build_target_loss_example,
    compute_capability_comparison,
    compute_fairness_audit,
    compute_forgetting_feedback,
    create_unique_p32_run_directory,
    training_plan_text,
    training_records_sha256,
    validate_continuation_training_plan,
    validate_p32_config,
    weighted_completion_loss,
)
from p0b_common import sanitize_text, write_json, write_text  # noqa: E402
from run_p3_1_b0 import (  # noqa: E402
    FORBIDDEN_WARNING_FRAGMENTS,
    TeeStream,
    average_completion_loss,
    clean_generation_config,
    evaluate_fingerprints,
    generate_capability_samples,
    initial_evaluation_metrics,
    release_gpu,
    write_capability_scores,
    write_scores,
)


SCRIPT_VERSION = "p3-2-feedback-1.0.0"
DATA_ROOT = Path("/root/autodl-tmp")
STAGES = ("proxy-score", "train-branches", "evaluate", "all")
PARENT_REQUIRED_FILES = (
    "resolved_config.json",
    "comparison.json",
    "fingerprint_manifest.json",
    "dolly_split_manifest.json",
    "b0_adapter_evaluation/metrics.json",
    "b0_merged_evaluation/metrics.json",
)
FORBIDDEN_WARNINGS = FORBIDDEN_WARNING_FRAGMENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "p3_2_feedback.json",
    )
    parser.add_argument("--p3-1-run", type=Path)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}顶层必须是JSON对象")
    return value


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in records
    )
    write_text(path, content)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def resolve_run_directory(path: Path, prefix: str) -> Path:
    resolved = path.expanduser().resolve()
    runs_root = (PROJECT_ROOT / "runs").resolve()
    if not resolved.is_relative_to(runs_root) or not resolved.name.startswith(prefix):
        raise ValueError(f"运行目录必须位于本项目runs/下且以{prefix}开头")
    if not resolved.is_dir():
        raise ValueError(f"运行目录不存在：{resolved}")
    return resolved


def successful_p3_1_candidates() -> list[Path]:
    candidates: list[Path] = []
    for run_dir in sorted((PROJECT_ROOT / "runs").glob("p3_1_b0_*")):
        try:
            comparison = load_json(run_dir / "comparison.json")
            resolved = load_json(run_dir / "resolved_config.json")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            comparison.get("p3_1_passed") is True
            and resolved.get("status") == "completed"
        ):
            candidates.append(run_dir.resolve())
    return candidates


def select_parent_p3_1(explicit: Path | None) -> tuple[Path, list[str]]:
    candidates = successful_p3_1_candidates()
    candidate_ids = [path.name for path in candidates]
    if explicit is not None:
        selected = resolve_run_directory(explicit, "p3_1_b0_")
        if selected not in candidates:
            raise RuntimeError("指定的P3-1运行没有通过验收")
    else:
        if not candidates:
            raise RuntimeError("找不到通过验收的P3-1运行")
        selected = candidates[-1]
    print("发现的P3-1通过候选：" + ", ".join(candidate_ids))
    print(f"实际引用的P3-1 run ID：{selected.name}")
    return selected, candidate_ids


def validate_parent_p3_1(
    parent: Path,
    config: dict[str, Any],
    *,
    require_model_artifacts: bool,
) -> dict[str, Any]:
    for relative in PARENT_REQUIRED_FILES:
        if not (parent / relative).is_file():
            raise RuntimeError(f"P3-1父运行缺少文件：{relative}")
    resolved = load_json(parent / "resolved_config.json")
    comparison = load_json(parent / "comparison.json")
    if resolved.get("status") != "completed" or comparison.get("p3_1_passed") is not True:
        raise RuntimeError("P3-1父运行没有通过")
    if resolved.get("run_id") != parent.name:
        raise RuntimeError("P3-1父运行目录名与run_id不一致")
    if resolved.get("model_id") != EXPECTED_MODEL_ID:
        raise RuntimeError("P3-1模型ID不正确")
    if resolved.get("revision") != EXPECTED_REVISION:
        raise RuntimeError("P3-1模型revision不正确")
    parent_config = resolved.get("training_config")
    if not isinstance(parent_config, dict):
        raise RuntimeError("P3-1 resolved_config缺少训练配置快照")
    inherited_fields = (
        "model_id",
        "revision",
        "dtype",
        "device",
        "local_files_only",
        "enable_thinking",
        "max_seq_length",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "task_type",
        "target_modules",
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "bf16",
        "gradient_checkpointing",
        "logging_steps",
        "report_to",
        "fingerprint_repeat",
        "normal_example_count",
        "fingerprint_count",
    )
    for field in inherited_fields:
        if parent_config.get(field) != config.get(field):
            raise RuntimeError(f"P3-2字段{field}没有继承P3-1成功配置")
    for metrics_path in (
        parent / "b0_adapter_evaluation" / "metrics.json",
        parent / "b0_merged_evaluation" / "metrics.json",
    ):
        metrics = load_json(metrics_path)
        if metrics.get("exact_match_count_by_repeat") != {"1": 32, "2": 32}:
            raise RuntimeError(f"P3-1 B0不是两轮32/32：{metrics_path}")
        if metrics.get("evaluation_passed") is not True:
            raise RuntimeError(f"P3-1 B0评估未通过：{metrics_path}")
    fingerprint_path = parent / "fingerprint_manifest.json"
    fingerprint_sha = PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.sha256"
    fingerprints = load_p3_fingerprint_manifest(fingerprint_path)
    expected_fingerprint_sha = fingerprint_sha.read_text(encoding="utf-8").split()[0]
    if sha256_file(fingerprint_path) != expected_fingerprint_sha:
        raise RuntimeError("P3-1父运行的指纹清单与冻结SHA256不一致")
    dolly_manifest = load_json(parent / "dolly_split_manifest.json")
    validate_dolly_split_manifest(dolly_manifest, require_files=False)
    expected_counts = {
        "normal_train": 1000,
        "proxy_reserved": 500,
        "capability_eval": 100,
    }
    for name, count in expected_counts.items():
        if dolly_manifest["splits"][name].get("count") != count:
            raise RuntimeError(f"P3-1 Dolly划分{name}数量不正确")
    snapshot = Path(str(resolved.get("snapshot_path", "")))
    if require_model_artifacts:
        required_artifacts = (
            parent / "b0_adapter" / "adapter_model.safetensors",
            parent / "b0_merged_model" / "model.safetensors",
            snapshot,
        )
        for path in required_artifacts:
            if not path.exists():
                raise RuntimeError(f"P3-2正式运行缺少父模型产物：{path}")
    return {
        "resolved": resolved,
        "comparison": comparison,
        "parent_config": parent_config,
        "fingerprints": fingerprints,
        "fingerprint_path": fingerprint_path,
        "dolly_manifest": dolly_manifest,
        "dolly_manifest_path": parent / "dolly_split_manifest.json",
        "snapshot": snapshot,
    }


def load_selected_dolly_splits(
    manifest: dict[str, Any], names: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    """只读取P3-2允许的数据；刻意不访问unseen_reserved。"""

    allowed = {"normal_train", "proxy_reserved", "capability_eval"}
    if not set(names).issubset(allowed):
        raise ValueError("P3-2禁止读取unseen_reserved")
    loaded: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, set[str]] = {}
    for name in names:
        info = manifest["splits"][name]
        path = Path(info["file_path"])
        if not path.is_file():
            raise RuntimeError(f"Dolly冻结文件不存在：{path}")
        if sha256_file(path) != info.get("file_sha256"):
            raise RuntimeError(f"Dolly冻结文件SHA256不一致：{name}")
        rows = load_jsonl(path)
        if len(rows) != info["count"]:
            raise RuntimeError(f"Dolly冻结文件行数不正确：{name}")
        content_hashes = {row["content_sha256"] for row in rows}
        if len(content_hashes) != len(rows):
            raise RuntimeError(f"Dolly冻结文件存在重复：{name}")
        loaded[name] = rows
        hashes[name] = content_hashes
    ordered = list(names)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if hashes[left] & hashes[right]:
                raise RuntimeError(f"Dolly划分存在交集：{left}/{right}")
    return loaded


def validate_cloud_environment(config: dict[str, Any]) -> None:
    if not PROJECT_ROOT.resolve().is_relative_to(DATA_ROOT):
        raise RuntimeError("P3-2正式运行必须位于/root/autodl-tmp/")
    required = int(float(config["minimum_free_disk_gib"]) * 1024**3)
    if shutil.disk_usage(PROJECT_ROOT).free < required:
        raise RuntimeError("数据盘剩余空间不足8GiB")


def adapter_state_sha256(model: Any, get_peft_model_state_dict: Any) -> str:
    state = get_peft_model_state_dict(model)
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_lora_config(config: dict[str, Any], LoraConfig: Any) -> Any:
    return LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["lora_bias"],
        task_type=config["task_type"],
        target_modules=config["target_modules"],
    )


def set_reproducible_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_weighted_lora(
    *,
    torch: Any,
    get_linear_schedule_with_warmup: Any,
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    maximum_gradient_norm: float,
    accumulation_steps: int,
    logging_steps: int,
    metrics_path: Path,
    device: str,
    label: str,
) -> tuple[list[float], list[dict[str, Any]]]:
    if len(plan) % accumulation_steps:
        raise ValueError(f"{label}训练记录不能被梯度累积步数整除")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    expected_steps = len(plan) // accumulation_steps
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=expected_steps,
    )
    write_text(metrics_path, "")
    model.train()
    original_use_cache = bool(model.config.use_cache)
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    losses: list[float] = []
    metrics: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for position, item in enumerate(plan, start=1):
            example = examples[int(item["record_index"])]
            batch = collate_training_examples(
                [example],
                pad_token_id=tokenizer.pad_token_id,
                tensor_factory=lambda value: torch.tensor(
                    value, dtype=torch.long, device=device
                ),
            )
            labels = batch.pop("labels")
            output = model(**batch)
            sample_weights = torch.tensor(
                [float(example["sample_weight"])],
                dtype=torch.float32,
                device=device,
            )
            loss, per_example, token_counts = weighted_completion_loss(
                torch, output.logits, labels, sample_weights
            )
            accumulated_loss += float(loss.detach().float().item())
            (loss / accumulation_steps).backward()
            del batch, labels, output, sample_weights, loss, per_example, token_counts
            if position % accumulation_steps:
                continue
            optimizer_step = position // accumulation_steps
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, maximum_gradient_norm
            )
            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError(f"{label}第{optimizer_step}步梯度非有限")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            average_loss = accumulated_loss / accumulation_steps
            accumulated_loss = 0.0
            losses.append(average_loss)
            row = {
                "optimizer_step": optimizer_step,
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
            metrics.append(row)
            append_jsonl(metrics_path, row)
            started = time.perf_counter()
            if optimizer_step % logging_steps == 0:
                print(
                    f"{label} step {optimizer_step}/{expected_steps} "
                    f"loss={average_loss:.8f} lr={row['learning_rate']:.8g}"
                )
    finally:
        model.config.use_cache = original_use_cache
    return losses, metrics


def measure_fingerprint_target_losses(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    fingerprints: dict[str, Any],
    maximum_length: int,
    device: str,
) -> dict[str, dict[str, Any]]:
    model.eval()
    results: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for row in fingerprints["fingerprints"]:
            example = build_target_loss_example(
                tokenizer, row, max_sequence_length=maximum_length
            )
            batch = collate_training_examples(
                [example],
                pad_token_id=tokenizer.pad_token_id,
                tensor_factory=lambda value: torch.tensor(
                    value, dtype=torch.long, device=device
                ),
            )
            labels = batch.pop("labels")
            output = model(**batch)
            weights = torch.ones(1, dtype=torch.float32, device=device)
            _loss, per_example, token_counts = weighted_completion_loss(
                torch, output.logits, labels, weights
            )
            fingerprint_id = row["fingerprint_id"]
            results[fingerprint_id] = {
                "loss": float(per_example[0].detach().float().item()),
                "target_token_count": int(token_counts[0].item()),
            }
            print(
                f"loss {fingerprint_id}={results[fingerprint_id]['loss']:.8f} "
                f"tokens={results[fingerprint_id]['target_token_count']}"
            )
            del batch, labels, output, weights, per_example, token_counts, _loss
    return results


def write_forgetting_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "fingerprint_id",
        "target_token_count",
        "loss_before",
        "loss_after",
        "loss_delta",
        "positive_delta",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def initial_comparison() -> dict[str, Any]:
    return {
        "parent_p3_1_passed": False,
        "proxy_training_completed": False,
        "feedback_valid": False,
        "b1_training_completed": False,
        "p_training_completed": False,
        "b1_adapter_evaluation_passed": False,
        "b1_merged_evaluation_passed": False,
        "b1_adapter_exact_match_rate_by_repeat": None,
        "b1_merged_exact_match_rate_by_repeat": None,
        "p_adapter_evaluation_passed": False,
        "p_merged_evaluation_passed": False,
        "p_adapter_exact_match_rate_by_repeat": None,
        "p_merged_exact_match_rate_by_repeat": None,
        "b1_adapter_and_merged_outputs_identical": False,
        "p_adapter_and_merged_outputs_identical": False,
        "capability_comparison_passed": False,
        "fairness_audit_passed": False,
        "unseen_reserved_accessed": False,
        "forbidden_warning_count": 0,
        "failure_reasons": ["P3-2尚未完成"],
        "p3_2_passed": False,
    }


def initialize_run(run_dir: Path, run_id: str, checked_at: str) -> None:
    for name in (
        "proxy_adapter",
        "b1_adapter",
        "b1_merged_model",
        "b1_evaluation",
        "b1_evaluation/adapter",
        "b1_evaluation/merged",
        "b1_evaluation/capability",
        "p_adapter",
        "p_merged_model",
        "p_evaluation",
        "p_evaluation/adapter",
        "p_evaluation/merged",
        "p_evaluation/capability",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "resolved_config.json",
        {
            "status": "not_run",
            "script_version": SCRIPT_VERSION,
            "run_id": run_id,
            "checked_at": checked_at,
            "stage_id": "P3-2",
            "experiment_tier": "pilot",
            "paper_usage": "方法预实验，不进入论文主结果",
        },
    )
    write_json(run_dir / "parent_p3_1.json", {"status": "not_validated"})
    write_text(run_dir / "proxy_training_metrics.jsonl", "")
    write_json(
        run_dir / "proxy_training_summary.json",
        {"training_completed": False, "errors": []},
    )
    write_text(run_dir / "fingerprint_forgetting_scores.csv", "")
    write_json(run_dir / "fingerprint_forgetting_scores.json", {"scores": []})
    write_json(run_dir / "fingerprint_weights.json", {"weights": []})
    write_json(
        run_dir / "weight_statistics.json",
        {"feedback_valid": False, "status": "not_run"},
    )
    write_text(run_dir / "continuation_training_order.jsonl", "")
    write_json(
        run_dir / "continuation_training_order_sha256.json",
        {"status": "not_run"},
    )
    for branch in ("b1", "p"):
        write_text(run_dir / f"{branch}_training_metrics.jsonl", "")
        write_json(
            run_dir / f"{branch}_training_summary.json",
            {"training_completed": False, "errors": []},
        )
        for mode in ("adapter", "merged"):
            directory = run_dir / f"{branch}_evaluation" / mode
            write_text(directory / "raw_generations.jsonl", "")
            write_scores(directory / "scores.csv", [])
            write_json(
                directory / "metrics.json", initial_evaluation_metrics("positive")
            )
        capability_dir = run_dir / f"{branch}_evaluation" / "capability"
        write_json(capability_dir / "loss.json", {"status": "not_run"})
        write_text(capability_dir / "raw_generations.jsonl", "")
        write_capability_scores(capability_dir / "scores.csv", [], set())
        write_json(capability_dir / "metrics.json", {"status": "not_run"})
        write_json(
            run_dir / f"{branch}_evaluation" / "metrics.json",
            {"status": "not_run"},
        )
    write_json(run_dir / "capability_comparison.json", {"status": "not_run"})
    write_json(run_dir / "fairness_audit.json", {"all_checks_passed": False})
    write_json(run_dir / "comparison.json", initial_comparison())
    write_text(run_dir / "summary.md", "# P3-2摘要\n\n- 状态：尚未开始\n")


def load_weights(path: Path) -> dict[str, float]:
    data = load_json(path)
    rows = data.get("weights")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("fingerprint_weights.json必须包含32个权重")
    weights = {
        str(row["fingerprint_id"]): float(row["weight"]) for row in rows
    }
    if len(weights) != 32:
        raise ValueError("fingerprint_weights.json包含重复ID")
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("fingerprint_weights.json包含无效权重")
    if abs(sum(weights.values()) / 32 - 1.0) >= 1e-6:
        raise ValueError("fingerprint_weights.json权重均值不是1")
    return weights


def render_summary(run_dir: Path, error: dict[str, str] | None = None) -> str:
    resolved = load_json(run_dir / "resolved_config.json")
    parent = load_json(run_dir / "parent_p3_1.json")
    comparison = load_json(run_dir / "comparison.json")
    weight_stats = load_json(run_dir / "weight_statistics.json")
    b1_summary = load_json(run_dir / "b1_training_summary.json")
    p_summary = load_json(run_dir / "p_training_summary.json")
    capability = load_json(run_dir / "capability_comparison.json")
    fairness = load_json(run_dir / "fairness_audit.json")
    b1_adapter = load_json(run_dir / "b1_evaluation/adapter/metrics.json")
    b1_merged = load_json(run_dir / "b1_evaluation/merged/metrics.json")
    p_adapter = load_json(run_dir / "p_evaluation/adapter/metrics.json")
    p_merged = load_json(run_dir / "p_evaluation/merged/metrics.json")
    lines = [
        "# P3-2代理遗忘测量与B1/P公平训练摘要",
        "",
        f"- 运行编号：`{resolved.get('run_id')}`",
        f"- 状态：`{resolved.get('status')}`",
        f"- P3-1父运行：`{parent.get('run_id')}`",
        f"- 模型：`{resolved.get('model_id')}`",
        f"- revision：`{resolved.get('revision')}`",
        "",
        "## 代理反馈",
        "",
        f"- 代理训练完成：`{comparison.get('proxy_training_completed')}`",
        "- loss_delta > 1e-4的指纹数："
        f"`{weight_stats.get('positive_delta_above_threshold_count')}`",
        f"- difficulty标准差：`{weight_stats.get('difficulty_std')}`",
        f"- 权重范围：`{weight_stats.get('weight_min')} - {weight_stats.get('weight_max')}`",
        f"- 权重均值：`{weight_stats.get('weight_mean')}`",
        f"- 反馈有效：`{weight_stats.get('feedback_valid')}`",
        "",
        "## B1与P发布前检查",
        "",
        f"- B1训练完成：`{b1_summary.get('training_completed')}`",
        f"- P训练完成：`{p_summary.get('training_completed')}`",
        f"- B1适配器每轮命中：`{b1_adapter.get('exact_match_count_by_repeat')}`",
        f"- B1合并模型每轮命中：`{b1_merged.get('exact_match_count_by_repeat')}`",
        f"- P适配器每轮命中：`{p_adapter.get('exact_match_count_by_repeat')}`",
        f"- P合并模型每轮命中：`{p_merged.get('exact_match_count_by_repeat')}`",
        f"- P相对B1能力loss变化：`{capability.get('p_vs_b1_relative_loss_change')}`",
        f"- 公平性审计：`{fairness.get('all_checks_passed')}`",
        "",
        f"- P3-2通过：`{comparison.get('p3_2_passed')}`",
    ]
    if comparison.get("failure_reasons"):
        lines.extend(["", "## 未通过原因", ""])
        lines.extend(f"- {reason}" for reason in comparison["failure_reasons"])
    if error:
        lines.extend(["", "## 执行错误", "", f"- {error['type']}：{error['message']}"])
    lines.extend(
        [
            "",
            "> 本阶段没有使用unseen_reserved，也没有执行P3-3。"
            "P3-2的发布前32/32不用于判断P优于B1。",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_context(
    config_path: Path,
    parent_path: Path,
    *,
    require_model_artifacts: bool,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = load_json(config_path)
    validate_p32_config(config)
    parent = resolve_run_directory(parent_path, "p3_1_b0_")
    parent_data = validate_parent_p3_1(
        parent, config, require_model_artifacts=require_model_artifacts
    )
    return {
        "config_path": config_path,
        "config": config,
        "parent": parent,
        **parent_data,
    }


def initialize_resolved_context(
    run_dir: Path,
    context: dict[str, Any],
    candidate_ids: list[str],
) -> dict[str, Any]:
    resolved = load_json(run_dir / "resolved_config.json")
    config = context["config"]
    parent = context["parent"]
    resolved.update(
        {
            "status": "validated",
            "git_commit": git_commit(),
            "model_id": config["model_id"],
            "revision": config["revision"],
            "dtype": config["dtype"],
            "device": config["device"],
            "local_files_only": True,
            "parent_run_ids": [parent.name],
            "parent_p3_1_directory": project_relative(parent),
            "p3_1_candidates": candidate_ids,
            "config_path": project_relative(context["config_path"]),
            "config_sha256": sha256_file(context["config_path"]),
            "config": config,
            "fingerprint_set_id": context["fingerprints"]["fingerprint_set_id"],
            "fingerprint_manifest_sha256": sha256_file(context["fingerprint_path"]),
            "dolly_split_manifest_sha256": sha256_file(
                context["dolly_manifest_path"]
            ),
            "dolly_dataset_revision": context["dolly_manifest"]["dataset_revision"],
            "snapshot_path": str(context["snapshot"]),
            "generation": {
                "enable_thinking": False,
                "do_sample": False,
                "max_new_tokens": 16,
                "repeats": 2,
                "system_prompt": None,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "use_model_defaults": False,
            },
            "unseen_reserved_accessed": False,
        }
    )
    write_json(run_dir / "resolved_config.json", resolved)
    parent_record = {
        "stage_id": "P3-1",
        "run_id": parent.name,
        "source_run_directory": project_relative(parent),
        "p3_1_passed": True,
        "model_id": context["resolved"]["model_id"],
        "model_revision": context["resolved"]["revision"],
        "b0_adapter_exact_match_rate_by_repeat": context["comparison"][
            "b0_adapter_exact_match_rate_by_repeat"
        ],
        "b0_merged_exact_match_rate_by_repeat": context["comparison"][
            "b0_merged_exact_match_rate_by_repeat"
        ],
        "fingerprint_set_id": context["fingerprints"]["fingerprint_set_id"],
        "normal_train_count": context["dolly_manifest"]["splits"]["normal_train"][
            "count"
        ],
        "proxy_reserved_count": context["dolly_manifest"]["splits"][
            "proxy_reserved"
        ]["count"],
        "capability_eval_count": context["dolly_manifest"]["splits"][
            "capability_eval"
        ]["count"],
        "b0_adapter_directory": project_relative(parent / "b0_adapter"),
        "b0_merged_model_directory": project_relative(parent / "b0_merged_model"),
        "unseen_reserved_accessed": False,
    }
    write_json(run_dir / "parent_p3_1.json", parent_record)
    comparison = load_json(run_dir / "comparison.json")
    comparison.update(
        {
            "parent_p3_1_passed": True,
            "failure_reasons": ["P3-2尚未完成"],
        }
    )
    write_json(run_dir / "comparison.json", comparison)
    return resolved


def execute_proxy_score(run_dir: Path, context: dict[str, Any]) -> bool:
    config = context["config"]
    parent = context["parent"]
    validate_cloud_environment(config)
    selected = load_selected_dolly_splits(
        context["dolly_manifest"],
        ("normal_train", "proxy_reserved", "capability_eval"),
    )
    proxy_rows = selected["proxy_reserved"]
    proxy_records = [
        {
            "sample_id": f"proxy:{row['original_index']}:{row['content_sha256'][:12]}",
            "sample_type": "proxy",
            "prompt": row["prompt"],
            "response": row["response"],
            "sample_weight": 1.0,
        }
        for row in proxy_rows
    ]
    proxy_plan = build_proxy_training_plan(
        proxy_records,
        epochs=config["proxy_num_train_epochs"],
        seed=config["proxy_data_seed"],
    )
    accumulation = config["proxy_gradient_accumulation_steps"]
    expected_steps = len(proxy_plan) // accumulation

    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        get_peft_model_state_dict,
    )
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU不支持BF16")
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(
        parent / "b0_merged_model",
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer缺少PAD和EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    proxy_examples = [
        build_completion_example(
            tokenizer,
            sample_id=row["sample_id"],
            sample_type=row["sample_type"],
            prompt=row["prompt"],
            response=row["response"],
            max_sequence_length=config["max_seq_length"],
            sample_weight=1.0,
        )
        for row in proxy_records
    ]
    if len(proxy_examples) != 500 or any(
        example["was_truncated"] for example in proxy_examples
    ):
        raise RuntimeError("代理训练必须包含500条未截断正常指令")

    set_reproducible_seed(torch, config["proxy_seed"])
    model = AutoModelForCausalLM.from_pretrained(
        parent / "b0_merged_model",
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    module_matches = resolve_target_modules(model, config["target_modules"])
    lora_config = build_lora_config(config, LoraConfig)
    model = get_peft_model(model, lora_config)
    parameters = parameter_summary(model)
    print("代理训练LoRA目标模块：")
    for target in config["target_modules"]:
        for full_name in module_matches[target]:
            print(f"- {full_name}")
    summary: dict[str, Any] = {
        **parameters,
        "training_completed": False,
        "seed": config["proxy_seed"],
        "data_seed": config["proxy_data_seed"],
        "normal_example_count": 500,
        "fingerprint_example_count": 0,
        "num_train_epochs": 3,
        "effective_batch_size": config["proxy_per_device_train_batch_size"]
        * accumulation,
        "expected_optimizer_step_count": expected_steps,
        "learning_rate": config["learning_rate"],
        "completion_only_loss": True,
        "actual_target_modules": module_matches,
        "errors": [],
    }
    write_json(run_dir / "proxy_training_summary.json", summary)
    losses, metric_rows = train_weighted_lora(
        torch=torch,
        get_linear_schedule_with_warmup=get_linear_schedule_with_warmup,
        model=model,
        tokenizer=tokenizer,
        examples=proxy_examples,
        plan=proxy_plan,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        maximum_gradient_norm=config["max_grad_norm"],
        accumulation_steps=accumulation,
        logging_steps=config["logging_steps"],
        metrics_path=run_dir / "proxy_training_metrics.jsonl",
        device=device,
        label="proxy",
    )
    summary.update(summarize_training_losses(losses, expected_steps))
    summary["training_metric_record_count"] = len(metric_rows)
    summary["adapter_state_sha256_before_save"] = adapter_state_sha256(
        model, get_peft_model_state_dict
    )
    model.save_pretrained(run_dir / "proxy_adapter", safe_serialization=True)
    tokenizer.save_pretrained(run_dir / "proxy_adapter")
    write_json(run_dir / "proxy_training_summary.json", summary)
    model = None
    release_gpu(torch)

    b0_model = AutoModelForCausalLM.from_pretrained(
        parent / "b0_merged_model",
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    before_losses = measure_fingerprint_target_losses(
        torch=torch,
        model=b0_model,
        tokenizer=tokenizer,
        fingerprints=context["fingerprints"],
        maximum_length=config["max_seq_length"],
        device=device,
    )
    b0_model = None
    release_gpu(torch)

    proxy_base = AutoModelForCausalLM.from_pretrained(
        parent / "b0_merged_model",
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    proxy_model = PeftModel.from_pretrained(
        proxy_base,
        run_dir / "proxy_adapter",
        is_trainable=False,
        local_files_only=True,
    )
    reloaded_hash = adapter_state_sha256(proxy_model, get_peft_model_state_dict)
    summary["adapter_state_sha256_after_reload"] = reloaded_hash
    summary["adapter_reload_state_identical"] = (
        reloaded_hash == summary["adapter_state_sha256_before_save"]
    )
    if not summary["adapter_reload_state_identical"]:
        raise RuntimeError("proxy_adapter保存前后状态SHA256不一致")
    write_json(run_dir / "proxy_training_summary.json", summary)
    after_losses = measure_fingerprint_target_losses(
        torch=torch,
        model=proxy_model,
        tokenizer=tokenizer,
        fingerprints=context["fingerprints"],
        maximum_length=config["max_seq_length"],
        device=device,
    )
    proxy_model = None
    proxy_base = None
    release_gpu(torch)

    scores, weights, statistics = compute_forgetting_feedback(
        before_losses, after_losses, config=config
    )
    write_forgetting_csv(run_dir / "fingerprint_forgetting_scores.csv", scores)
    write_json(
        run_dir / "fingerprint_forgetting_scores.json",
        {
            "schema_version": 1,
            "fingerprint_set_id": context["fingerprints"]["fingerprint_set_id"],
            "loss_definition": "assistant目标代号Token的float32平均交叉熵",
            "scores": scores,
        },
    )
    write_json(
        run_dir / "fingerprint_weights.json",
        {
            "schema_version": 1,
            "fingerprint_set_id": context["fingerprints"]["fingerprint_set_id"],
            "formula": "normalize(1 + 2 * clip(difficulty / (p90_positive + 1e-8), 0, 1))",
            "weights": [
                {
                    "fingerprint_id": row["fingerprint_id"],
                    "weight": weights[row["fingerprint_id"]],
                }
                for row in scores
            ],
        },
    )
    write_json(run_dir / "weight_statistics.json", statistics)
    resolved = load_json(run_dir / "resolved_config.json")
    comparison = load_json(run_dir / "comparison.json")
    comparison.update(
        {
            "proxy_training_completed": summary.get("training_completed", False),
            "feedback_valid": statistics["feedback_valid"],
        }
    )
    if not statistics["feedback_valid"]:
        reasons = [
            name for name, passed in statistics["checks"].items() if not passed
        ]
        comparison["failure_reasons"] = [
            "遗忘反馈非退化检查失败：" + ", ".join(reasons)
        ]
        resolved["status"] = "feedback_invalid"
        write_json(run_dir / "resolved_config.json", resolved)
        write_json(run_dir / "comparison.json", comparison)
        write_text(run_dir / "summary.md", render_summary(run_dir))
        print("P3-2遗忘反馈无效，已按要求停止，不训练B1/P。", file=sys.stderr)
        return False

    unit_weights = {
        row["fingerprint_id"]: 1.0
        for row in context["fingerprints"]["fingerprints"]
    }
    training_records = build_branch_training_records(
        selected["normal_train"],
        context["fingerprints"],
        fingerprint_repeat=config["fingerprint_repeat"],
        fingerprint_weights=unit_weights,
    )
    effective_batch = config["continuation_per_device_train_batch_size"] * config[
        "continuation_gradient_accumulation_steps"
    ]
    plan, order_sha = build_continuation_training_plan(
        training_records,
        seed=config["continuation_data_seed"],
        max_steps=config["continuation_max_steps"],
        effective_batch_size=effective_batch,
    )
    write_text(run_dir / "continuation_training_order.jsonl", training_plan_text(plan))
    records_sha = training_records_sha256(training_records)
    write_json(
        run_dir / "continuation_training_order_sha256.json",
        {
            "continuation_data_seed": config["continuation_data_seed"],
            "training_pool_record_count": len(training_records),
            "consumed_training_record_count": len(plan),
            "max_steps": config["continuation_max_steps"],
            "effective_batch_size": effective_batch,
            "training_examples_sha256": records_sha,
            "training_order_sha256": order_sha,
            "b1_and_p_must_read_this_same_file": True,
        },
    )
    resolved["status"] = "feedback_ready"
    resolved["continuation_training_examples_sha256"] = records_sha
    resolved["continuation_training_order_sha256"] = order_sha
    comparison["failure_reasons"] = ["B1/P尚未训练"]
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "comparison.json", comparison)
    write_text(run_dir / "summary.md", render_summary(run_dir))
    print("P3-2代理反馈通过，已生成固定B1/P共同训练顺序。")
    return True


def load_continuation_plan(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def train_continuation_branch(
    *,
    branch: str,
    run_dir: Path,
    context: dict[str, Any],
    normal_rows: list[dict[str, Any]],
    fingerprint_weights: dict[str, float],
    plan: list[dict[str, Any]],
    order_info: dict[str, Any],
    torch: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    PeftModel: Any,
    get_peft_model_state_dict: Any,
    get_linear_schedule_with_warmup: Any,
) -> dict[str, Any]:
    config = context["config"]
    parent = context["parent"]
    device = config["device"]
    records = build_branch_training_records(
        normal_rows,
        context["fingerprints"],
        fingerprint_repeat=config["fingerprint_repeat"],
        fingerprint_weights=fingerprint_weights,
    )
    examples_sha = training_records_sha256(records)
    if examples_sha != order_info["training_examples_sha256"]:
        raise RuntimeError(f"{branch}训练数据身份SHA256与固定顺序不一致")
    effective_batch = config["continuation_per_device_train_batch_size"] * config[
        "continuation_gradient_accumulation_steps"
    ]
    validate_continuation_training_plan(
        plan,
        records,
        expected_sha256=order_info["training_order_sha256"],
        max_steps=config["continuation_max_steps"],
        effective_batch_size=effective_batch,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        parent / "b0_adapter",
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer缺少PAD和EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    examples = [
        build_completion_example(
            tokenizer,
            sample_id=row["sample_id"],
            sample_type=row["sample_type"],
            prompt=row["prompt"],
            response=row["response"],
            max_sequence_length=config["max_seq_length"],
            sample_weight=row["sample_weight"],
        )
        for row in records
    ]
    if len(examples) != 1256 or any(row["was_truncated"] for row in examples):
        raise RuntimeError(f"{branch}训练数据数量或截断检查失败")

    set_reproducible_seed(torch, config["continuation_seed"])
    base = AutoModelForCausalLM.from_pretrained(
        context["snapshot"],
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(
        base,
        parent / "b0_adapter",
        is_trainable=True,
        local_files_only=True,
    )
    if set(model.peft_config) != {"default"}:
        raise RuntimeError(
            f"{branch}必须继续更新P3-1的单一default适配器，"
            "不能创建增量适配器"
        )
    parameters = parameter_summary(model)
    start_hash = adapter_state_sha256(model, get_peft_model_state_dict)
    lora_snapshot = {
        "r": config["lora_r"],
        "alpha": config["lora_alpha"],
        "dropout": config["lora_dropout"],
        "bias": config["lora_bias"],
        "task_type": config["task_type"],
        "target_modules": config["target_modules"],
    }
    summary: dict[str, Any] = {
        **parameters,
        "branch": branch,
        "training_completed": False,
        "parent_p3_1_run_id": parent.name,
        "starting_b0_adapter_state_sha256": start_hash,
        "continued_existing_adapter_name": "default",
        "saved_checkpoint_semantics": "P3-1 B0完整适配器状态加80步原位更新",
        "seed": config["continuation_seed"],
        "data_seed": config["continuation_data_seed"],
        "normal_example_count": 1000,
        "fingerprint_unique_count": 32,
        "fingerprint_repeat": 8,
        "fingerprint_repeated_count": 256,
        "training_pool_record_count": 1256,
        "consumed_training_record_count": len(plan),
        "max_steps": config["continuation_max_steps"],
        "effective_batch_size": effective_batch,
        "learning_rate": config["learning_rate"],
        "normal_example_weight": 1.0,
        "mean_fingerprint_weight": sum(fingerprint_weights.values()) / 32,
        "fingerprint_weight_mapping": fingerprint_weights,
        "lora_config": lora_snapshot,
        "training_examples_sha256": examples_sha,
        "training_order_sha256": order_info["training_order_sha256"],
        "completion_only_per_example_weighted_loss": True,
        "errors": [],
    }
    write_json(run_dir / f"{branch}_training_summary.json", summary)
    losses, metric_rows = train_weighted_lora(
        torch=torch,
        get_linear_schedule_with_warmup=get_linear_schedule_with_warmup,
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        plan=plan,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        maximum_gradient_norm=config["max_grad_norm"],
        accumulation_steps=config["continuation_gradient_accumulation_steps"],
        logging_steps=config["logging_steps"],
        metrics_path=run_dir / f"{branch}_training_metrics.jsonl",
        device=device,
        label=branch,
    )
    summary.update(
        summarize_training_losses(losses, config["continuation_max_steps"])
    )
    summary["training_metric_record_count"] = len(metric_rows)
    summary["adapter_state_sha256_before_save"] = adapter_state_sha256(
        model, get_peft_model_state_dict
    )
    summary["saved_adapter_contains_b0_and_continuation_state"] = True
    model.save_pretrained(run_dir / f"{branch}_adapter", safe_serialization=True)
    tokenizer.save_pretrained(run_dir / f"{branch}_adapter")
    write_json(run_dir / f"{branch}_training_summary.json", summary)
    model = None
    base = None
    tokenizer = None
    release_gpu(torch)
    return summary


def execute_train_branches(run_dir: Path, context: dict[str, Any]) -> bool:
    resolved = load_json(run_dir / "resolved_config.json")
    if resolved.get("status") != "feedback_ready":
        raise RuntimeError("当前P3-2目录尚未获得有效反馈，禁止训练B1/P")
    config = context["config"]
    if resolved.get("config_sha256") != sha256_file(context["config_path"]):
        raise RuntimeError("P3-2配置在代理反馈后发生变化")
    if resolved.get("parent_run_ids") != [context["parent"].name]:
        raise RuntimeError("P3-1父运行在代理反馈后发生变化")
    selected = load_selected_dolly_splits(
        context["dolly_manifest"], ("normal_train",)
    )
    p_weights = load_weights(run_dir / "fingerprint_weights.json")
    b1_weights = {
        row["fingerprint_id"]: 1.0
        for row in context["fingerprints"]["fingerprints"]
    }
    order_info = load_json(run_dir / "continuation_training_order_sha256.json")
    if (
        order_info.get("training_order_sha256")
        != resolved.get("continuation_training_order_sha256")
        or order_info.get("training_examples_sha256")
        != resolved.get("continuation_training_examples_sha256")
    ):
        raise RuntimeError("共同训练顺序或数据身份与resolved_config不一致")
    plan = load_continuation_plan(run_dir / "continuation_training_order.jsonl")

    from peft import PeftModel, get_peft_model_state_dict
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU不支持BF16")
    b1_summary = train_continuation_branch(
        branch="b1",
        run_dir=run_dir,
        context=context,
        normal_rows=selected["normal_train"],
        fingerprint_weights=b1_weights,
        plan=plan,
        order_info=order_info,
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        PeftModel=PeftModel,
        get_peft_model_state_dict=get_peft_model_state_dict,
        get_linear_schedule_with_warmup=get_linear_schedule_with_warmup,
    )
    p_summary = train_continuation_branch(
        branch="p",
        run_dir=run_dir,
        context=context,
        normal_rows=selected["normal_train"],
        fingerprint_weights=p_weights,
        plan=plan,
        order_info=order_info,
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        PeftModel=PeftModel,
        get_peft_model_state_dict=get_peft_model_state_dict,
        get_linear_schedule_with_warmup=get_linear_schedule_with_warmup,
    )
    if (
        b1_summary["starting_b0_adapter_state_sha256"]
        != p_summary["starting_b0_adapter_state_sha256"]
    ):
        raise RuntimeError("B1与P的B0起点状态SHA256不一致")
    fairness = compute_fairness_audit(
        parent_run_id_b1=b1_summary["parent_p3_1_run_id"],
        parent_run_id_p=p_summary["parent_p3_1_run_id"],
        revision_b1=config["revision"],
        revision_p=config["revision"],
        training_example_sha_b1=b1_summary["training_examples_sha256"],
        training_example_sha_p=p_summary["training_examples_sha256"],
        order_sha_b1=b1_summary["training_order_sha256"],
        order_sha_p=p_summary["training_order_sha256"],
        b1_summary=b1_summary,
        p_summary=p_summary,
        b1_weights=b1_weights,
        p_weights=p_weights,
    )
    fairness["same_starting_b0_adapter_state_sha256"] = (
        b1_summary["starting_b0_adapter_state_sha256"]
        == p_summary["starting_b0_adapter_state_sha256"]
    )
    fairness["all_checks_passed"] = (
        fairness["all_checks_passed"]
        and fairness["same_starting_b0_adapter_state_sha256"]
    )
    write_json(run_dir / "fairness_audit.json", fairness)
    if not fairness["all_checks_passed"]:
        raise RuntimeError("B1/P训练公平性审计失败")
    comparison = load_json(run_dir / "comparison.json")
    comparison.update(
        {
            "b1_training_completed": b1_summary.get("training_completed", False),
            "p_training_completed": p_summary.get("training_completed", False),
            "fairness_audit_passed": True,
            "failure_reasons": ["B1/P发布前评估尚未执行"],
        }
    )
    resolved["status"] = "branches_trained"
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "comparison.json", comparison)
    write_text(run_dir / "summary.md", render_summary(run_dir))
    print("P3-2 B1与P已从相同B0起点完成80步公平继续训练。")
    return True


def evaluate_branch(
    *,
    branch: str,
    run_dir: Path,
    context: dict[str, Any],
    capability_rows: list[dict[str, Any]],
    torch: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    PeftModel: Any,
    get_peft_model_state_dict: Any,
) -> dict[str, Any]:
    config = context["config"]
    device = config["device"]
    training_summary = load_json(run_dir / f"{branch}_training_summary.json")
    tokenizer = AutoTokenizer.from_pretrained(
        run_dir / f"{branch}_adapter",
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer缺少PAD和EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        context["snapshot"],
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    adapter_model = PeftModel.from_pretrained(
        base,
        run_dir / f"{branch}_adapter",
        is_trainable=False,
        local_files_only=True,
    )
    reload_hash = adapter_state_sha256(adapter_model, get_peft_model_state_dict)
    state_identical = (
        reload_hash == training_summary["adapter_state_sha256_before_save"]
    )
    if not state_identical:
        raise RuntimeError(f"{branch}适配器保存前后状态SHA256不一致")
    adapter_records, adapter_metrics = evaluate_fingerprints(
        torch=torch,
        model=adapter_model,
        tokenizer=tokenizer,
        manifest=context["fingerprints"],
        repeats=config["evaluation_repeats"],
        maximum_new_tokens=config["evaluation_max_new_tokens"],
        device=device,
        output_dir=run_dir / f"{branch}_evaluation" / "adapter",
        expected_mode="positive",
    )
    if adapter_metrics.get("evaluation_passed") is not True:
        raise RuntimeError(f"{branch}适配器没有在两轮中达到32/32")
    merged = adapter_model.merge_and_unload(safe_merge=True)
    merged.generation_config = clean_generation_config(
        merged, config["evaluation_max_new_tokens"]
    )
    merged_dir = run_dir / f"{branch}_merged_model"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    adapter_model = None
    base = None
    merged = None
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
    merged_records, merged_metrics = evaluate_fingerprints(
        torch=torch,
        model=reloaded_merged,
        tokenizer=merged_tokenizer,
        manifest=context["fingerprints"],
        repeats=config["evaluation_repeats"],
        maximum_new_tokens=config["evaluation_max_new_tokens"],
        device=device,
        output_dir=run_dir / f"{branch}_evaluation" / "merged",
        expected_mode="positive",
    )
    if merged_metrics.get("evaluation_passed") is not True:
        raise RuntimeError(f"{branch}合并模型没有在两轮中达到32/32")
    capability_loss = average_completion_loss(
        torch=torch,
        model=reloaded_merged,
        tokenizer=merged_tokenizer,
        records=capability_rows,
        maximum_length=config["max_seq_length"],
        device=device,
    )
    generations = generate_capability_samples(
        torch=torch,
        model=reloaded_merged,
        tokenizer=merged_tokenizer,
        records=capability_rows,
        maximum_new_tokens=config["evaluation_max_new_tokens"],
        device=device,
    )
    known_codes = set(P1_RESPONSES) | set(context["fingerprints"]["allowed_responses"])
    capability_dir = run_dir / f"{branch}_evaluation" / "capability"
    write_json(capability_dir / "loss.json", capability_loss)
    write_jsonl(capability_dir / "raw_generations.jsonl", generations)
    write_capability_scores(capability_dir / "scores.csv", generations, known_codes)
    branch_metrics = {
        "branch": branch,
        "adapter_reload_state_sha256": reload_hash,
        "adapter_reload_state_identical": state_identical,
        "adapter_evaluation_passed": adapter_metrics["evaluation_passed"],
        "merged_evaluation_passed": merged_metrics["evaluation_passed"],
        "adapter_and_merged_outputs_identical": outputs_identical_between_models(
            adapter_records, merged_records
        ),
        "average_completion_token_loss": capability_loss[
            "average_completion_token_loss"
        ],
        "thinking_tag_count": int(adapter_metrics["thinking_tag_count"])
        + int(merged_metrics["thinking_tag_count"])
        + sum(
            "<think>" in row["raw_output"] or "</think>" in row["raw_output"]
            for row in generations
        ),
    }
    write_json(capability_dir / "metrics.json", branch_metrics)
    write_json(run_dir / f"{branch}_evaluation" / "metrics.json", branch_metrics)
    reloaded_merged = None
    merged_tokenizer = None
    tokenizer = None
    release_gpu(torch)
    return {
        "adapter_records": adapter_records,
        "adapter_metrics": adapter_metrics,
        "merged_records": merged_records,
        "merged_metrics": merged_metrics,
        "capability_loss": capability_loss,
        "generations": generations,
        "branch_metrics": branch_metrics,
    }


def execute_evaluate(run_dir: Path, context: dict[str, Any]) -> bool:
    resolved = load_json(run_dir / "resolved_config.json")
    if resolved.get("status") != "branches_trained":
        raise RuntimeError("当前P3-2目录尚未完成B1/P训练，禁止评估")
    config = context["config"]
    if resolved.get("config_sha256") != sha256_file(context["config_path"]):
        raise RuntimeError("P3-2配置在分支训练后发生变化")
    selected = load_selected_dolly_splits(
        context["dolly_manifest"], ("capability_eval",)
    )
    from peft import PeftModel, get_peft_model_state_dict
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU不支持BF16")
    b1_result = evaluate_branch(
        branch="b1",
        run_dir=run_dir,
        context=context,
        capability_rows=selected["capability_eval"],
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        PeftModel=PeftModel,
        get_peft_model_state_dict=get_peft_model_state_dict,
    )
    p_result = evaluate_branch(
        branch="p",
        run_dir=run_dir,
        context=context,
        capability_rows=selected["capability_eval"],
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        PeftModel=PeftModel,
        get_peft_model_state_dict=get_peft_model_state_dict,
    )
    known_codes = set(P1_RESPONSES) | set(context["fingerprints"]["allowed_responses"])
    capability = compute_capability_comparison(
        b1_loss=float(
            b1_result["capability_loss"]["average_completion_token_loss"]
        ),
        p_loss=float(p_result["capability_loss"]["average_completion_token_loss"]),
        b1_generations=b1_result["generations"],
        p_generations=p_result["generations"],
        known_codes=known_codes,
        maximum_relative_increase=config["maximum_p_vs_b1_loss_increase"],
    )
    write_json(run_dir / "capability_comparison.json", capability)
    fairness = load_json(run_dir / "fairness_audit.json")
    b1_summary = load_json(run_dir / "b1_training_summary.json")
    p_summary = load_json(run_dir / "p_training_summary.json")
    proxy_summary = load_json(run_dir / "proxy_training_summary.json")
    weight_statistics = load_json(run_dir / "weight_statistics.json")
    terminal_text = (run_dir / "terminal_output.log").read_text(encoding="utf-8")
    warning_count = sum(terminal_text.count(value) for value in FORBIDDEN_WARNINGS)
    thinking_count = (
        b1_result["branch_metrics"]["thinking_tag_count"]
        + p_result["branch_metrics"]["thinking_tag_count"]
    )
    checks = [
        (
            proxy_summary.get("training_completed") is True,
            "代理训练未完成",
        ),
        (
            proxy_summary.get("all_losses_finite") is True,
            "代理训练loss出现NaN或Inf",
        ),
        (
            proxy_summary.get("only_lora_parameters_trainable") is True,
            "代理训练存在非LoRA可训练参数",
        ),
        (
            proxy_summary.get("adapter_reload_state_identical") is True,
            "代理适配器保存重载状态不一致",
        ),
        (
            weight_statistics.get("feedback_valid") is True,
            "遗忘反馈未通过非退化检查",
        ),
        (b1_summary.get("training_completed") is True, "B1训练未完成"),
        (p_summary.get("training_completed") is True, "P训练未完成"),
        (
            b1_summary.get("all_losses_finite") is True
            and b1_summary.get("only_lora_parameters_trainable") is True,
            "B1训练loss或可训练参数检查失败",
        ),
        (
            p_summary.get("all_losses_finite") is True
            and p_summary.get("only_lora_parameters_trainable") is True,
            "P训练loss或可训练参数检查失败",
        ),
        (
            b1_result["branch_metrics"]["adapter_reload_state_identical"] is True,
            "B1适配器保存重载状态不一致",
        ),
        (
            p_result["branch_metrics"]["adapter_reload_state_identical"] is True,
            "P适配器保存重载状态不一致",
        ),
        (
            b1_result["adapter_metrics"].get("evaluation_passed") is True,
            "B1适配器评估未通过",
        ),
        (
            b1_result["merged_metrics"].get("evaluation_passed") is True,
            "B1合并模型评估未通过",
        ),
        (
            p_result["adapter_metrics"].get("evaluation_passed") is True,
            "P适配器评估未通过",
        ),
        (
            p_result["merged_metrics"].get("evaluation_passed") is True,
            "P合并模型评估未通过",
        ),
        (
            b1_result["branch_metrics"]["adapter_and_merged_outputs_identical"]
            is True,
            "B1适配器与合并模型输出不一致",
        ),
        (
            p_result["branch_metrics"]["adapter_and_merged_outputs_identical"]
            is True,
            "P适配器与合并模型输出不一致",
        ),
        (
            capability.get("capability_comparison_passed") is True,
            "P相对B1能力检查未通过",
        ),
        (fairness.get("all_checks_passed") is True, "公平性审计未通过"),
        (thinking_count == 0, "输出包含思考标签"),
        (warning_count == 0, "日志包含弃用或采样配置警告"),
    ]
    failure_reasons = [reason for passed, reason in checks if not passed]
    comparison = load_json(run_dir / "comparison.json")
    comparison.update(
        {
            "b1_adapter_evaluation_passed": b1_result["adapter_metrics"][
                "evaluation_passed"
            ],
            "b1_merged_evaluation_passed": b1_result["merged_metrics"][
                "evaluation_passed"
            ],
            "b1_adapter_exact_match_rate_by_repeat": b1_result[
                "adapter_metrics"
            ]["exact_match_rate_by_repeat"],
            "b1_merged_exact_match_rate_by_repeat": b1_result[
                "merged_metrics"
            ]["exact_match_rate_by_repeat"],
            "p_adapter_evaluation_passed": p_result["adapter_metrics"][
                "evaluation_passed"
            ],
            "p_merged_evaluation_passed": p_result["merged_metrics"][
                "evaluation_passed"
            ],
            "p_adapter_exact_match_rate_by_repeat": p_result["adapter_metrics"][
                "exact_match_rate_by_repeat"
            ],
            "p_merged_exact_match_rate_by_repeat": p_result["merged_metrics"][
                "exact_match_rate_by_repeat"
            ],
            "b1_adapter_and_merged_outputs_identical": b1_result[
                "branch_metrics"
            ]["adapter_and_merged_outputs_identical"],
            "p_adapter_and_merged_outputs_identical": p_result["branch_metrics"][
                "adapter_and_merged_outputs_identical"
            ],
            "capability_comparison_passed": capability[
                "capability_comparison_passed"
            ],
            "fairness_audit_passed": fairness["all_checks_passed"],
            "thinking_tag_count": thinking_count,
            "forbidden_warning_count": warning_count,
            "failure_reasons": failure_reasons,
            "p3_2_passed": not failure_reasons,
        }
    )
    resolved["status"] = "completed"
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "comparison.json", comparison)
    write_text(run_dir / "summary.md", render_summary(run_dir))
    print(f"P3-2完成：{'通过' if comparison['p3_2_passed'] else '未通过'}")
    return bool(comparison["p3_2_passed"])


def update_failure(
    run_dir: Path,
    stage: str,
    exc: Exception,
) -> None:
    error = {"type": type(exc).__name__, "message": sanitize_text(exc)}
    resolved = load_json(run_dir / "resolved_config.json")
    resolved["status"] = f"{stage.replace('-', '_')}_failed"
    resolved["error"] = error
    comparison = load_json(run_dir / "comparison.json")
    comparison["failure_reasons"] = [f"{error['type']}：{error['message']}"]
    comparison["p3_2_passed"] = False
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "comparison.json", comparison)
    if stage == "proxy-score":
        summary = load_json(run_dir / "proxy_training_summary.json")
        summary.setdefault("errors", []).append(error)
        write_json(run_dir / "proxy_training_summary.json", summary)
    elif stage == "train-branches":
        for branch in ("b1", "p"):
            summary = load_json(run_dir / f"{branch}_training_summary.json")
            if summary.get("training_completed") is not True:
                summary.setdefault("errors", []).append(error)
                write_json(run_dir / f"{branch}_training_summary.json", summary)
    write_text(run_dir / "summary.md", render_summary(run_dir, error))
    print(f"P3-2 {stage}失败：{error['type']}：{error['message']}", file=sys.stderr)
    print(f"输出目录：{run_dir}")


def context_for_existing_run(
    args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    resolved = load_json(run_dir / "resolved_config.json")
    parent_value = resolved.get("parent_p3_1_directory")
    if not isinstance(parent_value, str) or not parent_value:
        raise RuntimeError("P3-2 resolved_config缺少P3-1父目录")
    stored_parent = Path(parent_value)
    if not stored_parent.is_absolute():
        stored_parent = PROJECT_ROOT / stored_parent
    if args.p3_1_run is not None:
        explicit = resolve_run_directory(args.p3_1_run, "p3_1_b0_")
        if explicit != stored_parent.resolve():
            raise RuntimeError("命令指定的P3-1父运行与当前P3-2目录不一致")
    context = prepare_context(
        args.config, stored_parent, require_model_artifacts=True
    )
    if resolved.get("config_sha256") != sha256_file(context["config_path"]):
        raise RuntimeError("P3-2配置与创建运行目录时不一致")
    return context


def main() -> int:
    args = parse_args()
    creating = args.stage in {"proxy-score", "all"}
    if creating:
        if args.run_dir is not None:
            print(f"{args.stage}不能传入--run-dir", file=sys.stderr)
            return 3
        checked_at = datetime.now().astimezone()
        try:
            run_id, run_dir = create_unique_p32_run_directory(
                PROJECT_ROOT / "runs", checked_at
            )
            initialize_run(run_dir, run_id, checked_at.isoformat(timespec="seconds"))
        except Exception as exc:
            print(f"无法创建P3-2运行目录：{sanitize_text(exc)}", file=sys.stderr)
            return 3
        log_mode = "w"
    else:
        if args.run_dir is None:
            print(f"{args.stage}必须传入--run-dir", file=sys.stderr)
            return 3
        try:
            run_dir = resolve_run_directory(args.run_dir, "p3_2_feedback_")
        except Exception as exc:
            print(f"无效P3-2运行目录：{sanitize_text(exc)}", file=sys.stderr)
            return 3
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
            print(f"P3-2阶段：{args.stage}")
            print(f"终端日志：{terminal_path}")
            current_stage = "proxy-score" if creating else args.stage
            try:
                if creating:
                    config = load_json(args.config.expanduser().resolve())
                    validate_p32_config(config)
                    parent, candidate_ids = select_parent_p3_1(args.p3_1_run)
                    context = prepare_context(
                        args.config, parent, require_model_artifacts=True
                    )
                    initialize_resolved_context(run_dir, context, candidate_ids)
                    feedback_ready = execute_proxy_score(run_dir, context)
                    if not feedback_ready:
                        print(f"输出目录：{run_dir}")
                        return 2
                    if args.stage == "proxy-score":
                        print(f"输出目录：{run_dir}")
                        return 0
                    current_stage = "train-branches"
                    branches_ready = execute_train_branches(run_dir, context)
                    if not branches_ready:
                        print(f"输出目录：{run_dir}")
                        return 2
                    current_stage = "evaluate"
                    passed = execute_evaluate(run_dir, context)
                    print(f"输出目录：{run_dir}")
                    return 0 if passed else 2

                context = context_for_existing_run(args, run_dir)
                if args.stage == "train-branches":
                    ready = execute_train_branches(run_dir, context)
                    print(f"输出目录：{run_dir}")
                    return 0 if ready else 2
                passed = execute_evaluate(run_dir, context)
                print(f"输出目录：{run_dir}")
                return 0 if passed else 2
            except Exception as exc:
                update_failure(run_dir, current_stage, exc)
                return 3


if __name__ == "__main__":
    raise SystemExit(main())
