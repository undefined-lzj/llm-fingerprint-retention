#!/usr/bin/env python3
"""P3-3：用同一份未见Dolly数据公平攻击B1/P并比较指纹保持率。"""

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
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any


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
    records_sha256,
    sha256_file,
)
from fingerprint.p32 import (  # noqa: E402
    EXPECTED_MODEL_ID,
    EXPECTED_REVISION,
    compute_capability_comparison,
    weighted_completion_loss,
)
from fingerprint.p33 import (  # noqa: E402
    ATTACK_STEPS,
    CAPABILITY_STEPS,
    EVALUATION_STEPS,
    PARENT_P3_1_RUN_ID,
    PARENT_P3_2_RUN_ID,
    PARENT_P3_3_RUN_ID,
    audit_dolly_splits,
    build_unseen_training_plan,
    build_unseen_training_records,
    compute_attack_fairness_audit,
    compute_capability_curve,
    compute_g2_precheck,
    compute_retention_outputs,
    create_unique_p33_run_directory,
    create_unique_p33r_run_directory,
    classify_p33r_result,
    training_plan_sha256,
    training_plan_text,
    training_records_sha256,
    validate_p33_config,
    validate_p33r_config,
    validate_unseen_training_plan,
)
from p0b_common import sanitize_text, write_json, write_text  # noqa: E402
from run_p3_1_b0 import (  # noqa: E402
    FORBIDDEN_WARNING_FRAGMENTS,
    TeeStream,
    average_completion_loss,
    evaluate_fingerprints,
    generate_capability_samples,
    release_gpu,
    write_capability_scores,
    write_scores,
)
from run_p3_2_feedback import (  # noqa: E402
    adapter_state_sha256,
    build_lora_config,
    measure_fingerprint_target_losses,
    set_reproducible_seed,
)


SCRIPT_VERSION = "p3-3-unseen-1.1.0"
DATA_ROOT = Path("/root/autodl-tmp")
STAGES = ("prepare", "train", "evaluate", "all")
METHODS = ("b1", "p")
PARENT_REQUIRED_FILES = (
    "resolved_config.json",
    "parent_p3_1.json",
    "comparison.json",
    "fairness_audit.json",
    "capability_comparison.json",
    "fingerprint_weights.json",
    "b1_evaluation/merged/metrics.json",
    "p_evaluation/merged/metrics.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "p3_3_unseen.json",
    )
    parser.add_argument(
        "--p3-2-run",
        type=Path,
        default=PROJECT_ROOT / "runs" / PARENT_P3_2_RUN_ID,
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}顶层必须是JSON对象")
    return value


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in records
        ),
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def project_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def resolve_run_directory(path: Path, prefix: str) -> Path:
    resolved = path.expanduser().resolve()
    runs_root = (PROJECT_ROOT / "runs").resolve()
    if not resolved.is_relative_to(runs_root) or not resolved.name.startswith(prefix):
        raise ValueError(f"运行目录必须位于本项目runs/下且以{prefix}开头")
    if not resolved.is_dir():
        raise ValueError(f"运行目录不存在：{resolved}")
    return resolved


def load_weights(path: Path) -> dict[str, float]:
    rows = load_json(path).get("weights")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("P3-2权重文件必须包含32条记录")
    weights = {str(row["fingerprint_id"]): float(row["weight"]) for row in rows}
    if len(weights) != 32:
        raise ValueError("P3-2权重文件包含重复指纹ID")
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("P3-2权重必须全部为有限正数")
    if abs(sum(weights.values()) / 32 - 1.0) >= 1e-6:
        raise ValueError("P3-2权重均值不是1")
    return weights


def validate_parent_p3_2(
    parent: Path,
    config: dict[str, Any],
    *,
    require_model_artifacts: bool,
) -> dict[str, Any]:
    parent = resolve_run_directory(parent, "p3_2_feedback_")
    if parent.name != PARENT_P3_2_RUN_ID:
        raise RuntimeError(f"P3-3只允许固定父运行{PARENT_P3_2_RUN_ID}")
    for relative in PARENT_REQUIRED_FILES:
        if not (parent / relative).is_file():
            raise RuntimeError(f"P3-2父运行缺少文件：{relative}")
    resolved = load_json(parent / "resolved_config.json")
    comparison = load_json(parent / "comparison.json")
    fairness = load_json(parent / "fairness_audit.json")
    capability = load_json(parent / "capability_comparison.json")
    parent_p3_1 = load_json(parent / "parent_p3_1.json")
    if resolved.get("status") != "completed" or comparison.get("p3_2_passed") is not True:
        raise RuntimeError("固定P3-2父运行没有通过")
    if resolved.get("run_id") != parent.name:
        raise RuntimeError("P3-2 resolved_config中的run_id与目录名不一致")
    if resolved.get("model_id") != EXPECTED_MODEL_ID or resolved.get("revision") != EXPECTED_REVISION:
        raise RuntimeError("P3-2模型ID或revision不正确")
    if resolved.get("unseen_reserved_accessed") is not False or comparison.get("unseen_reserved_accessed") is not False:
        raise RuntimeError("P3-2已经访问过unseen_reserved，不能作为本阶段父运行")
    if parent_p3_1.get("run_id") != PARENT_P3_1_RUN_ID or parent_p3_1.get("p3_1_passed") is not True:
        raise RuntimeError("P3-2引用的P3-1父运行不正确")
    if fairness.get("all_checks_passed") is not True:
        raise RuntimeError("P3-2公平性审计未通过")
    if capability.get("capability_comparison_passed") is not True:
        raise RuntimeError("P3-2正常能力检查未通过")
    if float(capability.get("p_vs_b1_relative_loss_change", math.inf)) > config["maximum_p_vs_b1_loss_increase"]:
        raise RuntimeError("P3-2中P相对B1的正常能力差异超过5%")
    parent_config = resolved.get("config")
    if not isinstance(parent_config, dict):
        raise RuntimeError("P3-2 resolved_config缺少配置快照")
    inherited_proxy_fields = {
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "lora_bias": "lora_bias",
        "task_type": "task_type",
        "target_modules": "target_modules",
        "max_seq_length": "max_seq_length",
        "bf16": "bf16",
        "gradient_checkpointing": "gradient_checkpointing",
        "proxy_seed": "seed",
        "proxy_per_device_train_batch_size": "per_device_train_batch_size",
        "proxy_gradient_accumulation_steps": "gradient_accumulation_steps",
    }
    for parent_field, p33_field in inherited_proxy_fields.items():
        if parent_config.get(parent_field) != config.get(p33_field):
            raise RuntimeError(
                f"P3-3字段{p33_field}没有继承P3-2代理配置{parent_field}"
            )
    expected_parent_learning_rate = (
        config["original_learning_rate"]
        if config.get("stage_id") == "P3-3R"
        else config["learning_rate"]
    )
    if parent_config.get("learning_rate") != expected_parent_learning_rate:
        raise RuntimeError("P3-2代理微调学习率与原P3-3配置不一致")
    for method in METHODS:
        metrics = load_json(parent / f"{method}_evaluation" / "merged" / "metrics.json")
        if metrics.get("exact_match_count_by_repeat") != {"1": 32, "2": 32} or metrics.get("evaluation_passed") is not True:
            raise RuntimeError(f"P3-2 {method.upper()}发布模型不是两轮32/32")
    p3_1_path = PROJECT_ROOT / "runs" / PARENT_P3_1_RUN_ID
    if not p3_1_path.is_dir():
        raise RuntimeError("找不到固定P3-1父运行")
    fingerprint_path = p3_1_path / "fingerprint_manifest.json"
    fingerprint_sha = PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.sha256"
    fingerprints = load_p3_fingerprint_manifest(fingerprint_path)
    frozen_sha_parts = fingerprint_sha.read_text(encoding="utf-8").strip().split()
    if len(frozen_sha_parts) != 2 or sha256_file(fingerprint_path) != frozen_sha_parts[0]:
        raise RuntimeError("P3-1父运行的指纹清单与冻结SHA256不一致")
    dolly_manifest_path = p3_1_path / "dolly_split_manifest.json"
    dolly_manifest = load_json(dolly_manifest_path)
    expected_counts = {
        "normal_train": 1000,
        "proxy_reserved": 500,
        "unseen_reserved": 1000,
        "capability_eval": 100,
    }
    for name, count in expected_counts.items():
        if dolly_manifest.get("splits", {}).get(name, {}).get("count") != count:
            raise RuntimeError(f"冻结Dolly划分{name}数量不正确")
    intersection_check = dolly_manifest.get("intersection_check")
    if (
        dolly_manifest.get("all_intersections_empty") is not True
        or not isinstance(intersection_check, dict)
        or any(value != 0 for value in intersection_check.values())
    ):
        raise RuntimeError("P3-1冻结清单没有证明四个Dolly集合互不重叠")
    weights = load_weights(parent / "fingerprint_weights.json")
    fingerprint_ids = {
        row["fingerprint_id"] for row in fingerprints["fingerprints"]
    }
    if set(weights) != fingerprint_ids:
        raise RuntimeError("P3-2权重与32条固定指纹不对应")
    model_paths = {
        method: parent / f"{method}_merged_model" for method in METHODS
    }
    if require_model_artifacts:
        for method, model_path in model_paths.items():
            if not (model_path / "model.safetensors").is_file():
                raise RuntimeError(f"云端缺少{method.upper()}合并模型权重：{model_path}")
            if not (model_path / "config.json").is_file():
                raise RuntimeError(f"云端缺少{method.upper()}合并模型配置")
    return {
        "parent": parent,
        "resolved": resolved,
        "comparison": comparison,
        "fairness": fairness,
        "capability": capability,
        "parent_p3_1": parent_p3_1,
        "p3_1_path": p3_1_path,
        "fingerprint_path": fingerprint_path,
        "fingerprints": fingerprints,
        "dolly_manifest_path": dolly_manifest_path,
        "dolly_manifest": dolly_manifest,
        "weights": weights,
        "model_paths": model_paths,
    }


def validate_parent_p3_3(
    parent: Path, original_config: dict[str, Any]
) -> dict[str, Any]:
    """验证唯一允许调整所依据的原P3-3弱攻击运行。"""

    parent = resolve_run_directory(parent, "p3_3_unseen_")
    if parent.name != PARENT_P3_3_RUN_ID:
        raise RuntimeError(f"P3-3R只允许引用固定原运行{PARENT_P3_3_RUN_ID}")
    required = (
        "resolved_config.json",
        "comparison.json",
        "g2_precheck.json",
        "fairness_audit.json",
        "unseen_training_order.jsonl",
        "unseen_training_order_sha256.json",
        "retention_curve.csv",
        "fingerprint_loss_curve.csv",
        "capability_curve.csv",
        "b1_attack/training_summary.json",
        "p_attack/training_summary.json",
    )
    for relative in required:
        if not (parent / relative).is_file():
            raise RuntimeError(f"原P3-3运行缺少文件：{relative}")
    resolved = load_json(parent / "resolved_config.json")
    comparison = load_json(parent / "comparison.json")
    g2 = load_json(parent / "g2_precheck.json")
    fairness = load_json(parent / "fairness_audit.json")
    if resolved.get("status") != "completed" or resolved.get("run_id") != parent.name:
        raise RuntimeError("原P3-3运行没有完整完成")
    if resolved.get("config") != original_config:
        raise RuntimeError("原P3-3运行的配置快照与冻结原配置不一致")
    if g2.get("g2_precheck") != "inconclusive_attack_too_weak":
        raise RuntimeError("原P3-3不是允许单次调整的弱攻击结果")
    expected_counts = {
        "0": {"b1": 32, "p": 32},
        "125": {"b1": 32, "p": 32},
        "375": {"b1": 32, "p": 32},
        "750": {"b1": 32, "p": 32},
    }
    if comparison.get("checkpoint_exact_counts") != expected_counts:
        raise RuntimeError("原P3-3不是所有检查点的32/32弱攻击结果")
    if comparison.get("fairness_audit_passed") is not True or fairness.get("all_checks_passed") is not True:
        raise RuntimeError("原P3-3公平性审计未通过")
    order_info = load_json(parent / "unseen_training_order_sha256.json")
    plan = load_jsonl(parent / "unseen_training_order.jsonl")
    actual_order_sha = training_plan_sha256(plan)
    if actual_order_sha != order_info.get("training_order_sha256"):
        raise RuntimeError("原P3-3训练顺序SHA256与实际文件不一致")
    summaries = {
        method: load_json(parent / f"{method}_attack/training_summary.json")
        for method in METHODS
    }
    original_initial_sha = summaries["b1"].get(
        "initial_attack_adapter_state_sha256"
    )
    if (
        not isinstance(original_initial_sha, str)
        or len(original_initial_sha) != 64
        or summaries["p"].get("initial_attack_adapter_state_sha256")
        != original_initial_sha
    ):
        raise RuntimeError("原P3-3两分支初始攻击适配器状态不一致")
    return {
        "path": parent,
        "resolved": resolved,
        "comparison": comparison,
        "g2": g2,
        "fairness": fairness,
        "order_info": order_info,
        "plan": plan,
        "training_summaries": summaries,
        "initial_attack_adapter_state_sha256": original_initial_sha,
    }


def validate_cloud_environment(config: dict[str, Any]) -> None:
    if not PROJECT_ROOT.resolve().is_relative_to(DATA_ROOT):
        raise RuntimeError("P3-3正式运行必须位于/root/autodl-tmp/")
    required = int(float(config["minimum_free_disk_gib"]) * 1024**3)
    if shutil.disk_usage(PROJECT_ROOT).free < required:
        raise RuntimeError("数据盘剩余空间不足8GiB")


def load_frozen_split(manifest: dict[str, Any], name: str) -> list[dict[str, Any]]:
    info = manifest["splits"][name]
    path = Path(str(info["file_path"]))
    if not path.is_file():
        raise RuntimeError(f"冻结Dolly文件不存在：{path}")
    if sha256_file(path) != info["file_sha256"]:
        raise RuntimeError(f"冻结Dolly文件SHA256不一致：{name}")
    rows = load_jsonl(path)
    if len(rows) != info["count"] or records_sha256(rows) != info["records_sha256"]:
        raise RuntimeError(f"冻结Dolly文件记录数或内容SHA256不一致：{name}")
    return rows


def initialize_run(
    run_dir: Path,
    run_id: str,
    checked_at: str,
    *,
    stage_id: str,
) -> None:
    for relative in (
        "b1_attack",
        "p_attack",
        "evaluations/step_0",
        "evaluations/step_125",
        "evaluations/step_375",
        "evaluations/step_750",
        "recovery",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "resolved_config.json",
        {
            "status": "not_run",
            "script_version": SCRIPT_VERSION,
            "run_id": run_id,
            "checked_at": checked_at,
            "stage_id": stage_id,
            "experiment_tier": "pilot",
            "paper_usage": "方法预实验，不进入论文主结果",
        },
    )
    write_json(run_dir / "parent_p3_2.json", {"status": "not_validated"})
    if stage_id == "P3-3R":
        write_json(run_dir / "parent_p3_3.json", {"status": "not_validated"})
        write_json(run_dir / "config_diff.json", {"config_audit_passed": False})
    write_json(run_dir / "unseen_data_audit.json", {"audit_passed": False})
    write_text(run_dir / "unseen_training_order.jsonl", "")
    write_json(run_dir / "unseen_training_order_sha256.json", {"status": "not_run"})
    for method in METHODS:
        write_text(run_dir / f"{method}_attack" / "training_metrics.jsonl", "")
        write_json(
            run_dir / f"{method}_attack" / "training_summary.json",
            {"method": method, "training_completed": False, "errors": []},
        )
    for name in (
        "fingerprint_loss_curve.csv",
        "retention_curve.csv",
        "paired_outcomes.csv",
        "capability_curve.csv",
    ):
        write_text(run_dir / name, "")
    write_json(run_dir / "fairness_audit.json", {"all_checks_passed": False})
    write_json(run_dir / "g2_precheck.json", {"g2_precheck": "not_run"})
    write_json(
        run_dir / "comparison.json",
        {
            "parent_p3_2_passed": False,
            "data_audit_passed": False,
            "b1_training_completed": False,
            "p_training_completed": False,
            "evaluation_completed": False,
            "p3_3_development_checks_passed": False,
            "g2_precheck": "not_run",
            "failure_reasons": [f"{stage_id}尚未完成"],
        },
    )
    write_text(run_dir / "summary.md", f"# {stage_id}摘要\n\n- 状态：尚未开始\n")


def _csv_rows_by_step(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row["checkpoint_step"]): row
            for row in csv.DictReader(handle)
        }


def _mean_target_losses(path: Path) -> dict[int, dict[str, float]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    grouped: dict[int, dict[str, list[float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["checkpoint_step"])
            method = str(row["method"])
            grouped.setdefault(step, {}).setdefault(method, []).append(
                float(row["target_token_loss"])
            )
    return {
        step: {
            method: sum(values) / len(values)
            for method, values in methods.items()
        }
        for step, methods in grouped.items()
    }


def render_summary(run_dir: Path, error: dict[str, str] | None = None) -> str:
    resolved = load_json(run_dir / "resolved_config.json")
    parent = load_json(run_dir / "parent_p3_2.json")
    comparison = load_json(run_dir / "comparison.json")
    g2 = load_json(run_dir / "g2_precheck.json")
    stage_id = str(resolved.get("stage_id", "P3-3"))
    lines = [
        f"# {stage_id}未见下游微调与B1/P指纹保持对比摘要",
        "",
        f"- 运行编号：`{resolved.get('run_id')}`",
        f"- 状态：`{resolved.get('status')}`",
        f"- P3-2父运行：`{parent.get('run_id')}`",
        f"- P3-1父运行：`{parent.get('parent_p3_1_run_id')}`",
        f"- 模型：`{resolved.get('model_id')}`",
        f"- revision：`{resolved.get('revision')}`",
        "",
        "## 固定未见攻击",
        "",
        f"- 未见数据审计通过：`{comparison.get('data_audit_passed')}`",
        f"- B1训练到750步：`{comparison.get('b1_training_completed')}`",
        f"- P训练到750步：`{comparison.get('p_training_completed')}`",
        f"- 公平性审计：`{comparison.get('fairness_audit_passed')}`",
        f"- 确定性输出一致：`{comparison.get('outputs_identical_across_repeats')}`",
        f"- 思考标签数量：`{comparison.get('thinking_tag_count')}`",
        "",
        "## G2方向筛查",
        "",
        f"- B1三个攻击后检查点平均保持率：`{g2.get('mean_retention_b1')}`",
        f"- P三个攻击后检查点平均保持率：`{g2.get('mean_retention_p')}`",
        f"- P减B1平均保持率差：`{g2.get('mean_retention_gap')}`",
        f"- 有信息检查点数量：`{g2.get('informative_checkpoint_count')}`",
        f"- G2预检查分类：`{g2.get('g2_precheck')}`",
        f"- {stage_id}开发检查通过：`{comparison.get('p3_3_development_checks_passed')}`",
    ]
    if stage_id == "P3-3R":
        parent_p33_path = run_dir / "parent_p3_3.json"
        config_diff_path = run_dir / "config_diff.json"
        parent_p33 = load_json(parent_p33_path) if parent_p33_path.is_file() else {}
        config_diff = load_json(config_diff_path) if config_diff_path.is_file() else {}
        original_dir_value = parent_p33.get("source_run_directory")
        original_dir = None
        if isinstance(original_dir_value, str) and original_dir_value:
            original_dir = Path(original_dir_value)
            if not original_dir.is_absolute():
                original_dir = PROJECT_ROOT / original_dir
        original_retention = (
            _csv_rows_by_step(original_dir / "retention_curve.csv")
            if original_dir is not None
            else {}
        )
        adjusted_retention = _csv_rows_by_step(run_dir / "retention_curve.csv")
        original_losses = (
            _mean_target_losses(original_dir / "fingerprint_loss_curve.csv")
            if original_dir is not None
            else {}
        )
        adjusted_losses = _mean_target_losses(run_dir / "fingerprint_loss_curve.csv")
        original_capability = (
            _csv_rows_by_step(original_dir / "capability_curve.csv")
            if original_dir is not None
            else {}
        )
        adjusted_capability = _csv_rows_by_step(run_dir / "capability_curve.csv")
        lines.extend(
            [
                "",
                "## 唯一允许的攻击强度调整",
                "",
                f"- 原P3-3运行：`{parent_p33.get('run_id')}`",
                f"- 原学习率：`{parent_p33.get('original_learning_rate')}`",
                f"- 调整后学习率：`{resolved.get('config', {}).get('learning_rate')}`",
                f"- 配置差异审计通过：`{config_diff.get('config_audit_passed')}`",
                f"- 除学习率外无实验参数变化：`{config_diff.get('only_experimental_change_is_learning_rate')}`",
                "",
                "### 原P3-3与P3-3R指纹命中及平均目标Token loss",
                "",
                "| 检查点 | 原B1/P命中 | P3-3R B1/P命中 | 原B1/P loss | P3-3R B1/P loss |",
                "| ---: | --- | --- | --- | --- |",
            ]
        )
        for step in EVALUATION_STEPS:
            old_ret = original_retention.get(step, {})
            new_ret = adjusted_retention.get(step, {})
            old_loss = original_losses.get(step, {})
            new_loss = adjusted_losses.get(step, {})
            lines.append(
                "| "
                f"{step} | {old_ret.get('b1_exact_count', '—')}/{old_ret.get('p_exact_count', '—')} "
                f"| {new_ret.get('b1_exact_count', '—')}/{new_ret.get('p_exact_count', '—')} "
                f"| {old_loss.get('b1', '—')}/{old_loss.get('p', '—')} "
                f"| {new_loss.get('b1', '—')}/{new_loss.get('p', '—')} |"
            )
        lines.extend(
            [
                "",
                "### 正常能力变化（completion loss）",
                "",
                "| 检查点 | 原P3-3 B1/P相对step 0 | P3-3R B1/P相对step 0 | P3-3R P相对B1 |",
                "| ---: | --- | --- | --- |",
            ]
        )
        for step in CAPABILITY_STEPS:
            old_row = original_capability.get(step, {})
            new_row = adjusted_capability.get(step, {})
            lines.append(
                "| "
                f"{step} | {old_row.get('b1_relative_to_step_0', '—')}/{old_row.get('p_relative_to_step_0', '—')} "
                f"| {new_row.get('b1_relative_to_step_0', '—')}/{new_row.get('p_relative_to_step_0', '—')} "
                f"| {new_row.get('p_relative_to_b1', '—')} |"
            )
        lines.extend(
            [
                "",
                f"- 调整后攻击是否具有信息量：`{bool(g2.get('informative_checkpoint_count', 0))}`",
                f"- P是否出现正向信号：`{g2.get('g2_precheck') == 'promising'}`",
            ]
        )
    if comparison.get("failure_reasons"):
        lines.extend(["", "## 未通过原因", ""])
        lines.extend(f"- {reason}" for reason in comparison["failure_reasons"])
    if error:
        lines.extend(["", "## 执行错误", "", f"- {error['type']}：{error['message']}"])
    lines.extend(
        [
            "",
            "> 本结果仅是单个开发训练—密钥组合的方向筛查，不能作为正式论文结论。",
            "> 本脚本不会再次调整攻击强度、重算P3-2权重或开始P4。",
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
    original_config_path = PROJECT_ROOT / "configs" / "training" / "p3_3_unseen.json"
    original_config = load_json(original_config_path)
    if config.get("stage_id") == "P3-3R":
        config_diff = validate_p33r_config(config, original_config)
        parent_p3_3 = validate_parent_p3_3(
            PROJECT_ROOT / "runs" / PARENT_P3_3_RUN_ID,
            original_config,
        )
        config_diff.update(
            {
                "original_config_path": project_relative(original_config_path),
                "adjusted_config_path": project_relative(config_path),
                "original_config_sha256": sha256_file(original_config_path),
                "adjusted_config_sha256": sha256_file(config_path),
                "original_p3_3_run_id": parent_p3_3["path"].name,
            }
        )
    else:
        validate_p33_config(config)
        config_diff = None
        parent_p3_3 = None
    parent_data = validate_parent_p3_2(
        parent_path, config, require_model_artifacts=require_model_artifacts
    )
    return {
        "config_path": config_path,
        "config": config,
        "stage_id": config["stage_id"],
        "is_adjustment": config["stage_id"] == "P3-3R",
        "original_config_path": original_config_path,
        "original_config": original_config,
        "config_diff": config_diff,
        "parent_p3_3": parent_p3_3,
        **parent_data,
    }


def execute_prepare(run_dir: Path, context: dict[str, Any]) -> None:
    config = context["config"]
    if context["is_adjustment"]:
        write_json(run_dir / "config_diff.json", context["config_diff"])
    validate_cloud_environment(config)
    print(f"固定P3-2 run ID：{context['parent'].name}")
    print(f"固定P3-1 run ID：{context['p3_1_path'].name}")
    splits = {
        name: load_frozen_split(context["dolly_manifest"], name)
        for name in ("normal_train", "proxy_reserved", "unseen_reserved", "capability_eval")
    }
    audit = audit_dolly_splits(splits, context["fingerprints"])
    write_json(run_dir / "unseen_data_audit.json", audit)
    if not audit["audit_passed"]:
        raise RuntimeError("冻结Dolly四集合交集或未见数据指纹污染审计失败")
    records = build_unseen_training_records(splits["unseen_reserved"])
    examples_sha = training_records_sha256(records)
    effective_batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    plan, order_sha = build_unseen_training_plan(
        records,
        seed=config["data_seed"],
        max_steps=config["max_steps"],
        effective_batch_size=effective_batch,
    )
    validate_unseen_training_plan(plan, records, expected_sha256=order_sha)
    if context["is_adjustment"]:
        original_plan = context["parent_p3_3"]["plan"]
        original_order_sha = context["parent_p3_3"]["order_info"][
            "training_order_sha256"
        ]
        validate_unseen_training_plan(
            original_plan,
            records,
            expected_sha256=original_order_sha,
        )
        if order_sha != original_order_sha or plan != original_plan:
            raise RuntimeError("P3-3R重算的训练顺序与原P3-3不一致")
        plan = original_plan
        order_sha = original_order_sha
        source_order = context["parent_p3_3"]["path"] / "unseen_training_order.jsonl"
        write_text(
            run_dir / "unseen_training_order.jsonl",
            source_order.read_text(encoding="utf-8"),
        )
    else:
        write_text(run_dir / "unseen_training_order.jsonl", training_plan_text(plan))
    order_info = {
        "status": "frozen",
        "seed": config["data_seed"],
        "record_count": len(plan),
        "unique_example_count": len(records),
        "epoch_count": 3,
        "optimizer_step_count": config["max_steps"],
        "effective_batch_size": effective_batch,
        "training_order_sha256": order_sha,
        "training_examples_sha256": examples_sha,
    }
    write_json(run_dir / "unseen_training_order_sha256.json", order_info)
    resolved = load_json(run_dir / "resolved_config.json")
    resolved.update(
        {
            "status": "prepared",
            "git_commit": git_commit(),
            "model_id": config["model_id"],
            "revision": config["revision"],
            "dtype": config["dtype"],
            "device": config["device"],
            "local_files_only": True,
            "parent_run_ids": (
                [PARENT_P3_3_RUN_ID, PARENT_P3_2_RUN_ID, PARENT_P3_1_RUN_ID]
                if context["is_adjustment"]
                else [PARENT_P3_2_RUN_ID, PARENT_P3_1_RUN_ID]
            ),
            "parent_p3_2_directory": project_relative(context["parent"]),
            "parent_p3_1_directory": project_relative(context["p3_1_path"]),
            "config_path": project_relative(context["config_path"]),
            "config_sha256": sha256_file(context["config_path"]),
            "config": config,
            "fingerprint_set_id": context["fingerprints"]["fingerprint_set_id"],
            "fingerprint_manifest_sha256": sha256_file(context["fingerprint_path"]),
            "dolly_split_manifest_sha256": sha256_file(context["dolly_manifest_path"]),
            "dolly_dataset_revision": context["dolly_manifest"]["dataset_revision"],
            "unseen_training_order_sha256": order_sha,
            "unseen_training_examples_sha256": examples_sha,
            "unseen_reserved_accessed": True,
            "training_data_usage": {
                "unseen_reserved": "训练",
                "capability_eval": "能力评估",
                "normal_train": "仅四集合交集审计",
                "proxy_reserved": "仅四集合交集审计",
                "fingerprint_examples": 0,
            },
            "generation": {
                "enable_thinking": False,
                "do_sample": False,
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
    if context["is_adjustment"]:
        write_json(run_dir / "config_diff.json", context["config_diff"])
        write_json(
            run_dir / "parent_p3_3.json",
            {
                "stage_id": "P3-3",
                "run_id": context["parent_p3_3"]["path"].name,
                "source_run_directory": project_relative(
                    context["parent_p3_3"]["path"]
                ),
                "status": "completed",
                "g2_precheck": context["parent_p3_3"]["g2"]["g2_precheck"],
                "original_learning_rate": context["original_config"][
                    "learning_rate"
                ],
                "adjusted_learning_rate": config["learning_rate"],
                "checkpoint_exact_counts": context["parent_p3_3"][
                    "comparison"
                ]["checkpoint_exact_counts"],
                "training_order_sha256": context["parent_p3_3"]["order_info"][
                    "training_order_sha256"
                ],
                "initial_attack_adapter_state_sha256": context["parent_p3_3"][
                    "initial_attack_adapter_state_sha256"
                ],
            },
        )
    write_json(
        run_dir / "parent_p3_2.json",
        {
            "stage_id": "P3-2",
            "run_id": context["parent"].name,
            "source_run_directory": project_relative(context["parent"]),
            "p3_2_passed": True,
            "parent_p3_1_run_id": context["p3_1_path"].name,
            "model_id": config["model_id"],
            "model_revision": config["revision"],
            "b1_release_exact_match_rate_by_repeat": context["comparison"]["b1_merged_exact_match_rate_by_repeat"],
            "p_release_exact_match_rate_by_repeat": context["comparison"]["p_merged_exact_match_rate_by_repeat"],
            "p_vs_b1_relative_capability_loss_change": context["capability"]["p_vs_b1_relative_loss_change"],
            "unseen_reserved_accessed_in_p3_2": False,
            "b1_merged_model_directory": project_relative(context["model_paths"]["b1"]),
            "p_merged_model_directory": project_relative(context["model_paths"]["p"]),
        },
    )
    comparison = load_json(run_dir / "comparison.json")
    comparison.update(
        {
            "parent_p3_2_passed": True,
            "data_audit_passed": True,
            "failure_reasons": [f"{context['stage_id']}训练和评估尚未完成"],
        }
    )
    write_json(run_dir / "comparison.json", comparison)
    write_text(run_dir / "summary.md", render_summary(run_dir))
    print(f"未见数据审计通过，固定训练顺序SHA256：{order_sha}")


def checkpoint_is_complete(path: Path) -> bool:
    return path.is_dir() and all(
        (path / name).is_file()
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "training_state.pt",
            "training_state.json",
        )
    )


def archive_partial_path(run_dir: Path, path: Path, label: str) -> None:
    if not path.exists():
        return
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    destination = run_dir / "recovery" / f"{label}_{timestamp}_{uuid.uuid4().hex[:8]}"
    shutil.move(str(path), destination)
    print(f"已将未完成产物移入恢复区：{destination}")


def latest_complete_checkpoint(branch_dir: Path, run_dir: Path, method: str) -> tuple[int, Path | None]:
    for temporary in sorted(branch_dir.glob(".step_*.incomplete_*")):
        archive_partial_path(run_dir, temporary, f"{method}_interrupted_checkpoint")
    latest_step = 0
    latest_path: Path | None = None
    completed_steps: list[int] = []
    for step in ATTACK_STEPS:
        path = branch_dir / f"step_{step}"
        if path.exists() and not checkpoint_is_complete(path):
            archive_partial_path(run_dir, path, f"{method}_incomplete_step_{step}")
            continue
        if checkpoint_is_complete(path):
            state = load_json(path / "training_state.json")
            if state.get("optimizer_step") != step or state.get("method") != method:
                raise RuntimeError(f"{method} step_{step}训练状态元数据不一致")
            latest_step = step
            latest_path = path
            completed_steps.append(step)
    if completed_steps and completed_steps != list(ATTACK_STEPS[: len(completed_steps)]):
        raise RuntimeError(f"{method}完整检查点序列不连续：{completed_steps}")
    return latest_step, latest_path


def load_and_validate_training_inputs(
    run_dir: Path, context: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    resolved = load_json(run_dir / "resolved_config.json")
    if resolved.get("status") not in {"prepared", "train_failed", "training", "trained"}:
        raise RuntimeError("P3-3运行目录尚未完成prepare")
    if resolved.get("config_sha256") != sha256_file(context["config_path"]):
        raise RuntimeError("P3-3配置在prepare后发生变化")
    unseen = load_frozen_split(context["dolly_manifest"], "unseen_reserved")
    records = build_unseen_training_records(unseen)
    if training_records_sha256(records) != resolved.get("unseen_training_examples_sha256"):
        raise RuntimeError("未见训练样本身份SHA256与prepare记录不一致")
    plan = load_jsonl(run_dir / "unseen_training_order.jsonl")
    order_info = load_json(run_dir / "unseen_training_order_sha256.json")
    validate_unseen_training_plan(
        plan,
        records,
        expected_sha256=order_info["training_order_sha256"],
    )
    if order_info["training_order_sha256"] != resolved.get("unseen_training_order_sha256"):
        raise RuntimeError("未见训练顺序SHA256与resolved_config不一致")
    return unseen, records, plan


def _trim_metrics_for_resume(
    metrics_path: Path, checkpoint_step: int, run_dir: Path, method: str
) -> list[dict[str, Any]]:
    rows = load_jsonl(metrics_path) if metrics_path.stat().st_size else []
    valid = [row for row in rows if int(row.get("optimizer_step", -1)) <= checkpoint_step]
    discarded = [row for row in rows if int(row.get("optimizer_step", -1)) > checkpoint_step]
    expected = list(range(1, checkpoint_step + 1))
    if [int(row["optimizer_step"]) for row in valid] != expected:
        raise RuntimeError(f"{method}现有训练指标与最近完整检查点不连续")
    if discarded:
        recovery = run_dir / "recovery" / f"{method}_discarded_metrics_after_step_{checkpoint_step}_{uuid.uuid4().hex[:8]}.jsonl"
        write_jsonl(recovery, discarded)
    write_jsonl(metrics_path, valid)
    return valid


def _save_attack_checkpoint(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    get_peft_model_state_dict: Any,
    destination: Path,
    method: str,
    optimizer_step: int,
    order_sha256: str,
) -> str:
    if destination.exists():
        if checkpoint_is_complete(destination):
            raise RuntimeError(f"拒绝覆盖已完成检查点：{destination}")
        raise RuntimeError(f"检查点目标已存在但不完整：{destination}")
    temporary = destination.parent / f".{destination.name}.incomplete_{uuid.uuid4().hex[:8]}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        adapter_sha = adapter_state_sha256(model, get_peft_model_state_dict)
        state = {
            "optimizer_step": optimizer_step,
            "method": method,
            "training_order_sha256": order_sha256,
            "adapter_state_sha256": adapter_sha,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "python_random_state": random.getstate(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        }
        torch.save(state, temporary / "training_state.pt")
        write_json(
            temporary / "training_state.json",
            {
                "optimizer_step": optimizer_step,
                "method": method,
                "training_order_sha256": order_sha256,
                "adapter_state_sha256": adapter_sha,
                "contains_optimizer_state": True,
                "contains_scheduler_state": True,
                "contains_rng_state": True,
                "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
        temporary.rename(destination)
        return adapter_sha
    except Exception:
        if temporary.exists():
            print(f"保留未完成检查点供诊断：{temporary}", file=sys.stderr)
        raise


def train_attack_branch(
    *,
    method: str,
    run_dir: Path,
    context: dict[str, Any],
    records: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    torch: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    LoraConfig: Any,
    PeftModel: Any,
    get_peft_model: Any,
    get_peft_model_state_dict: Any,
    get_linear_schedule_with_warmup: Any,
) -> dict[str, Any]:
    config = context["config"]
    branch_dir = run_dir / f"{method}_attack"
    metrics_path = branch_dir / "training_metrics.jsonl"
    summary_path = branch_dir / "training_summary.json"
    existing_summary = load_json(summary_path)
    latest_step, checkpoint_path = latest_complete_checkpoint(branch_dir, run_dir, method)
    if existing_summary.get("training_completed") is True and latest_step == config["max_steps"]:
        print(f"{method.upper()}攻击训练已完成，安全跳过")
        return existing_summary
    metrics = _trim_metrics_for_resume(metrics_path, latest_step, run_dir, method)
    losses = [float(row["loss"]) for row in metrics]
    order_info = load_json(run_dir / "unseen_training_order_sha256.json")
    examples_sha = training_records_sha256(records)
    if examples_sha != order_info["training_examples_sha256"]:
        raise RuntimeError(f"{method}训练样本SHA256不一致")
    device = config["device"]
    parent_model_path = context["model_paths"][method]
    tokenizer = AutoTokenizer.from_pretrained(
        parent_model_path, trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer缺少PAD和EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    examples = [
        build_completion_example(
            tokenizer,
            sample_id=row["sample_id"],
            sample_type="unseen",
            prompt=row["prompt"],
            response=row["response"],
            max_sequence_length=config["max_seq_length"],
            sample_weight=1.0,
        )
        for row in records
    ]
    if len(examples) != 1000 or any(example["was_truncated"] for example in examples):
        raise RuntimeError(f"{method}未见训练数据数量或截断检查失败")
    base = AutoModelForCausalLM.from_pretrained(
        parent_model_path,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    resolved_modules = resolve_target_modules(base, config["target_modules"])
    if latest_step:
        model = PeftModel.from_pretrained(
            base, checkpoint_path, is_trainable=True, local_files_only=True
        )
        if set(model.peft_config) != {"default"}:
            raise RuntimeError(f"{method}恢复后必须只有一个default下游适配器")
    else:
        set_reproducible_seed(torch, config["seed"])
        model = get_peft_model(base, build_lora_config(config, LoraConfig))
    parameters = parameter_summary(model)
    initial_hash = existing_summary.get("initial_attack_adapter_state_sha256")
    if latest_step == 0:
        initial_hash = adapter_state_sha256(model, get_peft_model_state_dict)
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        raise RuntimeError(f"{method}缺少初始下游适配器SHA256")
    lora_snapshot = {
        "r": config["lora_r"],
        "alpha": config["lora_alpha"],
        "dropout": config["lora_dropout"],
        "bias": config["lora_bias"],
        "task_type": config["task_type"],
        "target_modules": config["target_modules"],
    }
    summary = {
        **parameters,
        "method": method,
        "training_completed": False,
        "parent_p3_2_run_id": context["parent"].name,
        "parent_release_model_directory": project_relative(parent_model_path),
        "seed": config["seed"],
        "data_seed": config["data_seed"],
        "unseen_example_count": 1000,
        "fingerprint_training_example_count": 0,
        "consumed_training_record_count": len(plan),
        "max_steps": config["max_steps"],
        "checkpoint_steps": config["checkpoint_steps"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "effective_batch_size": config["per_device_train_batch_size"] * config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "lora_config": lora_snapshot,
        "resolved_target_modules": resolved_modules,
        "training_examples_sha256": examples_sha,
        "training_order_sha256": order_info["training_order_sha256"],
        "completion_only_loss": True,
        "initial_attack_adapter_state_sha256": initial_hash,
        "resumed_from_optimizer_step": latest_step,
        "errors": existing_summary.get("errors", []),
    }
    write_json(summary_path, summary)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["max_steps"],
    )
    if checkpoint_path is not None:
        state = torch.load(
            checkpoint_path / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if state["optimizer_step"] != latest_step or state["training_order_sha256"] != order_info["training_order_sha256"]:
            raise RuntimeError(f"{method}恢复状态与固定训练顺序不一致")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        random.setstate(state["python_random_state"])
        torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
        reloaded_hash = adapter_state_sha256(model, get_peft_model_state_dict)
        if reloaded_hash != state["adapter_state_sha256"]:
            raise RuntimeError(f"{method}恢复检查点适配器SHA256不一致")
        print(f"{method.upper()}从step {latest_step}安全恢复")
    else:
        optimizer.zero_grad(set_to_none=True)
    model.train()
    original_use_cache = bool(model.config.use_cache)
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats(device)
    accumulation = config["gradient_accumulation_steps"]
    accumulated_loss = 0.0
    started = time.perf_counter()
    try:
        for item in plan[latest_step * accumulation :]:
            example = examples[int(item["record_index"])]
            batch = collate_training_examples(
                [example],
                pad_token_id=tokenizer.pad_token_id,
                tensor_factory=lambda value: torch.tensor(value, dtype=torch.long, device=device),
            )
            labels = batch.pop("labels")
            output = model(**batch)
            weight = torch.ones(1, dtype=torch.float32, device=device)
            loss, _per_example, _token_counts = weighted_completion_loss(
                torch, output.logits, labels, weight
            )
            accumulated_loss += float(loss.detach().float().item())
            (loss / accumulation).backward()
            position = int(item["global_position"])
            del batch, labels, output, weight, loss, _per_example, _token_counts
            if position % accumulation:
                continue
            optimizer_step = position // accumulation
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config["max_grad_norm"])
            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError(f"{method}第{optimizer_step}步梯度非有限")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            average_loss = accumulated_loss / accumulation
            accumulated_loss = 0.0
            if not math.isfinite(average_loss):
                raise FloatingPointError(f"{method}第{optimizer_step}步loss非有限")
            losses.append(average_loss)
            metric = {
                "optimizer_step": optimizer_step,
                "loss": round(average_loss, 8),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            append_jsonl(metrics_path, metric)
            started = time.perf_counter()
            if optimizer_step % config["logging_steps"] == 0:
                print(f"{method} attack step {optimizer_step}/{config['max_steps']} loss={average_loss:.8f} lr={metric['learning_rate']:.8g}")
            if optimizer_step in ATTACK_STEPS:
                adapter_sha = _save_attack_checkpoint(
                    torch=torch,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    get_peft_model_state_dict=get_peft_model_state_dict,
                    destination=branch_dir / f"step_{optimizer_step}",
                    method=method,
                    optimizer_step=optimizer_step,
                    order_sha256=order_info["training_order_sha256"],
                )
                print(f"{method.upper()}已保存step {optimizer_step}，adapter SHA256={adapter_sha}")
    finally:
        model.config.use_cache = original_use_cache
    summary.update(summarize_training_losses(losses, config["max_steps"]))
    summary["training_metric_record_count"] = len(losses)
    summary["completed_checkpoint_steps"] = [
        step for step in ATTACK_STEPS if checkpoint_is_complete(branch_dir / f"step_{step}")
    ]
    summary["final_adapter_state_sha256"] = adapter_state_sha256(model, get_peft_model_state_dict)
    write_json(summary_path, summary)
    model = None
    base = None
    tokenizer = None
    release_gpu(torch)
    return summary


def execute_train(run_dir: Path, context: dict[str, Any]) -> None:
    validate_cloud_environment(context["config"])
    _unseen, records, plan = load_and_validate_training_inputs(run_dir, context)
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU不支持BF16")
    resolved = load_json(run_dir / "resolved_config.json")
    resolved["status"] = "training"
    write_json(run_dir / "resolved_config.json", resolved)
    summaries = {}
    for method in METHODS:
        summaries[method] = train_attack_branch(
            method=method,
            run_dir=run_dir,
            context=context,
            records=records,
            plan=plan,
            torch=torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
            LoraConfig=LoraConfig,
            PeftModel=PeftModel,
            get_peft_model=get_peft_model,
            get_peft_model_state_dict=get_peft_model_state_dict,
            get_linear_schedule_with_warmup=get_linear_schedule_with_warmup,
        )
    order_info = load_json(run_dir / "unseen_training_order_sha256.json")
    audit = load_json(run_dir / "unseen_data_audit.json")
    fairness = compute_attack_fairness_audit(
        b1_summary=summaries["b1"],
        p_summary=summaries["p"],
        order_sha256=order_info["training_order_sha256"],
        unseen_examples_sha256=order_info["training_examples_sha256"],
        data_audit=audit,
        config=context["config"],
    )
    if context["is_adjustment"]:
        original = context["parent_p3_3"]
        original_summaries = original["training_summaries"]
        fairness.update(
            {
                "same_training_order_as_original_p3_3": order_info[
                    "training_order_sha256"
                ]
                == original["order_info"]["training_order_sha256"],
                "same_checkpoint_steps_as_original_p3_3": all(
                    summary.get("checkpoint_steps")
                    == original_summaries[method].get("checkpoint_steps")
                    for method, summary in summaries.items()
                ),
                "same_max_steps_as_original_p3_3": all(
                    summary.get("max_steps")
                    == original_summaries[method].get("max_steps")
                    for method, summary in summaries.items()
                ),
                "same_lora_config_as_original_p3_3": all(
                    summary.get("lora_config")
                    == original_summaries[method].get("lora_config")
                    for method, summary in summaries.items()
                ),
                "same_random_seed_as_original_p3_3": all(
                    summary.get("seed") == original_summaries[method].get("seed")
                    for method, summary in summaries.items()
                ),
                "same_initial_attack_adapter_state_as_original_p3_3": all(
                    summary.get("initial_attack_adapter_state_sha256")
                    == original["initial_attack_adapter_state_sha256"]
                    for summary in summaries.values()
                ),
                "only_change_from_original_p3_3": "learning_rate",
                "only_change_from_original_p3_3_verified": context[
                    "config_diff"
                ].get("only_experimental_change_is_learning_rate")
                is True,
                "original_learning_rate": context["original_config"][
                    "learning_rate"
                ],
                "adjusted_learning_rate": context["config"]["learning_rate"],
            }
        )
        fairness["all_checks_passed"] = all(
            value for value in fairness.values() if isinstance(value, bool)
        )
    write_json(run_dir / "fairness_audit.json", fairness)
    if not fairness["all_checks_passed"]:
        raise RuntimeError("B1/P攻击训练公平性审计未通过")
    if any(summary.get("training_completed") is not True for summary in summaries.values()):
        raise RuntimeError("B1或P没有完整训练到750步")
    resolved["status"] = "trained"
    write_json(run_dir / "resolved_config.json", resolved)
    comparison = load_json(run_dir / "comparison.json")
    comparison.update(
        {
            "b1_training_completed": True,
            "p_training_completed": True,
            "fairness_audit_passed": True,
            "same_initial_attack_adapter_state": fairness["same_initial_attack_adapter_state"],
            "failure_reasons": ["P3-3评估尚未完成"],
        }
    )
    write_json(run_dir / "comparison.json", comparison)
    write_text(run_dir / "summary.md", render_summary(run_dir))
    print("B1与P均已完整训练到750步，公平性审计通过。")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False, allow_nan=False)
                values[field] = value
            writer.writerow(values)


def evaluation_directory_is_complete(path: Path, step: int) -> bool:
    required = [
        "raw_generations.jsonl",
        "scores.csv",
        "metrics.json",
        "target_losses.json",
        "evaluation_complete.json",
    ]
    if step in CAPABILITY_STEPS:
        required.append("capability_loss.json")
    if step == 750:
        required.extend(("capability_generations.jsonl", "capability_scores.csv"))
    return path.is_dir() and all(
        (path / name).is_file()
        for name in required
    )


def evaluate_attack_state(
    *,
    method: str,
    step: int,
    run_dir: Path,
    context: dict[str, Any],
    capability_rows: list[dict[str, Any]],
    torch: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    PeftModel: Any,
) -> dict[str, Any]:
    output_dir = run_dir / "evaluations" / f"step_{step}" / method
    if evaluation_directory_is_complete(output_dir, step):
        print(f"{method.upper()} step {step}评估已完成，安全复用")
        return {
            "records": load_jsonl(output_dir / "raw_generations.jsonl"),
            "metrics": load_json(output_dir / "metrics.json"),
            "target_losses": load_json(output_dir / "target_losses.json")["fingerprints"],
            "capability": load_json(output_dir / "capability_loss.json")
            if (output_dir / "capability_loss.json").is_file()
            else None,
            "generations": load_jsonl(output_dir / "capability_generations.jsonl")
            if (output_dir / "capability_generations.jsonl").is_file()
            else None,
        }
    if output_dir.exists():
        archive_partial_path(run_dir, output_dir, f"partial_evaluation_{method}_step_{step}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = context["config"]
    parent_model_path = context["model_paths"][method]
    tokenizer = AutoTokenizer.from_pretrained(
        parent_model_path, trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer缺少PAD和EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        parent_model_path,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(config["device"])
    checkpoint_path: Path | None = None
    if step > 0:
        checkpoint_path = run_dir / f"{method}_attack" / f"step_{step}"
        if not checkpoint_is_complete(checkpoint_path):
            raise RuntimeError(f"{method} step_{step}检查点不完整")
        model = PeftModel.from_pretrained(
            model, checkpoint_path, is_trainable=False, local_files_only=True
        )
    records, metrics = evaluate_fingerprints(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        manifest=context["fingerprints"],
        repeats=config["evaluation_repeats"],
        maximum_new_tokens=config["evaluation_max_new_tokens"],
        device=config["device"],
        output_dir=output_dir,
        expected_mode="positive",
    )
    records = [
        {**row, "method": method, "checkpoint_step": step} for row in records
    ]
    metrics.update(
        {
            "method": method,
            "checkpoint_step": step,
            "parent_release_model_directory": project_relative(parent_model_path),
            "attack_adapter_directory": project_relative(checkpoint_path)
            if checkpoint_path is not None
            else None,
            "retention_evaluation": True,
        }
    )
    write_jsonl(output_dir / "raw_generations.jsonl", records)
    write_scores(output_dir / "scores.csv", records)
    write_json(output_dir / "metrics.json", metrics)
    target_losses = measure_fingerprint_target_losses(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        fingerprints=context["fingerprints"],
        maximum_length=config["max_seq_length"],
        device=config["device"],
    )
    write_json(
        output_dir / "target_losses.json",
        {
            "method": method,
            "checkpoint_step": step,
            "loss_scope": "assistant target code tokens only",
            "internal_dtype": "float32",
            "fingerprints": target_losses,
        },
    )
    capability = None
    generations = None
    if step in CAPABILITY_STEPS:
        capability = average_completion_loss(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            records=capability_rows,
            maximum_length=config["max_seq_length"],
            device=config["device"],
        )
        write_json(output_dir / "capability_loss.json", capability)
    if step == config["capability_generation_step"]:
        generations = generate_capability_samples(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            records=capability_rows,
            maximum_new_tokens=config["evaluation_max_new_tokens"],
            device=config["device"],
        )
        known_codes = set(P1_RESPONSES) | set(context["fingerprints"]["allowed_responses"])
        write_jsonl(output_dir / "capability_generations.jsonl", generations)
        write_capability_scores(
            output_dir / "capability_scores.csv", generations, known_codes
        )
    write_json(
        output_dir / "evaluation_complete.json",
        {
            "method": method,
            "checkpoint_step": step,
            "completed": True,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
    model = None
    tokenizer = None
    release_gpu(torch)
    return {
        "records": records,
        "metrics": metrics,
        "target_losses": target_losses,
        "capability": capability,
        "generations": generations,
    }


def execute_evaluate(run_dir: Path, context: dict[str, Any]) -> bool:
    resolved = load_json(run_dir / "resolved_config.json")
    if resolved.get("status") not in {"trained", "evaluate_failed", "evaluating", "completed"}:
        raise RuntimeError("P3-3尚未完成两条攻击训练分支，禁止评估")
    if resolved.get("config_sha256") != sha256_file(context["config_path"]):
        raise RuntimeError("P3-3配置在训练后发生变化")
    for method in METHODS:
        summary = load_json(run_dir / f"{method}_attack" / "training_summary.json")
        if summary.get("training_completed") is not True:
            raise RuntimeError(f"{method.upper()}攻击训练没有完成750步")
        for step in ATTACK_STEPS:
            if not checkpoint_is_complete(run_dir / f"{method}_attack" / f"step_{step}"):
                raise RuntimeError(f"{method.upper()}缺少完整step_{step}检查点")
    capability_rows = load_frozen_split(context["dolly_manifest"], "capability_eval")
    from peft import PeftModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU不支持BF16")
    resolved["status"] = "evaluating"
    write_json(run_dir / "resolved_config.json", resolved)
    results: dict[str, dict[int, dict[str, Any]]] = {"b1": {}, "p": {}}
    for step in EVALUATION_STEPS:
        for method in METHODS:
            results[method][step] = evaluate_attack_state(
                method=method,
                step=step,
                run_dir=run_dir,
                context=context,
                capability_rows=capability_rows,
                torch=torch,
                AutoModelForCausalLM=AutoModelForCausalLM,
                AutoTokenizer=AutoTokenizer,
                PeftModel=PeftModel,
            )
    fingerprint_ids = [
        row["fingerprint_id"] for row in context["fingerprints"]["fingerprints"]
    ]
    evaluation_records = {
        method: {step: results[method][step]["records"] for step in EVALUATION_STEPS}
        for method in METHODS
    }
    retention_curve, paired_outcomes, retention_details = compute_retention_outputs(
        evaluation_records,
        fingerprint_ids,
        context["weights"],
    )
    write_csv(
        run_dir / "retention_curve.csv",
        retention_curve,
        [
            "checkpoint_step",
            "b1_exact_count",
            "p_exact_count",
            "b1_exact_rate",
            "p_exact_rate",
            "rate_difference_p_minus_b1",
        ],
    )
    write_csv(
        run_dir / "paired_outcomes.csv",
        paired_outcomes,
        [
            "checkpoint_step",
            "both_retained",
            "only_p_retained",
            "only_b1_retained",
            "neither_retained",
            "only_p_retained_ids",
            "only_b1_retained_ids",
        ],
    )
    loss_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for step in EVALUATION_STEPS:
            exact_by_id = {
                fingerprint_id: all(
                    row["parse_status"] == "exact_match"
                    for row in results[method][step]["records"]
                    if row["fingerprint_id"] == fingerprint_id
                )
                for fingerprint_id in fingerprint_ids
            }
            for fingerprint_id in fingerprint_ids:
                item = results[method][step]["target_losses"][fingerprint_id]
                loss_rows.append(
                    {
                        "fingerprint_id": fingerprint_id,
                        "feedback_weight": context["weights"][fingerprint_id],
                        "method": method,
                        "checkpoint_step": step,
                        "target_token_loss": item["loss"],
                        "target_token_count": item["target_token_count"],
                        "is_exact_match": exact_by_id[fingerprint_id],
                    }
                )
    write_csv(
        run_dir / "fingerprint_loss_curve.csv",
        loss_rows,
        [
            "fingerprint_id",
            "feedback_weight",
            "method",
            "checkpoint_step",
            "target_token_loss",
            "target_token_count",
            "is_exact_match",
        ],
    )
    capability_losses = {
        method: {
            step: float(results[method][step]["capability"]["average_completion_token_loss"])
            for step in CAPABILITY_STEPS
        }
        for method in METHODS
    }
    capability_curve = compute_capability_curve(capability_losses)
    write_csv(
        run_dir / "capability_curve.csv",
        capability_curve,
        [
            "checkpoint_step",
            "b1_completion_loss",
            "p_completion_loss",
            "b1_relative_to_step_0",
            "p_relative_to_step_0",
            "p_relative_to_b1",
        ],
    )
    known_codes = set(P1_RESPONSES) | set(context["fingerprints"]["allowed_responses"])
    capability_comparison = compute_capability_comparison(
        b1_loss=capability_losses["b1"][750],
        p_loss=capability_losses["p"][750],
        b1_generations=results["b1"][750]["generations"],
        p_generations=results["p"][750]["generations"],
        known_codes=known_codes,
        maximum_relative_increase=context["config"]["maximum_p_vs_b1_loss_increase"],
    )
    write_json(
        run_dir / "evaluations" / "step_750" / "capability_comparison.json",
        capability_comparison,
    )
    g2 = compute_g2_precheck(
        retention_curve,
        retention_details,
        capability_curve,
        step_750_smoke_passed=bool(capability_comparison["checks"]["b1_generation_smoke_passed"] and capability_comparison["checks"]["p_generation_smoke_passed"]),
        config=context["config"],
    )
    if context["is_adjustment"]:
        original_classification = g2["g2_precheck"]
        g2["base_classification_before_adjustment_mapping"] = original_classification
        g2["g2_precheck"] = classify_p33r_result(original_classification)
        g2["allowed_adjustment_index"] = 1
        g2["allowed_adjustment_limit"] = 1
        g2["further_attack_adjustment_allowed"] = False
    g2["paired_retention_details"] = retention_details
    write_json(run_dir / "g2_precheck.json", g2)
    fairness = load_json(run_dir / "fairness_audit.json")
    outputs_identical = all(
        results[method][step]["metrics"]["outputs_identical_across_repeats"]
        for method in METHODS
        for step in EVALUATION_STEPS
    )
    thinking_count = sum(
        int(results[method][step]["metrics"]["thinking_tag_count"])
        for method in METHODS
        for step in EVALUATION_STEPS
    ) + sum(
        "<think>" in row["raw_output"] or "</think>" in row["raw_output"]
        for method in METHODS
        for row in results[method][750]["generations"]
    )
    capability_within_limit = all(
        row["p_relative_to_b1"] <= context["config"]["maximum_p_vs_b1_loss_increase"]
        for row in capability_curve
    )
    terminal_text = (run_dir / "terminal_output.log").read_text(encoding="utf-8")
    warning_count = sum(
        terminal_text.count(fragment) for fragment in FORBIDDEN_WARNING_FRAGMENTS
    )
    checks = [
        (load_json(run_dir / "unseen_data_audit.json").get("audit_passed") is True, "未见数据审计未通过"),
        (all(load_json(run_dir / f"{method}_attack/training_summary.json").get("training_completed") is True for method in METHODS), "攻击训练未完整到750步"),
        (outputs_identical, "两次确定性查询不一致"),
        (thinking_count == 0, "输出包含思考标签"),
        (fairness.get("all_checks_passed") is True, "公平性审计未通过"),
        (g2["informative_checkpoint_count"] >= 1, "没有有信息检查点"),
        (capability_within_limit, "P正常能力相对B1恶化超过5%"),
        (capability_comparison["checks"]["b1_generation_smoke_passed"] is True and capability_comparison["checks"]["p_generation_smoke_passed"] is True, "step_750正常生成冒烟检查未通过"),
        (warning_count == 0, "日志包含弃用或无效采样参数警告"),
    ]
    failure_reasons = [reason for passed, reason in checks if not passed]
    comparison = load_json(run_dir / "comparison.json")
    comparison.update(
        {
            "parent_p3_2_passed": True,
            "data_audit_passed": True,
            "b1_training_completed": True,
            "p_training_completed": True,
            "evaluation_completed": True,
            "checkpoint_exact_counts": {
                str(row["checkpoint_step"]): {
                    "b1": row["b1_exact_count"],
                    "p": row["p_exact_count"],
                }
                for row in retention_curve
            },
            "mean_retention_b1": retention_details["mean_retention_b1"],
            "mean_retention_p": retention_details["mean_retention_p"],
            "mean_retention_gap": retention_details["mean_retention_gap"],
            "outputs_identical_across_repeats": outputs_identical,
            "thinking_tag_count": thinking_count,
            "fairness_audit_passed": fairness["all_checks_passed"],
            "capability_comparison_passed": capability_within_limit,
            "forbidden_warning_count": warning_count,
            "informative_checkpoint_count": g2["informative_checkpoint_count"],
            "g2_precheck": g2["g2_precheck"],
            "p3_3_development_checks_passed": not failure_reasons,
            "p3_3r_development_checks_passed": (
                not failure_reasons if context["is_adjustment"] else None
            ),
            "only_change_from_original_p3_3": (
                fairness.get("only_change_from_original_p3_3")
                if context["is_adjustment"]
                else None
            ),
            "failure_reasons": failure_reasons,
            "formal_conclusion_allowed": False,
        }
    )
    write_json(run_dir / "comparison.json", comparison)
    resolved["status"] = "completed"
    write_json(run_dir / "resolved_config.json", resolved)
    write_text(run_dir / "summary.md", render_summary(run_dir))
    print(f"{context['stage_id']}完成，G2预检查分类：{g2['g2_precheck']}")
    return not failure_reasons


def update_failure(run_dir: Path, stage: str, exc: Exception) -> None:
    error = {"type": type(exc).__name__, "message": sanitize_text(exc)}
    resolved = load_json(run_dir / "resolved_config.json")
    resolved["status"] = f"{stage}_failed"
    resolved["error"] = error
    write_json(run_dir / "resolved_config.json", resolved)
    comparison = load_json(run_dir / "comparison.json")
    comparison["failure_reasons"] = [f"{error['type']}：{error['message']}"]
    comparison["p3_3_development_checks_passed"] = False
    write_json(run_dir / "comparison.json", comparison)
    if stage == "train":
        for method in METHODS:
            path = run_dir / f"{method}_attack" / "training_summary.json"
            summary = load_json(path)
            if summary.get("training_completed") is not True:
                summary.setdefault("errors", []).append(error)
                write_json(path, summary)
    write_text(run_dir / "summary.md", render_summary(run_dir, error))
    print(
        f"{resolved.get('stage_id', 'P3-3')} {stage}失败："
        f"{error['type']}：{error['message']}",
        file=sys.stderr,
    )
    print(f"输出目录：{run_dir}")


def context_for_existing_run(
    args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    resolved = load_json(run_dir / "resolved_config.json")
    config_preview = load_json(args.config.expanduser().resolve())
    if resolved.get("stage_id") != config_preview.get("stage_id"):
        raise RuntimeError("运行目录阶段与命令指定配置不一致")
    parent_value = resolved.get("parent_p3_2_directory")
    if not isinstance(parent_value, str) or not parent_value:
        raise RuntimeError("P3-3 resolved_config缺少P3-2父目录")
    stored_parent = Path(parent_value)
    if not stored_parent.is_absolute():
        stored_parent = PROJECT_ROOT / stored_parent
    explicit = resolve_run_directory(args.p3_2_run, "p3_2_feedback_")
    if explicit != stored_parent.resolve():
        raise RuntimeError("命令指定的P3-2父运行与当前P3-3目录不一致")
    context = prepare_context(
        args.config, stored_parent, require_model_artifacts=True
    )
    if resolved.get("config_sha256") != sha256_file(context["config_path"]):
        raise RuntimeError("P3-3配置与创建运行目录时不一致")
    return context


def preflight_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """在创建运行目录前验证配置，避免无效调整留下空目录。"""

    config = load_json(config_path.expanduser().resolve())
    if config.get("stage_id") == "P3-3R":
        original = load_json(
            PROJECT_ROOT / "configs" / "training" / "p3_3_unseen.json"
        )
        return config, validate_p33r_config(config, original)
    validate_p33_config(config)
    return config, None


def main() -> int:
    args = parse_args()
    try:
        config_preview, _config_diff = preflight_config(args.config)
    except Exception as exc:
        print(f"配置审计失败：{sanitize_text(exc)}", file=sys.stderr)
        return 3
    stage_id = config_preview["stage_id"]
    is_adjustment = stage_id == "P3-3R"
    run_prefix = "p3_3r_unseen_lr5e4_" if is_adjustment else "p3_3_unseen_"
    creating = args.stage in {"prepare", "all"}
    if creating:
        if args.run_dir is not None:
            print(f"{args.stage}不能传入--run-dir", file=sys.stderr)
            return 3
        checked_at = datetime.now().astimezone()
        try:
            creator = (
                create_unique_p33r_run_directory
                if is_adjustment
                else create_unique_p33_run_directory
            )
            run_id, run_dir = creator(PROJECT_ROOT / "runs", checked_at)
            initialize_run(
                run_dir,
                run_id,
                checked_at.isoformat(timespec="seconds"),
                stage_id=stage_id,
            )
        except Exception as exc:
            print(f"无法创建{stage_id}运行目录：{sanitize_text(exc)}", file=sys.stderr)
            return 3
        log_mode = "w"
    else:
        if args.run_dir is None:
            print(f"{args.stage}必须传入--run-dir", file=sys.stderr)
            return 3
        try:
            run_dir = resolve_run_directory(args.run_dir, run_prefix)
        except Exception as exc:
            print(f"无效{stage_id}运行目录：{sanitize_text(exc)}", file=sys.stderr)
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
            print(f"{stage_id}阶段：{args.stage}")
            print(f"终端日志：{terminal_path}")
            current_stage = "prepare" if creating else args.stage
            try:
                if creating:
                    context = prepare_context(
                        args.config,
                        args.p3_2_run,
                        require_model_artifacts=True,
                    )
                    execute_prepare(run_dir, context)
                    if args.stage == "prepare":
                        print(f"输出目录：{run_dir}")
                        return 0
                    current_stage = "train"
                    execute_train(run_dir, context)
                    current_stage = "evaluate"
                    passed = execute_evaluate(run_dir, context)
                    print(f"输出目录：{run_dir}")
                    return 0 if passed else 2

                context = context_for_existing_run(args, run_dir)
                if args.stage == "train":
                    execute_train(run_dir, context)
                    print(f"输出目录：{run_dir}")
                    return 0
                passed = execute_evaluate(run_dir, context)
                print(f"输出目录：{run_dir}")
                return 0 if passed else 2
            except Exception as exc:
                update_failure(run_dir, current_stage, exc)
                return 3


if __name__ == "__main__":
    raise SystemExit(main())
