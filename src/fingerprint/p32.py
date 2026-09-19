"""P3-2代理遗忘反馈、加权训练与公平性审计的纯逻辑。"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from fingerprint.p3 import (
    build_completion_example,
    build_training_records,
    canonical_json,
    validate_p3_fingerprint_manifest,
)


EXPECTED_MODEL_ID = "Qwen/Qwen3-0.6B"
EXPECTED_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
EXPECTED_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def validate_p32_config(config: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "stage_id": "P3-2",
        "experiment_tier": "pilot",
        "model_id": EXPECTED_MODEL_ID,
        "revision": EXPECTED_REVISION,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "local_files_only": True,
        "enable_thinking": False,
        "max_seq_length": 512,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "task_type": "CAUSAL_LM",
        "learning_rate": 2e-4,
        "bf16": True,
        "gradient_checkpointing": False,
        "report_to": "none",
        "proxy_seed": 314159,
        "proxy_data_seed": 314159,
        "proxy_num_train_epochs": 3,
        "proxy_example_count": 500,
        "continuation_seed": 42,
        "continuation_data_seed": 271828,
        "continuation_max_steps": 80,
        "normal_example_count": 1000,
        "fingerprint_count": 32,
        "fingerprint_repeat": 8,
        "normal_example_weight": 1.0,
        "b1_fingerprint_weight": 1.0,
        "weight_multiplier": 2.0,
        "weight_scale_percentile": 0.9,
        "weight_scale_epsilon": 1e-8,
        "minimum_positive_delta_count": 8,
        "positive_delta_threshold": 1e-4,
        "minimum_difficulty_std": 1e-4,
        "minimum_weight_std": 0.01,
        "maximum_weight_mean_error": 1e-6,
        "evaluation_max_new_tokens": 16,
        "evaluation_repeats": 2,
        "evaluation_do_sample": False,
        "system_prompt": None,
        "capability_generation_count": 10,
        "maximum_p_vs_b1_loss_increase": 0.05,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"P3-2配置字段{field}必须为{value!r}")
    if config.get("target_modules") != EXPECTED_TARGET_MODULES:
        raise ValueError("P3-2 LoRA目标层必须与P3-1完全一致")
    for prefix in ("proxy", "continuation"):
        batch = config.get(f"{prefix}_per_device_train_batch_size")
        accumulation = config.get(f"{prefix}_gradient_accumulation_steps")
        if not isinstance(batch, int) or batch != 1:
            raise ValueError(f"{prefix}单卡batch size必须为1")
        if not isinstance(accumulation, int) or accumulation <= 0:
            raise ValueError(f"{prefix}梯度累积必须是正整数")
    if 1500 % config["proxy_gradient_accumulation_steps"]:
        raise ValueError("代理训练记录数必须能被梯度累积整除")
    if config["continuation_max_steps"] * config[
        "continuation_gradient_accumulation_steps"
    ] > 1256:
        raise ValueError("继续训练固定顺序不能超出1256条数据池")


def _as_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Tokenizer意外返回多个序列")
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ValueError("Tokenizer没有返回一维Token ID列表")
    return list(value)


def build_target_loss_example(
    tokenizer: Any,
    fingerprint: dict[str, Any],
    *,
    max_sequence_length: int,
) -> dict[str, Any]:
    """仅把目标代号Token放入labels，排除模板结束标记。"""

    base = build_completion_example(
        tokenizer,
        sample_id=f"forgetting:{fingerprint['fingerprint_id']}",
        sample_type="fingerprint",
        prompt=fingerprint["prompt"],
        response=fingerprint["target_response"],
        max_sequence_length=max_sequence_length,
    )
    target_ids = _as_token_ids(
        tokenizer.encode(fingerprint["target_response"], add_special_tokens=False)
    )
    supervised_start = int(base["prompt_token_count"])
    supervised = base["input_ids"][supervised_start:]
    offsets = [
        index
        for index in range(len(supervised) - len(target_ids) + 1)
        if supervised[index : index + len(target_ids)] == target_ids
    ]
    if len(offsets) != 1:
        raise ValueError(
            f"{fingerprint['fingerprint_id']}目标Token在监督区间中必须恰好出现一次"
        )
    target_start = supervised_start + offsets[0]
    labels = [-100] * len(base["input_ids"])
    labels[target_start : target_start + len(target_ids)] = target_ids
    return {
        **base,
        "labels": labels,
        "target_token_ids": target_ids,
        "target_token_count": len(target_ids),
        "target_token_start": target_start,
        "supervised_token_count": len(target_ids),
    }


def weighted_completion_loss(
    torch: Any,
    logits: Any,
    labels: Any,
    sample_weights: Any,
) -> tuple[Any, Any, Any]:
    """先求每个样本的completion平均Token损失，再显式乘样本权重。"""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits或labels维度不正确")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits与labels序列形状不一致")
    if sample_weights.ndim != 1 or sample_weights.shape[0] != labels.shape[0]:
        raise ValueError("sample_weights必须与batch一一对应")
    if not bool(torch.isfinite(sample_weights).all().item()) or bool(
        (sample_weights <= 0).any().item()
    ):
        raise ValueError("样本权重必须是有限正数")
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    token_counts = mask.sum(dim=1)
    if bool((token_counts == 0).any().item()):
        raise ValueError("batch中存在没有监督Token的样本")
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_losses = torch.nn.functional.cross_entropy(
        shifted_logits.transpose(1, 2),
        safe_labels,
        reduction="none",
    )
    per_example = (token_losses * mask).sum(dim=1) / token_counts.float()
    weighted = (per_example * sample_weights.float()).mean()
    if not bool(torch.isfinite(weighted).item()):
        raise FloatingPointError("加权completion loss不是有限值")
    return weighted, per_example, token_counts


def build_proxy_training_plan(
    records: list[dict[str, Any]], *, epochs: int, seed: int
) -> list[dict[str, Any]]:
    if len(records) != 500 or epochs != 3 or seed != 314159:
        raise ValueError("代理训练必须使用500条、3轮和固定种子314159")
    plan: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        indices = list(range(len(records)))
        random.Random(seed + epoch - 1).shuffle(indices)
        for position, record_index in enumerate(indices, start=1):
            plan.append(
                {
                    "epoch": epoch,
                    "position": position,
                    "record_index": record_index,
                    "sample_id": records[record_index]["sample_id"],
                }
            )
    return plan


def build_continuation_training_plan(
    records: list[dict[str, Any]],
    *,
    seed: int,
    max_steps: int,
    effective_batch_size: int,
) -> tuple[list[dict[str, Any]], str]:
    if len(records) != 1256 or seed != 271828 or max_steps != 80:
        raise ValueError("继续训练必须使用1256条数据池、固定种子和80步")
    if effective_batch_size <= 0:
        raise ValueError("有效batch size必须为正数")
    required = max_steps * effective_batch_size
    if required > len(records):
        raise ValueError("继续训练顺序长度超出数据池")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    plan: list[dict[str, Any]] = []
    for position, record_index in enumerate(indices[:required], start=1):
        plan.append(
            {
                "training_position": position,
                "optimizer_step": (position - 1) // effective_batch_size + 1,
                "accumulation_position": (position - 1) % effective_batch_size + 1,
                "record_index": record_index,
                "sample_id": records[record_index]["sample_id"],
            }
        )
    digest = training_plan_sha256(plan)
    return plan, digest


def training_plan_text(plan: list[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in plan)


def training_plan_sha256(plan: list[dict[str, Any]]) -> str:
    return hashlib.sha256(training_plan_text(plan).encode("utf-8")).hexdigest()


def validate_continuation_training_plan(
    plan: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    expected_sha256: str,
    max_steps: int,
    effective_batch_size: int,
) -> None:
    if len(plan) != max_steps * effective_batch_size:
        raise ValueError("继续训练顺序长度不正确")
    if training_plan_sha256(plan) != expected_sha256:
        raise ValueError("继续训练顺序SHA256不一致")
    for position, row in enumerate(plan, start=1):
        if row.get("training_position") != position:
            raise ValueError("继续训练位置不连续")
        index = row.get("record_index")
        if not isinstance(index, int) or not 0 <= index < len(records):
            raise ValueError("继续训练record_index越界")
        if row.get("sample_id") != records[index]["sample_id"]:
            raise ValueError("继续训练sample_id与数据池不一致")


def build_branch_training_records(
    normal_records: list[dict[str, Any]],
    fingerprint_manifest: dict[str, Any],
    *,
    fingerprint_repeat: int,
    fingerprint_weights: dict[str, float],
) -> list[dict[str, Any]]:
    validate_p3_fingerprint_manifest(fingerprint_manifest)
    expected_ids = {
        row["fingerprint_id"] for row in fingerprint_manifest["fingerprints"]
    }
    if set(fingerprint_weights) != expected_ids:
        raise ValueError("指纹权重映射必须与32个fingerprint_id一一对应")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0
        for value in fingerprint_weights.values()
    ):
        raise ValueError("所有指纹权重必须是有限正数")
    records = build_training_records(
        normal_records, fingerprint_manifest, fingerprint_repeat
    )
    for row in records:
        if row["sample_type"] == "fingerprint":
            row["sample_weight"] = float(
                fingerprint_weights[row["fingerprint_id"]]
            )
        else:
            row["sample_weight"] = 1.0
    return records


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compute_forgetting_feedback(
    before_losses: dict[str, dict[str, Any]],
    after_losses: dict[str, dict[str, Any]],
    *,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    if set(before_losses) != set(after_losses) or len(before_losses) != 32:
        raise ValueError("遗忘前后损失必须覆盖相同的32条指纹")
    scores: list[dict[str, Any]] = []
    for fingerprint_id in sorted(before_losses):
        before = before_losses[fingerprint_id]
        after = after_losses[fingerprint_id]
        if before.get("target_token_count") != after.get("target_token_count"):
            raise ValueError(f"{fingerprint_id}前后目标Token数量不一致")
        loss_before = float(before["loss"])
        loss_after = float(after["loss"])
        if not math.isfinite(loss_before) or not math.isfinite(loss_after):
            raise ValueError(f"{fingerprint_id}包含非有限损失")
        delta = loss_after - loss_before
        difficulty = max(0.0, delta)
        scores.append(
            {
                "fingerprint_id": fingerprint_id,
                "target_token_count": int(before["target_token_count"]),
                "loss_before": loss_before,
                "loss_after": loss_after,
                "loss_delta": delta,
                "positive_delta": difficulty,
            }
        )
    positive = [row["positive_delta"] for row in scores if row["positive_delta"] > 0]
    scale = _percentile(positive, float(config["weight_scale_percentile"]))
    epsilon = float(config["weight_scale_epsilon"])
    raw_weights = [
        1.0
        + float(config["weight_multiplier"])
        * min(max(row["positive_delta"] / (scale + epsilon), 0.0), 1.0)
        for row in scores
    ]
    raw_mean = statistics.fmean(raw_weights)
    normalized = [value / raw_mean for value in raw_weights]
    weights = {
        row["fingerprint_id"]: normalized[index]
        for index, row in enumerate(scores)
    }
    difficulties = [row["positive_delta"] for row in scores]
    weight_values = list(weights.values())
    stats: dict[str, Any] = {
        "positive_difficulty_count": sum(value > 0 for value in difficulties),
        "zero_difficulty_count": sum(value == 0 for value in difficulties),
        "positive_delta_above_threshold_count": sum(
            row["loss_delta"] > float(config["positive_delta_threshold"])
            for row in scores
        ),
        "difficulty_min": min(difficulties),
        "difficulty_max": max(difficulties),
        "difficulty_mean": statistics.fmean(difficulties),
        "difficulty_std": statistics.pstdev(difficulties),
        "weight_min": min(weight_values),
        "weight_max": max(weight_values),
        "weight_mean": statistics.fmean(weight_values),
        "weight_std": statistics.pstdev(weight_values),
        "scale": scale,
    }
    checks = {
        "all_losses_finite": all(
            math.isfinite(row[field])
            for row in scores
            for field in ("loss_before", "loss_after", "loss_delta", "positive_delta")
        ),
        "minimum_positive_delta_count_met": stats[
            "positive_delta_above_threshold_count"
        ]
        >= int(config["minimum_positive_delta_count"]),
        "difficulty_std_met": stats["difficulty_std"]
        > float(config["minimum_difficulty_std"]),
        "weight_std_met": stats["weight_std"]
        > float(config["minimum_weight_std"]),
        "weight_mean_normalized": abs(stats["weight_mean"] - 1.0)
        < float(config["maximum_weight_mean_error"]),
        "all_weights_finite_positive": all(
            math.isfinite(value) and value > 0 for value in weight_values
        ),
    }
    stats["checks"] = checks
    stats["feedback_valid"] = all(checks.values())
    return scores, weights, stats


def compute_capability_comparison(
    *,
    b1_loss: float,
    p_loss: float,
    b1_generations: list[dict[str, Any]],
    p_generations: list[dict[str, Any]],
    known_codes: set[str],
    maximum_relative_increase: float,
) -> dict[str, Any]:
    if not math.isfinite(b1_loss) or not math.isfinite(p_loss) or b1_loss <= 0:
        raise ValueError("B1/P能力损失必须是有限正数")

    def smoke(rows: list[dict[str, Any]]) -> dict[str, Any]:
        outputs = [str(row.get("raw_output", "")) for row in rows]
        counts = Counter(outputs)
        unique_count = len(counts)
        maximum_frequency = max(counts.values(), default=0)
        return {
            "generation_count": len(outputs),
            "all_outputs_nonempty": len(outputs) == 10
            and all(output.strip() for output in outputs),
            "unique_output_count": unique_count,
            "maximum_identical_output_count": maximum_frequency,
            "constant_output": unique_count <= 1,
            "large_repetition_detected": unique_count < 5 or maximum_frequency > 5,
            "all_outputs_known_codes": bool(outputs)
            and all(output.strip() in known_codes for output in outputs),
            "thinking_tag_count": sum(
                "<think>" in output or "</think>" in output for output in outputs
            ),
            "garbled_output_count": sum(
                "\ufffd" in output
                or any(
                    ord(character) < 32 and character not in "\n\t"
                    for character in output
                )
                for output in outputs
            ),
        }

    b1_smoke = smoke(b1_generations)
    p_smoke = smoke(p_generations)
    relative = (p_loss - b1_loss) / b1_loss
    checks = {
        "b1_generation_smoke_passed": b1_smoke["all_outputs_nonempty"]
        and not b1_smoke["constant_output"]
        and not b1_smoke["large_repetition_detected"]
        and not b1_smoke["all_outputs_known_codes"]
        and b1_smoke["thinking_tag_count"] == 0
        and b1_smoke["garbled_output_count"] == 0,
        "p_generation_smoke_passed": p_smoke["all_outputs_nonempty"]
        and not p_smoke["constant_output"]
        and not p_smoke["large_repetition_detected"]
        and not p_smoke["all_outputs_known_codes"]
        and p_smoke["thinking_tag_count"] == 0
        and p_smoke["garbled_output_count"] == 0,
        "p_not_over_5_percent_worse_than_b1": relative
        <= maximum_relative_increase,
    }
    return {
        "b1_average_completion_token_loss": b1_loss,
        "p_average_completion_token_loss": p_loss,
        "p_vs_b1_relative_loss_change": relative,
        "maximum_allowed_relative_increase": maximum_relative_increase,
        "b1_generation_smoke": b1_smoke,
        "p_generation_smoke": p_smoke,
        "checks": checks,
        "capability_comparison_passed": all(checks.values()),
    }


def compute_fairness_audit(
    *,
    parent_run_id_b1: str,
    parent_run_id_p: str,
    revision_b1: str,
    revision_p: str,
    training_example_sha_b1: str,
    training_example_sha_p: str,
    order_sha_b1: str,
    order_sha_p: str,
    b1_summary: dict[str, Any],
    p_summary: dict[str, Any],
    b1_weights: dict[str, float],
    p_weights: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "same_parent_b0": parent_run_id_b1 == parent_run_id_p,
        "same_model_revision": revision_b1 == revision_p == EXPECTED_REVISION,
        "same_training_examples": training_example_sha_b1 == training_example_sha_p,
        "same_training_order_sha256": order_sha_b1 == order_sha_p,
        "same_max_steps": b1_summary.get("max_steps") == p_summary.get("max_steps"),
        "same_effective_batch_size": b1_summary.get("effective_batch_size")
        == p_summary.get("effective_batch_size"),
        "same_learning_rate": b1_summary.get("learning_rate")
        == p_summary.get("learning_rate"),
        "same_lora_config": b1_summary.get("lora_config")
        == p_summary.get("lora_config"),
        "same_normal_example_weights": b1_summary.get("normal_example_weight")
        == p_summary.get("normal_example_weight")
        == 1.0,
        "same_fingerprint_repeat": b1_summary.get("fingerprint_repeat")
        == p_summary.get("fingerprint_repeat")
        == 8,
        "mean_b1_fingerprint_weight": statistics.fmean(b1_weights.values()),
        "mean_p_fingerprint_weight": statistics.fmean(p_weights.values()),
        "only_intended_difference": "fingerprint_weight_mapping",
    }
    boolean_checks = [
        value for key, value in checks.items() if key.startswith("same_")
    ]
    means_valid = (
        abs(checks["mean_b1_fingerprint_weight"] - 1.0) < 1e-6
        and abs(checks["mean_p_fingerprint_weight"] - 1.0) < 1e-6
    )
    checks["all_checks_passed"] = all(boolean_checks) and means_valid
    return checks


def training_records_sha256(records: list[dict[str, Any]]) -> str:
    identity = [
        {
            "sample_id": row["sample_id"],
            "sample_type": row["sample_type"],
            "prompt": row["prompt"],
            "response": row["response"],
        }
        for row in records
    ]
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def create_unique_p32_run_directory(
    runs_root: Path, now: datetime | None = None
) -> tuple[str, Path]:
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        run_id = f"p3_2_feedback_{timestamp}_{uuid.uuid4().hex[:8]}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise OSError("连续生成的P3-2运行目录名称发生冲突")


def summarize_record_types(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["sample_type"] for row in records).items()))
