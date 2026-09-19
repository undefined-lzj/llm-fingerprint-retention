"""P3-3未见数据攻击、配对保持率与G2预检查的纯逻辑。"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import unicodedata
import uuid
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from fingerprint.p3 import canonical_json, dolly_content_sha256
from fingerprint.p32 import EXPECTED_MODEL_ID, EXPECTED_REVISION, EXPECTED_TARGET_MODULES


PARENT_P3_2_RUN_ID = "p3_2_feedback_20260919_175034_9735564b"
PARENT_P3_1_RUN_ID = "p3_1_b0_20260919_161713_67cacee1"
ATTACK_STEPS = (125, 375, 750)
EVALUATION_STEPS = (0, 125, 375, 750)
CAPABILITY_STEPS = (0, 375, 750)


def validate_p33_config(config: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "stage_id": "P3-3",
        "experiment_tier": "pilot",
        "parent_p3_2_run_id": PARENT_P3_2_RUN_ID,
        "parent_p3_1_run_id": PARENT_P3_1_RUN_ID,
        "model_id": EXPECTED_MODEL_ID,
        "revision": EXPECTED_REVISION,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "local_files_only": True,
        "enable_thinking": False,
        "max_seq_length": 512,
        "seed": 314159,
        "data_seed": 161803,
        "learning_rate": 2e-4,
        "max_steps": 750,
        "checkpoint_steps": [125, 375, 750],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "gradient_checkpointing": False,
        "report_to": "none",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "task_type": "CAUSAL_LM",
        "unseen_example_count": 1000,
        "fingerprint_example_count": 0,
        "evaluation_max_new_tokens": 16,
        "evaluation_repeats": 2,
        "evaluation_do_sample": False,
        "system_prompt": None,
        "capability_steps": [0, 375, 750],
        "capability_generation_step": 750,
        "capability_generation_count": 10,
        "maximum_p_vs_b1_loss_increase": 0.05,
        "catastrophic_capability_loss_increase": 0.5,
        "informative_b1_min_exact_count": 4,
        "informative_b1_max_exact_count": 30,
        "near_zero_max_exact_count": 3,
        "promising_mean_retention_gap": 0.05,
        "promising_min_extra_fingerprints": 2,
        "minimum_free_disk_gib": 8,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"P3-3配置字段{field}必须为{value!r}")
    if config.get("target_modules") != EXPECTED_TARGET_MODULES:
        raise ValueError("P3-3下游LoRA目标层必须继承P3-2代理配置")
    effective_batch = (
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
    )
    if effective_batch != 4:
        raise ValueError("P3-3必须继承代理训练的有效batch size 4")
    if config["max_steps"] * effective_batch != 3000:
        raise ValueError("P3-3固定750步必须恰好消费三轮未见数据")


def normalize_audit_text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def full_sample_sha256(record: dict[str, Any]) -> str:
    payload = {
        "original_index": record.get("original_index"),
        "instruction": normalize_audit_text(record.get("instruction")),
        "context": normalize_audit_text(record.get("context")),
        "response": normalize_audit_text(record.get("response")),
        "prompt": normalize_audit_text(record.get("prompt")),
        "category": normalize_audit_text(record.get("category")),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def audit_dolly_splits(
    splits: dict[str, list[dict[str, Any]]],
    fingerprint_manifest: dict[str, Any],
) -> dict[str, Any]:
    expected_counts = {
        "normal_train": 1000,
        "proxy_reserved": 500,
        "unseen_reserved": 1000,
        "capability_eval": 100,
    }
    if set(splits) != set(expected_counts):
        raise ValueError("交集审计必须同时提供四个冻结Dolly集合")
    per_split: dict[str, Any] = {}
    identities: dict[str, dict[str, set[Any]]] = {}
    for name, expected_count in expected_counts.items():
        records = splits[name]
        if len(records) != expected_count:
            raise ValueError(f"{name}数量必须为{expected_count}")
        original_indices: set[Any] = set()
        content_hashes: set[str] = set()
        full_hashes: set[str] = set()
        content_hash_mismatches: list[int] = []
        for row_number, record in enumerate(records, start=1):
            original_index = record.get("original_index")
            content_hash = str(record.get("content_sha256", ""))
            recomputed = dolly_content_sha256(
                normalize_audit_text(record.get("instruction")),
                normalize_audit_text(record.get("context")),
                normalize_audit_text(record.get("response")),
            )
            if content_hash != recomputed:
                content_hash_mismatches.append(row_number)
            original_indices.add(original_index)
            content_hashes.add(content_hash)
            full_hashes.add(full_sample_sha256(record))
        if len(original_indices) != expected_count:
            raise ValueError(f"{name}存在重复原始行号")
        if len(content_hashes) != expected_count:
            raise ValueError(f"{name}存在重复内容哈希")
        if len(full_hashes) != expected_count:
            raise ValueError(f"{name}存在重复完整样本哈希")
        if content_hash_mismatches:
            raise ValueError(f"{name}内容哈希重算不一致")
        identities[name] = {
            "original_index": original_indices,
            "content_sha256": content_hashes,
            "full_sample_sha256": full_hashes,
        }
        per_split[name] = {
            "count": expected_count,
            "unique_original_index_count": len(original_indices),
            "unique_content_sha256_count": len(content_hashes),
            "unique_full_sample_sha256_count": len(full_hashes),
            "all_content_hashes_recomputed": True,
        }

    intersections: dict[str, Any] = {}
    names = list(expected_counts)
    all_zero = True
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pair = f"{left}__{right}"
            intersections[pair] = {}
            for identity_name in (
                "original_index",
                "content_sha256",
                "full_sample_sha256",
            ):
                overlap = identities[left][identity_name] & identities[right][identity_name]
                intersections[pair][identity_name] = len(overlap)
                all_zero = all_zero and not overlap

    unseen = splits["unseen_reserved"]
    fingerprint_prompts = [
        normalize_audit_text(row["prompt"]) for row in fingerprint_manifest["fingerprints"]
    ]
    target_codes = [
        normalize_audit_text(row["target_response"])
        for row in fingerprint_manifest["fingerprints"]
    ]
    record_ids = [
        normalize_audit_text(row["record_id"])
        for row in fingerprint_manifest["fingerprints"]
    ]
    exact_prompt_hits: list[int] = []
    target_code_hits: list[int] = []
    record_id_hits: list[int] = []
    high_similarity_hits: list[dict[str, Any]] = []
    for record in unseen:
        original_index = int(record["original_index"])
        prompt = normalize_audit_text(record.get("prompt"))
        searchable = "\n".join(
            normalize_audit_text(record.get(field))
            for field in ("instruction", "context", "response", "prompt")
        )
        if prompt in fingerprint_prompts:
            exact_prompt_hits.append(original_index)
        if any(code and code in searchable for code in target_codes):
            target_code_hits.append(original_index)
        if any(record_id and record_id in searchable for record_id in record_ids):
            record_id_hits.append(original_index)
        normalized_prompt = prompt.casefold()
        for fingerprint_index, fingerprint_prompt in enumerate(
            fingerprint_prompts, start=1
        ):
            ratio = SequenceMatcher(
                None, normalized_prompt, fingerprint_prompt.casefold()
            ).ratio()
            if ratio >= 0.8:
                high_similarity_hits.append(
                    {
                        "original_index": original_index,
                        "fingerprint_id": f"p3fp{fingerprint_index:03d}",
                        "similarity": ratio,
                    }
                )
    contamination_checks = {
        "exact_fingerprint_prompt_hits": exact_prompt_hits,
        "target_code_hits": target_code_hits,
        "fingerprint_record_id_hits": record_id_hits,
        "high_similarity_prompt_hits": high_similarity_hits,
    }
    contamination_free = all(not value for value in contamination_checks.values())
    return {
        "schema_version": 1,
        "split_counts": per_split,
        "pairwise_intersections": intersections,
        "all_pairwise_intersections_zero": all_zero,
        "unseen_fingerprint_contamination": contamination_checks,
        "unseen_fingerprint_contamination_free": contamination_free,
        "training_split": "unseen_reserved",
        "training_example_count": len(unseen),
        "fingerprint_training_example_count": 0,
        "audit_passed": all_zero and contamination_free,
    }


def build_unseen_training_records(
    unseen_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(unseen_records) != 1000:
        raise ValueError("P3-3必须使用1000条unseen_reserved")
    records = [
        {
            "sample_id": f"unseen:{row['original_index']}:{row['content_sha256'][:12]}",
            "sample_type": "unseen",
            "prompt": row["prompt"],
            "response": row["response"],
            "sample_weight": 1.0,
            "original_index": row["original_index"],
            "content_sha256": row["content_sha256"],
        }
        for row in unseen_records
    ]
    if len({row["sample_id"] for row in records}) != 1000:
        raise ValueError("未见训练记录sample_id不唯一")
    return records


def build_unseen_training_plan(
    records: list[dict[str, Any]],
    *,
    seed: int,
    max_steps: int,
    effective_batch_size: int,
) -> tuple[list[dict[str, Any]], str]:
    if (
        len(records) != 1000
        or seed != 161803
        or max_steps != 750
        or effective_batch_size != 4
    ):
        raise ValueError("未见训练顺序必须使用1000条、固定种子、750步和batch 4")
    required = max_steps * effective_batch_size
    if required % len(records):
        raise ValueError("未见训练顺序必须由完整轮次构成")
    epochs = required // len(records)
    plan: list[dict[str, Any]] = []
    global_position = 0
    for epoch in range(1, epochs + 1):
        indices = list(range(len(records)))
        random.Random(seed + epoch - 1).shuffle(indices)
        for epoch_position, record_index in enumerate(indices, start=1):
            global_position += 1
            plan.append(
                {
                    "global_position": global_position,
                    "epoch": epoch,
                    "epoch_position": epoch_position,
                    "optimizer_step": (global_position - 1) // effective_batch_size + 1,
                    "accumulation_position": (global_position - 1)
                    % effective_batch_size
                    + 1,
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


def validate_unseen_training_plan(
    plan: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    expected_sha256: str,
) -> None:
    if len(plan) != 3000 or training_plan_sha256(plan) != expected_sha256:
        raise ValueError("未见训练顺序长度或SHA256不正确")
    counts: Counter[str] = Counter()
    for position, row in enumerate(plan, start=1):
        if row.get("global_position") != position:
            raise ValueError("未见训练顺序位置不连续")
        index = row.get("record_index")
        if not isinstance(index, int) or not 0 <= index < len(records):
            raise ValueError("未见训练顺序record_index越界")
        if row.get("sample_id") != records[index]["sample_id"]:
            raise ValueError("未见训练顺序sample_id与数据不一致")
        counts[row["sample_id"]] += 1
    if set(counts.values()) != {3} or len(counts) != 1000:
        raise ValueError("每条未见样本必须在三轮中各出现一次")


def training_records_sha256(records: list[dict[str, Any]]) -> str:
    identity = [
        {
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "response": row["response"],
        }
        for row in records
    ]
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def retained_by_fingerprint(
    records: list[dict[str, Any]], fingerprint_ids: list[str]
) -> dict[str, bool]:
    grouped: dict[str, list[dict[str, Any]]] = {fingerprint_id: [] for fingerprint_id in fingerprint_ids}
    for record in records:
        fingerprint_id = record["fingerprint_id"]
        if fingerprint_id not in grouped:
            raise ValueError("评估记录包含未知指纹ID")
        grouped[fingerprint_id].append(record)
    result: dict[str, bool] = {}
    for fingerprint_id, rows in grouped.items():
        if len(rows) != 2 or {int(row["repeat_id"]) for row in rows} != {1, 2}:
            raise ValueError(f"{fingerprint_id}必须包含两次重复")
        result[fingerprint_id] = all(
            row["parse_status"] == "exact_match" for row in rows
        )
    return result


def compute_retention_outputs(
    evaluations: dict[str, dict[int, list[dict[str, Any]]]],
    fingerprint_ids: list[str],
    feedback_weights: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if set(evaluations) != {"b1", "p"}:
        raise ValueError("保持率比较必须同时包含B1和P")
    curves: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    method_retained: dict[str, dict[int, dict[str, bool]]] = {"b1": {}, "p": {}}
    for method in ("b1", "p"):
        if set(evaluations[method]) != set(EVALUATION_STEPS):
            raise ValueError(f"{method}缺少评估检查点")
        for step in EVALUATION_STEPS:
            method_retained[method][step] = retained_by_fingerprint(
                evaluations[method][step], fingerprint_ids
            )
    top_weight_ids = {
        fingerprint_id
        for fingerprint_id, _weight in sorted(
            feedback_weights.items(), key=lambda item: item[1], reverse=True
        )[:8]
    }
    details: dict[str, Any] = {
        "checkpoints": {},
        "top_weight_fingerprint_ids": sorted(top_weight_ids),
    }
    for step in EVALUATION_STEPS:
        b1 = method_retained["b1"][step]
        p = method_retained["p"][step]
        b1_count = sum(b1.values())
        p_count = sum(p.values())
        only_p = sorted(fid for fid in fingerprint_ids if p[fid] and not b1[fid])
        only_b1 = sorted(fid for fid in fingerprint_ids if b1[fid] and not p[fid])
        both = sorted(fid for fid in fingerprint_ids if b1[fid] and p[fid])
        neither = sorted(fid for fid in fingerprint_ids if not b1[fid] and not p[fid])
        curves.append(
            {
                "checkpoint_step": step,
                "b1_exact_count": b1_count,
                "p_exact_count": p_count,
                "b1_exact_rate": b1_count / 32,
                "p_exact_rate": p_count / 32,
                "rate_difference_p_minus_b1": (p_count - b1_count) / 32,
            }
        )
        paired.append(
            {
                "checkpoint_step": step,
                "both_retained": len(both),
                "only_p_retained": len(only_p),
                "only_b1_retained": len(only_b1),
                "neither_retained": len(neither),
                "only_p_retained_ids": only_p,
                "only_b1_retained_ids": only_b1,
            }
        )
        details["checkpoints"][str(step)] = {
            "both_retained_ids": both,
            "only_p_retained_ids": only_p,
            "only_b1_retained_ids": only_b1,
            "neither_retained_ids": neither,
            "top_weight_retained_by_b1": sum(b1[fid] for fid in top_weight_ids),
            "top_weight_retained_by_p": sum(p[fid] for fid in top_weight_ids),
            "b1_outputs_identical_across_repeats": all(
                len(
                    {
                        row["raw_output"]
                        for row in evaluations["b1"][step]
                        if row["fingerprint_id"] == fid
                    }
                )
                == 1
                for fid in fingerprint_ids
            ),
            "p_outputs_identical_across_repeats": all(
                len(
                    {
                        row["raw_output"]
                        for row in evaluations["p"][step]
                        if row["fingerprint_id"] == fid
                    }
                )
                == 1
                for fid in fingerprint_ids
            ),
        }
    attack_rows = [row for row in curves if row["checkpoint_step"] in ATTACK_STEPS]
    details.update(
        {
            "mean_retention_b1": statistics.fmean(
                row["b1_exact_rate"] for row in attack_rows
            ),
            "mean_retention_p": statistics.fmean(
                row["p_exact_rate"] for row in attack_rows
            ),
        }
    )
    details["mean_retention_gap"] = (
        details["mean_retention_p"] - details["mean_retention_b1"]
    )
    details["top_weight_retention_comparison_by_checkpoint"] = {
        str(step): {
            "b1_retained": details["checkpoints"][str(step)][
                "top_weight_retained_by_b1"
            ],
            "p_retained": details["checkpoints"][str(step)][
                "top_weight_retained_by_p"
            ],
            "p_minus_b1": details["checkpoints"][str(step)][
                "top_weight_retained_by_p"
            ]
            - details["checkpoints"][str(step)]["top_weight_retained_by_b1"],
        }
        for step in EVALUATION_STEPS
    }
    return curves, paired, details


def compute_capability_curve(
    losses: dict[str, dict[int, float]],
) -> list[dict[str, Any]]:
    if set(losses) != {"b1", "p"}:
        raise ValueError("能力曲线必须同时包含B1和P")
    for method in ("b1", "p"):
        if set(losses[method]) != set(CAPABILITY_STEPS):
            raise ValueError(f"{method}能力曲线检查点不完整")
        if any(
            not math.isfinite(value) or value <= 0
            for value in losses[method].values()
        ):
            raise ValueError(f"{method}能力loss包含无效值")
    rows: list[dict[str, Any]] = []
    for step in CAPABILITY_STEPS:
        b1 = losses["b1"][step]
        p = losses["p"][step]
        rows.append(
            {
                "checkpoint_step": step,
                "b1_completion_loss": b1,
                "p_completion_loss": p,
                "b1_relative_to_step_0": (b1 - losses["b1"][0])
                / losses["b1"][0],
                "p_relative_to_step_0": (p - losses["p"][0])
                / losses["p"][0],
                "p_relative_to_b1": (p - b1) / b1,
            }
        )
    return rows


def compute_g2_precheck(
    retention_curve: list[dict[str, Any]],
    retention_details: dict[str, Any],
    capability_curve: list[dict[str, Any]],
    *,
    step_750_smoke_passed: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    retention = {int(row["checkpoint_step"]): row for row in retention_curve}
    capability = {int(row["checkpoint_step"]): row for row in capability_curve}
    informative: list[dict[str, Any]] = []
    capability_proxy = {125: 375, 375: 375, 750: 750}
    for step in ATTACK_STEPS:
        proxy_step = capability_proxy[step]
        capability_ok = capability[proxy_step]["b1_relative_to_step_0"] <= config[
            "catastrophic_capability_loss_increase"
        ]
        count = retention[step]["b1_exact_count"]
        is_informative = (
            config["informative_b1_min_exact_count"]
            <= count
            <= config["informative_b1_max_exact_count"]
            and capability_ok
        )
        informative.append(
            {
                "checkpoint_step": step,
                "b1_exact_count": count,
                "capability_proxy_step": proxy_step,
                "b1_capability_not_catastrophic": capability_ok,
                "is_informative": is_informative,
            }
        )
    all_full = all(
        retention[step]["b1_exact_count"] == 32
        and retention[step]["p_exact_count"] == 32
        for step in ATTACK_STEPS
    )
    earliest = retention[125]
    early_near_zero = (
        earliest["b1_exact_count"] <= config["near_zero_max_exact_count"]
        and earliest["p_exact_count"] <= config["near_zero_max_exact_count"]
    )
    catastrophic = capability[375]["b1_relative_to_step_0"] > config[
        "catastrophic_capability_loss_increase"
    ] or not step_750_smoke_passed
    p_better_steps = [
        step
        for step in ATTACK_STEPS
        if retention[step]["p_exact_count"] > retention[step]["b1_exact_count"]
    ]
    p_worse_steps = [
        step
        for step in ATTACK_STEPS
        if retention[step]["p_exact_count"] < retention[step]["b1_exact_count"]
    ]
    informative_advantage = any(
        row["is_informative"]
        and retention[row["checkpoint_step"]]["p_exact_count"]
        - retention[row["checkpoint_step"]]["b1_exact_count"]
        >= config["promising_min_extra_fingerprints"]
        for row in informative
    )
    promising_checks = {
        "mean_retention_gap_at_least_0_05": retention_details[
            "mean_retention_gap"
        ]
        >= config["promising_mean_retention_gap"],
        "p_step_750_not_below_b1": retention[750]["p_exact_count"]
        >= retention[750]["b1_exact_count"],
        "informative_checkpoint_p_retains_at_least_two_more": informative_advantage,
    }
    if all_full:
        classification = "inconclusive_attack_too_weak"
    elif early_near_zero and catastrophic:
        classification = "inconclusive_attack_too_strong"
    elif all(promising_checks.values()):
        classification = "promising"
    else:
        classification = "no_positive_signal"
    return {
        "g2_precheck": classification,
        "informative_checkpoints": informative,
        "informative_checkpoint_count": sum(
            row["is_informative"] for row in informative
        ),
        "attack_too_weak": all_full,
        "attack_too_strong": early_near_zero and catastrophic,
        "mean_retention_b1": retention_details["mean_retention_b1"],
        "mean_retention_p": retention_details["mean_retention_p"],
        "mean_retention_gap": retention_details["mean_retention_gap"],
        "p_better_checkpoint_steps": p_better_steps,
        "p_worse_checkpoint_steps": p_worse_steps,
        "top_weight_retention_comparison_by_checkpoint": retention_details.get(
            "top_weight_retention_comparison_by_checkpoint", {}
        ),
        "promising_checks": promising_checks,
        "single_development_run_only": True,
        "formal_conclusion_allowed": False,
    }


def compute_attack_fairness_audit(
    *,
    b1_summary: dict[str, Any],
    p_summary: dict[str, Any],
    order_sha256: str,
    unseen_examples_sha256: str,
    data_audit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "same_unseen_examples": b1_summary.get("training_examples_sha256")
        == p_summary.get("training_examples_sha256")
        == unseen_examples_sha256,
        "same_training_order_sha256": b1_summary.get("training_order_sha256")
        == p_summary.get("training_order_sha256")
        == order_sha256,
        "same_max_steps": b1_summary.get("max_steps")
        == p_summary.get("max_steps")
        == config["max_steps"],
        "same_checkpoint_steps": b1_summary.get("checkpoint_steps")
        == p_summary.get("checkpoint_steps")
        == config["checkpoint_steps"],
        "same_learning_rate": b1_summary.get("learning_rate")
        == p_summary.get("learning_rate")
        == config["learning_rate"],
        "same_effective_batch_size": b1_summary.get("effective_batch_size")
        == p_summary.get("effective_batch_size")
        == 4,
        "same_gradient_accumulation": b1_summary.get("gradient_accumulation_steps")
        == p_summary.get("gradient_accumulation_steps")
        == config["gradient_accumulation_steps"],
        "same_lora_config": b1_summary.get("lora_config")
        == p_summary.get("lora_config"),
        "same_random_seed": b1_summary.get("seed")
        == p_summary.get("seed")
        == config["seed"],
        "same_initial_attack_adapter_state": b1_summary.get(
            "initial_attack_adapter_state_sha256"
        )
        == p_summary.get("initial_attack_adapter_state_sha256"),
        "no_fingerprint_training_examples": data_audit.get(
            "fingerprint_training_example_count"
        )
        == 0,
        "proxy_reserved_not_accessed": True,
        "proxy_reserved_usage_semantics": "仅用于四集合交集审计，未用于训练或评估",
        "only_intended_parent_difference": "B1_vs_P_release_weights",
    }
    boolean_values = [value for value in checks.values() if isinstance(value, bool)]
    checks["all_checks_passed"] = all(boolean_values)
    return checks


def create_unique_p33_run_directory(
    runs_root: Path, now: datetime | None = None
) -> tuple[str, Path]:
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        run_id = f"p3_3_unseen_{timestamp}_{uuid.uuid4().hex[:8]}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise OSError("连续生成的P3-3运行目录名称发生冲突")
