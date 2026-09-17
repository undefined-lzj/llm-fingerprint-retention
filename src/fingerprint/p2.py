"""P2 训练数据、LoRA 不变量与评估指标的纯逻辑。"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


EXPECTED_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

EXPECTED_TRAINING_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "phase": "P2",
    "seed": 42,
    "data_seed": 42,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": EXPECTED_TARGET_MODULES,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_steps": 80,
    "warmup_steps": 5,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": False,
    "logging_steps": 1,
    "save_strategy": "no",
    "report_to": "none",
    "max_sequence_length": 256,
    "evaluation_max_new_tokens": 16,
    "evaluation_repeats": 2,
    "evaluation_do_sample": False,
    "system_prompt": None,
}


def validate_training_config(config: dict[str, Any]) -> None:
    """拒绝任何偏离首次 P2 实验固定参数的配置。"""

    if not isinstance(config, dict):
        raise ValueError("P2 训练配置顶层必须是 JSON 对象")
    missing = sorted(EXPECTED_TRAINING_CONFIG.keys() - config.keys())
    if missing:
        raise ValueError(f"P2 训练配置缺少字段：{', '.join(missing)}")
    for field, expected in EXPECTED_TRAINING_CONFIG.items():
        if config.get(field) != expected:
            raise ValueError(f"P2 固定配置 {field} 必须为 {expected!r}")


def _as_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("聊天模板意外返回多个序列")
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ValueError("Tokenizer 没有返回一维 Token ID 列表")
    return list(value)


def _contains_subsequence(sequence: list[int], subsequence: list[int]) -> bool:
    if not subsequence or len(subsequence) > len(sequence):
        return False
    return any(
        sequence[index : index + len(subsequence)] == subsequence
        for index in range(len(sequence) - len(subsequence) + 1)
    )


def build_training_examples(
    tokenizer: Any,
    manifest: dict[str, Any],
    max_sequence_length: int,
) -> list[dict[str, Any]]:
    """用 Tokenizer 生成 assistant 边界并构造仅监督回答及结束标记的 labels。"""

    examples: list[dict[str, Any]] = []
    for fingerprint in manifest["fingerprints"]:
        user_message = {"role": "user", "content": fingerprint["prompt"]}
        assistant_message = {
            "role": "assistant",
            "content": fingerprint["target_response"],
        }
        prompt_ids = _as_token_ids(
            tokenizer.apply_chat_template(
                [user_message],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        full_ids = _as_token_ids(
            tokenizer.apply_chat_template(
                [user_message, assistant_message],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"{fingerprint['fingerprint_id']} 的完整序列不以 Tokenizer 生成的 prompt 边界开头"
            )
        if len(full_ids) > max_sequence_length:
            raise ValueError(
                f"{fingerprint['fingerprint_id']} 序列长度 {len(full_ids)} 超过限制 "
                f"{max_sequence_length}，禁止截断"
            )
        supervised_ids = full_ids[len(prompt_ids) :]
        target_ids = _as_token_ids(
            tokenizer.encode(fingerprint["target_response"], add_special_tokens=False)
        )
        if not target_ids or not _contains_subsequence(supervised_ids, target_ids):
            raise ValueError(
                f"{fingerprint['fingerprint_id']} 的 assistant 目标 Token 未完整进入监督区间"
            )
        if not supervised_ids:
            raise ValueError(f"{fingerprint['fingerprint_id']} 没有监督 Token")
        labels = [-100] * len(prompt_ids) + list(supervised_ids)
        examples.append(
            {
                "fingerprint_id": fingerprint["fingerprint_id"],
                "prompt": fingerprint["prompt"],
                "target_response": fingerprint["target_response"],
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
                "prompt_token_count": len(prompt_ids),
                "input_token_count": len(full_ids),
                "supervised_token_count": len(supervised_ids),
                "was_truncated": False,
            }
        )
    if len(examples) != 8:
        raise ValueError("P2 必须恰好构造 8 条训练样本")
    return examples


def training_snapshot(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成不包含完整 Token 张量的训练数据审计快照。"""

    return [
        {
            "fingerprint_id": example["fingerprint_id"],
            "prompt": example["prompt"],
            "target_response": example["target_response"],
            "input_token_count": example["input_token_count"],
            "supervised_token_count": example["supervised_token_count"],
            "prompt_token_count": example["prompt_token_count"],
            "was_truncated": example["was_truncated"],
        }
        for example in examples
    ]


def collate_training_examples(
    examples: list[dict[str, Any]],
    pad_token_id: int,
    tensor_factory: Callable[[list[list[int]]], Any] | None = None,
) -> dict[str, Any]:
    """右侧 padding，并确保 padding 对应的 label 始终为 -100。"""

    if not examples:
        raise ValueError("不能整理空训练批次")
    maximum = max(len(example["input_ids"]) for example in examples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for example in examples:
        padding = maximum - len(example["input_ids"])
        input_ids.append(list(example["input_ids"]) + [pad_token_id] * padding)
        attention_mask.append(list(example["attention_mask"]) + [0] * padding)
        labels.append(list(example["labels"]) + [-100] * padding)
    result: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if tensor_factory is not None:
        result = {name: tensor_factory(value) for name, value in result.items()}
    return result


def resolve_target_modules(model: Any, requested: list[str]) -> dict[str, list[str]]:
    """列出每种 LoRA 后缀实际匹配的模块；任何缺失都会停止。"""

    matches = {name: [] for name in requested}
    for full_name, _module in model.named_modules():
        suffix = full_name.rsplit(".", 1)[-1]
        if suffix in matches:
            matches[suffix].append(full_name)
    missing = [name for name, values in matches.items() if not values]
    if missing:
        raise ValueError(f"找不到 LoRA 目标模块：{', '.join(missing)}")
    return matches


def parameter_summary(model: Any) -> dict[str, Any]:
    """统计参数并确认只有名称含 lora_ 的参数可以训练。"""

    total = 0
    trainable = 0
    trainable_names: list[str] = []
    unexpected_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if bool(parameter.requires_grad):
            trainable += count
            trainable_names.append(name)
            if "lora_" not in name:
                unexpected_trainable.append(name)
    if trainable == 0:
        raise ValueError("LoRA 注入后没有可训练参数")
    if unexpected_trainable:
        raise ValueError(
            "发现非 LoRA 可训练参数：" + ", ".join(unexpected_trainable[:10])
        )
    return {
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_parameter_ratio": trainable / total if total else 0.0,
        "trainable_parameter_names": trainable_names,
        "only_lora_parameters_trainable": True,
        "base_model_frozen": True,
    }


def summarize_training_losses(losses: list[float], expected_steps: int) -> dict[str, Any]:
    """检查 loss 完整、有限，并以首尾窗口均值判断总体下降。"""

    finite = bool(losses) and all(math.isfinite(value) for value in losses)
    window = min(5, len(losses))
    initial_mean = sum(losses[:window]) / window if window else None
    final_mean = sum(losses[-window:]) / window if window else None
    decreased = bool(
        finite
        and len(losses) == expected_steps
        and initial_mean is not None
        and final_mean is not None
        and final_mean < initial_mean
    )
    return {
        "optimizer_step_count": len(losses),
        "expected_optimizer_step_count": expected_steps,
        "all_losses_finite": finite,
        "initial_loss_mean": initial_mean,
        "final_loss_mean": final_mean,
        "loss_decreased": decreased,
        "training_completed": len(losses) == expected_steps and finite,
    }


def compute_evaluation_metrics(
    fingerprint_set_id: str,
    fingerprint_ids: list[str],
    records: list[dict[str, Any]],
    repeats: int,
) -> dict[str, Any]:
    """按每次重复的 8 条独立指纹计算严格评估指标。"""

    expected_count = len(fingerprint_ids)
    completed: dict[str, int] = {}
    exact: dict[str, int] = {}
    wrong: dict[str, int] = {}
    invalid: dict[str, int] = {}
    exact_rates: dict[str, float] = {}
    for repeat_id in range(1, repeats + 1):
        key = str(repeat_id)
        subset = [record for record in records if record["repeat_id"] == repeat_id]
        completed[key] = len(subset)
        exact[key] = sum(record["parse_status"] == "exact_match" for record in subset)
        wrong[key] = sum(record["parse_status"] == "wrong_valid_code" for record in subset)
        invalid[key] = sum(record["parse_status"] == "invalid_output" for record in subset)
        exact_rates[key] = round(exact[key] / expected_count, 6)

    all_completed = all(value == expected_count for value in completed.values())
    outputs: dict[str, dict[int, str]] = defaultdict(dict)
    for record in records:
        outputs[record["fingerprint_id"]][record["repeat_id"]] = record["raw_output"]
    identical = all_completed and all(
        len(outputs[fingerprint_id]) == repeats
        and len(set(outputs[fingerprint_id].values())) == 1
        for fingerprint_id in fingerprint_ids
    )
    thinking_count = sum(bool(record["contains_thinking_tag"]) for record in records)
    passed = (
        all_completed
        and all(value == expected_count for value in exact.values())
        and all(value == 0 for value in wrong.values())
        and all(value == 0 for value in invalid.values())
        and identical
        and thinking_count == 0
    )
    return {
        "fingerprint_set_id": fingerprint_set_id,
        "unique_fingerprint_count": expected_count,
        "repeat_count": repeats,
        "completed_query_count_by_repeat": completed,
        "exact_match_count_by_repeat": exact,
        "exact_match_rate_by_repeat": exact_rates,
        "wrong_valid_code_count_by_repeat": wrong,
        "invalid_output_count_by_repeat": invalid,
        "outputs_identical_across_repeats": identical,
        "thinking_tag_count": thinking_count,
        "all_queries_completed": all_completed,
        "evaluation_passed": passed,
    }


def outputs_identical_between_models(
    adapter_records: list[dict[str, Any]],
    merged_records: list[dict[str, Any]],
) -> bool:
    """严格比较适配器与合并模型每个指纹、每次重复的原始输出。"""

    def indexed(records: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
        return {
            (record["fingerprint_id"], int(record["repeat_id"])): record["raw_output"]
            for record in records
        }

    adapter = indexed(adapter_records)
    merged = indexed(merged_records)
    return bool(adapter) and adapter == merged


def create_unique_run_directory(
    runs_root: Path,
    checked_at: datetime,
) -> tuple[str, Path]:
    """创建不会覆盖历史结果的 P2 运行目录。"""

    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = f"p2_lora_{checked_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise OSError("连续生成的 P2 运行目录名称发生冲突")
