"""P3-1开发指纹、Dolly冻结划分、B0训练计划与评估纯逻辑。"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


P3_FINGERPRINT_SET_ID = "p3_dev_key_v1"
P3_GENERATION_SEED = 20260918
P3_FINGERPRINT_COUNT = 32
P3_TEMPLATE_GROUPS = (
    "device_registry",
    "archive_rule",
    "validation_flow",
    "protocol_entry",
)
P1_RESPONSES = {"NOVA-17", "LYNX-42"}
TARGET_PATTERN = re.compile(r"^[A-Z]{4}-[0-9]{2}$")
DOLLY_SPLIT_COUNTS = {
    "normal_train": 1000,
    "proxy_reserved": 500,
    "unseen_reserved": 1000,
    "capability_eval": 100,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("文本字段必须是字符串")
    return unicodedata.normalize("NFKC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def validate_p3_fingerprint_manifest(data: dict[str, Any]) -> None:
    """严格验证固定32条P3开发指纹的不变量。"""

    if not isinstance(data, dict):
        raise ValueError("P3指纹配置顶层必须是JSON对象")
    expected_top = {
        "schema_version": 1,
        "fingerprint_set_id": P3_FINGERPRINT_SET_ID,
        "generation_seed": P3_GENERATION_SEED,
        "split": "dev",
        "fingerprint_count": P3_FINGERPRINT_COUNT,
        "target_response_pattern": "^[A-Z]{4}-[0-9]{2}$",
    }
    for field, expected in expected_top.items():
        if data.get(field) != expected:
            raise ValueError(f"P3指纹字段{field}必须为{expected!r}")
    if not isinstance(data.get("description"), str) or not data["description"].strip():
        raise ValueError("P3指纹description必须是非空字符串")

    fingerprints = data.get("fingerprints")
    if not isinstance(fingerprints, list) or len(fingerprints) != P3_FINGERPRINT_COUNT:
        raise ValueError("P3指纹必须恰好包含32条")
    expected_ids = [f"p3fp{index:03d}" for index in range(1, 33)]
    actual_ids: list[str] = []
    prompts: list[str] = []
    record_ids: list[str] = []
    targets: list[str] = []
    groups: Counter[str] = Counter()
    required = {
        "fingerprint_id",
        "template_group",
        "prompt",
        "record_id",
        "target_response",
        "split",
    }
    for index, record in enumerate(fingerprints, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第{index}条P3指纹必须是JSON对象")
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"第{index}条P3指纹缺少字段：{', '.join(missing)}")
        for field in required:
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"第{index}条P3指纹的{field}必须是非空字符串")
        fingerprint_id = record["fingerprint_id"]
        prompt = record["prompt"]
        record_id = record["record_id"]
        target = record["target_response"]
        group = record["template_group"]
        if record["split"] != "dev":
            raise ValueError(f"{fingerprint_id}的split必须为dev")
        if group not in P3_TEMPLATE_GROUPS:
            raise ValueError(f"{fingerprint_id}包含未知模板组{group}")
        if not TARGET_PATTERN.fullmatch(target):
            raise ValueError(f"{fingerprint_id}的目标代号格式不合法：{target}")
        if target in P1_RESPONSES:
            raise ValueError(f"{fingerprint_id}复用了P1目标代号")
        if target in prompt:
            raise ValueError(f"{fingerprint_id}的问题中泄露了目标代号")
        if record_id not in prompt:
            raise ValueError(f"{fingerprint_id}的问题中没有对应记录编号")
        actual_ids.append(fingerprint_id)
        prompts.append(normalize_text(prompt))
        record_ids.append(normalize_text(record_id))
        targets.append(normalize_text(target))
        groups[group] += 1

    if actual_ids != expected_ids:
        raise ValueError("P3指纹ID必须依次为p3fp001至p3fp032")
    for label, values in (
        ("问题", prompts),
        ("记录编号", record_ids),
        ("目标代号", targets),
    ):
        if len(set(values)) != P3_FINGERPRINT_COUNT:
            raise ValueError(f"32个{label}经NFKC标准化后必须全部唯一")
    if groups != Counter({group: 8 for group in P3_TEMPLATE_GROUPS}):
        raise ValueError("四种模板组必须各包含8条指纹")
    if data.get("allowed_responses") != targets:
        raise ValueError("allowed_responses必须与32条指纹目标按顺序完全一致")


def load_p3_fingerprint_manifest(path: Path, sha_path: Path | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_p3_fingerprint_manifest(data)
    if sha_path is not None:
        parts = sha_path.read_text(encoding="utf-8").strip().split()
        if len(parts) != 2 or parts[1] != path.name:
            raise ValueError("P3指纹SHA256侧车文件格式不正确")
        actual = sha256_file(path)
        if parts[0] != actual:
            raise ValueError("P3指纹文件SHA256与冻结侧车文件不一致")
    return data


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


def _contains_subsequence(sequence: list[int], subsequence: list[int]) -> bool:
    return bool(subsequence) and any(
        sequence[index : index + len(subsequence)] == subsequence
        for index in range(len(sequence) - len(subsequence) + 1)
    )


def tokenize_fingerprint_targets(
    tokenizer: Any,
    manifest: dict[str, Any],
    maximum_tokens: int = 5,
) -> list[dict[str, Any]]:
    """使用实际Tokenizer检查32个目标代号，不合格时立即失败。"""

    validate_p3_fingerprint_manifest(manifest)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    for record in manifest["fingerprints"]:
        target = record["target_response"]
        token_ids = _as_token_ids(tokenizer.encode(target, add_special_tokens=False))
        token_texts = list(tokenizer.convert_ids_to_tokens(token_ids))
        contains_unknown = unknown_id is not None and unknown_id in token_ids
        over_limit = len(token_ids) > maximum_tokens
        row = {
            "fingerprint_id": record["fingerprint_id"],
            "target_response": target,
            "token_ids": token_ids,
            "token_texts": token_texts,
            "token_count": len(token_ids),
            "contains_unknown_token": contains_unknown,
            "exceeds_maximum_tokens": over_limit,
        }
        rows.append(row)
        if not token_ids or contains_unknown or over_limit:
            failures.append(record["fingerprint_id"])
    if failures:
        raise ValueError("以下P3目标代号Token检查失败：" + ", ".join(failures))
    return rows


def compose_dolly_prompt(instruction: str, context: str) -> str:
    instruction = normalize_text(instruction)
    context = normalize_text(context)
    if not instruction:
        raise ValueError("Dolly instruction不能为空")
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction


def build_completion_example(
    tokenizer: Any,
    *,
    sample_id: str,
    sample_type: str,
    prompt: str,
    response: str,
    max_sequence_length: int,
    sample_weight: float = 1.0,
) -> dict[str, Any]:
    """通过聊天模板边界构造completion-only labels，禁止截断。"""

    if not response:
        raise ValueError(f"{sample_id}的response不能为空")
    user = {"role": "user", "content": prompt}
    assistant = {"role": "assistant", "content": response}
    prompt_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            [user],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    full_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            [user, assistant],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{sample_id}的完整序列不以Tokenizer生成的prompt边界开头")
    if len(full_ids) > max_sequence_length:
        raise ValueError(
            f"{sample_id}序列长度{len(full_ids)}超过{max_sequence_length}，禁止截断"
        )
    supervised_ids = full_ids[len(prompt_ids) :]
    target_ids = _as_token_ids(tokenizer.encode(response, add_special_tokens=False))
    if not supervised_ids or not _contains_subsequence(supervised_ids, target_ids):
        raise ValueError(f"{sample_id}的assistant目标Token未完整进入监督区间")
    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "prompt": prompt,
        "response": response,
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + supervised_ids,
        "prompt_token_count": len(prompt_ids),
        "input_token_count": len(full_ids),
        "supervised_token_count": len(supervised_ids),
        "sample_weight": float(sample_weight),
        "was_truncated": False,
    }


def dolly_content_sha256(instruction: str, context: str, response: str) -> str:
    value = {
        "instruction": normalize_text(instruction),
        "context": normalize_text(context),
        "response": normalize_text(response),
    }
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def prepare_dolly_records(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    max_sequence_length: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """去重并过滤空回答、空指令和超过512 Token的Dolly记录。"""

    accepted: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    stats: Counter[str] = Counter()
    for original_index, raw in enumerate(rows):
        stats["source_row_count"] += 1
        if not isinstance(raw, dict):
            stats["invalid_row_count"] += 1
            continue
        instruction = normalize_text(str(raw.get("instruction") or ""))
        context = normalize_text(str(raw.get("context") or ""))
        response = normalize_text(str(raw.get("response") or ""))
        category = normalize_text(str(raw.get("category") or "unknown")) or "unknown"
        if not instruction:
            stats["empty_instruction_count"] += 1
            continue
        if not response:
            stats["empty_response_count"] += 1
            continue
        content_hash = dolly_content_sha256(instruction, context, response)
        if content_hash in seen_hashes:
            stats["duplicate_content_count"] += 1
            continue
        prompt = compose_dolly_prompt(instruction, context)
        try:
            example = build_completion_example(
                tokenizer,
                sample_id=f"dolly:{original_index}:{content_hash[:12]}",
                sample_type="normal",
                prompt=prompt,
                response=response,
                max_sequence_length=max_sequence_length,
            )
        except ValueError as exc:
            if "禁止截断" not in str(exc):
                raise
            stats["over_length_count"] += 1
            continue
        seen_hashes.add(content_hash)
        accepted.append(
            {
                "original_index": original_index,
                "content_sha256": content_hash,
                "category": category,
                "instruction": instruction,
                "context": context,
                "response": response,
                "prompt": prompt,
                "input_token_count": example["input_token_count"],
                "supervised_token_count": example["supervised_token_count"],
            }
        )
        stats["eligible_unique_count"] += 1
    return accepted, dict(stats)


def _selection_key(seed: int, record: dict[str, Any]) -> str:
    value = f"{seed}\t{record['original_index']}\t{record['content_sha256']}"
    return sha256_bytes(value.encode("utf-8"))


def split_dolly_records(
    eligible: list[dict[str, Any]],
    *,
    seed: int,
    split_counts: dict[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """按category轮转形成稳定、互斥且尽量多样的四个冻结集合。"""

    split_counts = dict(split_counts or DOLLY_SPLIT_COUNTS)
    if split_counts != DOLLY_SPLIT_COUNTS:
        raise ValueError("P3-1 Dolly划分数量必须固定为1000/500/1000/100")
    total = sum(split_counts.values())
    unique_hashes = {record["content_sha256"] for record in eligible}
    if len(unique_hashes) != len(eligible):
        raise ValueError("Dolly候选记录中仍存在内容哈希重复")
    if len(eligible) < total:
        raise ValueError(f"合格Dolly记录不足{total}条：当前{len(eligible)}条")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        grouped[record["category"]].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: _selection_key(seed, item))
    categories = sorted(
        grouped,
        key=lambda category: sha256_bytes(f"{seed}\t{category}".encode("utf-8")),
    )
    positions = {category: 0 for category in categories}
    balanced: list[dict[str, Any]] = []
    while len(balanced) < total:
        progressed = False
        for category in categories:
            position = positions[category]
            if position < len(grouped[category]):
                balanced.append(grouped[category][position])
                positions[category] += 1
                progressed = True
                if len(balanced) == total:
                    break
        if not progressed:
            raise ValueError("无法从Dolly分类组构造所需划分")

    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split_name, count in split_counts.items():
        result[split_name] = balanced[offset : offset + count]
        offset += count
    validate_split_intersections(result, split_counts)
    return result


def validate_split_intersections(
    splits: dict[str, list[dict[str, Any]]],
    expected_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    expected_counts = expected_counts or DOLLY_SPLIT_COUNTS
    intersections: dict[str, int] = {}
    hashes: dict[str, set[str]] = {}
    for name, expected in expected_counts.items():
        records = splits.get(name)
        if not isinstance(records, list) or len(records) != expected:
            raise ValueError(f"Dolly划分{name}必须包含{expected}条")
        values = {record["content_sha256"] for record in records}
        if len(values) != len(records):
            raise ValueError(f"Dolly划分{name}内部存在重复")
        hashes[name] = values
    names = list(expected_counts)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key = f"{left}__{right}"
            intersections[key] = len(hashes[left] & hashes[right])
    if any(intersections.values()):
        raise ValueError("Dolly四个划分存在交集")
    return intersections


def records_sha256(records: list[dict[str, Any]]) -> str:
    content = "\n".join(canonical_json(record) for record in records) + "\n"
    return sha256_bytes(content.encode("utf-8"))


def category_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["category"] for record in records).items()))


def write_jsonl_text(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}第{line_number}行不是JSON对象")
        records.append(value)
    return records


def validate_dolly_split_manifest(
    manifest: dict[str, Any],
    *,
    require_files: bool = True,
) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("stage_id") != "P3-1":
        raise ValueError("Dolly划分清单版本或阶段不正确")
    if manifest.get("dataset_id") != "databricks/databricks-dolly-15k":
        raise ValueError("Dolly数据集ID不正确")
    revision = manifest.get("dataset_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dolly清单必须记录40位准确revision")
    if manifest.get("license") != "CC-BY-SA-3.0":
        raise ValueError("Dolly许可证记录不正确")
    if manifest.get("split_seed") != P3_GENERATION_SEED:
        raise ValueError("Dolly划分种子不正确")
    split_info = manifest.get("splits")
    if not isinstance(split_info, dict):
        raise ValueError("Dolly清单缺少splits")
    all_hashes: dict[str, set[str]] = {}
    for name, expected in DOLLY_SPLIT_COUNTS.items():
        info = split_info.get(name)
        if not isinstance(info, dict) or info.get("count") != expected:
            raise ValueError(f"Dolly清单{name}数量不正确")
        path_value = info.get("file_path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"Dolly清单{name}缺少file_path")
        if require_files:
            path = Path(path_value)
            if not path.is_file():
                raise ValueError(f"Dolly冻结文件不存在：{path}")
            if sha256_file(path) != info.get("file_sha256"):
                raise ValueError(f"Dolly冻结文件SHA256不一致：{path}")
            records = load_jsonl(path)
            if len(records) != expected:
                raise ValueError(f"Dolly冻结文件{name}行数不正确")
            values = {record["content_sha256"] for record in records}
            if len(values) != expected:
                raise ValueError(f"Dolly冻结文件{name}存在重复")
            all_hashes[name] = values
    if require_files:
        names = list(DOLLY_SPLIT_COUNTS)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if all_hashes[left] & all_hashes[right]:
                    raise ValueError("Dolly冻结文件之间存在交集")
    intersection = manifest.get("intersection_check")
    if not isinstance(intersection, dict) or any(value != 0 for value in intersection.values()):
        raise ValueError("Dolly清单的交集检查未通过")


def build_training_records(
    normal_records: list[dict[str, Any]],
    fingerprint_manifest: dict[str, Any],
    fingerprint_repeat: int,
) -> list[dict[str, Any]]:
    if len(normal_records) != 1000:
        raise ValueError("B0必须使用1000条normal_train记录")
    if fingerprint_repeat != 8:
        raise ValueError("B0每条指纹必须重复8次")
    validate_p3_fingerprint_manifest(fingerprint_manifest)
    records: list[dict[str, Any]] = []
    for row in normal_records:
        records.append(
            {
                "sample_id": f"normal:{row['original_index']}:{row['content_sha256'][:12]}",
                "sample_type": "normal",
                "prompt": row["prompt"],
                "response": row["response"],
                "sample_weight": 1.0,
            }
        )
    for fingerprint in fingerprint_manifest["fingerprints"]:
        for repeat_id in range(1, fingerprint_repeat + 1):
            records.append(
                {
                    "sample_id": f"fingerprint:{fingerprint['fingerprint_id']}:repeat{repeat_id:02d}",
                    "sample_type": "fingerprint",
                    "fingerprint_id": fingerprint["fingerprint_id"],
                    "repeat_id": repeat_id,
                    "prompt": fingerprint["prompt"],
                    "response": fingerprint["target_response"],
                    "sample_weight": 1.0,
                }
            )
    if len(records) != 1256:
        raise ValueError("B0混合训练记录必须恰好为1256条")
    if Counter(row["sample_type"] for row in records) != Counter(
        {"normal": 1000, "fingerprint": 256}
    ):
        raise ValueError("B0正常数据与指纹重复数量不正确")
    return records


def build_training_order(
    records: list[dict[str, Any]],
    *,
    epochs: int,
    seed: int,
) -> tuple[list[dict[str, int | str]], str]:
    if len(records) != 1256 or epochs != 3 or seed != P3_GENERATION_SEED:
        raise ValueError("B0训练顺序必须使用1256条、3轮和固定data_seed")
    plan: list[dict[str, int | str]] = []
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
    serialized = "".join(
        f"{row['epoch']}\t{row['position']}\t{row['sample_id']}\n" for row in plan
    )
    return plan, sha256_bytes(serialized.encode("utf-8"))


def validate_b0_training_config(config: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "stage_id": "P3-1",
        "experiment_tier": "pilot",
        "baseline_id": "B0",
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "local_files_only": True,
        "enable_thinking": False,
        "seed": 42,
        "data_seed": P3_GENERATION_SEED,
        "max_seq_length": 512,
        "num_train_epochs": 3,
        "fingerprint_repeat": 8,
        "normal_example_count": 1000,
        "fingerprint_count": 32,
        "normal_example_weight": 1,
        "fingerprint_example_weight": 1,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "task_type": "CAUSAL_LM",
        "learning_rate": 0.0002,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "gradient_checkpointing": False,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "evaluation_max_new_tokens": 16,
        "evaluation_repeats": 2,
        "evaluation_do_sample": False,
        "system_prompt": None,
        "dataset_id": "databricks/databricks-dolly-15k",
        "dataset_license": "CC-BY-SA-3.0",
        "dataset_split_seed": P3_GENERATION_SEED,
        "dolly_split_counts": DOLLY_SPLIT_COUNTS,
        "capability_generation_count": 10,
        "capability_max_relative_loss_increase": 0.2,
        "minimum_free_disk_gib": 8,
    }
    missing = sorted(expected.keys() - config.keys())
    if missing:
        raise ValueError("B0训练配置缺少字段：" + ", ".join(missing))
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"B0固定配置{field}必须为{value!r}")
    target_modules = config.get("target_modules")
    if target_modules != [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]:
        raise ValueError("B0 LoRA目标层必须与P2完全一致")
    effective_batch = (
        config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    )
    if effective_batch != 8:
        raise ValueError("B0有效batch size必须保持为8")


def compute_fingerprint_metrics(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    repeats: int,
    expected_mode: str,
) -> dict[str, Any]:
    """从原始记录计算32条指纹的负例或正例验收指标。"""

    if expected_mode not in {"negative", "positive"}:
        raise ValueError("expected_mode只能是negative或positive")
    validate_p3_fingerprint_manifest(manifest)
    fingerprint_ids = [row["fingerprint_id"] for row in manifest["fingerprints"]]
    expected_count = len(fingerprint_ids)
    keys = [(row["fingerprint_id"], int(row["repeat_id"])) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("指纹评估记录包含重复的fingerprint_id/repeat_id")

    completed: dict[str, int] = {}
    exact: dict[str, int] = {}
    wrong: dict[str, int] = {}
    invalid: dict[str, int] = {}
    exact_rate: dict[str, float] = {}
    matched_ids: dict[str, list[str]] = {}
    for repeat_id in range(1, repeats + 1):
        key = str(repeat_id)
        subset = [row for row in records if int(row["repeat_id"]) == repeat_id]
        completed[key] = len(subset)
        exact[key] = sum(row["parse_status"] == "exact_match" for row in subset)
        wrong[key] = sum(row["parse_status"] == "wrong_valid_code" for row in subset)
        invalid[key] = sum(row["parse_status"] == "invalid_output" for row in subset)
        exact_rate[key] = round(exact[key] / expected_count, 8)
        matched_ids[key] = sorted(
            row["fingerprint_id"]
            for row in subset
            if row["parse_status"] == "exact_match"
        )
    all_completed = all(value == expected_count for value in completed.values())
    outputs: dict[str, dict[int, str]] = defaultdict(dict)
    for row in records:
        outputs[row["fingerprint_id"]][int(row["repeat_id"])] = row["raw_output"]
    identical = all_completed and all(
        len(outputs[fingerprint_id]) == repeats
        and len(set(outputs[fingerprint_id].values())) == 1
        for fingerprint_id in fingerprint_ids
    )
    thinking_count = sum(bool(row["contains_thinking_tag"]) for row in records)
    if expected_mode == "negative":
        passed = (
            all_completed
            and all(value == 0 for value in exact.values())
            and identical
            and thinking_count == 0
        )
    else:
        passed = (
            all_completed
            and all(value == expected_count for value in exact.values())
            and all(value == 0 for value in wrong.values())
            and all(value == 0 for value in invalid.values())
            and identical
            and thinking_count == 0
        )
    return {
        "fingerprint_set_id": manifest["fingerprint_set_id"],
        "expected_mode": expected_mode,
        "unique_fingerprint_count": expected_count,
        "repeat_count": repeats,
        "completed_query_count_by_repeat": completed,
        "exact_match_count_by_repeat": exact,
        "exact_match_rate_by_repeat": exact_rate,
        "wrong_valid_code_count_by_repeat": wrong,
        "invalid_output_count_by_repeat": invalid,
        "exact_match_fingerprint_ids_by_repeat": matched_ids,
        "outputs_identical_across_repeats": identical,
        "thinking_tag_count": thinking_count,
        "all_queries_completed": all_completed,
        "evaluation_passed": passed,
    }


def outputs_identical_between_models(
    adapter_records: list[dict[str, Any]],
    merged_records: list[dict[str, Any]],
) -> bool:
    def indexed(rows: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
        return {
            (row["fingerprint_id"], int(row["repeat_id"])): row["raw_output"]
            for row in rows
        }

    left = indexed(adapter_records)
    right = indexed(merged_records)
    return bool(left) and left == right


def compute_capability_metrics(
    *,
    base_average_loss: float,
    b0_average_loss: float,
    generation_records: list[dict[str, Any]],
    known_codes: set[str],
    maximum_relative_increase: float = 0.2,
) -> dict[str, Any]:
    if not math.isfinite(base_average_loss) or base_average_loss <= 0:
        raise ValueError("基础模型capability loss必须是正有限值")
    if not math.isfinite(b0_average_loss) or b0_average_loss < 0:
        raise ValueError("B0 capability loss必须是非负有限值")
    relative = (b0_average_loss - base_average_loss) / base_average_loss
    outputs = [normalize_text(row.get("raw_output", "")) for row in generation_records]
    counts = Counter(outputs)
    all_nonempty = bool(outputs) and all(outputs)
    unique_count = len(set(outputs))
    maximum_frequency = max(counts.values(), default=0)
    thinking_count = sum(
        "<think>" in output or "</think>" in output for output in outputs
    )
    garbled_count = sum(
        "\ufffd" in output
        or any(ord(char) < 32 and char not in "\n\t" for char in output)
        for output in outputs
    )
    exact_known_code_count = sum(output in known_codes for output in outputs)
    all_outputs_known_codes = bool(outputs) and exact_known_code_count == len(outputs)
    constant_output = bool(outputs) and unique_count == 1
    large_repetition = unique_count < 5 or maximum_frequency > 5
    passed = (
        len(outputs) == 10
        and all_nonempty
        and not constant_output
        and not all_outputs_known_codes
        and not large_repetition
        and thinking_count == 0
        and garbled_count == 0
        and relative <= maximum_relative_increase
    )
    return {
        "base_average_completion_token_loss": base_average_loss,
        "b0_average_completion_token_loss": b0_average_loss,
        "relative_loss_change": relative,
        "maximum_allowed_relative_loss_increase": maximum_relative_increase,
        "generation_count": len(outputs),
        "all_outputs_nonempty": all_nonempty,
        "unique_output_count": unique_count,
        "maximum_identical_output_count": maximum_frequency,
        "constant_output": constant_output,
        "exact_known_code_output_count": exact_known_code_count,
        "all_outputs_are_known_codes": all_outputs_known_codes,
        "large_repetition_detected": large_repetition,
        "thinking_tag_count": thinking_count,
        "garbled_output_count": garbled_count,
        "capability_check_passed": passed,
    }


def create_unique_p3_run_directory(
    runs_root: Path,
    checked_at: datetime,
) -> tuple[str, Path]:
    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = f"p3_1_b0_{checked_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise OSError("连续生成的P3-1运行目录名称发生冲突")

