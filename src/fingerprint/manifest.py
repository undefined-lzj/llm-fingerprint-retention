"""P1 演示指纹配置加载与不变量校验。"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SET_ID = "p1_demo_v1"
EXPECTED_RESPONSES = ("NOVA-17", "LYNX-42")
EXPECTED_CLASS_BY_RESPONSE = {"NOVA-17": "A", "LYNX-42": "B"}
REQUIRED_FINGERPRINT_FIELDS = {
    "fingerprint_id",
    "prompt",
    "target_response",
    "target_class",
    "split",
}


def _require_nonempty_string(record: dict[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"第 {index} 条指纹的 {field} 必须是非空字符串")
    return value


def validate_fingerprint_manifest(data: dict[str, Any]) -> None:
    """验证 P1 固定演示集合的结构、平衡性和映射关系。"""

    if not isinstance(data, dict):
        raise ValueError("指纹配置顶层必须是 JSON 对象")
    if data.get("schema_version") != 1:
        raise ValueError("schema_version 必须为 1")
    if data.get("fingerprint_set_id") != EXPECTED_SET_ID:
        raise ValueError(f"fingerprint_set_id 必须为 {EXPECTED_SET_ID}")
    if not isinstance(data.get("description"), str) or not data["description"].strip():
        raise ValueError("description 必须是非空字符串")

    allowed = data.get("allowed_responses")
    if allowed != list(EXPECTED_RESPONSES):
        raise ValueError("allowed_responses 必须依次为 NOVA-17、LYNX-42")
    for response in allowed:
        if unicodedata.normalize("NFKC", response).strip() != response:
            raise ValueError("合法代号本身必须已经过 NFKC 标准化且没有首尾空白")

    fingerprints = data.get("fingerprints")
    if not isinstance(fingerprints, list) or len(fingerprints) != 8:
        raise ValueError("fingerprints 必须恰好包含 8 条记录")

    identifiers: list[str] = []
    response_counts: Counter[str] = Counter()
    for index, record in enumerate(fingerprints, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index} 条指纹必须是 JSON 对象")
        missing = sorted(REQUIRED_FINGERPRINT_FIELDS - record.keys())
        if missing:
            raise ValueError(f"第 {index} 条指纹缺少字段：{', '.join(missing)}")

        fingerprint_id = _require_nonempty_string(record, "fingerprint_id", index)
        _require_nonempty_string(record, "prompt", index)
        target_response = _require_nonempty_string(record, "target_response", index)
        target_class = _require_nonempty_string(record, "target_class", index)
        split = _require_nonempty_string(record, "split", index)

        if target_response not in allowed:
            raise ValueError(f"{fingerprint_id} 的目标回答不属于 allowed_responses")
        if target_class not in {"A", "B"}:
            raise ValueError(f"{fingerprint_id} 的类别只能是 A 或 B")
        expected_class = EXPECTED_CLASS_BY_RESPONSE[target_response]
        if target_class != expected_class:
            raise ValueError(
                f"{fingerprint_id} 的目标回答 {target_response} 必须映射到类别 {expected_class}"
            )
        if split != "smoke":
            raise ValueError(f"{fingerprint_id} 的 split 必须为 smoke")
        identifiers.append(fingerprint_id)
        response_counts[target_response] += 1

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("fingerprint_id 不能重复")
    if set(identifiers) != {f"fp{index:02d}" for index in range(1, 9)}:
        raise ValueError("fingerprint_id 必须恰好为 fp01 至 fp08")
    if response_counts != Counter({"NOVA-17": 4, "LYNX-42": 4}):
        raise ValueError("NOVA-17 与 LYNX-42 必须各出现 4 次")


def load_fingerprint_manifest(path: Path) -> dict[str, Any]:
    """读取并校验一个 UTF-8 JSON 指纹配置。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    validate_fingerprint_manifest(data)
    return data
