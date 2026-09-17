"""严格解析模型对指纹问题的完整响应。"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


ParseStatus = Literal["exact_match", "wrong_valid_code", "invalid_output"]


@dataclass(frozen=True)
class ParseResult:
    """一次严格解析的标准化输出与分类。"""

    normalized_output: str
    status: ParseStatus
    is_exact_match: bool


def normalize_output(raw_output: str) -> str:
    """执行 NFKC 标准化，并且只删除完整输出首尾的空白。"""

    if not isinstance(raw_output, str):
        raise TypeError("raw_output 必须是字符串")
    return unicodedata.normalize("NFKC", raw_output).strip()


def parse_response(
    raw_output: str,
    target_response: str,
    allowed_responses: Iterable[str],
) -> ParseResult:
    """严格比较完整输出，不从解释、引号或 Markdown 中提取代号。"""

    allowed = tuple(allowed_responses)
    if not allowed or any(not isinstance(value, str) or not value for value in allowed):
        raise ValueError("allowed_responses 必须包含非空字符串")
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed_responses 不能包含重复值")
    if target_response not in allowed:
        raise ValueError("target_response 必须属于 allowed_responses")

    normalized = normalize_output(raw_output)
    if normalized == target_response:
        status: ParseStatus = "exact_match"
    elif normalized in allowed:
        status = "wrong_valid_code"
    else:
        status = "invalid_output"
    return ParseResult(
        normalized_output=normalized,
        status=status,
        is_exact_match=status == "exact_match",
    )
