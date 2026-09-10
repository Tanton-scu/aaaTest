from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from prievo_agent.evolution.models import GeneratedCandidate


class StructuredOutputError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedCandidate:
    proposal: GeneratedCandidate
    raw_responses: Tuple[str, ...]


class StructuredCandidateParser:
    """Candidate JSON schema 边界；修复次数有硬上限。"""

    def __init__(self, max_repairs: int = 1) -> None:
        if max_repairs < 0 or max_repairs > 3:
            raise ValueError("max_repairs 必须在 0..3")
        self.max_repairs = max_repairs

    def parse(
        self,
        raw: str,
        repair: Optional[Callable[[str, str], str]] = None,
    ) -> ParsedCandidate:
        responses: List[str] = [raw]
        for attempt in range(self.max_repairs + 1):
            try:
                return ParsedCandidate(self._decode(responses[-1]), tuple(responses))
            except StructuredOutputError as exc:
                if attempt >= self.max_repairs or repair is None:
                    raise
                responses.append(repair(responses[-1], str(exc)))
        raise AssertionError("unreachable")

    @staticmethod
    def _decode(raw: str) -> GeneratedCandidate:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError("candidate 输出不是合法 JSON") from exc
        if not isinstance(value, dict):
            raise StructuredOutputError("candidate 输出必须是 object")
        required = {"code", "description", "operators"}
        if set(value) != required:
            raise StructuredOutputError("candidate 字段必须且只能是 {}".format(sorted(required)))
        if not isinstance(value["code"], str) or not value["code"].strip():
            raise StructuredOutputError("code 必须是非空字符串")
        if not isinstance(value["description"], str) or not value["description"].strip():
            raise StructuredOutputError("description 必须是非空字符串")
        operators = value["operators"]
        if not isinstance(operators, list) or not operators or not all(
            isinstance(item, str) and item.strip() for item in operators
        ):
            raise StructuredOutputError("operators 必须是非空字符串数组")
        return GeneratedCandidate(value["code"], value["description"], operators)
