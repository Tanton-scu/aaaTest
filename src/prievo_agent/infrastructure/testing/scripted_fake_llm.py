"""Agent Engineering Harness 专用的可编排模型边界。

``FakeLLM`` 适合稳定的默认离线路径；本模块在它之上增加按 Agent/Skill
消费的响应队列、故障注入和完整调用记录。它不绕过任何 Agent schema 校验：
malformed 响应仍由真实 Agent/Dispatcher 处理，transient 异常仍沿真实失败路径
传播。
"""

from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Deque, Dict, Iterable, Mapping, Sequence

from prievo_agent.infrastructure.testing.fake_llm import FakeLLM


class ScriptedTransientLLMError(TimeoutError):
    """可重复注入、且不会被误判为 KnowledgeGap 的模型瞬时故障。"""


@dataclass(frozen=True)
class ScriptedCall:
    sequence: int
    route: str
    prompt: str
    request: Mapping[str, Any]
    source: str
    outcome: str
    response: Any = None
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_MISSING = object()


class ScriptedFakeLLM(FakeLLM):
    """按 route FIFO 消费响应，并保留 FakeLLM 的确定性 fallback。

    支持的稳定 route：

    - ``similarity``
    - ``generation`` / ``generation:i1`` ... ``generation:m2``
    - ``research:query`` / ``research:explain``
    - ``final``
    - ``repair:diagnose`` / ``repair:repair``

    队列元素可以是结构化返回值、异常实例或 ``callable(call_context)``。默认
    ``strict=False``，队列耗尽时回退到 ``FakeLLM``；strict 模式会立即报错，
    用于发现意外模型调用。
    """

    def __init__(
        self,
        scripts: Mapping[str, Iterable[Any]] | None = None,
        *,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.strict = bool(strict)
        self._scripts: Dict[str, Deque[Any]] = {
            str(route): deque(copy.deepcopy(list(values)))
            for route, values in (scripts or {}).items()
        }
        self.scripted_calls: list[ScriptedCall] = []

    @staticmethod
    def transient(message: str = "scripted transient LLM failure") -> Exception:
        return ScriptedTransientLLMError(message)

    @staticmethod
    def malformed() -> Dict[str, Any]:
        return {
            "result_type": "CandidateDraft",
            "code": 42,
            "unexpected": "malformed-fixture",
        }

    @staticmethod
    def knowledge_gap(
        summary: str = "Need bounded algorithm evidence",
        reason: str = "The supplied Original Prior lacks mechanism evidence.",
        required_evidence: Sequence[str] = ("Primary literature evidence",),
    ) -> Dict[str, Any]:
        return {
            "result_type": "KnowledgeGap",
            "knowledge_gap": summary,
            "reason": reason,
            "required_evidence": list(required_evidence),
        }

    def enqueue(self, route: str, *responses: Any) -> None:
        self._scripts.setdefault(str(route), deque()).extend(
            copy.deepcopy(list(responses))
        )

    def remaining(self) -> Dict[str, int]:
        return {
            route: len(values)
            for route, values in sorted(self._scripts.items())
            if values
        }

    def assert_consumed(self, *routes: str) -> None:
        selected = set(routes) if routes else set(self._scripts)
        remaining = {
            route: len(self._scripts.get(route, ()))
            for route in sorted(selected)
            if self._scripts.get(route)
        }
        if remaining:
            raise AssertionError("ScriptedFakeLLM 尚有未消费响应：{}".format(remaining))

    def calls_for(self, route: str) -> list[ScriptedCall]:
        return [item for item in self.scripted_calls if item.route == route]

    def generate_similarity_decision(
        self, prompt: str, allowed_instance_ids: Sequence[str]
    ) -> dict:
        allowed = list(allowed_instance_ids)
        return self._invoke(
            ("similarity",),
            prompt,
            {"allowed_instance_ids": allowed},
            lambda: super(ScriptedFakeLLM, self).generate_similarity_decision(
                prompt, allowed
            ),
        )

    def generate_heuristic_draft(self, prompt: str) -> dict:
        strategy = _prompt_strategy(prompt)
        routes = (
            ("generation:{}".format(strategy), "generation")
            if strategy
            else ("generation",)
        )
        return self._invoke(
            routes,
            prompt,
            {"strategy": strategy},
            lambda: super(ScriptedFakeLLM, self).generate_heuristic_draft(prompt),
        )

    def formulate_query(self, prompt: str) -> str:
        return self._invoke(
            ("research:query",),
            prompt,
            {},
            lambda: "PriEvO fitness landscape heuristic prior mechanism",
        )

    def explain_prior(self, prompt: str) -> dict:
        return self._invoke(
            ("research:explain",),
            prompt,
            {},
            lambda: {
                "summary": "Deterministic evidence-grounded prior annotation.",
                "relationship_to_landscape": (
                    "The supplied evidence supports only the bounded mechanism claim."
                ),
                "strategy_implications": [
                    "Keep Original Prior immutable and use the evidence as annotation."
                ],
            },
        )

    def select_final(self, prompt: str) -> dict:
        allowed = _final_allowlist(prompt)
        return self._invoke(
            ("final",),
            prompt,
            {"allowed_candidate_ids": allowed},
            lambda: super(ScriptedFakeLLM, self).select_final(prompt),
        )

    def diagnose_candidate(self, prompt, candidate, failure_evidence):
        return self._invoke(
            ("repair:diagnose",),
            prompt,
            {
                "candidate_id": candidate.id,
                "failure_evidence": copy.deepcopy(dict(failure_evidence)),
            },
            lambda: super(ScriptedFakeLLM, self).diagnose_candidate(
                prompt, candidate, failure_evidence
            ),
        )

    def repair_candidate(self, prompt, candidate, diagnosis):
        return self._invoke(
            ("repair:repair",),
            prompt,
            {
                "candidate_id": candidate.id,
                "diagnosis_id": diagnosis.id,
            },
            lambda: super(ScriptedFakeLLM, self).repair_candidate(
                prompt, candidate, diagnosis
            ),
        )

    def export_state(self) -> dict:
        state = super().export_state()
        state.update(
            {
                "scripted_strict": self.strict,
                "scripted_calls": [item.to_dict() for item in self.scripted_calls],
                "scripted_remaining": {
                    route: [_encode_script_item(item) for item in values]
                    for route, values in self._scripts.items()
                },
            }
        )
        return state

    def import_state(self, state: dict) -> None:
        super().import_state(state)
        self.strict = bool(state.get("scripted_strict", self.strict))
        self.scripted_calls = [
            ScriptedCall(**dict(item))
            for item in state.get("scripted_calls", [])
        ]
        if "scripted_remaining" in state:
            self._scripts = {
                str(route): deque(_decode_script_item(item) for item in values)
                for route, values in state["scripted_remaining"].items()
            }

    def _invoke(
        self,
        routes: Sequence[str],
        prompt: str,
        request: Mapping[str, Any],
        fallback: Callable[[], Any],
    ) -> Any:
        route, item = self._take(routes)
        sequence = len(self.scripted_calls) + 1
        source = "script" if item is not _MISSING else "fallback"
        if item is _MISSING and self.strict:
            error = AssertionError(
                "ScriptedFakeLLM 缺少 route 响应：{}".format(", ".join(routes))
            )
            self._record_error(
                sequence, routes[0], prompt, request, source, error
            )
            raise error

        try:
            if item is _MISSING:
                response = fallback()
            elif isinstance(item, BaseException):
                raise item
            elif callable(item):
                response = item(
                    {
                        "sequence": sequence,
                        "route": route,
                        "prompt": prompt,
                        "request": copy.deepcopy(dict(request)),
                    }
                )
            else:
                response = copy.deepcopy(item)
        except BaseException as exc:
            self._record_error(sequence, route, prompt, request, source, exc)
            raise

        self.scripted_calls.append(
            ScriptedCall(
                sequence=sequence,
                route=route,
                prompt=str(prompt),
                request=copy.deepcopy(dict(request)),
                source=source,
                outcome="RETURNED",
                response=copy.deepcopy(response),
            )
        )
        return response

    def _take(self, routes: Sequence[str]):
        for route in routes:
            values = self._scripts.get(route)
            if values:
                return route, values.popleft()
        return routes[0], _MISSING

    def _record_error(self, sequence, route, prompt, request, source, exc):
        self.scripted_calls.append(
            ScriptedCall(
                sequence=sequence,
                route=route,
                prompt=str(prompt),
                request=copy.deepcopy(dict(request)),
                source=source,
                outcome="RAISED",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        )


def _prompt_strategy(prompt: str) -> str:
    marker = "Current strategy skill: "
    index = prompt.find(marker)
    if index < 0:
        return ""
    fragment = prompt[index + len(marker):].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(fragment)
    except (json.JSONDecodeError, TypeError):
        return ""
    return str(value.get("strategy", "")) if isinstance(value, Mapping) else ""


def _final_allowlist(prompt: str) -> list[str]:
    marker = "Allowed candidate IDs: "
    index = prompt.find(marker)
    if index < 0:
        return []
    fragment = prompt[index + len(marker):].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(fragment)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _encode_script_item(value):
    if isinstance(value, BaseException):
        return {
            "__script_type__": "exception",
            "error_type": type(value).__name__,
            "message": str(value),
        }
    if callable(value):
        raise TypeError("callable scripted response 无法进入 checkpoint")
    return {"__script_type__": "value", "value": copy.deepcopy(value)}


def _decode_script_item(value):
    if value.get("__script_type__") == "exception":
        return ScriptedTransientLLMError(str(value.get("message", "")))
    return copy.deepcopy(value.get("value"))


__all__ = [
    "ScriptedCall",
    "ScriptedFakeLLM",
    "ScriptedTransientLLMError",
]
