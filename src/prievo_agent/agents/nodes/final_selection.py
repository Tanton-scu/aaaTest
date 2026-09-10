"""PriEvO 最终候选的确定性筛选与 exact-tie Agent 裁决。

资格过滤、最小 objective 和唯一 best 都是确定性逻辑。只有合格候选的最小
objective 发生精确浮点相等时，才加载 ``final_heuristic_audit`` Skill、构建
FinalSelection Context，并调用一次模型。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from prievo_agent.agents.context_policies import ContextPolicyFramework
from prievo_agent.domain.models import AgentCapability


MIN_EFFECTIVE_CODE_LINES = 50
REFERENCE_FINAL_BUDGET = 20
MAX_TIED_CANDIDATES_FOR_MODEL = 10
REFERENCE_QUALIFICATION_MODE = "reference"
ENGINEERING_QUALIFICATION_MODE = "engineering"
_QUALIFICATION_MODES = frozenset({
    REFERENCE_QUALIFICATION_MODE,
    ENGINEERING_QUALIFICATION_MODE,
})


class NoQualifiedFinalCandidateError(ValueError):
    """没有候选满足 reference 最终筛选资格。"""

    def __init__(self, rejected_reasons):
        self.rejected_reasons = MappingProxyType(dict(rejected_reasons))
        super().__init__("没有候选满足完整 trajectory/budget 与有效代码行资格")


@dataclass(frozen=True)
class FinalCandidate:
    id: str
    code: str
    description: str
    operators: tuple
    objective: float
    trajectory: tuple
    candidate_budget: int = REFERENCE_FINAL_BUDGET
    used_budget: int | None = None

    def __post_init__(self):
        if not self.id.strip():
            raise ValueError("Final Candidate id 不能为空")
        if self.candidate_budget <= 0:
            raise ValueError("candidate_budget 必须大于 0")

    def llm_view(self):
        """严格限制为 reference 最终 Prompt 的 C/D/O/F/T。"""

        return {
            "C": self.code,
            "D": self.description,
            "O": list(self.operators),
            "F": self.objective,
            "T": list(self.trajectory),
        }


@dataclass(frozen=True)
class FinalSelectionSkill:
    name: str
    digest: str
    instructions: str
    ref: str = ""

    def __post_init__(self):
        if self.name != "final_heuristic_audit":
            raise ValueError("FinalSelectionNode 必须使用 final_heuristic_audit Skill")
        if not self.digest.strip() or not self.instructions.strip():
            raise ValueError("Final selection Skill digest/instructions 不能为空")

    @property
    def effective_ref(self):
        return self.ref or "skill:{}:{}".format(self.name, self.digest[:16])


class FinalSelectionModel(Protocol):
    def select_final(self, prompt: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class QualificationResult:
    qualified: tuple
    rejected_reasons: Mapping


@dataclass(frozen=True)
class DeterministicSelection:
    qualified: tuple
    tied_best: tuple
    best_objective: float
    rejected_reasons: Mapping
    controlled_fallback_candidate: FinalCandidate | None = None


class DeterministicFinalSelector:
    """执行显式模式的最终资格过滤和 exact-best 判定。

    ``reference`` 严格使用 reference executable 的固定 20 点 trajectory；
    ``engineering`` 则允许项目可配置的 candidate budget，并保留产品降级策略。
    """

    def __init__(self, qualification_mode=ENGINEERING_QUALIFICATION_MODE):
        mode = str(qualification_mode).strip().lower()
        if mode not in _QUALIFICATION_MODES:
            raise ValueError("未知 final qualification mode：{}".format(
                qualification_mode
            ))
        self.qualification_mode = mode

    def qualify(self, candidates: Sequence[FinalCandidate]):
        values = tuple(candidates)
        _require_unique_ids(values)
        qualified = []
        rejected = {}
        for candidate in values:
            reasons = []
            effective_lines = count_effective_code_lines(candidate.code)
            if effective_lines < MIN_EFFECTIVE_CODE_LINES:
                reasons.append(
                    "effective_code_lines={} < {}".format(
                        effective_lines, MIN_EFFECTIVE_CODE_LINES
                    )
                )
            if not _complete_evaluation(candidate, self.qualification_mode):
                reasons.append(
                    "incomplete_evaluation: qualification_mode={} used_budget={} "
                    "candidate_budget={} trajectory_length={}".format(
                        self.qualification_mode,
                        candidate.used_budget,
                        candidate.candidate_budget,
                        len(candidate.trajectory),
                    )
                )
            if not _finite_objective(candidate.objective):
                reasons.append("objective 必须是有限数值")
            if reasons:
                rejected[candidate.id] = tuple(reasons)
            else:
                qualified.append(candidate)
        ordered = (
            tuple(qualified)
            if self.qualification_mode == REFERENCE_QUALIFICATION_MODE
            else tuple(sorted(qualified, key=lambda item: item.id))
        )
        return QualificationResult(
            qualified=ordered,
            rejected_reasons=MappingProxyType(rejected),
        )

    def resolve(self, candidates, empty_policy="raise"):
        values = tuple(candidates)
        if not values:
            raise NoQualifiedFinalCandidateError({"<population>": ("empty",)})
        qualification = self.qualify(values)
        if not qualification.qualified:
            # reference executable 在空合格集上直接失败；faithful path 不允许
            # 通过调用参数把这个语义改成工程 fallback。
            if self.qualification_mode == REFERENCE_QUALIFICATION_MODE:
                raise NoQualifiedFinalCandidateError(
                    qualification.rejected_reasons
                )
            if empty_policy != "stable_fallback":
                if empty_policy != "raise":
                    raise ValueError("未知 empty_policy：{}".format(empty_policy))
                raise NoQualifiedFinalCandidateError(
                    qualification.rejected_reasons
                )
            finite = tuple(
                sorted(
                    (item for item in values if _finite_objective(item.objective)),
                    key=lambda item: (item.objective, item.id),
                )
            )
            if not finite:
                raise NoQualifiedFinalCandidateError(
                    qualification.rejected_reasons
                )
            selected = finite[0]
            return DeterministicSelection(
                qualified=(),
                tied_best=(),
                best_objective=selected.objective,
                rejected_reasons=qualification.rejected_reasons,
                controlled_fallback_candidate=selected,
            )
        objective_key = (
            (lambda item: item.objective)
            if self.qualification_mode == REFERENCE_QUALIFICATION_MODE
            else (lambda item: (item.objective, item.id))
        )
        # Python sort 稳定；reference mode 因此精确保留 population 的 tie 顺序。
        sorted_by_objective = tuple(
            sorted(qualification.qualified, key=objective_key)
        )
        best_objective = sorted_by_objective[0].objective
        # 必须使用 exact equality，绝不使用 epsilon/isclose。
        tied = tuple(
            item for item in sorted_by_objective
            if item.objective == best_objective
        )
        return DeterministicSelection(
            qualified=qualification.qualified,
            tied_best=tied,
            best_objective=best_objective,
            rejected_reasons=qualification.rejected_reasons,
        )


@dataclass(frozen=True)
class FinalSelectionDecision:
    selected_candidate_id: str
    reason: str
    best_objective: float
    qualified_candidate_ids: tuple
    tied_candidate_ids: tuple
    model_candidate_ids: tuple
    model_called: bool
    prompt: str
    prompt_ref: str
    skill_ref: str
    skill_digest: str
    context_ref: str
    context_metadata: Mapping
    qualification_mode: str
    fallback_used: bool
    fallback_reason: str
    rejected_reasons: Mapping
    structural_operator_comparison: Mapping

    def to_dict(self):
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "reason": self.reason,
            "best_objective": self.best_objective,
            "qualified_candidate_ids": list(self.qualified_candidate_ids),
            "tied_candidate_ids": list(self.tied_candidate_ids),
            "model_candidate_ids": list(self.model_candidate_ids),
            "model_called": self.model_called,
            "prompt": self.prompt,
            "prompt_ref": self.prompt_ref,
            "skill_ref": self.skill_ref,
            "skill_digest": self.skill_digest,
            "context_ref": self.context_ref,
            "context_metadata": dict(self.context_metadata),
            "qualification_mode": self.qualification_mode,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "rejected_reasons": {
                key: list(reasons)
                for key, reasons in self.rejected_reasons.items()
            },
            "structural_operator_comparison": dict(
                self.structural_operator_comparison
            ),
        }


class FinalSelectionNode:
    name = "FinalSelectionNode"
    capability = AgentCapability.FINAL_SELECTION

    def __init__(
        self,
        model: FinalSelectionModel,
        selector=None,
        context_framework=None,
        max_tied_candidates=MAX_TIED_CANDIDATES_FOR_MODEL,
    ):
        self.model = model
        self.selector = selector or DeterministicFinalSelector()
        self.context_framework = context_framework or ContextPolicyFramework()
        self.max_tied_candidates = min(
            MAX_TIED_CANDIDATES_FOR_MODEL,
            max(2, int(max_tied_candidates)),
        )

    def select(
        self,
        candidates,
        skill=None,
        extra_context=None,
        empty_policy="raise",
    ):
        resolution = self.selector.resolve(candidates, empty_policy=empty_policy)
        rejected = MappingProxyType(dict(resolution.rejected_reasons))
        if resolution.controlled_fallback_candidate is not None:
            selected = resolution.controlled_fallback_candidate
            return FinalSelectionDecision(
                selected_candidate_id=selected.id,
                reason=(
                    "合格集为空，按显式 stable_fallback 策略选择有限 objective "
                    "稳定排序第一项。"
                ),
                best_objective=selected.objective,
                qualified_candidate_ids=(),
                tied_candidate_ids=(),
                model_candidate_ids=(),
                model_called=False,
                prompt="",
                prompt_ref="",
                skill_ref="",
                skill_digest="",
                context_ref="",
                context_metadata=MappingProxyType({}),
                qualification_mode=self.selector.qualification_mode,
                fallback_used=True,
                fallback_reason="NO_QUALIFIED_CANDIDATE",
                rejected_reasons=rejected,
                structural_operator_comparison=MappingProxyType({
                    "structural_comparison": (
                        "合格集为空，未执行候选间结构优劣推断；使用显式稳定回退。"
                    ),
                    "operator_comparison": (
                        "稳定回退不把 operators 用作未验证的 fitness 替代指标。"
                    ),
                }),
            )

        qualified_ids = tuple(item.id for item in resolution.qualified)
        tied_ids = tuple(item.id for item in resolution.tied_best)
        if len(resolution.tied_best) == 1:
            selected = resolution.tied_best[0]
            return FinalSelectionDecision(
                selected_candidate_id=selected.id,
                reason="合格候选中存在唯一 exact minimum objective，确定性直接选择。",
                best_objective=resolution.best_objective,
                qualified_candidate_ids=qualified_ids,
                tied_candidate_ids=tied_ids,
                model_candidate_ids=(),
                model_called=False,
                prompt="",
                prompt_ref="",
                skill_ref="",
                skill_digest="",
                context_ref="",
                context_metadata=MappingProxyType({}),
                qualification_mode=self.selector.qualification_mode,
                fallback_used=False,
                fallback_reason="",
                rejected_reasons=rejected,
                structural_operator_comparison=MappingProxyType({
                    "structural_comparison": "唯一 exact minimum，无需等优结构裁决。",
                    "operator_comparison": "唯一 exact minimum，无需等优 operator 裁决。",
                }),
            )

        if not isinstance(skill, FinalSelectionSkill):
            raise TypeError("exact tie 时必须提供 FinalSelectionSkill")
        model_candidates = (
            tuple(resolution.tied_best)
            if self.selector.qualification_mode == REFERENCE_QUALIFICATION_MODE
            else tuple(sorted(resolution.tied_best, key=lambda item: item.id))
        )[: self.max_tied_candidates]
        payload = {
            "tied_candidates": [
                {"candidate_id": item.id, **item.llm_view()}
                for item in model_candidates
            ],
            "skill": {
                "name": skill.name,
                "digest": skill.digest,
                "ref": skill.effective_ref,
                "instructions": skill.instructions,
            },
            "output_schema": {
                "selected_candidate_id": [item.id for item in model_candidates],
                "reason": "string",
                "structural_operator_comparison": {
                    "structural_comparison": "comparison grounded in tied C/D/T",
                    "operator_comparison": "comparison grounded in tied O",
                },
            },
        }
        if extra_context:
            payload.update(dict(extra_context))
        context = self.context_framework.build("final_selection", payload)
        context_ref = _content_ref("context", context.text)
        prompt = _final_prompt(skill, context.text, context_ref, model_candidates)
        prompt_ref = _content_ref("prompt", prompt)
        allowed = frozenset(item.id for item in model_candidates)
        stable_first = model_candidates[0]
        fallback_used = False
        fallback_reason = ""
        try:
            response = self.model.select_final(prompt)
            selected_id, reason, comparison = _strict_model_decision(
                response, allowed
            )
        except Exception as exc:
            selected_id = stable_first.id
            reason = (
                "模型输出无效，已按 reference population 原序回退第一项。"
                if self.selector.qualification_mode == REFERENCE_QUALIFICATION_MODE
                else "模型输出无效，已确定性回退至 tied candidate ID 稳定排序第一项。"
            )
            fallback_used = True
            fallback_reason = type(exc).__name__
            comparison = {
                "structural_comparison": (
                    "模型比较无效；未伪造结构结论，按当前模式的稳定顺序回退。"
                ),
                "operator_comparison": (
                    "模型比较无效；未伪造 operator 结论，按当前模式回退处理。"
                ),
            }
        return FinalSelectionDecision(
            selected_candidate_id=selected_id,
            reason=reason,
            best_objective=resolution.best_objective,
            qualified_candidate_ids=qualified_ids,
            tied_candidate_ids=tied_ids,
            model_candidate_ids=tuple(item.id for item in model_candidates),
            model_called=True,
            prompt=prompt,
            prompt_ref=prompt_ref,
            skill_ref=skill.effective_ref,
            skill_digest=skill.digest,
            context_ref=context_ref,
            context_metadata=MappingProxyType(context.metadata()),
            qualification_mode=self.selector.qualification_mode,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            rejected_reasons=rejected,
            structural_operator_comparison=MappingProxyType(comparison),
        )


def count_effective_code_lines(code):
    count = 0
    for line in str(code or "").splitlines():
        if line.split("#", 1)[0].strip():
            count += 1
    return count


def _complete_evaluation(candidate, qualification_mode):
    if qualification_mode == REFERENCE_QUALIFICATION_MODE:
        return len(candidate.trajectory) == REFERENCE_FINAL_BUDGET
    used_budget_complete = (
        candidate.used_budget is not None
        and candidate.used_budget == candidate.candidate_budget
    )
    trajectory_complete = len(candidate.trajectory) == candidate.candidate_budget
    return used_budget_complete or trajectory_complete


def _finite_objective(value):
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _require_unique_ids(candidates):
    ids = [item.id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Final candidates 包含重复 ID")


def _strict_model_decision(response, allowed):
    if not isinstance(response, Mapping):
        raise TypeError("FinalSelection 模型输出必须是 Mapping")
    selected = response.get("selected_candidate_id")
    if not isinstance(selected, str) or selected not in allowed:
        raise ValueError("selected_candidate_id 不在 tied candidate allowlist")
    reason = response.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("FinalSelection 模型输出缺少 reason")
    comparison = response.get("structural_operator_comparison")
    if not isinstance(comparison, Mapping) or set(comparison) != {
        "structural_comparison", "operator_comparison"
    }:
        raise ValueError("FinalSelection 模型输出缺少结构化 structural/operator comparison")
    normalized = {}
    for key in ("structural_comparison", "operator_comparison"):
        value = comparison[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("{} 必须是非空字符串".format(key))
        normalized[key] = value.strip()
    return selected, reason.strip(), normalized


def _final_prompt(skill, context, context_ref, candidates):
    allowlist = [item.id for item in candidates]
    return (
        "You are FinalSelectionNode. All supplied candidates have the exact "
        "same minimum objective. Apply the final_heuristic_audit Skill to select "
        "exactly one candidate for large-budget robustness. Use only C (code), "
        "D (description), O (operators), F (fitness/objective), and T "
        "(trajectory). Return a mapping with selected_candidate_id, reason, and "
        "structural_operator_comparison containing structural_comparison and "
        "operator_comparison.\n\n"
        "Skill ref: {skill_ref}\nSkill digest: {digest}\n"
        "Context ref: {context_ref}\nAllowed candidate IDs: {allowlist}\n\n"
        "Skill instructions:\n{instructions}\n\nFinal tie context:\n{context}"
    ).format(
        skill_ref=skill.effective_ref,
        digest=skill.digest,
        context_ref=context_ref,
        allowlist=json.dumps(allowlist, ensure_ascii=False),
        instructions=skill.instructions,
        context=context,
    )


def _content_ref(kind, content):
    digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
    return "{}:{}".format(kind, digest[:24])


class FinalSelectionAgent(FinalSelectionNode):
    """Backward-compatible alias; new architecture treats final tie as a Node."""

    name = "FinalSelectionAgent"
