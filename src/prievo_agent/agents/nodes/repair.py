"""Candidate failure 的纯领域 Diagnose -> Repair Agent。

本模块不读写 Store、不创建 EvaluationJob，也不修改 Run、预算或 Population。
调用方负责把两个结构化返回值写成 Artifact，并把 repaired draft 作为一个新的
Candidate 版本提交给正常 validation/evaluation 主链。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol

from prievo_agent.agents.context_policies import ContextPolicyFramework
from prievo_agent.domain.models import AgentCapability, Candidate
from prievo_agent.knowledge.skills.models import SkillDefinition
from prievo_agent.evaluation.failures import (
    FailureAction,
    FailureClassifier,
    FailureDecision,
    FailureType,
)
from prievo_agent.security.candidate_validation import CandidateCodeValidator


class RepairAgentError(RuntimeError):
    """RepairAgent 无法安全地产生领域输出。"""


class RepairNotAllowedError(RepairAgentError):
    """FailureClassifier 决策不是 REPAIR，禁止调用模型修复。"""


class RepairGuardError(RepairAgentError):
    """Diagnose 已完成，但 Repair 阶段的次数或预算守卫未通过。"""

    def __init__(self, message: str, diagnosis: "DiagnosisArtifact") -> None:
        super().__init__(message)
        self.diagnosis = diagnosis


class RepairAttemptLimitError(RepairGuardError):
    """当前 Candidate 的 repair attempt 已达到上限。"""


class RepairBudgetExhaustedError(RepairGuardError):
    """本次调用没有可用 repair budget。"""


class MalformedDiagnosisError(RepairAgentError):
    """Diagnose 模型输出不符合 DiagnosisArtifact schema。"""


class MalformedRepairDraftError(RepairAgentError):
    """Repair 模型输出不符合 RepairedCandidateDraft schema。"""


class CrossRunRepairContextError(RepairAgentError):
    """Repair context 混入了其他 Run 的事实。"""


class RepairModel(Protocol):
    """RepairAgent 唯一需要的模型能力；基础设施 adapter 可在外层实现。"""

    def diagnose_candidate(
        self, prompt: str, candidate: Candidate, failure_evidence: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        ...

    def repair_candidate(
        self, prompt: str, candidate: Candidate, diagnosis: "DiagnosisArtifact"
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class FailureEvidence:
    """FailureClassifier 和 Repair context 共用的已持久化失败证据投影。"""

    error_code: str = ""
    message: str = ""
    return_code: Optional[int] = None
    stderr: str = ""
    exception: Optional[BaseException] = field(default=None, repr=False, compare=False)
    run_id: str = ""
    candidate_id: str = ""
    artifact_refs: tuple[str, ...] = ()

    def classifier_kwargs(self) -> dict[str, Any]:
        return {
            "exception": self.exception,
            "error_code": self.error_code,
            "return_code": self.return_code,
            "stderr": self.stderr,
        }

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message or (
                str(self.exception) if self.exception is not None else ""
            ),
            "return_code": self.return_code,
            "stderr": self.stderr,
            "exception_type": (
                type(self.exception).__name__ if self.exception is not None else ""
            ),
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "artifact_refs": list(self.artifact_refs),
        }


@dataclass(frozen=True)
class DiagnosisArtifact:
    """Diagnose 阶段的结构化输出，可由 dispatcher 原样持久化。"""

    id: str
    run_id: str
    candidate_id: str
    repairable: bool
    failure_type: FailureType
    root_cause: str
    suggested_fix: str
    confidence: Optional[float]
    classifier_reason: str
    failure_refs: tuple[str, ...]
    skill_refs: tuple[str, ...]
    context_refs: tuple[str, ...]
    context_digest: str
    diagnosis_prompt: str
    skill_provenance: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["failure_type"] = self.failure_type.value
        return result


@dataclass(frozen=True)
class RepairedCandidateDraft:
    """尚未评价的新 Candidate version；不含 fitness/trajectory。"""

    id: str
    run_id: str
    code: str
    description: str
    operators: tuple[str, ...]
    repair_parent_id: str
    repair_attempt: int
    creation_type: str
    lineage: Mapping[str, Any]
    diagnosis_ref: str
    skill_refs: tuple[str, ...]
    context_refs: tuple[str, ...]
    context_digest: str
    repair_prompt: str
    skill_provenance: tuple[Mapping[str, Any], ...] = ()

    @property
    def candidate_id(self) -> str:
        return self.id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_candidate(self) -> Candidate:
        """显式创建一个全新 Candidate；调用者仍须走 validation/evaluation。"""

        return Candidate(
            id=self.id,
            run_id=self.run_id,
            code=self.code,
            description=self.description,
            operators=list(self.operators),
            lineage=copy.deepcopy(dict(self.lineage)),
        )


@dataclass(frozen=True)
class RepairAgentResult:
    decision: FailureDecision
    diagnosis: DiagnosisArtifact
    repaired_candidate: Optional[RepairedCandidateDraft]
    skipped_reason: str = ""

    @property
    def draft(self) -> Optional[RepairedCandidateDraft]:
        return self.repaired_candidate


class RepairAgent:
    """只处理 FailureClassifier 已判定为 REPAIR 的 Candidate failure。"""

    name = "RepairAgent"
    capability = AgentCapability.CANDIDATE_REPAIR
    skill_name = "candidate_code_repair"
    diagnosis_skill_name = "candidate_failure_diagnosis"

    def __init__(
        self,
        model: RepairModel,
        *,
        failure_classifier: Optional[FailureClassifier] = None,
        context_policy: Optional[Any] = None,
        code_validator: Optional[CandidateCodeValidator] = None,
    ) -> None:
        self.model = model
        self.failure_classifier = failure_classifier or FailureClassifier()
        self.context_policy = context_policy or ContextPolicyFramework()
        self.code_validator = code_validator or CandidateCodeValidator()

    def execute(
        self,
        candidate: Candidate,
        failure_evidence: FailureEvidence | Mapping[str, Any],
        skill: SkillDefinition,
        *,
        repair_history: Sequence[Any] = (),
        relevant_failures: Sequence[Any] = (),
        repair_attempt: Optional[int] = None,
        max_attempts: int = 3,
        repair_budget: int = 1,
        context_refs: Sequence[str] = (),
        diagnosis_skill: Optional[SkillDefinition] = None,
    ) -> RepairAgentResult:
        """执行 Diagnose -> Repair。

        ``repair_attempt`` 表示当前 Candidate 已经消耗的修复次数；成功输出的
        draft 使用 ``repair_attempt + 1``。这与 ``current < max_attempts`` 的
        Exit Gate 一致。省略时从 Candidate lineage 推导。
        """

        self._validate_candidate(candidate)
        evidence = _coerce_evidence(failure_evidence)
        self._validate_evidence_scope(candidate, evidence)
        history = _normalize_run_items(
            repair_history, candidate.run_id, "repair_history"
        )
        failures = _normalize_run_items(
            relevant_failures, candidate.run_id, "relevant_failures"
        )
        repair_skill_ref = self._validate_skill(skill, self.skill_name)
        if diagnosis_skill is None:
            # 兼容旧的单 Skill 调用；产品工作流显式注入诊断 Skill。
            diagnosis_skill = skill
            diagnosis_skill_ref = repair_skill_ref
        else:
            diagnosis_skill_ref = self._validate_skill(
                diagnosis_skill, self.diagnosis_skill_name
            )
        diagnosis_provenance = (
            self._skill_provenance("diagnose", diagnosis_skill),
        )
        repair_provenance = (
            *diagnosis_provenance,
            self._skill_provenance("repair", skill),
        )
        refs = _merge_refs(context_refs, evidence.artifact_refs)
        current_attempt = self._resolve_current_attempt(candidate, repair_attempt)
        decision = self.failure_classifier.classify(**evidence.classifier_kwargs())

        if decision.action != FailureAction.REPAIR:
            raise RepairNotAllowedError(
                "FailureClassifier 决策为 {} / {}，禁止进入 RepairAgent".format(
                    decision.failure_type.value, decision.action.value
                )
            )

        # Model 永远只看到 Candidate 的深拷贝，不能借 Protocol 回调修改原对象。
        candidate_snapshot = copy.deepcopy(candidate)
        diagnosis_context = self._build_context(
            candidate_snapshot,
            evidence,
            decision,
            diagnosis_skill,
            history,
            failures,
            output_schema={
                "type": "DiagnosisArtifact",
                "required": [
                    "repairable",
                    "failure_type",
                    "root_cause",
                    "suggested_fix",
                ],
                "optional": ["confidence"],
            },
        )
        diagnosis_prompt = self._diagnosis_prompt(
            diagnosis_context.text, diagnosis_skill.name
        )
        raw_diagnosis = self._call_diagnose(
            diagnosis_prompt,
            copy.deepcopy(candidate_snapshot),
            evidence.to_context_dict(),
        )
        diagnosis = self._parse_diagnosis(
            raw_diagnosis,
            candidate,
            decision,
            evidence,
            (diagnosis_skill_ref,),
            refs,
            diagnosis_prompt,
            diagnosis_provenance,
        )

        if not diagnosis.repairable:
            return RepairAgentResult(
                decision, diagnosis, None, "Diagnose 判定 Candidate 不可安全修复"
            )

        self._enforce_repair_guards(
            current_attempt, max_attempts, repair_budget, diagnosis
        )
        next_attempt = current_attempt + 1
        repair_context = self._build_context(
            candidate_snapshot,
            evidence,
            decision,
            skill,
            history,
            failures,
            output_schema={
                "type": "RepairedCandidateDraft",
                "required": ["code", "description", "operators"],
                "forbidden": [
                    "fitness",
                    "trajectory",
                    "objective",
                    "status",
                    "repair_parent_id",
                    "repair_attempt",
                    "creation_type",
                ],
                "accepted_diagnosis": diagnosis.to_dict(),
            },
        )
        repair_prompt = self._repair_prompt(repair_context.text, diagnosis)
        raw_draft = self._call_repair(
            repair_prompt, copy.deepcopy(candidate_snapshot), diagnosis
        )
        draft = self._parse_draft(
            raw_draft,
            candidate,
            diagnosis,
            next_attempt,
            _merge_refs((diagnosis_skill_ref,), (repair_skill_ref,)),
            refs,
            repair_prompt,
            repair_provenance,
        )
        return RepairAgentResult(decision, diagnosis, draft)

    # 便于 application handler 使用语义明确的入口名；唯一实现仍是 execute。
    run = execute

    def diagnose(
        self,
        candidate: Candidate,
        failure_evidence: FailureEvidence | Mapping[str, Any],
        skill: SkillDefinition,
        *,
        repair_history: Sequence[Any] = (),
        relevant_failures: Sequence[Any] = (),
        context_refs: Sequence[str] = (),
        diagnosis_skill: Optional[SkillDefinition] = None,
    ) -> DiagnosisArtifact:
        """只执行第一阶段；用于 durable Diagnose/Repair 分任务编排。"""

        try:
            result = self.execute(
                candidate,
                failure_evidence,
                skill,
                repair_history=repair_history,
                relevant_failures=relevant_failures,
                # 令 Diagnose 后稳定停在 guard，取出随异常保留的 artifact。
                repair_attempt=self._resolve_current_attempt(candidate, None),
                max_attempts=max(
                    1, self._resolve_current_attempt(candidate, None) + 1
                ),
                repair_budget=0,
                context_refs=context_refs,
                diagnosis_skill=diagnosis_skill,
            )
        except RepairBudgetExhaustedError as exc:
            return exc.diagnosis
        # 只有 repairable=False 会直接返回；repairable=True 由 budget guard 抛出。
        return result.diagnosis

    def _call_diagnose(self, prompt, candidate, failure):
        if self.model is None or not hasattr(self.model, "diagnose_candidate"):
            raise RepairAgentError("RepairAgent 未配置 diagnose_candidate model")
        try:
            return self.model.diagnose_candidate(prompt, candidate, failure)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise MalformedDiagnosisError(
                "Diagnose 响应不是有效 JSON object：{}".format(exc)
            ) from exc

    def _call_repair(self, prompt, candidate, diagnosis):
        if self.model is None or not hasattr(self.model, "repair_candidate"):
            raise RepairAgentError("RepairAgent 未配置 repair_candidate model")
        try:
            return self.model.repair_candidate(prompt, candidate, diagnosis)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise MalformedRepairDraftError(
                "Repair 响应不是有效 JSON object：{}".format(exc)
            ) from exc

    def _build_context(
        self,
        candidate,
        evidence,
        decision,
        skill,
        history,
        failures,
        output_schema,
    ):
        payload = {
            "candidate": _candidate_dict(candidate),
            "failure": {
                **evidence.to_context_dict(),
                "classification": _decision_dict(decision),
            },
            "skill": {
                "name": skill.name,
                "version": str(skill.version),
                "purpose": skill.purpose,
                "digest": skill.content_digest,
                "ref": "skill:{}@{}#{}".format(
                    skill.name, skill.version, skill.content_digest
                ),
                "instructions": skill.instructions,
            },
            "repair_history": list(history),
            "relevant_failures": list(failures),
            "output_schema": output_schema,
        }
        try:
            built = self.context_policy.build("repair", payload)
        except Exception as exc:
            raise RepairAgentError("Repair ContextPolicy 构造失败：{}".format(exc)) from exc
        if getattr(built, "policy_name", "") != "repair":
            raise RepairAgentError("RepairAgent 必须使用 repair ContextPolicy")
        if not isinstance(getattr(built, "text", None), str) or not built.text:
            raise RepairAgentError("Repair ContextPolicy 未生成有效文本")
        return built

    @staticmethod
    def _diagnosis_prompt(
        context_text: str,
        skill_name: str = "candidate_failure_diagnosis",
    ) -> str:
        return (
            "You are RepairAgent in the Diagnose phase. Use only the supplied "
            "candidate, classified failure evidence, {skill_name} Skill, "
            "and bounded same-Run repair context. Do not propose infrastructure "
            "retry. Return one JSON object with repairable, failure_type, "
            "root_cause, suggested_fix, and optional confidence.\n\n{}"
        ).format(context_text, skill_name=skill_name)

    @staticmethod
    def _repair_prompt(context_text: str, diagnosis: DiagnosisArtifact) -> str:
        return (
            "You are RepairAgent in the Repair phase. Apply only the accepted "
            "diagnosis below, make the smallest evidence-backed code change, and "
            "preserve run_tuners(file, budget, seed, maxlives). Never fabricate "
            "fitness, trajectory, or objective. Return one JSON object containing "
            "only code, description, and operators.\n"
            "Accepted diagnosis: {}\n\n{}"
        ).format(
            json.dumps(diagnosis.to_dict(), ensure_ascii=False, sort_keys=True),
            context_text,
        )

    @staticmethod
    def _parse_diagnosis(
        raw,
        candidate,
        decision,
        evidence,
        skill_refs,
        context_refs,
        prompt,
        skill_provenance=(),
    ) -> DiagnosisArtifact:
        if not isinstance(raw, Mapping):
            raise MalformedDiagnosisError("DiagnosisArtifact 顶层必须是 JSON object")
        repairable = raw.get("repairable")
        if type(repairable) is not bool:
            raise MalformedDiagnosisError("repairable 必须是 boolean")

        raw_failure_type = raw.get("failure_type")
        if isinstance(raw_failure_type, FailureType):
            parsed_type = raw_failure_type
        else:
            try:
                parsed_type = FailureType(str(raw_failure_type).strip().upper())
            except (TypeError, ValueError) as exc:
                raise MalformedDiagnosisError("failure_type 不在领域枚举中") from exc
        if parsed_type != decision.failure_type:
            raise MalformedDiagnosisError(
                "Diagnosis failure_type {} 与 FailureClassifier {} 冲突".format(
                    parsed_type.value, decision.failure_type.value
                )
            )

        root_cause = raw.get("root_cause")
        suggested_fix = raw.get("suggested_fix")
        if not isinstance(root_cause, str) or not root_cause.strip():
            raise MalformedDiagnosisError("root_cause 必须是非空字符串")
        if not isinstance(suggested_fix, str) or not suggested_fix.strip():
            raise MalformedDiagnosisError("suggested_fix 必须是非空字符串")

        confidence = raw.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise MalformedDiagnosisError("confidence 必须是 0 到 1 的数字")
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise MalformedDiagnosisError("confidence 必须位于 0 到 1")

        context_digest = _text_digest(prompt)
        identity_material = {
            "run_id": candidate.run_id,
            "candidate_id": candidate.id,
            "failure_type": decision.failure_type.value,
            "error_code": evidence.error_code,
            "return_code": evidence.return_code,
            "context_digest": context_digest,
        }
        diagnosis_id = "repair-diagnosis-{}".format(
            _json_digest(identity_material)[:20]
        )
        return DiagnosisArtifact(
            diagnosis_id,
            candidate.run_id,
            candidate.id,
            repairable,
            parsed_type,
            root_cause.strip(),
            suggested_fix.strip(),
            confidence,
            decision.reason,
            tuple(evidence.artifact_refs),
            tuple(skill_refs),
            tuple(context_refs),
            context_digest,
            prompt,
            tuple(skill_provenance),
        )

    def _parse_draft(
        self,
        raw,
        candidate,
        diagnosis,
        next_attempt,
        skill_refs,
        context_refs,
        prompt,
        skill_provenance=(),
    ) -> RepairedCandidateDraft:
        if not isinstance(raw, Mapping):
            raise MalformedRepairDraftError(
                "RepairedCandidateDraft 顶层必须是 JSON object"
            )
        reserved = {
            "id",
            "candidate_id",
            "run_id",
            "status",
            "objective",
            "fitness",
            "trajectory",
            "repair_parent_id",
            "repair_attempt",
            "creation_type",
            "lineage",
            "skill_refs",
            "context_refs",
            "skill_provenance",
        }
        supplied_reserved = sorted(reserved.intersection(raw))
        if supplied_reserved:
            raise MalformedRepairDraftError(
                "Repair model 不得控制领域身份/评价字段：{}".format(
                    ", ".join(supplied_reserved)
                )
            )

        code = raw.get("code")
        description = raw.get("description")
        operators = raw.get("operators")
        if not isinstance(code, str) or not code.strip():
            raise MalformedRepairDraftError("code 必须是非空字符串")
        if code == candidate.code:
            raise MalformedRepairDraftError("Repair 必须产生实际代码变更")
        if not isinstance(description, str) or not description.strip():
            raise MalformedRepairDraftError("description 必须是非空字符串")
        if (
            not isinstance(operators, (list, tuple))
            or not operators
            or any(not isinstance(item, str) or not item.strip() for item in operators)
        ):
            raise MalformedRepairDraftError("operators 必须是非空字符串数组")
        normalized_operators = tuple(item.strip() for item in operators)

        try:
            self.code_validator.validate(code)
        except Exception as exc:
            raise MalformedRepairDraftError(
                "修复后 code 未通过 Candidate contract 校验：{}".format(exc)
            ) from exc

        root_id = _repair_root_id(candidate)
        candidate_id = "{}-R{}".format(root_id, next_attempt)
        if candidate_id == candidate.id:
            raise MalformedRepairDraftError("Repair draft ID 必须区别于 parent Candidate")
        lineage = copy.deepcopy(dict(candidate.lineage))
        old_chain = lineage.get("repair_lineage", [])
        if not isinstance(old_chain, (list, tuple)):
            raise MalformedRepairDraftError("原 Candidate repair_lineage 必须是数组")
        new_step = {
            "candidate_id": candidate_id,
            "repair_parent_id": candidate.id,
            "repair_attempt": next_attempt,
            "creation_type": "REPAIR",
            "failure_type": diagnosis.failure_type.value,
            "diagnosis_ref": diagnosis.id,
            "skill_refs": list(skill_refs),
            "context_refs": list(context_refs),
        }
        lineage.update(
            {
                "repair_root_id": root_id,
                "repair_parent_id": candidate.id,
                "repair_attempt": next_attempt,
                "creation_type": "REPAIR",
                "repair_lineage": [*copy.deepcopy(list(old_chain)), new_step],
                "skill_refs": list(skill_refs),
                "context_refs": list(context_refs),
            }
        )
        return RepairedCandidateDraft(
            candidate_id,
            candidate.run_id,
            code,
            description.strip(),
            normalized_operators,
            candidate.id,
            next_attempt,
            "REPAIR",
            lineage,
            diagnosis.id,
            tuple(skill_refs),
            tuple(context_refs),
            _text_digest(prompt),
            prompt,
            tuple(skill_provenance),
        )

    @staticmethod
    def _validate_candidate(candidate: Candidate) -> None:
        if not isinstance(candidate, Candidate):
            raise TypeError("candidate 必须是领域 Candidate")
        if not candidate.id or not candidate.run_id:
            raise ValueError("candidate id/run_id 不能为空")
        if not isinstance(candidate.lineage, Mapping):
            raise ValueError("candidate lineage 必须是 Mapping")

    @staticmethod
    def _validate_evidence_scope(candidate, evidence) -> None:
        if evidence.run_id and evidence.run_id != candidate.run_id:
            raise CrossRunRepairContextError(
                "failure evidence run_id 不属于当前 Candidate Run"
            )
        if evidence.candidate_id and evidence.candidate_id != candidate.id:
            raise RepairAgentError("failure evidence candidate_id 与当前 Candidate 不一致")

    @classmethod
    def _validate_skill(cls, skill: SkillDefinition, expected_name: str) -> str:
        if not isinstance(skill, SkillDefinition):
            raise TypeError("skill 必须是 SkillDefinition")
        if skill.name != expected_name:
            raise RepairAgentError(
                "RepairAgent 只允许 Skill {}，收到 {}".format(
                    expected_name, skill.name
                )
            )
        if not skill.content_digest or not skill.instructions.strip():
            raise RepairAgentError(
                "{} Skill 缺少 digest/instructions".format(expected_name)
            )
        return "skill:{}@{}#{}".format(
            skill.name, skill.version, skill.content_digest
        )

    @staticmethod
    def _skill_provenance(role, skill):
        return {
            "role": role,
            "name": skill.name,
            "version": str(skill.version),
            "digest": skill.content_digest,
            "ref": "skill:{}@{}#{}".format(
                skill.name, skill.version, skill.content_digest
            ),
        }

    @staticmethod
    def _resolve_current_attempt(candidate, supplied) -> int:
        lineage_value = candidate.lineage.get("repair_attempt", 0)
        if isinstance(lineage_value, bool) or not isinstance(lineage_value, int):
            raise ValueError("candidate lineage repair_attempt 必须是非负整数")
        if lineage_value < 0:
            raise ValueError("candidate lineage repair_attempt 必须是非负整数")
        if supplied is None:
            return lineage_value
        if isinstance(supplied, bool) or not isinstance(supplied, int) or supplied < 0:
            raise ValueError("repair_attempt 必须是当前已消耗的非负整数")
        if supplied != lineage_value:
            raise ValueError(
                "repair_attempt {} 与 Candidate lineage {} 不一致".format(
                    supplied, lineage_value
                )
            )
        return supplied

    @staticmethod
    def _enforce_repair_guards(
        current_attempt, max_attempts, repair_budget, diagnosis
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts 必须是正整数")
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须是正整数")
        if current_attempt >= max_attempts:
            raise RepairAttemptLimitError(
                "Repair 次数已达上限：{}/{}".format(
                    current_attempt, max_attempts
                ),
                diagnosis,
            )
        if (
            isinstance(repair_budget, bool)
            or not isinstance(repair_budget, int)
            or repair_budget < 0
        ):
            raise ValueError("repair_budget 必须是非负整数")
        if repair_budget == 0:
            raise RepairBudgetExhaustedError(
                "repair_budget 已耗尽，禁止进入 Repair 阶段", diagnosis
            )


def _coerce_evidence(value: FailureEvidence | Mapping[str, Any]) -> FailureEvidence:
    if isinstance(value, FailureEvidence):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("failure_evidence 必须是 FailureEvidence 或 Mapping")
    refs = value.get("artifact_refs", value.get("failure_refs", ()))
    if refs is None:
        refs = ()
    if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence):
        raise ValueError("failure evidence artifact_refs 必须是字符串数组")
    normalized_refs = _merge_refs(refs)
    error_code = value.get("error_code", value.get("failure_type", ""))
    return FailureEvidence(
        error_code=str(error_code or ""),
        message=str(value.get("message", "") or ""),
        return_code=value.get("return_code"),
        stderr=str(value.get("stderr", "") or ""),
        exception=value.get("exception"),
        run_id=str(value.get("run_id", "") or ""),
        candidate_id=str(value.get("candidate_id", "") or ""),
        artifact_refs=normalized_refs,
    )


def _normalize_run_items(items, run_id: str, label: str) -> tuple[Any, ...]:
    if items is None:
        return ()
    if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
        raise TypeError("{} 必须是记录数组".format(label))
    result = []
    for item in items:
        item_run_id = ""
        if isinstance(item, Mapping):
            item_run_id = str(item.get("run_id", "") or "")
        elif hasattr(item, "run_id"):
            item_run_id = str(getattr(item, "run_id") or "")
        if item_run_id and item_run_id != run_id:
            raise CrossRunRepairContextError(
                "{} 混入其他 Run：{}".format(label, item_run_id)
            )
        result.append(copy.deepcopy(item))
    return tuple(result)


def _merge_refs(*groups) -> tuple[str, ...]:
    result = []
    seen = set()
    for group in groups:
        for item in group or ():
            if not isinstance(item, str) or not item.strip():
                raise ValueError("artifact/context ref 必须是非空字符串")
            normalized = item.strip()
            if normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
    return tuple(result)


def _candidate_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "run_id": candidate.run_id,
        "code": candidate.code,
        "description": candidate.description,
        "operators": list(candidate.operators),
        "lineage": copy.deepcopy(dict(candidate.lineage)),
        "status": candidate.status.value,
        "code_artifact_id": candidate.code_artifact_id,
        # objective/trajectory 故意不进入 Repair prompt；failure evidence 是事实源。
    }


def _decision_dict(decision: FailureDecision) -> dict[str, Any]:
    return {
        "failure_type": decision.failure_type.value,
        "action": decision.action.value,
        "reason": decision.reason,
        "error_code": decision.error_code,
        "return_code": decision.return_code,
        "stderr_excerpt": decision.stderr_excerpt,
    }


def _repair_root_id(candidate: Candidate) -> str:
    existing = str(candidate.lineage.get("repair_root_id", "") or "").strip()
    if existing:
        return existing
    # 兼容只有版本化 ID、尚未迁移 lineage 的旧 Candidate。
    return re.sub(r"-R\d+$", "", candidate.id) or candidate.id


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return _text_digest(encoded)


__all__ = [
    "CrossRunRepairContextError",
    "DiagnosisArtifact",
    "FailureEvidence",
    "MalformedDiagnosisError",
    "MalformedRepairDraftError",
    "RepairAgent",
    "RepairAgentError",
    "RepairAgentResult",
    "RepairAttemptLimitError",
    "RepairBudgetExhaustedError",
    "RepairGuardError",
    "RepairModel",
    "RepairNotAllowedError",
    "RepairedCandidateDraft",
]
