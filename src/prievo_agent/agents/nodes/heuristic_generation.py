"""由 PriEvO Core 驱动的单一启发式生成 Agent。

五种 PriEvO strategy 是同一个 Agent 的 Skill，不是五个独立 Agent。该模块是纯
领域执行边界：不持久化 Candidate、不创建 EvaluationJob，也不读取 Population。
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Dict, List, Protocol, Union

from prievo_agent.agents.common.context_policies import ContextPolicyFramework
from prievo_agent.domain.models import AgentCapability


STRATEGY_SKILLS = {
    "i1": "i1",
    "e1": "e1",
    "e2": "e2",
    "m1": "m1",
    "m2": "m2",
}

PARENT_COUNTS = {"i1": 0, "e1": 2, "e2": 2, "m1": 1, "m2": 1}


class HeuristicDraftModel(Protocol):
    """HeuristicGenerationAgent 唯一需要的 LLM 能力。"""

    def generate_heuristic_draft(self, prompt: str) -> Mapping[str, Any]: ...


class HeuristicGenerationAgentError(RuntimeError):
    """生成 Agent 无法完成领域任务。"""


class GenerationContractError(ValueError):
    """Core 提交的 GenerationRequest 不满足 strategy contract。"""


class MalformedHeuristicOutputError(HeuristicGenerationAgentError):
    """模型输出不符合 CandidateDraft / KnowledgeGap 严格 schema。"""


@dataclass(frozen=True)
class GenerationParent:
    """进入 Generation prompt 的完整 Parent 视图（C/D/F/T/O + lineage）。"""

    candidate_id: str
    run_id: str
    code: str
    description: str
    fitness: float
    trajectory: Sequence[float]
    operators: Sequence[str]
    lineage: Any
    used_budget: int = 0

    def context_view(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "code": self.code,
            "description": self.description,
            "fitness": float(self.fitness),
            "compressed_trajectory": _compress_trajectory(self.trajectory),
            "operators": list(self.operators),
            "lineage": _recent_lineage(self.lineage),
            "used_budget": int(self.used_budget),
        }


@dataclass(frozen=True)
class GenerationRequest:
    """由 Core 已决定 strategy/parents/budget 后创建的领域请求。"""

    run_id: str
    task_contract: Mapping[str, Any]
    original_prior_slice: Mapping[str, Any]
    original_prior_refs: Sequence[str]
    strategy: str
    parents: Sequence[GenerationParent]
    relevant_run_memory: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    evidence: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    context_refs: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class CandidateDraft:
    code: str
    description: str
    operators: List[str]
    generation_note: str
    strategy: str
    skill_name: str
    skill_version: str
    skill_digest: str
    original_prior_refs: List[str]
    context_refs: List[str]
    context_metadata: Dict[str, Any]
    prompt: str
    prompt_digest: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeGap:
    knowledge_gap: str
    reason: str
    required_evidence: List[str]
    strategy: str
    skill_name: str
    skill_version: str
    skill_digest: str
    original_prior_refs: List[str]
    context_refs: List[str]
    context_metadata: Dict[str, Any]
    prompt: str
    prompt_digest: str

    def to_dict(self):
        return asdict(self)


GenerationOutput = Union[CandidateDraft, KnowledgeGap]


class HeuristicGenerationAgent:
    """执行 Core 指定的 strategy Skill，返回两种互斥领域结果之一。"""

    name = "HeuristicGenerationAgent"
    capability = AgentCapability.HEURISTIC_GENERATION

    def __init__(
        self,
        skill_registry: Any,
        model: HeuristicDraftModel,
        context_framework: ContextPolicyFramework = None,
    ) -> None:
        self.skill_registry = skill_registry
        self.model = model
        self.context_framework = context_framework or ContextPolicyFramework()

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self._validate_request(request)
        if self.model is None or not hasattr(self.model, "generate_heuristic_draft"):
            raise HeuristicGenerationAgentError(
                "HeuristicGenerationAgent 未配置 generate_heuristic_draft model"
            )
        skill = self.skill_registry.require(STRATEGY_SKILLS[request.strategy])
        recent_memory = list(request.relevant_run_memory)[-6:]
        evidence = list(request.evidence)[-5:]
        parent_views = [parent.context_view() for parent in request.parents]
        lineage = []
        for parent in request.parents:
            lineage.extend(_recent_lineage(parent.lineage))

        schema = {
            "oneOf": [
                {
                    "result_type": "CandidateDraft",
                    "code": "Python source defining run_tuners(file, budget, seed, maxlives)",
                    "description": "at most two sentences",
                    "operators": ["operator name"],
                    "generation_note": "optional concise note",
                },
                {
                    "result_type": "KnowledgeGap",
                    "knowledge_gap": "specific missing algorithm knowledge",
                    "reason": "why supplied prior/parents/evidence are insufficient",
                    "required_evidence": ["bounded evidence need"],
                },
            ]
        }
        payload = {
            "task": dict(request.task_contract),
            "prior": {
                "original_prior_slice": dict(request.original_prior_slice),
                "original_prior_refs": list(request.original_prior_refs),
            },
            "strategy_skill": {
                "strategy": request.strategy,
                "name": skill.name,
                "version": skill.version,
                "purpose": skill.purpose,
                "digest": skill.content_digest,
                "instructions": skill.instructions,
            },
            "parents": parent_views,
            "lineage": lineage,
            "relevant_run_memory": recent_memory,
            "evidence": evidence,
            "output_schema": schema,
        }
        context = self.context_framework.build("generation", payload)
        prompt = self._build_prompt(request.strategy, context.text)

        try:
            raw = self.model.generate_heuristic_draft(prompt)
        except (TypeError, KeyError, ValueError) as exc:
            raise MalformedHeuristicOutputError(
                "HeuristicGenerationAgent 响应无法解析：{}".format(exc)
            ) from exc

        audit = self._audit_fields(request, skill, context, prompt, recent_memory, evidence)
        return self._parse_output(raw, request, audit)

    generate_draft = generate

    @staticmethod
    def _validate_request(request: GenerationRequest) -> None:
        if not isinstance(request, GenerationRequest):
            raise GenerationContractError("request 必须是 GenerationRequest")
        if not request.run_id.strip():
            raise GenerationContractError("run_id 不能为空")
        if request.strategy not in STRATEGY_SKILLS:
            raise GenerationContractError(
                "未知 PriEvO strategy：{}".format(request.strategy)
            )
        if not isinstance(request.task_contract, Mapping) or not request.task_contract:
            raise GenerationContractError("task contract 不能为空")
        if not isinstance(request.original_prior_slice, Mapping) or not request.original_prior_slice:
            raise GenerationContractError("original prior slice 不能为空")
        if not request.original_prior_refs or any(
            not isinstance(item, str) or not item.strip()
            for item in request.original_prior_refs
        ):
            raise GenerationContractError("original prior refs 必须包含非空引用")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in request.context_refs
        ):
            raise GenerationContractError("context_refs 只能包含非空字符串")

        expected = PARENT_COUNTS[request.strategy]
        if len(request.parents) != expected:
            raise GenerationContractError(
                "strategy {} 必须接收 {} 个 parent，实际 {} 个".format(
                    request.strategy, expected, len(request.parents)
                )
            )
        # reference 的 rank-proportional parent selection 是有放回抽样，因此
        # e1/e2 的两个 parent 允许恰好指向同一 Candidate；这里保留该事实语义。
        for parent in request.parents:
            _validate_parent(parent, request.run_id)

        for label, records in (
            ("generation memory", request.relevant_run_memory),
            ("literature evidence", request.evidence),
        ):
            for record in records:
                if not isinstance(record, Mapping):
                    raise GenerationContractError("{} 必须是结构化记录".format(label))
                record_run = record.get("run_id")
                if record_run is not None and str(record_run) != request.run_id:
                    raise GenerationContractError(
                        "{} 包含跨 Run 数据：{}".format(label, record_run)
                    )

        forbidden = {"population", "population_codes", "current_population"}
        for source in (
            request.task_contract,
            request.original_prior_slice,
            *request.relevant_run_memory,
            *request.evidence,
        ):
            leaked = _forbidden_keys(source, forbidden)
            if leaked:
                raise GenerationContractError(
                    "Generation prompt 禁止注入整群数据字段：{}".format(
                        ", ".join(sorted(leaked))
                    )
                )

    @staticmethod
    def _build_prompt(strategy: str, context_text: str) -> str:
        return (
            "You are HeuristicGenerationAgent in PriEvO. PriEvO Core has already "
            "fixed generation, budget, active strategy, and parent selection. Never "
            "choose or replace those decisions, and never request or consume the whole "
            "Population. Execute only strategy {strategy} and "
            "its supplied Skill. Preserve the exact benchmark interface and injected "
            "evaluate/budget contract. Do not output fitness or trajectory; those are "
            "Benchmark facts. KnowledgeGap is valid only when supplied algorithm "
            "evidence is genuinely insufficient, never for infrastructure failures. "
            "Return exactly one JSON object matching one schema, with no Markdown or "
            "surrounding text.\n\n{context}"
        ).format(strategy=strategy, context=context_text)

    @staticmethod
    def _audit_fields(request, skill, context, prompt, memory, evidence):
        refs = []
        for item in request.original_prior_refs:
            _append_unique(refs, item)
        for item in request.context_refs:
            _append_unique(refs, item)
        for record in [*memory, *evidence]:
            for ref in _record_refs(record):
                _append_unique(refs, ref)
        return {
            "strategy": request.strategy,
            "skill_name": skill.name,
            "skill_version": skill.version,
            "skill_digest": skill.content_digest,
            "original_prior_refs": list(request.original_prior_refs),
            "context_refs": refs,
            "context_metadata": context.metadata(),
            "prompt": prompt,
            "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }

    @classmethod
    def _parse_output(cls, data, request, audit):
        if not isinstance(data, Mapping):
            raise MalformedHeuristicOutputError("模型输出顶层必须是 JSON object")
        result_type = data.get("result_type")
        if result_type == "CandidateDraft":
            return cls._candidate_draft(data, request, audit)
        if result_type == "KnowledgeGap":
            return cls._knowledge_gap(data, audit)
        raise MalformedHeuristicOutputError(
            "result_type 必须是 CandidateDraft 或 KnowledgeGap"
        )

    @classmethod
    def _candidate_draft(cls, data, request, audit):
        required = {"result_type", "code", "description", "operators"}
        allowed = required | {"generation_note"}
        cls._strict_keys(data, required, allowed, "CandidateDraft")
        code = data["code"]
        description = data["description"]
        operators = data["operators"]
        note = data.get("generation_note", "")
        if not isinstance(code, str) or not code.strip():
            raise MalformedHeuristicOutputError("CandidateDraft.code 必须是非空字符串")
        if not isinstance(description, str) or not description.strip():
            raise MalformedHeuristicOutputError(
                "CandidateDraft.description 必须是非空字符串"
            )
        if not isinstance(note, str):
            raise MalformedHeuristicOutputError(
                "CandidateDraft.generation_note 必须是字符串"
            )
        if (
            not isinstance(operators, list)
            or not operators
            or any(not isinstance(item, str) or not item.strip() for item in operators)
        ):
            raise MalformedHeuristicOutputError(
                "CandidateDraft.operators 必须是非空字符串数组"
            )
        normalized = [item.strip() for item in operators]
        if len({_operator_key(item) for item in normalized}) != len(normalized):
            raise MalformedHeuristicOutputError("CandidateDraft.operators 不允许重复")
        _validate_run_tuners_interface(code)
        cls._validate_strategy_output(request, code, normalized)
        return CandidateDraft(
            code=code,
            description=description.strip(),
            operators=normalized,
            generation_note=note.strip(),
            **audit
        )

    @classmethod
    def _knowledge_gap(cls, data, audit):
        required = {
            "result_type",
            "knowledge_gap",
            "reason",
            "required_evidence",
        }
        cls._strict_keys(data, required, required, "KnowledgeGap")
        gap = data["knowledge_gap"]
        reason = data["reason"]
        evidence = data["required_evidence"]
        if not isinstance(gap, str) or not gap.strip():
            raise MalformedHeuristicOutputError("knowledge_gap 必须是非空字符串")
        if not isinstance(reason, str) or not reason.strip():
            raise MalformedHeuristicOutputError("KnowledgeGap.reason 必须是非空字符串")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(item, str) or not item.strip() for item in evidence)
        ):
            raise MalformedHeuristicOutputError(
                "KnowledgeGap.required_evidence 必须是非空字符串数组"
            )
        combined = "{} {}".format(gap, reason).lower()
        infra_markers = (
            "network", "database", "connection", "timeout", "redis", "mysql",
            "worker crash", "api error", "网络", "数据库", "连接失败", "超时",
        )
        if any(marker in combined for marker in infra_markers):
            raise MalformedHeuristicOutputError(
                "基础设施失败不能伪装成 KnowledgeGap"
            )
        return KnowledgeGap(
            knowledge_gap=gap.strip(),
            reason=reason.strip(),
            required_evidence=[item.strip() for item in evidence],
            **audit
        )

    @staticmethod
    def _strict_keys(data, required, allowed, label):
        missing = required - set(data)
        unknown = set(data) - allowed
        if missing:
            raise MalformedHeuristicOutputError(
                "{} 缺少字段：{}".format(label, ", ".join(sorted(missing)))
            )
        if unknown:
            raise MalformedHeuristicOutputError(
                "{} 包含禁止/未知字段：{}".format(
                    label, ", ".join(sorted(unknown))
                )
            )

    @staticmethod
    def _validate_strategy_output(request, code, operators):
        parents = list(request.parents)
        output_ops = {_operator_key(item) for item in operators}
        parent_ops = {
            _operator_key(item) for parent in parents for item in parent.operators
        }

        if request.strategy == "e1":
            if output_ops & parent_ops:
                raise MalformedHeuristicOutputError(
                    "i1/e1 contract：e1 必须使用与 parent 完全不同的 operators"
                )
            if any(code.strip() == parent.code.strip() for parent in parents):
                raise MalformedHeuristicOutputError("e1 不能原样复制 parent code")
        elif request.strategy == "e2":
            if not output_ops & parent_ops:
                raise MalformedHeuristicOutputError(
                    "e2 必须继承或重组至少一个 parent operator"
                )
            if any(code.strip() == parent.code.strip() for parent in parents):
                raise MalformedHeuristicOutputError("e2 必须形成新的重组 Candidate")
        elif request.strategy == "m1":
            parent = parents[0]
            structure_changed = _python_structure(code) != _python_structure(parent.code)
            operators_changed = output_ops != {
                _operator_key(item) for item in parent.operators
            }
            if code.strip() == parent.code.strip() or not (
                structure_changed or operators_changed
            ):
                raise MalformedHeuristicOutputError(
                    "m1 必须对 parent operator/代码结构做聚焦修订"
                )
        elif request.strategy == "m2":
            parent = parents[0]
            expected_ops = [_operator_key(item) for item in parent.operators]
            if [_operator_key(item) for item in operators] != expected_ops:
                raise MalformedHeuristicOutputError(
                    "m2 只能微调参数，不得改变 operator structure"
                )
            if _python_structure(code) != _python_structure(parent.code):
                raise MalformedHeuristicOutputError(
                    "m2 只能微调参数，不得改变 Python algorithm structure"
                )
            if _literal_signature(code) == _literal_signature(parent.code):
                raise MalformedHeuristicOutputError(
                    "m2 必须产生可观察的内部参数变化"
                )


def _validate_parent(parent, run_id):
    if not isinstance(parent, GenerationParent):
        raise GenerationContractError("parents 必须是 GenerationParent")
    if not parent.candidate_id.strip():
        raise GenerationContractError("parent candidate_id 不能为空")
    if parent.run_id != run_id:
        raise GenerationContractError(
            "parent {} 来自其他 Run：{}".format(parent.candidate_id, parent.run_id)
        )
    if not parent.code.strip() or not parent.description.strip():
        raise GenerationContractError("parent 必须包含 Code 与 Description")
    try:
        fitness = float(parent.fitness)
    except (TypeError, ValueError) as exc:
        raise GenerationContractError("parent Fitness 必须是数值") from exc
    if not isfinite(fitness):
        raise GenerationContractError("parent Fitness 必须是有限数值")
    try:
        valid_trajectory = bool(parent.trajectory) and all(
            isfinite(float(value)) for value in parent.trajectory
        )
    except (TypeError, ValueError):
        valid_trajectory = False
    if not valid_trajectory:
        raise GenerationContractError("parent 必须包含有效 Trajectory")
    if not parent.operators or any(
        not isinstance(item, str) or not item.strip() for item in parent.operators
    ):
        raise GenerationContractError("parent 必须包含 Operators")
    if parent.lineage is None:
        raise GenerationContractError("parent 必须包含 lineage")


def _validate_run_tuners_interface(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise MalformedHeuristicOutputError(
            "CandidateDraft.code 不是有效 Python：{}".format(exc.msg)
        ) from exc
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_tuners"
    ]
    if len(functions) != 1:
        raise MalformedHeuristicOutputError(
            "CandidateDraft.code 必须且只能定义一个顶层 run_tuners"
        )
    args = functions[0].args
    positional = [item.arg for item in getattr(args, "posonlyargs", [])] + [
        item.arg for item in args.args
    ]
    if (
        positional != ["file", "budget", "seed", "maxlives"]
        or args.vararg is not None
        or args.kwarg is not None
        or args.kwonlyargs
        or args.defaults
    ):
        raise MalformedHeuristicOutputError(
            "run_tuners 接口必须精确为 (file, budget, seed, maxlives)"
        )


class _NormalizeConstants(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            value = "<bool>"
        elif isinstance(node.value, (int, float, complex)):
            value = "<number>"
        elif isinstance(node.value, str):
            value = "<string>"
        elif node.value is None:
            value = "<none>"
        else:
            value = "<constant>"
        return ast.copy_location(ast.Constant(value=value), node)


def _python_structure(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise MalformedHeuristicOutputError(
            "Candidate/parent code 无法进行结构比较：{}".format(exc.msg)
        ) from exc
    normalized = _NormalizeConstants().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _literal_signature(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise MalformedHeuristicOutputError(
            "Candidate/parent code 无法提取参数：{}".format(exc.msg)
        ) from exc
    return tuple(
        (type(node.value).__name__, repr(node.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
    )


def _operator_key(value):
    return str(value).strip().casefold()


def _compress_trajectory(values, max_points=12):
    numeric = [float(value) for value in values]
    if len(numeric) <= max_points:
        return numeric
    indexes = {
        round(index * (len(numeric) - 1) / float(max_points - 1))
        for index in range(max_points)
    }
    return [numeric[index] for index in sorted(indexes)]


def _recent_lineage(lineage):
    if isinstance(lineage, Mapping):
        return [dict(lineage)]
    if isinstance(lineage, Sequence) and not isinstance(lineage, (str, bytes)):
        return [dict(item) if isinstance(item, Mapping) else item for item in lineage[-3:]]
    return [lineage]


def _record_refs(record):
    ref_keys = ("artifact_ref", "artifact_id", "chunk_ref", "ref", "source_ref")
    refs = []
    for key in ref_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            _append_unique(refs, value)
    return refs


def _append_unique(items, value):
    normalized = str(value).strip()
    if normalized and normalized not in items:
        items.append(normalized)


def _forbidden_keys(value, forbidden):
    found = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().casefold() in forbidden:
                found.add(str(key))
            found.update(_forbidden_keys(nested, forbidden))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found.update(_forbidden_keys(item, forbidden))
    return found
