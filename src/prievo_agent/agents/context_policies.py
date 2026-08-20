"""PriEvO 五类 Agent 的独立、白名单式 Context Policy。

Policy 只负责决定“哪些结构化信息可以进入哪一类 Context section”。字符预算和
通用压缩继续复用 :class:`AgentContextBuilder`。未知 payload 字段不会进入 Prompt，
只会以字段名出现在 metadata，便于 Harness 断言没有发生信息泄漏。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from prievo_agent.agents.context import AgentContextBuilder


PINNED = "pinned"
RECENT = "recent"
EVIDENCE = "evidence"


@dataclass(frozen=True)
class PolicySection:
    name: str
    channel: str
    keys: tuple
    items: tuple


@dataclass(frozen=True)
class PolicyMaterial:
    policy_name: str
    sections: tuple
    pinned: tuple
    recent: tuple
    evidence: tuple
    included_keys: tuple
    excluded_keys: tuple
    omitted_items: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyBuildResult:
    policy_name: str
    text: str
    sections: tuple
    pinned: tuple
    recent: tuple
    evidence: tuple
    included_keys: tuple
    excluded_keys: tuple
    omitted_items: Mapping
    context_metadata: Mapping

    def metadata(self):
        return {
            "policy_name": self.policy_name,
            "included_keys": list(self.included_keys),
            "excluded_keys": list(self.excluded_keys),
            "omitted_items": dict(self.omitted_items),
            "sections": [
                {
                    "name": section.name,
                    "channel": section.channel,
                    "keys": list(section.keys),
                    "item_count": len(section.items),
                }
                for section in self.sections
            ],
            "context": dict(self.context_metadata),
        }


@dataclass(frozen=True)
class _FieldRule:
    key: str
    label: str
    section: str
    channel: str = PINNED
    max_items: int = 0
    keep_recent: bool = False


class BaseContextPolicy:
    name = "base"
    rules = ()

    def prepare(self, payload):
        if not isinstance(payload, Mapping):
            raise TypeError("Context payload 必须是结构化 Mapping")
        supplied = dict(payload)
        allowed = {rule.key for rule in self.rules}
        included = []
        omitted = {}
        grouped = []

        for rule in self.rules:
            if rule.key not in supplied or _empty(supplied[rule.key]):
                continue
            value, omitted_count = _bounded(
                supplied[rule.key], rule.max_items, rule.keep_recent
            )
            if omitted_count:
                omitted[rule.key] = omitted_count
            item = _render_item(rule.label, value)
            included.append(rule.key)
            _append_group(grouped, rule.section, rule.channel, rule.key, item)

        sections = tuple(
            PolicySection(
                name=group["name"],
                channel=group["channel"],
                keys=tuple(group["keys"]),
                items=tuple(group["items"]),
            )
            for group in grouped
        )
        return PolicyMaterial(
            policy_name=self.name,
            sections=sections,
            pinned=_items_for_channel(sections, PINNED),
            recent=_items_for_channel(sections, RECENT),
            evidence=_items_for_channel(sections, EVIDENCE),
            included_keys=tuple(included),
            excluded_keys=tuple(sorted(set(supplied) - allowed)),
            omitted_items=dict(sorted(omitted.items())),
        )


class SimilarityContextPolicy(BaseContextPolicy):
    """只允许 FLA 相似性判断所需信息；无 Memory、RAG 或 Population。"""

    name = "similarity"
    rules = (
        _FieldRule("target_landscape", "Target landscape profile", "Landscape"),
        _FieldRule("fla_metrics", "Eight FLA metrics", "Landscape"),
        _FieldRule("metric_semantics", "Metric semantics", "Landscape"),
        _FieldRule(
            "top5_candidates", "Top-5 historical candidates", "Candidates",
            max_items=5,
        ),
        _FieldRule(
            "numeric_distance_ranking", "Numeric distance ranking", "Candidates",
            max_items=5,
        ),
        _FieldRule("skill", "Semantic similarity skill", "Skill"),
        _FieldRule("output_schema", "Output schema", "Contract"),
    )


class GenerationContextPolicy(BaseContextPolicy):
    """核心生成上下文；lineage 只保留最近三步。"""

    name = "generation"
    rules = (
        _FieldRule("task", "Task / benchmark contract", "Contract"),
        _FieldRule("prior", "Original PriEvO prior", "Prior"),
        _FieldRule("strategy_skill", "Current strategy skill", "Strategy"),
        _FieldRule("parents", "Current parent candidates", "Parents"),
        _FieldRule(
            "lineage", "Recent parent evolution history", "Parents",
            max_items=3, keep_recent=True,
        ),
        _FieldRule(
            "relevant_run_memory", "Relevant run-local generation memory", "Memory",
            channel=RECENT, max_items=6, keep_recent=True,
        ),
        _FieldRule(
            "evidence", "Existing literature evidence", "Evidence",
            channel=EVIDENCE, max_items=5, keep_recent=True,
        ),
        _FieldRule("output_schema", "CandidateDraft / KnowledgeGap schema", "Contract"),
    )


class ResearchContextPolicy(BaseContextPolicy):
    """只解释当前知识缺口；不读取 Generation/Repair 的无关历史。"""

    name = "research"
    rules = (
        _FieldRule("knowledge_gap", "Knowledge gap", "Gap"),
        _FieldRule("prior_slice", "Original prior slice", "Prior"),
        _FieldRule("landscape_summary", "Target landscape summary", "Landscape"),
        _FieldRule("strategy", "Current strategy", "Strategy"),
        _FieldRule("parent_summary", "Necessary parent summary", "Parent"),
        _FieldRule("skill", "Prior research skill", "Skill"),
        _FieldRule(
            "research_history", "Recent run-local research history", "Memory",
            channel=RECENT, max_items=5, keep_recent=True,
        ),
        _FieldRule(
            "evidence", "Retrieved literature evidence", "Evidence",
            channel=EVIDENCE, max_items=5, keep_recent=True,
        ),
        _FieldRule("output_schema", "Research output schema", "Contract"),
    )


class RepairContextPolicy(BaseContextPolicy):
    """只包含失败 Candidate 自身及有限修复历史。"""

    name = "repair"
    rules = (
        _FieldRule("candidate", "Failed candidate", "Candidate"),
        _FieldRule("failure", "Classified candidate failure", "Failure"),
        _FieldRule("skill", "Candidate repair skill", "Skill"),
        _FieldRule(
            "repair_history", "Recent repair history", "Repair history",
            channel=RECENT, max_items=3, keep_recent=True,
        ),
        _FieldRule(
            "relevant_failures", "Relevant run-local failure records", "Failure history",
            channel=RECENT, max_items=3, keep_recent=True,
        ),
        _FieldRule("output_schema", "RepairDecision / repaired draft schema", "Contract"),
    )


class FinalSelectionContextPolicy(BaseContextPolicy):
    """最终等优选择只看 tied candidates 和审查 Skill。"""

    name = "final_selection"
    rules = (
        _FieldRule("tied_candidates", "Tied final candidates", "Candidates"),
        _FieldRule("skill", "Final heuristic audit skill", "Skill"),
        _FieldRule("output_schema", "FinalSelectionDecision schema", "Contract"),
    )


DEFAULT_POLICIES = (
    SimilarityContextPolicy(),
    GenerationContextPolicy(),
    ResearchContextPolicy(),
    RepairContextPolicy(),
    FinalSelectionContextPolicy(),
)


class ContextPolicyFramework:
    """统一入口；选择独立 Policy 后复用已有 AgentContextBuilder。"""

    def __init__(self, builder=None, policies=DEFAULT_POLICIES):
        self.builder = builder or AgentContextBuilder()
        self._policies = {policy.name: policy for policy in policies}

    def build(self, policy_name, payload):
        normalized = _policy_name(policy_name)
        try:
            policy = self._policies[normalized]
        except KeyError as exc:
            raise KeyError("未知 Context Policy：{}".format(policy_name)) from exc
        material = policy.prepare(payload)
        built = self.builder.build(
            pinned=material.pinned,
            recent_memory=material.recent,
            # 规格禁止自动跨 Run 注入 long-term dataset memory。
            long_term_memory=(),
            evidence=material.evidence,
        )
        return PolicyBuildResult(
            policy_name=material.policy_name,
            text=built.text,
            sections=material.sections,
            pinned=material.pinned,
            recent=material.recent,
            evidence=material.evidence,
            included_keys=material.included_keys,
            excluded_keys=material.excluded_keys,
            omitted_items=material.omitted_items,
            context_metadata=built.metadata(),
        )

    @property
    def policy_names(self):
        return tuple(sorted(self._policies))


def build(policy_name, payload, builder=None):
    """模块级统一构建入口。"""

    return ContextPolicyFramework(builder=builder).build(policy_name, payload)


def _policy_name(value):
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "similarityagent": "similarity",
        "similarity_agent": "similarity",
        "heuristicgenerationagent": "generation",
        "heuristic_generation_agent": "generation",
        "generation_agent": "generation",
        "priorresearchagent": "research",
        "prior_research_agent": "research",
        "repairagent": "repair",
        "repair_agent": "repair",
        "finalselectionagent": "final_selection",
        "final_selection_agent": "final_selection",
        "finalselection": "final_selection",
    }
    return aliases.get(normalized, normalized)


def _append_group(groups, name, channel, key, item):
    for group in groups:
        if group["name"] == name and group["channel"] == channel:
            group["keys"].append(key)
            group["items"].append(item)
            return
    groups.append({
        "name": name,
        "channel": channel,
        "keys": [key],
        "items": [item],
    })


def _items_for_channel(sections, channel):
    result = []
    for section in sections:
        if section.channel != channel:
            continue
        result.extend(
            "[{}] {}".format(section.name, item) for item in section.items
        )
    return tuple(result)


def _bounded(value, max_items, keep_recent):
    if not max_items or not isinstance(value, (list, tuple)):
        return value, 0
    omitted = max(0, len(value) - max_items)
    if not omitted:
        return value, 0
    bounded = value[-max_items:] if keep_recent else value[:max_items]
    return bounded, omitted


def _render_item(label, value):
    return "{}: {}".format(
        label,
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
    )


def _empty(value):
    return value is None or value == "" or value == [] or value == () or value == {}
