"""PriEvO numeric Top-5 之后的受约束语义筛选 Node。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from prievo_agent.domain.models import AgentCapability
from prievo_agent.knowledge.prior.models import (
    LANDSCAPE_METRICS,
    LandscapeProfile,
    SimilarInstance,
)

from prievo_agent.agents.context_policies import ContextPolicyFramework
from prievo_agent.agents.models import SimilarityDecision


class SimilaritySelectionNodeError(RuntimeError):
    """SimilaritySelectionNode 无法产生有效决策。"""


class MalformedSimilarityDecisionError(SimilaritySelectionNodeError):
    """LLM 响应不符合受约束 SimilarityDecision schema。

    Agent 不在这里偷偷回退或编造 Prior。Dispatcher/Coordinator 可以据此执行
    明确的 retry/degraded policy，并把失败原因持久化。
    """


class SimilaritySelectionNode:
    """从 deterministic numeric Top-5 中选择 1 至 3 个历史实例。

    该 Agent 的职责止于语义选择。真实 optimizer/operator evidence 必须由后续
    PriorExtractor 使用选中 ID 从 repository 读取。
    """

    name = "SimilaritySelectionNode"
    capability = AgentCapability.SEMANTIC_SIMILARITY
    skill_name = "semantic_similarity_selection"

    def __init__(
        self, skill_registry: Any, model: Any, context_framework=None
    ) -> None:
        self.skill_registry = skill_registry
        self.model = model
        self.context_framework = context_framework or ContextPolicyFramework()

    def select(
        self,
        target: LandscapeProfile,
        numeric_candidates: Sequence[SimilarInstance],
        metric_semantics: Mapping[str, Any],
    ) -> SimilarityDecision:
        """执行一次无 Memory、无 RAG 的语义筛选。

        malformed output 会显式抛出 ``MalformedSimilarityDecisionError``；调用方
        负责 retry 或降级，Agent 自身不会用 numeric ranking 冒充 LLM 决策。
        """

        candidates = tuple(numeric_candidates)
        self._validate_inputs(candidates, metric_semantics)
        skill = self.skill_registry.require(self.skill_name)
        context = self.context_framework.build(
            "similarity",
            self._context_payload(
                target, candidates, metric_semantics, skill
            ),
        )
        prompt = self._build_prompt(context.text)

        if self.model is None or not hasattr(
            self.model, "generate_similarity_decision"
        ):
            raise SimilaritySelectionNodeError("SimilaritySelectionNode 未配置 semantic selection model")

        allowed_ids = [item.instance_name for item in candidates]
        try:
            raw_decision = self.model.generate_similarity_decision(
                prompt, allowed_ids
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise MalformedSimilarityDecisionError(
                "SimilaritySelectionNode 响应不是有效 JSON object：{}".format(exc)
            ) from exc

        return self._parse_decision(
            raw_decision,
            allowed_ids,
            skill.name,
            skill.content_digest,
            prompt,
            context.metadata(),
        )

    # 集成层可以使用更显式的方法名；保持唯一实现，避免两套选择语义。
    select_similar_instances = select

    @staticmethod
    def _validate_inputs(
        candidates: Sequence[SimilarInstance], metric_semantics: Mapping[str, Any]
    ) -> None:
        if not candidates:
            raise ValueError("numeric Top-5 候选不能为空")
        if len(candidates) > 5:
            raise ValueError("SimilaritySelectionNode 只接受 numeric Top-5（最多 5 个候选）")

        ids = [item.instance_name for item in candidates]
        if any(not instance_id for instance_id in ids):
            raise ValueError("numeric candidate instance_name 不能为空")
        if len(ids) != len(set(ids)):
            raise ValueError("numeric Top-5 中存在重复 instance ID")

        missing_semantics = [
            metric
            for metric in LANDSCAPE_METRICS
            if metric not in metric_semantics or not metric_semantics[metric]
        ]
        if missing_semantics:
            raise ValueError(
                "metric semantics 缺少八项 FLA 定义：{}".format(
                    ", ".join(missing_semantics)
                )
            )

        for item in candidates:
            missing = [
                metric for metric in LANDSCAPE_METRICS if metric not in item.raw_metrics
            ]
            if missing:
                raise ValueError(
                    "候选 {} 缺少 FLA metrics：{}".format(
                        item.instance_name, ", ".join(missing)
                    )
                )

    @staticmethod
    def _context_payload(
        target: LandscapeProfile,
        candidates: Sequence[SimilarInstance],
        metric_semantics: Mapping[str, Any],
        skill,
    ):
        target_metrics = {
            metric: float(target.metrics[metric]) for metric in LANDSCAPE_METRICS
        }
        definitions = {
            metric: metric_semantics[metric] for metric in LANDSCAPE_METRICS
        }
        ranked_candidates = [
            {
                "rank": rank,
                "instance_id": item.instance_name,
                "numeric_distance": float(item.distance),
                "metrics": {
                    metric: float(item.raw_metrics[metric])
                    for metric in LANDSCAPE_METRICS
                },
            }
            for rank, item in enumerate(candidates, 1)
        ]
        return {
            "target_landscape": {
                "instance_id": target.instance_name,
                "sample_count": target.sample_count,
                "analyzer": target.analyzer,
                "metrics": target_metrics,
            },
            "fla_metrics": target_metrics,
            "metric_semantics": definitions,
            "top5_candidates": ranked_candidates,
            "numeric_distance_ranking": [
                {
                    "rank": item["rank"],
                    "instance_id": item["instance_id"],
                    "numeric_distance": item["numeric_distance"],
                }
                for item in ranked_candidates
            ],
            "skill": {
                "name": skill.name,
                "version": str(skill.version),
                "digest": skill.content_digest,
                "instructions": skill.instructions,
            },
            "output_schema": {
                "selected_instance_ids": [
                    "exact ID from numeric Top-5"
                ],
                "reason_summary": "concise synthesis based on all eight metrics",
                "metric_evidence": {
                    "selected exact instance ID": {
                        metric: "comparison and value-significance evidence"
                        for metric in LANDSCAPE_METRICS
                    }
                },
            },
        }

    @staticmethod
    def _build_prompt(context_text: str) -> str:
        return (
            "You are SimilaritySelectionNode. Select historical instances whose fitness "
            "landscapes are semantically closest to the target.\n\n"
            "The supplied ContextPolicy output contains the authoritative Skill, "
            "landscape evidence, ranked Top-5 and schema.\n\n"
            "Hard constraints:\n"
            "1. Select 1 to 3 unique instance IDs, ordered most-to-least similar.\n"
            "2. Use exact, case-sensitive IDs from numeric_top5_ranked_candidates only.\n"
            "3. Compare all eight metrics: FDC, FBD, PLO, Skewness, Kurtosis, "
            "CL, MIE, and NBC, using the supplied definitions and value significance.\n"
            "4. metric_evidence must be keyed only by selected IDs and contain one "
            "entry for every one of the eight metrics.\n"
            "5. Do not generate optimizer code, operator evidence, or a Prior.\n"
            "6. Return exactly one JSON object and no Markdown or surrounding text.\n\n"
            "Bounded Similarity context:\n{}"
        ).format(
            context_text,
        )


    @staticmethod
    def _parse_decision(
        data: Any,
        allowed_ids: Sequence[str],
        skill_name: str,
        skill_digest: str,
        prompt: str,
        context_metadata,
    ) -> SimilarityDecision:
        if not isinstance(data, Mapping):
            raise MalformedSimilarityDecisionError(
                "SimilarityDecision 顶层必须是 JSON object"
            )

        selected = data.get("selected_instance_ids")
        if (
            not isinstance(selected, list)
            or isinstance(selected, (str, bytes))
            or not 1 <= len(selected) <= 3
            or any(not isinstance(item, str) or not item for item in selected)
        ):
            raise MalformedSimilarityDecisionError(
                "selected_instance_ids 必须包含 1 至 3 个非空字符串"
            )
        if len(selected) != len(set(selected)):
            raise MalformedSimilarityDecisionError(
                "selected_instance_ids 不允许重复"
            )

        allowed = set(allowed_ids)
        outside = [item for item in selected if item not in allowed]
        if outside:
            raise MalformedSimilarityDecisionError(
                "SimilarityDecision 包含 Top-5 allowlist 外 ID：{}".format(
                    ", ".join(outside)
                )
            )

        reason = data.get("reason_summary")
        if not isinstance(reason, str) or not reason.strip():
            raise MalformedSimilarityDecisionError("reason_summary 必须是非空字符串")

        evidence = data.get("metric_evidence")
        if not isinstance(evidence, Mapping):
            raise MalformedSimilarityDecisionError("metric_evidence 必须是 JSON object")
        if set(evidence) != set(selected):
            raise MalformedSimilarityDecisionError(
                "metric_evidence 必须且只能覆盖 selected_instance_ids"
            )

        normalized_evidence: dict[str, dict[str, Any]] = {}
        for instance_id in selected:
            per_metric = evidence[instance_id]
            if not isinstance(per_metric, Mapping):
                raise MalformedSimilarityDecisionError(
                    "{} 的 metric_evidence 必须是 JSON object".format(instance_id)
                )
            missing = [
                metric for metric in LANDSCAPE_METRICS if metric not in per_metric
            ]
            if missing:
                raise MalformedSimilarityDecisionError(
                    "{} 的 metric_evidence 缺少：{}".format(
                        instance_id, ", ".join(missing)
                    )
                )
            normalized_evidence[instance_id] = {
                metric: per_metric[metric] for metric in LANDSCAPE_METRICS
            }

        return SimilarityDecision(
            selected_instance_ids=list(selected),
            reason_summary=reason.strip(),
            metric_evidence=normalized_evidence,
            skill_name=skill_name,
            skill_digest=skill_digest,
            selection_prompt=prompt,
            context_metadata=dict(context_metadata),
        )
