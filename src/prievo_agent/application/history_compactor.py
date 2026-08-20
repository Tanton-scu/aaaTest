"""仅在同 Run Agent 历史超过阈值时生成可追溯的低方差摘要。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from .run_local_memory import RunLocalMemoryService


class MalformedHistorySummaryError(ValueError):
    pass


class HistoryCompactor:
    """Recent Window 原样保留，早期历史只在超阈值时调用一次模型摘要。"""

    SUMMARY_TYPES = {
        "generation": "GENERATION_HISTORY_SUMMARY",
        "research": "RESEARCH_HISTORY_SUMMARY",
        "repair": "REPAIR_HISTORY_SUMMARY",
    }

    def __init__(
        self,
        store,
        skill_registry,
        model,
        working_memory=None,
        threshold=12,
        refresh_batch_size=None,
        max_input_chars=12_000,
    ):
        self.store = store
        self.skill = skill_registry.require("history_summary")
        self.model = model
        self.memory = RunLocalMemoryService(store, working_memory)
        self.threshold = max(4, int(threshold))
        self.refresh_batch_size = max(
            4, int(refresh_batch_size or max(4, self.threshold // 2))
        )
        self.max_input_chars = max(2_000, int(max_input_chars))

    def context_records(
        self,
        run_id,
        dataset_id,
        scope,
        *,
        recent_window,
    ):
        scope = str(scope).strip().lower()
        if scope not in self.SUMMARY_TYPES:
            raise ValueError("HistoryCompactor scope 必须是 generation/research/repair")
        recent_window = max(2, int(recent_window))
        durable = self.memory.durable(run_id, scope, limit=1_000_000)
        source = [
            item
            for item in durable
            if item.get("memory_type") not in self.SUMMARY_TYPES.values()
        ]
        if len(source) <= self.threshold:
            # 正常路径完全不调用 LLM；仍优先走 cache miss -> durable warm。
            return self.memory.load(run_id, scope, limit=recent_window)

        recent = source[-recent_window:]
        eligible = source[:-recent_window]
        previous = self._latest_summary(durable)
        previous_payload = _decode(previous.get("content", "")) if previous else {}
        if not isinstance(previous_payload, Mapping):
            previous_payload = {}
        covered_ids = {
            str(value)
            for value in previous_payload.get("covered_memory_ids", ())
            if str(value)
        }
        if previous is not None and not covered_ids:
            # 兼容本轮早期生成的摘要 schema。
            source_ids = {_projection_id(item) for item in eligible}
            covered_ids = source_ids.intersection(
                str(ref) for ref in previous_payload.get("source_refs", ())
            )
        uncovered = [
            item for item in eligible if _projection_id(item) not in covered_ids
        ]

        # 用水位线抑制“阈值后每新增一条就调用一次 LLM”。未达到批次的事实
        # 原样随 Context 返回，不因延迟摘要而丢失。
        if previous is not None and len(uncovered) < self.refresh_batch_size:
            return [previous, *uncovered, *recent]
        batch = uncovered[: self.refresh_batch_size]
        if not batch:
            return [previous, *recent] if previous is not None else recent
        source_digest = _digest(
            [
                previous_payload.get("source_digest", ""),
                *[
                    {
                        "id": _projection_id(item),
                        "type": item.get("memory_type", ""),
                        "content": item.get("content", ""),
                        "evidence": item.get("evidence_artifact_id", ""),
                    }
                    for item in batch
                ],
            ]
        )
        try:
            summary = self._create_summary(
                str(run_id),
                str(dataset_id or ""),
                scope,
                batch,
                source_digest,
                previous=previous,
                previous_payload=previous_payload,
            )
        except Exception as exc:
            self.store.append_event(
                str(run_id),
                "HISTORY_SUMMARY_FAILED",
                "早期历史摘要失败，已降级为已有摘要与近期窗口",
                scope=scope,
                source_count=len(eligible),
                refresh_batch_count=len(batch),
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
                skill_digest=self.skill.content_digest,
            )
            return ([previous] if previous is not None else []) + uncovered + recent
        return [summary, *uncovered[len(batch) :], *recent]

    def _latest_summary(self, durable):
        summary_types = set(self.SUMMARY_TYPES.values())
        for item in reversed(durable):
            if item.get("memory_type") in summary_types:
                return item
        return None

    def _create_summary(
        self,
        run_id,
        dataset_id,
        scope,
        batch,
        source_digest,
        *,
        previous=None,
        previous_payload=None,
    ):
        previous_payload = previous_payload or {}
        prior_refs = {
            str(ref) for ref in previous_payload.get("source_refs", ()) if str(ref)
        }
        allowed_refs = sorted(prior_refs.union(_stable_refs(batch)))
        covered_memory_ids = [
            str(value)
            for value in previous_payload.get("covered_memory_ids", ())
            if str(value)
        ]
        if previous is not None and not covered_memory_ids:
            covered_memory_ids = [
                str(ref)
                for ref in previous_payload.get("source_refs", ())
                if str(ref).startswith("memory-")
            ]
        for memory_id in (_projection_id(item) for item in batch):
            if memory_id and memory_id not in covered_memory_ids:
                covered_memory_ids.append(memory_id)
        rolling_input = {
            "previous_summary": _summary_prompt_view(previous, previous_payload),
            "new_history_batch": _bounded_history(batch, self.max_input_chars),
        }
        prompt_refs = set(_stable_refs(batch))
        if previous is not None:
            prompt_refs.update(
                {
                    str(previous.get("agent_memory_id", "")),
                    str(previous_payload.get("summary_artifact_id", "")),
                }
            )
        prompt_refs.discard("")
        prompt = (
            "You are ContextCompactor. Apply the history_summary Skill only to the "
            "supplied rolling summary and new earlier-history batch from one Run and "
            "one Agent scope. Preserve prior facts and do not infer missing facts. "
            "Return JSON only with keys summary, strategy_changes, operator_changes, "
            "fitness_trend, important_failures, constraints.\n"
            "Skill name/version/digest: {}/{}/{}\n"
            "Skill instructions:\n{}\n"
            "Run ID: {}\nScope: {}\nAllowed prompt refs: {}\n"
            "Rolling input: {}"
        ).format(
            self.skill.name,
            self.skill.version,
            self.skill.content_digest,
            self.skill.instructions,
            run_id,
            scope,
            json.dumps(sorted(prompt_refs), ensure_ascii=False),
            json.dumps(rolling_input, ensure_ascii=False, sort_keys=True, default=str),
        )
        raw = self._call_model(prompt)
        summary = _validate_summary(raw)
        summary.update(
            {
                "run_id": run_id,
                "scope": scope,
                "source_digest": source_digest,
                "source_count": len(covered_memory_ids),
                "covered_memory_ids": covered_memory_ids,
                # provenance 由代码从历史事实抽取，模型无权新增或删除。
                "source_refs": allowed_refs,
                "skill_name": self.skill.name,
                "skill_version": str(self.skill.version),
                "skill_digest": self.skill.content_digest,
            }
        )
        prompt_artifact = self.store.put_artifact(
            run_id, "HISTORY_SUMMARY_PROMPT", prompt.encode("utf-8"), "text/plain"
        )
        summary["prompt_artifact_id"] = prompt_artifact.id
        summary_artifact = self.store.put_artifact(
            run_id,
            "HISTORY_SUMMARY",
            json.dumps(summary, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        summary["summary_artifact_id"] = summary_artifact.id
        memory = self.memory.persist(
            run_id=run_id,
            dataset_id=dataset_id,
            scope=scope,
            memory_type=self.SUMMARY_TYPES[scope],
            subject="{} scope early history rolling summary".format(scope),
            content=summary,
            evidence_artifact_id=summary_artifact.id,
            identity_material=source_digest,
        )
        self.store.append_event(
            run_id,
            "HISTORY_SUMMARY_CREATED",
            "同 Run Agent 历史达到滚动水位线，已分批压缩并保留近期窗口",
            scope=scope,
            agent_memory_id=memory.id,
            source_count=len(covered_memory_ids),
            refresh_batch_count=len(batch),
            recent_window_excluded=True,
            prompt_artifact_id=prompt_artifact.id,
            summary_artifact_id=summary_artifact.id,
            skill_digest=self.skill.content_digest,
        )
        return next(
            item
            for item in self.memory.durable(run_id, scope, limit=1_000_000)
            if item.get("agent_memory_id") == memory.id
        )

    def _call_model(self, prompt):
        if hasattr(self.model, "summarize_history"):
            return self.model.summarize_history(prompt)
        if hasattr(self.model, "_agent_json"):
            return self.model._agent_json(
                prompt,
                "Return the exact HistorySummary JSON object requested. No Markdown.",
            )
        raise TypeError("当前模型未实现 summarize_history 结构化能力")


def _projection_id(item):
    if not isinstance(item, Mapping):
        return ""
    metadata = item.get("metadata")
    nested = metadata.get("agent_memory_id", "") if isinstance(metadata, Mapping) else ""
    return str(item.get("agent_memory_id") or nested or "")


def _summary_prompt_view(previous, payload):
    if previous is None or not isinstance(payload, Mapping):
        return None
    keys = (
        "summary",
        "strategy_changes",
        "operator_changes",
        "fitness_trend",
        "important_failures",
        "constraints",
        "source_count",
        "source_digest",
    )
    result = {key: payload.get(key) for key in keys if key in payload}
    result["agent_memory_id"] = _projection_id(previous)
    result["summary_artifact_id"] = str(payload.get("summary_artifact_id", ""))
    return result


def _bounded_history(values, max_chars):
    """给滚动摘要构造有硬上限的事实投影，绝不把完整历史反复回灌。"""

    result = []
    remaining = max(512, int(max_chars))
    count = max(1, len(values))
    per_item = max(256, remaining // count)
    for item in values:
        content = _decode(item.get("content", ""))
        view = {
            "agent_memory_id": _projection_id(item),
            "memory_type": str(item.get("memory_type", "")),
            "subject": str(item.get("subject", "")),
            "evidence_artifact_id": str(item.get("evidence_artifact_id", "")),
            "content": content,
        }
        encoded = json.dumps(
            view, ensure_ascii=False, sort_keys=True, default=str
        )
        if len(encoded) > per_item:
            view["content"] = encoded[: max(64, per_item - 180)] + "…[truncated]"
        result.append(view)
    encoded_result = json.dumps(
        result, ensure_ascii=False, sort_keys=True, default=str
    )
    if len(encoded_result) > max_chars:
        # 极端长 ID/subject 仍受最终硬上限保护；截断只影响摘要优化输入，
        # authoritative memory 与 provenance refs 均未改变。
        return [{"bounded_history_excerpt": encoded_result[:max_chars]}]
    return result


def _validate_summary(value):
    if not isinstance(value, Mapping):
        raise MalformedHistorySummaryError("HistorySummary 必须是 JSON object")
    required = {
        "summary",
        "strategy_changes",
        "operator_changes",
        "fitness_trend",
        "important_failures",
        "constraints",
    }
    if set(value) != required:
        raise MalformedHistorySummaryError(
            "HistorySummary 字段必须严格等于 {}".format(sorted(required))
        )
    result = dict(value)
    if not isinstance(result["summary"], str) or not result["summary"].strip():
        raise MalformedHistorySummaryError("HistorySummary.summary 不能为空")
    if not isinstance(result["fitness_trend"], str):
        raise MalformedHistorySummaryError("fitness_trend 必须是字符串")
    for key in (
        "strategy_changes",
        "operator_changes",
        "important_failures",
        "constraints",
    ):
        if not isinstance(result[key], list) or not all(
            isinstance(item, str) for item in result[key]
        ):
            raise MalformedHistorySummaryError("{} 必须是 string list".format(key))
    return result


def _stable_refs(values):
    refs = set()

    def visit(value, key=""):
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            # AgentMemory.content 是 canonical JSON 字符串；provenance 提取必须
            # 继续向内读取，否则 Candidate/Artifact refs 会在摘要时丢失。
            if key == "content":
                decoded = _decode(value)
                if decoded is not value:
                    visit(decoded, key)
                    return
            if (
                key.endswith("_ref")
                or key.endswith("_id")
                or key.endswith("_refs")
                or key in {"agent_memory_id", "evidence_artifact_id"}
            ) and value:
                refs.add(value)

    visit(values)
    return refs


def _decode(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _digest(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["HistoryCompactor", "MalformedHistorySummaryError"]
