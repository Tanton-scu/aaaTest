"""Run/Agent scope 隔离的分层 Memory 应用服务。

MySQL/SQLite 的 ``AgentMemory`` 是事实源；Redis/InMemory 只保存近期窗口。该服务
集中完成稳定写入、cache miss 回温和结构化 Context 投影，避免各 Agent workflow
自行实现不一致的跨 Run 查询。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from prievo_agent.domain.models import AgentMemory


class _DurableMemoryOnly:
    def load_with_fallback(self, run_id, scope, store, limit=None):
        return [
            _memory_projection(item, scope)
            for item in store.agent_memories_for_run(
                run_id, scope, limit=max(1, int(limit or 30))
            )
        ]

    def load_recent(self, run_id, limit=None, scope="generation"):
        return []

    def append(self, run_id, memory_type, content, scope="generation", **metadata):
        return False


class RunLocalMemoryService:
    """单 Run、单 Agent scope 的 durable-first Memory 门面。"""

    def __init__(self, store, cache=None):
        self.store = store
        self.cache = cache if cache is not None else _DurableMemoryOnly()

    def load(self, run_id, scope, limit=6):
        values = self.cache.load_with_fallback(
            str(run_id), str(scope), self.store, limit=max(1, int(limit))
        )
        result = []
        for raw in values:
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            item_scope = str(item.get("scope", scope)).strip().lower()
            item_run_id = str(item.get("run_id", run_id))
            if item_scope != str(scope).strip().lower() or item_run_id != str(run_id):
                raise ValueError("Agent Memory 禁止跨 Run 或跨 scope 注入")
            item["run_id"] = str(run_id)
            item["scope"] = item_scope
            result.append(item)
        return result[-max(1, int(limit)):]

    def durable(self, run_id, scope, limit=1000):
        return [
            _memory_projection(item, scope)
            for item in self.store.agent_memories_for_run(
                str(run_id), str(scope), limit=max(1, int(limit))
            )
        ]

    def persist(
        self,
        *,
        run_id,
        dataset_id,
        scope,
        memory_type,
        subject,
        content,
        evidence_artifact_id="",
        identity_material="",
    ):
        canonical = _canonical_content(content)
        identity = str(identity_material or canonical)
        memory_id = "memory-{}".format(
            hashlib.sha256(
                "{}|{}|{}|{}".format(
                    run_id, scope, memory_type, identity
                ).encode("utf-8")
            ).hexdigest()[:24]
        )
        durable_scope = list(
            self.store.agent_memories_for_run(
                str(run_id), str(scope), limit=1_000_000
            )
        )
        existing = next(
            (
                item
                for item in durable_scope
                if item.id == memory_id
            ),
            None,
        )
        created_at = datetime.now(timezone.utc)
        if durable_scope:
            latest = max(item.created_at for item in durable_scope)
            if created_at <= latest:
                # Windows 的 wall clock 分辨率可能让同一批记录时间完全相同；
                # 显式单调化可避免近期窗口按 content-addressed ID 偶然重排。
                created_at = latest + timedelta(microseconds=1)
        memory = existing or AgentMemory(
            memory_id,
            str(run_id),
            str(dataset_id or ""),
            str(memory_type),
            str(subject),
            canonical,
            str(evidence_artifact_id or ""),
            created_at,
        )
        if existing is None:
            self.store.add_agent_memory(memory)

        cached = self.cache.load_recent(str(run_id), limit=30, scope=str(scope))
        if not any(_projection_id(item) == memory.id for item in cached):
            self.cache.append(
                str(run_id),
                memory.memory_type,
                memory.content,
                scope=str(scope),
                agent_memory_id=memory.id,
                subject=memory.subject,
                evidence_artifact_id=memory.evidence_artifact_id,
            )
        return memory


def memory_context(values):
    """把 cache/durable 投影转换为适合独立 ContextPolicy 的结构化记录。"""

    result = []
    for raw in values:
        item = dict(raw)
        content = item.get("content", "")
        try:
            decoded = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError:
            decoded = content
        metadata = item.get("metadata", {})
        result.append(
            {
                "agent_memory_id": _projection_id(item),
                "memory_type": str(item.get("memory_type", "")),
                "subject": str(
                    item.get("subject", metadata.get("subject", ""))
                ),
                "content": decoded,
                "evidence_artifact_id": str(
                    item.get(
                        "evidence_artifact_id",
                        metadata.get("evidence_artifact_id", ""),
                    )
                ),
            }
        )
    return result


def _memory_projection(memory, scope):
    return {
        "run_id": memory.run_id,
        "scope": str(scope),
        "memory_type": memory.memory_type,
        "content": memory.content,
        "subject": memory.subject,
        "evidence_artifact_id": memory.evidence_artifact_id,
        "agent_memory_id": memory.id,
        "metadata": {
            "agent_memory_id": memory.id,
            "subject": memory.subject,
            "evidence_artifact_id": memory.evidence_artifact_id,
        },
        "created_at": memory.created_at.isoformat(),
    }


def _projection_id(item):
    if not isinstance(item, Mapping):
        return ""
    metadata = item.get("metadata")
    nested = metadata.get("agent_memory_id", "") if isinstance(metadata, Mapping) else ""
    return str(item.get("agent_memory_id") or nested or "")


def _canonical_content(content):
    if isinstance(content, str):
        return content
    return json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


__all__ = ["RunLocalMemoryService", "memory_context"]
