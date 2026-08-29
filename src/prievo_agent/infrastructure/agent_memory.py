import json
import logging
from datetime import datetime, timezone


logger = logging.getLogger("prievo.agent_memory")


DEFAULT_MEMORY_SCOPE = "generation"
MEMORY_SCOPES = frozenset({"planner", "generation", "research", "repair"})

# 旧版本以 memory_type 表达记录用途。新版本以 scope 隔离近期窗口，并保留
# 原 memory_type 作为可读的记录类型；该映射让旧持久数据仍可按 scope 回源。
_MEMORY_TYPES_BY_SCOPE = {
    "generation": frozenset({
        "generation", "GENERATION", "GENERATION_SUMMARY", "EVOLUTION_ADVICE",
        "EVOLUTION_STRATEGY", "FINAL_SUMMARY", "GENERATION_EVALUATION",
        "GENERATION_HISTORY_SUMMARY",
    }),
    "planner": frozenset({
        "planner", "PLANNER", "PLANNING_EXPERIENCE", "PLANNER_SUMMARY",
        "PLANNER_HISTORY_SUMMARY",
    }),
    "research": frozenset({
        "research", "RESEARCH", "KNOWLEDGE_GAP", "LITERATURE_EVIDENCE",
        "LITERATURE_INSIGHT", "RESEARCH_RESULT", "PRIOR_EXPLANATION",
        "RESEARCH_HISTORY_SUMMARY",
    }),
    "repair": frozenset({
        "repair", "REPAIR", "CANDIDATE_FAILURE", "REPAIR_ADVICE",
        "REPAIR_DECISION", "FAILURE_LESSON", "REPAIR_HISTORY_SUMMARY",
    }),
}


def normalize_memory_scope(scope=DEFAULT_MEMORY_SCOPE):
    normalized = str(scope or DEFAULT_MEMORY_SCOPE).strip().lower()
    if normalized not in MEMORY_SCOPES:
        raise ValueError("未知 Agent Memory scope：{}".format(scope))
    return normalized


def memory_types_for_scope(scope):
    """返回一个 scope 可读取的直接值和历史 memory_type 值。"""

    return tuple(sorted(_MEMORY_TYPES_BY_SCOPE[normalize_memory_scope(scope)]))


def memory_scope_for_type(memory_type):
    value = str(memory_type)
    for scope, values in _MEMORY_TYPES_BY_SCOPE.items():
        if value in values:
            return scope
    raise ValueError("memory_type 尚未映射到 run-local scope：{}".format(value))


class _FallbackMemoryMixin:
    def load_with_fallback(self, run_id, scope, store, limit=None):
        """缓存为空时只从同 Run、同 scope 的持久历史回源并回温。"""

        normalized_scope = normalize_memory_scope(scope)
        count = min(
            getattr(self, "max_items", 30),
            max(1, int(limit or getattr(self, "max_items", 30))),
        )
        recent = self.load_recent(run_id, limit=count, scope=normalized_scope)
        if recent:
            return recent
        durable = list(
            store.agent_memories_for_run(run_id, normalized_scope, limit=count)
        )
        fallback_items = []
        for memory in durable:
            metadata = {
                "agent_memory_id": memory.id,
                "subject": memory.subject,
                "evidence_artifact_id": memory.evidence_artifact_id,
            }
            fallback_items.append(
                _memory_item(
                    memory.memory_type,
                    memory.content,
                    metadata,
                    normalized_scope,
                    created_at=memory.created_at,
                )
            )
            self.append(
                run_id,
                memory.memory_type,
                memory.content,
                scope=normalized_scope,
                **metadata
            )
        warmed = self.load_recent(run_id, limit=count, scope=normalized_scope)
        return warmed or fallback_items


class NullAgentWorkingMemory(_FallbackMemoryMixin):
    def append(self, run_id, memory_type, content, scope=DEFAULT_MEMORY_SCOPE, **metadata):
        normalize_memory_scope(scope)
        return False

    def load_recent(self, run_id, limit=None, scope=DEFAULT_MEMORY_SCOPE):
        normalize_memory_scope(scope)
        return []


class InMemoryAgentWorkingMemory(_FallbackMemoryMixin):
    """Demo/Test adapter；语义与 Redis bounded list 一致。"""

    def __init__(self, max_items=40, detailed_items=10):
        self.max_items = max_items
        self.detailed_items = max(1, int(detailed_items))
        self.items = {}

    def append(self, run_id, memory_type, content, scope=DEFAULT_MEMORY_SCOPE, **metadata):
        normalized_scope = normalize_memory_scope(scope)
        values = self.items.setdefault((str(run_id), normalized_scope), [])
        values.append(_memory_item(memory_type, content, metadata, normalized_scope))
        del values[:-self.max_items]
        return True

    def load_recent(self, run_id, limit=None, scope=DEFAULT_MEMORY_SCOPE):
        normalized_scope = normalize_memory_scope(scope)
        values = self.items.get((str(run_id), normalized_scope), [])
        return list(values[-(limit or self.max_items):])


class RedisAgentWorkingMemory(_FallbackMemoryMixin):
    """Redis 只是可丢失的近期工作区；Artifact/MySQL 仍是事实源。"""

    def __init__(self, client, ttl_seconds=86400, max_items=40, detailed_items=10):
        self.client = client
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_items = max(1, int(max_items))
        self.detailed_items = max(1, min(10, int(detailed_items)))

    def append(self, run_id, memory_type, content, scope=DEFAULT_MEMORY_SCOPE, **metadata):
        normalized_scope = normalize_memory_scope(scope)
        key = self._key(run_id, normalized_scope)
        payload = json.dumps(
            _memory_item(memory_type, content, metadata, normalized_scope),
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            pipe = self.client.pipeline()
            pipe.rpush(key, payload)
            pipe.ltrim(key, -self.max_items, -1)
            pipe.expire(key, self.ttl_seconds)
            # 迁移期仅镜像默认 generation 到旧 key，使已有调用方和旧 Redis
            # 数据可平滑升级；research/repair 从不写入这个无 scope 的旧 key。
            if normalized_scope == DEFAULT_MEMORY_SCOPE:
                legacy_key = self._legacy_key(run_id)
                pipe.rpush(legacy_key, payload)
                pipe.ltrim(legacy_key, -self.max_items, -1)
                pipe.expire(legacy_key, self.ttl_seconds)
            pipe.execute()
            return True
        except Exception as exc:
            logger.warning("Redis Agent Working Memory 写入失败，Run 将继续：%s", exc)
            return False

    def load_recent(self, run_id, limit=None, scope=DEFAULT_MEMORY_SCOPE):
        normalized_scope = normalize_memory_scope(scope)
        count = min(self.max_items, max(1, int(limit or self.max_items)))
        try:
            raw_items = self.client.lrange(
                self._key(run_id, normalized_scope), -count, -1
            )
            result = self._decode(raw_items, normalized_scope)
            if not result and normalized_scope == DEFAULT_MEMORY_SCOPE:
                legacy = self.client.lrange(self._legacy_key(run_id), -count, -1)
                result = self._decode(legacy, normalized_scope)
            return result
        except Exception as exc:
            logger.warning("Redis Agent Working Memory 读取失败，回退持久化上下文：%s", exc)
            return []

    @staticmethod
    def _key(run_id, scope=DEFAULT_MEMORY_SCOPE):
        scope = normalize_memory_scope(scope)
        return "prievo:run:{}:agent:{}:recent".format(run_id, scope)

    @staticmethod
    def _legacy_key(run_id):
        return "prievo:agent:{}:working_memory".format(run_id)

    @staticmethod
    def _decode(raw_items, expected_scope):
        result = []
        for raw in raw_items:
            try:
                item = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(item, dict) or not item.get("memory_type"):
                continue
            item_scope = normalize_memory_scope(
                item.get("scope", DEFAULT_MEMORY_SCOPE)
            )
            if item_scope == expected_scope:
                result.append(item)
        return result


def _memory_item(memory_type, content, metadata, scope, created_at=None):
    return {
        "memory_type": str(memory_type),
        "scope": normalize_memory_scope(scope),
        "content": str(content)[:2000],
        "metadata": dict(metadata),
        "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
    }
