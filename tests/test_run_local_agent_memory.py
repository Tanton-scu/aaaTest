import tempfile
import unittest
from pathlib import Path

from prievo_agent.domain.models import AgentMemory
from prievo_agent.infrastructure.working_memory import (
    InMemoryAgentWorkingMemory,
    RedisAgentWorkingMemory,
)
from prievo_agent.infrastructure.local.sqlite_store import SQLiteRuntimeStore


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.operations = []

    def rpush(self, key, value): self.operations.append(("rpush", key, value)); return self
    def ltrim(self, key, start, end): self.operations.append(("ltrim", key, start, end)); return self
    def expire(self, key, ttl): self.operations.append(("expire", key, ttl)); return self
    def execute(self):
        for name, *args in self.operations:
            getattr(self.client, name)(*args)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def pipeline(self): return FakePipeline(self)
    def rpush(self, key, value): self.values.setdefault(key, []).append(value)
    def ltrim(self, key, start, end): self.values[key] = self.values.get(key, [])[start:]
    def expire(self, key, ttl): self.ttls[key] = ttl
    def lrange(self, key, start, end): return self.values.get(key, [])[start:]


class RunLocalWorkingMemoryTests(unittest.TestCase):
    def test_in_memory_adapter_isolates_run_and_scope(self):
        memory = InMemoryAgentWorkingMemory(max_items=5)
        memory.append("run-1", "GENERATION_SUMMARY", "run-1 generation")
        memory.append(
            "run-1", "LITERATURE_EVIDENCE", "run-1 research", scope="research"
        )
        memory.append("run-2", "GENERATION_SUMMARY", "run-2 generation")

        self.assertEqual(
            ["run-1 generation"],
            [item["content"] for item in memory.load_recent("run-1")],
        )
        self.assertEqual(
            ["run-1 research"],
            [
                item["content"]
                for item in memory.load_recent("run-1", scope="research")
            ],
        )
        self.assertEqual(
            ["run-2 generation"],
            [item["content"] for item in memory.load_recent("run-2")],
        )
        self.assertEqual([], memory.load_recent("run-2", scope="repair"))

    def test_redis_uses_run_and_scope_key(self):
        client = FakeRedis()
        memory = RedisAgentWorkingMemory(client, ttl_seconds=900, max_items=5)

        memory.append("run-1", "GENERATION_SUMMARY", "generation")
        memory.append(
            "run-1", "LITERATURE_EVIDENCE", "evidence", scope="research"
        )

        key = "prievo:run:run-1:agent:research:recent"
        generation_key = "prievo:run:run-1:agent:generation:recent"
        self.assertIn(key, client.values)
        self.assertIn(generation_key, client.values)
        self.assertEqual(900, client.ttls[key])
        self.assertEqual(
            ["evidence"],
            [
                item["content"]
                for item in memory.load_recent("run-1", scope="research")
            ],
        )
        self.assertEqual(
            ["generation"],
            [item["content"] for item in memory.load_recent("run-1")],
        )


class RunLocalPersistentFallbackTests(unittest.TestCase):
    def test_sqlite_query_and_cache_warm_never_cross_run_or_scope(self):
        with tempfile.TemporaryDirectory(prefix="prievo-run-memory-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                values = [
                    AgentMemory(
                        "m-run1-generation", "run-1", "dataset-a",
                        "EVOLUTION_STRATEGY", "generation", "run-1 generation",
                    ),
                    AgentMemory(
                        "m-run1-research", "run-1", "dataset-a",
                        "LITERATURE_INSIGHT", "research", "run-1 research",
                    ),
                    AgentMemory(
                        "m-run1-repair", "run-1", "dataset-a",
                        "FAILURE_LESSON", "repair", "run-1 repair",
                    ),
                    AgentMemory(
                        "m-run2-generation", "run-2", "dataset-a",
                        "EVOLUTION_STRATEGY", "generation", "run-2 secret",
                    ),
                    # memory_type 也允许直接使用 scope。
                    AgentMemory(
                        "m-run1-direct-scope", "run-1", "dataset-a",
                        "research", "research", "run-1 direct research",
                    ),
                ]
                for item in values:
                    store.add_agent_memory(item)

                persistent = store.agent_memories_for_run(
                    "run-1", "research", limit=10
                )
                redis = FakeRedis()
                cache = RedisAgentWorkingMemory(
                    redis, ttl_seconds=900, max_items=10
                )
                warmed = cache.load_with_fallback(
                    "run-1", "research", store, limit=10
                )
            finally:
                store.close()

        self.assertEqual(
            {"m-run1-research", "m-run1-direct-scope"},
            {item.id for item in persistent},
        )
        self.assertEqual(
            {"run-1 research", "run-1 direct research"},
            {item["content"] for item in warmed},
        )
        self.assertIn(
            "prievo:run:run-1:agent:research:recent", redis.values
        )
        self.assertEqual(
            {"run-1 research", "run-1 direct research"},
            {
                item["content"]
                for item in cache.load_recent("run-1", scope="research")
            },
        )
        self.assertNotIn("run-2 secret", {item["content"] for item in warmed})
        self.assertNotIn("run-1 generation", {item["content"] for item in warmed})
        self.assertNotIn("run-1 repair", {item["content"] for item in warmed})


if __name__ == "__main__":
    unittest.main()
