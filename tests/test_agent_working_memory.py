import unittest

from prievo_agent.infrastructure.working_memory import RedisAgentWorkingMemory


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


class AgentWorkingMemoryTest(unittest.TestCase):
    def test_redis_memory_is_bounded_and_has_ttl(self):
        client = FakeRedis()
        memory = RedisAgentWorkingMemory(client, ttl_seconds=3600, max_items=5)
        for index in range(8):
            self.assertTrue(memory.append("run-1", "GENERATION_SUMMARY", str(index)))
        items = memory.load_recent("run-1")
        key = "prievo:agent:run-1:working_memory"
        self.assertEqual(5, len(items))
        self.assertEqual(["3", "4", "5", "6", "7"], [item["content"] for item in items])
        self.assertEqual(3600, client.ttls[key])


if __name__ == "__main__":
    unittest.main()
