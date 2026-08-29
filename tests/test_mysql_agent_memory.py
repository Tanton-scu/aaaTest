import os
import tempfile
import unittest
import uuid
from pathlib import Path

from prievo_agent.domain.models import AgentMemory, OptimizationTask, Run, ToolCallRecord
from prievo_agent.infrastructure.mysql_store import MySQLRuntimeStore


@unittest.skipUnless(os.getenv("DATABASE_URL", "").startswith("mysql+pymysql://"),
                     "需要 Full Mode MySQL")
class MySQLAgentMemoryTest(unittest.TestCase):
    def test_v2_memory_and_existing_tool_call_table_are_writable(self):
        suffix = uuid.uuid4().hex[:10]
        with tempfile.TemporaryDirectory() as directory:
            store = MySQLRuntimeStore(os.environ["DATABASE_URL"], Path(directory))
            try:
                task = OptimizationTask(
                    "task-" + suffix, "mysql-agent", "minimize", 1, 10,
                    dataset_id="xgboost-Covtype", generations=0, population_size=1,
                )
                run = Run("run-" + suffix, task.id, dataset_id=task.dataset_id)
                store.add_task(task)
                store.add_run(run)
                store.add_agent_memory(AgentMemory(
                    "memory-" + suffix, run.id, run.dataset_id,
                    "EVOLUTION_STRATEGY", "test", "bounded experience",
                ))
                store.record_tool_call(ToolCallRecord(
                    "tool-" + suffix, run.id, "literature_search", "COMPLETED",
                    {"reason": "test"}, {"result_count": 1},
                ))
                memories = store.agent_memories_for_dataset(run.dataset_id, 20)
                migration = store._one(
                    "SELECT version_no FROM schema_migrations WHERE version_no=2", ()
                )
                tool = store._one("SELECT status FROM tool_calls WHERE id=%s", ("tool-" + suffix,))
            finally:
                store.close()
        self.assertIn("memory-" + suffix, {item.id for item in memories})
        self.assertEqual(2, migration["version_no"])
        self.assertEqual("COMPLETED", tool["status"])


if __name__ == "__main__":
    unittest.main()
