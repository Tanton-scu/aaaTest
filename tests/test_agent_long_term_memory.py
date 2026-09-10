import tempfile
import unittest
from pathlib import Path

from prievo_agent.domain.models import AgentMemory
from prievo_agent.infrastructure.local.sqlite_store import SQLiteRuntimeStore


class AgentLongTermMemoryTest(unittest.TestCase):
    def test_dataset_memory_can_exclude_current_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                store.add_agent_memory(AgentMemory(
                    "m1", "old-run", "dataset-a", "FAILURE_LESSON",
                    "timeout", "Bound candidate execution before retry.",
                ))
                store.add_agent_memory(AgentMemory(
                    "m2", "new-run", "dataset-a", "EVOLUTION_STRATEGY",
                    "generation 1", "Prefer local revision.",
                ))
                values = store.agent_memories_for_dataset(
                    "dataset-a", limit=5, exclude_run_id="new-run"
                )
            finally:
                store.close()
        self.assertEqual(["m1"], [item.id for item in values])


if __name__ == "__main__":
    unittest.main()
