import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prievo_agent.domain.models import (
    AgentCapability,
    AgentTask,
    AgentTaskStatus,
    OptimizationTask,
    Run,
)
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore


class SQLiteAgentTaskTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prievo-agent-task-")
        root = Path(self.temp.name)
        self.store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
        task = OptimizationTask("task-1", "测试", "minimize", 2, 20)
        self.store.add_task(task)
        self.store.add_run(Run("run-1", task.id))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_idempotent_create_claim_complete_and_list(self):
        task = AgentTask(
            "agent-task-1", "run-1", "SEMANTIC_SIMILARITY_SELECTION",
            AgentCapability.SEMANTIC_SIMILARITY, "similarity:top5:v1",
            ["artifact-top5"],
        )
        persisted, created = self.store.add_agent_task(task)
        duplicate, duplicate_created = self.store.add_agent_task(
            AgentTask(
                "agent-task-other", "run-1", "SEMANTIC_SIMILARITY_SELECTION",
                AgentCapability.SEMANTIC_SIMILARITY, "similarity:top5:v1",
                ["artifact-other"],
            )
        )
        self.assertTrue(created)
        self.assertFalse(duplicate_created)
        self.assertEqual(persisted.id, duplicate.id)

        claimed = self.store.claim_agent_task(task.id, "SimilaritySelectionNode")
        self.assertEqual(AgentTaskStatus.CLAIMED, claimed.status)
        self.assertEqual(1, claimed.attempts)
        with self.assertRaises(RuntimeError):
            self.store.claim_agent_task(task.id, "SimilaritySelectionNode-2")

        completed = self.store.complete_agent_task(
            task.id, claimed.claim_token, ["artifact-decision"]
        )
        self.assertEqual(AgentTaskStatus.COMPLETED, completed.status)
        self.assertEqual(["artifact-decision"], completed.output_artifact_refs)
        self.assertEqual([task.id], [item.id for item in self.store.agent_tasks_for_run("run-1")])

    def test_failure_requeues_until_max_attempts(self):
        task = AgentTask(
            "agent-task-failure", "run-1", "LITERATURE_EVIDENCE",
            AgentCapability.LITERATURE_EVIDENCE, "research:gap:v1", max_attempts=2,
        )
        self.store.add_agent_task(task)
        claimed = self.store.claim_agent_task(task.id, "LiteratureEvidenceResolver")
        first = self.store.fail_agent_task(
            task.id, claimed.claim_token, "transient"
        )
        self.assertEqual(AgentTaskStatus.PENDING, first.status)
        claimed = self.store.claim_agent_task(task.id, "LiteratureEvidenceResolver")
        final = self.store.fail_agent_task(
            task.id, claimed.claim_token, "malformed twice"
        )
        self.assertEqual(AgentTaskStatus.FAILED, final.status)
        self.assertEqual("malformed twice", final.error_message)


if __name__ == "__main__":
    unittest.main()
