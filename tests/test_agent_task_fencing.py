from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prievo_agent.domain.models import (
    AgentCapability,
    AgentTask,
    AgentTaskStatus,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.local.sqlite_store import SQLiteRuntimeStore


class AgentTaskFencingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="prievo-agent-task-fencing-"
        )
        root = Path(self.temporary.name)
        self.store = SQLiteRuntimeStore(
            root / "state.sqlite3", root / "artifacts"
        )
        self.now = datetime(2026, 8, 13, tzinfo=timezone.utc)
        optimization = OptimizationTask(
            "task-agent-fencing", "agent fencing", "minimize", 1, 5
        )
        self.run = Run(
            "run-agent-fencing", optimization.id, status=RunStatus.RUNNING
        )
        self.store.add_task(optimization)
        self.store.add_run(self.run)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def _task(self, identifier="agent-task-fencing", max_attempts=3):
        task = AgentTask(
            identifier,
            self.run.id,
            "HEURISTIC_GENERATION",
            AgentCapability.HEURISTIC_GENERATION,
            "fencing:" + identifier,
            max_attempts=max_attempts,
        )
        self.store.add_agent_task(task)
        return task

    def test_a_orphan_b_claim_then_stale_a_cannot_complete_or_fail(self):
        task = self._task()
        claimed_a = self.store.claim_agent_task(
            task.id, "GenerationAgent-A", self.now, 10,
            claim_token="claim-a",
        )
        self.assertEqual("claim-a", claimed_a.claim_token)
        self.assertEqual(self.now + timedelta(seconds=10), claimed_a.lease_expires_at)

        expired = self.now + timedelta(seconds=10)
        self.assertEqual(1, self.store.recover_orphan_agent_tasks(expired))
        pending = self.store.get_agent_task(task.id)
        self.assertEqual(AgentTaskStatus.PENDING, pending.status)
        self.assertEqual("", pending.claim_token)

        claimed_b = self.store.claim_agent_task(
            task.id, "GenerationAgent-B", expired, 10,
            claim_token="claim-b",
        )
        self.assertEqual(2, claimed_b.attempts)
        with self.assertRaisesRegex(RuntimeError, "lease"):
            self.store.complete_agent_task(
                task.id, "claim-a", ["artifact-from-stale-a"],
                expired + timedelta(seconds=1),
            )
        with self.assertRaisesRegex(RuntimeError, "lease"):
            self.store.fail_agent_task(
                task.id, "claim-a", "stale a failure",
                expired + timedelta(seconds=1),
            )

        current = self.store.get_agent_task(task.id)
        self.assertEqual(AgentTaskStatus.CLAIMED, current.status)
        self.assertEqual("GenerationAgent-B", current.claimed_by)
        self.assertEqual("claim-b", current.claim_token)
        self.assertEqual([], current.output_artifact_refs)

        completed = self.store.complete_agent_task(
            task.id, "claim-b", ["artifact-from-b"],
            expired + timedelta(seconds=1),
        )
        self.assertEqual(AgentTaskStatus.COMPLETED, completed.status)
        self.assertEqual(["artifact-from-b"], completed.output_artifact_refs)
        self.assertEqual("", completed.claim_token)
        self.assertIsNone(completed.lease_expires_at)

    def test_cancel_fences_claimed_task_and_orphan_at_max_becomes_failed(self):
        claimed_task = self._task("agent-task-cancel")
        claimed = self.store.claim_agent_task(
            claimed_task.id, "RepairAgent", self.now, 10,
            claim_token="cancelled-claim",
        )
        self.store.request_run_cancel(self.run.id, "用户取消")
        cancelled = self.store.get_agent_task(claimed_task.id)
        self.assertEqual(AgentTaskStatus.CANCELLED, cancelled.status)
        self.assertEqual("", cancelled.claim_token)
        with self.assertRaisesRegex(RuntimeError, "lease"):
            self.store.complete_agent_task(
                claimed_task.id, claimed.claim_token, ["late-output"],
                self.now + timedelta(seconds=1),
            )

        # 另建 active Run 验证最后一次 claim 过期不会永远卡 PENDING。
        optimization = OptimizationTask(
            "task-agent-max", "agent max", "minimize", 1, 5
        )
        active = Run("run-agent-max", optimization.id, status=RunStatus.RUNNING)
        self.store.add_task(optimization)
        self.store.add_run(active)
        terminal_task = AgentTask(
            "agent-task-max", active.id, "LITERATURE_EVIDENCE",
            AgentCapability.LITERATURE_EVIDENCE, "fencing:max", max_attempts=1,
        )
        self.store.add_agent_task(terminal_task)
        self.store.claim_agent_task(
            terminal_task.id, "LiteratureEvidenceResolver", self.now, 5,
            claim_token="last-claim",
        )
        self.assertEqual(
            1,
            self.store.recover_orphan_agent_tasks(
                self.now + timedelta(seconds=5)
            ),
        )
        exhausted = self.store.get_agent_task(terminal_task.id)
        self.assertEqual(AgentTaskStatus.FAILED, exhausted.status)
        self.assertEqual("", exhausted.claim_token)
        with self.assertRaisesRegex(RuntimeError, "重试耗尽"):
            self.store.claim_agent_task(
                terminal_task.id, "LiteratureEvidenceResolver",
                self.now + timedelta(seconds=6), 5,
            )


if __name__ == "__main__":
    unittest.main()
