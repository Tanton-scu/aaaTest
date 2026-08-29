import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prievo_agent.application.recovery_manager import RecoveryManager
from prievo_agent.domain.models import (
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


class RecoveryManagerTest(unittest.TestCase):
    def test_startup_recovers_orphan_agent_task_and_runtime_owner(self):
        from prievo_agent.domain.models import AgentCapability, AgentTask, AgentTaskStatus

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                now = datetime(2026, 8, 12, tzinfo=timezone.utc)
                task = OptimizationTask("task-orphan", "orphan", "minimize", 1, 4)
                run = Run("run-orphan", task.id, status=RunStatus.RUNNING)
                store.add_task(task)
                store.add_run(run)
                agent_task = AgentTask(
                    "agent-task-orphan", run.id, "HEURISTIC_GENERATION",
                    AgentCapability.HEURISTIC_GENERATION, "orphan-task",
                )
                store.add_agent_task(agent_task)
                claimed_task = store.claim_agent_task(
                    agent_task.id, "dead-agent-process", now, 1
                )
                self.assertTrue(
                    store.claim_run_lease(
                        run.id,
                        "dead-runtime",
                        claimed_task.updated_at.astimezone(timezone.utc),
                        1,
                    )
                )
                self.assertEqual(
                    0,
                    store.recover_orphan_agent_tasks(
                        claimed_task.updated_at.astimezone(timezone.utc)
                        + timedelta(milliseconds=500),
                        orphan_seconds=1,
                    ),
                )
                self.assertEqual(
                    AgentTaskStatus.CLAIMED,
                    store.get_agent_task(agent_task.id).status,
                )

                scheduled = []
                report = RecoveryManager(
                    clock=lambda: (
                        claimed_task.updated_at.astimezone(timezone.utc)
                        + timedelta(seconds=2)
                    ),
                    agent_task_orphan_seconds=1,
                ).recover(store, scheduled.append)

                recovered_run = store.get_run(run.id)
                recovered_task = store.get_agent_task(agent_task.id)
                self.assertEqual(1, report.orphan_agent_tasks)
                self.assertEqual(1, report.orphan_runtime_leases)
                self.assertEqual(AgentTaskStatus.PENDING, recovered_task.status)
                self.assertEqual("", recovered_task.claimed_by)
                self.assertEqual("", recovered_run.runtime_owner_id)
                self.assertIsNone(recovered_run.runtime_lease_expires_at)
                self.assertEqual([run.id], scheduled)
            finally:
                store.close()

    def test_startup_schedules_only_active_runs_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRuntimeStore(
                Path(directory) / "state.sqlite3",
                Path(directory) / "artifacts",
            )
            try:
                for suffix, status in (
                    ("pending", RunStatus.PENDING),
                    ("running", RunStatus.RUNNING),
                    ("paused", RunStatus.PAUSED),
                    ("done", RunStatus.COMPLETED),
                ):
                    task = OptimizationTask(
                        "task-{}".format(suffix), suffix, "minimize", 1, 2
                    )
                    run = Run(
                        "run-{}".format(suffix), task.id, status=status
                    )
                    store.add_task(task)
                    store.add_run(run)

                scheduled = []
                now = datetime(2026, 8, 12, tzinfo=timezone.utc)
                first = RecoveryManager(clock=lambda: now).recover(
                    store, scheduled.append
                )
                second = RecoveryManager(clock=lambda: now).recover(
                    store, scheduled.append
                )
            finally:
                store.close()

        expected = ("run-pending", "run-running")
        self.assertEqual(expected, first.scheduled_run_ids)
        self.assertEqual(expected, second.scheduled_run_ids)
        self.assertEqual(list(expected) * 2, scheduled)
        self.assertEqual(0, first.stale_evaluation_jobs)
        self.assertEqual(0, first.reconciled_agent_tasks)

    def test_recovery_first_requeues_expired_evaluation_lease(self):
        from prievo_agent.domain.models import Candidate, EvaluationJobStatus
        from prievo_agent.runtime.evaluation_queue import EvaluationQueueService

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                now = datetime(2026, 8, 12, tzinfo=timezone.utc)
                task = OptimizationTask("task", "recovery", "minimize", 1, 2)
                run = Run("run", task.id, status=RunStatus.RUNNING)
                store.add_task(task)
                store.add_run(run)
                store.add_candidate(Candidate("candidate", run.id, "code", "d", [], {}))
                job = EvaluationQueueService(store, clock=lambda: now).submit(
                    run.id, task.id, "candidate", 101, 1
                )
                claimed = store.claim_next_job("dead-worker", now, 1)
                self.assertEqual(EvaluationJobStatus.RUNNING, claimed.status)

                scheduled = []
                report = RecoveryManager(
                    clock=lambda: now + timedelta(seconds=2)
                ).recover(store, scheduled.append)
                recovered = store.get_evaluation_job(job.id)
            finally:
                store.close()

        self.assertEqual(1, report.stale_evaluation_jobs)
        self.assertEqual(EvaluationJobStatus.PENDING, recovered.status)
        self.assertEqual(["run"], scheduled)


if __name__ == "__main__":
    unittest.main()
