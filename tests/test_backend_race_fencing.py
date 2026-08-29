from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prievo_agent.algorithm.prievo_engine import PriEvOEngine
from prievo_agent.domain.models import (
    AgentCapability,
    AgentTask,
    AgentTaskStatus,
    Candidate,
    CandidateStatus,
    EvaluationJobStatus,
    EvaluationResult,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.evaluation_queue import EvaluationQueueService
from prievo_agent.runtime.lifecycle import RunLifecycleService
from prievo_agent.runtime.persistent_runtime import RuntimeLeaseConflict
from prievo_agent.runtime.state_machine import RunStateMachine


class _ExpectedPrepareStop(RuntimeError):
    pass


class _LeaseObservingEngine(PriEvOEngine):
    def __init__(self, *args, entered=None, release=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = entered
        self.release = release
        self.prepare_calls = 0

    def prepare(self, run_id, runtime_owner_id=""):
        self.prepare_calls += 1
        current = self.store.get_run(run_id)
        if current.status != RunStatus.RUNNING:
            raise AssertionError("prepare 必须在 RUNNING CAS 之后执行")
        if current.runtime_owner_id != runtime_owner_id or not runtime_owner_id:
            raise AssertionError("prepare 必须持有当前 Runtime owner lease")
        self.entered.set()
        if not self.release.wait(timeout=3):
            raise AssertionError("测试未释放 prepare")
        raise _ExpectedPrepareStop("fixture 在第一个 prepare 副作用前停止")


class BackendRaceFencingTest(unittest.TestCase):
    def test_stale_run_snapshot_cannot_resurrect_cancel_or_rollback_lease(self):
        with tempfile.TemporaryDirectory(prefix="prievo-run-cas-") as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            artifacts = root / "artifacts"
            first = SQLiteRuntimeStore(database, artifacts)
            task = OptimizationTask("task-cas", "run cas", "minimize", 1, 5)
            run = Run("run-cas", task.id, status=RunStatus.PAUSED)
            first.add_task(task)
            first.add_run(run)
            stale_paused = first.get_run(run.id)

            control = SQLiteRuntimeStore(database, artifacts)
            try:
                control.request_run_cancel(run.id, "并发取消")
            finally:
                control.close()

            with self.assertRaisesRegex(RuntimeError, "CAS"):
                RunLifecycleService(first, RunStateMachine()).start(stale_paused)
            self.assertEqual(RunStatus.CANCELLED, first.get_run(run.id).status)
            first.close()

            lease_store = SQLiteRuntimeStore(database, artifacts)
            active_task = OptimizationTask(
                "task-lease-rollback", "lease rollback", "minimize", 1, 5
            )
            active = Run(
                "run-lease-rollback", active_task.id, status=RunStatus.RUNNING
            )
            lease_store.add_task(active_task)
            lease_store.add_run(active)
            now = datetime(2026, 8, 13, tzinfo=timezone.utc)
            self.assertTrue(
                lease_store.claim_run_lease(active.id, "owner-a", now, 10)
            )
            stale = lease_store.get_run(active.id)
            self.assertTrue(
                lease_store.renew_run_lease(
                    active.id, "owner-a", now + timedelta(seconds=5), 10
                )
            )
            expected_expiry = now + timedelta(seconds=15)
            stale.reserved_evaluations = 999
            stale.runtime_lease_expires_at = now + timedelta(seconds=10)
            lease_store.save_run(stale)
            persisted = lease_store.get_run(active.id)
            self.assertEqual(expected_expiry, persisted.runtime_lease_expires_at)
            self.assertEqual(0, persisted.reserved_evaluations)
            with self.assertRaisesRegex(RuntimeError, "CAS"):
                lease_store.fail_run(
                    active.id, "stale executor failure",
                    now + timedelta(seconds=6),
                )
            self.assertEqual(
                RunStatus.RUNNING, lease_store.get_run(active.id).status
            )
            lease_store.close()

    def test_expired_evaluation_owner_cannot_settle_before_sweeper(self):
        with tempfile.TemporaryDirectory(prefix="prievo-expired-settle-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            now = datetime(2026, 8, 13, tzinfo=timezone.utc)
            task = OptimizationTask("task-expired", "expired", "minimize", 2, 4)
            run = Run("run-expired", task.id, status=RunStatus.RUNNING)
            candidate = Candidate("candidate-expired", run.id, "code", "d", [], {})
            store.add_task(task)
            store.add_run(run)
            store.add_candidate(candidate)
            job = EvaluationQueueService(store, clock=lambda: now).submit(
                run.id, task.id, candidate.id, seed=1, budget=2
            )
            claimed = store.claim_next_job("worker-expired", now, 10)
            self.assertEqual(job.id, claimed.id)
            expired = now + timedelta(seconds=10)
            artifact = store.put_artifact(
                run.id, "EVALUATION_RESULT", b"expired", "application/json"
            )
            result = EvaluationResult(
                "result-expired", run.id, candidate.id, 0.1, [0.1], {}, 1
            )
            with self.assertRaisesRegex(RuntimeError, "lease"):
                store.complete_job_success(
                    job.id, "worker-expired", result, artifact.id, expired
                )
            with self.assertRaisesRegex(RuntimeError, "lease"):
                store.complete_job_failure(
                    job.id, "worker-expired", "LATE", "late", False,
                    expired, expired,
                )
            persisted = store.get_evaluation_job(job.id)
            self.assertEqual(EvaluationJobStatus.RUNNING, persisted.status)
            self.assertEqual(2, store.get_run(run.id).reserved_evaluations)
            self.assertEqual(0, store.get_run(run.id).consumed_evaluations)
            store.close()

    def test_only_running_run_can_enqueue_evaluation_jobs(self):
        with tempfile.TemporaryDirectory(prefix="prievo-terminal-claim-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            now = datetime(2026, 8, 13, tzinfo=timezone.utc)
            for status in (
                RunStatus.PENDING,
                RunStatus.PAUSED,
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            ):
                suffix = status.value.lower()
                task = OptimizationTask(
                    "task-" + suffix, suffix, "minimize", 1, 2
                )
                run = Run("run-" + suffix, task.id, status=status)
                candidate = Candidate(
                    "candidate-" + suffix, run.id, "code", suffix, [], {}
                )
                store.add_task(task)
                store.add_run(run)
                store.add_candidate(candidate)
                with self.subTest(status=status.value):
                    with self.assertRaisesRegex(RuntimeError, "RUNNING"):
                        EvaluationQueueService(store, clock=lambda: now).submit(
                            run.id, task.id, candidate.id, seed=1, budget=1
                        )
                    self.assertEqual(
                        0, store.get_run(run.id).reserved_evaluations
                    )
                    self.assertEqual(
                        [], list(store.evaluation_jobs_for_run(run.id))
                    )
            store.close()

    def test_complete_and_fail_atomically_fence_all_unsettled_work(self):
        with tempfile.TemporaryDirectory(prefix="prievo-terminal-cleanup-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            now = datetime(2026, 8, 13, tzinfo=timezone.utc)
            for terminal in (RunStatus.COMPLETED, RunStatus.FAILED):
                suffix = terminal.value.lower()
                task = OptimizationTask(
                    "task-cleanup-" + suffix, suffix, "minimize", 2, 8
                )
                run = Run(
                    "run-cleanup-" + suffix, task.id, status=RunStatus.RUNNING
                )
                store.add_task(task)
                store.add_run(run)
                owner = "runtime-" + suffix
                self.assertTrue(store.claim_run_lease(run.id, owner, now, 30))

                candidates = []
                jobs = []
                queue = EvaluationQueueService(store, clock=lambda: now)
                for index in range(2):
                    candidate = Candidate(
                        "candidate-cleanup-{}-{}".format(suffix, index),
                        run.id, "code", suffix, [], {},
                    )
                    store.add_candidate(candidate)
                    candidates.append(candidate)
                    jobs.append(queue.submit(
                        run.id, task.id, candidate.id, seed=index, budget=2
                    ))
                claimed_job = store.claim_next_job(
                    "evaluation-owner-" + suffix, now, 30
                )
                self.assertIn(claimed_job.id, {job.id for job in jobs})

                pending_agent = AgentTask(
                    "agent-pending-" + suffix, run.id, "PRIOR_RESEARCH",
                    AgentCapability.PRIOR_RESEARCH, "pending:" + suffix,
                )
                claimed_agent = AgentTask(
                    "agent-claimed-" + suffix, run.id, "CANDIDATE_REPAIR",
                    AgentCapability.CANDIDATE_REPAIR, "claimed:" + suffix,
                )
                store.add_agent_task(pending_agent)
                store.add_agent_task(claimed_agent)
                claimed_agent = store.claim_agent_task(
                    claimed_agent.id, "RepairAgent", now, 30,
                    claim_token="agent-token-" + suffix,
                )

                transition_at = now + timedelta(seconds=1)
                if terminal == RunStatus.COMPLETED:
                    persisted = store.complete_run(
                        run.id, owner, transition_at, 1, "best-" + suffix
                    )
                    expected_code = "RUN_COMPLETED"
                else:
                    persisted = store.fail_run(
                        run.id, "fixture failure", transition_at, owner_id=owner
                    )
                    expected_code = "RUN_FAILED"

                self.assertEqual(terminal, persisted.status)
                self.assertEqual(0, persisted.reserved_evaluations)
                self.assertEqual("", persisted.runtime_owner_id)
                self.assertIsNone(persisted.runtime_lease_expires_at)
                terminal_jobs = list(store.evaluation_jobs_for_run(run.id))
                self.assertEqual(
                    {EvaluationJobStatus.CANCELLED},
                    {job.status for job in terminal_jobs},
                )
                self.assertEqual(
                    {expected_code}, {job.error_code for job in terminal_jobs}
                )
                self.assertEqual(
                    {CandidateStatus.INVALID},
                    {store.candidate_by_id(item.id).status for item in candidates},
                )
                terminal_agents = list(store.agent_tasks_for_run(run.id))
                self.assertEqual(
                    {AgentTaskStatus.CANCELLED},
                    {item.status for item in terminal_agents},
                )
                self.assertEqual({""}, {item.claim_token for item in terminal_agents})
                self.assertEqual(
                    {None}, {item.lease_expires_at for item in terminal_agents}
                )

                late_result = EvaluationResult(
                    "late-result-" + suffix, run.id, claimed_job.candidate_id,
                    0.1, [0.1], {}, 1,
                )
                late_artifact = store.put_artifact(
                    run.id, "EVALUATION_RESULT", b"late", "application/json"
                )
                with self.assertRaisesRegex(RuntimeError, "lease"):
                    store.complete_job_success(
                        claimed_job.id, "evaluation-owner-" + suffix,
                        late_result, late_artifact.id,
                        transition_at + timedelta(seconds=1),
                    )
                with self.assertRaisesRegex(RuntimeError, "lease"):
                    store.complete_agent_task(
                        claimed_agent.id, claimed_agent.claim_token,
                        ["late-agent-output"],
                        transition_at + timedelta(seconds=1),
                    )
            store.close()

    def test_only_runtime_lease_winner_may_enter_prepare(self):
        with tempfile.TemporaryDirectory(prefix="prievo-prepare-lease-") as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            artifacts = root / "artifacts"
            setup = SQLiteRuntimeStore(database, artifacts)
            task = OptimizationTask("task-prepare", "prepare", "minimize", 1, 2)
            run = Run("run-prepare", task.id)
            setup.add_task(task)
            setup.add_run(run)
            setup.close()

            entered = threading.Event()
            release = threading.Event()
            second_store = SQLiteRuntimeStore(database, artifacts)
            prior_root = root / "unused-prior"
            second = _LeaseObservingEngine(
                second_store, None, prior_root,
                entered=threading.Event(), release=threading.Event(),
            )
            errors = []
            first_holder = []

            def first_execute():
                first_store = SQLiteRuntimeStore(database, artifacts)
                first = _LeaseObservingEngine(
                    first_store, None, prior_root,
                    entered=entered, release=release,
                )
                first_holder.append(first)
                try:
                    first.run(run.id)
                except _ExpectedPrepareStop:
                    return
                except Exception as exc:  # pragma: no cover - 诊断
                    errors.append(exc)
                finally:
                    first_store.close()

            thread = threading.Thread(target=first_execute)
            thread.start()
            self.assertTrue(entered.wait(timeout=2), "winner 未进入 prepare")
            try:
                with self.assertRaises(RuntimeLeaseConflict):
                    second.run(run.id)
                self.assertEqual(0, second.prepare_calls)
            finally:
                release.set()
                thread.join(timeout=3)
                second_store.close()
            self.assertFalse(thread.is_alive())
            self.assertEqual([], errors)
            self.assertEqual(1, first_holder[0].prepare_calls)


if __name__ == "__main__":
    unittest.main()
