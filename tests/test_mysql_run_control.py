import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
from prievo_agent.infrastructure.mysql_store import MySQLRuntimeStore
from prievo_agent.evaluation.queue import EvaluationQueueService


@unittest.skipUnless(
    os.getenv("DATABASE_URL", "").startswith("mysql+pymysql://"),
    "需要 Full Mode MySQL",
)
class MySQLRunControlTest(unittest.TestCase):
    def test_v4_runtime_lease_pause_and_atomic_cancel(self):
        suffix = uuid.uuid4().hex[:12]
        now = datetime(2026, 8, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = MySQLRuntimeStore(
                os.environ["DATABASE_URL"], Path(directory)
            )
            try:
                task = OptimizationTask(
                    "task-control-" + suffix, "mysql-control", "minimize", 2, 6,
                    dataset_id="xgboost-Covtype", generations=1, population_size=2,
                )
                run = Run(
                    "run-control-" + suffix, task.id,
                    status=RunStatus.RUNNING, dataset_id=task.dataset_id,
                )
                store.add_task(task)
                store.add_run(run)
                self.assertTrue(
                    store.claim_run_lease(run.id, "runtime-a", now, 30)
                )
                self.assertFalse(
                    store.claim_run_lease(run.id, "runtime-b", now, 30)
                )
                agent_task = AgentTask(
                    "agent-control-" + suffix, run.id, "HEURISTIC_GENERATION",
                    AgentCapability.HEURISTIC_GENERATION, "control-agent-" + suffix,
                )
                store.add_agent_task(agent_task)
                claimed_agent = store.claim_agent_task(
                    agent_task.id, "live-agent", now, 1
                )
                self.assertEqual(
                    0,
                    store.recover_orphan_agent_tasks(
                        claimed_agent.updated_at.astimezone(timezone.utc)
                        + timedelta(milliseconds=500),
                        orphan_seconds=1,
                    ),
                )
                self.assertEqual(
                    1,
                    store.recover_orphan_agent_tasks(
                        claimed_agent.updated_at.astimezone(timezone.utc)
                        + timedelta(seconds=2),
                        orphan_seconds=1,
                    ),
                )
                self.assertEqual(
                    AgentTaskStatus.PENDING,
                    store.get_agent_task(agent_task.id).status,
                )
                paused = store.request_run_pause(run.id, "mysql pause fixture")
                self.assertTrue(paused.pause_requested)
                self.assertEqual(RunStatus.RUNNING, paused.status)

                for index in range(2):
                    candidate = Candidate(
                        "candidate-control-{}-{}".format(suffix, index), run.id,
                        "def run_tuners(file, budget, seed, maxlives): return {}".format(index),
                        "mysql control", [], {},
                    )
                    store.add_candidate(candidate)
                    EvaluationQueueService(store, clock=lambda: now).submit(
                        run.id, task.id, candidate.id, index, 2
                    )
                self.assertEqual(4, store.get_run(run.id).reserved_evaluations)

                cancelled = store.request_run_cancel(run.id, "mysql cancel fixture")

                self.assertEqual(RunStatus.CANCELLED, cancelled.status)
                self.assertEqual(0, cancelled.reserved_evaluations)
                self.assertEqual("", cancelled.runtime_owner_id)
                self.assertEqual(
                    {EvaluationJobStatus.CANCELLED},
                    {job.status for job in store.evaluation_jobs_for_run(run.id)},
                )
                self.assertIsNone(
                    store.claim_next_job("late-mysql-worker", now + timedelta(seconds=1), 30)
                )
                migration = store._one(
                    "SELECT checksum FROM schema_migrations WHERE version_no=4", ()
                )
                self.assertEqual(64, len(migration["checksum"]))
                identity_migration = store._one(
                    "SELECT checksum FROM schema_migrations WHERE version_no=6", ()
                )
                self.assertEqual(64, len(identity_migration["checksum"]))
            finally:
                store.close()

    def test_p0_run_cas_and_expired_evaluation_settlement_fencing(self):
        suffix = uuid.uuid4().hex[:12]
        now = datetime(2026, 8, 13, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = MySQLRuntimeStore(os.environ["DATABASE_URL"], root / "a")
            second = MySQLRuntimeStore(
                os.environ["DATABASE_URL"], root / "b", ensure_schema=False
            )
            try:
                task = OptimizationTask(
                    "task-p0-" + suffix, "mysql p0", "minimize", 2, 6
                )
                paused = Run(
                    "run-paused-p0-" + suffix, task.id,
                    status=RunStatus.PAUSED,
                )
                first.add_task(task)
                first.add_run(paused)
                stale = first.get_run(paused.id)
                second.request_run_cancel(paused.id, "并发取消")
                with self.assertRaisesRegex(RuntimeError, "CAS"):
                    first.start_run(paused.id, stale.status, now)
                self.assertEqual(
                    RunStatus.CANCELLED, first.get_run(paused.id).status
                )

                active = Run(
                    "run-active-p0-" + suffix, task.id,
                    status=RunStatus.RUNNING,
                )
                first.add_run(active)
                self.assertTrue(
                    first.claim_run_lease(active.id, "runtime-a", now, 10)
                )
                stale_active = first.get_run(active.id)
                self.assertTrue(
                    second.renew_run_lease(
                        active.id, "runtime-a", now + timedelta(seconds=5), 10
                    )
                )
                stale_active.reserved_evaluations = 999
                first.save_run(stale_active)
                persisted_active = first.get_run(active.id)
                self.assertEqual(
                    now + timedelta(seconds=15),
                    persisted_active.runtime_lease_expires_at,
                )
                self.assertEqual(0, persisted_active.reserved_evaluations)

                candidate = Candidate(
                    "candidate-p0-" + suffix, active.id, "code", "p0", [], {}
                )
                first.add_candidate(candidate)
                job = EvaluationQueueService(first, clock=lambda: now).submit(
                    active.id, task.id, candidate.id, 1, 2
                )
                claimed = first.claim_next_job("worker-p0", now, 10)
                self.assertEqual(job.id, claimed.id)
                artifact = first.put_artifact(
                    active.id, "EVALUATION_RESULT", b"expired", "application/json"
                )
                result = EvaluationResult(
                    "result-p0-" + suffix, active.id, candidate.id,
                    0.1, [0.1], {}, 1,
                )
                expired = now + timedelta(seconds=10)
                with self.assertRaisesRegex(RuntimeError, "lease"):
                    first.complete_job_success(
                        job.id, "worker-p0", result, artifact.id, expired
                    )
                with self.assertRaisesRegex(RuntimeError, "lease"):
                    first.complete_job_failure(
                        job.id, "worker-p0", "LATE", "late", False,
                        expired, expired,
                    )
                self.assertEqual(
                    EvaluationJobStatus.RUNNING,
                    first.get_evaluation_job(job.id).status,
                )

                terminal_task = OptimizationTask(
                    "task-terminal-p0-" + suffix, "terminal", "minimize", 1, 2
                )
                terminal = Run(
                    "run-terminal-p0-" + suffix, terminal_task.id,
                    status=RunStatus.COMPLETED,
                )
                terminal_candidate = Candidate(
                    "candidate-terminal-p0-" + suffix, terminal.id,
                    "code", "terminal", [], {},
                )
                first.add_task(terminal_task)
                first.add_run(terminal)
                first.add_candidate(terminal_candidate)
                with self.assertRaisesRegex(RuntimeError, "RUNNING"):
                    EvaluationQueueService(first, clock=lambda: now).submit(
                        terminal.id, terminal_task.id, terminal_candidate.id, 1, 1
                    )
                self.assertEqual(0, first.get_run(terminal.id).reserved_evaluations)
                self.assertEqual([], list(first.evaluation_jobs_for_run(terminal.id)))
                # active job 仍是 RUNNING，terminal job 必须保持不可认领。
                self.assertIsNone(first.claim_next_job("late-worker", now, 30))
            finally:
                second.close()
                first.close()

    def test_terminal_transition_atomically_cleans_jobs_agents_and_budget(self):
        suffix = uuid.uuid4().hex[:12]
        now = datetime(2026, 8, 13, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = MySQLRuntimeStore(os.environ["DATABASE_URL"], Path(directory))
            try:
                for terminal in (RunStatus.COMPLETED, RunStatus.FAILED):
                    case = "{}-{}".format(terminal.value.lower(), suffix)
                    task = OptimizationTask(
                        "task-cleanup-" + case, case, "minimize", 2, 8
                    )
                    run = Run(
                        "run-cleanup-" + case, task.id,
                        status=RunStatus.RUNNING,
                    )
                    candidate = Candidate(
                        "candidate-cleanup-" + case, run.id,
                        "code", case, [], {},
                    )
                    store.add_task(task)
                    store.add_run(run)
                    store.add_candidate(candidate)
                    owner = "runtime-cleanup-" + case
                    self.assertTrue(store.claim_run_lease(run.id, owner, now, 30))
                    job = EvaluationQueueService(store, clock=lambda: now).submit(
                        run.id, task.id, candidate.id, seed=1, budget=2
                    )
                    claimed_job = store.claim_next_job(
                        "evaluation-cleanup-" + case, now, 30
                    )
                    self.assertEqual(job.id, claimed_job.id)

                    pending_agent = AgentTask(
                        "agent-pending-" + case, run.id, "LITERATURE_EVIDENCE",
                        AgentCapability.LITERATURE_EVIDENCE, "pending:" + case,
                    )
                    claimed_agent = AgentTask(
                        "agent-claimed-" + case, run.id, "CANDIDATE_REPAIR",
                        AgentCapability.CANDIDATE_REPAIR, "claimed:" + case,
                    )
                    store.add_agent_task(pending_agent)
                    store.add_agent_task(claimed_agent)
                    claimed_agent = store.claim_agent_task(
                        claimed_agent.id, "RepairAgent", now, 30,
                        claim_token="agent-token-" + case,
                    )

                    transitioned_at = now + timedelta(seconds=1)
                    if terminal == RunStatus.COMPLETED:
                        persisted = store.complete_run(
                            run.id, owner, transitioned_at, 1, candidate.id
                        )
                        expected_code = "RUN_COMPLETED"
                    else:
                        persisted = store.fail_run(
                            run.id, "fixture failure", transitioned_at,
                            owner_id=owner,
                        )
                        expected_code = "RUN_FAILED"

                    self.assertEqual(terminal, persisted.status)
                    self.assertEqual(0, persisted.reserved_evaluations)
                    self.assertEqual("", persisted.runtime_owner_id)
                    terminal_job = store.get_evaluation_job(job.id)
                    self.assertEqual(EvaluationJobStatus.CANCELLED, terminal_job.status)
                    self.assertEqual(expected_code, terminal_job.error_code)
                    self.assertEqual(
                        CandidateStatus.INVALID,
                        store.candidate_by_id(candidate.id).status,
                    )
                    self.assertEqual(
                        {AgentTaskStatus.CANCELLED},
                        {item.status for item in store.agent_tasks_for_run(run.id)},
                    )
                    self.assertEqual(
                        {""},
                        {item.claim_token for item in store.agent_tasks_for_run(run.id)},
                    )
                    with self.assertRaisesRegex(RuntimeError, "lease"):
                        store.complete_agent_task(
                            claimed_agent.id, claimed_agent.claim_token,
                            ["late-agent-output"],
                            transitioned_at + timedelta(seconds=1),
                        )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
