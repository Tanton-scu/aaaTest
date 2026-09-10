from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from prievo_agent.cli.evaluation_worker import (
    EvaluationWorkerLoop,
    EvaluationWorkerSettings,
    main,
    unique_worker_id,
)
from prievo_agent.domain.models import EvaluationJobStatus
from prievo_agent.evaluation.queue import WorkerCrashed


class _Clock:
    def __init__(self):
        self.current = 0.0

    def __call__(self):
        return self.current


class _StopEvent:
    def __init__(self, clock, stop_after):
        self.clock = clock
        self.stop_after = stop_after
        self.waits = []
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, seconds):
        self.waits.append(float(seconds))
        self.clock.current += float(seconds)
        if len(self.waits) >= self.stop_after:
            self.stopped = True
        return self.stopped


class _Worker:
    def __init__(self, outcomes=(), recovered=0):
        self.outcomes = list(outcomes)
        self.recovered = recovered
        self.recover_calls = 0
        self.run_calls = 0

    def recover_stale(self):
        self.recover_calls += 1
        return self.recovered

    def run_once(self):
        self.run_calls += 1
        if not self.outcomes:
            return None
        value = self.outcomes.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _job(status):
    return SimpleNamespace(status=status)


class EvaluationWorkerSettingsTest(unittest.TestCase):
    def test_environment_config_is_validated_and_public_view_hides_credentials(self):
        settings = EvaluationWorkerSettings.from_environment(
            "/runtime",
            "/datasets",
            {
                "DATABASE_URL": (
                    "mysql+pymysql://private-user:private-password@mysql/prievo"
                ),
                "EVALUATION_TIMEOUT_SECONDS": "12",
                "EVALUATION_LEASE_SECONDS": "35",
                "EVALUATION_WORKER_POLL_SECONDS": "0.2",
                "EVALUATION_WORKER_MAX_POLL_SECONDS": "1.5",
                "EVALUATION_STALE_SWEEP_SECONDS": "8",
            },
        )

        public = settings.public_dict()
        self.assertEqual(35, settings.lease_seconds)
        self.assertEqual(12.0, settings.evaluation_timeout_seconds)
        self.assertNotIn("private-user", str(public))
        self.assertNotIn("private-password", str(public))
        self.assertIn("no background heartbeat", public["heartbeat_mode"])

    def test_lease_must_cover_timeout_without_background_heartbeat(self):
        with self.assertRaisesRegex(ValueError, "lease_seconds"):
            EvaluationWorkerSettings(
                "mysql+pymysql://user:pass@mysql/prievo",
                Path("/runtime"),
                Path("/datasets"),
                lease_seconds=10,
                evaluation_timeout_seconds=10,
            )

    def test_worker_id_is_unique_per_process_incarnation_and_bounded(self):
        first = unique_worker_id(
            "worker prefix", "host-A", process_id=42, nonce="nonce-a"
        )
        second = unique_worker_id(
            "worker prefix", "host-A", process_id=42, nonce="nonce-b"
        )

        self.assertNotEqual(first, second)
        self.assertLessEqual(len(first), 120)
        self.assertNotIn(" ", first)
        self.assertIn("host-A", first)


class EvaluationWorkerLoopTest(unittest.TestCase):
    def test_idle_polling_has_exponential_but_bounded_delay_and_graceful_stop(self):
        clock = _Clock()
        stop = _StopEvent(clock, stop_after=4)
        worker = _Worker()

        report = EvaluationWorkerLoop(
            worker,
            "worker-idle",
            stop_event=stop,
            idle_poll_seconds=0.1,
            max_idle_poll_seconds=0.25,
            stale_sweep_seconds=10,
            monotonic=clock,
        ).run()

        self.assertEqual([0.1, 0.2, 0.25, 0.25], stop.waits)
        self.assertEqual(4, report.cycles)
        self.assertEqual(4, report.idle_polls)
        self.assertEqual(0, report.processed_jobs)

    def test_completed_job_resets_idle_delay_and_stale_sweep_is_periodic(self):
        clock = _Clock()
        stop = _StopEvent(clock, stop_after=3)
        worker = _Worker(
            [None, _job(EvaluationJobStatus.SUCCESS), None, None],
            recovered=1,
        )

        report = EvaluationWorkerLoop(
            worker,
            "worker-active",
            stop_event=stop,
            idle_poll_seconds=0.1,
            max_idle_poll_seconds=0.4,
            stale_sweep_seconds=0.15,
            monotonic=clock,
        ).run()

        self.assertEqual([0.1, 0.1, 0.2], stop.waits)
        self.assertEqual(1, report.successful_jobs)
        self.assertEqual(1, report.processed_jobs)
        self.assertGreaterEqual(worker.recover_calls, 2)
        self.assertEqual(worker.recover_calls, report.recovered_stale_jobs)

    def test_lost_lease_is_fenced_and_loop_can_stop_without_claiming_again(self):
        clock = _Clock()
        stop = _StopEvent(clock, stop_after=1)
        worker = _Worker([WorkerCrashed("lease lost")])

        report = EvaluationWorkerLoop(
            worker,
            "worker-fenced",
            stop_event=stop,
            idle_poll_seconds=0.1,
            max_idle_poll_seconds=0.2,
            stale_sweep_seconds=5,
            monotonic=clock,
        ).run()

        self.assertEqual(1, report.lost_lease_jobs)
        self.assertEqual(0, report.processed_jobs)
        self.assertEqual(1, worker.run_calls)

    def test_once_mode_recovers_stale_and_claims_at_most_one_job(self):
        worker = _Worker(
            [
                _job(EvaluationJobStatus.RETRY_WAIT),
                _job(EvaluationJobStatus.SUCCESS),
            ],
            recovered=2,
        )
        report = EvaluationWorkerLoop(worker, "worker-once").run_one_available()

        self.assertEqual(1, worker.run_calls)
        self.assertEqual(2, report.recovered_stale_jobs)
        self.assertEqual(1, report.processed_jobs)
        self.assertEqual(1, report.retry_wait_jobs)


class EvaluationWorkerCliTest(unittest.TestCase):
    def test_health_check_connects_mysql_without_schema_mutation(self):
        instances = []

        class _Connection:
            def __init__(self):
                self.pings = []

            def ping(self, reconnect=False):
                self.pings.append(reconnect)

        class _Store:
            def __init__(self, database_url, artifact_root, ensure_schema):
                self.database_url = database_url
                self.artifact_root = artifact_root
                self.ensure_schema = ensure_schema
                self.connection = _Connection()
                self.closed = False
                instances.append(self)

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            exit_code = main(
                ["--root", directory, "--health-check"],
                environ={
                    "DATABASE_URL": "mysql+pymysql://user:pass@mysql/prievo",
                },
                store_factory=_Store,
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(instances))
        self.assertFalse(instances[0].ensure_schema)
        self.assertEqual([False], instances[0].connection.pings)
        self.assertTrue(instances[0].closed)


class ComposeInfrastructureContractTest(unittest.TestCase):
    def test_compose_only_declares_mysql_and_redis_infrastructure(self):
        compose = (
            Path(__file__).resolve().parents[1] / "docker-compose.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("  mysql:\n", compose)
        self.assertIn("  redis:\n", compose)
        self.assertNotIn("  app:\n", compose)
        self.assertNotIn("  worker:\n", compose)
        self.assertIn("${MYSQL_PORT:-3306}:3306", compose)
        self.assertIn("${REDIS_PORT:-6379}:6379", compose)

if __name__ == "__main__":
    unittest.main()
