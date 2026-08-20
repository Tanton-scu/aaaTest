import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.full_mode_smoke import wait_for_full_mode_run


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class _Store:
    def __init__(self, jobs=()):
        self.jobs = list(jobs)
        self.closed = False

    def evaluation_jobs_for_run(self, _run_id):
        return list(self.jobs)

    def close(self):
        self.closed = True


class _Composition:
    def __init__(self, store):
        self.store = store

    def open_store(self):
        return self.store


class _Facade:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.index = 0

    def get_run(self, run_id):
        index = min(self.index, len(self.statuses) - 1)
        self.index += 1
        return {"run_id": run_id, "status": self.statuses[index]}


def _job(status="PENDING", attempts=0):
    return SimpleNamespace(
        id="job-1",
        status=status,
        attempts=attempts,
        worker_id=None,
        error_code=None,
    )


class FullModeSmokeDeadlineTest(unittest.TestCase):
    def test_delivery_image_contains_the_documented_test_suite(self):
        project = Path(__file__).resolve().parents[1]
        dockerfile = (project / "Dockerfile").read_text(encoding="utf-8")
        ignored = {
            line.strip().strip("/")
            for line in (project / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("COPY tests ./tests", dockerfile)
        self.assertNotIn("tests", ignored)

    def test_compose_cannot_split_app_demo_from_full_worker(self):
        project = Path(__file__).resolve().parents[1]
        compose = (project / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertEqual(2, compose.count("PRIEVO_MODE: full"))
        self.assertNotIn("PRIEVO_MODE: ${PRIEVO_MODE", compose)

    def test_completed_run_returns_and_closes_monitor_store(self):
        clock = _Clock()
        store = _Store()
        result = wait_for_full_mode_run(
            _Facade(["RUNNING", "COMPLETED"]),
            _Composition(store),
            "run-1",
            timeout_seconds=10,
            worker_ready_timeout_seconds=3,
            poll_seconds=1,
            monotonic=clock,
            sleeper=clock.sleep,
        )
        self.assertEqual("COMPLETED", result["status"])
        self.assertTrue(store.closed)

    def test_unclaimed_job_fails_before_global_timeout(self):
        clock = _Clock()
        store = _Store([_job()])
        with self.assertRaisesRegex(RuntimeError, "未领取首个 Job"):
            wait_for_full_mode_run(
                _Facade(["RUNNING"]),
                _Composition(store),
                "run-1",
                timeout_seconds=10,
                worker_ready_timeout_seconds=2,
                poll_seconds=1,
                monotonic=clock,
                sleeper=clock.sleep,
            )
        self.assertEqual(2.0, clock.value)
        self.assertTrue(store.closed)

    def test_global_timeout_applies_before_any_job_exists(self):
        clock = _Clock()
        store = _Store()
        with self.assertRaisesRegex(RuntimeError, "全局截止时间"):
            wait_for_full_mode_run(
                _Facade(["RUNNING"]),
                _Composition(store),
                "run-1",
                timeout_seconds=3,
                worker_ready_timeout_seconds=2,
                poll_seconds=1,
                monotonic=clock,
                sleeper=clock.sleep,
            )
        self.assertEqual(3.0, clock.value)
        self.assertTrue(store.closed)

    def test_failed_terminal_run_is_not_reported_as_success(self):
        store = _Store([_job("DEAD", attempts=3)])
        with self.assertRaisesRegex(RuntimeError, "提前进入终态 FAILED"):
            wait_for_full_mode_run(
                _Facade(["FAILED"]),
                _Composition(store),
                "run-1",
                timeout_seconds=3,
                worker_ready_timeout_seconds=2,
            )
        self.assertTrue(store.closed)

    def test_worker_timeout_cannot_exceed_global_timeout(self):
        store = _Store()
        with self.assertRaisesRegex(ValueError, "不能大于"):
            wait_for_full_mode_run(
                _Facade(["RUNNING"]),
                _Composition(store),
                "run-1",
                timeout_seconds=2,
                worker_ready_timeout_seconds=3,
            )
        self.assertFalse(store.closed)


if __name__ == "__main__":
    unittest.main()
