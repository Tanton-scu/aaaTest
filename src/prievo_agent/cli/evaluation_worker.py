"""Independent Evaluation Worker process.

The worker does not serve HTTP. It only claims durable EvaluationJob rows from
MySQL. Redis is used for optional notifications; job ownership, lease, status,
attempts and budget are stored in MySQL.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from prievo_agent.datasets.registry import DatasetRegistry
from prievo_agent.infrastructure.mysql_store import MySQLRuntimeStore
from prievo_agent.infrastructure.redis_events import PublishingStore, RedisEventBus
from prievo_agent.runtime.evaluation_queue import EvaluationWorker, WorkerCrashed


logger = logging.getLogger("prievo.evaluation_worker")
DEFAULT_DATABASE_URL = "mysql+pymysql://prievo:prievo@127.0.0.1:3306/prievo"
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"


@dataclass(frozen=True)
class EvaluationWorkerSettings:
    database_url: str
    runtime_root: Path
    dataset_root: Path
    redis_url: str = DEFAULT_REDIS_URL
    worker_id_prefix: str = "evaluation-worker"
    lease_seconds: int = 30
    evaluation_timeout_seconds: float = 10.0
    idle_poll_seconds: float = 0.25
    max_idle_poll_seconds: float = 2.0
    stale_sweep_seconds: float = 10.0
    ensure_schema: bool = True

    def __post_init__(self):
        if not self.database_url.startswith("mysql+pymysql://"):
            raise ValueError("Evaluation Worker requires mysql+pymysql:// DATABASE_URL")
        if not 0.05 <= self.idle_poll_seconds <= 30:
            raise ValueError("idle_poll_seconds must be within 0.05..30 seconds")
        if not self.idle_poll_seconds <= self.max_idle_poll_seconds <= 60:
            raise ValueError(
                "max_idle_poll_seconds must be >= idle_poll_seconds and <= 60 seconds"
            )
        if not 1 <= self.stale_sweep_seconds <= 3600:
            raise ValueError("stale_sweep_seconds must be within 1..3600 seconds")
        if not 0 < self.evaluation_timeout_seconds <= 300:
            raise ValueError("evaluation_timeout_seconds must be within (0, 300] seconds")
        if self.lease_seconds <= self.evaluation_timeout_seconds:
            raise ValueError(
                "lease_seconds must be greater than evaluation timeout when no "
                "background heartbeat exists"
            )
        if self.lease_seconds > 3600:
            raise ValueError("lease_seconds must not exceed 3600 seconds")

    @classmethod
    def from_environment(cls, runtime_root, dataset_root, environ=None):
        values = dict(os.environ if environ is None else environ)
        return cls(
            database_url=values.get("DATABASE_URL", DEFAULT_DATABASE_URL),
            runtime_root=Path(runtime_root),
            dataset_root=Path(dataset_root),
            redis_url=values.get("REDIS_URL", DEFAULT_REDIS_URL),
            worker_id_prefix=values.get(
                "EVALUATION_WORKER_ID_PREFIX", "evaluation-worker"
            ),
            lease_seconds=_env_int(values, "EVALUATION_LEASE_SECONDS", 30),
            evaluation_timeout_seconds=_env_float(
                values, "EVALUATION_TIMEOUT_SECONDS", 10.0
            ),
            idle_poll_seconds=_env_float(
                values, "EVALUATION_WORKER_POLL_SECONDS", 0.25
            ),
            max_idle_poll_seconds=_env_float(
                values, "EVALUATION_WORKER_MAX_POLL_SECONDS", 2.0
            ),
            stale_sweep_seconds=_env_float(
                values, "EVALUATION_STALE_SWEEP_SECONDS", 10.0
            ),
            ensure_schema=_env_bool(values, "EVALUATION_WORKER_ENSURE_SCHEMA", True),
        )

    def public_dict(self):
        return {
            "runtime_root": str(self.runtime_root),
            "dataset_root": str(self.dataset_root),
            "database": "mysql",
            "redis_notifications": bool(self.redis_url),
            "worker_id_prefix": self.worker_id_prefix,
            "lease_seconds": self.lease_seconds,
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
            "idle_poll_seconds": self.idle_poll_seconds,
            "max_idle_poll_seconds": self.max_idle_poll_seconds,
            "stale_sweep_seconds": self.stale_sweep_seconds,
            "ensure_schema": self.ensure_schema,
            "heartbeat_mode": "renew before/after benchmark; no background heartbeat",
        }


@dataclass
class EvaluationWorkerLoopReport:
    worker_id: str
    cycles: int = 0
    processed_jobs: int = 0
    successful_jobs: int = 0
    retry_wait_jobs: int = 0
    dead_jobs: int = 0
    recovered_stale_jobs: int = 0
    lost_lease_jobs: int = 0
    idle_polls: int = 0

    def to_dict(self):
        return asdict(self)


class EvaluationWorkerLoop:
    """Bounded polling loop with stale recovery for one worker."""

    def __init__(
        self,
        worker,
        worker_id,
        stop_event=None,
        idle_poll_seconds=0.25,
        max_idle_poll_seconds=2.0,
        stale_sweep_seconds=10.0,
        monotonic=None,
    ):
        if idle_poll_seconds <= 0:
            raise ValueError("idle_poll_seconds must be positive")
        if max_idle_poll_seconds < idle_poll_seconds:
            raise ValueError("max_idle_poll_seconds cannot be smaller than idle_poll_seconds")
        if stale_sweep_seconds <= 0:
            raise ValueError("stale_sweep_seconds must be positive")
        self.worker = worker
        self.worker_id = worker_id
        self.stop_event = stop_event or threading.Event()
        self.idle_poll_seconds = float(idle_poll_seconds)
        self.max_idle_poll_seconds = float(max_idle_poll_seconds)
        self.stale_sweep_seconds = float(stale_sweep_seconds)
        self.monotonic = monotonic or time.monotonic

    def run(self, max_cycles=None):
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        report = EvaluationWorkerLoopReport(self.worker_id)
        report.recovered_stale_jobs += self._recover_stale()
        next_sweep = self.monotonic() + self.stale_sweep_seconds
        idle_delay = self.idle_poll_seconds

        while not self.stop_event.is_set():
            if max_cycles is not None and report.cycles >= max_cycles:
                break
            now = self.monotonic()
            if now >= next_sweep:
                report.recovered_stale_jobs += self._recover_stale()
                next_sweep = now + self.stale_sweep_seconds

            report.cycles += 1
            try:
                job = self.worker.run_once()
            except WorkerCrashed as exc:
                report.lost_lease_jobs += 1
                logger.warning(
                    "Worker lost evaluation lease; result will not be settled: %s",
                    exc,
                )
                self.stop_event.wait(idle_delay)
                idle_delay = min(self.max_idle_poll_seconds, idle_delay * 2)
                continue

            if job is None:
                report.idle_polls += 1
                self.stop_event.wait(idle_delay)
                idle_delay = min(self.max_idle_poll_seconds, idle_delay * 2)
                continue

            report.processed_jobs += 1
            status = job.status.value
            if status == "SUCCESS":
                report.successful_jobs += 1
            elif status == "RETRY_WAIT":
                report.retry_wait_jobs += 1
            elif status == "DEAD":
                report.dead_jobs += 1
            idle_delay = self.idle_poll_seconds

        return report

    def run_one_available(self):
        report = EvaluationWorkerLoopReport(self.worker_id, cycles=1)
        report.recovered_stale_jobs = self._recover_stale()
        try:
            job = self.worker.run_once()
        except WorkerCrashed as exc:
            report.lost_lease_jobs = 1
            logger.warning("One-shot worker lost evaluation lease: %s", exc)
            return report
        if job is None:
            report.idle_polls = 1
            return report
        report.processed_jobs = 1
        if job.status.value == "SUCCESS":
            report.successful_jobs = 1
        elif job.status.value == "RETRY_WAIT":
            report.retry_wait_jobs = 1
        elif job.status.value == "DEAD":
            report.dead_jobs = 1
        return report

    def _recover_stale(self):
        recovered = int(self.worker.recover_stale())
        if recovered:
            logger.warning("Recovered %s expired Evaluation leases from MySQL", recovered)
        return recovered


def main(argv=None, environ=None, store_factory=None, worker_factory=None):
    parser = argparse.ArgumentParser(description="Start PriEvO Evaluation Worker")
    parser.add_argument("--root", type=Path, default=Path(".prievo-runtime"))
    parser.add_argument("--dataset-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Process at most one Job and exit")
    mode.add_argument("--health-check", action="store_true", help="Check MySQL and exit")
    mode.add_argument("--print-config", action="store_true", help="Print sanitized settings")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]
    dataset_root = args.dataset_root or project_root / "resources" / "datasets"
    settings = EvaluationWorkerSettings.from_environment(
        args.root, dataset_root, environ=environ
    )
    if args.print_config:
        import json

        print(json.dumps(settings.public_dict(), ensure_ascii=False, indent=2))
        return 0

    raw_store_factory = store_factory or MySQLRuntimeStore
    if args.health_check:
        store = raw_store_factory(
            settings.database_url,
            settings.runtime_root / "artifacts",
            ensure_schema=False,
        )
        try:
            store.connection.ping(reconnect=False)
            print("Evaluation Worker health check passed: MySQL is reachable")
            return 0
        finally:
            store.close()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    store = raw_store_factory(
        settings.database_url,
        settings.runtime_root / "artifacts",
        ensure_schema=settings.ensure_schema,
    )
    if settings.redis_url:
        store = PublishingStore(store, RedisEventBus(settings.redis_url))
    stop_event = threading.Event()
    worker_id = unique_worker_id(settings.worker_id_prefix)
    make_worker = worker_factory or _build_worker
    worker = make_worker(store, settings, worker_id)
    loop = EvaluationWorkerLoop(
        worker,
        worker_id,
        stop_event=stop_event,
        idle_poll_seconds=settings.idle_poll_seconds,
        max_idle_poll_seconds=settings.max_idle_poll_seconds,
        stale_sweep_seconds=settings.stale_sweep_seconds,
    )
    _install_signal_handlers(stop_event)
    logger.info("Evaluation Worker started: %s", worker_id)
    logger.info("Worker settings (sanitized): %s", settings.public_dict())
    try:
        report = loop.run_one_available() if args.once else loop.run()
        logger.info("Evaluation Worker stopped: %s", report.to_dict())
        return 0
    finally:
        store.close()


def unique_worker_id(prefix, hostname=None, process_id=None, nonce=None):
    host = hostname or socket.gethostname()
    pid = os.getpid() if process_id is None else int(process_id)
    suffix = nonce or uuid.uuid4().hex[:10]
    parts = [_safe_id(prefix), _safe_id(host), str(pid), _safe_id(suffix)]
    return "-".join(item for item in parts if item)[:120]


def _build_worker(store, settings, worker_id):
    from prievo_agent.algorithm.dataset_evaluator import DatasetEvaluator

    registry = DatasetRegistry(settings.dataset_root)
    evaluator = DatasetEvaluator(
        registry, timeout_seconds=settings.evaluation_timeout_seconds
    )
    return EvaluationWorker(
        store,
        evaluator,
        worker_id=worker_id,
        lease_seconds=settings.lease_seconds,
    )


def _install_signal_handlers(stop_event):
    def request_stop(signum, _frame):
        logger.info(
            "Received signal %s: stop claiming new jobs and wait for current benchmark",
            signum,
        )
        stop_event.set()

    for name in ("SIGTERM", "SIGINT"):
        value = getattr(signal, name, None)
        if value is not None:
            signal.signal(value, request_stop)


def _safe_id(value):
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return normalized.strip("-._") or "worker"


def _env_int(values: Mapping[str, str], name, default):
    try:
        return int(values.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be an integer".format(name)) from exc


def _env_float(values: Mapping[str, str], name, default):
    try:
        return float(values.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a number".format(name)) from exc


def _env_bool(values: Mapping[str, str], name, default):
    raw = str(values.get(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("{} must be true/false".format(name))


if __name__ == "__main__":
    raise SystemExit(main())
