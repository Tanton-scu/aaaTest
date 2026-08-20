"""Full Mode 独立 Evaluation Worker 进程。

该进程只通过 MySQL durable EvaluationJob 领取工作。Redis 仅用于可选事件通知；
Job ownership、lease、status 和 budget 均以 MySQL 为事实源。当前
``EvaluationWorker`` 在 benchmark 前后续租，但没有后台 heartbeat，因此配置
会强制 lease 大于候选子进程 timeout，不能声称长任务期间存在周期 heartbeat。
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


@dataclass(frozen=True)
class EvaluationWorkerSettings:
    database_url: str
    runtime_root: Path
    dataset_root: Path
    redis_url: str = ""
    worker_id_prefix: str = "evaluation-worker"
    lease_seconds: int = 30
    evaluation_timeout_seconds: float = 10.0
    idle_poll_seconds: float = 0.25
    max_idle_poll_seconds: float = 2.0
    stale_sweep_seconds: float = 10.0
    ensure_schema: bool = True

    def __post_init__(self):
        if not self.database_url.startswith("mysql+pymysql://"):
            raise ValueError(
                "Evaluation Worker 需要 mysql+pymysql:// DATABASE_URL"
            )
        if not 0.05 <= self.idle_poll_seconds <= 30:
            raise ValueError("idle_poll_seconds 必须在 0.05～30 秒")
        if not self.idle_poll_seconds <= self.max_idle_poll_seconds <= 60:
            raise ValueError(
                "max_idle_poll_seconds 必须不小于初始 polling，且不超过 60 秒"
            )
        if not 1 <= self.stale_sweep_seconds <= 3600:
            raise ValueError("stale_sweep_seconds 必须在 1～3600 秒")
        if not 0 < self.evaluation_timeout_seconds <= 300:
            raise ValueError("evaluation_timeout_seconds 必须在 (0, 300] 秒")
        if self.lease_seconds <= self.evaluation_timeout_seconds:
            raise ValueError(
                "没有后台 heartbeat 时 lease_seconds 必须大于 evaluation timeout"
            )
        if self.lease_seconds > 3600:
            raise ValueError("lease_seconds 不得超过 3600 秒")

    @classmethod
    def from_environment(
        cls,
        runtime_root,
        dataset_root,
        environ=None,
    ):
        values = dict(os.environ if environ is None else environ)
        mode = values.get("PRIEVO_MODE", "full").strip().lower()
        if mode != "full":
            raise ValueError("独立 Evaluation Worker 只用于 PRIEVO_MODE=full")
        return cls(
            database_url=values.get("DATABASE_URL", ""),
            runtime_root=Path(runtime_root),
            dataset_root=Path(dataset_root),
            redis_url=values.get("REDIS_URL", ""),
            worker_id_prefix=values.get(
                "EVALUATION_WORKER_ID_PREFIX", "evaluation-worker"
            ),
            lease_seconds=_env_int(
                values, "EVALUATION_LEASE_SECONDS", 30
            ),
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
            ensure_schema=_env_bool(
                values, "EVALUATION_WORKER_ENSURE_SCHEMA", True
            ),
        )

    def public_dict(self):
        """不暴露数据库口令的日志/CLI 配置投影。"""

        return {
            "runtime_root": str(self.runtime_root),
            "dataset_root": str(self.dataset_root),
            "redis_notifications": bool(self.redis_url),
            "worker_id_prefix": self.worker_id_prefix,
            "lease_seconds": self.lease_seconds,
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
            "idle_poll_seconds": self.idle_poll_seconds,
            "max_idle_poll_seconds": self.max_idle_poll_seconds,
            "stale_sweep_seconds": self.stale_sweep_seconds,
            "ensure_schema": self.ensure_schema,
            "heartbeat_mode": "benchmark 前后续租；无后台 heartbeat",
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
    """一个 Worker 的有界 idle polling 与 stale recovery 循环。"""

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
            raise ValueError("idle_poll_seconds 必须大于 0")
        if max_idle_poll_seconds < idle_poll_seconds:
            raise ValueError("max_idle_poll_seconds 不能小于 idle_poll_seconds")
        if stale_sweep_seconds <= 0:
            raise ValueError("stale_sweep_seconds 必须大于 0")
        self.worker = worker
        self.worker_id = worker_id
        self.stop_event = stop_event or threading.Event()
        self.idle_poll_seconds = float(idle_poll_seconds)
        self.max_idle_poll_seconds = float(max_idle_poll_seconds)
        self.stale_sweep_seconds = float(stale_sweep_seconds)
        self.monotonic = monotonic or time.monotonic

    def run(self, max_cycles=None):
        """运行到 stop；``max_cycles`` 只用于确定性测试/诊断。"""

        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles 必须大于 0")
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
                # 失去 lease 后 fencing 已阻止当前进程结算。让 stale recovery 或
                # 其他 Worker 接管，不把陈旧结果写回。
                report.lost_lease_jobs += 1
                logger.warning("Worker 已失去评价 lease，本次结果不会结算：%s", exc)
                self.stop_event.wait(idle_delay)
                idle_delay = min(
                    self.max_idle_poll_seconds, idle_delay * 2
                )
                continue

            if job is None:
                report.idle_polls += 1
                self.stop_event.wait(idle_delay)
                idle_delay = min(
                    self.max_idle_poll_seconds, idle_delay * 2
                )
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
        """启动时回收 stale job，并最多 claim 一个当前可用 Job；不 idle wait。"""

        report = EvaluationWorkerLoopReport(self.worker_id, cycles=1)
        report.recovered_stale_jobs = self._recover_stale()
        try:
            job = self.worker.run_once()
        except WorkerCrashed as exc:
            report.lost_lease_jobs = 1
            logger.warning("单次 Worker 已失去评价 lease：%s", exc)
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
            logger.warning("已从 MySQL 回收 %s 个过期 Evaluation lease", recovered)
        return recovered


def main(argv=None, environ=None, store_factory=None, worker_factory=None):
    parser = argparse.ArgumentParser(
        description="启动 PriEvO Full Mode 独立 Evaluation Worker"
    )
    parser.add_argument("--root", type=Path, default=Path("/var/lib/prievo"))
    parser.add_argument("--dataset-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once", action="store_true", help="最多处理一个当前可用 Job 后退出"
    )
    mode.add_argument(
        "--health-check", action="store_true", help="只检查 MySQL 连接后退出"
    )
    mode.add_argument(
        "--print-config", action="store_true", help="打印脱敏配置后退出"
    )
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
            print("Evaluation Worker 健康检查通过：MySQL 可连接")
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
    logger.info("Evaluation Worker 已启动：%s", worker_id)
    logger.info("Worker 配置（已脱敏）：%s", settings.public_dict())
    try:
        report = loop.run_one_available() if args.once else loop.run()
        logger.info("Evaluation Worker 已停止：%s", report.to_dict())
        return 0
    finally:
        # 收到 SIGTERM 后不再 claim 新 Job；正在执行的受监督子进程先结算或超时，
        # 然后才走到这里关闭连接。
        store.close()


def unique_worker_id(prefix, hostname=None, process_id=None, nonce=None):
    """即使副本配置相同 prefix，也生成唯一 incarnation fencing token。"""

    host = hostname or socket.gethostname()
    pid = os.getpid() if process_id is None else int(process_id)
    suffix = nonce or uuid.uuid4().hex[:10]
    parts = [_safe_id(prefix), _safe_id(host), str(pid), _safe_id(suffix)]
    return "-".join(item for item in parts if item)[:120]


def _build_worker(store, settings, worker_id):
    # 延迟导入，让 --print-config/--health-check 与循环单测不必加载完整
    # LLM/API 依赖；真正 Worker 启动仍使用正式 executable evaluator。
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
        logger.info("收到信号 %s：停止领取新 Job，等待当前 benchmark 收尾", signum)
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
        raise ValueError("{} 必须是整数".format(name)) from exc


def _env_float(values: Mapping[str, str], name, default):
    try:
        return float(values.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError("{} 必须是数字".format(name)) from exc


def _env_bool(values: Mapping[str, str], name, default):
    raw = str(values.get(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("{} 必须是 true/false".format(name))


if __name__ == "__main__":
    raise SystemExit(main())
