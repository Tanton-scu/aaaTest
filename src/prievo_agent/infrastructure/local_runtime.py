from pathlib import Path
import os
from threading import Lock

from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore
from prievo_agent.algorithm.prievo_engine import PriEvOEngine
from prievo_agent.datasets.registry import DatasetRegistry
from prievo_agent.infrastructure.mysql_store import MySQLRuntimeStore
from prievo_agent.infrastructure.redis_events import NullEventBus, PublishingStore, RedisEventBus
from prievo_agent.infrastructure.agent_memory import (
    NullAgentWorkingMemory,
    RedisAgentWorkingMemory,
)


class LocalRuntimeComposition:
    """本地 V1 composition root：集中选择具体 adapter，不泄漏到 Application。"""

    def __init__(self, runtime_root: Path, project_root=None) -> None:
        self.runtime_root = Path(runtime_root)
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3])
        self.dataset_registry = DatasetRegistry(self.project_root / "resources" / "datasets")
        self.mode = os.getenv("PRIEVO_MODE", "demo").lower()
        if self.mode not in {"demo", "full"}:
            raise ValueError("PRIEVO_MODE 只能是 demo/full")
        self.llm_mode = _llm_mode_from_environment()
        self.research_faithful_mode = _env_bool(
            "RESEARCH_FAITHFUL_MODE", default=False
        )
        # Demo 保持单进程可启动；Full Mode 只提交/等待 durable Job，真正 claim
        # 由 docker-compose worker 服务完成。
        self.evaluation_execution_mode = (
            "external" if self.mode == "full" else "inline"
        )
        self.evaluator_mode = os.getenv("PRIEVO_EVALUATOR_MODE", "real").strip().lower()
        if self.evaluator_mode not in {"real", "fake"}:
            raise ValueError("PRIEVO_EVALUATOR_MODE 只能是 real/fake")
        if self.mode == "full" and self.evaluator_mode == "fake":
            raise ValueError("PRIEVO_EVALUATOR_MODE=fake 只允许 PRIEVO_MODE=demo")
        self.evaluation_timeout_seconds = _env_float(
            "EVALUATION_TIMEOUT_SECONDS", default=10.0
        )
        if not 0 < self.evaluation_timeout_seconds <= 300:
            raise ValueError("EVALUATION_TIMEOUT_SECONDS 必须在 (0, 300] 秒")
        redis_url = os.getenv("REDIS_URL", "")
        self.event_bus = RedisEventBus(redis_url) if self.mode == "full" and redis_url else NullEventBus()
        self.agent_working_memory = (
            RedisAgentWorkingMemory(
                self.event_bus.client,
                ttl_seconds=int(os.getenv("AGENT_MEMORY_TTL_SECONDS", "86400")),
                max_items=int(os.getenv("AGENT_MEMORY_MAX_ITEMS", "40")),
                detailed_items=int(os.getenv("AGENT_MEMORY_DETAILED_ITEMS", "10")),
            )
            if isinstance(self.event_bus, RedisEventBus)
            else NullAgentWorkingMemory()
        )
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._store_init_lock = Lock()
        self._mysql_schema_ready = False
        self._sqlite_schema_ready = False
        self._datasets_synced = False
        store = self.open_store()
        store.close()

    def health_snapshot(self):
        """执行真实依赖探测，并明确区分进程存活与服务就绪度。"""
        database_status = "UP"
        database_error = ""
        job_counts = {}
        store = None
        try:
            store = self.open_store()
            connection = getattr(store, "connection", None)
            if connection is None and hasattr(store, "store"):
                connection = getattr(store.store, "connection", None)
            if connection is None:
                raise RuntimeError("RuntimeStore 未暴露可探测的数据库连接")
            if self.mode == "full":
                connection.ping(reconnect=True)
            else:
                connection.execute("SELECT 1").fetchone()
            if self.mode == "full":
                for run in store.list_runs():
                    for job in store.evaluation_jobs_for_run(run.id):
                        status = job.status.value
                        job_counts[status] = job_counts.get(status, 0) + 1
        except Exception as exc:
            database_status = "DOWN"
            database_error = "{}: {}".format(type(exc).__name__, str(exc)[:240])
        finally:
            if store is not None:
                store.close()

        redis_status = "DISABLED"
        redis_error = ""
        if isinstance(self.event_bus, RedisEventBus):
            try:
                self.event_bus.client.ping()
                redis_status = "UP"
            except Exception as exc:
                redis_status = "DOWN"
                redis_error = "{}: {}".format(type(exc).__name__, str(exc)[:240])

        worker = {
            "status": "NOT_APPLICABLE",
            "verification": "inline evaluator 与 API 进程同生命周期",
            "durable_job_counts": job_counts,
        }
        readiness_status = "UP"
        readiness_reasons = []
        if database_status == "DOWN":
            readiness_status = "DOWN"
            readiness_reasons.append("数据库 ping 失败")
        elif redis_status == "DOWN":
            readiness_status = "DEGRADED"
            readiness_reasons.append("可选 Redis 通知/缓存不可用，数据库事实源仍可用")
        if self.mode == "full":
            worker = {
                "status": "UNVERIFIED",
                "verification": (
                    "当前未持久化独立 worker heartbeat；durable EvaluationJob "
                    "只能说明队列状态，不能伪装成 worker 存活证明"
                ),
                "durable_job_counts": job_counts,
            }
            if readiness_status == "UP":
                readiness_status = "DEGRADED"
            readiness_reasons.append("Evaluation Worker 活性未验证")

        status = readiness_status
        return {
            "status": status,
            "liveness": {"status": "UP", "reason": "API 进程可响应"},
            "readiness": {
                "status": readiness_status,
                "reasons": readiness_reasons,
            },
            "mode": self.mode,
            "database": "mysql" if self.mode == "full" else "sqlite",
            "redis": "notification-cache" if isinstance(
                self.event_bus, RedisEventBus
            ) else "disabled",
            "components": {
                "database": {
                    "status": database_status,
                    "error": database_error,
                },
                "redis": {
                    "status": redis_status,
                    "error": redis_error,
                    "required": False,
                },
                "evaluation_worker": worker,
            },
            "llm": self.llm_mode,
            "dataset_count": len(self.dataset_registry.list_datasets()),
            "research_faithful_mode": self.research_faithful_mode,
            "evaluation_execution_mode": self.evaluation_execution_mode,
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
            "evaluator_mode": self.evaluator_mode,
        }

    def open_store(self):
        if self.mode == "full":
            database_url = os.getenv("DATABASE_URL", "")
            if not database_url.startswith("mysql+pymysql://"):
                raise RuntimeError("Full Mode 必须配置 mysql+pymysql:// DATABASE_URL")
            with self._store_init_lock:
                ensure_schema = not self._mysql_schema_ready
                store = MySQLRuntimeStore(
                    database_url, self.runtime_root / "artifacts",
                    ensure_schema=ensure_schema,
                )
                self._mysql_schema_ready = True
                try:
                    if not self._datasets_synced:
                        for dataset in self.dataset_registry.list_datasets():
                            store.sync_dataset(dataset)
                        self._datasets_synced = True
                except Exception:
                    store.close()
                    raise
            return PublishingStore(store, self.event_bus)
        with self._store_init_lock:
            store = SQLiteRuntimeStore(
                self.runtime_root / "state.sqlite3", self.runtime_root / "artifacts",
                ensure_schema=not self._sqlite_schema_ready,
            )
            self._sqlite_schema_ready = True
        return store

    def execute(self, run_id: str) -> None:
        store = self.open_store()
        try:
            PriEvOEngine(
                store, self.dataset_registry,
                self.project_root / "resources" / "prior_knowledge",
                agent_working_memory=self.agent_working_memory,
            research_faithful_mode=self.research_faithful_mode,
            evaluation_execution_mode=self.evaluation_execution_mode,
            evaluation_timeout_seconds=self.evaluation_timeout_seconds,
            evaluator_mode=self.evaluator_mode,
        ).run(run_id)
        finally:
            store.close()


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("{} 必须是 true/false".format(name))


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError("{} 必须是数字".format(name)) from exc


def _llm_mode_from_environment():
    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    ark_api_key = os.getenv("ARK_API_KEY", "").strip()
    values = {
        "LLM_API_ENDPOINT": os.getenv("LLM_API_ENDPOINT", "").strip(),
        "LLM_API_KEY/ARK_API_KEY": llm_api_key or ark_api_key,
        "LLM_MODEL": os.getenv("LLM_MODEL", "").strip(),
    }
    configured = [name for name, value in values.items() if value]
    if configured and len(configured) != len(values):
        missing = sorted(set(values) - set(configured))
        raise RuntimeError(
            "LLM 配置不完整；请同时填写 {}，或全部留空使用 FakeLLM".format(
                ", ".join(missing)
            )
        )
    return "openai-compatible" if configured else "deterministic-fake"
