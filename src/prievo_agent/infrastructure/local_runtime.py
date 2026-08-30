from pathlib import Path
import os
from threading import Lock

from prievo_agent.algorithm.prievo_engine import PriEvOEngine
from prievo_agent.datasets.registry import DatasetRegistry
from prievo_agent.infrastructure.agent_memory import (
    NullAgentWorkingMemory,
    RedisAgentWorkingMemory,
)
from prievo_agent.infrastructure.mysql_store import MySQLRuntimeStore
from prievo_agent.infrastructure.redis_events import NullEventBus, PublishingStore, RedisEventBus


DEFAULT_DATABASE_URL = "mysql+pymysql://prievo:prievo@127.0.0.1:3306/prievo"
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"


class LocalRuntimeComposition:
    """产品运行时装配根。

    GitHub-ready 版本只保留完整后端路径：FastAPI 接收请求并调度 Run，MySQL 是唯一
    durable facts source，Redis 只做事件通知和近期 Agent memory cache，候选评价由独立
    evaluation worker 进程消费 MySQL 中的 EvaluationJob。

    Docker 不是业务必需品；它只是本机快速拉起 MySQL/Redis 的方式。App 和 Worker 都可以
    直接用 Python 命令启动。
    """

    def __init__(self, runtime_root: Path, project_root=None) -> None:
        self.runtime_root = Path(runtime_root)
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3])
        self.dataset_registry = DatasetRegistry(self.project_root / "resources" / "datasets")
        self.database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL).strip()
        if not self.database_url.startswith("mysql+pymysql://"):
            raise RuntimeError(
                "DATABASE_URL 必须是 mysql+pymysql://...；本项目只保留 MySQL 完整模式"
            )

        self.llm_mode = _llm_mode_from_environment()
        self.research_faithful_mode = _env_bool(
            "RESEARCH_FAITHFUL_MODE", default=False
        )
        self.evaluation_timeout_seconds = _env_float(
            "EVALUATION_TIMEOUT_SECONDS", default=10.0
        )
        if not 0 < self.evaluation_timeout_seconds <= 300:
            raise ValueError("EVALUATION_TIMEOUT_SECONDS 必须在 (0, 300] 秒")

        redis_url = os.getenv("REDIS_URL", DEFAULT_REDIS_URL).strip()
        self.event_bus = RedisEventBus(redis_url) if redis_url else NullEventBus()
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
        self._datasets_synced = False
        store = self.open_store()
        store.close()

    def health_snapshot(self):
        """返回真实依赖健康状态；不伪装 worker 心跳。"""

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
            connection.ping(reconnect=True)
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

        readiness_status = "UP"
        readiness_reasons = []
        if database_status == "DOWN":
            readiness_status = "DOWN"
            readiness_reasons.append("MySQL ping 失败")
        elif redis_status == "DOWN":
            readiness_status = "DEGRADED"
            readiness_reasons.append("Redis 通知/缓存不可用；MySQL facts source 仍可用")

        worker = {
            "status": "UNVERIFIED",
            "verification": (
                "当前未实现持久化 worker heartbeat；durable EvaluationJob 只能说明队列状态，"
                "不能伪装成 worker 存活证明"
            ),
            "durable_job_counts": job_counts,
        }
        if readiness_status == "UP":
            readiness_status = "DEGRADED"
        readiness_reasons.append("Evaluation Worker 活性未通过 heartbeat 验证")

        return {
            "status": readiness_status,
            "liveness": {"status": "UP", "reason": "API 进程可响应"},
            "readiness": {
                "status": readiness_status,
                "reasons": readiness_reasons,
            },
            "database": "mysql",
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
            "evaluation_execution_mode": "external",
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
            "evaluator_mode": "real",
        }

    def open_store(self):
        with self._store_init_lock:
            ensure_schema = not self._mysql_schema_ready
            store = MySQLRuntimeStore(
                self.database_url,
                self.runtime_root / "artifacts",
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

    def execute(self, run_id: str) -> None:
        store = self.open_store()
        try:
            PriEvOEngine(
                store,
                self.dataset_registry,
                self.project_root / "resources" / "prior_knowledge",
                agent_working_memory=self.agent_working_memory,
                research_faithful_mode=self.research_faithful_mode,
                evaluation_execution_mode="external",
                evaluation_timeout_seconds=self.evaluation_timeout_seconds,
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
    if len(configured) != len(values):
        missing = sorted(set(values) - set(configured))
        raise RuntimeError(
            "LLM 配置不完整；请同时填写 {}。GitHub-ready 版本不再提供离线模型兜底。".format(
                ", ".join(missing)
            )
        )
    return "openai-compatible"
