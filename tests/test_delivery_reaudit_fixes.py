import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from prievo_agent.application.tools.tool_governance import (
    CandidateCodeAuditTool,
    CandidateInspectionTool,
    ToolCallDenied,
    ToolGovernanceGateway,
)
from prievo_agent.domain.models import (
    Candidate,
    OptimizationTask,
    Run,
    ToolCallRecord,
)
from prievo_agent.infrastructure.local_runtime import LocalRuntimeComposition
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore


class CandidateInspectionIsolationTest(unittest.TestCase):
    def test_sqlite_tool_audit_same_tick_preserves_insertion_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask("task-order", "排序", "minimize", 1, 2)
                run = Run("run-order", task.id)
                store.add_task(task)
                store.add_run(run)
                same_tick = datetime(2026, 8, 13, tzinfo=timezone.utc)
                # 故意让 UUID 字典序与插入顺序相反；不能按随机 ID 排序。
                store.record_tool_call(ToolCallRecord(
                    "tool-z-first", run.id, "fixture", "COMPLETED", {}, {},
                    same_tick,
                ))
                store.record_tool_call(ToolCallRecord(
                    "tool-a-second", run.id, "fixture", "DENIED", {}, {},
                    same_tick,
                ))
                self.assertEqual(
                    ["tool-z-first", "tool-a-second"],
                    [item.id for item in store.tool_calls_for_run(run.id)],
                )
            finally:
                store.close()

    def test_candidate_code_audit_is_static_governed_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask("task-audit", "代码审查", "minimize", 3, 30)
                run = Run("run-audit", task.id)
                store.add_task(task)
                store.add_run(run)
                tool = CandidateCodeAuditTool(ToolGovernanceGateway(store))

                passed = tool.audit(
                    run.id,
                    "def run_tuners(file, budget, seed, maxlives):\n    return []\n",
                    "提交评估前做静态审查",
                    candidate_id="candidate-audit",
                )

                self.assertEqual("PASSED", passed["status"])
                self.assertEqual("run_tuners", passed["function_name"])
                self.assertIn("python_ast_parse", passed["checks"])
                self.assertTrue(passed["tool_call_ref"].startswith("tool-call-"))
                self.assertEqual(
                    "candidate_code_audit",
                    list(store.tool_calls_for_run(run.id))[-1].tool_name,
                )

                with self.assertRaises(Exception):
                    tool.audit(
                        run.id,
                        "import os\ndef run_tuners(file, budget, seed, maxlives):\n    return []\n",
                        "拦截带 import 的候选代码",
                        candidate_id="candidate-bad",
                    )
                failed = list(store.tool_calls_for_run(run.id))[-1]
                self.assertEqual("FAILED", failed.status)
                self.assertEqual("candidate_code_audit", failed.tool_name)
            finally:
                store.close()

    def test_cross_run_candidate_is_denied_audited_and_not_leaked_as_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask("task", "隔离测试", "minimize", 3, 30)
                store.add_task(task)
                store.add_run(Run("run-a", task.id))
                store.add_run(Run("run-b", task.id))
                store.add_candidate(Candidate(
                    "candidate-b", "run-b", "secret code", "secret description",
                    ["Revise"], {},
                ))
                store.add_candidate(Candidate(
                    "candidate-a", "run-a", "safe code", "safe description",
                    ["Revise"], {"parents": ["candidate-b"]},
                ))
                tool = CandidateInspectionTool(
                    ToolGovernanceGateway(store), store
                )

                allowed = tool.inspect("run-a", "candidate-a", "诊断当前候选")
                self.assertEqual(
                    [{"id": "candidate-b", "forbidden": True}],
                    allowed["parents"],
                )
                self.assertNotIn("secret description", str(allowed))

                with self.assertRaisesRegex(ToolCallDenied, "不属于当前 Run"):
                    tool.inspect("run-a", "candidate-b", "诊断失败候选")

                calls = list(store.tool_calls_for_run("run-a"))
                self.assertEqual("DENIED", calls[-1].status)
                self.assertEqual(
                    "candidate-b", calls[-1].request["input_metadata"]["candidate_id"]
                )
                denied_events = [
                    event for event in store.events_for_run("run-a")
                    if event.event_type == "TOOL_CALL_DENIED"
                ]
                self.assertEqual(1, len(denied_events))
                self.assertIn("跨 Run", denied_events[0].payload["denied_reason"])
            finally:
                store.close()


class _FakeMySQLStore:
    instances = []
    synced_dataset_ids = []

    def __init__(self, database_url, artifact_root, ensure_schema=True):
        self.database_url = database_url
        self.artifact_root = artifact_root
        self.ensure_schema = ensure_schema
        self.connection = self
        self.closed = False
        self.__class__.instances.append(self)

    def sync_dataset(self, dataset):
        self.__class__.synced_dataset_ids.append(dataset.id)

    def ping(self, reconnect=True):
        return True

    def list_runs(self):
        return []

    def evaluation_jobs_for_run(self, run_id):
        return []

    def close(self):
        self.closed = True


class RuntimeHealthAndInitializationTest(unittest.TestCase):
    def setUp(self):
        _FakeMySQLStore.instances = []
        _FakeMySQLStore.synced_dataset_ids = []

    def test_full_mode_syncs_dataset_metadata_only_during_initialization(self):
        environment = {
            "DATABASE_URL": "mysql+pymysql://u:p@db/prievo",
            "REDIS_URL": "",
            "LLM_API_ENDPOINT": "https://llm.example",
            "LLM_API_KEY": "secret",
            "LLM_MODEL": "model",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, environment, clear=False
        ), patch(
            "prievo_agent.infrastructure.local_runtime.MySQLRuntimeStore",
            _FakeMySQLStore,
        ):
            composition = LocalRuntimeComposition(Path(directory))
            expected_ids = {
                item.id for item in composition.dataset_registry.list_datasets()
            }
            first = composition.open_store()
            first.close()
            second = composition.open_store()
            second.close()
            snapshot = composition.health_snapshot()

        self.assertEqual(expected_ids, set(_FakeMySQLStore.synced_dataset_ids))
        self.assertEqual(
            len(expected_ids), len(_FakeMySQLStore.synced_dataset_ids),
            "后续读请求不得重复写入全部 Dataset metadata",
        )
        self.assertTrue(_FakeMySQLStore.instances[0].ensure_schema)
        self.assertTrue(all(
            not item.ensure_schema for item in _FakeMySQLStore.instances[1:]
        ))
        self.assertEqual("DEGRADED", snapshot["status"])
        self.assertEqual(
            "UNVERIFIED",
            snapshot["components"]["evaluation_worker"]["status"],
        )
        self.assertIn(
            "heartbeat",
            snapshot["components"]["evaluation_worker"]["verification"],
        )

    def test_database_ping_failure_makes_readiness_down(self):
        environment = {
            "DATABASE_URL": "mysql+pymysql://u:p@db/prievo",
            "REDIS_URL": "",
            "LLM_API_ENDPOINT": "https://llm.example",
            "LLM_API_KEY": "secret",
            "LLM_MODEL": "model",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, environment, clear=False
        ), patch(
            "prievo_agent.infrastructure.local_runtime.MySQLRuntimeStore",
            _FakeMySQLStore,
        ):
            composition = LocalRuntimeComposition(Path(directory))
            with patch.object(
                composition, "open_store", side_effect=OSError("database offline")
            ):
                snapshot = composition.health_snapshot()

        self.assertEqual("DOWN", snapshot["status"])
        self.assertEqual("DOWN", snapshot["readiness"]["status"])
        self.assertEqual("DOWN", snapshot["components"]["database"]["status"])
        self.assertIn("database offline", snapshot["components"]["database"]["error"])


if __name__ == "__main__":
    unittest.main()
