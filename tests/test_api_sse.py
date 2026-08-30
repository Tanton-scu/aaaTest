import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from prievo_agent.api import create_app


class ApiSseAcceptanceTest(unittest.TestCase):
    def test_create_disconnect_reconnect_and_observe_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 容器/CI 文件系统抖动可能超过 90ms；用更宽的启动隔离窗验证
            # POST 没有等待后台 evolution，而不是对机器性能做脆弱断言。
            first_app = create_app(root, start_delay_seconds=0.5)
            with TestClient(first_app) as client:
                datasets = client.get("/api/datasets").json()
                self.assertIn("xgboost-Covtype", {item["id"] for item in datasets})
                self.assertEqual(422, client.post("/api/runs", json={
                    "dataset_id": "../arbitrary.csv"
                }).status_code)
                started = time.monotonic()
                response = client.post(
                    "/api/runs",
                    json={"dataset_id": "xgboost-Covtype", "generations": 1,
                          "population_size": 2, "candidate_budget": 3,
                          "random_seed": 7},
                )
                elapsed = time.monotonic() - started
                self.assertEqual(202, response.status_code)
                self.assertLess(elapsed, 0.25, "POST 不应等待演化运行结束")
                run_id = response.json()["run_id"]

            # 原客户端已经断开；新应用实例只依赖持久化目录恢复历史。
            second_app = create_app(root, auto_start=False)
            with TestClient(second_app) as client:
                detail = client.get("/api/runs/{}".format(run_id))
                self.assertEqual(200, detail.status_code)
                payload = detail.json()
                self.assertEqual("COMPLETED", payload["status"])
                self.assertEqual("xgboost-Covtype", payload["dataset_id"])
                # 10 次演化评价 + 2 个 Final Optimization 独立 seed clone；
                # 真实 evaluator 还会保留失败 prior/Repair，Candidate 事实数可更大。
                self.assertGreaterEqual(len(payload["candidates"]), 12)
                self.assertEqual(
                    12,
                    sum(item["status"] == "EVALUATED" for item in payload["candidates"]),
                )
                self.assertEqual(42, payload["budget"]["total"])
                self.assertEqual(42, payload["budget"]["consumed"])
                self.assertTrue(any(item["lineage"] for item in payload["candidates"]))

                events = client.get("/api/runs/{}/events".format(run_id)).json()["items"]
                self.assertIn("RUN_COMPLETED", [item["event_type"] for item in events])
                artifacts = client.get("/api/runs/{}/artifacts".format(run_id)).json()["items"]
                self.assertIn("FINAL_HEURISTIC", [item["kind"] for item in artifacts])
                self.assertIn(
                    "FINAL_OPTIMIZATION_REPORT",
                    [item["kind"] for item in artifacts],
                )

                candidates = client.get(
                    "/api/runs/{}/candidates".format(run_id)
                ).json()
                self.assertEqual(run_id, candidates["run_id"])
                self.assertEqual(len(payload["candidates"]), len(candidates["items"]))

                trace = client.get("/api/runs/{}/trace".format(run_id))
                legacy_trace = client.get(
                    "/api/runs/{}/agent-trace".format(run_id)
                )
                self.assertEqual(200, trace.status_code)
                self.assertEqual(trace.json(), legacy_trace.json())
                self.assertIn("tasks", trace.json())

                metrics = client.get("/api/runs/{}/metrics".format(run_id)).json()
                self.assertEqual(run_id, metrics["run_id"])
                self.assertEqual(payload["budget"]["consumed"], metrics["budget_consumed"])
                self.assertIsNotNone(metrics["best_fitness"])
                self.assertEqual(1, len(metrics["generation_durations"]))

                stream = client.get("/api/runs/{}/events/stream".format(run_id))
                self.assertEqual(200, stream.status_code)
                self.assertIn("event: RUN_COMPLETED", stream.text)
                self.assertIn("运行已完成", stream.text)
                self.assertIn("目标 Dataset", client.get("/").text)

    def test_cancel_and_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(Path(directory), auto_start=False)
            with TestClient(app) as client:
                created = client.post("/api/runs", json={"dataset_id": "sqlite"})
                run_id = created.json()["run_id"]
                cancelled = client.post("/api/runs/{}/cancel".format(run_id))
                self.assertEqual("CANCELLED", cancelled.json()["status"])
                self.assertEqual(409, client.post("/api/runs/{}/resume".format(run_id)).status_code)
                self.assertEqual(404, client.get("/api/runs/not-found").status_code)

    def test_pause_endpoint_records_cooperative_request_without_lying_about_paused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = create_app(root, auto_start=False)
            with TestClient(app) as client:
                created = client.post(
                    "/api/runs", json={"dataset_id": "sqlite"}
                )
                self.assertEqual(202, created.status_code)
                run_id = created.json()["run_id"]
                # auto_start=False 的 PENDING 尚无 Runtime/safe point，不能伪装暂停。
                pending_pause = client.post(
                    "/api/runs/{}/pause".format(run_id)
                )
                self.assertEqual(409, pending_pause.status_code)

            # 测试只在 durable store 将 Run 置 RUNNING；pause endpoint 只能落请求。
            from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore
            from prievo_agent.runtime.lifecycle import RunLifecycleService
            from prievo_agent.runtime.state_machine import RunStateMachine

            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                run = store.get_run(run_id)
                RunLifecycleService(store, RunStateMachine()).start(run)
            finally:
                store.close()

            app = create_app(root, auto_start=False)
            with TestClient(app) as client:
                requested = client.post(
                    "/api/runs/{}/pause".format(run_id)
                )
                self.assertEqual(202, requested.status_code)
                payload = requested.json()
                self.assertEqual("RUNNING", payload["status"])
                self.assertTrue(payload["control"]["pause_requested"])
                event_types = [
                    item["event_type"]
                    for item in client.get(
                        "/api/runs/{}/events".format(run_id)
                    ).json()["items"]
                ]
                self.assertIn("RUN_PAUSE_REQUESTED", event_types)
                self.assertNotIn("RUN_PAUSED", event_types)


if __name__ == "__main__":
    unittest.main()
