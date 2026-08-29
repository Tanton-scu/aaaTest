import tempfile
import threading
import unittest
from pathlib import Path

from prievo_agent.application.run_facade import RunApplicationFacade
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


class PeriodicCoordinatorSweepTest(unittest.TestCase):
    def test_commit_without_reconcile_is_recovered_exactly_once(self):
        with tempfile.TemporaryDirectory(prefix="prievo-periodic-sweep-") as directory:
            root = Path(directory)
            db_path = root / "state.sqlite3"
            artifacts = root / "artifacts"

            def store_factory():
                return SQLiteRuntimeStore(db_path, artifacts)

            store = store_factory()
            task = OptimizationTask("task", "fixture", "minimize", 1, 5)
            run = Run("run", task.id)
            store.add_task(task)
            store.add_run(run)
            source = store.put_artifact(
                run.id, "GENERATION_REQUEST", b"{}", "application/json"
            )
            store.close()

            scheduled = threading.Event()
            facade = RunApplicationFacade(
                store_factory,
                lambda _run_id: scheduled.set(),
                auto_start=False,
            )
            try:
                first = facade.reconcile_active_agent_tasks()
                second = facade.reconcile_active_agent_tasks()
                check = store_factory()
                try:
                    tasks = list(check.agent_tasks_for_run(run.id))
                finally:
                    check.close()
            finally:
                facade.shutdown()

            self.assertEqual(1, first["created_task_count"])
            self.assertEqual(0, second["created_task_count"])
            self.assertEqual([first["created_task_ids"][0]], [tasks[0].id])
            self.assertEqual([source.id], list(tasks[0].input_artifact_refs))
            self.assertTrue(scheduled.is_set())


if __name__ == "__main__":
    unittest.main()
