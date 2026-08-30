import tempfile
import threading
import time
import unittest
from pathlib import Path

from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.domain.models import OptimizationTask, Run, RunStatus
from prievo_agent.infrastructure.testing.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.testing.fake_llm import FakeLLM
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.lifecycle import RunLifecycleService
from prievo_agent.runtime.persistent_runtime import PersistentEvolutionRuntime
from prievo_agent.runtime.state_machine import RunStateMachine


class SlowEvaluator(FakeEvaluator):
    def evaluate(self, candidate, task):
        time.sleep(0.04)
        return super().evaluate(candidate, task)


class RuntimeCancellationRaceTest(unittest.TestCase):
    def test_persisted_cancel_is_not_overwritten_by_stale_running_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            artifacts = root / "artifacts"
            setup = SQLiteRuntimeStore(database, artifacts)
            task = OptimizationTask("task-cancel-race", "取消竞态", "minimize", 3, 60)
            run = Run("run-cancel-race", task.id)
            setup.add_task(task)
            setup.add_run(run)
            setup.close()

            def execute():
                worker_store = SQLiteRuntimeStore(database, artifacts)
                try:
                    PersistentEvolutionRuntime(
                        worker_store,
                        SlowEvaluator(),
                        PriEvoEvolutionCore(FakeLLM(), population_size=3),
                        total_generations=2,
                    ).execute(run.id)
                finally:
                    worker_store.close()

            thread = threading.Thread(target=execute)
            thread.start()
            control = SQLiteRuntimeStore(database, artifacts)
            try:
                deadline = time.monotonic() + 2
                while control.get_run(run.id).status != RunStatus.RUNNING:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.005)
                current = control.get_run(run.id)
                RunLifecycleService(control, RunStateMachine()).cancel(current)
            finally:
                control.close()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())

            verify = SQLiteRuntimeStore(database, artifacts)
            try:
                self.assertEqual(RunStatus.CANCELLED, verify.get_run(run.id).status)
                event_types = [item.event_type for item in verify.events_for_run(run.id)]
                self.assertIn("RUN_CANCELLED", event_types)
                self.assertNotIn("RUN_COMPLETED", event_types)
            finally:
                verify.close()


if __name__ == "__main__":
    unittest.main()
