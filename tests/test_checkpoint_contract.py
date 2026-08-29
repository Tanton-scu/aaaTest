from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.infrastructure.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.persistent_runtime import PersistentEvolutionRuntime


class CheckpointRuntimeContractTest(unittest.TestCase):
    def _runtime(self, store, *, faithful=False, mode="inline"):
        return PersistentEvolutionRuntime(
            store,
            FakeEvaluator(),
            PriEvoEvolutionCore(FakeLLM(), population_size=1),
            total_generations=2,
            research_faithful_mode=faithful,
            evaluation_execution_mode=mode,
        )

    def test_faithful_mode_cannot_change_across_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRuntimeStore(
                Path(directory) / "state.sqlite3", Path(directory) / "artifacts"
            )
            try:
                runtime = self._runtime(store, faithful=True)
                with self.assertRaisesRegex(RuntimeError, "faithful mode"):
                    runtime._validate_checkpoint_runtime_contract(
                        {
                            "strategy": {
                                "total_generations": 2,
                                "research_faithful_mode": False,
                                "evaluation_execution_mode": "inline",
                            }
                        }
                    )
            finally:
                store.close()

    def test_evaluation_execution_mode_cannot_change_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRuntimeStore(
                Path(directory) / "state.sqlite3", Path(directory) / "artifacts"
            )
            try:
                runtime = self._runtime(store, mode="external")
                with self.assertRaisesRegex(RuntimeError, "evaluation execution mode"):
                    runtime._validate_checkpoint_runtime_contract(
                        {
                            "strategy": {
                                "total_generations": 2,
                                "research_faithful_mode": False,
                                "evaluation_execution_mode": "inline",
                            }
                        }
                    )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
