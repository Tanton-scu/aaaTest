from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prievo_agent.application.orchestration.agent_dispatcher import AgentDispatchError
from prievo_agent.application.workflows.generation_workflow import DurableGenerationWorkflow
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.domain.prior import (
    LANDSCAPE_METRICS,
    InstanceSpecificPrior,
    LandscapeProfile,
    SemanticRefinement,
)
from prievo_agent.infrastructure.testing.scripted_fake_llm import ScriptedFakeLLM
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore


def _prior():
    return InstanceSpecificPrior(
        LandscapeProfile(
            "fixture",
            {name: 0.1 for name in LANDSCAPE_METRICS},
            100,
            "fixture",
        ),
        [],
        SemanticRefinement(["source"], "fixture", "fixture"),
        [],
        "fixture-v1",
    )


def _draft():
    return {
        "result_type": "CandidateDraft",
        "code": (
            "def run_tuners(file, budget, seed, maxlives):\n"
            "    value = 3\n"
            "    return value\n"
        ),
        "description": "bounded redrive fixture",
        "operators": ["Fixture"],
        "generation_note": "redriven",
    }


class AgentTaskRedriveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="agent-redrive-")
        root = Path(self.temporary.name)
        self.store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
        self.project = Path(__file__).resolve().parents[1]
        self.task = OptimizationTask("task", "redrive", "minimize", 3, 20)
        self.run = Run("run", self.task.id)
        self.store.add_task(self.task)
        self.store.add_run(self.run)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def _workflow(self, model):
        return DurableGenerationWorkflow(
            self.store, SkillRegistry(self.project / "skills"), model
        )

    def test_transient_once_is_automatically_redriven_to_success(self):
        model = ScriptedFakeLLM(
            {"generation:i1": [ScriptedFakeLLM.transient("once"), _draft()]},
            strict=True,
        )

        draft, _, _ = self._workflow(model).generate(
            self.run, self.task, _prior(), ["prior-ref"], "i1", [], 0, 0
        )

        task = self.store.agent_tasks_for_run(self.run.id)[0]
        events = [item.event_type for item in self.store.events_for_run(self.run.id)]
        self.assertEqual("redriven", draft.generation_note)
        self.assertEqual("COMPLETED", task.status.value)
        self.assertEqual(2, task.attempts)
        self.assertEqual(1, events.count("AGENT_TASK_REDRIVE_SCHEDULED"))
        self.assertEqual(2, len(model.scripted_calls))

    def test_persistent_malformed_exhausts_exact_max_attempts(self):
        model = ScriptedFakeLLM(
            {"generation:i1": [ScriptedFakeLLM.malformed()] * 3},
            strict=True,
        )

        with self.assertRaises(AgentDispatchError):
            self._workflow(model).generate(
                self.run, self.task, _prior(), ["prior-ref"], "i1", [], 0, 0
            )

        task = self.store.agent_tasks_for_run(self.run.id)[0]
        events = [item.event_type for item in self.store.events_for_run(self.run.id)]
        self.assertEqual("FAILED", task.status.value)
        self.assertEqual(3, task.attempts)
        self.assertEqual(2, events.count("AGENT_TASK_REDRIVE_SCHEDULED"))
        self.assertEqual(3, len(model.scripted_calls))
        self.assertFalse(any(
            item.kind == "CANDIDATE_DRAFT"
            for item in self.store.artifacts_for_run(self.run.id)
        ))


if __name__ == "__main__":
    unittest.main()
