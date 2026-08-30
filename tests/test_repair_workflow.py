from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.application.workflows.repair_workflow import DurableRepairWorkflow
from prievo_agent.domain.models import (
    Candidate,
    EvaluationJob,
    EvaluationJobStatus,
    OptimizationTask,
    Run,
)
from prievo_agent.infrastructure.testing.fake_llm import FakeLLM
from prievo_agent.infrastructure.agent_memory import InMemoryAgentWorkingMemory
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore
from prievo_agent.security.heuristic_worker import validate_heuristic_source


class _NonRepairableModel(FakeLLM):
    def repair_candidate(self, prompt, candidate, diagnosis):
        raise AssertionError("repairable=false 时不得进入 Repair phase")

    def diagnose_candidate(self, prompt, candidate, failure_evidence):
        result = super().diagnose_candidate(prompt, candidate, failure_evidence)
        result.update({
            "repairable": False,
            "root_cause": "Persisted evidence is insufficient for a safe fix.",
            "suggested_fix": "Do not create a draft; retain the diagnosis.",
        })
        return result


class DurableRepairWorkflowTest(unittest.TestCase):
    def test_failure_creates_new_version_without_overwriting_parent(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask(
                    "task", "fixture", "minimize", 2, 20,
                    dataset_id="fixture", population_size=1,
                )
                run = Run("run", task.id, dataset_id="fixture")
                store.add_task(task)
                store.add_run(run)
                original_code = (
                    "def run_tuners(file, budget, seed, maxlives):\n"
                    "    raise ValueError('broken')\n"
                )
                parent = Candidate(
                    "candidate", run.id, original_code, "broken",
                    ["Sampling"], {"repair_attempt": 0},
                )
                store.add_candidate(parent)
                failed = EvaluationJob(
                    "job", run.id, task.id, parent.id, 101, 2, "key",
                    status=EvaluationJobStatus.DEAD,
                    attempts=1,
                    error_code="RUNTIME_ERROR",
                    error_message="boom",
                )
                model = FakeLLM()
                cache = InMemoryAgentWorkingMemory()
                workflow = DurableRepairWorkflow(
                    store,
                    SkillRegistry(project / "skills"),
                    model,
                    working_memory=cache,
                )

                repaired, failure_ref, draft_ref = workflow.repair(
                    run, task, parent, failed
                )

                self.assertEqual("candidate-R1", repaired.id)
                self.assertEqual(parent.id, repaired.lineage["repair_parent_id"])
                self.assertEqual(1, repaired.lineage["repair_attempt"])
                self.assertNotEqual(parent.code, repaired.code)
                validate_heuristic_source(repaired.code)
                persisted_parent = store.candidate_by_id(parent.id)
                self.assertEqual(original_code, persisted_parent.code)
                self.assertEqual(repaired.code, store.candidate_by_id(repaired.id).code)
                self.assertTrue(failure_ref)
                self.assertTrue(draft_ref)
                tasks = [
                    item for item in store.agent_tasks_for_run(run.id)
                    if item.task_type == "CANDIDATE_REPAIR"
                ]
                self.assertEqual(1, len(tasks))
                self.assertEqual("COMPLETED", tasks[0].status.value)
                self.assertEqual(
                    ["diagnose", "repair"],
                    [item["phase"] for item in model.repair_agent_calls],
                )
                diagnosis_prompt = model.repair_agent_calls[0]["prompt"]
                repair_prompt = model.repair_agent_calls[1]["prompt"]
                self.assertIn("candidate_failure_diagnosis", diagnosis_prompt)
                self.assertIn("candidate_code_repair", repair_prompt)
                artifact_payloads = {
                    item.kind: json.loads(store.artifact_content(item.id))
                    for item in store.artifacts_for_run(run.id)
                    if item.kind in {
                        "REPAIR_DECISION", "REPAIRED_CANDIDATE_DRAFT"
                    }
                }
                decision_provenance = artifact_payloads[
                    "REPAIR_DECISION"
                ]["diagnosis"]["skill_provenance"]
                draft_provenance = artifact_payloads[
                    "REPAIRED_CANDIDATE_DRAFT"
                ]["skill_provenance"]
                self.assertEqual(
                    ["candidate_failure_diagnosis"],
                    [item["name"] for item in decision_provenance],
                )
                self.assertEqual(
                    ["candidate_failure_diagnosis", "candidate_code_repair"],
                    [item["name"] for item in draft_provenance],
                )
                calls = list(store.tool_calls_for_run(run.id))
                self.assertEqual(1, len(calls))
                self.assertEqual("candidate_inspection", calls[0].tool_name)
                self.assertEqual("COMPLETED", calls[0].status)
                self.assertEqual("RepairAgent", calls[0].request["caller"])
                self.assertEqual(
                    parent.id,
                    calls[0].request["input_metadata"]["candidate_id"],
                )
                self.assertIn("started_at", calls[0].request)
                self.assertIn("finished_at", calls[0].response)
                memories = store.agent_memories_for_run(
                    run.id, "repair", limit=10
                )
                self.assertEqual(1, len(memories))
                self.assertEqual("REPAIR_DECISION", memories[0].memory_type)
                self.assertEqual(draft_ref, memories[0].evidence_artifact_id)
                cached = cache.load_recent(run.id, scope="repair")
                self.assertEqual(memories[0].id, cached[0]["metadata"]["agent_memory_id"])

                second_failure = EvaluationJob(
                    "job-2", run.id, task.id, repaired.id, 102, 2, "key-2",
                    status=EvaluationJobStatus.DEAD,
                    attempts=1,
                    error_code="RUNTIME_ERROR",
                    error_message="still broken",
                )
                second, _, _ = workflow.repair(
                    run, task, repaired, second_failure
                )
                self.assertEqual("candidate-R2", second.id)
                second_diagnosis_prompt = model.repair_agent_calls[-2]["prompt"]
                self.assertIn(memories[0].id, second_diagnosis_prompt)
                self.assertIn("REPAIRED_DRAFT_CREATED", second_diagnosis_prompt)
                self.assertEqual(
                    2, len(store.agent_memories_for_run(run.id, "repair", 10))
                )
            finally:
                store.close()

    def test_nonrepairable_is_structured_completed_result_without_draft(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask(
                    "task-negative", "fixture", "minimize", 2, 20,
                    dataset_id="fixture", population_size=1,
                )
                run = Run("run-negative", task.id, dataset_id="fixture")
                store.add_task(task)
                store.add_run(run)
                parent = Candidate(
                    "candidate-negative",
                    run.id,
                    (
                        "def run_tuners(file, budget, seed, maxlives):\n"
                        "    raise ValueError('broken')\n"
                    ),
                    "not safely repairable",
                    ["Sampling"],
                    {"repair_attempt": 0},
                )
                store.add_candidate(parent)
                failed = EvaluationJob(
                    "job-negative", run.id, task.id, parent.id, 101, 2,
                    "key-negative", status=EvaluationJobStatus.DEAD,
                    attempts=1, error_code="RUNTIME_ERROR",
                    error_message="insufficient evidence",
                )
                model = _NonRepairableModel()
                result = DurableRepairWorkflow(
                    store,
                    SkillRegistry(project / "skills"),
                    model,
                ).repair(run, task, parent, failed)

                self.assertFalse(result.repairable)
                self.assertIsNone(result.repaired_candidate)
                self.assertIsNone(result.repaired_candidate_draft_artifact_id)
                self.assertTrue(result.failure_artifact_id)
                self.assertTrue(result.repair_decision_artifact_id)
                self.assertFalse(result.diagnosis["repairable"])
                self.assertIn("不可安全修复", result.skipped_reason)
                self.assertEqual(
                    (None, result.failure_artifact_id, None), tuple(result)
                )
                self.assertEqual(
                    ["diagnose"],
                    [item["phase"] for item in model.repair_agent_calls],
                )
                self.assertEqual(1, len(store.candidates_for_run(run.id)))
                kinds = [item.kind for item in store.artifacts_for_run(run.id)]
                self.assertIn("REPAIR_DECISION", kinds)
                self.assertNotIn("REPAIRED_CANDIDATE_DRAFT", kinds)
                tasks = [
                    item for item in store.agent_tasks_for_run(run.id)
                    if item.task_type == "CANDIDATE_REPAIR"
                ]
                self.assertEqual(1, len(tasks))
                self.assertEqual("COMPLETED", tasks[0].status.value)
                memories = store.agent_memories_for_run(
                    run.id, "repair", limit=10
                )
                self.assertEqual(1, len(memories))
                memory = json.loads(memories[0].content)
                self.assertEqual("NOT_REPAIRABLE", memory["outcome"])
                self.assertFalse(memory["repairable"])
                self.assertEqual(
                    result.repair_decision_artifact_id,
                    memories[0].evidence_artifact_id,
                )
                event_types = [
                    item.event_type for item in store.events_for_run(run.id)
                ]
                self.assertIn("REPAIR_SKIPPED_NOT_REPAIRABLE", event_types)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
