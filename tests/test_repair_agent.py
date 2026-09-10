import copy
import unittest

from prievo_agent.agents.nodes.repair import (
    CrossRunRepairContextError,
    FailureEvidence,
    MalformedDiagnosisError,
    MalformedRepairDraftError,
    RepairAgent,
    RepairAttemptLimitError,
    RepairBudgetExhaustedError,
    RepairNotAllowedError,
)
from prievo_agent.domain.models import Candidate
from prievo_agent.knowledge.skills.models import SkillDefinition
from prievo_agent.evaluation.failures import FailureAction, FailureType


ORIGINAL_CODE = """def run_tuners(file, budget, seed, maxlives):
    return {"objective": 10, "seed": seed}
"""

REPAIRED_CODE_1 = """def run_tuners(file, budget, seed, maxlives):
    return {"objective": 9, "seed": seed, "budget": budget}
"""

REPAIRED_CODE_2 = """def run_tuners(file, budget, seed, maxlives):
    return {"objective": 8, "seed": seed, "budget": min(budget, maxlives)}
"""


class ScriptedRepairModel:
    def __init__(self, diagnoses=None, drafts=None, mutate_received_candidate=False):
        self.diagnoses = list(diagnoses or [])
        self.drafts = list(drafts or [])
        self.calls = []
        self.mutate_received_candidate = mutate_received_candidate

    def diagnose_candidate(self, prompt, candidate, failure_evidence):
        self.calls.append(("diagnose", prompt, candidate.id, dict(failure_evidence)))
        if self.mutate_received_candidate:
            candidate.code = "MODEL_TRIED_TO_MUTATE_PARENT"
            candidate.lineage["model_mutation"] = True
        return self.diagnoses.pop(0)

    def repair_candidate(self, prompt, candidate, diagnosis):
        self.calls.append(("repair", prompt, candidate.id, diagnosis.id))
        if self.mutate_received_candidate:
            candidate.description = "MODEL_TRIED_TO_MUTATE_PARENT"
        return self.drafts.pop(0)


def skill():
    return SkillDefinition(
        name="candidate_code_repair",
        version="1",
        purpose="candidate repair",
        instructions=(
            "Apply the classified failure fix only. Preserve run_tuners and return "
            "a new RepairedCandidateDraft."
        ),
        content_digest="repair-skill-sha256",
    )


def diagnosis_skill():
    return SkillDefinition(
        name="candidate_failure_diagnosis",
        version="3",
        purpose="candidate failure diagnosis",
        instructions=(
            "Classify only persisted failure facts. "
            "DIAGNOSIS_SKILL_MARKER_4831"
        ),
        content_digest="diagnosis-skill-sha256",
    )


def candidate(candidate_id="C17", code=ORIGINAL_CODE, lineage=None):
    return Candidate(
        id=candidate_id,
        run_id="run-7",
        code=code,
        description="failed candidate",
        operators=["REVISE"],
        lineage=copy.deepcopy(lineage or {"generation": 2, "parents": ["C9"]}),
    )


def diagnosis(failure_type, repairable=True):
    return {
        "repairable": repairable,
        "failure_type": failure_type,
        "root_cause": "The candidate implementation violates the observed contract.",
        "suggested_fix": "Apply one bounded code-level fix for the observed evidence.",
        "confidence": 0.9,
    }


def draft(code=REPAIRED_CODE_1):
    return {
        "code": code,
        "description": "minimal evidence-backed repair",
        "operators": ["REVISE"],
    }


class RepairAgentTest(unittest.TestCase):
    def test_product_path_uses_distinct_diagnose_and_repair_skills(self):
        model = ScriptedRepairModel(
            [diagnosis("RUNTIME_ERROR")], [draft()]
        )
        result = RepairAgent(model).execute(
            candidate(),
            FailureEvidence(error_code="RUNTIME_ERROR"),
            skill(),
            diagnosis_skill=diagnosis_skill(),
        )

        diagnosis_prompt = model.calls[0][1]
        repair_prompt = model.calls[1][1]
        self.assertIn("candidate_failure_diagnosis", diagnosis_prompt)
        self.assertIn("DIAGNOSIS_SKILL_MARKER_4831", diagnosis_prompt)
        self.assertIn("diagnosis-skill-sha256", diagnosis_prompt)
        self.assertIn(
            "skill:candidate_failure_diagnosis@3#diagnosis-skill-sha256",
            diagnosis_prompt,
        )
        self.assertNotIn("Apply the classified failure fix only", diagnosis_prompt)
        self.assertIn("candidate_code_repair", repair_prompt)
        self.assertIn("Apply the classified failure fix only", repair_prompt)
        self.assertIn("repair-skill-sha256", repair_prompt)
        self.assertIn(
            "skill:candidate_code_repair@1#repair-skill-sha256",
            repair_prompt,
        )

        self.assertEqual(
            ("skill:candidate_failure_diagnosis@3#diagnosis-skill-sha256",),
            result.diagnosis.skill_refs,
        )
        self.assertEqual(
            {
                "role": "diagnose",
                "name": "candidate_failure_diagnosis",
                "version": "3",
                "digest": "diagnosis-skill-sha256",
                "ref": (
                    "skill:candidate_failure_diagnosis@3#"
                    "diagnosis-skill-sha256"
                ),
            },
            dict(result.diagnosis.skill_provenance[0]),
        )
        self.assertEqual(
            ["candidate_failure_diagnosis", "candidate_code_repair"],
            [item["name"] for item in result.draft.skill_provenance],
        )
        self.assertEqual(
            ["diagnose", "repair"],
            [item["role"] for item in result.draft.skill_provenance],
        )
        self.assertEqual(
            list(result.draft.skill_refs),
            result.draft.lineage["skill_refs"],
        )

    def test_candidate_failure_matrix_runs_diagnose_then_repair(self):
        matrix = [
            ("SYNTAX_ERROR", FailureType.SYNTAX_ERROR),
            ("RUNTIME_ERROR", FailureType.RUNTIME_ERROR),
            ("INTERFACE_ERROR", FailureType.INTERFACE_ERROR),
            ("ALGORITHM_TIMEOUT", FailureType.ALGORITHM_TIMEOUT),
            ("ALGORITHM_OOM", FailureType.ALGORITHM_OOM),
        ]
        for error_code, expected_type in matrix:
            with self.subTest(error_code=error_code):
                model = ScriptedRepairModel(
                    [diagnosis(expected_type.value)], [draft()]
                )
                result = RepairAgent(model).execute(
                    candidate(),
                    FailureEvidence(
                        error_code=error_code,
                        run_id="run-7",
                        candidate_id="C17",
                        artifact_refs=("failure-artifact-17",),
                    ),
                    skill(),
                    repair_history=[
                        {"run_id": "run-7", "attempt": value}
                        for value in range(5)
                    ],
                    relevant_failures=[
                        {"run_id": "run-7", "failure": value}
                        for value in range(4)
                    ],
                    context_refs=("candidate-code-artifact-17",),
                )

                self.assertEqual(expected_type, result.decision.failure_type)
                self.assertEqual(FailureAction.REPAIR, result.decision.action)
                self.assertEqual(["diagnose", "repair"], [item[0] for item in model.calls])
                self.assertEqual(expected_type, result.diagnosis.failure_type)
                self.assertEqual("C17-R1", result.draft.id)
                self.assertEqual("C17", result.draft.repair_parent_id)
                self.assertEqual(1, result.draft.repair_attempt)
                self.assertEqual("REPAIR", result.draft.creation_type)
                self.assertNotIn("attempt\": 0", model.calls[0][1])
                self.assertIn("attempt\": 4", model.calls[0][1])

    def test_infrastructure_failures_never_call_diagnose_or_repair_model(self):
        for error_code in ("NETWORK", "WORKER_CRASH", "TRANSIENT_INFRA"):
            with self.subTest(error_code=error_code):
                model = ScriptedRepairModel()
                with self.assertRaises(RepairNotAllowedError):
                    RepairAgent(model).execute(
                        candidate(),
                        FailureEvidence(error_code=error_code),
                        skill(),
                    )
                self.assertEqual([], model.calls)

    def test_nonrepairable_diagnosis_stops_before_repair_phase(self):
        model = ScriptedRepairModel([diagnosis("LOGIC_FAILURE", False)])
        result = RepairAgent(model).execute(
            candidate(), FailureEvidence(error_code="LOGIC_FAILURE"), skill()
        )

        self.assertFalse(result.diagnosis.repairable)
        self.assertIsNone(result.draft)
        self.assertIn("不可安全修复", result.skipped_reason)
        self.assertEqual(["diagnose"], [item[0] for item in model.calls])

    def test_repair_creates_immutable_parent_child_version_chain_and_refs(self):
        original = candidate()
        original_snapshot = copy.deepcopy(original)
        first_model = ScriptedRepairModel(
            [diagnosis("RUNTIME_ERROR")],
            [draft(REPAIRED_CODE_1)],
            mutate_received_candidate=True,
        )
        first = RepairAgent(first_model).execute(
            original,
            FailureEvidence(
                error_code="RUNTIME_ERROR",
                artifact_refs=("failure-ref-1",),
            ),
            skill(),
            context_refs=("context-ref-1",),
        ).draft

        self.assertEqual(original_snapshot, original)
        self.assertEqual("C17-R1", first.id)
        self.assertEqual("C17", first.lineage["repair_root_id"])
        self.assertEqual("C17", first.lineage["repair_parent_id"])
        self.assertEqual(
            ("context-ref-1", "failure-ref-1"), first.context_refs
        )
        self.assertTrue(first.skill_refs[0].startswith("skill:candidate_code_repair@1#"))
        self.assertEqual(1, len(first.skill_refs))
        self.assertEqual(first.skill_refs, tuple(first.lineage["skill_refs"]))
        self.assertEqual(first.context_refs, tuple(first.lineage["context_refs"]))

        repaired_parent = first.to_candidate()
        second_model = ScriptedRepairModel(
            [diagnosis("ALGORITHM_TIMEOUT")], [draft(REPAIRED_CODE_2)]
        )
        second = RepairAgent(second_model).execute(
            repaired_parent,
            FailureEvidence(error_code="ALGORITHM_TIMEOUT"),
            skill(),
        ).draft

        self.assertEqual("C17-R2", second.id)
        self.assertEqual("C17-R1", second.repair_parent_id)
        self.assertEqual(2, second.repair_attempt)
        self.assertEqual(2, len(second.lineage["repair_lineage"]))
        self.assertEqual(
            ["C17-R1", "C17-R2"],
            [item["candidate_id"] for item in second.lineage["repair_lineage"]],
        )
        self.assertEqual("C17-R1", second.lineage["repair_parent_id"])

    def test_attempt_limit_preserves_diagnosis_and_skips_repair_call(self):
        parent = candidate(
            "C17-R1",
            REPAIRED_CODE_1,
            {
                "repair_root_id": "C17",
                "repair_attempt": 1,
                "repair_lineage": [
                    {
                        "candidate_id": "C17-R1",
                        "repair_parent_id": "C17",
                        "repair_attempt": 1,
                    }
                ],
            },
        )
        model = ScriptedRepairModel([diagnosis("INTERFACE_ERROR")])

        with self.assertRaises(RepairAttemptLimitError) as raised:
            RepairAgent(model).execute(
                parent,
                FailureEvidence(error_code="INTERFACE_ERROR"),
                skill(),
                max_attempts=1,
            )

        self.assertEqual(parent.id, raised.exception.diagnosis.candidate_id)
        self.assertEqual(["diagnose"], [item[0] for item in model.calls])

    def test_zero_repair_budget_preserves_diagnosis_and_skips_repair_call(self):
        model = ScriptedRepairModel([diagnosis("SYNTAX_ERROR")])

        with self.assertRaises(RepairBudgetExhaustedError) as raised:
            RepairAgent(model).execute(
                candidate(),
                FailureEvidence(error_code="SYNTAX_ERROR"),
                skill(),
                repair_budget=0,
            )

        self.assertTrue(raised.exception.diagnosis.repairable)
        self.assertEqual(["diagnose"], [item[0] for item in model.calls])

    def test_diagnose_only_does_not_call_repair_model(self):
        model = ScriptedRepairModel([diagnosis("RUNTIME_ERROR")])
        artifact = RepairAgent(model).diagnose(
            candidate(), FailureEvidence(error_code="RUNTIME_ERROR"), skill()
        )

        self.assertEqual(FailureType.RUNTIME_ERROR, artifact.failure_type)
        self.assertEqual(["diagnose"], [item[0] for item in model.calls])

    def test_malformed_diagnosis_is_rejected(self):
        malformed = diagnosis("RUNTIME_ERROR")
        malformed["repairable"] = "yes"
        model = ScriptedRepairModel([malformed])

        with self.assertRaises(MalformedDiagnosisError):
            RepairAgent(model).execute(
                candidate(), FailureEvidence(error_code="RUNTIME_ERROR"), skill()
            )
        self.assertEqual(["diagnose"], [item[0] for item in model.calls])

    def test_diagnosis_cannot_override_failure_classifier(self):
        model = ScriptedRepairModel([diagnosis("SYNTAX_ERROR")])
        with self.assertRaises(MalformedDiagnosisError):
            RepairAgent(model).execute(
                candidate(), FailureEvidence(error_code="ALGORITHM_OOM"), skill()
            )

    def test_malformed_or_identity_overwriting_draft_is_rejected(self):
        bad_drafts = [
            {
                "code": "def broken(:\n",
                "description": "broken",
                "operators": ["REVISE"],
            },
            {**draft(), "candidate_id": "C17"},
            draft(ORIGINAL_CODE),
        ]
        for bad_draft in bad_drafts:
            with self.subTest(draft=bad_draft):
                model = ScriptedRepairModel(
                    [diagnosis("SYNTAX_ERROR")], [bad_draft]
                )
                with self.assertRaises(MalformedRepairDraftError):
                    RepairAgent(model).execute(
                        candidate(),
                        FailureEvidence(error_code="SYNTAX_ERROR"),
                        skill(),
                    )

    def test_context_rejects_cross_run_history(self):
        model = ScriptedRepairModel()
        with self.assertRaises(CrossRunRepairContextError):
            RepairAgent(model).execute(
                candidate(),
                FailureEvidence(error_code="RUNTIME_ERROR"),
                skill(),
                relevant_failures=[{"run_id": "run-other", "failure": "x"}],
            )
        self.assertEqual([], model.calls)

    def test_wrong_skill_is_rejected_before_model_call(self):
        wrong = SkillDefinition("revise", "1", "", "instruction", "digest")
        model = ScriptedRepairModel()
        with self.assertRaisesRegex(Exception, "candidate_code_repair"):
            RepairAgent(model).execute(
                candidate(), FailureEvidence(error_code="RUNTIME_ERROR"), wrong
            )
        self.assertEqual([], model.calls)


if __name__ == "__main__":
    unittest.main()
