import unittest

from prievo_agent.agents.context import AgentContextBuilder
from prievo_agent.agents.context_policies import ContextPolicyFramework
from prievo_agent.agents.final_selection import (
    DeterministicFinalSelector,
    FinalCandidate,
    FinalSelectionAgent,
    FinalSelectionSkill,
    NoQualifiedFinalCandidateError,
    REFERENCE_QUALIFICATION_MODE,
)


def code(lines=50):
    return "\n".join("value_{0} = {0}".format(index) for index in range(lines))


def candidate(
    candidate_id,
    objective,
    lines=50,
    trajectory_length=20,
    candidate_budget=20,
    used_budget=None,
):
    return FinalCandidate(
        id=candidate_id,
        code=code(lines),
        description="{} robust tuner".format(candidate_id),
        operators=("explore", "refine"),
        objective=objective,
        trajectory=tuple(float(index) for index in range(trajectory_length)),
        candidate_budget=candidate_budget,
        used_budget=used_budget,
    )


def skill():
    return FinalSelectionSkill(
        name="final_heuristic_audit",
        digest="f" * 64,
        ref="skill:final_heuristic_audit:v1",
        instructions=(
            "Compare long-budget convergence, implementation integrity, and "
            "trajectory stability. Select only an allowed candidate ID."
        ),
    )


class RecordingModel:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def select_final(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def agent(model):
    return FinalSelectionAgent(
        model,
        context_framework=ContextPolicyFramework(
            AgentContextBuilder(max_chars=40000)
        ),
    )


class FinalSelectionAgentTests(unittest.TestCase):
    def test_unique_exact_best_is_selected_without_llm(self):
        model = RecordingModel(AssertionError("unique best 不应调用模型"))

        decision = agent(model).select([
            candidate("c2", 0.20),
            candidate("c1", 0.10),
            candidate("c3", 0.30),
        ])

        self.assertEqual("c1", decision.selected_candidate_id)
        self.assertFalse(decision.model_called)
        self.assertEqual([], model.prompts)
        self.assertEqual("", decision.prompt)
        self.assertEqual("", decision.skill_ref)
        self.assertFalse(decision.fallback_used)

    def test_exact_tie_calls_llm_once_and_preserves_refs(self):
        model = RecordingModel({
            "selected_candidate_id": "c2",
            "reason": "c2 has the stronger long-budget exploration mechanism.",
            "structural_operator_comparison": {
                "structural_comparison": "c2 has the more bounded control flow.",
                "operator_comparison": "c2 combines exploration and refinement.",
            },
        })

        decision = agent(model).select([
            candidate("c2", 0.10),
            candidate("c1", 0.10),
            candidate("c3", 0.20),
        ], skill=skill())

        self.assertEqual("c2", decision.selected_candidate_id)
        self.assertTrue(decision.model_called)
        self.assertEqual(1, len(model.prompts))
        self.assertEqual(("c1", "c2"), decision.tied_candidate_ids)
        self.assertEqual(("c1", "c2"), decision.model_candidate_ids)
        self.assertTrue(decision.prompt_ref.startswith("prompt:"))
        self.assertTrue(decision.context_ref.startswith("context:"))
        self.assertEqual(skill().effective_ref, decision.skill_ref)
        self.assertIn(skill().effective_ref, decision.prompt)
        self.assertIn(decision.context_ref, decision.prompt)
        self.assertFalse(decision.fallback_used)
        self.assertIn(
            "bounded control flow",
            decision.structural_operator_comparison["structural_comparison"],
        )
        serialized = decision.to_dict()
        self.assertEqual(["c1", "c2"], serialized["tied_candidate_ids"])
        self.assertEqual(
            decision.context_metadata["policy_name"],
            serialized["context_metadata"]["policy_name"],
        )

    def test_tie_is_exact_float_equality_not_epsilon(self):
        model = RecordingModel(AssertionError("near-equal objective 不应触发 LLM"))

        decision = agent(model).select([
            candidate("c1", 0.1),
            candidate("c2", 0.100000000001),
        ], skill=skill())

        self.assertEqual("c1", decision.selected_candidate_id)
        self.assertEqual(("c1",), decision.tied_candidate_ids)
        self.assertFalse(decision.model_called)
        self.assertEqual([], model.prompts)

    def test_qualification_filters_incomplete_trajectory_and_short_code(self):
        model = RecordingModel(AssertionError("过滤后唯一 best 不应调用 LLM"))
        valid_by_used_budget = candidate(
            "valid", 0.30, trajectory_length=3, used_budget=20
        )
        decision = agent(model).select([
            candidate("short", 0.01, lines=49),
            candidate("incomplete", 0.02, trajectory_length=19, used_budget=19),
            valid_by_used_budget,
        ])

        self.assertEqual("valid", decision.selected_candidate_id)
        self.assertEqual(("valid",), decision.qualified_candidate_ids)
        self.assertEqual({"incomplete", "short"}, set(decision.rejected_reasons))

        with self.assertRaises(NoQualifiedFinalCandidateError) as captured:
            agent(model).select([
                candidate("only-short", 0.1, lines=4),
                candidate("only-incomplete", 0.2, trajectory_length=4),
            ])
        self.assertEqual(
            {"only-incomplete", "only-short"},
            set(captured.exception.rejected_reasons),
        )

        controlled = agent(model).select([
            candidate("z-short", 0.1, lines=4),
            candidate("a-short", 0.1, lines=4),
        ], empty_policy="stable_fallback")
        self.assertEqual("a-short", controlled.selected_candidate_id)
        self.assertTrue(controlled.fallback_used)
        self.assertEqual("NO_QUALIFIED_CANDIDATE", controlled.fallback_reason)
        self.assertFalse(controlled.model_called)

    def test_reference_qualification_requires_exactly_twenty_trajectory_points(self):
        model = RecordingModel(AssertionError("reference unique best 不应调用模型"))
        reference_agent = FinalSelectionAgent(
            model,
            selector=DeterministicFinalSelector(REFERENCE_QUALIFICATION_MODE),
            context_framework=ContextPolicyFramework(
                AgentContextBuilder(max_chars=40000)
            ),
        )

        with self.assertRaises(NoQualifiedFinalCandidateError) as captured:
            reference_agent.select([
                candidate(
                    "used-budget-only",
                    0.1,
                    trajectory_length=3,
                    candidate_budget=3,
                    used_budget=3,
                )
            ], empty_policy="stable_fallback")
        self.assertIn(
            "qualification_mode=reference",
            captured.exception.rejected_reasons["used-budget-only"][0],
        )

        decision = reference_agent.select([
            candidate(
                "fixed-reference-budget",
                0.1,
                trajectory_length=20,
                candidate_budget=3,
                used_budget=3,
            )
        ])
        self.assertEqual("fixed-reference-budget", decision.selected_candidate_id)
        self.assertEqual(REFERENCE_QUALIFICATION_MODE, decision.qualification_mode)
        self.assertFalse(decision.model_called)

    def test_malformed_or_out_of_range_model_output_falls_back_stably(self):
        tied = [candidate("z-candidate", 0.1), candidate("a-candidate", 0.1)]
        for response in [
            "a-candidate",
            {"selected_candidate_id": "not-in-tie", "reason": "invalid"},
            {"selected_candidate_id": "z-candidate"},
            RuntimeError("transient model failure"),
        ]:
            with self.subTest(response=response):
                model = RecordingModel(response)
                decision = agent(model).select(tied, skill=skill())
                self.assertEqual("a-candidate", decision.selected_candidate_id)
                self.assertTrue(decision.model_called)
                self.assertEqual(1, len(model.prompts))
                self.assertTrue(decision.fallback_used)
                self.assertTrue(decision.fallback_reason)

    def test_reference_tie_and_fallback_preserve_population_order(self):
        reference_agent = FinalSelectionAgent(
            RecordingModel(RuntimeError("malformed")),
            selector=DeterministicFinalSelector(REFERENCE_QUALIFICATION_MODE),
            context_framework=ContextPolicyFramework(
                AgentContextBuilder(max_chars=40000)
            ),
        )
        tied = [candidate("z-first", 0.1), candidate("a-second", 0.1)]
        decision = reference_agent.select(tied, skill=skill())
        self.assertEqual(("z-first", "a-second"), decision.tied_candidate_ids)
        self.assertEqual(("z-first", "a-second"), decision.model_candidate_ids)
        self.assertEqual("z-first", decision.selected_candidate_id)
        self.assertIn("population 原序", decision.reason)

    def test_tie_prompt_contains_only_c_d_o_f_t_and_no_memory_or_rag(self):
        model = RecordingModel({
            "selected_candidate_id": "c1",
            "reason": "stable",
            "structural_operator_comparison": {
                "structural_comparison": "c1 has stable bounded structure.",
                "operator_comparison": "c1 operator set is coherent.",
            },
        })
        memory_secret = "MEMORY_SECRET_8182"
        rag_secret = "RAG_SECRET_9293"
        population_secret = "NON_TIED_POPULATION_SECRET_4141"

        decision = agent(model).select([
            candidate("c1", 0.1),
            candidate("c2", 0.1),
            candidate("non-tied", 0.2),
        ], skill=skill(), extra_context={
            "memory": memory_secret,
            "rag": rag_secret,
            "population": population_secret,
        })

        prompt = model.prompts[0]
        self.assertIn('"C"', prompt)
        self.assertIn('"D"', prompt)
        self.assertIn('"O"', prompt)
        self.assertIn('"F"', prompt)
        self.assertIn('"T"', prompt)
        self.assertIn("c1 robust tuner", prompt)
        self.assertIn("c2 robust tuner", prompt)
        self.assertNotIn("non-tied robust tuner", prompt)
        self.assertNotIn(memory_secret, prompt)
        self.assertNotIn(rag_secret, prompt)
        self.assertNotIn(population_secret, prompt)
        self.assertEqual(
            {"memory", "population", "rag"},
            set(decision.context_metadata["excluded_keys"]),
        )
        self.assertNotIn(memory_secret, str(dict(decision.context_metadata)))
        self.assertNotIn(rag_secret, str(dict(decision.context_metadata)))


if __name__ == "__main__":
    unittest.main()
