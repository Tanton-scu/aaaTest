from __future__ import annotations

import unittest
from pathlib import Path

from prievo_agent.agents.context import AgentContextBuilder
from prievo_agent.agents.context_policies import ContextPolicyFramework
from prievo_agent.agents.heuristic_generation import (
    CandidateDraft,
    GenerationContractError,
    GenerationParent,
    GenerationRequest,
    HeuristicGenerationAgent,
    KnowledgeGap,
    MalformedHeuristicOutputError,
)
from prievo_agent.domain.models import AgentCapability
from prievo_agent.infrastructure.skill_registry import SkillRegistry


PARENT_CODE_A = """\
def run_tuners(file, budget, seed, maxlives):
    exploration = 0.2
    return exploration
"""

PARENT_CODE_B = """\
def run_tuners(file, budget, seed, maxlives):
    samples = 2
    return samples
"""


class _Model:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def generate_heuristic_draft(self, prompt):
        self.prompts.append(prompt)
        return self.result


def _parent(candidate_id, code, operators):
    return GenerationParent(
        candidate_id=candidate_id,
        run_id="run-1",
        code=code,
        description="evaluated parent",
        fitness=0.25,
        trajectory=[0.9, 0.6, 0.25],
        operators=operators,
        lineage=[
            {"candidate_id": "old-1", "strategy": "i1"},
            {"candidate_id": "old-2", "strategy": "e2"},
            {"candidate_id": "old-3", "strategy": "m1"},
            {"candidate_id": candidate_id, "strategy": "e2"},
        ],
        used_budget=20,
    )


def _request(strategy, parents=(), memory=(), evidence=(), task=None):
    return GenerationRequest(
        run_id="run-1",
        task_contract=task or {
            "dataset_id": "xgboost-Covtype",
            "objective": "minimize",
            "generation": 3,
            "budget": 20,
            "population_size": 10,
            "interface": "run_tuners(file, budget, seed, maxlives)",
        },
        original_prior_slice={
            "optimizer": "PriEvO evidence",
            "operator_components": ["TPE", "Random Search"],
        },
        original_prior_refs=["prior:instance-a:optimizer-1"],
        strategy=strategy,
        parents=list(parents),
        relevant_run_memory=list(memory),
        evidence=list(evidence),
        context_refs=["generation-request:3:e2:0"],
    )


def _candidate(code, operators, note=""):
    return {
        "result_type": "CandidateDraft",
        "code": code,
        "description": "A bounded generated heuristic.",
        "operators": list(operators),
        "generation_note": note,
    }


class HeuristicGenerationAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.skills = SkillRegistry(root / "skills")

    def _agent(self, result):
        model = _Model(result)
        framework = ContextPolicyFramework(AgentContextBuilder(max_chars=24000))
        return HeuristicGenerationAgent(self.skills, model, framework), model

    def test_i1_has_no_parent_and_preserves_skill_context_audit(self):
        code = """\
def run_tuners(file, budget, seed, maxlives):
    return 3
"""
        memory = [{"run_id": "run-1", "artifact_ref": "memory-1", "note": "stable"}]
        evidence = [{"run_id": "run-1", "chunk_ref": "paper-1:methods:2"}]
        agent, model = self._agent(_candidate(code, ["Random Search"]))

        result = agent.generate(_request("i1", memory=memory, evidence=evidence))

        self.assertIsInstance(result, CandidateDraft)
        self.assertEqual("HeuristicGenerationAgent", agent.name)
        self.assertEqual(AgentCapability.HEURISTIC_GENERATION, agent.capability)
        self.assertEqual("synthesize", result.skill_name)
        self.assertTrue(result.skill_digest)
        self.assertTrue(result.prompt_digest)
        self.assertIn("This is the PriEvO `i1` strategy", result.prompt)
        self.assertIn("prior:instance-a:optimizer-1", result.prompt)
        self.assertIn("memory-1", result.prompt)
        self.assertIn("paper-1:methods:2", result.prompt)
        self.assertIn("memory-1", result.context_refs)
        self.assertIn("paper-1:methods:2", result.context_refs)
        self.assertNotIn("WHOLE_POPULATION", result.prompt)
        self.assertEqual(1, len(model.prompts))
        self.assertEqual("generation", result.context_metadata["policy_name"])
        self.assertEqual(result.code, result.to_dict()["code"])

    def test_parent_cardinality_is_fixed_by_strategy(self):
        p1 = _parent("p1", PARENT_CODE_A, ["TPE"])
        p2 = _parent("p2", PARENT_CODE_B, ["CMA-ES"])
        cases = [
            ("i1", [p1]),
            ("e1", [p1]),
            ("e2", [p1]),
            ("m1", [p1, p2]),
            ("m2", []),
        ]
        for strategy, parents in cases:
            with self.subTest(strategy=strategy), self.assertRaises(
                GenerationContractError
            ):
                self._agent({})[0].generate(_request(strategy, parents))

    def test_e1_requires_two_parents_and_totally_different_operators(self):
        parents = [
            _parent("p1", PARENT_CODE_A, ["TPE"]),
            _parent("p2", PARENT_CODE_B, ["CMA-ES"]),
        ]
        code = """\
def run_tuners(file, budget, seed, maxlives):
    result = seed
    return result
"""
        good = self._agent(_candidate(code, ["Random Search"]))[0].generate(
            _request("e1", parents)
        )
        self.assertEqual(["Random Search"], good.operators)

        with self.assertRaisesRegex(MalformedHeuristicOutputError, "完全不同"):
            self._agent(_candidate(code, ["tpe"]))[0].generate(
                _request("e1", parents)
            )

    def test_e2_must_inherit_or_recombine_parent_operator(self):
        parents = [
            _parent("p1", PARENT_CODE_A, ["TPE"]),
            _parent("p2", PARENT_CODE_B, ["CMA-ES"]),
        ]
        code = """\
def run_tuners(file, budget, seed, maxlives):
    if seed:
        return 1
    return 2
"""
        result = self._agent(_candidate(code, ["TPE", "CMA-ES"]))[0].generate(
            _request("e2", parents)
        )
        self.assertEqual("recombine", result.skill_name)

        with self.assertRaisesRegex(MalformedHeuristicOutputError, "继承或重组"):
            self._agent(_candidate(code, ["Random Search"]))[0].generate(
                _request("e2", parents)
            )

    def test_m1_requires_focused_structural_revision(self):
        parent = _parent("p1", PARENT_CODE_A, ["TPE"])
        revised = """\
def run_tuners(file, budget, seed, maxlives):
    exploration = 0.3
    if exploration > 0.1:
        return exploration
    return 0.1
"""
        result = self._agent(_candidate(revised, ["TPE"]))[0].generate(
            _request("m1", [parent])
        )
        self.assertEqual("revise", result.skill_name)

        parameter_only = PARENT_CODE_A.replace("0.2", "0.3")
        with self.assertRaisesRegex(MalformedHeuristicOutputError, "聚焦修订"):
            self._agent(_candidate(parameter_only, ["TPE"]))[0].generate(
                _request("m1", [parent])
            )

    def test_m2_allows_parameter_change_but_rejects_structure_or_operator_change(self):
        parent = _parent("p1", PARENT_CODE_A, ["TPE"])
        tuned = PARENT_CODE_A.replace("0.2", "0.35")
        result = self._agent(_candidate(tuned, ["TPE"]))[0].generate(
            _request("m2", [parent])
        )
        self.assertEqual("fine_tune", result.skill_name)

        structural = tuned.replace(
            "    return exploration", "    exploration += 0.1\n    return exploration"
        )
        for output in [
            _candidate(structural, ["TPE"]),
            _candidate(tuned, ["TPE", "Restart"]),
            _candidate(PARENT_CODE_A + "\n# comment only\n", ["TPE"]),
        ]:
            with self.subTest(output=output), self.assertRaises(
                MalformedHeuristicOutputError
            ):
                self._agent(output)[0].generate(_request("m2", [parent]))

    def test_knowledge_gap_is_second_valid_output_but_not_infra_failure(self):
        valid = {
            "result_type": "KnowledgeGap",
            "knowledge_gap": "How does a decision-tree surrogate fit this rugged landscape?",
            "reason": "The supplied Prior names the mechanism but gives no relation to NBC.",
            "required_evidence": ["Primary source explaining surrogate behavior under high NBC"],
        }
        result = self._agent(valid)[0].generate(_request("i1"))
        self.assertIsInstance(result, KnowledgeGap)
        self.assertEqual("synthesize", result.skill_name)
        self.assertEqual(result.knowledge_gap, result.to_dict()["knowledge_gap"])

        invalid = dict(valid)
        invalid["reason"] = "MySQL connection timeout"
        with self.assertRaisesRegex(MalformedHeuristicOutputError, "基础设施"):
            self._agent(invalid)[0].generate(_request("i1"))

    def test_strict_schema_rejects_fabricated_evaluation_facts_and_bad_interface(self):
        code = """\
def run_tuners(file, budget, seed, maxlives):
    return 1
"""
        with_fitness = _candidate(code, ["Random Search"])
        with_fitness["fitness"] = 0.01
        bad_interface = _candidate(
            "def run_tuners(file, budget):\n    return 1\n", ["Random Search"]
        )
        for output in [with_fitness, bad_interface, [], {}]:
            with self.subTest(output=output), self.assertRaises(
                MalformedHeuristicOutputError
            ):
                self._agent(output)[0].generate(_request("i1"))

    def test_cross_run_memory_and_whole_population_are_rejected_before_llm(self):
        cross_run = [{"run_id": "run-2", "artifact_ref": "foreign-memory"}]
        agent, model = self._agent({})
        with self.assertRaisesRegex(GenerationContractError, "跨 Run"):
            agent.generate(_request("i1", memory=cross_run))
        self.assertEqual([], model.prompts)

        task = {
            "objective": "minimize",
            "population": [{"code": "WHOLE_POPULATION_CODE"}],
        }
        with self.assertRaisesRegex(GenerationContractError, "整群"):
            agent.generate(_request("i1", task=task))
        self.assertEqual([], model.prompts)


if __name__ == "__main__":
    unittest.main()
