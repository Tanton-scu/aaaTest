from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.application.workflows.generation_workflow import DurableGenerationWorkflow
from prievo_agent.agents.nodes.heuristic_generation import KnowledgeGap
from prievo_agent.domain.models import (
    AgentMemory,
    Candidate,
    CandidateStatus,
    EvaluationResult,
    OptimizationTask,
    Run,
)
from prievo_agent.domain.prior import (
    InstanceSpecificPrior,
    LandscapeProfile,
    SemanticRefinement,
)
from prievo_agent.infrastructure.agent_memory import (
    InMemoryAgentWorkingMemory,
    RedisAgentWorkingMemory,
)
from prievo_agent.infrastructure.testing.fake_llm import FakeLLM
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore


class _UnavailableRedis:
    def lrange(self, *_args, **_kwargs):
        raise ConnectionError("test redis unavailable")

    def pipeline(self):
        raise ConnectionError("test redis unavailable")


class _KnowledgeGapModel:
    def generate_heuristic_draft(self, _prompt):
        return {
            "result_type": "KnowledgeGap",
            "knowledge_gap": "missing bounded evidence",
            "reason": "current context is insufficient",
            "required_evidence": ["one paper section"],
        }


class GenerationWorkflowMemoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="prievo-generation-memory-"
        )
        work = Path(self.temporary.name)
        self.store = SQLiteRuntimeStore(
            work / "state.sqlite3", work / "artifacts"
        )
        self.project_root = Path(__file__).resolve().parents[1]
        self.skills = SkillRegistry(self.project_root / "skills")
        self.task = OptimizationTask(
            "task-memory",
            "fixture",
            "minimize",
            3,
            60,
            dataset_id="fixture",
            generations=1,
            population_size=1,
        )
        self.run = Run("run-memory", self.task.id, dataset_id="fixture")
        self.other_run = Run(
            "run-other", self.task.id, dataset_id="fixture"
        )
        self.store.add_task(self.task)
        self.store.add_run(self.run)
        self.store.add_run(self.other_run)
        self.parent = Candidate(
            "parent-memory",
            self.run.id,
            "def run_tuners(file, budget, seed, maxlives):\n"
            "    value = 1\n"
            "    return value\n",
            "parent",
            ["Random Search"],
            {"generation": 0},
            CandidateStatus.EVALUATED,
            0.5,
        )
        self.store.add_candidate(self.parent)
        self.store.add_result(
            EvaluationResult(
                "result-parent-memory",
                self.run.id,
                self.parent.id,
                0.5,
                [0.9, 0.5],
                {"x": 1},
                2,
            )
        )
        self.prior = InstanceSpecificPrior(
            LandscapeProfile(
                "fixture",
                {
                    name: 0.1
                    for name in (
                        "FDC",
                        "FBD",
                        "PLO",
                        "Skewness",
                        "Kurtosis",
                        "CL",
                        "MIE",
                        "NBC",
                    )
                },
                100,
                "fixture",
            ),
            [],
            SemanticRefinement(["source"], "fixture", "fake"),
            [],
            "prior-v1",
        )

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def _workflow(self, model=None, memory=None):
        return DurableGenerationWorkflow(
            self.store,
            self.skills,
            model or FakeLLM(),
            working_memory=memory,
        )

    def _generate(self, workflow, sequence=0, relevant_run_memory=()):
        return workflow.generate(
            self.run,
            self.task,
            self.prior,
            ["prior-ref"],
            "m2",
            [self.parent],
            1,
            sequence,
            relevant_run_memory=relevant_run_memory,
        )

    def test_candidate_draft_memory_is_durable_stable_and_used_by_next_prompt(self):
        first_model = FakeLLM()
        first_cache = InMemoryAgentWorkingMemory(max_items=10)
        first_workflow = self._workflow(first_model, first_cache)

        first, request_ref, draft_ref = self._generate(first_workflow, sequence=0)
        # 重放同一逻辑请求不会把它自己的记忆重新注入，也不会重调模型。
        replay, replay_request, replay_draft = self._generate(
            first_workflow, sequence=0
        )

        self.assertEqual(request_ref, replay_request)
        self.assertEqual(draft_ref, replay_draft)
        self.assertEqual(first.code, replay.code)
        self.assertEqual(1, len(first_model.generation_agent_calls))
        memories = self.store.agent_memories_for_run(
            self.run.id, "generation", limit=10
        )
        self.assertEqual(1, len(memories))
        self.assertEqual(draft_ref, memories[0].evidence_artifact_id)
        self.assertTrue(memories[0].id.startswith("agent-memory-"))
        summary = json.loads(memories[0].content)
        self.assertEqual(
            {"generation": 1, "sequence": 0, "strategy": "m2"},
            {key: summary[key] for key in ("generation", "sequence", "strategy")},
        )

        # 模拟 Redis 近期窗口丢失：新 cache 从 durable store 回源并回温，
        # 回源内容必须真实进入后续 Generation prompt。
        cold_cache = InMemoryAgentWorkingMemory(max_items=10)
        second_model = FakeLLM()
        second, second_request_ref, _ = self._generate(
            self._workflow(second_model, cold_cache), sequence=1
        )
        self.assertIn(first.description, second.prompt)
        request_payload = json.loads(
            self.store.artifact_content(second_request_ref).decode("utf-8")
        )
        self.assertEqual(1, len(request_payload["relevant_run_memory"]))
        self.assertEqual(
            self.run.id,
            request_payload["relevant_run_memory"][0]["run_id"],
        )
        self.assertEqual(
            "generation",
            request_payload["relevant_run_memory"][0]["scope"],
        )
        self.assertEqual(
            memories[0].id,
            cold_cache.load_recent(self.run.id, scope="generation")[0][
                "metadata"
            ]["agent_memory_id"],
        )

    def test_parent_context_reconstructs_recent_three_step_durable_lineage(self):
        ancestor = Candidate(
            "ancestor-memory",
            self.run.id,
            self.parent.code,
            "ancestor",
            ["Sampling"],
            {"generation": 1, "operator": "i1", "parents": []},
            CandidateStatus.EVALUATED,
            0.9,
        )
        middle = Candidate(
            "middle-memory",
            self.run.id,
            self.parent.code,
            "middle",
            ["Recombine"],
            {"generation": 2, "operator": "e2", "parents": [ancestor.id]},
            CandidateStatus.EVALUATED,
            0.7,
        )
        self.parent.lineage.update(
            {"generation": 3, "operator": "m2", "parents": [middle.id]}
        )
        for candidate, objective in ((ancestor, 0.9), (middle, 0.7)):
            self.store.add_candidate(candidate)
            self.store.add_result(
                EvaluationResult(
                    "result-" + candidate.id,
                    self.run.id,
                    candidate.id,
                    objective,
                    [objective],
                    {"x": 1},
                    1,
                )
            )
        self.store.add_candidate(self.parent)

        view = self._workflow()._parent_view(self.parent)

        self.assertEqual(
            [ancestor.id, middle.id, self.parent.id],
            [item["candidate_id"] for item in view["lineage"]],
        )
        self.assertEqual(
            ["i1", "e2", "m2"],
            [item["creation_strategy"] for item in view["lineage"]],
        )
        self.assertNotIn("code", view["lineage"][0])

    def test_successful_evaluation_adds_fitness_trajectory_memory_idempotently(self):
        cache = InMemoryAgentWorkingMemory(max_items=10)
        workflow = self._workflow(FakeLLM(), cache)
        draft, _, draft_ref = self._generate(workflow, sequence=0)
        candidate = Candidate(
            "evaluated-child",
            self.run.id,
            draft.code,
            draft.description,
            list(draft.operators),
            {
                "generation": 1,
                "operator": "m2",
                "parents": [self.parent.id],
                "candidate_draft_artifact_id": draft_ref,
            },
            CandidateStatus.EVALUATED,
            0.25,
            evaluation_artifact_id="evaluation-artifact-child",
        )
        result = EvaluationResult(
            "result-evaluated-child",
            self.run.id,
            candidate.id,
            0.25,
            [0.8, 0.4, 0.25],
            {"x": 2},
            3,
        )

        first = workflow.record_evaluation(
            self.run, self.task, candidate, result
        )
        replay = workflow.record_evaluation(
            self.run, self.task, candidate, result
        )

        self.assertEqual(first.id, replay.id)
        memories = list(
            self.store.agent_memories_for_run(
                self.run.id, "generation", limit=20
            )
        )
        evaluation_memories = [
            item for item in memories
            if item.memory_type == "GENERATION_EVALUATION"
        ]
        self.assertEqual(1, len(evaluation_memories))
        payload = json.loads(evaluation_memories[0].content)
        self.assertEqual(0.25, payload["fitness"])
        self.assertEqual([0.8, 0.4, 0.25], payload["trajectory"])
        self.assertEqual([self.parent.id], payload["parent_ids"])
        self.assertEqual(
            1,
            sum(
                item.event_type == "GENERATION_MEMORY_UPDATED"
                for item in self.store.events_for_run(self.run.id)
            ),
        )

    def test_redis_unavailable_falls_back_to_same_run_same_scope_only(self):
        values = (
            AgentMemory(
                "same-run-generation",
                self.run.id,
                "fixture",
                "GENERATION_SUMMARY",
                "generation",
                "same-run-generation-marker",
                "artifact-generation",
            ),
            AgentMemory(
                "same-run-research",
                self.run.id,
                "fixture",
                "LITERATURE_INSIGHT",
                "research",
                "same-run-research-secret",
                "artifact-research",
            ),
            AgentMemory(
                "other-run-generation",
                self.other_run.id,
                "fixture",
                "GENERATION_SUMMARY",
                "generation",
                "other-run-generation-secret",
                "artifact-other",
            ),
        )
        for memory in values:
            self.store.add_agent_memory(memory)
        unavailable = RedisAgentWorkingMemory(
            _UnavailableRedis(), ttl_seconds=900, max_items=10
        )

        output, request_ref, _ = self._generate(
            self._workflow(FakeLLM(), unavailable), sequence=2
        )
        request_payload = json.loads(
            self.store.artifact_content(request_ref).decode("utf-8")
        )
        serialized = json.dumps(
            request_payload["relevant_run_memory"], ensure_ascii=False
        )
        self.assertIn("same-run-generation-marker", serialized)
        self.assertIn("same-run-generation-marker", output.prompt)
        self.assertNotIn("same-run-research-secret", serialized)
        self.assertNotIn("other-run-generation-secret", serialized)

    def test_explicit_memory_is_merged_deduplicated_bounded_and_cross_run_rejected(self):
        explicit = [
            {
                "run_id": self.run.id,
                "scope": "generation",
                "agent_memory_id": "explicit-{}".format(index),
                "content": "explicit-memory-{}".format(index),
            }
            for index in range(7)
        ]
        explicit.append(dict(explicit[-1]))
        _, request_ref, _ = self._generate(
            self._workflow(), sequence=3, relevant_run_memory=explicit
        )
        payload = json.loads(
            self.store.artifact_content(request_ref).decode("utf-8")
        )
        included = payload["relevant_run_memory"]
        self.assertEqual(6, len(included))
        self.assertEqual(
            len(included),
            len({item["agent_memory_id"] for item in included}),
        )
        self.assertEqual("explicit-6", included[-1]["agent_memory_id"])

        with self.assertRaisesRegex(ValueError, "禁止跨 Run"):
            self._generate(
                self._workflow(),
                sequence=4,
                relevant_run_memory=[
                    {
                        "run_id": self.other_run.id,
                        "scope": "generation",
                        "content": "cross-run-secret",
                    }
                ],
            )

    def test_knowledge_gap_does_not_create_generation_summary(self):
        output, _, gap_ref = self._generate(
            self._workflow(_KnowledgeGapModel()), sequence=5
        )

        self.assertIsInstance(output, KnowledgeGap)
        self.assertTrue(gap_ref)
        self.assertEqual(
            [],
            self.store.agent_memories_for_run(
                self.run.id, "generation", limit=10
            ),
        )


if __name__ == "__main__":
    unittest.main()
