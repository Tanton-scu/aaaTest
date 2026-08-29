from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.application.generation_workflow import DurableGenerationWorkflow
from prievo_agent.application.prior_research_workflow import (
    DurablePriorResearchWorkflow,
)
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.domain.prior import (
    LANDSCAPE_METRICS,
    InstanceSpecificPrior,
    LandscapeProfile,
    OperatorEvidence,
    OptimizerEvidence,
    SemanticRefinement,
)
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


class _GapThenDraftModel:
    def __init__(self):
        self.generation_prompts = []
        self.query_prompts = []
        self.explanation_prompts = []

    def generate_heuristic_draft(self, prompt):
        self.generation_prompts.append(prompt)
        if len(self.generation_prompts) == 1:
            return {
                "result_type": "KnowledgeGap",
                "knowledge_gap": (
                    "Decision Tree Surrogate 与高 NBC landscape 的关系证据不足"
                ),
                "reason": "Original Prior 只有名称，缺少机制解释。",
                "required_evidence": [
                    "Primary literature about tree surrogates on rugged landscapes"
                ],
            }
        return {
            "result_type": "CandidateDraft",
            "code": (
                "def run_tuners(file, budget, seed, maxlives):\n"
                "    tuning_step = 3\n"
                "    return tuning_step\n"
            ),
            "description": "Best-effort Prior-guided heuristic after bounded research.",
            "operators": ["Decision Tree Surrogate"],
            "generation_note": "resumed-after-prior-research",
        }

    def formulate_query(self, prompt):
        self.query_prompts.append(prompt)
        return "decision tree surrogate rugged landscape nearest better clustering"

    def explain_prior(self, prompt):
        self.explanation_prompts.append(prompt)
        return {
            "summary": "Tree surrogates model bounded non-linear response partitions.",
            "relationship_to_landscape": (
                "The retrieved source supports using partitions as an annotation for "
                "a rugged high-NBC landscape."
            ),
            "strategy_implications": [
                "Use the immutable tree-surrogate Prior as bounded inspiration."
            ],
        }


class _LiteratureBackend:
    def __init__(self, empty=False, secret=""):
        self.empty = empty
        self.secret = secret
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        if self.empty:
            return []
        return [
            {
                "paper_id": "paper-tree-2025",
                "title": "Tree Surrogates for Configuration Optimization",
                "section": "Methods",
                "chunk_id": "paper-tree-2025:methods:2",
                "content": (
                    "Tree partitions capture non-linear parameter interactions."
                ),
                "score": 0.91,
                "identifier": "doi:10.1000/tree-fixture",
                "source_path": "data/papers/tree-fixture.pdf",
                "rerank_score": 0.91,
            }
        ]


def _prior():
    target = LandscapeProfile(
        "fixture-target",
        {
            metric: float(index + 1) / 10.0
            for index, metric in enumerate(LANDSCAPE_METRICS)
        },
        100,
        "fixture FLA",
    )
    operator = OperatorEvidence(
        "op-tree",
        "Decision Tree Surrogate",
        "Surrogate",
        "Partition the response space.",
        "tree = DecisionTreeRegressor()",
        "history-a",
        "tuner-a",
        "rank1",
    )
    optimizer = OptimizerEvidence(
        "tuner-a",
        "rank1",
        "history-a",
        "Tree surrogate optimizer.",
        "def run_tuners(file, budget, seed, maxlives):\n    return 1\n",
        [operator],
    )
    return InstanceSpecificPrior(
        target,
        [],
        SemanticRefinement(["history-a"], "fixture", "fixture"),
        [optimizer],
        "prior-evidence-v1",
    )


class PriorResearchWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prievo-prior-research-")
        root = Path(self.temporary.name)
        self.store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
        self.project = Path(__file__).resolve().parents[1]
        self.skills = SkillRegistry(self.project / "skills")

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def _run(self, suffix, empty=False):
        task = OptimizationTask(
            "task-" + suffix,
            "prior research " + suffix,
            "minimize",
            3,
            30,
            dataset_id="fixture-target",
            generations=1,
            population_size=1,
            random_seed=9,
        )
        run = Run("run-" + suffix, task.id, dataset_id=task.dataset_id)
        self.store.add_task(task)
        self.store.add_run(run)
        model = _GapThenDraftModel()
        backend = _LiteratureBackend(empty=empty)
        research = DurablePriorResearchWorkflow(
            self.store, self.skills, model, backend
        )
        workflow = DurableGenerationWorkflow(
            self.store,
            self.skills,
            model,
            prior_research_workflow=research,
        )
        return run, task, model, backend, workflow

    def test_gap_research_resume_is_exact_once_recoverable_and_prior_immutable(self):
        run, task, model, backend, workflow = self._run("found")
        prior = _prior()
        prior_before = copy.deepcopy(prior)

        draft, stable_request_ref, draft_ref = workflow.generate(
            run,
            task,
            prior,
            ["artifact-original-prior"],
            "i1",
            [],
            generation=0,
            sequence=0,
        )

        self.assertEqual("resumed-after-prior-research", draft.generation_note)
        self.assertEqual(2, len(model.generation_prompts))
        self.assertEqual(1, len(model.query_prompts))
        self.assertEqual(1, len(model.explanation_prompts))
        self.assertEqual(1, len(backend.queries))
        self.assertIn("paper-tree-2025:methods:2", draft.prompt)
        self.assertIn("PRIOR_RESEARCH", draft.prompt)
        self.assertIn("artifact-original-prior", draft.prompt)
        self.assertEqual(prior_before, prior)

        artifacts = list(self.store.artifacts_for_run(run.id))
        by_kind = {}
        for artifact in artifacts:
            by_kind.setdefault(artifact.kind, []).append(artifact)
        for kind in (
            "GENERATION_REQUEST",
            "KNOWLEDGE_GAP",
            "PRIOR_RESEARCH_QUERY_PROMPT",
            "PRIOR_RESEARCH_EXPLANATION_PROMPT",
            "PRIOR_EXPLANATION",
            "LITERATURE_EVIDENCE",
            "GENERATION_RESUME_REQUEST",
            "RESUMED_CANDIDATE_DRAFT",
        ):
            self.assertEqual(1, len(by_kind.get(kind, [])), kind)
        self.assertEqual(draft_ref, by_kind["RESUMED_CANDIDATE_DRAFT"][0].id)

        request = _payload(self.store, by_kind["GENERATION_REQUEST"][0].id)
        resume = _payload(self.store, by_kind["GENERATION_RESUME_REQUEST"][0].id)
        evidence = _payload(self.store, by_kind["LITERATURE_EVIDENCE"][0].id)
        explanation = _payload(self.store, by_kind["PRIOR_EXPLANATION"][0].id)
        review_skill = self.skills.require("literature_evidence_review")
        explanation_skill = self.skills.require("prior_explanation")
        provenance = {
            item["name"]: item for item in evidence["skill_provenance"]
        }
        self.assertEqual(
            {"prior_explanation", "literature_evidence_review"},
            set(provenance),
        )
        self.assertEqual(
            {
                "version": str(review_skill.version),
                "digest": review_skill.content_digest,
                "ref": "skill:literature_evidence_review@{}#{}".format(
                    review_skill.version, review_skill.content_digest
                ),
            },
            {
                key: provenance["literature_evidence_review"][key]
                for key in ("version", "digest", "ref")
            },
        )
        self.assertEqual(
            explanation_skill.content_digest,
            provenance["prior_explanation"]["digest"],
        )
        self.assertEqual(review_skill.content_digest, evidence["skill_digest"])
        self.assertEqual(review_skill.content_digest, explanation["skill_digest"])
        self.assertIn("literature_evidence_review", model.explanation_prompts[0])
        self.assertIn(review_skill.content_digest, model.explanation_prompts[0])
        self.assertNotIn(review_skill.instructions, model.query_prompts[0])
        self.assertEqual(stable_request_ref, resume["stable_generation_request_id"])
        self.assertEqual(request["original_prior_slice"], resume["original_prior_slice"])
        self.assertEqual(
            {
                request["original_prior_digest"],
                resume["original_prior_digest"],
                evidence["original_prior_digest"],
                explanation["original_prior_digest"],
            },
            {request["original_prior_digest"]},
        )
        self.assertEqual(
            by_kind["KNOWLEDGE_GAP"][0].id,
            evidence["knowledge_gap_artifact_id"],
        )

        tasks = list(self.store.agent_tasks_for_run(run.id))
        self.assertEqual(
            {
                "HEURISTIC_GENERATION": 1,
                "PRIOR_RESEARCH": 1,
                "HEURISTIC_GENERATION_RESUME": 1,
            },
            {
                task_type: sum(item.task_type == task_type for item in tasks)
                for task_type in {
                    "HEURISTIC_GENERATION",
                    "PRIOR_RESEARCH",
                    "HEURISTIC_GENERATION_RESUME",
                }
            },
        )
        events = list(self.store.events_for_run(run.id))
        self.assertEqual(
            1,
            sum(event.event_type == "TOOL_CALL_COMPLETED" for event in events),
        )
        expected_trace = [
            "GENERATION_REQUESTED",
            "PRIOR_RESEARCH_REQUESTED",
            "TOOL_CALL_STARTED",
            "TOOL_CALL_COMPLETED",
            "PRIOR_RESEARCH_COMPLETED",
            "GENERATION_RESUMED_AFTER_RESEARCH",
        ]
        positions = {
            event_type: next(
                index
                for index, event in enumerate(events)
                if event.event_type == event_type
            )
            for event_type in expected_trace
        }
        self.assertEqual(
            expected_trace, sorted(expected_trace, key=lambda item: positions[item])
        )
        memories = self.store.agent_memories_for_run(
            run.id, "research", limit=10
        )
        self.assertEqual(1, len(memories))
        research_memory = json.loads(memories[0].content)
        self.assertEqual(
            "decision tree surrogate rugged landscape nearest better clustering",
            research_memory["query"],
        )
        self.assertEqual(
            ["paper-tree-2025:methods:2"],
            [item["chunk_ref"] for item in research_memory["retrieved_papers"]],
        )
        self.assertEqual(
            {"prior_explanation", "literature_evidence_review"},
            {
                item["name"]
                for item in research_memory["skill_provenance"]
            },
        )
        self.assertEqual(
            by_kind["LITERATURE_EVIDENCE"][0].id,
            memories[0].evidence_artifact_id,
        )

        # 模拟同一个 generation step 在进程恢复后重新进入：全部 durable 输出复用。
        recovered, recovered_request_ref, recovered_draft_ref = workflow.generate(
            run,
            task,
            prior,
            ["artifact-original-prior"],
            "i1",
            [],
            generation=0,
            sequence=0,
        )
        self.assertEqual(stable_request_ref, recovered_request_ref)
        self.assertEqual(draft_ref, recovered_draft_ref)
        self.assertEqual(draft.code, recovered.code)
        self.assertEqual(2, len(model.generation_prompts))
        self.assertEqual(1, len(model.query_prompts))
        self.assertEqual(1, len(model.explanation_prompts))
        self.assertEqual(1, len(backend.queries))
        self.assertEqual(
            1,
            len(self.store.agent_memories_for_run(run.id, "research", 10)),
        )
        tasks_after = list(self.store.agent_tasks_for_run(run.id))
        self.assertEqual(len(tasks), len(tasks_after))
        events_after = list(self.store.events_for_run(run.id))
        self.assertEqual(
            1,
            sum(
                event.event_type == "GENERATION_RESUMED_AFTER_RESEARCH"
                for event in events_after
            ),
        )

    def test_empty_rag_is_audited_and_still_resumes_generation(self):
        run, task, model, backend, workflow = self._run("empty", empty=True)

        draft, _, _ = workflow.generate(
            run,
            task,
            _prior(),
            ["artifact-original-prior-empty"],
            "i1",
            [],
            generation=0,
            sequence=0,
        )

        self.assertEqual("resumed-after-prior-research", draft.generation_note)
        self.assertEqual(2, len(model.generation_prompts))
        self.assertEqual(1, len(model.query_prompts))
        self.assertEqual([], model.explanation_prompts)
        self.assertEqual(1, len(backend.queries))
        self.assertIn('"evidence_status": "EMPTY"', draft.prompt)
        evidence_artifact = next(
            item
            for item in self.store.artifacts_for_run(run.id)
            if item.kind == "LITERATURE_EVIDENCE"
        )
        evidence = _payload(self.store, evidence_artifact.id)
        self.assertEqual("EMPTY", evidence["status"])
        self.assertEqual([], evidence["items"])
        events = list(self.store.events_for_run(run.id))
        resumed = next(
            event
            for event in events
            if event.event_type == "GENERATION_RESUMED_AFTER_RESEARCH"
        )
        self.assertEqual("EMPTY", resumed.payload["evidence_status"])
        self.assertEqual(
            1,
            sum(event.event_type == "TOOL_CALL_COMPLETED" for event in events),
        )

    def test_research_history_and_evidence_never_cross_run(self):
        foreign = self.store.put_artifact(
            "run-foreign",
            "PRIOR_EXPLANATION",
            json.dumps(
                {
                    "knowledge_gap_artifact_id": "foreign-gap",
                    "summary": "FOREIGN_RUN_SECRET_7741",
                    "evidence_refs": [],
                },
                sort_keys=True,
            ).encode("utf-8"),
            "application/json",
        )
        run, task, model, backend, workflow = self._run("isolated")

        draft, _, _ = workflow.generate(
            run,
            task,
            _prior(),
            ["artifact-original-prior-isolated"],
            "i1",
            [],
            generation=0,
            sequence=0,
        )

        all_prompts = "\n".join(
            [*model.generation_prompts, *model.query_prompts, *model.explanation_prompts]
        )
        self.assertNotIn("FOREIGN_RUN_SECRET_7741", all_prompts)
        self.assertNotIn(foreign.id, all_prompts)
        self.assertNotIn("FOREIGN_RUN_SECRET_7741", draft.prompt)


def _payload(store, artifact_id):
    return json.loads(store.artifact_content(artifact_id).decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
