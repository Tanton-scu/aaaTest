"""可重复执行的 Agent Engineering Harness。

Harness 运行真实 Runtime/Coordinator/Blackboard/Registry/ContextPolicy、SQLite
事实库、Artifact、ToolGateway 与 Trace；只替换 LLM、固定文献、固定数据和 Redis
这类不可控边界。每个场景都输出结构化断言，不能用“最终有 Candidate”掩盖
路由、基数或因果链错误。
"""

from __future__ import annotations

import copy
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from prievo_agent.algorithm.executable_dataset_evaluator import (
    ExecutableDatasetEvaluator,
)
from prievo_agent.algorithm.prievo_engine import PriEvOEngine
from prievo_agent.application.agent_trace import AgentTraceQuery
from prievo_agent.application.blackboard import Blackboard
from prievo_agent.application.durable_agent_coordinator import (
    DurableAgentCoordinator,
)
from prievo_agent.application.final_selection_workflow import (
    DurableFinalSelectionWorkflow,
)
from prievo_agent.application.generation_workflow import DurableGenerationWorkflow
from prievo_agent.application.prior_research_workflow import (
    DurablePriorResearchWorkflow,
)
from prievo_agent.application.repair_workflow import DurableRepairWorkflow
from prievo_agent.application.tool_governance import (
    ToolCallDenied,
    ToolGovernanceGateway,
    ToolPolicy,
)
from prievo_agent.datasets.registry import DatasetRegistry
from prievo_agent.domain.errors import (
    AlgorithmOOMError,
    AlgorithmTimeoutError,
    BudgetExhaustedError,
    CandidateSyntaxError,
)
from prievo_agent.domain.models import (
    AgentMemory,
    Candidate,
    EvaluationJob,
    EvaluationJobStatus,
    EvaluationResult,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.domain.prior import (
    LANDSCAPE_METRICS,
    InstanceSpecificPrior,
    LandscapeProfile,
    OperatorEvidence,
    OptimizerEvidence,
    SemanticRefinement,
)
from prievo_agent.infrastructure.agent_memory import RedisAgentWorkingMemory
from prievo_agent.infrastructure.scripted_fake_llm import ScriptedFakeLLM
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.checkpoint_harness import CheckpointRecoveryHarness
from prievo_agent.runtime.lifecycle import RunLifecycleService
from prievo_agent.runtime.queue_harness import QueueReliabilityHarness
from prievo_agent.runtime.state_machine import RunStateMachine


@dataclass(frozen=True)
class HarnessAssertion:
    name: str
    expected: Any
    actual: Any
    passed: bool
    detail: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class HarnessScenario:
    name: str
    category: str
    passed: bool
    duration_ms: int
    assertions: Sequence[HarnessAssertion]
    expected_trace: Sequence[str] = field(default_factory=tuple)
    actual_trace: Sequence[str] = field(default_factory=tuple)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "assertions": [item.to_dict() for item in self.assertions],
            "expected_trace": list(self.expected_trace),
            "actual_trace": list(self.actual_trace),
            "evidence": _jsonable(dict(self.evidence)),
            "error": self.error,
        }


class _Assertions:
    def __init__(self):
        self.items: List[HarnessAssertion] = []
        self.expected_trace: List[str] = []
        self.actual_trace: List[str] = []

    def equal(self, name, expected, actual, detail=""):
        self.items.append(
            HarnessAssertion(
                name,
                _jsonable(expected),
                _jsonable(actual),
                expected == actual,
                detail,
            )
        )

    def true(self, name, actual, detail=""):
        self.equal(name, True, bool(actual), detail)

    def subsequence(self, name, expected, actual, *, exact=False):
        expected_values = list(expected)
        actual_values = list(actual)
        passed = (
            expected_values == actual_values
            if exact
            else _is_subsequence(expected_values, actual_values)
        )
        self.items.append(
            HarnessAssertion(
                name,
                expected_values,
                actual_values,
                passed,
                "exact trace" if exact else "ordered subsequence trace",
            )
        )
        self.expected_trace = expected_values
        self.actual_trace = actual_values


class AgentEngineeringHarness:
    """正常/故障 Agent Path 与 Backend Reliability 的统一离线 Harness。"""

    SCENARIOS = (
        ("similarity_generation_mainline", "agent-normal", "_mainline"),
        ("knowledge_gap_research_resume", "agent-normal", "_knowledge_gap"),
        ("final_exact_tie", "agent-normal", "_final_tie"),
        ("llm_malformed_output", "agent-fault", "_malformed"),
        ("llm_transient_failure", "agent-fault", "_transient"),
        ("rag_empty_result", "agent-fault", "_rag_empty"),
        ("candidate_syntax_failure", "candidate-fault", "_candidate_syntax"),
        ("candidate_timeout", "candidate-fault", "_candidate_timeout"),
        ("candidate_oom", "candidate-fault", "_candidate_oom"),
        ("repair_success", "agent-fault", "_repair_success"),
        ("repair_beyond_limit", "agent-fault", "_repair_limit"),
        ("evaluation_transient_retry", "backend", "_queue_transient"),
        ("worker_crash", "backend", "_worker_crash"),
        ("lease_expire_recovery", "backend", "_lease_expire"),
        ("duplicate_evaluation_submit", "backend", "_duplicate_evaluation"),
        ("budget_contention", "backend", "_budget_contention"),
        ("redis_unavailable", "memory", "_redis_unavailable"),
        ("duplicate_runtime_event", "backend", "_duplicate_event"),
        ("run_crash", "recovery", "_run_crash"),
        ("checkpoint_recovery", "recovery", "_checkpoint_recovery"),
        ("pause", "lifecycle", "_pause"),
        ("resume", "lifecycle", "_resume"),
        ("cancel", "lifecycle", "_cancel"),
        ("tool_unauthorized", "security", "_tool_unauthorized"),
    )

    def __init__(self, project_root, *, strict=True):
        self.project_root = Path(project_root).resolve()
        self.strict = bool(strict)
        self._queue_cache = None
        self._recovery_cache = None
        self._lifecycle_cache = None
        self._compatibility = {}

    def run(self, report_path=None):
        started = time.perf_counter()
        scenarios = []
        with tempfile.TemporaryDirectory(prefix="prievo-agent-engineering-harness-") as directory:
            isolated_root = Path(directory)
            for index, (name, category, method_name) in enumerate(self.SCENARIOS, 1):
                case_root = isolated_root / "{:02d}-{}".format(index, name)
                case_root.mkdir(parents=True, exist_ok=True)
                scenarios.append(
                    self._run_scenario(
                        name, category, getattr(self, method_name), case_root
                    )
                )

        passed_count = sum(item.passed for item in scenarios)
        assertion_count = sum(len(item.assertions) for item in scenarios)
        passed_assertions = sum(
            assertion.passed
            for item in scenarios
            for assertion in item.assertions
        )
        report = {
            "schema_version": "agent-engineering-harness-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "passed": passed_count == len(scenarios),
            "scenario_count": len(scenarios),
            "passed_scenario_count": passed_count,
            "scenario_pass_rate": (
                passed_count / len(scenarios) if scenarios else 0.0
            ),
            "assertion_count": assertion_count,
            "passed_assertion_count": passed_assertions,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "isolation": {
                "database": "one temporary SQLite database per scenario",
                "artifacts": "one temporary Artifact directory per scenario",
                "redis": "in-process unavailable-client fixture; Redis is not Source of Truth",
                "llm": "ScriptedFakeLLM FIFO queues with full prompt/call records",
                "datasets": "fixed repository dataset plus temporary evaluator fixture",
            },
            "scenarios": [item.to_dict() for item in scenarios],
            # 兼容早期 smoke test，同时让 key 直接定位失败场景。
            "checks": {item.name: item.passed for item in scenarios},
            **self._compatibility,
        }
        if report_path is not None:
            destination = Path(report_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if self.strict and not report["passed"]:
            failed = [item.name for item in scenarios if not item.passed]
            raise AssertionError(
                "Agent Engineering Harness 未通过：{}".format(", ".join(failed))
            )
        return report

    def _run_scenario(self, name, category, method, case_root):
        assertions = _Assertions()
        started = time.perf_counter()
        evidence = {}
        error = ""
        try:
            evidence = method(case_root, assertions) or {}
        except Exception as exc:  # Harness 必须继续运行并汇总全部失败。
            error = "{}: {}".format(type(exc).__name__, exc)
            assertions.items.append(
                HarnessAssertion(
                    "scenario_execution",
                    "no exception",
                    error,
                    False,
                    "场景异常也进入可机器读取报告",
                )
            )
        passed = bool(assertions.items) and all(
            item.passed for item in assertions.items
        )
        return HarnessScenario(
            name=name,
            category=category,
            passed=passed,
            duration_ms=round((time.perf_counter() - started) * 1000),
            assertions=tuple(assertions.items),
            expected_trace=tuple(assertions.expected_trace),
            actual_trace=tuple(assertions.actual_trace),
            evidence=evidence,
            error=error,
        )

    # ------------------------------------------------------------------
    # Agent normal/fault paths

    def _mainline(self, root, check):
        store = self._store(root)
        model = ScriptedFakeLLM()
        try:
            task = OptimizationTask(
                "task-mainline",
                "Agent Harness 真实主链",
                "minimize",
                3,
                90,
                dataset_id="xgboost-Covtype",
                generations=1,
                population_size=2,
                random_seed=2026,
            )
            run = Run("run-mainline", task.id, dataset_id=task.dataset_id)
            store.add_task(task)
            store.add_run(run)
            completed = PriEvOEngine(
                store,
                DatasetRegistry(self.project_root / "resources" / "datasets"),
                self.project_root / "resources" / "prior_knowledge",
                llm=model,
            ).run(run.id)
            tasks = list(store.agent_tasks_for_run(run.id))
            events = list(store.events_for_run(run.id))
            artifacts = list(store.artifacts_for_run(run.id))
            board = Blackboard.from_store(store, run.id)
            trace = AgentTraceQuery(store).trace(run.id)
            counts = _counts(item.task_type for item in tasks)
            expected_counts = {
                "HEURISTIC_GENERATION": 8,
            }
            check.equal("run_completed", "COMPLETED", completed.status.value)
            check.equal("agent_task_cardinality", expected_counts, counts)
            check.true(
                "all_tasks_completed",
                all(item.status.value == "COMPLETED" for item in tasks),
            )
            check.equal("blackboard_task_projection", len(tasks), len(board.tasks))
            check.equal(
                "generation_draft_cardinality",
                8,
                sum(item.event_type == "CANDIDATE_DRAFT_MATERIALIZED" for item in events),
            )
            operator_selections = [
                item.payload
                for item in events
                if item.event_type == "OPERATOR_SELECTION_COMPLETED"
            ]
            check.equal(
                "operator_selection_cardinality",
                1,
                len(operator_selections),
            )
            check.equal(
                "generation_5p_combined_cardinality",
                10,
                operator_selections[0]["combined_count"],
            )
            check.equal(
                "generation_4p_offspring_cardinality",
                8,
                operator_selections[0]["offspring_count"],
            )
            check.equal(
                "generation_one_uses_early_selection",
                "fitness-diversity",
                operator_selections[0]["strategy"],
            )
            check.true(
                "final_decision_artifact",
                any(item.kind == "FINAL_SELECTION_DECISION" for item in artifacts),
            )
            check.equal("trace_task_count", len(tasks), trace["summary"]["task_count"])
            generation_prompts = [
                item.prompt for item in model.scripted_calls
                if item.route.startswith("generation")
            ]
            check.equal("generation_model_calls", 8, len(generation_prompts))
            check.true(
                "context_contains_strategy_skill",
                all("Current strategy skill" in prompt for prompt in generation_prompts),
            )
            check.true(
                "context_excludes_whole_population",
                all("current_population" not in prompt for prompt in generation_prompts),
            )
            event_types = [item.event_type for item in events]
            check.subsequence(
                "mainline_trace",
                [
                    "SIMILARITY_CANDIDATES_READY",
                    "PRIOR_EXECUTION_COMPATIBILITY_AUDITED",
                    "LANDSCAPE_ANALYZED",
                    "GENERATION_PLANNED",
                    "GENERATION_REQUESTED",
                    "CANDIDATE_DRAFT_MATERIALIZED",
                    "OPERATOR_SELECTION_COMPLETED",
                    "RUN_COMPLETED",
                ],
                event_types,
            )
            self._compatibility = {
                "candidate_count": len(list(store.candidates_for_run(run.id))),
                "agent_task_count": len(tasks),
                "generation_llm_calls": len(generation_prompts),
                "similarity_llm_calls": len(model.calls_for("similarity")),
                "final_llm_calls": len(model.calls_for("final")),
            }
            return {
                "run_id": run.id,
                "task_counts": counts,
                "artifact_count": len(artifacts),
                "trace_summary": trace["summary"],
                "llm_call_routes": [item.route for item in model.scripted_calls],
            }
        finally:
            store.close()

    def _knowledge_gap(self, root, check):
        return self._research_resume(root, check, empty=False)

    def _rag_empty(self, root, check):
        return self._research_resume(root, check, empty=True)

    def _research_resume(self, root, check, *, empty):
        store = self._store(root)
        model = ScriptedFakeLLM(
            {
                "generation:i1": [
                    ScriptedFakeLLM.knowledge_gap(
                        "Decision Tree Surrogate mechanism evidence is missing",
                        "Original Prior names the operator but does not explain it.",
                    ),
                    _candidate_draft("resumed-after-research"),
                ],
                "research:query": [
                    "decision tree surrogate rugged fitness landscape"
                ],
                "research:explain": [
                    {
                        "summary": "Tree partitions model bounded non-linear response regions.",
                        "relationship_to_landscape": (
                            "The fixture evidence supports only this bounded annotation."
                        ),
                        "strategy_implications": [
                            "Use the immutable tree Prior as bounded inspiration."
                        ],
                    }
                ],
            },
            strict=True,
        )
        backend = _FixedLiteratureBackend(empty=empty)
        try:
            suffix = "empty" if empty else "found"
            task = OptimizationTask(
                "task-research-" + suffix,
                "research fixture",
                "minimize",
                3,
                30,
                dataset_id="fixture",
                generations=1,
                population_size=1,
                random_seed=9,
            )
            run = Run("run-research-" + suffix, task.id, dataset_id="fixture")
            store.add_task(task)
            store.add_run(run)
            skills = SkillRegistry(self.project_root / "skills")
            research = DurablePriorResearchWorkflow(store, skills, model, backend)
            generation = DurableGenerationWorkflow(
                store, skills, model, prior_research_workflow=research
            )
            original_prior = _fixture_prior()
            prior_snapshot = copy.deepcopy(original_prior)
            draft, request_ref, draft_ref = generation.generate(
                run,
                task,
                original_prior,
                ["artifact-original-prior"],
                "i1",
                [],
                generation=0,
                sequence=0,
            )
            tasks = list(store.agent_tasks_for_run(run.id))
            task_counts = _counts(item.task_type for item in tasks)
            expected_counts = {
                "HEURISTIC_GENERATION": 1,
                "HEURISTIC_GENERATION_RESUME": 1,
                "PRIOR_RESEARCH": 1,
            }
            artifacts = list(store.artifacts_for_run(run.id))
            artifact_counts = _counts(item.kind for item in artifacts)
            events = list(store.events_for_run(run.id))
            event_types = [item.event_type for item in events]
            trace = AgentTraceQuery(store).trace(run.id)
            expected_trace = [
                "GENERATION_REQUESTED",
                "PRIOR_RESEARCH_REQUESTED",
                "TOOL_CALL_STARTED",
                "TOOL_CALL_COMPLETED",
                "PRIOR_RESEARCH_COMPLETED",
                "GENERATION_RESUMED_AFTER_RESEARCH",
            ]
            check.equal("agent_task_cardinality", expected_counts, task_counts)
            check.equal("original_prior_immutable", prior_snapshot, original_prior)
            check.equal("literature_tool_calls", 1, len(backend.queries))
            check.equal(
                "durable_tool_call_cardinality",
                1,
                len(store.tool_calls_for_run(run.id)),
            )
            check.true("stable_generation_request", request_ref in draft.context_refs)
            check.true("resumed_draft_materialized", bool(draft_ref))
            check.subsequence("knowledge_gap_trace", expected_trace, event_types)
            if empty:
                evidence_ref = next(
                    item.id for item in artifacts if item.kind == "LITERATURE_EVIDENCE"
                )
                evidence = _artifact_json(store, evidence_ref)
                check.equal("rag_status", "EMPTY", evidence["status"])
                check.equal("rag_items", [], evidence["items"])
                check.equal("explanation_model_calls", 0, len(model.calls_for("research:explain")))
                check.true("empty_rag_still_resumed", "evidence_status\": \"EMPTY" in draft.prompt)
            else:
                check.equal("rag_status", 1, artifact_counts.get("LITERATURE_EVIDENCE", 0))
                check.equal("explanation_model_calls", 1, len(model.calls_for("research:explain")))
                check.true("evidence_in_resume_context", "fixture-tree:methods:2" in draft.prompt)
            check.equal("trace_task_count", 3, trace["summary"]["task_count"])
            model.assert_consumed("generation:i1", "research:query")
            if not empty:
                model.assert_consumed("research:explain")
            return {
                "run_id": run.id,
                "task_counts": task_counts,
                "artifact_counts": artifact_counts,
                "llm_calls": [item.to_dict() for item in model.scripted_calls],
                "agent_trace_summary": trace["summary"],
            }
        finally:
            store.close()

    def _final_tie(self, root, check):
        store = self._store(root)
        model = ScriptedFakeLLM(
            {
                "final": [
                    {
                        "selected_candidate_id": "candidate-a",
                        "reason": "Fixture chooses the stable trajectory.",
                        "structural_operator_comparison": {
                            "structural_comparison": (
                                "candidate-a uses the more bounded control structure."
                            ),
                            "operator_comparison": (
                                "candidate-a has the clearer operator composition."
                            ),
                        },
                    }
                ]
            },
            strict=True,
        )
        try:
            task = OptimizationTask("task-final", "final tie", "minimize", 3, 20)
            run = Run("run-final", task.id)
            store.add_task(task)
            store.add_run(run)
            candidates = []
            for identifier in ("candidate-a", "candidate-b"):
                candidate = Candidate(
                    identifier,
                    run.id,
                    _long_code(identifier),
                    "final tie " + identifier,
                    ["Fixture"],
                    {},
                    objective=0.1,
                )
                store.add_candidate(candidate)
                store.add_result(
                    EvaluationResult(
                        "result-" + identifier,
                        run.id,
                        identifier,
                        0.1,
                        [0.3, 0.2, 0.1],
                        {"fixture": identifier},
                        3,
                    )
                )
                candidates.append(candidate)
            selected, decision_ref, decision = DurableFinalSelectionWorkflow(
                store, SkillRegistry(self.project_root / "skills"), model
            ).select(run.id, candidates, 3)
            tasks = list(store.agent_tasks_for_run(run.id))
            events = [item.event_type for item in store.events_for_run(run.id)]
            check.equal("selected_candidate", "candidate-a", selected)
            check.equal("final_task_cardinality", 0, len(tasks))
            check.equal("final_model_call_cardinality", 1, len(model.calls_for("final")))
            check.equal("model_allowlist", ["candidate-a", "candidate-b"], model.scripted_calls[0].response and list(model.scripted_calls[0].request.get("allowed_candidate_ids", [])))
            # select_final request 没有额外参数；严格 allowlist 仍在完整 Prompt 中。
            check.true("prompt_has_exact_allowlist", 'Allowed candidate IDs: ["candidate-a", "candidate-b"]' in model.scripted_calls[0].prompt)
            check.true("decision_artifact", bool(decision_ref))
            check.true("model_called_flag", decision["model_called"])
            check.subsequence(
                "final_tie_trace",
                ["FINAL_TIE_DETECTED"],
                events,
            )
            model.assert_consumed("final")
            return {
                "decision_artifact_id": decision_ref,
                "task_id": "",
                "prompt": model.scripted_calls[0].prompt,
            }
        finally:
            store.close()

    def _malformed(self, root, check):
        from prievo_agent.application.agent_dispatcher import AgentDispatchError

        store = self._store(root)
        model = ScriptedFakeLLM(
            {"generation:i1": [ScriptedFakeLLM.malformed()] * 3},
            strict=True,
        )
        try:
            task = OptimizationTask("task-malformed", "malformed", "minimize", 3, 20)
            run = Run("run-malformed", task.id)
            store.add_task(task)
            store.add_run(run)
            workflow = DurableGenerationWorkflow(
                store, SkillRegistry(self.project_root / "skills"), model
            )
            caught = ""
            try:
                workflow.generate(
                    run, task, _fixture_prior(), ["original-prior"], "i1", [], 0, 0
                )
            except AgentDispatchError as exc:
                caught = type(exc.__cause__).__name__
            tasks = list(store.agent_tasks_for_run(run.id))
            events = [item.event_type for item in store.events_for_run(run.id)]
            check.equal("terminal_failure_type", "MalformedHeuristicOutputError", caught)
            check.equal("single_durable_task", 1, len(tasks))
            check.equal("max_attempts_consumed", 3, tasks[0].attempts)
            check.equal("task_terminal_status", "FAILED", tasks[0].status.value)
            check.equal("model_call_cardinality", 3, len(model.calls_for("generation:i1")))
            check.equal("bounded_redrive_events", 2, events.count("AGENT_TASK_REDRIVE_SCHEDULED"))
            check.equal("candidate_draft_absent", 0, sum(
                item.kind == "CANDIDATE_DRAFT"
                for item in store.artifacts_for_run(run.id)
            ))
            check.subsequence(
                "malformed_terminal_trace",
                [
                    "AGENT_TASK_FAILED",
                    "AGENT_TASK_REDRIVE_SCHEDULED",
                    "AGENT_TASK_FAILED",
                    "AGENT_TASK_REDRIVE_SCHEDULED",
                    "AGENT_TASK_FAILED",
                ],
                events,
            )
            return {
                "task_id": tasks[0].id,
                "model_calls": [item.to_dict() for item in model.scripted_calls],
            }
        finally:
            store.close()

    def _transient(self, root, check):
        store = self._store(root)
        model = ScriptedFakeLLM(
            {
                "generation:i1": [
                    ScriptedFakeLLM.transient(
                        "fixture provider temporarily unavailable"
                    ),
                    _candidate_draft("auto-redrive-success"),
                ]
            },
            strict=True,
        )
        try:
            task = OptimizationTask("task-transient", "transient", "minimize", 3, 20)
            run = Run("run-transient", task.id)
            store.add_task(task)
            store.add_run(run)
            workflow = DurableGenerationWorkflow(
                store, SkillRegistry(self.project_root / "skills"), model
            )
            draft, request_ref, draft_ref = workflow.generate(
                run, task, _fixture_prior(), ["original-prior"], "i1", [], 0, 0
            )
            tasks = list(store.agent_tasks_for_run(run.id))
            events = [item.event_type for item in store.events_for_run(run.id)]
            check.equal("single_durable_task", 1, len(tasks))
            check.equal("task_attempts", 2, tasks[0].attempts)
            check.equal("task_terminal_status", "COMPLETED", tasks[0].status.value)
            check.equal("model_call_cardinality", 2, len(model.calls_for("generation:i1")))
            check.equal("bounded_redrive_events", 1, events.count("AGENT_TASK_REDRIVE_SCHEDULED"))
            check.equal("draft_note", "auto-redrive-success", draft.generation_note)
            check.subsequence(
                "fault_retry_trace",
                [
                    "AGENT_TASK_FAILED",
                    "AGENT_TASK_REDRIVE_SCHEDULED",
                    "AGENT_TASK_CLAIMED",
                    "AGENT_TASK_COMPLETED",
                ],
                events,
            )
            return {
                "generation_request_artifact_id": request_ref,
                "candidate_draft_artifact_id": draft_ref,
                "model_calls": [item.to_dict() for item in model.scripted_calls],
            }
        finally:
            store.close()

    # ------------------------------------------------------------------
    # Candidate execution / Repair

    def _candidate_syntax(self, root, check):
        return self._candidate_fault(
            root,
            check,
            "def run_tuners(file, budget, seed, maxlives):\n    broken = (\n",
            CandidateSyntaxError,
        )

    def _candidate_timeout(self, root, check):
        return self._candidate_fault(
            root,
            check,
            "def run_tuners(file, budget, seed, maxlives):\n    while True:\n        pass\n",
            AlgorithmTimeoutError,
        )

    def _candidate_oom(self, root, check):
        return self._candidate_fault(
            root,
            check,
            "def run_tuners(file, budget, seed, maxlives):\n    raise MemoryError('fixture OOM')\n",
            AlgorithmOOMError,
        )

    def _candidate_fault(self, root, check, code, expected_error):
        dataset_root = root / "datasets"
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "fixture.csv").write_text(
            "x,mode,$<loss\n"
            "0,a,10\n0,b,9\n1,a,8\n1,b,7\n2,a,6\n"
            "2,b,5\n3,a,4\n3,b,3\n4,a,2\n4,b,1\n",
            encoding="utf-8",
        )
        task = OptimizationTask(
            "task-candidate-fault",
            "candidate fault",
            "minimize",
            1,
            3,
            dataset_id="fixture",
            generations=0,
            population_size=1,
        )
        candidate = Candidate(
            "candidate-fault", "run-candidate-fault", code, "fault", ["Fixture"], {}
        )
        evaluator = ExecutableDatasetEvaluator(
            DatasetRegistry(dataset_root), timeout_seconds=0.2
        )
        actual_error = ""
        try:
            evaluator.execute(candidate, task)
        except Exception as exc:
            actual_error = type(exc).__name__
        check.equal("typed_candidate_failure", expected_error.__name__, actual_error)
        return {
            "evaluator": getattr(
                evaluator, "version", type(evaluator).__name__
            ),
            "expected_error": expected_error.__name__,
            "candidate_code_digest_input": code,
        }

    def _repair_success(self, root, check):
        store, task, run, parent, job = self._repair_fixture(root, repair_attempt=0)
        model = ScriptedFakeLLM()
        try:
            repaired, failure_ref, draft_ref = DurableRepairWorkflow(
                store, SkillRegistry(self.project_root / "skills"), model
            ).repair(run, task, parent, job)
            tasks = list(store.agent_tasks_for_run(run.id))
            check.equal("new_version_id", "candidate-repair-R1", repaired.id)
            check.equal("parent_immutable", parent.code, store.candidate_by_id(parent.id).code)
            check.equal("repair_task_cardinality", 1, len(tasks))
            check.equal("repair_task_status", "COMPLETED", tasks[0].status.value)
            check.equal("repair_model_phases", ["repair:diagnose", "repair:repair"], [item.route for item in model.scripted_calls])
            check.true("failure_artifact", bool(failure_ref))
            check.true("draft_artifact", bool(draft_ref))
            return {
                "parent_candidate_id": parent.id,
                "repaired_candidate_id": repaired.id,
                "failure_artifact_id": failure_ref,
                "draft_artifact_id": draft_ref,
            }
        finally:
            store.close()

    def _repair_limit(self, root, check):
        from prievo_agent.application.agent_dispatcher import AgentDispatchError

        store, task, run, parent, job = self._repair_fixture(root, repair_attempt=3)
        model = ScriptedFakeLLM()
        try:
            errors = []
            workflow = DurableRepairWorkflow(
                store,
                SkillRegistry(self.project_root / "skills"),
                model,
                max_repair_attempts=3,
            )
            for _ in range(3):
                try:
                    workflow.repair(run, task, parent, job)
                except AgentDispatchError as exc:
                    errors.append(type(exc.__cause__).__name__)
            tasks = list(store.agent_tasks_for_run(run.id))
            check.equal("repair_attempt_errors", ["RepairAttemptLimitError"] * 3, errors)
            check.equal("single_repair_task", 1, len(tasks))
            check.equal("task_attempts", 3, tasks[0].attempts)
            check.equal("task_failed_terminal", "FAILED", tasks[0].status.value)
            check.equal("repair_phase_never_called", 0, len(model.calls_for("repair:repair")))
            check.equal("no_new_candidate_version", 1, len(store.candidates_for_run(run.id)))
            return {
                "task_id": tasks[0].id,
                "diagnosis_call_count": len(model.calls_for("repair:diagnose")),
                "errors": errors,
            }
        finally:
            store.close()

    def _repair_fixture(self, root, repair_attempt):
        store = self._store(root)
        task = OptimizationTask("task-repair", "repair", "minimize", 2, 20)
        run = Run("run-repair", task.id)
        store.add_task(task)
        store.add_run(run)
        parent = Candidate(
            "candidate-repair",
            run.id,
            "def run_tuners(file, budget, seed, maxlives):\n    raise ValueError('broken')\n",
            "broken",
            ["Sampling"],
            {"repair_attempt": repair_attempt},
        )
        store.add_candidate(parent)
        job = EvaluationJob(
            "job-repair",
            run.id,
            task.id,
            parent.id,
            101,
            2,
            "repair-key",
            status=EvaluationJobStatus.DEAD,
            attempts=1,
            error_code="RUNTIME_ERROR",
            error_message="fixture boom",
        )
        return store, task, run, parent, job

    # ------------------------------------------------------------------
    # Queue, recovery, lifecycle, memory and tool policy

    def _queue(self, root):
        if self._queue_cache is None:
            self._queue_cache = QueueReliabilityHarness().run(root.parent / "queue-shared")
        return self._queue_cache

    def _queue_transient(self, root, check):
        report = self._queue(root)
        check.equal("retry_attempts", 2, report.retry_attempts)
        check.equal("logical_result_once", 1, report.crash_logical_results)
        return report.to_dict()

    def _worker_crash(self, root, check):
        report = self._queue(root)
        check.equal("recovered_job_status", "SUCCESS", report.crash_recovered_status)
        check.equal("benchmark_reexecuted_after_crash", 2, report.crash_benchmark_calls)
        check.equal("single_logical_result", 1, report.crash_logical_results)
        return report.to_dict()

    def _lease_expire(self, root, check):
        report = self._queue(root)
        check.equal("lease_recovery_terminal_status", "SUCCESS", report.crash_recovered_status)
        check.equal("budget_charged_once", 3, report.crash_budget_charged)
        return report.to_dict()

    def _duplicate_evaluation(self, root, check):
        report = self._queue(root)
        check.equal("duplicate_jobs_created", 1, report.duplicate_jobs_created)
        return report.to_dict()

    def _budget_contention(self, root, check):
        report = self._queue(root)
        check.true("budget_overrun_blocked", report.budget_overrun_blocked)
        check.equal(
            "two_connections_one_budget_winner",
            1,
            report.concurrent_budget_winners,
        )
        return report.to_dict()

    def _redis_unavailable(self, root, check):
        store = self._store(root)
        try:
            store.add_agent_memory(
                AgentMemory(
                    "memory-1",
                    "run-memory",
                    "fixture",
                    "LITERATURE_INSIGHT",
                    "research",
                    "durable same-run evidence",
                )
            )
            memory = RedisAgentWorkingMemory(_UnavailableRedis(), max_items=5)
            write_ok = memory.append(
                "run-memory",
                "LITERATURE_INSIGHT",
                "ephemeral write",
                scope="research",
            )
            warmed = memory.load_with_fallback(
                "run-memory", "research", store, limit=5
            )
            check.equal("redis_write_degrades", False, write_ok)
            check.equal("durable_fallback_count", 1, len(warmed))
            check.equal("same_run_content", "durable same-run evidence", warmed[0]["content"])
            check.equal("scope_preserved", "research", warmed[0]["scope"])
            return {
                "redis_source_of_truth": False,
                "fallback_items": warmed,
            }
        finally:
            store.close()

    def _duplicate_event(self, root, check):
        store = self._store(root)
        try:
            task = OptimizationTask(
                "task-duplicate-event", "duplicate event", "minimize", 1, 2
            )
            run = Run("run-duplicate-event", task.id)
            store.add_task(task)
            store.add_run(run)
            payload = json.dumps({"fixture": "generation"}, sort_keys=True).encode("utf-8")
            first = store.put_artifact(run.id, "GENERATION_REQUEST", payload, "application/json")
            second = store.put_artifact(run.id, "GENERATION_REQUEST", payload, "application/json")
            for _ in range(2):
                store.append_event(
                    run.id,
                    "GENERATION_REQUESTED",
                    "at-least-once replay fixture",
                    artifact_id=first.id,
                )
            coordinator = DurableAgentCoordinator(store)
            first_created = coordinator.reconcile(run.id)
            second_created = coordinator.reconcile(run.id)
            tasks = list(store.agent_tasks_for_run(run.id))
            check.equal("content_addressed_artifact", first.id, second.id)
            check.equal("duplicate_events_retained_for_audit", 2, sum(
                item.event_type == "GENERATION_REQUESTED"
                for item in store.events_for_run(run.id)
            ))
            check.equal("first_reconcile_created", 1, len(first_created))
            check.equal("replay_reconcile_created", 0, len(second_created))
            check.equal("derived_task_exactly_once", 1, len(tasks))
            return {
                "artifact_id": first.id,
                "agent_task_id": tasks[0].id,
                "idempotency_key": tasks[0].idempotency_key,
            }
        finally:
            store.close()

    def _checkpoint(self, root):
        if self._recovery_cache is None:
            self._recovery_cache = CheckpointRecoveryHarness(self.project_root).run(
                root.parent / "checkpoint-shared"
            )
        return self._recovery_cache

    def _run_crash(self, root, check):
        report = self._checkpoint(root)
        check.true("recovery_event_present", report.recovery_event_present)
        check.true("interrupted_checkpoint_present", report.recovered["checkpoint_count"] >= 1)
        return report.to_dict()

    def _checkpoint_recovery(self, root, check):
        report = self._checkpoint(root)
        comparable = (
            "status",
            "objective",
            "final_digest",
            "consumed_budget",
            "evaluation_result_count",
            "generation",
        )
        for key in comparable:
            check.equal("recovered_matches_" + key, report.uninterrupted[key], report.recovered[key])
        check.equal("duplicate_evaluations", 0, report.duplicate_evaluations)
        return report.to_dict()

    def _lifecycle(self, root):
        if self._lifecycle_cache is None:
            store = self._store(root.parent / "lifecycle-shared")
            task = OptimizationTask("task-lifecycle", "lifecycle", "minimize", 1, 3)
            run = Run("run-lifecycle", task.id)
            store.add_task(task)
            store.add_run(run)
            service = RunLifecycleService(store, RunStateMachine())
            run = service.start(run)
            owner_id = "agent-harness-lifecycle"
            now = datetime.now(timezone.utc)
            if not store.claim_run_lease(run.id, owner_id, now, 30):
                raise AssertionError("Lifecycle harness 未取得 Runtime lease")
            run = service.pause(run, owner_id=owner_id, now=now)
            paused_status = store.get_run(run.id).status.value
            run = service.start(run)
            resumed_status = store.get_run(run.id).status.value
            service.cancel(run)
            cancelled_status = store.get_run(run.id).status.value
            events = [item.event_type for item in store.events_for_run(run.id)]
            self._lifecycle_cache = {
                "paused_status": paused_status,
                "resumed_status": resumed_status,
                "cancelled_status": cancelled_status,
                "events": events,
            }
            store.close()
        return self._lifecycle_cache

    def _pause(self, root, check):
        result = self._lifecycle(root)
        check.equal("pause_status", "PAUSED", result["paused_status"])
        check.subsequence("pause_trace", ["RUN_STARTED", "RUN_PAUSED"], result["events"])
        return result

    def _resume(self, root, check):
        result = self._lifecycle(root)
        check.equal("resume_status", "RUNNING", result["resumed_status"])
        check.subsequence("resume_trace", ["RUN_PAUSED", "RUN_RESUMED"], result["events"])
        return result

    def _cancel(self, root, check):
        result = self._lifecycle(root)
        check.equal("cancel_status", "CANCELLED", result["cancelled_status"])
        check.subsequence("cancel_trace", ["RUN_RESUMED", "RUN_CANCELLED"], result["events"])
        return result

    def _tool_unauthorized(self, root, check):
        store = self._store(root)
        try:
            task = OptimizationTask(
                "task-tool-denied", "tool denied", "minimize", 1, 2
            )
            run = Run("run-tool-denied", task.id)
            store.add_task(task)
            store.add_run(run)
            policy = ToolPolicy(
                "literature_search",
                frozenset({"PriorResearchAgent"}),
                1,
                "READ_ONLY",
                "LOCAL_LOW",
            )
            executed = []
            denied = False
            try:
                ToolGovernanceGateway(store).execute(
                    run.id,
                    "UnauthorizedAgent",
                    policy,
                    "fixture unauthorized call",
                    lambda: executed.append(True),
                )
            except ToolCallDenied:
                denied = True
            calls = list(store.tool_calls_for_run(run.id))
            events = [item.event_type for item in store.events_for_run(run.id)]
            check.true("call_denied", denied)
            check.equal("operation_not_executed", [], executed)
            check.equal("audit_row_cardinality", 1, len(calls))
            check.equal("audit_row_status", "DENIED", calls[0].status)
            check.subsequence("denied_trace", ["TOOL_CALL_DENIED"], events)
            return {"tool_call": asdict(calls[0])}
        finally:
            store.close()

    @staticmethod
    def _store(root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        return SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")


# 兼容既有脚本和测试入口；行为已经升级为完整工程 Harness。
AgentMainlineHarness = AgentEngineeringHarness


class _FixedLiteratureBackend:
    def __init__(self, empty=False):
        self.empty = bool(empty)
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        if self.empty:
            return []
        return [
            {
                "paper_id": "fixture-tree",
                "title": "Tree Surrogates for Configuration Optimization",
                "section": "Methods",
                "chunk_id": "fixture-tree:methods:2",
                "content": "Tree partitions capture bounded non-linear interactions.",
                "score": 0.91,
                "identifier": "doi:fixture/tree",
                "source_path": "fixture/tree.pdf",
                "rerank_score": 0.91,
            }
        ]


class _UnavailableRedis:
    def pipeline(self):
        raise ConnectionError("fixture Redis unavailable")

    def lrange(self, *args, **kwargs):
        raise ConnectionError("fixture Redis unavailable")


def _fixture_prior():
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


def _candidate_draft(note):
    return {
        "result_type": "CandidateDraft",
        "code": (
            "def run_tuners(file, budget, seed, maxlives):\n"
            "    used_budget = 0\n"
            "    lives = 0\n"
            "    history = {}\n"
            "    best = float('inf')\n"
            "    cursor = 0\n"
            "    while used_budget < budget and lives < maxlives:\n"
            "        configuration = []\n"
            "        index = seed + cursor * 3\n"
            "        for values in file.independent_set:\n"
            "            configuration.append(values[index % len(values)])\n"
            "            index //= len(values)\n"
            "        before = used_budget\n"
            "        used_budget, lives, history, best, score, mapped = evaluate(\n"
            "            used_budget, lives, history, best, configuration\n"
            "        )\n"
            "        cursor += 1\n"
            "        if used_budget == before and cursor > budget * 100:\n"
            "            break\n"
            "    return best\n"
        ),
        "description": "Harness deterministic candidate after bounded evidence.",
        "operators": ["Decision Tree Surrogate"],
        "generation_note": str(note),
    }


def _long_code(value):
    lines = ["def run_tuners(file, budget, seed, maxlives):"]
    lines.extend("    fixture_{} = {}".format(index, index) for index in range(55))
    lines.append("    return {!r}".format(value))
    return "\n".join(lines) + "\n"


def _artifact_json(store, artifact_id):
    return json.loads(store.artifact_content(artifact_id).decode("utf-8"))


def _counts(values: Iterable[str]) -> Dict[str, int]:
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _is_subsequence(expected: Sequence[str], actual: Sequence[str]) -> bool:
    cursor = iter(actual)
    return all(any(value == expected_value for value in cursor) for expected_value in expected)


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return _jsonable(value.__dict__)
    return str(value)


__all__ = [
    "AgentEngineeringHarness",
    "AgentMainlineHarness",
    "HarnessAssertion",
    "HarnessScenario",
]
