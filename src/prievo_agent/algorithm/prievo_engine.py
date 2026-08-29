import json
import logging
import csv
import uuid

from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.core.landscape_analysis import LandscapeAnalysisService
from prievo_agent.core.prior_retrieval import PriorRetrievalService
from prievo_agent.core.prior_compatibility import (
    PRIOR_COMPATIBILITY_POLICY_VERSION,
    optimizer_compatibility_rows,
)
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.prior_repository import CsvPriorRepository
from prievo_agent.infrastructure.llm_adapter import configured_llm
from prievo_agent.runtime.persistent_runtime import (
    PersistentEvolutionRuntime,
    RuntimeLeaseConflict,
)
from prievo_agent.runtime.evaluation_queue import utc_clock
from prievo_agent.runtime.lifecycle import RunLifecycleService
from prievo_agent.runtime.state_machine import RunStateMachine
from prievo_agent.domain.models import RunStatus
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.application.similarity_workflow import DurableSimilarityWorkflow
from prievo_agent.application.generation_workflow import DurableGenerationWorkflow
from prievo_agent.application.planning_workflow import DurablePlanningWorkflow
from prievo_agent.application.prior_research_workflow import (
    DurablePriorResearchWorkflow,
)
from prievo_agent.application.final_selection_workflow import (
    DurableFinalSelectionWorkflow,
)
from prievo_agent.application.repair_workflow import DurableRepairWorkflow
from prievo_agent.infrastructure.literature_hybrid import LocalHybridLiteratureRAG
from prievo_agent.runtime.final_optimization import FinalOptimizationService

from .dataset_evaluator import DatasetEvaluator
from .engine import AlgorithmEngine


logger = logging.getLogger("prievo.engine")


class PriEvOEngine(AlgorithmEngine):
    def __init__(self, store, dataset_registry, prior_root, llm=None,
                 agent_working_memory=None, research_faithful_mode=False,
                 evaluation_execution_mode="inline",
                 evaluation_timeout_seconds=10.0,
                 runtime_lease_seconds=600,
                 evaluator_mode="real"):
        self.store = store
        self.dataset_registry = dataset_registry
        self.prior_root = prior_root
        self.llm = llm or configured_llm() or FakeLLM()
        self.agent_working_memory = agent_working_memory
        self.research_faithful_mode = bool(research_faithful_mode)
        self.evaluation_execution_mode = evaluation_execution_mode
        self.evaluation_timeout_seconds = float(evaluation_timeout_seconds)
        self.runtime_lease_seconds = int(runtime_lease_seconds)
        self.evaluator_mode = str(evaluator_mode or "real").strip().lower()
        if not 0 < self.evaluation_timeout_seconds <= 300:
            raise ValueError("evaluation_timeout_seconds 必须在 (0, 300] 秒")
        if self.runtime_lease_seconds <= 0:
            raise ValueError("runtime_lease_seconds 必须大于 0")
        if self.evaluator_mode not in {"real", "fake"}:
            raise ValueError("evaluator_mode 只能是 real/fake")
        if self.evaluator_mode == "fake" and self.evaluation_execution_mode != "inline":
            raise ValueError("fake evaluator 只允许 inline/demo 路径")

    def prepare(self, run_id, runtime_owner_id=""):
        if not runtime_owner_id:
            raise RuntimeLeaseConflict("prepare 必须在 Runtime lease 内执行")
        self._require_prepare_lease(run_id, runtime_owner_id, "prepare 入口")
        run = self.store.get_run(run_id)
        task = self.store.get_task(run.task_id)
        if not run.dataset_id or run.dataset_id != task.dataset_id:
            raise ValueError("Run 与 OptimizationTask 的 Dataset ID 不一致")
        dataset = self.dataset_registry.load(run.dataset_id)
        repository = CsvPriorRepository(self.prior_root)
        requested_sample_size = min(100, _search_space_size(dataset))
        analysis = LandscapeAnalysisService(
            recorded_repository=repository
        ).analyze(
            dataset,
            seed=task.random_seed,
            sample_size=requested_sample_size,
        )
        self._require_prepare_lease(run_id, runtime_owner_id, "FLA 完成后")
        target = analysis.profile
        sample_size = len(analysis.samples.rows)
        sample_payload = {
            "dataset_id": run.dataset_id,
            "dataset_digest": dataset.info.digest,
            "random_seed": task.random_seed,
            "sample_count": sample_size,
            "sampling": analysis.samples.to_dict(),
            "fla_source": analysis.source,
            "fla_fallback_reason": analysis.fallback_reason,
        }
        sample_artifact = self.store.put_artifact(
            run.id, "LANDSCAPE_SAMPLE",
            json.dumps(sample_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            run.id, "LANDSCAPE_SAMPLED", "已从目标 Dataset 完成确定性地形采样",
            dataset_id=run.dataset_id, dataset_digest=dataset.info.digest,
            sample_count=sample_size, sample_artifact_id=sample_artifact.id,
            fla_source=analysis.source,
            fla_fallback_reason=analysis.fallback_reason,
        )
        logger.info("run_id=%s 正在检索 Dataset %s 的 instance-specific prior", run.id, run.dataset_id)
        project_root = self.prior_root.parents[1]
        skill_registry = SkillRegistry(project_root / "skills")
        # 数值 Top-5 与 semantic decision 严格分层；产品不装配 numeric fallback。
        retrieval = PriorRetrievalService(repository)
        numeric = retrieval.retrieve_numeric(target, top_k=5)
        self._require_prepare_lease(run_id, runtime_owner_id, "Similarity 前")
        refinement, top5_artifact_id, similarity_artifact_id = DurableSimilarityWorkflow(
            self.store, skill_registry, self.llm
        ).select(
            run.id,
            target,
            numeric,
            _metric_semantics(self.prior_root / "fl_metric.csv"),
        )
        self._require_prepare_lease(run_id, runtime_owner_id, "Similarity 后")
        prior = retrieval.extract(target, numeric, refinement)
        prior_refs = list(prior.refinement.selected_instances)
        compatibility_rows = optimizer_compatibility_rows(prior.optimizers)
        supported_prior_seeds = sum(
            row["status"] == "SUPPORTED" for row in compatibility_rows
        )
        compatibility_payload = {
            "policy_version": PRIOR_COMPATIBILITY_POLICY_VERSION,
            "dataset_id": run.dataset_id,
            "evidence_version": prior.evidence_version,
            "semantics": {
                "SUPPORTED": "通过静态契约，可进入受控 evaluator 尝试执行",
                "UNSUPPORTED": "仍进入 Prompt/Evidence，但不物化为 initial seed",
            },
            "total": len(compatibility_rows),
            "supported_count": supported_prior_seeds,
            "unsupported_count": len(compatibility_rows) - supported_prior_seeds,
            "records": compatibility_rows,
        }
        compatibility_artifact = self.store.put_artifact(
            run.id,
            "PRIOR_EXECUTION_COMPATIBILITY",
            json.dumps(
                compatibility_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            run.id,
            "PRIOR_EXECUTION_COMPATIBILITY_AUDITED",
            "已审计 instance-specific prior 的受控执行兼容性",
            artifact_id=compatibility_artifact.id,
            policy_version=PRIOR_COMPATIBILITY_POLICY_VERSION,
            total=len(compatibility_rows),
            supported_count=supported_prior_seeds,
            unsupported_count=len(compatibility_rows) - supported_prior_seeds,
        )
        artifact_payload = {
            "dataset_id": run.dataset_id,
            "dataset_digest": dataset.info.digest,
            "sample_size": sample_size,
            "landscape_metrics": target.metrics,
            "similar_instances": prior_refs,
            "selected_instances": list(prior.refinement.selected_instances),
            "evidence_version": prior.evidence_version,
            "fla_source": target.analyzer,
            "fla_execution_mode": analysis.source,
            "fla_fallback_reason": analysis.fallback_reason,
            "top5_artifact_id": top5_artifact_id,
            "similarity_decision_artifact_id": similarity_artifact_id,
            "prior_compatibility_artifact_id": compatibility_artifact.id,
            "prior_supported_seed_count": supported_prior_seeds,
            "prior_unsupported_seed_count": (
                len(compatibility_rows) - supported_prior_seeds
            ),
        }
        artifact = self.store.put_artifact(
            run.id, "INSTANCE_SPECIFIC_PRIOR",
            json.dumps(artifact_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            run.id, "LANDSCAPE_ANALYZED", "目标 Dataset 的 FLA 与 prior 已准备完成",
            dataset_id=run.dataset_id, dataset_digest=dataset.info.digest,
            prior_artifact_id=artifact.id,
        )
        return {
            "run": run, "task": task, "dataset": dataset,
            "prior": prior, "prior_refs": prior_refs,
        }

    def run(self, run_id):
        runtime_owner_id = "runtime-{}".format(uuid.uuid4().hex)
        now = utc_clock()
        if not self.store.claim_run_lease(
            run_id, runtime_owner_id, now, self.runtime_lease_seconds
        ):
            current = self.store.get_run(run_id)
            if current.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
                return current
            raise RuntimeLeaseConflict(
                "Run 已被其他 Runtime owner 持有：{}".format(run_id)
            )
        runtime_handoff = False
        try:
            current = self.store.get_run(run_id)
            if current.status == RunStatus.PENDING:
                RunLifecycleService(self.store, RunStateMachine()).start(
                    current, owner_id=runtime_owner_id
                )
            elif current.status != RunStatus.RUNNING:
                raise ValueError("PriEvO Engine 只接受 PENDING/RUNNING Run")
            context = self.prepare(run_id, runtime_owner_id)
            result = self._run_owned(run_id, context, runtime_owner_id)
            runtime_handoff = True
            return result
        finally:
            # PersistentRuntime 接管后会自行 release；prepare/装配异常则由这里释放。
            if not runtime_handoff:
                self.store.release_run_lease(run_id, runtime_owner_id)

    def _run_owned(self, run_id, context, runtime_owner_id):
        task = context["task"]
        core = PriEvoEvolutionCore(
            self.llm, population_size=task.population_size,
            seed=task.random_seed, prior=context["prior"],
        )
        project_root = self.prior_root.parents[1]
        skill_registry = SkillRegistry(project_root / "skills")
        prior_research_workflow = None
        if not self.research_faithful_mode:
            prior_research_workflow = DurablePriorResearchWorkflow(
                self.store,
                skill_registry,
                self.llm,
                LocalHybridLiteratureRAG(
                    [
                        project_root / "data" / "literature" / "corpus.json",
                        project_root / "data" / "literature" / "pdf_corpus.json",
                    ]
                ),
                working_memory=self.agent_working_memory,
            )
        # Full Mode 中 App 只生成 durable job，但 job identity 仍包含 evaluator
        # metadata；因此它必须和独立 Worker 使用完全相同的 timeout 配置。
        evaluator = (
            FakeEvaluator()
            if self.evaluator_mode == "fake"
            else DatasetEvaluator(
                self.dataset_registry,
                timeout_seconds=self.evaluation_timeout_seconds,
            )
        )
        runtime = PersistentEvolutionRuntime(
            self.store, evaluator, core,
            total_generations=task.generations,
            dataset_digest=context["dataset"].info.digest,
            prior_refs=context["prior_refs"],
            generation_workflow=DurableGenerationWorkflow(
                self.store,
                skill_registry,
                self.llm,
                prior_research_workflow=prior_research_workflow,
                working_memory=self.agent_working_memory,
            ),
            planning_workflow=DurablePlanningWorkflow(
                self.store,
                self.llm,
                working_memory=self.agent_working_memory,
            ),
            final_selection_workflow=DurableFinalSelectionWorkflow(
                self.store, skill_registry, self.llm,
                faithful_mode=self.research_faithful_mode,
            ),
            final_optimization_service=FinalOptimizationService(
                self.store,
                evaluator,
                evaluation_execution_mode=self.evaluation_execution_mode,
            ),
            repair_workflow=DurableRepairWorkflow(
                self.store, skill_registry, self.llm,
                working_memory=self.agent_working_memory,
            ),
            research_faithful_mode=self.research_faithful_mode,
            evaluation_execution_mode=self.evaluation_execution_mode,
            runtime_owner_id=runtime_owner_id,
            runtime_lease_seconds=self.runtime_lease_seconds,
        )
        logger.info("run_id=%s 开始 PriEvO evolution，Dataset=%s", run_id, task.dataset_id)
        result = runtime.execute(run_id, lease_already_claimed=True)
        logger.info("run_id=%s PriEvO evolution 已完成", run_id)
        return result

    def _require_prepare_lease(self, run_id, runtime_owner_id, stage):
        if not self.store.renew_run_lease(
            run_id, runtime_owner_id, utc_clock(), self.runtime_lease_seconds
        ):
            raise RuntimeLeaseConflict(
                "Runtime owner lease 已在{}丢失：{}".format(stage, run_id)
            )

    def resume(self, run_id):
        logger.info("run_id=%s 正在从 checkpoint 恢复", run_id)
        return self.run(run_id)


def _metric_semantics(path):
    """读取 reference-compatible metric definitions，完整提供 value significance。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["metric"]: {
            "full_name": row["full_name"],
            "description": row["description"],
            "calculation_method": row["calculation_method"],
            "value_significance": row["value_significance"],
        }
        for row in rows
    }


def _search_space_size(dataset):
    size = 1
    for axis in range(len(dataset.info.independent_columns)):
        size *= len({configuration[axis] for configuration in dataset.configurations})
    return size
