from __future__ import annotations

import hashlib
import inspect
import json
import time
import uuid
from typing import List, Optional

from prievo_agent import __version__
from prievo_agent.evolution.population import PriEvoEvolutionCore
from prievo_agent.evolution.schedule import is_early_generation, operators_for_generation
from prievo_agent.evolution.selection import select_population_early, select_population_late
from prievo_agent.evolution.serialization import candidate_to_dict
from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import (
    Candidate,
    CandidateStatus,
    CheckpointMetadata,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.domain.ports import CandidateEvaluator, RuntimeStore
from prievo_agent.evaluation.driver import (
    DurableEvaluationJobDriver,
    normalize_evaluation_execution_mode,
)
from prievo_agent.evaluation.queue import (
    EvaluationQueueService,
    EvaluationWorker,
    utc_clock,
)

from .lifecycle import RunLifecycleService
from .state_machine import RunStateMachine


CHECKPOINT_SCHEMA_VERSION = 4
SELECTION_CADENCE_VERSION = "generation-5p-to-p-v1"


class SimulatedHardDeath(RuntimeError):
    pass


class RuntimeLeaseConflict(RuntimeError):
    """另一个尚未过期的 Runtime owner 已持有该 Run。"""


class RuntimeLeaseLost(RuntimeError):
    """当前 Runtime owner 的 fencing lease 已丢失。"""


class _DeterministicPlan:
    def __init__(self, run_id, generation, sequence, strategy, policy):
        self.plan_id = "deterministic-plan-{}-{}-{}-{}".format(
            run_id, generation, sequence, strategy
        )
        self.run_id = run_id
        self.generation = generation
        self.sequence = sequence
        self.generation_strategy = strategy
        self.parent_selection_policy = policy
        self.decision_reason = "未装配 Planner workflow，沿用 Runtime 确定性策略"
        self.required_parent_count = {"i1": 0, "e1": 2, "e2": 2, "m1": 1, "m2": 1}[strategy]


class CooperativePause(RuntimeError):
    def __init__(self, run):
        super().__init__("Run 已在 durable safe point 暂停")
        self.run = run


class PersistentEvolutionRuntime:
    """按 generation checkpoint 推进、可跨进程恢复的确定性 Runtime。"""

    def __init__(
        self,
        store: RuntimeStore,
        evaluator: CandidateEvaluator,
        core: PriEvoEvolutionCore,
        total_generations: int = 2,
        dataset_digest="",
        prior_refs=None,
        planning_workflow=None,
        generation_workflow=None,
        final_selection_workflow=None,
        final_optimization_service=None,
        repair_workflow=None,
        research_faithful_mode=False,
        clock=None,
        sleeper=None,
        retry_wait_poll_seconds=0.25,
        evaluation_parameters_version=None,
        runtime_owner_id=None,
        runtime_lease_seconds=600,
        evaluation_execution_mode="inline",
    ) -> None:
        self.store = store
        self.evaluator = evaluator
        self.core = core
        self.total_generations = total_generations
        self.dataset_digest = dataset_digest
        self.prior_refs = list(prior_refs or [])
        self.planning_workflow = planning_workflow
        self.generation_workflow = generation_workflow
        self.final_selection_workflow = final_selection_workflow
        self.final_optimization_service = final_optimization_service
        self.repair_workflow = repair_workflow
        self.research_faithful_mode = bool(research_faithful_mode)
        self.clock = clock or utc_clock
        self.sleeper = sleeper or time.sleep
        self.retry_wait_poll_seconds = float(retry_wait_poll_seconds)
        if self.retry_wait_poll_seconds <= 0:
            raise ValueError("retry_wait_poll_seconds 必须大于 0")
        self.evaluator_version = _evaluator_version(evaluator)
        self.evaluation_parameters_version = (
            str(evaluation_parameters_version)
            if evaluation_parameters_version is not None
            else _evaluation_parameters_version(evaluator)
        )
        self.lifecycle = RunLifecycleService(store, RunStateMachine())
        self.queue = EvaluationQueueService(store, clock=self.clock)
        self.worker = EvaluationWorker(store, evaluator, clock=self.clock)
        self.evaluation_execution_mode = normalize_evaluation_execution_mode(
            evaluation_execution_mode
        )
        self.evaluation_driver = DurableEvaluationJobDriver(
            store,
            self.worker,
            mode=self.evaluation_execution_mode,
            clock=self.clock,
            sleeper=self.sleeper,
            max_poll_seconds=self.retry_wait_poll_seconds,
        )
        self.runtime_owner_id = runtime_owner_id or "runtime-{}".format(uuid.uuid4().hex)
        self.runtime_lease_seconds = int(runtime_lease_seconds)
        if self.runtime_lease_seconds <= 0:
            raise ValueError("runtime_lease_seconds 必须大于 0")

    def execute(
        self, run_id: str, crash_after_generation: Optional[int] = None,
        lease_already_claimed: bool = False,
    ) -> Run:
        if lease_already_claimed:
            acquired = self.store.renew_run_lease(
                run_id,
                self.runtime_owner_id,
                self.clock(),
                self.runtime_lease_seconds,
            )
        else:
            acquired = self.store.claim_run_lease(
                run_id,
                self.runtime_owner_id,
                self.clock(),
                self.runtime_lease_seconds,
            )
        if not acquired:
            current = self.store.get_run(run_id)
            if current.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
                return current
            raise RuntimeLeaseConflict(
                "Run 已被其他 Runtime owner 持有：{}".format(run_id)
            )
        try:
            try:
                return self._execute_owned(run_id, crash_after_generation)
            except CooperativePause as signal:
                return signal.run
            except RuntimeLeaseLost:
                current = self.store.get_run(run_id)
                if current.status == RunStatus.CANCELLED:
                    return current
                raise
            except Exception:
                # control request 可能恰好发生在“检查 flag”与 enqueue/claim 事务间；
                # Store 会正确拒绝新工作，此处把已持久化 CANCELLED 当正常终止。
                current = self.store.get_run(run_id)
                if current.status == RunStatus.CANCELLED:
                    return current
                raise
        finally:
            # 若 pause/cancel 已在事务中清空 owner，这里会安全返回 False。
            self.store.release_run_lease(run_id, self.runtime_owner_id)

    def _execute_owned(
        self, run_id: str, crash_after_generation: Optional[int] = None
    ) -> Run:
        run = self.store.get_run(run_id)
        task = self.store.get_task(run.task_id)
        if run.status == RunStatus.PENDING:
            run = self.lifecycle.start(
                run, owner_id=self.runtime_owner_id, now=self.clock()
            )
        elif run.status != RunStatus.RUNNING:
            raise ValueError("持久 Runtime 只接受 PENDING/RUNNING Run")

        self._require_runtime_lease(run.id, "执行入口")
        stale_jobs = int(self.store.recover_stale_jobs(self.clock(), run.id))
        if stale_jobs:
            self.store.append_event(
                run.id,
                EventType.RUN_RECONCILED.value,
                "恢复入口已重新调度 lease 过期的评价工作",
                stale_evaluation_jobs=stale_jobs,
            )
        checkpoint = self._optional_checkpoint(run.id)
        checkpoint_payload = (
            self._load_and_validate_checkpoint(run, checkpoint)
            if checkpoint is not None
            else None
        )
        self._reconcile_durable_evaluations(run, task)

        if checkpoint is None:
            if self.generation_workflow is None:
                population = self.core.propose(task, run)
                population = self._evaluate_new(run, task, population)
            else:
                population = []
                prior_seeds = self.core.propose_prior_seeds(run)
                for seed_candidate in prior_seeds:
                    try:
                        evaluated_values = self._evaluate_new(
                            run, task, [seed_candidate]
                        )
                        if not evaluated_values:
                            raise RuntimeError(
                                "faithful mode 已排除失败 prior/repair candidate"
                            )
                        evaluated_seed = evaluated_values[0]
                    except RuntimeError as exc:
                        self.store.append_event(
                            run.id,
                            "PRIOR_SEED_REJECTED",
                            "original prior seed 无法在受控 evaluator 中执行，已审计并跳过",
                            candidate_id=seed_candidate.id,
                            source_instance=seed_candidate.lineage.get("source_instance"),
                            optimizer=seed_candidate.lineage.get("optimizer"),
                            error_type=type(exc).__name__,
                        )
                        continue
                    population.append(evaluated_seed)
                    if len(population) >= self.core.population_size:
                        break
                i1_sequence = len(population)
                i1_attempts = 0
                max_i1_attempts = max(10, self.core.population_size * 5)
                while len(population) < self.core.population_size:
                    if i1_attempts >= max_i1_attempts:
                        raise RuntimeError(
                            "i1 连续生成/评价失败，无法补齐初始 population："
                            "valid={}/{} attempts={}".format(
                                len(population), self.core.population_size, i1_attempts
                            )
                        )
                    candidate = self._generate_durable_candidate(
                        run,
                        task,
                        population,
                        0,
                        "i1",
                        i1_sequence,
                    )
                    i1_sequence += 1
                    i1_attempts += 1
                    evaluated_values = self._evaluate_new(run, task, [candidate])
                    if not evaluated_values:
                        self.store.append_event(
                            run.id,
                            "INITIAL_I1_REJECTED",
                            "i1 Candidate 未形成有效评价，继续生成新版本补齐初始种群",
                            candidate_id=candidate.id,
                            attempt=i1_attempts,
                            valid_population_size=len(population),
                        )
                        continue
                    population.append(evaluated_values[0])
            cancelled = self._cancelled_run(run.id)
            if cancelled is not None:
                return cancelled
            population = select_population_early(population, self.core.population_size)
            initial_cursor = {
                "phase": "operator_boundary",
                "next_generation": 1,
                "next_operator_index": 0,
            }
            checkpoint = self._save_checkpoint(
                run, task, 0, population, cursor=initial_cursor, reason="initial_population"
            )
            cursor = initial_cursor
            controlled = self._control_safe_point(run, checkpoint)
            if controlled is not None:
                return controlled
            next_generation = 1
            next_operator_index = 0
        else:
            population, cursor = self._restore_checkpoint(
                run, task, checkpoint, payload=checkpoint_payload
            )
            next_generation = int(
                cursor.get("next_generation", checkpoint.generation + 1)
            )
            next_operator_index = int(cursor.get("next_operator_index", 0))
            self.store.append_event(
                run.id,
                EventType.RUN_RECOVERED.value,
                "已从 generation checkpoint 恢复运行",
                generation=checkpoint.generation,
                checkpoint_id=checkpoint.id,
            )

        for generation in range(next_generation, self.total_generations + 1):
            cancelled = self._cancelled_run(run.id)
            if cancelled is not None:
                return cancelled
            self.store.append_event(
                run.id,
                EventType.GENERATION_STARTED.value,
                "第 {} 代开始".format(generation),
                generation=generation,
            )
            # 一代内四个 strategy 必须共享同一 retained P。Operator checkpoint
            # 只保存 base population 和已完成的 offspring refs；四批全完成后才
            # 一次性执行 (P + 4P) -> P，避免后一个 strategy 偷看同代新个体。
            base_population = list(population)
            if generation == next_generation:
                generation_offspring = [
                    self._candidate_from_store(candidate_id)
                    for candidate_id in cursor.get("generation_offspring_ids", [])
                ] if checkpoint is not None else []
                completed_operator_count = int(
                    cursor.get("completed_operator_count", 0)
                ) if checkpoint is not None else 0
            else:
                generation_offspring = []
                completed_operator_count = 0
            selection_strategy = (
                "fitness-diversity"
                if is_early_generation(generation, self.total_generations)
                else "fitness-first"
            )
            generation_operators = list(
                operators_for_generation(generation, self.total_generations)
            )
            start_operator_index = (
                next_operator_index if generation == next_generation else 0
            )
            for operator_index, operator in enumerate(generation_operators):
                if operator_index < start_operator_index:
                    continue
                self.store.append_event(
                    run.id, "OPERATOR_BATCH_STARTED",
                    "operator {} 开始生成 {} 个 offspring".format(
                        operator, self.core.population_size,
                    ),
                    generation=generation, operator=operator,
                    offspring_count=self.core.population_size,
                )
                if self.generation_workflow is None:
                    offspring = self.core.evolve_operator(
                        base_population, run.id, generation, operator
                    )
                else:
                    offspring = self._generate_durable_batch(
                        run, task, base_population, generation, operator
                    )
                offspring = self._evaluate_new(run, task, offspring)
                cancelled = self._cancelled_run(run.id)
                if cancelled is not None:
                    return cancelled
                generation_offspring.extend(offspring)
                completed_operator_count += 1
                self.store.append_event(
                    run.id, "OPERATOR_BATCH_COMPLETED",
                    "operator {} 的 P 个 offspring 已评价并累计，等待本代统一选择".format(
                        operator
                    ),
                    generation=generation, operator=operator,
                    offspring_count=len(offspring),
                    accumulated_offspring_count=len(generation_offspring),
                )
                operator_cursor = {
                    "phase": "operator_boundary",
                    "next_generation": generation,
                    "next_operator_index": operator_index + 1,
                    "last_operator": operator,
                    "generation_offspring_ids": [
                        item.id for item in generation_offspring
                    ],
                    "completed_operator_count": completed_operator_count,
                    "selection_strategy": selection_strategy,
                }
                operator_checkpoint = self._save_checkpoint(
                    run,
                    task,
                    generation,
                    base_population,
                    cursor=operator_cursor,
                    reason="operator_batch_boundary",
                )
                controlled = self._control_safe_point(run, operator_checkpoint)
                if controlled is not None:
                    return controlled
            combined = [*base_population, *generation_offspring]
            if is_early_generation(generation, self.total_generations):
                population = select_population_early(
                    combined, self.core.population_size
                )
            else:
                population = select_population_late(
                    combined, self.core.population_size
                )
            selected_ids = {candidate.id for candidate in population}
            for candidate in combined:
                # Population membership 是算法关系，不复用 Candidate lifecycle status。
                candidate.status = CandidateStatus.EVALUATED
                self.store.add_candidate(candidate)
            self.store.append_event(
                run.id,
                "OPERATOR_SELECTION_COMPLETED",
                "本代 retained P 与四批 offspring 已统一按 5P -> P 筛选",
                generation=generation,
                operator="ALL_ACTIVE_STRATEGIES",
                strategy=selection_strategy,
                operator_batch_count=completed_operator_count,
                offspring_count=len(generation_offspring),
                combined_count=len(combined),
                population_ids=sorted(selected_ids),
            )
            self.store.append_event(
                run.id, "GENERATION_SELECTION_COMPLETED",
                "本代 {} 个 offspring 与 retained P 经一次 5P -> P 筛回 {} 个".format(
                    len(generation_offspring),
                    self.core.population_size,
                ),
                generation=generation,
                strategy=selection_strategy,
                offspring_count=len(generation_offspring),
                operator_batch_count=completed_operator_count,
                selection_count=1,
                combined_count=len(combined),
                population_ids=sorted(selected_ids),
            )
            run.generation = generation
            run = self.store.update_run_progress(
                run.id, self.runtime_owner_id, self.clock(), generation
            )
            self.store.append_event(
                run.id,
                EventType.GENERATION_COMPLETED.value,
                "第 {} 代完成".format(generation),
                generation=generation,
                population_ids=sorted(selected_ids),
            )
            generation_checkpoint = self._save_checkpoint(
                run,
                task,
                generation,
                population,
                cursor={
                    "phase": "generation_boundary",
                    "next_generation": generation + 1,
                    "next_operator_index": 0,
                },
                reason="generation_selection_boundary",
            )
            controlled = self._control_safe_point(run, generation_checkpoint)
            if controlled is not None:
                return controlled
            if crash_after_generation == generation:
                raise SimulatedHardDeath(
                    "在第 {} 代 checkpoint 后模拟硬退出".format(generation)
                )

        cancelled = self._cancelled_run(run.id)
        if cancelled is not None:
            return cancelled
        final_selection_decision_ref = ""
        if self.final_selection_workflow is None:
            best = min(population, key=lambda item: float(item.objective))
        else:
            selected_id, final_selection_decision_ref, _ = (
                self.final_selection_workflow.select(
                    run.id, population, task.evaluation_budget
                )
            )
            matching = [item for item in population if item.id == selected_id]
            if len(matching) != 1:
                raise RuntimeError("FinalSelection 选择了 population 之外的 Candidate")
            best = matching[0]
        for candidate in population:
            candidate.status = CandidateStatus.EVALUATED
            self.store.add_candidate(candidate)
        run.best_candidate_id = best.id
        final = self.store.put_artifact(
            run.id, "FINAL_HEURISTIC", best.code.encode("utf-8"), "text/x-python"
        )
        self.store.append_event(
            run.id,
            EventType.CANDIDATE_SELECTED.value,
            "已选出最终候选",
            candidate_id=best.id,
            objective=best.objective,
            final_artifact_id=final.id,
            final_selection_decision_artifact_id=final_selection_decision_ref,
        )
        final_optimization_report = None
        if self.final_optimization_service is not None:
            # 这是 reference executable 缺失后的明确工程补全：选择出的 heuristic
            # 保持不可变；每个 seed 以独立 clone 经同一 EvaluationJob/预算账本评价。
            final_optimization_report = self.final_optimization_service.optimize(
                run,
                task,
                best,
                self.dataset_digest,
                control_check=lambda stage: self._evaluation_wait_control(
                    run.id, "Final Optimization {}".format(stage)
                ),
            )
            self.store.append_event(
                run.id,
                "FINAL_CONFIGURATION_SELECTED",
                "已根据多随机种子 Final Optimization 报告确定最终配置",
                source_candidate_id=best.id,
                report_artifact_id=final_optimization_report.artifact_id,
                best_seed=final_optimization_report.best_seed,
                best_objective=final_optimization_report.best_objective,
                best_configuration=final_optimization_report.best_configuration,
                engineering_extension=True,
                reference_executable_missing=True,
            )
        # Final Optimization 由 EvaluationJob 短事务更新权威预算；完成 Run 前必须
        # 回读，避免 lifecycle.save_run 用调用前的内存快照覆盖新增消费。
        refreshed = self.store.get_run(run.id)
        run.consumed_evaluations = refreshed.consumed_evaluations
        run.reserved_evaluations = refreshed.reserved_evaluations
        run.pause_requested = refreshed.pause_requested
        run.cancel_requested = refreshed.cancel_requested
        run.control_reason = refreshed.control_reason
        final_safe_checkpoint = self._optional_checkpoint(run.id)
        if final_safe_checkpoint is not None:
            controlled = self._control_safe_point(run, final_safe_checkpoint)
            if controlled is not None:
                return controlled
        return self.lifecycle.complete(
            run, owner_id=self.runtime_owner_id, now=self.clock()
        )

    def _generate_durable_batch(
        self, run, task, population, generation, operator
    ):
        """逐个持久化 GenerationTask output；已完成 Draft 恢复时不重调 LLM。"""
        from prievo_agent.agents.nodes.heuristic_generation import KnowledgeGap

        offspring = []
        for index in range(self.core.population_size):
            self._pause_on_existing_safe_point(run)
            plan = self._plan_generation(
                run, task, population, generation, operator, index
            )
            parents = self._select_parents_for_plan(population, plan)
            offspring.append(
                self._generate_durable_candidate(
                    run, task, population, generation, operator, index, parents, plan
                )
            )
        return offspring

    def _generate_durable_candidate(
        self, run, task, population, generation, operator, index, parents=None, plan=None
    ):
        from prievo_agent.agents.nodes.heuristic_generation import KnowledgeGap

        plan = plan or self._plan_generation(
            run, task, population, generation, operator, index
        )
        parents = list(parents if parents is not None else self._select_parents_for_plan(population, plan))
        output, request_ref, output_ref = self.generation_workflow.generate(
            run,
            task,
            self.core.prior,
            self.prior_refs,
            plan.generation_strategy,
            parents,
            generation,
            index,
        )
        if isinstance(output, KnowledgeGap):
            self.store.append_event(
                run.id,
                "GENERATION_SUSPENDED_FOR_RESEARCH",
                "生成 Agent 发现真实知识缺口，等待 ResearchTask",
                generation=generation,
                operator=operator,
                sequence=index,
                generation_request_artifact_id=request_ref,
                knowledge_gap_artifact_id=output_ref,
            )
            raise RuntimeError(
                "GenerationTask 产生 KnowledgeGap；当前 faithful/纯领域模式未装配 LiteratureEvidence resume"
            )
        candidate = self.core.materialize_candidate(
            plan.generation_strategy,
            parents,
            run.id,
            generation,
            index,
            output.code,
            output.description,
            output.operators,
            {
                "generation_request_artifact_id": request_ref,
                "plan_id": plan.plan_id,
                "parent_selection_policy": plan.parent_selection_policy,
                "candidate_draft_artifact_id": output_ref,
                "skill_name": output.skill_name,
                "skill_version": output.skill_version,
                "skill_digest": output.skill_digest,
                "prompt_digest": output.prompt_digest,
                "prior_refs": list(output.original_prior_refs),
                "context_refs": list(output.context_refs),
                "generation_note": output.generation_note,
            },
        )
        candidate.generation = int(generation)
        candidate.plan_id = plan.plan_id
        candidate.generation_strategy = plan.generation_strategy
        candidate.code_digest = hashlib.sha256(
            candidate.code.encode("utf-8")
        ).hexdigest()
        candidate.selected_parent_ids = [parent.id for parent in parents]
        candidate.prior_refs = list(self.prior_refs)
        candidate.status = CandidateStatus.GENERATED
        # 先保存 GENERATED Candidate，再进入 validation/artifact/job 阶段。即使
        # 下一次 LLM 失败，之前成功生成的 Candidate 也已经是 durable fact。
        self.store.add_candidate(candidate)
        self.store.append_event(
            run.id,
            "CANDIDATE_DRAFT_MATERIALIZED",
            "CandidateDraft 已立即持久化为 Candidate",
            generation=generation,
            operator=operator,
            sequence=index,
            candidate_id=candidate.id,
            generation_request_artifact_id=request_ref,
            candidate_draft_artifact_id=output_ref,
        )
        return candidate

    def _evaluate_new(
        self, run: Run, task: OptimizationTask, candidates: List[Candidate]
    ) -> List[Candidate]:
        submitted_job_ids = []
        candidate_job_ids = {}
        for candidate in candidates:
            self._pause_on_existing_safe_point(run)
            try:
                existing = self.store.result_for_candidate(candidate.id)
            except KeyError:
                existing = None
            if existing is not None:
                self._assert_candidate_evaluation_identity(
                    run, task, candidate, seed=101
                )
                candidate.objective = existing.objective
                candidate.status = CandidateStatus.EVALUATED
                candidate.evaluation_artifact_id = self._candidate_from_store(
                    candidate.id
                ).evaluation_artifact_id
                self._record_generation_evaluation_memory(
                    run, task, candidate, existing
                )
                continue
            code_artifact = self.store.put_artifact(
                run.id, "CANDIDATE_CODE", candidate.code.encode("utf-8"), "text/x-python"
            )
            candidate.code_artifact_id = code_artifact.id
            candidate.status = CandidateStatus.VALIDATED
            self.store.add_candidate(candidate)
            self.store.append_event(
                run.id,
                EventType.CANDIDATE_GENERATED.value,
                "候选已生成",
                candidate_id=candidate.id,
                artifact_id=code_artifact.id,
                lineage=candidate.lineage,
            )
            job = self.queue.submit(
                run.id,
                task.id,
                candidate.id,
                seed=101,
                budget=task.evaluation_budget,
                dataset_digest=self.dataset_digest,
                evaluator_version=self.evaluator_version,
                evaluation_parameters_version=self.evaluation_parameters_version,
            )
            submitted_job_ids.append(job.id)
            candidate_job_ids[candidate.id] = job.id

        # 测试/嵌入方可能替换公开 worker seam 做故障注入；inline driver 每批读取
        # 当前 seam，external 模式仍不会调用它。
        self.evaluation_driver.worker = self.worker
        driven = self.evaluation_driver.drive(
            run.id,
            submitted_job_ids,
            control_check=lambda stage: self._evaluation_wait_control(
                run.id, stage
            ),
        )
        if driven.cancelled:
            return []

        evaluated_candidates = []
        for candidate in candidates:
            try:
                result = self.store.result_for_candidate(candidate.id)
            except KeyError as exc:
                failed_job_id = candidate_job_ids.get(candidate.id)
                failed_job = (
                    self.store.get_evaluation_job(failed_job_id)
                    if failed_job_id else None
                )
                if self.repair_workflow is not None and failed_job is not None:
                    from prievo_agent.evaluation.failures import (
                        FailureAction,
                        FailureClassifier,
                    )

                    decision = FailureClassifier().classify(
                        error_code=failed_job.error_code or "UNKNOWN"
                    )
                    if decision.action == FailureAction.REPAIR:
                        refreshed = self.store.get_run(run.id)
                        run.consumed_evaluations = refreshed.consumed_evaluations
                        run.reserved_evaluations = refreshed.reserved_evaluations
                        repair_result = self.repair_workflow.repair(
                            run, task, candidate, failed_job
                        )
                        repaired = repair_result.repaired_candidate
                        if repaired is None:
                            self.store.append_event(
                                run.id,
                                "CANDIDATE_REPAIR_SKIPPED",
                                "RepairAgent 判定当前失败不可安全修复，原 Candidate 保持 INVALID",
                                failed_candidate_id=candidate.id,
                                evaluation_job_id=failed_job.id,
                                repair_decision_artifact_id=(
                                    repair_result.repair_decision_artifact_id
                                ),
                                skipped_reason=repair_result.skipped_reason,
                            )
                            continue
                        if self.research_faithful_mode:
                            repaired.status = CandidateStatus.INVALID
                            self.store.add_candidate(repaired)
                            self.store.append_event(
                                run.id,
                                "FAITHFUL_REPAIR_EXCLUDED",
                                "faithful mode 保留 Repair 审计，但 repaired Candidate 不进入 population",
                                failed_candidate_id=candidate.id,
                                repaired_candidate_id=repaired.id,
                            )
                            continue
                        replacements = self._evaluate_new(
                            run, task, [repaired]
                        )
                        if len(replacements) != 1:
                            raise RuntimeError("repaired Candidate 未完成唯一评价") from exc
                        evaluated_candidates.append(replacements[0])
                        continue
                raise RuntimeError("候选评价未成功完成：{}".format(candidate.id)) from exc
            persisted = self._candidate_from_store(candidate.id)
            candidate.objective = result.objective
            candidate.status = persisted.status
            candidate.code_artifact_id = persisted.code_artifact_id
            candidate.evaluation_artifact_id = persisted.evaluation_artifact_id
            evaluated_candidates.append(candidate)
            self._record_generation_evaluation_memory(
                run, task, candidate, result
            )
        refreshed = self.store.get_run(run.id)
        run.consumed_evaluations = refreshed.consumed_evaluations
        run.reserved_evaluations = refreshed.reserved_evaluations
        return evaluated_candidates

    def _record_generation_evaluation_memory(self, run, task, candidate, result):
        if self.generation_workflow is None or not hasattr(
            self.generation_workflow, "record_evaluation"
        ):
            return
        try:
            self.generation_workflow.record_evaluation(
                run, task, candidate, result
            )
        except Exception as memory_error:
            # Memory 不是算法事实；评价与预算已经成功结算，不能因缓存/摘要
            # 辅助链失败而把 authoritative Result 反向标成失败。
            self.store.append_event(
                run.id,
                "GENERATION_MEMORY_DEGRADED",
                "评价已成功，但 generation memory 写入失败",
                candidate_id=candidate.id,
                error_type=type(memory_error).__name__,
                error_message=str(memory_error)[:500],
            )

    def _reconcile_durable_evaluations(self, run, task):
        """把 checkpoint 后的 Candidate/Job/Result 与 baseline 对账。

        - Result 已提交但 checkpoint 尚未更新：直接同步 Candidate，后续复用；
        - Candidate 已提交但 Job 尚未创建：用稳定 Candidate ID 创建幂等 Job；
        - Job 已存在：不重复预留预算、不重复提交。
        """

        jobs = list(self.store.evaluation_jobs_for_run(run.id))
        by_candidate = {}
        for job in jobs:
            by_candidate.setdefault(job.candidate_id, []).append(job)
        reused_results = 0
        jobs_created = 0
        for candidate in self.store.candidates_for_run(run.id):
            # Final Optimization clone 的 seed/budget/identity 由专用服务恢复；
            # 通用演化 reconcile 若按 task 默认值补 Job，会破坏多 seed 语义。
            if candidate.lineage.get("creation_type") == "FINAL_OPTIMIZATION":
                continue
            try:
                result = self.store.result_for_candidate(candidate.id)
            except KeyError:
                result = None
            if result is not None:
                self._assert_candidate_evaluation_identity(
                    run,
                    task,
                    candidate,
                    seed=int(candidate.lineage.get("evaluation_seed", 101)),
                )
                if candidate.status != CandidateStatus.EVALUATED:
                    candidate.status = CandidateStatus.EVALUATED
                    candidate.objective = result.objective
                    self.store.add_candidate(candidate)
                reused_results += 1
                continue
            if candidate.status == CandidateStatus.INVALID:
                continue
            if by_candidate.get(candidate.id):
                continue
            # Candidate code 是既存 durable fact；这里绝不重新调用 LLM。
            if not candidate.code_artifact_id:
                artifact = self.store.put_artifact(
                    run.id,
                    "CANDIDATE_CODE",
                    candidate.code.encode("utf-8"),
                    "text/x-python",
                )
                candidate.code_artifact_id = artifact.id
                candidate.status = CandidateStatus.VALIDATED
                self.store.add_candidate(candidate)
            self.queue.submit(
                run.id,
                task.id,
                candidate.id,
                seed=int(candidate.lineage.get("evaluation_seed", 101)),
                budget=task.evaluation_budget,
                dataset_digest=self.dataset_digest,
                evaluator_version=self.evaluator_version,
                evaluation_parameters_version=self.evaluation_parameters_version,
            )
            jobs_created += 1
        if reused_results or jobs_created:
            self.store.append_event(
                run.id,
                EventType.RUN_RECONCILED.value,
                "已对账 checkpoint 后 Candidate/Job/Result durable facts",
                reused_results=reused_results,
                evaluation_jobs_created=jobs_created,
            )
        return {
            "reused_results": reused_results,
            "evaluation_jobs_created": jobs_created,
        }

    def _require_runtime_lease(self, run_id, stage):
        if self.store.renew_run_lease(
            run_id,
            self.runtime_owner_id,
            self.clock(),
            self.runtime_lease_seconds,
        ):
            return
        current = self.store.get_run(run_id)
        if current.status == RunStatus.CANCELLED or current.cancel_requested:
            raise RuntimeLeaseLost("Run 已取消，Runtime 在{}停止".format(stage))
        raise RuntimeLeaseLost(
            "Runtime owner lease 已在{}丢失：{}".format(stage, run_id)
        )

    def _control_safe_point(self, run, checkpoint):
        current = self.store.get_run(run.id)
        if current.status == RunStatus.CANCELLED or current.cancel_requested:
            return current
        if current.pause_requested:
            self.lifecycle.pause(
                current, checkpoint.id, owner_id=self.runtime_owner_id,
                now=self.clock(),
            )
            return self.store.get_run(run.id)
        self._require_runtime_lease(run.id, "安全点")
        return None

    def _evaluation_wait_control(self, run_id, stage):
        """评价等待每轮的 control/lease 检查；不伪造 pause safe point。"""

        current = self.store.get_run(run_id)
        if current.status == RunStatus.CANCELLED or current.cancel_requested:
            return False
        # 只有已经存在一致 checkpoint 时才会真正 PAUSED；初始 population
        # 尚无 checkpoint 时继续等待当前评价完成，再在首次安全点确认 pause。
        self._pause_on_existing_safe_point(current)
        self._require_runtime_lease(run_id, stage)
        return True

    def _pause_on_existing_safe_point(self, run):
        """Candidate 边界发现 pause 时退回最近一致 checkpoint。

        checkpoint 后已落库的 Draft/Candidate/Result 保留，由 resume reconciliation
        复用；不存在初始 checkpoint 时继续完成初始 population，再安全暂停。
        """

        current = self.store.get_run(run.id)
        if current.status == RunStatus.CANCELLED or current.cancel_requested:
            raise RuntimeLeaseLost("Run 已取消，停止创建新工作")
        if not current.pause_requested:
            return
        checkpoint = self._optional_checkpoint(run.id)
        if checkpoint is None:
            return
        self.lifecycle.pause(
            current, checkpoint.id, owner_id=self.runtime_owner_id,
            now=self.clock(),
        )
        raise CooperativePause(self.store.get_run(run.id))

    def _wait_for_retry_or_cancel(self, run_id: str, job_ids: List[str]) -> bool:
        """等待最早 RETRY_WAIT ready；有界 sleep，期间持续检查 cancel。

        返回 ``False`` 表示 Run 已取消。PENDING 已 ready 却无法 claim、RUNNING 被
        其他 worker 占用或 ready RETRY_WAIT 仍无法 claim 都是明确调度异常，不以
        busy-spin 掩盖。
        """

        jobs = [self.store.get_evaluation_job(job_id) for job_id in job_ids]
        pending = [job for job in jobs if job.status.value == "PENDING"]
        if pending:
            raise RuntimeError(
                "evaluation queue 存在 PENDING job 但 worker 无法 claim：{}".format(
                    ", ".join(job.id for job in pending)
                )
            )
        running = [job for job in jobs if job.status.value == "RUNNING"]
        if running:
            raise RuntimeError(
                "evaluation queue job 已处于 RUNNING，但当前 worker 无进展：{}".format(
                    ", ".join(job.id for job in running)
                )
            )
        retrying = [job for job in jobs if job.status.value == "RETRY_WAIT"]
        if not retrying:
            raise RuntimeError("evaluation queue 无可 claim job，但仍有未知非终态 job")

        ready_at = min(job.available_at for job in retrying)
        stagnant_clock_reads = 0
        waited = False
        while True:
            if self._cancelled_run(run_id) is not None:
                return False
            self._pause_on_existing_safe_point(self.store.get_run(run_id))
            now = self.clock()
            remaining = (ready_at - now).total_seconds()
            if remaining <= 0:
                if not waited:
                    raise RuntimeError(
                        "evaluation queue 存在已 ready 的 RETRY_WAIT job，"
                        "但 worker 仍无法 claim：{}".format(
                            ", ".join(job.id for job in retrying)
                        )
                    )
                return True
            sleep_seconds = min(remaining, self.retry_wait_poll_seconds)
            self.sleeper(sleep_seconds)
            waited = True
            after = self.clock()
            if self._cancelled_run(run_id) is not None:
                return False
            self._pause_on_existing_safe_point(self.store.get_run(run_id))
            if after <= now:
                stagnant_clock_reads += 1
                if stagnant_clock_reads >= 3:
                    raise RuntimeError(
                        "retry sleeper 未推进 clock，拒绝无界等待/busy-spin"
                    )
            else:
                stagnant_clock_reads = 0

    def _save_checkpoint(
        self,
        run: Run,
        task: OptimizationTask,
        generation: int,
        population: List[Candidate],
        cursor=None,
        reason="safe_point",
    ) -> CheckpointMetadata:
        """保存只含稳定引用与可重放状态的 run-level snapshot。

        Candidate code/result payload 已分别 durable 保存，Checkpoint 不复制这些大
        对象；恢复时必须用 ID 回表并与 checkpoint 后事实做 reconciliation。
        """

        self._require_runtime_lease(run.id, "checkpoint 前")
        code_version = _code_version()
        refreshed = self.store.get_run(run.id)
        run.consumed_evaluations = refreshed.consumed_evaluations
        run.reserved_evaluations = refreshed.reserved_evaluations
        stable_population_ids = [item.id for item in population]
        population_artifact = self.store.put_artifact(
            run.id,
            "POPULATION_SNAPSHOT",
            json.dumps(
                {"population_ids": stable_population_ids},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8"),
            "application/json",
        )
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": run.id,
            "dataset_id": run.dataset_id,
            "dataset_digest": self.dataset_digest,
            "generation": generation,
            "population_ids": stable_population_ids,
            "population_artifact_id": population_artifact.id,
            "evaluation_references": {
                item.id: self.store.result_for_candidate(item.id).id
                for item in population
            },
            "consumed_budget": run.consumed_evaluations,
            "reserved_budget": run.reserved_evaluations,
            "remaining_budget": (
                task.total_budget
                - run.consumed_evaluations
                - run.reserved_evaluations
            ),
            "algorithm_cursor": dict(cursor or {}),
            "core_state": self.core.export_state(),
            "strategy": {
                "total_generations": self.total_generations,
                "population_size": self.core.population_size,
                "parent_count": self.core.parent_count,
                "research_faithful_mode": self.research_faithful_mode,
                "evaluation_execution_mode": self.evaluation_execution_mode,
                "selection_cadence": SELECTION_CADENCE_VERSION,
            },
            "code_version": code_version,
            "package_version": __version__,
            "model_adapter": type(self.core.llm).__name__,
            "prior_refs": list(self.prior_refs),
            "reason": reason,
        }
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        artifact = self.store.put_artifact(
            run.id, "CHECKPOINT", content, "application/json"
        )
        checkpoint_digest = hashlib.sha256(content).hexdigest()
        checkpoint = CheckpointMetadata(
            id="{}-checkpoint-{}".format(run.id, checkpoint_digest[:20]),
            run_id=run.id,
            generation=generation,
            population_ids=stable_population_ids,
            consumed_budget=run.consumed_evaluations,
            remaining_budget=(
                task.total_budget
                - run.consumed_evaluations
                - run.reserved_evaluations
            ),
            artifact_id=artifact.id,
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            code_version=code_version,
            dataset_id=run.dataset_id,
            prior_refs=self.prior_refs,
        )
        self.store.add_checkpoint(checkpoint)
        if not self.store.update_runtime_cursor(
            run.id, artifact.id, self.runtime_owner_id, self.clock()
        ):
            raise RuntimeLeaseLost("checkpoint 已写入，但 Runtime owner fencing 更新失败")
        run.runtime_cursor_artifact_id = artifact.id
        self.store.append_event(
            run.id,
            EventType.CHECKPOINT_SAVED.value,
            "generation checkpoint 已保存",
            checkpoint_id=checkpoint.id,
            generation=generation,
            artifact_id=artifact.id,
            reason=reason,
            cursor=dict(cursor or {}),
        )
        return checkpoint

    def _restore_checkpoint(
        self,
        run: Run,
        task: OptimizationTask,
        checkpoint: CheckpointMetadata,
        *,
        payload=None,
    ):
        payload = payload or self._load_and_validate_checkpoint(run, checkpoint)
        self.core.import_state(payload["core_state"])
        population = [
            self._candidate_from_store(item) for item in checkpoint.population_ids
        ]
        for candidate in population:
            # Result 是 checkpoint 外的事实源；存在则直接复用，不重复评价。
            self._assert_candidate_evaluation_identity(
                run,
                task,
                candidate,
                seed=int(candidate.lineage.get("evaluation_seed", 101)),
            )
            result = self.store.result_for_candidate(candidate.id)
            candidate.objective = result.objective
            candidate.status = CandidateStatus.EVALUATED
        return population, dict(payload.get("algorithm_cursor", {}))

    def _load_and_validate_checkpoint(
        self, run: Run, checkpoint: CheckpointMetadata
    ) -> dict:
        """在任何 Result reconciliation 前验证 checkpoint 的不可变契约。"""

        payload = json.loads(
            self.store.artifact_content(checkpoint.artifact_id).decode("utf-8")
        )
        if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("checkpoint schema version 不兼容")
        if payload["run_id"] != run.id or payload["generation"] != checkpoint.generation:
            raise RuntimeError("checkpoint identity 不一致")
        if payload.get("dataset_id", "") != run.dataset_id:
            raise RuntimeError("checkpoint Dataset ID 与 Run 不一致")
        if checkpoint.dataset_id != run.dataset_id:
            raise RuntimeError("checkpoint metadata Dataset ID 与 Run 不一致")
        if payload.get("dataset_digest", "") != self.dataset_digest:
            raise RuntimeError("checkpoint Dataset digest 与当前文件不一致")
        if payload["consumed_budget"] > run.consumed_evaluations:
            raise RuntimeError("checkpoint 预算领先于 durable Run，拒绝倒退恢复")
        self._validate_checkpoint_runtime_contract(payload)
        if checkpoint.code_version != _code_version():
            raise RuntimeError("checkpoint code version 与当前 Core 不一致")
        if list(payload.get("population_ids", [])) != list(checkpoint.population_ids):
            raise RuntimeError("checkpoint population references 不一致")
        return payload

    def _assert_candidate_evaluation_identity(
        self,
        run: Run,
        task: OptimizationTask,
        candidate: Candidate,
        *,
        seed: int,
    ) -> None:
        """复用已有 fitness 前核验完整 logical evaluation identity。"""

        self.queue.assert_candidate_identity(
            run_id=run.id,
            task_id=task.id,
            candidate_id=candidate.id,
            seed=seed,
            budget=task.evaluation_budget,
            dataset_digest=self.dataset_digest,
            evaluator_version=self.evaluator_version,
            evaluation_parameters_version=self.evaluation_parameters_version,
        )

    def _validate_checkpoint_runtime_contract(self, payload):
        """独立验证会改变恢复结果的 Runtime 配置，便于故障注入测试。"""

        strategy = payload.get("strategy", {})
        if strategy.get("total_generations") != self.total_generations:
            raise RuntimeError("checkpoint runtime strategy 不一致")
        if bool(strategy.get("research_faithful_mode", False)) != (
            self.research_faithful_mode
        ):
            raise RuntimeError("checkpoint faithful mode 与当前 Runtime 不一致")
        if strategy.get("evaluation_execution_mode", "inline") != (
            self.evaluation_execution_mode
        ):
            raise RuntimeError("checkpoint evaluation execution mode 不一致")
        if strategy.get("selection_cadence") != SELECTION_CADENCE_VERSION:
            raise RuntimeError("checkpoint 5P selection cadence 与当前 Runtime 不一致")

    def _plan_generation(self, run, task, population, generation, operator, index):
        if self.planning_workflow is None:
            return _DeterministicPlan(
                run.id,
                generation,
                index,
                operator,
                "null" if operator == "i1" else "roulette",
            )
        return self.planning_workflow.plan(
            run,
            task,
            population,
            generation,
            index,
            operator,
            previous_feedback=self._previous_plan_feedback(run.id),
        )

    def _select_parents_for_plan(self, population, plan):
        strategy = plan.generation_strategy
        required = int(plan.required_parent_count)
        if strategy == "i1" or plan.parent_selection_policy == "null":
            if required != 0:
                raise RuntimeError("null parent policy 只能用于无 parent strategy")
            return []
        if not population:
            raise ValueError("population 不能为空")
        if plan.parent_selection_policy == "roulette":
            return self.core.select_generation_parents(population, strategy)
        if plan.parent_selection_policy == "greedy":
            ranked = sorted(
                population,
                key=lambda item: (
                    float(item.objective)
                    if item.objective is not None else float("inf"),
                    item.id,
                ),
            )
            return ranked[:required]
        if plan.parent_selection_policy == "random":
            return [
                self.core.rng.choice(list(population))
                for _ in range(required)
            ]
        raise RuntimeError("未知 parent_selection_policy：{}".format(
            plan.parent_selection_policy
        ))

    def _previous_plan_feedback(self, run_id):
        if not hasattr(self.store, "generation_plans_for_run"):
            return None
        try:
            plans = list(self.store.generation_plans_for_run(run_id))
        except Exception:
            return None
        if not plans:
            return None
        latest = plans[-1]
        candidates = [
            item for item in self.store.candidates_for_run(run_id)
            if item.plan_id == latest.plan_id
        ]
        if not candidates:
            return {
                "previous_plan_id": latest.plan_id,
                "previous_generation_strategy": latest.generation_strategy,
                "previous_parent_selection_policy": latest.parent_selection_policy,
                "selected_parent_ids": [],
                "generated_candidate_id": "",
                "candidate_status": "PENDING_CANDIDATE",
            }
        candidate = sorted(candidates, key=lambda item: item.id)[-1]
        ranked = sorted(
            (
                item for item in self.store.candidates_for_run(run_id)
                if item.objective is not None
            ),
            key=lambda item: (float(item.objective), item.id),
        )
        rank_lookup = {item.id: rank for rank, item in enumerate(ranked, 1)}
        parent_values = []
        for parent_id in candidate.selected_parent_ids:
            try:
                parent = self.store.candidate_by_id(parent_id)
            except KeyError:
                continue
            if parent.objective is not None:
                parent_values.append(parent.objective)
        improvement = (
            min(parent_values) - candidate.objective
            if parent_values and candidate.objective is not None
            else None
        )
        return {
            "previous_plan_id": latest.plan_id,
            "previous_generation_strategy": latest.generation_strategy,
            "previous_parent_selection_policy": latest.parent_selection_policy,
            "selected_parent_ids": list(candidate.selected_parent_ids),
            "generated_candidate_id": candidate.id,
            "candidate_status": candidate.status.value,
            "fitness": candidate.objective,
            "rank": rank_lookup.get(candidate.id),
            "improvement": improvement,
            "repair_status": str(candidate.lineage.get("repair_status", "")),
            "repair_count": int(candidate.lineage.get("repair_count", 0) or 0),
            "failure_type": str(candidate.lineage.get("failure_type", "")),
        }

    def _candidate_from_store(self, candidate_id: str) -> Candidate:
        return self.store.candidate_by_id(candidate_id)

    def _optional_checkpoint(self, run_id: str) -> Optional[CheckpointMetadata]:
        try:
            return self.store.latest_checkpoint(run_id)
        except KeyError:
            return None

    def _cancelled_run(self, run_id: str) -> Optional[Run]:
        current = self.store.get_run(run_id)
        if current.status == RunStatus.CANCELLED:
            return current
        return None


def _code_version() -> str:
    """对所有会改变 checkpoint 恢复语义的 Core/调度/选择代码做指纹。"""

    material = "\n".join(
        (
            inspect.getsource(PriEvoEvolutionCore),
            inspect.getsource(operators_for_generation),
            inspect.getsource(is_early_generation),
            inspect.getsource(select_population_early),
            inspect.getsource(select_population_late),
            SELECTION_CADENCE_VERSION,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _evaluator_version(evaluator) -> str:
    declared = getattr(evaluator, "version", None)
    if declared is not None and str(declared).strip():
        return str(declared).strip()
    evaluator_type = type(evaluator)
    return "{}.{}".format(evaluator_type.__module__, evaluator_type.__qualname__)


def _evaluation_parameters_version(evaluator) -> str:
    for attribute in (
        "evaluation_parameters_version",
        "parameters_version",
        "parameter_version",
    ):
        declared = getattr(evaluator, attribute, None)
        if declared is not None and str(declared).strip():
            return str(declared).strip()

    # 这些参数会改变 executable evaluator 的可观察语义；registry/cache/call
    # counters 等运行态对象不能进入 identity。
    parameters = {}
    for attribute in ("timeout_seconds", "memory_limit_mb", "max_lives"):
        value = getattr(evaluator, attribute, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value is not None:
                parameters[attribute] = value
    if not parameters:
        return "parameters-v1"
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return "parameters-{}".format(digest)
