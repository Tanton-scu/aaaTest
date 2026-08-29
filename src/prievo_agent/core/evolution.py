from __future__ import annotations

import hashlib
import random
from typing import List, Sequence

from prievo_agent.domain.models import Candidate, OptimizationTask, Run
from prievo_agent.domain.ports import LLMPort
from prievo_agent.domain.prior import InstanceSpecificPrior

from .schedule import operators_for_generation
from .selection import select_parents
from .prior_compatibility import assess_prior_code


class PriEvoEvolutionCore:
    """从 PriEvO 提取的纯演化编排；I/O、评价和持久化均在端口之外。"""

    def __init__(
        self,
        llm: LLMPort,
        population_size: int = 2,
        parent_count: int = 2,
        seed: int = 2024,
        prior: InstanceSpecificPrior = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size 必须大于 0")
        self.llm = llm
        self.population_size = population_size
        self.parent_count = parent_count
        self.rng = random.Random(seed)
        self.prior = prior
        self.agent_prompt_context = ""

    def propose(self, task: OptimizationTask, run: Run) -> List[Candidate]:
        """兼容纯 Core/ResearchRunner：prior seeds 后以 legacy LLM port 补齐。"""
        population = self.propose_prior_seeds(run)
        while len(population) < self.population_size:
            population.append(self._generate("i1", [], run.id, 0, len(population)))
        return population

    def propose_prior_seeds(self, run: Run) -> List[Candidate]:
        """只物化 original prior seeds；durable Runtime 决定如何补齐 i1。"""
        population = []
        seen_code = set()
        if self.prior is not None:
            for optimizer in self.prior.optimizers:
                if not optimizer.code or optimizer.code in seen_code:
                    continue
                seen_code.add(optimizer.code)
                compatibility = assess_prior_code(optimizer.code)
                # 不兼容条目仍保留在 immutable prior、Generation Prompt 与
                # Evidence 中，但不能伪装成受控 evaluator 可执行 seed。
                if not compatibility.supported:
                    continue
                fingerprint = hashlib.sha256(optimizer.code.encode("utf-8")).hexdigest()[:12]
                population.append(
                    Candidate(
                        id="{}-g0-prior-{}".format(run.id, fingerprint),
                        run_id=run.id,
                        code=optimizer.code,
                        description=optimizer.description,
                        operators=[item.name for item in optimizer.operators],
                        lineage={
                            "operator": "preknowledge",
                            "generation": 0,
                            "source_instance": optimizer.source_instance,
                            "optimizer": optimizer.name,
                            "rank": optimizer.rank,
                            "evidence_version": self.prior.evidence_version,
                            "prior_compatibility_status": compatibility.status,
                            "prior_compatibility_policy": compatibility.policy_version,
                            "prior_code_sha256": compatibility.code_sha256,
                        },
                    )
                )
                if len(population) >= self.population_size:
                    break
        return population

    def evolve(
        self,
        population: Sequence[Candidate],
        run_id: str,
        generation: int,
        total_generations: int,
    ) -> List[Candidate]:
        """生成本代全部 operator batch；持久 Runtime 会在 batch 间执行评价和筛选。"""
        offspring = []
        for operator in operators_for_generation(generation, total_generations):
            offspring.extend(
                self.evolve_operator(population, run_id, generation, operator)
            )
        return offspring

    def evolve_operator(
        self,
        population: Sequence[Candidate],
        run_id: str,
        generation: int,
        operator: str,
    ) -> List[Candidate]:
        """与 PriEvO InterfaceEC 一致：单个 operator 生成 pop_size 个 offspring。"""
        if operator not in {"i1", "e1", "e2", "m1", "m2"}:
            raise ValueError("未知 PriEvO operator：{}".format(operator))
        offspring = []
        for index in range(self.population_size):
            parents = self.select_generation_parents(population, operator)
            offspring.append(
                self._generate(operator, parents, run_id, generation, index)
            )
        return offspring

    def select_generation_parents(
        self, population: Sequence[Candidate], operator: str
    ) -> List[Candidate]:
        """由 Core 决定 parent；Agent 只能消费该结果，不能重新选择。"""
        if operator not in {"i1", "e1", "e2", "m1", "m2"}:
            raise ValueError("未知 PriEvO operator：{}".format(operator))
        if operator == "i1":
            return []
        parent_count = 1 if operator in {"m1", "m2"} else self.parent_count
        return select_parents(population, parent_count, self.rng)

    def materialize_candidate(
        self,
        operator: str,
        parents: Sequence[Candidate],
        run_id: str,
        generation: int,
        index: int,
        code: str,
        description: str,
        operators: Sequence[str],
        audit_lineage=None,
    ) -> Candidate:
        """把已验证 Draft 转成稳定 Candidate；不执行 I/O 或评价。"""
        fingerprint = hashlib.sha256(
            "{}|{}|{}|{}|{}".format(
                run_id, generation, operator, index, code,
            ).encode("utf-8")
        ).hexdigest()[:12]
        lineage = {
            "operator": operator,
            "generation": generation,
            "parents": [parent.id for parent in parents],
            "sequence": index,
        }
        lineage.update(dict(audit_lineage or {}))
        return Candidate(
            id="{}-g{}-{}-{}".format(run_id, generation, operator, fingerprint),
            run_id=run_id,
            code=code,
            description=description,
            operators=list(operators),
            lineage=lineage,
        )

    def export_state(self) -> dict:
        llm_state = None
        if hasattr(self.llm, "export_state"):
            llm_state = self.llm.export_state()
        return {
            "rng_state": self.rng.getstate(),
            "llm_state": llm_state,
            "agent_prompt_context": self.agent_prompt_context,
        }

    def import_state(self, state: dict) -> None:
        self.rng.setstate(_nested_tuple(state["rng_state"]))
        if state.get("llm_state") is not None and hasattr(self.llm, "import_state"):
            self.llm.import_state(state["llm_state"])
        self.agent_prompt_context = str(state.get("agent_prompt_context", ""))

    def set_agent_prompt_context(self, context: str) -> None:
        self.agent_prompt_context = context or ""

    def _generate(
        self,
        operator: str,
        parents: Sequence[Candidate],
        run_id: str,
        generation: int,
        index: int,
    ) -> Candidate:
        generated = self.llm.generate_candidate(
            operator, parents, generation, self.agent_prompt_context
        )
        return self.materialize_candidate(
            operator,
            parents,
            run_id,
            generation,
            index,
            generated.code,
            generated.description,
            generated.operators,
        )


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value
