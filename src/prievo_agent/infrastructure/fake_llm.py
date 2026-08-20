from __future__ import annotations

import ast
import json
from typing import Dict, List, Sequence, Tuple

from prievo_agent.domain.prior import LANDSCAPE_METRICS

from prievo_agent.core.models import GeneratedCandidate
from prievo_agent.domain.models import Candidate


class FakeLLM:
    """记录结构化调用并返回确定性 heuristic proposal。"""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, List[str], int]] = []
        self.contexts: List[str] = []
        self.similarity_calls: List[Dict[str, object]] = []
        self.generation_agent_calls: List[Dict[str, object]] = []
        self.final_selection_calls: List[Dict[str, object]] = []
        self.repair_agent_calls: List[Dict[str, object]] = []
        self.history_summary_calls: List[Dict[str, object]] = []

    def export_state(self) -> dict:
        return {
            "calls": [list(item) for item in self.calls],
            "contexts": list(self.contexts),
            "similarity_calls": list(self.similarity_calls),
            "generation_agent_calls": list(self.generation_agent_calls),
            "final_selection_calls": list(self.final_selection_calls),
            "repair_agent_calls": list(self.repair_agent_calls),
            "history_summary_calls": list(self.history_summary_calls),
        }

    def import_state(self, state: dict) -> None:
        self.calls = [
            (str(item[0]), list(item[1]), int(item[2]))
            for item in state.get("calls", [])
        ]
        self.contexts = [str(item) for item in state.get("contexts", [])]
        self.similarity_calls = [
            dict(item) for item in state.get("similarity_calls", [])
        ]
        self.generation_agent_calls = [
            dict(item) for item in state.get("generation_agent_calls", [])
        ]
        self.final_selection_calls = [
            dict(item) for item in state.get("final_selection_calls", [])
        ]
        self.repair_agent_calls = [
            dict(item) for item in state.get("repair_agent_calls", [])
        ]
        self.history_summary_calls = [
            dict(item) for item in state.get("history_summary_calls", [])
        ]

    def generate_similarity_decision(
        self, prompt: str, allowed_instance_ids: Sequence[str]
    ) -> dict:
        """确定性选择 numeric ranking 的前两个，同时保留独立调用记录。"""

        allowed = list(allowed_instance_ids)
        selected = allowed[: min(2, len(allowed))]
        self.similarity_calls.append(
            {"prompt": prompt, "allowed_instance_ids": list(allowed)}
        )
        return {
            "selected_instance_ids": selected,
            "reason_summary": "Deterministic fake：按 numeric rank 选择前 {} 个实例".format(
                len(selected)
            ),
            "metric_evidence": {
                instance_id: {
                    metric: "deterministic fake evidence for {}".format(metric)
                    for metric in LANDSCAPE_METRICS
                }
                for instance_id in selected
            },
        }

    def generate_candidate(
        self,
        operator: str,
        parents: Sequence[Candidate],
        generation: int,
        agent_context: str = "",
    ) -> GeneratedCandidate:
        parent_ids = [parent.id for parent in parents]
        call_index = len(self.calls)
        self.calls.append((operator, parent_ids, generation))
        self.contexts.append(agent_context)
        formal_name = {
            "i1": "Synthesize",
            "e1": "Imitate",
            "e2": "Recombine",
            "m1": "Revise",
            "m2": "Fine-tune",
        }[operator]
        return GeneratedCandidate(
            code=(
                "def run_tuners(file, budget, seed, maxlives):\n"
                "    return {!r}\n".format(
                    "{}-g{}-call{}".format(operator, generation, call_index)
                )
            ),
            description="{} operator 的确定性生成结果".format(formal_name),
            operators=[formal_name, "Prior-guided Sampling"],
        )

    def generate_heuristic_draft(self, prompt: str) -> dict:
        """按 Prompt 中已经固定的 strategy/parents 生成可编排确定性 Draft。"""
        strategy_data = _prompt_json(prompt, "Current strategy skill")
        parents = _prompt_json(prompt, "Current parent candidates") or []
        strategy = str((strategy_data or {}).get("strategy", "i1"))
        call_index = len(self.generation_agent_calls)
        self.generation_agent_calls.append(
            {"strategy": strategy, "prompt": prompt, "parent_ids": [
                str(item.get("candidate_id", "")) for item in parents
            ]}
        )

        parent_operators = [
            str(operator)
            for parent in parents
            for operator in parent.get("operators", [])
        ]
        if strategy == "e1":
            operator = "Independent Exploration {}".format(call_index)
            while operator.casefold() in {
                item.casefold() for item in parent_operators
            }:
                operator += " X"
            operators = [operator]
            code = _simple_draft_code(call_index + 11)
        elif strategy == "e2":
            operators = [parent_operators[0] if parent_operators else "Recombine"]
            # parent selection 有放回，且早期 Draft 的 tuning_step 可能恰好等于
            # call_index+2。Fake 必须稳定满足真实 e2 contract，不能偶发原样复制。
            recombine_value = call_index + 2
            parent_codes = {str(item.get("code", "")).strip() for item in parents}
            code = _simple_draft_code(recombine_value)
            while code.strip() in parent_codes:
                recombine_value += 97
                code = _simple_draft_code(recombine_value)
        elif strategy == "m1" and parents:
            operators = list(parents[0].get("operators", [])) or ["Revise"]
            code = _focused_revision(
                str(parents[0].get("code", "")), call_index + 1
            )
        elif strategy == "m2" and parents:
            operators = list(parents[0].get("operators", [])) or ["Fine-tune"]
            code = _tune_first_number(
                str(parents[0].get("code", "")), call_index + 1
            )
        else:
            operators = ["Prior-guided Sampling"]
            code = _simple_draft_code(call_index + 1)
        return {
            "result_type": "CandidateDraft",
            "code": code,
            "description": "Deterministic FakeLLM {} candidate.".format(strategy),
            "operators": operators,
            "generation_note": "scripted-call-{}".format(call_index),
        }

    def select_final(self, prompt: str) -> dict:
        marker = "Allowed candidate IDs: "
        fragment = prompt.split(marker, 1)[1].split("\n", 1)[0]
        allowed = json.loads(fragment)
        self.final_selection_calls.append(
            {"prompt": prompt, "allowed_candidate_ids": list(allowed)}
        )
        return {
            "selected_candidate_id": sorted(allowed)[0],
            "reason": "Deterministic FakeLLM stable-ID audit selection.",
            "structural_operator_comparison": {
                "structural_comparison": (
                    "Deterministic fixture comparison based on tied C/D/T only."
                ),
                "operator_comparison": (
                    "Deterministic fixture comparison based on tied O only."
                ),
            },
        }

    def diagnose_candidate(self, prompt, candidate, failure_evidence):
        classification = dict(failure_evidence.get("classification", {}))
        failure_type = str(
            classification.get("failure_type")
            or failure_evidence.get("error_code")
            or "RUNTIME_ERROR"
        )
        self.repair_agent_calls.append({
            "phase": "diagnose",
            "candidate_id": candidate.id,
            "failure_type": failure_type,
            "prompt": prompt,
        })
        return {
            "repairable": True,
            "failure_type": failure_type,
            "root_cause": "Deterministic injected candidate contract defect.",
            "suggested_fix": "Replace the defective body with a bounded evaluate loop.",
            "confidence": 1.0,
        }

    def repair_candidate(self, prompt, candidate, diagnosis):
        self.repair_agent_calls.append({
            "phase": "repair",
            "candidate_id": candidate.id,
            "diagnosis_id": diagnosis.id,
            "prompt": prompt,
        })
        return {
            "code": _simple_draft_code(len(self.repair_agent_calls) + 3),
            "description": "Deterministic repaired candidate with bounded evaluate loop.",
            "operators": list(candidate.operators) or ["Bounded Sampling"],
        }

    def summarize_history(self, prompt):
        self.history_summary_calls.append({"prompt": prompt})
        return {
            "summary": "Deterministic summary of bounded earlier same-Run history.",
            "strategy_changes": [],
            "operator_changes": [],
            "fitness_trend": "Only persisted history values were considered.",
            "important_failures": [],
            "constraints": ["Keep stable source references and recent window."],
        }


def _prompt_json(prompt: str, label: str):
    marker = label + ": "
    index = prompt.find(marker)
    if index < 0:
        return None
    fragment = prompt[index + len(marker):].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(fragment)
        return value
    except (json.JSONDecodeError, TypeError):
        return None


def _simple_draft_code(value: int) -> str:
    return (
        "def run_tuners(file, budget, seed, maxlives):\n"
        "    proposal_bias = {}\n"
        "    used_budget = 0\n"
        "    lives = 0\n"
        "    history = {{}}\n"
        "    best = float('inf')\n"
        "    cursor = 0\n"
        "    while used_budget < budget and lives < maxlives:\n"
        # cursor 以 1 递增可遍历 mixed-radix configuration 空间；bias 仍让
        # 不同 Draft 产生不同真实轨迹，且不会因 step 与空间大小不互素而卡住。
        "        mixed_index = seed + proposal_bias + cursor\n"
        "        configuration = []\n"
        "        for values in file.independent_set:\n"
        "            configuration.append(values[mixed_index % len(values)])\n"
        "            mixed_index //= len(values)\n"
        "        before = used_budget\n"
        "        used_budget, lives, history, best, score, matched = evaluate(\n"
        "            used_budget, lives, history, best, configuration\n"
        "        )\n"
        "        cursor += 1\n"
        "        if used_budget == before and cursor > budget * 100:\n"
        "            break\n"
        "    return best\n".format(value)
    )


def _focused_revision(code: str, value: int) -> str:
    try:
        tree = ast.parse(code)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_tuners"
        )
        function.body.insert(
            0,
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="budget", ctx=ast.Load()),
                    ops=[ast.LtE()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=[ast.Return(value=ast.Constant(value=float(value)))],
                orelse=[],
            ),
        )
        ast.fix_missing_locations(tree)
        return ast.unparse(tree) + "\n"
    except (SyntaxError, StopIteration):
        return _simple_draft_code(value)


class _TuneNumber(ast.NodeTransformer):
    _protected_assignments = {
        "used_budget", "lives", "cursor", "before", "maxlives", "budget"
    }

    def __init__(self, delta):
        self.delta = delta
        self.changed = False

    def visit_Assign(self, node):
        names = {
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if names & self._protected_assignments:
            return node
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if (
            isinstance(node.target, ast.Name)
            and node.target.id in self._protected_assignments
        ):
            return node
        return self.generic_visit(node)

    def visit_Compare(self, node):
        names = {
            item.id
            for item in [node.left, *node.comparators]
            if isinstance(item, ast.Name)
        }
        if names & self._protected_assignments:
            return node
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if (
            not self.changed
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            self.changed = True
            tuned = (
                node.value + max(1, int(self.delta))
                if isinstance(node.value, int)
                else node.value + self.delta * 0.01
            )
            return ast.copy_location(
                ast.Constant(tuned), node
            )
        return node


class _TuneInfinityLiteral(ast.NodeTransformer):
    """保持算法结构与数值语义，只改变常见的可调初始化字面量表示。"""

    def __init__(self):
        self.changed = False

    def visit_Constant(self, node):
        if (
            not self.changed
            and isinstance(node.value, str)
            and node.value.casefold() == "inf"
        ):
            self.changed = True
            return ast.copy_location(ast.Constant("Infinity"), node)
        return node


def _tune_first_number(code: str, delta: int) -> str:
    try:
        tree = ast.parse(code)
        transformer = _TuneNumber(delta)
        tree = transformer.visit(tree)
        if not transformer.changed:
            infinity = _TuneInfinityLiteral()
            tree = infinity.visit(tree)
            if not infinity.changed:
                raise ValueError("no safely tunable literal")
        ast.fix_missing_locations(tree)
        return ast.unparse(tree) + "\n"
    except (SyntaxError, ValueError):
        return _simple_draft_code(delta)
