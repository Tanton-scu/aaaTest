"""实际执行 Candidate heuristic 的 Dataset evaluator。

与 reference PriEvO 一致，Candidate 必须执行
``run_tuners(file, budget, seed, maxlives)`` 并通过注入的 ``evaluate`` 消费唯一
configuration budget。这里不以 candidate code hash 代替算法执行。
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prievo_agent.domain.errors import (
    AlgorithmOOMError,
    AlgorithmTimeoutError,
    CandidateInterfaceError,
    CandidateRuntimeError,
    CandidateSyntaxError,
)
from prievo_agent.domain.models import EvaluationResult
from prievo_agent.security.heuristic_worker import (
    HeuristicSourceInterfaceError,
    HeuristicSourceSyntaxError,
    validate_heuristic_source,
)


@dataclass(frozen=True)
class HeuristicEvaluation:
    """子进程的权威 evaluation Trace；便于 Harness 做调用级断言。"""

    objective: float
    trajectory: tuple[float, ...]
    evaluated_configurations: tuple[tuple[Any, ...], ...]
    best_configuration: tuple[Any, ...]
    used_budget: int
    candidate_return: float
    calls: tuple[dict[str, Any], ...]


class ExecutableDatasetEvaluator:
    """Reference-compatible、跨平台尽力隔离的产品 evaluator。

    ``python -I``、最小环境、临时目录、wall timeout 和 POSIX rlimit 只是风险降低。
    Windows 无内置 rlimit，生产 Full Mode 应再使用低权限容器/微虚机、禁网和只读
    文件系统；本类不声称能抵御恶意 Python 的强安全沙箱。
    """

    def __init__(
        self,
        registry,
        *,
        timeout_seconds: float = 10.0,
        memory_limit_mb: int = 768,
        max_lives: int = 1000,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 300:
            raise ValueError("timeout_seconds 必须在 (0, 300] 秒")
        if memory_limit_mb < 64 or memory_limit_mb > 8192:
            raise ValueError("memory_limit_mb 必须在 [64, 8192]")
        if isinstance(max_lives, bool) or not isinstance(max_lives, int) or max_lives <= 0:
            raise ValueError("max_lives 必须是正整数")
        self.registry = registry
        self.timeout_seconds = float(timeout_seconds)
        self.memory_limit_mb = int(memory_limit_mb)
        self.max_lives = max_lives
        self.cache = {}
        self._worker_path = Path(__file__).resolve().parents[1] / "security" / "heuristic_worker.py"

    def evaluate(self, candidate, task) -> EvaluationResult:
        execution = self.execute(candidate, task)
        dataset = self._dataset(task.dataset_id)
        configuration = {
            name: execution.best_configuration[index]
            for index, name in enumerate(dataset.info.independent_columns)
        }
        configuration.update(
            {
                "dataset_id": task.dataset_id,
                "dataset_digest": dataset.info.digest,
            }
        )
        return EvaluationResult(
            id="result-{}".format(candidate.id),
            run_id=candidate.run_id,
            candidate_id=candidate.id,
            objective=execution.objective,
            trajectory=list(execution.trajectory),
            best_configuration=configuration,
            used_budget=execution.used_budget,
        )

    def execute(self, candidate, task) -> HeuristicEvaluation:
        if not getattr(task, "dataset_id", ""):
            raise CandidateInterfaceError("EvaluationTask 缺少 dataset_id")
        budget = getattr(task, "evaluation_budget", None)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise CandidateInterfaceError("evaluation_budget 必须是正整数")
        seed = getattr(task, "random_seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CandidateInterfaceError("random_seed 必须是整数")
        code = getattr(candidate, "code", None)
        try:
            validate_heuristic_source(code)
        except HeuristicSourceSyntaxError as exc:
            raise CandidateSyntaxError(str(exc)) from exc
        except HeuristicSourceInterfaceError as exc:
            raise CandidateInterfaceError(str(exc)) from exc

        dataset = self._dataset(task.dataset_id)
        if not dataset.configurations:
            raise CandidateInterfaceError("Dataset 没有 configuration")
        if budget > len(dataset.lookup):
            # Duplicate mapped configurations 不耗预算，无法凭空满足超过搜索空间的预算。
            raise CandidateInterfaceError(
                "evaluation_budget {} 超过 Dataset 唯一 configuration 数 {}".format(
                    budget, len(dataset.lookup)
                )
            )
        payload = {
            "code": code,
            "budget": budget,
            "seed": seed,
            "maxlives": self.max_lives,
            "dataset": _dataset_payload(dataset),
            "resource_limits": {
                "memory_bytes": self.memory_limit_mb * 1024 * 1024,
                "cpu_seconds": max(1, int(math.ceil(self.timeout_seconds))),
                "output_bytes": 2 * 1024 * 1024,
            },
        }
        response = self._run_worker(payload)
        return _parse_success(response, budget, len(dataset.info.independent_columns))

    def _dataset(self, dataset_id):
        dataset = self.cache.get(dataset_id)
        if dataset is None:
            dataset = self.registry.load(dataset_id)
            self.cache[dataset_id] = dataset
        return dataset

    def _run_worker(self, payload):
        if not self._worker_path.is_file():
            raise CandidateRuntimeError("heuristic_worker.py 不存在")
        with tempfile.TemporaryDirectory(prefix="prievo-heuristic-") as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "output.json"
            input_path.write_text(
                json.dumps(payload, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            environment = _minimal_environment()
            kwargs = {
                "args": [
                    sys.executable,
                    "-I",
                    str(self._worker_path),
                    str(input_path),
                    str(output_path),
                ],
                "cwd": str(root),
                "env": environment,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "timeout": self.timeout_seconds,
                "check": False,
            }
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                kwargs["start_new_session"] = True
            try:
                completed = subprocess.run(**kwargs)
            except subprocess.TimeoutExpired as exc:
                raise AlgorithmTimeoutError(
                    "Candidate algorithm 超过 {:.3f}s wall timeout".format(
                        self.timeout_seconds
                    )
                ) from exc

            if output_path.exists() and output_path.stat().st_size <= 2 * 1024 * 1024:
                try:
                    response = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise CandidateInterfaceError(
                        "Candidate worker 输出不是合法 JSON"
                    ) from exc
                if not isinstance(response, dict):
                    raise CandidateInterfaceError("Candidate worker 输出必须是 object")
                if response.get("status") == "ok" and completed.returncode == 0:
                    return response
                _raise_worker_failure(response, completed.returncode)

            _raise_missing_output(completed.returncode)


def _dataset_payload(dataset):
    features = list(dataset.info.independent_columns)
    configurations = [list(item) for item in dataset.configurations]
    objectives = [float(item) for item in dataset.objectives]
    if any(not math.isfinite(item) for item in objectives):
        raise CandidateInterfaceError("Dataset objective 包含非有限值")
    independent_set = []
    for index in range(len(features)):
        values = []
        for configuration in configurations:
            value = configuration[index]
            if value not in values:
                values.append(value)
        independent_set.append(sorted(values, key=lambda item: (type(item).__name__, repr(item))))
    return {
        "name": dataset.info.id,
        "features": features,
        "independent_set": independent_set,
        "configurations": configurations,
        "objectives": objectives,
        "dataset_digest": dataset.info.digest,
    }


def _parse_success(response, budget, dimensions):
    result = response.get("result")
    if not isinstance(result, dict):
        raise CandidateInterfaceError("Candidate worker 缺少结构化 result")
    try:
        objective = float(result["objective"])
        trajectory = tuple(float(item) for item in result["trajectory"])
        evaluated = tuple(
            tuple(item) for item in result["evaluated_configurations"]
        )
        best = tuple(result["best_configuration"])
        used_budget = int(result["used_budget"])
        candidate_return = float(result["candidate_return"])
        calls = tuple(dict(item) for item in result["calls"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateInterfaceError("Candidate worker result schema 无效") from exc
    if not math.isfinite(objective) or not math.isfinite(candidate_return):
        raise CandidateInterfaceError("Candidate objective/return 必须是有限值")
    if used_budget <= 0 or used_budget > budget:
        raise CandidateInterfaceError("Candidate used_budget 超出 (0, budget]")
    if len(trajectory) != used_budget or len(evaluated) != used_budget:
        raise CandidateInterfaceError(
            "trajectory/evaluated_configurations 必须逐项覆盖 unique budget 1..used_budget"
        )
    if len(best) != dimensions or any(len(item) != dimensions for item in evaluated):
        raise CandidateInterfaceError("Candidate configuration 维度不符合 Dataset")
    if not math.isclose(objective, candidate_return, rel_tol=1e-12, abs_tol=1e-12):
        raise CandidateInterfaceError("Candidate return 与 objective 不一致")
    if not math.isclose(objective, min(trajectory), rel_tol=1e-12, abs_tol=1e-12):
        raise CandidateInterfaceError("Candidate objective 与 trajectory best 不一致")
    if any(
        later > earlier + 1e-12
        for earlier, later in zip(trajectory, trajectory[1:])
    ):
        raise CandidateInterfaceError("最小化 trajectory 不得变差")
    return HeuristicEvaluation(
        objective,
        trajectory,
        evaluated,
        best,
        used_budget,
        candidate_return,
        calls,
    )


def _raise_worker_failure(response, return_code):
    status = str(response.get("status", ""))
    message = str(response.get("message", "") or "Candidate worker 失败")
    error_type = str(response.get("error_type", "") or "")
    detail = "{}：{}（退出码 {}）".format(status, message, return_code)
    if error_type:
        detail += " [{}]".format(error_type)
    if status == "syntax_error":
        raise CandidateSyntaxError(detail)
    if status == "interface_error":
        raise CandidateInterfaceError(detail)
    if status == "algorithm_oom":
        raise AlgorithmOOMError(detail)
    raise CandidateRuntimeError(detail)


def _raise_missing_output(return_code):
    oom_codes = {137, -9, -1073741801, 3221225495}
    if return_code in oom_codes:
        raise AlgorithmOOMError(
            "Candidate worker 被 SIGKILL/OOM-like 状态终止（退出码 {}）".format(
                return_code
            )
        )
    sigxcpu = getattr(signal, "SIGXCPU", None)
    if sigxcpu is not None and return_code == -int(sigxcpu):
        raise AlgorithmTimeoutError("Candidate worker 达到 CPU time limit")
    raise CandidateRuntimeError(
        "Candidate worker 未产生输出（退出码 {}）；可能为异常退出或依赖加载失败".format(
            return_code
        )
    )


def _minimal_environment():
    environment = {
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    # Windows Python/stdlib 初始化需要 SYSTEMROOT；不透传 API key、数据库密码等。
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


__all__ = ["ExecutableDatasetEvaluator", "HeuristicEvaluation"]

