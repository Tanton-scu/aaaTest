"""Reference-compatible Candidate heuristic 子进程。

该进程只依赖 Python 标准库。父进程使用 ``python -I``、最小环境变量、临时工作
目录、wall timeout，并在 POSIX 尽力设置 rlimit。它是风险降低层，不是 OS 级安全
沙箱：生产环境仍应把 Worker 放入无凭据、无网络、只读根文件系统的容器/微虚机。
"""

from __future__ import annotations

import ast
import builtins
import errno
import json
import math
import sys
import traceback
import types
from pathlib import Path


class HeuristicSourceSyntaxError(ValueError):
    pass


class HeuristicSourceInterfaceError(ValueError):
    pass


class _CandidateContractViolation(RuntimeError):
    pass


_ALLOWED_IMPORT_ROOTS = {
    # Reference heuristics 最常见的纯计算依赖；缺失的第三方包在运行期明确分类。
    "ConfigSpace",
    "bisect",
    "collections",
    "copy",
    "functools",
    "heapq",
    "itertools",
    "math",
    "numpy",
    "operator",
    "random",
    "scipy",
    "sklearn",
    "statistics",
    "time",
    # 注入的 reference-compatible 模块。
    "util",
}

_FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

_FORBIDDEN_ATTRIBUTES = {
    "FileHandler",
    "HTTPHandler",
    "SMTPHandler",
    "SocketHandler",
    "WatchedFileHandler",
    "fork",
    "popen",
    "spawn",
    "system",
}


def validate_heuristic_source(code: str) -> None:
    """验证语法、入口签名和静态风险边界，不导入或执行 Candidate。"""

    if not isinstance(code, str) or not code.strip():
        raise HeuristicSourceSyntaxError("Candidate code 不能为空")
    if len(code.encode("utf-8")) > 1_000_000:
        raise HeuristicSourceInterfaceError("Candidate code 超过 1 MiB 上限")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise HeuristicSourceSyntaxError("Candidate Python 语法无效") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 10_000:
        raise HeuristicSourceInterfaceError("Candidate AST 超过 10000 节点上限")

    entries = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_tuners"
    ]
    if len(entries) != 1 or isinstance(entries[0], ast.AsyncFunctionDef):
        raise HeuristicSourceInterfaceError(
            "Candidate 必须定义唯一同步 run_tuners 函数"
        )
    function = entries[0]
    parameters = tuple(item.arg for item in function.args.args)
    if (
        parameters != ("file", "budget", "seed", "maxlives")
        or function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.defaults
        or function.args.kw_defaults
    ):
        raise HeuristicSourceInterfaceError(
            "run_tuners 签名必须严格为 (file, budget, seed, maxlives)"
        )

    for node in nodes:
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise HeuristicSourceInterfaceError("Candidate 禁止 global/nonlocal")
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise HeuristicSourceInterfaceError("Candidate 禁止相对 import")
            _validate_import(node.module or "")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise HeuristicSourceInterfaceError("Candidate 禁止 dunder 名称")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise HeuristicSourceInterfaceError("Candidate 禁止私有/dunder 属性访问")
            if node.attr in _FORBIDDEN_ATTRIBUTES:
                raise HeuristicSourceInterfaceError(
                    "Candidate 禁止高风险属性：{}".format(node.attr)
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                raise HeuristicSourceInterfaceError(
                    "Candidate 禁止调用：{}".format(node.func.id)
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _FORBIDDEN_ATTRIBUTES
            ):
                raise HeuristicSourceInterfaceError(
                    "Candidate 禁止调用高风险属性：{}".format(node.func.attr)
                )


def _validate_import(name: str) -> None:
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_IMPORT_ROOTS:
        raise HeuristicSourceInterfaceError(
            "Candidate import 未获批准：{}".format(name or "<empty>")
        )
    if root == "util" and name not in {"util", "util.Evaluate"}:
        raise HeuristicSourceInterfaceError(
            "Candidate 只允许 import util.Evaluate"
        )


class _DatasetFile:
    def __init__(self, payload):
        self.name = str(payload["name"])
        self.independent_set = [list(values) for values in payload["independent_set"]]
        self.features = list(payload["features"])
        configurations = [tuple(item) for item in payload["configurations"]]
        objectives = [float(item) for item in payload["objectives"]]
        if not configurations or len(configurations) != len(objectives):
            raise _CandidateContractViolation("Dataset configurations/objectives 无效")
        if any(len(item) != len(self.features) for item in configurations):
            raise _CandidateContractViolation("Dataset configuration 维度不一致")
        self._dimension = len(self.features)
        self._lookup = types.MappingProxyType(dict(zip(configurations, objectives)))
        # 保留 reference 属性，但 Candidate 无法改写权威 lookup。
        self.dict_search = self._lookup
        self._configurations = configurations
        self._objectives = objectives
        self._parameter_types = [
            _infer_parameter_type([item[index] for item in configurations])
            for index in range(len(self.features))
        ]
        self._numeric_ranges = [
            _numeric_range([item[index] for item in configurations])
            if self._parameter_types[index] == "numeric"
            else None
            for index in range(len(self.features))
        ]

    def nearest(self, generated_config):
        requested = _plain_config(generated_config, self._dimension)
        requested_tuple = tuple(requested)
        if requested_tuple in self._lookup:
            return float(self._lookup[requested_tuple]), list(requested_tuple), True

        best_index = None
        best_distance = math.inf
        for index, known in enumerate(self._configurations):
            distance = self._distance(requested, known)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None or not math.isfinite(best_distance):
            raise _CandidateContractViolation(
                "generated_config 无法映射到 Dataset configuration"
            )
        return (
            float(self._objectives[best_index]),
            list(self._configurations[best_index]),
            False,
        )

    def _distance(self, requested, known):
        numeric_squared = 0.0
        non_numeric = 0.0
        for index, parameter_type in enumerate(self._parameter_types):
            left = requested[index]
            right = known[index]
            if parameter_type == "numeric":
                if (
                    isinstance(left, bool)
                    or not isinstance(left, (int, float))
                    or not math.isfinite(float(left))
                ):
                    return math.inf
                low, high = self._numeric_ranges[index]
                scale = high - low
                delta = 0.0 if scale == 0 else (float(left) - float(right)) / scale
                numeric_squared += delta * delta
            elif left != right:
                non_numeric += 1.0
        # 与 reference QueryDataset 一致：归一化连续欧氏距离 + categorical Hamming。
        return math.sqrt(numeric_squared) + non_numeric


class _EvaluationTrace:
    def __init__(self, dataset: _DatasetFile, budget: int):
        self.dataset = dataset
        self.budget = budget
        self.used_budget = 0
        self.consecutive_no_improve = 0
        self.best_result = math.inf
        self.best_configuration = None
        self.history = {}
        self.trajectory = []
        self.calls = []

    def evaluate(
        self,
        used_budget,
        consecutive_no_improve,
        history_configs,
        best_result,
        generated_config,
    ):
        # 参数保留 reference 签名；权威预算/历史由注入层维护，Candidate 不能伪造。
        if isinstance(used_budget, bool) or not isinstance(used_budget, int):
            raise _CandidateContractViolation("evaluate used_budget 必须是整数")
        if (
            isinstance(consecutive_no_improve, bool)
            or not isinstance(consecutive_no_improve, int)
        ):
            raise _CandidateContractViolation(
                "evaluate consecutive_no_improve 必须是整数"
            )
        if not isinstance(history_configs, dict):
            raise _CandidateContractViolation("evaluate history_configs 必须是 dict")

        score, mapped, exact = self.dataset.nearest(generated_config)
        mapped_key = tuple(mapped)
        duplicate = mapped_key in self.history
        if duplicate:
            self.consecutive_no_improve += 1
        else:
            if self.used_budget >= self.budget:
                raise _CandidateContractViolation(
                    "Candidate 请求的唯一 evaluation 超过 budget"
                )
            self.history[mapped_key] = score
            self.used_budget += 1
            if score < self.best_result:
                self.best_result = score
                self.best_configuration = list(mapped)
                self.consecutive_no_improve = 0
            else:
                self.consecutive_no_improve += 1
            self.trajectory.append(float(self.best_result))

        self.calls.append(
            {
                "call_index": len(self.calls) + 1,
                "requested_config": _plain_config(
                    generated_config, self.dataset._dimension
                ),
                "mapped_config": list(mapped),
                "score": score,
                "exact_match": exact,
                "duplicate": duplicate,
                "used_budget": self.used_budget,
                "best_result": self.best_result,
            }
        )
        return (
            self.used_budget,
            self.consecutive_no_improve,
            dict(self.history),
            self.best_result,
            score,
            list(mapped),
        )

    def checked_result(self, candidate_result):
        if self.used_budget <= 0:
            raise _CandidateContractViolation(
                "Candidate 未调用注入的 evaluate，不能形成真实评价"
            )
        if isinstance(candidate_result, bool) or not isinstance(
            candidate_result, (int, float)
        ):
            # 支持 numpy scalar 等提供 item() 的纯数值对象。
            item = getattr(candidate_result, "item", None)
            if not callable(item):
                raise _CandidateContractViolation(
                    "run_tuners 必须返回 evaluate 产生的数值 best_result"
                )
            candidate_result = item()
        if isinstance(candidate_result, bool) or not isinstance(
            candidate_result, (int, float)
        ):
            raise _CandidateContractViolation("run_tuners 返回值不是数值")
        candidate_result = float(candidate_result)
        if not math.isfinite(candidate_result):
            raise _CandidateContractViolation("run_tuners 返回值必须是有限数值")
        if not math.isclose(
            candidate_result, self.best_result, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise _CandidateContractViolation(
                "run_tuners 返回 best_result 与注入 evaluate Trace 不一致"
            )
        if len(self.trajectory) != self.used_budget:
            raise _CandidateContractViolation("trajectory 与 unique used_budget 不一致")
        return {
            "objective": float(self.best_result),
            "trajectory": list(self.trajectory),
            "evaluated_configurations": [list(item) for item in self.history],
            "best_configuration": list(self.best_configuration),
            "used_budget": self.used_budget,
            "candidate_return": candidate_result,
            "calls": list(self.calls),
        }


def _infer_parameter_type(values):
    if values and all(isinstance(item, bool) for item in values):
        return "boolean"
    if values and all(
        not isinstance(item, bool) and isinstance(item, (int, float)) for item in values
    ):
        return "numeric"
    return "categorical"


def _numeric_range(values):
    numbers = [float(item) for item in values]
    return min(numbers), max(numbers)


def _plain_config(value, expected_size):
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, (list, tuple)):
        # numpy arrays and similar objects may expose a bounded tolist projection.
        to_list = getattr(value, "tolist", None)
        if not callable(to_list):
            raise _CandidateContractViolation(
                "generated_config 必须是与 file.features 对齐的 sequence"
            )
        value = to_list()
    if not isinstance(value, (list, tuple)) or len(value) != expected_size:
        raise _CandidateContractViolation(
            "generated_config 维度必须等于 file.features"
        )
    result = []
    for item in value:
        scalar = getattr(item, "item", None)
        if callable(scalar):
            item = scalar()
        if isinstance(item, float) and not math.isfinite(item):
            raise _CandidateContractViolation("generated_config 包含非有限数值")
        if not isinstance(item, (str, int, float, bool)) and item is not None:
            raise _CandidateContractViolation(
                "generated_config 包含不可序列化参数类型"
            )
        result.append(item)
    return result


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("Candidate 禁止相对 import")
    _validate_import(name)
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins():
    names = {
        "ArithmeticError",
        "AssertionError",
        "Exception",
        "IndexError",
        "KeyError",
        "MemoryError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
        "__build_class__",
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
    }
    result = {name: getattr(builtins, name) for name in names}
    result["__import__"] = _safe_import
    return result


def _install_evaluate_module(evaluate):
    package = types.ModuleType("util")
    package.__path__ = []
    module = types.ModuleType("util.Evaluate")
    module.evaluate = evaluate
    package.Evaluate = module
    sys.modules["util"] = package
    sys.modules["util.Evaluate"] = module


def _apply_resource_limits(limits):
    """POSIX 尽力限制资源；Windows 由父进程 wall timeout/容器边界降级保护。"""

    try:
        import resource
    except ImportError:
        return
    memory_bytes = int(limits.get("memory_bytes", 0) or 0)
    cpu_seconds = int(limits.get("cpu_seconds", 0) or 0)
    output_bytes = int(limits.get("output_bytes", 0) or 0)
    try:
        if memory_bytes > 0:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        if cpu_seconds > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        if output_bytes > 0:
            resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except (OSError, ValueError):
        # 平台/容器不支持某个 rlimit 时继续依赖父进程 timeout；不是强沙箱声明。
        return


def _run(payload):
    validate_heuristic_source(payload["code"])
    dataset = _DatasetFile(payload["dataset"])
    budget = int(payload["budget"])
    maxlives = int(payload["maxlives"])
    seed = int(payload["seed"])
    if budget <= 0 or maxlives <= 0:
        raise _CandidateContractViolation("budget/maxlives 必须是正整数")
    trace = _EvaluationTrace(dataset, budget)
    _install_evaluate_module(trace.evaluate)
    namespace = {
        "__builtins__": _safe_builtins(),
        "__name__": "candidate_heuristic",
        "evaluate": trace.evaluate,
    }
    compiled = compile(payload["code"], "<candidate>", "exec")
    exec(compiled, namespace, namespace)
    run_tuners = namespace.get("run_tuners")
    if not callable(run_tuners):
        raise _CandidateContractViolation("run_tuners 未成功加载")
    candidate_result = run_tuners(dataset, budget, seed, maxlives)
    return trace.checked_result(candidate_result)


def _write(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        return 64
    input_path, output_path = args
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        _apply_resource_limits(payload.get("resource_limits", {}))
        result = _run(payload)
        _write(output_path, {"status": "ok", "result": result})
        return 0
    except HeuristicSourceSyntaxError as exc:
        _write(output_path, {"status": "syntax_error", "message": str(exc)})
        return 20
    except (HeuristicSourceInterfaceError, _CandidateContractViolation) as exc:
        _write(output_path, {"status": "interface_error", "message": str(exc)})
        return 21
    except MemoryError as exc:
        _write(output_path, {"status": "algorithm_oom", "message": str(exc)})
        return 22
    except OSError as exc:
        if exc.errno == errno.ENOMEM:
            _write(output_path, {"status": "algorithm_oom", "message": str(exc)})
            return 22
        _write(
            output_path,
            {
                "status": "runtime_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8)[-4000:],
            },
        )
        return 23
    except BaseException as exc:
        _write(
            output_path,
            {
                "status": "runtime_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8)[-4000:],
            },
        )
        return 23


if __name__ == "__main__":
    raise SystemExit(main())
