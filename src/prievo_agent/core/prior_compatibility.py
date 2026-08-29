"""Original-prior 与受控 evaluator 的静态兼容性审计。

这里的 ``SUPPORTED`` 只表示代码通过当前静态执行契约，可以进入受控 evaluator
尝试执行；它不承诺某个 heuristic 一定适合任意 Dataset，也不执行或改写 prior。
``prior_population.json`` 始终是不可变的 prompt/evidence 来源。
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from prievo_agent.security.heuristic_worker import (
    HeuristicSourceInterfaceError,
    HeuristicSourceSyntaxError,
    validate_heuristic_source,
)


PRIOR_COMPATIBILITY_POLICY_VERSION = "prior-seed-static-v1"
PRIOR_COMPATIBILITY_REPORT_SCHEMA = "prior-compatibility-report-v1"

# 这是产品镜像实际承诺给 Candidate worker 的 import surface。科学计算依赖由
# ``.[research]`` 安装；util.Evaluate 由 worker 注入。不得根据开发机偶然安装的包
# 动态改变结论，否则同一份 prior 在不同机器上会得到不同矩阵。
_PRODUCT_IMPORT_ROOTS = frozenset(
    {
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
        "util",
    }
)


@dataclass(frozen=True)
class PriorCompatibilityAssessment:
    status: str
    reason_code: str
    reason: str
    code_sha256: str
    import_roots: tuple[str, ...]
    blocked_import_roots: tuple[str, ...]
    policy_version: str = PRIOR_COMPATIBILITY_POLICY_VERSION

    @property
    def supported(self) -> bool:
        return self.status == "SUPPORTED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "code_sha256": self.code_sha256,
            "import_roots": list(self.import_roots),
            "blocked_import_roots": list(self.blocked_import_roots),
            "policy_version": self.policy_version,
        }


def assess_prior_code(code: str) -> PriorCompatibilityAssessment:
    """只读、无 import/exec 的静态审计，返回跨机器稳定的状态和原因码。"""

    source = code if isinstance(code, str) else ""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    tree = _parse_optional(source)
    import_roots = _import_roots(tree)
    blocked_roots = tuple(
        root for root in import_roots if root not in _PRODUCT_IMPORT_ROOTS
    )
    try:
        validate_heuristic_source(source)
    except HeuristicSourceSyntaxError:
        return _assessment(
            "UNSUPPORTED",
            "SOURCE_SYNTAX_INVALID",
            "源代码无法解析，不能作为受控 evaluator seed 执行",
            digest,
            import_roots,
            blocked_roots,
        )
    except HeuristicSourceInterfaceError:
        reason_code, reason = _interface_reason(source, tree, blocked_roots)
        return _assessment(
            "UNSUPPORTED",
            reason_code,
            reason,
            digest,
            import_roots,
            blocked_roots,
        )

    # 防止 worker 将来放宽 allowlist 后，开发机上的偶然依赖被误认为产品可用。
    if blocked_roots:
        return _assessment(
            "UNSUPPORTED",
            "IMPORT_POLICY_BLOCKED",
            "包含产品 Candidate worker 未承诺的 import root：{}".format(
                ", ".join(blocked_roots)
            ),
            digest,
            import_roots,
            blocked_roots,
        )
    return _assessment(
        "SUPPORTED",
        "STATIC_CONTRACT_SATISFIED",
        "通过受控 evaluator 静态源代码契约，可尝试作为 original-prior seed 执行",
        digest,
        import_roots,
        (),
    )


def build_prior_compatibility_report(
    records: Sequence[Mapping[str, Any]],
    *,
    source_sha256: str,
    source_path: str = "resources/prior_knowledge/prior_population.json",
) -> dict[str, Any]:
    """为 JSON prior records 生成不含时间/环境探测的可复现矩阵。"""

    matrix = []
    for index, record in enumerate(records):
        assessment = assess_prior_code(str(record.get("code", "")))
        row = {
            "index": index,
            "name": str(record.get("name", "")),
        }
        row.update(assessment.to_dict())
        matrix.append(row)
    supported = sum(row["status"] == "SUPPORTED" for row in matrix)
    return {
        "schema_version": PRIOR_COMPATIBILITY_REPORT_SCHEMA,
        "policy_version": PRIOR_COMPATIBILITY_POLICY_VERSION,
        "source_path": source_path.replace("\\", "/"),
        "source_sha256": source_sha256,
        "total": len(matrix),
        "supported_count": supported,
        "unsupported_count": len(matrix) - supported,
        "semantics": {
            "SUPPORTED": (
                "通过静态执行契约，可进入受控 evaluator 尝试执行；不等于对任意 Dataset 动态成功"
            ),
            "UNSUPPORTED": (
                "仍保留为 immutable prompt/evidence，但不得物化为可执行 initial seed"
            ),
        },
        "records": matrix,
    }


def load_prior_compatibility_report(
    source: Path,
    *,
    source_path: str = "resources/prior_knowledge/prior_population.json",
) -> dict[str, Any]:
    raw = Path(source).read_bytes()
    records = json.loads(raw.decode("utf-8"))
    if not isinstance(records, list):
        raise ValueError("prior_population.json 顶层必须是 array")
    return build_prior_compatibility_report(
        records,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_path=source_path,
    )


def optimizer_compatibility_rows(optimizers: Iterable[Any]) -> list[dict[str, Any]]:
    """为一次 instance-specific prior 输出精简且可审计的兼容性行。"""

    rows = []
    for optimizer in optimizers:
        assessment = assess_prior_code(getattr(optimizer, "code", ""))
        row = {
            "name": str(getattr(optimizer, "name", "")),
            "rank": str(getattr(optimizer, "rank", "")),
            "source_instance": str(getattr(optimizer, "source_instance", "")),
        }
        row.update(assessment.to_dict())
        rows.append(row)
    return rows


def _parse_optional(code: str) -> ast.Module | None:
    try:
        return ast.parse(code, mode="exec")
    except (SyntaxError, TypeError, ValueError):
        return None


def _import_roots(tree: ast.Module | None) -> tuple[str, ...]:
    if tree is None:
        return ()
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return tuple(sorted(roots))


def _interface_reason(
    code: str,
    tree: ast.Module | None,
    blocked_roots: tuple[str, ...],
) -> tuple[str, str]:
    if len(code.encode("utf-8")) > 1_000_000:
        return "SOURCE_SIZE_LIMIT", "源代码超过受控 evaluator 的 1 MiB 上限"
    if tree is None:
        return "SOURCE_SYNTAX_INVALID", "源代码无法解析，不能作为 seed 执行"
    entries = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_tuners"
    ]
    if len(entries) != 1 or not _valid_entrypoint(entries[0]):
        return (
            "ENTRYPOINT_CONTRACT_MISMATCH",
            "run_tuners 必须是唯一同步顶层函数，签名严格为 (file, budget, seed, maxlives)",
        )
    if any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(tree)):
        return (
            "STATE_SCOPE_POLICY_BLOCKED",
            "受控 evaluator 禁止 Candidate 使用 global/nonlocal 状态",
        )
    if blocked_roots:
        return (
            "IMPORT_POLICY_BLOCKED",
            "包含产品 Candidate worker 未承诺的 import root：{}".format(
                ", ".join(blocked_roots)
            ),
        )
    return (
        "SOURCE_SECURITY_POLICY_BLOCKED",
        "源代码违反受控 evaluator 的静态安全边界",
    )


def _valid_entrypoint(node: ast.AST) -> bool:
    if not isinstance(node, ast.FunctionDef):
        return False
    arguments = node.args
    return (
        tuple(item.arg for item in arguments.args)
        == ("file", "budget", "seed", "maxlives")
        and not arguments.posonlyargs
        and not arguments.kwonlyargs
        and arguments.vararg is None
        and arguments.kwarg is None
        and not arguments.defaults
        and not arguments.kw_defaults
    )


def _assessment(
    status: str,
    reason_code: str,
    reason: str,
    digest: str,
    imports: tuple[str, ...],
    blocked: tuple[str, ...],
) -> PriorCompatibilityAssessment:
    return PriorCompatibilityAssessment(
        status=status,
        reason_code=reason_code,
        reason=reason,
        code_sha256=digest,
        import_roots=imports,
        blocked_import_roots=blocked,
    )


__all__ = [
    "PRIOR_COMPATIBILITY_POLICY_VERSION",
    "PRIOR_COMPATIBILITY_REPORT_SCHEMA",
    "PriorCompatibilityAssessment",
    "assess_prior_code",
    "build_prior_compatibility_report",
    "load_prior_compatibility_report",
    "optimizer_compatibility_rows",
]
