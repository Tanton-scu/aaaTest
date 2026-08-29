from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Tuple

from prievo_agent.domain.errors import CandidateInvalidError


@dataclass(frozen=True)
class ValidationReport:
    function_name: str
    parameters: Tuple[str, ...]
    node_count: int


class CandidateCodeValidator:
    """V1 allowlist：单一 run_tuners 函数、无 import/危险 builtin/dunder。"""

    required_parameters = ("file", "budget", "seed", "maxlives")
    forbidden_calls = {
        "__import__", "breakpoint", "compile", "eval", "exec", "globals",
        "input", "locals", "open", "setattr", "delattr", "getattr",
    }

    def validate(self, code: str) -> ValidationReport:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise CandidateInvalidError("candidate Python 语法无效") from exc
        if len(list(ast.walk(tree))) > 500:
            raise CandidateInvalidError("candidate AST 超过 V1 大小上限")
        functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)]
        if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != "run_tuners":
            raise CandidateInvalidError("candidate 必须只定义一个 run_tuners 函数")
        function = functions[0]
        parameters = tuple(item.arg for item in function.args.args)
        if (
            parameters != self.required_parameters
            or function.args.vararg is not None
            or function.args.kwarg is not None
            or function.args.kwonlyargs
        ):
            raise CandidateInvalidError(
                "run_tuners 签名必须是 (file, budget, seed, maxlives)"
            )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
                raise CandidateInvalidError("candidate 包含禁止的 import/global 操作")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise CandidateInvalidError("candidate 包含禁止的 dunder 属性访问")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise CandidateInvalidError("candidate 包含禁止的 dunder 名称")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.forbidden_calls:
                    raise CandidateInvalidError("candidate 调用了禁止函数：{}".format(node.func.id))
        return ValidationReport("run_tuners", parameters, len(list(ast.walk(tree))))
