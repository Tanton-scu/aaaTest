from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from prievo_agent.domain.errors import CandidateInvalidError, EvaluationTimeoutError

from .candidate_validation import CandidateCodeValidator


_RUNNER = """import importlib.util, json, pathlib
root = pathlib.Path.cwd()
spec = importlib.util.spec_from_file_location('candidate', root / 'candidate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
payload = json.loads((root / 'input.json').read_text(encoding='utf-8'))
result = module.run_tuners(payload['file'], payload['budget'], payload['seed'], payload['maxlives'])
(root / 'output.json').write_text(json.dumps({'result': result}, ensure_ascii=False), encoding='utf-8')
"""


class BoundedCandidateExecutor:
    """受限 subprocess runner；它是风险降低层，不是强安全沙箱。"""

    def __init__(self, timeout_seconds: float = 2.0) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 30:
            raise ValueError("timeout_seconds 必须在 (0, 30]")
        self.timeout_seconds = timeout_seconds
        self.validator = CandidateCodeValidator()

    def execute(self, code: str, payload: Dict[str, Any]) -> Any:
        self.validator.validate(code)
        required = {"file", "budget", "seed", "maxlives"}
        if set(payload) != required:
            raise CandidateInvalidError("candidate 输入字段不符合执行契约")
        with tempfile.TemporaryDirectory(prefix="prievo-candidate-") as directory:
            root = Path(directory)
            (root / "candidate.py").write_text(code, encoding="utf-8")
            (root / "runner.py").write_text(_RUNNER, encoding="utf-8")
            (root / "input.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            environment = {
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
            }
            if os.name == "nt" and "SYSTEMROOT" in os.environ:
                environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(root / "runner.py")],
                    cwd=str(root),
                    env=environment,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise EvaluationTimeoutError("candidate 执行超过时间上限") from exc
            if completed.returncode != 0:
                raise CandidateInvalidError(
                    "candidate 执行失败（退出码 {}）".format(completed.returncode)
                )
            output = root / "output.json"
            if not output.exists() or output.stat().st_size > 1_000_000:
                raise CandidateInvalidError("candidate 未产生有效的受控输出")
            try:
                return json.loads(output.read_text(encoding="utf-8"))["result"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise CandidateInvalidError("candidate 输出不是合法 JSON 契约") from exc
