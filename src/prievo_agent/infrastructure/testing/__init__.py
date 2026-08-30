"""测试与工程 harness 使用的确定性 adapter。

这些 adapter 不属于生产运行入口：生产运行使用 MySQL、真实 LLM 和独立评估
Worker；这里的 FakeLLM / SQLite store 只服务单元测试、故障注入和可复现实验。
"""

from .fake_evaluator import FakeEvaluator
from .fake_llm import FakeLLM
from .scripted_fake_llm import ScriptedFakeLLM
from .sqlite_store import SQLiteRuntimeStore

__all__ = [
    "FakeEvaluator",
    "FakeLLM",
    "ScriptedFakeLLM",
    "SQLiteRuntimeStore",
]
