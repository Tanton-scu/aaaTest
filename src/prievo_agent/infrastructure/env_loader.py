from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    """加载本地 .env；不依赖 python-dotenv，避免给本地启动增加额外依赖。

    规则刻意保持简单：支持 KEY=VALUE、可选 export 前缀、单/双引号包裹值；
    空行和 # 注释会被忽略。默认不覆盖已经存在的系统环境变量。
    """

    env_path = Path(path)
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not _valid_env_key(key):
            continue
        parsed = _parse_env_value(value)
        if override or key not in os.environ:
            os.environ[key] = parsed
            loaded[key] = parsed
    return loaded


def _valid_env_key(key: str) -> bool:
    if not key[0].isalpha() and key[0] != "_":
        return False
    return all(ch.isalnum() or ch == "_" for ch in key)


def _parse_env_value(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    if (
        len(stripped) >= 2
        and stripped[0] == stripped[-1]
        and stripped[0] in {"'", '"'}
    ):
        return stripped[1:-1]
    return stripped
