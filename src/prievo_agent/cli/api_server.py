from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from prievo_agent.api import create_app
from prievo_agent.infrastructure.env_loader import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 PriEvO-Agent API 与运行面板")
    parser.add_argument("--root", type=Path, default=Path(".prievo-runtime"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="本地配置文件；默认读取当前目录 .env，已存在的系统环境变量优先",
    )
    args = parser.parse_args()
    loaded = load_env_file(args.env_file)
    if loaded:
        safe_keys = [
            key
            for key in sorted(loaded)
            if not key.endswith("KEY") and "PASSWORD" not in key and "SECRET" not in key
        ]
        if safe_keys:
            print("已加载本地配置：{}（敏感值不打印）".format(", ".join(safe_keys)))
        else:
            print("已加载本地配置（仅包含敏感值，已隐藏）")
    print("PriEvO-Agent 服务启动：http://{}:{}".format(args.host, args.port))
    uvicorn.run(create_app(args.root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
