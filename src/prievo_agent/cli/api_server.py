from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from prievo_agent.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 PriEvO-Agent API 与运行面板")
    parser.add_argument("--root", type=Path, default=Path(".prievo-runtime"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("PriEvO-Agent 服务启动：http://{}:{}".format(args.host, args.port))
    uvicorn.run(create_app(args.root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
