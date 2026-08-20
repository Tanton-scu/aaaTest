from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from prievo_agent.runtime.checkpoint_harness import CheckpointRecoveryHarness


def main() -> int:
    parser = argparse.ArgumentParser(description="PriEvO-Agent checkpoint 恢复演示")
    parser.add_argument("--output", help="保留 SQLite/artifact 证据的目录")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        report = CheckpointRecoveryHarness(project_root).run(output)
        evidence_path = str(output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="prievo-checkpoint-") as temp:
            report = CheckpointRecoveryHarness(project_root).run(Path(temp))
            evidence_path = "临时目录（运行结束后已清理）"
    print(
        json.dumps(
            {
                "message": "Checkpoint 跨进程恢复验证通过",
                "evidence_path": evidence_path,
                **report.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
