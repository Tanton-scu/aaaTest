from __future__ import annotations

import json
import tempfile
from pathlib import Path

from prievo_agent.runtime.queue_harness import QueueReliabilityHarness


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="prievo-queue-") as temp:
        report = QueueReliabilityHarness().run(Path(temp))
    print(
        json.dumps(
            {"message": "异步评价队列 Harness 通过", **report.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
