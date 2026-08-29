from __future__ import annotations

import json

from prievo_agent.runtime.prior_harness import PriorRetrievalHarness


def main() -> int:
    report = PriorRetrievalHarness().run()
    print(
        json.dumps(
            {"message": "Instance-specific Prior Retrieval Harness 通过", **report.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
