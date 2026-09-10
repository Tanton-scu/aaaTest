"""生成 Original Prior 与产品 evaluator 的可复现静态兼容矩阵。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prievo_agent.knowledge.prior.compatibility import load_prior_compatibility_report


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 original prior 静态执行兼容性")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("assets/prior/prior_population.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/prior_compatibility.json")
    )
    args = parser.parse_args()
    report = load_prior_compatibility_report(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Prior 兼容矩阵：总数 {total}，可执行 {supported_count}，"
        "不兼容 {unsupported_count}；报告 {path}".format(
            path=args.output, **report
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
