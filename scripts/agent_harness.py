"""运行 Agent Engineering Harness 并生成可审计 JSON 报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prievo_agent.devtools.harness.agent import AgentEngineeringHarness


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="运行 PriEvO Agent 工程 Harness")
    parser.add_argument(
        "--report",
        type=Path,
        default=project / "reports" / "agent_harness.json",
        help="JSON 报告路径（默认 reports/agent_harness.json）",
    )
    parser.add_argument(
        "--collect-all",
        action="store_true",
        help="即使存在失败场景也收集全部结果并以 JSON 输出",
    )
    args = parser.parse_args()

    report = AgentEngineeringHarness(
        project, strict=not args.collect_all
    ).run(args.report)
    print(json.dumps({
        "通过": report["passed"],
        "场景": "{}/{}".format(
            report["passed_scenario_count"], report["scenario_count"]
        ),
        "断言": "{}/{}".format(
            report["passed_assertion_count"], report["assertion_count"]
        ),
        "耗时毫秒": report["duration_ms"],
        "报告": str(args.report.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
