import unittest
from pathlib import Path

from prievo_agent.runtime.agent_harness import AgentMainlineHarness


class AgentHarnessTest(unittest.TestCase):
    def test_complete_agent_mainline_harness(self):
        report = AgentMainlineHarness(Path(__file__).resolve().parents[1]).run()
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
