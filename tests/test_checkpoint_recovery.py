from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prievo_agent.runtime.checkpoint_harness import CheckpointRecoveryHarness


class CheckpointRecoveryTest(unittest.TestCase):
    def test_real_process_restart_matches_uninterrupted_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prievo-test-recovery-") as temp:
            report = CheckpointRecoveryHarness(ROOT).run(Path(temp))
        self.assertTrue(report.recovery_event_present)
        self.assertEqual(0, report.duplicate_evaluations)
        self.assertEqual(
            report.uninterrupted["consumed_budget"],
            report.recovered["consumed_budget"],
        )
        self.assertEqual(27, report.recovered["evaluation_result_count"])


if __name__ == "__main__":
    unittest.main()
