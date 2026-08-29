from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prievo_agent.runtime.queue_harness import QueueReliabilityHarness


class EvaluationQueueTest(unittest.TestCase):
    def test_queue_retry_idempotency_budget_dead_letter_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prievo-test-queue-") as temp:
            report = QueueReliabilityHarness().run(Path(temp))
        self.assertEqual(2, report.retry_attempts)
        self.assertEqual(1, report.duplicate_jobs_created)
        self.assertTrue(report.budget_overrun_blocked)
        self.assertEqual("DEAD", report.invalid_status)
        self.assertEqual("DEAD", report.timeout_status)
        self.assertEqual("SUCCESS", report.crash_recovered_status)
        self.assertEqual(2, report.crash_benchmark_calls)
        self.assertEqual(1, report.crash_logical_results)
        self.assertEqual(3, report.crash_budget_charged)
        self.assertEqual(1, report.concurrent_budget_winners)


if __name__ == "__main__":
    unittest.main()
