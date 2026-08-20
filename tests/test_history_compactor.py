from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.application.history_compactor import HistoryCompactor
from prievo_agent.application.run_local_memory import RunLocalMemoryService
from prievo_agent.infrastructure.agent_memory import InMemoryAgentWorkingMemory
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


class HistoryCompactorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prievo-history-summary-")
        root = Path(self.temp.name)
        self.store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
        self.cache = InMemoryAgentWorkingMemory(max_items=30)
        self.model = FakeLLM()
        skills = SkillRegistry(Path(__file__).resolve().parents[1] / "skills")
        self.compactor = HistoryCompactor(
            self.store,
            skills,
            self.model,
            self.cache,
            threshold=4,
        )
        self.memory = RunLocalMemoryService(self.store, self.cache)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def _persist(self, run_id, index):
        return self.memory.persist(
            run_id=run_id,
            dataset_id="fixture",
            scope="generation",
            memory_type="GENERATION_SUMMARY",
            subject="generation {}".format(index),
            content={
                "strategy": "m{}".format(index % 2 + 1),
                "fitness": float(index),
                "candidate_id": "candidate-{}-{}".format(run_id, index),
                "evidence_artifact_id": "artifact-{}-{}".format(run_id, index),
            },
            evidence_artifact_id="artifact-{}-{}".format(run_id, index),
            identity_material=str(index),
        )

    def test_below_threshold_keeps_recent_window_without_llm(self):
        for index in range(4):
            self._persist("run-a", index)
        values = self.compactor.context_records(
            "run-a", "fixture", "generation", recent_window=3
        )
        self.assertEqual(3, len(values))
        self.assertEqual([], self.model.history_summary_calls)
        self.assertEqual([], [
            item for item in self.store.artifacts_for_run("run-a")
            if item.kind == "HISTORY_SUMMARY"
        ])

    def test_over_threshold_summarizes_only_early_history_once_and_keeps_refs(self):
        for index in range(7):
            self._persist("run-a", index)
        self._persist("run-b", 99)

        first = self.compactor.context_records(
            "run-a", "fixture", "generation", recent_window=3
        )
        second = self.compactor.context_records(
            "run-a", "fixture", "generation", recent_window=3
        )

        self.assertEqual(4, len(first))
        self.assertEqual(4, len(second))
        self.assertEqual(1, len(self.model.history_summary_calls))
        prompt = self.model.history_summary_calls[0]["prompt"]
        self.assertIn("history_summary", prompt)
        self.assertIn("candidate-run-a-0", prompt)
        self.assertNotIn("candidate-run-a-6", prompt, "recent window 不应被摘要")
        self.assertNotIn("candidate-run-b-99", prompt, "禁止跨 Run")
        summaries = [
            item for item in self.store.agent_memories_for_run(
                "run-a", "generation", 20
            )
            if item.memory_type == "GENERATION_HISTORY_SUMMARY"
        ]
        self.assertEqual(1, len(summaries))
        payload = json.loads(summaries[0].content)
        self.assertEqual(4, payload["source_count"])
        self.assertIn("candidate-run-a-0", payload["source_refs"])
        self.assertIn("artifact-run-a-3", payload["source_refs"])
        self.assertTrue(payload["skill_digest"])
        self.assertTrue(payload["prompt_artifact_id"])

    def test_rolling_watermark_avoids_per_record_resummary_and_bounds_delta(self):
        for index in range(7):
            self._persist("run-a", index)
        self.compactor.context_records(
            "run-a", "fixture", "generation", recent_window=3
        )
        self.assertEqual(1, len(self.model.history_summary_calls))

        # 新增一条只让一个 former-recent 进入 early 区，不应触发 LLM。
        self._persist("run-a", 7)
        values = self.compactor.context_records(
            "run-a", "fixture", "generation", recent_window=3
        )
        self.assertEqual(1, len(self.model.history_summary_calls))
        self.assertEqual(5, len(values), "旧摘要 + 1 条待汇总事实 + Recent Window")

        # 累计到完整批次才滚动一次，且第二次 Prompt 不重复完整旧历史。
        for index in (8, 9, 10):
            self._persist("run-a", index)
        self.compactor.context_records(
            "run-a", "fixture", "generation", recent_window=3
        )
        self.assertEqual(2, len(self.model.history_summary_calls))
        second_prompt = self.model.history_summary_calls[1]["prompt"]
        self.assertIn("previous_summary", second_prompt)
        self.assertIn("candidate-run-a-4", second_prompt)
        self.assertNotIn("candidate-run-a-0", second_prompt)
        self.assertLess(len(second_prompt), 20_000)

        summaries = [
            item
            for item in self.store.agent_memories_for_run(
                "run-a", "generation", 50
            )
            if item.memory_type == "GENERATION_HISTORY_SUMMARY"
        ]
        self.assertEqual(2, len(summaries))
        latest = json.loads(summaries[-1].content)
        self.assertEqual(8, latest["source_count"])
        self.assertEqual(8, len(latest["covered_memory_ids"]))


if __name__ == "__main__":
    unittest.main()
