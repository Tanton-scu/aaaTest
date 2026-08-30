from __future__ import annotations

import unittest

from prievo_agent.infrastructure.testing.scripted_fake_llm import (
    ScriptedFakeLLM,
    ScriptedTransientLLMError,
)


class ScriptedFakeLLMTest(unittest.TestCase):
    def test_agent_and_skill_queues_are_fifo_and_record_full_prompt(self):
        first = ScriptedFakeLLM.knowledge_gap("specific gap")
        second = {
            "result_type": "CandidateDraft",
            "code": "def run_tuners(file, budget, seed, maxlives):\n    return 1\n",
            "description": "fixture",
            "operators": ["Fixture"],
        }
        model = ScriptedFakeLLM(
            {"generation:i1": [first, second]}, strict=True
        )
        prompt = 'Current strategy skill: {"strategy": "i1"}\nfull-prompt'

        self.assertEqual("KnowledgeGap", model.generate_heuristic_draft(prompt)["result_type"])
        self.assertEqual("CandidateDraft", model.generate_heuristic_draft(prompt)["result_type"])
        self.assertEqual(["generation:i1", "generation:i1"], [
            item.route for item in model.scripted_calls
        ])
        self.assertEqual([prompt, prompt], [item.prompt for item in model.scripted_calls])
        self.assertEqual({}, model.remaining())

    def test_malformed_and_transient_faults_are_not_hidden(self):
        model = ScriptedFakeLLM(
            {
                "generation": [
                    ScriptedFakeLLM.malformed(),
                    ScriptedFakeLLM.transient("provider unavailable"),
                ]
            },
            strict=True,
        )
        prompt = 'Current strategy skill: {"strategy": "i1"}'

        self.assertEqual(42, model.generate_heuristic_draft(prompt)["code"])
        with self.assertRaisesRegex(ScriptedTransientLLMError, "provider unavailable"):
            model.generate_heuristic_draft(prompt)
        self.assertEqual(["RETURNED", "RAISED"], [
            item.outcome for item in model.scripted_calls
        ])
        self.assertEqual("ScriptedTransientLLMError", model.scripted_calls[-1].error_type)

    def test_serializable_queue_and_call_trace_survive_state_roundtrip(self):
        model = ScriptedFakeLLM(
            {"research:query": ["first", ScriptedFakeLLM.transient("retry")]},
            strict=True,
        )
        self.assertEqual("first", model.formulate_query("query prompt"))
        state = model.export_state()

        restored = ScriptedFakeLLM()
        restored.import_state(state)
        self.assertEqual(1, len(restored.scripted_calls))
        with self.assertRaisesRegex(ScriptedTransientLLMError, "retry"):
            restored.formulate_query("query prompt after recovery")

    def test_deterministic_fallback_is_explicitly_recorded(self):
        model = ScriptedFakeLLM()
        response = model.generate_similarity_decision("prompt", ["a", "b", "c"])

        self.assertEqual(["a", "b"], response["selected_instance_ids"])
        self.assertEqual("fallback", model.scripted_calls[0].source)
        self.assertEqual("prompt", model.scripted_calls[0].prompt)


if __name__ == "__main__":
    unittest.main()
