import unittest

from prievo_agent.agents.context import AgentContextBuilder


class AgentContextBuilderTest(unittest.TestCase):
    def test_pinned_context_survives_real_compression(self):
        builder = AgentContextBuilder(max_chars=800)
        result = builder.build(
            pinned=["run_id=R1; dataset_id=D1", "EvolutionAdvice=A1"],
            evidence=["evidence-{} {}".format(i, "x" * 180) for i in range(8)],
            recent_memory=["memory-{} {}".format(i, "y" * 120) for i in range(6)],
            long_term_memory=["lesson-{} {}".format(i, "z" * 140) for i in range(5)],
        )
        self.assertTrue(result.compression_applied)
        self.assertGreater(result.omitted_items, 0)
        self.assertIn("run_id=R1; dataset_id=D1", result.text)
        self.assertIn("EvolutionAdvice=A1", result.text)
        self.assertIn("Context compression: applied=true", result.text)
        self.assertLess(result.chars_after, result.chars_before)


if __name__ == "__main__":
    unittest.main()
