import json
import unittest

from prievo_agent.domain.errors import CandidateInvalidError, EvaluationTimeoutError
from prievo_agent.security import (
    BoundedCandidateExecutor,
    CandidateCodeValidator,
    StructuredCandidateParser,
)
from prievo_agent.security.structured_output import StructuredOutputError


GOOD = """def run_tuners(file, budget, seed, maxlives):
    return {'file': file, 'score': budget + seed + maxlives}
"""


class CandidateSecurityTest(unittest.TestCase):
    def test_structured_output_schema_and_bounded_repair(self):
        parser = StructuredCandidateParser(max_repairs=1)
        calls = []

        def repair(raw, error):
            calls.append((raw, error))
            return json.dumps({"code": GOOD, "description": "有效", "operators": ["Revise"]})

        parsed = parser.parse("not-json", repair)
        self.assertEqual(GOOD, parsed.proposal.code)
        self.assertEqual(2, len(parsed.raw_responses))
        self.assertEqual(1, len(calls))
        with self.assertRaises(StructuredOutputError):
            parser.parse("{}", lambda raw, error: "{}")

    def test_invalid_and_maliciousish_candidates_are_rejected(self):
        validator = CandidateCodeValidator()
        bad = [
            "def nope(file, budget, seed, maxlives): return 1",
            "import os\ndef run_tuners(file, budget, seed, maxlives): return 1",
            "def run_tuners(file, budget, seed, maxlives): return open(file).read()",
            "def run_tuners(file, budget, seed, maxlives): return __import__('os')",
        ]
        for code in bad:
            with self.subTest(code=code), self.assertRaises(CandidateInvalidError):
                validator.validate(code)

    def test_subprocess_execution_is_bounded_and_uses_contract(self):
        executor = BoundedCandidateExecutor(timeout_seconds=1)
        result = executor.execute(
            GOOD, {"file": "fixture.csv", "budget": 2, "seed": 3, "maxlives": 4}
        )
        self.assertEqual(9, result["score"])

        endless = """def run_tuners(file, budget, seed, maxlives):
    while True:
        pass
"""
        with self.assertRaises(EvaluationTimeoutError):
            BoundedCandidateExecutor(timeout_seconds=0.2).execute(
                endless, {"file": "x", "budget": 1, "seed": 1, "maxlives": 1}
            )


if __name__ == "__main__":
    unittest.main()
