from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prievo_agent.domain.errors import (
    AlgorithmOOMError,
    AlgorithmTimeoutError,
    CandidateInterfaceError,
    CandidateLogicError,
    CandidateRuntimeError,
    CandidateSyntaxError,
    EvaluationTimeoutError,
    InfrastructureTimeoutError,
    NetworkEvaluationError,
    TransientInfrastructureError,
    WorkerCrashError,
)
from prievo_agent.domain.models import (
    Candidate,
    EvaluationJobStatus,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.evaluation_queue import EvaluationQueueService, EvaluationWorker
from prievo_agent.runtime.failure_classifier import (
    FailureAction,
    FailureClassifier,
    FailureType,
)


class FailureClassifierTest(unittest.TestCase):
    def setUp(self):
        self.classifier = FailureClassifier()

    def test_explicit_exception_matrix_covers_every_failure_type(self):
        cases = [
            (NetworkEvaluationError("reset"), FailureType.NETWORK, FailureAction.RETRY),
            (WorkerCrashError("lost"), FailureType.WORKER_CRASH, FailureAction.RETRY),
            (TransientInfrastructureError("db busy"), FailureType.TRANSIENT_INFRA, FailureAction.RETRY),
            (CandidateSyntaxError("bad syntax"), FailureType.SYNTAX_ERROR, FailureAction.REPAIR),
            (CandidateRuntimeError("index"), FailureType.RUNTIME_ERROR, FailureAction.REPAIR),
            (CandidateInterfaceError("signature"), FailureType.INTERFACE_ERROR, FailureAction.REPAIR),
            (AlgorithmTimeoutError("loop"), FailureType.ALGORITHM_TIMEOUT, FailureAction.REPAIR),
            (AlgorithmOOMError("memory"), FailureType.ALGORITHM_OOM, FailureAction.REPAIR),
            (CandidateLogicError("no progress"), FailureType.LOGIC_FAILURE, FailureAction.REPAIR),
            (RuntimeError("unclassified"), FailureType.UNKNOWN, FailureAction.DEAD),
        ]
        for exception, expected_type, expected_action in cases:
            with self.subTest(expected_type=expected_type.value):
                decision = self.classifier.classify(exception=exception)
                self.assertEqual(expected_type, decision.failure_type)
                self.assertEqual(expected_action, decision.action)
                self.assertEqual(
                    expected_action == FailureAction.RETRY, decision.retryable
                )
                self.assertEqual(
                    expected_action == FailureAction.REPAIR, decision.repairable
                )

    def test_error_code_return_code_and_stderr_are_explicit_evidence(self):
        cases = [
            ({"error_code": "NETWORK_ERROR"}, FailureType.NETWORK),
            ({"error_code": "LEASE_EXPIRED"}, FailureType.WORKER_CRASH),
            ({"error_code": "INFRA_TIMEOUT"}, FailureType.TRANSIENT_INFRA),
            ({"stderr": "SyntaxError: invalid syntax"}, FailureType.SYNTAX_ERROR),
            ({"return_code": 1}, FailureType.RUNTIME_ERROR),
            ({"stderr": "invalid output: interface contract"}, FailureType.INTERFACE_ERROR),
            ({"error_code": "EVALUATION_TIMEOUT"}, FailureType.ALGORITHM_TIMEOUT),
            ({"return_code": 137}, FailureType.ALGORITHM_OOM),
            ({"error_code": "LOGIC_FAILURE"}, FailureType.LOGIC_FAILURE),
            ({}, FailureType.UNKNOWN),
        ]
        for evidence, expected in cases:
            with self.subTest(expected=expected.value):
                decision = self.classifier.classify(**evidence)
                self.assertEqual(expected, decision.failure_type)

    def test_candidate_timeout_and_infrastructure_timeout_have_different_actions(self):
        legacy_candidate_timeout = self.classifier.classify(
            exception=EvaluationTimeoutError("candidate process exceeded limit")
        )
        infrastructure_timeout = self.classifier.classify(
            exception=InfrastructureTimeoutError("object store timeout")
        )

        self.assertEqual(
            (FailureType.ALGORITHM_TIMEOUT, FailureAction.REPAIR),
            (legacy_candidate_timeout.failure_type, legacy_candidate_timeout.action),
        )
        self.assertEqual(
            (FailureType.TRANSIENT_INFRA, FailureAction.RETRY),
            (infrastructure_timeout.failure_type, infrastructure_timeout.action),
        )


class _RaisingEvaluator:
    def __init__(self, exception):
        self.exception = exception

    def evaluate(self, candidate, task):
        raise self.exception


class EvaluationWorkerClassifierIntegrationTest(unittest.TestCase):
    def _run_once(self, exception, suffix):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
        task = OptimizationTask("task-" + suffix, suffix, "minimize", 1, 3)
        run = Run("run-" + suffix, task.id, status=RunStatus.RUNNING)
        candidate = Candidate(
            "candidate-" + suffix,
            run.id,
            "def run_tuners(file, budget, seed, maxlives): return 1",
            suffix,
            ["fixture"],
            {},
        )
        store.add_task(task)
        store.add_run(run)
        store.add_candidate(candidate)
        job = EvaluationQueueService(store).submit(
            run.id, task.id, candidate.id, seed=1, budget=1, max_attempts=2
        )
        updated = EvaluationWorker(store, _RaisingEvaluator(exception)).run_once()
        events = list(store.events_for_run(run.id))
        return temporary, store, job, updated, events

    def test_worker_retries_only_infrastructure_timeout(self):
        temporary, store, job, updated, events = self._run_once(
            InfrastructureTimeoutError("redis timeout"), "infra-timeout"
        )
        try:
            self.assertEqual(EvaluationJobStatus.RETRY_WAIT, updated.status)
            event = events[-1]
            self.assertEqual("TRANSIENT_INFRA", event.payload["failure_type"])
            self.assertEqual("RETRY", event.payload["failure_action"])
        finally:
            store.close()
            temporary.cleanup()

    def test_worker_marks_algorithm_timeout_for_repair_without_backend_retry(self):
        temporary, store, job, updated, events = self._run_once(
            AlgorithmTimeoutError("candidate endless loop"), "algorithm-timeout"
        )
        try:
            self.assertEqual(EvaluationJobStatus.DEAD, updated.status)
            self.assertEqual(1, updated.attempts)
            event = events[-1]
            self.assertEqual("ALGORITHM_TIMEOUT", event.payload["failure_type"])
            self.assertEqual("REPAIR", event.payload["failure_action"])
        finally:
            store.close()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
