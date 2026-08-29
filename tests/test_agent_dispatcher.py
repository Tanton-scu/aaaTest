from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prievo_agent.agents.registry import AgentRegistry
from prievo_agent.application.agent_dispatcher import (
    AgentDispatchError,
    AgentTaskDispatcher,
)
from prievo_agent.domain.models import (
    AgentCapability,
    AgentTask,
    AgentTaskStatus,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


class _Handler:
    def __init__(self, capability, fail=False):
        self.name = "{}Handler".format(capability.value)
        self.capability = capability
        self.fail = fail
        self.calls = []

    def handle(self, task, board):
        self.calls.append((task, board))
        if self.fail:
            raise RuntimeError("fixture handler failure")
        projected = board.task_by_ref(task.id)
        return {
            "task_id": task.id,
            "claimed_status": projected.status.value,
            "input_refs": list(projected.input_artifact_refs),
        }


class _ArtifactWriter:
    def __init__(self, store):
        self.store = store
        self.calls = []

    def __call__(self, task, result):
        self.calls.append((task.id, result))
        artifact = self.store.put_artifact(
            task.run_id,
            "DISPATCH_RESULT",
            json.dumps(result, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        return [artifact]


class AgentTaskDispatcherTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
        optimization = OptimizationTask(
            "task-dispatch", "dispatcher", "minimize", 3, 99,
            dataset_id="fixture", generations=4, population_size=10,
        )
        self.run = Run(
            "run-dispatch", optimization.id, RunStatus.RUNNING,
            generation=2, consumed_evaluations=17, reserved_evaluations=3,
            best_candidate_id="candidate-best", dataset_id="fixture",
        )
        self.store.add_task(optimization)
        self.store.add_run(self.run)
        self.input_artifact = self.store.put_artifact(
            self.run.id, "TOP5_CANDIDATES", b"{}", "application/json"
        )

    def tearDown(self):
        self.store.close()
        self.temporary_directory.cleanup()

    def _task(self, identifier, capability, max_attempts=2):
        task = AgentTask(
            identifier,
            self.run.id,
            "TEST_DISPATCH",
            capability,
            "{}:idempotency".format(identifier),
            [self.input_artifact.id],
            max_attempts=max_attempts,
        )
        self.store.add_agent_task(task)
        return task

    def test_success_rebuilds_blackboard_persists_refs_and_is_idempotent(self):
        handler = _Handler(AgentCapability.SEMANTIC_SIMILARITY)
        writer = _ArtifactWriter(self.store)
        task = self._task(
            "agent-task-success", AgentCapability.SEMANTIC_SIMILARITY
        )
        before = self.store.get_run(self.run.id)
        dispatcher = AgentTaskDispatcher(
            self.store, AgentRegistry([handler]), writer
        )

        result = dispatcher.dispatch(task.id)
        repeated = dispatcher.dispatch(task.id)

        self.assertEqual(AgentTaskStatus.COMPLETED, result.status)
        self.assertTrue(result.executed)
        self.assertEqual(1, len(result.artifact_refs))
        self.assertEqual(result.artifact_refs, repeated.artifact_refs)
        self.assertFalse(repeated.executed)
        self.assertEqual(1, len(handler.calls))
        self.assertEqual(1, len(writer.calls))
        handled_task, board = handler.calls[0]
        self.assertEqual(AgentTaskStatus.CLAIMED, handled_task.status)
        self.assertEqual(
            AgentTaskStatus.CLAIMED, board.task_by_ref(task.id).status
        )
        self.assertEqual(
            (self.input_artifact.id,),
            board.task_by_ref(task.id).input_artifact_refs,
        )
        persisted = self.store.get_agent_task(task.id)
        self.assertEqual(list(result.artifact_refs), persisted.output_artifact_refs)
        events = list(self.store.events_for_run(self.run.id))
        self.assertEqual(
            ["AGENT_TASK_CLAIMED", "AGENT_TASK_COMPLETED"],
            [item.event_type for item in events],
        )
        self.assertEqual(
            list(result.artifact_refs), events[-1].payload["output_artifact_refs"]
        )
        after = self.store.get_run(self.run.id)
        self.assertEqual(before.status, after.status)
        self.assertEqual(before.generation, after.generation)
        self.assertEqual(before.consumed_evaluations, after.consumed_evaluations)
        self.assertEqual(before.reserved_evaluations, after.reserved_evaluations)
        self.assertEqual(before.best_candidate_id, after.best_candidate_id)
        self.assertEqual([], list(self.store.candidates_for_run(self.run.id)))

    def test_handler_failure_requeues_then_enters_terminal_failed(self):
        handler = _Handler(AgentCapability.PRIOR_RESEARCH, fail=True)
        task = self._task(
            "agent-task-failure", AgentCapability.PRIOR_RESEARCH,
            max_attempts=2,
        )
        dispatcher = AgentTaskDispatcher(
            self.store, AgentRegistry([handler]), _ArtifactWriter(self.store)
        )

        with self.assertRaises(AgentDispatchError) as first:
            dispatcher.dispatch(task.id)
        self.assertEqual(AgentTaskStatus.PENDING, first.exception.result.status)
        self.assertTrue(first.exception.result.will_retry)
        self.assertEqual(1, self.store.get_agent_task(task.id).attempts)

        with self.assertRaises(AgentDispatchError) as second:
            dispatcher.dispatch(task.id)
        self.assertEqual(AgentTaskStatus.FAILED, second.exception.result.status)
        self.assertFalse(second.exception.result.will_retry)
        self.assertEqual(2, self.store.get_agent_task(task.id).attempts)
        events = list(self.store.events_for_run(self.run.id))
        failures = [item for item in events if item.event_type == "AGENT_TASK_FAILED"]
        self.assertEqual([True, False], [item.payload["will_retry"] for item in failures])
        self.assertEqual(2, len(handler.calls))

    def test_unregistered_capability_is_audited_without_claiming(self):
        handler = _Handler(AgentCapability.SEMANTIC_SIMILARITY)
        task = self._task(
            "agent-task-wrong-capability", AgentCapability.FINAL_SELECTION
        )
        dispatcher = AgentTaskDispatcher(
            self.store, AgentRegistry([handler]), _ArtifactWriter(self.store)
        )

        with self.assertRaises(AgentDispatchError) as raised:
            dispatcher.dispatch(task.id)

        result = raised.exception.result
        self.assertEqual(AgentTaskStatus.PENDING, result.status)
        self.assertFalse(result.executed)
        self.assertTrue(result.will_retry)
        persisted = self.store.get_agent_task(task.id)
        self.assertEqual(AgentTaskStatus.PENDING, persisted.status)
        self.assertEqual(0, persisted.attempts)
        self.assertEqual([], handler.calls)
        events = list(self.store.events_for_run(self.run.id))
        self.assertEqual(["AGENT_TASK_FAILED"], [item.event_type for item in events])
        self.assertEqual("ROUTING", events[0].payload["failure_stage"])
        self.assertTrue(events[0].payload["will_retry"])

    def test_unpersisted_artifact_ref_follows_failure_policy(self):
        handler = _Handler(AgentCapability.CANDIDATE_REPAIR)
        task = self._task(
            "agent-task-bad-ref", AgentCapability.CANDIDATE_REPAIR,
            max_attempts=1,
        )
        dispatcher = AgentTaskDispatcher(
            self.store,
            AgentRegistry([handler]),
            lambda task, result: ["artifact-not-persisted"],
        )

        with self.assertRaises(AgentDispatchError) as raised:
            dispatcher.dispatch(task.id)

        self.assertEqual(AgentTaskStatus.FAILED, raised.exception.result.status)
        self.assertFalse(raised.exception.result.will_retry)
        self.assertEqual([], self.store.get_agent_task(task.id).output_artifact_refs)
        event = list(self.store.events_for_run(self.run.id))[-1]
        self.assertEqual("AGENT_TASK_FAILED", event.event_type)
        self.assertEqual("ValueError", event.payload["error_type"])

    def test_handler_crossing_lease_boundary_cannot_write_or_settle(self):
        current = [datetime(2026, 8, 13, tzinfo=timezone.utc)]
        handler = _Handler(AgentCapability.HEURISTIC_GENERATION)
        original_handle = handler.handle

        def slow_handle(task, board):
            result = original_handle(task, board)
            current[0] += timedelta(seconds=6)
            return result

        handler.handle = slow_handle
        writer = _ArtifactWriter(self.store)
        task = self._task(
            "agent-task-expired-handler", AgentCapability.HEURISTIC_GENERATION
        )
        dispatcher = AgentTaskDispatcher(
            self.store,
            AgentRegistry([handler]),
            writer,
            clock=lambda: current[0],
            lease_seconds=5,
        )

        with self.assertRaises(AgentDispatchError) as raised:
            dispatcher.dispatch(task.id)

        self.assertIn("lease", raised.exception.result.error_message)
        self.assertEqual([], writer.calls)
        expired = self.store.get_agent_task(task.id)
        self.assertEqual(AgentTaskStatus.CLAIMED, expired.status)
        self.assertEqual(1, self.store.recover_orphan_agent_tasks(current[0]))
        self.assertEqual(
            AgentTaskStatus.PENDING, self.store.get_agent_task(task.id).status
        )
        event_types = [
            event.event_type for event in self.store.events_for_run(self.run.id)
        ]
        self.assertEqual(["AGENT_TASK_CLAIMED"], event_types)


if __name__ == "__main__":
    unittest.main()
