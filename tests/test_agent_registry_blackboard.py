from __future__ import annotations

import tempfile
import unittest
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prievo_agent.agents.registry import (
    AgentNotRegisteredError,
    AgentRegistry,
    DuplicateAgentRegistrationError,
)
from prievo_agent.application.orchestration.blackboard import Blackboard
from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import (
    AgentCapability,
    AgentTask,
    AgentTaskStatus,
    OptimizationTask,
    Run,
)
from prievo_agent.infrastructure.local.sqlite_store import SQLiteRuntimeStore


class _Agent:
    def __init__(self, name: str, capability: AgentCapability):
        self.name = name
        self.capability = capability


class AgentRegistryTests(unittest.TestCase):
    def test_capability_matches_exactly_one_handler(self):
        similarity = _Agent(
            "SimilaritySelectionNode", AgentCapability.SEMANTIC_SIMILARITY
        )
        generation = _Agent(
            "HeuristicGenerationAgent", AgentCapability.HEURISTIC_GENERATION
        )
        registry = AgentRegistry([similarity, generation])
        task = AgentTask(
            "agent-task-1",
            "run-1",
            "SEMANTIC_SIMILARITY_SELECTION",
            AgentCapability.SEMANTIC_SIMILARITY,
            "run-1:similarity",
        )

        self.assertIs(registry.match(task), similarity)
        self.assertIs(
            registry.resolve(AgentCapability.HEURISTIC_GENERATION), generation
        )
        self.assertEqual(len(registry), 2)
        self.assertEqual(
            registry.registration_for("SEMANTIC_SIMILARITY").name,
            "SimilaritySelectionNode",
        )

    def test_duplicate_capability_and_missing_capability_fail_explicitly(self):
        registry = AgentRegistry(
            [_Agent("SimilaritySelectionNode", AgentCapability.SEMANTIC_SIMILARITY)]
        )

        with self.assertRaises(DuplicateAgentRegistrationError):
            registry.register(
                _Agent("AnotherSimilaritySelectionNode", AgentCapability.SEMANTIC_SIMILARITY)
            )
        with self.assertRaises(AgentNotRegisteredError):
            registry.resolve(AgentCapability.FINAL_SELECTION)


class BlackboardProjectionTests(unittest.TestCase):
    def test_projection_rebuilds_from_durable_records_and_exposes_only_refs(self):
        with tempfile.TemporaryDirectory(prefix="prievo-blackboard-") as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            artifacts = root / "artifacts"
            store = SQLiteRuntimeStore(database, artifacts)
            task = OptimizationTask("task-1", "Blackboard", "minimize", 1)
            run = Run("run-1", task.id)
            store.add_task(task)
            store.add_run(run)
            input_artifact = store.put_artifact(
                run.id,
                "TOP5_CANDIDATES",
                b'{"code":"must-not-enter-blackboard","population":[1,2,3]}',
                "application/json",
            )
            decision_artifact = store.put_artifact(
                run.id,
                "SIMILARITY_DECISION",
                b'{"selected_instance_ids":["historical-1"]}',
                "application/json",
            )
            open_task = AgentTask(
                "agent-task-open",
                run.id,
                "SEMANTIC_SIMILARITY_SELECTION",
                AgentCapability.SEMANTIC_SIMILARITY,
                "run-1:similarity",
                input_artifact_refs=[input_artifact.id],
            )
            completed_task = AgentTask(
                "agent-task-completed",
                run.id,
                "FINAL_SELECTION",
                AgentCapability.FINAL_SELECTION,
                "run-1:final-selection",
                output_artifact_refs=[decision_artifact.id],
                status=AgentTaskStatus.COMPLETED,
            )
            store.add_agent_task(open_task)
            store.add_agent_task(completed_task)
            event = store.append_event(
                run.id,
                EventType.AGENT_TASK_CREATED.value,
                "已创建语义相似度任务",
                agent_task_id=open_task.id,
                input_artifact_refs=[input_artifact.id],
                population=["must", "not", "leak"],
                prior={"must": "not leak"},
                code="must-not-leak",
            )
            store.close()

            # 关闭并重新打开数据库，证明投影来自 durable store，而非进程内对象。
            reopened = SQLiteRuntimeStore(database, artifacts)
            try:
                board = Blackboard.from_store(reopened, run.id)
            finally:
                reopened.close()

            self.assertEqual(
                [item.id for item in board.open_tasks()], [open_task.id]
            )
            self.assertEqual(
                board.task_by_ref(open_task.id).input_artifact_refs,
                (input_artifact.id,),
            )
            self.assertEqual(
                board.artifact_by_ref(input_artifact.id).kind, "TOP5_CANDIDATES"
            )
            self.assertEqual(
                [item.id for item in board.artifacts_by_kind("SIMILARITY_DECISION")],
                [decision_artifact.id],
            )
            event_view = board.event_by_sequence(event.sequence)
            self.assertEqual(event_view.task_refs, (open_task.id,))
            self.assertEqual(event_view.artifact_refs, (input_artifact.id,))
            self.assertFalse(hasattr(event_view, "payload"))
            self.assertFalse(hasattr(board, "population"))
            self.assertFalse(hasattr(board, "prior"))
            self.assertFalse(hasattr(board.artifacts[0], "content"))
            with self.assertRaises(FrozenInstanceError):
                board.tasks[0].status = AgentTaskStatus.CANCELLED

    def test_projection_rejects_cross_run_records(self):
        class _BrokenStore:
            def agent_tasks_for_run(self, run_id):
                return [
                    AgentTask(
                        "wrong-task",
                        "another-run",
                        "FINAL_SELECTION",
                        AgentCapability.FINAL_SELECTION,
                        "wrong",
                    )
                ]

            def events_for_run(self, run_id):
                return []

            def artifacts_for_run(self, run_id):
                return []

        with self.assertRaisesRegex(ValueError, "混入其他 Run"):
            Blackboard.from_store(_BrokenStore(), "run-1")


if __name__ == "__main__":
    unittest.main()
