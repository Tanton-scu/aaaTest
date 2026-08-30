import os
import sys
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prievo_agent.domain.models import (
    AgentCapability,
    AgentTask,
    AgentTaskStatus,
    OptimizationTask,
    Run,
)
from prievo_agent.infrastructure.mysql_store import MySQLRuntimeStore


@unittest.skipUnless(
    os.getenv("DATABASE_URL", "").startswith("mysql+pymysql://"),
    "需要 Full Mode MySQL",
)
class MySQLAgentTaskTest(unittest.TestCase):
    def test_v3_idempotency_concurrent_claim_and_terminal_transitions(self):
        suffix = uuid.uuid4().hex[:12]
        database_url = os.environ["DATABASE_URL"]
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            store = MySQLRuntimeStore(database_url, artifact_root)
            try:
                optimization = OptimizationTask(
                    "task-" + suffix, "mysql-agent-task", "minimize", 1, 10,
                    dataset_id="xgboost-Covtype", generations=1,
                    population_size=1,
                )
                run = Run(
                    "run-" + suffix, optimization.id,
                    dataset_id=optimization.dataset_id,
                )
                store.add_task(optimization)
                store.add_run(run)

                task = AgentTask(
                    "agent-task-" + suffix, run.id, "SIMILARITY_SELECTION",
                    AgentCapability.SEMANTIC_SIMILARITY,
                    "similarity:top5:v1", ["artifact-top5"], max_attempts=2,
                )
                persisted, created = store.add_agent_task(task)
                duplicate, duplicate_created = store.add_agent_task(AgentTask(
                    "agent-task-duplicate-" + suffix, run.id,
                    "SIMILARITY_SELECTION",
                    AgentCapability.SEMANTIC_SIMILARITY,
                    task.idempotency_key, ["artifact-other"], max_attempts=2,
                ))
                self.assertTrue(created)
                self.assertFalse(duplicate_created)
                self.assertEqual(persisted.id, duplicate.id)

                barrier = threading.Barrier(2)

                def claim(agent_name):
                    connection = MySQLRuntimeStore(
                        database_url, artifact_root, ensure_schema=False,
                    )
                    try:
                        barrier.wait(timeout=5)
                        return connection.claim_agent_task(task.id, agent_name)
                    finally:
                        connection.close()

                outcomes = []
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(claim, "SimilaritySelectionNode-A"),
                        executor.submit(claim, "SimilaritySelectionNode-B"),
                    ]
                    for future in futures:
                        try:
                            outcomes.append(("ok", future.result(timeout=10)))
                        except RuntimeError as exc:
                            outcomes.append(("conflict", str(exc)))

                self.assertEqual(1, sum(kind == "ok" for kind, _ in outcomes))
                self.assertEqual(1, sum(kind == "conflict" for kind, _ in outcomes))
                claimed = store.get_agent_task(task.id)
                self.assertEqual(AgentTaskStatus.CLAIMED, claimed.status)
                self.assertEqual(1, claimed.attempts)
                completed = store.complete_agent_task(
                    task.id, claimed.claim_token,
                    ["artifact-similarity-decision"],
                )
                self.assertEqual(AgentTaskStatus.COMPLETED, completed.status)
                self.assertEqual(
                    ["artifact-similarity-decision"],
                    completed.output_artifact_refs,
                )

                retry_task = AgentTask(
                    "agent-task-retry-" + suffix, run.id, "LITERATURE_EVIDENCE",
                    AgentCapability.LITERATURE_EVIDENCE, "research:gap:v1",
                    ["artifact-gap"], max_attempts=2,
                )
                store.add_agent_task(retry_task)
                claimed_retry = store.claim_agent_task(
                    retry_task.id, "LiteratureEvidenceResolver"
                )
                first_failure = store.fail_agent_task(
                    retry_task.id, claimed_retry.claim_token,
                    "fixture transient model failure",
                )
                self.assertEqual(AgentTaskStatus.PENDING, first_failure.status)
                claimed_retry = store.claim_agent_task(
                    retry_task.id, "LiteratureEvidenceResolver"
                )
                final_failure = store.fail_agent_task(
                    retry_task.id, claimed_retry.claim_token,
                    "fixture repeated model failure",
                )
                self.assertEqual(AgentTaskStatus.FAILED, final_failure.status)
                self.assertEqual(2, final_failure.attempts)

                listed = store.agent_tasks_for_run(run.id)
                self.assertEqual({task.id, retry_task.id}, {item.id for item in listed})
                migration = store._one(
                    "SELECT version_no FROM schema_migrations WHERE version_no=5",
                    (),
                )
                self.assertEqual(5, migration["version_no"])

                fencing_task = AgentTask(
                    "agent-task-fencing-" + suffix, run.id,
                    "HEURISTIC_GENERATION",
                    AgentCapability.HEURISTIC_GENERATION,
                    "fencing:" + suffix, max_attempts=3,
                )
                store.add_agent_task(fencing_task)
                now = datetime(2026, 8, 13, tzinfo=timezone.utc)
                claimed_a = store.claim_agent_task(
                    fencing_task.id, "GenerationAgent-A", now, 5,
                    claim_token="mysql-claim-a-" + suffix,
                )
                expired = now + timedelta(seconds=5)
                self.assertEqual(
                    1, store.recover_orphan_agent_tasks(expired)
                )
                claimed_b = store.claim_agent_task(
                    fencing_task.id, "GenerationAgent-B", expired, 5,
                    claim_token="mysql-claim-b-" + suffix,
                )
                with self.assertRaisesRegex(RuntimeError, "lease"):
                    store.complete_agent_task(
                        fencing_task.id, claimed_a.claim_token,
                        ["artifact-stale-a"], expired + timedelta(seconds=1),
                    )
                current = store.get_agent_task(fencing_task.id)
                self.assertEqual(AgentTaskStatus.CLAIMED, current.status)
                self.assertEqual(claimed_b.claim_token, current.claim_token)
                store.complete_agent_task(
                    fencing_task.id, claimed_b.claim_token,
                    ["artifact-winner-b"], expired + timedelta(seconds=1),
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
