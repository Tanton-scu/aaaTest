from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from prievo_agent.domain.models import (
    ArtifactMetadata,
    Candidate,
    CandidateStatus,
    CheckpointMetadata,
    EvaluationResult,
    EvaluationJob,
    EvaluationJobStatus,
    Event,
    GenerationPlan,
    OptimizationTask,
    PreviousPlanFeedback,
    Run,
    RunStatus,
    AgentMemory,
    AgentTask,
    AgentTaskStatus,
    AgentCapability,
    TraceRecord,
    ToolCallRecord,
)
from prievo_agent.domain.errors import (
    ArtifactIntegrityError,
    BudgetExhaustedError,
    EvaluationIdentityConflictError,
)
from prievo_agent.infrastructure.agent_memory import memory_types_for_scope


def _dt(value: datetime) -> str:
    return value.isoformat()


def _same_result(row: sqlite3.Row, result: EvaluationResult) -> bool:
    return (
        row["id"] == result.id
        and row["run_id"] == result.run_id
        and row["candidate_id"] == result.candidate_id
        and row["objective"] == result.objective
        and json.loads(row["trajectory_json"]) == result.trajectory
        and json.loads(row["best_configuration_json"])
        == result.best_configuration
        and row["used_budget"] == result.used_budget
    )


def _same_job_identity(row: sqlite3.Row, job: EvaluationJob) -> bool:
    return (
        row["idempotency_key"] == job.idempotency_key
        and row["run_id"] == job.run_id
        and row["task_id"] == job.task_id
        and row["candidate_id"] == job.candidate_id
        and row["seed"] == job.seed
        and row["budget"] == job.budget
    )


class SQLiteRuntimeStore:
    """SQLite current state/event metadata + filesystem 大 artifact。"""

    def __init__(self, database_path: Path, artifact_root: Path, ensure_schema=True,
                 durable=True) -> None:
        self.database_path = Path(database_path)
        self.artifact_root = Path(artifact_root)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.database_path), timeout=10)
        self.connection.row_factory = sqlite3.Row
        if ensure_schema:
            self.connection.execute("PRAGMA journal_mode=WAL")
        # Full Mode 使用 MySQL；本地 API Demo 可关闭 fsync 以保持快速 202 响应。
        # 直接构造 SQLite store 时默认 NORMAL，checkpoint recovery 测试仍覆盖持久模式。
        self.connection.execute("PRAGMA synchronous={}".format(
            "NORMAL" if durable else "OFF"
        ))
        self.connection.execute("PRAGMA foreign_keys=ON")
        if ensure_schema:
            self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, objective TEXT NOT NULL,
                evaluation_budget INTEGER NOT NULL, total_budget INTEGER NOT NULL,
                created_at TEXT NOT NULL, dataset_id TEXT NOT NULL DEFAULT '',
                generations INTEGER NOT NULL DEFAULT 2,
                population_size INTEGER NOT NULL DEFAULT 3,
                random_seed INTEGER NOT NULL DEFAULT 2024
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, status TEXT NOT NULL,
                generation INTEGER NOT NULL, consumed_evaluations INTEGER NOT NULL,
                reserved_evaluations INTEGER NOT NULL,
                best_candidate_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                dataset_id TEXT NOT NULL DEFAULT '',
                pause_requested INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                control_reason TEXT NOT NULL DEFAULT '',
                runtime_cursor_artifact_id TEXT NOT NULL DEFAULT '',
                runtime_owner_id TEXT NOT NULL DEFAULT '',
                runtime_lease_expires_at TEXT,
                FOREIGN KEY(task_id) REFERENCES tasks(id)
            );
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, code TEXT NOT NULL,
                description TEXT NOT NULL, operators_json TEXT NOT NULL,
                lineage_json TEXT NOT NULL, status TEXT NOT NULL, objective REAL,
                code_artifact_id TEXT, evaluation_artifact_id TEXT,
                generation INTEGER NOT NULL DEFAULT 0,
                plan_id TEXT NOT NULL DEFAULT '',
                generation_strategy TEXT NOT NULL DEFAULT '',
                code_digest TEXT NOT NULL DEFAULT '',
                selected_parent_ids_json TEXT NOT NULL DEFAULT '[]',
                prior_refs_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            CREATE TABLE IF NOT EXISTS generation_plans (
                plan_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                generation INTEGER NOT NULL, sequence INTEGER NOT NULL,
                generation_strategy TEXT NOT NULL,
                parent_selection_policy TEXT NOT NULL,
                decision_reason TEXT NOT NULL,
                required_parent_count INTEGER NOT NULL,
                previous_feedback_json TEXT NOT NULL DEFAULT '{}',
                context_artifact_id TEXT NOT NULL DEFAULT '',
                prompt_artifact_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(run_id, generation, generation_strategy, sequence)
            );
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, candidate_id TEXT UNIQUE NOT NULL,
                objective REAL NOT NULL, trajectory_json TEXT NOT NULL,
                best_configuration_json TEXT NOT NULL, used_budget INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL, sequence INTEGER NOT NULL, event_type TEXT NOT NULL,
                message TEXT NOT NULL, payload_json TEXT NOT NULL, occurred_at TEXT NOT NULL,
                PRIMARY KEY(run_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                media_type TEXT NOT NULL, size INTEGER NOT NULL, digest TEXT NOT NULL,
                uri TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, generation INTEGER NOT NULL,
                population_ids_json TEXT NOT NULL, consumed_budget INTEGER NOT NULL,
                remaining_budget INTEGER NOT NULL, artifact_id TEXT NOT NULL,
                schema_version INTEGER NOT NULL, code_version TEXT NOT NULL,
                created_at TEXT NOT NULL, dataset_id TEXT NOT NULL DEFAULT '',
                prior_refs_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS evaluation_jobs (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL, seed INTEGER NOT NULL, budget INTEGER NOT NULL,
                idempotency_key TEXT UNIQUE NOT NULL, status TEXT NOT NULL,
                attempts INTEGER NOT NULL, max_attempts INTEGER NOT NULL,
                available_at TEXT NOT NULL, lease_expires_at TEXT, worker_id TEXT,
                result_id TEXT, error_code TEXT, error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_memories (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
                memory_type TEXT NOT NULL, subject TEXT NOT NULL, content TEXT NOT NULL,
                evidence_artifact_id TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_agent_memories_dataset
                ON agent_memories(dataset_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, tool_name TEXT NOT NULL,
                status TEXT NOT NULL, request_json TEXT NOT NULL,
                response_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS generation_plans (
                plan_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                generation INTEGER NOT NULL, sequence INTEGER NOT NULL,
                generation_strategy TEXT NOT NULL,
                parent_selection_policy TEXT NOT NULL,
                decision_reason TEXT NOT NULL,
                required_parent_count INTEGER NOT NULL DEFAULT 0,
                previous_feedback_json TEXT NOT NULL DEFAULT '{}',
                context_artifact_id TEXT NOT NULL DEFAULT '',
                prompt_artifact_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(run_id, generation, generation_strategy, sequence)
            );
            CREATE INDEX IF NOT EXISTS ix_generation_plans_run_generation
                ON generation_plans(run_id, generation, generation_strategy, sequence);
            CREATE TABLE IF NOT EXISTS trace_records (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, span_type TEXT NOT NULL,
                actor TEXT NOT NULL, status TEXT NOT NULL, payload_json TEXT NOT NULL,
                plan_id TEXT NOT NULL DEFAULT '', candidate_id TEXT NOT NULL DEFAULT '',
                evaluation_job_id TEXT NOT NULL DEFAULT '', skill_name TEXT NOT NULL DEFAULT '',
                context_refs_json TEXT NOT NULL DEFAULT '[]',
                tool_call_id TEXT NOT NULL DEFAULT '', latency_ms INTEGER NOT NULL DEFAULT 0,
                token_usage_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_type TEXT NOT NULL,
                required_capability TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                input_artifact_refs_json TEXT NOT NULL,
                output_artifact_refs_json TEXT NOT NULL,
                status TEXT NOT NULL, claimed_by TEXT NOT NULL,
                attempts INTEGER NOT NULL, max_attempts INTEGER NOT NULL,
                error_message TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                claim_token TEXT NOT NULL DEFAULT '',
                lease_expires_at TEXT,
                UNIQUE(run_id, idempotency_key),
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            CREATE INDEX IF NOT EXISTS ix_agent_tasks_run_status
                ON agent_tasks(run_id, status, created_at);
            """
        )
        self._add_column("tasks", "dataset_id", "TEXT NOT NULL DEFAULT ''")
        self._add_column("tasks", "generations", "INTEGER NOT NULL DEFAULT 2")
        self._add_column("tasks", "population_size", "INTEGER NOT NULL DEFAULT 3")
        self._add_column("tasks", "random_seed", "INTEGER NOT NULL DEFAULT 2024")
        self._add_column("runs", "dataset_id", "TEXT NOT NULL DEFAULT ''")
        self._add_column("runs", "pause_requested", "INTEGER NOT NULL DEFAULT 0")
        self._add_column("runs", "cancel_requested", "INTEGER NOT NULL DEFAULT 0")
        self._add_column("runs", "control_reason", "TEXT NOT NULL DEFAULT ''")
        self._add_column(
            "runs", "runtime_cursor_artifact_id", "TEXT NOT NULL DEFAULT ''"
        )
        self._add_column("runs", "runtime_owner_id", "TEXT NOT NULL DEFAULT ''")
        self._add_column("runs", "runtime_lease_expires_at", "TEXT")
        self._add_column("candidates", "generation", "INTEGER NOT NULL DEFAULT 0")
        self._add_column("candidates", "plan_id", "TEXT NOT NULL DEFAULT ''")
        self._add_column("candidates", "generation_strategy", "TEXT NOT NULL DEFAULT ''")
        self._add_column("candidates", "code_digest", "TEXT NOT NULL DEFAULT ''")
        self._add_column("candidates", "selected_parent_ids_json", "TEXT NOT NULL DEFAULT '[]'")
        self._add_column("candidates", "prior_refs_json", "TEXT NOT NULL DEFAULT '[]'")
        self._add_column("candidates", "created_at", "TEXT NOT NULL DEFAULT ''")
        self._add_column("checkpoints", "dataset_id", "TEXT NOT NULL DEFAULT ''")
        self._add_column("checkpoints", "prior_refs_json", "TEXT NOT NULL DEFAULT '[]'")
        self._add_column("agent_tasks", "claim_token", "TEXT NOT NULL DEFAULT ''")
        self._add_column("agent_tasks", "lease_expires_at", "TEXT")
        self.connection.execute(
            """CREATE INDEX IF NOT EXISTS ix_agent_tasks_lease
               ON agent_tasks(status, lease_expires_at)"""
        )
        duplicate_job = self.connection.execute(
            """SELECT candidate_id,COUNT(*) AS job_count FROM evaluation_jobs
               GROUP BY candidate_id HAVING COUNT(*)>1 LIMIT 1"""
        ).fetchone()
        if duplicate_job is not None:
            raise EvaluationIdentityConflictError(
                "SQLite schema upgrade 发现同一 Candidate 的多个 legacy logical jobs：{}；"
                "请先审计数据，不允许自动猜测 authoritative identity".format(
                    duplicate_job["candidate_id"]
                )
            )
        self.connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_evaluation_job_candidate
               ON evaluation_jobs(candidate_id)"""
        )
        self.connection.commit()

    def add_agent_task(self, task):
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO agent_tasks(
               id,run_id,task_type,required_capability,idempotency_key,
               input_artifact_refs_json,output_artifact_refs_json,status,
               claimed_by,attempts,max_attempts,error_message,created_at,updated_at,
               claim_token,lease_expires_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (task.id, task.run_id, task.task_type, task.required_capability.value,
             task.idempotency_key, json.dumps(task.input_artifact_refs),
             json.dumps(task.output_artifact_refs), task.status.value,
             task.claimed_by, task.attempts, task.max_attempts, task.error_message,
             _dt(task.created_at), _dt(task.updated_at), task.claim_token,
             _dt(task.lease_expires_at) if task.lease_expires_at else None),
        )
        self.connection.commit()
        if cursor.rowcount == 1:
            return task, True
        row = self.connection.execute(
            "SELECT * FROM agent_tasks WHERE run_id=? AND idempotency_key=?",
            (task.run_id, task.idempotency_key),
        ).fetchone()
        return self._agent_task(row), False

    def get_agent_task(self, task_id):
        row = self.connection.execute(
            "SELECT * FROM agent_tasks WHERE id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._agent_task(row)

    def agent_tasks_for_run(self, run_id):
        rows = self.connection.execute(
            "SELECT * FROM agent_tasks WHERE run_id=? ORDER BY created_at,id", (run_id,)
        ).fetchall()
        return [self._agent_task(row) for row in rows]

    def claim_agent_task(
        self, task_id, agent_name, now=None, lease_seconds=300, claim_token=""
    ):
        if lease_seconds <= 0:
            raise ValueError("AgentTask lease_seconds 必须大于 0")
        now = now or datetime.now(timezone.utc)
        token = claim_token or "agent-claim-{}".format(uuid.uuid4().hex)
        expires = now + timedelta(seconds=lease_seconds)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute(
                """UPDATE agent_tasks SET status=?,claimed_by=?,claim_token=?,
                   lease_expires_at=?,attempts=attempts+1,error_message='',updated_at=?
                   WHERE id=? AND status=? AND attempts<max_attempts
                     AND EXISTS(SELECT 1 FROM runs
                       WHERE runs.id=agent_tasks.run_id
                         AND runs.cancel_requested=0
                         AND runs.status IN (?,?))""",
                (AgentTaskStatus.CLAIMED.value, str(agent_name)[:120], token,
                 _dt(expires), _dt(now), task_id, AgentTaskStatus.PENDING.value,
                 RunStatus.PENDING.value, RunStatus.RUNNING.value),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "AgentTask 已被领取、重试耗尽或所属 Run 非 active"
                )
            self.connection.commit()
            return self.get_agent_task(task_id)
        except Exception:
            self.connection.rollback()
            raise

    def renew_agent_task_lease(self, task_id, claim_token, now, lease_seconds):
        if not claim_token:
            raise ValueError("AgentTask claim_token 不能为空")
        if lease_seconds <= 0:
            raise ValueError("AgentTask lease_seconds 必须大于 0")
        expires = now + timedelta(seconds=lease_seconds)
        cursor = self.connection.execute(
            """UPDATE agent_tasks SET lease_expires_at=?,updated_at=?
               WHERE id=? AND status=? AND claim_token=?
                 AND lease_expires_at IS NOT NULL AND lease_expires_at>?
                 AND EXISTS(SELECT 1 FROM runs
                   WHERE runs.id=agent_tasks.run_id AND runs.cancel_requested=0
                     AND runs.status IN (?,?))""",
            (_dt(expires), _dt(now), task_id, AgentTaskStatus.CLAIMED.value,
             claim_token, _dt(now), RunStatus.PENDING.value,
             RunStatus.RUNNING.value),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def complete_agent_task(
        self, task_id, claim_token, output_artifact_refs, now=None
    ):
        if not claim_token:
            raise ValueError("AgentTask claim_token 不能为空")
        now = now or datetime.now(timezone.utc)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute(
                """UPDATE agent_tasks SET status=?,output_artifact_refs_json=?,
                   claim_token='',lease_expires_at=NULL,updated_at=?
                   WHERE id=? AND status=? AND claim_token=?
                     AND lease_expires_at IS NOT NULL AND lease_expires_at>?
                     AND EXISTS(SELECT 1 FROM runs
                       WHERE runs.id=agent_tasks.run_id AND runs.cancel_requested=0
                         AND runs.status IN (?,?))""",
                (AgentTaskStatus.COMPLETED.value,
                 json.dumps(list(output_artifact_refs)), _dt(now), task_id,
                 AgentTaskStatus.CLAIMED.value, claim_token, _dt(now),
                 RunStatus.PENDING.value, RunStatus.RUNNING.value),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "AgentTask lease 已丢失：过期、接管、取消或非 active Run"
                )
            self.connection.commit()
            return self.get_agent_task(task_id)
        except Exception:
            self.connection.rollback()
            raise

    def fail_agent_task(self, task_id, claim_token, error_message, now=None):
        if not claim_token:
            raise ValueError("AgentTask claim_token 不能为空")
        now = now or datetime.now(timezone.utc)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT * FROM agent_tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            task = self._agent_task(row)
            status = (
                AgentTaskStatus.PENDING
                if task.attempts < task.max_attempts
                else AgentTaskStatus.FAILED
            )
            cursor = self.connection.execute(
                """UPDATE agent_tasks SET status=?,claimed_by='',claim_token='',
                   lease_expires_at=NULL,error_message=?,updated_at=?
                   WHERE id=? AND status=? AND claim_token=?
                     AND lease_expires_at IS NOT NULL AND lease_expires_at>?
                     AND EXISTS(SELECT 1 FROM runs
                       WHERE runs.id=agent_tasks.run_id AND runs.cancel_requested=0
                         AND runs.status IN (?,?))""",
                (status.value, str(error_message)[:2000], _dt(now), task_id,
                 AgentTaskStatus.CLAIMED.value, claim_token, _dt(now),
                 RunStatus.PENDING.value, RunStatus.RUNNING.value),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "AgentTask lease 已丢失：拒绝 stale failure 覆盖当前 owner"
                )
            self.connection.commit()
            return self.get_agent_task(task_id)
        except Exception:
            self.connection.rollback()
            raise

    def recover_orphan_agent_tasks(self, now, orphan_seconds=300):
        """回收 lease 已过期的 claim；旧 schema NULL lease 以 timeout 兜底。"""

        cutoff = now.astimezone(timezone.utc) - timedelta(seconds=orphan_seconds)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self.connection.execute(
                """SELECT agent_tasks.*,runs.status AS run_status,
                          runs.cancel_requested AS run_cancel_requested
                   FROM agent_tasks JOIN runs ON runs.id=agent_tasks.run_id
                   WHERE agent_tasks.status=? AND (
                     (agent_tasks.lease_expires_at IS NOT NULL
                      AND agent_tasks.lease_expires_at<=?) OR
                     (agent_tasks.lease_expires_at IS NULL
                      AND agent_tasks.updated_at<=?))""",
                (AgentTaskStatus.CLAIMED.value, _dt(now), _dt(cutoff)),
            ).fetchall()
            recovered = 0
            for row in rows:
                if bool(row["run_cancel_requested"]) or row["run_status"] in {
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                }:
                    target = AgentTaskStatus.CANCELLED
                    message = "所属 Run 已终止，过期 AgentTask 已取消"
                elif row["attempts"] >= row["max_attempts"]:
                    target = AgentTaskStatus.FAILED
                    message = "AgentTask lease 过期且已达到最大尝试次数"
                else:
                    target = AgentTaskStatus.PENDING
                    message = "AgentTask lease 过期，已重新入队"
                recovered += self.connection.execute(
                    """UPDATE agent_tasks SET status=?,claimed_by='',claim_token='',
                       lease_expires_at=NULL,error_message=?,updated_at=?
                       WHERE id=? AND status=? AND claim_token=?""",
                    (target.value, message, _dt(now), row["id"],
                     AgentTaskStatus.CLAIMED.value, row["claim_token"]),
                ).rowcount
            self.connection.commit()
            return recovered
        except Exception:
            self.connection.rollback()
            raise

    def recover_orphan_runs(self, now):
        cursor = self.connection.execute(
            """UPDATE runs SET runtime_owner_id='',runtime_lease_expires_at=NULL
               WHERE status IN (?,?) AND runtime_owner_id<>''
                 AND runtime_lease_expires_at IS NOT NULL
                 AND runtime_lease_expires_at<=?""",
            (RunStatus.PENDING.value,RunStatus.RUNNING.value,_dt(now)),
        )
        self.connection.commit()
        return cursor.rowcount

    @staticmethod
    def _agent_task(row):
        return AgentTask(
            row["id"], row["run_id"], row["task_type"],
            AgentCapability(row["required_capability"]), row["idempotency_key"],
            json.loads(row["input_artifact_refs_json"]),
            json.loads(row["output_artifact_refs_json"]),
            AgentTaskStatus(row["status"]), row["claimed_by"], row["attempts"],
            row["max_attempts"], row["error_message"],
            datetime.fromisoformat(row["created_at"]),
            datetime.fromisoformat(row["updated_at"]),
            row["claim_token"] or "",
            (
                datetime.fromisoformat(row["lease_expires_at"])
                if row["lease_expires_at"] else None
            ),
        )

    def add_agent_memory(self, memory):
        self.connection.execute(
            """INSERT OR REPLACE INTO agent_memories
               VALUES(?,?,?,?,?,?,?,?)""",
            (memory.id, memory.run_id, memory.dataset_id, memory.memory_type,
             memory.subject, memory.content, memory.evidence_artifact_id,
             _dt(memory.created_at)),
        )
        self.connection.commit()

    def agent_memories_for_dataset(self, dataset_id, limit=5, exclude_run_id=""):
        query = "SELECT * FROM agent_memories WHERE dataset_id=?"
        params = [dataset_id]
        if exclude_run_id:
            query += " AND run_id<>?"
            params.append(exclude_run_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        rows = self.connection.execute(query, tuple(params)).fetchall()
        return [AgentMemory(
            row["id"], row["run_id"], row["dataset_id"], row["memory_type"],
            row["subject"], row["content"], row["evidence_artifact_id"],
            datetime.fromisoformat(row["created_at"]),
        ) for row in rows]

    def agent_memories_for_run(self, run_id, scope, limit=30):
        memory_types = memory_types_for_scope(scope)
        placeholders = ",".join("?" for _ in memory_types)
        query = (
            "SELECT * FROM agent_memories WHERE run_id=? "
            "AND memory_type IN ({}) "
            "ORDER BY created_at DESC,id DESC LIMIT ?"
        ).format(placeholders)
        rows = self.connection.execute(
            query, (run_id, *memory_types, max(1, int(limit)))
        ).fetchall()
        rows.reverse()
        return [AgentMemory(
            row["id"], row["run_id"], row["dataset_id"], row["memory_type"],
            row["subject"], row["content"], row["evidence_artifact_id"],
            datetime.fromisoformat(row["created_at"]),
        ) for row in rows]

    def record_tool_call(self, record):
        self.connection.execute(
            """INSERT INTO tool_calls VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET status=excluded.status,
               response_json=excluded.response_json""",
            (record.id, record.run_id, record.tool_name, record.status,
             json.dumps(record.request, ensure_ascii=False, sort_keys=True),
             json.dumps(record.response, ensure_ascii=False, sort_keys=True),
             _dt(record.created_at)),
        )
        self.connection.commit()

    def tool_calls_for_run(self, run_id):
        rows = self.connection.execute(
            # Windows 的系统时钟可能让连续调用得到完全相同的 timestamp；SQLite
            # 用插入 rowid 作为同 tick 的审计顺序，不让随机 UUID 伪造先后关系。
            "SELECT * FROM tool_calls WHERE run_id=? ORDER BY created_at,rowid",
            (run_id,),
        ).fetchall()
        return [ToolCallRecord(
            row["id"], row["run_id"], row["tool_name"], row["status"],
            json.loads(row["request_json"]), json.loads(row["response_json"]),
            datetime.fromisoformat(row["created_at"]),
        ) for row in rows]

    def add_generation_plan(self, plan):
        feedback = (
            plan.previous_feedback.__dict__
            if plan.previous_feedback is not None
            else {}
        )
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO generation_plans(
               plan_id,run_id,generation,sequence,generation_strategy,
               parent_selection_policy,decision_reason,required_parent_count,
               previous_feedback_json,context_artifact_id,prompt_artifact_id,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                plan.plan_id,
                plan.run_id,
                plan.generation,
                plan.sequence,
                plan.generation_strategy,
                plan.parent_selection_policy,
                plan.decision_reason,
                plan.required_parent_count,
                json.dumps(feedback, ensure_ascii=False, sort_keys=True),
                plan.context_artifact_id,
                plan.prompt_artifact_id,
                _dt(plan.created_at),
            ),
        )
        self.connection.commit()
        return self.generation_plan_by_id(plan.plan_id), cursor.rowcount == 1

    def generation_plan_by_id(self, plan_id):
        row = self.connection.execute(
            "SELECT * FROM generation_plans WHERE plan_id=?", (plan_id,)
        ).fetchone()
        if row is None:
            raise KeyError(plan_id)
        return self._generation_plan(row)

    def generation_plans_for_run(self, run_id):
        rows = self.connection.execute(
            """SELECT * FROM generation_plans WHERE run_id=?
               ORDER BY generation,generation_strategy,sequence""",
            (run_id,),
        ).fetchall()
        return [self._generation_plan(row) for row in rows]

    def record_trace(self, record):
        self.connection.execute(
            """INSERT INTO trace_records(
               id,run_id,span_type,actor,status,payload_json,plan_id,candidate_id,
               evaluation_job_id,skill_name,context_refs_json,tool_call_id,
               latency_ms,token_usage_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET status=excluded.status,
               payload_json=excluded.payload_json,latency_ms=excluded.latency_ms,
               token_usage_json=excluded.token_usage_json""",
            (
                record.id,
                record.run_id,
                record.span_type,
                record.actor,
                record.status,
                json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
                record.plan_id,
                record.candidate_id,
                record.evaluation_job_id,
                record.skill_name,
                json.dumps(record.context_refs, ensure_ascii=False),
                record.tool_call_id,
                record.latency_ms,
                json.dumps(record.token_usage, ensure_ascii=False, sort_keys=True),
                _dt(record.created_at),
            ),
        )
        self.connection.commit()

    def trace_records_for_run(self, run_id):
        rows = self.connection.execute(
            "SELECT * FROM trace_records WHERE run_id=? ORDER BY created_at,id",
            (run_id,),
        ).fetchall()
        return [self._trace(row) for row in rows]

    def _generation_plan(self, row):
        feedback_payload = json.loads(row["previous_feedback_json"] or "{}")
        feedback = (
            PreviousPlanFeedback(**feedback_payload)
            if feedback_payload else None
        )
        return GenerationPlan(
            plan_id=row["plan_id"],
            run_id=row["run_id"],
            generation=row["generation"],
            sequence=row["sequence"],
            generation_strategy=row["generation_strategy"],
            parent_selection_policy=row["parent_selection_policy"],
            decision_reason=row["decision_reason"],
            required_parent_count=row["required_parent_count"],
            previous_feedback=feedback,
            context_artifact_id=row["context_artifact_id"],
            prompt_artifact_id=row["prompt_artifact_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _trace(self, row):
        return TraceRecord(
            id=row["id"],
            run_id=row["run_id"],
            span_type=row["span_type"],
            actor=row["actor"],
            status=row["status"],
            payload=json.loads(row["payload_json"] or "{}"),
            plan_id=row["plan_id"],
            candidate_id=row["candidate_id"],
            evaluation_job_id=row["evaluation_job_id"],
            skill_name=row["skill_name"],
            context_refs=json.loads(row["context_refs_json"] or "[]"),
            tool_call_id=row["tool_call_id"],
            latency_ms=row["latency_ms"],
            token_usage=json.loads(row["token_usage_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _add_column(self, table, name, definition):
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info({})".format(table))}
        if name not in columns:
            self.connection.execute("ALTER TABLE {} ADD COLUMN {} {}".format(table, name, definition))

    def add_task(self, task: OptimizationTask) -> None:
        self.connection.execute(
            """INSERT INTO tasks(id,name,objective,evaluation_budget,total_budget,created_at,
               dataset_id,generations,population_size,random_seed) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (task.id, task.name, task.objective, task.evaluation_budget, task.total_budget,
             _dt(task.created_at), task.dataset_id, task.generations,
             task.population_size, task.random_seed),
        )
        self.connection.commit()

    def create_task_run(self, task, run, event_type, message, payload):
        event = Event(1, run.id, event_type, message, dict(payload))
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                """INSERT INTO tasks(id,name,objective,evaluation_budget,total_budget,created_at,
                   dataset_id,generations,population_size,random_seed) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (task.id,task.name,task.objective,task.evaluation_budget,task.total_budget,
                 _dt(task.created_at),task.dataset_id,task.generations,
                 task.population_size,task.random_seed),
            )
            self.connection.execute(
                """INSERT INTO runs(id,task_id,status,generation,consumed_evaluations,
                   reserved_evaluations,best_candidate_id,created_at,updated_at,dataset_id)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (run.id,run.task_id,run.status.value,run.generation,run.consumed_evaluations,
                 run.reserved_evaluations,run.best_candidate_id,_dt(run.created_at),
                 _dt(run.updated_at),run.dataset_id),
            )
            self.connection.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?)",
                (run.id,1,event_type,message,
                 json.dumps(payload,ensure_ascii=False,sort_keys=True),_dt(event.occurred_at)),
            )
            self.connection.commit()
            return event
        except Exception:
            self.connection.rollback()
            raise

    def add_run(self, run: Run) -> None:
        self.connection.execute(
            """INSERT INTO runs(id,task_id,status,generation,consumed_evaluations,
               reserved_evaluations,best_candidate_id,created_at,updated_at,dataset_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run.id, run.task_id, run.status.value, run.generation, run.consumed_evaluations,
             run.reserved_evaluations, run.best_candidate_id, _dt(run.created_at),
             _dt(run.updated_at), run.dataset_id),
        )
        self.connection.commit()

    def get_task(self, task_id: str) -> OptimizationTask:
        row = self.connection.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return OptimizationTask(
            id=row["id"], name=row["name"], objective=row["objective"],
            evaluation_budget=row["evaluation_budget"], total_budget=row["total_budget"],
            created_at=datetime.fromisoformat(row["created_at"]),
            dataset_id=row["dataset_id"], generations=row["generations"],
            population_size=row["population_size"], random_seed=row["random_seed"],
        )

    def get_run(self, run_id: str) -> Run:
        row = self.connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return Run(
            id=row["id"], task_id=row["task_id"], status=RunStatus(row["status"]),
            generation=row["generation"], consumed_evaluations=row["consumed_evaluations"],
            reserved_evaluations=row["reserved_evaluations"],
            best_candidate_id=row["best_candidate_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            dataset_id=row["dataset_id"],
            pause_requested=bool(row["pause_requested"]),
            cancel_requested=bool(row["cancel_requested"]),
            control_reason=row["control_reason"],
            runtime_cursor_artifact_id=row["runtime_cursor_artifact_id"],
            runtime_owner_id=row["runtime_owner_id"],
            runtime_lease_expires_at=(
                datetime.fromisoformat(row["runtime_lease_expires_at"])
                if row["runtime_lease_expires_at"] else None
            ),
        )

    def list_runs(self) -> Iterable[Run]:
        rows = self.connection.execute(
            "SELECT id FROM runs ORDER BY created_at DESC"
        ).fetchall()
        return [self.get_run(row["id"]) for row in rows]

    def save_run(self, run: Run) -> None:
        """仅保存算法投影；控制、预算与 lease 只能由各自的原子命令修改。

        该兼容入口保留给旧 harness。产品 Runtime/Lifecycle 使用下方窄接口，
        从而避免一个过期 ``Run`` 快照复活 cancel 或回滚新的 lease/预算事实。
        """
        self.connection.execute(
            """UPDATE runs SET generation=?,best_candidate_id=?,updated_at=?
               WHERE id=?""",
            (run.generation, run.best_candidate_id, _dt(run.updated_at), run.id),
        )
        self.connection.commit()

    def start_run(self, run_id, expected_status, now, owner_id=""):
        expected = (
            expected_status.value
            if isinstance(expected_status, RunStatus)
            else str(expected_status)
        )
        query = (
            "UPDATE runs SET status=?,pause_requested=0,control_reason='',updated_at=? "
            "WHERE id=? AND status=? AND cancel_requested=0"
        )
        params = [RunStatus.RUNNING.value, _dt(now), run_id, expected]
        if owner_id:
            query += " AND runtime_owner_id=? AND runtime_lease_expires_at>?"
            params.extend((owner_id, _dt(now)))
        else:
            # 控制面 resume 没有 owner token，只能启动当前确实无人持有的 Run。
            query += " AND runtime_owner_id=''"
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            updated = self.connection.execute(query, tuple(params)).rowcount
            if updated != 1:
                raise RuntimeError("Run 启动 CAS 失败：状态、取消请求或 Runtime lease 已变化")
            self.connection.commit()
            return self.get_run(run_id)
        except Exception:
            self.connection.rollback()
            raise

    def pause_run(self, run_id, owner_id, now):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            updated = self.connection.execute(
                """UPDATE runs SET status=?,pause_requested=0,
                   runtime_owner_id='',runtime_lease_expires_at=NULL,updated_at=?
                   WHERE id=? AND status=? AND runtime_owner_id=?
                     AND runtime_lease_expires_at>? AND cancel_requested=0""",
                (RunStatus.PAUSED.value, _dt(now), run_id,
                 RunStatus.RUNNING.value, owner_id, _dt(now)),
            ).rowcount
            if updated != 1:
                raise RuntimeError("Run 暂停 CAS 失败：Runtime lease 或控制状态已变化")
            self.connection.commit()
            return self.get_run(run_id)
        except Exception:
            self.connection.rollback()
            raise

    def complete_run(self, run_id, owner_id, now, generation, best_candidate_id):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            active_jobs = self.connection.execute(
                """SELECT id,candidate_id,budget FROM evaluation_jobs
                   WHERE run_id=? AND status IN (?,?,?)""",
                (run_id, EvaluationJobStatus.PENDING.value,
                 EvaluationJobStatus.RETRY_WAIT.value,
                 EvaluationJobStatus.RUNNING.value),
            ).fetchall()
            released = sum(int(row["budget"]) for row in active_jobs)
            updated = self.connection.execute(
                """UPDATE runs SET status=?,generation=?,best_candidate_id=?,
                   reserved_evaluations=0,runtime_owner_id='',
                   runtime_lease_expires_at=NULL,updated_at=?
                   WHERE id=? AND status=? AND runtime_owner_id=?
                     AND runtime_lease_expires_at>? AND cancel_requested=0
                     AND pause_requested=0 AND reserved_evaluations=?""",
                (RunStatus.COMPLETED.value, int(generation), best_candidate_id,
                 _dt(now), run_id, RunStatus.RUNNING.value, owner_id, _dt(now),
                 released),
            ).rowcount
            if updated != 1:
                raise RuntimeError(
                    "Run 完成 CAS 失败：Runtime lease、控制状态或预算对账已变化"
                )
            self._cancel_terminal_work(
                run_id, now, "RUN_COMPLETED", "Run 已完成，未结算工作已终止"
            )
            self.connection.commit()
            return self.get_run(run_id)
        except Exception:
            self.connection.rollback()
            raise

    def fail_run(self, run_id, reason, now, owner_id=""):
        terminal_jobs = (
            EvaluationJobStatus.PENDING.value,
            EvaluationJobStatus.RETRY_WAIT.value,
            EvaluationJobStatus.RUNNING.value,
        )
        query = (
            "UPDATE runs SET status=?,control_reason=?,reserved_evaluations=0,"
            "runtime_owner_id='',runtime_lease_expires_at=NULL,updated_at=? "
            "WHERE id=? AND status IN (?,?) AND cancel_requested=0 "
            "AND reserved_evaluations=?"
        )
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            active_jobs = self.connection.execute(
                """SELECT id,candidate_id,budget FROM evaluation_jobs
                   WHERE run_id=? AND status IN (?,?,?)""",
                (run_id, *terminal_jobs),
            ).fetchall()
            released = sum(int(row["budget"]) for row in active_jobs)
            params = [
                RunStatus.FAILED.value, str(reason)[:2000], _dt(now), run_id,
                RunStatus.PENDING.value, RunStatus.RUNNING.value, released,
            ]
            if owner_id:
                query += " AND runtime_owner_id=? AND runtime_lease_expires_at>?"
                params.extend((owner_id, _dt(now)))
            else:
                # Facade 没有 owner token 时只能结算无人持有的 Run；否则一个刚
                # 失败/释放的旧 executor 可把新 owner 已接管的 RUNNING 覆成 FAILED。
                query += " AND runtime_owner_id=''"
            updated = self.connection.execute(query, tuple(params)).rowcount
            if updated != 1:
                raise RuntimeError(
                    "Run 失败 CAS 未命中：状态、取消请求、Runtime lease 或预算对账已变化"
                )
            self._cancel_terminal_work(
                run_id, now, "RUN_FAILED", "Run 已失败，未结算工作已终止"
            )
            self.connection.commit()
            return self.get_run(run_id)
        except Exception:
            self.connection.rollback()
            raise

    def _cancel_terminal_work(self, run_id, now, error_code, message):
        """在调用方已持有写事务时终止 Run 的全部未结算工作。"""

        job_statuses = (
            EvaluationJobStatus.PENDING.value,
            EvaluationJobStatus.RETRY_WAIT.value,
            EvaluationJobStatus.RUNNING.value,
        )
        self.connection.execute(
            """UPDATE candidates SET status=? WHERE id IN(
                 SELECT candidate_id FROM evaluation_jobs
                 WHERE run_id=? AND status IN (?,?,?))""",
            (CandidateStatus.INVALID.value, run_id, *job_statuses),
        )
        self.connection.execute(
            """UPDATE evaluation_jobs SET status=?,worker_id=NULL,
               lease_expires_at=NULL,error_code=?,error_message=?
               WHERE run_id=? AND status IN (?,?,?)""",
            (EvaluationJobStatus.CANCELLED.value, error_code, message, run_id,
             *job_statuses),
        )
        self.connection.execute(
            """UPDATE agent_tasks SET status=?,claimed_by='',claim_token='',
               lease_expires_at=NULL,error_message=?,updated_at=?
               WHERE run_id=? AND status IN (?,?)""",
            (AgentTaskStatus.CANCELLED.value, message, _dt(now), run_id,
             AgentTaskStatus.PENDING.value, AgentTaskStatus.CLAIMED.value),
        )

    def update_run_progress(self, run_id, owner_id, now, generation):
        cursor = self.connection.execute(
            """UPDATE runs SET generation=?,updated_at=? WHERE id=? AND status=?
               AND runtime_owner_id=? AND runtime_lease_expires_at>?
               AND cancel_requested=0""",
            (int(generation), _dt(now), run_id, RunStatus.RUNNING.value,
             owner_id, _dt(now)),
        )
        self.connection.commit()
        if cursor.rowcount != 1:
            raise RuntimeError("Run 进度 fencing 失败：Runtime lease 或控制状态已变化")
        return self.get_run(run_id)

    def request_run_pause(self, run_id: str, reason: str) -> Run:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            updated = self.connection.execute(
                """UPDATE runs SET pause_requested=1,control_reason=?,updated_at=?
                   WHERE id=? AND status=? AND cancel_requested=0""",
                (str(reason)[:2000], _dt(datetime.now().astimezone()), run_id,
                 RunStatus.RUNNING.value),
            ).rowcount
            if updated != 1:
                raise RuntimeError("Run 当前状态不允许请求暂停")
            self.connection.commit()
            return self.get_run(run_id)
        except Exception:
            self.connection.rollback()
            raise

    def request_run_cancel(self, run_id: str, reason: str) -> Run:
        """原子取消未认领工作，并且每份 reservation 只释放一次。"""

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT * FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["status"] not in {
                RunStatus.PENDING.value,
                RunStatus.RUNNING.value,
                RunStatus.PAUSED.value,
            }:
                raise RuntimeError("Run 当前状态不允许取消")
            cancellable = (
                EvaluationJobStatus.PENDING.value,
                EvaluationJobStatus.RETRY_WAIT.value,
            )
            released = self.connection.execute(
                """SELECT COALESCE(SUM(budget),0) AS amount
                   FROM evaluation_jobs WHERE run_id=? AND status IN (?,?)""",
                (run_id, *cancellable),
            ).fetchone()["amount"]
            self.connection.execute(
                """UPDATE candidates SET status=? WHERE id IN(
                     SELECT candidate_id FROM evaluation_jobs
                     WHERE run_id=? AND status IN (?,?))""",
                (CandidateStatus.INVALID.value, run_id, *cancellable),
            )
            self.connection.execute(
                """UPDATE evaluation_jobs SET status=?,worker_id=NULL,
                   lease_expires_at=NULL,error_code='RUN_CANCELLED',
                   error_message='Run 已取消' WHERE run_id=? AND status IN (?,?)""",
                (EvaluationJobStatus.CANCELLED.value, run_id, *cancellable),
            )
            self.connection.execute(
                """UPDATE agent_tasks SET status=?,claimed_by='',claim_token='',
                   lease_expires_at=NULL,error_message='Run 已取消',updated_at=?
                   WHERE run_id=? AND status IN (?,?)""",
                (AgentTaskStatus.CANCELLED.value, _dt(datetime.now(timezone.utc)),
                 run_id, AgentTaskStatus.PENDING.value,
                 AgentTaskStatus.CLAIMED.value),
            )
            updated = self.connection.execute(
                """UPDATE runs SET status=?,cancel_requested=1,pause_requested=0,
                   control_reason=?,reserved_evaluations=reserved_evaluations-?,
                   runtime_owner_id='',runtime_lease_expires_at=NULL,updated_at=?
                   WHERE id=? AND reserved_evaluations>=?""",
                (RunStatus.CANCELLED.value, str(reason)[:2000], int(released),
                 _dt(datetime.now().astimezone()), run_id, int(released)),
            ).rowcount
            if updated != 1:
                raise RuntimeError("取消时预算 reservation 对账失败")
            self.connection.commit()
            return self.get_run(run_id)
        except Exception:
            self.connection.rollback()
            raise

    def claim_run_lease(self, run_id, owner_id, now, lease_seconds):
        from datetime import timedelta

        if lease_seconds <= 0:
            raise ValueError("runtime lease_seconds 必须大于 0")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            expires = now + timedelta(seconds=lease_seconds)
            updated = self.connection.execute(
                """UPDATE runs SET runtime_owner_id=?,runtime_lease_expires_at=?
                   WHERE id=? AND status IN (?,?) AND cancel_requested=0
                     AND (runtime_owner_id='' OR runtime_owner_id=?
                          OR runtime_lease_expires_at IS NULL
                          OR runtime_lease_expires_at<=?)""",
                (owner_id, _dt(expires), run_id, RunStatus.PENDING.value,
                 RunStatus.RUNNING.value, owner_id, _dt(now)),
            ).rowcount
            self.connection.commit()
            return updated == 1
        except Exception:
            self.connection.rollback()
            raise

    def renew_run_lease(self, run_id, owner_id, now, lease_seconds):
        from datetime import timedelta

        expires = now + timedelta(seconds=lease_seconds)
        cursor = self.connection.execute(
            """UPDATE runs SET runtime_lease_expires_at=? WHERE id=?
               AND runtime_owner_id=? AND status=? AND cancel_requested=0
               AND runtime_lease_expires_at>?""",
            (_dt(expires), run_id, owner_id, RunStatus.RUNNING.value, _dt(now)),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def release_run_lease(self, run_id, owner_id):
        cursor = self.connection.execute(
            """UPDATE runs SET runtime_owner_id='',runtime_lease_expires_at=NULL
               WHERE id=? AND runtime_owner_id=?""",
            (run_id, owner_id),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def update_runtime_cursor(self, run_id, artifact_id, owner_id="", now=None):
        query = "UPDATE runs SET runtime_cursor_artifact_id=? WHERE id=?"
        params = [artifact_id, run_id]
        if owner_id:
            now = now or datetime.now().astimezone()
            query += (
                " AND runtime_owner_id=? AND status=? AND cancel_requested=0"
                " AND runtime_lease_expires_at>?"
            )
            params.extend((owner_id, RunStatus.RUNNING.value, _dt(now)))
        cursor = self.connection.execute(query, tuple(params))
        self.connection.commit()
        return cursor.rowcount == 1

    def add_candidate(self, candidate: Candidate) -> None:
        self.connection.execute(
            """INSERT INTO candidates(
               id,run_id,code,description,operators_json,lineage_json,status,
               objective,code_artifact_id,evaluation_artifact_id,generation,
               plan_id,generation_strategy,code_digest,selected_parent_ids_json,
               prior_refs_json,created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET status=excluded.status,
               objective=excluded.objective, code_artifact_id=excluded.code_artifact_id,
               evaluation_artifact_id=excluded.evaluation_artifact_id,
               generation=excluded.generation,plan_id=excluded.plan_id,
               generation_strategy=excluded.generation_strategy,
               code_digest=excluded.code_digest,
               selected_parent_ids_json=excluded.selected_parent_ids_json,
               prior_refs_json=excluded.prior_refs_json""",
            (candidate.id, candidate.run_id, candidate.code, candidate.description,
             json.dumps(candidate.operators, ensure_ascii=False),
             json.dumps(candidate.lineage, ensure_ascii=False, sort_keys=True),
             candidate.status.value, candidate.objective, candidate.code_artifact_id,
             candidate.evaluation_artifact_id, candidate.generation,
             candidate.plan_id, candidate.generation_strategy,
             candidate.code_digest, json.dumps(candidate.selected_parent_ids, ensure_ascii=False),
             json.dumps(candidate.prior_refs, ensure_ascii=False), _dt(candidate.created_at)),
        )
        self.connection.commit()

    def _candidate(self, row: sqlite3.Row) -> Candidate:
        lineage = json.loads(row["lineage_json"])
        return Candidate(
            id=row["id"], run_id=row["run_id"], code=row["code"],
            description=row["description"], operators=json.loads(row["operators_json"]),
            lineage=lineage, status=CandidateStatus(row["status"]),
            objective=row["objective"], code_artifact_id=row["code_artifact_id"],
            evaluation_artifact_id=row["evaluation_artifact_id"],
            generation=int(row["generation"] or lineage.get("generation", 0) or 0),
            plan_id=row["plan_id"] or str(lineage.get("plan_id", "")),
            generation_strategy=(
                row["generation_strategy"] or str(lineage.get("operator", ""))
            ),
            code_digest=row["code_digest"] or "",
            selected_parent_ids=(
                json.loads(row["selected_parent_ids_json"])
                if row["selected_parent_ids_json"] else list(lineage.get("parents", []))
            ),
            prior_refs=(
                json.loads(row["prior_refs_json"])
                if row["prior_refs_json"] else list(lineage.get("prior_refs", []))
            ),
            created_at=(
                datetime.fromisoformat(row["created_at"])
                if row["created_at"] else datetime.now(timezone.utc)
            ),
        )

    def candidates_for_run(self, run_id: str) -> Iterable[Candidate]:
        rows = self.connection.execute(
            "SELECT * FROM candidates WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [self._candidate(row) for row in rows]

    def candidate_by_id(self, candidate_id: str) -> Candidate:
        row = self.connection.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return self._candidate(row)

    def add_result(self, result: EvaluationResult) -> None:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            candidate = self.connection.execute(
                "SELECT run_id FROM candidates WHERE id=?", (result.candidate_id,)
            ).fetchone()
            if candidate is None:
                raise KeyError(result.candidate_id)
            if candidate["run_id"] != result.run_id:
                raise EvaluationIdentityConflictError(
                    "EvaluationResult 与 Candidate 的 run 归属不一致"
                )
            existing = self.connection.execute(
                "SELECT * FROM evaluation_results WHERE candidate_id=? OR id=?",
                (result.candidate_id, result.id),
            ).fetchall()
            if existing:
                if len(existing) != 1 or not _same_result(existing[0], result):
                    raise EvaluationIdentityConflictError(
                        "Candidate 已有不同的 authoritative EvaluationResult；重评必须 clone Candidate"
                    )
                self.connection.commit()
                return
            inserted = self.connection.execute(
                "INSERT INTO evaluation_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                (result.id, result.run_id, result.candidate_id, result.objective,
                 json.dumps(result.trajectory),
                 json.dumps(result.best_configuration, ensure_ascii=False, sort_keys=True),
                 result.used_budget),
            ).rowcount
            if inserted != 1:
                raise RuntimeError("EvaluationResult 插入未产生唯一 authoritative fact")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def result_for_candidate(self, candidate_id: str) -> EvaluationResult:
        row = self.connection.execute(
            "SELECT * FROM evaluation_results WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return EvaluationResult(
            id=row["id"], run_id=row["run_id"], candidate_id=row["candidate_id"],
            objective=row["objective"], trajectory=json.loads(row["trajectory_json"]),
            best_configuration=json.loads(row["best_configuration_json"]),
            used_budget=row["used_budget"],
        )

    def append_event(self, run_id: str, event_type: str, message: str, **payload: object) -> Event:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            sequence = self.connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            event = Event(sequence, run_id, event_type, message, dict(payload))
            self.connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, sequence, event_type, message,
                 json.dumps(payload, ensure_ascii=False, sort_keys=True), _dt(event.occurred_at)),
            )
            self.connection.commit()
            return event
        except Exception:
            self.connection.rollback()
            raise

    def events_for_run(self, run_id: str) -> Iterable[Event]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY sequence", (run_id,)
        ).fetchall()
        return [Event(row["sequence"], row["run_id"], row["event_type"], row["message"],
                      json.loads(row["payload_json"]), datetime.fromisoformat(row["occurred_at"]))
                for row in rows]

    def put_artifact(self, run_id: str, kind: str, content: bytes, media_type: str) -> ArtifactMetadata:
        import hashlib
        digest = hashlib.sha256(content).hexdigest()
        run_key = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
        artifact_id = "artifact-{}-{}-{}".format(run_key, kind.lower(), digest[:16])
        run_root = self.artifact_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        path = run_root / artifact_id
        temp_path = run_root / (artifact_id + ".tmp-{}".format(os.getpid()))
        temp_path.write_bytes(content)
        if hashlib.sha256(temp_path.read_bytes()).hexdigest() != digest:
            temp_path.unlink(missing_ok=True)
            raise ArtifactIntegrityError("artifact digest 校验失败")
        os.replace(str(temp_path), str(path))
        metadata = ArtifactMetadata(artifact_id, run_id, kind, media_type, len(content), digest, str(path.resolve()))
        self.connection.execute(
            "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (metadata.id, metadata.run_id, metadata.kind, metadata.media_type,
             metadata.size, metadata.digest, metadata.uri),
        )
        self.connection.commit()
        return metadata

    def artifact_content(self, artifact_id: str) -> bytes:
        row = self.connection.execute("SELECT uri, digest FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        try:
            content = Path(row["uri"]).read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(
                "artifact 文件缺失：{}".format(artifact_id)
            ) from exc
        import hashlib
        if hashlib.sha256(content).hexdigest() != row["digest"]:
            raise ArtifactIntegrityError("artifact 已损坏：{}".format(artifact_id))
        return content

    def artifacts_for_run(self, run_id: str) -> Iterable[ArtifactMetadata]:
        rows = self.connection.execute("SELECT * FROM artifacts WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        return [ArtifactMetadata(row["id"], row["run_id"], row["kind"], row["media_type"],
                                 row["size"], row["digest"], row["uri"]) for row in rows]

    def add_checkpoint(self, checkpoint: CheckpointMetadata) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO checkpoints(id,run_id,generation,population_ids_json,consumed_budget,
               remaining_budget,artifact_id,schema_version,code_version,created_at,dataset_id,
               prior_refs_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (checkpoint.id, checkpoint.run_id, checkpoint.generation,
             json.dumps(checkpoint.population_ids), checkpoint.consumed_budget,
             checkpoint.remaining_budget, checkpoint.artifact_id,
             checkpoint.schema_version, checkpoint.code_version, _dt(checkpoint.created_at),
             checkpoint.dataset_id, json.dumps(checkpoint.prior_refs)),
        )
        self.connection.commit()

    def latest_checkpoint(self, run_id: str) -> CheckpointMetadata:
        row = self.connection.execute(
            """SELECT checkpoints.* FROM checkpoints
               JOIN runs ON runs.id=checkpoints.run_id
               WHERE checkpoints.run_id=?
               ORDER BY CASE
                 WHEN runs.runtime_cursor_artifact_id<>''
                  AND checkpoints.artifact_id=runs.runtime_cursor_artifact_id
                 THEN 0 WHEN runs.runtime_cursor_artifact_id='' THEN 0 ELSE 1 END,
                 checkpoints.created_at DESC,checkpoints.rowid DESC LIMIT 1""",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return CheckpointMetadata(
            row["id"], row["run_id"], row["generation"],
            json.loads(row["population_ids_json"]), row["consumed_budget"],
            row["remaining_budget"], row["artifact_id"], row["schema_version"],
            row["code_version"], datetime.fromisoformat(row["created_at"]),
            row["dataset_id"], json.loads(row["prior_refs_json"]),
        )

    def _job(self, row: sqlite3.Row) -> EvaluationJob:
        return EvaluationJob(
            id=row["id"], run_id=row["run_id"], task_id=row["task_id"],
            candidate_id=row["candidate_id"], seed=row["seed"], budget=row["budget"],
            idempotency_key=row["idempotency_key"], status=EvaluationJobStatus(row["status"]),
            attempts=row["attempts"], max_attempts=row["max_attempts"],
            available_at=datetime.fromisoformat(row["available_at"]),
            lease_expires_at=(datetime.fromisoformat(row["lease_expires_at"]) if row["lease_expires_at"] else None),
            worker_id=row["worker_id"], result_id=row["result_id"],
            error_code=row["error_code"], error_message=row["error_message"],
        )

    def get_evaluation_job(self, job_id: str) -> EvaluationJob:
        row = self.connection.execute("SELECT * FROM evaluation_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job(row)

    def evaluation_jobs_for_run(self, run_id: str) -> Iterable[EvaluationJob]:
        rows = self.connection.execute(
            "SELECT * FROM evaluation_jobs WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [self._job(row) for row in rows]

    def enqueue_job(self, job: EvaluationJob) -> tuple:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            run_row = self.connection.execute("SELECT * FROM runs WHERE id=?", (job.run_id,)).fetchone()
            task_row = self.connection.execute("SELECT * FROM tasks WHERE id=?", (job.task_id,)).fetchone()
            candidate_row = self.connection.execute(
                "SELECT * FROM candidates WHERE id=?", (job.candidate_id,)
            ).fetchone()
            if run_row is None or task_row is None or candidate_row is None:
                raise KeyError(job.run_id)
            if run_row["task_id"] != job.task_id:
                raise EvaluationIdentityConflictError(
                    "EvaluationJob 的 Task 不属于目标 Run"
                )
            if candidate_row["run_id"] != job.run_id:
                raise EvaluationIdentityConflictError(
                    "EvaluationJob 的 Candidate 不属于目标 Run"
                )
            if (
                bool(run_row["cancel_requested"])
                or run_row["status"] != RunStatus.RUNNING.value
            ):
                raise RuntimeError(
                    "Run 非 RUNNING，拒绝创建或重放 EvaluationJob"
                )

            candidate_jobs = self.connection.execute(
                "SELECT * FROM evaluation_jobs WHERE candidate_id=? ORDER BY id",
                (job.candidate_id,),
            ).fetchall()
            if candidate_jobs:
                if (
                    len(candidate_jobs) == 1
                    and _same_job_identity(candidate_jobs[0], job)
                ):
                    self.connection.commit()
                    return self._job(candidate_jobs[0]), False
                raise EvaluationIdentityConflictError(
                    "Candidate 已绑定其他 logical evaluation identity；重评必须 clone Candidate"
                )

            key_collision = self.connection.execute(
                "SELECT * FROM evaluation_jobs WHERE idempotency_key=?",
                (job.idempotency_key,),
            ).fetchone()
            if key_collision is not None:
                raise EvaluationIdentityConflictError(
                    "EvaluationJob idempotency key 与其他 Candidate 冲突"
                )
            existing_result = self.connection.execute(
                "SELECT id FROM evaluation_results WHERE candidate_id=?",
                (job.candidate_id,),
            ).fetchone()
            if existing_result is not None:
                raise EvaluationIdentityConflictError(
                    "Candidate 已有 authoritative EvaluationResult，但缺少可验证的原 Job；重评必须 clone Candidate"
                )
            if job.budget <= 0 or job.max_attempts <= 0:
                raise ValueError("EvaluationJob budget/max_attempts 必须大于 0")
            available = task_row["total_budget"] - run_row["consumed_evaluations"] - run_row["reserved_evaluations"]
            if job.budget > available:
                raise BudgetExhaustedError(
                    "预算不足：需要 {}，剩余可预留 {}".format(job.budget, available)
                )
            self.connection.execute(
                "INSERT INTO evaluation_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job.id, job.run_id, job.task_id, job.candidate_id, job.seed, job.budget,
                 job.idempotency_key, job.status.value, job.attempts, job.max_attempts,
                 _dt(job.available_at), None, None, None, None, None),
            )
            candidate_updated = self.connection.execute(
                "UPDATE candidates SET status=? WHERE id=? AND run_id=?",
                (CandidateStatus.EVALUATION_PENDING.value, job.candidate_id, job.run_id),
            ).rowcount
            if candidate_updated != 1:
                raise RuntimeError("EvaluationJob Candidate 归属校验发生并发冲突")
            self.connection.execute(
                "UPDATE runs SET reserved_evaluations=reserved_evaluations+? WHERE id=?",
                (job.budget, job.run_id),
            )
            self.connection.commit()
            return job, True
        except Exception:
            self.connection.rollback()
            raise

    def claim_next_job(self, worker_id: str, now: datetime, lease_seconds: int):
        from datetime import timedelta
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                """SELECT evaluation_jobs.* FROM evaluation_jobs
                   JOIN runs ON runs.id=evaluation_jobs.run_id
                   WHERE evaluation_jobs.status IN (?, ?)
                     AND evaluation_jobs.available_at<=?
                     AND runs.cancel_requested=0 AND runs.status=?
                   ORDER BY evaluation_jobs.available_at, evaluation_jobs.id LIMIT 1""",
                (EvaluationJobStatus.PENDING.value, EvaluationJobStatus.RETRY_WAIT.value,
                 _dt(now), RunStatus.RUNNING.value),
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            lease = now + timedelta(seconds=lease_seconds)
            updated = self.connection.execute(
                """UPDATE evaluation_jobs SET status=?, attempts=attempts+1,
                   lease_expires_at=?, worker_id=? WHERE id=? AND status IN (?, ?)""",
                (EvaluationJobStatus.RUNNING.value, _dt(lease), worker_id, row["id"],
                 EvaluationJobStatus.PENDING.value, EvaluationJobStatus.RETRY_WAIT.value),
            ).rowcount
            if updated != 1:
                self.connection.rollback()
                return None
            self.connection.execute("UPDATE candidates SET status=? WHERE id=?",
                                    (CandidateStatus.EVALUATING.value, row["candidate_id"]))
            self.connection.commit()
            return self.get_evaluation_job(row["id"])
        except Exception:
            self.connection.rollback()
            raise

    def renew_job_lease(
        self, job_id: str, worker_id: str, now: datetime, lease_seconds: int
    ) -> bool:
        """仅允许当前、尚未过期的 RUNNING owner 延长 lease。"""

        from datetime import timedelta

        if lease_seconds <= 0:
            raise ValueError("lease_seconds 必须大于 0")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            lease = now + timedelta(seconds=lease_seconds)
            updated = self.connection.execute(
                """UPDATE evaluation_jobs SET lease_expires_at=?
                   WHERE id=? AND status=? AND worker_id=?
                     AND lease_expires_at IS NOT NULL AND lease_expires_at>?""",
                (
                    _dt(lease), job_id, EvaluationJobStatus.RUNNING.value,
                    worker_id, _dt(now),
                ),
            ).rowcount
            self.connection.commit()
            return updated == 1
        except Exception:
            self.connection.rollback()
            raise

    def complete_job_success(
        self, job_id: str, worker_id: str, result: EvaluationResult,
        artifact_id: str, now: datetime
    ) -> EvaluationJob:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute("SELECT * FROM evaluation_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            job = self._job(row)
            if (
                job.status != EvaluationJobStatus.RUNNING
                or job.worker_id != worker_id
                or job.lease_expires_at is None
                or job.lease_expires_at <= now
            ):
                raise RuntimeError("评价 lease 已丢失：仅当前 RUNNING owner 可以成功结算")
            run_row = self.connection.execute(
                "SELECT status,cancel_requested FROM runs WHERE id=?", (job.run_id,)
            ).fetchone()
            if (
                run_row is None
                or run_row["status"] != RunStatus.RUNNING.value
                or bool(run_row["cancel_requested"])
            ):
                raise RuntimeError("Run 已不处于 active 状态，拒绝评价成功结算")
            if result.run_id != job.run_id or result.candidate_id != job.candidate_id:
                raise RuntimeError("评价结果与 job 的 run/candidate 不一致")
            if result.used_budget < 0 or result.used_budget > job.budget:
                raise RuntimeError("评价实际预算必须位于 0 到预留预算之间")
            candidate_row = self.connection.execute(
                "SELECT run_id FROM candidates WHERE id=?", (job.candidate_id,)
            ).fetchone()
            if candidate_row is None or candidate_row["run_id"] != job.run_id:
                raise EvaluationIdentityConflictError(
                    "EvaluationJob 与 Candidate 的 durable 归属不一致"
                )
            existing_results = self.connection.execute(
                "SELECT id FROM evaluation_results WHERE candidate_id=? OR id=?",
                (result.candidate_id, result.id),
            ).fetchall()
            if existing_results:
                raise EvaluationIdentityConflictError(
                    "Candidate 或 Result ID 已绑定 authoritative EvaluationResult；拒绝第二次成功结算"
                )
            updated = self.connection.execute(
                """UPDATE evaluation_jobs SET status=?, result_id=?, lease_expires_at=NULL,
                   worker_id=NULL, error_code=NULL, error_message=NULL
                   WHERE id=? AND status=? AND worker_id=?
                     AND lease_expires_at IS NOT NULL AND lease_expires_at>?""",
                (
                    EvaluationJobStatus.SUCCESS.value, result.id, job_id,
                    EvaluationJobStatus.RUNNING.value, worker_id, _dt(now),
                ),
            ).rowcount
            if updated != 1:
                raise RuntimeError("评价 lease 已丢失：成功结算 fencing 条件未命中")
            self.connection.execute(
                "INSERT INTO evaluation_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                (result.id, result.run_id, result.candidate_id, result.objective,
                 json.dumps(result.trajectory),
                 json.dumps(result.best_configuration, ensure_ascii=False, sort_keys=True),
                 result.used_budget),
            )
            candidate_updated = self.connection.execute(
                """UPDATE candidates SET status=?, objective=?, evaluation_artifact_id=?
                   WHERE id=? AND run_id=?""",
                (
                    CandidateStatus.EVALUATED.value,
                    result.objective,
                    artifact_id,
                    result.candidate_id,
                    result.run_id,
                ),
            ).rowcount
            if candidate_updated != 1:
                raise RuntimeError("EvaluationResult Candidate 更新发生并发归属冲突")
            budget_updated = self.connection.execute(
                """UPDATE runs SET reserved_evaluations=reserved_evaluations-?,
                   consumed_evaluations=consumed_evaluations+?
                   WHERE id=? AND reserved_evaluations>=?""",
                (job.budget, result.used_budget, job.run_id, job.budget),
            ).rowcount
            if budget_updated != 1:
                raise RuntimeError("评价预算预留不足，拒绝成功结算")
            self.connection.commit()
            return self.get_evaluation_job(job_id)
        except Exception:
            self.connection.rollback()
            raise

    def complete_job_failure(
        self, job_id: str, worker_id: str, error_code: str, message: str,
        retryable: bool, available_at: datetime, now: datetime
    ) -> EvaluationJob:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute("SELECT * FROM evaluation_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            job = self._job(row)
            if (
                job.status != EvaluationJobStatus.RUNNING
                or job.worker_id != worker_id
                or job.lease_expires_at is None
                or job.lease_expires_at <= now
            ):
                raise RuntimeError("评价 lease 已丢失：仅当前 RUNNING owner 可以失败结算")
            run_row = self.connection.execute(
                "SELECT status,cancel_requested FROM runs WHERE id=?", (job.run_id,)
            ).fetchone()
            if (
                run_row is None
                or run_row["status"] != RunStatus.RUNNING.value
                or bool(run_row["cancel_requested"])
            ):
                raise RuntimeError("Run 已不处于 active 状态，拒绝评价失败结算")
            should_retry = retryable and job.attempts < job.max_attempts
            status = EvaluationJobStatus.RETRY_WAIT if should_retry else EvaluationJobStatus.DEAD
            updated = self.connection.execute(
                """UPDATE evaluation_jobs SET status=?, available_at=?, lease_expires_at=NULL,
                   worker_id=NULL, error_code=?, error_message=?
                   WHERE id=? AND status=? AND worker_id=?
                     AND lease_expires_at IS NOT NULL AND lease_expires_at>?""",
                (
                    status.value, _dt(available_at), error_code, message, job_id,
                    EvaluationJobStatus.RUNNING.value, worker_id, _dt(now),
                ),
            ).rowcount
            if updated != 1:
                raise RuntimeError("评价 lease 已丢失：失败结算 fencing 条件未命中")
            if status == EvaluationJobStatus.DEAD:
                budget_updated = self.connection.execute(
                    """UPDATE runs SET reserved_evaluations=reserved_evaluations-?
                       WHERE id=? AND reserved_evaluations>=?""",
                    (job.budget, job.run_id, job.budget),
                ).rowcount
                if budget_updated != 1:
                    raise RuntimeError("评价预算预留不足，拒绝失败结算")
                self.connection.execute(
                    """UPDATE candidates SET status=?
                       WHERE id=? AND status IN (?, ?)""",
                    (
                        CandidateStatus.INVALID.value,
                        job.candidate_id,
                        CandidateStatus.EVALUATING.value,
                        CandidateStatus.EVALUATION_PENDING.value,
                    ),
                )
            self.connection.commit()
            return self.get_evaluation_job(job_id)
        except Exception:
            self.connection.rollback()
            raise

    def recover_stale_jobs(self, now: datetime, run_id: str = "") -> int:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            query = (
                "SELECT * FROM evaluation_jobs WHERE status=? AND lease_expires_at<=?"
            )
            params = [EvaluationJobStatus.RUNNING.value, _dt(now)]
            if run_id:
                query += " AND run_id=?"
                params.append(run_id)
            rows = self.connection.execute(query, tuple(params)).fetchall()
            for row in rows:
                job = self._job(row)
                run_row = self.connection.execute(
                    "SELECT cancel_requested,status FROM runs WHERE id=?", (job.run_id,)
                ).fetchone()
                cancelled = bool(run_row["cancel_requested"]) or (
                    run_row["status"] == RunStatus.CANCELLED.value
                )
                if cancelled:
                    self.connection.execute(
                        """UPDATE evaluation_jobs SET status=?,worker_id=NULL,
                           lease_expires_at=NULL,error_code='RUN_CANCELLED',
                           error_message='Run 已取消，stale running job 已结算' WHERE id=?""",
                        (EvaluationJobStatus.CANCELLED.value, job.id),
                    )
                    self.connection.execute(
                        """UPDATE runs SET reserved_evaluations=reserved_evaluations-?
                           WHERE id=? AND reserved_evaluations>=?""",
                        (job.budget,job.run_id,job.budget),
                    )
                    self.connection.execute(
                        "UPDATE candidates SET status=? WHERE id=?",
                        (CandidateStatus.INVALID.value,job.candidate_id),
                    )
                elif job.attempts >= job.max_attempts:
                    self.connection.execute(
                        """UPDATE evaluation_jobs SET status=?, worker_id=NULL,
                           lease_expires_at=NULL, error_code=?, error_message=? WHERE id=?""",
                        (EvaluationJobStatus.DEAD.value, "LEASE_EXHAUSTED",
                         "worker lease 多次过期，已进入 dead letter", job.id),
                    )
                    self.connection.execute(
                        "UPDATE runs SET reserved_evaluations=reserved_evaluations-? WHERE id=?",
                        (job.budget, job.run_id),
                    )
                    self.connection.execute(
                        "UPDATE candidates SET status=? WHERE id=?",
                        (CandidateStatus.INVALID.value,job.candidate_id),
                    )
                else:
                    self.connection.execute(
                        """UPDATE evaluation_jobs SET status=?, available_at=?, worker_id=NULL,
                           lease_expires_at=NULL, error_code=?, error_message=? WHERE id=?""",
                        (EvaluationJobStatus.PENDING.value, _dt(now), "LEASE_EXPIRED",
                         "worker lease 过期，已重新入队", job.id),
                    )
            self.connection.commit()
            return len(rows)
        except Exception:
            self.connection.rollback()
            raise
