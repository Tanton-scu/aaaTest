import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlparse

from prievo_agent.domain.errors import (
    ArtifactIntegrityError,
    BudgetExhaustedError,
    EvaluationIdentityConflictError,
)
from prievo_agent.domain.models import (
    ArtifactMetadata, Candidate, CandidateStatus, CheckpointMetadata,
    EvaluationJob, EvaluationJobStatus, EvaluationResult, Event,
    GenerationPlan, OptimizationTask, PreviousPlanFeedback, Run, RunStatus,
    AgentCapability, AgentMemory, AgentTask, AgentTaskStatus, ToolCallRecord,
    TraceRecord,
)
from prievo_agent.infrastructure.agent_memory import memory_types_for_scope


def _dt(value):
    return value.isoformat()


def _parsed(value):
    return datetime.fromisoformat(value) if value else None


def _same_result(row, result):
    return (
        row["id"] == result.id
        and row["run_id"] == result.run_id
        and row["candidate_id"] == result.candidate_id
        and row["objective"] == result.objective
        and MySQLRuntimeStore._json(row["trajectory_json"]) == result.trajectory
        and MySQLRuntimeStore._json(row["best_configuration_json"])
        == result.best_configuration
        and row["used_budget"] == result.used_budget
    )


def _same_job_identity(row, job):
    return (
        row["idempotency_key"] == job.idempotency_key
        and row["run_id"] == job.run_id
        and row["task_id"] == job.task_id
        and row["candidate_id"] == job.candidate_id
        and row["seed"] == job.seed
        and row["budget"] == job.budget
    )


class MySQLRuntimeStore:
    """Full Mode 的持久事实源，接口与 SQLite Demo Store 一致。"""

    def __init__(self, database_url, artifact_root, ensure_schema=True):
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError("Full Mode 需要安装 pymysql") from exc
        parsed = urlparse(database_url.replace("mysql+pymysql://", "mysql://", 1))
        self.connection = pymysql.connect(
            host=parsed.hostname or "mysql", port=parsed.port or 3306,
            user=unquote(parsed.username or "prievo"),
            password=unquote(parsed.password or ""), database=parsed.path.lstrip("/"),
            charset="utf8mb4", autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        if ensure_schema:
            self._create_schema()

    def close(self):
        self.connection.close()

    def _create_schema(self):
        statements = [
            """CREATE TABLE IF NOT EXISTS tasks(
              id VARCHAR(96) PRIMARY KEY,name VARCHAR(160) NOT NULL,objective VARCHAR(32) NOT NULL,
              evaluation_budget INT NOT NULL,total_budget INT NOT NULL,created_at VARCHAR(40) NOT NULL,
              dataset_id VARCHAR(120) NOT NULL,generations INT NOT NULL,population_size INT NOT NULL,
              random_seed BIGINT NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS runs(
              id VARCHAR(96) PRIMARY KEY,task_id VARCHAR(96) NOT NULL,status VARCHAR(24) NOT NULL,
              generation INT NOT NULL,consumed_evaluations INT NOT NULL,reserved_evaluations INT NOT NULL,
              best_candidate_id VARCHAR(180),created_at VARCHAR(40) NOT NULL,updated_at VARCHAR(40) NOT NULL,
              dataset_id VARCHAR(120) NOT NULL,INDEX ix_runs_status(status,created_at),
              CONSTRAINT fk_run_task FOREIGN KEY(task_id) REFERENCES tasks(id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS candidates(
              id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,code LONGTEXT NOT NULL,
              description TEXT NOT NULL,operators_json JSON NOT NULL,lineage_json JSON NOT NULL,
              status VARCHAR(32) NOT NULL,objective DOUBLE,code_artifact_id VARCHAR(180),
              evaluation_artifact_id VARCHAR(180),INDEX ix_candidates_run(run_id,status),
              CONSTRAINT fk_candidate_run FOREIGN KEY(run_id) REFERENCES runs(id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS candidate_parents(
              candidate_id VARCHAR(180) NOT NULL,parent_id VARCHAR(180) NOT NULL,
              PRIMARY KEY(candidate_id,parent_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS population_memberships(
              run_id VARCHAR(96) NOT NULL,generation INT NOT NULL,candidate_id VARCHAR(180) NOT NULL,
              role_name VARCHAR(24) NOT NULL DEFAULT 'MEMBER',PRIMARY KEY(run_id,generation,candidate_id))
              ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS evaluation_results(
              id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,candidate_id VARCHAR(180) UNIQUE NOT NULL,
              objective DOUBLE NOT NULL,trajectory_json JSON NOT NULL,best_configuration_json JSON NOT NULL,
              used_budget INT NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS run_events(
              run_id VARCHAR(96) NOT NULL,sequence_no BIGINT NOT NULL,event_type VARCHAR(64) NOT NULL,
              message TEXT NOT NULL,payload_json JSON NOT NULL,occurred_at VARCHAR(40) NOT NULL,
              PRIMARY KEY(run_id,sequence_no)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS artifacts(
              id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,kind VARCHAR(64) NOT NULL,
              media_type VARCHAR(96) NOT NULL,size_bytes BIGINT NOT NULL,digest CHAR(64) NOT NULL,
              uri TEXT NOT NULL,INDEX ix_artifacts_run(run_id,kind)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS checkpoints(
              id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,generation INT NOT NULL,
              population_ids_json JSON NOT NULL,consumed_budget INT NOT NULL,remaining_budget INT NOT NULL,
              artifact_id VARCHAR(180) NOT NULL,schema_version INT NOT NULL,code_version VARCHAR(64) NOT NULL,
              created_at VARCHAR(40) NOT NULL,dataset_id VARCHAR(120) NOT NULL,prior_refs_json JSON NOT NULL,
              UNIQUE KEY uq_checkpoint_generation(run_id,generation)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS evaluation_jobs(
              id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,task_id VARCHAR(96) NOT NULL,
              candidate_id VARCHAR(180) NOT NULL,seed BIGINT NOT NULL,budget INT NOT NULL,
              idempotency_key VARCHAR(255) UNIQUE NOT NULL,status VARCHAR(32) NOT NULL,attempts INT NOT NULL,
              max_attempts INT NOT NULL,available_at VARCHAR(40) NOT NULL,lease_expires_at VARCHAR(40),
              worker_id VARCHAR(120),result_id VARCHAR(180),error_code VARCHAR(64),error_message TEXT,
              INDEX ix_jobs_ready(status,available_at)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS datasets_metadata(
              dataset_id VARCHAR(120) PRIMARY KEY,filename VARCHAR(180) NOT NULL,row_count BIGINT NOT NULL,
              digest CHAR(64) NOT NULL,validated_at VARCHAR(40) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS tool_calls(
              id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,tool_name VARCHAR(96) NOT NULL,
              status VARCHAR(24) NOT NULL,request_json JSON NOT NULL,response_json JSON NOT NULL,
              created_at VARCHAR(40) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
        ]
        checksum=hashlib.sha256("\n".join(statements).encode()).hexdigest()
        with self.connection.cursor() as cursor:
            cursor.execute("""CREATE TABLE IF NOT EXISTS schema_migrations(
              version_no INT PRIMARY KEY,description VARCHAR(255) NOT NULL,checksum CHAR(64) NOT NULL,
              applied_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6))
              ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cursor.execute("SELECT checksum FROM schema_migrations WHERE version_no=1")
            applied=cursor.fetchone()
            if applied and applied["checksum"]!=checksum:
                raise RuntimeError("MySQL V001 schema checksum 漂移，请创建新 migration")
            for statement in statements:
                cursor.execute(statement)
            if not applied:
                cursor.execute("INSERT INTO schema_migrations VALUES(1,'V5 initial full mode',%s,NOW(6))",
                               (checksum,))
        self.connection.commit()
        self._apply_agent_memory_migration()
        self._apply_agent_task_migration()
        self._apply_run_control_migration()
        self._apply_agent_task_fencing_migration()
        self._apply_evaluation_identity_migration()
        self._apply_planning_trace_migration()

    def _apply_agent_memory_migration(self):
        statements = [
            """CREATE TABLE IF NOT EXISTS agent_memories(
              id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,
              dataset_id VARCHAR(120) NOT NULL,memory_type VARCHAR(48) NOT NULL,
              subject VARCHAR(255) NOT NULL,content TEXT NOT NULL,
              evidence_artifact_id VARCHAR(180) NOT NULL DEFAULT '',created_at VARCHAR(40) NOT NULL,
              INDEX ix_agent_memory_dataset(dataset_id,created_at))
              ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
        ]
        checksum = hashlib.sha256("\n".join(statements).encode()).hexdigest()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT checksum FROM schema_migrations WHERE version_no=2")
            applied = cursor.fetchone()
            if applied and applied["checksum"] != checksum:
                raise RuntimeError("MySQL V002 schema checksum 漂移，请创建新 migration")
            for statement in statements:
                cursor.execute(statement)
            if not applied:
                cursor.execute(
                    "INSERT INTO schema_migrations VALUES(2,'Agent long-term memory',%s,NOW(6))",
                    (checksum,),
                )
        self.connection.commit()

    def _apply_agent_task_migration(self):
        statements = [
            """CREATE TABLE IF NOT EXISTS agent_tasks(
              id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,
              task_type VARCHAR(64) NOT NULL,required_capability VARCHAR(64) NOT NULL,
              idempotency_key VARCHAR(255) NOT NULL,input_artifact_refs_json JSON NOT NULL,
              output_artifact_refs_json JSON NOT NULL,status VARCHAR(24) NOT NULL,
              claimed_by VARCHAR(120) NOT NULL DEFAULT '',attempts INT NOT NULL,
              max_attempts INT NOT NULL,error_message TEXT NOT NULL,
              created_at VARCHAR(40) NOT NULL,updated_at VARCHAR(40) NOT NULL,
              UNIQUE KEY uq_agent_task_run_idempotency(run_id,idempotency_key),
              INDEX ix_agent_tasks_run_status(run_id,status,created_at),
              CONSTRAINT fk_agent_task_run FOREIGN KEY(run_id) REFERENCES runs(id))
              ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
        ]
        checksum = hashlib.sha256("\n".join(statements).encode()).hexdigest()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT checksum FROM schema_migrations WHERE version_no=3")
            applied = cursor.fetchone()
            if applied and applied["checksum"] != checksum:
                raise RuntimeError("MySQL V003 schema checksum 漂移，请创建新 migration")
            for statement in statements:
                cursor.execute(statement)
            if not applied:
                cursor.execute(
                    "INSERT INTO schema_migrations VALUES(3,'Durable agent tasks',%s,NOW(6))",
                    (checksum,),
                )
        self.connection.commit()

    def _apply_run_control_migration(self):
        """V004：只新增控制/lease 字段并移除 checkpoint 的代际唯一约束。

        V001--V003 的 statement material 保持原样，避免篡改已应用 checksum。
        逐列检查用于承受 MySQL DDL 隐式提交后的中途重启。
        """

        column_definitions = {
            "pause_requested": "TINYINT(1) NOT NULL DEFAULT 0",
            "cancel_requested": "TINYINT(1) NOT NULL DEFAULT 0",
            "control_reason": "VARCHAR(2000) NOT NULL DEFAULT ''",
            "runtime_cursor_artifact_id": "VARCHAR(180) NOT NULL DEFAULT ''",
            "runtime_owner_id": "VARCHAR(120) NOT NULL DEFAULT ''",
            "runtime_lease_expires_at": "VARCHAR(40) NULL",
        }
        material = [
            "ALTER TABLE runs ADD COLUMN {} {}".format(name, definition)
            for name, definition in column_definitions.items()
        ] + ["ALTER TABLE checkpoints DROP INDEX uq_checkpoint_generation"]
        checksum = hashlib.sha256("\n".join(material).encode()).hexdigest()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT checksum FROM schema_migrations WHERE version_no=4")
            applied = cursor.fetchone()
            if applied and applied["checksum"] != checksum:
                raise RuntimeError("MySQL V004 schema checksum 漂移，请创建新 migration")
            if applied:
                self.connection.commit()
                return
            cursor.execute(
                """SELECT COLUMN_NAME FROM information_schema.COLUMNS
                   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='runs'"""
            )
            existing_columns = {row["COLUMN_NAME"] for row in cursor.fetchall()}
            for name, definition in column_definitions.items():
                if name not in existing_columns:
                    cursor.execute(
                        "ALTER TABLE runs ADD COLUMN {} {}".format(name, definition)
                    )
            cursor.execute(
                """SELECT INDEX_NAME FROM information_schema.STATISTICS
                   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='checkpoints'
                     AND INDEX_NAME='uq_checkpoint_generation'"""
            )
            if cursor.fetchone():
                cursor.execute(
                    "ALTER TABLE checkpoints DROP INDEX uq_checkpoint_generation"
                )
            cursor.execute(
                "INSERT INTO schema_migrations VALUES(4,'Run control, cursor and lease',%s,NOW(6))",
                (checksum,),
            )
        self.connection.commit()

    def _apply_agent_task_fencing_migration(self):
        """V005：AgentTask 每次 claim 的 token 与有界 lease。

        V001--V004 的 statement/checksum material 保持不变；新字段只允许由本
        migration 追加。逐项探测用于承受 MySQL DDL 隐式提交后的中途重启。
        """

        column_definitions = {
            "claim_token": "VARCHAR(120) NOT NULL DEFAULT ''",
            "lease_expires_at": "VARCHAR(40) NULL",
        }
        index_name = "ix_agent_tasks_lease"
        material = [
            "ALTER TABLE agent_tasks ADD COLUMN {} {}".format(name, definition)
            for name, definition in column_definitions.items()
        ] + [
            "CREATE INDEX {} ON agent_tasks(status,lease_expires_at)".format(
                index_name
            )
        ]
        checksum = hashlib.sha256("\n".join(material).encode()).hexdigest()
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT checksum FROM schema_migrations WHERE version_no=5"
            )
            applied = cursor.fetchone()
            if applied and applied["checksum"] != checksum:
                raise RuntimeError("MySQL V005 schema checksum 漂移，请创建新 migration")
            if applied:
                self.connection.commit()
                return
            cursor.execute(
                """SELECT COLUMN_NAME FROM information_schema.COLUMNS
                   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='agent_tasks'"""
            )
            existing_columns = {row["COLUMN_NAME"] for row in cursor.fetchall()}
            for name, definition in column_definitions.items():
                if name not in existing_columns:
                    cursor.execute(
                        "ALTER TABLE agent_tasks ADD COLUMN {} {}".format(
                            name, definition
                        )
                    )
            cursor.execute(
                """SELECT INDEX_NAME FROM information_schema.STATISTICS
                   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='agent_tasks'
                     AND INDEX_NAME=%s""",
                (index_name,),
            )
            if not cursor.fetchone():
                cursor.execute(
                    "CREATE INDEX {} ON agent_tasks(status,lease_expires_at)".format(
                        index_name
                    )
                )
            cursor.execute(
                """INSERT INTO schema_migrations
                   VALUES(5,'AgentTask claim token and lease fencing',%s,NOW(6))""",
                (checksum,),
            )
        self.connection.commit()

    def _apply_evaluation_identity_migration(self):
        """V006：一个 Candidate 在数据库层最多绑定一个 logical evaluation。"""

        index_name = "uq_evaluation_job_candidate"
        statement = (
            "ALTER TABLE evaluation_jobs ADD UNIQUE KEY "
            "{}(candidate_id)".format(index_name)
        )
        checksum = hashlib.sha256(statement.encode()).hexdigest()
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT checksum FROM schema_migrations WHERE version_no=6"
            )
            applied = cursor.fetchone()
            if applied and applied["checksum"] != checksum:
                raise RuntimeError("MySQL V006 schema checksum 漂移，请创建新 migration")
            if applied:
                self.connection.commit()
                return
            cursor.execute(
                """SELECT INDEX_NAME FROM information_schema.STATISTICS
                   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='evaluation_jobs'
                     AND INDEX_NAME=%s""",
                (index_name,),
            )
            if not cursor.fetchone():
                cursor.execute(
                    """SELECT candidate_id,COUNT(*) AS job_count
                       FROM evaluation_jobs GROUP BY candidate_id
                       HAVING COUNT(*)>1 LIMIT 1"""
                )
                duplicate = cursor.fetchone()
                if duplicate:
                    raise EvaluationIdentityConflictError(
                        "MySQL V006 发现同一 Candidate 的多个 legacy logical jobs：{}；"
                        "请先审计数据，不允许自动猜测 authoritative identity".format(
                            duplicate["candidate_id"]
                        )
                    )
                cursor.execute(statement)
            cursor.execute(
                """INSERT INTO schema_migrations
                   VALUES(6,'One logical evaluation per Candidate',%s,NOW(6))""",
                (checksum,),
            )
        self.connection.commit()

    def _apply_planning_trace_migration(self):
        """V007：GenerationPlan、Candidate 一等索引字段与结构化 Trace。"""

        candidate_columns = {
            "generation": "INT NOT NULL DEFAULT 0",
            "plan_id": "VARCHAR(180) NOT NULL DEFAULT ''",
            "generation_strategy": "VARCHAR(24) NOT NULL DEFAULT ''",
            "code_digest": "CHAR(64) NOT NULL DEFAULT ''",
            "selected_parent_ids_json": "JSON NOT NULL",
            "prior_refs_json": "JSON NOT NULL",
            "created_at": "VARCHAR(40) NOT NULL DEFAULT ''",
        }
        statements = [
            """CREATE TABLE IF NOT EXISTS generation_plans(
              plan_id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,
              generation INT NOT NULL,sequence_no INT NOT NULL,
              generation_strategy VARCHAR(24) NOT NULL,
              parent_selection_policy VARCHAR(24) NOT NULL,
              decision_reason TEXT NOT NULL,required_parent_count INT NOT NULL,
              previous_feedback_json JSON NOT NULL,context_artifact_id VARCHAR(180) NOT NULL DEFAULT '',
              prompt_artifact_id VARCHAR(180) NOT NULL DEFAULT '',created_at VARCHAR(40) NOT NULL,
              UNIQUE KEY uq_generation_plan_slot(run_id,generation,generation_strategy,sequence_no),
              INDEX ix_generation_plan_run(run_id,generation,generation_strategy,sequence_no))
              ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS trace_records(
              id VARCHAR(180) PRIMARY KEY,run_id VARCHAR(96) NOT NULL,
              span_type VARCHAR(64) NOT NULL,actor VARCHAR(96) NOT NULL,
              status VARCHAR(32) NOT NULL,payload_json JSON NOT NULL,
              plan_id VARCHAR(180) NOT NULL DEFAULT '',candidate_id VARCHAR(180) NOT NULL DEFAULT '',
              evaluation_job_id VARCHAR(180) NOT NULL DEFAULT '',skill_name VARCHAR(96) NOT NULL DEFAULT '',
              context_refs_json JSON NOT NULL,tool_call_id VARCHAR(180) NOT NULL DEFAULT '',
              latency_ms INT NOT NULL DEFAULT 0,token_usage_json JSON NOT NULL,
              created_at VARCHAR(40) NOT NULL,INDEX ix_trace_records_run(run_id,created_at))
              ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
        ] + [
            "ALTER TABLE candidates ADD COLUMN {} {}".format(name, definition)
            for name, definition in candidate_columns.items()
        ]
        checksum = hashlib.sha256("\n".join(statements).encode()).hexdigest()
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT checksum FROM schema_migrations WHERE version_no=7"
            )
            applied = cursor.fetchone()
            if applied and applied["checksum"] != checksum:
                raise RuntimeError("MySQL V007 schema checksum 漂移，请创建新 migration")
            if applied:
                self.connection.commit()
                return
            cursor.execute(
                """SELECT COLUMN_NAME FROM information_schema.COLUMNS
                   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='candidates'"""
            )
            existing_columns = {row["COLUMN_NAME"] for row in cursor.fetchall()}
            for name, definition in candidate_columns.items():
                if name in existing_columns:
                    continue
                default_definition = definition
                if name in {"selected_parent_ids_json", "prior_refs_json"}:
                    default_definition = "JSON NOT NULL DEFAULT ('[]')"
                cursor.execute(
                    "ALTER TABLE candidates ADD COLUMN {} {}".format(
                        name, default_definition
                    )
                )
            for statement in statements[:2]:
                cursor.execute(statement)
            cursor.execute(
                """INSERT INTO schema_migrations
                   VALUES(7,'Planning, candidate indexes and structured trace',%s,NOW(6))""",
                (checksum,),
            )
        self.connection.commit()

    def add_agent_task(self, task):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO agent_tasks(
                      id,run_id,task_type,required_capability,idempotency_key,
                      input_artifact_refs_json,output_artifact_refs_json,status,
                      claimed_by,attempts,max_attempts,error_message,created_at,updated_at)
                      VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (task.id,task.run_id,task.task_type,
                     task.required_capability.value,task.idempotency_key,
                     json.dumps(task.input_artifact_refs,ensure_ascii=False),
                     json.dumps(task.output_artifact_refs,ensure_ascii=False),
                     task.status.value,task.claimed_by,task.attempts,
                     task.max_attempts,task.error_message,_dt(task.created_at),
                     _dt(task.updated_at)),
                )
            self.connection.commit()
            return task, True
        except Exception as exc:
            self.connection.rollback()
            if not exc.args or exc.args[0] != 1062:
                raise
            row = self._one(
                "SELECT * FROM agent_tasks WHERE run_id=%s AND idempotency_key=%s",
                (task.run_id,task.idempotency_key),
            )
            if row is None:
                # 主键冲突但 idempotency identity 不同，不应伪装成幂等命中。
                raise
            return self._agent_task(row), False

    def get_agent_task(self, task_id):
        row = self._one("SELECT * FROM agent_tasks WHERE id=%s", (task_id,))
        if row is None:
            raise KeyError(task_id)
        return self._agent_task(row)

    def agent_tasks_for_run(self, run_id):
        return [
            self._agent_task(row)
            for row in self._all(
                "SELECT * FROM agent_tasks WHERE run_id=%s ORDER BY created_at,id",
                (run_id,),
            )
        ]

    def claim_agent_task(
        self, task_id, agent_name, now=None, lease_seconds=300, claim_token=""
    ):
        if lease_seconds <= 0:
            raise ValueError("AgentTask lease_seconds 必须大于 0")
        now = now or datetime.now().astimezone()
        token = claim_token or "agent-claim-{}".format(uuid.uuid4().hex)
        expires = now + timedelta(seconds=lease_seconds)
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE agent_tasks JOIN runs ON runs.id=agent_tasks.run_id
                      SET agent_tasks.status=%s,agent_tasks.claimed_by=%s,
                      agent_tasks.claim_token=%s,agent_tasks.lease_expires_at=%s,
                      agent_tasks.attempts=agent_tasks.attempts+1,
                      agent_tasks.error_message='',agent_tasks.updated_at=%s
                      WHERE agent_tasks.id=%s AND agent_tasks.status=%s
                        AND agent_tasks.attempts<agent_tasks.max_attempts
                        AND runs.cancel_requested=0 AND runs.status IN(%s,%s)""",
                    (AgentTaskStatus.CLAIMED.value,str(agent_name)[:120],token,
                     _dt(expires),_dt(now),task_id,AgentTaskStatus.PENDING.value,
                     RunStatus.PENDING.value,RunStatus.RUNNING.value),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "AgentTask 已被领取、重试耗尽或所属 Run 非 active"
                    )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return self.get_agent_task(task_id)

    def renew_agent_task_lease(self,task_id,claim_token,now,lease_seconds):
        if not claim_token:
            raise ValueError("AgentTask claim_token 不能为空")
        if lease_seconds<=0:
            raise ValueError("AgentTask lease_seconds 必须大于 0")
        expires=now+timedelta(seconds=lease_seconds)
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE agent_tasks JOIN runs ON runs.id=agent_tasks.run_id
                      SET agent_tasks.lease_expires_at=%s,agent_tasks.updated_at=%s
                      WHERE agent_tasks.id=%s AND agent_tasks.status=%s
                        AND agent_tasks.claim_token=%s
                        AND agent_tasks.lease_expires_at IS NOT NULL
                        AND agent_tasks.lease_expires_at>%s
                        AND runs.cancel_requested=0 AND runs.status IN(%s,%s)""",
                    (_dt(expires),_dt(now),task_id,AgentTaskStatus.CLAIMED.value,
                     claim_token,_dt(now),RunStatus.PENDING.value,
                     RunStatus.RUNNING.value),
                )
                renewed=cursor.rowcount==1
                if not renewed:
                    # 固定时钟下续租到相同值时 MySQL 默认报告 0 changed rows；
                    # 只读确认全部 fencing 条件仍成立即可视为成功。
                    cursor.execute(
                        """SELECT agent_tasks.id FROM agent_tasks
                          JOIN runs ON runs.id=agent_tasks.run_id
                          WHERE agent_tasks.id=%s AND agent_tasks.status=%s
                            AND agent_tasks.claim_token=%s
                            AND agent_tasks.lease_expires_at>%s
                            AND agent_tasks.lease_expires_at>=%s
                            AND runs.cancel_requested=0 AND runs.status IN(%s,%s)""",
                        (task_id,AgentTaskStatus.CLAIMED.value,claim_token,
                         _dt(now),_dt(expires),RunStatus.PENDING.value,
                         RunStatus.RUNNING.value),
                    )
                    renewed=cursor.fetchone() is not None
            self.connection.commit(); return renewed
        except Exception:
            self.connection.rollback(); raise

    def complete_agent_task(
        self, task_id, claim_token, output_artifact_refs, now=None
    ):
        if not claim_token:
            raise ValueError("AgentTask claim_token 不能为空")
        now = now or datetime.now().astimezone()
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE agent_tasks JOIN runs ON runs.id=agent_tasks.run_id
                      SET agent_tasks.status=%s,
                      agent_tasks.output_artifact_refs_json=%s,
                      agent_tasks.claim_token='',agent_tasks.lease_expires_at=NULL,
                      agent_tasks.updated_at=%s
                      WHERE agent_tasks.id=%s AND agent_tasks.status=%s
                        AND agent_tasks.claim_token=%s
                        AND agent_tasks.lease_expires_at IS NOT NULL
                        AND agent_tasks.lease_expires_at>%s
                        AND runs.cancel_requested=0 AND runs.status IN(%s,%s)""",
                    (AgentTaskStatus.COMPLETED.value,
                     json.dumps(list(output_artifact_refs),ensure_ascii=False),
                     _dt(now),task_id,AgentTaskStatus.CLAIMED.value,claim_token,
                     _dt(now),RunStatus.PENDING.value,RunStatus.RUNNING.value),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "AgentTask lease 已丢失：过期、接管、取消或非 active Run"
                    )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return self.get_agent_task(task_id)

    def fail_agent_task(self, task_id, claim_token, error_message, now=None):
        if not claim_token:
            raise ValueError("AgentTask claim_token 不能为空")
        now = now or datetime.now().astimezone()
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM agent_tasks WHERE id=%s FOR UPDATE", (task_id,)
                )
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(task_id)
                task = self._agent_task(row)
                retry = task.attempts < task.max_attempts
                status = (
                    AgentTaskStatus.PENDING if retry else AgentTaskStatus.FAILED
                )
                cursor.execute(
                    """UPDATE agent_tasks JOIN runs ON runs.id=agent_tasks.run_id
                      SET agent_tasks.status=%s,agent_tasks.claimed_by='',
                      agent_tasks.claim_token='',agent_tasks.lease_expires_at=NULL,
                      agent_tasks.error_message=%s,agent_tasks.updated_at=%s
                      WHERE agent_tasks.id=%s AND agent_tasks.status=%s
                        AND agent_tasks.claim_token=%s
                        AND agent_tasks.lease_expires_at IS NOT NULL
                        AND agent_tasks.lease_expires_at>%s
                        AND runs.cancel_requested=0 AND runs.status IN(%s,%s)""",
                    (status.value,str(error_message)[:2000],_dt(now),task_id,
                     AgentTaskStatus.CLAIMED.value,claim_token,_dt(now),
                     RunStatus.PENDING.value,RunStatus.RUNNING.value),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "AgentTask lease 已丢失：拒绝 stale failure 覆盖当前 owner"
                    )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return self.get_agent_task(task_id)

    def recover_orphan_agent_tasks(self, now, orphan_seconds=300):
        from datetime import timezone

        cutoff = now.astimezone(timezone.utc) - timedelta(seconds=orphan_seconds)
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """SELECT agent_tasks.*,runs.status AS run_status,
                              runs.cancel_requested AS run_cancel_requested
                      FROM agent_tasks JOIN runs ON runs.id=agent_tasks.run_id
                      WHERE agent_tasks.status=%s AND (
                        (agent_tasks.lease_expires_at IS NOT NULL
                         AND agent_tasks.lease_expires_at<=%s) OR
                        (agent_tasks.lease_expires_at IS NULL
                         AND agent_tasks.updated_at<=%s)) FOR UPDATE""",
                    (AgentTaskStatus.CLAIMED.value,_dt(now),_dt(cutoff)),
                )
                rows = list(cursor.fetchall())
                recovered = 0
                for row in rows:
                    if bool(row["run_cancel_requested"]) or row["run_status"] in {
                        RunStatus.COMPLETED.value,RunStatus.FAILED.value,
                        RunStatus.CANCELLED.value,
                    }:
                        target=AgentTaskStatus.CANCELLED
                        message="所属 Run 已终止，过期 AgentTask 已取消"
                    elif row["attempts"]>=row["max_attempts"]:
                        target=AgentTaskStatus.FAILED
                        message="AgentTask lease 过期且已达到最大尝试次数"
                    else:
                        target=AgentTaskStatus.PENDING
                        message="AgentTask lease 过期，已重新入队"
                    cursor.execute(
                        """UPDATE agent_tasks SET status=%s,claimed_by='',claim_token='',
                          lease_expires_at=NULL,error_message=%s,updated_at=%s
                          WHERE id=%s AND status=%s AND claim_token=%s""",
                        (target.value,message,_dt(now),row["id"],
                         AgentTaskStatus.CLAIMED.value,row["claim_token"]),
                    )
                    recovered += cursor.rowcount
            self.connection.commit()
            return recovered
        except Exception:
            self.connection.rollback()
            raise

    def recover_orphan_runs(self, now):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """UPDATE runs SET runtime_owner_id='',runtime_lease_expires_at=NULL
                  WHERE status IN(%s,%s) AND runtime_owner_id<>''
                    AND runtime_lease_expires_at IS NOT NULL
                    AND runtime_lease_expires_at<=%s""",
                (RunStatus.PENDING.value,RunStatus.RUNNING.value,_dt(now)),
            )
            count = cursor.rowcount
        self.connection.commit()
        return count

    @staticmethod
    def _agent_task(row):
        return AgentTask(
            row["id"],row["run_id"],row["task_type"],
            AgentCapability(row["required_capability"]),row["idempotency_key"],
            MySQLRuntimeStore._json(row["input_artifact_refs_json"]),
            MySQLRuntimeStore._json(row["output_artifact_refs_json"]),
            AgentTaskStatus(row["status"]),row["claimed_by"],row["attempts"],
            row["max_attempts"],row["error_message"],
            _parsed(row["created_at"]),_parsed(row["updated_at"]),
            row.get("claim_token") or "",_parsed(row.get("lease_expires_at")),
        )

    def add_agent_memory(self, memory):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO agent_memories VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                   ON DUPLICATE KEY UPDATE subject=VALUES(subject),content=VALUES(content),
                   evidence_artifact_id=VALUES(evidence_artifact_id)""",
                (memory.id,memory.run_id,memory.dataset_id,memory.memory_type,
                 memory.subject,memory.content,memory.evidence_artifact_id,
                 _dt(memory.created_at)),
            )
        self.connection.commit()

    def agent_memories_for_dataset(self, dataset_id, limit=5, exclude_run_id=""):
        query = "SELECT * FROM agent_memories WHERE dataset_id=%s"
        params = [dataset_id]
        if exclude_run_id:
            query += " AND run_id<>%s"
            params.append(exclude_run_id)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(int(limit))
        return [AgentMemory(
            row["id"],row["run_id"],row["dataset_id"],row["memory_type"],
            row["subject"],row["content"],row["evidence_artifact_id"],
            _parsed(row["created_at"]),
        ) for row in self._all(query, tuple(params))]

    def agent_memories_for_run(self, run_id, scope, limit=30):
        memory_types = memory_types_for_scope(scope)
        placeholders = ",".join("%s" for _ in memory_types)
        query = (
            "SELECT * FROM agent_memories WHERE run_id=%s "
            "AND memory_type IN ({}) "
            "ORDER BY created_at DESC,id DESC LIMIT %s"
        ).format(placeholders)
        rows = self._all(
            query, (run_id, *memory_types, max(1, int(limit)))
        )
        rows.reverse()
        return [AgentMemory(
            row["id"],row["run_id"],row["dataset_id"],row["memory_type"],
            row["subject"],row["content"],row["evidence_artifact_id"],
            _parsed(row["created_at"]),
        ) for row in rows]

    def record_tool_call(self, record):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO tool_calls VALUES(%s,%s,%s,%s,%s,%s,%s)
                   ON DUPLICATE KEY UPDATE status=VALUES(status),response_json=VALUES(response_json)""",
                (record.id,record.run_id,record.tool_name,record.status,
                 json.dumps(record.request,ensure_ascii=False),
                 json.dumps(record.response,ensure_ascii=False),_dt(record.created_at)),
            )
        self.connection.commit()

    def tool_calls_for_run(self, run_id):
        return [ToolCallRecord(
            row["id"],row["run_id"],row["tool_name"],row["status"],
            self._json(row["request_json"]),self._json(row["response_json"]),
            _parsed(row["created_at"]),
        ) for row in self._all(
            "SELECT * FROM tool_calls WHERE run_id=%s ORDER BY created_at,id", (run_id,)
        )]

    def add_generation_plan(self, plan):
        feedback = (
            plan.previous_feedback.__dict__
            if plan.previous_feedback is not None
            else {}
        )
        with self.connection.cursor() as cursor:
            cursor.execute(
                """INSERT IGNORE INTO generation_plans(
                  plan_id,run_id,generation,sequence_no,generation_strategy,
                  parent_selection_policy,decision_reason,required_parent_count,
                  previous_feedback_json,context_artifact_id,prompt_artifact_id,created_at)
                  VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    plan.plan_id, plan.run_id, plan.generation, plan.sequence,
                    plan.generation_strategy, plan.parent_selection_policy,
                    plan.decision_reason, plan.required_parent_count,
                    json.dumps(feedback, ensure_ascii=False, sort_keys=True),
                    plan.context_artifact_id, plan.prompt_artifact_id,
                    _dt(plan.created_at),
                ),
            )
            created = cursor.rowcount == 1
        self.connection.commit()
        return self.generation_plan_by_id(plan.plan_id), created

    def generation_plan_by_id(self, plan_id):
        row = self._one(
            "SELECT * FROM generation_plans WHERE plan_id=%s", (plan_id,)
        )
        if not row:
            raise KeyError(plan_id)
        return self._generation_plan(row)

    def generation_plans_for_run(self, run_id):
        return [
            self._generation_plan(row)
            for row in self._all(
                """SELECT * FROM generation_plans WHERE run_id=%s
                   ORDER BY generation,generation_strategy,sequence_no""",
                (run_id,),
            )
        ]

    def record_trace(self, record):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO trace_records(
                  id,run_id,span_type,actor,status,payload_json,plan_id,candidate_id,
                  evaluation_job_id,skill_name,context_refs_json,tool_call_id,
                  latency_ms,token_usage_json,created_at)
                  VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                  ON DUPLICATE KEY UPDATE status=VALUES(status),
                  payload_json=VALUES(payload_json),latency_ms=VALUES(latency_ms),
                  token_usage_json=VALUES(token_usage_json)""",
                (
                    record.id, record.run_id, record.span_type, record.actor,
                    record.status, json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
                    record.plan_id, record.candidate_id, record.evaluation_job_id,
                    record.skill_name, json.dumps(record.context_refs, ensure_ascii=False),
                    record.tool_call_id, record.latency_ms,
                    json.dumps(record.token_usage, ensure_ascii=False, sort_keys=True),
                    _dt(record.created_at),
                ),
            )
        self.connection.commit()

    def trace_records_for_run(self, run_id):
        return [
            self._trace(row)
            for row in self._all(
                "SELECT * FROM trace_records WHERE run_id=%s ORDER BY created_at,id",
                (run_id,),
            )
        ]

    def add_task(self, task):
        with self.connection.cursor() as cursor:
            cursor.execute("""INSERT INTO tasks VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                           (task.id,task.name,task.objective,task.evaluation_budget,task.total_budget,
                            _dt(task.created_at),task.dataset_id,task.generations,
                            task.population_size,task.random_seed))
        self.connection.commit()

    def create_task_run(self, task, run, event_type, message, payload):
        event = Event(1, run.id, event_type, message, dict(payload))
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO tasks VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (task.id,task.name,task.objective,task.evaluation_budget,
                     task.total_budget,_dt(task.created_at),task.dataset_id,
                     task.generations,task.population_size,task.random_seed),
                )
                cursor.execute(
                    """INSERT INTO runs(
                      id,task_id,status,generation,consumed_evaluations,
                      reserved_evaluations,best_candidate_id,created_at,updated_at,dataset_id)
                      VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (run.id,run.task_id,run.status.value,run.generation,
                     run.consumed_evaluations,run.reserved_evaluations,
                     run.best_candidate_id,_dt(run.created_at),_dt(run.updated_at),
                     run.dataset_id),
                )
                cursor.execute(
                    "INSERT INTO run_events VALUES(%s,%s,%s,%s,%s,%s)",
                    (run.id,1,event_type,message,
                     json.dumps(payload,ensure_ascii=False),_dt(event.occurred_at)),
                )
            self.connection.commit()
            return event
        except Exception:
            self.connection.rollback()
            raise

    def add_run(self, run):
        with self.connection.cursor() as cursor:
            cursor.execute("""INSERT INTO runs(
              id,task_id,status,generation,consumed_evaluations,reserved_evaluations,
              best_candidate_id,created_at,updated_at,dataset_id)
              VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                           (run.id,run.task_id,run.status.value,run.generation,
                            run.consumed_evaluations,run.reserved_evaluations,run.best_candidate_id,
                            _dt(run.created_at),_dt(run.updated_at),run.dataset_id))
        self.connection.commit()

    def get_task(self, task_id):
        row = self._one("SELECT * FROM tasks WHERE id=%s", (task_id,))
        if not row: raise KeyError(task_id)
        return OptimizationTask(row["id"],row["name"],row["objective"],row["evaluation_budget"],
                                row["total_budget"],_parsed(row["created_at"]),row["dataset_id"],
                                row["generations"],row["population_size"],row["random_seed"])

    def get_run(self, run_id):
        row = self._one("SELECT * FROM runs WHERE id=%s", (run_id,))
        if not row: raise KeyError(run_id)
        return Run(row["id"],row["task_id"],RunStatus(row["status"]),row["generation"],
                   row["consumed_evaluations"],row["reserved_evaluations"],row["best_candidate_id"],
                   _parsed(row["created_at"]),_parsed(row["updated_at"]),row["dataset_id"],
                   bool(row["pause_requested"]),bool(row["cancel_requested"]),
                   row["control_reason"] or "",row["runtime_cursor_artifact_id"] or "",
                   row["runtime_owner_id"] or "",_parsed(row["runtime_lease_expires_at"]))

    def list_runs(self):
        return [self.get_run(row["id"]) for row in self._all("SELECT id FROM runs ORDER BY created_at DESC")]

    def save_run(self, run):
        """仅保存算法投影；控制、预算与 lease 由窄事务命令维护。"""
        with self.connection.cursor() as cursor:
            cursor.execute(
                """UPDATE runs SET generation=%s,best_candidate_id=%s,updated_at=%s
                  WHERE id=%s""",
                (run.generation,run.best_candidate_id,_dt(run.updated_at),run.id),
            )
        self.connection.commit()

    def start_run(self,run_id,expected_status,now,owner_id=""):
        expected=(expected_status.value if isinstance(expected_status,RunStatus)
                  else str(expected_status))
        query=("UPDATE runs SET status=%s,pause_requested=0,control_reason='',updated_at=%s "
               "WHERE id=%s AND status=%s AND cancel_requested=0")
        params=[RunStatus.RUNNING.value,_dt(now),run_id,expected]
        if owner_id:
            query += " AND runtime_owner_id=%s AND runtime_lease_expires_at>%s"
            params.extend((owner_id,_dt(now)))
        else:
            query += " AND runtime_owner_id=''"
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(query,tuple(params))
                if cursor.rowcount!=1:
                    raise RuntimeError("Run 启动 CAS 失败：状态、取消请求或 Runtime lease 已变化")
            self.connection.commit(); return self.get_run(run_id)
        except Exception:
            self.connection.rollback(); raise

    def pause_run(self,run_id,owner_id,now):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE runs SET status=%s,pause_requested=0,
                      runtime_owner_id='',runtime_lease_expires_at=NULL,updated_at=%s
                      WHERE id=%s AND status=%s AND runtime_owner_id=%s
                        AND runtime_lease_expires_at>%s AND cancel_requested=0""",
                    (RunStatus.PAUSED.value,_dt(now),run_id,RunStatus.RUNNING.value,
                     owner_id,_dt(now)),
                )
                if cursor.rowcount!=1:
                    raise RuntimeError("Run 暂停 CAS 失败：Runtime lease 或控制状态已变化")
            self.connection.commit(); return self.get_run(run_id)
        except Exception:
            self.connection.rollback(); raise

    def complete_run(self,run_id,owner_id,now,generation,best_candidate_id):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """SELECT id,candidate_id,budget FROM evaluation_jobs
                      WHERE run_id=%s AND status IN(%s,%s,%s) FOR UPDATE""",
                    (run_id,EvaluationJobStatus.PENDING.value,
                     EvaluationJobStatus.RETRY_WAIT.value,
                     EvaluationJobStatus.RUNNING.value),
                )
                active_jobs=list(cursor.fetchall())
                released=sum(int(row["budget"]) for row in active_jobs)
                cursor.execute(
                    """UPDATE runs SET status=%s,generation=%s,best_candidate_id=%s,
                      reserved_evaluations=0,runtime_owner_id='',
                      runtime_lease_expires_at=NULL,updated_at=%s
                      WHERE id=%s AND status=%s AND runtime_owner_id=%s
                        AND runtime_lease_expires_at>%s AND cancel_requested=0
                        AND pause_requested=0 AND reserved_evaluations=%s""",
                    (RunStatus.COMPLETED.value,int(generation),best_candidate_id,_dt(now),
                     run_id,RunStatus.RUNNING.value,owner_id,_dt(now),released),
                )
                if cursor.rowcount!=1:
                    raise RuntimeError(
                        "Run 完成 CAS 失败：Runtime lease、控制状态或预算对账已变化"
                    )
                self._cancel_terminal_work(
                    cursor,run_id,now,"RUN_COMPLETED",
                    "Run 已完成，未结算工作已终止",
                )
            self.connection.commit(); return self.get_run(run_id)
        except Exception:
            self.connection.rollback(); raise

    def fail_run(self,run_id,reason,now,owner_id=""):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """SELECT id,candidate_id,budget FROM evaluation_jobs
                      WHERE run_id=%s AND status IN(%s,%s,%s) FOR UPDATE""",
                    (run_id,EvaluationJobStatus.PENDING.value,
                     EvaluationJobStatus.RETRY_WAIT.value,
                     EvaluationJobStatus.RUNNING.value),
                )
                active_jobs=list(cursor.fetchall())
                released=sum(int(row["budget"]) for row in active_jobs)
                query=("UPDATE runs SET status=%s,control_reason=%s,"
                       "reserved_evaluations=0,runtime_owner_id='',"
                       "runtime_lease_expires_at=NULL,updated_at=%s "
                       "WHERE id=%s AND status IN (%s,%s) "
                       "AND cancel_requested=0 AND reserved_evaluations=%s")
                params=[RunStatus.FAILED.value,str(reason)[:2000],_dt(now),run_id,
                        RunStatus.PENDING.value,RunStatus.RUNNING.value,released]
                if owner_id:
                    query += " AND runtime_owner_id=%s AND runtime_lease_expires_at>%s"
                    params.extend((owner_id,_dt(now)))
                else:
                    query += " AND runtime_owner_id=''"
                cursor.execute(query,tuple(params))
                if cursor.rowcount!=1:
                    raise RuntimeError(
                        "Run 失败 CAS 未命中：状态、取消请求、Runtime lease 或预算对账已变化"
                    )
                self._cancel_terminal_work(
                    cursor,run_id,now,"RUN_FAILED",
                    "Run 已失败，未结算工作已终止",
                )
            self.connection.commit(); return self.get_run(run_id)
        except Exception:
            self.connection.rollback(); raise

    @staticmethod
    def _cancel_terminal_work(cursor,run_id,now,error_code,message):
        """在调用方已持有 MySQL 事务时终止 Run 的全部未结算工作。"""

        statuses=(EvaluationJobStatus.PENDING.value,
                  EvaluationJobStatus.RETRY_WAIT.value,
                  EvaluationJobStatus.RUNNING.value)
        cursor.execute(
            """UPDATE candidates SET status=%s WHERE id IN(
              SELECT candidate_id FROM evaluation_jobs
              WHERE run_id=%s AND status IN(%s,%s,%s))""",
            (CandidateStatus.INVALID.value,run_id,*statuses),
        )
        cursor.execute(
            """UPDATE evaluation_jobs SET status=%s,worker_id=NULL,
              lease_expires_at=NULL,error_code=%s,error_message=%s
              WHERE run_id=%s AND status IN(%s,%s,%s)""",
            (EvaluationJobStatus.CANCELLED.value,error_code,message,run_id,
             *statuses),
        )
        cursor.execute(
            """UPDATE agent_tasks SET status=%s,claimed_by='',claim_token='',
              lease_expires_at=NULL,error_message=%s,updated_at=%s
              WHERE run_id=%s AND status IN(%s,%s)""",
            (AgentTaskStatus.CANCELLED.value,message,_dt(now),run_id,
             AgentTaskStatus.PENDING.value,AgentTaskStatus.CLAIMED.value),
        )

    def update_run_progress(self,run_id,owner_id,now,generation):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """UPDATE runs SET generation=%s,updated_at=%s WHERE id=%s AND status=%s
                  AND runtime_owner_id=%s AND runtime_lease_expires_at>%s
                  AND cancel_requested=0""",
                (int(generation),_dt(now),run_id,RunStatus.RUNNING.value,
                 owner_id,_dt(now)),
            )
            updated=cursor.rowcount==1
        self.connection.commit()
        if not updated:
            raise RuntimeError("Run 进度 fencing 失败：Runtime lease 或控制状态已变化")
        return self.get_run(run_id)

    def request_run_pause(self, run_id, reason):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE runs SET pause_requested=1,control_reason=%s,updated_at=%s
                      WHERE id=%s AND status=%s AND cancel_requested=0""",
                    (str(reason)[:2000],_dt(datetime.now().astimezone()),run_id,
                     RunStatus.RUNNING.value),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Run 当前状态不允许请求暂停")
            self.connection.commit()
            return self.get_run(run_id)
        except Exception:
            self.connection.rollback()
            raise

    def request_run_cancel(self, run_id, reason):
        """在同一行锁事务内取消可取消工作，并恰好释放其预算预留。"""

        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM runs WHERE id=%s FOR UPDATE",(run_id,))
                run = cursor.fetchone()
                if not run:
                    raise KeyError(run_id)
                if run["status"] not in {
                    RunStatus.PENDING.value,RunStatus.RUNNING.value,RunStatus.PAUSED.value,
                }:
                    raise RuntimeError("Run 当前状态不允许取消")
                statuses = (
                    EvaluationJobStatus.PENDING.value,
                    EvaluationJobStatus.RETRY_WAIT.value,
                )
                cursor.execute(
                    """SELECT COALESCE(SUM(budget),0) AS amount FROM evaluation_jobs
                      WHERE run_id=%s AND status IN(%s,%s) FOR UPDATE""",
                    (run_id,*statuses),
                )
                released = int(cursor.fetchone()["amount"] or 0)
                cursor.execute(
                    """UPDATE candidates SET status=%s WHERE id IN(
                      SELECT candidate_id FROM evaluation_jobs
                      WHERE run_id=%s AND status IN(%s,%s))""",
                    (CandidateStatus.INVALID.value,run_id,*statuses),
                )
                cursor.execute(
                    """UPDATE evaluation_jobs SET status=%s,worker_id=NULL,
                      lease_expires_at=NULL,error_code='RUN_CANCELLED',
                      error_message='Run 已取消' WHERE run_id=%s AND status IN(%s,%s)""",
                    (EvaluationJobStatus.CANCELLED.value,run_id,*statuses),
                )
                cursor.execute(
                    """UPDATE agent_tasks SET status=%s,claimed_by='',claim_token='',
                      lease_expires_at=NULL,error_message='Run 已取消',updated_at=%s
                      WHERE run_id=%s AND status IN(%s,%s)""",
                    (AgentTaskStatus.CANCELLED.value,_dt(datetime.now().astimezone()),
                     run_id,AgentTaskStatus.PENDING.value,
                     AgentTaskStatus.CLAIMED.value),
                )
                cursor.execute(
                    """UPDATE runs SET status=%s,cancel_requested=1,pause_requested=0,
                      control_reason=%s,reserved_evaluations=reserved_evaluations-%s,
                      runtime_owner_id='',runtime_lease_expires_at=NULL,updated_at=%s
                      WHERE id=%s AND reserved_evaluations>=%s""",
                    (RunStatus.CANCELLED.value,str(reason)[:2000],released,
                     _dt(datetime.now().astimezone()),run_id,released),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("取消时预算 reservation 对账失败")
            self.connection.commit()
            return self.get_run(run_id)
        except Exception:
            self.connection.rollback()
            raise

    def claim_run_lease(self, run_id, owner_id, now, lease_seconds):
        if lease_seconds <= 0:
            raise ValueError("runtime lease_seconds 必须大于 0")
        expires = now + timedelta(seconds=lease_seconds)
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE runs SET runtime_owner_id=%s,runtime_lease_expires_at=%s
                      WHERE id=%s AND status IN(%s,%s) AND cancel_requested=0
                        AND (runtime_owner_id='' OR runtime_owner_id=%s
                             OR runtime_lease_expires_at IS NULL
                             OR runtime_lease_expires_at<=%s)""",
                    (owner_id,_dt(expires),run_id,RunStatus.PENDING.value,
                     RunStatus.RUNNING.value,owner_id,_dt(now)),
                )
                claimed = cursor.rowcount == 1
            self.connection.commit()
            return claimed
        except Exception:
            self.connection.rollback()
            raise

    def renew_run_lease(self, run_id, owner_id, now, lease_seconds):
        expires = now + timedelta(seconds=lease_seconds)
        with self.connection.cursor() as cursor:
            cursor.execute(
                """UPDATE runs SET runtime_lease_expires_at=%s WHERE id=%s
                  AND runtime_owner_id=%s AND status=%s AND cancel_requested=0
                  AND runtime_lease_expires_at>%s""",
                (_dt(expires),run_id,owner_id,RunStatus.RUNNING.value,_dt(now)),
            )
            renewed = cursor.rowcount == 1
            if not renewed:
                cursor.execute(
                    """SELECT id FROM runs WHERE id=%s AND runtime_owner_id=%s
                      AND status=%s AND cancel_requested=0
                      AND runtime_lease_expires_at>%s AND runtime_lease_expires_at>=%s""",
                    (run_id,owner_id,RunStatus.RUNNING.value,_dt(now),_dt(expires)),
                )
                renewed = cursor.fetchone() is not None
        self.connection.commit()
        return renewed

    def release_run_lease(self, run_id, owner_id):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """UPDATE runs SET runtime_owner_id='',runtime_lease_expires_at=NULL
                  WHERE id=%s AND runtime_owner_id=%s""",
                (run_id,owner_id),
            )
            released = cursor.rowcount == 1
        self.connection.commit()
        return released

    def update_runtime_cursor(self, run_id, artifact_id, owner_id="", now=None):
        query = "UPDATE runs SET runtime_cursor_artifact_id=%s WHERE id=%s"
        params = [artifact_id,run_id]
        if owner_id:
            now = now or datetime.now().astimezone()
            query += (" AND runtime_owner_id=%s AND status=%s AND cancel_requested=0"
                      " AND runtime_lease_expires_at>%s")
            params.extend((owner_id,RunStatus.RUNNING.value,_dt(now)))
        with self.connection.cursor() as cursor:
            cursor.execute(query,tuple(params))
            updated = cursor.rowcount == 1
        self.connection.commit()
        return updated

    def add_candidate(self, candidate):
        with self.connection.cursor() as cursor:
            cursor.execute("""INSERT INTO candidates(
                id,run_id,code,description,operators_json,lineage_json,status,
                objective,code_artifact_id,evaluation_artifact_id,generation,
                plan_id,generation_strategy,code_digest,selected_parent_ids_json,
                prior_refs_json,created_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE status=VALUES(status),objective=VALUES(objective),
                code_artifact_id=VALUES(code_artifact_id),evaluation_artifact_id=VALUES(evaluation_artifact_id),
                generation=VALUES(generation),plan_id=VALUES(plan_id),
                generation_strategy=VALUES(generation_strategy),
                code_digest=VALUES(code_digest),
                selected_parent_ids_json=VALUES(selected_parent_ids_json),
                prior_refs_json=VALUES(prior_refs_json)""",
                           (candidate.id,candidate.run_id,candidate.code,candidate.description,
                            json.dumps(candidate.operators,ensure_ascii=False),
                            json.dumps(candidate.lineage,ensure_ascii=False),candidate.status.value,
                            candidate.objective,candidate.code_artifact_id,candidate.evaluation_artifact_id,
                            candidate.generation,candidate.plan_id,candidate.generation_strategy,
                            candidate.code_digest,json.dumps(candidate.selected_parent_ids,ensure_ascii=False),
                            json.dumps(candidate.prior_refs,ensure_ascii=False),_dt(candidate.created_at)))
            for parent in candidate.lineage.get("parents", []):
                cursor.execute("INSERT IGNORE INTO candidate_parents VALUES(%s,%s)",
                               (candidate.id,parent))
        self.connection.commit()

    def _candidate(self, row):
        lineage = self._json(row["lineage_json"])
        return Candidate(row["id"],row["run_id"],row["code"],row["description"],
                         self._json(row["operators_json"]),lineage,
                         CandidateStatus(row["status"]),row["objective"],row["code_artifact_id"],
                         row["evaluation_artifact_id"],
                         int(row.get("generation") or lineage.get("generation", 0) or 0),
                         row.get("plan_id") or str(lineage.get("plan_id", "")),
                         row.get("generation_strategy") or str(lineage.get("operator", "")),
                         row.get("code_digest") or "",
                         self._json(row.get("selected_parent_ids_json") or "[]"),
                         self._json(row.get("prior_refs_json") or "[]"),
                         _parsed(row.get("created_at")) or utc_now())

    def candidates_for_run(self, run_id):
        return [self._candidate(row) for row in self._all(
            "SELECT * FROM candidates WHERE run_id=%s ORDER BY id",(run_id,))]

    def candidate_by_id(self, candidate_id):
        row=self._one("SELECT * FROM candidates WHERE id=%s",(candidate_id,))
        if not row: raise KeyError(candidate_id)
        return self._candidate(row)

    def add_result(self, result):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT run_id FROM candidates WHERE id=%s FOR UPDATE",
                    (result.candidate_id,),
                )
                candidate = cursor.fetchone()
                if candidate is None:
                    raise KeyError(result.candidate_id)
                if candidate["run_id"] != result.run_id:
                    raise EvaluationIdentityConflictError(
                        "EvaluationResult 与 Candidate 的 run 归属不一致"
                    )
                cursor.execute(
                    "SELECT * FROM evaluation_results WHERE candidate_id=%s OR id=%s FOR UPDATE",
                    (result.candidate_id, result.id),
                )
                existing = list(cursor.fetchall())
                if existing:
                    if len(existing) != 1 or not _same_result(existing[0], result):
                        raise EvaluationIdentityConflictError(
                            "Candidate 已有不同的 authoritative EvaluationResult；重评必须 clone Candidate"
                        )
                    self.connection.commit()
                    return
                cursor.execute(
                    "INSERT INTO evaluation_results VALUES(%s,%s,%s,%s,%s,%s,%s)",
                    (result.id,result.run_id,result.candidate_id,result.objective,
                     json.dumps(result.trajectory),
                     json.dumps(result.best_configuration,ensure_ascii=False),
                     result.used_budget),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "EvaluationResult 插入未产生唯一 authoritative fact"
                    )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def result_for_candidate(self, candidate_id):
        row=self._one("SELECT * FROM evaluation_results WHERE candidate_id=%s",(candidate_id,))
        if not row: raise KeyError(candidate_id)
        return EvaluationResult(row["id"],row["run_id"],row["candidate_id"],row["objective"],
                                self._json(row["trajectory_json"]),
                                self._json(row["best_configuration_json"]),row["used_budget"])

    def append_event(self, run_id, event_type, message, **payload):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT id FROM runs WHERE id=%s FOR UPDATE",(run_id,))
                if not cursor.fetchone(): raise KeyError(run_id)
                cursor.execute("SELECT COALESCE(MAX(sequence_no),0)+1 AS next_seq FROM run_events WHERE run_id=%s",(run_id,))
                sequence=cursor.fetchone()["next_seq"]
                event=Event(sequence,run_id,event_type,message,payload)
                cursor.execute("INSERT INTO run_events VALUES(%s,%s,%s,%s,%s,%s)",
                               (run_id,sequence,event_type,message,json.dumps(payload,ensure_ascii=False),_dt(event.occurred_at)))
            self.connection.commit()
            return event
        except Exception:
            self.connection.rollback(); raise

    def events_for_run(self, run_id):
        return [Event(row["sequence_no"],row["run_id"],row["event_type"],row["message"],
                      self._json(row["payload_json"]),_parsed(row["occurred_at"]))
                for row in self._all("SELECT * FROM run_events WHERE run_id=%s ORDER BY sequence_no",(run_id,))]

    def put_artifact(self, run_id, kind, content, media_type):
        digest=hashlib.sha256(content).hexdigest(); run_key=hashlib.sha256(run_id.encode()).hexdigest()[:8]
        artifact_id="artifact-{}-{}-{}".format(run_key,kind.lower(),digest[:16])
        run_root=self.artifact_root/run_id; run_root.mkdir(parents=True,exist_ok=True)
        path=run_root/artifact_id; temp=run_root/(artifact_id+".tmp-{}".format(os.getpid()))
        temp.write_bytes(content)
        if hashlib.sha256(temp.read_bytes()).hexdigest()!=digest:
            temp.unlink(missing_ok=True); raise ArtifactIntegrityError("artifact digest 校验失败")
        os.replace(str(temp),str(path))
        value=ArtifactMetadata(artifact_id,run_id,kind,media_type,len(content),digest,str(path.resolve()))
        with self.connection.cursor() as cursor:
            cursor.execute("""INSERT INTO artifacts VALUES(%s,%s,%s,%s,%s,%s,%s)
              ON DUPLICATE KEY UPDATE uri=VALUES(uri),digest=VALUES(digest),size_bytes=VALUES(size_bytes)""",
                           (value.id,value.run_id,value.kind,value.media_type,value.size,value.digest,value.uri))
        self.connection.commit(); return value

    def artifact_content(self, artifact_id):
        row=self._one("SELECT uri,digest FROM artifacts WHERE id=%s",(artifact_id,))
        if not row: raise KeyError(artifact_id)
        try: content=Path(row["uri"]).read_bytes()
        except FileNotFoundError as exc: raise ArtifactIntegrityError("artifact 文件缺失") from exc
        if hashlib.sha256(content).hexdigest()!=row["digest"]: raise ArtifactIntegrityError("artifact 已损坏")
        return content

    def artifacts_for_run(self, run_id):
        return [ArtifactMetadata(row["id"],row["run_id"],row["kind"],row["media_type"],
                                 row["size_bytes"],row["digest"],row["uri"])
                for row in self._all("SELECT * FROM artifacts WHERE run_id=%s ORDER BY id",(run_id,))]

    def add_checkpoint(self, checkpoint):
        with self.connection.cursor() as cursor:
            cursor.execute("INSERT IGNORE INTO checkpoints VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                           (checkpoint.id,checkpoint.run_id,checkpoint.generation,
                            json.dumps(checkpoint.population_ids),checkpoint.consumed_budget,
                            checkpoint.remaining_budget,checkpoint.artifact_id,checkpoint.schema_version,
                            checkpoint.code_version,_dt(checkpoint.created_at),checkpoint.dataset_id,
                            json.dumps(checkpoint.prior_refs)))
            for candidate_id in checkpoint.population_ids:
                cursor.execute("INSERT IGNORE INTO population_memberships VALUES(%s,%s,%s,'MEMBER')",
                               (checkpoint.run_id,checkpoint.generation,candidate_id))
        self.connection.commit()

    def latest_checkpoint(self, run_id):
        row=self._one("""SELECT checkpoints.* FROM checkpoints
          JOIN runs ON runs.id=checkpoints.run_id WHERE checkpoints.run_id=%s
          ORDER BY CASE
            WHEN runs.runtime_cursor_artifact_id<>''
             AND checkpoints.artifact_id=runs.runtime_cursor_artifact_id
            THEN 0 WHEN runs.runtime_cursor_artifact_id='' THEN 0 ELSE 1 END,
            checkpoints.created_at DESC,checkpoints.id DESC LIMIT 1""",(run_id,))
        if not row: raise KeyError(run_id)
        return CheckpointMetadata(row["id"],row["run_id"],row["generation"],
            self._json(row["population_ids_json"]),row["consumed_budget"],row["remaining_budget"],
            row["artifact_id"],row["schema_version"],row["code_version"],_parsed(row["created_at"]),
            row["dataset_id"],self._json(row["prior_refs_json"]))

    def _job(self,row):
        return EvaluationJob(row["id"],row["run_id"],row["task_id"],row["candidate_id"],
            row["seed"],row["budget"],row["idempotency_key"],EvaluationJobStatus(row["status"]),
            row["attempts"],row["max_attempts"],_parsed(row["available_at"]),
            _parsed(row["lease_expires_at"]),row["worker_id"],row["result_id"],
            row["error_code"],row["error_message"])

    def get_evaluation_job(self,job_id):
        row=self._one("SELECT * FROM evaluation_jobs WHERE id=%s",(job_id,))
        if not row: raise KeyError(job_id)
        return self._job(row)

    def evaluation_jobs_for_run(self,run_id):
        return [self._job(row) for row in self._all(
            "SELECT * FROM evaluation_jobs WHERE run_id=%s ORDER BY id",(run_id,))]

    def enqueue_job(self,job):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM runs WHERE id=%s FOR UPDATE",(job.run_id,)); run=cursor.fetchone()
                cursor.execute("SELECT * FROM tasks WHERE id=%s",(job.task_id,)); task=cursor.fetchone()
                cursor.execute(
                    "SELECT * FROM candidates WHERE id=%s FOR UPDATE",
                    (job.candidate_id,),
                )
                candidate = cursor.fetchone()
                if not run or not task or not candidate:
                    raise KeyError(job.run_id)
                if run["task_id"] != job.task_id:
                    raise EvaluationIdentityConflictError(
                        "EvaluationJob 的 Task 不属于目标 Run"
                    )
                if candidate["run_id"] != job.run_id:
                    raise EvaluationIdentityConflictError(
                        "EvaluationJob 的 Candidate 不属于目标 Run"
                    )
                if (bool(run["cancel_requested"])
                        or run["status"]!=RunStatus.RUNNING.value):
                    raise RuntimeError(
                        "Run 非 RUNNING，拒绝创建或重放 EvaluationJob"
                    )
                cursor.execute(
                    "SELECT * FROM evaluation_jobs WHERE candidate_id=%s FOR UPDATE",
                    (job.candidate_id,),
                )
                candidate_jobs = list(cursor.fetchall())
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
                cursor.execute(
                    "SELECT * FROM evaluation_jobs WHERE idempotency_key=%s FOR UPDATE",
                    (job.idempotency_key,),
                )
                if cursor.fetchone() is not None:
                    raise EvaluationIdentityConflictError(
                        "EvaluationJob idempotency key 与其他 Candidate 冲突"
                    )
                cursor.execute(
                    "SELECT id FROM evaluation_results WHERE candidate_id=%s FOR UPDATE",
                    (job.candidate_id,),
                )
                if cursor.fetchone() is not None:
                    raise EvaluationIdentityConflictError(
                        "Candidate 已有 authoritative EvaluationResult，但缺少可验证的原 Job；重评必须 clone Candidate"
                    )
                if job.budget <= 0 or job.max_attempts <= 0:
                    raise ValueError("EvaluationJob budget/max_attempts 必须大于 0")
                available=task["total_budget"]-run["consumed_evaluations"]-run["reserved_evaluations"]
                if job.budget>available:
                    raise BudgetExhaustedError("预算不足：需要 {}，剩余可预留 {}".format(job.budget,available))
                cursor.execute("INSERT INTO evaluation_jobs VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL,NULL,NULL,NULL,NULL)",
                    (job.id,job.run_id,job.task_id,job.candidate_id,job.seed,job.budget,
                     job.idempotency_key,job.status.value,job.attempts,job.max_attempts,_dt(job.available_at)))
                cursor.execute(
                    "UPDATE candidates SET status=%s WHERE id=%s AND run_id=%s",
                    (CandidateStatus.EVALUATION_PENDING.value,job.candidate_id,
                     job.run_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "EvaluationJob Candidate 归属校验发生并发冲突"
                    )
                cursor.execute("UPDATE runs SET reserved_evaluations=reserved_evaluations+%s WHERE id=%s",
                               (job.budget,job.run_id))
            self.connection.commit(); return job,True
        except Exception:
            self.connection.rollback(); raise

    def claim_next_job(self,worker_id,now,lease_seconds):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute("""SELECT evaluation_jobs.* FROM evaluation_jobs
                  JOIN runs ON runs.id=evaluation_jobs.run_id
                  WHERE evaluation_jobs.status IN(%s,%s)
                    AND evaluation_jobs.available_at<=%s
                    AND runs.cancel_requested=0 AND runs.status=%s
                  ORDER BY evaluation_jobs.available_at,evaluation_jobs.id
                  LIMIT 1 FOR UPDATE SKIP LOCKED""",
                  (EvaluationJobStatus.PENDING.value,EvaluationJobStatus.RETRY_WAIT.value,
                   _dt(now),RunStatus.RUNNING.value))
                row=cursor.fetchone()
                if not row: self.connection.commit(); return None
                lease=now+timedelta(seconds=lease_seconds)
                cursor.execute("""UPDATE evaluation_jobs SET status=%s,attempts=attempts+1,
                  lease_expires_at=%s,worker_id=%s WHERE id=%s AND status IN(%s,%s)""",
                  (EvaluationJobStatus.RUNNING.value,_dt(lease),worker_id,row["id"],
                   EvaluationJobStatus.PENDING.value,EvaluationJobStatus.RETRY_WAIT.value))
                if cursor.rowcount!=1: self.connection.rollback(); return None
                cursor.execute("UPDATE candidates SET status=%s WHERE id=%s",
                               (CandidateStatus.EVALUATING.value,row["candidate_id"]))
            self.connection.commit(); return self.get_evaluation_job(row["id"])
        except Exception:
            self.connection.rollback(); raise

    def renew_job_lease(self,job_id,worker_id,now,lease_seconds):
        """仅允许当前、尚未过期的 RUNNING owner 延长 lease。"""
        if lease_seconds<=0:
            raise ValueError("lease_seconds 必须大于 0")
        try:
            self.connection.begin()
            lease=now+timedelta(seconds=lease_seconds)
            with self.connection.cursor() as cursor:
                cursor.execute("""UPDATE evaluation_jobs SET lease_expires_at=%s
                  WHERE id=%s AND status=%s AND worker_id=%s
                    AND lease_expires_at IS NOT NULL AND lease_expires_at>%s""",
                  (_dt(lease),job_id,EvaluationJobStatus.RUNNING.value,worker_id,_dt(now)))
                renewed=cursor.rowcount==1
                # MySQL 默认只报告实际变更行数；固定时钟下，claim 后立即续租会
                # 写入同一时间戳而返回 0。此时只读确认 fencing 条件仍成立。
                if not renewed:
                    cursor.execute("""SELECT id FROM evaluation_jobs
                      WHERE id=%s AND status=%s AND worker_id=%s
                        AND lease_expires_at IS NOT NULL
                        AND lease_expires_at>%s AND lease_expires_at>=%s""",
                      (job_id,EvaluationJobStatus.RUNNING.value,worker_id,
                       _dt(now),_dt(lease)))
                    renewed=cursor.fetchone() is not None
            self.connection.commit(); return renewed
        except Exception:
            self.connection.rollback(); raise

    def complete_job_success(self,job_id,worker_id,result,artifact_id,now):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM evaluation_jobs WHERE id=%s FOR UPDATE",(job_id,)); row=cursor.fetchone()
                if not row: raise KeyError(job_id)
                job=self._job(row)
                if (job.status!=EvaluationJobStatus.RUNNING or job.worker_id!=worker_id
                        or job.lease_expires_at is None or job.lease_expires_at<=now):
                    raise RuntimeError("评价 lease 已丢失：仅当前 RUNNING owner 可以成功结算")
                cursor.execute(
                    "SELECT status,cancel_requested FROM runs WHERE id=%s FOR UPDATE",
                    (job.run_id,),
                )
                active_run = cursor.fetchone()
                if (
                    active_run is None
                    or active_run["status"] != RunStatus.RUNNING.value
                    or bool(active_run["cancel_requested"])
                ):
                    raise RuntimeError("Run 已不处于 active 状态，拒绝评价成功结算")
                if result.run_id!=job.run_id or result.candidate_id!=job.candidate_id:
                    raise RuntimeError("评价结果与 job 的 run/candidate 不一致")
                if result.used_budget<0 or result.used_budget>job.budget:
                    raise RuntimeError("评价实际预算必须位于 0 到预留预算之间")
                cursor.execute(
                    "SELECT run_id FROM candidates WHERE id=%s FOR UPDATE",
                    (job.candidate_id,),
                )
                candidate = cursor.fetchone()
                if candidate is None or candidate["run_id"] != job.run_id:
                    raise EvaluationIdentityConflictError(
                        "EvaluationJob 与 Candidate 的 durable 归属不一致"
                    )
                cursor.execute(
                    "SELECT id FROM evaluation_results WHERE candidate_id=%s OR id=%s FOR UPDATE",
                    (result.candidate_id, result.id),
                )
                if cursor.fetchone() is not None:
                    raise EvaluationIdentityConflictError(
                        "Candidate 或 Result ID 已绑定 authoritative EvaluationResult；拒绝第二次成功结算"
                    )
                cursor.execute("""UPDATE evaluation_jobs SET status=%s,result_id=%s,lease_expires_at=NULL,
                  worker_id=NULL,error_code=NULL,error_message=NULL
                  WHERE id=%s AND status=%s AND worker_id=%s
                    AND lease_expires_at IS NOT NULL AND lease_expires_at>%s""",
                  (EvaluationJobStatus.SUCCESS.value,result.id,job_id,
                   EvaluationJobStatus.RUNNING.value,worker_id,_dt(now)))
                if cursor.rowcount!=1:
                    raise RuntimeError("评价 lease 已丢失：成功结算 fencing 条件未命中")
                cursor.execute("INSERT INTO evaluation_results VALUES(%s,%s,%s,%s,%s,%s,%s)",
                    (result.id,result.run_id,result.candidate_id,result.objective,
                     json.dumps(result.trajectory),json.dumps(result.best_configuration,ensure_ascii=False),result.used_budget))
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "EvaluationResult 插入未产生唯一 authoritative fact"
                    )
                cursor.execute(
                    """UPDATE candidates SET status=%s,objective=%s,
                      evaluation_artifact_id=%s WHERE id=%s AND run_id=%s""",
                    (CandidateStatus.EVALUATED.value,result.objective,artifact_id,
                     result.candidate_id,result.run_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("EvaluationResult Candidate 更新发生并发归属冲突")
                cursor.execute("""UPDATE runs SET reserved_evaluations=reserved_evaluations-%s,
                  consumed_evaluations=consumed_evaluations+%s
                  WHERE id=%s AND reserved_evaluations>=%s""",
                  (job.budget,result.used_budget,job.run_id,job.budget))
                if cursor.rowcount!=1:
                    raise RuntimeError("评价预算预留不足，拒绝成功结算")
            self.connection.commit(); return self.get_evaluation_job(job_id)
        except Exception:
            self.connection.rollback(); raise

    def complete_job_failure(self,job_id,worker_id,error_code,message,retryable,available_at,now):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM evaluation_jobs WHERE id=%s FOR UPDATE",(job_id,)); row=cursor.fetchone()
                if not row: raise KeyError(job_id)
                job=self._job(row)
                if (job.status!=EvaluationJobStatus.RUNNING or job.worker_id!=worker_id
                        or job.lease_expires_at is None or job.lease_expires_at<=now):
                    raise RuntimeError("评价 lease 已丢失：仅当前 RUNNING owner 可以失败结算")
                cursor.execute(
                    "SELECT status,cancel_requested FROM runs WHERE id=%s FOR UPDATE",
                    (job.run_id,),
                )
                active_run = cursor.fetchone()
                if (
                    active_run is None
                    or active_run["status"] != RunStatus.RUNNING.value
                    or bool(active_run["cancel_requested"])
                ):
                    raise RuntimeError("Run 已不处于 active 状态，拒绝评价失败结算")
                retry=retryable and job.attempts<job.max_attempts
                status=EvaluationJobStatus.RETRY_WAIT if retry else EvaluationJobStatus.DEAD
                cursor.execute("""UPDATE evaluation_jobs SET status=%s,available_at=%s,lease_expires_at=NULL,
                  worker_id=NULL,error_code=%s,error_message=%s
                  WHERE id=%s AND status=%s AND worker_id=%s
                    AND lease_expires_at IS NOT NULL AND lease_expires_at>%s""",
                  (status.value,_dt(available_at),error_code,message,job_id,
                   EvaluationJobStatus.RUNNING.value,worker_id,_dt(now)))
                if cursor.rowcount!=1:
                    raise RuntimeError("评价 lease 已丢失：失败结算 fencing 条件未命中")
                if status==EvaluationJobStatus.DEAD:
                    cursor.execute("""UPDATE runs SET reserved_evaluations=reserved_evaluations-%s
                      WHERE id=%s AND reserved_evaluations>=%s""",
                      (job.budget,job.run_id,job.budget))
                    if cursor.rowcount!=1:
                        raise RuntimeError("评价预算预留不足，拒绝失败结算")
                    cursor.execute("""UPDATE candidates SET status=%s
                      WHERE id=%s AND status IN(%s,%s)""",
                      (CandidateStatus.INVALID.value,job.candidate_id,
                       CandidateStatus.EVALUATING.value,
                       CandidateStatus.EVALUATION_PENDING.value))
            self.connection.commit(); return self.get_evaluation_job(job_id)
        except Exception:
            self.connection.rollback(); raise

    def recover_stale_jobs(self,now,run_id=""):
        try:
            self.connection.begin()
            with self.connection.cursor() as cursor:
                query="SELECT * FROM evaluation_jobs WHERE status=%s AND lease_expires_at<=%s"
                params=[EvaluationJobStatus.RUNNING.value,_dt(now)]
                if run_id:
                    query += " AND run_id=%s"
                    params.append(run_id)
                cursor.execute(query+" FOR UPDATE",tuple(params)); rows=cursor.fetchall()
                for row in rows:
                    job=self._job(row)
                    cursor.execute(
                        "SELECT cancel_requested,status FROM runs WHERE id=%s FOR UPDATE",
                        (job.run_id,),
                    )
                    run=cursor.fetchone()
                    cancelled=bool(run["cancel_requested"]) or run["status"]==RunStatus.CANCELLED.value
                    if cancelled:
                        cursor.execute("""UPDATE evaluation_jobs SET status=%s,worker_id=NULL,
                          lease_expires_at=NULL,error_code='RUN_CANCELLED',
                          error_message='Run 已取消，stale running job 已结算' WHERE id=%s""",
                          (EvaluationJobStatus.CANCELLED.value,job.id))
                        cursor.execute("""UPDATE runs SET reserved_evaluations=reserved_evaluations-%s
                          WHERE id=%s AND reserved_evaluations>=%s""",
                          (job.budget,job.run_id,job.budget))
                        cursor.execute("UPDATE candidates SET status=%s WHERE id=%s",
                                       (CandidateStatus.INVALID.value,job.candidate_id))
                    elif job.attempts>=job.max_attempts:
                        cursor.execute("""UPDATE evaluation_jobs SET status=%s,worker_id=NULL,
                          lease_expires_at=NULL,error_code='LEASE_EXHAUSTED',
                          error_message='worker lease 多次过期，已进入 dead letter' WHERE id=%s""",
                          (EvaluationJobStatus.DEAD.value,job.id))
                        cursor.execute("UPDATE runs SET reserved_evaluations=reserved_evaluations-%s WHERE id=%s",
                                       (job.budget,job.run_id))
                        cursor.execute("UPDATE candidates SET status=%s WHERE id=%s",
                                       (CandidateStatus.INVALID.value,job.candidate_id))
                    else:
                        cursor.execute("""UPDATE evaluation_jobs SET status=%s,available_at=%s,worker_id=NULL,
                          lease_expires_at=NULL,error_code='LEASE_EXPIRED',
                          error_message='worker lease 过期，已重新入队' WHERE id=%s""",
                          (EvaluationJobStatus.PENDING.value,_dt(now),job.id))
            self.connection.commit(); return len(rows)
        except Exception:
            self.connection.rollback(); raise

    def sync_dataset(self,info):
        from prievo_agent.domain.models import utc_now
        with self.connection.cursor() as cursor:
            cursor.execute("""INSERT INTO datasets_metadata VALUES(%s,%s,%s,%s,%s)
              ON DUPLICATE KEY UPDATE filename=VALUES(filename),row_count=VALUES(row_count),
              digest=VALUES(digest),validated_at=VALUES(validated_at)""",
              (info.id,info.filename,info.row_count,info.digest,_dt(utc_now())))
        self.connection.commit()

    def _one(self,sql,params=()):
        self.connection.ping(reconnect=True)
        with self.connection.cursor() as cursor:
            cursor.execute(sql,params)
            value = cursor.fetchone()
        # autocommit=False 下只读查询也会打开 REPEATABLE READ snapshot；若不在
        # 这里结束事务，同一 Store 会看不到其他 Worker 刚提交的 claim/result。
        self.connection.commit()
        return value

    def _all(self,sql,params=()):
        self.connection.ping(reconnect=True)
        with self.connection.cursor() as cursor:
            cursor.execute(sql,params)
            values = list(cursor.fetchall())
        self.connection.commit()
        return values

    def _generation_plan(self, row):
        feedback_payload = self._json(row["previous_feedback_json"] or "{}")
        feedback = (
            PreviousPlanFeedback(**feedback_payload)
            if feedback_payload else None
        )
        return GenerationPlan(
            plan_id=row["plan_id"],
            run_id=row["run_id"],
            generation=row["generation"],
            sequence=row["sequence_no"],
            generation_strategy=row["generation_strategy"],
            parent_selection_policy=row["parent_selection_policy"],
            decision_reason=row["decision_reason"],
            required_parent_count=row["required_parent_count"],
            previous_feedback=feedback,
            context_artifact_id=row["context_artifact_id"],
            prompt_artifact_id=row["prompt_artifact_id"],
            created_at=_parsed(row["created_at"]),
        )

    def _trace(self, row):
        return TraceRecord(
            id=row["id"],
            run_id=row["run_id"],
            span_type=row["span_type"],
            actor=row["actor"],
            status=row["status"],
            payload=self._json(row["payload_json"] or "{}"),
            plan_id=row["plan_id"],
            candidate_id=row["candidate_id"],
            evaluation_job_id=row["evaluation_job_id"],
            skill_name=row["skill_name"],
            context_refs=self._json(row["context_refs_json"] or "[]"),
            tool_call_id=row["tool_call_id"],
            latency_ms=row["latency_ms"],
            token_usage=self._json(row["token_usage_json"] or "{}"),
            created_at=_parsed(row["created_at"]),
        )

    @staticmethod
    def _json(value):
        return json.loads(value) if isinstance(value,str) else value
