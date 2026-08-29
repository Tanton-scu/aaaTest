"""Durable AgentTask 的单次领取与执行边界。

Dispatcher 只负责 capability 路由、任务状态、Blackboard 投影、Agent handler、
结构化 Artifact 写入和审计 Event。它不读取或修改 Run 状态、budget、Candidate、
population 或 PriEvO strategy。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import datetime, timezone
from typing import Iterable

from prievo_agent.agents.registry import AgentNotRegisteredError, AgentRegistry
from prievo_agent.application.blackboard import Blackboard
from prievo_agent.domain.events import EventType
from prievo_agent.domain.errors import LLMTimeoutError
from prievo_agent.domain.models import AgentTaskStatus


@dataclass(frozen=True)
class DispatchResult:
    """一次 dispatch 的明确结果；失败异常也会携带该对象。"""

    task_id: str
    status: AgentTaskStatus
    agent_name: str = ""
    artifact_refs: tuple[str, ...] = ()
    executed: bool = False
    will_retry: bool = False
    error_message: str = ""


class AgentDispatchError(RuntimeError):
    """Handler/Artifact/路由失败；durable task 状态已先行结算。"""

    def __init__(self, result: DispatchResult) -> None:
        self.result = result
        super().__init__(result.error_message or "AgentTask dispatch 失败")


class AgentTaskDispatcher:
    """领取并执行一个 durable AgentTask。

    ``artifact_writer`` 契约为 ``artifact_writer(task, handler_result)``，返回
    一个或多个 ArtifactMetadata/Artifact ID。Writer 必须先把 Artifact 写入同一
    durable store；Dispatcher 校验引用后才允许任务进入 COMPLETED。
    """

    def __init__(
        self, store, registry: AgentRegistry, artifact_writer,
        clock=None, lease_seconds: int = 300,
    ) -> None:
        self.store = store
        self.registry = registry
        self.artifact_writer = artifact_writer
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lease_seconds = int(lease_seconds)
        if self.lease_seconds <= 0:
            raise ValueError("AgentTask lease_seconds 必须大于 0")

    def dispatch(self, task_id: str) -> DispatchResult:
        task = self.store.get_agent_task(task_id)
        if task.status != AgentTaskStatus.PENDING:
            return self._not_executed(task)

        try:
            registration = self.registry.registration_for(
                task.required_capability
            )
        except (AgentNotRegisteredError, ValueError) as exc:
            result = DispatchResult(
                task.id,
                task.status,
                executed=False,
                will_retry=True,
                error_message=str(exc),
            )
            self.store.append_event(
                task.run_id,
                EventType.AGENT_TASK_FAILED.value,
                "AgentTask 路由失败，等待 capability 注册后重试",
                agent_task_id=task.id,
                required_capability=task.required_capability.value,
                failure_stage="ROUTING",
                will_retry=True,
                error_type=type(exc).__name__,
            )
            raise AgentDispatchError(result) from exc

        try:
            claimed = self.store.claim_agent_task(
                task.id, registration.name, self.clock(), self.lease_seconds
            )
        except RuntimeError as exc:
            # 另一个 Dispatcher 可能刚刚领取或完成；刷新 durable state，重复
            # dispatch 不执行 handler，也不制造虚假的 FAILED Event。
            current = self.store.get_agent_task(task.id)
            if current.status != AgentTaskStatus.PENDING:
                return self._not_executed(current, registration.name)
            result = DispatchResult(
                current.id,
                current.status,
                registration.name,
                executed=False,
                will_retry=True,
                error_message=str(exc),
            )
            raise AgentDispatchError(result) from exc

        self.store.append_event(
            claimed.run_id,
            EventType.AGENT_TASK_CLAIMED.value,
            "{} 已领取 AgentTask".format(registration.name),
            agent_task_id=claimed.id,
            agent=registration.name,
            required_capability=claimed.required_capability.value,
            attempt=claimed.attempts,
        )

        try:
            board = Blackboard.from_store(self.store, claimed.run_id)
            handler_result = registration.handler.handle(claimed, board)
            if not self.store.renew_agent_task_lease(
                claimed.id, claimed.claim_token, self.clock(), self.lease_seconds
            ):
                raise RuntimeError(
                    "AgentTask lease 已在 handler 后丢失，拒绝写入/结算旧 owner 结果"
                )
            artifact_refs = self._write_artifacts(claimed, handler_result)
            completed = self.store.complete_agent_task(
                claimed.id, claimed.claim_token, artifact_refs, self.clock()
            )
        except Exception as exc:
            return self._fail_and_raise(claimed, registration.name, exc)

        self.store.append_event(
            completed.run_id,
            EventType.AGENT_TASK_COMPLETED.value,
            "{} 已完成 AgentTask".format(registration.name),
            agent_task_id=completed.id,
            agent=registration.name,
            required_capability=completed.required_capability.value,
            attempt=completed.attempts,
            output_artifact_refs=list(artifact_refs),
        )
        return DispatchResult(
            completed.id,
            completed.status,
            registration.name,
            tuple(artifact_refs),
            executed=True,
        )

    def dispatch_with_retries(self, task_id: str, retry_predicate=None) -> DispatchResult:
        """对模型边界的可重试失败进行有界同步 redrive。

        上限完全来自 durable ``AgentTask.max_attempts``；每次失败先由
        :meth:`dispatch` 持久化并增加 attempts，再决定是否重驱，因此不会形成
        无状态 busy loop。瞬时 provider 错误与 malformed structured output 都可
        重新请求；持续 malformed 在恰好耗尽 max_attempts 后进入 FAILED。
        业务/编程错误默认不自动重试，仍原样向上抛出。
        """

        should_retry = retry_predicate or _bounded_model_redrive
        while True:
            try:
                return self.dispatch(task_id)
            except AgentDispatchError as exc:
                cause = exc.__cause__ or exc
                if not exc.result.will_retry or not should_retry(cause):
                    raise
                current = self.store.get_agent_task(task_id)
                # fail_agent_task 已保证 attempts < max_attempts 才回到 PENDING；
                # 这里保留显式 guard，防止自定义 Store 破坏该约定。
                if current.attempts >= current.max_attempts:
                    raise
                self.store.append_event(
                    current.run_id,
                    "AGENT_TASK_REDRIVE_SCHEDULED",
                    "Agent 模型边界失败，按 durable max_attempts 有界重驱",
                    agent_task_id=current.id,
                    failure_type=type(cause).__name__,
                    completed_attempts=current.attempts,
                    next_attempt=current.attempts + 1,
                    max_attempts=current.max_attempts,
                )

    def _write_artifacts(self, task, handler_result) -> tuple[str, ...]:
        raw_refs = self.artifact_writer(task, handler_result)
        refs = _artifact_refs(raw_refs)
        if not refs:
            raise ValueError("artifact_writer 必须生成至少一个 Artifact ref")
        existing = {
            artifact.id
            for artifact in self.store.artifacts_for_run(task.run_id)
        }
        missing = [ref for ref in refs if ref not in existing]
        if missing:
            raise ValueError(
                "artifact_writer 返回了未持久化或属于其他 Run 的引用：{}".format(
                    ", ".join(missing)
                )
            )
        return refs

    def _fail_and_raise(self, task, agent_name: str, exc: Exception):
        try:
            failed = self.store.fail_agent_task(
                task.id, task.claim_token, str(exc), self.clock()
            )
        except RuntimeError as fencing_exc:
            current = self.store.get_agent_task(task.id)
            result = DispatchResult(
                current.id,
                current.status,
                agent_name,
                executed=True,
                will_retry=False,
                error_message=str(fencing_exc),
            )
            # 旧 owner 不得生成虚假的 AGENT_TASK_FAILED；权威状态属于接管者。
            raise AgentDispatchError(result) from exc
        will_retry = failed.status == AgentTaskStatus.PENDING
        self.store.append_event(
            failed.run_id,
            EventType.AGENT_TASK_FAILED.value,
            "{} 执行 AgentTask 失败{}".format(
                agent_name, "，已重新入队" if will_retry else "，已进入终态"
            ),
            agent_task_id=failed.id,
            agent=agent_name,
            required_capability=failed.required_capability.value,
            attempt=failed.attempts,
            will_retry=will_retry,
            error_type=type(exc).__name__,
        )
        result = DispatchResult(
            failed.id,
            failed.status,
            agent_name,
            executed=True,
            will_retry=will_retry,
            error_message=str(exc),
        )
        raise AgentDispatchError(result) from exc

    @staticmethod
    def _not_executed(task, agent_name: str = "") -> DispatchResult:
        return DispatchResult(
            task.id,
            task.status,
            agent_name or task.claimed_by,
            tuple(task.output_artifact_refs),
            executed=False,
            will_retry=task.status == AgentTaskStatus.PENDING,
            error_message=task.error_message,
        )


def _artifact_refs(values) -> tuple[str, ...]:
    if isinstance(values, str) or hasattr(values, "id"):
        values = [values]
    if values is None:
        return ()
    if not isinstance(values, Iterable):
        raise TypeError("artifact_writer 返回值必须是 Artifact ref 或其 iterable")
    refs = []
    for value in values:
        ref = value if isinstance(value, str) else getattr(value, "id", None)
        if not isinstance(ref, str) or not ref.strip():
            raise TypeError("Artifact ref 必须是非空字符串或带 id 的 metadata")
        refs.append(ref)
    return tuple(dict.fromkeys(refs))


def _bounded_model_redrive(exc: BaseException) -> bool:
    """只识别模型协议/瞬时边界，不吞掉任意领域或代码错误。"""

    if isinstance(exc, (TimeoutError, ConnectionError, LLMTimeoutError, json.JSONDecodeError)):
        return True
    return type(exc).__name__.startswith("Malformed")
