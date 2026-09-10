"""Evaluation failure 的纯证据分类与 Retry/Repair/Dead 决策。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from prievo_agent.domain.errors import (
    AlgorithmOOMError,
    AlgorithmTimeoutError,
    CandidateInterfaceError,
    CandidateInvalidError,
    CandidateLogicError,
    CandidateRuntimeError,
    CandidateSyntaxError,
    InfrastructureTimeoutError,
    NetworkEvaluationError,
    TransientEvaluationError,
    TransientInfrastructureError,
    WorkerCrashError,
)


class FailureType(str, Enum):
    NETWORK = "NETWORK"
    WORKER_CRASH = "WORKER_CRASH"
    TRANSIENT_INFRA = "TRANSIENT_INFRA"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    INTERFACE_ERROR = "INTERFACE_ERROR"
    ALGORITHM_TIMEOUT = "ALGORITHM_TIMEOUT"
    ALGORITHM_OOM = "ALGORITHM_OOM"
    LOGIC_FAILURE = "LOGIC_FAILURE"
    UNKNOWN = "UNKNOWN"


class FailureAction(str, Enum):
    RETRY = "RETRY"
    REPAIR = "REPAIR"
    DEAD = "DEAD"


_ACTIONS = {
    FailureType.NETWORK: FailureAction.RETRY,
    FailureType.WORKER_CRASH: FailureAction.RETRY,
    FailureType.TRANSIENT_INFRA: FailureAction.RETRY,
    FailureType.SYNTAX_ERROR: FailureAction.REPAIR,
    FailureType.RUNTIME_ERROR: FailureAction.REPAIR,
    FailureType.INTERFACE_ERROR: FailureAction.REPAIR,
    FailureType.ALGORITHM_TIMEOUT: FailureAction.REPAIR,
    FailureType.ALGORITHM_OOM: FailureAction.REPAIR,
    FailureType.LOGIC_FAILURE: FailureAction.REPAIR,
    FailureType.UNKNOWN: FailureAction.DEAD,
}


@dataclass(frozen=True)
class FailureDecision:
    failure_type: FailureType
    action: FailureAction
    reason: str
    error_code: str = ""
    return_code: Optional[int] = None
    stderr_excerpt: str = ""

    @property
    def retryable(self) -> bool:
        return self.action == FailureAction.RETRY

    @property
    def repairable(self) -> bool:
        return self.action == FailureAction.REPAIR


class FailureClassifier:
    """只使用异常类型和显式 process evidence，不根据 Agent 猜测分类。"""

    def classify(
        self,
        exception: Optional[BaseException] = None,
        error_code: str = "",
        return_code: Optional[int] = None,
        stderr: str = "",
    ) -> FailureDecision:
        normalized_code = str(error_code or "").strip().upper()
        stderr_text = str(stderr or "")

        by_exception = self._from_exception(exception)
        if by_exception is not None:
            return self._decision(
                by_exception,
                "由明确异常类型 {} 分类".format(type(exception).__name__),
                normalized_code,
                return_code,
                stderr_text,
            )

        by_code = _ERROR_CODES.get(normalized_code)
        if by_code is not None:
            return self._decision(
                by_code,
                "由明确 error_code {} 分类".format(normalized_code),
                normalized_code,
                return_code,
                stderr_text,
            )

        by_process = self._from_process_evidence(return_code, stderr_text)
        if by_process is not None:
            failure_type, reason = by_process
            return self._decision(
                failure_type,
                reason,
                normalized_code,
                return_code,
                stderr_text,
            )

        return self._decision(
            FailureType.UNKNOWN,
            "没有足够的异常、error_code 或进程证据",
            normalized_code,
            return_code,
            stderr_text,
        )

    @staticmethod
    def _from_exception(exception: Optional[BaseException]):
        if exception is None:
            return None
        if isinstance(exception, NetworkEvaluationError):
            return FailureType.NETWORK
        if isinstance(exception, WorkerCrashError):
            return FailureType.WORKER_CRASH
        if isinstance(
            exception,
            (InfrastructureTimeoutError, TransientInfrastructureError),
        ):
            return FailureType.TRANSIENT_INFRA
        if isinstance(exception, CandidateSyntaxError) or isinstance(
            exception, SyntaxError
        ):
            return FailureType.SYNTAX_ERROR
        if isinstance(exception, CandidateRuntimeError):
            return FailureType.RUNTIME_ERROR
        if isinstance(exception, CandidateInterfaceError):
            return FailureType.INTERFACE_ERROR
        if isinstance(exception, AlgorithmTimeoutError):
            return FailureType.ALGORITHM_TIMEOUT
        if isinstance(exception, AlgorithmOOMError) or isinstance(
            exception, MemoryError
        ):
            return FailureType.ALGORITHM_OOM
        if isinstance(exception, CandidateLogicError):
            return FailureType.LOGIC_FAILURE
        if isinstance(exception, CandidateInvalidError):
            # 旧 validator 只提供 CandidateInvalidError；它仍明确说明问题来自
            # Candidate contract，而不是可重试基础设施。
            return FailureType.INTERFACE_ERROR
        if isinstance(exception, ConnectionError):
            return FailureType.NETWORK
        if isinstance(exception, TransientEvaluationError):
            return FailureType.TRANSIENT_INFRA
        return None

    @staticmethod
    def _from_process_evidence(return_code, stderr):
        lowered = stderr.lower()
        if any(
            marker in lowered
            for marker in (
                "out of memory",
                "oom-kill",
                "oom killed",
                "memoryerror",
                "cannot allocate memory",
            )
        ):
            return FailureType.ALGORITHM_OOM, "stderr 包含明确 OOM 证据"
        if return_code in {137, -9}:
            return (
                FailureType.ALGORITHM_OOM,
                "子进程退出码 {} 表明 SIGKILL/OOM 风险".format(return_code),
            )
        if "syntaxerror" in lowered:
            return FailureType.SYNTAX_ERROR, "stderr 包含 SyntaxError"
        if any(
            marker in lowered
            for marker in (
                "interface contract",
                "invalid output",
                "missing run_tuners",
                "unexpected signature",
            )
        ):
            return FailureType.INTERFACE_ERROR, "stderr 包含接口契约错误"
        if any(
            marker in lowered
            for marker in (
                "traceback (most recent call last)",
                "zerodivisionerror",
                "indexerror",
                "keyerror",
                "typeerror",
            )
        ):
            return FailureType.RUNTIME_ERROR, "stderr 包含 Candidate runtime traceback"
        if return_code not in (None, 0):
            return (
                FailureType.RUNTIME_ERROR,
                "Candidate 子进程以非零退出码 {} 结束".format(return_code),
            )
        return None

    @staticmethod
    def _decision(failure_type, reason, error_code, return_code, stderr):
        return FailureDecision(
            failure_type,
            _ACTIONS[failure_type],
            reason,
            error_code,
            return_code,
            stderr[:1000],
        )


_ERROR_CODES = {
    "NETWORK": FailureType.NETWORK,
    "NETWORK_ERROR": FailureType.NETWORK,
    "CONNECTION_RESET": FailureType.NETWORK,
    "WORKER_CRASH": FailureType.WORKER_CRASH,
    "LEASE_EXPIRED": FailureType.WORKER_CRASH,
    "TRANSIENT_INFRA": FailureType.TRANSIENT_INFRA,
    "TRANSIENT_BENCHMARK_ERROR": FailureType.TRANSIENT_INFRA,
    "INFRASTRUCTURE_TIMEOUT": FailureType.TRANSIENT_INFRA,
    "INFRA_TIMEOUT": FailureType.TRANSIENT_INFRA,
    "SYNTAX_ERROR": FailureType.SYNTAX_ERROR,
    "CANDIDATE_SYNTAX_ERROR": FailureType.SYNTAX_ERROR,
    "RUNTIME_ERROR": FailureType.RUNTIME_ERROR,
    "CANDIDATE_RUNTIME_ERROR": FailureType.RUNTIME_ERROR,
    "INTERFACE_ERROR": FailureType.INTERFACE_ERROR,
    "CANDIDATE_INVALID": FailureType.INTERFACE_ERROR,
    "INVALID_OUTPUT": FailureType.INTERFACE_ERROR,
    "ALGORITHM_TIMEOUT": FailureType.ALGORITHM_TIMEOUT,
    "EVALUATION_TIMEOUT": FailureType.ALGORITHM_TIMEOUT,
    "ALGORITHM_OOM": FailureType.ALGORITHM_OOM,
    "OOM": FailureType.ALGORITHM_OOM,
    "LOGIC_FAILURE": FailureType.LOGIC_FAILURE,
}


def classify_failure(
    exception: Optional[BaseException] = None,
    error_code: str = "",
    return_code: Optional[int] = None,
    stderr: str = "",
) -> FailureDecision:
    """无状态便捷入口。"""

    return FailureClassifier().classify(
        exception=exception,
        error_code=error_code,
        return_code=return_code,
        stderr=stderr,
    )

