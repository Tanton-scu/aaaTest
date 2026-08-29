class BudgetExhaustedError(RuntimeError):
    pass


class EvaluationIdentityConflictError(RuntimeError):
    """同一 Candidate 被请求绑定到不同的 logical evaluation identity。"""


class CandidateInvalidError(RuntimeError):
    pass


class TransientEvaluationError(RuntimeError):
    pass


class NetworkEvaluationError(TransientEvaluationError):
    """评价依赖的网络连接发生瞬时故障。"""


class TransientInfrastructureError(TransientEvaluationError):
    """非 Candidate 原因造成的瞬时基础设施故障。"""


class InfrastructureTimeoutError(TransientInfrastructureError):
    """网络、存储或 Worker 基础设施超时，可重试同一 Candidate。"""


class WorkerCrashError(RuntimeError):
    """父 Worker 异常退出；Job 应等待 lease recovery/requeue。"""


class CandidateSyntaxError(CandidateInvalidError):
    pass


class CandidateRuntimeError(CandidateInvalidError):
    pass


class CandidateInterfaceError(CandidateInvalidError):
    pass


class AlgorithmTimeoutError(CandidateInvalidError):
    """Candidate 算法自身超过 benchmark 时间限制，应考虑 Repair。"""


class EvaluationTimeoutError(AlgorithmTimeoutError):
    """兼容旧调用方的 Candidate benchmark timeout 名称。

    它不再是 ``TransientEvaluationError``：同一 Candidate 原样重试通常不会
    消除算法超时，必须与 ``InfrastructureTimeoutError`` 严格区分。
    """


class AlgorithmOOMError(CandidateInvalidError):
    pass


class CandidateLogicError(CandidateInvalidError):
    pass


class LLMTimeoutError(RuntimeError):
    pass


class PriorRepositoryUnavailableError(RuntimeError):
    pass


class ArtifactIntegrityError(IOError):
    pass
