from dataclasses import asdict, dataclass
from typing import Any, Dict, List

@dataclass(frozen=True)
class SimilarityDecision:
    """SimilarityAgent 的结构化、可审计输出。

    ``selected_instance_ids`` 只表示对 numeric Top-5 的语义筛选结果；它不携带、
    也不生成 optimizer/operator Prior。``selection_prompt`` 与 Skill 身份被保留，
    便于调用方将这次决策原样写入 prompt/decision Artifact。
    """

    selected_instance_ids: List[str]
    reason_summary: str
    metric_evidence: Dict[str, Dict[str, Any]]
    skill_name: str
    skill_digest: str
    selection_prompt: str
    context_metadata: Dict[str, Any]

    def to_dict(self):
        return asdict(self)
