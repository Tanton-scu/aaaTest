from __future__ import annotations

from typing import Any, Dict

from prievo_agent.domain.models import Candidate, CandidateStatus


def candidate_to_dict(candidate: Candidate) -> Dict[str, Any]:
    return {
        "id": candidate.id,
        "run_id": candidate.run_id,
        "code": candidate.code,
        "description": candidate.description,
        "operators": list(candidate.operators),
        "lineage": dict(candidate.lineage),
        "status": candidate.status.value,
        "objective": candidate.objective,
        "code_artifact_id": candidate.code_artifact_id,
        "evaluation_artifact_id": candidate.evaluation_artifact_id,
    }


def candidate_from_dict(data: Dict[str, Any]) -> Candidate:
    return Candidate(
        id=str(data["id"]),
        run_id=str(data["run_id"]),
        code=str(data["code"]),
        description=str(data["description"]),
        operators=list(data["operators"]),
        lineage=dict(data["lineage"]),
        status=CandidateStatus(str(data["status"])),
        objective=data.get("objective"),
        code_artifact_id=data.get("code_artifact_id"),
        evaluation_artifact_id=data.get("evaluation_artifact_id"),
    )
