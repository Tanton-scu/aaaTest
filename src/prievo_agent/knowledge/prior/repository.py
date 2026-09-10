from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

from prievo_agent.knowledge.prior.models import (
    LANDSCAPE_METRICS,
    LandscapeProfile,
    OperatorEvidence,
    OptimizerEvidence,
)


def _split(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split("_") if item.strip()]


class StructuredPriorRepository:
    def __init__(
        self,
        profiles: Sequence[LandscapeProfile],
        optimizers_by_instance: Dict[str, List[OptimizerEvidence]],
        version: str = "fixture-v1",
    ) -> None:
        self._profiles = list(profiles)
        self._optimizers = {
            key.strip().lower(): list(value)
            for key, value in optimizers_by_instance.items()
        }
        self._version = version

    def landscape_profiles(self) -> Sequence[LandscapeProfile]:
        return list(self._profiles)

    def strong_optimizers(self, instance_name: str) -> List[OptimizerEvidence]:
        return list(self._optimizers.get(instance_name.strip().lower(), []))

    def evidence_version(self) -> str:
        return self._version


class CsvPriorRepository:
    """读取 PriEvO `pre_knowledge` 的结构化 CSV/JSON schema，不依赖图数据库。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._profiles = self._load_profiles()
        self._optimizers = self._load_evidence()
        digest = hashlib.sha256()
        for name in [
            "dataset_fl.csv",
            "dataset_top3.csv",
            "tuner_operator.csv",
            "operator_detail.csv",
            "operator_module.csv",
            "prior_population.json",
        ]:
            path = self.root / name
            if path.exists():
                digest.update(path.read_bytes())
        self._version = digest.hexdigest()[:16]

    def landscape_profiles(self) -> Sequence[LandscapeProfile]:
        return list(self._profiles)

    def strong_optimizers(self, instance_name: str) -> List[OptimizerEvidence]:
        return list(self._optimizers.get(instance_name.strip().lower(), []))

    def evidence_version(self) -> str:
        return self._version

    def _rows(self, name: str) -> List[dict]:
        with (self.root / name).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _load_profiles(self) -> List[LandscapeProfile]:
        profiles = []
        for row in self._rows("dataset_fl.csv"):
            profiles.append(
                LandscapeProfile(
                    row["dataset"].strip(),
                    {metric: float(row[metric]) for metric in LANDSCAPE_METRICS},
                    100,
                    "PriEvO reference FLA",
                )
            )
        return profiles

    def _load_evidence(self) -> Dict[str, List[OptimizerEvidence]]:
        tuner_ops = {
            row["tuner"].strip().lower(): _split(row["operator"])
            for row in self._rows("tuner_operator.csv")
        }
        module_prefixes = {
            row["operator"].strip(): row["module"].strip()
            for row in self._rows("operator_module.csv")
        }
        details = {row["operator"].strip(): row for row in self._rows("operator_detail.csv")}
        prior_path = self.root / "prior_population.json"
        prior_records = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else []
        prior = {str(row.get("name", "")).lower(): row for row in prior_records}
        result: Dict[str, List[OptimizerEvidence]] = {}
        rank_columns = [
            ("rank1", "rank1_tuners"),
            ("rank2", "rank2_tuners"),
            ("rank3", "rank3_tuners"),
        ]
        for row in self._rows("dataset_top3.csv"):
            instance = row["dataset"].strip()
            evidence = []
            seen = set()
            for rank, column in rank_columns:
                for tuner in _split(row[column]):
                    key = tuner.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    operators = []
                    for operator_id in tuner_ops.get(key, []):
                        detail = details.get(operator_id, {})
                        prefix = operator_id[:2] if operator_id[:2] in module_prefixes else operator_id[:1]
                        operators.append(
                            OperatorEvidence(
                                operator_id,
                                detail.get("name", "op_{}".format(operator_id)),
                                module_prefixes.get(prefix, "Unknown"),
                                detail.get("description", ""),
                                detail.get("code_block", ""),
                                instance,
                                tuner,
                                rank,
                            )
                        )
                    record = prior.get(key, {})
                    evidence.append(
                        OptimizerEvidence(
                            tuner,
                            rank,
                            instance,
                            str(record.get("description", "")),
                            str(record.get("code", "")),
                            operators,
                        )
                    )
            result[instance.lower()] = evidence
        return result
