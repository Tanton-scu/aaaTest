from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from prievo_agent.domain.models import RunStatus
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore

from .harness import LifecycleHarness


@dataclass(frozen=True)
class RecoveryComparison:
    uninterrupted: Dict[str, object]
    recovered: Dict[str, object]
    recovery_event_present: bool
    duplicate_evaluations: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "uninterrupted": self.uninterrupted,
            "recovered": self.recovered,
            "recovery_event_present": self.recovery_event_present,
            "duplicate_evaluations": self.duplicate_evaluations,
        }


class CheckpointRecoveryHarness:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()

    def run(self, output_root: Path) -> RecoveryComparison:
        output_root = Path(output_root).resolve()
        uninterrupted_root = output_root / "uninterrupted"
        recovered_root = output_root / "interrupted"
        uninterrupted_root.mkdir(parents=True, exist_ok=True)
        recovered_root.mkdir(parents=True, exist_ok=True)

        self._worker(uninterrupted_root, "run-uninterrupted", None, {0})
        self._worker(recovered_root, "run-recovered", 1, {75})
        self._worker(recovered_root, "run-recovered", None, {0})

        uninterrupted = self._summary(uninterrupted_root, "run-uninterrupted")
        recovered = self._summary(recovered_root, "run-recovered")
        comparable_keys = {
            "status",
            "objective",
            "final_digest",
            "consumed_budget",
            "evaluation_result_count",
            "generation",
        }
        for key in comparable_keys:
            if uninterrupted[key] != recovered[key]:
                raise AssertionError(
                    "恢复运行与不中断运行不一致：{} {} != {}".format(
                        key, uninterrupted[key], recovered[key]
                    )
                )
        expected_evaluations = 27
        duplicates = int(recovered["evaluation_result_count"]) - expected_evaluations
        if duplicates != 0:
            raise AssertionError("恢复后出现重复逻辑评价")
        return RecoveryComparison(
            uninterrupted=uninterrupted,
            recovered=recovered,
            recovery_event_present=bool(recovered["recovery_event_present"]),
            duplicate_evaluations=duplicates,
        )

    def _worker(
        self,
        root: Path,
        run_id: str,
        crash_after: int,
        expected_codes: set,
    ) -> None:
        command = [
            sys.executable,
            "-m",
            "prievo_agent.cli.checkpoint_worker",
            "--root",
            str(root),
            "--run-id",
            run_id,
        ]
        if crash_after is not None:
            command.extend(["--crash-after", str(crash_after)])
        env = os.environ.copy()
        src = str(self.project_root / "src")
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            command,
            cwd=str(self.project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode not in expected_codes:
            raise AssertionError(
                "checkpoint 子进程退出码异常：{}\nstdout={}\nstderr={}".format(
                    result.returncode, result.stdout, result.stderr
                )
            )

    def _summary(self, root: Path, run_id: str) -> Dict[str, object]:
        store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
        try:
            run = store.get_run(run_id)
            if run.status != RunStatus.COMPLETED:
                raise AssertionError("恢复 Harness 的 Run 未完成")
            inspection = LifecycleHarness(store).inspect_completed(run_id)
            best = store.candidate_by_id(run.best_candidate_id)
            result = store.result_for_candidate(best.id)
            final = next(
                item
                for item in store.artifacts_for_run(run_id)
                if item.kind == "FINAL_HEURISTIC"
            )
            events = list(store.events_for_run(run_id))
            return {
                "status": run.status.value,
                "objective": result.objective,
                "final_digest": final.digest,
                "consumed_budget": run.consumed_evaluations,
                "evaluation_result_count": inspection.evaluation_result_count,
                "generation": run.generation,
                "checkpoint_count": sum(
                    1 for event in events if event.event_type == "CHECKPOINT_SAVED"
                ),
                "recovery_event_present": any(
                    event.event_type == "RUN_RECOVERED" for event in events
                ),
            }
        finally:
            store.close()
