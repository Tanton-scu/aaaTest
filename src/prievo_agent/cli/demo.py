from __future__ import annotations

import json
import tempfile
from pathlib import Path

from prievo_agent.application.run_facade import RunApplicationFacade
from prievo_agent.infrastructure.local_runtime import LocalRuntimeComposition
from prievo_agent.runtime.harness import LifecycleHarness


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="prievo-demo-") as directory:
        composition = LocalRuntimeComposition(Path(directory))
        facade = RunApplicationFacade(
            composition.open_store, composition.execute, auto_start=True,
            dataset_registry=composition.dataset_registry,
        )
        created = facade.create_run(
            "xgboost-Covtype", generations=2, population_size=3,
            candidate_budget=3, random_seed=7,
        )
        facade.shutdown()
        detail = facade.get_run(created["run_id"])
        store = composition.open_store()
        try:
            inspection = LifecycleHarness(store).inspect_completed(
                created["run_id"]
            ).to_dict()
        finally:
            store.close()
        report = {
            "message": "PriEvO-Agent 确定性持久运行完成",
            "run": detail,
            "events": facade.events(created["run_id"]),
            "artifacts": facade.artifacts(created["run_id"]),
            "metrics": facade.metrics(created["run_id"]),
            "lifecycle_inspection": inspection,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
