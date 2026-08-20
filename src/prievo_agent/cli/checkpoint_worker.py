from __future__ import annotations

import argparse
import os
from pathlib import Path

from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.infrastructure.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.persistent_runtime import (
    PersistentEvolutionRuntime,
    SimulatedHardDeath,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="PriEvO-Agent checkpoint 子进程")
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--crash-after", type=int, default=-1)
    args = parser.parse_args()

    root = Path(args.root)
    store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
    try:
        try:
            run = store.get_run(args.run_id)
        except KeyError:
            task = OptimizationTask(
                id="task-{}".format(args.run_id),
                name="Checkpoint 恢复演示",
                objective="minimize",
                evaluation_budget=3,
                total_budget=81,
            )
            run = Run(id=args.run_id, task_id=task.id)
            store.add_task(task)
            store.add_run(run)
            store.append_event(run.id, EventType.RUN_CREATED.value, "运行已创建")

        runtime = PersistentEvolutionRuntime(
            store,
            FakeEvaluator(),
            PriEvoEvolutionCore(FakeLLM(), population_size=3),
            total_generations=2,
        )
        runtime.execute(
            run.id,
            crash_after_generation=(
                args.crash_after if args.crash_after >= 0 else None
            ),
        )
        return 0
    except SimulatedHardDeath:
        # 不执行 finally/close，模拟进程在已提交 checkpoint 后突然死亡。
        os._exit(75)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
