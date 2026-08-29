from __future__ import annotations

from typing import Dict

from prievo_agent.domain.models import CandidateStatus


class RunMetricsQuery:
    """从事实源派生 V1 指标，不引入第二套可变计数器。"""

    def __init__(self, store) -> None:
        self.store = store

    def snapshot(self, run_id: str) -> Dict[str, object]:
        run = self.store.get_run(run_id)
        events = list(self.store.events_for_run(run_id))
        candidates = list(self.store.candidates_for_run(run_id))
        jobs = list(self.store.evaluation_jobs_for_run(run_id))
        evaluated = [
            event.payload["objective"]
            for event in events
            if event.event_type == "CANDIDATE_EVALUATED" and "objective" in event.payload
        ]
        generation_started = {
            event.payload.get("generation"): event.occurred_at
            for event in events
            if event.event_type == "GENERATION_STARTED"
        }
        generation_durations = []
        for event in events:
            if event.event_type != "GENERATION_COMPLETED":
                continue
            generation = event.payload.get("generation")
            started = generation_started.get(generation)
            if started is not None:
                generation_durations.append(
                    {
                        "generation": generation,
                        "duration_seconds": max(
                            0.0, (event.occurred_at - started).total_seconds()
                        ),
                    }
                )
        invalid_count = sum(
            item.status == CandidateStatus.INVALID for item in candidates
        )
        terminal = run.status.value in {"COMPLETED", "FAILED", "CANCELLED"}
        return {
            "run_id": run.id,
            "run_duration_seconds": (
                max(0.0, (run.updated_at - run.created_at).total_seconds())
                if terminal
                else None
            ),
            "generation_durations": generation_durations,
            "candidate_count": len(candidates),
            "candidate_invalid_rate": (
                invalid_count / len(candidates) if candidates else 0.0
            ),
            "evaluation_attempts": sum(item.attempts for item in jobs),
            "evaluation_retries": sum(max(0, item.attempts - 1) for item in jobs),
            "literature_tool_calls": sum(
                event.event_type == "LITERATURE_SEARCHED" for event in events
            ),
            "budget_consumed": run.consumed_evaluations,
            "best_fitness": min(evaluated) if evaluated else None,
            "fitness_improvement": (
                evaluated[0] - min(evaluated) if evaluated else None
            ),
        }
