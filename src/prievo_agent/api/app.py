from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import suppress
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, Response

from prievo_agent import __version__
from prievo_agent.application.orchestration.run_facade import (
    RunApplicationFacade,
    RunConflict,
    RunNotFound,
)
from prievo_agent.infrastructure.local_runtime import LocalRuntimeComposition

from .schemas import CreateRunRequest


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
TERMINAL_EVENTS = {
    "COMPLETED": "RUN_COMPLETED",
    "FAILED": "RUN_FAILED",
    "CANCELLED": "RUN_CANCELLED",
}
logger = logging.getLogger(__name__)


def create_app(
    runtime_root: Path,
    auto_start: bool = True,
    start_delay_seconds: float = 0.0,
) -> FastAPI:
    composition = LocalRuntimeComposition(runtime_root)
    facade = RunApplicationFacade(
        composition.open_store,
        composition.execute,
        auto_start,
        start_delay_seconds,
        composition.dataset_registry,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if auto_start:
            await asyncio.to_thread(facade.recover_startup)
        sweep_seconds = max(
            1.0, float(os.getenv("COORDINATOR_SWEEP_SECONDS", "30"))
        )
        stop_sweep = asyncio.Event()

        async def periodic_reconcile():
            while not stop_sweep.is_set():
                try:
                    await asyncio.wait_for(stop_sweep.wait(), timeout=sweep_seconds)
                    continue
                except asyncio.TimeoutError:
                    pass
                try:
                    await asyncio.to_thread(facade.reconcile_active_agent_tasks)
                except Exception:
                    # 周期兜底失败不终止 API；下一周期或 startup recovery 仍会重试。
                    logger.exception("周期 AgentTask reconciliation sweep 失败")

        sweep_task = asyncio.create_task(periodic_reconcile())
        try:
            yield
        finally:
            stop_sweep.set()
            sweep_task.cancel()
            with suppress(asyncio.CancelledError):
                await sweep_task
            facade.shutdown()

    app = FastAPI(title="PriEvO-Agent", version=__version__, lifespan=lifespan)
    app.state.facade = facade

    @app.get("/api/health")
    def health():
        snapshot = composition.health_snapshot()
        snapshot["message"] = (
            "PriEvO-Agent 服务可用"
            if snapshot["status"] == "UP"
            else "PriEvO-Agent 依赖状态需要关注"
        )
        return snapshot

    @app.get("/api/health/live")
    def liveness():
        return {"status": "UP", "message": "API 进程可响应"}

    @app.get("/api/health/ready")
    def readiness(response: Response):
        snapshot = composition.health_snapshot()
        ready = snapshot["readiness"]
        if ready["status"] == "DOWN":
            response.status_code = 503
        return ready

    @app.get("/api/datasets")
    def datasets():
        return [item.to_dict() for item in composition.dataset_registry.list_datasets()]

    @app.post("/api/runs", status_code=202)
    def create_run(body: CreateRunRequest):
        try:
            return facade.create_run(
                body.dataset_id, generations=body.generations,
                population_size=body.population_size,
                candidate_budget=body.candidate_budget,
                random_seed=body.random_seed,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/runs")
    def list_runs():
        return {"items": facade.list_runs()}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return _translate(lambda: facade.get_run(run_id))

    @app.get("/api/runs/{run_id}/events")
    def events(run_id: str, after: int = Query(default=0, ge=0)):
        return _translate(lambda: {"items": facade.events(run_id, after)})

    @app.get("/api/runs/{run_id}/artifacts")
    def artifacts(run_id: str):
        return _translate(lambda: {"items": facade.artifacts(run_id)})

    @app.get("/api/runs/{run_id}/artifacts/{artifact_id}/content")
    def artifact_content(run_id: str, artifact_id: str):
        content = _translate(lambda: facade.artifact_content(run_id, artifact_id))
        return Response(content, media_type="application/octet-stream")

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str):
        return _translate(lambda: facade.metrics(run_id))

    @app.get("/api/runs/{run_id}/agent-trace")
    def agent_trace(run_id: str):
        return _translate(lambda: facade.agent_trace(run_id))

    @app.get("/api/runs/{run_id}/trace")
    def trace(run_id: str):
        """规范 Trace 路径；旧 agent-trace 路径继续兼容。"""
        return _translate(lambda: facade.agent_trace(run_id))

    @app.get("/api/runs/{run_id}/candidates")
    def candidates(run_id: str):
        return _translate(lambda: facade.candidates(run_id))

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        return _translate(lambda: facade.cancel(run_id))

    @app.post("/api/runs/{run_id}/pause", status_code=202)
    def pause(run_id: str):
        return _translate(lambda: facade.pause(run_id))

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    def resume(run_id: str):
        return _translate(lambda: facade.resume(run_id))

    @app.get("/api/runs/{run_id}/events/stream")
    def stream_events(
        request: Request,
        run_id: str,
        after: int = Query(default=0, ge=0),
        last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
    ):
        cursor = max(after, int(last_event_id or 0))
        _translate(lambda: facade.get_run(run_id))

        async def generate():
            nonlocal cursor
            idle_rounds = 0
            terminal_event_seen = False
            while True:
                if await request.is_disconnected():
                    return
                items = facade.events(run_id, cursor)
                if items:
                    idle_rounds = 0
                    for item in items:
                        cursor = item["sequence"]
                        terminal_event_seen = terminal_event_seen or (
                            item["event_type"] in TERMINAL_EVENTS.values()
                        )
                        yield "id: {}\nevent: {}\ndata: {}\n\n".format(
                            cursor,
                            item["event_type"],
                            json.dumps(item, ensure_ascii=False),
                        )
                else:
                    idle_rounds += 1
                status = facade.get_run(run_id)["status"]
                if status in TERMINAL and not facade.events(run_id, cursor):
                    # Run 状态与 terminal Event 分属相邻事务。只有确认相应
                    # durable Event 已经写入且发送后才关闭，避免在提交窗口漏尾。
                    if not terminal_event_seen:
                        terminal_event_seen = any(
                            item["event_type"] == TERMINAL_EVENTS[status]
                            for item in facade.events(run_id, 0)
                            if item["sequence"] <= cursor
                        )
                    if terminal_event_seen:
                        return
                if idle_rounds % 10 == 0:
                    yield ": keep-alive\n\n"
                if composition.mode == "full":
                    await asyncio.to_thread(composition.event_bus.wait,run_id,0.5)
                else:
                    await asyncio.sleep(0.02)

        return StreamingResponse(generate(), media_type="text/event-stream")

    dashboard = Path(__file__).with_name("dashboard.html")

    @app.get("/")
    def dashboard_page():
        return FileResponse(dashboard)

    return app


def _translate(operation):
    try:
        return operation()
    except RunNotFound as exc:
        raise HTTPException(404, "Run 不存在") from exc
    except RunConflict as exc:
        raise HTTPException(409, str(exc)) from exc
