# API、SSE 与运行面板

FastAPI 只负责协议转换，所有用例进入 `RunApplicationFacade`，演化循环继续位于 Runtime。`POST /api/runs` 持久化 Run 后把工作提交给后台执行器，并以 `202 Accepted` 立即返回；因此 HTTP 请求生命周期与演化生命周期解耦。

Run、事件和 artifact 元数据以 SQLite/文件系统为事实源。客户端断开后，可以使用相同 `run_id` 查询当前状态、完整事件历史与 artifact；SSE 的 `Last-Event-ID` 或 `after` 游标可从最后确认的事件继续读取。SSE 只发布已持久化的 Run 事件，不宣称 token streaming。

## 启动

```powershell
$env:PYTHONPATH = "src"
python -m prievo_agent.cli.api_server --root .prievo-runtime
```

浏览器访问 `http://127.0.0.1:8000/`。面板展示 Run 状态、代数、预算、候选、lineage、轨迹、事件和 artifact 元数据。

## 失败语义

- Run 不存在：HTTP 404；
- 当前状态不允许 cancel/resume：HTTP 409；
- 应用正常关闭时等待已接受的本进程任务收束；进程异常退出后的恢复仍由 generation checkpoint 负责。
