# ADR-0013：候选执行采用验证后受限子进程

## 决策

LLM 原始输出先经过严格 schema 和有限 repair；候选代码再经过 AST/interface allowlist，最后才进入专用临时目录中的 isolated Python 子进程。超时与 invalid candidate 使用不同错误类型。

## 后果

常见错误与恶意式 fixture 可安全拒绝或限时终止，但普通子进程不被称为安全沙箱。未来真实 benchmark adapter 必须显式组合此边界或更强容器隔离，不能直接 `exec` candidate 文本。
