# Candidate 安全边界与威胁模型

## 保护目标

V1 防止格式错误的 LLM 输出直接变成可执行代码，并对常见危险操作、语法/接口错误、无限循环和失控标准输出提供确定性拒绝或边界。

## 已实现

- Candidate 结构化输出必须严格包含 `code`、`description`、`operators`；保留原始响应，repair 最多 0..3 次，默认 1 次；
- Python AST 必须只定义 `run_tuners(file, budget, seed, maxlives)`，限制 AST 大小，拒绝 import/global、危险 builtins 和 dunder 访问；
- 验证通过后才可进入独立 Python `-I` 子进程；使用专用临时目录、精简环境变量、受控 JSON 输入/输出、丢弃 stdout/stderr，并设置硬超时；
- schema/AST/契约/执行错误分类为 `CandidateInvalidError`，超时分类为 `EvaluationTimeoutError`，供评价队列采用不同重试策略。

## 明确限制

这不是强安全沙箱。AST allowlist 和普通子进程不能抵御熟练攻击者的所有 Python object-graph 绕过，也没有操作系统级网络、文件系统、CPU 或内存隔离。Windows 本地路径目前仅有 wall-clock timeout，没有可靠的 per-process memory limit。因此该 runner 只适合可信开发/研究环境中的风险降低；不应执行任意互联网用户提交的敌意代码。

生产多租户场景必须增加独立低权限身份、容器/VM、只读根文件系统、网络 deny-by-default、CPU/内存/pid 配额与宿主级监控。达到这些条件之前，产品和文档不得声称“secure sandbox”。
