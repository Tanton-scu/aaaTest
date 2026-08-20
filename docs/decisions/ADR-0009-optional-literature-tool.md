# ADR-0009：Literature Retrieval 是有预算的可选 Tool

- 状态：已接受
- 日期：2026-08-07

## 决策

Literature retrieval 只服务 generation/reflection 知识缺口，以 structured query 和 provenance-rich evidence 交互。v1 使用 curated local corpus + BM25/section chunks/primary preference；每 Run/每次结果均有限额并持久 trace。

它与 Instance Prior 使用独立 domain/port/artifact/event/harness。当前不引入 embedding DB 或 MCP。

## 后果

项目具备真实而有限的文献辅助用例，能够解释 evidence 如何影响后续 decision，同时在无文献/预算用尽时不阻塞 PriEvO 主链。
