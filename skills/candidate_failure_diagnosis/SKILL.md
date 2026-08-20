---
name: candidate_failure_diagnosis
version: 1
purpose: 基于候选、父代、EvaluationJob 与输出事实诊断失败并形成有界修复建议
---

# Candidate Failure Diagnosis

## Workflow

1. 必须先通过 CandidateInspectionTool 读取持久化事实。
2. 区分代码校验、超时、运行异常、预算耗尽和结果缺失；不得猜测未记录的 stdout/stderr。
3. 修复建议必须保持 `run_tuners(file, budget, seed, maxlives)` 接口。
4. 不得绕过 EvaluationJob、budget、candidate validation 或 subprocess security。
5. 输出 diagnosis、likely cause、最小 repair steps 和下一代 prompt advice；不输出 Chain-of-Thought。

