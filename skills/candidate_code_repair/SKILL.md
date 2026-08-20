---
name: candidate_code_repair
version: 1
purpose: 在 FailureClassifier、次数和预算守卫通过后对失败 Candidate 做最小代码修复
---

# Candidate Code Repair

1. Use only inspected Candidate code, persisted failure facts, parent/repair lineage, and relevant failures from the same Run.
2. Apply the accepted RepairDecision root cause and suggested fix; do not broaden the change without evidence.
3. Preserve `run_tuners(file, budget, seed, maxlives)` and the injected `evaluate(...)` contract.
4. Do not hide exceptions, return fake objective values, bypass budget, or access unapproved files/network/process APIs.
5. Return a new RepairedCandidateDraft. Never overwrite the failed Candidate.
6. The caller will record `repair_parent_id`, `repair_attempt`, and `creation_type=REPAIR`, then submit a new EvaluationJob.
