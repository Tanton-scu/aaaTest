---
name: prior_explanation
version: 1
purpose: 在不覆盖 original PriEvO Prior 的前提下解释晦涩的 operator 或 tuner mechanism
---

# Prior Explanation

1. Restate the bounded KnowledgeGap and cite the exact original Prior reference.
2. Explain only mechanisms supported by the supplied Prior slice and LiteratureEvidence.
3. Separate source facts, agent annotation, and hypotheses that still require benchmark validation.
4. Preserve provenance: original prior ref, paper ref, chunk ref, retrieval scores, and source metadata.
5. Never rewrite, delete, or replace the original PriEvO Prior.
6. Return a concise `PriorExplanation` plus evidence references; do not generate Candidate code unless a later GenerationTask requests it.
