---
name: history_summary
version: 1
purpose: 仅当单个 Run 的 Agent 历史超过阈值时压缩早期记录
---

# History Summary

Summarize only the supplied earlier history while preserving:

- strategy and important operator changes;
- fitness trend and meaningful trajectory changes;
- important failures and attempted repairs;
- established design constraints;
- stable Candidate, Artifact, Event, and Evidence references.

Do not summarize the recent 2-3 lineage steps, invent missing facts, merge different Runs, or remove provenance. Return a low-variance factual summary suitable for a bounded prompt context.
