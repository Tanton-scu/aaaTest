---
name: semantic_similarity_selection
version: 1
purpose: 从 deterministic FLA numeric Top-5 中按八项 metric 语义选择 1 至 3 个相似历史实例
---

# Semantic Similarity Selection

You are selecting historical instances whose fitness landscapes are closest to a target instance.

1. Use only the supplied numeric Top-5 candidates. Never introduce another dataset.
2. Compare all eight metrics: FDC, FBD, PLO, Skewness, Kurtosis, CL, MIE, and NBC.
3. Apply the supplied metric definitions and value significance. Explain which values are close, which differ, and why each difference matters or has limited impact.
4. Select at least one and at most three instance IDs, ordered from most to least similar.
5. Do not create optimizer code or Prior. The deterministic PriorExtractor will load evidence for the selected IDs.
6. Return the required structured schema only: `selected_instance_ids`, `reason_summary`, and per-instance `metric_evidence`.

本 Skill 封装 `references/prievo/prior_util/dataset_llm_recommend.py` 的真实语义筛选约束；numeric distance ranking 是输入证据，不是可改写事实。
