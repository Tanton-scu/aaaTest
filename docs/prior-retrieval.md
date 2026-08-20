# Instance-Specific Prior Retrieval

## 与论文/源码一致的语义

该模块查询的是“与目标 CTP 实例 landscape 相似、且有历史实验证据的实例/优化器/算子”，不是自然语言文档检索。

```text
目标实例 samples
→ FDC/FBD/PLO/Skewness/Kurtosis/CL/MIE/NBC
→ 结构化历史实例的数值相似度 top-k
→ 可选 LLM metric-semantic refinement（仅能从 top-k 选 1-3 个）
→ rank1/rank2/rank3 strong optimizers
→ optimizer → module → operator description/code/provenance
→ InstanceSpecificPrior artifact
→ initial population prior seeds
```

## 数值规则

`NumericalLandscapeRetriever` 保持 `references/prievo/prior_util/dataset_distance.py`：

- FBD/PLO/MIE/NBC：`max - x`，表示 smaller-better；
- FDC/Skewness/Kurtosis：`1 - abs(x)/max_abs`，表示 close-to-zero；
- CL：原值，表示 larger-better；
- 各维基于历史 candidates min-max，target 先 clamp 到历史 raw range；
- 8 维等权 Euclidean distance；排除同名目标实例，取 top-5。

常量且 `max_abs=0` 时明确回退 0.5，修复 reference 的除零边界，但不改变正常数据排名。

## Landscape analyzer

- `PflaccoLandscapeAnalyzer`：可选 research adapter，按 reference 使用 `pflacco` 计算 8 个正式指标，并保留 mixed distance、PLO 与 CL 逻辑；安装 `.[research]` 后使用。
- `RecordedLandscapeAnalyzer`：Harness/离线复现消费已由 reference-compatible FLA 计算的 metric record，不声称自己重新计算指标。

这一区分避免在缺少 `pflacco` 时伪造新指标。

## 结构化 evidence

`CsvPriorRepository` 直接适配 `dataset_fl.csv`、`dataset_top3.csv`、`tuner_operator.csv`、`operator_detail.csv`、`operator_module.csv`、`prior_population.json`。每个 operator evidence 保存 source instance、optimizer、rank、module、operator ID/name/description/code。文件 digest 汇总为 evidence version。

当前只读验证：74 profiles；Dune 对应 7 个 strong optimizers、25 个 operator records，7 个均有历史 heuristic code。v1 不需要 Neo4j；CSV/typed objects 已满足查询。

## Semantic refinement 与降级

真实 refiner 只能从 numeric top-k 选 1–3 个名称，并需保存 reason/raw response/refiner。无 Key、timeout、schema 错误或越界选择时，确定性回退 numeric top-3，并在 artifact 标记 `fallback_used=true`。Harness 使用 Fake refiner，不冒充真实 LLM reasoning。

## Trace 与 artifact

- `LANDSCAPE_PROFILE` + `LANDSCAPE_ANALYZED`：完整 metrics、sample count、analyzer。
- `INSTANCE_SPECIFIC_PRIOR` + `PRIOR_RETRIEVED`：每个 numeric distance/raw/normalized metrics、semantic selection/reason/raw response、optimizer/operator empirical evidence、evidence version。

## 独立 Harness

```powershell
$env:PYTHONPATH='src'
python -m prievo_agent.cli.prior_demo
```

该 Harness 与后续 Literature RAG eval 完全分离。
