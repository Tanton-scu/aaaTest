# PriEvO Core 迁移说明

## 已迁移且保持忠实的语义

- `core/schedule.py`：按论文/README 的 generation schedule，总代数前半按 `i1/e1/e2/m1`，后半按 `e1/e2/m1/m2`。
- `core/selection.py::select_parents`：保持 `prob_rank.parent_selection` 的 `1/(rank+1+population_size)` 权重与有放回抽样。
- `select_population_early`：保持 `pop_greedy.population_management` 的 fitness tier、每层至少一个和 operator diversity 排序。
- `select_population_late`：保持 objective 优先、operator diversity tie-break。
- `PriEvoEvolutionCore.evolve_operator`：保留 i1 Synthesize、e1 Imitate、e2 Recombine、m1 Revise、m2 Fine-tune 的 parent 数量和 lineage 语义；每个 operator 严格生成 `pop_size` 个 offspring。
- `PersistentEvolutionRuntime/ResearchRunner`：四个 operator 分别生成并评价 `pop_size` 个，汇总成 `4n` 后与原 `n` 合并，再按论文的 early/late 规则从 `5n` 一次筛回 `n`。

## 边界提取

- `CandidateEvaluator`：core 不执行 candidate code；FakeEvaluator 当前只验证契约。
- `LLMPort`：生成请求包含 operator、parents 和 generation，FakeLLM 返回结构化 `GeneratedCandidate` 并保留英文 heuristic code。
- `RuntimeStore.put_artifact`：摘要输出不由 core 写文件。
- prior access 尚未接入实现；阶段 09 将以单独 `PriorRepository` 提供 `InstanceSpecificPrior`，不会混入 LLM/RAG。

## 有意改变

- reference 当前可执行循环在每个 operator batch 后调用一次 population management，但 README 的 per-paper workflow 明确写为每代生成 `4n` 后执行 dual-criterion selection。这里按论文描述和产品需求采用整代 `5n → n`，并通过事件显式记录。
- reference individual 使用松散 dict 和 `operators` 文本；目标项目使用 typed `Candidate`、`operators: list[str]` 和显式 lineage，并提供稳定序列化。
- reference 使用全局 `random.seed(2024)`；目标 core 注入私有 `random.Random(2024)`，避免跨 Run 污染，同时保持同权重公式。
- 原始 LLM prompt 大段内容尚未复制；阶段 11 在结构化 skill/tool 边界成熟后迁移。当前 FakeLLM 只验证 operator dispatch。
- 原始 `joblib/SIGALRM/exec/temp CSV` 未迁入 core；评价执行属于 infrastructure/runtime，阶段 08/14 实现。
- 原始最终 `best_stability` 的 LLM 二次选择未迁入；当前最小 objective 确定性选择。该行为复现性不足，后续若保留只能作为可选建议并留 trace。

## Characterization 证据

`tests/test_prievo_core.py` 直接只读加载：

- `references/prievo/prievo/methods/selection/prob_rank.py`
- `references/prievo/prievo/methods/management/pop_greedy.py`

在固定 fixture/seed 上比较 selected candidate IDs；另测 schedule、serialization、LLM call/lineage、research runner。随机 LLM 文本和完整 benchmark 不追求逐值等价，锁定的是 operator/selection/boundary invariants。

## 当前尚非“完整 PriEvO”

FLA、structured prior retrieval、真实 prompt、candidate code process execution 和 CTP dataset evaluation 尚待后续阶段。当前可以可信表述为“提取了 PriEvO-derived evolution core 的调度与选择主干”，不能声称已复现论文完整 74-instance 实验。
