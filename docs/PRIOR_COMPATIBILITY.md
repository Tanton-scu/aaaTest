# Original Prior 执行兼容性

`resources/prior_knowledge/prior_population.json` 的 31 条历史 heuristic 始终作为
不可变的 PriEvO Prompt/Evidence 保存。它们并不因此自动获得产品 Candidate worker
的执行权限。

`scripts/audit_prior_compatibility.py` 使用与产品 worker 相同的 AST、入口签名和
import 安全契约进行无 `import`、无 `exec` 的静态审计。当前矩阵为：

- 31 条总记录；
- 21 条 `SUPPORTED`：可进入受控 evaluator 尝试执行，不保证对任意 Dataset 成功；
- 10 条 `UNSUPPORTED`：继续进入 Prior/Generation Prompt 和 Evidence，但不会物化为
  initial population seed。

不兼容条目是 HEBO、PromiseTune、SWAY、CMAES、ACO、ResTune、ROBOTune、
Hyperband、BOHB、DEHB。原因包括产品未承诺的第三方依赖、`global/nonlocal`
状态和不符合严格 `run_tuners(file,budget,seed,maxlives)` 的入口签名。项目不通过
盲装 torch/pymoo/hpbandster 等重依赖来绕开边界。

每次 Run 会额外保存 `PRIOR_EXECUTION_COMPATIBILITY` Artifact，并写
`PRIOR_EXECUTION_COMPATIBILITY_AUDITED` Event；全库机器报告位于
`reports/prior_compatibility.json`。

生成命令：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python scripts/audit_prior_compatibility.py
```
