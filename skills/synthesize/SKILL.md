---
name: synthesize
version: 1
purpose: PriEvO i1 策略，从 instance-specific operator prior 合成新的 HPO heuristic
---

# Synthesize (i1)

Create one new parameter-tuning heuristic from scratch.

1. Use the supplied PriEvO task and benchmark contract without changing `run_tuners(file, budget, seed, maxlives)`.
2. Use the supplied instance-specific reference operator components as inspiration; components may be combined, adapted, or extended.
3. Generate configurations only from `file.independent_set`. Treat every value as a categorical enumeration unless an explicit encoder is used.
4. Do not implement or bypass the injected `evaluate(...)` function, budget accounting, duplicate handling, or termination contract.
5. Return only `code`, a description of at most two sentences, `operators`, and an optional generation note. Do not invent fitness or trajectory.

This is the PriEvO `i1` strategy. It has no parent.
