---
name: e2
version: 1
purpose: PriEvO e2 / Recombine，继承并调整父代有效 operators 产生新 heuristic
---

# Recombine (e2)

Create one new heuristic by inheriting, referencing, and adapting effective core operators from the supplied parents.

1. Compare parent objectives, trajectories, budget use, stability, convergence, and operator composition.
2. Preserve mechanisms supported by benchmark evidence; combine them into a coherent design instead of concatenating code blindly.
3. The new logic must be recognizably inspired by effective parent operators and may use the supplied Prior operator references.
4. Preserve the exact `run_tuners` and injected `evaluate` contract.
5. Return only the CandidateDraft schema, without fabricated fitness or trajectory.

This is the PriEvO `e2` exploitation strategy and normally receives two selected parents.
