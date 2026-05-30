# Validation checklist (before GitHub push)

Use this while `DRL_chemo_github/` lives inside `DRL_Chemotherapy/`.  
Goal: **new `main(...)` pipeline matches your existing MATLAB results** (same `rng(42)`, same agents, same metrics).

## Setup

```matlab
cd DRL_Chemotherapy
main('eval', struct('run_cohort', false))   % quick test
main('eval')                               % full evaluation
```

Trained agents: `../savedAgents/DQN/PaperStyle` and `../savedAgents/TD3/PaperStyle`.

Outputs: `DRL_chemo_github/results/eval_<timestamp>.mat`, plus `data/pmp_data.mat` and `data/rl_data.mat`.

## 1. PMP benchmark

| Check | Legacy | New pipeline |
|-------|--------|--------------|
| Run | `pmp_shooting.m` or `solve_pmp` at root | `main('pmp')` |
| Output | `optimal_control_results.mat` / figures | `../data/pmp_data.mat` |
| Metrics | ~87.7% tumor reduction (paper) | `pmp_data.metrics.tumor_reduction` |

Compare: final burden, total drug, total cost `J`, and control profile shape (bang–singular).

## 2. Nominal DQN / TD3

| Check | Legacy | New pipeline |
|-------|--------|--------------|
| Run | `main_simulate_dqn_td3.m` (mode 1, PaperStyle runs) | `main('eval', struct('run_cohort',false))` or `main('simulate','nominal')` |
| Agents | `savedAgents/*/20260329_*_PaperStyle` | same via `PaperStyle` link |

Compare per agent:

- Tumor reduction (%)
- Total cost `J`
- Total drug
- Dosage / burden curves (visual)

## 3. Cohort robustness (n = 100)

| Check | Legacy | New pipeline |
|-------|--------|--------------|
| Run | `main_simulate_dqn_td3.m` (mode 2, n=100) | `main('eval')` or `main('simulate','cohort',100)` |
| Perturbation | ±10% on R, Phi | `cfg.cohort_variation = 0.10` |

Compare mean ± std of tumor reduction and total cost across the cohort.  
Use the same parallel pool / `rng` if you need bit-for-bit parity (cohort uses `parfor` with per-patient random draws).

## 4. Paper figures

```matlab
main('eval')
main('figures')
```

Figures write to `DRL_chemo_github/results/figures/*.png`.  
Overlay with prior EPS/PNG from `main_simulate_dqn_td3` or `generate_figures` at root.

## 5. When results match

- Copy any final reference `.mat` snippets into `results/reference/` (optional).
- Remove or ignore nested `savedAgents/` symlinks before push; document that users train locally.
- Push **only** `DRL_chemo_github/` (standalone clone). Update `config.m` parent detection still works when the repo is cloned alone.

## Known differences (acceptable if documented)

- `solve_pmp.m` (fsolve) vs `pmp_shooting.m` (iterative shooting): small numerical differences possible.
- Figure format: pipeline exports PNG; legacy may use EPS.
- `main('ablation')` uses untrained nets unless you train per action count.
