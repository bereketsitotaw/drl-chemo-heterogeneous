# Reproducibility — DRL heterogeneous chemotherapy

Maps the manuscript *Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning* to this repository.

## Four reproducibility categories

| Category | Status | Notes |
|----------|--------|-------|
| **1. Code / scientific config** | Reproducible from git | `scripts/config.m` pins $(\alpha,\beta,\gamma)=(500,400,4000)$, $n=21$, $T=10$, $U_{\max}=3$, DQN actions $=7$, cohort $100\times\pm10\%$. Locked by `configs/assert_publication_defaults.m` + CI greps. |
| **2. Environment** | Documented | MATLAB R2020a+ (paper: R2024b); Reinforcement Learning Toolbox; Optimization Toolbox (`fsolve`); Parallel Computing Toolbox recommended. See `requirements.txt`. |
| **3. Trained agents** | Not in git (by design) | Checkpoints gitignored. Reproduce with `main('train')` (~2–5 h) or request author `PaperStyle` weights. Local nested workspace may already provide `../savedAgents/*/PaperStyle`. |
| **4. Numerical results / figures** | Partially pinned | Table III/IV numbers in `configs/reference_metrics.m`. PMP soft-checked by `main('smoke')`. Full RL nominal/cohort needs agents; figures from `main('eval')` + `main('figures')`. |

## Model (Section II)

| Paper | Code |
|-------|------|
| Eq. (1) trait ODE, \(n=21\) | `tumor_dynamics.m`, `config.m` (`R_vec`, `Phi_vec`, `M_vec`, `N0`) |
| Eq. (2) cost \(J\) | `config.m` (`alpha`, `beta`, `gamma`), `compute_metrics.m` |
| Eq. (3)–(5) PMP | `solve_pmp.m` |
| \(u \in [0,3]\), \(T=10\) yr, \(\Delta t=0.1\) | `config.m`, `build_env3.m` |

## RL (Section III)

| Paper | Code |
|-------|------|
| MDP reward \(-(\beta^T N + \gamma u)\,\Delta t\), terminal \(-\alpha^T N(T)\) | `build_env3.m` (`stepFunction`) |
| DQN, 7 actions (Table I) | `build_agent.m`, `cfg.train.dqn_actions = 7` |
| TD3 | `build_agent.m` |
| Table II hyperparameters | `build_agent.m`, `run_training.m` |

## Experiments (Section IV)

| Experiment | Command |
|------------|---------|
| PMP nominal (Fig. 1) | `main('pmp')` or `main('smoke')` |
| Train DQN / TD3 (~3k episodes) | `main('train')` |
| Nominal comparison (Fig. 2) | `main('eval', struct('run_cohort',false))` then `main('figures', 2)` |
| 100-patient cohort, \(U[0.9,1.1]\) on \(R,\Phi\) | `main('eval')` (`cfg.num_patients=100`, `cfg.cohort_variation=0.10`) |
| Figs. 3–4 | `main('figures', 3)` / `main('figures', 4)` |
| DQN action ablation (Table I) | `main('ablation')` — train separate agents per action count for published numbers |

## Suggested workflow

1. `main('test')` — unit smoke (no agents)
2. `main('validate')` — publication pin self-check
3. `main('smoke')` — PMP solve + Table III soft pin
4. `main('train')` — or copy pretrained `agent_final.mat` into `savedAgents/*/PaperStyle/`
5. `main('eval')` then `main('figures')`

## Agent provenance (local nested workspace)

When this repo is nested under `DRL_Chemotherapy/`, `config.m` shares `../savedAgents` and `../data`. As of the deep-pass audit:

| Agent | Active `PaperStyle` | Origin tag |
|-------|---------------------|------------|
| DQN | `20260530_113808_Clean` | May 30 2026 clean train |
| TD3 | `20260530_120643_Clean` | May 30 2026 clean train |

Standalone clones have empty `savedAgents/` placeholders until `main('train')` or an author transfer.

## Local layout (pre-push)

| Folder | Role |
|--------|------|
| `DRL_Chemotherapy/` | Full PhD workspace (legacy scripts, `savedAgents/`, `OfflineRL/`) — **not** the published git root |
| `DRL_Chemotherapy/DRL_chemo_github/` | Lean export — **this repository** |

See `docs/VALIDATION.md` for matching legacy results before publishing.

## Notes

- From project root, parent `main(...)` may delegate into this folder.
- When nested, `config.m` reads `../savedAgents` and `../data` from the parent workspace.
- Figure export uses PNG under `results/figures/` (vector EPS can be added in `generate_figures.m`).
- MPC comparisons in early drafts are optional; Fig. 4 in the paper shows DQN and TD3 only.
- `main('test')` runs unit smoke tests (not full evaluation). Use `main('eval')` for paper evaluation.
