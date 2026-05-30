# Reproducibility — DRL heterogeneous chemotherapy

Maps the manuscript *Adaptive Chemotherapy via Deep Reinforcement Learning for Heterogeneous Populations* to this repository.

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
| Table II hyperparameters | `build_agent.m`, `drl_pipeline.m` (`train_one_agent`) |

## Experiments (Section IV)

| Experiment | Command |
|------------|---------|
| PMP nominal (Fig. 1) | `main('pmp')` |
| Train DQN / TD3 (~3k episodes) | `main('train')` |
| Nominal comparison (Fig. 2) | `main('collect')` then `main('figures', 2)` |
| 100-patient cohort, \(U[0.9,1.1]\) on \(R,\Phi\) | `main('collect')` uses `cfg.num_patients=100`, `cfg.cohort_variation=0.10` |
| Figs. 3–4 | `main('figures', 3)` / `main('figures', 4)` |
| DQN action ablation (Table I) | `main('ablation')` — train separate agents per action count for published numbers |

## Suggested workflow

1. `main('pmp')`
2. `main('train')` — or copy pretrained `agent_final.mat` into `savedAgents/*/PaperStyle/`
3. `main('collect')`
4. `main('figures')`

## Local layout (pre-push)

| Folder | Role |
|--------|------|
| `DRL_Chemotherapy/` | Full PhD workspace (legacy scripts, `savedAgents/`, `OfflineRL/`) |
| `DRL_Chemotherapy/DRL_chemo_github/` | Lean export — **push this folder** when validation passes |

See `docs/VALIDATION.md` for matching legacy results before publishing.

## Notes

- From project root, `main(...)` delegates into this folder.
- When nested, `config.m` reads `../savedAgents` and `../data` from the parent workspace.
- Figure export uses PNG under `results/figures/` (vector EPS can be added in `generate_figures.m`).
- MPC comparisons in early drafts are optional; Fig. 4 in the paper shows DQN and TD3 only.
