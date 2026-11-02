# Adaptive Chemotherapy via Deep Reinforcement Learning

MATLAB code for *Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning* (Kidane, Motayed, Wang).

Compares **DQN** (discrete dosing) and **TD3** (continuous dosing) against a **PMP** open-loop benchmark on a 21-subpopulation heterogeneous tumor model, with cohort robustness over 100 virtual patients (±10% on growth and drug sensitivity).

**This repository contains code only** — no pretrained `.mat` agents (train locally; see below). If you prefer not to retrain, contact the authors and we can share the trained checkpoints on request.

## Quick start

```matlab
git clone https://github.com/bereketsitotaw/drl-chemo-heterogeneous.git
cd drl-chemo-heterogeneous

main('test')                 % lightweight smoke unit tests
main('validate')             % publication pin self-check
main('smoke')                % PMP + Table III soft pin (~minutes)
main('pmp')                  % PMP benchmark
main('train')                % DQN + TD3 (~2–5 hours)
main('eval')                 % PMP + nominal + 100-patient cohort
main('figures')              % display Figs. 1–4
```

Quick checks without cohort:

```matlab
main('eval', struct('run_cohort', false))
```

Export figure PNGs:

```matlab
main('figures', struct('save_png', true))
```

## Paper → code

| `main(...)` | Paper |
|-------------|-------|
| `'pmp'` / `'smoke'` | Fig. 1 — PMP benchmark |
| `'eval'` + `'figures', 2` | Fig. 2 — nominal DQN / TD3 vs PMP |
| `'eval'` + `'figures', 3` | Fig. 3 — cohort Pareto |
| `'eval'` + `'figures', 4` | Fig. 4 — median robustness trajectories |
| `'train'` + `'ablation'` | Table I — DQN action count (train per action count) |

Cost weights locked to $(\alpha,\beta,\gamma)=(500,400,4000)$. Details: `docs/REPRODUCIBILITY_DRL.md`, training: `docs/TRAINING.md`, agents: `docs/AGENT_PROVENANCE.md`.

## Layout

```
main.m
scripts/          % model, PMP, RL env, train, eval, figures
configs/          % publication asserts, reference pins, smoke settings
tests/            % lightweight smoke suite
docs/
results/
  generated/      % runtime mats (gitignored)
  reference/      % pin docs (tracked)
savedAgents/      % README only in git; agents created by main('train')
```

## Trained agents (local only)

After `main('train')`, checkpoints are saved under:

```
savedAgents/DQN/<timestamp>_Clean/agent_final.mat
savedAgents/TD3/<timestamp>_Clean/agent_final.mat
```

Evaluation uses `PaperStyle` symlinks (created automatically). These files are **gitignored**.

**Pretrained weights on request:** The released code is intended for full reproduction via `main('train')`. If you need the authors’ trained DQN/TD3 checkpoints (May 2026 clean training) to skip retraining, please open a GitHub issue or email the corresponding author listed in the paper.

## Requirements

- MATLAB **R2020a+** (R2024b used in the paper)
- **Reinforcement Learning Toolbox**
- **Parallel Computing Toolbox** (recommended)
- Optimization Toolbox (`fsolve` for PMP)

## Citation

```bibtex
@article{kidane2026adaptivechemo,
  author  = {Kidane, Bereket Sitotaw and Motayed, Md Samiul Haque and Wang, Shuo},
  title   = {Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning},
  journal = {TBD},
  year    = {2026},
  note    = {University of Texas at Arlington}
}
```

## License

MIT — see [LICENSE](LICENSE).
