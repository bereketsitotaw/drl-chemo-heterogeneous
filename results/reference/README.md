# Reference outputs

Paper Table III / IV numeric pins live in `configs/reference_metrics.m` (always tracked).

| Controller | Final tumor (avg) | Reduction (%) | Total drug |
|------------|-------------------|---------------|------------|
| PMP | 1.23 | 87.72 | 27.50 |
| DQN | 1.31 | 86.93 | 26.70 |
| TD3 | 1.09 | 89.11 | 26.90 |

Cohort (Table IV): DQN mean reduction 82.78%; TD3 84.07%.

Optional: place small reference `.mat` snippets or figure checksums here for regression checks. Full solutions and trained agents are **not** versioned (see `savedAgents/README.md`).

Generated runtime mats go to `results/generated/` (and legacy `../data/` when nested).
