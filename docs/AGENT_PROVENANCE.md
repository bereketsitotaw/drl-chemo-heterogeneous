# Agent provenance

Pretrained DQN/TD3 checkpoints are **not** shipped in git. This note records how evaluation discovers agents and what the authors’ local May 2026 clean runs correspond to.

## Discovery order (`config.resolve_agent_subdir`)

1. `savedAgents/<DQN|TD3>/PaperStyle/agent_final.mat` (preferred evaluation default)
2. Else newest `savedAgents/<DQN|TD3>/*_Clean/agent_final.mat`
3. Else expect `PaperStyle` (error at load time if missing)

When this repository is nested under `DRL_Chemotherapy/`, `config.m` sets `workspace_dir` to the parent so agents/data are shared.

## Author local runs (parent workspace, not in git)

Observed on the deep-pass machine (compare-only; not copied into the published tree):

| Agent | Symlink | Target folder | Artifact |
|-------|---------|---------------|----------|
| DQN | `PaperStyle` → | `20260530_113808_Clean` | `agent_final.mat`, `stats_final.mat`, `training_manifest.mat`, `training_curve.png` |
| TD3 | `PaperStyle` → | `20260530_120643_Clean` | same |

Backups retained under `PaperStyle_backup_*` pointing at earlier PaperStyle / Clean runs. Older `*_NominalBenchmark` and `20260329_*_PaperStyle` folders remain for forensics only.

## Reproduce without author weights

```matlab
main('train')   % ~2–5 hours; promotes new runs to PaperStyle
main('eval')
```

## Request pretrained weights

Open a GitHub issue or email the corresponding author (see paper / `CITATION.cff`). Place received `agent_final.mat` under `savedAgents/DQN/PaperStyle/` and `savedAgents/TD3/PaperStyle/`.
