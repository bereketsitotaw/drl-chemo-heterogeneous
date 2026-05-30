# Pretrained agents (not in git)

Agent checkpoints (`.mat`) are **not** stored in this repository to keep the repo small. You can reproduce them with `main('train')`, or **contact the authors** if you would like the pretrained DQN/TD3 agents shared directly.

## Train locally

```matlab
cd drl-chemo-heterogeneous   % after clone
main('train')                % DQN + TD3 (~2–5 hours; RL Toolbox + parpool)
main('eval')                 % evaluate saved agents
main('figures')              % display paper figures
```

Agents are written to `savedAgents/DQN/<timestamp>_Clean/` and `savedAgents/TD3/<timestamp>_Clean/`, then linked as `PaperStyle` for evaluation.

## Requirements

- MATLAB R2020a+ with **Reinforcement Learning Toolbox**
- **Parallel Computing Toolbox** recommended

See `docs/TRAINING.md` for hyperparameters and `docs/REPRODUCIBILITY_DRL.md` for paper mapping.
