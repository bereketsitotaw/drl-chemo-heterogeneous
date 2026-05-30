# Clean training — DQN & TD3

Train **fresh** agents from randomized initial states (paper setup). Old `PaperStyle` runs are backed up, not deleted.

## One command (both agents, ~2–5 hours total)

```matlab
cd DRL_Chemotherapy/DRL_chemo_github
main('train')
```

This will:

1. Start an 8-worker `parpool` (if needed)
2. Train **DQN** (7 actions, seed 42) → `savedAgents/DQN/<timestamp>_Clean/`
3. Train **TD3** (seed 43)
4. Back up old `PaperStyle` folders and link new runs as `PaperStyle`
5. Save `results/training_manifest_*.mat`

## Train one agent only

```matlab
main('train', struct('agents', 'DQN'))
main('train', struct('agents', 'TD3'))
```

## Options

```matlab
opts = struct( ...
    'agents', 'both', ...
    'max_episodes', 3000, ...
    'run_tag', 'Clean', ...
    'promote', true, ...      % false = keep old PaperStyle paths
    'pool_workers', 8, ...
    'rng_base', 42);
main('train', opts);
```

## After training

```matlab
main('eval')       % evaluate new PaperStyle agents
main('figures')
```

## Hyperparameters (Table II)

| Setting | Value |
|---------|--------|
| Episodes | 3000 (early stop if avg reward ≤ −286) |
| Steps / episode | 100 (dt=0.1, T=10 yr) |
| DQN actions | 7 |
| Networks | 400 → 300, ReLU (+ batch norm on TD3 actor) |
| Training init | Random N₀ (not nominal) |
| Seeds | DQN 42, TD3 43 |

## Outputs per run

```
savedAgents/DQN/<timestamp>_Clean/
  agent_final.mat
  stats_final.mat
  training_manifest.mat
  training_curve.png
```
