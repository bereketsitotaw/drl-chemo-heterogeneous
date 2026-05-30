# Push to GitHub (code only — no agent weights)

## What is published

- MATLAB pipeline (`main.m`, `scripts/`, docs)
- **No** `agent_final.mat`, `stats_final.mat`, or `data/*.mat`
- **No** `results/` or figure PNGs

Users clone, run `main('train')`, then `main('eval')` on their machine.

## Push steps

```bash
cd /Users/berekets.kidane/Desktop/PHD/Year2/Projects/Cancer/DRL_Chemotherapy/DRL_chemo_github

git init
git add .
git status    # confirm savedAgents/ only has README.md, no .mat files
git commit -m "Initial release: DRL heterogeneous chemotherapy reproduction code"
git branch -M main
git remote add origin https://github.com/bereketsitotaw/drl-chemo-heterogeneous.git
git push -u origin main
```

Commit with **your** name/email only.

## What stays on your Mac only

Parent folder `DRL_Chemotherapy/savedAgents/` — all training runs, including:

- `20260530_113808_Clean` (DQN)
- `20260530_120643_Clean` (TD3)
- Older `NominalBenchmark` / backup folders

Optional: delete `DRL_chemo_github/savedAgents/DQN/Clean_20260530` copies if you created them earlier (not needed for git).

## Pretrained agents (on request)

The public repo stays code-only. README states that users may **contact you** for trained checkpoints if they do not wish to retrain.

When someone asks, you can share (e.g. zip via email, institutional cloud, or a one-off GitHub Release):

- `savedAgents/DQN/20260530_113808_Clean/`
- `savedAgents/TD3/20260530_120643_Clean/`

They should place folders under `savedAgents/` and run `main('eval')`, or symlink as `PaperStyle`.

Optional later: publish the same zip as a formal **GitHub Release** asset and link it in README.
