# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `configs/` publication asserts, Table III/IV reference pins, reproducibility + smoke settings
- `tests/run_smoke_tests.m` (7 cases) via `main('test')`
- `main('smoke')` PMP plumbing + soft Table III pin; `main('validate')` config pin check
- `docs/AGENT_PROVENANCE.md` and expanded `docs/REPRODUCIBILITY_DRL.md` (4-category status)
- `.github/workflows/repo-smoke.yml` path + publication-pin guards
- `results/generated/` layout (mirrors workspace `data/` mats)
- `CHANGELOG.md`, `requirements.txt`, `results/README.md` (structural hygiene)

### Changed
- `main('test')` now runs unit smoke tests (evaluation remains `main('eval')`)
- `config.resolve_agent_subdir` prefers `PaperStyle` over a non-existent `Clean_20260530` folder name
- Pipeline banners print publication cost pins; PMP/eval also write `results/generated/`
- Expanded `.gitignore` for generated mats / MATLAB / OS artifacts; track empty `savedAgents/*/.gitkeep`

### Fixed
- Agent discovery no longer prefers a Mac-local `Clean_20260530` name; public clones resolve via `PaperStyle` / newest `*_Clean` only
- `docs/GITHUB_PUSH.md` no longer embeds a machine-absolute `/Users/...` path

### Scientific
- **No changes** to dynamics, cost weights, reward map, PMP shooting law, or Table II hyperparameters
