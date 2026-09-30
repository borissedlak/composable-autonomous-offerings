# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Research code for *Composable Autonomous Offerings*: GP / Deep GP models of individual
service capabilities, used to predict end-to-end performance of dynamically composed
service chains under resource constraints. Results feed papers (ICSOC, TSC, SummerSOC).

- `agent/components/` — core library (GP models, optimizer, SLO registry, RASK, CAO)
- `agent/components/legacy/` — old code; don't modify or import from it unless asked
- `notebooks/<venue>/` — experiment notebooks per paper; `notebooks/archived/` is frozen
- `statics/` — datasets, candidates, and experiment outputs
- `figures/` — exported figures for the papers
- Root `*Client.py` — Docker / HTTP / Prometheus clients for the physical testbed

## Environment

- Python venv at `.venv/` — run things with `.venv/bin/python`, never the system Python
- Dependencies: `requirements.txt` (pin with `~=` like the existing entries)
- The testbed (Docker, Prometheus) is usually NOT running locally — don't try to start
  containers or hit endpoints unless I ask

## How I want you to work

- **Ask before large refactors.** This is research code; prefer small, targeted edits.
- **Don't silently change experiment semantics** (hyperparameters, seeds, SLO definitions,
  β penalty, data splits). If a change affects results, say so explicitly.
- **Notebooks:** edit cells in place with NotebookEdit; don't rewrite whole notebooks.
  Keep outputs unless I ask to clear them. Don't re-run long training cells without asking.
- **Reproducibility:** fix random seeds; write outputs to `statics/` or `figures/` with
  descriptive filenames, not to the repo root.
- **Statistics:** state assumptions (distribution, independence) and prefer robust /
  non-parametric methods for tail-latency analysis. Explain what a result *means*, not
  just the number.
- **Plots:** see the `paper-figure` skill.
- **Commits:** only when I ask. Short, lowercase, imperative messages. Never commit
  `*.joblib`, `.idea/`, or `__pycache__/`.

## Conventions

- Style: follow surrounding code; no new formatter/linter config.
- Type hints on new public functions; short docstrings only where non-obvious.
- Prefer numpy/pandas vectorization over Python loops on metric data.
- Use `torch`/`gpytorch` for GP work; `scikit-learn` GPs only where already used.

