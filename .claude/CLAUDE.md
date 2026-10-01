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
- **Explanations (markdown cells, comments, answers):** write so a reader can follow
  without guessing.
  - Define every symbol and term where it first appears: what it is, its unit, and
    where it comes from (e.g. "X, the execution time of the arriving item, in seconds").
  - Never use a pronoun whose referent is unclear; name the thing ("the backlog",
    not "it").
  - Give the reason behind every non-obvious step or number, not just the result
    (e.g. why the slope is −1, not only that it is).
  - Describe what a plot shows (axes, units, what to look at) before interpreting it.
  - Prefer one concrete example with real numbers over an abstract statement.
- **No hardcoded result numbers in text:** never write a number that the code computes
  (a quantile, a ratio, a coverage, a fitted parameter) as a literal in a markdown cell
  or a code comment; it goes stale as soon as a parameter changes.
  - Numbers that support a conclusion are printed from variables with f-strings, or the
    conclusion cell is rendered from code:
    `display(Markdown(f"The bound is {ratio:.2f}× the true p95 ..."))`.
  - Static markdown describes results qualitatively ("the bound is about twice the true
    p95", "coverage stays at 100%") and points to the output that holds the number.
  - Code comments explain *why*, never *what the result was*; don't write
    `# gives 0.287`.
  - Exempt: the definition of a parameter itself (`arrival_rate = 3.0  # items/s`),
    physical or mathematical constants, and numbers inside a derivation that follow
    from the stated parameters (e.g. ρ = λμ = 0.24), as long as those parameters are
    named next to them.
- **Method / baseline sections:** follow the `method-section-structure` skill (overview with setup
  cell, one subsection per method with its computation directly after it, results last).
- **Notebook coherence:** run the `notebook-review` skill after a substantial change to a
  notebook (a changed assumption, a new section, or edits to ≥ 3 cells) and the first time
  I work on a notebook in a session. Report findings and proposals; don't fix without asking.
- **Commits:** only when I ask. Short, lowercase, imperative messages. Never commit
  `*.joblib`, `.idea/`, or `__pycache__/`.

## Conventions

- Style: follow surrounding code; no new formatter/linter config.
- Type hints on new public functions; short docstrings only where non-obvious.
- Prefer numpy/pandas vectorization over Python loops on metric data.
- Use `torch`/`gpytorch` for GP work; `scikit-learn` GPs only where already used.

