---
name: notebook-review
description: Check whether a Jupyter notebook still reads as one coherent argument (stale text,
  broken story, hidden state, clutter) and propose simplifications or restructurings. Use after
  substantial notebook changes, when starting work on a notebook, before a submission, or when
  asked to review, clean up, or restructure a notebook.
---

# Notebook review

Report findings; don't fix anything without asking.

## 1. Read the notebook as a reader would
- Large notebooks exceed the Read tool's limit: dump cell sources and text outputs with a short
  script (cell index, type, source, stream outputs), not the images.
- Write down the notebook's argument in 3–5 sentences: the question, the steps, the conclusion.
  If that is hard to write, that is the first finding.

## 2. Check

**Stale text**
- Markdown conclusions ("What this shows", quoted numbers, "above/below", "X× looser") that no
  longer match the current outputs or parameters. Quote the sentence and the output it contradicts.
- Text that holds only for one parameter setting (e.g. Poisson arrivals) without saying so.
- Computed numbers hardcoded as literals in markdown or code comments (see the "No hardcoded
  result numbers" rule in CLAUDE.md), even if they are still correct today.

**Broken story**
- Sections that no longer follow from each other, or whose question is no longer answered.
- Symbols or terms used before they are defined (see the explanation rule in CLAUDE.md).
- Figures that no markdown cell explains.

**Hidden state**
- Cells that depend on execution order or on a variable set further down.
- Variables reused with a different meaning (e.g. `rng`, `n_grid`, `latency`).
- Outputs a clean top-to-bottom run would not reproduce. If unsure, execute a copy in the
  scratchpad (`jupyter nbconvert --to notebook --execute`) and compare; never re-execute the
  original in place without asking.

**Accumulated clutter**
- OLD/NEW code kept side by side, commented-out blocks, revision notes that have served their
  purpose, duplicated helpers that also exist in `agent/` or another notebook.

## 3. Report
1. The argument in 3–5 sentences (from step 1).
2. Findings grouped by the four categories above, each with the cell index and a one-line
   description. Skip empty categories.
3. At most three proposals to simplify or restructure, ranked by benefit. For each: what it
   removes or moves, and why the notebook reads better afterwards. Typical proposals: split the
   notebook in two, move helpers into a module under `agent/`, drop OLD code once a change is
   settled, merge sections that answer the same question.
