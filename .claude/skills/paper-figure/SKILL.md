---
name: paper-figure
description: Create, restyle, or export a matplotlib/seaborn figure intended for a paper
  (ICSOC, TSC, SummerSOC). Use when asked to make a figure "for the paper", export a plot,
  or fix a figure's styling or readability.
---

# Paper figure

## 1. Clarify before plotting
- Which claim does the figure support? Write it as one sentence in the cell's markdown
  above the plot; the figure should make that claim visible at a glance.
- Which data file in `statics/` (or which notebook variable) is the source? Name it in
  a code comment.

## 2. Style (match the existing figures)
- `plt.style.use('default')` and `plt.rcParams.update({'font.size': 12})` at the top of
  the cell, not per axis.
- Size: single panel `figsize=(9, 4.5)`; row of two panels `(15, 4.8)`. Figures are scaled
  down in LaTeX, so keep fonts ≥ 12 and lines ≥ 1.5 pt.
- Colors: one fixed color per method or process, reused across all figures of a paper
  (e.g. PPG always green, oracle always purple). Also distinguish series by marker or
  linestyle, so the figure reads in grayscale.
- Axis labels in words with units: `latency d (s)`, `utilization ρ = λμ`. Use log axes
  for tails and for ε.
- Reference lines (target ε, ratio = 1, violation boundary) as thin black lines with
  a legend entry.
- No suptitle in paper exports (the caption does that); per-panel titles only if they
  distinguish panels.

## 3. Export
- Save to `figures/<experiment>/<snake_case_name>.pdf` (vector, for LaTeX) and a `.jpg`
  next to it (dpi=300, `facecolor='white'`) for slides.
- Always `bbox_inches='tight'`; also `plt.show()` so it renders in the notebook.
- Don't overwrite an existing figure file without saying so; mention the path in the reply.

## 4. Explain the figure (per the CLAUDE.md explanation rule)
In the markdown cell below the plot, in this order:
1. What is plotted: each axis, its unit, and what one line/point/bar represents.
2. What to look at: the one feature that carries the claim (a gap, a crossing, a slope).
3. What it means, with one concrete number read off the plot.

## 5. Checklist before finishing
- [ ] Source data named; any aggregation (mean, p95, median over configs) stated in the
      axis label or a comment
- [ ] Every symbol in labels and legend is defined in the text
- [ ] Readable in grayscale and at half size
- [ ] Numbers quoted in the markdown match the printed output
