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
- **TSC notebooks (`notebooks/tsc/`): use `pegasus/figures.py`.** It holds the approved style
  (reference figures: `draft_problem_figure.py` and B.1 on All Chains in `2a_base_case.ipynb`):
  - draw inside `with paper_style():` (default style, font 12, scoped to the block so later
    notebook figures are unchanged);
  - every method has a fixed color (`METHOD_COLORS`) **and** marker + line style
    (`METHOD_STYLES`), Pegasus thicker (`method_linewidth`); add a new method to both dicts
    instead of styling it locally;
  - shade the failure zone (below the measured quantile, SLO violated) in `FAILURE_COLOR`
    and label it inside the axes with `FAILURE_TEXT_COLOR` text ("below the measured p95:
    no bound"); optionally the wasteful zone in `WASTE_COLOR`;
  - bands over seeds/quantiles with `alpha=BAND_ALPHA, linewidth=0`;
  - validity: filled marker = holds, hollow marker = does not hold, legend from
    `method_handle` + `holds_handles`;
  - categorical axes: group names written once inside the axes, short tick labels, and the
    categories sorted by a meaningful quantity (e.g. chains by number of services);
  - lines only along an ordered quantity (a sweep). Independent instances (chains, services)
    get a dodged dot plot: one column per method within a category (`dodge_offsets`), a
    thin vertical bar for the range over seeds, legend `method_handle(name, line=False)`;
  - legend inside the axes where possible, via `legend_inside` (tries several locations,
    extends a log y-axis upward if the legend would hide data; re-set the ticks after it);
    move labels that would collide with it (e.g. group names) outside the frame
    (`ax.secondary_xaxis("top")`);
  - export with `save_figure(fig, name)` (→ `figures/tsc/<name>.pdf` + `.jpg`).
- Elsewhere: `plt.style.use('default')` and `plt.rcParams.update({'font.size': 12})` at the
  top of the cell, not per axis, and the same rules as above by hand.
- Size and layout (TSC): `fig, ax = paper_figure()` from `pegasus/figures.py`: exactly `FIGSIZE`
  (6.5 × 4 in) with the plot area at fixed margins, the same in every evaluation figure, so that all
  files have the same size and LaTeX scales them alike. Never `tight_layout` or a cropped export.
  The margins hold one row of tick labels and one axis label per axis: put a second value in
  parentheses (`"3 (7)"`), not in a second tick row; keep the y label no longer than the tightness
  label; write log ticks as plain numbers (`plain_log_ticks`); widen a linear x axis whose end
  labels stick out with `fit_x_tick_labels`. Two related figures side by side in the width of one:
  `half_figure("left" | "right")` (e.g. `tightness_sweep_figure(..., half=...)` in B.4): short x
  labels, y labels only on the left, the legend split over both halves; two halves with their own
  y axes (`shared_y=False`) get `fit_half_pair` so short tick labels leave no empty strip; in
  LaTeX include both at the same height. Figures are scaled down in LaTeX, so keep fonts ≥ 12 and lines ≥ 1.5 pt.
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
- TSC: `save_figure(fig, name)` saves at exactly the figure size (the fixed margins leave no white
  border) and prints a note if a label reaches beyond the figure: fix the label, don't ignore the
  note. Elsewhere: `bbox_inches='tight', pad_inches=0`. Also `plt.show()` so it renders in the notebook.
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
