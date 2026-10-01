---
name: method-section-structure
description: Structure for notebook sections that introduce and compare methods or
  baselines (bounds, compositions, models). Use when adding a method or baseline to a
  notebook, restructuring such a section, or writing its results.
---

# Method / baseline sections

Every section that involves methods or baselines has three parts, in this order.

## 1. Top-level description (section overview)
- `## N. Title`, then one paragraph on what the section answers.
- A bullet list of its subsections, one line each: `**Na. Name:** what it is`.
  Baselines are marked "(baseline)"; the results subsection comes last.
- One sentence on the setup cell and on where results are compared.
- Directly after the overview: **one setup cell** with everything all subsections
  share (configurations, simulated ground truth, beliefs, a results container
  such as `node_bounds[...]`). Nothing method-specific goes here.

## 2. One subsection per method or baseline
- `### Na. Name of the fundamental method` (e.g. "Union Bound", "Martingale Bound",
  "Composition by Independence (Baseline)"). Name strategies after the methods they
  combine ("union bound, martingale nodes"), never after a project label.
- Markdown: what the method does, every symbol defined, its assumptions, and whether
  it is a guarantee. Name the arrival process (or other setting) wherever a statement
  holds only for it.
- **Directly after it, the cell that computes it**: a few lines that call the
  implementation and store the result (in the container of the setup cell or in
  clearly named variables). No plotting, no comparison.
- Implementations live in modules, never in the notebook: methods in
  `snc_bounds.py`, baselines in `baselines.py`, models in `service_gp.py`.
- A method that does not apply in this section (e.g. propagated arrivals for a
  single service) gets one sentence saying so instead of a cell.

## 3. Results subsection (last)
- `### Nz. ...` compares only: table, plot, and a "What this shows" conclusion
  rendered from the variables (CLAUDE.md: no hardcoded result numbers), naming the
  setting it was computed for (e.g. "Poisson arrivals, λ = 4/s").
- A single configuration cannot show coverage; when a result comes from one
  configuration, say so and point to the section that sweeps configurations.

## Exception: sweeps
Sections that apply all methods to the same simulated data across many
configurations or depths (e.g. depth studies) keep **one loop** over the sweep, with
all methods inside it: splitting would store or re-simulate the data per method. The
methods are explained in earlier sections; the sweep section only references them.

## Checklist
- [ ] Overview lists every subsection; setup cell directly after it
- [ ] Every method/baseline: explanation, then its computation cell, nothing between
- [ ] Results subsection last, conclusions rendered and naming the setting
- [ ] No implementation code in the notebook; numbers unchanged after restructuring
