"""Style of the paper figures of the TSC paper: one color, marker and line style per method, the light-red zone where a
method is no bound, and the export to figures/tsc. The style follows draft_problem_figure.py; see the paper-figure skill.
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.ticker import LogFormatterSciNotation

from .evaluation import METHOD_COLORS, PROPOSED, PROPOSED_SNC

FIGURES = Path(__file__).resolve().parents[3] / "figures" / "tsc"
# marker and line style per method, so that the methods can be told apart in grayscale; the color is METHOD_COLORS
METHOD_STYLES = {PROPOSED: ("o", "-"), PROPOSED_SNC: ("D", "-"), "GP Sum": ("s", "--"), "DNN Sum": ("X", "--"), "GP Queue": ("^", "-."),
                 "Quantile Union": ("v", ":"), "Conformal": ("P", ":"), "Oracle": ("*", "--")}
FAILURE_COLOR, FAILURE_TEXT_COLOR = "mistyrose", "darkred"   # zone where a method is no bound (below the measured quantile)
WASTE_COLOR = "0.9"                                           # optional zone where a method wastes resources (far above it)
BAND_ALPHA = 0.2                                              # bands over seeds or quantiles, drawn without an edge
FIGSIZE = (6.0, 3.7)                                          # (width, height) in inches of every single-panel evaluation figure, so that
                                                              # LaTeX scales them all by the same factor and their fonts match; small
                                                              # enough that the fonts stay readable in one column of the paper, large
                                                              # enough that two half figures still fit their legends
# Fixed layout (inches) of every evaluation figure: the plot area sits at the same place in every figure (paper_figure,
# half_figure), so that every exported file has exactly the size of its canvas and the same plot area. The margins hold
# one row of tick labels and one axis label on each axis, measured from the widest labels of all figures; HALF_GAP, the
# left margin of a right half figure, keeps the last tick label of the left half and the first of the right half apart.
# MARGIN_TOP holds half a line of text, the part of a tick label at the top of the vertical axis that lies above it.
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_BOTTOM, MARGIN_TOP, HALF_GAP = 0.88, 0.1, 0.55, 0.09, 0.2


def tightness_label(level_label: str) -> str:
    """Label of every tightness axis: the latency of a method divided by the measured quantile level_label (e.g. "p95")."""
    return f"tightness: latency ÷ measured {level_label}"


def tightness_tick(value: float) -> str:
    """Tick label of every tightness axis: the ratio with a multiplication sign, e.g. 2×."""
    return f"{value:g}×"


@contextmanager
def paper_style():
    """The style of the paper figures inside a `with` block, without changing the figures drawn after it."""
    with plt.style.context("default"), plt.rc_context({"font.size": 12}):
        yield


def method_linewidth(name: str) -> float:
    """The two versions of Pegasus are drawn thicker than the baselines."""
    return 2.5 if name in (PROPOSED, PROPOSED_SNC) else 1.8


def method_handle(name: str, filled: bool = True, line: bool = True) -> Line2D:
    """Legend entry of a method: its color, marker and line style (without the line for a dot plot, line=False)."""
    marker, linestyle = METHOD_STYLES[name]
    return Line2D([], [], linestyle=linestyle if line else "", marker=marker, color=METHOD_COLORS[name], markerfacecolor=METHOD_COLORS[name] if filled else "white",
                  markersize=7, linewidth=method_linewidth(name))


def holds_handles(subject: str = "") -> tuple:
    """Legend entries of the filled marker (the method holds: it reaches the confidence level under every seed) and the
    hollow marker (it does not); subject is appended, e.g. " on the chain". The figure's description defines "holds"."""
    marker = lambda face: Line2D([], [], linestyle="", marker="o", markersize=7, markerfacecolor=face, markeredgecolor="dimgray", markeredgewidth=1.5)
    return ([marker("dimgray"), marker("white")],
            [f"holds{subject} on all seeds", f"does not hold{subject}"])


def dodge_offsets(n: int, width: float = 0.7) -> np.ndarray:
    """Horizontal offsets of n series side by side within one category of width 1 (a dodged dot plot): a category whose
    instances are independent, such as one chain, gets one column per method instead of a line through the categories."""
    return (np.arange(n) - (n - 1) / 2) * width / max(n - 1, 1)


def legend_inside(ax, handles: list, labels: list, segments: list, locations: tuple = ("upper center", "upper left", "upper right"),
                  max_doublings: int = 3, **options):
    """A legend inside the axes that hides as little data as possible. segments: (x, y_low, y_high) in data
    coordinates of everything drawn, a dot as (x, y, y) and a bar or band from y_low to y_high; a value above the top
    of the axis counts as at the top, where it is cut off. Every location (as for ax.legend) is tried in turn; if the
    legend hides data at all of them, the top of the logarithmic y-axis is doubled (at most max_doublings times),
    which moves the data down and away from a legend at the top, and the locations are tried again. If data stays
    hidden, the legend goes to the location that hides the fewest segments, with a printed note. options: as for
    ax.legend (ncol, fontsize, ...). Returns the legend."""
    top = ax.get_ylim()[1]
    x, low, high = (np.asarray(values, dtype=float) for values in zip(*segments)) if segments else (np.empty(0),) * 3
    low, high = np.minimum(low, top), np.minimum(high, top)
    for doubling in range(max_doublings + 1):
        hidden_at = {}
        for location in locations:
            legend = ax.legend(handles, labels, loc=location, **options)
            ax.figure.canvas.draw()
            box = legend.get_window_extent().transformed(ax.transData.inverted())
            hidden_at[location] = int(((x >= box.x0) & (x <= box.x1) & (high >= box.y0) & (low <= box.y1)).sum())
            if not hidden_at[location]:
                return legend
        if doubling < max_doublings:
            ax.set_ylim(ax.get_ylim()[0], 2 * ax.get_ylim()[1])
    best = min(hidden_at, key=hidden_at.get)
    print(f"legend_inside: the legend hides {hidden_at[best]} points or bands at its best location ({best})")
    return ax.legend(handles, labels, loc=best, **options)


def _figure(width: float, left: float):
    """Figure of the height of FIGSIZE and the given width, with the plot area at the fixed margins (left: its left margin)."""
    height = FIGSIZE[1]
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((left / width, MARGIN_BOTTOM / height, 1 - (left + MARGIN_RIGHT) / width, 1 - (MARGIN_BOTTOM + MARGIN_TOP) / height))
    return fig, ax


def paper_figure():
    """Figure and axes of an evaluation figure: FIGSIZE with the plot area at the fixed margins, the same in every figure.
    Do not call tight_layout, which would move the plot area."""
    return _figure(FIGSIZE[0], MARGIN_LEFT)


def half_figure(side: str, shared_y: bool = True):
    """Figure and axes of one of two panels that sit side by side in the width of one FIGSIZE figure (side: "left" or
    "right"). Both have the height of FIGSIZE and a plot area of the same size at the same height as every other
    figure, so that their axes line up; in LaTeX both are included at the same height. shared_y: only the left one has
    room for the labels of the vertical axis, which the right one shares, and the left file is the wider one; with
    shared_y=False (two different quantities on the vertical axes) both have that room and are FIGSIZE[0] / 2 wide."""
    right_left = HALF_GAP if shared_y else MARGIN_LEFT
    plot_width = (FIGSIZE[0] - MARGIN_LEFT - right_left - 2 * MARGIN_RIGHT) / 2
    left = MARGIN_LEFT if side == "left" else right_left
    return _figure(left + plot_width + MARGIN_RIGHT, left)


def fit_half_pair(axes: list) -> None:
    """Size two half figures with their own vertical axes (half_figure(side, shared_y=False), axes: [left, right]) so
    that each left margin is just as wide as the labels of its vertical axis, and the two plot areas share the rest of
    FIGSIZE[0] equally: short tick labels (0.1, 1) leave no empty strip left of the axis label, and the pair stays
    exactly one figure wide with two plot areas of the same size. Call when the vertical axes are complete."""
    lefts = []
    for ax in axes:
        ax.figure.canvas.draw()
        labels = ax.yaxis.get_tightbbox(ax.figure.canvas.get_renderer())   # tick labels and axis label, in pixels
        lefts.append((ax.bbox.x0 - labels.x0) / ax.figure.dpi + 0.02)        # 0.02 in so that the axis label does not touch the border
    plot_width = (FIGSIZE[0] - sum(lefts) - 2 * MARGIN_RIGHT) / 2
    height = FIGSIZE[1]
    for ax, left in zip(axes, lefts):
        width = left + plot_width + MARGIN_RIGHT
        ax.figure.set_size_inches(width, height)
        ax.set_position((left / width, MARGIN_BOTTOM / height, plot_width / width, 1 - (MARGIN_BOTTOM + MARGIN_TOP) / height))


def fit_x_tick_labels(ax) -> None:
    """Widen a linear horizontal axis until its first and last tick labels lie inside the figure: a long label at the
    end of the axis (e.g. "heaviest (2.35)") would otherwise reach beyond the fixed margins and be cut off."""
    renderer = ax.figure.canvas.get_renderer()
    for _ in range(5):   # the labels move with the axis, so a few steps settle it
        ax.figure.canvas.draw()
        boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels() if label.get_text()]
        if not boxes:
            return
        beyond_left = max(0.0, ax.figure.bbox.x0 - min(box.x0 for box in boxes))
        beyond_right = max(0.0, max(box.x1 for box in boxes) - ax.figure.bbox.x1)
        if not beyond_left and not beyond_right:
            return
        low, high = ax.get_xlim()
        per_pixel = (high - low) / ax.bbox.width
        ax.set_xlim(low - (beyond_left + 2) * per_pixel, high + (beyond_right + 2) * per_pixel)   # 2 pixels to spare


def plain_log_ticks(axis) -> None:
    """Tick labels of a logarithmic axis (ax.yaxis or ax.xaxis) as plain numbers (0.4, 0.001) instead of 4×10⁻¹, at the
    ticks that matplotlib labels anyway, so that the labels fit the fixed margins."""
    class Plain(LogFormatterSciNotation):
        def __call__(self, x, pos=None):
            return f"{x:g}" if super().__call__(x, pos) else ""
    axis.set_major_formatter(Plain())
    axis.set_minor_formatter(Plain(labelOnlyBase=False, minor_thresholds=(1, 0.4)))


def save_figure(fig, name: str) -> None:
    """figures/tsc/<name>.pdf for LaTeX and .jpg for slides, at exactly the size of the figure (no cropping; the fixed
    margins of paper_figure and half_figure leave no empty border). Prints a note if a label reaches beyond the figure
    and is cut off, which means that the margins are too small for it."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # the plot areas, their axes with tick and axis labels, and their legends, in inches; fig.get_tightbbox and
    # ax.get_tightbbox clip the labels to the figure and would miss a cut-off label, the boxes of the parts themselves do not
    boxes = [box for ax in fig.axes for box in (ax.bbox, ax.xaxis.get_tightbbox(renderer), ax.yaxis.get_tightbbox(renderer),
                                                 ax.get_legend().get_window_extent(renderer) if ax.get_legend() else None) if box is not None]
    drawn = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    width, height = fig.get_size_inches()
    beyond = {side: round(value, 2) for side, value in (("left", -drawn.x0), ("right", drawn.x1 - width), ("bottom", -drawn.y0),
                                                         ("top", drawn.y1 - height)) if value > 0.01}
    if beyond:
        print(f"save_figure: {name} reaches beyond the figure by {beyond} inches")
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.jpg", dpi=300, facecolor="white")
