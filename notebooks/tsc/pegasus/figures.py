"""Style of the paper figures of the TSC paper: one color, marker and line style per method, the light-red zone where a
method is no bound, and the export to figures/tsc. The style follows draft_problem_figure.py; see the paper-figure skill.
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .evaluation import METHOD_COLORS, PROPOSED, PROPOSED_SNC

FIGURES = Path(__file__).resolve().parents[3] / "figures" / "tsc"
# marker and line style per method, so that the methods can be told apart in grayscale; the color is METHOD_COLORS
METHOD_STYLES = {PROPOSED: ("o", "-"), PROPOSED_SNC: ("D", "-"), "GP Sum": ("s", "--"), "DNN Sum": ("X", "--"), "Allen-Cunneen": ("^", "-."),
                 "Quantile Union": ("v", ":"), "Conformal": ("P", ":"), "Oracle": ("*", "--")}
FAILURE_COLOR, FAILURE_TEXT_COLOR = "mistyrose", "darkred"   # zone where a method is no bound (below the measured quantile)
WASTE_COLOR = "0.9"                                           # optional zone where a method wastes resources (far above it)
BAND_ALPHA = 0.2                                              # bands over seeds or quantiles, drawn without an edge


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


def save_figure(fig, name: str) -> None:
    """figures/tsc/<name>.pdf for LaTeX and .jpg for slides."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.jpg", dpi=300, bbox_inches="tight", facecolor="white")
