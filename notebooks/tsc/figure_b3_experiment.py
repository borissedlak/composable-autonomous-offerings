"""Figure that explains the proposed experiment B3 (varying depth) of new_experiments.md.

Claim of the figure: with the merge rule "wait for both children of the same frame", a hierarchy of any depth gives
one item per frame at every service, so the depth can be varied (1 to 6 levels) while every service keeps the same
arrival rate.

Panels:
 (a) the application, a tree of detectors (leaves) and transformation services (aggregators), here with 3 levels;
 (b) the merge rule at one aggregator, as a timeline of one frame;
 (c) the hierarchies that are run, each one separately;
 (d) the phases of one run, counted in frames.

Run from `notebooks/tsc/` with the venv. Writes figures/tsc/b3_experiment_setup.pdf and .jpg.

Source of the data: none. The figure is a schematic; every number in it follows from the parameters below, which are
the ones of the section "Proposed experiment for B3" in new_experiments.md. The times of panel (b) are example values.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

plt.style.use('default')
plt.rcParams.update({'font.size': 12})

OUTPUT = Path(__file__).resolve().parents[2] / "figures" / "tsc"

# Parameters of the experiment, as in new_experiments.md
levels_required = [1, 2, 3, 4]                       # numbers of levels that must be run
levels_optional = [5, 6]                             # numbers of levels that are run only if the testbed has the cores
levels_shown = 3                                     # number of levels of the tree in panel (a)
utilization = 0.5                                    # arrival rate x mean processing time of the slowest service
slowest_processing = (0.010, 0.040)                  # s, range of the mean processing time of the slowest service of a run
calibration_frames, calibration_rate = 200, 2.0      # frames, frames/s
warmup_frames, measured_frames, n_batches = 500, 5000, 10
quantile_label = "p95"                               # the ground truth is the 95th percentile of the end-to-end latency

# Example times of panel (b), in ms since the arrival of frame k. The two children release their results at different
# times, and the aggregator is still busy with frame k-1 when the second result arrives, so all three phases appear.
child_processing = {"child 1:\nobject detector": (6, 20), "child 2:\nface detector": (2, 12)}   # (start, end) of the processing of frame k
previous_frame = (14, 26)                            # (start, end) of the processing of frame k-1 at the aggregator
aggregator_processing = 10                           # ms, processing time of frame k at the aggregator

# One color and one marker per kind of service, in all panels
KINDS = {"object": dict(color="tab:blue", marker="o", label="object detector (leaf)"),
         "face": dict(color="tab:orange", marker="s", label="face detector (leaf)"),
         "aggregator": dict(color="tab:green", marker="D", label="transformation service (aggregator)")}
leaf_kind = lambda index: "object" if index % 2 == 0 else "face"   # the leaves alternate between the two detectors


def balanced_tree(levels: int) -> tuple[dict, list]:
    """Nodes {(level, index): kind} and edges [(child, parent)] of a tree where every aggregator has two children; level 1 is the root."""
    nodes = {(level, i): "aggregator" if level < levels else leaf_kind(i) for level in range(1, levels + 1) for i in range(2 ** (level - 1))}
    edges = [((level + 1, 2 * i + side), (level, i)) for level in range(1, levels) for i in range(2 ** (level - 1)) for side in (0, 1)]
    return nodes, edges


def draw_balanced(ax, levels: int, left: float, width: float) -> None:
    """Small balanced tree with the root on top, between x = left and x = left + width."""
    nodes, edges = balanced_tree(levels)
    position = {(level, i): (left + width * (i + 0.5) / 2 ** (level - 1), 1 - level) for level, i in nodes}
    draw_small(ax, nodes, edges, position)


def draw_narrow(ax, levels: int, left: float, step: float) -> None:
    """Small narrow tree: at every level the aggregator joins the result of the level below with one new leaf."""
    nodes, edges, position = {}, [], {}
    for level in range(1, levels):
        spine, leaf = ("spine", level), ("leaf", level)
        nodes[spine], nodes[leaf] = "aggregator", leaf_kind(level - 1)
        position[spine] = (left + level * step, 1 - level)
        position[leaf] = (left + (level - 0.5) * step, -level)
        edges.append((leaf, spine))
        edges.append((("spine", level + 1), spine))
    last = ("spine", levels)                         # the lowest level has no aggregator below it: its node is a leaf
    nodes[last], position[last] = leaf_kind(levels - 1), (left + (levels - 0.5) * step + step / 2, 1 - levels)
    draw_small(ax, nodes, edges, position)


def draw_small(ax, nodes: dict, edges: list, position: dict) -> None:
    for child, parent in edges:
        ax.plot(*zip(position[child], position[parent]), color="0.4", lw=1.0, zorder=1)
    for kind, style in KINDS.items():
        points = [position[node] for node, node_kind in nodes.items() if node_kind == kind]
        if points:
            ax.plot(*zip(*points), ls="none", marker=style["marker"], ms=5.5, color=style["color"], mec="black", mew=0.5, zorder=2)


fig = plt.figure(figsize=(15, 13))
# One grid per row, so that the legend below panel (b) has its own space above panel (c)
top = fig.add_gridspec(1, 2, left=0.02, right=0.99, top=0.94, bottom=0.63, width_ratios=[1.45, 1], wspace=0.3)
ax_tree, ax_merge = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
ax_sizes = fig.add_subplot(fig.add_gridspec(1, 1, left=0.02, right=0.99, top=0.455, bottom=0.2)[0, 0])
ax_run = fig.add_subplot(fig.add_gridspec(1, 1, left=0.02, right=0.99, top=0.15, bottom=0.03)[0, 0])

# --- (a) the application ---------------------------------------------------------
nodes, edges = balanced_tree(levels_shown)
box_width, box_height, between = {"leaf": 1.0, "aggregator": 1.6}, 0.17, 0.45   # widths in x units; one x unit is about one inch
width_of = lambda node: box_width["aggregator" if nodes[node] == "aggregator" else "leaf"]
x_source = 0.7
column_x = {levels_shown: x_source + between + box_width["leaf"] / 2}
for level in range(levels_shown - 1, 0, -1):
    column_x[level] = column_x[level + 1] + width_of((level + 1, 0)) / 2 + between + box_width["aggregator"] / 2
position = {(level, i): (column_x[level], 1 - (i + 0.5) / 2 ** (level - 1)) for level, i in nodes}
box_text = {"object": "object\ndetector", "face": "face\ndetector", "aggregator": "transformation"}
arrow = dict(arrowstyle="-|>", color="0.25", lw=1.5, shrinkA=0, shrinkB=0)
for node, kind in nodes.items():
    (x, y), width = position[node], width_of(node)
    ax_tree.add_patch(FancyBboxPatch((x - width / 2, y - box_height / 2), width, box_height, boxstyle="round,pad=0,rounding_size=0.03",
                                     fc=KINDS[kind]["color"], alpha=0.25, ec="none"))
    ax_tree.add_patch(FancyBboxPatch((x - width / 2, y - box_height / 2), width, box_height, boxstyle="round,pad=0,rounding_size=0.03",
                                     fc="none", ec=KINDS[kind]["color"], lw=1.8))
    ax_tree.text(x, y, box_text[kind], ha="center", va="center")
for child, parent in edges:
    (x_child, y_child), (x_parent, y_parent) = position[child], position[parent]
    ax_tree.annotate("", (x_parent - width_of(parent) / 2, y_parent), (x_child + width_of(child) / 2, y_child), arrowprops=arrow)
x_leaf, x_root = column_x[levels_shown] - box_width["leaf"] / 2, column_x[1] + box_width["aggregator"] / 2
ax_tree.text(x_source - 0.08, 0.5, "frames", ha="right", va="center")
for (level, i), (x, y) in position.items():
    if level == levels_shown:                        # an arriving frame is sent to all leaves at the same moment
        ax_tree.annotate("", (x_leaf, y), (x_source, 0.5), arrowprops=arrow)
ax_tree.annotate("", (x_root + 0.35, 0.5), (x_root, 0.5), arrowprops=arrow)
ax_tree.text(x_root + 0.4, 0.5, "result", ha="left", va="center")
headers = {levels_shown: "leaves (map)", 1: "root (reduce)"}
for level in range(1, levels_shown + 1):
    ax_tree.text(column_x[level], 1.06, f"level {level}\n{headers.get(level, 'aggregators (reduce)')}", ha="center", va="bottom")
ax_tree.annotate("", (x_root + 0.35, -0.07), (x_source, -0.07), arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.2, shrinkA=0, shrinkB=0))
ax_tree.text((x_source + x_root + 0.35) / 2, -0.11, "end-to-end latency of a frame:\nfrom its arrival until the root releases its result\n"
             "(every service is one FIFO queue with one worker)", ha="center", va="top")
ax_tree.set_xlim(0, x_root + 1.05)
ax_tree.set_ylim(-0.42, 1.2)
ax_tree.axis("off")
ax_tree.set_title(f"(a) Application: every frame is sent to all leaves\n(here {levels_shown} levels, {len(nodes)} services)", pad=10)

# --- (b) the merge rule ----------------------------------------------------------
complete = max(end for _, end in child_processing.values())       # frame k is one item as soon as both results are there
first_result = min(end for _, end in child_processing.values())
start = max(complete, previous_frame[1])                          # FIFO: frame k starts when the worker has finished frame k-1
end = start + aggregator_processing
rows = [*child_processing, "frame k at\nthe aggregator", "worker of\nthe aggregator"]
height = 0.56
row_y = {name: len(rows) - 1 - r for r, name in enumerate(rows)}
for (name, (begin, finish)), kind in zip(child_processing.items(), ("object", "face")):
    ax_merge.broken_barh([(begin, finish - begin)], (row_y[name] - height / 2, height), fc=KINDS[kind]["color"], ec="black", alpha=0.75)
    ax_merge.text((begin + finish) / 2, row_y[name], "frame k", ha="center", va="center")
    ax_merge.annotate("", (finish, row_y[rows[2]] + height / 2), (finish, row_y[name] - height / 2), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
phases = [("waiting for the result of the other child", first_result, complete, dict(fc="white", hatch="///")),
          ("waiting in the queue of the aggregator", complete, start, dict(fc="0.8", hatch="..")),
          ("processing", start, end, dict(fc=KINDS["aggregator"]["color"], alpha=0.75))]
for label, begin, finish, style in phases:
    ax_merge.broken_barh([(begin, finish - begin)], (row_y[rows[2]] - height / 2, height), ec="black", **style)
for label, (begin, finish), color in (("frame k−1", previous_frame, "0.6"), ("frame k", (start, end), KINDS["aggregator"]["color"])):
    ax_merge.broken_barh([(begin, finish - begin)], (row_y[rows[3]] - height / 2, height), fc=color, ec="black", alpha=0.75)
    ax_merge.text((begin + finish) / 2, row_y[rows[3]], label, ha="center", va="center")
for time, label in ((complete, "complete"), (start, "start"), (end, "end")):
    ax_merge.plot([time, time], [-0.5, len(rows) - 0.42], color="black", lw=1.0, ls=":")
    ax_merge.text(time, len(rows) - 0.38, label, ha="center", va="bottom")
ax_merge.set_yticks([row_y[name] for name in rows], rows)
ax_merge.set_ylim(-0.5, len(rows) - 0.05)
ax_merge.set_xlim(0, end + 3)
ax_merge.set_xlabel("time since the arrival of frame k (ms), example values")
ax_merge.spines[["top", "right"]].set_visible(False)
ax_merge.legend(handles=[Patch(ec="black", **style) for _, _, _, style in phases] + [Line2D([], [], color="black", lw=1.0, ls=":")],
                labels=[label for label, *_ in phases] + ["the three times stored per frame and service"],
                loc="upper center", bbox_to_anchor=(0.4, -0.17), frameon=False)
ax_merge.set_title("(b) Merge rule: an aggregator waits for both children,\nthen processes the frame as one item", pad=10)

# --- (c) the hierarchies ---------------------------------------------------------
gap, narrow_step = 0.9, 0.42
tree_width = {1: 1.8, 2: 1.8, 3: 2.2, 4: 3.2, 5: 4.8, 6: 7.6}
label_y, group_y = -max(levels_optional) + 0.3, 1.5
left, spans = 0.0, {}
for levels in levels_required + levels_optional:
    draw_balanced(ax_sizes, levels, left, tree_width[levels])
    ax_sizes.text(left + tree_width[levels] / 2, label_y, f"{levels} level{'s' if levels > 1 else ''}\n{2 ** levels - 1} service{'s' if levels > 1 else ''}", ha="center", va="top")
    spans[levels] = (left, left + tree_width[levels])
    left += tree_width[levels] + gap
narrow_left = left
for levels in levels_optional:
    width = (levels + 0.5) * narrow_step
    draw_narrow(ax_sizes, levels, left, narrow_step)
    ax_sizes.text(left + width / 2 + narrow_step / 2, label_y, f"{levels} levels\n{2 * levels - 1} services", ha="center", va="top")
    left += width + gap
groups = [("required", spans[levels_required[0]][0], spans[levels_required[-1]][1]),
          ("only if the testbed has the cores", spans[levels_optional[0]][0], spans[levels_optional[-1]][1]),
          ("otherwise: narrow trees\nof the same depth", narrow_left, left - gap + narrow_step)]
for label, begin, finish in groups:
    ax_sizes.plot([begin, finish], [group_y, group_y], color="black", lw=1.0, marker="|", ms=8)
    ax_sizes.text((begin + finish) / 2, group_y + 0.15, label, ha="center", va="bottom")
ax_sizes.legend(handles=[Line2D([], [], ls="none", marker=style["marker"], ms=8, color=style["color"], mec="black", mew=0.5) for style in KINDS.values()],
                labels=[style["label"] for style in KINDS.values()], loc="lower left", bbox_to_anchor=(0.0, 0.2), frameon=False)
ax_sizes.set_xlim(-0.4, left - gap + narrow_step + 0.4)
ax_sizes.set_ylim(label_y - 1.6, group_y + 1.6)
ax_sizes.axis("off")
ax_sizes.set_title("(c) Hierarchies: every tree is run separately, because every tree needs its own ground truth", pad=4)

# --- (d) one run -----------------------------------------------------------------
batch = measured_frames // n_batches
rates = [utilization / processing for processing in slowest_processing]   # frames/s, highest rate first
ax_run.broken_barh([(0, calibration_frames)], (0, 1), fc="white", ec="black", hatch="///")
ax_run.broken_barh([(calibration_frames, warmup_frames)], (0, 1), fc="0.8", ec="black")
for b in range(n_batches):
    begin = calibration_frames + warmup_frames + b * batch
    ax_run.broken_barh([(begin, batch)], (0, 1), fc="tab:purple", alpha=0.75 if b % 2 == 0 else 0.45, ec="black")
    ax_run.text(begin + batch / 2, 0.5, str(b + 1), ha="center", va="center")
pointer = dict(arrowstyle="-", color="black", lw=1.0)
ax_run.annotate(f"calibration: {calibration_frames} frames at {calibration_rate:g} frames/s (nothing queues);\n"
                f"measures the mean processing times, sets the arrival rate",
                (calibration_frames / 2, 1), (0, 1.35), ha="left", va="bottom", arrowprops=pointer)
ax_run.annotate(f"warm-up: {warmup_frames} frames, not measured,\nso that the queues fill to their usual level",
                (calibration_frames + warmup_frames / 2, 0), (calibration_frames, -0.35), ha="left", va="top", arrowprops=pointer)
measured_start = calibration_frames + warmup_frames
ax_run.text(measured_start + measured_frames, 1.35, f"measurement: {measured_frames} frames = {n_batches} consecutive batches of {batch} frames;\n"
            f"ground truth: {quantile_label} of the end-to-end latency, with a confidence interval", ha="right", va="bottom")
ax_run.text(measured_start + measured_frames, -0.35, f"arrival rate (Poisson) = {utilization:g} ÷ mean processing time of the slowest service\n"
            f"= {min(rates):g} to {max(rates):g} frames/s, so the measurement takes {measured_frames / min(rates):g} s to {measured_frames / max(rates):g} s", ha="right", va="top")
ax_run.set_xlim(-20, measured_start + measured_frames + 20)
ax_run.set_ylim(-1.9, 2.6)
ax_run.axis("off")
ax_run.set_title("(d) One run, to scale in frames: the length of a run is a number of frames, not a duration", pad=4)

OUTPUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT / "b3_experiment_setup.pdf", bbox_inches='tight')
fig.savefig(OUTPUT / "b3_experiment_setup.jpg", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
