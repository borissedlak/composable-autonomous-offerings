"""Fork-join trees of services (experiment B3 of new_experiments.md): topologies, their leaf-to-root paths, and the
maximum over these paths through which every composition of a chain extends to a tree.

A tree is given by `children`: one list per service with the indices of its two children (empty for a leaf); service
0 is the root. A frame enters every leaf at its arrival time, and an aggregator processes a frame once both children
have released it (simulate_tree in simulation.py). Every service then performs only maxima and sums of times, so the
end-to-end latency of a frame is the maximum, over the leaf-to-root paths, of its latency through the path run as an
isolated chain with the same arrivals and execution times (exact for services with one worker). A bound per path at
epsilon / P, with P paths, therefore bounds the tree at epsilon (union bound over the paths).
"""
import numpy as np


def balanced_tree(levels: int, leaf_services=("s1", "s2"), aggregator_service="s3") -> tuple:
    """Balanced tree with `levels` levels and 2^levels - 1 services in heap order (children of i: 2i + 1, 2i + 2).
    Returns children and the service of every position: the leaves alternate between leaf_services, every inner
    service is the aggregator."""
    n_services, first_leaf = 2 ** levels - 1, 2 ** (levels - 1) - 1
    children = [[2 * i + 1, 2 * i + 2] if i < first_leaf else [] for i in range(n_services)]
    services = [aggregator_service if i < first_leaf else leaf_services[(i - first_leaf) % len(leaf_services)] for i in range(n_services)]
    return children, services


def narrow_tree(levels: int, leaf_services=("s1", "s2"), aggregator_service="s3") -> tuple:
    """Narrow tree with `levels` levels and 2 levels - 1 services: aggregators 0 to levels - 2 form a chain from the
    root down, and every aggregator joins the one below it (two leaves at the bottom) with one new leaf. It has a
    path of `levels` services and `levels` paths in all."""
    if levels == 1:
        return [[]], [leaf_services[0]]
    n_aggregators = levels - 1
    leaves = list(range(n_aggregators, 2 * levels - 1))
    children = [[j + 1 if j < n_aggregators - 1 else leaves[-1], leaves[j]] for j in range(n_aggregators)] + [[] for _ in leaves]
    services = [aggregator_service] * n_aggregators + [leaf_services[k % len(leaf_services)] for k in range(len(leaves))]
    return children, services


def leaf_to_root_paths(children: list) -> list:
    """Every path from a leaf to the root, as a list of service positions starting at the leaf."""
    paths, stack = [], [(0, [0])]
    while stack:
        position, to_root = stack.pop()
        if not children[position]:
            paths.append(to_root[::-1])
        stack.extend((child, to_root + [child]) for child in children[position])
    return sorted(paths)


def children_first(children: list) -> list:
    """The positions in an order in which every service comes after its children."""
    order, stack = [], [(0, False)]
    while stack:
        position, expanded = stack.pop()
        if expanded:
            order.append(position)
        else:
            stack.append((position, True))
            stack.extend((child, False) for child in children[position])
    return order


def max_over_paths(paths: list, path_value, path_key=None) -> np.ndarray:
    """Maximum over the paths of path_value(path) (one value per configuration). Paths with the same path_key(path)
    are computed once (e.g. paths with the same services when all services have one worker)."""
    values = {}
    for path in paths:
        key = tuple(path) if path_key is None else path_key(path)
        if key not in values:
            values[key] = path_value(path)
    return np.max(list(values.values()), axis=0)


def max_path_sum(paths: list, step_value) -> np.ndarray:
    """Maximum over the paths of the sum of step_value(position, previous) along the path (previous: the position
    before it on the path, None for the leaf). Each (position, previous) pair is computed once: a tree with N services
    has N such pairs, however many paths share them."""
    steps = {}
    totals = []
    for path in paths:
        for previous, position in zip([None] + path[:-1], path):
            if (position, previous) not in steps:
                steps[(position, previous)] = step_value(position, previous)
        totals.append(sum(steps[(position, previous)] for previous, position in zip([None] + path[:-1], path)))
    return np.max(totals, axis=0)
