"""Transitive closure computation with SCC-based optimization.

Primary algorithm: SCC decomposition + per-component Floyd-Warshall + DAG
reachability propagation. Falls back to full Floyd-Warshall for tiny matrices
or single-SCC graphs. Produces EXACT same results as full Floyd-Warshall.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from prefgraph._kernels import floyd_warshall_tc_numba, floyd_warshall_tc_serial

# Threshold below which we skip SCC overhead
_SCC_THRESHOLD = 10

# Threshold for switching to parallel Floyd-Warshall within an SCC
_PARALLEL_THRESHOLD = 500


def floyd_warshall_transitive_closure(
    adjacency: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """
    Compute transitive closure of a boolean adjacency matrix.

    Uses SCC decomposition for graphs with T >= 10: decomposes into
    strongly connected components, runs Floyd-Warshall only within each
    SCC, then propagates reachability across the condensed DAG. This is
    exact (not an approximation) and provides 10-10000x speedup on
    typical revealed preference data where most SCCs are small.

    Falls back to direct Floyd-Warshall for T < 10 or single-SCC graphs.

    Args:
        adjacency: T x T boolean adjacency matrix where adjacency[i,j] = True
            means there is a direct edge from i to j

    Returns:
        T x T boolean transitive closure matrix where result[i,j] = True
        means there exists a path from i to j (direct or indirect)

    Example:
        >>> import numpy as np
        >>> adj = np.array([
        ...     [False, True, False],
        ...     [False, False, True],
        ...     [False, False, False]
        ... ])
        >>> closure = floyd_warshall_transitive_closure(adj)
        >>> closure[0, 2]  # A reaches C through B
        True
    """
    adjacency_c = np.ascontiguousarray(adjacency, dtype=np.bool_)
    T = adjacency_c.shape[0]

    if T < _SCC_THRESHOLD:
        return floyd_warshall_tc_serial(adjacency_c)

    return scc_transitive_closure(adjacency_c)


def scc_transitive_closure(
    adjacency: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """
    Compute transitive closure using SCC decomposition.

    Algorithm:
    1. Find SCCs via scipy.sparse.csgraph (O(V+E))
    2. For each SCC with size > 1, run Floyd-Warshall on the subgraph
    3. Build condensed DAG and topological sort it
    4. Propagate reachability across SCCs in reverse topological order

    Produces identical output to full Floyd-Warshall.

    Args:
        adjacency: T x T boolean adjacency matrix

    Returns:
        T x T boolean transitive closure matrix
    """
    from prefgraph.graph.scc import (
        find_sccs,
        build_condensed_dag,
        topological_order_dag,
    )

    T = adjacency.shape[0]
    n_components, labels = find_sccs(adjacency)

    # Single SCC - fall back to full Floyd-Warshall
    if n_components == 1:
        if T < _PARALLEL_THRESHOLD:
            return floyd_warshall_tc_serial(adjacency)
        else:
            return floyd_warshall_tc_numba(adjacency)

    # Initialize closure with original edges + self-loops
    closure = adjacency.copy()
    np.fill_diagonal(closure, True)

    # Group nodes by SCC
    scc_nodes: list[NDArray[np.intp]] = []
    for c in range(n_components):
        scc_nodes.append(np.where(labels == c)[0])

    # Step 1: Compute TC within each non-trivial SCC
    for c in range(n_components):
        nodes = scc_nodes[c]
        if len(nodes) <= 1:
            continue

        # Extract subgraph and compute its TC
        sub_adj = np.ascontiguousarray(adjacency[np.ix_(nodes, nodes)], dtype=np.bool_)
        sub_tc = floyd_warshall_tc_serial(sub_adj)

        # Write back into closure
        closure[np.ix_(nodes, nodes)] = sub_tc

    # Step 2: Build condensed DAG and topological sort
    dag = build_condensed_dag(adjacency, labels, n_components)
    topo_order = topological_order_dag(dag)

    # Step 3: Propagate reachability in reverse topological order
    # For each SCC, compute the full set of reachable nodes (as a boolean row)
    # reachable[c] = all nodes reachable from any node in SCC c
    reachable = [np.zeros(T, dtype=np.bool_) for _ in range(n_components)]

    # Initialize: each SCC can reach its own nodes (already have internal TC)
    for c in range(n_components):
        nodes = scc_nodes[c]
        # Reachable from this SCC = union of all rows in closure for nodes in this SCC
        for i in nodes:
            reachable[c] |= closure[i]

    # Process in reverse topological order
    for c in reversed(topo_order):
        # For each DAG successor of c, merge their reachable sets
        for succ in range(n_components):
            if dag[c, succ]:
                reachable[c] |= reachable[succ]

        # Write back: all nodes in SCC c can reach everything in reachable[c]
        nodes = scc_nodes[c]
        for i in nodes:
            closure[i] |= reachable[c]

    return closure


def _floyd_warshall_direct(
    adjacency: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Direct Floyd-Warshall without SCC optimization (for testing/comparison)."""
    adjacency_c = np.ascontiguousarray(adjacency, dtype=np.bool_)
    T = adjacency_c.shape[0]
    if T < _PARALLEL_THRESHOLD:
        return floyd_warshall_tc_serial(adjacency_c)
    else:
        return floyd_warshall_tc_numba(adjacency_c)


def floyd_warshall_with_path_reconstruction(
    adjacency: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """
    Compute transitive closure with path reconstruction capability.

    In addition to the closure matrix, returns a "next" matrix that
    allows reconstructing the shortest path between any two nodes.

    Args:
        adjacency: T x T boolean adjacency matrix

    Returns:
        Tuple of:
        - closure: T x T boolean transitive closure matrix
        - next_node: T x T matrix where next_node[i,j] is the next node
          on the path from i to j (-1 if no path exists)

    Example:
        >>> closure, next_node = floyd_warshall_with_path_reconstruction(adj)
        >>> # Reconstruct path from 0 to 2
        >>> path = [0]
        >>> while path[-1] != 2:
        ...     path.append(next_node[path[-1], 2])
    """
    T = adjacency.shape[0]
    closure = adjacency.copy()

    # Initialize next_node matrix
    # next_node[i,j] = j if direct edge exists, -1 otherwise
    next_node = np.full((T, T), -1, dtype=np.int64)
    for i in range(T):
        for j in range(T):
            if adjacency[i, j]:
                next_node[i, j] = j

    # Ensure reflexivity
    np.fill_diagonal(closure, True)
    for i in range(T):
        next_node[i, i] = i

    # Floyd-Warshall with path tracking
    for k in range(T):
        for i in range(T):
            for j in range(T):
                if not closure[i, j] and closure[i, k] and closure[k, j]:
                    closure[i, j] = True
                    next_node[i, j] = next_node[i, k]

    return closure, next_node


def reconstruct_path(
    next_node: NDArray[np.int64], start: int, end: int
) -> list[int] | None:
    """
    Reconstruct path from start to end using the next_node matrix.

    Args:
        next_node: Matrix from floyd_warshall_with_path_reconstruction
        start: Starting node index
        end: Ending node index

    Returns:
        List of node indices forming the path, or None if no path exists
    """
    if next_node[start, end] == -1:
        return None

    path = [start]
    current = start
    while current != end:
        current = next_node[current, end]
        if current == -1:
            return None
        path.append(current)
        if len(path) > next_node.shape[0]:
            # Safety check to prevent infinite loops
            return None

    return path
