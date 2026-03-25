"""Strongly Connected Components and Feedback Arc/Vertex Set utilities.

Provides graph decomposition primitives used by the SCC-optimized
transitive closure and Houtman-Maks algorithms.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def find_sccs(
    adjacency: NDArray[np.bool_],
) -> tuple[int, NDArray[np.int32]]:
    """
    Find strongly connected components of a directed graph.

    Uses scipy.sparse.csgraph.connected_components with strong connection.

    Args:
        adjacency: T x T boolean adjacency matrix

    Returns:
        Tuple of (n_components, labels) where labels[i] is the SCC id for node i
    """
    sparse = csr_matrix(adjacency.astype(np.int8))
    n_components, labels = connected_components(
        sparse, directed=True, connection="strong"
    )
    return n_components, labels.astype(np.int32)


def build_condensed_dag(
    adjacency: NDArray[np.bool_],
    labels: NDArray[np.int32],
    n_components: int,
) -> NDArray[np.bool_]:
    """
    Build the condensed DAG from SCC labels.

    Each SCC becomes a single node. An edge exists from SCC c1 to SCC c2
    if any edge goes from a node in c1 to a node in c2 (c1 != c2).

    Args:
        adjacency: T x T boolean adjacency matrix
        labels: SCC label for each node
        n_components: Number of SCCs

    Returns:
        n_components x n_components boolean DAG adjacency matrix
    """
    dag = np.zeros((n_components, n_components), dtype=np.bool_)

    # Find cross-SCC edges using vectorized operations
    rows, cols = np.nonzero(adjacency)
    src_labels = labels[rows]
    dst_labels = labels[cols]
    cross_mask = src_labels != dst_labels
    dag[src_labels[cross_mask], dst_labels[cross_mask]] = True

    return dag


def topological_order_dag(
    dag: NDArray[np.bool_],
) -> list[int]:
    """
    Topological sort of a DAG using Kahn's algorithm.

    Args:
        dag: n x n boolean DAG adjacency matrix (must be acyclic)

    Returns:
        List of node indices in topological order
    """
    n = dag.shape[0]
    in_degree = dag.sum(axis=0).astype(np.int64)

    # Queue of nodes with no incoming edges
    queue: list[int] = []
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    order: list[int] = []
    head = 0

    while head < len(queue):
        node = queue[head]
        head += 1
        order.append(node)

        # Decrease in-degree of successors
        for j in range(n):
            if dag[node, j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

    return order


def greedy_feedback_vertex_set(
    adjacency: NDArray[np.bool_],
) -> list[int]:
    """
    Compute a greedy Feedback Vertex Set (approximate minimum).

    Repeatedly removes the node involved in the most cycles (highest
    out_degree - in_degree differential), breaking cycles until the
    graph is acyclic (no more SCCs of size > 1).

    This is a heuristic with good practical performance. The current
    Houtman-Maks implementation is also greedy, so quality is comparable.

    Args:
        adjacency: T x T boolean adjacency matrix

    Returns:
        List of node indices to remove to make the graph acyclic
    """
    T = adjacency.shape[0]
    if T <= 1:
        return []

    removed: list[int] = []
    active = np.ones(T, dtype=np.bool_)

    while True:
        # Find SCCs of the active subgraph
        active_nodes = np.where(active)[0]
        if len(active_nodes) < 2:
            break

        sub_adj = adjacency[np.ix_(active_nodes, active_nodes)]
        n_comp, sub_labels = find_sccs(sub_adj)

        # Find non-trivial SCCs (size > 1)
        scc_sizes = np.bincount(sub_labels, minlength=n_comp)
        has_nontrivial = np.any(scc_sizes > 1)

        if not has_nontrivial:
            break

        # Score each node in non-trivial SCCs by cycle participation
        # Use (out_degree + in_degree) within the SCC as proxy
        best_node_local = -1
        best_score = -1

        for c in range(n_comp):
            if scc_sizes[c] <= 1:
                continue

            scc_mask = sub_labels == c
            scc_local = np.where(scc_mask)[0]
            scc_sub = sub_adj[np.ix_(scc_local, scc_local)]

            out_deg = scc_sub.sum(axis=1)
            in_deg = scc_sub.sum(axis=0)
            scores = out_deg + in_deg

            local_best = int(np.argmax(scores))
            if scores[local_best] > best_score:
                best_score = int(scores[local_best])
                best_node_local = int(active_nodes[scc_local[local_best]])

        if best_node_local < 0:
            break

        removed.append(best_node_local)
        active[best_node_local] = False
        # Zero out the row and column in adjacency to avoid re-extraction overhead
        adjacency[best_node_local, :] = False
        adjacency[:, best_node_local] = False

    return removed
