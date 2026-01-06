"""Floyd-Warshall algorithm for transitive closure computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def floyd_warshall_transitive_closure(
    adjacency: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """
    Compute transitive closure using vectorized Floyd-Warshall algorithm.

    For revealed preference analysis, this computes R* (indirect revealed
    preference) from R (direct revealed preference). If R[i,j] means
    "bundle i is revealed preferred to bundle j", then R*[i,j] means
    "bundle i is revealed preferred to bundle j through a chain of
    direct preferences".

    The algorithm iteratively checks if path i -> k -> j exists for each k,
    using NumPy broadcasting for vectorization.

    Args:
        adjacency: T x T boolean adjacency matrix where adjacency[i,j] = True
            means there is a direct edge from i to j

    Returns:
        T x T boolean transitive closure matrix where result[i,j] = True
        means there exists a path from i to j (direct or indirect)

    Complexity:
        Time: O(T^3)
        Space: O(T^2)

    Example:
        >>> import numpy as np
        >>> # A -> B -> C
        >>> adj = np.array([
        ...     [False, True, False],
        ...     [False, False, True],
        ...     [False, False, False]
        ... ])
        >>> closure = floyd_warshall_transitive_closure(adj)
        >>> closure[0, 2]  # A reaches C through B
        True
    """
    T = adjacency.shape[0]
    closure = adjacency.copy()

    # Ensure reflexivity (every node reaches itself)
    np.fill_diagonal(closure, True)

    # Floyd-Warshall: for each intermediate vertex k
    # closure[i,j] = closure[i,j] OR (closure[i,k] AND closure[k,j])
    for k in range(T):
        # Vectorized update using broadcasting
        # closure[:, k:k+1] is column k (shape T x 1)
        # closure[k:k+1, :] is row k (shape 1 x T)
        # Their AND produces T x T matrix of paths through k
        closure = closure | (closure[:, k : k + 1] & closure[k : k + 1, :])

    return closure


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
