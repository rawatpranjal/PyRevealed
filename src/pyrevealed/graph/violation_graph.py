"""ViolationGraph: NetworkX-based visualization of GARP violations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import networkx as nx
    from pyrevealed.core.result import GARPResult
    from pyrevealed.core.session import ConsumerSession


class ViolationGraph:
    """
    NetworkX-based graph for visualizing revealed preference relations and violations.

    Nodes represent observations (bundles chosen at specific prices).
    Edges represent revealed preference relations:
    - 'weak' edges: direct revealed preference (R): p_i @ x_i >= p_i @ x_j
    - 'strict' edges: strict revealed preference (P): p_i @ x_i > p_i @ x_j

    GARP violations are cycles containing at least one strict preference edge.

    Example:
        >>> from pyrevealed import ConsumerSession, check_garp
        >>> from pyrevealed.graph import ViolationGraph
        >>> session = ConsumerSession(prices, quantities)
        >>> result = check_garp(session)
        >>> graph = ViolationGraph(session, result)
        >>> fig, ax = graph.plot(highlight_violations=True)
    """

    def __init__(
        self, session: ConsumerSession, garp_result: GARPResult
    ) -> None:
        """
        Initialize ViolationGraph from session and GARP result.

        Args:
            session: ConsumerSession with the choice data
            garp_result: Result from check_garp containing preference matrices
        """
        self.session = session
        self.garp_result = garp_result
        self._graph: nx.DiGraph | None = None

    @property
    def graph(self) -> nx.DiGraph:
        """Lazily build and return the NetworkX graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph from preference matrices."""
        import networkx as nx

        G = nx.DiGraph()
        T = self.session.num_observations

        # Add nodes with attributes
        for i in range(T):
            G.add_node(
                i,
                bundle=self.session.quantities[i].tolist(),
                prices=self.session.prices[i].tolist(),
                expenditure=float(self.session.own_expenditures[i]),
                label=f"Obs {i}",
            )

        # Add edges for revealed preferences
        R = self.garp_result.direct_revealed_preference
        P = self.garp_result.strict_revealed_preference

        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                if R[i, j]:
                    edge_type = "strict" if P[i, j] else "weak"
                    G.add_edge(i, j, relation=edge_type)

        return G

    def get_violation_subgraph(self) -> nx.DiGraph:
        """
        Extract subgraph containing only nodes involved in violation cycles.

        Returns:
            NetworkX DiGraph with only violation-related nodes and edges
        """
        violation_nodes: set[int] = set()
        for cycle in self.garp_result.violations:
            violation_nodes.update(cycle)

        return self.graph.subgraph(violation_nodes).copy()

    def to_adjacency_matrix(self) -> NDArray[np.bool_]:
        """
        Return adjacency matrix of the preference graph.

        Returns:
            T x T boolean matrix where result[i,j] = True if edge i->j exists
        """
        import networkx as nx

        return nx.to_numpy_array(self.graph, dtype=bool)

    def find_all_cycles(self) -> list[list[int]]:
        """
        Find all simple cycles in the preference graph.

        Returns:
            List of cycles, where each cycle is a list of node indices
        """
        import networkx as nx

        return list(nx.simple_cycles(self.graph))

    def get_edge_list(self) -> list[tuple[int, int, str]]:
        """
        Get list of all edges with their types.

        Returns:
            List of (source, target, relation_type) tuples
        """
        return [
            (u, v, d["relation"])
            for u, v, d in self.graph.edges(data=True)
        ]

    def plot(
        self,
        figsize: tuple[int, int] = (10, 8),
        highlight_violations: bool = True,
        show_edge_labels: bool = False,
        layout: str = "spring",
        ax: Any = None,
    ) -> tuple[Any, Any]:
        """
        Plot the violation graph using matplotlib.

        Args:
            figsize: Figure size as (width, height)
            highlight_violations: Color violation cycle nodes in red
            show_edge_labels: Show edge relation types ('weak'/'strict')
            layout: Graph layout algorithm:
                - 'spring': Force-directed layout (default)
                - 'circular': Nodes arranged in a circle
                - 'kamada_kawai': Kamada-Kawai layout
            ax: Optional matplotlib axes to draw on

        Returns:
            Tuple of (figure, axes) matplotlib objects
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, seed=42)

        # Determine node colors based on violations
        violation_nodes: set[int] = set()
        if highlight_violations:
            for cycle in self.garp_result.violations:
                violation_nodes.update(cycle)

        node_colors = [
            "salmon" if i in violation_nodes else "lightblue"
            for i in self.graph.nodes()
        ]

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=700,
            ax=ax,
        )

        # Separate edges by type
        weak_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("relation") == "weak"
        ]
        strict_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("relation") == "strict"
        ]

        # Draw weak preference edges (gray, thin)
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=weak_edges,
            style="solid",
            alpha=0.4,
            edge_color="gray",
            arrows=True,
            arrowsize=15,
            ax=ax,
        )

        # Draw strict preference edges (red, thick)
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=strict_edges,
            style="solid",
            edge_color="darkred",
            width=2,
            arrows=True,
            arrowsize=20,
            ax=ax,
        )

        # Draw labels
        labels = {i: f"{i}" for i in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, ax=ax, font_size=10)

        # Optional edge labels
        if show_edge_labels:
            edge_labels = {
                (u, v): d["relation"][0].upper()  # 'W' or 'S'
                for u, v, d in self.graph.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                self.graph, pos, edge_labels, ax=ax, font_size=8
            )

        title = "Revealed Preference Graph"
        if not self.garp_result.is_consistent:
            title += f" ({len(self.garp_result.violations)} violations)"
        ax.set_title(title)
        ax.axis("off")

        return fig, ax
