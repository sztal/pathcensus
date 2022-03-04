"""Test utilities."""
from typing import Union, Any
import numpy as np
from scipy.sparse import spmatrix, isspmatrix
import igraph as ig


def rowsums(X: Union[np.ndarray, spmatrix]) -> np.ndarray:
    """Calculate row sums of a matrix."""
    if isspmatrix(X):
        return np.array(X.sum(1)).flatten()
    return X.sum(1)

def get_largest_component(graph: ig.Graph, **kwds: Any) -> ig.Graph:
    """Get largest component of a graph.

    ``**kwds`` are passed to :py:meth:`igraph.Graph.components`.
    """
    vids = None
    for component in graph.components(**kwds):
        if vids is None or len(component) > len(vids):
            vids = component
    return graph.induced_subgraph(vids)

# Random graphs ---------------------------------------------------------------

def make_er_graph(n, dbar):
    """Make ER random graph with given average degree."""
    p = dbar / (n-1)
    return ig.Graph.Erdos_Renyi(n, p=p, directed=False)

def make_rgg(n, dbar):
    """Make random geometric graph with given average degree."""
    radius = np.sqrt(dbar/(np.pi*(n-1)))
    return ig.Graph.GRG(n, radius=radius, torus=True)

def add_random_weights(graph, m0=1, m1=10):
    """Add random integer weights between ``m0`` and ``m1``
    to a :py:class:`igraph.Graph` instance.
    """
    graph = graph.copy()
    graph.es["weight"] = np.random.randint(m0, m1, (graph.ecount(),))
    return graph

# Motifs ----------------------------------------------------------------------

def make_triangle():
    """Make a simple triangle graph (undirected)."""
    G = ig.Graph(directed=False)
    G.add_vertices(3)
    G.add_edges([(0, 1), (1, 2), (2, 0)])
    return G

def make_quadrangle(weak=0):
    """Make a simple quadrangle graph (undirected)
    with the number of chords equal to ``weak``.
    """
    G = ig.Graph(directed=False)
    G.add_vertices(4)
    G.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    if weak >= 1:
        G.add_edge(0, 2)
    if weak == 2:
        G.add_edge(1, 3)
    return G
