"""Arbitrary classes of which instances can be interpreted
as :py:mod:`scipy` sparse matrices or 2D square :py:mod:`numpy` arrays
can be registered as abstract subclasses of :class:`GraphABC`.
This way all main classes/functions implemented in :mod:`pathcensus`
can automatically interpret them as graph-like objects allowing
seemless integration with many different data formats and third-party
packages such as :py:mod:`networkx` or :py:mod:`igraph`.

In order to register a class a function for converting its instances to
:py:class:`scipy.sparse.spmatrix` (CRS format) needs to be defined.
The conversion is handled by the :py:func:`pathcensus.graph.adjacency` function
which can be overloaded through the single dispatch mechanism. In particular,
it should be called on arrays/sparse matrices extracted from graph classes
to ensure standardized format. See the example below.

Graph classes defined by :py:mod:`networkx`, :py:mod:`igraph`
and :py:mod:`graph_tool` are registered automatically provided
the packages are installed.

Below is an example in which a custom conversion from a list of list
format is registered. Arguably, the below implementation is naive and
handles the conversion by simply converting to a :py:mod:`numpy` array,
without checking wheter the array is really 2D and square, but it illustrates
the main logic of registering custom graph-like classes.

.. doctest:: graph-abc

    >>> import numpy as np
    >>> from pathcensus import GraphABC, adjacency, PathCensus

    >>> def _adj_list(graph: list) -> spmatrix:
    ...     \"\"\"Adjacency matrix from list of lists.\"\"\"
    ...     A = np.array(graph)
    ...     return adjacency(A)

    >>> GraphABC.register_type(list, _adj_list)
    >>> # A triangle graph
    >>> A = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    >>> # Calculate path census
    >>> paths = PathCensus(A)
    >>> paths.census("global")
       t  tw  th  q0  qw  qh
    0  1   3   3   0   0   0
"""
from abc import ABC
from typing import Callable, Any
from functools import singledispatch
import numpy as np
from scipy.sparse import spmatrix, csr_matrix


class GraphABC(ABC):
    """Abstract Base Class (ABC) for registering different graphs classes.

    Any kind of graph object from different libraries can be registered
    as a subclass of ABC as long as also a function for converting it into
    a sparse adjacency matrix is provided at the registration time.
    In particular, :py:class:`scipy.sparse.spmatrix` objects are
    automatically recognized as graph-like objects.

    This allows all graph-based functions/methods/class defined in
    :py:mod:`pathcensus` to operate flexibly on any sort of graph-like
    objects / graph implementations.

    Provided the packages are installed methods for handling graph objects
    from :py:mod:`networkx`, :py:mod:`igraph` and :py:mod:`graph_tool`
    are automatically registered.

    See class method :py:meth:`register_graph` for more info.
    """
    @classmethod
    def register_type(
        cls,
        subclass: type,
        adj: Callable[..., spmatrix]
    ) -> None:
        """Register type as a subclass of :py:class:`pathcensus.graph.GraphABC`.

        Parameters
        ----------
        subclass
            Any graph-like class.
        adj
            Function for converting `subclass` graphs to sparse adjacency matrices.
            It should use the following signature `(graph, **kwds) -> spmatrix`.
            The return matrix must be in the format in which `(i, j)` indicate
            an edge from `i` to `j` (in the case a network is directed).
            Using ``**kwds`` is optional and in general it is best to
            implement `adj` in such a way that using ``**kwds`` is not necessary,
            in particular for detecting whether a graph is weighted and
            converting it to a weighted adjacency matrix if necessary.
            This way :py:mod:`pathcensus` will be able automatically choose
            to use weighted methods for weighted graphs.
        """
        cls.register(subclass)
        adjacency.register(adj)


@singledispatch
def adjacency(graph: GraphABC) -> spmatrix:
    """Get (unweighted) adjacency matrix of a graph."""
    raise TypeError(f"cannot handle '{type(graph)}' object")


# Register 2D square numpy arrays as graph-like class -------------------------

def _adj_numpy(graph: np.ndarray) -> spmatrix:
    """Convert 2D square array to sparse matrix."""
    if graph.ndim != 2:
        raise AttributeError("only 2D arrays are accepted")
    if graph.shape[0] != graph.shape[1]:
        raise AttributeError("array is not square")
    i, j = graph.nonzero()
    data = graph[i, j]
    adj  = csr_matrix((data, (i, j)), shape=graph.shape, dtype=graph.dtype)
    return adj

GraphABC.register_type(spmatrix, _adj_numpy)

# Register sparse matrices as graph-like class --------------------------------

def _adj_spmat(graph: spmatrix) -> spmatrix:
    """Adjacency matrix from sparse matrix.

    It just converts it to CSR format and ensures that no zeros
    are represented explicitly.
    """
    graph = graph.tocsr()
    graph.eliminate_zeros()
    return graph

GraphABC.register_type(spmatrix, _adj_spmat)


# Register networkx networks as graph-like class ------------------------------
try:
    import networkx as nx     # tyoe: ignore
    def _adj_nx(graph: nx.Graph, **kwds: Any) -> spmatrix:
        """Adjacency matrix from :py:class:`networkx.Graph` object."""
        adj = nx.convert_matrix.to_scipy_sparse_matrix(graph, **kwds)
        return adjacency(adj)
    # Register as GraphABC subclass
    GraphABC.register_type(nx.Graph, _adj_nx)
except ModuleNotFoundError:
    pass


# Register igraph networks as graph-like class --------------------------------
try:
    import igraph as ig    # type: ignore
    def _adj_ig(graph: ig.Graph, **kwds: Any) -> spmatrix:
        """Adjacency matrix from :py:class:`igraph.Graph` object."""
        if graph.is_weighted():
            attribute = "weight"
        else:
            attribute = None
        kwds = { "attribute": attribute, **kwds }
        adj = graph.get_adjacency_sparse(**kwds)

        if kwds["attribute"] is None:
            adj.data[:] = 1

        return adjacency(adj)
    # Register as GraphABC subclass
    GraphABC.register_type(ig.Graph, _adj_ig)
except ModuleNotFoundError:
    pass


# Register graph_tool networks as graph-like class ----------------------------
try:
    # pylint: disable=import-error
    import graph_tool.all as gt   # type: ignore
    def _adj_gt(graph: gt.Graph, **kwds: Any) -> spmatrix:
        """Adjacency matrix from :py:class:`graph_tool.Graph` object."""
        if "weight" in graph.edge_properties:
            weight = graph.edge_properties["weight"]
        else:
            weight = None
        kwds = { "weight": weight, **kwds }
        adj = gt.adjacency(graph, **kwds).T

        if kwds["weight"] is None:
            adj.data[:] = 1

        return adjacency(adj)
    # Register as GraphABC subclass
    GraphABC.register_type(gt.Graph, _adj_gt)
except ModuleNotFoundError:
    pass
