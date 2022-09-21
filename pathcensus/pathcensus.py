""":class:`PathCensus` is the basis which structural similarity and
complementarity coefficients are derived from. In its raw form
it is a set of counts of wedge and head triples (2-paths) and quadruples
(3-paths) traversing an ``(i, j)`` edge as well as counts of corresponding
triangles (3-cycles) and quadrangles (4-cycles).
In the weighted case there are separate counts of triangles
and quadrangles for wedge and head paths as average weights are defined
differently for these two cases.

.. note::
    Path/cycle counts and structural coefficients are returned
    as :class:`pandas.Series` or :class:`pandas.DataFrame` objects indexed
    properly with integer node indices corresponding to the ordering of
    rows in the underlying adjacency matrix of the network.

.. note::
    Path census calculations are relatively efficient as the main
    workhorse functions are just-in-time (JIT) compiled to highly
    optimized C code using :py:mod:`numba` package. Moreover,
    the path census algorithm is based on a state-of-the-art
    graphlet counting algorithm proposed by
    :cite:t:`ahmedEfficientGraphletCounting2015`
    which has worst-case asymptotic computational complexity of
    :math:`O(md^2_{\\text{max}})`, where :math:`m` is the number
    of edges and :math:`d_{\\text{max}}` is the maximum degree.

    Moreover, some additional optimizations are used to speed up
    the calculations in the case of highly heterogeneous degree
    distributions (e.g. power laws). See ``min_di`` argument in
    :meth:`PathCensus.count_paths`.

Node- and graph-level counts are derived from edge-level counts according
to simple aggregations rules. They are defined in definition classes
implemented in :py:mod:`pathcensus.definitions` submodule.

.. seealso::

    :class:`pathcensus.definitions.PathDefinitionsUnweighted`
    for the naming scheme used for unweighted counts.

    :class:`pathcensus.definitions.PathDefinitionsWeighted`
    for the naming scheme used for weighted counts.

Below a node-level census for a simple triangle graph is counted.

.. doctest:: census-triangle

    >>> import numpy as np
    >>> from pathcensus import PathCensus
    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> P = PathCensus(G)
    >>> # Census calculations can be also parallelized
    >>> # by default all available threads are used
    >>> P = PathCensus(G, parallel=True)
    >>> # But the number of threads can be set explicitly
    >>> P = PathCensus(G, num_threads=2)
    >>> P.census("nodes")
       t  tw  th  q0  qw  qh
    i
    0  1   2   2   0   0   0
    1  1   2   2   0   0   0
    2  1   2   2   0   0   0

Structural similarity
---------------------

Structural similarity coefficients (:meth:`PathCensus.similarity`)
as well as their corresponding clustering (:meth:`PathCensus.tclust`)
and closure coefficients (:meth:`PathCensus.tclosure`)
are defined quite simply in terms of ratios of 3-cycles (triangles)
to 2- (triples) counted at the levels of edges, nodes or globaly within an
entire graph. The figure below presents a summary of the underlying geometric
motivation as well as the main properties of structural similarity
coefficients, including the differences relative to local clustering and
closure coefficient
:cite:p:`wattsCollectiveDynamicsSmallworld1998,yinLocalClosureCoefficient2019`.

.. figure:: /figures/sim.svg
    :align: center
    :alt: Overview of the properties of structural similarity coefficients

    Overview of the properties of structural similarity coefficients.

Below node-wise structural similarity coefficients are counted for
a simple triangle graph.

.. doctest:: simcoef-nodes

    >>> import numpy as np
    >>> from pathcensus import PathCensus
    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> P = PathCensus(G)
    >>> P.similarity("nodes")
    i
    0    1.0
    1    1.0
    2    1.0
    dtype: float64

Structural complementarity
--------------------------

Structural complementarity (:meth:`PathCensus.complementarity`)
coefficients and their corresponding clustering
(:meth:`PathCensus.qclust`) and closure coefficients
(:meth:`PathCensus.qclosure`) are defined in terms of ratios of 4-cycles
(quadrangles) to 3-paths (quadruples) and can be defined at the levels
of edges, nodes and entire graphs. The figure below present a summary
of the underlying geometric motivation and some of the main properties.

.. figure:: /figures/comp.svg
    :align: center
    :alt: Overview of the properties of structural complementarity coefficients

    Overview of the properties of structural complementarity coefficients.

Below we calculate complementarity coefficients for nodes
in a 4-clique graph. Note that complementarity coefficients are all
zeros as there are not quadrangles without any chordal edges.

.. doctest:: compcoefs-nodes

    >>> import numpy as np
    >>> from pathcensus import PathCensus
    >>> G = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    >>> P = PathCensus(G)
    >>> P.complementarity("nodes")
    i
    0    0.0
    1    0.0
    2    0.0
    3    0.0
    dtype: float64

Weighted coefficients
---------------------

Structural coefficients can be also defined for weighted networks in which
case paths and cycles are weighted according to the arithmetic average
over edge weights defining an underlying path
(so closing or chordal edges in triangles/quadrangles are ignored).
This can be seen as an extension of the weighted clustering coefficient
proposed by :cite:t:`barratArchitectureComplexWeighted2004`.
Indeed, our formulation of the weighted clustering based on triangles
is equivalent to it.
The figure below presents a summary of the weighting rules.

.. figure:: /figures/weighted.svg
    :align: center
    :alt: Overview of weighted path/cycle counts

    Overview of weighted path/cycle counts.

Edge weights should be detected automatically in most cases provided that
a standard name of edge weight attribute (``"weight"``) is used.
However, weighted computations may be also enabled/disabled explicitly
by using ``weighted`` argument.

.. doctest:: coefs-weighted

    >>> import numpy as np
    >>> from pathcensus import PathCensus
    >>> G = np.array([[0,2,3],[2,0,11],[3,11,0]])
    >>> PathCensus(G).census("nodes")
       twc   thc    tw    th  q0wc  q0hc  qw  qh
    i
    0  2.5  6.75   5.0  13.5   0.0   0.0   0.0   0.0
    1  6.5  4.75  13.0   9.5   0.0   0.0   0.0   0.0
    2  7.0  4.50  14.0   9.0   0.0   0.0   0.0   0.0
    >>> PathCensus(G, weighted=False).census("nodes")
       t  tw  th  q0  qw  qh
    i
    0  1   2   2   0   0   0
    1  1   2   2   0   0   0
    2  1   2   2   0   0   0
"""
from __future__ import annotations
from typing import Union, Literal, Optional, Any, Tuple, Dict
import numpy as np
from scipy.sparse import csr_matrix
import numba
import pandas as pd
from . import types, adjacency
from .core.graph import Graph
from .core.parallel import count_paths_parallel
from .types import UInt, Float
from .definitions import PathDefinitionsUnweighted, PathDefinitionsWeighted


class PathCensus:
    """Path census and structural coefficients calculations
    for undirected graphs.

    Attributes
    ----------
    graph
        :class:`pathcensus.core.graph.Graph` instance
        for calculating path census.
    counts
        Data frame with path/cycle counts per edge.
        Initialization may be postponed.

    Notes
    -----
    Naming scheme used for denoting counts is documented in the docstring
    for ``definitions`` attribute (i.e. ``self.definitions``).

    .. testsetup:: pathcensus

        import numpy as np
        from pathcensus import PathCensus
    """
    # pylint: disable=too-many-public-methods
    class Meta:
        """Container class with various metadata such
        as lists of possible values of arguments of different
        methods of :class:`PathCensus`.

        **Fields**

        mode
            Allowed values of ``mode`` argument in structural
            coefficients methods.
        undef
            Allowed values of ``undefined`` argument in structural
            coefficients methods.
        """
        mode = ("nodes", "edges", "global")
        undef = ("nan", "zero")

    def __init__(
        self,
        graph: types.GraphABC,
        weighted: Optional[bool] = None,
        validate: bool = True,
        adj_kws: Optional[Dict] = None,
        count_paths: bool = True,
        **kwds: Any
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        graph
            Graph-like object registered with
            :py:class:`pathcensus.types.GraphABC` abstract base class
            and a registered single dispatch method registered on
            :py:class:`pathcensus.utils.adjacency`. Sparse matrices
            work out of the box.
        weighted
            Should the graph be interpreted as weighted.
            Determined automatically if ``None``.
        validate
            Should input graph be validate for correctness
            (i.e. checked if is undirected).
        adj_kws
            Additional keyword params passed to
            :py:func:`pathcensus.utils.adjacency`.
        count_paths
            Should `counts` attribute be initialized immediately.
            It can be initialized later using :py:meth:`count` method.
        **kwds
            Passed to :py:meth:`count_paths`.
        """
        adj_kws    = adj_kws or {}
        graph_kws  = dict(weighted=weighted, validate=validate, **adj_kws)
        self.graph = self.get_graph(graph, **graph_kws)

        # Setup path definition objects
        if self.weighted:
            self.definitions = PathDefinitionsWeighted()
        else:
            self.definitions = PathDefinitionsUnweighted()

        self.counts = None
        if count_paths:
            self.count(**kwds)

    def count(self, **kwds: Any) -> None:
        """Count paths and set `self.counts` attribute.

        ``**kwds`` are passed to :py:meth:`count_paths`.
        """
        E, counts = self.count_paths(self.graph, **kwds)
        self.counts = self._make_counts(E, counts)

    # Properties --------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.graph.n_nodes

    @property
    def vcount(self) -> int:
        return self.n_nodes

    @property
    def n_edges(self) -> int:
        return self.counts.shape[0]
    @property
    def ecount(self) -> int:
        return self.n_edges

    @property
    def weighted(self) -> bool:
        return self.graph.weighted

    @property
    def degree(self) -> np.ndarray:
        """Get degree sequence of the underlying graph."""
        return self.graph.D

    @property
    def strength(self) -> np.ndarray:
        """Get strength sequence of the underlying graph
        (or degree sequence in the unweighted case).
        """
        if self.weighted:
            return self.graph.S
        return self.graph.D

    @property
    def tdf(self) -> pd.DataFrame:
        """Data frame with triple/triangle counts per edge."""
        cols = [
            name for name in self.definitions.get_column_names()
            if name in self.definitions["sim"]
        ]
        return self.counts[cols]

    @property
    def qdf(self) -> pd.DataFrame:
        """Data frame with quadruple/quadrangle counts per edge."""
        cols = [
            name for name in self.definitions.get_column_names()
            if name in self.definitions["comp"]
        ]
        return self.counts[cols]

    # Static & class methods --------------------------------------------------

    @classmethod
    def get_graph(
        cls,
        graph: types.GraphABC,
        weighted: Optional[bool] = None,
        validate: bool = True,
        **kwds: Any
    ) -> Graph:
        """Get graph object for path counting.

        Parameters
        ----------
        graph
            A compatibe graph object registered with `paths.types.GraphABC`.
        n_nodes
            Number of nodes. Used only when `graph` is passed as an edgelist.
        weighted
            Should the graph be interpreted as weighted graph.
            If ``None`` then it is determined based on the number of
            unique values of non-zero values in the adjacency matrix.
        validate
            Should input graph be validate for correctness
            (i.e. checked if is undirected).
        **kwds
            Passed to :py:func:`pathcensus.utils.adjacency`.
        """
        A = adjacency(graph, **kwds)

        if validate and (A != A.T).count_nonzero() > 0:
            raise AttributeError("only undirected graphs are accepted")

        n_nodes = A.shape[0]
        E = np.ascontiguousarray(np.array(A.nonzero(), dtype=UInt).T)

        if weighted is None:
            weighted = any(A.data != 1)
        if weighted:
            W = A.data.astype(Float)
        else:
            W = None

        G = Graph(n_nodes, E, W)
        return G

    @classmethod
    def count_paths(
        cls,
        graph: Union[types.GraphABC, Graph],
        *,
        parallel: Optional[bool] = None,
        num_threads: Optional[int] = None,
        graph_kws: Optional[Dict] = None,
        min_di: bool = True,
        **kwds: Any
    ) -> Tuple[int, np.ndarray]:
        """Count paths and cycles in a graph.

        Parameters
        ----------
        graph
            :py:class:`pathcensus.core.graph.Graph` instance.
            or graph-like object that can be converted to it.

        parallel
            Should parallel counting algorithm be used.
            When ``None`` it is used by default for graphs
            with at least one million edges.
        num_threads
            Number of threads to use when ``parallel=True``.
        batch_size
            Batch size to use when running with ``parallel=True``.
        graph_kws
            Additional keyword arguments passed to :py:meth:`get_graph`.
            Used only when `graph` is not already in the JIT-compiled form.
        min_di
            Should `di < dj` rule for iterating over edges be used.
            This way the most expensive loop of the `PathCensus` algorithm
            for computing edge-wise path/cycle counts always iterates over
            neighbors of the lower degree node in an ``(i, j)`` edge.
            Almost always should be set to ``True``.
            The argument is used mostly for testing purposes.
        **kwds
            Passed to :py:func:`pathcensus.core.parallel.count_paths_parallel`
            when ``parallel=True``.

        Notes
        -----
        The ``parallel=True`` argument may not work and lead to segmentation
        faults on some MacOS machines.

        Returns
        -------
        n_nodes
            Number of nodes.
        counts
            Path and cycles counts.
        """
        if isinstance(graph, types.GraphABC):
            graph = cls.get_graph(graph, **(graph_kws or {}))

        if min_di:
            E = graph.get_min_di_edges()
        else:
            E = graph.get_edges()

        if parallel is None:
            # Use parallel algorithm when at least 100k edges
            parallel = graph.n_edges >= 1e5
        if not num_threads or num_threads <= 0:
            # pylint: disable=no-member
            num_threads = numba.config.NUMBA_NUM_THREADS
        if parallel and num_threads > 1:
            orig_num_threads = numba.get_num_threads()
            numba.set_num_threads(num_threads)
            try:
                E, counts = count_paths_parallel(graph, **kwds)
            finally:
                numba.set_num_threads(orig_num_threads)
        else:
            E, counts = graph.count_paths(E)
        return E, counts

    # Auxiliary methods -------------------------------------------------------

    def get_counts(
        self,
        mode: Literal[Meta.mode] = Meta.mode[0], # type: ignore
    ) -> pd.DataFrame:
        """Get (possibly aggregated) path counts.

        Parameters
        ----------
        mode
            Should node, edge or global counts be calculated.
        """
        self._check_mode(mode)
        counts = self.counts

        if mode == "nodes":
            counts = counts.groupby(level="i") \
                .sum() \
                .reindex(np.arange(self.n_nodes), copy=False) \
                .fillna(0)
        elif mode == "global":
            counts = counts.sum().to_frame().T

        return counts

    # Similarity coefficients -------------------------------------------------

    def tclust(
        self,
        *,
        undefined: Literal[Meta.undef] = Meta.undef[0], # type: ignore
        counts: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Triangle-based local clustering (node-wise).

        It is equivalent to local clustering coefficient
        :cite:p:`wattsCollectiveDynamicsSmallworld1998`.

        Parameters
        ----------
        undefined
            If ``'nan'`` the nodes with undefined values are treated
            as NaNs. If ``'zero'`` then they are considered zeros.
        counts
            Path counts data frame to use. Mostly for internal use.

        Notes
        -----
        It is defined as the ratio of triangles including a focal node ``i``
        to the number of wedge triples centered at it:

        .. math::

            s^W_i = \\frac{2T_i}{t^W_i}

        .. figure:: /figures/t-wedge.svg
            :align: center
            :alt: Wedge triple

            Wedge triple.

        Examples
        --------

        .. doctest:: pathcensus

            >>> # Triangle graph
            >>> A = np.array([[0,1,1], [1,0,1], [1,1,0]])
            >>> PathCensus(A).tclust()
            i
            0    1.0
            1    1.0
            2    1.0
            dtype: float64
        """
        df    = counts if counts is not None else self.get_counts("nodes")
        num   = df[self._a("twc")]
        denom = df[self._a("tw")]
        return self._divide(num, denom, undefined=undefined)

    def tclosure(
        self,
        *,
        undefined: Literal[Meta.undef] = Meta.undef[0], # type: ignore
        counts: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Triangle-based local closure coefficient (node-wise).

        It is equivalent to local closure coefficient
        :cite:p:`yinLocalClosureCoefficient2019`.

        Parameters
        ----------
        undefined
            If ``'nan'`` the nodes with undefined values are treated
            as NaNs. If ``'zero'`` then they are considered zeros.
        counts
            Path counts data frame to use. Mostly for internal use.

        Notes
        -----
        It is defined as the ratio of the number of triangles including
        a focal node ``i`` to the number of head triples starting from it:

        .. math::

            s^H_i = \\frac{2T_i}{t^H_i}

        .. figure:: /figures/t-head.svg
            :align: center
            :alt: Head triple

            Head triple.

        Examples
        --------

        .. doctest:: pathcensus

            >>> # Triangle graph
            >>> A = np.array([[0,1,1],[1,0,1],[1,1,0]])
            >>> PathCensus(A).tclosure()
            i
            0    1.0
            1    1.0
            2    1.0
            dtype: float64
        """
        df    = counts if counts is not None else self.get_counts("nodes")
        num   = df[self._a("thc")]
        denom = df[self._a("th")]
        return self._divide(num, denom, undefined=undefined)

    def similarity(
        self,
        mode: Literal[Meta.mode] = Meta.mode[0], # type: ignore
        *,
        undefined: Literal[Meta.undef] = Meta.undef[0], # type: ignore
        counts: Optional[pd.DataFrame] = None
    ) -> Union[pd.Series, float]:
        """Structural similarity coefficients.

        Parameters
        ----------
        mode
            Should it be calculated for nodes, edges or globally
            (equivalent to global clustering).
        undefined
            If ``'nan'`` the nodes with undefined values are treated
            as NaNs. If ``'zero'`` then they are considered zeros.
        counts
            Path counts data frame to use. Mostly for internal use.

        Notes
        -----
        It is defined as the ratio of triangles including a focal node ``i``
        to the total number of both wedge and head triples:

        .. math::

            s_i = \\frac{4T_i}{t^W_i + t^H_i}

        See Also
        --------
        simcoefs : structural similarity coefficients
        coefs : structural coefficients

        Examples
        --------

        .. doctest:: pathcensus

            >>> # Triangle graph
            >>> A = np.array([[0,1,1], [1,0,1], [1,1,0]])
            >>> PathCensus(A).similarity("edges")
            i  j
            0  1    1.0
               2    1.0
            1  0    1.0
               2    1.0
            2  0    1.0
               1    1.0
            dtype: float64
            >>> PathCensus(A).similarity("nodes")
            i
            0    1.0
            1    1.0
            2    1.0
            dtype: float64
            >>> PathCensus(A).similarity("global")
            1.0
        """
        df    = counts if counts is not None else self.get_counts(mode)
        num   = df[self._a("twc")] + df[self._a("thc")]
        denom = df[self._a("tw")] + df[self._a("th")]
        return self._divide(num, denom, undefined=undefined)

    # Complementarity coefficients --------------------------------------------

    def qclust(
        self,
        *,
        undefined: Literal[Meta.undef] = Meta.undef[0], # type: ignore
        counts: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Quadrangle-based local clustering coefficient (node-wise).

        Parameters
        ----------
        undefined
            If ``'nan'`` the nodes with undefined values are treated
            as NaNs. If ``'zero'`` then they are considered zeros.
        counts
            Path counts data frame to use. Mostly for internal use.

        Notes
        -----
        It is defined as the ratio of quadrangles including a focal node ``i``
        and the number of wedge quadruples with ``i`` at the second position
        (this is to avoid double counting and make the number of wedge and
        head quadruples per quadrangle equal):

        .. math::

            c^W_i = \\frac{2Q_i}{q^W_i}

        .. figure:: /figures/q-wedge.svg
            :align: center
            :alt: Wedge quadruple

            Wedge quadruple.

        Examples
        --------

        .. doctest:: pathcensus

            >>> # Quadrangle graph
            >>> A = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
            >>> PathCensus(A).qclust()
            i
            0    1.0
            1    1.0
            2    1.0
            3    1.0
            dtype: float64
        """
        df    = counts if counts is not None else self.get_counts("nodes")
        num   = self._qcount(df, which="wedge")
        denom = df[self._a("qw")]
        return self._divide(num, denom, undefined=undefined)

    def qclosure(
        self,
        *,
        undefined: Literal[Meta.undef] = Meta.undef[0], # type: ignore
        counts: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Quadrangle-based local closure coefficient.

        Parameters
        ----------
        undefined
            If ``'nan'`` the nodes with undefined values are treated
            as NaNs. If ``'zero'`` then they are considered zeros.
        counts
            Path counts data frame to use. Mostly for internal use.

        Notes
        -----
        It is defined as the ratio of quadrangles including a focal node ``i``
        and the number of head quadruples starting from it.

        .. math::

            c^H_i = \\frac{2Q_i}{q^H_i}

        .. figure:: /figures/q-head.svg
            :align: center
            :alt: Head quadruple

            Head quadruple.

        Examples
        --------

        .. doctest:: pathcensus

            >>> # Quadrangle graph
            >>> A = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
            >>> PathCensus(A).qclosure()
            i
            0    1.0
            1    1.0
            2    1.0
            3    1.0
            dtype: float64
        """
        df    = counts if counts is not None else self.get_counts("nodes")
        num   = self._qcount(df, which="head")
        denom = df[self._a("qh")]
        return self._divide(num, denom, undefined=undefined)

    def complementarity(
        self,
        mode: Literal[Meta.mode] = Meta.mode[0], # type: ignore
        *,
        undefined: Literal[Meta.undef] = Meta.undef[0], # type: ignore
        counts: Optional[pd.DataFrame] = None
    ) -> Union[pd.Series, float]:
        """Structural complementarity coefficients.

        Parameters
        ----------
        mode
            Should it be calculated for nodes, edges or globally
            (equivalent to global clustering).
        undefined
            If ``'nan'`` the nodes with undefined values are treated
            as NaNs. If ``'zero'`` then they are considered zeros.
        counts
            Path counts data frame to use. Mostly for internal use.

        Notes
        -----
        The node-wise coefficient is defined as the ratio of quadrangles
        including a focal node ``i`` and the total number of both wedge
        and head quadruples:

        .. math::

            c_i = \\frac{4Q_i}{q^W_i + q^H_i}

        The edge-wise coefficient is defined as the ratio of quadrangles
        including an ``(i, j)`` edge and the number of quadruples starting
        at it:

        .. math::

            c_{ij} = \\frac{2Q_{ij}}{q_{ij}}

        The global coefficient is defined as the ratio of sums of quadrangles
        to the sum of quadruples (wedge or head):

        .. math::

            c
            = \\frac{2\\sum_i Q_i}{\\sum_i q^W_i}
            = \\frac{2\\sum_i Q_i}{\\sum_i q^H_i}

        See Also
        --------
        compcoefs : structural complementarity coefficients
        coefs : structural coefficients

        Examples
        --------

        .. doctest:: pathcensus

            >>> # Quadrangle graph
            >>> A = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
            >>> PathCensus(A).complementarity("edges")
            i  j
            0  1    1.0
               3    1.0
            1  0    1.0
               2    1.0
            2  1    1.0
               3    1.0
            3  0    1.0
               2    1.0
            dtype: float64
            >>> PathCensus(A).complementarity("nodes")
            i
            0    1.0
            1    1.0
            2    1.0
            3    1.0
            dtype: float64
            >>> PathCensus(A).complementarity("global")
            1.0
        """
        df  = counts if counts is not None else self.get_counts(mode)
        num = self._qcount(df, which="wedge") \
            + self._qcount(df, which="head")
        denom = df[self._a("qw")] + df[self._a("qh")]
        return self._divide(num, denom, undefined=undefined)

    # Summaries ---------------------------------------------------------------

    def simcoefs(
        self,
        mode: Literal[Meta.mode] = Meta.mode[0], # type: ignore
        *,
        undefined: Literal[Meta.undef] = Meta.undef[0], # type: ignore
        census: bool = False,
        counts: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Calculate similarity coefficients including clustering
        and closure coefficients when ``mode="nodes"`` or their
        node-wise averages when ``mode="global"``.

        Parameters
        ----------
        mode
            Should node, edge or global counts be calculated.
        undefined
            If ``'nan'`` the nodes with undefined values are treated
            as NaNs. If ``'zero'`` then they are considered zeros.
        census
            If ``True`` then path census data is added.
            as columns in the front of the data frame.
        counts
            Path counts data frame to use. Mostly for internal use.

        Examples
        --------

        .. doctest:: pathcensus

            >>> # Triangle graph
            >>> A = np.array([[0,1,1], [1,0,1], [1,1,0]])
            >>> PathCensus(A).simcoefs("edges")
                sim
                i j
                0 1  1.0
                  2  1.0
                1 0  1.0
                  2  1.0
                2 0  1.0
                  1  1.0
            >>> PathCensus(A).simcoefs("nodes")
               sim  tclust  tclosure
            i
            0  1.0     1.0       1.0
            1  1.0     1.0       1.0
            2  1.0     1.0       1.0
            >>> PathCensus(A).simcoefs("global")
               sim_g  sim  tclust  tclosure
            0    1.0  1.0     1.0       1.0
        """
        counts = counts if counts is not None else self.get_counts(mode)
        kwds = dict(undefined=undefined)

        if mode == "edges":
            coefs = pd.DataFrame({
                "sim": self.similarity(mode, counts=counts, **kwds),
            }, index=counts.index)
        elif mode == "nodes":
            coefs = pd.DataFrame({
                "sim": self.similarity(mode, counts=counts, **kwds),
                "tclust": self.tclust(counts=counts, **kwds),
                "tclosure": self.tclosure(counts=counts, **kwds),
            })
        else:
            coefs = self.simcoefs(mode="nodes", **kwds).mean().to_frame().T
            coefs.insert(0, "sim_g", self.similarity(mode, counts=counts, **kwds))

        if census:
            paths = self.census(mode)
            coefs = pd.concat([coefs, paths], axis=1)

        return coefs

    def compcoefs(
        self,
        mode: Literal[Meta.mode] = Meta.mode[0], # type: ignore
        *,
        undefined: Literal[Meta.undef] = Meta.undef[0], # type: ignore
        census: bool = False,
        counts: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Calculate complementarity coefficients including clustering
        and closure coefficients when ``mode="nodes"`` or their
        node-wise averages when ``mode="global"``.

        Parameters
        ----------
        mode
            Should node, edge or global counts be calculated.
        undefined
            If ``'nan'`` the nodes with undefined values are treated
            as NaNs. If ``'zero'`` then they are considered zeros.
        census
            If ``True`` then path census data is added.
        counts
            Path counts data frame to use. Mostly for internal use.

        Examples
        --------

        .. doctest:: pathcensus

            >>> # Quadrangle graph
            >>> A = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
            >>> PathCensus(A).compcoefs("edges")
                comp
            i j
            0 1   1.0
              3   1.0
            1 0   1.0
              2   1.0
            2 1   1.0
              3   1.0
            3 0   1.0
              2   1.0
            >>> PathCensus(A).compcoefs("nodes")
               comp  qclust  qclosure
            i
            0   1.0     1.0       1.0
            1   1.0     1.0       1.0
            2   1.0     1.0       1.0
            3   1.0     1.0       1.0
            >>> PathCensus(A).compcoefs("global")
               comp_g  comp  qclust  qclosure
            0     1.0   1.0     1.0       1.0
        """
        counts = counts if counts is not None else self.get_counts(mode)
        kwds = dict(undefined=undefined)

        if mode == "edges":
            coefs = pd.DataFrame({
                "comp": self.complementarity(mode, counts=counts, **kwds),
            }, index=counts.index)
        elif mode == "nodes":
            coefs = pd.DataFrame({
                "comp": self.complementarity(mode, counts=counts, **kwds),
                "qclust": self.qclust(counts=counts, **kwds),
                "qclosure": self.qclosure(counts=counts, **kwds)
            })
        else:
            coefs = self.compcoefs(mode="nodes", **kwds).mean().to_frame().T
            coefs.insert(0, "comp_g", self.complementarity(mode, counts=counts, **kwds))

        if census:
            paths = self.census(mode)
            coefs = pd.concat([coefs, paths], axis=1)

        return coefs

    def coefs(
        self,
        mode: Literal[Meta.mode] = Meta.mode[0], # type: ignore
        **kwds
    ) -> pd.DataFrame:
        """Calculate structural coefficients.

        Parameters
        ----------
        mode
            Should node, edge or global counts be calculated.
        undefined
            If ``'nan'`` the nodes with undefined values are treated
            as NaNs. If ``'zero'`` then they are considered zeros.
        census
            If ``True`` then path census data is added.
        counts
            Path counts data frame to use. Mostly for internal use.

        See Also
        --------
        simcoefs : structural similarity coefficients
        compcoefs: structural complementarity coefficients
        """
        if "counts" not in kwds:
            kwds["counts"] = self.get_counts(mode)

        census = kwds.pop("census", False)

        skw = kwds.copy()
        ckw = kwds

        scoefs = self.simcoefs(mode, **skw)
        ccoefs = self.compcoefs(mode, **ckw)
        coefs = pd.concat([scoefs, ccoefs], axis=1)

        if census:
            paths = self.census(mode)
            coefs = pd.concat([coefs, paths], axis=1)

        return coefs

    def census(
        self,
        mode: Literal[Meta.mode] = Meta.mode[0], # type: ignore
        *,
        counts: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Calculate path census.

        Parameters
        ----------
        mode
            Should node, edge or global counts be calculated.
        counts
            Path counts data frame to use. Mostly for internal use.

        Examples
        --------

        .. doctest:: pathcensus

            >>> # Triangle graph
            >>> A = np.array([[0,1,1], [1,0,1], [1,1,0]])
            >>> PathCensus(A).census()
               t  tw  th  q0  qw  qh
            i
            0  1   2   2   0   0   0
            1  1   2   2   0   0   0
            2  1   2   2   0   0   0
        """
        self._check_mode(mode)
        if counts is None:
            counts = self.get_counts(mode).copy()

        arules = self.definitions.aggregation.get(mode, {})
        for k, v in arules.items():
            if self.weighted:
                counts[k] /= v
            else:
                counts[k] //= v

        return counts

    # Serialization -----------------------------------------------------------

    def dump(self) -> Tuple[
        int, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """Dump to raw data in the form of arrays and the number of nodes.

        Returns
        -------
        n_nodes
            Number of nodes.
        E
            Edgelist array.
        W
            Optional edge weights array.
        counts
            Path counts array. May be ``None``.
        """
        return self.n_nodes, self.graph.E, self.graph.W, self.counts

    @classmethod
    def from_dump(
        cls,
        n_nodes: int,
        E: np.ndarray,
        W: Optional[np.ndarray] = None,
        counts: Optional[np.ndarray] = None,
        adj_kws: Optional[Dict] = None,
        **kwds: Any
    ) -> PathCensus:      # type: ignore
        """Construct from the output of :py:meth:`dump`.

        Parameters
        ----------
        n_nodes
            Number of nodes.
        E
            Edgelist array.
        W
            Optional edge weights array.
        counts
            Optional path counts array. It is calculated on-the-fly
            when ``None``.
        adj_kws
            Passed to :py:func:`pathcensus.utils.adjacency`.
        **kwds
            Passed to :py:meth:`count_paths`
            when `counts` is ``None``.
        """
        if W is None:
            weighted = False
            W = np.full(len(E), 1)
        else:
            weighted = True

        i, j = E[:, 1:].T
        A = csr_matrix((W, (i, j)), shape=(n_nodes, n_nodes))
        adj_kws = adj_kws or {}
        paths = cls(A, weighted=weighted, adj_kws=adj_kws, count_paths=False)
        if counts is None:
            paths.count(**kwds)
        else:
            paths.counts = counts
        return paths

    # Internals ---------------------------------------------------------------

    def _a(self, name: str) -> str:
        """Resolve possibly aliased path name."""
        return self.definitions.resolve(name)

    def _make_counts(
        self,
        E: np.ndarray,
        counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make path counts data frame."""
        swaps = self.definitions.get_swap_rules()
        cols  = self.definitions.get_column_ids()
        names = self.definitions.get_column_names()

        E2 = E.copy()
        E2[:, [0, 1]] = E[:, [1, 0]]
        E = np.vstack((E, E2))

        if not self.weighted:
            counts = counts.astype(UInt)

        counts2 = counts.copy()
        for u, v in swaps:
            counts2[:, [u, v]] = counts[:, [v, u]]
        counts = np.vstack((counts, counts2))

        # Drop unnecessary columns
        counts = counts[:, cols]

        counts = pd.DataFrame(
            data=counts,
            columns=names,
            index=pd.MultiIndex.from_arrays(
                arrays=E.T,
                names=["i", "j"]
            )
        )
        return counts.sort_index()

    def _check_undef(self, val: str) -> None:
        if val not in self.Meta.undef:
            raise ValueError(f"'undefined' has to be one of {self.Meta.undef}")

    def _check_mode(self, val: str) -> None:
        if val not in self.Meta.mode:
            raise ValueError(f"'mode' has to be one of {self.Meta.mode}")

    def _divide(
        self,
        x: Union[int, float],
        y: Union[int, float],
        *,
        undefined: Literal[Meta.undef] = Meta.undef[0], # type: ignore
    ) -> float:
        with np.errstate(invalid="ignore"):
            out = x / y
        if undefined == "zero":
            out[np.isnan(out) | np.isinf(out)] = 0.0
        else:
            out[np.isinf(out)] = np.nan
        if out.size == 1:
            out = out.iloc[0]
        return out

    def _qcount(
        self,
        df: pd.DataFrame,
        which: Literal["wedge", "head"],
    ) -> pd.Series:
        """Get quadrangle count."""
        if which == "wedge":
            col = "q0wc"
        elif which == "head":
            col = "q0hc"
        else:
            raise ValueError("incorrect 'which' value")

        q = df[self._a(col)].copy()
        return q

    def _rev_index(self, s: pd.Series) -> pd.Series:
        s = s.copy()
        s.index.names = s.index.names[::-1]
        return s.swaplevel()
