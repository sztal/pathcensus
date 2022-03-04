"""Exponential Random Graph Models (ERGM) with local constraints are
such ERGMs in which sufficient statistics are defined at the level of
individual nodes (or globally for the entire graph). In other words, their
values for each node can be set independently. Unlike ERGMs with non-local
constraints which are notoriously problematic
(e.g. due to degenerate convergence and non-projectivity)
they are analytically solvable. Prime examples of ERGMs with local constraints
are configuration models which induce maximum entropy distributions over
graphs with ``N`` nodes with arbitrary expected degree sequence and/or
strength sequence constraints.

The :py:mod:`pathcensus.nullmodels` submodule implements several such
ERGMs which are most appropriate for statistical calibration of strucutral
coefficients. They can be applied to simple undirected and unweighted/weighted
networks.

See Also
--------
ERGM : base class for ERGMs
pathcensus.nullmodels.ubcm : Undirected Binary Configuration Model
    (fixed expected degree sequence)
pathcensus.nullmodels.uecm : Undirected Enhanced Configuration Model
    (fixed expected degree and strength sequences assuming positive integer weights)


.. note::
    The ERGM functionalities provided by :py:mod:`pathcensus` are simple
    wrappers around the :py:mod:`NEMtropy` package.
"""
# pylint: disable=abstract-method
from typing import Any, Union, Mapping, Optional
from typing import Literal, Callable, Iterable, Tuple
from types import MappingProxyType
import io
import contextlib
import warnings
from functools import cached_property
import numpy as np
from numba import njit
from scipy.sparse import spmatrix, csr_matrix
from scipy.sparse.linalg import LinearOperator
from NEMtropy import UndirectedGraph, DirectedGraph
from ..types import GraphABC
from ..utils import relerr


# Base ERGM -------------------------------------------------------------------

class ERGM:
    """Generic base class for Exponential Random Graph Models
    with local (i.e. node-level) constraints.

    Attributes
    ----------
    statistics
        2D (float) array with sufficient statistics for nodes.
        First axis is for nodes and second for differen statistics.
    fit_args
        Dictionary with arguments used in the last call of :py:meth:`fit`.
        ``None`` if the model has not been fitted yet.

    Notes
    -----
    The following class attributes are required and need to be defined on
    concrete subclasses.

    names
        Mapping from names of sufficient statistics to attribute
        names in the :py:mod:`NEMtropy` solver class storing fitted
        model parameters. They must be provided in an order consistent with
        ``statistics``. This is a class attribute which must be defined on
        subclasses implementing particular models. The mapping must have
        stable order (starting from ``python3.6`` an ordinary ``dict`` will do).
        However, it is usually better to use mapping proxy objects instead
        of dicts as they are not mutable.
    labels
        Mapping from abbreviated labels to full names of sufficient statistics.
    models
        Model names as defined in :py:mod:`NEMtropy` allowed for the specific
        type of model. Must be implemented on a subclass as a class attribute.
        The first model on the list should will be used by default.
    """
    # pylint: disable=too-many-public-methods,function-redefined
    names  = None
    aliases = None
    models  = None
    # Default maximum allowed relative error for validation
    default_rtol = 1e-1
    # Default fit methods kwds
    default_fit_kwds = None
    # Solver methods
    methods = ("auto", "newton", "fixed-point")
    # Allowed values of 'which' argument in `_get_stat`
    _stat_which = ("observed", "expected", "parameters")

    def __init__(
        self,
        statistics: Union[np.ndarray, GraphABC],
        **kwds: Any
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        statistics
            Array with sufficient statistics or a graph-like object
            (registered properly with :py:class:`pathcensus.types.GraphABC`).
        **kwds
            Passed to :py:meth:`extract_statistics` when `statistics` is
            passed a graph-like object.
        """
        if isinstance(statistics, GraphABC):
            statistics = self.extract_statistics(statistics, **kwds)

        statistics = statistics.astype(float)
        if statistics.ndim == 1:
            statistics = statistics.reshape(-1, 1)

        self.validate_statistics_shape(statistics)
        self.validate_statistics_values(statistics)

        self.statistics = statistics
        self.fit_args = {}

    # Properties --------------------------------------------------------------

    @property
    def fullname(self) -> str:
        """Full name of model. May be reimplemented on concrete
        subclass to allow using shortened class names.
        """
        return self.__class__.__name__

    @property
    def models(self) -> Tuple[str]:
        if not self.models:
            cn = self.__class__.__name__
            raise NotImplementedError(
                f"it seems '{cn}' does not define allowed models"
            )
        return tuple(self.models)

    @property
    def default_model(self) -> str:
        return self.models[0]

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the underlying graph."""
        return len(self.statistics)

    @property
    def n_stats(self) -> int:
        """Number of sufficient statistics."""
        return len(self.names)

    @property
    def directed(self) -> bool:
        """Is model directed."""
        raise NotImplementedError

    @property
    def weighted(self) -> bool:
        """Is model weighted."""
        raise NotImplementedError

    @property
    def expected_statistics(self) -> np.ndarray:
        """Model-based expected values of sufficient statistics."""
        raise NotImplementedError

    @property
    def names(self) -> Mapping:
        """Mapping from names to :py:mod:`NEMtropy` solver attribute names
        corresponding to sufficient statistics.
        """
        names = self.__class__.names
        if not names:
            cn = self.__class__.__name__
            raise NotImplementedError(
                f"it seems that '{cn}' does not define any names"
            )
        if isinstance(names, Mapping):
            return dict(names)
        return dict(list(names))

    @property
    def labels(self) -> Mapping:
        """Mapping from short labels to full names corresponding to sufficient
        statistics.
        """
        if not self.aliases:
            cn = self.__class__.__name__
            raise NotImplementedError(
                f"it seems '{cn}' does not define any aliases"
            )
        return { self.aliases[n]: n for n in self.names }

    @property
    def fp_threshold(self) -> int:
        """Threshold on the number of nodes after which by default the
        fixed-point solver is used instead of the Newton method solver.
        """
        return 500

    @cached_property
    def solver(self) -> Union[UndirectedGraph, DirectedGraph]:
        """:py:mod:`NEMtropy` graph solver instance."""
        return self.get_nemtropy_graph()

    @property
    def pijfunc(self) -> Callable:
        """JIT-compiled function calculating :math:`p_{ij}`'s
        based on the model.
        """
        raise NotImplementedError

    @property
    def wijfunc(self) -> Callable:
        """JIT-compiled function sampling edge weights :math:`w_{ij}`
        based on the model.
        """
        self._only_weighted()
        raise NotImplementedError

    @property
    def Ewijfunc(self) -> Callable:
        """JIT-compiled function calculating expected edge weights
        :math:`\\mathbb{E}[w_{ij}]` (conditional on being present)
        based on the model.
        """
        self._only_weighted()
        raise NotImplementedError

    @property
    def pmv(self) -> Callable:
        """JIT-compiled function calculating :math:`Pv`
        where :math:`P` is the edge probability matrix
        and :math:`v` is an arbitrary vector.
        """
        return lambda v: get_pmv(self.X, v, self.pijfunc)

    @property
    def rpmv(self) -> Callable:
        """JIT-compiled function calculating :math:`vP`
        where :math:`P` is the edge probability matrix
        and :math:`v` is an abitrary vector.
        """
        if self.directed:
            raise NotImplementedError
        return self.pmv

    @property
    def wmv(self) -> Callable:
        """JIT-compiled function calculating :math:`Wv`
        where :math:`W` is the matrix of expected edge weights
        and :math:`v` is an arbitrary vector.
        """
        self._only_weighted()
        return lambda v: get_wmv(self.X, v, self.pijfunc, self.Ewijfunc)

    @property
    def rwmv(self) -> Callable:
        """JIT-compiled function calculating :math:`vW`
        where :math:`W` is the matrix of expected edge weights
        and :math:`v` is an arbitrary vector.
        """
        if self.directed:
            raise NotImplementedError
        return self.wmv

    # Parameters properties and getters ---------------------------------------

    def get_stat(
        self,
        stat: Union[int, str],
        expected: bool = False
    ) -> np.ndarray:
        """Get sufficient statistic array by index or label.

        Parameters
        ----------
        stat
            Index or label of a sufficient statistic.
        expected
            Should observed or expected statistic be returned.
        """
        which = "expected" if expected else "observed"
        return self._get_stat(stat, which=which)

    def get_param(self, stat: Union[int, str]) -> np.ndarray:
        """Get parameter array associated with a given sufficient statistic.

        ``None`` is returned if the model is not yet fitted.

        Parameters
        ----------
        stat
            Index or label of a sufficient statistic.
        """
        return self._get_stat(stat, which="parameters")

    @property
    def X(self) -> Optional[np.ndarray]:
        """Array with fitted model parameters (1D).

        Raises
        ------
        ValueError
            If model is not fitted.
        """
        self.check_fitted()
        return np.concatenate([
            getattr(self.solver, attr) for attr in self.names.values()
        ])

    @property
    def parameters(self) -> Optional[np.ndarray]:
        """Array with fitted model parameters shaped as ``self.statistics``.

        Raises
        ------
        ValueError
            If model is not fitted.
        """
        return self._get_param_array(self.X)

    @property
    def error(self) -> np.ndarray:
        """Get maximum overall absolute error of the fit."""
        self.check_fitted()
        return self.solver.error

    def get_P(
        self,
        *,
        dense: bool = False
    ) -> Union[LinearOperator, np.ndarray]:
        """Get matrix of edge probabilities.

        Parameters
        ----------
        dense
            If ``True`` then a dense array is returned.
            Otherwise a :py:class:`scipy.sparse.linalg.LinearOperator`
            is returned.
        """
        n = self.n_nodes
        P = LinearOperator(
            shape=(n, n),
            matvec=self.pmv,
            rmatvec=self.rpmv,
            dtype=self.X.dtype
        )
        if dense:
            P = P@np.eye(n)
        return P

    def get_W(
        self,
        *,
        dense: bool = False
    ) -> Union[LinearOperator, np.ndarray]:
        """Get matrix of expected edge weights.

        Parameters
        ----------
        dense
            If ``True`` then a dense array is returned.
            Otherwise a :py:class:`scipy.sparse.linalg.LinearOperator`
            is returned.

        Raises
        ------
        NotImplementedError
            If called on a model instance which is not weighted.
        """
        self._only_weighted()
        n = self.n_nodes
        P = LinearOperator(
            shape=(n, n),
            matvec=self.wmv,
            rmatvec=self.rwmv,
            dtype=self.X.dtype
        )
        if dense:
            P = P@np.eye(n)
        return P


    # Validation methods ------------------------------------------------------

    def validate_statistics_shape(self, statistics: np.ndarray) -> None:
        """Raise ``ValueError`` if ``statistics`` has an incorrect shape
        which is not consistent with the class attribute ``cls.names``.
        """
        if statistics.ndim != 2:
            raise ValueError("'statistics' array does not have two axes")
        ncol = statistics.shape[1]
        if ncol != self.n_stats:
            cnm  = self.__class__.__name__
            raise ValueError(
                f"'statistics' array has {ncol} columns while "
                f"'{cnm}' class defines {self.n_stats} sufficient statistics"
            )

    def validate_statistics_values(self, statistics: np.ndarray) -> None:
        """Raise if ``statistics`` contain incorrect values.

        It must be implemented on a subclass.

        Notes
        -----
        Validation of the shape of ``statistics`` is implemented
        independently in :py:meth:`validate_statistics_shape`
        which is a generic method which in most cases does not need
        to be implemented on subclasses.
        """
        raise NotImplementedError

    def relerr(self) -> np.ndarray:
        """Get error of the fitted expected statistics relative
        to the observed sufficient statistics as
        ``|expected - observed| / |observed|``.
        """
        self.check_fitted()
        return relerr(self.expected_statistics, self.statistics)

    def is_valid(self, rtol: Optional[float] = None) -> bool:
        """Check if model is approximately correct or that the relative
        difference ``|expected - observed| / |observed|`` is not greater
        than ``rtol``.

        Parameters
        ----------
        rtol
            Maximum allowed relative difference.
            Class attribute ``default_rtol`` is used when ``None``.
        """
        rtol = self.default_rtol if rtol is None else rtol
        return self.relerr().max() <= rtol

    def validate(self, rtol: Optional[float] = None) -> None:
        """Raise ``ValueError`` if the relative difference
        ``|expected - observed| / |observed|``, is greater than ``rtol``.

        Parameters
        ----------
        rtol
            Maximum allowed relative difference.
            Class attribute ``default_rtol`` is used when ``None``.

        Returns
        -------
        self
            The same model instance if the error is not raised.
        """
        rtol = self.default_rtol if rtol is None else rtol
        e = self.relerr().max()
        is_valid = e <= rtol
        if not is_valid:
            raise ValueError(
                f"maximum relative error, {e}, is greater than {rtol}"
            )
        return self

    def is_fitted(self) -> bool:
        """Check if model instance is fitted
        (this does not check quality of the fit).
        """
        return self.solver.x is not None

    def check_fitted(self) -> None:
        """Raise `ValueError` if model is not fitted."""
        if not self.is_fitted():
            raise ValueError("model is not fitted; use 'fit' method")

    # Statistics getter methods -----------------------------------------------

    def extract_statistics(self, graph: GraphABC) -> np.ndarray:
        """Extract array of sufficient statistics from a graph-like object."""
        raise NotImplementedError

    # NEMtropy wrapper methods ------------------------------------------------

    def get_nemtropy_graph(self) -> Union[UndirectedGraph, DirectedGraph]:
        """Get :py:mod:`NEMtropy` graph representation instance
        appropriate for a given type of model.
        """
        raise NotImplementedError

    def fit(
        self,
        model: Optional[str] = None,
        method: Literal[methods] = methods[0],   # type: ignore
        **kwds
    ) -> float:
        """Fit model parameters to the observed sufficient statistics
        and returns the overall maximum absolute error.

        Parameters
        ----------
        model
            Type of model to use. Default value defined in
            ``self.default_model`` is used when ``None``.
        method
            Solver method to use. If ``"auto"`` then either Newton or fixed-point
            method is used depending on the number of nodes with the threshold
            defined by ``self.fp_threshold``.
        **kwds
            Passed to NEMtropy solver method ``solve_tool``.

        Notes
        -----
        Some of the ``**kwds`` may be prefilled (but can be overriden)
        with default values defined on ``default_fit_kwds`` class attribute.

        Returns
        -------
        self
            Fitted model.
        """
        if method not in self.methods:
            raise ValueError(f"'method' has to be one of {self.methods}")
        if method == "auto":
            if self.n_nodes < self.fp_threshold:
                method = "newton"
            else:
                method = "fixed-point"

        model = model or self.default_model
        kwds  = { **(self.default_fit_kwds or {}), **kwds }
        kwds  = dict(model=model, method=method, **kwds)

        with \
        warnings.catch_warnings(), \
        contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            self.solver.solve_tool(**kwds)
        self.fit_args = MappingProxyType(kwds)
        return self

    def sample_one(self) -> spmatrix:
        """Sample a graph instance as sparse matrix from the model.

        Returns
        -------
        A
            Graph instance represented as a sparse matrix (CSR format).
        """
        n = self.n_nodes
        if self.weighted:
            E, W = sample_edgelist_weighted(self.X, n, self.pijfunc, self.wijfunc)
            return self._make_adj(E, W)
        E = sample_edgelist_unweighted(self.X, n, self.pijfunc)
        return self._make_adj(E)

    def sample(self, n: int) -> Iterable[spmatrix]:
        """Generate `n` instances sampled from the model.

        Yields
        ------
        A
            Graph instance represented as a sparse matrix (CSR format)
        """
        for _ in range(n):
            yield self.sample_one()


    # Internals ---------------------------------------------------------------

    def _get_param_array(self, X: np.ndarray) -> np.ndarray:
        """Get array of model parameters shaped as ``self.statistics``."""
        return X.reshape(self.n_stats, -1).T

    def _get_stat(
        self,
        stat: Union[int, str],
        *,
        which: Literal[_stat_which] = _stat_which[0]   # type: ignore
    ) -> np.ndarray:
        """Get particular sufficient statistic by index or label.

        Parameters
        ----------
        stat
            Index or label of a statistic to extract.
        which
            Should observed or expected sufficient statistics be returned
            or alternatively their associated model parameters.
        """
        if which not in self._stat_which:
            raise ValueError(f"'which' has to be one of {self._stat_which}")

        if which == "observed":
            statistics = self.statistics
        elif which == "expected":
            statistics = self.expected_statistics
        else:
            statistics = self.parameters

        if isinstance(stat, int):
            return statistics[:, stat]
        if isinstance(stat, str):
            for idx, name in enumerate(self.names):
                if stat == name:
                    return statistics[:, idx]
        raise ValueError(
            f"there is no statistic/parameter with label/index '{stat}'"
        )

    def _make_adj(
        self,
        E: np.ndarray,
        W: Optional[np.ndarray] = None
    ) -> spmatrix:
        """Make adjacency from edgelist and optional edge weighted array."""
        n = self.n_nodes
        if W is None:
            W = np.ones(len(E), dtype=np.uint8)
        i, j = E.T
        A = csr_matrix((W, (i, j)), shape=(n, n))

        if not self.directed:
            A += A.T

        return A

    def _only_weighted(self) -> None:
        """Raise if model is not weighted."""
        if not self.weighted:
            cn = self.__class__.__name__
            raise AttributeError(f"'{cn}' is not weighted")

# Soft Configuration Model ----------------------------------------------------

class SoftConfigurationModel(ERGM):
    """Base class for soft configuration models."""
    def validate_statistics_values(self, statistics: np.ndarray) -> None:
        """Raise if degree sequence contains negative values."""
        for i, name in enumerate(self.names):
            stat = statistics[:, i]
            if np.any(stat <= 0):
                raise ValueError(f"{name} sequence contains non-positive values")

# Undirected Soft Configuration Model -----------------------------------------

class UndirectedSoftConfigurationModel(SoftConfigurationModel):
    """Base class for undirected soft configuration models."""
    aliases = MappingProxyType({
        "degree": "d",
        "strength": "s"
    })

    @property
    def directed(self) -> bool:
        return False


# Compiled routines -----------------------------------------------------------

@njit(boundscheck=False, nogil=True, cache=True)
def get_pmv(
    X: np.ndarray,
    v: np.ndarray,
    pijfunc: Callable[[np.ndarray, int, int], float]
) -> np.ndarray:
    """Calculate :math:`Pv` where :math:`P` is edge probability matrix
    and :math:`v` an arbitrary vector.

    Parameters
    ----------
    X
        1D array of model parameters.
    v
        Arbitrary vector.
    pijfunc
        JIT-compiled function (in no-python mode) calculating edge
        probabilities :math:`p_{ij}`. It should have the following
        signature: ``(X, i, j) -> float``, where ``X`` is a 1D array
        of model parameters. The return value must be a float in ``[0, 1]``.
    """
    v = v.flatten()
    u = np.zeros_like(v, dtype=X.dtype)
    n = len(v)

    for i in range(n):
        for j in range(n):
            pij   = pijfunc(X, i, j)
            u[i] += pij * v[j]

    return u

@njit(boundscheck=False, nogil=True, cache=True)
def get_wmv(
    X: np.ndarray,
    v: np.ndarray,
    pijfunc: Callable[[np.ndarray, int, int], float],
    Ewijfunc: Callable[[np.ndarray, int, int], float]
) -> np.ndarray:
    """Calculate :math:`Wv` where :math:`W` is expected edge weight matrix
    and :math:`v` is an arbitrary vector.

    Parameters
    ----------
    X
        1D array of model parameters.
    v
        Arbitrary vector.
    pijfunc
        JIT-compiled function (in no-python mode) calculating edge
        probabilities :math:`p_{ij}`. It should have the following
        signature: ``(X, i, j) -> float``, where ``X`` is a 1D array
        of model parameters. The return value must be a float in ``[0, 1]``.
    Ewijfunc
        JIT-compiled function (in no-python mode) calculating expected
        edge weights :math:`\\mathbb{E}[p_{ij}]`. It should have the
        following signature ``(X, i, j) -> float``, where ``X`` is a 1D array
        of model parameters. The return value must be a positive float.
    """
    v = v.flatten()
    u = np.zeros_like(v, dtype=X.dtype)
    n = len(v)

    for i in range(n):
        for j in range(n):
            pij   = pijfunc(X, i, j)
            wij   = Ewijfunc(X, i, j)
            u[i] += pij * wij * v[j]

    return u

@njit(boundscheck=False, nogil=True, cache=True)
def sample_edgelist_unweighted(
    X: np.ndarray,
    n_nodes: int,
    pijfunc: Callable[[np.ndarray, int, int], float]
) -> np.ndarray:
    """Sample edgelist array from an ERGM.

    Parameters
    ----------
    X
        1D array of model parameters.
    n_nodes
        Number of nodes in hte underlying graph.
    pijfunc
        JIT-compiled function (in no-python mode) calculating edge
        probabilities :math:`p_{ij}`. It should have the following
        signature: ``(X, i, j) -> float``, where ``X`` is a 1D array
        of model parameters. The return value must be a float in ``[0, 1]``.

    Returns
    -------
    E
        Edgelist array.
    """
    edges = []
    for i in range(1, n_nodes):
        for j in range(i):
            pij = pijfunc(X, i, j)

            if np.random.rand() <= pij:
                edges.append([i, j])

    return np.array(edges)

@njit(boundscheck=False, nogil=True, cache=True)
def sample_edgelist_weighted(
    X: np.ndarray,
    n_nodes: int,
    pijfunc: Callable[[np.ndarray, int, int], float],
    wijfunc: Callable[[np.ndarray, int, int], Union[int, float]]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Sample edgelist array from an ERGM.

    Parameters
    ----------
    X
        1D array of model parameters.
    n_nodes
        Number of nodes in the underlying graph.
    weighted
        Is the model weighted
    pijfunc
        JIT-compiled function (in no-python mode) calculating edge
        probabilities :math:`p_{ij}`. It should have the following
        signature: ``(X, i, j) -> float``, where ``X`` is a 1D array
        of model parameters. The return value must be a float in ``[0, 1]``.
    wijfunc
        JIT-compiled function (in no-python mode) sampling edge weights
        :math:`w_{ij}`. It should have the following signature:
        ``(X, i, j) -> float/int``, where ``X`` is a 1D array of model
        arameters. The return value must be a positive int/float.

    Returns
    -------
    E
        Edgelist array.
    W
        1D array with edge weights.
    """
    edges = []
    weights = []
    for i in range(1, n_nodes):
        for j in range(i):
            pij = pijfunc(X, i, j)

            if np.random.rand() <= pij:
                edges.append([i, j])
                w = wijfunc(X, i, j)
                weights.append(w)

    return np.array(edges), np.array(weights)
