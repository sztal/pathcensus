"""Undirected Enhanced Configuration Model (UECM)
induces a maximum entropy probability distribution over
networks of a given size such that it has specific expected
degree and strength sequences. It can be used to model undirected
weighted networks with edge weights being positive integers
(with no upper bound). See :cite:p:`vallaranoFastScalableLikelihood2021`
for details.

See Also
--------
UECM : UECM class


Examples
--------

.. testsetup:: uecm

    import numpy as np
    from pathcensus.nullmodels import UECM

.. doctest:: uecm

    >>> import random
    >>> import igraph as ig
    >>> # Make a ER random graph with random integer weights
    >>> random.seed(27732)
    >>> G = ig.Graph.Erdos_Renyi(20, p=.2)
    >>> G.es["weight"] = np.random.randint(1, 11, G.ecount())
    >>> # Initialize UECM from the graph object
    >>> uecm = UECM(G)
    >>> # Alternatively initialize from an array of sufficient statistics
    >>> # 1st column - degree sequence; 2nd column - strength sequence
    >>> D = np.array(G.degree())
    >>> S = np.array(G.strength(weights="weight"))
    >>> stats = np.column_stack([D, S])
    >>> uecm = UECM(stats).fit()
    >>> # Check fit error
    >>> round(uecm.error, 6)
    0.0
    >>> # Mean absolute deviation of the fitted expected degree sequence
    >>> # from the observed sequence
    >>> (np.abs(uecm.ED - uecm.D) <= 1e-6).all()
    True
    >>> # Mean absolute deviation of the fitted expected strength sequence
    >>> # from the observed sequence
    >>> (np.abs(uecm.ES - uecm.S) <= 1e-6).all()
    True
    >>> # Sample a single instance
    >>> uecm.sample_one()    # doctest: +ELLIPSIS
    <20x20 sparse matrix of type '<class 'numpy.int64'>'
    	with ... stored elements in Compressed Sparse Row format>
    >>> # Sample multiple instances (generator)
    >>> for instance in uecm.sample(10): pass
"""
from typing import Callable
from types import MappingProxyType
import numpy as np
from numba import njit
from NEMtropy import UndirectedGraph
from .base import UndirectedSoftConfigurationModel
from ..utils import rowsums
from ..types import GraphABC
from .. import adjacency



class UECM(UndirectedSoftConfigurationModel):
    """Undirected Enhanced Configuration Model.

    This is a soft configuration model for undirected weighted networks with
    unbounded positive integer weights which belongs to the family
    of Exponential Random Graph Models (ERGMs) with local constraints.
    It induces a maximum entropy probability distribution over a set of
    networks with :math:`N` nodes such that it yields a specific degree sequence
    and a specific strenght sequence on average.

    Attributes
    ----------
    statistics
        2D (float) array with sufficient statistics for nodes.
        In this case there are two sufficient statistics, that is,
        the degree sequence and the strength sequence.
    fit_args
        Dictionary with arguments used in the last call of :py:meth:`fit`.
        ``None`` if the model has not been fitted yet.

    Notes
    -----
    The following important class attributes are also defined:

    labels
        Mapping from abbreviated labels to full names identifying sufficient
        statistics.
    models
        Model names as defined in :py:mod:`NEMtropy` allowed for the specific
        type of model.
    """
    names = MappingProxyType({ "degree": "x", "strength": "y" })
    models = ("ecm_exp", "ecm")
    # Default `fit` method keyword arguments
    default_fit_kwds = MappingProxyType({"initial_guess": "strengths_minor"})

    @property
    def weighted(self) -> bool:
        return True

    @property
    def expected_statistics(self) -> np.ndarray:
        """Expected sufficient statistics."""
        return np.column_stack([
            self.solver.expected_dseq,
            self.solver.expected_strength_seq
        ])

    @property
    def D(self) -> np.ndarray:
        """Observed degree sequence."""
        return self.get_stat("degree", expected=False)

    @property
    def ED(self) -> np.ndarray:
        """Expected degree sequence."""
        return self.get_stat("degree", expected=True)

    @property
    def S(self) -> np.ndarray:
        """Observed strength sequence."""
        return self.get_stat("strength", expected=False)

    @property
    def ES(self) -> np.ndarray:
        """Expected strength sequence."""
        return self.get_stat("strength", expected=True)

    @property
    def pijfunc(self) -> Callable:
        """JIT-compiled routine for calculating :math:`p_{ij}`."""
        return uecm_pij

    @property
    def wijfunc(self) -> Callable:
        """JIT-compiled routine sampling :math:`w_{ij}`."""
        return uecm_wij

    @property
    def Ewijfunc(self) -> Callable:
        """JIT-compiled routing for calculating :math:`\\mathbb{E}[w_{ij}]`
        (conditional on the edge being present).
        """
        return uecm_Ewij

    def extract_statistics(self, graph: GraphABC) -> np.ndarray:
        """Extract sufficient statistics from a graph-like object."""
        A = adjacency(graph).copy()
        S = rowsums(A)
        A.data[:] = 1
        D = rowsums(A)
        return np.column_stack([ D, S ])

    # NEMtropy wrapper methods ------------------------------------------------

    def get_nemtropy_graph(self) -> UndirectedGraph:
        """Get :py:mod:`NEMtropy` graph representation instance."""
        return UndirectedGraph(degree_sequence=self.D, strength_sequence=self.S)


# Sampler routines ------------------------------------------------------------

@njit(boundscheck=False, nogil=True, cache=True)
def uecm_pij(X: np.ndarray, i: int, j: int) -> float:
    """Calculate edge probability :math:`p_{ij}` in UECM model.

    Parameters
    ----------
    X
        1D array of model parameters.
    i, j
        Node indices.
    """
    if i == j:
        return 0
    n  = len(X) // 2
    xx = X[i]*X[j]
    yy = X[i+n]*X[j+n]
    return xx*yy / (1 - yy + xx*yy)

@njit(boundscheck=False, nogil=True, cache=True)
def uecm_wij(X: np.ndarray, i: int, j: int) -> int:
    """Sample edge weight :math:`w_{ij}` in UECM model.

    Parameters
    ----------
    X
        1D Array of model parameters.
    i, j
        Node indices.
    """
    if i == j:
        return 0
    n  = len(X) // 2
    yy = X[i+n]*X[j+n]
    return np.random.geometric(1-yy)

@njit(boundscheck=False, nogil=True, cache=True)
def uecm_Ewij(X: np.ndarray, i: int, j: int) -> float:
    """Calculate expected edge weight :math:`\\mathbb{E}[w_{ij}]`
    (conditional on the edge being present) in UECM model.

    Parameters
    ----------
    X
        1D array od model parameters.
    i, j
        Node indices.
    """
    n  = len(X) // 2
    yy = X[i+n]*X[j+n]
    return 1 / (1-yy)
