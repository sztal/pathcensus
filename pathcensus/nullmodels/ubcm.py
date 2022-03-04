"""Undirected Binary Configuration Model (UBCM) induces a maximum entropy
probability distribution over networks of a given size such that it has
a specific expected degree sequence. It can be used to model undirected
unweighted networks. See :cite:p:`vallaranoFastScalableLikelihood2021`
for details.

See Also
--------
UBCM : UBCM class


Examples
--------

.. testsetup:: ubcm

    import numpy as np
    from pathcensus.nullmodels import UBCM

.. doctest:: ubcm

    >>> # Make simple ER random graph using `igraph`
    >>> import random
    >>> import igraph as ig
    >>> random.seed(101)
    >>> G = ig.Graph.Erdos_Renyi(20, p=.2)
    >>> # Initialize UBCM directly from the graph object
    >>> ubcm = UBCM(G)
    >>> # Alternatively, initialize from degree sequence array
    >>> D = np.array(G.degree())
    >>> ubcm = UBCM(D).fit()
    >>> # Check fit error
    >>> round(ubcm.error, 6)
    0.0
    >>> # Mean absolute deviation of the fitted expected degree sequence
    >>> # from the observed sequence
    >>> (np.abs(ubcm.ED - ubcm.D) <= 1e-6).all()
    True
    >>> # Set seed of null model sampler and sample ensemble instance
    >>> from pathcensus.utils import set_seed
    >>> set_seed(17)
    >>> ubcm.sample_one()
    <20x20 sparse matrix of type '<class 'numpy.uint8'>'
    	with 84 stored elements in Compressed Sparse Row format>
    >>> # Sample multiple instances (generator)
    >>> for instance in ubcm.sample(10): pass
"""
from typing import Callable
from types import MappingProxyType
import numpy as np
from numba import njit
from NEMtropy import UndirectedGraph
from .base import UndirectedSoftConfigurationModel
from .. import adjacency
from ..utils import rowsums
from ..types import GraphABC


class UBCM(UndirectedSoftConfigurationModel):
    """Undirected Binary Configuration Model.

    This is a soft configuration model for undirected unweighted networks
    which belongs to the family of Exponential Random Graph Models (ERGMs)
    with local constraints. It induces a maximum entropy probability distribution
    over a set of networks with :math:`N` nodes such that it yields a specific
    degree sequence on average.

    Attributes
    ----------
    statistics
        2D (float) array with sufficient statistics for nodes.
        In this case there is only one sufficient statistic, that is,
        the degree sequence.
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
    names = MappingProxyType({"degree": "x"})
    models = ("cm_exp", "cm")
    # Default `fit` method keyword arguments
    default_fit_kwds = MappingProxyType({"initial_guess": "chung_lu"})

    @property
    def fullname(self) -> str:
        return "Undirected Binary Configuration Model"

    @property
    def weighted(self) -> bool:
        return False

    @property
    def expected_statistics(self) -> np.ndarray:
        """Expected sufficient statistics."""
        return self._get_param_array(self.solver.expected_dseq)

    @property
    def D(self) -> np.ndarray:
        """Observed degree sequence."""
        return self.get_stat("degree", expected=False)

    @property
    def ED(self) -> np.ndarray:
        """Expected degree sequence."""
        return self.get_stat("degree", expected=True)

    @property
    def pijfunc(self) -> Callable:
        r"""JIT-compiled routine for calculating :math:`p_{ij}`."""
        return ubcm_pij

    def extract_statistics(self, graph: GraphABC) -> np.ndarray:
        """Extract sufficient statistics from a graph-like object."""
        A = adjacency(graph).copy()
        A.data[:] = 1
        return rowsums(A)

    # NEMtropy wrapper methods ------------------------------------------------

    def get_nemtropy_graph(self) -> UndirectedGraph:
        """Get :py:mod:`NEMtropy` graph representation instance."""
        return UndirectedGraph(degree_sequence=self.D)


# Sampler routines ------------------------------------------------------------

@njit(boundscheck=False, nogil=True, cache=True)
def ubcm_pij(X: np.ndarray, i: int, j: int) -> float:
    """Calculate edge probability :math:`p_{ij}` in UBCM model.

    Parameters
    ----------
    X
        1D Array of model parameters.
    i, j
        Node indices.
    """
    if i == j:
        return 0
    xx = X[i]*X[j]
    return xx / (1 + xx)
