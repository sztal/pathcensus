"""Internal routines for path/cycle counting."""
from typing import Tuple
import numpy as np
import numba
from .graph import Graph
from .types import Float


@numba.njit(parallel=True, boundscheck=False, nogil=True, cache=False)
def count_paths_parallel(
    graph: Graph,
    batch_size: int = 10,
    min_di: bool = True,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Count paths and cycles using parallel algorithm.

    Parameters
    ----------
    graph
        Compiled :py:class:`pathcensus.core.graph.Graph` instance.
    batch_size
        Number of edges processed in one batch.
        Usually should not be very large, but also cannot be too small.
        The default value often works quite well.
    min_di
        Should `di < dj` rule for iterating over edges be used.
        See :meth:`pathcensus.PathCensus.count_paths` for details.
        Almost always should be set to ``True``.
        The argument is used mostly for testing purposes.
    shuffle
        Should rows of the edge array be first reshuffled randomly.
        This often improves performance by decreasing the likelihood
        of concurrent accesses to the same elements of the edge array
        by different threads.
    """
    if min_di:
        E = graph.get_min_di_edges()
    else:
        E = graph.get_edges()

    if shuffle:
        np.random.shuffle(E)

    n_edges = len(E)
    counts = np.zeros((n_edges, graph.npaths), dtype=Float)
    n_batches = int(np.ceil(n_edges / batch_size))

    for i in numba.prange(n_batches):   # pylint: disable=not-an-iterable
        start = batch_size * i
        batch = E[start:start+batch_size]
        _, _counts = graph.count_paths(batch)
        counts[start:start+batch_size] = _counts

    return (
        np.ascontiguousarray(E[:, 1:]),
        np.ascontiguousarray(counts)
    )
