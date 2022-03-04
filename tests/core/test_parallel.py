"""Basic tests for parallel counting algorithms."""
# pylint: disable=redefined-outer-name
import pytest
import numpy as np
from tests.utils import make_er_graph, add_random_weights
from pathcensus import PathCensus


@pytest.fixture(scope="session")
def unweighted_graph():
    """Get unweighted graph."""
    return make_er_graph(10000, dbar=20)

@pytest.fixture(scope="session")
def weighted_graph(unweighted_graph):
    """Get weighted graph."""
    return add_random_weights(unweighted_graph)


class TestParallelPathCounting:
    """Basic tests of parallelized path counting algorithm(s).
    """
    def test_parallel_unweighted(self, unweighted_graph):
        graph = unweighted_graph
        s1 = PathCensus(graph, parallel=True).counts.values
        s2 = PathCensus(graph, parallel=False).counts.values
        assert np.allclose(s1, s2)

    def test_parallel_weighted(self, weighted_graph):
        graph = weighted_graph
        s1 = PathCensus(graph, parallel=True).counts.values
        s2 = PathCensus(graph, parallel=False).counts.values
        assert np.allclose(s1, s2)
