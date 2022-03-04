"""Test of the graph class."""
# pylint: disable=redefined-outer-name
import pytest
from numpy.testing import assert_array_almost_equal
from pathcensus import PathCensus
from tests.utils import add_random_weights


@pytest.fixture(scope="session")
def weighted_random_graph(base_random_graph):
    return add_random_weights(base_random_graph)


class TestGraphConversion:
    """Unit tests for graph converters."""
    def test_networkx_unweighted(self, base_random_graph):
        G  = base_random_graph  # graph in `igraph` format
        N  = G.to_networkx()
        P1 = PathCensus(G)
        P2 = PathCensus(N)
        assert_array_almost_equal(P1.counts.values, P2.counts.values)

    def test_networkx_weighted(self, weighted_random_graph):
        G  = weighted_random_graph  # graph in `igraph` format
        N  = G.to_networkx()
        P1 = PathCensus(G)
        P2 = PathCensus(N)
        assert_array_almost_equal(P1.counts.values, P2.counts.values)
