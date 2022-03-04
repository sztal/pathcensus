"""Test of the graph class."""
# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from pathcensus import PathCensus
from tests.utils import add_random_weights


@pytest.fixture(scope="session")
def weighted_random_graph(base_random_graph):
    """Get weighted random graph."""
    return add_random_weights(base_random_graph)


class TestGraph:
    """Unit tests for ``Graph`` class used for path/cycle counting."""

    def test_degree(self, base_random_graph):
        """Test degree sequence getter."""
        G = base_random_graph
        D0 = np.array(G.degree())
        D1 = PathCensus.get_graph(G).degree()
        assert np.allclose(D0, D1)

    def test_strength(self, weighted_random_graph):
        """Test strength sequence getter."""
        G = weighted_random_graph
        S0 = np.array(G.strength(weights="weight"))
        S1 = PathCensus.get_graph(G).strength()
        assert np.allclose(S0, S1)
