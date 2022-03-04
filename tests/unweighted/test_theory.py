"""Tests of theoretical results."""
# pylint: disable=redefined-outer-name,cyclic-import
import pytest
import numpy as np
from pathcensus import PathCensus
from tests.utils import get_largest_component


@pytest.fixture(scope="session")
def random_graph_connected(random_graph):
    G, _ = random_graph
    G = get_largest_component(G)
    P = PathCensus(G)
    return G, P


class TestTheory:
    """Test various theoretical results concerning
    local similarity and complementarity coefficients.
    """
    @staticmethod
    def weighted_average(x, w):
        m = np.isnan(x)
        if m.all():
            return 0
        x = x[~m]
        w = w[~m]
        return (x * w).sum() / w.sum()

    def test_similarity_node_edge_sum(self, random_graph_connected):
        """Test whether node similarity is a weighted average
        of corresponding edge similarities.
        """
        _, P = random_graph_connected
        edge = P.simcoefs("edges", census=True, undefined="nan") \
            .groupby(level="i") \
            .apply(lambda df: \
                self.weighted_average(df["sim"], df["tw"] + df["th"])
            )
        node = P.similarity("nodes", undefined="zero")
        assert np.allclose(edge, node)

    def test_similarity_node_edge_minmax_bounds(self, random_graph_connected):
        """Test whether node similarity is bounded between
        minimum and maximum edge similarity.
        """
        _, P = random_graph_connected
        gdf = P.similarity("edges").groupby(level="i").agg([min, max])
        s_node = P.similarity("nodes", undefined="undefined")
        s_emin = gdf["min"]
        s_emax = gdf["max"]
        assert s_node.between(s_emin, s_emax).all()

    def test_complementarity_node_edge_sum(self, random_graph_connected):
        """Test whether node complementarity is a weighted average
        of corresponding edge complementairyt coefficients.
        """
        _, P = random_graph_connected
        edge = P.compcoefs("edges", census=True, undefined="nan") \
            .groupby(level="i") \
            .apply(lambda df: \
                self.weighted_average(df["comp"], df["qw"] + df["qh"])
            )
        node = P.complementarity("nodes", undefined="zero")
        assert np.allclose(edge, node)

    def test_complementarity_node_edge_minmax_bounds(self, random_graph_connected):
        """Test whether node complementarity is bounded between
        minimum and maximum edge complementarity.
        """
        _, P = random_graph_connected
        gdf = P.complementarity("edges").groupby(level="i").agg([min, max])
        c_node = P.complementarity("nodes", undefined="zero")
        c_emin = gdf["min"]
        c_emax = gdf["max"]
        assert c_node.between(c_emin, c_emax).all()
