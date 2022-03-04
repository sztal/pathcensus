"""Configuration for unweighted path counting tests."""
import pytest
from pathcensus import PathCensus


@pytest.fixture(scope="session")
def random_graph(base_random_graph):
    """Get graph with its corresponding path census."""
    graph = base_random_graph
    return graph, PathCensus(graph)


@pytest.fixture(scope="session")
def simple_motif(base_simple_motif):
    """Get motif name and its corresponding path census."""
    motif, G = base_simple_motif
    return motif, PathCensus(G)
