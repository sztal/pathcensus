"""Configuration for weighted path counting tests."""
# pylint: disable=cyclic-import
import pytest
import numpy as np
from pathcensus import PathCensus
from tests.utils import add_random_weights

_SEEDS = (324, 7171)

@pytest.fixture(scope="session", params=_SEEDS)
def random_graph(base_random_graph, request):
    """Get graph with its corresponding path census."""
    seed = request.param
    graph = base_random_graph
    np.random.seed(seed)
    graph = add_random_weights(graph)
    return graph, PathCensus(graph)


@pytest.fixture(scope="session", params=_SEEDS)
def simple_motif(base_simple_motif, request):
    """Get motif name and its corresponding path census."""
    seed = request.param
    motif, G = base_simple_motif
    np.random.seed(seed)
    G = add_random_weights(G)
    return motif, PathCensus(G)
