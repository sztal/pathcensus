"""Shared configuration for unit tests."""
import random
from itertools import product
import pytest
from tests.utils import make_er_graph, make_rgg
from tests.utils import make_triangle, make_quadrangle


FAMILY = ("erdos_renyi", "geometric")
VCOUNTS = (20, 100, 500)
KBARS = (2, 10)
RANDOM_SEEDS = (1, 10)
MOTIFS = (
    "triangle",
    "quadrangle",
)

_params = list(product(FAMILY, VCOUNTS, KBARS, RANDOM_SEEDS))

@pytest.fixture(scope="session", params=_params)
def base_random_graph(request):
    """Fixture for generating multiple Erdős–Rényi or geometric
    random graphs with different numbers of nodes and average degrees equal
    to ``10`` or ``20``. They are automatically passed to all test methods
    for testing consistency between different methods of path counting.
    """
    family, n, dbar, seed = request.param
    random.seed(seed)
    if family == "geometric":
        graph = make_rgg(n, dbar)
    else:
        graph = make_er_graph(n, dbar)
    if dbar <= 5:
        # Add isolated node with the last id
        # to test degree calculations in 'Graph' class
        graph.add_vertex()
    return graph


@pytest.fixture(scope="session", params=MOTIFS)
def base_simple_motif(request):
    """Fixture for generating small graphs with simple motifs
    (e.g. triangles and quadrangles) for testing correctness
    of path/motif counting routines and relational coefficients.
    """
    motif = request.param
    if motif == "triangle":
        G = make_triangle()
    elif motif == "quadrangle":
        G = make_quadrangle(weak=0)
    else:
        raise ValueError(f"unknown motif '{motif}'")
    return motif, G
