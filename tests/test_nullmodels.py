"""Unit tests for :mod:`pathcensus.nullmodels`."""
# pylint: disable=redefined-outer-name
import random
from itertools import product
import pytest
import numpy as np
from pathcensus.nullmodels import UBCM, UECM
from pathcensus.utils import rowsums, set_numba_seed
from pathcensus.utils import relclose
from tests.utils import make_er_graph, make_rgg, add_random_weights
from tests.utils import get_largest_component

FAMILY = ("erdos_renyi", "geometric")
SEEDS = (20, 40)

_params  = list(product(FAMILY, SEEDS))
_methods = ("newton", "fixed-point")
_ubcm_params = list(product(["cm_exp", "cm"], _methods))
_uecm_params = list(product(["ecm_exp", "ecm"], _methods))

@pytest.fixture(scope="session", params=_params)
def small_graph(request):
    """Generate some small graphs (ER and RGG)."""
    family, seed = request.param
    random.seed(seed)
    if family == "geometric":
        graph = get_largest_component(make_rgg(50, 5))
    else:
        graph = get_largest_component(make_er_graph(50, 5))
    return graph, seed

@pytest.fixture(scope="session", params=_ubcm_params)
def small_graph_ubcm(request, small_graph):
    """Generate some small graphs (ER and RGG)."""
    model, method = request.param
    graph, seed = small_graph
    ubcm = UBCM(graph)
    ubcm.fit(model, method)
    return ubcm, seed, graph

@pytest.fixture(scope="session", params=_uecm_params)
def small_graph_uecm(request, small_graph):
    model, method = request.param
    graph, seed = small_graph
    np.random.seed(seed)
    graph = add_random_weights(graph)
    uecm = UECM(graph)
    uecm.fit(model, method)
    return uecm, seed, graph


class TestUBCM:
    """Unit tests for Unweighted Binary Configuration Model."""
    def test_ubcm(self, small_graph_ubcm):
        """Test whether the expected degree sequence in UBCM approximates
        the observed sequence.
        """
        ubcm, *_ = small_graph_ubcm
        rtol = 1e-6 if ubcm.fit_args["method"] == "newton" else 1e-3

        assert ubcm.is_fitted()
        assert ubcm.is_valid(rtol)

        P = ubcm.get_P(dense=True)
        assert relclose(P.sum(axis=1), ubcm.D, rtol=rtol)

    def test_ubcm_sampling(self, small_graph_ubcm):
        """Test convergence of the average over degree sequences sampled
        from UBCM towards the observed sequence.
        """
        ubcm, seed, _ = small_graph_ubcm
        rtol = 1e-1 if ubcm.fit_args["method"] == "newton" else 1e-1

        D = ubcm.D
        E = np.zeros_like(D, dtype=float)
        n = 1000

        set_numba_seed(seed)

        for rand in ubcm.sample(n):
            E += rowsums(rand)

        E = E / n
        assert relclose(D, E, rtol=rtol)

    def test_ubcm_seed(self, small_graph_ubcm):
        """Test if setting random seed for sampling works correctly."""
        ubcm, seed, _ = small_graph_ubcm

        set_numba_seed(seed)
        A1 = ubcm.sample_one()
        set_numba_seed(seed)
        A2 = ubcm.sample_one()
        assert (A1 != A2).count_nonzero() == 0


class TestUECM:
    """Unit tests for Unweighted Enhanced Configuration Model."""
    def test_uecm(self, small_graph_uecm):
        """Test whether the expected degree and strength sequences in UECM
        approximate the observed sequences.
        """
        uecm, *_ = small_graph_uecm
        rtol = 1e-1 if uecm.fit_args["method"] == "newton" else 2e-1

        assert uecm.is_fitted()
        assert uecm.is_valid(rtol)

        P = uecm.get_P(dense=True)
        W = uecm.get_W(dense=True)
        assert relclose(P.sum(axis=1), uecm.D, rtol=rtol)
        assert relclose(W.sum(axis=1), uecm.S, rtol=rtol)

    def test_uecm_sampling(self, small_graph_uecm):
        """Test convergence of the averages over degree and strength
        sequences sampled from UECM towards the observed sequences.
        """
        uecm, seed, _ = small_graph_uecm
        rtol = 1e-1 if uecm.fit_args["method"] == "newton" else 2e-1

        D = uecm.D
        S = uecm.S

        ED = np.zeros_like(D, dtype=float)
        ES = np.zeros_like(S, dtype=float)

        n = 1000
        set_numba_seed(seed)

        for rand in uecm.sample(n):
            ES += rowsums(rand)
            rand.data[:] = 1
            ED += rowsums(rand)

        ED /= n
        ES /= n

        assert relclose(D, ED, rtol=rtol)
        assert relclose(S, ES, rtol=rtol)
