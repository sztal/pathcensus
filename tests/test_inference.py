"""Unit tests for :class:`pathcensus.inference.Inference`."""
# pylint: disable=redefined-outer-name,too-many-locals
import random
from itertools import product
import pytest
import numpy as np
from pathcensus import PathCensus
from pathcensus.nullmodels import UBCM, UECM
from pathcensus.inference import Inference
from pathcensus.utils import set_seed
from tests.utils import make_er_graph, make_rgg
from tests.utils import get_largest_component, add_random_weights

SEEDS    = (20,)
WEIGHTED = (False, True)
PARAMS   = list(product(SEEDS, WEIGHTED))

def _make_graph(func, seed, weighted=False):
    random.seed(seed)
    set_seed(random=seed)
    graph = get_largest_component(func(50, 7))
    if weighted:
        set_seed(numpy=seed)
        graph = add_random_weights(graph)
        model = UECM
    else:
        model = UBCM
    model = model(graph)
    model.fit()
    model.validate()
    return graph, model, seed

@pytest.fixture(scope="session", params=PARAMS)
def er_graph_inference(request):
    """Inference object for small ER graph."""
    seed, weighted = request.param
    return _make_graph(make_er_graph, seed, weighted=weighted)

@pytest.fixture(scope="session", params=PARAMS)
def rgg_graph_inference(request):
    seed, weighted = request.param
    return _make_graph(make_rgg, seed, weighted=weighted)


class TestInference:
    """Unit test for :class:`pathcensus.inference.Inference`.

    In general the test check whether coefficients are insignificant
    in ER graphs and whether similarity coefficients are significant
    in RGGs.
    """
    @pytest.mark.parametrize("mode", ["nodes", "global"])
    @pytest.mark.parametrize("alpha", [.01, .05])
    def test_er_pvalues(self, er_graph_inference, mode, alpha):
        """Test p-values in an ER graph.

        In general it is expected that there will be no significant
        values.
        """
        graph, model, seed = er_graph_inference

        def stats(graph):
            return PathCensus(graph).coefs(mode)

        inference = Inference(graph, model, stats)
        set_seed(numba=seed)

        if mode == "nodes":
            inference.aggregate_by = "units"

        n = 200 if graph.is_weighted() else 100
        data, null = inference.init_comparison(n)

        pvals = inference.estimate_pvalues(data, null, alpha=alpha)
        if len(pvals) == 1:
            threshold = 1
        else:
            se = alpha*(1-alpha) / np.sqrt(pvals.shape[1])
            threshold = alpha + 1*se
        assert (pvals.values <= alpha).mean() <= threshold

    @pytest.mark.parametrize("mode", ["nodes", "global"])
    @pytest.mark.parametrize("alpha", [.01, .05])
    def test_rgg_pvalues(self, rgg_graph_inference, mode, alpha):
        graph, model, seed = rgg_graph_inference

        def stats(graph):
            return PathCensus(graph).similarity(mode)

        inference = Inference(graph, model, stats)
        set_seed(numba=seed)

        if mode == "nodes":
            inference.aggregate_by = "units"

        n = 200 if graph.is_weighted() else 100
        data, null = inference.init_comparison(n)

        pvals = inference.estimate_pvalues(data, null, alpha=alpha)
        assert (pvals <= alpha).mean() > alpha

    def test_simulate_null_with_seed(self, er_graph_inference):
        """Test effect of passing random seed to :py:mod:`numba`."""
        graph, model, seed = er_graph_inference

        def stats(graph):
            return PathCensus(graph).similarity("nodes", undefined="zero")

        inference = Inference(graph, model, stats)

        set_seed(numba=seed)
        null1 = inference.simulate_null(100)
        set_seed(numba=seed)
        null2 = inference.simulate_null(100)

        assert (null1.index == null2.index).all()
        assert np.allclose(null1, null2)
