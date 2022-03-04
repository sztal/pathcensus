"""Test unweighted path counting methods."""
# pylint: disable=redefined-outer-name,too-few-public-methods
# pylint: disable=too-many-branches
from collections import defaultdict
import pytest
from pytest import approx
import numpy as np
from pathcensus.definitions import PathDefinitionsUnweighted
from pathcensus.utils import rowsums


@pytest.fixture(scope="session")
def paths_edges(random_graph):
    """Fixture for generating path census data frames
    for edges and node/global counts based `random_graph` fixture.
    """
    _, P = random_graph
    E = P.census("edges")
    return E, P
@pytest.fixture(scope="session")
def paths_edges_nodes(paths_edges):
    """Get edge and node path/cycle counts."""
    E, P = paths_edges
    return E, P.census("nodes")
@pytest.fixture(scope="session")
def paths_edges_global(paths_edges):
    """Get edge and global path/cycle counts."""
    E, P = paths_edges
    return E, P.census("global")

@pytest.fixture(scope="session")
def triangle_counts(random_graph):
    """Get graph, path census and triangles enumerated with :mod:`igraph`."""
    G, P = random_graph
    T = np.array(G.cliques(min=3, max=3))
    return G, T, P


class TestPathCounting:
    """Tests of different path counting methods.

    All main path counting methods are defined for overall graph counts,
    node counts and node-pair (edge) counts. The below tests check whether
    the results of all different counting methods are consistent in a sense
    that they give the same answers after proper summing.
    """
    class TestAggregationConsistency:
        """Tests of aggregation consistency between edge, node
        and global counts.
        """
        paths = PathDefinitionsUnweighted().get_column_names()

        @pytest.mark.parametrize("path", paths)
        def test_edges_to_nodes(self, path, paths_edges_nodes):
            """Check consistency between edge and node counts
            of paths and cycles.
            """
            E, N = paths_edges_nodes
            m0 = N[path].dropna()
            m1 = E[path].groupby(level="i").sum() \
                .reindex(N.index) \
                .fillna(0)

            arules = PathDefinitionsUnweighted().aggregation.get("nodes", {})
            m1 /= arules.get(path, 1)
            assert (m0 == m1).all()

        @pytest.mark.parametrize("path", paths)
        def test_edges_to_global(self, path, paths_edges_global):
            """Check consistency between edge and global counts
            of paths and cycles.
            """
            E, G = paths_edges_global
            m0 = G[path].iloc[0]
            m1 = E[path].sum()

            arules = PathDefinitionsUnweighted().aggregation.get("global", {})
            m1 /= arules.get(path, 1)
            assert m0 == m1


    class TestSimpleMotifs:
        """Test agreement with counts expected for simple motifs
        such as triangle, quadrangle and star.
        """
        simcoefs = ("sim_g", "sim", "tclust", "tclosure")
        compcoefs = ("comp_g", "comp", "qclust", "qclosure")

        def approx_in(self, obj, vals, allow_nan=False, **kwds):
            """Auxiliary method for approximate testing if
            values in ``obj`` are in ``vals``.
            """
            x = obj.values
            l = np.zeros_like(x, dtype=bool)
            for val in vals:
                if allow_nan:
                    l |= np.isnan(x) | np.isclose(x, val, **kwds)
                else:
                    l |= np.isclose(x, val, **kwds)
            return l.all()

        @pytest.mark.parametrize("undefined", ["nan", "zero"])
        def test_simple_motifs_global(self, simple_motif, undefined):
            """Check values of global structural coefficients
            in simple motifs.
            """
            motif, P = simple_motif
            kwds = dict(undefined=undefined)

            sim  = P.simcoefs("global", **kwds)
            comp = P.compcoefs("global", **kwds)

            if motif == "triangle":
                assert self.approx_in(sim, [1])
                assert self.approx_in(comp, [0], allow_nan=True)

            elif motif == "quadrangle":
                assert self.approx_in(sim, [0])
                assert self.approx_in(comp, [1])

        @pytest.mark.parametrize("undefined", ["nan", "zero"])
        def test_simple_motifs_nodes(self, simple_motif, undefined):
            """Check values of node-wise structural coefficients
            in simple motifs.
            """
            motif, P = simple_motif
            kwds = dict(undefined=undefined)

            sim  = P.simcoefs("nodes", **kwds)
            comp = P.compcoefs("nodes", **kwds)

            if motif == "triangle":
                assert self.approx_in(sim, [1])
                assert self.approx_in(comp, [0], allow_nan=True)

            elif motif == "quadrangle":
                assert self.approx_in(sim, [0])
                assert self.approx_in(comp, [1])

        @pytest.mark.parametrize("undefined", ["nan", "zero"])
        def test_simple_motifs_edges(self, simple_motif, undefined):
            """Check values of edge-wise structural coefficients
            in simple motifs.
            """
            motif, P = simple_motif
            kwds = dict(undefined=undefined)

            sim  = P.similarity("edges", **kwds)
            comp = P.complementarity("edges", **kwds)

            if motif == "triangle":
                assert self.approx_in(sim, [1])
                assert self.approx_in(comp, [0], allow_nan=True)

            elif motif == "quadrangle":
                assert self.approx_in(sim, [0])
                assert self.approx_in(comp, [1])

    class TestCountingAgainstOtherImplementations:
        """Test path counting against triangle counting methods
        implemented in :py:mod:`igraph` as well as naive implementations
        using :py:mod:`numpy` arrays.
        """
        def test_triangles_edges(self, triangle_counts):
            """Test triangle counts for edges against
            :py:mod:`igraph` implementation.
            """
            _, T, P = triangle_counts
            t1 = P.tdf["t"].to_dict()
            t0 = defaultdict(lambda: 0)
            for i, j, k in T:
                for key in [(i, j), (i, k), (j, k)]:
                    u, v = key
                    for link in [(u, v), (v, u)]:
                        t0[link] += 1
            for link in t1:
                if link not in t0:
                    t0[link] = 0
            t0 = dict(t0)
            assert t0 == t1

        def test_triangles_nodes(self, triangle_counts):
            """Test triangle counts for nodes against
            :py:mod:`igraph` implementation.
            """
            _, T, P = triangle_counts
            t1 = (P.tdf["t"].groupby(level="i").sum() // 2).to_dict()
            t0 = defaultdict(lambda: 0)
            for triple in T:
                for i in triple:
                    t0[i] += 1
            for i in t1:
                if i not in t0:
                    t0[i] = 0
            t0 = dict(t0)
            assert t0 == t1

        def test_triangles_global(self, triangle_counts):
            """Test global triangle counts
            against :py:mod:`igraph` implementation.
            """
            _, T, P = triangle_counts
            t0 = len(T)
            t1 = P.tdf["t"].sum() / 6
            assert t0 == t1

        @pytest.mark.parametrize("undefined", ["nan", "zero"])
        def test_clustering_local(self, random_graph, undefined):
            """Test local clustering coefficient calculations
            against the :py:mod:`igraph` implementation.
            """
            G, M = random_graph
            t0 = np.array(G.transitivity_local_undirected(mode=undefined))
            t1 = M.tclust(undefined=undefined).values
            nan0 = np.isnan(t0)
            nan1 = np.isnan(t1)
            assert np.array_equal(nan0, nan1)
            assert np.allclose(t0[~nan0], t1[~nan1])

        @pytest.mark.parametrize("undefined", ["nan", "zero"])
        def test_clustering_global(self, random_graph, undefined):
            """Test global clustering coefficient calculations
            against the :py:mod:`igraph` implementation.
            """
            G, M = random_graph
            t0 = G.transitivity_undirected(mode=undefined)
            t1 = M.similarity("global", undefined=undefined)
            assert t0 == approx(t1)

        def test_node_paths_wedge_triples(self, random_graph):
            """Test against naive :py:mod:`numpy` implementation."""
            G, M = random_graph
            k = np.array(G.degree())
            t0 = k*(k-1)
            t1 = M.tdf["tw"].groupby(level="i").sum() \
                .reindex(np.arange(G.vcount())) \
                .fillna(0)
            assert np.array_equal(t0, t1)

        def test_node_paths_head_triples(self, random_graph):
            """Test against naive :py:mod:`numpy` implementation."""
            G, M = random_graph
            A = G.get_adjacency_sparse()
            k = np.array(G.degree())
            t0 = A@(k-1)
            t1 = M.tdf["th"].groupby(level="i").sum() \
                .reindex(np.arange(G.vcount())) \
                .fillna(0)
            assert np.array_equal(t0, t1)

        def test_node_paths_wedge_quadruples(self, random_graph):
            """Test against naive :py:mod:`numpy` implementation."""
            G, M = random_graph
            A = G.get_adjacency_sparse()
            T = (A@A).multiply(A)
            k = np.array(G.degree())
            q0 = (k-1)*(A@(k-1)) - rowsums(T)
            q1 = M.qdf["qw"].groupby(level="i").sum() \
                .reindex(np.arange(G.vcount())) \
                .fillna(0)
            assert np.array_equal(q0, q1)

        def test_node_paths_head_quadruples(self, random_graph):
            """Test against naive :py:mod:`numpy` implementation."""
            G, M = random_graph
            P = M.census("edges")
            A = G.get_adjacency_sparse()
            T = (A@A).multiply(A)
            k = np.array(G.degree())
            k2 = A@(k-1)
            q0 = A@k2 - k*(k-1) - rowsums(T)
            q1 = P["qh"].groupby(level="i").sum() \
                .reindex(np.arange(G.vcount())) \
                .fillna(0)
            assert np.array_equal(q0, q1)

        def test_tclust(self, random_graph):
            """Test against naive :py:mod:`numpy` implementation."""
            G, M = random_graph
            A = G.get_adjacency_sparse()
            T = (A@A).multiply(A)
            k = np.array(G.degree())
            tw = k*(k-1) // 2
            t = rowsums(T) // 2
            tclust0 = np.array(G.transitivity_local_undirected(mode="zero"))
            tclust1 = t / np.where(tw == 0, 1, tw)
            tclust2 = M.tclust(undefined="zero").values
            assert np.allclose(tclust0, tclust1) and np.allclose(tclust1, tclust2)

        def test_tclosure(self, random_graph):
            """Test against naive :py:mod:`numpy` implementation."""
            G, M = random_graph
            A = G.get_adjacency_sparse()
            T = (A@A).multiply(A)
            k = np.array(G.degree())
            th = A@(k-1)
            t = rowsums(T) // 2
            tclo0 = 2*t / np.where(th == 0, 1, th)
            tclo1 = M.tclosure(undefined="zero").values
            assert np.allclose(tclo0, tclo1)

        def test_similarity(self, random_graph):
            """Test against naive :py:mod:`numpy` implementation."""
            G, M = random_graph
            A = G.get_adjacency_sparse()
            T = (A@A).multiply(A)
            k = np.array(G.degree())
            tw = k*(k-1)
            th = A@(k-1)
            t = rowsums(T) // 2
            tclust = 2*t / np.where(tw == 0, 1, tw)
            tclo = 2*t / np.where(th == 0, 1, th)
            with np.errstate(invalid="ignore", divide="ignore"):
                sim0 = (tw*tclust + th*tclo) / (tw + th)
            sim0[np.isnan(sim0)] = 0
            sim1 = M.similarity("nodes", undefined="zero")
            assert np.allclose(sim0, sim1)


    class TestConsistencyBounds:
        """Test consistency in terms of bounds between open
        and closed paths. In particular, closed paths (e.g. triangles)
        cannot be more frequent than their open counterparts.
        Moreover, relational coefficients (similarity and complementarity)
        must be bounded between their min/max of their corresponding
        clustering and closure coefficients.
        """
        @pytest.mark.parametrize("mode", ["edges", "nodes", "global"])
        def test_path_counts_consistency(self, random_graph, mode):
            _, P = random_graph
            P = P.census(mode)
            assert (P.values >= 0).all()
            assert (P["t"] <= P["tw"]).all()
            assert (P["t"] <= P["th"]).all()
            assert (P["q0"] <= P["qw"]).all()
            assert (P["q0"] <= P["qh"]).all()

        @pytest.mark.parametrize("mode", ["edges", "nodes", "global"])
        def test_similarity_coefs_consistency(self, random_graph, mode):
            _, P = random_graph
            C = P.coefs(mode).dropna()
            vals = C.values
            assert (vals >= -1e-6).all() and (vals <= 1+1e-6).all()
            if mode == "nodes":
                m0 = C[["tclust", "tclosure"]].min(axis=1)
                m1 = C[["tclust", "tclosure"]].max(axis=1)
                assert (C["sim"].between(m0, m1)).all()

        @pytest.mark.parametrize("mode", ["edges", "nodes", "global"])
        def test_complementarity_coefs_consistency(self, random_graph, mode):
            _, P = random_graph
            C = P.coefs(mode).dropna()
            vals = C.values
            assert (vals >= -1e-6).all() and (vals <= 1+1e-6).all()
            if mode == "nodes":
                m0 = C[["qclust", "qclosure"]].min(axis=1)
                m1 = C[["qclust", "qclosure"]].max(axis=1)
                assert (C["comp"].between(m0, m1)).all()
