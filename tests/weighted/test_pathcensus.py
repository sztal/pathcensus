"""Test weighted path counting methods."""
# pylint: disable=redefined-outer-name,too-few-public-methods
# pylint: disable=too-many-branches
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from pathcensus.definitions import PathDefinitionsWeighted
from pathcensus import PathCensus


@pytest.fixture(scope="session")
def paths_edges(random_graph):
    """Fixture for generating path census data frames
    for edges and node/global counts based `random_graph` fixture.
    """
    _, S = random_graph
    E = S.census("edges")
    return E, S
@pytest.fixture(scope="session")
def paths_edges_nodes(paths_edges):
    """Get edge and node path/cycle counts."""
    E, S = paths_edges
    return E, S.census("nodes")
@pytest.fixture(scope="session")
def paths_edges_global(paths_edges):
    """Get edge and global path/cycle counts."""
    E, S = paths_edges
    return E, S.census("global")

@pytest.fixture(scope="session")
def graph_weights_one(random_graph):
    """Pair of :py:class:`pathcensus.PathCensus` objects for weighted and
    unweighted version of the same graph with all weights equal to ``1``.
    """
    G, _ = random_graph
    G.es["weight"] = np.ones((G.ecount(),))
    P0 = PathCensus(G, weighted=False)
    P1 = PathCensus(G, weighted=True)
    return P0, P1

@pytest.fixture(scope="session")
def graph_weights_uniform(random_graph):
    """Pair of :py:class:`pathcensus.PathCensus` objects for weighted and
    unweighted version of the same graph with all weights being uniform
    but other than ``1``.
    """
    G, _ = random_graph
    G.es["weight"] = 3*np.ones((G.ecount(),))
    P0 = PathCensus(G, weighted=False)
    P1 = PathCensus(G, weighted=True)
    return P0, P1


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
        paths = PathDefinitionsWeighted().get_column_names()

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

            arules = PathDefinitionsWeighted().aggregation.get("nodes", {})
            m1 /= arules.get(path, 1)
            assert np.allclose(m0, m1)

        @pytest.mark.parametrize("path", paths)
        def test_edges_to_global(self, path, paths_edges_global):
            """Check consistency between edge and global counts
            of paths and cycles.
            """
            E, G = paths_edges_global
            m0 = G[path].iloc[0]
            m1 = E[path].sum()

            arules = PathDefinitionsWeighted().aggregation.get("global", {})
            m1 /= arules.get(path, 1)
            assert m0 == approx(m1)


    class TestCountingAgainstOtherImplementations:
        """Test weighted path counting against mean weighted local
        clustering coefficient as defined by Barrat et al.
        and implemented in :py:mod:`igraph`.

        In general, weighted `t`-clustering should be equal to
        the method by Barrat et al.
        """
        @pytest.mark.parametrize("undefined", ["nan", "zero"])
        def test_mean_local_clustering(self, random_graph, undefined):
            G, P = random_graph
            c0 = G.transitivity_avglocal_undirected(weights="weight", mode=undefined)
            c1 = P.tclust(undefined=undefined).mean(skipna=False)
            assert np.isnan([c0, c1]).all() or c0 == approx(c1)


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
            C = P.census(mode)
            tol = 1e-6
            assert (C.values >= 0).all()
            assert (C["twc"] <= C["tw"] + tol).all()
            assert (C["thc"] <= C["th"] + tol).all()
            assert (C["q0wc"] <= C["qw"] + tol).all()
            assert (C["q0hc"] <= C["qh"] + tol).all()

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


    class TestConsistencyWithUnweightedMethods:
        """Test whether weighted counts with uniform weights
        are consistent with the unweighted counts etc.
        """
        @staticmethod
        def to_unweighted(df):
            """Combine weighted counts so they have the same columns
            as unweighted counts.
            """
            return pd.DataFrame({
                "t": (df["twc"] + df["thc"]) / 2,
                "tw": df["tw"],
                "th": df["th"],
                "q0": (df["q0wc"] + df["q0hc"]) / 2,
                "qw": df["qw"],
                "qh": df["qh"]
            })

        @pytest.mark.parametrize("mode", ["edges", "nodes", "global"])
        def test_path_counts_consistency(self, graph_weights_one, mode):
            """Test consistency of path counts."""
            P0, P1 = graph_weights_one
            assert P1.weighted
            p0 = P0.census(mode)
            p1 = self.to_unweighted(P1.census(mode))
            assert np.allclose(p0.values, p1.values)

        @pytest.mark.parametrize("mode", ["edges", "nodes", "global"])
        def test_coefs_consistency(self, graph_weights_uniform, mode):
            """Test consistency of coefficients."""
            P0, P1 = graph_weights_uniform
            assert P1.weighted
            c0 = P0.coefs(mode, undefined="zero")
            c1 = P1.coefs(mode, undefined="zero")
            assert np.allclose(c0.values, c1.values)


    class TestSimpleMotifs:
        """Test agreement with counts expected for simple motifs
        such as triangle, quadrangle and star.
        """
        simcoefs = ("sim_g", "sim", "tclust", "tclosure")
        compcoefs = ("comp_g", "comp", "qclust", "qclosure")

        def approx_in(self, obj, vals, allow_nan=False, **kwds):
            """Auxiliary method for approximate testing if
            values in ``objs`` are in ``vals``.
            """
            x = obj.values
            l = np.zeros_like(x, dtype=bool)
            for val in vals:
                if allow_nan:
                    l |= np.isnan(x) | np.isclose(x, val, **kwds)
                else:
                    l |= np.isclose(x, val, **kwds)
            return l.all()

        def approx_between(self, obj, lo, hi, allow_nan=False, tol=1e-6):
            """Auxiliary method for approximate testing if
            valuesin ``obj`` are between ``lo`` and ``hi``.
            """
            x = obj.values
            l = np.isnan(x) if allow_nan else np.zeros_like(x, dtype=bool)
            return (l | (x >= lo-tol) | (x <= hi+tol)).all()

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
