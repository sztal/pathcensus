"""Simple JIT-compiled graph class for calculating path census."""
from typing import Optional, Tuple
import numpy as np
import numba
from numba.typed import List   # pylint: disable=no-name-in-module
from numba.experimental import jitclass
from .types import UInt, Float
from ..definitions import PathDefinitionsUnweighted, PathDefinitionsWeighted


_NPATHS_UNWEIGHTED = PathDefinitionsUnweighted().npaths
_NPATHS_WEIGHTED   = PathDefinitionsWeighted().npaths


@jitclass([
    ("E", UInt[:, ::1]),
    ("D", UInt[::1]),
    ("W", numba.optional(Float[::1])),
    ("S", numba.optional(Float[::1])),
    ("_strides", UInt[::1])
])
class Graph:
    """Graph represented as an edgelist. The edgelist representation
    is efficient for edge-oriented algorithms such as path census.

    Attributes
    ----------
    n_nodes
        Number of nodes in the graph.
    E
        Edgelist as 2D C-contiguous array of
        unsigned 64-bit integers. It has to be sorted
        by head indices.
    W
        Optional 1D array of 64-bit floats storing edge weights.
    """
    def __init__(
        self,
        n_nodes: int,
        E: np.ndarray,
        W: Optional[np.ndarray] = None
    ) -> None:
        # Sort edge array so i <= j
        # and save sorting order indices for sorting weight array
        o1, o2 = self._get_ij_ordering(E)
        E = E[o1][o2]

        # Add edge ids
        eid = np.arange(len(E), dtype=UInt)
        _E = np.ascontiguousarray(np.column_stack((eid, E)))
        self.E = _E

        # Make strides array for efficient access
        # to node neighborhoods.
        self._strides = self._make_strides(n_nodes, self.E)

        # Set node degree array
        self.D = self.degree()

        if W is not None:
            if self.E.shape[0] != W.shape[0]:
                raise AttributeError("'E' and 'W' have to be of the same length")
            _W = W[o1][o2]
            self.W = np.ascontiguousarray(_W)
            # Set node strength array
            self.S = self.strength()
        else:
            self.W = None
            self.S = None

    # Properties --------------------------------------------------------------

    @property
    def vcount(self):
        """Number of nodes."""
        return len(self._strides) - 1
    @property
    def n_nodes(self):
        return self.vcount

    @property
    def ecount(self):
        """Number of edges."""
        return len(self.E) // 2
    @property
    def n_edges(self):
        return self.ecount

    @property
    def vids(self):
        return np.arange(self.vcount, dtype=UInt)

    @property
    def eids(self):
        return np.arange(self.ecount, dtype=UInt)

    @property
    def directed(self) -> bool:
        return False

    @property
    def weighted(self) -> bool:
        return self.W is not None

    @property
    def npaths(self) -> int:
        if self.weighted:
            return _NPATHS_WEIGHTED
        return _NPATHS_UNWEIGHTED

    # Methods -----------------------------------------------------------------

    def get_edges(self) -> np.ndarray:
        """Get undirected edges without self-loops (`i < j`)."""
        E = self.E
        mask = E[:, 1] < E[:, 2]
        return E[mask]

    def get_min_di_edges(self) -> np.ndarray:
        """Get undirected edges without self loops (`di <= dj`)."""
        E  = self.E
        D  = self.D
        i  = E[:, 1]
        j  = E[:, 2]
        di = D[i]
        dj = D[j]

        mask = (di < dj) | ((di == dj) & (i < j))
        return E[mask]

    def degree(self) -> np.ndarray:
        """Get node degrees."""
        return self._strides[1:] - self._strides[:-1]

    def strength(self) -> np.ndarray:
        """Get node strengths."""
        S = np.empty((self.vcount,), dtype=Float)
        for i, seq in enumerate(zip(self._strides[:-1], self._strides[1:])):
            start, end = seq
            S[i] = self.W[start:end].sum()
        return S

    def N(self, i: int) -> np.ndarray:
        """Get 1-neighborhood of ``i``."""
        start, end = self._strides[i:i+2]
        return self.E[start:end, np.array([0, 2])]

    # Path counting methods ---------------------------------------------------

    def _count_paths_unweighted(
        self,
        E: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Count unweighted paths.

        Parameters
        ----------
        E
            Array with edge and source/target indices
            to consider in the calculations.

        Returns
        -------
        E
            2D array of edges as source and target indices (only i < j).
        counts
            2D array with path counts per edge.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        counts = np.empty((len(E), self.npaths), dtype=Float)
        # Array for keeping track of node roles
        role = np.zeros((self.n_nodes,), dtype=np.uint8)

        D  = self.D   # degree sequence

        # Main loop
        for idx, edge in enumerate(E):
            _, i, j = edge

            _zero = Float(0)

            t = tw = th = _zero
            q = qw = qh = _zero

            star_i = set(List.empty_list(UInt))
            star_j = set(List.empty_list(UInt))
            tri_ij = set(List.empty_list(UInt))

            for _, k in self.N(i):
                if k == j:
                    continue

                star_i.add(k)
                role[k] = 1
                # Count wedge triples
                tw += 1

            for _, k in self.N(j):
                if k == i:
                    continue

                if role[k] == 1:
                    # Count triangles
                    t += 1
                    star_i.remove(k)
                    tri_ij.add(k)
                    role[k] = 3
                else:
                    star_j.add(k)
                    role[k] = 2

                # Count head triples
                th += 1

            for k in star_i:
                for _, l in self.N(k):
                    if l == i:
                        continue
                    # Count strong quadrangles
                    if role[l] == 2:
                        q += 1

            # Count wedge and head quadruples
            # and clear `role` vector.
            for k in star_i:
                qw += D[k] - 1
                role[k] = 0
            for k in star_j:
                qh += D[k] - 1
                role[k] = 0
            for k in tri_ij:
                n = D[k] - 2
                qw += n
                qh += n
                role[k] = 0

            counts[idx] = (
                t, tw, th,
                q, qw, qh
            )

        return E, counts

    def _count_paths_weighted(
        self,
        E: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Count weighted paths.

        Parameters
        ----------
        E
            Array with edge and source/target indices
            to consider in the calculations.

        Returns
        -------
        E
            2D array of edges as source and target indices (only i < j).
        counts
            2D array with weighted path counts per edge.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        counts = np.zeros((len(E), self.npaths), dtype=Float)
        # Array for keeping track of node roles
        role = np.zeros((self.n_nodes,), dtype=np.uint8)

        D = self.D    # degree sequence
        S = self.S    # strength sequence
        W = self.W    # edge weights array

        # Weight arrays for links from `i` and `j` to their neighbors
        Wi = np.empty((self.n_nodes,), dtype=Float)
        Wj = np.empty((self.n_nodes,), dtype=Float)

        # Main loop
        for idx, edge in enumerate(E):
            ij, i, j = edge
            wij = W[ij]

            _zero = Float(0)

            twc = thc = tw = th = _zero
            q0wc = q0hc = qw = qh = _zero

            star_i = set(List.empty_list(UInt))
            star_j = set(List.empty_list(UInt))
            tri_ij = set(List.empty_list(UInt))

            for ik, k in self.N(i):
                if k == j:
                    continue
                star_i.add(k)
                role[k] = 1
                wik = W[ik]
                Wi[k] = wik
                # Count wedge triples
                w = (wij + wik) / 2
                tw += w

            for jk, k in self.N(j):
                if k == i:
                    continue
                wjk = W[jk]
                Wj[k] = wjk
                w = (wij + wjk) / 2
                # Handle triangles
                if role[k] == 1:
                    star_i.remove(k)
                    tri_ij.add(k)
                    role[k] = 3
                    # Count triangles
                    # both with wedge and head weighting
                    twc += (wij + Wi[k]) / 2
                    thc += w
                else:
                    star_j.add(k)
                    role[k] = 2
                # Count head triples
                th += w

            for k in star_i:
                wik = Wi[k]
                for kl, l in self.N(k):
                    if l == i:
                        continue
                    # Count strong quadrangles
                    if role[l] == 2:
                        wkl = W[kl]
                        wjl = Wj[l]
                        q0wc += (wij + wik + wkl) / 3
                        q0hc += (wij + wjl + wkl) / 3

            # Count wedge and head quadruples of `i`
            # and clear `role` vector.
            for k in star_i:
                wik = Wi[k]
                qw += ((D[k]-1)*(wij+wik) + S[k] - wik) / 3
                role[k] = 0
            for k in star_j:
                wjk = Wj[k]
                qh += ((D[k]-1)*(wij+wjk) + S[k] - wjk) / 3
                role[k] = 0
            for k in tri_ij:
                wik = Wi[k]
                wjk = Wj[k]
                dk = D[k]
                sk = S[k]
                qw += ((dk-2)*(wij+wik) + sk - wjk - wik) / 3
                qh += ((dk-2)*(wij+wjk) + sk - wjk - wik) / 3
                role[k] = 0

            counts[idx] = (
                twc, thc, tw, th,
                q0wc, q0hc, qw, qh
            )

        return E, counts

    def count_paths(
        self,
        E: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Count paths.

        Parameters
        ----------
        E
            Array with edge and source/target indices
            to consider in the calculations.

        Returns
        -------
        E
            2D array of edges as source and target indices (only i < j).
        paths
            2D array with weighted path counts per edge.
        """
        if self.weighted:
            E, counts = self._count_paths_weighted(E)
        else:
            E, counts = self._count_paths_unweighted(E)
        return (
            np.ascontiguousarray(E[:, 1:]),
            np.ascontiguousarray(counts)
        )

    # Internals ---------------------------------------------------------------

    def _make_strides(self, n_nodes, E):
        strides = np.zeros((n_nodes+1,), dtype=UInt)
        last = eid = UInt(0)
        for eid, i in E[:, :2]:
            diff = i - last
            if diff > 0:
                strides[i] = eid
                if diff > 1:
                    for k in range(last+1, i):
                        strides[k] = eid
            last = i
        for k in range(last+1, len(strides)):
            strides[k] = len(E)
        return strides

    def _get_ij_ordering(self, E):
        o1 = np.argsort(E[:, 1])
        o2 = np.argsort(E[o1, 0])
        return o1, o2
