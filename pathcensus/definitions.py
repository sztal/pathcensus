"""Path counting and aggregation definitions."""
from typing import List, Tuple, Dict, Iterable


class PathDefinitions:
    """Base class for path definitions.

    See Also
    --------
    pathcensus.definitions.PathDefinitionsUnweighted : unweighted definitions
    pathcensus.definitions.PathDefinitionsWeighted : weighted definitions
    """
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            instance = super().__new__(cls)
            instance.__init__()
            cls.__instance = instance
        return cls.__instance

    def __iter__(self) -> Iterable[str]:
        yield from self.definitions["sim"]
        yield from self.definitions["comp"]

    def __getitem__(self, key):
        return self.definitions[key]

    @property
    def definitions(self) -> Dict:
        """Raw path definitions. Weighted and unweighted definitions
        are derived from this.
        """
        return dict(
            sim=("twc", "thc", "tw", "th"),
            comp=("q0wc", "q0hc", "qw", "qh"),
            swap=(
                ("twc", "thc"), ("tw", "th"),
                ("q0wc", "q0hc"), ("qw", "qh"),
            )
        )

    @property
    def aggregation(self) -> Dict:
        """Aggregation rules for computing node and global counts
        from edge counts. The integers specify the factor to divide
        by the sum over edge counts to get a corresponding node/global count.
        """
        tri = ("twc", "thc")
        quad = (
            "q0wc", "q0hc",
        )
        opaths = ("tw", "th", "qw", "qh")
        dct = {
            "nodes": {
                **{ k: 2 for k in (*tri, *quad) },
            },
            "global": {
                **{ k: 6 for k in tri },
                **{ k: 8 for k in quad },
                **{ k: 2 for k in opaths }
            }
        }
        out = {}
        for mode, agg in dct.items():
            d = {}
            for k, v in agg.items():
                d[self.resolve(k)] = v
            out[mode] = d
        return out

    @property
    def aliases(self) -> Dict:
        """Aliases mapping actual path names to raw names."""
        return {}

    @property
    def npaths(self) -> int:
        """Number of different path/cycle counts."""
        return len(self.definitions["sim"]) + len(self.definitions["comp"])

    def resolve(self, name) -> str:
        """Resolve path name alias."""
        if name in self.aliases:
            return self.aliases[name]
        if name in self.list():
            return name
        raise ValueError(f"incorrect path name '{name}'")

    def list(self) -> List[str]:
        """List path names."""
        return list(self)

    def enumerate(self) -> List[Tuple[str, int]]:
        """Enumerate path names."""
        return [ (p, i) for i, p in enumerate(self) ]

    def get_swap_rules(self) -> List[Tuple[int, int]]:
        """Get swap rules for counting reversed paths.

        They define indices of pairs of columns which need to be
        swaped in order to get reversed paths. Note that reverses
        of wedge paths are head paths and vice versa. In the case
        of weighted paths also wedge/head cycle counts need to be
        reversed.
        """
        omap = dict(self.enumerate())
        return [
            (omap[left], omap[right])
            for left, right in self["swap"]
        ]

    def get_column_names(self) -> List[str]:
        """Get names of path columns used once the reverse counting
        is done.
        """
        return list(self)

    def get_column_ids(self) -> List[int]:
        """Get indices of path columns to leave
        once reversed counting is done.
        """
        omap = dict(self.enumerate())
        return [ omap[path] for path in self.get_column_names() ]


class PathDefinitionsUnweighted(PathDefinitions):
    """Unweighted path definitions.

    **Similarity-related paths**

    t
        Triangles.
    tw
        Wedge-triples around ``i`` (i.e. ``k-i-j`` paths).
    th
        Head-triples originating from ``i`` (i.e. ``i-j-k`` paths).

    **Complementarity-related paths**

    q0
        Strong quadrangles.
    qw
        Wedge-quadruples around ``i`` (i.e. ``k-i-j-l`` paths).
    qh
        Head-quadruples originating from ``i`` (i.e. ``i-j-k-l`` paths).
    """
    @property
    def definitions(self) -> Dict:
        return dict(
            sim=("t", "tw", "th"),
            comp=(
                "q0", "qw", "qh"
            ),
            swap=(
                ("tw", "th"),
                ("qw", "qh")
            )
        )

    @property
    def aliases(self) -> Dict:
        return {
            "twc": "t",
            "thc": "t",
            "q0wc": "q0",
            "q0hc": "q0",
        }


class PathDefinitionsWeighted(PathDefinitions):
    """Weighted path definitions.

    **Similarity-related paths**

    twc
        Closed wedge triples or triangles weighted by ``ij`` and ``ik`` edges.
    thc
        Closed head triples or triangles weighted by ``ij`` and ``jk`` edges.
    tw
        Wedge triples.
    th
        Head triples.


    **Complementarity-related paths**

    q0wc
        Closed wedge quadruples with no chords (strong quadrangles)
        weighted by ``ij``, ``jk``, and ``il`` edges.
    qw
        Wedge quadruples.
    q0hc
        Closed head quadruples with no chords (strong quadrangles)
        weighted by ``ij``, ``jk`` and ``kl`` edges.
    qh
        Head quadruples.
    """
