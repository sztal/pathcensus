"""Approximate inference for arbitrary graph statistics,
including structural coefficients, can be conducted using
samples from appropriate Exponential Random Graph Models.
The following generic algorithm can be used to solve a wide
range of inferential problems:

#. Calculate statistics of interest on an observed graph.
#. Sample ``R`` randomized instances from an appropriate null model.
#. Calculate graph statistics on null model samples.
#. Compare observed and null model values.

:class:`Inference` class implements the above approach.
It is comaptible with any registered class of graph-like objects
and any properly implemented subclass of
:class:`pathcensus.nullmodels.base.ERGM` representing a null model
to sample from.

.. seealso::

    :mod:`pathcensus.graph` for seemless ``pathcensus`` integration
    with arbitrary graph-like classes.

    :mod:`pathcensus.nullmodels` for available null models.

This simulation-based approach is relatively efficient for graph-
and node-level statistics but can be very computationally expensive
when used for edge-level analyses. Hence, is this case it is often
useful to use various coarse-graining strategies to reduce the number
of unique combinations of values of sufficient statistics.

.. seealso::

    :class:`pathcensus.graph.GraphABC` for the abstract class
    for graph-like objects.

    :mod:`pathcensus.nullmodels` for compatible ERGM classes.

    :meth:`pathcensus.inference.Inference.coarse_grain`
    for coarse-graining methods.

Below is a simple example of an estimation of p-values of node-wise
structural similarity coefficients in an Erdős–Rényi random graph.
The result, of course, should not be statistically significant.
We use the default significance level of :math:`\\alpha = 0.05` and
Benjamini-Hochberg FDR correction for multiple testing.

.. testsetup:: inference

    import numpy as np
    np.random.seed(34)

.. doctest:: inference

    >>> import numpy as np
    >>> from scipy import sparse as sp
    >>> from pathcensus import PathCensus
    >>> from pathcensus.inference import Inference
    >>> from pathcensus.nullmodels import UBCM
    >>> np.random.seed(34)
    >>> # Generate ER random graph (roughly)
    >>> A = sp.random(100, 100, density=0.05, dtype=int, data_rvs=lambda n: np.ones(n))
    >>> A = (A + A.T).astype(bool).astype(int)
    >>> ubcm = UBCM(A)
    >>> err = ubcm.fit()
    >>> infer = Inference(A, ubcm, lambda g: PathCensus(g).similarity())
    >>> data, null = infer.init_comparison(100)
    >>> pvals = infer.estimate_pvalues(data, null, alternative="greater")
    >>> # Structural similarity coefficient values
    >>> # should not be significant more often than 5% of times
    >>> # (BH FDR correction is used)
    >>> (pvals <= 0.05).mean() <= 0.05
    True
"""
from __future__ import annotations
from typing import Any, Optional, Mapping
from typing import Sequence, Tuple, Dict, Callable, Literal, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection_twostage
from tqdm.auto import tqdm
from .nullmodels.base import ERGM
from .types import GraphABC, Data


class Inference:
    """Generic approximate statistical inference based on
    arbitrary null models with node-level sufficient statistics.

    The methods implemented by this class are based on sampling
    from null models so they are may not be very efficient, in particular
    for edge-level statistics. On the other hand, they allow to conduct
    statistical inferency for arbitrary graph statistics.

    Attributes
    ----------
    graph
        Graph-like object representing an observed network.
    model
        Fitted instance of a subclass of
        :py:class:`pathcensus.nullmodels.ERGM`.
    statistics
        Function for calculating graph statistics of interest with the
        following signature::

            (graph, **kwds) -> DataFrame / Series

        The first argument must be a graph-like object (e.g. a sparse matrix),
        ``**kwds`` can be used to pass additional arguments
        (only keyword args are allowed) if necessary.
        The return value must be either a :py:class:`pandas.DataFrame`
        or :py:class:`pandas.Series`.
    aggregate_by
        Mode of aggregation for determining null distribution.
        If ``"stats"`` then null distribution is aggregated within
        unique combinations of values of the sufficient statistics
        (possibly coarse-grained, see :py:meth:`init_comparison`).
        If ``"units"`` then null distribution is aggregated within individual
        units (e.g. nodes). This is often useful in analyses at the level
        of nodes but may require too many samples for edge-level analyses.
    """
    _alternative  = ("greater", "less")
    _aggregate_by = ("stats", "units")
    _filter_index = ("values", "range")
    index_names   = ("i", "j")

    @dataclass
    class Levels:
        """Container class for storing information on unit,
        sufficient statistics and other index levels in observed
        and null model data.
        """
        units: Tuple[str, ...]
        stats: Tuple[str, ...]
        other: Tuple[str, ...]

        def __bool__(self) -> bool:
            return bool(self.units or self.stats or self.other)

    def __init__(
        self,
        graph: GraphABC,
        model: ERGM,
        statistics: Callable[[GraphABC], Data],
        *,
        aggregate_by: Literal[_aggregate_by] = _aggregate_by[0] # type: ignore
    ) -> None:
        """Initialization method."""
        self.graph = graph
        self.model = model
        self.statistics = statistics
        self.aggregate_by = self._check_vals(
            aggregate_by=aggregate_by,
            allowed=self._aggregate_by
        )

    def __call__(
        self,
        graph: GraphABC,
        _stats: Optional[np.ndarray] = None,
        **kwds: Any
    ) -> Data:
        """This method should be called to actually calculate graph statistics.

        Parameters
        ----------
        graph
            Graph-like object to calculate statistics for.
        stats
            Array of sufficient statistics for nodes.
            If ``None`` then `self.model.statistics` is used.
        **kwds
            Passed to graph statistics function.
        """
        data  = self.statistics(graph, **kwds)
        if _stats is None:
            _stats = self.model.extract_statistics(graph)
            # _stats = self.model.statistics
        out = self._postprocess_data(data, _stats)
        return out

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the observed network."""
        return self.model.n_nodes

    def get_levels(self, data: Data) -> Levels:
        """Get index levels descriptor from a data object."""
        nodes = tuple(l for l in data.index.names if l in self.index_names)
        allowed_stats = [ f"{l}{i}" for l in self.model.labels for i in nodes ]
        stats = tuple(l for l in data.index.names if l in allowed_stats)
        other = tuple(
            l for l in data.index.names
            if l and l not in (*nodes, *stats)
        )
        return self.Levels(nodes, stats, other)

    def simulate_null(
        self,
        n: int,
        *,
        progress: bool = False,
        progress_kws: Optional[Dict] = None,
        use_observed_stats: bool = True,
        **kwds: Any
    ) -> Data:
        """Get data frame of null model samples of strucutral coefficients.

        Parameters
        ----------
        n
            Number of samples.
        progress
            Should progress bar be showed.
        progress_kws
            Keyword arguments for customizing progress bar when
            ``progress=True``. Passed to :py:func:`tqdm.tqdm`.
        use_observed_stats
            If ``True`` then simulated data is indexed with
            sufficient statistics from the observed network.
            This often helps to accumulate enough observations
            faster at the expense of not fully exact conditioning.
        **kwds
            Keyword arguments passed to :py:meth:`statistics`.

        Returns
        -------
        null
            Data frame with simulated null distribution.
        """
        rand = []
        keys = []

        simulator = self.model.sample(n)
        progress_kws = {
            **(progress_kws or {}),
            "disable": not progress,
            "total": n
        }

        for i, graph in tqdm(enumerate(simulator), **progress_kws):
            keys.append(i)
            _stats = self.model.statistics if use_observed_stats else None
            rand.append(self(graph, _stats=_stats, **kwds))

        null = pd.concat(rand, keys=keys, names=["_sample"])

        return null

    def filter_index(
        self,
        data: Data,
        target: Data,
        *,
        how: Literal[_filter_index] = _filter_index[0],     # type: ignore
        levels: Optional[Union[Sequence[str], Sequence[int]]] = None
    ) -> Data:
        """Filter ``data`` by index with respect to ``target``.

        Parameters
        ----------
        data
            Data to filter.
        target
            Dataset with target index.
        how
            How index should be filtered.
            Either by unique combinations of values or just contained
            to the range of values for separate levels in ``target``.
        levels
            Levels to use for filtering. If ``None`` then either
            ``self.levels.units`` or ``self.levels.stats`` is used depending
            on the value of ``self.aggregate_by``.

        Returns
        -------
        data
            Filtered copy of ``data``.
        """
        how = self._check_vals(how=how, allowed=self._filter_index)

        l = self.get_levels(target)
        if levels is None:
            levels = l.stats if self.aggregate_by == "stats" else l.units

        # Filter by unique combinations of values in `target`
        if how == "values":
            # Determinex index values in `data`
            remove = [ n for n in data.index.names if n not in levels ]
            didx   = data.reset_index(remove).index if remove else data.index
            # Determine index values in `target`
            remove = [n for n in target.index.names if n not in levels ]
            tidx   = target.reset_index(remove).index if remove else target.index
            data   = data[didx.isin(tidx)]
        # Filter down to ranges of levels in `target`
        else:
            for level in levels:
                tidx  = target.index.get_level_values(level).values
                didx  = data.index.get_level_values(level).values
                minx  = min(tidx)
                maxx  = max(tidx)
                data  = data[(didx >= minx) & (didx <= maxx)]

        return data

    def init_comparison(
        self,
        n: int,
        *,
        filter_index: Union[bool, Literal[_filter_index]] = False,    # type: ignore
        sample_index: bool = False,
        null_kws: Optional[Dict] = None,
        **kwds: Any
    ) -> Tuple[Data, Data, Levels]:
        """Initialize data for a comparison with a null model
        and determine index level names.

        Parameters
        ----------
        n
            Number of null model samples.
        filter_index
            If ``True`` or ``"values"`` then ``null`` will be filtered to
            contain only observations with index values matching those in
            ``data`` with levels used for the comparison selected based on
            ``self.aggregate_by``. If ``"range"`` then null model samples
            will be filtered to be in the range of index values in the
            observed data.
        sample_index
            If ``False`` then ``_sample`` index with sample ids
            is dropped from ``null`` data frame with null model samples.
        null_kws:
            Keyword args passed to :py:meth:`simulate_null`.
        **kwds
            Passed to :py:meth:`statistics` method used for calculating
            statistics of interest.

        Notes
        -----
        Estimating distributions of edge-wise statistics conditional
        on sufficient statistics of the participating nodes may
        require really large number of samples and in general is
        not really feasible for large networks (in particular weighted).
        The same applies, although probably to slightly lesser degree,
        to node-wise statistics when ``use_observed_stats=False``
        is passed to :py:meth:`simulate_null`.

        Efficient methods for solving these problems will be implemented
        in the future.

        Returns
        -------
        data
            Observed graph statistics.
        null
            Null distribution samples.
        """
        if isinstance(filter_index, bool):
            if filter_index:
                filter_index = "range"
        else:
            filter_index = self._check_vals(
                filter_index=filter_index,
                allowed=self._filter_index
            )

        data = self(self.graph, **kwds)
        self._validate_data(data)

        null_kws = (null_kws or {})
        null = self.simulate_null(n, **(null_kws or {}), **kwds)

        levels = self.get_levels(data)
        remove = levels.units if self.aggregate_by == "stats" else levels.stats
        remove = [ *remove, *levels.other ]
        if remove:
            null.reset_index(remove, drop=True, inplace=True)

        data = pd.concat([data], keys=[0], names=["_"])
        null = pd.concat([null], keys=[0], names=["_"])

        data = self._remove_unnamed_indexes(data)
        null = self._remove_unnamed_indexes(null)

        if not sample_index:
            null.reset_index("_sample", drop=True, inplace=True)

        if filter_index:
            null = self.filter_index(null, data, how=filter_index)

        return data, null

    def postprocess(self, data: Data, target: Data) -> Data:
        """Postprocess data after running a comparison.

        This mainly involves sanitizing index names after aggregation
        as well as setting proper shape and types for outputs
        so ``data`` has the same general form as ``target``.
        """
        if isinstance(data, pd.Series) and isinstance(target, pd.DataFrame):
            data = data.to_frame().T
            data.index = target.index
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = self.postprocess_index(data)
        return data

    def postprocess_index(self, data: Data) -> Data:
        """Postprocess index after running a comparison.

        This involves getting rid of temporary index names
        used when running comparisons as well as any unnamed indexes.
        Moreover sufficient statistics indexes are removed
        if ``self.aggregate_by == "unit"`` or ensuring that observed values of
        sufficient statistics are used in the index
        (instead of coarse-grained values) if ``self.aggregate == "stats"``.
        """
        levels = self.get_levels(data)
        # Remove sufficient statistics indexes
        remove = [ l for l in levels.stats if l in data.index.names ]
        # Remove auxiliary index `_`
        if "_" in data.index.names:
            remove.append("_")
        if remove:
            data.reset_index(remove, drop=True, inplace=True)
        # Remove unnamed indexes
        remove = [ i for i, n in enumerate(data.index.names) if n is None ]
        if remove:
            data.reset_index(remove, drop=True, inplace=True)
        # Add indexes if necessary
        if self.aggregate_by == "stats":
            data = self.add_stats_index(data, self.model.statistics)
        return data

    def estimate_pvalues(
        self,
        data: pd.DataFrame,
        null: pd.DataFrame,
        *,
        alternative: Literal[_alternative] = _alternative[0],   # type: ignore
        adjust: bool = True,
        resolution: int = 1000,
        **kwds: Any
    ) -> Data:
        """Estimate p-values of node/edge/global coefficients
        based on sampling from a configuraiton model
        (as returned by :py:meth:`init_comparison`).

        Parameters
        ----------
        data
            Data frame with observed graph statistics.
        null
            Data frame with simulated null distribution
            of graph statistics.
        alternative
            Type of test two perform.
            Currently only one-sided tests are supported.
        adjust
            Should p-values be adjusted. Benjamini-Hochberg FDR correction is
            used by default when ``True``.
        resolution
            Resolution of p-value estimation. It specifies the number
            of quantiles to comapre observed values against.
            For instance, if ``resolution=100`` then p-values
            will be accurate only up to ``0.01``.
            This parameter controls the amount of memory consumed
            by the estimation process.
        **kwds
            Passed as additional arguents to :meth:`adjust_pvalues`
            when ``adjust=True``.

        See Also
        --------
        adjust_pvalues : p-value adjustment method

        Returns
        -------
        pvalues
            P-values for statistics as :py:class:`pandas.Series`
            (for one graph statistic) or :py:class:`pandas.DataFrame`
            (for multiple statistics).
        """
        alternative = self._check_vals(
            alternative=alternative,
            allowed=self._alternative
        )
        levels = self.get_levels(data)

        if levels.units:
            null = self.filter_index(null, data, how="values")

        # Quantile data frame
        if self.aggregate_by == "units":
            keys = levels.units
        else:
            keys = levels.stats

        if keys:
            qdf = null.groupby(level=keys)
        else:
            qdf = null

        qdf = qdf.quantile(np.arange(0, resolution+1) / resolution) \
            .reset_index(level=-1, drop=True)

        idx = qdf.index.to_frame().rename(columns={0: None})
        idx.insert(0, "_", 0)
        idx = pd.MultiIndex.from_frame(idx)
        qdf.index = idx

        # Estimate p-values
        if alternative == "greater":
            pvals = data.le(qdf)
        else:
            pvals = data.ge(qdf)

        pvals = pvals.fillna(True)
        if levels.units:
            pvals = pvals.groupby(level=levels.units)

        pvals = pvals.mean()

        if adjust and levels.units:
            adjust_kws = { **kwds, "copy": False }
            pvals = self.adjust_pvalues(pvals, **adjust_kws)

        return self.postprocess(pvals, target=data)

    @staticmethod
    def adjust_pvalues(
        pvals: Data,
        *,
        alpha: float = 0.05,
        copy: True = bool,
        **kwds: Any
    ) -> Data:
        """Adjust p-values for multiple testing.

        Benjamini-Hochberg-Yekuteli two-stage procedure implemented in
        :py:func:`statsmodels.multitest.fdrcorrection_twostage`
        is used.

        Parameters
        ----------
        pvals
            Data frame / series with p-values for different coefficients
            in columns.
        alpha
            Desired type I error rate after the adjustement.
        copy
            Should copy of ``pvals`` be returned.
        **kwds
            Additional arguments passed to
            :py:func:`statsmodels.multitest.fdrcorrection_twostage`
        """
        if copy:
            pvals = pvals.copy()
        shape = pvals.shape
        pv = pvals.values.flatten()
        _, pv, *_ = fdrcorrection_twostage(pv, alpha=alpha, **kwds)
        pvals.values[:] = np.clip(pv.reshape(shape), 0, 1)
        return pvals

    @staticmethod
    def add_index(
        data: Data,
        idx: Mapping[str, Sequence],
        *,
        prepend: bool = False,
        drop_unnamed: bool = True,
        copy: bool = True
    ) -> Data:
        """Add index to a data frame or series.

        Parameters
        ----------
        data
            Data frame or series.
        idx
            Mapping from index names to sequences of values.
        prepend
            Should new indexes be prepended or appended
            to the existing indexes.
        drop_unnamed
            Should unnamed indexes be droppped during the process.
            Unnamed indexes are usually generic indexes which are
            redundant after adding additional indexes.
        copy
            Should a copy be returned.
        """
        idx = pd.DataFrame(idx)

        if copy:
            data = data.copy()

        idf = data.index.to_frame(index=False)

        if drop_unnamed:
            use = [ i for i, n in enumerate(data.index.names) if n is not None ]
            idf = idf.iloc[:, use]

        objs = [ idx, idf ] if prepend else [ idf, idx ]
        objs = [ d for d in objs if not d.empty ]
        if objs:
            idf  = pd.concat(objs, axis=1, ignore_index=False)
        else:
            # Return if there no resulting indexes
            return data.reset_index(drop=True)

        if len(idf) == 0:
            return data
        if len(idf) == 1:
            idx = pd.Index(idf.iloc[:, 0])
        else:
            idx = pd.MultiIndex.from_frame(idf)

        data.index = idx
        return data

    def add_stats_index(
        self,
        data: Data,
        stats: Optional[np.ndarray] = None
    ) -> Data:
        """Add indexes with sufficient statistics.

        Parameters
        ----------
        data
            Data frame or series with graph statistics.
        stats
            Array of sufficient statistics.
            Use ``self.model.statistics`` if ``None``.
        """
        idx = {}
        levels = self.get_levels(data)
        if stats is None:
            stats = self.model.statistics
        for u in levels.units:
            for i, l in enumerate(self.model.labels):
                vals = stats[data.index.get_level_values(u), i]
                name = f"{l}{u}"
                idx[name] = vals
        return self.add_index(
            data=data,
            idx=idx,
            prepend=False,
            drop_unnamed=True,
            copy=False
        )

    # Internals ---------------------------------------------------------------

    def _postprocess_data(
        self,
        data: Data,
        stats: np.ndarray
    ) -> Data:
        """Post-process data with calculated graph statistics.

        Parameters
        ----------
        data
            Calculate graph statistics.
        stats
            Array with sufficient statistics for nodes.
        """
        if np.isscalar(data):
            data = pd.Series([data])

        if stats.ndim == 1:
            stats = stats[:, None]

        if self.aggregate_by == "stats":
            data = self.add_stats_index(data, stats)

        return data

    def _remove_unnamed_indexes(
        self,
        data: Data
    ) -> Data:
        """Remove unnamed indexes."""
        remove = []
        for i, name in enumerate(data.index.names):
            if name is None:
                remove.append(i)
        return data.reset_index(level=remove, drop=True)

    def _check_vals(
        self,
        *,
        allowed: Sequence[str],
        **kwds: Any
    ) -> str:
        """Check if value is okay and return."""
        if len(kwds) != 1:
            raise ValueError("exactly two keyword arguments are expected")
        allowed = tuple(allowed)
        key = list(kwds.keys())[0]
        val = list(kwds.values())[0]
        if val not in allowed:
            raise ValueError(f"'{key}' has to be one of {allowed}")
        return val

    def _validate_data(self, data: Data) -> None:
        """Check if `data` has correct indexes and shape."""
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            m = "'data' has to be either 'Series' or 'DataFrame' instance"
            raise TypeError(m)

        levels = self.get_levels(data)

        if levels.stats and not levels.units:
            raise AttributeError(
                "'data' has sufficient statistics "
                "but no node/edge indexes"
            )

        if self.aggregate_by == "stats" and levels.units and not levels.stats:
            raise AttributeError(
                "'data' has node/edge indexes but no "
                "index with sufficient statistics"
            )
