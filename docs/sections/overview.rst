Overview
========

Welcome to the documentation of ``pathcensus`` package.
It is a Python (3.8+) implementation of **structural similarity and
complementarity coefficients** for undirected (un)weighted networks based
on efficient counting of 2- and 3-paths (triples and quadruples)
and 3- and 4-cycles (triangles and quadrangles).

**Structural coefficients are graph-theoretic
measures of the extent to which relations at different levels
(of edges, nodes or entire networks) are driven by similarity or
complementarity between different nodes**. Even though they are defined
in purely combinatorial manner they are motivated by geometric arguments
which link them to the family of latent space/random geometric graph models.
In particular, the geometric view allow the identification of network motifs
charactersitic for similarity (triangles) and complementarity (quadrangles).
They can be seen as a generalization of the well-known
local and global clustering coefficients which summarize the structure
of a network in terms of density of ego subgraph(s).

Even though it is a Python package ``pathcensus`` is performant as its main
workhorse functions are just-in-time (JIT) compiled to efficient C code
thanks to the `numba`_ library. It is compatible with `numpy`_
arrays and `scipy`_ sparse matrices making it easy to use in practice.
Moreover, it allows registering graph classes implemented by different
third-party packages such as `networkx`_ so they can be converted
automatically to sparse matrices. Conversion methods for `networkx`_,
`igraph`_ and `graph-tool`_ are registered automatically
provided the packages are installed.

For the sake of convenience ``pathcensus`` provides also implementations
of most appropriate null models for statistical calibration of structural
coefficients which are simple wrappers around the excellent `NEMtropy`_
package :cite:p:`vallaranoFastScalableLikelihood2021`. It also defines
the :py:mod:`pathcensus.inference` submodule with utility class for
facilitating approximate statistical inference based on sampling from
null models.

.. include:: /sections/citation.rst
.. include:: /links.rst
