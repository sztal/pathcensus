=============================
``pathcensus`` package
=============================

.. image:: https://github.com/sztal/pathcensus/actions/workflows/tests.yml/badge.svg
 :target: https://github.com/sztal/pathcensus

.. image:: https://codecov.io/gh/sztal/pathcensus/branch/master/graph/badge.svg?token=HP4hLAnagg
 :target: https://codecov.io/gh/sztal/pathcensus


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

**NOTE**

    ``pathcensus`` uses the ``A_{ij} = 1`` convention to indicate
    that a node `i` sends a tie to a node `j`. Functions converting
    graph-like objects to arrays / sparse matrices need to be aware
    of that.

**NOTE**

    ``pathcensus`` is compatible only with Python versions supported
    by `numba`_. In practice it means that it is compatible with all
    versions (starting from 3.8) except for the latest one, which usually
    starts to be supported by `numba`_ with some (often significant)
    delay.


For the sake of convenience ``pathcensus`` also provides implementations
of most appropriate null models for statistical calibration of structural
coefficients which are simple wrappers around the excellent `NEMtropy`_
package. It also defines the ``pathcensus.inference`` submodule with
utility class for facilitating approximate statistical inference based on
sampling from null models.

See ``examples`` subfolder and the main documentation for more details.

At the command line via pip:

.. code-block::

    # Install from PyPI
    pip install pathcensus

The current development version (not guaranteed to be stable)
can be installed directly from the `github repo`_

.. code-block::

    pip install git+ssh://git@github.com/sztal/pathcensus.git


How to cite?
============

You find the package useful? Please cite our work properly.

**Main theory paper**

    Talaga, S., & Nowak, A. (2022). Structural measures of similarity
    and complementarity in complex networks. *Scientific Reports*, 12(1), 16580.
    https://doi.org/10.1038/s41598-022-20710-w


Usage
=====

**NOTE**

    Main internal functions for calculating path census are JIT-compiled
    when used for the first time. Thus, the first initialization of a
    ``PathCensus`` object may be quite slow as its execution time will include
    the time required for compilation. However, this happens only once.

We will use `igraph`_ to generate graphs used in examples. However, even though
it is automatically integrated with ``pathcensus``, `igraph`_ is not
a dependency and needs to be installed separately.

.. code-block:: python

    # Main imports used in the examples below
    import random
    import numpy as np
    import igraph as ig
    from pathcensus import PathCensus

    # Set random and numpy rng seeds
    random.seed(303)
    np.random.seed(101)

More detailed examples can be found in the official documentation.


Path census & structural coefficients
-------------------------------------

Path census is a set of counts of different paths and cycles per edge, node
or in the entire graph. The counts are subsequently used to calculate different
kinds of structural coefficients.

.. code-block:: python

    # Generate simple undirected ER random graph
    G = ig.Graph.Erdos_Renyi(100, p=.05, directed=False)
    # Initialize path census object.
    # it precomputed path/cycle counts at the level of edges.
    # Other counts are derived from them.
    P = PathCensus(G)

    # Get edge-level census
    P.census("edges")
    # Get node-level census
    P.census("nodes")   # or just P.census()
    # Get global census
    P.census("global")

    # Column definitions
    ?P.definitions

Once path census is computed it can be used to calculate structural
coefficients.

.. code-block:: python

    # Similarity coefficients
    P.tclust()     # triangle-clustering equivalent to local clustering coefficient
    P.tclosure()   # triangle-closure equivalent to local closure coefficient
    P.similarity() # structural similarity (weighted average of clustering and closure)

    # Edge-wise similarity
    P.similarity("edges")
    # Global similarity (equivalent to global clustering coefficient)
    P.similarity("global")

The figure below sums up the design of structural similarity coefficients,
their geometric motivation and some of the main properties.

.. image:: /docs/figures/sim.svg
    :align: center


.. code-block:: python

    # Complementarity coefficients
    P.qclust()          # quadrangle-based clustering
    P.qclosure()        # quadrangle-based closure
    P.complementarity() # structural complementarity (weighted average of clustering and closure)

    # Edge-wise complementarity
    P.complementarity("edges")
    # Global complementarity
    P.complementarity("global")

The figure below sums up the design and the geometric motivation of
complementarity coefficients as well as their main properties.

.. image:: /docs/figures/comp.svg
    :align: center

Similarity and/or complementarity coefficients may be calculated in one
go using appropriate methods as shown below.

.. code-block:: python

    # Similarity + corresponding clustering and closure coefs
    P.simcoefs()           # node-wise
    P.simcoefs("global")   # global

    # Complementarity + corresponding clustering and closure coefs
    P.compcoefs()          # node-wise
    P.compcoefs("global")  # global

    # All coefficients
    P.coefs()
    # All coefficients + full path census
    P.coefs(census=True)


Weighted coefficients
---------------------

Below we create an ER random graph with random integer edge weights
between 1 and 10. As long as edge weights are assigned to an edge property
of the standard name (``"weight"``) they should be detected automatically
and ``pathcensus`` will calculate weighted census. However, unweighted census
may be enforced by using ``weighted=False``.

.. code-block:: python

    G = ig.Graph.Erdos_Renyi(100, p=0.05, directed=False)
    G.es["weight"] = np.random.randint(1, 11, G.ecount())

    P = PathCensus(G)
    P.weighted   # True
    # Get all coefficients and full path census
    P.coefs(census=True)

    # Use unweighted census
    P = PathCensus(G, weighted=False)
    P.weighted   # False
    P.coefs(census=True)

Below is the summary of the construction of weighted coefficients.

.. image:: /docs/figures/weighted.svg
    :align: center


Parallel ``PathCensus`` algorithm
---------------------------------

``PathCensus`` objects may be initialized using parallelized algorithms
by using ``parallel=True``.

**NOTE**

    Parallel algorithms require an extra compilation step so the first
    time ``parallel=True`` is used there will be a significant extra
    overhead.

**NOTE**

    The ``parallel=True`` argument may not work and lead to segmentation
    faults on some MacOS machines.

.. code-block:: python

    # By default all available threads are used
    P = PathCensus(G, parallel=True)

    # Use specific number of threads
    P = PathCensus(G, parallel=True, num_threads=2)


Other features
==============

Other main features of ``pathcensus`` are:

#. Null models based on the ERGM family.
#. Utilities for conducting statistical inference based on null models.
#. Integration with arbitrary classes of graph-like objects.

All these features are documented in the official documentation.


Testing
=======

The repository with the package source code can be cloned easily
from the `github repo`_.

.. code-block::

    git clone git@github.com:sztal/pathcensus.git

It is recommended to work within an isolated virtual environment.
This can be done easily for instance using `conda`_.
Remember about using a proper Python version (i.e. 3.8+).

.. code-block::

    conda create --name my-env python=3.8
    conda activate my-env

After entering the directory in which ``pathcensus`` repository
was cloned it is enough to install the package locally.

.. code-block:: bash

    pip install .
    # Or in developer/editable mode
    pip install --editable .

In order to run tests it is necessary to install also test dependencies.

.. code-block:: bash

    pip install -r ./requirements-tests.txt
    # Now tests can be run
    pytest
    # Or alternatively
    make test
    # And to run linter
    make lint

And similarly for building the documentation from source.

.. code-block:: bash

    pip install -r ./requirements-docs.txt
    # Now documentation can be built
    make docs

Tests targeting different Python versions can be run using `tox`_ test
automation framework. You may first need to install `tox`_
(e.g. ``pip install tox``).

.. code-block:: bash

    make test-all
    # Or alternatively
    tox

Test coverage
-------------

Unit test coverage report can be generated easily.

.. code-block::

    make coverage
    # Report can be displayed again after running coverage
    make cov-report


Feedback
========

If you have any suggestions or questions about **Path census** feel free to email me
at stalaga@protonmail.com.

If you encounter any errors or problems with **Path census**, please let me know!
Open an Issue at the GitHub http://github.com/sztal/pathcensus main repository.


Authors
=======

* Szymon Talaga <stalaga@protonmail.com>



.. _github repo: https://github.com/sztal/pathcensus
.. _examples: https://github.com/sztal/pathcensus/tree/master/examples
.. _conda: https://docs.conda.io/en/latest/
.. _tox: https://tox.wiki/en/latest/
.. _numpy: https://numpy.org/
.. _scipy: https://scipy.org/
.. _numba: https://numba.pydata.org/
.. _networkx: https://networkx.org/
.. _igraph: https://igraph.org/python/
.. _graph-tool: https://graph-tool.skewed.de/
.. _NEMtropy: https://pypi.org/project/NEMtropy/
