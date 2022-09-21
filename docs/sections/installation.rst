============
Installation
============

At the command line via pip

.. code-block::

    pip install pathcensus

The current development version (not guaranteed to be stable)
can be installed directly from the `github repo`_

.. code-block::

    pip install git+ssh://git@github.com/sztal/pathcensus.git


Development & testing
---------------------

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


.. include:: /links.rst

Test coverage
~~~~~~~~~~~~~

Unit test coverage report can be generated easily.

.. code-block::

    make coverage
    # Report can be displayed again after running coverage
    make cov-report
