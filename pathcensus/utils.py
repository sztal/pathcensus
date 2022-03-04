"""Utility functions."""
from typing import Union, Optional
import random as _random
import numpy as np
from scipy.sparse import isspmatrix, spmatrix
from .core.random import set_numba_seed
from . import adjacency  # pylint: disable=unused-import


def set_seed(
    all: Optional[int] = None,
    *,
    random: Optional[int] = None,
    numpy: Optional[int] = None,
    numba: Optional[int] = None,
) -> None:
    """Set seeds of random number generators.

    Parameters
    ----------
    random
        Seed value for :py:mod:`random` generator.
    numpy
        Seed value for :py:mod:`numpy` generator.
    numba
        Seed value for py:mod:`numba` generator.
    all
        Seed value used for all generators.
        Cannot be used jointly with other arguments.

    Raises
    ------
    ValueError
        If 'all' is used with other arguments or no seed is set.
    """
    # pylint: disable=redefined-builtin
    any_seed = random is not None or numpy is not None or numba is not None
    if all is not None and any_seed:
        raise ValueError("'all' cannot be used with other arguments")
    if all is None and not any_seed:
        raise ValueError("no random generator module selected")

    if all is not None:
        random = numpy = numba = all

    if random is not None:
        _random.seed(random)
    if numpy is not None:
        np.random.seed(numpy)
    if numba is not None:
        set_numba_seed(numba)

def rowsums(X: Union[np.ndarray, spmatrix]) -> np.ndarray:
    """Calculate row sums of a matrix."""
    if isspmatrix(X):
        return np.array(X.sum(1)).flatten()
    return X.sum(1)

def relerr(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Relative error ``|(x1 - x2)| / |x2|``."""
    return np.abs(x1 - x2) / np.abs(x2)

def relclose(x1: np.ndarray, x2: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    """Are two arrays relatively close.

    ``rtol`` defines the maximum allowed relative difference
    between ``x1`` and ``x2`` relative to the magnitude of ``x2``.
    """
    return (relerr(x1, x2) <= rtol).all()
