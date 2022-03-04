"""Numba random number generator utilities."""
import numpy as np
from numba import njit

@njit
def set_numba_seed(seed: int) -> None:
    """Set seed for Numba random numbers generator."""
    np.random.seed(seed)
