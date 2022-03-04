"""Core types used by compiled code."""
import numba
from ..types import UInt, Float

UInt  = numba.from_dtype(UInt)
Float = numba.from_dtype(Float)
