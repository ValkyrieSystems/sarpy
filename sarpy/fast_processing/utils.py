"""Numba optimized utility functions"""

__classification__ = "UNCLASSIFIED"

import numba


@numba.njit(parallel=True)
def parallel_copyto(dst, src):
    for rowidx in numba.prange(dst.shape[0]):
        for colidx in numba.prange(dst.shape[1]):
            dst[rowidx, colidx] = src[rowidx, colidx]
