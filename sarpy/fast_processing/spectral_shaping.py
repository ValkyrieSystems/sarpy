"""Utility for applying spectral shaping to a detected image"""

__classification__ = "UNCLASSIFIED"

import numba
import numpy as np

# This filter is defined in the AGI Algorithm Description Document.
DEFAULT_FILTER = np.array([[ 0.0000, -0.0372, -0.2284, -0.0372,  0.0000],
                           [-0.0372,  0.0306,  0.1881,  0.0306, -0.0372],
                           [-0.2284,  0.1881,  1.3363,  0.1881, -0.2284],
                           [-0.0372,  0.0306,  0.1881,  0.0306, -0.0372],
                           [ 0.0000, -0.0372, -0.2284, -0.0372,  0.0000]]).astype(np.float32)


@numba.njit
def _copy_boundaries(base_data, new_data, m, n):
    """Copy over boundary values"""
    new_data[:m, :] = base_data[:m, :]
    new_data[-m:, :] = base_data[-m:, :]
    new_data[m:-m, :n] = base_data[m:-m, :n]
    new_data[m:-m, -n:] = base_data[m:-m, -n:]


@numba.njit(parallel=True)
def apply_filter(data):
    """numba parallelized 2D filter"""
    shape = data.shape
    filt = DEFAULT_FILTER
    filt_shape = filt.shape
    row_pad = filt_shape[0] // 2
    col_pad = filt_shape[1] // 2
    out = np.empty(data.shape, data.dtype)
    _copy_boundaries(data, out, row_pad, col_pad)
    for rowidx in numba.prange(shape[0]-2*row_pad):
        for colidx in numba.prange(shape[1]-2*col_pad):
            out_val = np.float32(0.0)
            for filt_m in range(0, filt_shape[0]):
                for filt_n in range(0, filt_shape[1]):
                    out_val += data[rowidx + filt_m, colidx + filt_n] * filt[filt_m, filt_n]
            out[rowidx + row_pad, colidx + col_pad] = out_val
    return out
