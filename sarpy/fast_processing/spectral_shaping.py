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


SPECTRAL_SHAPING_DATABASE_PARAMETERS = {'SVA': {'n_2': 8,
                                                'n_0': 16,
                                                'lim_n': 9},
                                        'DSVA': {'n_2': 8,
                                                   'n_0': 16,
                                                   'lim_n': 9},
                                        'JIQ': {'n_2': 8,
                                                   'n_0': 16,
                                                   'lim_n': 9},
                                        'Taylor': {'n_2': 8,
                                                   'n_0': 24,
                                                   'lim_n': 9},
                                        'Uniform': {'n_2': 8,
                                                   'n_0': 16,
                                                   'lim_n': 9}}


def compute_spectral_shaping_parameters(c_l, c_h, wgt_type):
    db_key = (wgt_type if wgt_type in SPECTRAL_SHAPING_DATABASE_PARAMETERS
              else 'Uniform')
    db_params = SPECTRAL_SHAPING_DATABASE_PARAMETERS[db_key]
    log_ch_cl = np.log10(c_h / c_l)
    x_2 = 10**(np.log10(c_h) - db_params['n_2'] / 32 * log_ch_cl)
    x_0 = 10**(np.log10(x_2) - db_params['n_0'] / 32 * log_ch_cl)
    return {'x_0': x_0,
            'x_2': x_2,
            'lim_n': db_params['lim_n']}


@numba.njit
def _copy_boundaries(base_data, new_data, m, n):
    """Copy over boundary values"""
    new_data[:m, :] = base_data[:m, :]
    new_data[-m:, :] = base_data[-m:, :]
    new_data[m:-m, :n] = base_data[m:-m, :n]
    new_data[m:-m, -n:] = base_data[m:-m, -n:]


@numba.njit(parallel=True)
def apply_filter(data, x_0, x_2, lim_n):
    """numba parallelized spectral shaping filter"""
    shape = data.shape
    filt = DEFAULT_FILTER
    filt_shape = filt.shape
    row_pad = filt_shape[0] // 2
    col_pad = filt_shape[1] // 2
    out = np.empty(data.shape, data.dtype)
    _copy_boundaries(data, out, row_pad, col_pad)
    for rowidx in numba.prange(shape[0]-2*row_pad):
        for colidx in numba.prange(shape[1]-2*col_pad):
            amp_filt = np.float32(0.0)
            amp_max = np.float32(0.0)
            for filt_m in range(0, filt_shape[0]):
                for filt_n in range(0, filt_shape[1]):
                    amp_max = max(amp_max, data[rowidx + filt_m, colidx + filt_n])
                    amp_filt += data[rowidx + filt_m, colidx + filt_n] * filt[filt_m, filt_n]

            amp_orig = data[rowidx + row_pad, colidx + col_pad]
            if amp_max <= x_0:
                amp_weighted = amp_orig
            elif amp_max >= x_2:
                amp_weighted = amp_filt
            else:
                wgt = np.log10(amp_max / x_0) / np.log10(x_2 / x_0)
                amp_weighted = (1.0 - wgt) * amp_orig + wgt * amp_filt

            out[rowidx + row_pad, colidx + col_pad] = max(amp_weighted, amp_orig * 10**(-lim_n/20))
    return out
