
__classification__ = "UNCLASSIFIED"

import numba
import numpy as np

from sarpy.fast_processing import benchmark


@numba.njit(parallel=True)
def _mean(data):
    """numba parallelized numpy.mean"""
    # TODO only compute over pixels covered by the valid data polygon
    return np.mean(data)  # TODO sarpy has nan and inf protection return np.mean(data[np.isfinite(data)])


@numba.njit(parallel=True)
def _median(data):
    """numba parallelized median"""
    # Compute min/max to determine range of histogram
    max_vals = np.empty(data.shape[0], data.dtype)
    min_vals = np.empty(data.shape[0], data.dtype)
    for rowidx in numba.prange(data.shape[0]):
        max_vals[rowidx] = data[rowidx, 0]
        min_vals[rowidx] = data[rowidx, 0]
        for colidx in range(1, data.shape[1]):
            if data[rowidx, colidx] > max_vals[rowidx]:
                max_vals[rowidx] = data[rowidx, colidx]
            if data[rowidx, colidx] < min_vals[rowidx]:
                min_vals[rowidx] = data[rowidx, colidx]

    min_val = min(min_vals)
    max_val = max(max_vals)
    if max_val == min_val:
        return min_val

    # Compute histogram with bins based upon data size
    npts = data.shape[0] * data.shape[1]
    num_threads = numba.get_num_threads()
    num_rows = data.shape[0]
    num_bins = int(np.round(np.sqrt(npts)))
    hist_0 = min_val
    hist_ss = (max_val - min_val) / (num_bins - 1)
    hist_inv_ss = 1 / hist_ss
    hist_edges = hist_0 + np.arange(num_bins + 1) * hist_ss
    chunk_edges = np.linspace(0, num_rows, num_threads+1).astype(np.int64)
    chunk_start = chunk_edges[:-1]
    chunk_stop = chunk_edges[1:]
    chunk_counts = np.zeros((num_threads, num_bins), dtype=np.int64)
    for chunkidx in numba.prange(num_threads):
        for rowidx in range(chunk_start[chunkidx], chunk_stop[chunkidx]):
            for colidx in range(data.shape[1]):
                binidx = int(round((data[rowidx, colidx] - hist_0) * hist_inv_ss))
                if data[rowidx, colidx] >= hist_edges[binidx]:
                    chunk_counts[chunkidx, binidx] += 1
                else:
                    chunk_counts[chunkidx, binidx-1] += 1

    counts = np.sum(chunk_counts, axis=0)
    desired_index = npts // 2  # This is equivalent to numpy.percentile(data, 50, method='higher')
    accum = 0
    for binidx in range(num_bins):
        if (accum + counts[binidx]) > desired_index:
            break
        else:
            accum += counts[binidx]

    # Threaded culling of data values to those in histogram bin containing the median value
    vals = np.empty(shape=(counts[binidx], ), dtype=data.dtype)
    chunk_save_start = np.cumsum(chunk_counts[:, binidx])

    for chunkidx in numba.prange(num_threads):
        if chunkidx == 0:
            chunk_save_idx = 0
        else:
            chunk_save_idx = chunk_save_start[chunkidx - 1]
        for rowidx in range(chunk_start[chunkidx], chunk_stop[chunkidx]):
            for colidx in range(data.shape[1]):
                if hist_edges[binidx] <= data[rowidx, colidx] < hist_edges[binidx+1]:
                    vals[chunk_save_idx] = data[rowidx, colidx]
                    chunk_save_idx += 1

    # Compute median value from culled data
    perc = float(desired_index - accum) / (counts[binidx] - 1) * 100
    with numba.objmode(med_val='f8'):
        med_val = np.percentile(vals, perc)
    return med_val


@numba.njit(parallel=True)
def density(data, dmin=30, mmult=40):
    """
    A monochromatic logarithmic density remapping function.

    This is a digested version of contents presented in a 1994 publication
    entitled "Softcopy Display of SAR Data" by Kevin Mangis. It is unclear where
    this was first published or where it may be publicly available.

    Args
    ----
    data: `numpy.ndarray`
        2D array of amplitude data
    dmin : float|int
        A dynamic range parameter. Lower this widens the range, will raising it
        narrows the range. This was historically fixed at 30.
    mmult : float|int
        A contrast parameter. Low values will result is higher contrast and quicker
        saturation, while high values will decrease contrast and slower saturation.
        There is some balance between the competing effects in the `dmin` and `mmult`
        parameters.

    Returns
    -------
    `numpy.ndarray`
        2D array of remapped uint8 data

    """

    data_mean = _mean(data)
    return amp_to_dens_uint8(data, dmin=dmin, mmult=mmult, data_mean=data_mean)


def _gdm_cutoff_values(data_mean, data_median, weighting, graze_rad, slope_rad):
    # This is a subset of the GDM remap algorithm defined in the AGI Algorithm Description Document.
    # Only those parts of the algorithm relevant to existing SarPy capabilities are included here.
    # If a weighting name other than 'uniform' is specified, 'taylor' parameters will be chosen.
    c3 = {'taylor': 1.33, 'uniform': 1.23}.get(weighting, 1.33)
    w1 = {'taylor': 0.77, 'uniform': 0.77}.get(weighting, 0.77)
    w2 = -0.0422
    w5 = -1.95
    a1 = data_median
    a2 = np.sin(graze_rad) / np.cos(slope_rad)
    a5 = np.log10(data_median / data_mean)
    cl_init = a1 * w1
    ch_init = a1 * 10**(c3 + w2*a2 + w5*a5)
    r_min = 24
    r_max = 40
    r = 20*np.log10(ch_init / cl_init)
    if r < r_min:
        beta = 10**((r - r_min) / 40.0)         # pragma: nocover
    elif r > r_max:
        beta = 10**((r - r_max) / 40.0)         # pragma: nocover
    else:
        beta = 1.0
    cl = cl_init * beta
    ch = ch_init / beta

    return cl, ch


def gdm_parameters(sicd_metadata):
    """Compute the metadata parameters needed for GDM remap

    Args
    ----
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object

    Returns
    -------
    dict
        Dictionary of keyword arguments for passing to `gdm()`
    """

    def _guess_sicd_weighting(sicd_metadata):
        row_uniform = True
        if sicd_metadata.Grid.Row.WgtFunct is not None:
            row_uniform = np.all(sicd_metadata.Grid.Row.WgtFunct == sicd_metadata.Grid.Row.WgtFunct[0])
        col_uniform = True
        if sicd_metadata.Grid.Col.WgtFunct is not None:
            col_uniform = np.all(sicd_metadata.Grid.Row.WgtFunct == sicd_metadata.Grid.Row.WgtFunct[0])

        if row_uniform and col_uniform:
            return 'uniform'
        else:
            return 'other'

    return {
         'weighting': _guess_sicd_weighting(sicd_metadata),
         'graze_deg': sicd_metadata.SCPCOA.GrazeAng,
         'slope_deg': sicd_metadata.SCPCOA.SlopeAng,
    }


def gdm(data, *, weighting, graze_deg, slope_deg):
    """
    The Density remap using image specific parameters

    Args
    ----
    data: `numpy.ndarray`
        2D array of floating point amplitude data
    weighting : str
        The spectral taper weighting ('taylor' | 'uniform')
    graze_deg: float|int
        Graze angle (degrees)
    slope_deg: float|int
        Slope angle (degrees)

    Returns
    -------
    `numpy.ndarray`
        2D array of remapped uint8 data

    """

    data_mean = _mean(data)
    with benchmark.howlong('median'):
        if data.size < 5e8:
            data_median = np.median(data)
        else:
            data_median = _median(data)
    c_l, c_h = _gdm_cutoff_values(data_mean, data_median, weighting, np.deg2rad(graze_deg), np.deg2rad(slope_deg))
    if c_l == 0:
        c_l = np.finfo(data.real.dtype).tiny
    return amp_to_dens_uint8(data,
                             dmin=30,
                             mmult=c_h / c_l,
                             data_mean=c_l / 0.8)


@numba.njit(parallel=True)
def amp_to_dens_uint8(data, *, dmin, mmult, data_mean):
    """
    Convert to density data for remap.

    This is a digested version of contents presented in a 1994 pulication
    entitled "Softcopy Display of SAR Data" by Kevin Mangis. It is unclear where
    this was first published or where it may be publicly available.

    Parameters
    ----------
    data : numpy.ndarray
        2D array of floating point amplitude data
    dmin : float|int
        A dynamic range parameter. Lowering this widens the range, while raising it
        narrows the range. This was historically fixed at 30.
    mmult : float|int
        A contrast parameter. Low values will result in higher contrast and quicker
        saturation, while high values will decrease contrast and slower saturation.
        There is some balance between the competing effects in the `dmin` and `mmult`
        parameters.
    data_mean : None|float|int
        The data mean (for this or the parent array for continuity), which will
        be calculated if not provided.

    Returns
    -------
    numpy.ndarray
    """

    data_type = data.dtype.type
    C_L = 0.8*data_mean
    C_H = mmult*C_L  # decreasing mmult will result in higher contrast (and quicker saturation)
    slope = data_type((255 - dmin)/np.log10(C_H/C_L))
    constant = data_type(dmin - (slope*np.log10(C_L)))

    out = np.empty_like(data, dtype=np.uint8)
    for rowidx in numba.prange(out.shape[0]):
        for colidx in numba.prange(out.shape[1]):

            if data[rowidx, colidx] == 0:
                dens = 0.0
            else:
                dens = slope*np.log10(data[rowidx, colidx]) + constant

            if dens < 0:
                out[rowidx, colidx] = 0
            elif dens > 255:
                out[rowidx, colidx] = 255
            else:
                out[rowidx, colidx] = dens
    return out
