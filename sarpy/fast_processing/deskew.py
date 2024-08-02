"""Utility for centering the kspace content of a SICD in one dimension"""

__classification__ = "UNCLASSIFIED"

import copy

import numba
import numpy as np
import numpy.polynomial.polynomial as npp


@numba.njit(parallel=True)
def _apply_phase_poly(array, phase_poly, row_0, row_ss, col_0, col_ss):
    """numba parallelized phase poly application"""
    out = np.empty_like(array)
    for rowidx in numba.prange(out.shape[0]):
        row_val = row_0 + rowidx * row_ss
        col_poly = phase_poly[-1, :]
        for ndx in range(phase_poly.shape[0] - 1, 0, -1):
            col_poly = col_poly * row_val + phase_poly[ndx - 1, :]
        for colidx in range(out.shape[1]):
            col_val = col_0 + colidx * col_ss
            phase_val = col_poly[-1]
            for ndx in range(col_poly.shape[0] - 1, 0, -1):
                phase_val = phase_val * col_val + col_poly[ndx - 1]

            out[rowidx, colidx] = array[rowidx, colidx] * np.exp(
                1j * 2 * np.pi * phase_val
            )

    return out


def _update_grid_metadata(phase_poly, mdata):
    """Update the metadata following a deskew operation"""
    for dim in ["Row", "Col"]:
        axis_index = {"Row": 0, "Col": 1}[dim]
        axis_mdata = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[dim]
        delta_k_coa_poly = np.array([[0.0]]) if axis_mdata.DeltaKCOAPoly is None else axis_mdata.DeltaKCOAPoly.Coefs
        phase_poly_der = npp.polyder(-phase_poly, axis=axis_index) * axis_mdata.Sgn

        max_dims = np.amax([delta_k_coa_poly.shape, phase_poly_der.shape], axis=0)
        pad = max_dims - phase_poly_der.shape
        phase_poly_der = np.pad(
            phase_poly_der, ((0, pad[0]), (0, pad[1])), mode="constant"
        )
        pad = max_dims - delta_k_coa_poly.shape
        delta_k_coa_poly = np.pad(
            delta_k_coa_poly, ((0, pad[0]), (0, pad[1])), mode="constant"
        )

        updated_poly = delta_k_coa_poly + phase_poly_der
        axis_mdata.DeltaKCOAPoly = updated_poly


def get_deskew_phase_poly(sicd_metadata, axis):
    """Return phase polynomial to deskew specified axis

    Parameters
    ----------
    sicd_metadata : SICDType
        SICD metadata
    axis : {'Row', 'Col'}
        Which axis to deskew

    Returns
    -------
    phase_poly : ndarray
        Array of phase polynomial coefficients
    """
    axis_index = {"Row": 0, "Col": 1}[axis]
    axis_mdata = {'Row': sicd_metadata.Grid.Row, 'Col': sicd_metadata.Grid.Col}[axis]

    phase_poly = np.array([[0.0]])
    if axis_mdata.DeltaKCOAPoly is None:
        return phase_poly

    if np.all(axis_mdata.DeltaKCOAPoly == 0):
        return phase_poly

    phase_poly = npp.polyint(axis_mdata.DeltaKCOAPoly.Coefs, axis=axis_index) * axis_mdata.Sgn
    return phase_poly


def apply_phase_poly(array, phase_poly, sicd_metadata):
    """Metadata aware phase poly application

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    phase_poly : ndarray
        Array of phase polynomial coefficients
    sicd_metadata : SICDType
        SICD metadata

    Returns
    -------
    array_out : ndarray
        2D array of adjusted complex pixels
    sicd_metadata_out : SICDType
        Updated SICD metadata
    """
    sicd_metadata_out = copy.deepcopy(sicd_metadata)
    row_ss = sicd_metadata.Grid.Row.SS
    row_0 = (sicd_metadata.ImageData.FirstRow - sicd_metadata.ImageData.SCPPixel.Row) * row_ss
    col_ss = sicd_metadata.Grid.Col.SS
    col_0 = (sicd_metadata.ImageData.FirstCol - sicd_metadata.ImageData.SCPPixel.Col) * col_ss

    array_out = _apply_phase_poly(array, phase_poly, row_0, row_ss, col_0, col_ss)
    _update_grid_metadata(phase_poly, sicd_metadata_out)

    return array_out, sicd_metadata_out


def sicd_to_sicd(array, sicd_metadata, axis):
    """Deskew complex data array

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    sicd_metadata : SICDType
        SICD metadata
    axis : {'Row', 'Col'}
        Which axis to deskew

    Returns
    -------
    array_deskew : ndarray
        2D array of deskewed complex pixels
    sicd_metadata_deskew : SICDType
        Updated SICD metadata
    """
    phase_poly = get_deskew_phase_poly(sicd_metadata, axis)
    if np.all(phase_poly == 0):
        return array, sicd_metadata
    else:
        return apply_phase_poly(array, phase_poly, sicd_metadata)
