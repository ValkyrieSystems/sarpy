"""Utililty for changing the sidelobe weighting of a SICD"""

__classification__ = "UNCLASSIFIED"

import copy
import logging

import numpy as np
import numpy.polynomial.polynomial as npp
import scipy.fft
import scipy.interpolate as spi

from sarpy.fast_processing import read_sicd
from sarpy.io.complex.sicd_elements.Grid import WgtTypeType
from sarpy.processing.sicd.spectral_taper import Taper
import sarpy.processing.sicd.windows as windows  # TODO migrate to sarpy2

import sarpy.fast_processing.backend
from sarpy.fast_processing import benchmark


def sicd_to_sicd(data, sicd_metadata, new_weights, window_name, window_parameters=None):
    """Apply sidelobe weighting to a SICD

    Args
    ----
    data: `numpy.ndarray`
        SICD pixels
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    new_weights: `numpy.ndarray`
        1-D array of desired weighting.  Will be applied in both row and col directions.
        Existing weighting will be removed.
    window_name: str
        Name of the window to record in SICD metadata
    window_parameters: dict
        Key/Value pairs of window parameters to record in SICD metadata

    Returns
    -------
    `numpy.ndarray`
        SICD pixels
    `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object

    """
    # TODO make sure Chips are supported
    new_weights = np.asarray(new_weights)
    for direction in ('Row', 'Col'):
        existing_window = _get_sicd_wgt_funct(sicd_metadata, direction, len(new_weights))
        both_windows = new_weights / np.maximum(existing_window, 0.01 * np.max(existing_window))

        # deskew
        with benchmark.howlong("deskew"):
            data = _deskew(data, sicd_metadata, direction, forward=False)

        # FFT # Mult # IFFT
        with benchmark.howlong("fft_window_ifft"):
            data = _fft_window_ifft(data, sicd_metadata, direction, both_windows)

        with benchmark.howlong("redeskew"):
            # reskew?  # TODO update metadata instead of reskewing
            data = _deskew(data, sicd_metadata, direction, forward=True)

    new_sicd_metadata = updated_sicd_metadata(sicd_metadata, new_weights, window_name, window_parameters)
    return data, new_sicd_metadata


# based on sarpy.processing.sicd.normalize_sicd.apply_skew_poly
def _apply_skew_poly(
        input_data: np.ndarray,
        delta_kcoa_poly: np.ndarray,
        row_array: np.ndarray,
        col_array: np.ndarray,
        fft_sgn: int,
        dimension: int,
        forward: bool = False) -> np.ndarray:
    """
    Performs the skew operation on the complex array, according to the provided
    delta kcoa polynomial.

    Parameters
    ----------
    input_data : np.ndarray
        The input data.
    delta_kcoa_poly : np.ndarray
        The delta kcoa polynomial to use.
    row_array : np.ndarray
        The row array, should agree with input_data first dimension definition.
    col_array : np.ndarray
        The column array, should agree with input_data second dimension definition.
    fft_sgn : int
        The fft sign to use.
    dimension : int
        The dimension to apply along.
    forward : bool
        If True, this shifts forward (i.e. skews), otherwise applies in inverse
        (i.e. deskew) direction.

    Returns
    -------
    np.ndarray
    """

    if np.all(delta_kcoa_poly == 0):
        return input_data

    delta_kcoa_poly_int = npp.polyint(delta_kcoa_poly, axis=dimension)
    if forward:
        fft_sgn *= -1
    return input_data*np.exp(1j*fft_sgn*2*np.pi*npp.polygrid2d(
        row_array, col_array, delta_kcoa_poly_int), dtype=input_data.dtype)


# TODO rewrite/optimize.  This is mostly a copy/paste of sarpy.processing.sicd.spectral_taper._apply_1d_spectral_taper
def _deskew(cdata, mdata, axis, forward):
    axis_index = {'Row': 0, 'Col': 1}[axis]
    axis_mdata = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[axis]

    xrow = (np.arange(mdata.ImageData.FirstRow, mdata.ImageData.FirstRow + mdata.ImageData.NumRows)
            - mdata.ImageData.SCPPixel.Row) * mdata.Grid.Row.SS
    ycol = (np.arange(mdata.ImageData.FirstCol, mdata.ImageData.FirstCol + mdata.ImageData.NumCols)
            - mdata.ImageData.SCPPixel.Col) * mdata.Grid.Col.SS

    delta_k_coa_poly = np.array([[0.0]]) if axis_mdata.DeltaKCOAPoly is None else axis_mdata.DeltaKCOAPoly.Coefs

    if not np.all(delta_k_coa_poly == 0):
        with benchmark.howlong('apply_skew'):
            cdata = _apply_skew_poly(cdata, delta_k_coa_poly, row_array=xrow, col_array=ycol,
                                     fft_sgn=axis_mdata.Sgn, dimension=axis_index, forward=forward)
    return cdata


def _fft_window_ifft(cdata, mdata, axis, window_vals):
    axis_index = {'Row': 0, 'Col': 1}[axis]
    axis_mdata = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[axis]
    nyquist_bw = 1.0 / axis_mdata.SS
    ipr_bw = axis_mdata.ImpRespBW
    osf = nyquist_bw / ipr_bw

    # Zero pad the image data to avoid IPR wrap-around, then
    # find a good FFT size which creates some additional zero pad.
    axis_size = cdata.shape[axis_index]
    wrap_around_pad = int(min(200 * osf, 0.1 * axis_size))
    good_fft_size = scipy.fft.next_fast_len(axis_size + wrap_around_pad)

    with benchmark.howlong("fft"):
        # Forward transform without FFTSHIFT so the DC bin is at index=0
        if axis_mdata.Sgn == -1:
            cdata_fft = scipy.fft.fft(cdata, n=good_fft_size, axis=axis_index, workers=-1)
        else:
            cdata_fft = scipy.fft.ifft(cdata, n=good_fft_size, axis=axis_index, workers=-1)

    with benchmark.howlong("taper"):
        # Interpolate the taper to cover the spectral support bandwidth and extend the
        # taper window's end points into the over sample region of the spectrum.
        func = spi.interp1d(np.linspace(-ipr_bw / 2, ipr_bw / 2, len(window_vals)), window_vals, kind='cubic',
                            bounds_error=False, fill_value=(window_vals[0], window_vals[-1]))
        padded_taper = func(scipy.fft.fftfreq(good_fft_size, axis_mdata.SS))

        # Apply the taper to the spectrum.
        taper_2d = padded_taper[:, np.newaxis] if axis == 'Row' else padded_taper[np.newaxis, :]
    with benchmark.howlong("multiply"):
        cdata_fft *= taper_2d

    with benchmark.howlong("ifft"):
        # Inverse transform without FFTSHIFT and trim back to the original image size.
        nrows, ncols = cdata.shape
        if axis_mdata.Sgn == -1:
            cdata = scipy.fft.ifft(cdata_fft, n=good_fft_size, axis=axis_index, workers=-1)[:nrows, :ncols]
        else:
            cdata = scipy.fft.fft(cdata_fft, n=good_fft_size, axis=axis_index, workers=-1)[:nrows, :ncols]

    return cdata


def updated_sicd_metadata(sicd_metadata, new_weights, window_name, window_parameters=None):
    """Update SICD metadata to describe a new weighting

    Args
    ----
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    new_weights: `numpy.ndarray`
        1-D array of desired weighting.  Will be applied in both row and col diretions.
        Existing weighting will be removed.
    window_name: str
        Name of the window to record in SICD metadata
    window_parameters: dict
        Key/Value pairs of window parameters to record in SICD metadata

    Returns
    -------
    `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object

    """
    window_parameters = window_parameters or {}

    mdata = copy.deepcopy(sicd_metadata)

    # Update the SICD metadata to account for the spectral weighting changes
    row_inv_osf = mdata.Grid.Row.ImpRespBW * mdata.Grid.Row.SS
    col_inv_osf = mdata.Grid.Col.ImpRespBW * mdata.Grid.Col.SS

    old_row_wgts = _get_sicd_wgt_funct(mdata, 'Row', len(new_weights))
    old_col_wgts = _get_sicd_wgt_funct(mdata, 'Col', len(new_weights))
    old_coh_amp_gain = np.mean(old_row_wgts) * np.mean(old_col_wgts) * row_inv_osf * col_inv_osf
    old_rms_amp_gain = np.sqrt(np.mean(old_row_wgts**2) * np.mean(old_col_wgts**2) * row_inv_osf * col_inv_osf)

    new_row_wgts = new_weights
    new_col_wgts = new_weights
    new_coh_amp_gain = np.mean(new_row_wgts) * np.mean(new_col_wgts) * row_inv_osf * col_inv_osf
    new_rms_amp_gain = np.sqrt(np.mean(new_row_wgts**2) * np.mean(new_col_wgts**2) * row_inv_osf * col_inv_osf)

    coh_pwr_gain = (new_coh_amp_gain / old_coh_amp_gain) ** 2
    rms_pwr_gain = (new_rms_amp_gain / old_rms_amp_gain) ** 2

    mdata.Grid.Row.WgtType = WgtTypeType(WindowName=window_name.upper(), Parameters=window_parameters)
    mdata.Grid.Col.WgtType = WgtTypeType(WindowName=window_name.upper(), Parameters=window_parameters)

    taper_is_uniform = np.all(new_weights == new_weights[0])
    mdata.Grid.Row.WgtFunct = None if taper_is_uniform else new_weights
    mdata.Grid.Col.WgtFunct = None if taper_is_uniform else new_weights

    ipr_half_power_width = windows.find_half_power(new_weights, oversample=16)
    mdata.Grid.Row.ImpRespWid = ipr_half_power_width / mdata.Grid.Row.ImpRespBW
    mdata.Grid.Col.ImpRespWid = ipr_half_power_width / mdata.Grid.Col.ImpRespBW

    if mdata.Radiometric:
        if mdata.Radiometric.NoiseLevel:
            if mdata.Radiometric.NoiseLevel.NoiseLevelType == "ABSOLUTE":
                mdata.Radiometric.NoiseLevel.NoisePoly.Coefs *= rms_pwr_gain

        if mdata.Radiometric.RCSSFPoly:
            mdata.Radiometric.RCSSFPoly.Coefs /= coh_pwr_gain

        if mdata.Radiometric.SigmaZeroSFPoly:
            mdata.Radiometric.SigmaZeroSFPoly.Coefs /= rms_pwr_gain

        if mdata.Radiometric.BetaZeroSFPoly:
            mdata.Radiometric.BetaZeroSFPoly.Coefs /= rms_pwr_gain

        if mdata.Radiometric.GammaZeroSFPoly:
            mdata.Radiometric.GammaZeroSFPoly.Coefs /= rms_pwr_gain

    return mdata


def _get_sicd_wgt_funct(mdata, axis, desired_size=513):
    axis_mdata = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[axis]

    if axis_mdata.WgtFunct is not None:
        # Get the SICD WgtFunct values and interpolate as needed to produce a window of the desired size.
        wgts = _fit_1d_window(axis_mdata.WgtFunct, desired_size=desired_size)

    else:
        # The SICD WgtFunct values do not exist, so if WindowName is not specified,
        # or the WindowName is "UNIFORM", then return a window of all ones.
        # If WindowName is specified as other than "UNIFORM" then we can not presume to
        # know what the WindowName means because it is not clearly defined in the SICD spec.
        window_name = axis_mdata.WgtType.WindowName if axis_mdata.WgtType else "UNIFORM"

        if window_name.upper() == 'UNIFORM':
            wgts = np.ones(desired_size)
        else:
            raise ValueError(f'SICD/Grid/{axis}/WgtFunct is not part of the SICD metadata, but there appears '
                             f'to be a window of type "{window_name}" applied to the {axis} axis spectrum.')

    return wgts


def _fit_1d_window(window_vals, desired_size):
    if len(window_vals) == desired_size:
        w = window_vals
    else:
        x = np.linspace(0, 1, len(window_vals))
        f = spi.interp1d(x, window_vals, kind='cubic')
        w = f(np.linspace(0, 1, desired_size))
    return w


def main(args=None):
    """CLI for changing the sidelobe control of a SICD"""
    import argparse
    import pathlib
    import sarpy.io.complex
    import sarpy.io.complex.sicd

    parser = argparse.ArgumentParser()
    parser.add_argument('input_sicd', type=pathlib.Path)
    parser.add_argument('output_sicd', type=pathlib.Path)
    parser.add_argument('--sidelobe-control', required=True,
                        choices=['Uniform', 'Taylor'], default='Uniform',
                        help="Desired sidelobe control")
    parser.add_argument('--fft-backend', choices=['auto', 'mkl', 'scipy'], default='auto',
                        help="Which FFT backend to use.  Default 'auto', which will use mkl if available")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose logging (may be repeated)")
    config = parser.parse_args(args)

    loglevels = [logging.WARNING, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[min(config.verbose, len(loglevels)-1)])

    with sarpy.fast_processing.backend.set_fft_backend(config.fft_backend):
        with benchmark.howlong('sidelobe_control'):
            with benchmark.howlong('read'):
                sicd_pixels, sicd_meta = read_sicd.read_from_file(config.input_sicd)

            window_name = config.sidelobe_control.upper()
            taper = Taper(window_name)
            new_window = taper.get_vals(65, sym=True)
            new_params = taper.window_pars
            new_pixels, new_meta = sicd_to_sicd(sicd_pixels, sicd_meta,
                                                new_window, window_name, new_params)

            with benchmark.howlong('write'):
                with sarpy.io.complex.sicd.SICDWriter(str(config.output_sicd), new_meta) as writer:
                    writer.write_chip(new_pixels)


if __name__ == '__main__':
    main()
