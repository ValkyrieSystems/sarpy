"""Utility for changing the sidelobe control of a SICD and adjusting the oversample ratio"""

__classification__ = "UNCLASSIFIED"

import copy
import logging

import numpy as np
import scipy.fft
import scipy.interpolate as spi

from sarpy.io.complex.sicd_elements.Grid import WgtTypeType
from sarpy.processing.sicd.spectral_taper import Taper
import sarpy.processing.sicd.windows as windows  # TODO migrate to sarpy2

import sarpy.fast_processing.backend
import sarpy.fast_processing.metadata
from sarpy.fast_processing import adjust_sicd_osr
from sarpy.fast_processing import benchmark
from sarpy.fast_processing import deskew
from sarpy.fast_processing import read_sicd
from sarpy.fast_processing import sidelobe_control
from sarpy.fast_processing import utils
from sarpy.fast_processing import write_sicd


def sicd_to_sicd(data, sicd_metadata, desired_osr,
                 new_weights, window_name, window_parameters=None):
    """Apply sidelobe weighting to a SICD

    Args
    ----
    data: `numpy.ndarray`
        SICD pixels
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    desired_osr: float
        Desired osr for output data
    new_weights: `numpy.ndarray`
        1-D array of desired weighting.  Will be applied to both row and col axes.
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
    mdata = sicd_metadata
    inwork_data = data
    data = None
    new_weights = np.asarray(new_weights)
    for axis_index, axis in enumerate(('Row', 'Col')):
        resamp_params = _get_sicd_resamp_params(mdata, axis, desired_osr)
        existing_weights = _get_sicd_wgt_funct(mdata, axis, len(new_weights))
        both_windows = new_weights / np.maximum(existing_weights, 0.01 * np.max(existing_weights))
        if np.allclose(both_windows, 1) and resamp_params['fft1_size'] == resamp_params['fft2_size']:
            continue

        # deskew
        with benchmark.howlong(f"{axis} deskew"):
            deskew_data, mdata = deskew.sicd_to_sicd(inwork_data, mdata, axis)
            inwork_data = None

        fft1_shape = list(deskew_data.shape)
        fft1_shape[axis_index] = resamp_params['fft1_size']
        fft2_shape = list(deskew_data.shape)
        fft2_shape[axis_index] = resamp_params['fft2_size']
        out_shape = list(deskew_data.shape)
        out_shape[axis_index] = resamp_params['num_samps_out']

        with benchmark.howlong(f"{axis} fft1 in copy"):
            # Fourier Transform input data
            fft1_buff = np.zeros(shape=fft1_shape, dtype=deskew_data.dtype)
            fft1_in_slices = [slice(None), slice(None)]
            fft1_in_slices[axis_index] = slice(resamp_params['insert_offset'], resamp_params['insert_offset'] + resamp_params['num_samps_in'])
            utils.parallel_copyto(fft1_buff[tuple(fft1_in_slices)], deskew_data)
            deskew_data = None

        with benchmark.howlong(f"{axis} fft1"):
            # Forward transform without FFTSHIFT so the DC bin is at index=0
            if resamp_params['sgn'] == -1:
                fft1_data = scipy.fft.fft(fft1_buff, n=resamp_params['fft1_size'], axis=axis_index, norm="forward",  workers=-1)
            else:
                fft1_data = scipy.fft.ifft(fft1_buff, n=resamp_params['fft1_size'], axis=axis_index, norm="backward", workers=-1)
            fft1_buff = None

        with benchmark.howlong(f"compute {axis} taper"):
            # Interpolate the taper to cover the spectral support bandwidth and extend the
            # taper window's end points into the over sample region of the spectrum.
            func = spi.interp1d(np.linspace(-resamp_params['ipr_bw'] / 2, resamp_params['ipr_bw'] / 2,
                                            len(both_windows)),
                                both_windows, kind='cubic',
                                bounds_error=False, fill_value=(both_windows[0], both_windows[-1]))
            padded_taper = func(scipy.fft.fftfreq(resamp_params['fft1_size'], resamp_params['ss']))

            taper_2d = (padded_taper[:, np.newaxis] if axis == 'Row' else padded_taper[np.newaxis, :]).astype(fft1_data.dtype)

        with benchmark.howlong(f"apply {axis} taper"):
            fft1_data *= taper_2d

        with benchmark.howlong(f"{axis} fft transfer copy"):
            # Copy from fft1 to fft2
            min_size = min(resamp_params['fft1_size'], resamp_params['fft2_size'])
            neg_start = min_size//2
            pos_end = min_size - neg_start
            fft_transfer_slices1 = [slice(None), slice(None)]
            fft_transfer_slices1[axis_index] = (slice(-neg_start, None))
            fft_transfer_slices2 = [slice(None), slice(None)]
            fft_transfer_slices2[axis_index] = (slice(None, pos_end))
            fft2_buff = np.zeros(shape=fft2_shape, dtype=fft1_data.dtype)
            utils.parallel_copyto(fft2_buff[tuple(fft_transfer_slices1)],
                                  fft1_data[tuple(fft_transfer_slices1)])
            utils.parallel_copyto(fft2_buff[tuple(fft_transfer_slices2)],
                                  fft1_data[tuple(fft_transfer_slices2)])
            fft1_data = None

        with benchmark.howlong(f"{axis} apply phase"):
            # Apply phase shift so that the reference index will be an integer
            phase_vec = np.exp(-resamp_params['sgn']*2*np.pi*1j*scipy.fft.fftfreq(resamp_params['fft2_size'])
                               * resamp_params['frac_shift']).astype(np.complex64)
            phase_vec_slices = [np.newaxis, np.newaxis]
            phase_vec_slices[axis_index] = slice(None)
            fft2_buff *= phase_vec[tuple(phase_vec_slices)]

        with benchmark.howlong(f"{axis} ifft"):
            # Inverse transform without FFTSHIFT and trim back to the original image size.
            if resamp_params['sgn'] == -1:
                fft2_data = scipy.fft.ifft(fft2_buff, n=resamp_params['fft2_size'], axis=axis_index, norm="forward", workers=-1)
            else:
                fft2_data = scipy.fft.fft(fft2_buff, n=resamp_params['fft2_size'], axis=axis_index, norm="backward", workers=-1)
            fft2_buff = None

        with benchmark.howlong("crop output"):
            inwork_data = np.zeros(shape=out_shape, dtype=fft2_data.dtype)
            fft2_out_slices = [slice(None), slice(None)]
            fft2_out_slices[axis_index] = slice(resamp_params['extract_offset'], resamp_params['extract_offset'] + resamp_params['num_samps_out'])
            utils.parallel_copyto(inwork_data,
                                  fft2_data[tuple(fft2_out_slices)])
            fft2_data = None

        mdata = updated_sicd_metadata(mdata, axis, resamp_params, existing_weights,
                                      new_weights, window_name, window_parameters)

    return inwork_data, mdata


def updated_sicd_metadata(sicd_metadata, axis, resamp_params, existing_weights,
                          new_weights, window_name, window_parameters=None):
    """Update SICD metadata to describe a new weighting

    Args
    ----
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    axis: str
        SICD axis for which to update parameters.  'Row' or 'Col'
    resamp_params: dict
        Resampling parameters used
    existing_weights: `numpy.ndarray`
        1-D array of existing weighting.
    new_weights: `numpy.ndarray`
        1-D array of desired weighting.  Will be applied in both row and col directions.
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
    taper_is_uniform = np.all(new_weights == new_weights[0])
    ipr_half_power_width = windows.find_half_power(new_weights, oversample=16)

    mdata = copy.deepcopy(sicd_metadata)

    if axis == 'Row':
        mdata.Grid.Row.SS = sicd_metadata.Grid.Row.SS / resamp_params['resample_rate']
        mdata.ImageData.SCPPixel.Row = resamp_params['resampled_scp_index']
        mdata.ImageData.FirstRow = 0
        mdata.ImageData.NumRows = resamp_params['num_samps_out']
        mdata.ImageData.FullImage.NumRows = resamp_params['num_samps_out']
        if mdata.ImageData.ValidData:
            for rowcol in mdata.ImageData.ValidData:
                rowcol.Row = int(round((rowcol.Row - sicd_metadata.ImageData.SCPPixel.Row) * resamp_params['resample_rate']
                                       + resamp_params['resampled_scp_index']))
        mdata.Grid.Row.WgtType = WgtTypeType(WindowName=window_name.upper(), Parameters=window_parameters)
        mdata.Grid.Row.WgtFunct = None if taper_is_uniform else new_weights
        mdata.Grid.Row.ImpRespWid = ipr_half_power_width / mdata.Grid.Row.ImpRespBW
    else:
        mdata.Grid.Col.SS = sicd_metadata.Grid.Col.SS / resamp_params['resample_rate']
        mdata.ImageData.SCPPixel.Col = resamp_params['resampled_scp_index']
        mdata.ImageData.FirstCol = 0
        mdata.ImageData.NumCols = resamp_params['num_samps_out']
        mdata.ImageData.FullImage.NumCols = resamp_params['num_samps_out']
        if mdata.ImageData.ValidData:
            for rowcol in mdata.ImageData.ValidData:
                rowcol.Col = int(round((rowcol.Col - sicd_metadata.ImageData.SCPPixel.Col) * resamp_params['resample_rate']
                                       + resamp_params['resampled_scp_index']))
        mdata.Grid.Col.WgtType = WgtTypeType(WindowName=window_name.upper(), Parameters=window_parameters)
        mdata.Grid.Col.WgtFunct = None if taper_is_uniform else new_weights
        mdata.Grid.Col.ImpRespWid = ipr_half_power_width / mdata.Grid.Col.ImpRespBW

    # Update the SICD metadata to account for the spectral weighting changes
    old_coh_amp_gain = np.mean(existing_weights)
    old_rms_amp_gain = np.sqrt(np.mean(existing_weights**2))

    new_coh_amp_gain = np.mean(new_weights)
    new_rms_amp_gain = np.sqrt(np.mean(new_weights**2))

    coh_pwr_gain = (new_coh_amp_gain / old_coh_amp_gain) ** 2
    rms_pwr_gain = (new_rms_amp_gain / old_rms_amp_gain) ** 2

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
    return sidelobe_control._get_sicd_wgt_funct(mdata, axis, desired_size)


def _fit_1d_window(window_vals, desired_size):
    return sidelobe_control._fit_1d_window(window_Vals, desired_size)


def _get_sicd_resamp_params(mdata, direction, desired_osr):
    return adjust_sicd_osr._get_sicd_resamp_params(mdata, direction, desired_osr)


def main(args=None):
    """CLI for changing the sidelobe control of a SICD and adjusting the oversample ratio"""
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument('input_sicd', type=pathlib.Path)
    parser.add_argument('output_sicd', type=pathlib.Path)
    parser.add_argument('--sidelobe-control', required=True,
                        choices=['Uniform', 'Taylor'], default='Uniform',
                        help="Desired sidelobe control")
    parser.add_argument('--desired-osr', required=True, type=float,
                        help="Desired oversample ratio for output SICD")
    parser.add_argument('--fft-backend', choices=['auto', 'mkl', 'scipy'], default='auto',
                        help="Which FFT backend to use.  Default 'auto', which will use mkl if available")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose logging (may be repeated)")
    config = parser.parse_args(args)
    assert config.desired_osr > 1.0

    loglevels = [logging.WARNING, logging.INFO, logging.DEBUG]
    loglevel = loglevels[min(config.verbose, len(loglevels)-1)]
    sarpy.fast_processing.backend.initialize_logging(loglevel)

    with sarpy.fast_processing.backend.set_fft_backend(config.fft_backend):
        with benchmark.howlong('Weight and Adjust OSR'):
            with benchmark.howlong('read'):
                sicd_pixels, sicd_meta = read_sicd.read_from_file(config.input_sicd)

            window_name = config.sidelobe_control.upper()
            taper = Taper(window_name)
            new_window = taper.get_vals(65, sym=True)
            new_params = taper.window_pars
            new_pixels, new_meta = sicd_to_sicd(sicd_pixels, sicd_meta, config.desired_osr,
                                                new_window, window_name, new_params)
            sicd_pixels = None

            sarpy.fast_processing.metadata.add_sicd_processing(
                new_meta,
                pathlib.Path(__file__).name,
                parameters={
                    "input_osr_row": 1.0 / (sicd_meta.Grid.Row.SS * sicd_meta.Grid.Row.ImpRespBW),
                    "input_osr_col": 1.0 / (sicd_meta.Grid.Col.SS * sicd_meta.Grid.Col.ImpRespBW),
                    "sidelobe_control": config.sidelobe_control,
                    "desired_osr": config.desired_osr,
                    "fft_backend": config.fft_backend,
                },
            )


            with benchmark.howlong('write'):
                write_sicd.write_to_file(config.output_sicd, new_pixels, new_meta)


if __name__ == '__main__':
    main()
