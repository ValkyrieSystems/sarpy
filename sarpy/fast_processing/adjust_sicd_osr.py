"""Utility for changing the sampling rate of a SICD"""

__classification__ = "UNCLASSIFIED"

import copy
import logging

import numba
import numpy as np
import numpy.polynomial.polynomial as npp
import scipy.fft

from sarpy.fast_processing import read_sicd
import sarpy.fast_processing.backend
from sarpy.fast_processing import benchmark


def sicd_to_sicd(data, sicd_metadata, desired_osr):
    """Enforce a minimum oversample ratio to a SICD

    Args
    ----
    data: `numpy.ndarray`
        SICD pixels
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    desired_osr: float
        Desired osr for output data

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
    for direction in ('Row', 'Col'):
        resamp_params = _get_sicd_resamp_params(mdata, direction, desired_osr)
        if resamp_params['fft1_size'] == resamp_params['fft2_size']:
            continue

        # deskew
        with benchmark.howlong(f"{direction} deskew"):
            deskew_data = _deskew(inwork_data, mdata, direction, forward=False)
            del inwork_data

        # FFT # Mult # IFFT
        with benchmark.howlong(f"{direction} fft_pad_ifft"):
            upsamp_data = _fft_pad_ifft(deskew_data, direction, resamp_params)
            del deskew_data
            mdata = updated_sicd_metadata(mdata, direction, resamp_params)

        with benchmark.howlong(f"{direction} reskew"):
            # reskew?  # TODO update metadata instead of reskewing
            inwork_data = _deskew(upsamp_data, mdata, direction, forward=True)
            del upsamp_data

    return inwork_data, mdata


@numba.njit(parallel=True)
def _apply_phase_poly(
        input_data,
        phase_poly,
        row_0,
        row_ss,
        col_0,
        col_ss):
    """numba parallelized phase poly application"""
    out = np.empty_like(input_data)
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

            out[rowidx, colidx] = input_data[rowidx, colidx] * np.exp(1j*2*np.pi*phase_val)

    return out


def _deskew(cdata, mdata, axis, forward):
    axis_index = {'Row': 0, 'Col': 1}[axis]
    axis_mdata = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[axis]

    row_ss = mdata.Grid.Row.SS
    row_0 = (mdata.ImageData.FirstRow - mdata.ImageData.SCPPixel.Row) * row_ss
    col_ss = mdata.Grid.Col.SS
    col_0 = (mdata.ImageData.FirstCol - mdata.ImageData.SCPPixel.Col) * col_ss

    delta_k_coa_poly = np.array([[0.0]]) if axis_mdata.DeltaKCOAPoly is None else axis_mdata.DeltaKCOAPoly.Coefs

    if not np.all(delta_k_coa_poly == 0):
        phase_poly = npp.polyint(delta_k_coa_poly, axis=axis_index) * axis_mdata.Sgn
        if forward:
            phase_poly *= -1
        with benchmark.howlong('apply_skew'):
            cdata = _apply_phase_poly(cdata, phase_poly, row_0, row_ss, col_0, col_ss)
    return cdata


def _fft_pad_ifft(cdata, direction, resamp_params):

    # Make resamp params local
    fft1_size = resamp_params['fft1_size']
    fft2_size = resamp_params['fft2_size']
    insert_offset = resamp_params['insert_offset']
    num_samps_in = resamp_params['num_samps_in']
    frac_shift = resamp_params['frac_shift']
    extract_offset = resamp_params['extract_offset']
    num_samps_out = resamp_params['num_samps_out']

    axis_index = {'Row': 0, 'Col': 1}[direction]
    fft1_shape = list(cdata.shape)
    fft1_shape[axis_index] = fft1_size
    fft2_shape = list(cdata.shape)
    fft2_shape[axis_index] = fft2_size
    out_shape = list(cdata.shape)
    out_shape[axis_index] = num_samps_out

    with benchmark.howlong("fft"):
        # Fourier Transform input data
        fft1_buff = np.zeros_like(cdata, shape=fft1_shape)
        fft1_in_slices = [slice(None), slice(None)]
        fft1_in_slices[axis_index] = slice(insert_offset, insert_offset + num_samps_in)
        fft1_buff[tuple(fft1_in_slices)] = cdata
        fft1 = scipy.fft.fft(fft1_buff, n=fft1_size, axis=axis_index, norm="forward", workers=-1)
        del fft1_buff

    with benchmark.howlong("apply phase"):
        # Apply phase shift so that the reference index will be an integer
        phase_vec = np.exp(2*np.pi*1j*scipy.fft.fftfreq(fft2_size) * frac_shift).astype(np.complex64)
        phase_vec_slices = [np.newaxis, np.newaxis]
        phase_vec_slices[axis_index] = slice(None)

        fft2_buff = np.zeros_like(cdata, shape=fft2_shape)
        min_size = min(fft1_size, fft2_size)
        neg_start = min_size//2
        pos_end = min_size - neg_start
        fft_transfer_slices1 = [slice(None), slice(None)]
        fft_transfer_slices1[axis_index] = (slice(-neg_start, None))
        fft_transfer_slices2 = [slice(None), slice(None)]
        fft_transfer_slices2[axis_index] = (slice(None, pos_end))
        fft2_buff[tuple(fft_transfer_slices1)] = fft1[tuple(fft_transfer_slices1)]
        fft2_buff[tuple(fft_transfer_slices2)] = fft1[tuple(fft_transfer_slices2)]
        del fft1
        fft2_buff = fft2_buff * phase_vec[tuple(phase_vec_slices)]

    with benchmark.howlong("ifft"):
        # Back Transform data to desired sampling
        fft2 = scipy.fft.ifft(fft2_buff, n=fft2_size, axis=axis_index, norm="forward", workers=-1)
        del fft2_buff

    with benchmark.howlong("crop output"):
        out_cdata = np.zeros_like(cdata, shape=out_shape)
        fft2_out_slices = [slice(None), slice(None)]
        fft2_out_slices[axis_index] = slice(extract_offset, extract_offset + num_samps_out)
        out_cdata = fft2[tuple(fft2_out_slices)]
        del fft2

    return out_cdata


def updated_sicd_metadata(sicd_metadata, direction, resamp_params):
    """Update SICD metadata to new reference location and sample spacing"""

    mdata = copy.deepcopy(sicd_metadata)
    if direction == 'Row':
        mdata.Grid.Row.SS = sicd_metadata.Grid.Row.SS / resamp_params['resample_rate']
        mdata.ImageData.SCPPixel.Row = resamp_params['new_ref_index']
        mdata.ImageData.NumRows = resamp_params['num_samps_out']
        mdata.ImageData.FullImage.NumRows = resamp_params['num_samps_out']
        if mdata.ImageData.ValidData:
            for rowcol in mdata.ImageData.ValidData:
                rowcol.Row = int(round((rowcol.Row - sicd_metadata.ImageData.SCPPixel.Row) * resamp_params['resample_rate']
                                       + resamp_params['new_ref_index']))
    else:
        mdata.Grid.Col.SS = sicd_metadata.Grid.Col.SS / resamp_params['resample_rate']
        mdata.ImageData.SCPPixel.Col = resamp_params['new_ref_index']
        mdata.ImageData.NumCols = resamp_params['num_samps_out']
        mdata.ImageData.FullImage.NumCols = resamp_params['num_samps_out']
        if mdata.ImageData.ValidData:
            for rowcol in mdata.ImageData.ValidData:
                rowcol.Col = int(round((rowcol.Col - sicd_metadata.ImageData.SCPPixel.Col) * resamp_params['resample_rate']
                                       + resamp_params['new_ref_index']))

    return mdata


def _get_sicd_resamp_params(mdata, direction, desired_osr):
    grid_dir = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[direction]
    ref_index = {'Row': mdata.ImageData.SCPPixel.Row, 'Col': mdata.ImageData.SCPPixel.Col}[direction]
    num_samps = {'Row': mdata.ImageData.NumRows, 'Col': mdata.ImageData.NumCols}[direction]
    current_osr = 1 / (grid_dir.SS * grid_dir.ImpRespBW)
    minimum_pad = int(min(200 * current_osr, 0.1 * num_samps))
    fft1_size = scipy.fft.next_fast_len(num_samps + minimum_pad)
    fft2_size = scipy.fft.next_fast_len(int(np.ceil(fft1_size * desired_osr / current_osr)))
    rsr = fft2_size / fft1_size
    effective_pad = fft1_size - num_samps
    fft1_in_offset = effective_pad // 2

    padded_ref_index = ref_index + fft1_in_offset
    resampled_index = padded_ref_index * rsr
    output_index = int(np.floor(resampled_index))
    frac_index = resampled_index - output_index
    keep_low = int(np.ceil(ref_index * rsr))
    keep_high = int(np.ceil(((num_samps - 1) - ref_index) * rsr))
    keep_total = keep_low + keep_high + 1
    fft2_out_offset = output_index - keep_low

    return {'fft1_size': fft1_size,
            'fft2_size': fft2_size,
            'insert_offset': fft1_in_offset,
            'num_samps_in': num_samps,
            'frac_shift': frac_index,
            'extract_offset': fft2_out_offset,
            'num_samps_out': keep_total,
            'new_ref_index': keep_low,
            'resample_rate': rsr}


def main(args=None):
    """CLI for changing the sampling rate of a SICD"""
    import argparse
    import pathlib
    import sarpy.io.complex
    import sarpy.io.complex.sicd

    parser = argparse.ArgumentParser()
    parser.add_argument('input_sicd', type=pathlib.Path)
    parser.add_argument('output_sicd', type=pathlib.Path)
    parser.add_argument('--desired-osr', required=True, type=float,
                        help="Desired oversample ratio for output SICD")
    parser.add_argument('--fft-backend', choices=['auto', 'mkl', 'scipy'], default='auto',
                        help="Which FFT backend to use.  Default 'auto', which will use mkl if available")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose logging (may be repeated)")
    config = parser.parse_args(args)

    loglevels = [logging.WARNING, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[min(config.verbose, len(loglevels)-1)])

    with sarpy.fast_processing.backend.set_fft_backend(config.fft_backend):
        with benchmark.howlong('adjust_oversample_ratio'):
            with benchmark.howlong('read'):
                sicd_pixels, sicd_meta = read_sicd.read_from_file(config.input_sicd)

            new_pixels, new_meta = sicd_to_sicd(sicd_pixels, sicd_meta, config.desired_osr)

            with benchmark.howlong('write'):
                with sarpy.io.complex.sicd.SICDWriter(str(config.output_sicd), new_meta) as writer:
                    writer.write_chip(new_pixels)


if __name__ == '__main__':
    main()
