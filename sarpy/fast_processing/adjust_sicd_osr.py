"""Utility for changing the sampling rate of a SICD"""

__classification__ = "UNCLASSIFIED"

import copy
import logging

import numpy as np
import scipy.fft

import sarpy.fast_processing.backend
from sarpy.fast_processing import benchmark
from sarpy.fast_processing import deskew
from sarpy.fast_processing import read_sicd


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
    for axis in ('Row', 'Col'):
        resamp_params = _get_sicd_resamp_params(mdata, axis, desired_osr)
        if resamp_params['fft1_size'] == resamp_params['fft2_size']:
            continue

        # deskew
        with benchmark.howlong(f"{axis} deskew"):
            deskew_data, mdata = deskew.sicd_to_sicd(inwork_data, mdata, axis)
            if inwork_data is not data:
                del inwork_data

        # FFT # Mult # IFFT
        with benchmark.howlong(f"{axis} fft_pad_ifft"):
            inwork_data = _fft_pad_ifft(deskew_data, axis, resamp_params)
            del deskew_data
            mdata = updated_sicd_metadata(mdata, axis, resamp_params)

    return inwork_data, mdata


def _fft_pad_ifft(cdata, axis, resamp_params):

    # Make resamp params local
    fft1_size = resamp_params['fft1_size']
    fft2_size = resamp_params['fft2_size']
    insert_offset = resamp_params['insert_offset']
    num_samps_in = resamp_params['num_samps_in']
    frac_shift = resamp_params['frac_shift']
    extract_offset = resamp_params['extract_offset']
    num_samps_out = resamp_params['num_samps_out']

    axis_index = {'Row': 0, 'Col': 1}[axis]
    fft1_shape = list(cdata.shape)
    fft1_shape[axis_index] = fft1_size
    fft2_shape = list(cdata.shape)
    fft2_shape[axis_index] = fft2_size
    out_shape = list(cdata.shape)
    out_shape[axis_index] = num_samps_out

    with benchmark.howlong("fft1 in copy"):
        # Fourier Transform input data
        fft1_buff = np.zeros_like(cdata, shape=fft1_shape)
        fft1_in_slices = [slice(None), slice(None)]
        fft1_in_slices[axis_index] = slice(insert_offset, insert_offset + num_samps_in)
        fft1_buff[tuple(fft1_in_slices)] = cdata

    with benchmark.howlong("fft"):
        # Perform forward transform
        fft1 = scipy.fft.fft(fft1_buff, n=fft1_size, axis=axis_index, norm="forward", workers=-1)
        del fft1_buff

    with benchmark.howlong("fft transfer copy"):
        # Copy from fft1 to fft2
        min_size = min(fft1_size, fft2_size)
        neg_start = min_size//2
        pos_end = min_size - neg_start
        fft_transfer_slices1 = [slice(None), slice(None)]
        fft_transfer_slices1[axis_index] = (slice(-neg_start, None))
        fft_transfer_slices2 = [slice(None), slice(None)]
        fft_transfer_slices2[axis_index] = (slice(None, pos_end))
        fft2_buff = np.zeros_like(cdata, shape=fft2_shape)
        fft2_buff[tuple(fft_transfer_slices1)] = fft1[tuple(fft_transfer_slices1)]
        fft2_buff[tuple(fft_transfer_slices2)] = fft1[tuple(fft_transfer_slices2)]
        del fft1

    with benchmark.howlong("apply phase"):
        # Apply phase shift so that the reference index will be an integer
        phase_vec = np.exp(2*np.pi*1j*scipy.fft.fftfreq(fft2_size) * frac_shift).astype(np.complex64)
        phase_vec_slices = [np.newaxis, np.newaxis]
        phase_vec_slices[axis_index] = slice(None)
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


def updated_sicd_metadata(sicd_metadata, axis, resamp_params):
    """Update SICD metadata to new reference location and sample spacing"""

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

    return mdata


def _get_sicd_resamp_params(mdata, direction, desired_osr):
    grid_dir = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[direction]
    scp_index = {'Row': mdata.ImageData.SCPPixel.Row - mdata.ImageData.FirstRow,
                 'Col': mdata.ImageData.SCPPixel.Col - mdata.ImageData.FirstCol}[direction]
    num_samps = {'Row': mdata.ImageData.NumRows, 'Col': mdata.ImageData.NumCols}[direction]
    current_osr = 1 / (grid_dir.SS * grid_dir.ImpRespBW)
    minimum_pad = int(min(200 * current_osr, 0.1 * num_samps))
    fft1_size = scipy.fft.next_fast_len(num_samps + minimum_pad)
    effective_pad = fft1_size - num_samps
    insert_offset = effective_pad // 2
    fft2_size = scipy.fft.next_fast_len(int(np.ceil(fft1_size * desired_osr / current_osr)))
    rsr = fft2_size / fft1_size

    padded_scp_index = scp_index + insert_offset
    resampled_scp_index = padded_scp_index * rsr
    floor_scp_index = int(np.floor(resampled_scp_index))
    frac_shift = resampled_scp_index - floor_scp_index
    extract_offset = int(np.floor(insert_offset * rsr))
    num_samps_out = int(np.ceil((insert_offset + num_samps - 1) * rsr)) - extract_offset + 1
    resampled_scp_index = floor_scp_index - extract_offset

    return {'fft1_size': fft1_size,
            'fft2_size': fft2_size,
            'insert_offset': insert_offset,
            'num_samps_in': num_samps,
            'frac_shift': frac_shift,
            'extract_offset': extract_offset,
            'num_samps_out': num_samps_out,
            'resampled_scp_index': resampled_scp_index,
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
