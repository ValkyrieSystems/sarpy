__classification__ = "UNCLASSIFIED"

import copy

import numba
import numpy as np
import scipy.fft

from sarpy.fast_processing import benchmark
from sarpy.fast_processing import deskew

# from AGI ADD (IIQ SVA parameters)
EDGE_GLINT_THRESHOLD = 0.2
EDGE_GLINT_MAX_WEIGHT = 0.45


@numba.njit
def copy_boundaries(base_data, new_data, m, n):
    """Copy over boundary values"""
    new_data[:m, :] = base_data[:m, :]
    new_data[-m:, :] = base_data[-m:, :]
    new_data[m:-m, :n] = base_data[m:-m, :n]
    new_data[m:-m, -n:] = base_data[m:-m, -n:]


@numba.njit
def check_diff_signs(num1, num2):
    """Check if two values have different signs"""
    return True if np.sign(num1) != np.sign(num2) else False


@numba.njit
def compute_min_abs(c1, c2):
    """Return value with the minimum absolute value"""
    if abs(c1) < abs(c2):
        return c1
    else:
        return c2


@numba.njit
def compute_e(data, m, n, osr):
    """Compute value of E to test for edge-glint retention"""
    v1 = data[m, n - osr // 2] - data[m, n + osr // 2]
    v2 = data[m, n] - data[m, n - osr] - data[m, n + osr]

    # prevent divide by zero
    E_denom = v1.real**2 + v1.imag**2
    if E_denom == 0:
        E_denom = np.finfo(data.real.dtype).tiny

    E = 2 * abs(v1.real * v2.real + v1.imag * v2.imag) / (3 * E_denom)

    return E


@numba.njit
def compute_uncoup_conv(comp, wm, wn, qm, qn, p):
    """Compute convolution for uncoupled SVA using given weights"""
    return comp + wm * (wn * p + qm) + wn * qn


@numba.njit
def min_uncoup_conv(comp, qm, qn, p, wn_max):
    """Compute minimum convolution for uncoupled SVA between different weights"""
    # compute convolution using (wm, wn) = (0, wn_max), (1/2, 0) and (1/2, wn_max)
    wm_min = 0
    wm_max = 0.5
    wn_min = 0

    conv1 = compute_uncoup_conv(comp, wm_min, wn_max, qm, qn, p)
    conv2 = compute_uncoup_conv(comp, wm_max, wn_min, qm, qn, p)
    conv3 = compute_uncoup_conv(comp, wm_max, wn_max, qm, qn, p)

    # test if convolution signs are opposite of unweighted component
    for conv in [conv1, conv2, conv3]:
        if check_diff_signs(comp, conv):
            # set pixel value to 0
            min_conv = 0.0
            return min_conv

    # otherwise, take the value with the minimum magnitude
    min_conv = comp
    for conv in [conv1, conv2, conv3]:
        min_conv = compute_min_abs(min_conv, conv)

    return min_conv


@numba.njit(parallel=True)
def uncoup_sva(
    data,
    row_kctr_poly_rad,
    col_kctr_poly_rad,
    edge_glint_threshold=EDGE_GLINT_THRESHOLD,
    edge_glint_max_weight=EDGE_GLINT_MAX_WEIGHT,
):
    """
    Perform separate-IQ, dimensions-uncoupled SVA on a 2x Nyquist image in both the row and column directions.

    Implementation of IQ-Separately, Two Dimensions Uncoupled SVA based on:
    Stankwitz, H.C. et al. "Nonlinear Apodization for Sidelobe Control in SAR Imagery" IEEE Trans Aero. and Elec. Systems, Vol. 31, 267 (1995)

    This algorithm assumes an image sampled at 2x Nyquist, which differs from the paper.

    The implementation of edge-glint retention (EGR) originates from the SVA algorithm definition found in AGI ADD 2.8.16.
    """
    # create new data array and copy boundary values
    sva_data = np.zeros_like(data)
    copy_boundaries(data, sva_data, 2, 2)

    # calculate poly coefficients per 2 samples
    row_kctr_poly_rad_per_2_samp = np.asarray(row_kctr_poly_rad) * 2
    col_kctr_poly_rad_per_2_samp = np.asarray(col_kctr_poly_rad) * 2

    for m in numba.prange(2, data.shape[0] - 2):
        # evaluate polys
        row_kctr_col_poly = row_kctr_poly_rad_per_2_samp[-1, :]
        for ndx in range(row_kctr_poly_rad_per_2_samp.shape[0] - 2, -1, -1):
            row_kctr_col_poly = (
                row_kctr_col_poly * m + row_kctr_poly_rad_per_2_samp[ndx, :]
            )
        col_kctr_col_poly = col_kctr_poly_rad_per_2_samp[-1, :]
        for ndx in range(col_kctr_poly_rad_per_2_samp.shape[0] - 2, -1, -1):
            col_kctr_col_poly = (
                col_kctr_col_poly * m + col_kctr_poly_rad_per_2_samp[ndx, :]
            )

        for n in numba.prange(2, data.shape[1] - 2):
            row_kctr = row_kctr_col_poly[-1]
            for ndx in range(row_kctr_col_poly.shape[0] - 2, -1, -1):
                row_kctr = row_kctr * n + row_kctr_col_poly[ndx]
            col_kctr = col_kctr_col_poly[-1]
            for ndx in range(col_kctr_col_poly.shape[0] - 2, -1, -1):
                col_kctr = col_kctr * n + col_kctr_col_poly[ndx]

            # create phasors
            pos_row_phasor = np.exp(1j * row_kctr)
            neg_row_phasor = np.conjugate(pos_row_phasor)

            pos_col_phasor = np.exp(1j * col_kctr)
            neg_col_phasor = np.conjugate(pos_col_phasor)

            tc = data[m - 2, n] * neg_row_phasor
            bc = data[m + 2, n] * pos_row_phasor
            ml = data[m, n - 2] * neg_col_phasor
            mr = data[m, n + 2] * pos_col_phasor
            tl = data[m - 2, n - 2] * neg_row_phasor * neg_col_phasor
            br = data[m + 2, n + 2] * pos_row_phasor * pos_col_phasor
            tr = data[m - 2, n + 2] * neg_row_phasor * pos_col_phasor
            bl = data[m + 2, n - 2] * pos_row_phasor * neg_col_phasor

            # apply edge-glint retention
            wn_max = 0.5
            if edge_glint_threshold > 0:
                E = compute_e(data, m, n, 2)
                # test whether to apply EGR
                if E <= edge_glint_threshold:
                    wn_max = edge_glint_max_weight

            # real component values
            real_compon = data[m, n].real
            qm = tc.real + bc.real
            qn = ml.real + mr.real
            p = tl.real + br.real + tr.real + bl.real

            # get minimum real convolution using different weightings
            real_conv = min_uncoup_conv(real_compon, qm, qn, p, wn_max)

            # imaginary component values
            imag_compon = data[m, n].imag
            qm = tc.imag + bc.imag
            qn = ml.imag + mr.imag
            p = tl.imag + br.imag + tr.imag + bl.imag

            # get minimum imaginary convolution using different weightings
            imag_conv = min_uncoup_conv(imag_compon, qm, qn, p, wn_max)

            sva_data[m, n] = complex(real_conv, imag_conv)
    return sva_data


@numba.njit
def compute_w1(round_osr, ce_osr, ws):
    """
    Compute the value of w1 used in the D-SVA algorithm.

    For oversample ratios where w1 takes on a negative value, the implementation of D-SVA as stated in the paper does not perform as expected.
    To correct this, a different computation for w1 is introduced for the affected osrs. This allows the algorithm to perform more
    consistently across all osrs.
    """
    calc = np.sinc(ce_osr * ws) - np.cos(np.pi * ce_osr * ws)
    # make adjustment for negative w1
    if calc < 0:
        if round_osr == ce_osr:
            w1 = 1 / (np.pi / 2 * np.abs(calc))
        else:
            w1 = 1 / (
                np.pi / 2 * (np.sinc(round_osr * ws) - np.cos(np.pi * round_osr * ws))
            )
    else:
        if round_osr == ce_osr:
            w1 = 1 / (2 * calc)
        else:
            w1 = 1 / (2 * (np.sinc(round_osr * ws) - np.cos(np.pi * round_osr * ws)))
    return w1


@numba.njit
def compute_a(osr, ws, w1):
    """Compute the value of a used in D-SVA"""
    a = 1 - 2 * w1 * np.sinc(ws * osr)
    return a


@numba.njit
def min_d_sva_conv(compon, q, w1, a):
    """Compute the minimum convolution for D-SVA"""
    conv = a * compon + w1 * q
    # if signs are different, set pixel value to 0
    if check_diff_signs(compon, conv):
        min_conv = 0
    # otherwise, set pixel value to the minimum magnitude
    else:
        min_conv = compute_min_abs(compon, conv)
    return min_conv


@numba.njit
def d_sva_helper(
    data,
    w1_row,
    w1_col,
    row_osr,
    col_osr,
    m,
    n,
    a_fl,
    a_col,
    pos_row_phasor,
    neg_row_phasor,
    pos_col_phasor,
    neg_col_phasor,
):
    """Compute the minimum convolution for D-SVA using both floor and ceiling oversample ratios"""
    tc = data[m - row_osr, n] * neg_row_phasor
    bc = data[m + row_osr, n] * pos_row_phasor
    ml = data[m, n - col_osr] * neg_col_phasor
    mr = data[m, n + col_osr] * pos_col_phasor

    # real component values
    real_compon = data[m, n].real
    qm = tc.real + bc.real
    qn = ml.real + mr.real
    real_conv_m = min_d_sva_conv(real_compon, qm, w1_row, a_fl)
    real_conv_n = min_d_sva_conv(real_compon, qn, w1_col, a_col)

    # compare real convolution computed along the m and n axes, choose the minimum magnitude
    real_conv = compute_min_abs(real_conv_m, real_conv_n)

    # imaginary component values
    imag_compon = data[m, n].imag
    qm = tc.imag + bc.imag
    qn = ml.imag + mr.imag
    imag_conv_m = min_d_sva_conv(imag_compon, qm, w1_row, a_fl)
    imag_conv_n = min_d_sva_conv(imag_compon, qn, w1_col, a_col)

    # compare imaginary convolution computed along the m and n axes, choose the minimum magnitude
    imag_conv = compute_min_abs(imag_conv_m, imag_conv_n)

    return real_conv, imag_conv


@numba.njit(parallel=True)
def d_sva(
    data,
    row_osr,
    col_osr,
    row_kctr_poly_rad,
    col_kctr_poly_rad,
    edge_glint_threshold=EDGE_GLINT_THRESHOLD,
    edge_glint_max_weight=EDGE_GLINT_MAX_WEIGHT,
):
    """
    Perform Double-SVA on an image in both the row and column directions.

    Works on data sampled at any oversample ratio by performing SVA using both the floor and ceiling osrs.

    Implementation of Double-SVA based on:
    Liu, Min et al. "A Novel Sidelobe Reduction Algorithm Based on Two-Dimensional Sidelobe Correction Using D-SVA for Squint SAR Images."
    Sensors (Basel, Switzerland) vol. 18,3 783. 5 Mar. 2018, doi:10.3390/s18030783

    This implementation has been modified from the paper to work more consistently across all osrs. See the compute_w1 function.

    The implementation of edge-glint retention (EGR) originates from the SVA algorithm description found in AGI ADD 2.8.16.
    """
    # compute floor and ceiling osrs
    fl_row_osr = int(np.floor(row_osr))
    fl_col_osr = int(np.floor(col_osr))

    ce_row_osr = int(np.ceil(row_osr))
    ce_col_osr = int(np.ceil(col_osr))

    # compute values of variables
    ws_row = 1 / row_osr
    ws_col = 1 / col_osr

    w1_fl_row = compute_w1(fl_row_osr, ce_row_osr, ws_row)
    w1_fl_col_no_egr = compute_w1(fl_col_osr, ce_col_osr, ws_col)
    w1_ce_row = compute_w1(ce_row_osr, ce_row_osr, ws_row)
    w1_ce_col_no_egr = compute_w1(ce_col_osr, ce_col_osr, ws_col)

    a_fl_row = compute_a(fl_row_osr, ws_row, w1_fl_row)
    a_fl_col = compute_a(fl_col_osr, ws_col, w1_fl_col_no_egr)
    a_ce_row = compute_a(ce_row_osr, ws_row, w1_ce_row)
    a_ce_col = compute_a(ce_col_osr, ws_col, w1_ce_col_no_egr)

    # create new data array and copy boundary values
    sva_data = np.zeros_like(data)
    copy_boundaries(data, sva_data, ce_row_osr, ce_col_osr)

    # test if given row and column osrs are integer
    has_int_osr = (fl_row_osr == ce_row_osr) and (fl_col_osr == ce_col_osr)

    # convert poly coefficients to numpy array
    row_kctr_poly_rad = np.asarray(row_kctr_poly_rad)
    col_kctr_poly_rad = np.asarray(col_kctr_poly_rad)

    for m in numba.prange(ce_row_osr, data.shape[0] - ce_row_osr):
        # evaluate polys
        row_kctr_col_poly = row_kctr_poly_rad[-1, :]
        for ndx in range(row_kctr_poly_rad.shape[0] - 2, -1, -1):
            row_kctr_col_poly = row_kctr_col_poly * m + row_kctr_poly_rad[ndx, :]
        col_kctr_col_poly = col_kctr_poly_rad[-1, :]
        for ndx in range(col_kctr_poly_rad.shape[0] - 2, -1, -1):
            col_kctr_col_poly = col_kctr_col_poly * m + col_kctr_poly_rad[ndx, :]

        for n in numba.prange(ce_col_osr, data.shape[1] - ce_col_osr):
            row_kctr = row_kctr_col_poly[-1]
            for ndx in range(row_kctr_col_poly.shape[0] - 2, -1, -1):
                row_kctr = row_kctr * n + row_kctr_col_poly[ndx]
            col_kctr = col_kctr_col_poly[-1]
            for ndx in range(col_kctr_col_poly.shape[0] - 2, -1, -1):
                col_kctr = col_kctr * n + col_kctr_col_poly[ndx]

            # use floor osr to create phasors
            fl_row_kctr = fl_row_osr * row_kctr
            fl_col_kctr = fl_col_osr * col_kctr

            fl_pos_row_phasor = np.exp(1j * fl_row_kctr)
            fl_neg_row_phasor = np.conjugate(fl_pos_row_phasor)

            fl_pos_col_phasor = np.exp(1j * fl_col_kctr)
            fl_neg_col_phasor = np.conjugate(fl_pos_col_phasor)

            if not has_int_osr:
                # use ceiling osr to create phasors
                ce_row_kctr = ce_row_osr * row_kctr
                ce_col_kctr = ce_col_osr * col_kctr

                ce_pos_row_phasor = np.exp(1j * ce_row_kctr)
                ce_neg_row_phasor = np.conjugate(ce_pos_row_phasor)

                ce_pos_col_phasor = np.exp(1j * ce_col_kctr)
                ce_neg_col_phasor = np.conjugate(ce_pos_col_phasor)

            # preserve original values
            w1_fl_col = w1_fl_col_no_egr
            w1_ce_col = w1_ce_col_no_egr

            # apply edge-glint retention
            if edge_glint_threshold > 0:
                if (w1_fl_col > edge_glint_max_weight) and (fl_col_osr >= 2):
                    # compute E using floor osr
                    E_fl = compute_e(data, m, n, fl_col_osr)

                    # apply EGR using the floor osr
                    if E_fl < edge_glint_threshold:
                        w1_fl_col = edge_glint_max_weight

                if (w1_ce_col > edge_glint_max_weight) and (ce_col_osr >= 2):
                    # compute E using ceiling osr
                    E_ce = compute_e(data, m, n, ce_col_osr)

                    # apply EGR using ceiling osr
                    if E_ce <= edge_glint_threshold:
                        w1_ce_col = edge_glint_max_weight

            # if osr is integer in both dimensions, compute convolution using only one integer osr in each dimension
            if has_int_osr:
                fl_out = d_sva_helper(
                    data,
                    w1_fl_row,
                    w1_fl_col,
                    fl_row_osr,
                    fl_col_osr,
                    m,
                    n,
                    a_fl_row,
                    a_fl_col,
                    fl_pos_row_phasor,
                    fl_neg_row_phasor,
                    fl_pos_col_phasor,
                    fl_neg_col_phasor,
                )

                real_conv = fl_out[0]
                imag_conv = fl_out[1]
            # otherwise, compute using both floor and ceiling osrs in each dimension
            else:
                fl_out = d_sva_helper(
                    data,
                    w1_fl_row,
                    w1_fl_col,
                    fl_row_osr,
                    fl_col_osr,
                    m,
                    n,
                    a_fl_row,
                    a_fl_col,
                    fl_pos_row_phasor,
                    fl_neg_row_phasor,
                    fl_pos_col_phasor,
                    fl_neg_col_phasor,
                )
                ce_out = d_sva_helper(
                    data,
                    w1_ce_row,
                    w1_ce_col,
                    ce_row_osr,
                    ce_col_osr,
                    m,
                    n,
                    a_ce_row,
                    a_ce_col,
                    ce_pos_row_phasor,
                    ce_neg_row_phasor,
                    ce_pos_col_phasor,
                    ce_neg_col_phasor,
                )
                # choose the pixel values with the minimum magnitude
                real_conv = compute_min_abs(fl_out[0], ce_out[0])
                imag_conv = compute_min_abs(fl_out[1], ce_out[1])

            sva_data[m, n] = complex(real_conv, imag_conv)
    return sva_data


@numba.njit
def pad_row(row, size):
    """Zero pad a row on both sides while preserving center sample"""
    if size > row.shape[0]:
        padded_row = np.zeros(size, dtype=row.dtype)

        # difference between center sample positions in original row and padded row
        start_ndx = size // 2 - row.shape[0] // 2

        end_ndx = start_ndx + row.shape[0]
        padded_row[start_ndx:end_ndx] = row
    else:
        padded_row = np.copy(row)

    return padded_row


def do_fft(data, fft_sign):
    """Perform the fft along with the required fft shifts directly on the given data based on the fft_sign"""
    if fft_sign > 0:
        data[:] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(data)))
    else:
        data[:] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data)))


@numba.njit
def upsample(row, input_osr, upsample_osr, fft_sign):
    """Upsample a row to the given osr"""
    with numba.objmode():
        # fft in
        do_fft(row, fft_sign)

    # calculate upsample size and pad row accordingly
    upsample_size = int(np.round(row.shape[0] / input_osr * upsample_osr))
    upsample_row = pad_row(row, upsample_size)

    with numba.objmode():
        # fft out
        do_fft(upsample_row, -fft_sign)

    return upsample_row


@numba.njit
def jiq_sva_1d(row, osr, edge_glint_threshold, edge_glint_max_weight):
    """
    Perform joint-IQ SVA on one row in the column direction.

    The implementation of edge-glint retention (EGR) originates from the SVA algorithm description found in AGI ADD 2.8.16.
    """
    # get nearby pixel values
    denom = row[: -2 * osr] + row[2 * osr :]
    denom[denom == 0] = np.finfo(row.dtype).tiny

    # obtain weightings
    wgt = -np.real(row[osr:-osr] / denom)
    wgt = np.clip(wgt, 0, 0.5)

    # apply edge-glint retention
    if edge_glint_threshold > 0:
        v1 = (
            row[(osr + 1) // 2 : -(osr + osr // 2)]
            - row[osr + osr // 2 : -((osr + 1) // 2)]
        )
        v1[v1 == 0] = np.finfo(row.dtype).tiny
        v2 = row[osr:-osr] - row[: -2 * osr] - row[2 * osr :]
        E = (
            2
            * np.abs(v1.real * v2.real + v1.imag * v2.imag)
            / (3 * (v1.real**2 + v1.imag**2))
        )
        # test whether to apply EGR for each pixel
        apply_egr = np.logical_and(
            E < edge_glint_threshold, wgt > edge_glint_max_weight
        )
        wgt[apply_egr] = edge_glint_max_weight

    # create results array and copy over boundary values unaffected by SVA
    upsample_results = np.zeros_like(row)
    upsample_results[:osr] = row[:osr]
    upsample_results[-osr:] = row[-osr:]

    # do sva
    upsample_results[osr:-osr] = row[osr:-osr] + wgt * denom

    return upsample_results


@numba.njit(parallel=True)
def joint_iq_sva(
    data,
    input_osr,
    upsample_osr,
    fft_size,
    fft_sign,
    edge_glint_threshold=EDGE_GLINT_THRESHOLD,
    edge_glint_max_weight=EDGE_GLINT_MAX_WEIGHT,
):
    """
    Perform joint-IQ SVA on an image in the column direction.

    Works at any oversample ratio by independently upsampling each row to integer Nyquist and performing SVA.

    Assumes that the spectra is centered in the column direction throughout the entire image.
    """
    if input_osr == upsample_osr:
        # no upsampling required
        sva_results = np.zeros_like(data)
        for m in numba.prange(data.shape[0]):
            row = data[m, :]
            sva_results[m, :] = jiq_sva_1d(
                row, upsample_osr, edge_glint_threshold, edge_glint_max_weight
            )
        return sva_results

    # calculate required increment to obtain closest osr to original image
    inc = int(upsample_osr / input_osr)

    # create results array
    results_row_length = int(np.round(data.shape[1] / input_osr * upsample_osr) // inc)
    results = np.zeros((data.shape[0], results_row_length), dtype=data.dtype)

    for m in numba.prange(data.shape[0]):
        row = data[m, :]

        # zero pad row to given fft_size
        padded_row = pad_row(row, fft_size)

        # upsample row to given osr
        upsample_row = upsample(padded_row, input_osr, upsample_osr, fft_sign)

        # perform sva
        sva_row = jiq_sva_1d(
            upsample_row, upsample_osr, edge_glint_threshold, edge_glint_max_weight
        )

        # calculate start of image in sva_row that preserves the center sample
        ndx = sva_row.shape[0] // 2 - results_row_length // 2 * inc

        # pick every nth pixel (based on inc) to obtain output image
        for n in range(results_row_length):
            results[m, n] = sva_row[ndx]
            ndx += inc

    return results


@numba.njit
def two_dim_joint_iq_sva(
    data,
    row_osr,
    col_osr,
    row_upsample_osr,
    col_upsample_osr,
    row_fft_size,
    col_fft_size,
    fft_sign,
    edge_glint_threshold=EDGE_GLINT_THRESHOLD,
    edge_glint_max_weight=EDGE_GLINT_MAX_WEIGHT,
):
    """
    Perform joint-IQ SVA on an image in both the row and column directions.

    Works on data sampled at any oversampling ratio.

    Assumes that the spectra is centered in both dimensions throughout the entire image.
    """
    # perform sva in the column direction
    results = joint_iq_sva(
        data,
        col_osr,
        col_upsample_osr,
        col_fft_size,
        fft_sign,
        edge_glint_threshold,
        edge_glint_max_weight,
    )

    results = np.transpose(results)

    # perform sva in the row direction
    results = joint_iq_sva(
        results,
        row_osr,
        row_upsample_osr,
        row_fft_size,
        fft_sign,
        edge_glint_threshold=0,
        edge_glint_max_weight=0,
    )

    results = np.transpose(results)

    return results


def _get_jiq_params_sicd(mdata, direction, interim_osr, decimation):
    assert decimation == int(decimation)
    assert decimation >= 1
    assert interim_osr == int(interim_osr)
    assert interim_osr >= 2
    grid_dir = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[direction]
    scp_index = {'Row': mdata.ImageData.SCPPixel.Row - mdata.ImageData.FirstRow,
                 'Col': mdata.ImageData.SCPPixel.Col - mdata.ImageData.FirstCol}[direction]
    num_samps = {'Row': mdata.ImageData.NumRows, 'Col': mdata.ImageData.NumCols}[direction]
    current_osr = 1 / (grid_dir.SS * grid_dir.ImpRespBW)
    minimum_pad = int(min(200 * current_osr, 0.1 * num_samps))
    fft1_size = scipy.fft.next_fast_len(num_samps + minimum_pad)
    effective_pad = fft1_size - num_samps
    insert_offset = effective_pad // 2
    fft2_size = scipy.fft.next_fast_len(int(np.ceil(fft1_size * interim_osr / current_osr)))
    rsr = fft2_size / fft1_size
    post_decim_rsr = rsr / decimation

    padded_scp_index = scp_index + insert_offset
    resampled_scp_index = padded_scp_index * rsr
    floor_scp_index = int(np.floor(resampled_scp_index))
    frac_shift = resampled_scp_index - floor_scp_index
    extract_offset = int(np.floor(insert_offset * rsr))
    extract_offset -= (extract_offset - floor_scp_index) % decimation
    decim_scp_index = (floor_scp_index - extract_offset) / decimation
    num_samps_out = int(np.ceil((insert_offset + num_samps - 1)
                                * post_decim_rsr - extract_offset / decimation)) + 1

    return {'fft1_size': fft1_size,
            'fft2_size': fft2_size,
            'insert_offset': insert_offset,
            'num_samps_in': num_samps,
            'frac_shift': frac_shift,
            'extract_offset': extract_offset,
            'interim_osr': interim_osr,
            'decimation': decimation,
            'num_samps_out': num_samps_out,
            'decim_scp_index': decim_scp_index,
            'post_decim_rsr': post_decim_rsr}


def update_sicd_metadata(sicd_metadata, direction, jiq_params_sicd):
    """Update SICD metadata to new reference location and sample spacing"""

    mdata = copy.deepcopy(sicd_metadata)
    if direction == 'Row':
        mdata.Grid.Row.SS = sicd_metadata.Grid.Row.SS / jiq_params_sicd['post_decim_rsr']
        mdata.ImageData.SCPPixel.Row = jiq_params_sicd['decim_scp_index']
        mdata.ImageData.FirstRow = 0
        mdata.ImageData.NumRows = jiq_params_sicd['num_samps_out']
        mdata.ImageData.FullImage.NumRows = jiq_params_sicd['num_samps_out']
        if mdata.ImageData.ValidData:
            for rowcol in mdata.ImageData.ValidData:
                rowcol.Row = int(round((rowcol.Row - sicd_metadata.ImageData.SCPPixel.Row)
                                       * jiq_params_sicd['post_decim_rsr'] + jiq_params_sicd['decim_scp_index']))
    else:
        mdata.Grid.Col.SS = sicd_metadata.Grid.Col.SS / jiq_params_sicd['post_decim_rsr']
        mdata.ImageData.SCPPixel.Col = jiq_params_sicd['decim_scp_index']
        mdata.ImageData.FirstCol = 0
        mdata.ImageData.NumCols = jiq_params_sicd['num_samps_out']
        mdata.ImageData.FullImage.NumCols = jiq_params_sicd['num_samps_out']
        if mdata.ImageData.ValidData:
            for rowcol in mdata.ImageData.ValidData:
                rowcol.Col = int(round((rowcol.Col - sicd_metadata.ImageData.SCPPixel.Col)
                                       * jiq_params_sicd['post_decim_rsr'] + jiq_params_sicd['decim_scp_index']))

    return mdata


def jiq_sicd(data, sicd_metadata, desired_osr=3, decimation=2,
             edge_glint_threshold=EDGE_GLINT_THRESHOLD, edge_glint_max_weight=EDGE_GLINT_MAX_WEIGHT):
    mdata = sicd_metadata

    with benchmark.howlong("Col deskew"):
        data, mdata = deskew.sicd_to_sicd(data, mdata, 'Col')

    # perform jiq sva
    col_jiq_params = _get_jiq_params_sicd(mdata, 'Col', desired_osr, decimation)
    with benchmark.howlong("Col sva"):
        data = jiq_1d_params(data, col_jiq_params['fft1_size'], col_jiq_params['fft2_size'], col_jiq_params['interim_osr'],
                             col_jiq_params['insert_offset'], col_jiq_params['num_samps_in'], col_jiq_params['frac_shift'],
                             col_jiq_params['extract_offset'], col_jiq_params['decimation'], col_jiq_params['num_samps_out'],
                             edge_glint_threshold, edge_glint_max_weight)
        mdata = update_sicd_metadata(mdata, 'Col', col_jiq_params)

    with benchmark.howlong("Row deskew"):
        data, mdata = deskew.sicd_to_sicd(data, mdata, 'Row')

    with benchmark.howlong("Transpose"):
        data = np.transpose(data)

    # perform jiq sva
    row_jiq_params = _get_jiq_params_sicd(mdata, 'Row', desired_osr, decimation)
    with benchmark.howlong("Row sva"):
        data = jiq_1d_params(data, row_jiq_params['fft1_size'], row_jiq_params['fft2_size'], row_jiq_params['interim_osr'],
                             row_jiq_params['insert_offset'], row_jiq_params['num_samps_in'], row_jiq_params['frac_shift'],
                             row_jiq_params['extract_offset'], row_jiq_params['decimation'], row_jiq_params['num_samps_out'],
                             edge_glint_threshold, edge_glint_max_weight)
        mdata = update_sicd_metadata(mdata, 'Row', row_jiq_params)

    with benchmark.howlong("Transpose"):
        data = np.transpose(data)

    return data, mdata


@numba.njit(parallel=True)
def jiq_1d_params(data, fft1_size, fft2_size, interim_osr,
                  insert_offset, num_samps_in, frac_shift,
                  extract_offset, decimation, num_samps_out,
                  edge_glint_threshold, edge_glint_max_weight):

    # create results array
    result = np.zeros((data.shape[0], num_samps_out), dtype=data.dtype)
    phase_vec = np.zeros((fft2_size,), dtype=data.dtype)
    with numba.objmode():
        phase_vec[:] = np.exp(2*np.pi*1j*np.fft.fftfreq(fft2_size) * frac_shift).astype(np.complex64)

    for m in numba.prange(data.shape[0]):
        row = data[m, :]
        fft1_buff = np.zeros((fft1_size,), dtype=row.dtype)
        fft1_buff[insert_offset: insert_offset + num_samps_in] = row
        fft1 = np.zeros_like(fft1_buff)
        with numba.objmode():
            fft1[:] = scipy.fft.fft(fft1_buff, norm="forward", workers=1)
        fft2_buff = np.zeros((fft2_size,), dtype=row.dtype)
        min_size = min(fft1_size, fft2_size)
        neg_start = min_size//2
        pos_end = min_size - neg_start
        fft2_buff[-neg_start:] = fft1[-neg_start:]
        fft2_buff[:pos_end] = fft1[:pos_end]
        fft2_buff = fft2_buff * phase_vec
        fft2 = np.zeros_like(fft2_buff)
        with numba.objmode():
            fft2[:] = scipy.fft.ifft(fft2_buff, norm="forward", workers=1)

        sva_row = jiq_sva_1d(fft2, interim_osr, edge_glint_threshold, edge_glint_max_weight)
        result[m, :] = sva_row[extract_offset:extract_offset+(decimation*num_samps_out):decimation]

    return result
