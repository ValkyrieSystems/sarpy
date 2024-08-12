__classification__ = "UNCLASSIFIED"

import numpy as np
import numpy.polynomial.polynomial as npp
import pytest

from sarpy.fast_processing import sva


def create_target(tp):
    """Create test target to apply SVA on"""
    # obtain target parameters
    ROW_SIZE = tp["data_size"][0]
    COL_SIZE = tp["data_size"][1]
    OSR = tp["osr"]
    AMP = tp["amp"]
    ROW_SHIFT = tp["shifts"][0]
    COL_SHIFT = tp["shifts"][1]

    # calculate sampling dimensions based on given osr
    samp_row_start = int((1 - 1 / OSR) / 2 * ROW_SIZE)
    samp_row_end = int(ROW_SIZE - (1 - 1 / OSR) / 2 * ROW_SIZE)
    samp_row_dim = slice(samp_row_start, samp_row_end)

    samp_col_start = int((1 - 1 / OSR) / 2 * COL_SIZE)
    samp_col_end = int(COL_SIZE - (1 - 1 / OSR) / 2 * COL_SIZE)
    samp_col_dim = slice(samp_col_start, samp_col_end)

    samp_dim = (samp_row_dim, samp_col_dim)

    # set target ampltiude
    freq_data = np.zeros((ROW_SIZE, COL_SIZE))
    freq_data[samp_dim] = AMP

    # apply phase shift to target
    krow_phase = (np.arange(ROW_SIZE) - ROW_SIZE // 2) / ROW_SIZE * ROW_SHIFT
    kcol_phase = (np.arange(COL_SIZE) - COL_SIZE // 2) / COL_SIZE * COL_SHIFT
    kphase = np.exp(
        2 * np.pi * 1j * (krow_phase[:, np.newaxis] + kcol_phase[np.newaxis, :])
    )
    phase_freq_data = freq_data * kphase

    # create target
    ipr = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(phase_freq_data)))

    # apply polys
    if "kctr" in tp:
        ROW_KCTR = tp["kctr"][0]
        COL_KCTR = tp["kctr"][1]
        row_phase = np.arange(ROW_SIZE) * ROW_KCTR
        col_phase = np.arange(COL_SIZE) * COL_KCTR
        phase = np.exp(
            -2 * np.pi * 1j * (row_phase[:, np.newaxis] + col_phase[np.newaxis, :])
        )
        ipr = ipr * phase
    elif "tgt_phase_poly" in tp:
        phase = np.exp(
            -2
            * np.pi
            * 1j
            * npp.polygrid2d(
                np.arange(ROW_SIZE), np.arange(COL_SIZE), tp["tgt_phase_poly"]
            )
        )
        ipr = ipr * phase

    return ipr


def get_mainlobe_dim(tp, os):
    """Calculate mainlobe region"""
    ROW_SIZE = tp["data_size"][0]
    COL_SIZE = tp["data_size"][1]
    ROW_SHIFT = int(np.floor(tp["shifts"][0]))
    COL_SHIFT = int(np.floor(tp["shifts"][1]))

    # calculate target center based on row and column shifts
    center = (ROW_SIZE // 2 + ROW_SHIFT, COL_SIZE // 2 + COL_SHIFT)

    # calculate mainlobe dimensions based on given offsets
    row_dim = slice(center[0] - os[0], center[0] + os[1] + 1)
    col_dim = slice(center[1] - os[0], center[1] + os[1] + 1)

    return (row_dim, col_dim)


def get_peak(ipr, td):
    """Find max value in the mainlobe region"""
    peak = np.max(ipr[td])
    return peak


def convert_to_db(ipr):
    """Convert ipr to decibels"""
    # prevent log divide by 0 by setting 0 values to minimum absolute value != 0
    ipr[ipr == 0] = np.min(np.abs(ipr[ipr != 0]))
    db_ipr = 20 * np.log10(np.abs(ipr))

    return db_ipr


def clean_edges(ipr, nyq):
    """Set edges unaffected by SVA to minimum value in the image"""
    ce_nyq = int(np.ceil(nyq))
    min_val = np.min(ipr[ce_nyq:-ce_nyq, ce_nyq:-ce_nyq])
    ipr[:ce_nyq, :] = min_val
    ipr[-ce_nyq:, :] = min_val
    ipr[:, :ce_nyq] = min_val
    ipr[:, -ce_nyq:] = min_val


def get_outside_max(ipr, tp, md):
    """Get maximum value outside of the mainlobe region"""
    # create mask around the mainlobe
    mainlobe_mask = np.zeros((tp["data_size"][0], tp["data_size"][1]))

    for dim in md:
        mainlobe_mask[dim] = 1

    # calculate max value outside of mainlobe
    masked_ipr = np.ma.masked_array(ipr, mask=mainlobe_mask)
    outside_max = np.max(masked_ipr)
    return outside_max


def run_one_target_test(tp, tc):
    """Run test involving one created target"""
    # create target using given parameters
    test_ipr = create_target(tp)

    # create polys to pass into SVA
    if "kctr" in tp:
        row_kctr_poly_radians_per_sample = 2 * np.pi * np.asarray([[tp["kctr"][0]]])
        col_kctr_poly_radians_per_sample = 2 * np.pi * np.asarray([[tp["kctr"][1]]])
    elif "tgt_phase_poly" in tp:
        tgt_phase_radians = 2 * np.pi * np.asarray(tp["tgt_phase_poly"])
        row_kctr_poly_radians_per_sample = npp.polyder(tgt_phase_radians, axis=0)
        col_kctr_poly_radians_per_sample = npp.polyder(tgt_phase_radians, axis=1)
    else:
        row_kctr_poly_radians_per_sample = np.asarray([[0]])
        col_kctr_poly_radians_per_sample = np.asarray([[0]])

    if tc["sva_type"] == "uncoupled":
        # apply uncoupled SVA
        results_ipr = sva.uncoup_sva(
            test_ipr,
            row_kctr_poly_radians_per_sample,
            col_kctr_poly_radians_per_sample,
        )
    elif tc["sva_type"] == "double":
        # apply Double SVA
        results_ipr = sva.d_sva(
            test_ipr,
            tp["osr"],
            tp["osr"],
            row_kctr_poly_radians_per_sample,
            col_kctr_poly_radians_per_sample,
        )
    elif tc["sva_type"] == "joint_iq":
        # apply joint-IQ SVA
        results_ipr = sva.two_dim_joint_iq_sva(
            test_ipr,
            tp["osr"],
            tp["osr"],
            tc["upsample_osr"],
            tc["upsample_osr"],
            tc["fft_size"],
            tc["fft_size"],
            fft_sign=1,
        )
    else:
        raise ValueError("Invalid SVA type given")

    # convert ipr values to decibels for comparison
    db_test_ipr, db_results_ipr = convert_to_db(test_ipr), convert_to_db(results_ipr)

    # set edges unaffected by SVA to the minimum value of the image
    clean_edges(db_results_ipr, tp["osr"])

    # calculate pre-SVA and post-SVA peak values
    offsets = tc["offsets"]
    mainlobe_dim = get_mainlobe_dim(tp, offsets)
    old_peak, new_peak = (
        get_peak(db_test_ipr, mainlobe_dim),
        get_peak(db_results_ipr, mainlobe_dim),
    )

    # test if post-sva peak is similar to pre-sva peak using given tolerance
    assert new_peak == pytest.approx(old_peak, abs=tc["tol"])

    # test if peak value is significantly greater than max value outside of mainlobe region using given tolerance
    outside_max = get_outside_max(db_results_ipr, tp, mainlobe_dim)
    assert new_peak - outside_max > tc["peak_diff"]


def run_two_target_test(tp1, tp2, tc):
    """Run test involving two created targets"""
    # create two targets using given parameters and combine them into one image
    ipr1 = create_target(tp1)
    ipr2 = create_target(tp2)
    test_ipr = ipr1 + ipr2

    zero_poly = np.asarray([[0.0]])
    if tc["sva_type"] == "uncoupled":
        # apply uncoupled SVA
        results_ipr = sva.uncoup_sva(
            test_ipr,
            zero_poly,
            zero_poly,
        )
    elif tc["sva_type"] == "double":
        # apply Double SVA
        results_ipr = sva.d_sva(
            test_ipr,
            tp1["osr"],
            tp1["osr"],
            zero_poly,
            zero_poly,
        )
    elif tc["sva_type"] == "joint_iq":
        # apply joint-IQ SVA
        results_ipr = sva.two_dim_joint_iq_sva(
            test_ipr,
            tp1["osr"],
            tp1["osr"],
            tc["upsample_osr"],
            tc["upsample_osr"],
            tc["fft_size"],
            tc["fft_size"],
            fft_sign=1,
        )
    else:
        raise ValueError("Invalid SVA type given")

    # convert ipr values to decibels for comparison
    db_ipr1, db_ipr2 = convert_to_db(ipr1), convert_to_db(ipr2)
    db_test_ipr, db_results_ipr = convert_to_db(test_ipr), convert_to_db(results_ipr)

    # set edges unaffected by SVA to the minimum value of the image
    clean_edges(db_results_ipr, tp1["osr"])

    # calculate pre-SVA and post-SVA peak values
    offsets = tc["offsets"]
    mainlobe1_dim = get_mainlobe_dim(tp1, offsets)
    mainlobe2_dim = get_mainlobe_dim(tp2, offsets)
    old_peak1, new_peak1 = (
        get_peak(db_ipr1, mainlobe1_dim),
        get_peak(db_results_ipr, mainlobe1_dim),
    )
    old_peak2, new_peak2 = (
        get_peak(db_ipr2, mainlobe2_dim),
        get_peak(db_results_ipr, mainlobe2_dim),
    )

    # test if post-sva peaks are similar to pre-sva peaks using given tolerance
    assert new_peak1 == pytest.approx(old_peak1, abs=tc["tol1"])
    assert new_peak2 == pytest.approx(old_peak2, abs=tc["tol2"])

    # test if main peak value is significantly greater than max value outside of mainlobe region using given tolerance
    outside_max = get_outside_max(db_results_ipr, tp1, (mainlobe1_dim, mainlobe2_dim))
    assert new_peak1 - outside_max > tc["peak_diff"]


def calc_center(row):
    """Calculate center sample in the given row"""
    return row[row.shape[0] // 2]


@pytest.mark.parametrize(
    "osr,sva_type,tol,peak_diff",
    [
        (2.0, "uncoupled", 0.001, 80),
        (2.0, "double", 0.01, 60),
        (1.5, "double", 0.01, 60),
        (1.25, "double", 0.01, 40),
    ],
)
def test_centered_target(osr, sva_type, tol, peak_diff):
    target_params = dict(data_size=(1024, 1024), osr=osr, amp=1, shifts=(0, 0))
    test_conditions = dict(
        sva_type=sva_type, offsets=(1, 1), tol=tol, peak_diff=peak_diff
    )
    run_one_target_test(target_params, test_conditions)


@pytest.mark.parametrize(
    "osr, sva_type, tol, peak_diff",
    [
        (2.0, "uncoupled", 0.001, 80),
        (2.0, "double", 0.01, 60),
        (1.5, "double", 0.01, 60),
        (1.25, "double", 0.01, 60),
    ],
)
def test_shifted_target(osr, sva_type, tol, peak_diff):
    target_params = dict(
        data_size=(1024, 1024),
        osr=osr,
        amp=1,
        shifts=(np.random.rand(), np.random.rand()),
    )
    test_conditions = dict(
        sva_type=sva_type, offsets=(1, 2), tol=tol, peak_diff=peak_diff
    )
    run_one_target_test(target_params, test_conditions)


@pytest.mark.parametrize(
    "osr, sva_type, target2_shift, tol1, tol2, peak_diff",
    [
        (2.0, "uncoupled", 7, 0.1, 10, 60),
        (2.0, "double", 7, 0.1, 10, 50),
        (1.5, "double", 8, 0.1, 15, 50),
        (1.25, "double", 8, 0.1, 15, 50),
    ],
)
def test_buried_target(osr, sva_type, target2_shift, tol1, tol2, peak_diff):
    target1_params = dict(
        data_size=(1024, 1024),
        osr=osr,
        amp=1,
        shifts=(np.random.rand(), np.random.rand()),
    )
    target1_shifts = target1_params["shifts"]
    target2_params = dict(
        data_size=(1024, 1024),
        osr=osr,
        amp=0.04,
        shifts=(target1_shifts[0] + target2_shift, target1_shifts[1]),
    )
    test_conditions = dict(
        sva_type=sva_type, offsets=(2, 3), tol1=tol1, tol2=tol2, peak_diff=peak_diff
    )
    run_two_target_test(target1_params, target2_params, test_conditions)


@pytest.mark.parametrize(
    "osr,sva_type,tol,peak_diff,kctr",
    [
        (2.0, "uncoupled", 0.1, 80, (0.3, 0.4)),
        (2.0, "double", 0.1, 60, (-0.3, 0.4)),
        (1.5, "double", 0.1, 60, (0.3, -0.4)),
        (1.25, "double", 0.1, 60, (-0.3, -0.4)),
    ],
)
def test_constant_shifted_spectra(osr, sva_type, tol, peak_diff, kctr):
    target_params = dict(
        data_size=(1024, 1024),
        osr=osr,
        amp=1,
        shifts=(np.random.rand(), np.random.rand()),
        kctr=kctr,
    )
    test_conditions = dict(
        sva_type=sva_type, offsets=(1, 2), tol=tol, peak_diff=peak_diff
    )
    run_one_target_test(target_params, test_conditions)


@pytest.mark.parametrize(
    "osr,sva_type,tol,peak_diff,tgt_phase_poly",
    [
        (
            2.0,
            "uncoupled",
            0.1,
            80,
            [
                [2.79859927e-02, 4.61178993e-07, -1.50323162e-07],
                [3.19950324e-04, -9.55916637e-08, 2.11746801e-10],
                [2.02447668e-07, -2.11108534e-10, -9.47282378e-14],
            ],
        ),
        (
            2.0,
            "double",
            0.1,
            60,
            [
                [-2.38969940e-01, -1.11531905e-04, -1.86553997e-07],
                [-1.16553933e-04, -3.46075058e-07, 1.78556430e-10],
                [2.96869022e-07, -2.07658357e-10, -1.13006436e-13],
            ],
        ),
        (
            1.5,
            "double",
            0.1,
            60,
            [
                [1.72502592e-01, 4.77756507e-04, -3.86998030e-07],
                [7.57418005e-05, 3.01818002e-07, -3.43175809e-10],
                [1.94841004e-07, -2.94380311e-10, -3.13729003e-13],
            ],
        ),
        (
            1.25,
            "double",
            0.1,
            60,
            [
                [4.37297371e-01, 1.48484260e-04, 2.06584474e-07],
                [2.42018475e-04, -3.11471305e-07, 2.48010167e-10],
                [-2.83514777e-07, -3.46671139e-10, 3.63480793e-13],
            ],
        ),
    ],
)
def test_varying_shifted_spectra(osr, sva_type, tol, peak_diff, tgt_phase_poly):
    target_params = dict(
        data_size=(1024, 1024),
        osr=osr,
        amp=1,
        shifts=(np.random.rand(), np.random.rand()),
        tgt_phase_poly=tgt_phase_poly,
    )
    test_conditions = dict(
        sva_type=sva_type, offsets=(1, 2), tol=tol, peak_diff=peak_diff
    )
    run_one_target_test(target_params, test_conditions)


@pytest.mark.parametrize(
    "data_size, osr, upsample_osr",
    [
        ((1024, 1025), 2, 3),
        ((1024, 1025), 2, 4),
        ((1024, 1025), 1.5, 2),
        ((1024, 1025), 1.5, 3),
        ((1024, 1025), 1.25, 4),
        ((1024, 1025), 1.25, 5),
        (
            (int(1000 * np.random.rand()) + 5, int(1000 * np.random.rand()) + 5),
            2 * np.random.rand(),
            3,
        ),
        (
            (int(1000 * np.random.rand()) + 5, int(1000 * np.random.rand()) + 5),
            2 * np.random.rand(),
            4,
        ),
    ],
)
def test_joint_iq_upsample(data_size, osr, upsample_osr):
    target_params = dict(
        data_size=data_size,
        osr=osr,
        amp=1,
        shifts=(np.random.rand(), np.random.rand()),
    )
    target = create_target(target_params)
    for m in range(data_size[0]):
        row = target[m, :]
        row_copy = np.copy(row)
        upsample_row = sva.upsample(row_copy, osr, upsample_osr, 1)
        assert calc_center(row) == pytest.approx(calc_center(upsample_row))
    target_transpose = np.transpose(target)
    for n in range(data_size[1]):
        row = target_transpose[n, :]
        row_copy = np.copy(row)
        upsample_row = sva.upsample(row_copy, osr, upsample_osr, 1)
        assert calc_center(row) == pytest.approx(calc_center(upsample_row))


@pytest.mark.parametrize(
    "osr,upsample_osr,sva_type,tol,peak_diff",
    [
        (2.0, 2, "joint_iq", 0.1, 50),
        (1.5, 3, "joint_iq", 0.1, 50),
        (1.25, 5, "joint_iq", 0.1, 50),
    ],
)
def test_joint_iq_centered_target(osr, upsample_osr, sva_type, tol, peak_diff):
    target_params = dict(
        data_size=(1024, 1024),
        osr=osr,
        amp=1,
        shifts=(0, 0),
    )
    test_conditions = dict(
        sva_type=sva_type,
        upsample_osr=upsample_osr,
        fft_size=target_params["data_size"][0],
        offsets=(1, 2),
        tol=tol,
        peak_diff=peak_diff,
    )
    run_one_target_test(target_params, test_conditions)


@pytest.mark.parametrize(
    "osr,upsample_osr,sva_type,tol,peak_diff",
    [
        (2.0, 2, "joint_iq", 0.1, 50),
        (1.5, 3, "joint_iq", 0.1, 50),
        (1.25, 5, "joint_iq", 0.1, 50),
    ],
)
def test_joint_iq_shifted_target(osr, upsample_osr, sva_type, tol, peak_diff):
    target_params = dict(
        data_size=(1024, 1024),
        osr=osr,
        amp=1,
        shifts=(np.random.rand(), np.random.rand()),
    )
    test_conditions = dict(
        sva_type=sva_type,
        upsample_osr=upsample_osr,
        fft_size=target_params["data_size"][0],
        offsets=(1, 2),
        tol=tol,
        peak_diff=peak_diff,
    )
    run_one_target_test(target_params, test_conditions)


@pytest.mark.parametrize(
    "osr,upsample_osr,sva_type,target2_shift,tol1,tol2,peak_diff",
    [
        (2.0, 2, "joint_iq", 7, 0.1, 15, 50),
        (1.5, 3, "joint_iq", 7, 0.1, 15, 50),
        (1.25, 5, "joint_iq", 7, 0.1, 15, 50),
    ],
)
def test_joint_iq_buried_target(
    osr, upsample_osr, sva_type, target2_shift, tol1, tol2, peak_diff
):
    target1_params = dict(
        data_size=(1024, 1024),
        osr=osr,
        amp=1,
        shifts=(np.random.rand(), np.random.rand()),
    )
    target1_shifts = target1_params["shifts"]
    target2_params = dict(
        data_size=(1024, 1024),
        osr=osr,
        amp=0.04,
        shifts=(target1_shifts[0] + target2_shift, target1_shifts[1]),
    )
    test_conditions = dict(
        sva_type=sva_type,
        upsample_osr=upsample_osr,
        fft_size=target1_params["data_size"][0],
        offsets=(2, 3),
        tol1=tol1,
        tol2=tol2,
        peak_diff=peak_diff,
    )
    run_two_target_test(target1_params, target2_params, test_conditions)
