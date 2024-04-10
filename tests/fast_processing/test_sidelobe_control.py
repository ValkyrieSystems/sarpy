__classification__ = "UNCLASSIFIED"

import numpy as np
import numpy.polynomial.polynomial as npp

import sarpy.fast_processing.sidelobe_control


def test_apply_phase_poly():
    rng = np.random.default_rng(12345)
    input_data = (
        rng.random((47, 51, 2), dtype=np.float32).view(dtype=np.complex64).squeeze()
    )
    phase_poly = np.linspace(-10, 10, 12).reshape((3, 4))
    r0 = -2.4
    rss = 0.11
    c0 = 8.1
    css = 1.3
    xx, yy = np.meshgrid(
        r0 + rss * np.arange(input_data.shape[0]),
        c0 + css * np.arange(input_data.shape[1]),
        indexing="ij",
    )
    expected_output = (
        np.exp(1j * 2 * np.pi * npp.polyval2d(xx, yy, phase_poly)) * input_data
    )
    actual_output = sarpy.fast_processing.sidelobe_control._apply_phase_poly(
        input_data,
        phase_poly,
        row_0=r0,
        row_ss=rss,
        col_0=c0,
        col_ss=css,
    )
    assert np.allclose(actual_output, expected_output)
