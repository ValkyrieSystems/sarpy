__classification__ = "UNCLASSIFIED"

import numpy as np
import pytest
import scipy.interpolate

import sarpy.fast_processing.projection as sfpp


@pytest.mark.parametrize(
    "interp_func, scipy_order, rtol",
    [(sfpp._do_interp_bilinear, 1, 1e-5), (sfpp._do_interp_bicubic, 3, 0.1)],
)
def test__do_interp(interp_func, scipy_order, rtol):
    # scipy doesn't implement the cubic spline variant used by sarpy.fast_processing,
    # so we'll use a smoothed input where they should have closer perormance
    data = scipy.misc.ascent().astype(np.float32) * .1
    data = scipy.ndimage.uniform_filter(data, 20)

    rbs = scipy.interpolate.RectBivariateSpline(
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
        data,
        kx=scipy_order,
        ky=scipy_order,
        s=0,
    )

    # check no-op on pixels
    x = np.arange(data.shape[0] - 3) + 1
    y = np.arange(data.shape[1] - 3) + 1
    scipy_computed = rbs(x, y)
    sarpy_computed = np.empty((x.size, y.size))
    for xidx, xval in enumerate(x):
        for yidx, yval in enumerate(x):
            sarpy_computed[xidx, yidx] = interp_func(data, xval, yval)
    np.testing.assert_allclose(scipy_computed, sarpy_computed)

    # check between pixels
    x += np.arange(data.shape[0] - 3) + 1.5
    y = np.arange(data.shape[1] - 3) + 1.5
    scipy_computed = rbs(x, y)
    sarpy_computed = np.empty((x.size, y.size))
    for xidx, xval in enumerate(x):
        for yidx, yval in enumerate(x):
            sarpy_computed[xidx, yidx] = interp_func(data, xval, yval)
    np.testing.assert_allclose(scipy_computed, sarpy_computed, rtol=rtol)

    # Check boundry logic
    assert interp_func(data, 0, -1) == 0.0
    assert interp_func(data, 0, -0.00001) == 0.0
    assert interp_func(data, 0, data.shape[1] - 1) == 0.0
    assert interp_func(data, 0, data.shape[1] - 1.000001) == pytest.approx(
        rbs(0, data.shape[1] - 1.000001)
    )

    assert interp_func(data, -1, 0) == 0.0
    assert interp_func(data, -0.00001, 0) == 0.0
    assert interp_func(data, data.shape[1] - 1, 0) == 0.0
    assert interp_func(data, data.shape[1] - 1.000001, 0) == pytest.approx(
        rbs(data.shape[1] - 1.000001, 0)
    )
