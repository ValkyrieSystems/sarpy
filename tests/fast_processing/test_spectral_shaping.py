__classification__ = "UNCLASSIFIED"

import numpy as np
import scipy.signal

import sarpy.fast_processing.spectral_shaping as fpss


def test_apply_filter():
    rng = np.random.default_rng(12345)

    input_data = rng.random((47, 51), dtype=np.float32)
    scipy_result = scipy.signal.convolve(input_data,
                                         fpss.DEFAULT_FILTER,
                                         mode='valid')
    fpss_result = fpss.apply_filter(input_data, -1.0, 0.0, 1000.0)
    mask = scipy_result > 0
    assert np.allclose(fpss_result[2:-2,2:-2][mask], scipy_result[mask], atol=1e-6)
