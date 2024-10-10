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
    fpss_result = fpss.apply_filter(input_data)
    assert np.allclose(fpss_result[2:-2,2:-2], scipy_result, atol=1e-6)
