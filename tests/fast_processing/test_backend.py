__classification__ = "UNCLASSIFIED"

import numpy as np
import scipy

import pytest

import sarpy.fast_processing.backend


try:
    import mkl_fft._scipy_fft_backend
    MKL_FFT_AVAILABLE = True
except ImportError:
    MKL_FFT_AVAILABLE = False


@pytest.mark.skipif(not MKL_FFT_AVAILABLE, reason="requires mkl_fft package")
def test_fft_backend():
    np.random.seed(0)
    data = np.random.random(1025)
    with sarpy.fast_processing.backend.set_fft_backend('auto'):
        auto_result = scipy.fft.fft(data)
    with sarpy.fast_processing.backend.set_fft_backend('scipy'):
        scipy_result = scipy.fft.fft(data)
    with sarpy.fast_processing.backend.set_fft_backend('mkl'):
        mkl_result = scipy.fft.fft(data)

    direct_mkl_result = mkl_fft._scipy_fft_backend.fft(data)
    assert np.all(mkl_result == direct_mkl_result)

    assert not np.all(scipy_result == mkl_result)
    np.testing.assert_allclose(scipy_result, mkl_result)
    assert np.all(auto_result == mkl_result)


@pytest.mark.skipif(MKL_FFT_AVAILABLE, reason="requires no mkl_fft package")
def test_fft_backend_nomkl():
    np.random.seed(0)
    data = np.random.random(1025)
    with sarpy.fast_processing.backend.set_fft_backend('auto'):
        auto_result = scipy.fft.fft(data)
    with sarpy.fast_processing.backend.set_fft_backend('scipy'):
        scipy_result = scipy.fft.fft(data)
    with pytest.raises(ValueError, match="mkl_fft package not available"):
        sarpy.fast_processing.backend.set_fft_backend('mkl')

    assert np.all(auto_result == scipy_result)
