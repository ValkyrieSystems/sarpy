"""Configure backend libraries"""

__classification__ = "UNCLASSIFIED"

import logging

import scipy.fft

try:
    import mkl_fft._scipy_fft_backend
    MKL_FFT_AVAILABLE = True
except ImportError:
    MKL_FFT_AVAILABLE = False


def find_scipy_backend(backend):
    """Return a backend suitable for being passed to scipy.fft.*backend

    Args
    ----
    backend: str or Object
        Name of backend to use.  "mkl" will use Intel MKL, "auto" will use MKL if available,
        other values are returned unchanged

    Returns: str or Object
        object suitable for being passed to scipy.fft.*backend functions
    """

    if backend == "auto":
        if MKL_FFT_AVAILABLE:
            backend = "mkl"
        else:
            backend = "scipy"

    if backend == "mkl":
        if not MKL_FFT_AVAILABLE:
            raise ValueError("mkl_fft package not available")
        backend = mkl_fft._scipy_fft_backend

    return backend


def set_fft_backend(backend):
    """Context manager to set the FFT backend"""
    scipy_backend = find_scipy_backend(backend)
    logging.info(f"Using FFT backend {scipy_backend}")
    return scipy.fft.set_backend(scipy_backend)
