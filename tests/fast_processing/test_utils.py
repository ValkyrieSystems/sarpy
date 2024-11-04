__classification__ = "UNCLASSIFIED"

import numpy as np
import numpy.testing as npt

from sarpy.fast_processing import utils


def _get_complex_data(shape=(47, 51)):
    rng = np.random.default_rng(12345)
    complex_data = (rng.random(shape, dtype=np.float32)
                    + rng.random(shape, dtype=np.float32) * 1j).astype(np.complex64)
    return complex_data


def test_parallel_copyto():
    in_shape = (1000, 1000)
    input_data = _get_complex_data(in_shape)
    out_shape = (1000, 1500)
    copy_slices = [slice(None), slice(-in_shape[1]//2, None)]

    numpy_copy = np.zeros(out_shape, dtype=np.complex64)
    numpy_copy[tuple(copy_slices)] = input_data[tuple(copy_slices)]

    utils_copy = np.zeros(out_shape, dtype=np.complex64)
    utils.parallel_copyto(utils_copy[tuple(copy_slices)], input_data[tuple(copy_slices)])

    npt.assert_array_equal(numpy_copy, utils_copy)
