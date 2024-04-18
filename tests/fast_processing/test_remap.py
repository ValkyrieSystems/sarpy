__classification__ = "UNCLASSIFIED"

import numba
import numpy as np

import pytest

import sarpy.fast_processing.remap


@pytest.mark.parametrize('size', ((947, 997), (937, 998), (np.random.random(2) * 1024).astype(int)))
def test_median_sizes(size):
    data = np.random.random(size)
    med_val = sarpy.fast_processing.remap._median(data)

    assert med_val == np.percentile(data, 50, interpolation='higher')

def test_median_const():
    data = np.ones((937, 998))
    assert sarpy.fast_processing.remap._median(data) == 1.0
