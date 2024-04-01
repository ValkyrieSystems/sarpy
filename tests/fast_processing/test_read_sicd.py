import pathlib

import numpy as np
import pytest

import sarpy.fast_processing.read_sicd

import tests

complex_file_types = tests.find_test_data_files(
    pathlib.Path(__file__).parents[1] / "io/complex/complex_file_types.json"
)


@pytest.fixture(scope="module")
def sicd_file():
    for file in complex_file_types.get("SICD", []):
        if pathlib.Path(file).name == "sicd_example_1_PFA_RE32F_IM32F_HH.nitf":
            return file
    pytest.skip("sicd test file not found")


def test_read_from_file(sicd_file):
    sicd_pixels, sicd_meta = sarpy.fast_processing.read_sicd.read_from_file(sicd_file)
    assert sicd_pixels.dtype == np.dtype('complex64')
    assert sicd_pixels.shape == (sicd_meta.ImageData.NumRows, sicd_meta.ImageData.NumCols)
