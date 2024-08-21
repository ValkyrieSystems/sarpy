import pathlib

import numpy as np
import pytest

import sarpy.fast_processing.read_sicd
import sarpy.fast_processing.write_sicd

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


def test_write_to_file(sicd_file, tmp_path):
    sicd_pixels, sicd_meta = sarpy.fast_processing.read_sicd.read_from_file(sicd_file)

    out_file = tmp_path / 'out.sicd'
    sarpy.fast_processing.write_sicd.write_to_file(out_file, sicd_pixels, sicd_meta)
    sicd_pixels2, sicd_meta2 = sarpy.fast_processing.read_sicd.read_from_file(out_file)

    np.testing.assert_array_equal(sicd_pixels, sicd_pixels2)
    assert sicd_meta.CollectionInfo.CoreName == sicd_meta2.CollectionInfo.CoreName
