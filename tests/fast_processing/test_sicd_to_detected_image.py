import pathlib

import pytest

import sarpy.fast_processing.sicd_to_detected_image as stdi

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


def test_smoke(sicd_file, tmp_path):
    sidd_filename = tmp_path / 'smoke.sidd'
    stdi.main([str(sicd_file), str(sidd_filename)])
    assert sidd_filename.exists()
