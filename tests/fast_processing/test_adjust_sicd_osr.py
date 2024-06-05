__classification__ = "UNCLASSIFIED"

import pathlib

import pytest

import sarpy.fast_processing.adjust_sicd_osr as aso

import tests


complex_file_types = tests.find_test_data_files(
    pathlib.Path(__file__).parents[1] / "io/complex/complex_file_types.json"
)


@pytest.fixture(scope="module")
def sicd_file():
    for file in complex_file_types.get("SICD", []):
        if pathlib.Path(file).name == "sicd_example_RMA_RGZERO_RE16I_IM16I.nitf":
            return file
    pytest.skip("sicd test file not found")


@pytest.mark.parametrize('osr_ratio', [1.1, 1.5, 2.0])
def test_smoke(sicd_file, tmp_path, osr_ratio):
    out_sicd_file = tmp_path / 'smoke.sicd'
    aso.main([str(sicd_file), str(out_sicd_file), '--desired-osr', str(osr_ratio)])
    assert out_sicd_file.exists()
