__classification__ = "UNCLASSIFIED"

import pathlib

import pytest

import sarpy.fast_processing.sidelobe_control

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


@pytest.mark.parametrize('sidelobe_control', ['Uniform', 'Taylor'])
def test_smoke(sicd_file, tmp_path, sidelobe_control):
    out_sicd_file = tmp_path / 'smoke.sicd'
    sarpy.fast_processing.sidelobe_control.main([str(sicd_file),
                                                 str(out_sicd_file),
                                                 '--sidelobe-control', sidelobe_control])
    assert out_sicd_file.exists()
