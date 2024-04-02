import copy
import pathlib
import tempfile

import pytest

import sarpy.fast_processing.sicd_to_detected_image as stdi
import sarpy.consistency.sidd_consistency
import sarpy.io.complex

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


@pytest.fixture(scope="module")
def sicd_file_without_rc_plane(sicd_file):
    with sarpy.io.complex.open(str(sicd_file)) as reader:
        sicd_pixels = reader[...]
        sicd_meta = reader.sicd_meta

    new_meta = copy.deepcopy(sicd_meta)
    assert new_meta.RadarCollection.Area.Plane is not None
    new_meta.RadarCollection.Area.Plane = None

    with tempfile.TemporaryDirectory() as tmpdir:
        out_sicd = pathlib.Path(tmpdir) / 'no_rc_plane.sicd'
        with sarpy.io.complex.sicd.SICDWriter(str(out_sicd), new_meta) as writer:
            writer.write_chip(sicd_pixels)
        yield out_sicd


@pytest.mark.parametrize('sidd_version', [1, 2, 3])
def test_smoke(sicd_file, tmp_path, sidd_version):
    sidd_filename = tmp_path / 'smoke.sidd'
    stdi.main([str(sicd_file), str(sidd_filename), '--sidd-version', str(sidd_version)])
    assert sidd_filename.exists()

    # TODO need a convenient way of extracting the XML
    expected_urn = f'urn:SIDD:{sidd_version}.0.0'
    assert expected_urn.encode() in sidd_filename.read_bytes()

    assert sarpy.consistency.sidd_consistency.check_file(str(sidd_filename))


def test_smoke_no_rc_plane(sicd_file_without_rc_plane, tmp_path):
    sidd_filename = tmp_path / 'smoke.sidd'
    stdi.main([str(sicd_file_without_rc_plane), str(sidd_filename)])
    assert sidd_filename.is_file()

    # TODO need a convenient way of extracting the XML
    expected_urn = f'urn:SIDD:3.0.0'
    assert expected_urn.encode() in sidd_filename.read_bytes()

    assert sarpy.consistency.sidd_consistency.check_file(str(sidd_filename))
