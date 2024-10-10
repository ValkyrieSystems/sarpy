import pathlib

import pytest

import sarpy.io.complex

import sarpy.fast_processing.metadata

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


def test_add_sicd_processing(sicd_file):
    with sarpy.io.complex.open(str(sicd_file)) as reader:
        sicd_meta = reader.sicd_meta

    sicd_meta.ImageFormation.Processings = None

    procs = (
        {"proc_type": "first"},
        {"proc_type": "second", "applied": False},
        {"proc_type": "third", "parameters": {"k0": "v0", "k2": "v2"}},
    )
    for index, proc in enumerate(procs):
        sarpy.fast_processing.metadata.add_sicd_processing(
            sicd_meta, **proc,
        )
        assert len(sicd_meta.ImageFormation.Processings) == index + 1
        assert proc["proc_type"] in sicd_meta.ImageFormation.Processings[index].Type
        assert sicd_meta.ImageFormation.Processings[index].Applied == proc.get("applied", True)
        if "parameters" in proc:
            assert sicd_meta.ImageFormation.Processings[index].Parameters.to_dict() == proc["parameters"]
        else:
            assert not sicd_meta.ImageFormation.Processings[index].Parameters
