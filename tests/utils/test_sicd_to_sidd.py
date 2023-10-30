import copy
import logging
import multiprocessing as mp
import os
import pathlib
import tempfile

import numpy as np
import pytest

from tests import find_test_data_files
from sarpy.io.complex.converter import open_complex
from sarpy.io.general.base import SarpyIOError
from sarpy.io.product.sidd import SIDDDetails, SIDDReader
from sarpy.processing.ortho_rectify import NearestNeighborMethod
from sarpy.processing.sidd import sidd_product_creation
import sarpy.utils.sicd_to_sidd
import sarpy.visualization.remap as remap

test_data_info = find_test_data_files(pathlib.Path(__file__).parent / 'utils_file_types.json')
sml_sicd_files = test_data_info.get('SICD_small', [])
big_sicd_files = test_data_info.get('SICD_big', [])

logging.basicConfig(level=logging.WARNING)


class MockPool():
    """When this class replaces the multiprocessing.Pool class, worker functions are run in the main python thread"""
    def __init__(self, processes=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def close(self):
        pass

    def join(self):
        pass

    def imap_unordered(self, mp_worker, multiproc_iter):
        return [mp_worker(pars) for pars in multiproc_iter]


def sicd_to_sidd_test_helper(capsys, sicd_file, prod_type="detected", method="nearest", remap="gdm",
                             version="3", num_cpus=1, window=None, pars=None, out_file=None, blocksize=None):

    with tempfile.TemporaryDirectory() as tempdir:
        sidd_dir_path = pathlib.Path(tempdir)

        args = [str(sicd_file), str(sidd_dir_path)]
        args += ["--type", prod_type] if prod_type else []
        args += ["--method", method] if method else []
        args += ["--remap", remap] if remap else []
        args += ["--block_size_mb", str(blocksize)] if blocksize else []
        args += ["--number_of_processes", str(num_cpus)] if num_cpus else []
        args += ["--version", str(version)] if version else []
        args += ['--window', window] if window else []
        args += (['--pars'] + pars) if pars else []
        args += ['--output_filename', out_file] if out_file else []

        sarpy.utils.sicd_to_sidd.main(args)

        captured = capsys.readouterr()
        assert captured.err == ''

        output_sidd_files = list(sidd_dir_path.iterdir())
        assert len(output_sidd_files) >= 1

        prod_class = {'detected': 'Detected Image',
                      'csi': 'Color Subaperture Image',
                      'dynamic': 'Dynamic Image'}.get(prod_type, 'Unknown')
        mean_values  = []
        for file_path in output_sidd_files:
            sidd_obj = SIDDDetails(str(file_path))
            assert sidd_obj.is_sidd
            assert all(meta.ProductCreation.ProductClass  == prod_class for meta in sidd_obj.sidd_meta)

            with SIDDReader(str(file_path)) as sidd:
                for seg in sidd.get_data_segments():
                    mean_values.append(np.mean(seg[:, :]))

        global_mean = np.mean(mean_values)
        assert global_mean > 0

        return global_mean


def test_sicd_to_sidd_help(capsys):
    with pytest.raises(SystemExit):
        sarpy.utils.sicd_to_sidd.main(['--help'])

    captured = capsys.readouterr()

    assert captured.err == ''
    assert captured.out.startswith('usage:')


@pytest.mark.parametrize("sicd_file", sml_sicd_files[:1])
def test_remap_scaling(capsys, sicd_file):
    # Check to see if the remaped values have approximately the same mean value independent of product type.
    sidd_means = []
    for prod_type in ['detected', 'csi', 'dynamic']:
        sidd_means.append(sicd_to_sidd_test_helper(capsys, sicd_file, prod_type=prod_type,
                                                   method="nearest", remap="density", out_file="out.nitf"))

    assert (max(sidd_means) - min(sidd_means)) / np.mean(sidd_means) < 0.05


@pytest.mark.parametrize("sicd_file", sml_sicd_files[:1])
@pytest.mark.parametrize("prod_type", ['detected', 'csi', 'dynamic'])
def test_sicd_to_sidd_no_taper_filename(capsys, sicd_file, prod_type):
    sicd_to_sidd_test_helper(capsys, sicd_file, prod_type=prod_type, method="nearest", remap="gdm", out_file="out.nitf")


@pytest.mark.parametrize("sicd_file", sml_sicd_files[:1])
@pytest.mark.parametrize("prod_type", ['detected', 'csi', 'dynamic'])
def test_sicd_to_sidd_no_taper(capsys, sicd_file, prod_type):
    sicd_to_sidd_test_helper(capsys, sicd_file, prod_type=prod_type, method="nearest", remap="gdm")


@pytest.mark.parametrize("sicd_file", sml_sicd_files[:1])
def test_sicd_to_sidd_with_taper(capsys, sicd_file):
    sicd_to_sidd_test_helper(capsys, sicd_file, prod_type="detected", method="nearest", remap="gdm",
                             window="taylor", pars=["5", "-35"])


cases = []
cases += [(big_sicd_files[0], 'detected')] if big_sicd_files else []
cases += [(sml_sicd_files[0], 'csi')] if sml_sicd_files else []
cases += [(sml_sicd_files[0], 'dynamic')] if sml_sicd_files else []
@pytest.mark.parametrize("sicd_file,prod_type", cases)
def test_sicd_to_sidd_no_taper_mp(sicd_file, prod_type, capsys):
    sicd_to_sidd_test_helper(capsys, sicd_file, prod_type=prod_type, method="nearest", remap="gdm", num_cpus=2)


@pytest.mark.parametrize("sicd_file", big_sicd_files[:1])
def test_sicd_to_sidd_no_taper_mp_spline(sicd_file, capsys):
    sicd_to_sidd_test_helper(capsys, sicd_file, prod_type="detected", method="spline_3", remap="gdm", num_cpus=4)


@pytest.mark.parametrize("sicd_file", sml_sicd_files[:1])
@pytest.mark.parametrize("method", ['nearest', 'spline_3'])
@pytest.mark.parametrize("prod_type", ['detected', 'csi', 'dynamic'])
def test_sicd_to_sidd_no_taper_mp_coverage(sicd_file, method, prod_type, capsys, monkeypatch):
    # This test is the same as test_sicd_to_sidd_no_taper_mp except that monkeypatching the Pool
    # class alows the _mp_worker function to be run in the same thread as the rest of the python
    # so that coverage stats can be obtained.
    monkeypatch.setattr(mp, 'Pool', MockPool)

    sicd_to_sidd_test_helper(capsys, sicd_file, prod_type=prod_type, method=method, remap="gdm", num_cpus=2)


@pytest.mark.parametrize("sicd_file", sml_sicd_files[:1])
@pytest.mark.parametrize("remap", ['density', 'high_contrast', 'brighter', 'darker',
                                   'pedf', 'gdm', 'linear', 'log', 'nrl'])
def test_sicd_to_sidd_remap_coverage(monkeypatch, capsys, sicd_file, remap):
    monkeypatch.setattr(mp, 'Pool', MockPool)

    sicd_to_sidd_test_helper(capsys, sicd_file, prod_type="detected", method="nearest", remap=remap, num_cpus=2)


@pytest.mark.parametrize("sicd_file", sml_sicd_files[:1])
def test_sicd_to_sidd_create_dynamic_image_sidd_exception(sicd_file):
    with tempfile.TemporaryDirectory() as tempdir:
        existing_file = pathlib.Path(tempdir) / 'dummy.nitf'
        existing_file.touch()

        reader = open_complex(sicd_file)
        ortho_helper = NearestNeighborMethod(reader)

        with pytest.raises(SarpyIOError, match=f"File {tempdir}/dummy.nitf already exists"):
            sidd_product_creation.create_dynamic_image_sidd(ortho_helper, tempdir, output_file='dummy.nitf')


@pytest.mark.parametrize("sicd_file", sml_sicd_files[:1])
def test_sicd_to_sicd_mp_worker_exceptions(sicd_file):
    pars = {'sicd_filename': sicd_file,
            '_obj_type_': 'OrthorectificationIterator',
            'ortho_helper': {'_obj_type_': 'NearestNeighborMethod',
                              'index': 0},
            'remap_function': {'_obj_type_': 'Density',
                               'override_name': 'density',
                               'bit_depth': 8,
                               'max_output_value': 255,
                               'data_mean': 2000.0},
            'calculator': {'_obj_type_': 'FullResolutionFetcher',
                           'dimension': 1,
                           'index': 0,
                           'block_size': 10.0}
            }

    pars2 = copy.deepcopy(pars)
    pars2['ortho_helper']['_obj_type_'] = 'Dummy'
    with pytest.raises(ValueError, match="Unknown ortho_helper object type: Dummy"):
        sidd_product_creation._mp_worker(pars2)

    pars2 = copy.deepcopy(pars)
    pars2['remap_function']['_obj_type_'] = 'Dummy'
    with pytest.raises(ValueError, match="Unknown remap_function object type: Dummy"):
        sidd_product_creation._mp_worker(pars2)

    pars2 = copy.deepcopy(pars)
    pars2['calculator']['_obj_type_'] = 'Dummy'
    with pytest.raises(ValueError, match="Unknown calculator object type: Dummy"):
        sidd_product_creation._mp_worker(pars2)

    pars2 = copy.deepcopy(pars)
    pars2['_obj_type_'] = 'Dummy'
    with pytest.raises(ValueError, match="Unknown OrthoIterator object type: Dummy"):
        sidd_product_creation._mp_worker(pars2)

    with pytest.raises(SarpyIOError, match="output_directory `/dummy`"):
        sidd_product_creation.create_detected_image_sidd('dummy', '/dummy')
    with pytest.raises(SarpyIOError, match="output_directory `/dummy`"):
        sidd_product_creation.create_csi_sidd('dummy', '/dummy')
    with pytest.raises(SarpyIOError, match="output_directory `/dummy`"):
        sidd_product_creation.create_dynamic_image_sidd('dummy', '/dummy')

    with pytest.raises(TypeError, match="ortho_helper is required to be an instance of OrthorectificationHelper"):
        sidd_product_creation.create_detected_image_sidd('dummy', '/tmp')
    with pytest.raises(TypeError, match="ortho_helper is required to be an instance of OrthorectificationHelper"):
        sidd_product_creation.create_csi_sidd('dummy', '/tmp')
    with pytest.raises(TypeError, match="ortho_helper is required to be an instance of OrthorectificationHelper"):
        sidd_product_creation.create_dynamic_image_sidd('dummy', '/tmp')


def test_sidd_prod_create_validate_filename():
    full_filename = sidd_product_creation._validate_filename('/tmp', '/dev/dummy.nitf', {})

    assert full_filename == '/tmp/dummy.nitf'

    with tempfile.TemporaryDirectory() as tempdir:
        existing_file = pathlib.Path(tempdir) / 'dummy.nitf'
        existing_file.touch()

        with pytest.raises(SarpyIOError, match=f"File {tempdir}/dummy.nitf already exists"):
            full_filename = sidd_product_creation._validate_filename(tempdir, '/dev/dummy.nitf', {})


def test_sidd_prod_create_validate_remap():
    with pytest.raises(TypeError, match="remap_function must be an instance of MonochromaticRemap"):
        sidd_product_creation._validate_remap_function(None)

    remap_function = remap.MonochromaticRemap(bit_depth=32)
    with pytest.raises(TypeError, match="remap_function usage for SIDD requires 8 or 16 bit output"):
        sidd_product_creation._validate_remap_function(remap_function)


def test_sidd_prod_create_affinity():
    actual_affinity = os.sched_getaffinity(0)
    num_cpus = mp.cpu_count()

    affinity_obj = sidd_product_creation.CPUAffinity()
    assert affinity_obj._saved_affinity == actual_affinity
    assert affinity_obj.get() == actual_affinity

    test_affinity = {num_cpus - 1}
    affinity_obj.set(test_affinity)
    assert affinity_obj._saved_affinity == actual_affinity
    assert affinity_obj.get() == test_affinity

    affinity_obj.restore()
    assert affinity_obj._saved_affinity == actual_affinity
    assert affinity_obj.get() == actual_affinity


def test_sidd_prod_create_affinity_no_getaffinity(monkeypatch):
    monkeypatch.delattr('os.sched_getaffinity')
    monkeypatch.delattr('os.sched_setaffinity')

    affinity_obj = sidd_product_creation.CPUAffinity({0})
    assert affinity_obj.get() is None
