import pathlib

import numpy as np

import sarpy.fast_processing.write_sidd
import sarpy.io.product.sidd3_elements.SIDD as sarpy_sidd3
import sarpy.io.product.sidd


def test_write_to_file(tmp_path):
    sidd_meta = sarpy_sidd3.SIDDType.from_xml_file(pathlib.Path(__file__).parents[1] / 'data/example.sidd.xml')

    assert sidd_meta.Display.PixelType == 'MONO8I'
    sidd_pixels = np.random.default_rng(seed=123).integers(0,
                                                           256,
                                                           size=sidd_meta.Measurement.PixelFootprint.get_array(),
                                                           dtype=np.uint8)

    out_file = tmp_path / 'out.sidd'
    sarpy.fast_processing.write_sidd.write_to_file(out_file, sidd_pixels, sidd_meta)

    with sarpy.io.product.sidd.SIDDReader(str(out_file)) as reader:
        read_pixels = reader[...]

    np.testing.assert_array_equal(sidd_pixels, read_pixels)
