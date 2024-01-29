"""Utility for processing a SICD to a ground plane detected image SIDD"""
import contextlib
import pathlib
import time

import numba
import numpy as np

# TODO Refactor from sarpy2
import sarpy.processing.sidd.sidd_structure_creation
import sarpy.processing.sicd.spectral_taper
import sarpy.processing.ortho_rectify

from sarpy.fast_processing import projection
from sarpy.fast_processing import sidelobe_control
from sarpy.fast_processing import remap


@contextlib.contextmanager
def howlong(label):
    """Print how long a section of code took

    Args
    ----
    label: str
        String printed alongside duration

    """
    start = time.perf_counter()
    try:
        yield start
    finally:
        print(f'{label}: {time.perf_counter() - start}')


def sicd_to_sidd(data, sicd_metadata, proj_helper, ortho_bounds):
    """Produce a SIDD from a SICD

    Args
    ----
    data: `numpy.ndarray`
        SICD pixel array.  2D array of complex values.
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object

    Returns
    -------
    numpy.ndarray
        SIDD pixel array. 2D array of uint8.
    sarpy.io.product.sidd3_elements.SIDD.SIDDType
        SIDD Metadata object
    """

    # amplitude
    with howlong('amplitude'):
        data = _amplitude(data)

    # project
    with howlong('projection'):
        # TODO replace projection_helper with output plane and grid computation
        # TODO compute output plane/grid, store in SICD metadata
        # TODO adjust output plane based on chipped extent
        # TODO compute SIDD metadata from SICD metadata
        # TODO create callables for SICD <--> SIDD coordinates
        data = projection.project(data, sicd_metadata, proj_helper, ortho_bounds)

    with howlong('remap'):
        _clip_zero_inplace(data)  # projection interpolation could result in small negative values
        gdm_params = remap.gdm_parameters(sicd_metadata)
        data = remap.gdm(data, **gdm_params)

    sidd_metadata = _create_sidd_metadata(proj_helper, ortho_bounds)

    return data, sidd_metadata


def _create_sidd_metadata(proj, bounds):
    """Generate the SIDD metadata for the supplied projection helper

    Args
    ----
    proj: sarpy.processing.ortho_rectify.projection_helper.PGRatPolyProjection
        Projection helper
    bounds: array-like
        Output area bounds.  [min row, max row, min column, max column]

    Returns
    -------
    sarpy.io.product.sidd3_elements.SIDD.SIDDType
        SIDD Metadata object

    """

    # legacy create_sidd_structure requires and ortho_helper
    class DummyOrthoHelper:
        def __init__(self, proj_helper):
            self.proj_helper = proj_helper

        def bounds_to_rectangle(self, bounds):
            # Copied from OrthorectificationHelper

            bounds = self.validate_bounds(bounds)
            coords = np.zeros((4, 2), dtype=np.int32)
            coords[0, :] = (bounds[0], bounds[2])  # row min, col min
            coords[1, :] = (bounds[0], bounds[3])  # row min, col max
            coords[2, :] = (bounds[1], bounds[3])  # row max, col max
            coords[3, :] = (bounds[1], bounds[2])  # row max, col min
            return bounds, coords

        @staticmethod
        def validate_bounds(bounds):
            import sarpy.processing.ortho_rectify.ortho_methods
            return sarpy.processing.ortho_rectify.ortho_methods.OrthorectificationHelper.validate_bounds(bounds)

    ortho_helper = DummyOrthoHelper(proj)
    sidd_metadata = sarpy.processing.sidd.sidd_structure_creation.create_sidd_structure_v3(ortho_helper,
                                                                                           bounds,
                                                                                           'Detectected Image',
                                                                                           'MONO8I')
    return sidd_metadata


@numba.njit(parallel=True)
def _amplitude(data):
    """Numba parallelized np.abs"""
    return np.abs(data)


@numba.njit(parallel=True)
def _clip_zero_inplace(data):
    """Inplace clip minimum values to zero"""
    # Explicit numba loops are faster than np.clip
    for row in numba.prange(data.shape[0]):
        for col in numba.prange(data.shape[1]):
            if data[row, col] < 0:
                data[row, col] = 0
    return data


def main(args=None):
    """CLI utility for creating SIDD v3 NITFs from SICDs"""
    import argparse
    import sarpy.io.complex
    import sarpy.io.product.sidd

    parser = argparse.ArgumentParser()
    parser.add_argument('input_sicd', type=pathlib.Path, help="Path to input SICD")
    parser.add_argument('output_sidd', type=pathlib.Path, help="Path to write SIDD NITF")
    parser.add_argument('--sidelobe-control', choices=['Skip', 'Uniform', 'Taylor'], default='Skip',
                        help="Desired sidelobe control.  'Skip' retains weighting of input SICD.")
    config = parser.parse_args(args)

    with howlong('read'):
        with sarpy.io.complex.open(str(config.input_sicd)) as reader:
            sicd_pixels = reader[...]
            sicd_metadata = reader.sicd_meta
            # TODO refactor these to run directly from SICD XML and move to sicd_to_sicd()
            proj_helper, ortho_bounds = _projection_info(reader)

    if config.sidelobe_control != 'Skip':
        with howlong('sidelobe'):
            window_name = config.sidelobe_control.upper()
            taper = sarpy.processing.sicd.spectral_taper.Taper(window_name)
            new_window = taper.get_vals(65, sym=True)
            new_params = taper.window_pars
            sicd_pixels, sicd_metadata = sidelobe_control.sicd_to_sicd(sicd_pixels, sicd_metadata,
                                                                       new_window, window_name, new_params)

    new_pixels, new_meta = sicd_to_sidd(sicd_pixels, sicd_metadata,
                                        proj_helper=proj_helper, ortho_bounds=ortho_bounds)

    with howlong('write'):
        with sarpy.io.product.sidd.SIDDWriter(str(config.output_sidd), new_meta) as writer:
            writer.write_chip(new_pixels)


def _projection_info(reader):
    """Compute information necessary for ground projection"""
    # TODO refactor this function to run from SICD XML
    from sarpy.processing.ortho_rectify import NearestNeighborMethod
    from sarpy.processing.ortho_rectify.base import OrthorectificationIterator

    ortho_helper = NearestNeighborMethod(reader)
    ortho_iterator = OrthorectificationIterator(ortho_helper)
    return ortho_helper.proj_helper, ortho_iterator.ortho_bounds


if __name__ == '__main__':
    main()
