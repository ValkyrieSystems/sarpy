"""Utility for processing a SICD to a ground plane detected image SIDD"""

__classification__ = "UNCLASSIFIED"

import logging
import pathlib

import numba
import numpy as np
import numpy.polynomial.polynomial as npp

# TODO Refactor from sarpy2
import sarpy.geometry.point_projection
import sarpy.processing.ortho_rectify
import sarpy.processing.sicd.spectral_taper
import sarpy.processing.sidd.sidd_structure_creation

from sarpy.fast_processing import adjust_sicd_osr
from sarpy.fast_processing import benchmark
from sarpy.fast_processing import projection
from sarpy.fast_processing import sidelobe_control
from sarpy.fast_processing import sva
from sarpy.fast_processing import read_sicd
from sarpy.fast_processing import remap


def sicd_to_sidd(data, sicd_metadata, proj_helper, ortho_bounds, sidd_version=3):
    """Produce a SIDD from a SICD

    Args
    ----
    data: `numpy.ndarray`
        SICD pixel array.  2D array of complex values sampled on the SICD grid.
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    sidd_version: int
        Version of SIDD metadata to produce

    Returns
    -------
    numpy.ndarray
        SIDD pixel array. 2D array of uint8.
    sarpy.io.product.sidd3_elements.SIDD.SIDDType
        SIDD Metadata object
    """

    # amplitude
    with benchmark.howlong('amplitude'):
        data = _amplitude(data)

    # project
    with benchmark.howlong('projection'):
        # TODO replace projection_helper with output plane and grid computation
        # TODO compute output plane/grid, store in SICD metadata
        # TODO adjust output plane based on chipped extent
        # TODO compute SIDD metadata from SICD metadata
        # TODO create callables for SICD <--> SIDD coordinates
        data = projection.project(data, sicd_metadata, proj_helper, ortho_bounds)

    with benchmark.howlong('remap'):
        _clip_zero_inplace(data)  # projection interpolation could result in small negative values
        gdm_params = remap.gdm_parameters(sicd_metadata)
        data = remap.gdm(data, **gdm_params)

    sidd_metadata = _create_sidd_metadata(proj_helper, ortho_bounds, sidd_version)

    return data, sidd_metadata


def _create_sidd_metadata(proj, bounds, sidd_version):
    """Generate the SIDD metadata for the supplied projection helper

    Args
    ----
    proj: sarpy.processing.ortho_rectify.projection_helper.PGRatPolyProjection
        Projection helper
    bounds: array-like
        Output area bounds.  [min row, max row, min column, max column]
    sidd_version: int
        Version of SIDD metadata to produce

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
    sidd_metadata = sarpy.processing.sidd.sidd_structure_creation.create_sidd_structure(
        ortho_helper,
        bounds,
        'Detectected Image',
        'MONO8I',
        version=sidd_version)
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

def _scale_input_and_shift(coefs, scales, new_origins):
    coefs = np.array(coefs)
    shape = coefs.shape
    assert len(shape) == len(scales) == len(new_origins)
    for axis_ndx, (scale, new_origin) in enumerate(zip(scales, new_origins)):
        moved_coefs = np.moveaxis(coefs, axis_ndx, -1)
        flattened_coefs = moved_coefs.reshape((-1, shape[axis_ndx]))
        for ndx in range(flattened_coefs.shape[0]):
            poly = npp.Polynomial(flattened_coefs[ndx, :])
            shifted_poly = poly.convert(domain=[0, scale], window=[new_origin, new_origin+1])
            num_coefs = len(shifted_poly.coef)
            flattened_coefs[ndx, :num_coefs] = shifted_poly.coef
        inflated_coefs = flattened_coefs.reshape(moved_coefs.shape)
        coefs = np.moveaxis(inflated_coefs, -1, axis_ndx)

    return coefs

def _kctr_polys_from_sicd_meta(sicd_metadata):
    scales = (sicd_metadata.Grid.Row.SS, sicd_metadata.Grid.Col.SS)
    shifts = (sicd_metadata.ImageData.SCPPixel.Row - sicd_metadata.ImageData.FirstRow,
              sicd_metadata.ImageData.SCPPixel.Col - sicd_metadata.ImageData.FirstCol)
    row_deltakcoa = (sicd_metadata.Grid.Row.DeltaKCOAPoly.Coefs if sicd_metadata.Grid.Row.DeltaKCOAPoly is not None
                     else np.array([[0.0]]))
    row_kctr_poly_rad = (sicd_metadata.Grid.Row.Sgn * (2 * np.pi) * scales[0]
                         * _scale_input_and_shift(row_deltakcoa,
                                                  scales, shifts))
    col_deltakcoa = (sicd_metadata.Grid.Col.DeltaKCOAPoly.Coefs if sicd_metadata.Grid.Col.DeltaKCOAPoly is not None
                     else np.array([[0.0]]))
    col_kctr_poly_rad = (sicd_metadata.Grid.Col.Sgn * (2 * np.pi) * scales[1]
                         * _scale_input_and_shift(col_deltakcoa,
                                                  scales, shifts))

    return row_kctr_poly_rad, col_kctr_poly_rad

def main(args=None):
    """CLI utility for creating SIDD v3 NITFs from SICDs"""
    import argparse
    import sarpy.io.complex
    import sarpy.io.product.sidd

    parser = argparse.ArgumentParser()
    parser.add_argument('input_sicd', type=pathlib.Path, help="Path to input SICD")
    parser.add_argument('output_sidd', type=pathlib.Path, help="Path to write SIDD NITF")
    parser.add_argument('--sidelobe-control', choices=['Skip', 'Uniform', 'Taylor', 'SVA', 'DSVA', 'JIQ'],
                        default='Skip', help="Desired sidelobe control.  'Skip' retains weighting of input SICD.")
    parser.add_argument('--sidd-version', default=3, type=int, choices=[1, 2, 3],
                        help="The version of the SIDD standard used.")
    parser.add_argument('--fft-backend', choices=['auto', 'mkl', 'scipy'], default='auto',
                        help="Which FFT backend to use.  Default 'auto', which will use mkl if available")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose logging (may be repeated)")
    config = parser.parse_args(args)

    loglevels = [logging.WARNING, logging.INFO, logging.DEBUG]
    loglevel = loglevels[min(config.verbose, len(loglevels)-1)]
    logging.basicConfig(level=loglevel)
    logging.info(f"Log level set to {logging.getLevelName(loglevel)}")

    with sarpy.fast_processing.backend.set_fft_backend(config.fft_backend):
        with benchmark.howlong("SICDtoSIDD"):
            with benchmark.howlong('read'):
                sicd_pixels, sicd_metadata = read_sicd.read_from_file(config.input_sicd)

            if config.sidelobe_control != 'Skip':
                with benchmark.howlong('sidelobe'):
                    if config.sidelobe_control.upper() in {'SVA', 'DSVA', 'JIQ'}:
                        window_name = 'Uniform'
                        taper = sarpy.processing.sicd.spectral_taper.Taper(window_name)
                        new_window = taper.get_vals(65, sym=True)
                        new_params = taper.window_pars
                        sicd_pixels, sicd_metadata = sidelobe_control.sicd_to_sicd(sicd_pixels,
                                                                                   sicd_metadata,
                                                                                   new_window,
                                                                                   window_name,
                                                                                   new_params)
                        if config.sidelobe_control.upper() == 'SVA':
                            sicd_pixels, sicd_metadata = adjust_sicd_osr.sicd_to_sicd(sicd_pixels,
                                                                                      sicd_metadata,
                                                                                      2.0)
                            row_kctr_poly_rad, col_kctr_poly_rad = _kctr_polys_from_sicd_meta(sicd_metadata)
                            sicd_pixels = sva.uncoup_sva(sicd_pixels,
                                                         row_kctr_poly_rad,
                                                         col_kctr_poly_rad)
                        elif config.sidelobe_control.upper() == 'DSVA':
                            row_nyq_rate = 1 / (sicd_metadata.Grid.Row.SS * sicd_metadata.Grid.Row.ImpRespBW)
                            col_nyq_rate = 1 / (sicd_metadata.Grid.Col.SS * sicd_metadata.Grid.Col.ImpRespBW)
                            row_kctr_poly_rad, col_kctr_poly_rad = _kctr_polys_from_sicd_meta(sicd_metadata)
                            sicd_pixels = sva.d_sva(sicd_pixels,
                                                    row_nyq_rate,
                                                    col_nyq_rate,
                                                    row_kctr_poly_rad,
                                                    col_kctr_poly_rad)
                        elif config.sidelobe_control.upper() == 'JIQ':
                            sicd_pixels, sicd_metadata = sva.jiq_sicd(sicd_pixels,
                                                                      sicd_metadata)
                    else:
                        window_name = config.sidelobe_control.upper()
                        taper = sarpy.processing.sicd.spectral_taper.Taper(window_name)
                        new_window = taper.get_vals(65, sym=True)
                        new_params = taper.window_pars
                        sicd_pixels, sicd_metadata = sidelobe_control.sicd_to_sicd(sicd_pixels,
                                                                                   sicd_metadata,
                                                                                   new_window,
                                                                                   window_name,
                                                                                   new_params)

            with sarpy.io.complex.open(str(config.input_sicd)) as reader:
                # TODO refactor these to run directly from SICD XML and move to sicd_to_sicd()
                proj_helper, ortho_bounds = _projection_info(reader, sicd_metadata)

            sidd_pixels, sidd_meta = sicd_to_sidd(sicd_pixels, sicd_metadata,
                                                  proj_helper=proj_helper, ortho_bounds=ortho_bounds,
                                                  sidd_version=config.sidd_version)

            with benchmark.howlong('write'):
                with sarpy.io.product.sidd.SIDDWriter(str(config.output_sidd), sidd_meta,
                                                      check_existence=False) as writer:
                    writer.write_chip(sidd_pixels)


def _projection_info(reader, sicd_meta):
    """Compute information necessary for ground projection"""
    # TODO refactor this function to run from SICD XML
    from sarpy.processing.ortho_rectify import NearestNeighborMethod
    from sarpy.processing.ortho_rectify import projection_helper

    # Based on sarpy.processing.ortho_rectify.ortho_methods.OrthorectificationHelper.set_index_and_proj_helper
    try:
        plane = sicd_meta.RadarCollection.Area.Plane
        row_sample_spacing = plane.XDir.LineSpacing
        col_sample_spacing = plane.YDir.SampleSpacing
        default_ortho_bounds = np.array([plane.XDir.FirstLine, plane.XDir.FirstLine + plane.XDir.NumLines,
                                         plane.YDir.FirstSample, plane.YDir.FirstSample + plane.YDir.NumSamples],
                                        dtype=np.uint32)
    except AttributeError:
        delta_xrow = 1.0 / sicd_meta.Grid.Row.ImpRespBW
        delta_ycol = 1.0 / sicd_meta.Grid.Col.ImpRespBW
        m_spxy_il = sarpy.geometry.point_projection.image_to_slant_sensitivity(sicd_meta, delta_xrow, delta_ycol)

        graz = np.radians(sicd_meta.SCPCOA.GrazeAng)
        twst = np.radians(sicd_meta.SCPCOA.TwistAng)
        m_gpxy_spxy = np.array([
            [1.0 / np.cos(graz), 0],
            [np.tan(graz) * np.tan(twst), (1.0 / np.cos(twst))]
        ])

        gpxy_resolutions = np.abs(m_gpxy_spxy @ m_spxy_il @ np.array([delta_xrow, delta_ycol]))

        sample_spacing = 0.886 * min(gpxy_resolutions) / 1.5
        row_sample_spacing = sample_spacing
        col_sample_spacing = sample_spacing
        default_ortho_bounds = None

    ph_kwargs = {
        'sicd': sicd_meta,
        'row_spacing': row_sample_spacing,
        'col_spacing': col_sample_spacing,
    }
    proj_helper = projection_helper.PGProjection(**ph_kwargs)

    ortho_helper = NearestNeighborMethod(
        reader,
        proj_helper=proj_helper,
    )
    ortho_helper._sicd = sicd_meta

    # Finish up sarpy.processing.ortho_rectify.ortho_metods.OrthorectificationHelper.set_index_and_proj_helper
    if default_ortho_bounds is not None:
        _, ortho_rectangle = ortho_helper.bounds_to_rectangle(default_ortho_bounds)
        ortho_helper._default_physical_bounds = ortho_helper.proj_helper.ortho_to_ecf(ortho_rectangle)

    return ortho_helper.proj_helper, ortho_helper.get_valid_ortho_bounds()


if __name__ == '__main__':
    main()
