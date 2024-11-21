"""Utility for processing a SICD to a ground plane detected image SIDD"""

__classification__ = "UNCLASSIFIED"

import logging
import pathlib
import tracemalloc

import numba
import numpy as np
import numpy.polynomial.polynomial as npp

# TODO Refactor from sarpy2
import sarpy.geometry.point_projection
import sarpy.io.product.sidd2_elements.ProductProcessing
import sarpy.processing.ortho_rectify
import sarpy.processing.sicd.spectral_taper
import sarpy.processing.sidd.sidd_structure_creation

import sarpy.fast_processing.backend
import sarpy.fast_processing.metadata
from sarpy.fast_processing import adjust_sicd_osr
from sarpy.fast_processing import benchmark
from sarpy.fast_processing import projection
from sarpy.fast_processing import sidelobe_control
from sarpy.fast_processing import spectral_shaping
from sarpy.fast_processing import sva
from sarpy.fast_processing import read_sicd
from sarpy.fast_processing import remap
from sarpy.fast_processing import weight_and_adjust_osr
from sarpy.fast_processing import write_sidd


def sicd_to_sidd(data, sicd_metadata, sidelobe_control,
                 sidd_version=3, apply_spectral_shaping=True,
                 bit_depth=8):
    """Produce a SIDD from a SICD

    Args
    ----
    data: `numpy.ndarray`
        SICD pixel array.  2D array of complex values sampled on the SICD grid.
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    sidelobe_control: str
        Sidelobe control applied
    sidd_version: int, optional
        Version of SIDD metadata to produce
    apply_spectral_shaping: bool
        Indicates whether to apply spectral shaping prior to remap
    bit_depth: int
        Indicates bits to use for output.  Must be 8 or 16.

    Returns
    -------
    numpy.ndarray
        SIDD pixel array. 2D array of uint8.
    sarpy.io.product.sidd3_elements.SIDD.SIDDType
        SIDD Metadata object
    """
    assert bit_depth in [8, 16]
    proj_helper, ortho_bounds = _projection_info(sicd_metadata)

    # amplitude
    with benchmark.howlong('amplitude'):
        amp_data = _amplitude(data)
        data = None

    # Precompute remap parameters
    if bit_depth == 8:
        with benchmark.howlong('gdm parameters'):
            gdm_params = remap.gdm_metadata_parameters(sicd_metadata)
            amp_to_dens_params = remap.gdm_remap_parameters(amp_data, **gdm_params)
        if apply_spectral_shaping:
            with benchmark.howlong('spectral shaping'):
                ss_params = spectral_shaping.compute_spectral_shaping_parameters(amp_to_dens_params['c_l'],
                                                                                 amp_to_dens_params['c_h'],
                                                                                 sidelobe_control)
                shaped_data = spectral_shaping.apply_filter(amp_data,
                                                            ss_params['x_0'],
                                                            ss_params['x_2'],
                                                            ss_params['lim_n'])
                amp_data = None
        else:
            shaped_data = amp_data
            amp_data = None
        with benchmark.howlong('perform density remap'):
            remap_data = remap.amp_to_dens(shaped_data,
                                           dmin=amp_to_dens_params['dmin'],
                                           mmult=amp_to_dens_params['mmult'],
                                           data_mean=amp_to_dens_params['data_mean'])
            shaped_data = None
    else:
        with benchmark.howlong('linear remap parameters'):
            linear_params = remap.linear_remap_parameters(amp_data)
        with benchmark.howlong('perform linear remap'):
            remap_data = remap.linear_remap(amp_data,
                                            min_input_val=linear_params['min_input_val'],
                                            max_input_val=linear_params['max_input_val'],
                                            min_output_val=linear_params['min_output_val'],
                                            max_output_val=linear_params['max_output_val'])
        amp_data = None

    # project
    with benchmark.howlong('projection'):
        # TODO replace projection_helper with output plane and grid computation
        # TODO compute output plane/grid, store in SICD metadata
        # TODO adjust output plane based on chipped extent
        # TODO compute SIDD metadata from SICD metadata
        # TODO create callables for SICD <--> SIDD coordinates
        proj_data = projection.project(remap_data, sicd_metadata, proj_helper, ortho_bounds)
        remap_data = None

    with benchmark.howlong('output formatting'):
        if bit_depth == 8:
            _clip_inplace(proj_data, 0, 2**8-1)  # projection interpolation could result in small negative values
            output_data = proj_data.astype(np.uint8)
        else:
            _clip_inplace(proj_data, 0, 2**16-1)  # projection interpolation could result in small negative values
            output_data = proj_data.astype(np.uint16)
        proj_data = None

    sidd_metadata = _create_sidd_metadata(proj_helper, ortho_bounds, sidd_version, bit_depth)

    return output_data, sidd_metadata


def _create_sidd_metadata(proj, bounds, sidd_version, bit_depth):
    """Generate the SIDD metadata for the supplied projection helper

    Args
    ----
    proj: sarpy.processing.ortho_rectify.projection_helper.PGRatPolyProjection
        Projection helper
    bounds: array-like
        Output area bounds.  [min row, max row, min column, max column]
    sidd_version: int
        Version of SIDD metadata to produce
    bit_depth: int
        Indicates bits to use for output.  Must be 8 or 16.

    Returns
    -------
    sarpy.io.product.sidd3_elements.SIDD.SIDDType
        SIDD Metadata object

    """

    # legacy create_sidd_structure requires an ortho_helper
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
        'Detected Image',
        f'MONO{bit_depth}I',
        version=sidd_version)
    _propagate_proc_metadata(ortho_helper.proj_helper.sicd, sidd_metadata)
    return sidd_metadata


def _propagate_proc_metadata(sicd_meta, sidd_meta):
    """Propagate SICD/ImageFormation/Processing parameters to SIDD/ProductProcessing/ProcessingModule"""
    if not sicd_meta.ImageFormation.Processings:
        return
    if sidd_meta.ProductProcessing is None:
        sidd_meta.ProductProcessing = sarpy.io.product.sidd2_elements.ProductProcessing.ProductProcessingType()
    for sicd_proc in sicd_meta.ImageFormation.Processings:
        new_pm = {"ModuleName": "", "name": sicd_proc.Type}
        if sicd_proc.Parameters:
            new_pm["ModuleParameters"] = sicd_proc.Parameters.to_dict()
        sidd_meta.ProductProcessing.addProcessingModule(new_pm)


@numba.njit(parallel=True)
def _amplitude(data):
    """Numba parallelized np.abs"""
    return np.abs(data)


@numba.njit(parallel=True)
def _clip_inplace(data, min_val, max_val):
    """Inplace clip values to min_val and max_val"""
    # Explicit numba loops are faster than np.clip
    for row in numba.prange(data.shape[0]):
        for col in numba.prange(data.shape[1]):
            if data[row, col] < min_val:
                data[row, col] = min_val
            elif data[row, col] > max_val:
                data[row, col] = max_val
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
    """CLI utility for creating SIDD NITFs from SICDs"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_sicd', type=pathlib.Path, help="Path to input SICD")
    parser.add_argument('output_sidd', type=pathlib.Path, help="Path to write SIDD NITF")
    parser.add_argument('--sidelobe-control', choices=['Skip', 'Uniform', 'Taylor', 'SVA', 'DSVA', 'JIQ'],
                        default='Skip', help="Desired sidelobe control. Default: %(default)s,"
                        " which retains weighting of input SICD.")
    parser.add_argument('--spectral-shaping', action=argparse.BooleanOptionalAction, default=True,
                        help="Apply spectral shaping (only applicable for 8-bit output)")
    parser.add_argument('--pre-detection-osr', default=None, type=str, choices=['5/4', '4/3', '3/2', '2/1'],
                        help="Oversample ratio prior to detection.  Will impact quality of DSVA and JIQ."
                        "  SVA sets to 2/1  JIQ defaults to 3/2")
    parser.add_argument('--egr-threshold', default=0.2, type=float,
                        help="Threshold for applying EGR. Default: %(default)s, 0 turns EGR off.")
    parser.add_argument('--egr-max-weight', default=0.45, type=float,
                        help="Max weight used by EGR. Default: %(default)s, set lower to increase correction.")
    parser.add_argument('--sidd-version', default=3, type=int, choices=[1, 2, 3],
                        help="The version of the SIDD standard used.  Default: %(default)s")
    parser.add_argument('--bit-depth', default=8, type=int, choices=[8, 16],
                        help="The number of bits to use for each output pixel."
                        " 16-bit output is experimental. Default: %(default)s")
    parser.add_argument('--fft-backend', choices=['auto', 'mkl', 'scipy'], default='auto',
                        help="Which FFT backend to use. Default: %(default)s, which will use mkl if available")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose logging (may be repeated)")
    parser.add_argument('--log-memory-usage', action='store_true',
                        help="Enable logging of memory usage (may significantly reduce performance)")
    config = parser.parse_args(args)

    loglevels = [logging.WARNING, logging.INFO, logging.DEBUG]
    loglevel = loglevels[min(config.verbose, len(loglevels)-1)]
    sarpy.fast_processing.backend.initialize_logging(loglevel)
    logging.info(f"Log level set to {logging.getLevelName(loglevel)}")

    logging.info(f"Memory usage logging enabled: {config.log_memory_usage}")
    if config.log_memory_usage:
        tracemalloc.start()

    sidelobe_option = config.sidelobe_control.upper()
    if sidelobe_option in {'SVA', 'DSVA', 'JIQ'}:
        window_name = 'Uniform'
    else:
        window_name = sidelobe_option
    pre_detection_osr = '2/1' if sidelobe_option == 'SVA' else config.pre_detection_osr
    if pre_detection_osr is None and sidelobe_option == 'JIQ':
        pre_detection_osr = '3/2'
    if pre_detection_osr is not None:
        quotient = pre_detection_osr.split('/')
        numerator = float(quotient[0])
        denominator = float(quotient[1])
        target_osr = numerator / denominator
    else:
        target_osr = None

    with sarpy.fast_processing.backend.set_fft_backend(config.fft_backend):
        with benchmark.howlong("SICDtoSIDD"):
            with benchmark.howlong('read'):
                sicd_pixels, sicd_metadata = read_sicd.read_from_file(config.input_sicd)

            apply_weighting = (window_name != 'SKIP')
            adjust_osr = (target_osr is not None and sidelobe_option != "JIQ")
            if apply_weighting:
                taper = sarpy.processing.sicd.spectral_taper.Taper(window_name)
                new_window = taper.get_vals(65, sym=True)
                new_params = taper.window_pars

            if apply_weighting and adjust_osr:
                with benchmark.howlong('Weighting and OSR Adjust'):
                    osr_pixels, sicd_metadata = weight_and_adjust_osr.sicd_to_sicd(sicd_pixels,
                                                                                   sicd_metadata,
                                                                                   target_osr,
                                                                                   new_window,
                                                                                   window_name,
                                                                                   new_params)
            elif apply_weighting:
                with benchmark.howlong('Weighting'):
                    osr_pixels, sicd_metadata = sidelobe_control.sicd_to_sicd(sicd_pixels,
                                                                              sicd_metadata,
                                                                              new_window,
                                                                              window_name,
                                                                              new_params)
            elif adjust_osr:
                with benchmark.howlong('Adjust OSR'):
                    osr_pixels, sicd_metadata = adjust_sicd_osr.sicd_to_sicd(sicd_pixels,
                                                                             sicd_metadata,
                                                                             target_osr)
            else:
                osr_pixels = sicd_pixels
            sicd_pixels = None

            if sidelobe_option in {'SVA', 'DSVA', 'JIQ'}:
                with benchmark.howlong('Apodization'):
                    if sidelobe_option == 'SVA':
                        with benchmark.howlong('Apply 2D Independent IQ SVA'):
                            row_kctr_poly_rad, col_kctr_poly_rad = _kctr_polys_from_sicd_meta(sicd_metadata)
                            sp_pixels = sva.uncoup_sva(osr_pixels,
                                                       row_kctr_poly_rad,
                                                       col_kctr_poly_rad,
                                                       edge_glint_threshold=config.egr_threshold,
                                                       edge_glint_max_weight=config.egr_max_weight)
                            osr_pixels = None
                    elif sidelobe_option == 'DSVA':
                        with benchmark.howlong('Apply Double SVA'):
                            row_nyq_rate = 1 / (sicd_metadata.Grid.Row.SS * sicd_metadata.Grid.Row.ImpRespBW)
                            col_nyq_rate = 1 / (sicd_metadata.Grid.Col.SS * sicd_metadata.Grid.Col.ImpRespBW)
                            row_kctr_poly_rad, col_kctr_poly_rad = _kctr_polys_from_sicd_meta(sicd_metadata)
                            sp_pixels = sva.d_sva(osr_pixels,
                                                  row_nyq_rate,
                                                  col_nyq_rate,
                                                  row_kctr_poly_rad,
                                                  col_kctr_poly_rad,
                                                  edge_glint_threshold=config.egr_threshold,
                                                  edge_glint_max_weight=config.egr_max_weight)
                            osr_pixels = None
                    else:
                        with benchmark.howlong('Apply Joint IQ SVA'):
                            sp_pixels, sicd_metadata = sva.jiq_sicd(osr_pixels,
                                                                    sicd_metadata,
                                                                    desired_osr=numerator,
                                                                    decimation=denominator,
                                                                    edge_glint_threshold=config.egr_threshold,
                                                                    edge_glint_max_weight=config.egr_max_weight)
                            osr_pixels = None
            else:
                sp_pixels = osr_pixels
                osr_pixels = None

            sarpy.fast_processing.metadata.add_sicd_processing(
                sicd_metadata,
                pathlib.Path(__file__).name,
                parameters={
                    "sidelobe_control": config.sidelobe_control,
                    "spectral_shaping": config.spectral_shaping,
                    "pre_detection_osr": config.pre_detection_osr,
                    "egr_threshold": config.egr_threshold,
                    "egr_max_weight": config.egr_max_weight,
                    "fft_backend": config.fft_backend,
                },
            )
            sidd_pixels, sidd_meta = sicd_to_sidd(sp_pixels, sicd_metadata,
                                                  sidelobe_control=sidelobe_option,
                                                  sidd_version=config.sidd_version,
                                                  apply_spectral_shaping=config.spectral_shaping,
                                                  bit_depth=config.bit_depth)
            sp_pixels = None

            with benchmark.howlong('write'):
                write_sidd.write_to_file(str(config.output_sidd), sidd_pixels, sidd_meta)


def _projection_info(sicd_meta):
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

    # legacy OrthoHelper requires an SICDTypeReader
    from sarpy.io.complex.base import SICDTypeReader
    class DummyReader(SICDTypeReader):
        def __init__(self, sicd_meta):
            super().__init__(data_segment=None, sicd_meta=sicd_meta)

        def get_sicds_as_tuple(self):
            return (self.sicd_meta, )

    ortho_helper = NearestNeighborMethod(
        DummyReader(sicd_meta),
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
