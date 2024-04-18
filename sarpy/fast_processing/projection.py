"""Project detected data corresponding to a SICD grid to the ground"""

__classification__ = "UNCLASSIFIED"

import numba
import numpy as np


def project(data, sicd_metadata, proj_helper, ortho_bounds):
    """Project detected data corresponding to a SICD grid to the ground

    Args
    ----
    data: `numpy.ndarray`
        SICD pixels as amplitude
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    proj_helper: sarpy.processing.ortho_rectify.projection_helper.PGRatPolyProjection
        Projection helper
    ortho_bounds: list-like
        output grid bounds [min_x, max_x, min_y, max_y]

    Returns
    -------
    numpy.ndarray
        projected image
    """

    # TODO chips
    input_origin_row = 0
    input_origin_col = 0

    data = _do_numba_projection(proj_helper, ortho_bounds, input_origin_row, input_origin_col, data)

    # TODO compute and return SIDD Valid Data Polygon?
    #   Should probably get computed along side the "proj_helper"
    return data


def _do_numba_projection(proj_helper, ortho_bounds, input_origin_row, input_origin_col, value_array):
    min_xy = np.asarray(ortho_bounds[::2])
    max_xy = np.asarray(ortho_bounds[1::2])

    num_points = 51
    spacing = (max_xy - min_xy) / (num_points - 3)
    output_iac = min_xy - spacing + spacing * np.arange(num_points)[:, np.newaxis]

    mesh = np.stack(np.meshgrid(*output_iac.T, indexing='ij'), axis=-1)
    pixel_coord_mesh = proj_helper.ortho_to_pixel(mesh)

    ortho_array = np.empty((ortho_bounds[1] - ortho_bounds[0],
                            ortho_bounds[3] - ortho_bounds[2]), dtype=value_array.dtype)
    _tie_point_resample(value_array,
                        input_origin_row,
                        input_origin_col,
                        pixel_coord_mesh[..., 0],
                        pixel_coord_mesh[..., 1],
                        output_iac[0, 0] - min_xy[0],
                        output_iac[0, 1] - min_xy[1],
                        spacing[0],
                        spacing[1],
                        ortho_array)
    return ortho_array


@numba.njit
def _cubic_interpolate(y0, y1, y2, y3, x):
    """Catmull-Rom cubic spline"""
    return y1 + 0.5 * x*(y2 - y0 + x*(2.0*y0 - 5.0*y1 + 4.0*y2 - y3 + x*(3.0*(y1 - y2) + y3 - y0)))


@numba.njit
def _bicubic_interpolate(z00, z01, z02, z03,
                         z10, z11, z12, z13,
                         z20, z21, z22, z23,
                         z30, z31, z32, z33,
                         x, y):
    z0 = _cubic_interpolate(z00, z01, z02, z03, y)
    z1 = _cubic_interpolate(z10, z11, z12, z13, y)
    z2 = _cubic_interpolate(z20, z21, z22, z23, y)
    z3 = _cubic_interpolate(z30, z31, z32, z33, y)

    z = _cubic_interpolate(z0, z1, z2, z3, x)

    return z


@numba.njit
def _do_interp_bilinear(input_data,
                        row,
                        col):
    if (row < 0
            or col < 0
            or row >= input_data.shape[0] - 1
            or col >= input_data.shape[1] - 1):
        return 0
    int_row = np.int64(row)
    int_col = np.int64(col)

    top_left = input_data[int_row, int_col]
    top_right = input_data[int_row, int_col + 1]
    bottom_left = input_data[int_row + 1, int_col]
    bottom_right = input_data[int_row + 1, int_col + 1]
    row_frac = row - int_row
    col_frac = col - int_col
    one = 1.0
    return ((top_left * (one - col_frac) + top_right * col_frac) * (one - row_frac) +
            (bottom_left * (one - col_frac) + bottom_right * col_frac) * row_frac)


@numba.njit
def _do_interp_bicubic(input_data,
                       row,
                       col):
    if (row < 0
            or col < 0
            or row >= input_data.shape[0] - 1
            or col >= input_data.shape[1] - 1):
        return 0
    int_row = np.int64(row)
    int_col = np.int64(col)
    # These bools indicate whether the first/last row/column in the 4x4 neighborhood of the point
    # to interpolate is supported by data, or whether it'll need to be synthesized
    first_row = (int_row > 0)
    first_col = (int_col > 0)
    last_row = (int_row < input_data.shape[0] - 2)
    last_col = (int_col < input_data.shape[1] - 2)

    z11 = input_data[int_row, int_col]
    z12 = input_data[int_row, int_col + 1]
    z21 = input_data[int_row + 1, int_col]
    z22 = input_data[int_row + 1, int_col + 1]

    if first_row:
        z01 = input_data[int_row-1, int_col]
        z02 = input_data[int_row-1, int_col + 1]
    else:
        z01 = 2*z11 - z21
        z02 = 2*z12 - z22

    if last_row:
        z31 = input_data[int_row + 2, int_col]
        z32 = input_data[int_row + 2, int_col + 1]
    else:
        z31 = 2*z21 - z11
        z32 = 2*z22 - z12

    if first_col:
        z10 = input_data[int_row, int_col - 1]
        z20 = input_data[int_row + 1, int_col - 1]
    else:
        z10 = 2*z11 - z12
        z20 = 2*z21 - z22

    if last_col:
        z13 = input_data[int_row, int_col + 2]
        z23 = input_data[int_row + 1, int_col + 2]
    else:
        z13 = 2*z21 - z11
        z23 = 2*z22 - z12

    if first_row and first_col:
        z00 = input_data[int_row - 1, int_col - 1]
    else:
        z00 = 2*z01 - z02

    if first_row and last_col:
        z03 = input_data[int_row - 1, int_col + 2]
    else:
        z03 = 2*z02 - z01

    if last_row and first_col:
        z30 = input_data[int_row + 2, int_col - 1]
    else:
        z30 = 2*z31 - z32

    if last_row and last_col:
        z33 = input_data[int_row + 2, int_col + 2]
    else:
        z33 = 2*z32 - z31

    row_frac = row - int_row
    col_frac = col - int_col

    return _bicubic_interpolate(z00, z01, z02, z03,
                                z10, z11, z12, z13,
                                z20, z21, z22, z23,
                                z30, z31, z32, z33,
                                row_frac, col_frac)


@numba.njit(parallel=True)
def _tie_point_resample(input_data,
                        input_origin_row,
                        input_origin_col,
                        tie_points_row,
                        tie_points_col,
                        first_tie_point_row,
                        first_tie_point_col,
                        tie_point_row_spacing,
                        tie_point_col_spacing,
                        output_data):

    for rowidx in numba.prange(output_data.shape[0]):
        tie_point_row = (rowidx - first_tie_point_row) / tie_point_row_spacing
        for colidx in range(output_data.shape[1]):
            tie_point_col = (colidx - first_tie_point_col) / tie_point_col_spacing
            row = _do_interp_bilinear(tie_points_row, tie_point_row, tie_point_col)
            col = _do_interp_bilinear(tie_points_col, tie_point_row, tie_point_col)

            output_data[rowidx, colidx] = _do_interp_bicubic(input_data,
                                                             row - input_origin_row,
                                                             col - input_origin_col)
