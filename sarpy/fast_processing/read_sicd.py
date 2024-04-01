"""Read a SICD file"""

__classification__ = "UNCLASSIFIED"

import numpy as np

import sarpy.io.complex

DEFAULT_BLOCK_SIZE = 128 << 20


def read_from_file(filename, blocksize=DEFAULT_BLOCK_SIZE):
    """Read SICD data and metadata from a file

    Args
    ----
    filename: str or path-like
        Path to SICD file
    blocksize: int
        Approximate number of bytes to read at a time

    Returns
    -------
    sicd_pixels: `numpy.ndarray`
        SICD pixel array.  2D array of complex values.
    sicd_meta: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    """
    with sarpy.io.complex.open(str(filename)) as reader:
        # Retrieving blocks from the reader is much more memory efficient for large SICDs
        # The reader will allocate working space to hold the entire requested data
        sicd_pixels = np.empty(reader.data_size, dtype="complex64")
        num_splits = int(np.ceil(sicd_pixels.nbytes / blocksize))
        first_row = 0
        for split in np.array_split(sicd_pixels, num_splits, axis=0):
            split[...] = reader[first_row:first_row + split.shape[0], :]
            first_row += split.shape[0]
    return sicd_pixels, reader.sicd_meta
