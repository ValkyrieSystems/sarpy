"""Write a SICD file"""

__classification__ = "UNCLASSIFIED"

import numpy as np

import sarpy.io.complex

DEFAULT_BLOCK_SIZE = 128 << 20


def write_to_file(filename, sicd_pixels, sicd_metadata, blocksize=DEFAULT_BLOCK_SIZE):
    """Write SICD data and metadata to a file

    Args
    ----
    filename: str or path-like
        Path to SICD file
    sicd_pixels: `numpy.ndarray`
        SICD pixels
    sicd_metadata: `sarpy.io.complex.sicd_elements.SICD.SICDType`
        SICD Metadata object
    blocksize: int
        Approximate number of bytes to write at a time

    Returns
    -------
    None
    """
    with sarpy.io.complex.sicd.SICDWriter(str(filename), sicd_metadata, check_existence=False) as writer:
        # Writing blocks is much more memory efficient for large files
        # The writer will allocate working space to hold the entire chip
        num_splits = min(int(np.ceil(sicd_pixels.nbytes / blocksize)), sicd_pixels.shape[0])
        first_row = 0
        for split in np.array_split(sicd_pixels, num_splits, axis=0):
            writer.write_chip(split, start_indices=(first_row, 0))
            first_row += split.shape[0]
    return
