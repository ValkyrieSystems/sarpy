"""Write a SIDD file"""

__classification__ = "UNCLASSIFIED"

import numpy as np

import sarpy.io.product

DEFAULT_BLOCK_SIZE = 128 << 20


def write_to_file(filename, sidd_pixels, sidd_metadata, blocksize=DEFAULT_BLOCK_SIZE):
    """Write sidd data and metadata to a file

    Args
    ----
    filename: str or path-like
        Path to sidd file
    sidd_pixels: `numpy.ndarray`
        sidd pixels
    sidd_metadata: `sarpy.io.product.sidd_elements.sidd.siddType`
        SIDD Metadata object
    blocksize: int
        Approximate number of bytes to write at a time

    Returns
    -------
    None
    """
    with sarpy.io.product.sidd.SIDDWriter(str(filename), sidd_metadata, check_existence=False) as writer:
        # Writing blocks is much more memory efficient for large files
        # The writer will allocate working space to hold the entire chip
        num_splits = min(int(np.ceil(sidd_pixels.nbytes / blocksize)), sidd_pixels.shape[0])
        first_row = 0
        for split in np.array_split(sidd_pixels, num_splits, axis=0):
            writer.write_chip(split, start_indices=(first_row, 0))
            first_row += split.shape[0]
    return
