#!/usr/bin/env python3

# call: python3 derotate_with_astrometry.py input_dir output_dir
# script to sep -> astrometry -> rotate all images so north is up
# writes images as derotated_ORIGINAL_NAME.fits in output_dir
# writing this code to fix VATT images when the derotator was broken

import argparse
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

FITS_EXTS = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz")

def find_fits_files(input_dir):
    """Return a sorted list of FITS files directly inside input_dir."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory")
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.name.lower().endswith(FITS_EXTS)
    )

def parse_args():
    """return a parser object with the command line args"""
    parser = argparse.ArgumentParser(
        description="Interactively select a solar system object across a "
                    "directory of FITS frames, stack aligned cutouts, and "
                    "export a stretch-adjusted greyscale image."
    )
    parser.add_argument("input_dir", help="directory containing FITS images")
    parser.add_argument("output_dir", help="path to write the final image (e.g. output.png)")
    return parser.parse_args()

def load_image_data(path):
    """Return the first 2D data array found in a FITS file."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                data = np.asarray(hdu.data, dtype=float)
                if data.ndim > 2:
                    data = data[0]
                return data
    raise ValueError(f"no 2D image data found in {path.name}")

def derotate(data_arr, output_path):
    """Solve astrometry for the image, write a derotated (N-up) image to output_path"""