#!/usr/bin/env python3
"""
make_cutout.py

Create a central NxN pixel cutout from a PNG or FITS image.

USAGE:
    python3 make_cutout.py path/to/image N [--to_fits] [--to_png]

ARGUMENTS:
    path/to/image   Path to the input image. Must end in .png, .fits, or .fit
    N               Integer size (in pixels) of the square cutout. The script
                    extracts the central N x N pixels of the image.

OPTIONAL FLAGS:
    --to_fits       If the input image is a PNG, also save an additional copy
                    of the cutout as a FITS file (grayscale pixel data, no
                    photometric calibration implied). Ignored if input is
                    already FITS.
    --to_png        If the input image is a FITS file, also save an additional
                    copy of the cutout as a PNG (linearly scaled to 8-bit for
                    display purposes only -- not flux calibrated). Ignored if
                    input is already PNG.

OUTPUT:
    Cutouts are written to a "cropped_images" directory created (if needed)
    inside the same directory as the original image. The output filename is
    the original filename with "_cropped_N_by_N" appended before the file
    extension, e.g.:

        my_image.png  ->  cropped_images/my_image_cropped_200_by_200.png

    If --to_fits or --to_png is used, an additional file with the same base
    name but the other extension is also written to the same directory.

ERRORS:
    The script raises an error and exits if N is larger than either
    dimension of the input image.
"""

import argparse
import os
import sys

import numpy as np
from astropy.io import fits
from PIL import Image


def load_image(path):
    """Load image data as a 2D (or 3D for RGB PNG) numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        img = Image.open(path)
        data = np.array(img)
        return data, "png", img.mode
    elif ext in (".fits", ".fit"):
        with fits.open(path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        return data, "fits", header
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Must be .png, .fits, or .fit")


def central_cutout(data, n):
    """Return the central n x n cutout of a 2D or 3D (H, W, channels) array."""
    height, width = data.shape[0], data.shape[1]

    if n > height or n > width:
        raise ValueError(
            f"Requested cutout size {n} exceeds image dimensions ({width}x{height})."
        )

    y_start = (height - n) // 2
    x_start = (width - n) // 2

    if data.ndim == 2:
        return data[y_start:y_start + n, x_start:x_start + n]
    else:
        return data[y_start:y_start + n, x_start:x_start + n, ...]


def save_png(data, path, mode=None):
    if data.ndim == 2 and mode not in ("L", None):
        # grayscale array but original mode indicated something else; just use array as-is
        pass
    Image.fromarray(data).save(path)


def save_png_from_fits_data(data, path):
    """Scale FITS data linearly to 8-bit and save as PNG (display only)."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("FITS data contains no finite values to scale for PNG output.")
    lo, hi = np.min(finite), np.max(finite)
    if hi == lo:
        scaled = np.zeros_like(data, dtype=np.uint8)
    else:
        scaled = (data - lo) / (hi - lo)
        scaled = np.clip(scaled, 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    Image.fromarray(scaled).save(path)


def save_fits_from_png_data(data, path):
    """Convert PNG array data to FITS. Collapses RGB(A) to grayscale via averaging."""
    if data.ndim == 3:
        data = data[..., :3].mean(axis=2)
    hdu = fits.PrimaryHDU(data=data)
    hdu.writeto(path, overwrite=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create a central NxN pixel cutout from a PNG or FITS image."
    )
    parser.add_argument("image_path", help="Path to the input .png or .fits image")
    parser.add_argument("num_pixels", type=int, help="Size (N) of the central NxN cutout")
    parser.add_argument(
        "--to_fits",
        action="store_true",
        help="Also save a FITS copy of the cutout (only applies when input is PNG)",
    )
    parser.add_argument(
        "--to_png",
        action="store_true",
        help="Also save a PNG copy of the cutout (only applies when input is FITS)",
    )

    args = parser.parse_args()

    image_path = args.image_path
    n = args.num_pixels

    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    if n <= 0:
        print("Error: num_pixels must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    try:
        data, filetype, extra = load_image(image_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        cutout = central_cutout(data, n)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    src_dir = os.path.dirname(os.path.abspath(image_path))
    out_dir = os.path.join(src_dir, "cropped_images")
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    suffix = f"_cropped_{n}_by_{n}"

    if filetype == "png":
        out_png_path = os.path.join(out_dir, f"{base_name}{suffix}.png")
        save_png(cutout, out_png_path, mode=extra)
        print(f"Saved PNG cutout: {out_png_path}")

        if args.to_fits:
            out_fits_path = os.path.join(out_dir, f"{base_name}{suffix}.fits")
            save_fits_from_png_data(cutout, out_fits_path)
            print(f"Saved FITS cutout: {out_fits_path}")

    else:  # fits
        out_fits_path = os.path.join(out_dir, f"{base_name}{suffix}.fits")
        header = extra
        hdu = fits.PrimaryHDU(data=cutout, header=header)
        hdu.writeto(out_fits_path, overwrite=True)
        print(f"Saved FITS cutout: {out_fits_path}")

        if args.to_png:
            out_png_path = os.path.join(out_dir, f"{base_name}{suffix}.png")
            save_png_from_fits_data(cutout, out_png_path)
            print(f"Saved PNG cutout: {out_png_path}")


if __name__ == "__main__":
    main()