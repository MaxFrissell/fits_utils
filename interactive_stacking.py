#!/usr/bin/env python3
"""
interactive_stacking.py

For every FITS image in an input directory:
  1. Display the image (default zscale stretch).
  2. Click on the solar system object, or press 's' to skip the frame,
     or 'q' to stop the selection pass early.

For every frame that was not skipped, a square cutout centered on the
clicked point is extracted and the cutouts are median-combined into a
single stacked image. A second GUI then lets you adjust the zscale
contrast (and fine-tune vmin/vmax directly) with a live preview before
saving the final image to disk as inverted greyscale (black = higher
pixel values).

Usage:
    python3 interactive_stacking.py input_dir output_file_path [-s/--size 355]

Controls in the frame-selection window:
    left click  - record the object position and move to the next frame
    s           - skip this frame (no cutout is made)
    q           - stop selecting frames (proceeds to stacking with
                  whatever has been collected so far)

Controls in the stretch-adjustment window:
    Contrast slider - recomputes vmin/vmax via ZScaleInterval(contrast=...)
    vmin / vmax     - manual fine-tuning, overrides the contrast preset
    Save button     - writes output_file_path and closes the window
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval

import matplotlib
try:
    matplotlib.use("TkAgg")
except ImportError:
    pass  # fall back to whatever interactive backend is already configured
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

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


class PointPicker:
    """Show one image and let the user click the object, skip, or quit."""

    def __init__(self, data, title):
        self.data = data
        self.point = None
        self.skipped = False
        self.quit = False

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title(title)

        vmin, vmax = ZScaleInterval().get_limits(data)
        self.ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        self.ax.set_title(
            f"{title}\nclick object  |  's' skip  |  'q' quit selection"
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.point = (event.xdata, event.ydata)
        plt.close(self.fig)

    def _on_key(self, event):
        if event.key == "s":
            self.skipped = True
            plt.close(self.fig)
        elif event.key == "q":
            self.quit = True
            plt.close(self.fig)

    def run(self):
        plt.show()
        return self.point, self.skipped, self.quit


def collect_cutouts(files, size):
    """Walk every file, letting the user pick a point, and cut it out."""
    cutouts = []
    for path in files:
        try:
            data = load_image_data(path)
        except ValueError as exc:
            print(f"  skipping {path.name}: {exc}", file=sys.stderr)
            continue

        point, skipped, quit_ = PointPicker(data, path.name).run()

        if quit_:
            print("stopping frame selection early (quit requested).")
            break
        if skipped or point is None:
            print(f"  skipped {path.name}")
            continue

        x, y = point
        cutout = Cutout2D(
            data, position=(x, y), size=(size, size),
            mode="partial", fill_value=np.nan,
        )
        cutouts.append(cutout.data)
        print(f"  added {path.name} at pixel ({x:.1f}, {y:.1f})")

    return cutouts


def stack_cutouts(cutouts):
    """Median-combine a list of equally-sized cutouts, ignoring NaN padding."""
    stack = np.array(cutouts)
    with np.errstate(all="ignore"):
        combined = np.nanmedian(stack, axis=0)
    # any pixel that was NaN in every frame (e.g. a corner off every cutout)
    # falls back to the overall median so the final image has no NaNs.
    fallback = np.nanmedian(combined)
    return np.nan_to_num(combined, nan=fallback)


class StretchAdjuster:
    """Live zscale/contrast preview; saves inverted greyscale on demand."""

    def __init__(self, data, output_path):
        self.data = data
        self.output_path = output_path

        vmin, vmax = ZScaleInterval(contrast=0.25).get_limits(data)
        data_min, data_max = float(np.nanmin(data)), float(np.nanmax(data))

        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        plt.subplots_adjust(bottom=0.28)

        self.im = self.ax.imshow(
            data, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax
        )
        self.ax.set_title("adjust stretch, then click Save")

        ax_contrast = plt.axes([0.2, 0.17, 0.6, 0.03])
        ax_vmin = plt.axes([0.2, 0.12, 0.6, 0.03])
        ax_vmax = plt.axes([0.2, 0.07, 0.6, 0.03])
        ax_save = plt.axes([0.4, 0.01, 0.2, 0.045])

        self.s_contrast = Slider(ax_contrast, "contrast", 0.01, 1.5, valinit=0.25)
        self.s_vmin = Slider(ax_vmin, "vmin", data_min, data_max, valinit=vmin)
        self.s_vmax = Slider(ax_vmax, "vmax", data_min, data_max, valinit=vmax)
        self.b_save = Button(ax_save, "Save")

        self.s_contrast.on_changed(self._on_contrast)
        self.s_vmin.on_changed(self._on_manual)
        self.s_vmax.on_changed(self._on_manual)
        self.b_save.on_clicked(self._on_save)

    def _on_contrast(self, val):
        vmin, vmax = ZScaleInterval(contrast=val).get_limits(self.data)
        for slider, new_val in ((self.s_vmin, vmin), (self.s_vmax, vmax)):
            slider.eventson = False
            slider.set_val(new_val)
            slider.eventson = True
        self._update(vmin, vmax)

    def _on_manual(self, _val):
        self._update(self.s_vmin.val, self.s_vmax.val)

    def _update(self, vmin, vmax):
        self.im.set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()

    def _on_save(self, _event):
        plt.imsave(
            self.output_path,
            self.data,
            cmap="gray_r",       # reversed greyscale: black = high value
            vmin=self.s_vmin.val,
            vmax=self.s_vmax.val,
            origin="lower",
        )
        print(f"saved stacked image to {self.output_path}")
        plt.close(self.fig)

    def run(self):
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively select a solar system object across a "
                    "directory of FITS frames, stack aligned cutouts, and "
                    "export a stretch-adjusted greyscale image."
    )
    parser.add_argument("input_dir", help="directory containing FITS images")
    parser.add_argument("output_file_path", help="path to write the final image (e.g. output.png)")
    parser.add_argument(
        "-s", "--size", type=int, default=355,
        help="side length in pixels of the square cutout (default: 355)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        files = find_fits_files(args.input_dir)
    except NotADirectoryError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"no FITS files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"found {len(files)} FITS files in {args.input_dir}")
    cutouts = collect_cutouts(files, args.size)

    if not cutouts:
        print("no frames were selected; nothing to stack.", file=sys.stderr)
        sys.exit(1)

    print(f"stacking {len(cutouts)} cutout(s) into a {args.size}x{args.size} image...")
    stacked = stack_cutouts(cutouts)

    StretchAdjuster(stacked, args.output_file_path).run()


if __name__ == "__main__":
    main()