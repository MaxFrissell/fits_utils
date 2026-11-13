#!/usr/bin/env python3
"""
interactive_stacking.py

For every FITS image in an input directory:
  1. Display the image (default zscale stretch).
  2. Click on the solar system object to drop a crosshair on it, click
     again anywhere to move the crosshair, then press Enter to confirm
     that frame's position. Press 's' to skip the frame instead, or 'q'
     to stop the selection pass early.
     The confirmed click is then snapped to the centroid of the nearest
     source found by SEP (Source Extractor as a Python library), and that
     centroid is used as the stacking centre. Use --no-sep to use the raw
     cursor position instead.

(optional, -r/--rotate)
  3. Rotation pass. Cutouts (default 1000 px) centred on the confirmed
     points are processed in order. The FIRST image is plate-solved with
     astrometry.net (local `solve-field` if installed, otherwise the web
     service via astroquery) and rotated automatically so north is up.
     If solving fails you get a manual rotation window for it instead.
     Every later image n is shown overlaid on the already-rotated image
     n-1, and you rotate it until the sources line up. Rotation is about
     the cutout centre, i.e. about the object, so the object stays
     centred.

For every frame that was not skipped, a square cutout centred on the
confirmed point is extracted (and rotated, if requested) and the cutouts
are median-combined into a single stacked image. A final GUI then lets
you adjust the zscale contrast (and fine-tune vmin/vmax directly) with a
live preview before saving the image to disk as inverted greyscale
(black = higher pixel values).

Usage:
    python3 interactive_stacking.py input_dir output_file_path [-s/--size 335]
                                    [-r/--rotate] [--rot-size 1000]
                                    [--pixscale ARCSEC_PER_PIX]
                                    [--astrometry-key KEY] [--no-astrometry]

Controls in the frame-selection window:
    left click  - drop/move a crosshair at the object's position
    Enter       - confirm the crosshair position and move to the next frame
    s           - skip this frame (no cutout is made)
    q           - stop selecting frames (proceeds with whatever has been
                  collected so far)

Controls in the rotation window:
    angle slider   - rotation in degrees (positive = counter-clockwise)
    left / right   - nudge angle by -/+ 1 degree
    down / up      - nudge angle by -/+ 0.1 degree
    blend slider   - (overlay only) crossfade between image n-1 (cyan)
                     and image n (red); aligned sources look grey
    Enter          - confirm the angle and move to the next frame
    s              - skip this frame (the first image is never skipped:
                     astrometry rotates it, or you do by hand if the
                     solve fails)

Controls in the stretch-adjustment window:
    Contrast slider - recomputes vmin/vmax via ZScaleInterval(contrast=...)
    vmin / vmax     - manual fine-tuning, overrides the contrast preset
    Save button     - writes output_file_path and closes the window
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from scipy import ndimage

try:
    import sep
except ImportError:
    sep = None

import matplotlib
try:
    matplotlib.use("TkAgg")
except ImportError:
    pass  # fall back to whatever interactive backend is already configured
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button

# matplotlib's default keymap binds 's' to "save figure" (pops up a file
# dialog), 'q'/'ctrl+w' to "close figure", and left/right (plus c/v) to
# back/forward navigation. All of these collide with our own keys.
plt.rcParams["keymap.save"] = []
plt.rcParams["keymap.quit"] = []
plt.rcParams["keymap.back"] = []
plt.rcParams["keymap.forward"] = []

FITS_EXTS = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz")

# scipy.ndimage.rotate with a positive angle looks clockwise when the array
# is shown with origin="lower", so we negate it to make positive = CCW on
# screen. If the direction feels backwards on your data, flip this to +1.
ROT_SIGN = -1


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
    """Show one image; click to crosshair a point, Enter confirms it."""

    def __init__(self, data, title):
        self.data = data
        self.title = title
        self.candidate = None
        self.point = None
        self.skipped = False
        self.quit = False
        self._vline = None
        self._hline = None

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title(title)

        vmin, vmax = ZScaleInterval().get_limits(data)
        self.ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        self._refresh_title()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _refresh_title(self):
        if self.candidate is None:
            status = "click to mark the object"
        else:
            x, y = self.candidate
            status = f"marked ({x:.1f}, {y:.1f}) — Enter to confirm, click to move"
        self.ax.set_title(f"{self.title}\n{status}  |  's' skip  |  'q' quit selection")

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.candidate = (event.xdata, event.ydata)

        if self._vline is not None:
            self._vline.remove()
            self._hline.remove()
        self._vline = self.ax.axvline(event.xdata, color="lime", lw=0.8)
        self._hline = self.ax.axhline(event.ydata, color="lime", lw=0.8)

        self._refresh_title()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "enter":
            if self.candidate is not None:
                self.point = self.candidate
                plt.close(self.fig)
        elif event.key == "s":
            self.skipped = True
            plt.close(self.fig)
        elif event.key == "q":
            self.quit = True
            plt.close(self.fig)

    def run(self):
        plt.show()
        return self.point, self.skipped, self.quit


def detect_sources(data, thresh):
    """Run SEP on a full frame and return its structured array of sources.

    Coordinates are 0-indexed pixel centres, matching matplotlib's
    imshow(origin="lower") axes. Raises on SEP failure.
    """
    arr = np.ascontiguousarray(data, dtype=np.float32)  # native byte order
    mask = ~np.isfinite(arr)
    if mask.all():
        raise ValueError("image has no finite pixels")
    if mask.any():
        arr = arr.copy()
        arr[mask] = np.median(arr[~mask])
    bkg = sep.Background(arr, mask=mask)
    sub = arr - bkg
    return sep.extract(sub, thresh, err=bkg.globalrms, mask=mask)


def refine_position(data, point, thresh):
    """Snap a clicked (x, y) to the centroid of the nearest SEP source.

    Falls back to the original point (with a message) if nothing is found
    or SEP fails.
    """
    x0, y0 = point
    try:
        objs = detect_sources(data, thresh)
    except Exception as exc:  # SEP raises plain Exceptions
        print(f"    SEP failed ({exc}); using cursor position", file=sys.stderr)
        return point

    if len(objs) == 0:
        print("    SEP found no sources; using cursor position", file=sys.stderr)
        return point

    dist = np.hypot(objs["x"] - x0, objs["y"] - y0)
    i = int(np.argmin(dist))
    x, y = float(objs["x"][i]), float(objs["y"][i])
    print(f"    SEP: {len(objs)} sources, nearest is {dist[i]:.1f} px from click")
    return x, y


def collect_cutouts(files, cut_size, use_sep=True, sep_thresh=3.0):
    """Walk every file, letting the user pick a point, and cut it out.

    If use_sep, the confirmed point is replaced by the centroid of the
    nearest SEP-detected source.

    Returns (names, cutouts, first): two parallel lists, plus
    first = (full_frame_data, (x, y)) for the first accepted frame (used
    for astrometry in rotation mode), or None if nothing was accepted.
    """
    names, cutouts = [], []
    first = None
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

        if use_sep:
            point = refine_position(data, point, sep_thresh)

        x, y = point
        cutout = Cutout2D(
            data, position=(x, y), size=(cut_size, cut_size),
            mode="partial", fill_value=np.nan,
        )
        if first is None:
            first = (data, (x, y))
        names.append(path.name)
        cutouts.append(cutout.data.astype(np.float32))
        print(f"  added {path.name} at pixel ({x:.1f}, {y:.1f})")

    return names, cutouts, first


# --------------------------------------------------------------------------
# rotation step
# --------------------------------------------------------------------------

def rotate_nan(arr, angle_deg):
    """Rotate about the array centre; pixels rotated in from outside are NaN.

    order=1 (bilinear) is used on purpose: higher-order splines prefilter
    the whole array, so a single NaN would contaminate every pixel.
    """
    return ndimage.rotate(
        arr, ROT_SIGN * angle_deg, reshape=False,
        order=1, mode="constant", cval=np.nan,
    )


def normalize(cut):
    """Scale a cutout to 0-1 using zscale limits (NaNs preserved)."""
    vmin, vmax = ZScaleInterval().get_limits(cut)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.clip((cut - vmin) / (vmax - vmin), 0.0, 1.0)


def center_crop(arr, size):
    h, w = arr.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr[y0:y0 + size, x0:x0 + size]


def _sources_for_astrometry(data, thresh, nmax=200):
    """Brightest SEP sources as 1-indexed (FITS-convention) x, y arrays."""
    objs = detect_sources(data, thresh)
    objs = objs[objs["flux"] > 0]
    order = np.argsort(-objs["flux"])[:nmax]
    return objs["x"][order] + 1.0, objs["y"][order] + 1.0


def _solve_local(x, y, width, height, pixscale):
    """Solve with a local astrometry.net `solve-field`; returns WCS or None."""
    with tempfile.TemporaryDirectory() as tmp:
        xyls = os.path.join(tmp, "src.xyls")
        Table({"X": x, "Y": y}).write(xyls, format="fits", overwrite=True)
        cmd = [
            "solve-field", xyls,
            "--width", str(width), "--height", str(height),
            "--no-plots", "--overwrite", "--dir", tmp, "--cpulimit", "120",
        ]
        if pixscale:
            cmd += ["--scale-units", "arcsecperpix",
                    "--scale-low", f"{0.8 * pixscale:.4f}",
                    "--scale-high", f"{1.2 * pixscale:.4f}"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return None
        wcs_path = os.path.join(tmp, "src.wcs")
        if not os.path.exists(wcs_path):
            return None
        return WCS(fits.getheader(wcs_path))


def _solve_web(x, y, width, height, pixscale, api_key):
    """Solve via nova.astrometry.net (astroquery); returns WCS or None."""
    from astroquery.astrometry_net import AstrometryNet
    ast = AstrometryNet()
    ast.api_key = api_key
    settings = {}
    if pixscale:
        settings.update(scale_units="arcsecperpix", scale_type="ul",
                        scale_lower=0.8 * pixscale, scale_upper=1.2 * pixscale)
    header = ast.solve_from_source_list(
        x.tolist(), y.tolist(), width, height, solve_timeout=600, **settings)
    return WCS(header) if header else None


def north_angle_from_wcs(wcs, x, y, step_arcsec=30.0):
    """Rotation (degrees, CCW on screen) that puts north up at pixel (x, y).

    x, y are 1-indexed FITS pixel coordinates. Also returns True if the
    image is mirrored (east is to the right once north is up), which a
    pure rotation cannot fix.
    """
    ra, dec = (float(v) for v in wcs.all_pix2world(x, y, 1))
    step = step_arcsec / 3600.0
    sgn = -1.0 if dec + step > 90.0 else 1.0       # avoid stepping over the pole
    nx, ny = (float(v) for v in wcs.all_world2pix(ra, dec + sgn * step, 1))
    cosd = max(np.cos(np.radians(dec)), 1e-6)
    ex, ey = (float(v) for v in wcs.all_world2pix(ra + step / cosd, dec, 1))

    nvec = sgn * np.array([nx - x, ny - y])        # direction of north on the image
    evec = np.array([ex - x, ey - y])              # direction of east on the image
    angle = float(np.degrees(np.arctan2(nvec[0], nvec[1])))
    mirrored = bool(nvec[0] * evec[1] - nvec[1] * evec[0] < 0)
    return angle, mirrored


def solve_first_image(first, sep_thresh, pixscale, api_key):
    """Plate-solve the first frame; return the angle putting north up, or None."""
    data, (x, y) = first
    height, width = data.shape

    try:
        xs, ys = _sources_for_astrometry(data, sep_thresh)
    except Exception as exc:
        print(f"  astrometry: source extraction failed ({exc})", file=sys.stderr)
        return None
    if len(xs) < 8:
        print(f"  astrometry: only {len(xs)} sources detected; too few to solve",
              file=sys.stderr)
        return None

    try:
        if shutil.which("solve-field"):
            print(f"  astrometry: solving locally with {len(xs)} sources...")
            wcs = _solve_local(xs, ys, width, height, pixscale)
        elif api_key:
            print(f"  astrometry: solving on nova.astrometry.net with {len(xs)} "
                  "sources (this can take a few minutes)...")
            wcs = _solve_web(xs, ys, width, height, pixscale, api_key)
        else:
            print("  astrometry: no local `solve-field` and no API key "
                  "(--astrometry-key or $ASTROMETRY_API_KEY)", file=sys.stderr)
            return None
    except Exception as exc:
        print(f"  astrometry: solve failed ({exc})", file=sys.stderr)
        return None

    if wcs is None:
        print("  astrometry: no solution found", file=sys.stderr)
        return None

    angle, mirrored = north_angle_from_wcs(wcs, x + 1.0, y + 1.0)
    if mirrored:
        print("  astrometry: WARNING image appears mirrored (east is right when "
              "north is up); rotation alone cannot fix this", file=sys.stderr)
    return angle


class RotationAligner:
    """Pick a rotation angle for one cutout.

    With ref=None just the image is shown (used for the first frame, to get
    north up). With a ref, the reference (cyan) and the current frame (red)
    are blended; where they coincide the result is neutral grey.
    """

    def __init__(self, norm, title, ref=None, start_angle=0.0):
        self.norm = norm
        self.ref = None if ref is None else np.nan_to_num(ref)
        self.title = title
        self.blend = 0.5
        self.angle = ((start_angle + 180.0) % 360.0) - 180.0
        self.confirmed = False
        self.skipped = False

        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        self.fig.canvas.manager.set_window_title(title)
        plt.subplots_adjust(bottom=0.2 if self.ref is not None else 0.15)

        if self.ref is None:
            self.im = self.ax.imshow(
                self._render(), origin="lower", cmap="gray", vmin=0, vmax=1)
        else:
            self.im = self.ax.imshow(self._render(), origin="lower")

        h, w = norm.shape
        self.ax.add_patch(Circle(
            ((w - 1) / 2, (h - 1) / 2), radius=15, fill=False,
            edgecolor="yellow", lw=0.8, alpha=0.7))

        y_angle = 0.10 if self.ref is not None else 0.06
        ax_angle = plt.axes([0.2, y_angle, 0.6, 0.03])
        self.s_angle = Slider(ax_angle, "angle (deg)", -180, 180, valinit=self.angle)
        self.s_angle.on_changed(self._on_angle)

        if self.ref is not None:
            ax_blend = plt.axes([0.2, 0.05, 0.6, 0.03])
            self.s_blend = Slider(ax_blend, "blend", 0.0, 1.0, valinit=self.blend)
            self.s_blend.on_changed(self._on_blend)

        self._refresh_title()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _render(self):
        cur = np.nan_to_num(rotate_nan(self.norm, self.angle))
        if self.ref is None:
            return cur
        a = self.blend
        return np.dstack([a * cur, (1 - a) * self.ref, (1 - a) * self.ref]).clip(0, 1)

    def _refresh_title(self):
        skip = "  |  's' skip" if self.ref is not None else ""
        self.ax.set_title(
            f"{self.title}   angle = {self.angle:+.2f}°\n"
            f"←/→ ±1°   ↑/↓ ±0.1°   |  Enter confirm{skip}"
        )

    def _redraw(self):
        self.im.set_data(self._render())
        self._refresh_title()
        self.fig.canvas.draw_idle()

    def _on_angle(self, val):
        self.angle = val
        self._redraw()

    def _on_blend(self, val):
        self.blend = val
        self._redraw()

    def _on_key(self, event):
        steps = {"left": -1.0, "right": 1.0, "down": -0.1, "up": 0.1}
        if event.key in steps:
            new = float(np.clip(self.s_angle.val + steps[event.key], -180, 180))
            self.s_angle.set_val(new)   # triggers _on_angle
        elif event.key == "enter":
            self.confirmed = True
            plt.close(self.fig)
        elif event.key == "s" and self.ref is not None:
            self.skipped = True
            plt.close(self.fig)

    def run(self):
        plt.show()
        if self.skipped or not self.confirmed:
            return None
        return self.angle


def rotate_cutouts(names, cutouts, size, first=None, use_astrometry=True,
                   sep_thresh=3.0, pixscale=None, api_key=None):
    """Rotate every cutout into alignment, one frame at a time.

    The first frame is rotated north-up automatically using astrometry
    (falling back to a manual window if that fails). Each later frame n is
    shown over the already-rotated frame n-1 and rotated by hand.

    Returns a list of rotated cutouts cropped to size x size.
    """
    norms = [normalize(c) for c in cutouts]

    angle0 = None
    if use_astrometry and first is not None:
        angle0 = solve_first_image(first, sep_thresh, pixscale, api_key)
    if angle0 is not None:
        print(f"  {names[0]}: astrometry rotation {angle0:+.2f}° (north up)")
    else:
        angle0 = RotationAligner(
            norms[0], f"{names[0]} — set north up by hand"
        ).run()
        if angle0 is None:
            print("first image was not confirmed; cannot align the rest.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"  {names[0]}: manual rotation {angle0:+.2f}°")

    prev_name, prev_angle = names[0], angle0
    prev_ref = rotate_nan(norms[0], angle0)
    rotated = [rotate_nan(cutouts[0], angle0)]

    for name, cut, norm in zip(names[1:], cutouts[1:], norms[1:]):
        angle = RotationAligner(
            norm, f"{name} (red) over {prev_name} (cyan)",
            ref=prev_ref, start_angle=prev_angle,
        ).run()
        if angle is None:
            print(f"  skipped {name} in rotation step")
            continue
        print(f"  {name}: rotated {angle:+.2f}°")
        rotated.append(rotate_nan(cut, angle))
        # the next frame is compared against this one
        prev_name, prev_angle = name, angle
        prev_ref = rotate_nan(norm, angle)

    return [center_crop(r, size) for r in rotated]


# --------------------------------------------------------------------------
# stacking / stretch
# --------------------------------------------------------------------------

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
                    "directory of FITS frames, optionally rotate the frames "
                    "into alignment, stack aligned cutouts, and export a "
                    "stretch-adjusted greyscale image."
    )
    parser.add_argument("input_dir", help="directory containing FITS images")
    parser.add_argument("output_file_path", help="path to write the final image (e.g. output.png)")
    parser.add_argument(
        "-s", "--size", type=int, default=335,
        help="side length in pixels of the final square cutout (default: 335)",
    )
    parser.add_argument(
        "-r", "--rotate", action="store_true",
        help="after selecting the object in every frame, interactively rotate "
             "each frame into alignment with the first before stacking",
    )
    parser.add_argument(
        "--rot-size", type=int, default=1000,
        help="side length in pixels of the cutouts shown during the rotation "
             "step (default: 1000)",
    )
    parser.add_argument(
        "--no-sep", action="store_true",
        help="use the raw cursor position instead of snapping to the nearest "
             "SEP-detected source",
    )
    parser.add_argument(
        "--sep-thresh", type=float, default=3.0,
        help="SEP detection threshold in units of the background RMS "
             "(default: 3.0)",
    )
    parser.add_argument(
        "--pixscale", type=float, default=None,
        help="approximate pixel scale in arcsec/pixel; narrows the "
             "astrometry.net search to +/-20%% of this (strongly recommended)",
    )
    parser.add_argument(
        "--astrometry-key", default=os.environ.get("ASTROMETRY_API_KEY"),
        help="nova.astrometry.net API key, used only if there is no local "
             "`solve-field` (default: $ASTROMETRY_API_KEY)",
    )
    parser.add_argument(
        "--no-astrometry", action="store_true",
        help="with --rotate, skip astrometry and rotate the first image by hand",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.rotate and args.rot_size < args.size:
        print(f"--rot-size ({args.rot_size}) must be >= --size ({args.size})",
              file=sys.stderr)
        sys.exit(1)

    try:
        files = find_fits_files(args.input_dir)
    except NotADirectoryError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"no FITS files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"found {len(files)} FITS files in {args.input_dir}")

    use_sep = not args.no_sep
    use_astrometry = args.rotate and not args.no_astrometry
    if (use_sep or use_astrometry) and sep is None:
        print("the 'sep' package is not installed (pip install sep); it is "
              "needed for centroiding and for astrometry source extraction. "
              "Install it, or rerun with --no-sep (and --no-astrometry if "
              "rotating)", file=sys.stderr)
        sys.exit(1)

    cut_size = args.rot_size if args.rotate else args.size
    names, cutouts, first = collect_cutouts(
        files, cut_size, use_sep=use_sep, sep_thresh=args.sep_thresh)

    if not cutouts:
        print("no frames were selected; nothing to stack.", file=sys.stderr)
        sys.exit(1)

    if args.rotate:
        print(f"rotation step: {len(cutouts)} frame(s), {args.rot_size}x{args.rot_size} px")
        cutouts = rotate_cutouts(
            names, cutouts, args.size, first=first,
            use_astrometry=use_astrometry, sep_thresh=args.sep_thresh,
            pixscale=args.pixscale, api_key=args.astrometry_key,
        )

    print(f"stacking {len(cutouts)} cutout(s) into a {args.size}x{args.size} image...")
    stacked = stack_cutouts(cutouts)

    StretchAdjuster(stacked, args.output_file_path).run()


if __name__ == "__main__":
    main()