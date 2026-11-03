"""Fig. 3(g): metalens phase map of the TM lens, redrawn from the analytic lens phase.

Why this file exists (audit 2026-10-15): the 1500 x 1500 PNG inside the deck (slide 3, md5 a5ef136aee93cc1fe61687af53f0a491)
has no generating script on disk. This script redraws it from the formula and the design numbers.

Model (no simulation):
    phi(rho) = -(2 pi / lam) * (sqrt(rho^2 + f^2) - f)            wrapped to [-180, +180) deg
    rho      = distance from the centre of the quadrant that contains the pixel
    centres  = (+-XC, +-XC) um,  XC = 100 / sqrt(2 pi) = 39.894228 um   (the lens centre of each quadrant)
    f = 200 um, lam = 1.55 um          (design block of tm_focus_summary.json)
Frame: x and y from -100 to +100 um, 1500 x 1500 pixels, pixel centres (7.5 px per um), row 0 = y +100 um.
The deck picture is a CONTINUOUS wrapped phase (it is not quantised to 8 levels); --levels N quantises, for comparison only.
Colour map: the cyclic map of the deck colourbar picture (slide 3, md5 f1b0d13e4b724a0a7f7f39db11a38d22: top +180 deg,
0 deg = #F7F7F7, +90 deg ~ #D65F07, -90 deg = #0072B4, +-180 deg = #1A1C30). It is stored below as 121 nodes (every 3 deg,
sampled from that colourbar); linear interpolation between nodes reproduces the colourbar to <= 1.1 / 255.

Usage (outside the cluster; no solver, no RSoft, small memory):
    python -I -B fig3g_phase_map.py [--out PATH] [--levels N]
Output default: fig3g_phase_map_regen.png beside this script.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM_JSON = r"<WORK_ROOT>\tm_focus_261011\tm_focus_summary.json"   # key "design"

# fall-back constants if the json is not reachable (identical to its design block)
F_UM = 200.0
LAM_UM = 1.55
XC_UM = 100.0 / np.sqrt(2.0 * np.pi)          # 39.894228040143275
HALF_UM = 100.0                                 # half width of the drawn frame
NPIX = 1500

# phase (deg) -> RGB, nodes every 3 deg from -180 to +180 (121 nodes), sampled from the deck colourbar
_LUT = [
    (26,28,48), (25,30,53), (23,32,57), (22,34,62), (20,37,66), (19,39,71),
    (18,41,76), (16,43,80), (15,46,85), (13,48,90), (12,50,94), (11,53,99),
    (10,57,106), (8,61,111), (7,64,116), (6,68,122), (5,71,128), (4,74,133),
    (2,78,138), (1,82,145), (0,85,150), (0,88,153), (0,91,156), (0,94,159),
    (0,96,162), (0,100,165), (0,102,168), (0,105,171), (0,108,174), (0,111,177),
    (0,114,180), (7,118,182), (14,122,184), (21,126,186), (28,129,188), (35,133,190),
    (42,137,192), (49,141,194), (56,144,196), (63,148,198), (70,152,200), (80,157,203),
    (90,164,206), (101,169,209), (111,174,212), (121,180,215), (131,186,218), (141,191,221),
    (152,196,224), (162,203,227), (172,208,230), (180,212,232), (187,216,233), (194,220,235),
    (202,224,237), (210,228,239), (217,231,240), (224,236,242), (232,239,244), (239,243,245),
    (247,247,247), (247,243,239), (247,240,232), (246,236,224), (246,232,216), (246,229,209),
    (246,225,200), (246,221,193), (245,217,186), (245,214,178), (245,210,170), (244,206,163),
    (243,202,156), (242,198,149), (241,194,142), (240,190,135), (240,185,128), (239,181,121),
    (238,177,114), (237,173,107), (236,169,100), (234,162,90), (232,154,81), (229,147,72),
    (227,140,62), (225,132,53), (223,125,44), (220,118,34), (218,110,25), (216,103,16),
    (214,95,6), (209,92,7), (205,90,8), (200,87,9), (196,84,10), (191,82,11),
    (186,79,12), (182,76,13), (177,73,14), (172,71,15), (168,68,16), (161,65,17),
    (154,62,18), (148,60,20), (141,57,21), (134,54,22), (128,52,23), (121,49,24),
    (114,46,26), (108,44,27), (101,41,28), (94,40,30), (86,38,32), (79,37,34),
    (71,36,36), (63,34,38), (56,33,40), (48,32,42), (41,31,44), (34,29,46),
    (26,28,48),
]
LUT_PHASE = np.arange(-180.0, 180.0 + 1e-9, 3.0)
LUT_RGB = np.asarray(_LUT, dtype=float)
assert LUT_RGB.shape == (121, 3) and LUT_PHASE.size == 121


def design():
    """f, lam, XC from tm_focus_summary.json design block (fall back to the constants above)."""
    try:
        with open(PARAM_JSON, "r", encoding="utf-8") as fh:
            d = json.load(fh)["design"]
        return float(d["f_um"]), float(d["lam_um"]), float(d["XC_um"]), "json"
    except Exception:
        return F_UM, LAM_UM, float(XC_UM), "constants"


def phase_deg(f_um=None, lam_um=None, xc_um=None, half_um=HALF_UM, npix=NPIX, levels=0):
    """Wrapped lens phase (deg, [-180, 180)) on the npix x npix pixel-centre grid; row 0 = +half_um."""
    d_f, d_lam, d_xc, _ = design()
    f_um = d_f if f_um is None else f_um
    lam_um = d_lam if lam_um is None else lam_um
    xc_um = d_xc if xc_um is None else xc_um
    c = (np.arange(npix) + 0.5) * (2.0 * half_um / npix) - half_um          # pixel centres, um
    x = c[None, :] * np.ones((npix, 1))
    y = -c[:, None] * np.ones((1, npix))                                     # row 0 = +half_um
    rho = np.hypot(np.abs(x) - xc_um, np.abs(y) - xc_um)                     # distance to the centre of the own quadrant
    ph = -(360.0 / lam_um) * (np.sqrt(rho * rho + f_um * f_um) - f_um)
    if levels and levels > 1:
        step = 360.0 / levels
        ph = np.rint(ph / step) * step
    return (ph + 180.0) % 360.0 - 180.0


def colour(ph):
    """Phase (deg) -> uint8 RGB through the 121-node cyclic map."""
    rgb = np.stack([np.interp(ph, LUT_PHASE, LUT_RGB[:, k]) for k in range(3)], axis=-1)
    return np.rint(rgb).astype(np.uint8)


def render(levels=0):
    return colour(phase_deg(levels=levels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "fig3g_phase_map_regen.png"))
    ap.add_argument("--levels", type=int, default=0, help="0 = continuous (the deck picture); N = quantise to N levels")
    a = ap.parse_args()
    f_um, lam_um, xc_um, src = design()
    img = render(a.levels)
    Image.fromarray(img, "RGB").save(a.out)
    print("f_um %.4f lam_um %.4f XC_um %.9f (source: %s) levels %d -> %s %s" % (f_um, lam_um, xc_um, src, a.levels, a.out, img.shape))


if __name__ == "__main__":
    main()
