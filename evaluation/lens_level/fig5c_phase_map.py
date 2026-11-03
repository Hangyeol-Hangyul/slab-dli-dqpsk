"""Fig. 5(c) "Lens zoning" phase map, redrawn from the analytic 8-level lens phase of the lens model.

Why this file exists (audit 2026-10-15, row Fig. 5(c)): the 1400 x 1400 PNG in the deck (slide 5, picture "138", md5 aeb94c3a8947997c29cb6c1a13f80ba8,
embedded in Downloads\\Fig5c_LPA_metalens_test_r2.svg) was made in a design tool and has no generating script on disk. This script redraws it from the
lens model and the design numbers; fig5c_compare_with_deck.py compares the result with the deck picture pixel by pixel.

Model (no simulation; the same lens as facts_v3 H1 / lens_total_li.py):
    psi(rho)  = (2 pi / lam) * (sqrt(rho^2 + f^2) - f)        accumulated lens phase delay, deg   (lam = 1.55 um, f = 200 um)
    rho       = distance from the lens centre (XC, XC) um, measured from the beam centre at the quadrant corner (0, 0);
                XC = 39.894 um as drawn in the picture (design value 100 / sqrt(2 pi) = 39.894228 um, rounded to 3 decimals)
    level     = floor(psi / 45 deg) mod 8                      8 phase levels (N = 8); the centre disc is level 0 (psi in [0, 45) deg)
    colour    = LEVEL_RGB[level]                               the 8 colours of the picture (= the 8 bands of the colourbar in the SVG, +-1/255)
Frame: one quadrant, x and y from 0 to 150 um (APER_um), 1400 x 1400 pixels, pixel centres; native orientation = the PNG as embedded
(lens centre near the LOWER-LEFT, row 0 = y 150 um). The deck shows the picture with flipH + flipV (author's intent, 2026-10-15): --orientation
published rotates the native image by 180 deg, which is what slide 5 displays (lens centre near the UPPER-RIGHT, at about (110, 110) um on the axes).

Conventions that are NOT given by the formula and were fixed from the picture: (i) level boundaries sit at psi = k * 45 deg (floor, not round);
(ii) the 8 colours; (iii) the frame (0-150 um, beam centre at the corner); (iv) XC = 39.894 um (3-decimal rounding of the design 39.894228 um; with the
unrounded value 834 of 1,960,000 pixels flip to the neighbouring level, every one within 0.003 pixel of a level boundary; with 39.894 none does).
fig5c_compare_with_deck.py runs controls that break each of them.

--zones prints the zone-ring radii of the lens model (local period P(rho) = lam * sqrt(rho^2 + f^2) / (rho * Lambda), P = 12, 6, 4, 3, 2 Lambda)
and the zone power shares integrated over the Gaussian quadrant, next to ZONE_W of lens_total_li.py.

Usage (outside the cluster; no solver, no RSoft, small memory):
    python -I -B fig5c_phase_map.py [--out PATH] [--orientation native|published] [--zones]
Output default: fig5c_phase_map_regen_native.png (or ..._published.png) beside this script.
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
XC_UM = 100.0 / np.sqrt(2.0 * np.pi)          # 39.894228040143275 (design value)
XC_DRAW_UM = 39.894                            # the value the deck picture was drawn with (reproduces it bit for bit)
APER_UM = 150.0                                # quadrant side
LAMG_UM = 1.31539                              # bottom metagrating period Lambda
W_UM = 100.0                                   # Gaussian beam radius (1/e^2 intensity), centred on the quadrant corner
NPIX = 1400
NLEV = 8

# level 0..7 = centre disc outward; exactly the 8 colours of the deck picture (aeb94c3a). The SVG colourbar (top band = level 0) has the same 8 colours
# except three channels that differ by 1/255: (231,151,77) vs (230,150,76), (53,143,195) vs (52,142,195), (16,45,83) vs (16,44,82).
LEVEL_RGB = np.array([
    (82, 38, 33), (180, 75, 14), (230, 150, 76), (246, 219, 189),
    (191, 218, 234), (52, 142, 195), (0, 92, 158), (16, 44, 82),
], dtype=np.uint8)
SVG_BAR_RGB = np.array([
    (82, 38, 33), (180, 75, 14), (231, 151, 77), (246, 219, 189),
    (191, 218, 234), (53, 143, 195), (0, 92, 158), (16, 45, 83),
], dtype=np.uint8)

# ZONE_W of lens_total_li.py lines 101-102 (percent); ">12L" = 100 - sum
ZONE_W = {"2-3L": 2.666, "3-4L": 7.462, "4-6L": 25.6, "6-12L": 47.9}
ZONE_W[">12L"] = 100.0 - sum(ZONE_W.values())


def design():
    """f, lam, XC, aperture, Lambda from tm_focus_summary.json design block (fall back to the constants above)."""
    try:
        with open(PARAM_JSON, "r", encoding="utf-8") as fh:
            d = json.load(fh)["design"]
        return (float(d["f_um"]), float(d["lam_um"]), float(d["XC_um"]), float(d["APER_um"]), float(d["LAMG_um"]), "json")
    except Exception:
        return F_UM, LAM_UM, float(XC_UM), APER_UM, LAMG_UM, "constants"


def psi_deg(f_um=None, lam_um=None, xc_um=None, aper_um=None, npix=NPIX, offset_px=0.5):
    """Lens phase delay psi (deg, unwrapped) on the npix x npix grid; row 0 = y aper_um, column 0 = x 0. offset_px 0.5 = pixel centres."""
    d_f, d_lam, _, d_ap, _, _ = design()
    f_um = d_f if f_um is None else f_um
    lam_um = d_lam if lam_um is None else lam_um
    xc_um = XC_DRAW_UM if xc_um is None else xc_um
    aper_um = d_ap if aper_um is None else aper_um
    s = aper_um / npix
    x = (np.arange(npix) + offset_px) * s                                   # um from the left edge
    y = aper_um - (np.arange(npix) + offset_px) * s                          # um from the bottom edge; row 0 = top
    rho = np.hypot(x[None, :] - xc_um, y[:, None] - xc_um)
    return (360.0 / lam_um) * (np.sqrt(rho * rho + f_um * f_um) - f_um)


def level_map(psi, phase_offset_deg=0.0):
    """Level index 0..7: floor((psi + offset) / 45 deg) mod 8."""
    return np.floor((psi + phase_offset_deg) / (360.0 / NLEV)).astype(np.int64) % NLEV


def render(orientation="native", **kw):
    """uint8 RGB image. 'native' = as embedded in the SVG/deck; 'published' = flipH + flipV (rotated by 180 deg), as slide 5 shows it."""
    off = kw.pop("phase_offset_deg", 0.0)
    img = LEVEL_RGB[level_map(psi_deg(**kw), off)]
    if orientation == "published":
        img = img[::-1, ::-1].copy()
    elif orientation != "native":
        raise ValueError(orientation)
    return img


def zone_table():
    """Zone ring radii (um) of the lens model and zone power shares (%) of the Gaussian quadrant."""
    f_um, lam_um, xc_um, aper_um, lamg_um, _ = design()

    def rho_of(p_over_lamg):                       # P(rho) = lam sqrt(rho^2 + f^2) / rho   ->   rho = f s / sqrt(1 - s^2), s = lam / P
        sn = lam_um / (p_over_lamg * lamg_um)
        return f_um * sn / np.sqrt(1.0 - sn * sn)

    pl = [12.0, 6.0, 4.0, 3.0, 2.0]
    radii = {("%gL" % p): float(rho_of(p)) for p in pl}
    n = 3000
    s = aper_um / n
    c = (np.arange(n) + 0.5) * s
    inten = np.exp(-2.0 * ((c[None, :]) ** 2 + (c[:, None]) ** 2) / W_UM ** 2)       # Gaussian beam centred on the corner (0, 0), field exp(-r^2/w^2)
    rho = np.hypot(c[None, :] - xc_um, c[:, None] - xc_um)
    tot = inten.sum()
    edges = [0.0] + [radii["%gL" % p] for p in pl] + [1e9]
    names = [">12L", "6-12L", "4-6L", "3-4L", "2-3L", "<2L"]
    shares = {}
    for k, nm in enumerate(names):
        m = (rho >= edges[k]) & (rho < edges[k + 1])
        shares[nm] = float(100.0 * inten[m].sum() / tot)
    return radii, shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--orientation", default="native", choices=["native", "published"])
    ap.add_argument("--zones", action="store_true", help="print zone ring radii and zone power shares (no image written)")
    a = ap.parse_args()
    f_um, lam_um, xc_um, aper_um, lamg_um, src = design()
    if a.zones:
        radii, shares = zone_table()
        print("f_um %.4f lam_um %.4f XC_um %.9f APER_um %.1f LAMG_um %.5f (source: %s)" % (f_um, lam_um, xc_um, aper_um, lamg_um, src))
        print("ring radii um (P = n Lambda):", {k: round(v, 3) for k, v in radii.items()})
        print("zone share % (Gaussian quadrant, 3000 x 3000):", {k: round(v, 3) for k, v in shares.items()})
        print("ZONE_W lens_total_li.py       :", {k: round(v, 3) for k, v in ZONE_W.items()})
        return
    out = a.out or os.path.join(HERE, "fig5c_phase_map_regen_%s.png" % a.orientation)
    img = render(a.orientation)
    Image.fromarray(img, "RGB").save(out)
    print("f_um %.4f lam_um %.4f XC_design_um %.9f XC_drawn_um %.3f (source: %s) orientation %s -> %s %s" % (f_um, lam_um, xc_um, XC_DRAW_UM, src, a.orientation, out, img.shape))


if __name__ == "__main__":
    main()
