# -*- coding: utf-8 -*-
"""Beam maps for Fig 4 (e) ZY / TE and (f) ZX / TM of the fixed 2-D input grating (warm_xsec_min_rf050_W1), plus an exact
re-rasterisation check of the XY-cut SVGs written by in2d_geometry.py.

Method (a RECONSTRUCTION from RCWA order powers, not a full-wave field):
  * order powers = Li-factorised RCWA, nG 1793 (sim752 input2d.json, the gate rung), normal incidence from air, 1550 nm;
  * the two lanes are LANE-AVERAGED (E_x lane and E_y lane order by order): the design is exactly sigma_d-symmetric, and
    liy is not x <-> y invariant (driver Main_rcwa_input2d.py:106-109), so the per-lane difference is factorisation error,
    not physics; both PNGs therefore carry the same numbers (mirror lanes);
  * Gaussian beam, 1/e^2 intensity radius W0 = 100 um (the paper beam; the earlier panel_1_input_grating.py used 50 um); the diffracted beam's
    own waist is W0 cos(theta) = 28.9 um, z_R about 2.44 mm in SiO2 >> 0.375 mm path inside the frame (width change ~1 %);
  * every order keeps the incident x-footprint, so the plotted quantity -- z-directed power flux density normalised to the
    incident peak -- is eta_m * exp(-2 (x - x_m(z))^2 / W0^2) along the ray x_m(z) = |z| tan(theta_m) * sign(m);
  * orders are summed INCOHERENTLY (no inter-order fringes; same choice as panel_1_input_grating.py);
  * order coefficients are taken angle-independent across the beam's +-0.57 deg angular spread.
Raster frame = the existing Fig 4 (f) picture frame, measured from the PPTX: X (or Y) -258.6 .. 305.6 um, Z -230.5 .. 102.1 um,
1520 x 896 px. Colour: white -> #D55E00 (TM) / #0072B2 (TE), linear in the plotted power, 0 .. 1 (as the existing colourbars).
Outputs go to the upload folder; numbers to in2d_beams_out.txt."""
import io
import json
import os
import re

import numpy as np
from PIL import Image

HERE = r"<WORK_ROOT>\paper_scdc_261011_v3\fig4_beams_w100"
CELL = r"<DATA_ROOT>\Result\261008_scdc_in2dli\261008_sim752_in2dLi_warm_xsec_min_rf050_W1"
UP = r"<PAPER_ROOT>\Figure\261012_Figure_4_beams_w100"
J = json.load(io.open(os.path.join(CELL, "input2d.json"), encoding="utf-8"))
NG = 1793
LAM, P, N_SUB, W0 = 1.55, J["period_nm"] * 1e-3, float(J["key_n_sub"]), 100.0
X0, X1, Z0, Z1, NX, NZ = -258.6, 305.6, -230.5, 102.1, 1520, 896
out = []


def g(k):
    return float(J["ng%d_%s" % (NG, k)])


def lane(pol):
    """In-plane orders of one launch. pol 'Ex' -> x-z plane (orders (m,0)), 'Ey' -> y-z plane (orders (0,m))."""
    pre = "air_%s_" % pol
    if pol == "Ex":
        o = {+1: "+1+0", -1: "-1+0"}
        oop = ("+0+1", "+0-1")
    else:
        o = {+1: "+0+1", -1: "+0-1"}
        oop = ("+1+0", "-1+0")
    d = {}
    for m, key in o.items():
        d[m] = dict(p=g(pre + "T_o%s_p" % key), s=g(pre + "T_o%s_s" % key))
    d[0] = dict(tot=g(pre + "T_o+0+0_Ex") + g(pre + "T_o+0+0_Ey"))
    d["R"] = g(pre + "R_o+0+0_Ex") + g(pre + "R_o+0+0_Ey")
    d["oop"] = sum(g(pre + "T_o%s_%s" % (k, c)) for k in oop for c in ("p", "s"))
    d["energy"] = g(pre + "energy")
    return d


def beam_map(d):
    xs = X0 + (np.arange(NX) + 0.5) * (X1 - X0) / NX
    zs = Z1 - (np.arange(NZ) + 0.5) * (Z1 - Z0) / NZ          # row 0 = top (z = Z1)
    XX, ZZ = np.meshgrid(xs, zs)
    gauss = lambda xc: np.exp(-2.0 * (XX - xc) ** 2 / W0 ** 2)
    I = np.zeros_like(XX)
    air = ZZ >= 0
    I[air] = (1.0 + d["R"]) * gauss(0.0)[air]                 # incident + specular reflection (incoherent)
    sub = ~air
    th = np.arcsin(LAM / (N_SUB * P))
    for m in (+1, -1):
        eta = d[m]["p"] + d[m]["s"]
        I[sub] += eta * gauss(np.sign(m) * np.abs(ZZ) * np.tan(th))[sub]
    I[sub] += d[0]["tot"] * gauss(0.0)[sub]
    return I, np.degrees(th)


def save_rgb(I, colour, path):
    c = np.array([int(colour[k:k + 2], 16) / 255.0 for k in (1, 3, 5)])
    a = np.clip(I, 0.0, 1.0)[..., None]
    rgb = 1.0 - (1.0 - c) * a
    Image.fromarray(np.round(rgb * 255).astype(np.uint8), "RGB").save(path, dpi=(770, 770))


def save_bar(colour, path, w=32, h=448):
    c = np.array([int(colour[k:k + 2], 16) / 255.0 for k in (1, 3, 5)])
    t = np.linspace(1.0, 0.0, h)[:, None, None]                 # top = 1
    rgb = np.repeat(1.0 - (1.0 - c) * t, w, axis=1)
    img = np.round(rgb * 255).astype(np.uint8)
    img[0, :] = img[-1, :] = 0
    img[:, 0] = img[:, -1] = 0
    Image.fromarray(img, "RGB").save(path)


def lane_avg():
    a, b = lane("Ex"), lane("Ey")
    d = {m: dict(p=0.5 * (a[m]["p"] + b[m]["p"]), s=0.5 * (a[m]["s"] + b[m]["s"])) for m in (+1, -1)}
    d[0] = dict(tot=0.5 * (a[0]["tot"] + b[0]["tot"]))
    for k in ("R", "oop", "energy"):
        d[k] = 0.5 * (a[k] + b[k])
    return d


for pol, colour, name in (("avg", "#D55E00", "beam_ZX_TM_input"), ("avg", "#0072B2", "beam_ZY_TE_input")):
    d = lane_avg()
    I, th = beam_map(d)
    save_rgb(I, colour, os.path.join(UP, name + ".png"))
    out.append("%s (%s): theta %.6f deg | designed (+1) p %.7f s %.3e | (-1) p %.3e s %.3e tot %.4e | (0) T %.4e | R(0,0) %.4e | "
               "out-of-plane %.4e | energy %.12f | map max air %.4f sub %.4f" % (
                   name, pol, th, d[+1]["p"], d[+1]["s"], d[-1]["p"], d[-1]["s"], d[-1]["p"] + d[-1]["s"], d[0]["tot"], d["R"],
                   d["oop"], d["energy"], I[: int(NZ * Z1 / (Z1 - Z0))].max(), I[int(NZ * Z1 / (Z1 - Z0)) + 2:].max()))
    zx = -X1 / np.tan(np.radians(th))
    out.append("   +1 beam centre leaves the right frame edge (%.1f um) at z = %.1f um" % (X1, zx))
save_bar("#D55E00", os.path.join(UP, "colorbar_TM_white_to_D55E00.png"))
save_bar("#0072B2", os.path.join(UP, "colorbar_TE_white_to_0072B2.png"))

# ---- exact re-rasterisation check of the SVGs (even-odd crossing test at pixel centres) ----
N = 256
px = J["period_nm"] / N
for tag in ("L1", "L2"):
    s = io.open(os.path.join(UP, "in2d_%s_xycut_cell.svg" % tag), encoding="utf-8").read()
    dstr = re.search(r'<path d="([^"]+)"', s).group(1)
    loops = []
    for seg in dstr.split("Z"):
        pts = re.findall(r"(-?[0-9.]+),(-?[0-9.]+)", seg)
        if pts:
            loops.append(np.array([[float(a), float(b)] for a, b in pts]))
    cx = (np.arange(N) + 0.5) * px
    inside = np.zeros((N, N), bool)                            # [ix, iy]
    for iy in range(N):
        ysvg = (N - (iy + 0.5)) * px
        cnt = np.zeros(N, int)
        for lp in loops:
            a = lp
            b = np.roll(lp, -1, axis=0)
            m = (a[:, 1] > ysvg) != (b[:, 1] > ysvg)
            if not np.any(m):
                continue
            xi = a[m, 0] + (ysvg - a[m, 1]) * (b[m, 0] - a[m, 0]) / (b[m, 1] - a[m, 1])
            cnt += (xi[None, :] > cx[:, None]).sum(axis=1)
        inside[:, iy] = cnt % 2 == 1
    rho = np.load(os.path.join(CELL, "rho%s.npy" % tag[1])) > 0.5
    out.append("SVG %s re-raster vs rho: mismatching pixels %d of %d" % (tag, int((inside != rho).sum()), N * N))
io.open(os.path.join(HERE, "in2d_beams_out.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("ok", len(out))
