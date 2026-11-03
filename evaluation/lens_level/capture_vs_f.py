# -*- coding: utf-8 -*-
r"""
capture_vs_f.py -- 8 x 8 um PD capture vs focal length for the L3 quadrant lens,
including the N=8 Lambda/3 site staircase and an LPA breakdown power-law overlay.

Reuses the exact validated Grid/cap machinery from review_lens_physics.py and
review_net_f.py (copied, not imported -- same convention those two scripts use).
Writes only into this review folder. Never touches <DATA_ROOT>\Simulation.

Scenarios computed for f = 136, 150, 170, 200, 225, 250 um:
  (1) cap1  -- ideal continuous hyperbolic phase, standard grid (dx=0.5, pad=4).
               Must reproduce review_net_f.log cap_B0 at f=136/170/200/250.
  (2) cap2  -- (1) replaced by the N=8 Lambda/3 site staircase, built exactly as
               review_lens_physics.py section F8 ("site_N8"): site-binned phase,
               then rounded to 8 levels. Must reproduce -0.289 dB at f=136
               (loss relative to the continuous phase evaluated on the SAME fine
               grid -- F8's "cont", i.e. site_N8_loss_dB).
  (3) cap3  -- (2) with an LPA breakdown amplitude overlay
               B(P) = B0 * (2*Lambda/P)^k dB, applied as 10^(B/20) in amplitude,
               using the continuous local period P(rho,f) at every fine-grid
               sample (not the site-snapped period) -- the formula is evaluated
               everywhere with NO floor/clip at P = 2*Lambda, i.e. it runs down
               to whatever P_min the aperture presents at that f.
  (4) power fraction of the input (Gaussian-weighted) below P = 2,3,4,5 Lambda,
      on the standard grid, exactly as review_lens_physics.py's f_scan (F(f)).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import json
import time

OUT = r"<WORK_ROOT>\wf_lpa\lens\review"
LAM = 1.55
LAMG = 1.31539
W = 100.0
APER = 150.0
XC = W / np.sqrt(2.0 * np.pi)
YC = XC
K0 = 2.0 * np.pi / LAM
T0 = time.time()


def log(*a):
    print(*a, "t=%.1f" % (time.time() - T0), flush=True)


class Grid:
    def __init__(self, dx, pad=4):
        self.dx = dx
        self.n = int(round(APER / dx))
        self.x = (np.arange(self.n) + 0.5) * dx
        self.X, self.Y = np.meshgrid(self.x, self.x)
        self.A = np.exp(-(self.X ** 2 + self.Y ** 2) / W ** 2)
        self.pin = float(np.sum(self.A ** 2) * dx * dx)
        self.rho = np.hypot(self.X - XC, self.Y - YC)
        self.pad = pad
        self.npd = self.n * pad
        k = 2.0 * np.pi * np.fft.fftfreq(self.npd, d=dx)
        self.k = k
        self._H = {}

    def P(self, f):
        return LAM * np.sqrt(self.rho ** 2 + f ** 2) / np.maximum(self.rho, 1e-9)

    def phi(self, f):
        return -K0 * (np.sqrt(self.rho ** 2 + f ** 2) - f)

    def H(self, f):
        if f not in self._H:
            KX, KY = np.meshgrid(self.k, self.k)
            kz2 = K0 ** 2 - KX ** 2 - KY ** 2
            del KX, KY
            self._H[f] = np.where(kz2 >= 0, np.exp(1j * np.sqrt(np.clip(kz2, 0, None)) * f), 0.0)
        return self._H[f]

    def field_at(self, E0, f, xt, yt):
        Ep = np.zeros((self.npd, self.npd), dtype=complex)
        Ep[:self.n, :self.n] = E0
        S = np.fft.fft2(Ep) * self.H(f)
        x_first = self.dx / 2.0
        Mx = np.exp(1j * np.outer(xt - x_first, self.k))
        My = np.exp(1j * np.outer(yt - x_first, self.k))
        return (My @ S @ Mx.T) / (self.npd * self.npd)


DT = 0.1
AX = DT * (np.arange(80) + 0.5) - 4.0     # exact 8 um box, 80 cells


def pd_field(g, E0, f, cx=XC, cy=YC):
    return g.field_at(E0, f, cx + AX, cy + AX)


def cap(g, E0, f, cx=XC, cy=YC):
    E = pd_field(g, E0, f, cx, cy)
    return float(np.sum(np.abs(E) ** 2) * DT * DT) / g.pin


def db(a, b):
    return float(10.0 * np.log10(a / b))


def lev(p):
    return (np.round(np.mod(p, 2 * np.pi) * 8 / (2 * np.pi)) % 8) * (2 * np.pi / 8)


F_LIST = [136.0, 150.0, 170.0, 200.0, 225.0, 250.0]
B0K_LIST = [(-2.25, 1), (-2.25, 2), (-3.57, 1), (-3.57, 2)]
L_LIST = [2, 3, 4, 5]

# reference values from review_net_f.log (cap_B0), for the reproduction check
CHECK_CAP_B0 = {136.0: 0.8994, 170.0: 0.8844, 200.0: 0.8676, 250.0: 0.8325}
CHECK_SITE_N8_DB_AT_136 = -0.289

g_std = Grid(0.5, 4)
site = LAMG / 3.0
g_fine = Grid(site / 2.0, 4)
log("grids built: std n=%d npd=%d | fine n=%d npd=%d" % (g_std.n, g_std.npd, g_fine.n, g_fine.npd))

# site-center geometry on the fine grid (fixed across f)
ix = np.floor(g_fine.X / site).astype(np.int64)
iy = np.floor(g_fine.Y / site).astype(np.int64)
xs = (ix + 0.5) * site
ys = (iy + 0.5) * site
rs = np.hypot(xs - XC, ys - YC)

rows = []
for f in F_LIST:
    row = {"f": f}

    # ---- (1) ideal continuous phase, standard grid ----
    ph1 = g_std.phi(f)
    cap1 = cap(g_std, g_std.A * np.exp(1j * ph1), f)
    row["cap1_ideal"] = cap1
    g_std._H.pop(f, None)

    # ---- power fraction below N*Lambda (standard grid, F8 f_scan convention) ----
    Pstd = g_std.P(f)
    w2 = g_std.A ** 2
    w2sum = float(w2.sum())
    row["pow_below"] = {str(L): float(w2[Pstd < L * LAMG].sum() / w2sum) for L in L_LIST}

    # ---- (2) N=8 Lambda/3 site staircase, exactly as review_lens_physics.py F8 ----
    ph_fine = g_fine.phi(f)                                   # continuous phase, fine grid ("cont")
    ph_site = -K0 * (np.sqrt(rs ** 2 + f ** 2) - f)            # phase at site centre, pre-quantisation
    cont_fine = cap(g_fine, g_fine.A * np.exp(1j * ph_fine), f)
    cap2 = cap(g_fine, g_fine.A * np.exp(1j * lev(ph_site)), f)
    loss2_vs_cont_dB = db(cap2, cont_fine)
    row["cap2_siteN8"] = cap2
    row["cont_fine_check"] = cont_fine
    row["loss2_siteN8_vs_cont_dB"] = loss2_vs_cont_dB

    # ---- (3) (2) + LPA breakdown power-law overlay, no floor at P=2*Lambda ----
    Pfine = g_fine.P(f)   # continuous local period at every fine-grid sample
    cap3 = {}
    for B0, kk in B0K_LIST:
        Bd = B0 * (2.0 * LAMG / Pfine) ** kk
        amp3 = g_fine.A * (10.0 ** (Bd / 20.0))
        c3 = cap(g_fine, amp3 * np.exp(1j * lev(ph_site)), f)
        key = "B%g_k%d" % (B0, kk)
        cap3[key] = dict(cap=c3,
                          loss_vs_cap1_dB=db(c3, cap1),
                          loss_vs_cap2_dB=db(c3, cap2))
    row["cap3"] = cap3
    g_fine._H.clear()

    rows.append(row)
    log("f=%g cap1=%.4f cap2=%.4f loss2=%.4f dB cap3=%s" %
        (f, cap1, cap2, loss2_vs_cont_dB,
         {k: round(v["cap"], 4) for k, v in cap3.items()}))

# ---- reproduction checks ----
checks = {"cap1_vs_review_net_f": {}, "siteN8_at_136": None}
for f, expect in CHECK_CAP_B0.items():
    got = next(r["cap1_ideal"] for r in rows if r["f"] == f)
    checks["cap1_vs_review_net_f"][str(f)] = dict(expected=expect, got=round(got, 4),
                                                    diff=round(got - expect, 6))
row136 = next(r for r in rows if r["f"] == 136.0)
checks["siteN8_at_136"] = dict(expected_dB=CHECK_SITE_N8_DB_AT_136,
                                got_dB=round(row136["loss2_siteN8_vs_cont_dB"], 5),
                                diff_dB=round(row136["loss2_siteN8_vs_cont_dB"] - CHECK_SITE_N8_DB_AT_136, 5))

# ---- argmax f for each (B0,k), over the sampled F_LIST ----
argmax = {}
for B0, kk in B0K_LIST:
    key = "B%g_k%d" % (B0, kk)
    best = max(rows, key=lambda r: r["cap3"][key]["cap"])
    argmax[key] = dict(f=best["f"], cap=best["cap3"][key]["cap"],
                        loss_vs_cap1_dB=best["cap3"][key]["loss_vs_cap1_dB"])

elapsed = time.time() - T0
out = dict(rows=rows, checks=checks, argmax=argmax, elapsed_s=elapsed)
with open(OUT + r"\capture_vs_f.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)

log("CHECKS", json.dumps(checks))
log("ARGMAX", json.dumps(argmax))
log("done, elapsed=%.1f s" % elapsed)
