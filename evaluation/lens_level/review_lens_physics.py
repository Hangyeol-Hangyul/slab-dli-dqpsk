# -*- coding: utf-8 -*-
r"""
review_lens_physics.py -- physics checks the evaluator did not make.
All captures use an EXACT 8 x 8 um box at the design focus (PD fixed), via own ASM +
separable-DFT zoom (validated against a Rayleigh-Sommerfeld integral in
review_lens_check.py to 1e-5).
 (a) random per-zone phase error restricted to P < P0 (the evaluator applied it to ALL zones)
 (b) systematic (coherent) phase error in P < P0: step and power law
 (c) two-arm coherent overlap with an arm-B-only phase error in P < P0
 (d) ghost transmitted orders n = 0, -1, +2 from P < P0: own-PD and neighbour-PD transfer
 (e) extrapolation of B(P) beyond the largest measured P (5 Lambda)
 (f) f scan of baseline capture and power below 5 Lambda / 24.8 um
 (g) lens-level site quantisation (N = 8 on the Lambda/3 site grid) vs the 0.224 dB line
Writes only into this review folder.
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
R = {}


def log(*a):
    print(*a, "t=%.0f" % (time.time() - T0), flush=True)


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


def zones(g, f):
    per = (np.sqrt(g.rho ** 2 + f ** 2) - f) / LAM
    rz = np.floor(per).astype(np.int64)
    th = np.arctan2(g.Y - YC, g.X - XC)
    zid = np.zeros(g.X.shape, dtype=np.int64)
    for r in np.unique(rz):
        m = rz == r
        rn = float(np.median(np.maximum(g.rho[m], 1e-9)))
        Pn = LAM * np.sqrt(rn ** 2 + f ** 2) / rn
        ns = max(1, int(round(2 * np.pi * rn / Pn)))
        az = np.clip(np.floor((th[m] + np.pi) / (2 * np.pi) * ns).astype(np.int64), 0, ns - 1)
        zid[m] = r * 100000 + az
    return zid


g = Grid(0.5, 4)
F = 136.0
P136 = g.P(F)
PHI136 = g.phi(F)
E_base = g.A * np.exp(1j * PHI136)
C0 = cap(g, E_base, F)
R["f136_base_exact8"] = C0
log("base f136 exact8", C0)

# ---------------- (a) random per-zone phase error only in P < P0
zid = zones(g, F)
uniq, inv = np.unique(zid, return_inverse=True)
inv = inv.reshape(zid.shape)
ra = []
for p0lab, p0 in (("2L", 2 * LAMG), ("3L", 3 * LAMG), ("4L", 4 * LAMG)):
    m = P136 < p0
    for sig in (10.0, 20.0):
        for scope in ("all", "below_P0"):
            ls = []
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                v = rng.normal(0.0, sig, size=uniq.shape)[inv] * np.pi / 180.0
                if scope == "below_P0":
                    v = np.where(m, v, 0.0)
                ls.append(db(cap(g, g.A * np.exp(1j * (PHI136 + v)), F), C0))
            ra.append(dict(P0=p0lab, sigma=sig, scope=scope, loss_dB_mean=float(np.mean(ls)), seeds=ls))
            log("(a)", p0lab, sig, scope, np.round(ls, 4))
R["a_phase_noise_scope"] = ra

# ---------------- (b) systematic phase error in P < P0 (amplitude untouched)
rb = []
for p0lab, p0 in (("3L", 3 * LAMG), ("4L", 4 * LAMG), ("6L", 6 * LAMG)):
    m = P136 < p0
    for dlt in (30.0, 60.0, 90.0):
        v = np.where(m, dlt * np.pi / 180.0, 0.0)
        c = cap(g, g.A * np.exp(1j * (PHI136 + v)), F)
        rb.append(dict(model="step", P0=p0lab, delta_deg=dlt, loss_dB=db(c, C0)))
        log("(b) step", p0lab, dlt, db(c, C0))
for d0 in (30.0, 60.0):
    for kk in (1, 2):
        v = d0 * np.pi / 180.0 * (2 * LAMG / P136) ** kk
        c = cap(g, g.A * np.exp(1j * (PHI136 + v)), F)
        rb.append(dict(model="powerlaw", delta0_deg_at_2L=d0, k=kk, loss_dB=db(c, C0)))
        log("(b) powerlaw", d0, kk, db(c, C0))
R["b_systematic_phase"] = rb

# ---------------- (c) two-arm overlap: arm B carries a relative phase error in P < P0
rc = []
EA0 = pd_field(g, E_base, F)
S0 = float(np.abs(np.sum(np.conj(EA0) * EA0)) * DT * DT)
for p0lab, p0 in (("3L", 3 * LAMG), ("4L", 4 * LAMG), ("6L", 6 * LAMG)):
    m = P136 < p0
    for dlt in (20.0, 45.0, 90.0):
        EB = pd_field(g, g.A * np.exp(1j * (PHI136 + np.where(m, dlt * np.pi / 180.0, 0.0))), F)
        S = float(np.abs(np.sum(np.conj(EA0) * EB)) * DT * DT)
        cB = float(np.sum(np.abs(EB) ** 2) * DT * DT)
        rc.append(dict(P0=p0lab, dpsi_err_deg=dlt, overlap_loss_dB=db(S, S0),
                       armB_capture_loss_dB=db(cB, S0)))
        log("(c)", p0lab, dlt, db(S, S0), db(cB, S0))
R["c_two_arm_overlap"] = rc

# ---------------- (d) ghost transmitted orders from P < P0
rd = []
pds = {"own": (XC, YC), "nbr_x": (-XC, YC), "nbr_y": (XC, -YC), "nbr_diag": (-XC, -YC)}
for p0lab, p0 in (("3L", 3 * LAMG), ("4L", 4 * LAMG), ("6L", 6 * LAMG)):
    m = P136 < p0
    for n in (0, -1, 2):
        prop = np.abs(n) * LAM / P136 < 1.0
        mm = m & prop
        G = np.where(mm, g.A * np.exp(1j * n * PHI136), 0.0)
        pg = float(np.sum(np.abs(G) ** 2) * g.dx * g.dx) / g.pin
        row = dict(P0=p0lab, order=n, ghost_power_frac_of_Pin_if_all=pg)
        for key, (cx, cy) in pds.items():
            if pg > 0:
                row["box_frac_of_ghost_" + key] = cap(g, G, F, cx, cy) / pg
            else:
                row["box_frac_of_ghost_" + key] = 0.0
        rd.append(row)
        log("(d)", p0lab, n, json.dumps({k: (round(v, 7) if isinstance(v, float) else v) for k, v in row.items()}))
R["d_ghost_orders"] = rd

# ---------------- (e) extrapolation of B beyond the largest measured P = 5 Lambda
re_ = []
P5 = 5 * LAMG
for B5 in (-0.3, -0.65, -1.0):
    for mode in ("zero_above", "hold", "invP", "invP2"):
        Bdb = np.full(P136.shape, B5)
        hi = P136 > P5
        if mode == "zero_above":
            Bdb[hi] = 0.0
        elif mode == "invP":
            Bdb[hi] = B5 * (P5 / P136[hi])
        elif mode == "invP2":
            Bdb[hi] = B5 * (P5 / P136[hi]) ** 2
        for fx in (136.0, 381.0):
            Pf = g.P(fx)
            if fx != 136.0:
                Bf = np.full(Pf.shape, B5)
                hif = Pf > P5
                if mode == "zero_above":
                    Bf[hif] = 0.0
                elif mode == "invP":
                    Bf[hif] = B5 * (P5 / Pf[hif])
                elif mode == "invP2":
                    Bf[hif] = B5 * (P5 / Pf[hif]) ** 2
            else:
                Bf = Bdb
            ph = g.phi(fx)
            cb = cap(g, g.A * np.exp(1j * ph), fx)
            cc = cap(g, g.A * 10.0 ** (Bf / 20.0) * np.exp(1j * ph), fx)
            re_.append(dict(B_flat_to_5L_dB=B5, extrap=mode, f=fx, loss_dB=db(cc, cb)))
            log("(e)", B5, mode, fx, db(cc, cb))
R["e_extrapolation"] = re_

# ---------------- (f) f scan
rf = []
w2 = g.A ** 2
for fx in (136.0, 170.0, 200.0, 250.0, 300.0, 381.0, 437.0):
    Pf = g.P(fx)
    cb = cap(g, g.A * np.exp(1j * g.phi(fx)), fx)
    rf.append(dict(f=fx, capture_exact8=cb,
                   pow_below_3L=float(w2[Pf < 3 * LAMG].sum() / w2.sum()),
                   pow_below_5L=float(w2[Pf < 5 * LAMG].sum() / w2.sum()),
                   pow_below_24p8=float(w2[Pf < 24.8].sum() / w2.sum())))
    log("(f)", json.dumps({k: round(v, 5) for k, v in rf[-1].items()}))
    g._H.pop(fx, None) if fx not in (136.0, 381.0) else None
R["f_scan"] = rf
del g

# ---------------- (g) site quantisation on the Lambda/3 grid, N = 8
rg = []
site = LAMG / 3.0
gq = Grid(site / 2.0, 4)
for fx in (136.0, 381.0):
    ph = gq.phi(fx)
    # site index of each sample (2 x 2 samples per site)
    ix = (np.floor(gq.X / site)).astype(np.int64)
    iy = (np.floor(gq.Y / site)).astype(np.int64)
    xs = (ix + 0.5) * site
    ys = (iy + 0.5) * site
    rs = np.hypot(xs - XC, ys - YC)
    ph_site = -K0 * (np.sqrt(rs ** 2 + fx ** 2) - fx)
    lev = lambda p: (np.round(np.mod(p, 2 * np.pi) * 8 / (2 * np.pi)) % 8) * (2 * np.pi / 8)
    cc = cap(gq, gq.A * np.exp(1j * ph), fx)
    c8 = cap(gq, gq.A * np.exp(1j * lev(ph)), fx)
    cs = cap(gq, gq.A * np.exp(1j * ph_site), fx)
    c8s = cap(gq, gq.A * np.exp(1j * lev(ph_site)), fx)
    row = dict(f=fx, cont=cc, N8_fine=c8, site_cont=cs, site_N8=c8s,
               N8_fine_loss_dB=db(c8, cc), site_cont_loss_dB=db(cs, cc), site_N8_loss_dB=db(c8s, cc))
    rg.append(row)
    log("(g)", json.dumps({k: round(v, 5) for k, v in row.items()}))
    gq._H.clear()
R["g_site_quantisation"] = rg

R["elapsed_s"] = time.time() - T0
with open(OUT + r"\review_lens_physics.json", "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=2)
log("done")
