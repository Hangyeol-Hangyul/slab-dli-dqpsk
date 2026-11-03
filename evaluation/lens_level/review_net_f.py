# -*- coding: utf-8 -*-
r"""
review_net_f.py -- net absolute capture (exact 8 x 8 um box at the design focus) vs f
under power-law breakdown B(P) = B0 * (2 Lambda / P)^k, to show whether the
lens-level verdict should be judged at a fixed f or on net capture.
Imports the validated Grid/cap helpers from review_lens_physics.py by copy (no import side effects).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import json

OUT = r"<WORK_ROOT>\wf_lpa\lens\review"
LAM = 1.55
LAMG = 1.31539
W = 100.0
APER = 150.0
XC = W / np.sqrt(2.0 * np.pi)
K0 = 2.0 * np.pi / LAM
dx = 0.5
n = int(round(APER / dx))
x = (np.arange(n) + 0.5) * dx
X, Y = np.meshgrid(x, x)
A = np.exp(-(X ** 2 + Y ** 2) / W ** 2)
pin = float(np.sum(A ** 2) * dx * dx)
rho = np.hypot(X - XC, Y - XC)
npd = 4 * n
k = 2.0 * np.pi * np.fft.fftfreq(npd, d=dx)
KX, KY = np.meshgrid(k, k)
kz2 = K0 ** 2 - KX ** 2 - KY ** 2
del KX, KY
DT = 0.1
AX = DT * (np.arange(80) + 0.5) - 4.0
Mx = np.exp(1j * np.outer(XC + AX - dx / 2.0, k))


def cap(E0, f):
    H = np.where(kz2 >= 0, np.exp(1j * np.sqrt(np.clip(kz2, 0, None)) * f), 0.0)
    Ep = np.zeros((npd, npd), dtype=complex)
    Ep[:n, :n] = E0
    S = np.fft.fft2(Ep) * H
    E = (Mx @ S @ Mx.T) / (npd * npd)
    return float(np.sum(np.abs(E) ** 2) * DT * DT) / pin


rows = []
for f in (136.0, 170.0, 200.0, 250.0, 300.0):
    P = LAM * np.sqrt(rho ** 2 + f ** 2) / np.maximum(rho, 1e-9)
    ph = -K0 * (np.sqrt(rho ** 2 + f ** 2) - f)
    c0 = cap(A * np.exp(1j * ph), f)
    row = dict(f=f, cap_B0=c0)
    for B0 in (-1.0, -3.0):
        for kk in (1, 2):
            Bd = B0 * (2 * LAMG / P) ** kk
            c = cap(A * 10.0 ** (Bd / 20.0) * np.exp(1j * ph), f)
            row["cap_B%g_k%d" % (B0, kk)] = c
            row["loss_B%g_k%d_dB" % (B0, kk)] = 10 * np.log10(c / c0)
    rows.append(row)
    print(json.dumps({kk2: round(v, 4) for kk2, v in row.items()}), flush=True)
with open(OUT + r"\review_net_f.json", "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=2)
