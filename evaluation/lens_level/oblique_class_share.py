r"""power share of the four oblique directions (psi 45 / 135 / 225 / 315 +- 22.5 deg) and of +-x / +-y cones per P band, NE quadrant, f 200 (numpy only) [E]."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="2"
import numpy as np
LAM, LAMG, W, APER, f = 1.55, 1.31539, 100.0, 150.0, 200.0
XC = W / np.sqrt(2 * np.pi); h = LAMG / 3; n = int(round(APER / h)); x = (np.arange(n) + 0.5) * h
X, Y = np.meshgrid(x, x); PW = np.exp(-2 * (X ** 2 + Y ** 2) / W ** 2)
rho = np.hypot(X - XC, Y - XC); PL = LAM * np.sqrt(rho ** 2 + f ** 2) / np.maximum(rho, 1e-9) / LAMG
psi = np.degrees(np.arctan2(-(Y - XC), -(X - XC))) % 360
cls = {"+x(0)": 0, "45": 45, "+y(90)": 90, "135": 135, "-x(180)": 180, "225": 225, "-y(270)": 270, "315": 315}
T = PW.sum()
print("band (L)       lens%  " + "  ".join("%7s" % k for k in cls))
for lo, hi in ((2, 4), (4, 6), (6, 12), (12, 1e9), (0, 1e9)):
    m = (PL > lo) & (PL <= hi)
    row = []
    for k, c in cls.items():
        d = np.abs((psi - c + 180) % 360 - 180)
        row.append(100 * PW[m & (d < 22.5)].sum() / PW[m].sum())
    print("(%g, %g]  %6.2f  " % (lo, hi, 100 * PW[m].sum() / T) + "  ".join("%7.1f" % v for v in row))
