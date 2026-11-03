"""Share of the quadrant power of the Gaussian beam by local period of the hyperbolic metalens (numpy only, no solver).
Model as in SI Note S10 / facts_focal: field exp(-r^2/100^2) (intensity radius 100 um), beam centre at the shared corner,
quadrant [0,150]^2 um (hard cut), lens centre (39.894, 39.894) um, f = 200 um, lambda 1.55 um, P_loc = lam*sqrt(rho^2+f^2)/rho.
Step 1: reproduce the published zone shares (2.666, 7.462, 25.6, 47.9, 16.37 %; below 3L 2.7 %, below 5L 20.4 %).
Step 2: share of power with P_loc below 8 L (level width P/8 < one period L of the bottom metagrating), and per-level width table.
"""
import numpy as np

LAM = 1.55
LAMBDA = 1.31539   # um, bottom metagrating period
F = 200.0
C = 39.894
h = 0.05
xs = np.arange(h / 2, 150, h)
X, Y = np.meshgrid(xs, xs, indexing="ij")
I = np.exp(-2.0 * (X ** 2 + Y ** 2) / 100.0 ** 2)
rho = np.hypot(X - C, Y - C)
P = LAM * np.sqrt(rho ** 2 + F ** 2) / np.maximum(rho, 1e-9)
PL = P / LAMBDA
tot = I.sum()

def share(lo, hi):
    m = (PL >= lo) & (PL < hi)
    return 100.0 * I[m].sum() / tot

out = []
def log(s):
    print(s)
    out.append(s)

log("grid h = %.3f um, rho_max = %.2f um, P_loc/L min = %.3f" % (h, rho.max(), PL.min()))
log("zone shares (percent of quadrant power inside 150 x 150 um):")
for lo, hi, ref in ((0, 2, None), (2, 3, 2.666), (3, 4, 7.462), (4, 6, 25.6), (6, 12, 47.9), (12, 1e9, 16.37)):
    log("  %5s <= P/L < %6s : %7.3f %%   (SI Table S15: %s)" % (lo, hi if hi < 1e8 else "inf", share(lo, hi), ref))
log("below 3L: %.3f %% (published 2.7 %% at f = 200); below 5L: %.3f %% (published 20.4 %%)" % (share(0, 3), share(0, 5)))
log("")
log("Per-level width of an 8-level staircase = P/8; atoms per level = M/8 with M = 3 P / L")
for M in (6, 9, 12, 15, 18, 24, 36, 48):
    P_um = M * LAMBDA / 3.0
    log("  M=%2d  P=%7.3f um (%.2f L)  level width P/8 = %.3f um = %.3f L = %.2f atoms" % (M, P_um, P_um / LAMBDA, P_um / 8, P_um / 8 / LAMBDA, M / 8.0))
log("")
for k in (4, 6, 8, 12, 16, 24):
    log("share with P/L < %2d (level narrower than %.3f L): %.2f %%" % (k, k / 8.0, share(0, k)))
open(r"<WORK_ROOT>\paper_scdc_261011_v3\review\r7_memo_261013\designflow\lens_zone_share_out.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
