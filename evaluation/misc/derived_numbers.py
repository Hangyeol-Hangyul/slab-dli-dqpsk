"""Derived numbers for the designflow memos (numpy only, no solver). Inputs are quoted values with their sources.

Part A (memo 9): lens power with level width P/8 below the effective seam width delta of SI Note S5.
  delta values (SI Note S5, si_v2_r6.md line 163): par(+) 1.313 um, par(-) 0.865 um, perp 0.923 um.
  Lens model identical to lens_zone_share.py (reproduces SI Table S15 shares to 0.04 pp).
Part B (Fig 1(b) quantitative request): shares from Fougstedt 2020 Table 1 and Hirokawa 2021.
"""
import numpy as np

out = []
def log(s):
    print(s)
    out.append(s)

# ---------------- Part A
LAM, LAMBDA, F, C = 1.55, 1.31539, 200.0, 39.894
h = 0.05
xs = np.arange(h / 2, 150, h)
X, Y = np.meshgrid(xs, xs, indexing="ij")
I = np.exp(-2.0 * (X ** 2 + Y ** 2) / 100.0 ** 2)
rho = np.hypot(X - C, Y - C)
P = LAM * np.sqrt(rho ** 2 + F ** 2) / np.maximum(rho, 1e-9)
tot = I.sum()
def share_below_P(P_um):
    return 100.0 * I[P < P_um].sum() / tot

log("PART A: share of quadrant power with level width P/8 < delta  <=>  P < 8 delta")
for name, d in (("par(-) 0.865", 0.865), ("perp 0.923", 0.923), ("par(+) 1.313", 1.313), ("one period L = 1.31539", LAMBDA)):
    log("  delta = %-24s -> P < %.3f um = %.3f L : %.2f %% of lens power" % (name, 8 * d, 8 * d / LAMBDA, share_below_P(8 * d)))
log("  (check: P<8L reproduces 63.48 %% of lens_zone_share_out.txt: %.2f %%)" % share_below_P(8 * LAMBDA))
log("")
log("level width P/8 per ramp class vs the seam widths (um):")
for M in (6, 9, 12, 15, 18, 24, 36, 48):
    Pm = M * LAMBDA / 3.0
    log("  x%02d  P/8 = %.3f um  (delta/(P/8): par- %.2f, perp %.2f, par+ %.2f)" % (M, Pm / 8, 0.865 / (Pm / 8), 0.923 / (Pm / 8), 1.313 / (Pm / 8)))

# ---------------- Part B
log("")
log("PART B: Fougstedt 2020 Table 1 (22 nm, 400G coherent RX ASIC), W")
t = dict(CDC=0.97, Interp=0.14, DynEq=3.25, CPE=0.50, ProdDec=0.07, StairDec=0.14, ADC=1.42)
dsp = t["CDC"] + t["Interp"] + t["DynEq"] + t["CPE"]
log("  DSP (CDC+Interp+DynEq+CPE) = %.2f W" % dsp)
for fec in ("ProdDec", "StairDec"):
    tot_df = dsp + t[fec]
    log("  DSP + %s = %.2f W (paper abstract: 5 W); DynEq share %.1f %%, CDC share %.1f %%, CPE %.1f %%, Interp %.1f %%, FEC %.1f %%"
        % (fec, tot_df, 100 * t["DynEq"] / tot_df, 100 * t["CDC"] / tot_df, 100 * t["CPE"] / tot_df, 100 * t["Interp"] / tot_df, 100 * t[fec] / tot_df))
    log("     ADC share of (DSP+FEC+ADC) = %.1f %%; (DSP+FEC+ADC) = %.2f W = %.1f pJ/bit at 400 Gb/s" % (100 * t["ADC"] / (tot_df + t["ADC"]), tot_df + t["ADC"], (tot_df + t["ADC"]) / 400e9 * 1e12))
log("PART B2: Hirokawa 2021: 4 ADCs 4 x 235 mW = %.0f mW, 4 x 702 mW = %.0f mW; per bit at 224 Gb/s: %.2f / %.2f pJ/bit"
    % (4 * 235, 4 * 702, 4 * 0.235 / 224e9 * 1e12, 4 * 0.702 / 224e9 * 1e12))
log("PART B3: module-level (Arista 400ZR 16 W typ / 20 W max; EFFECT nITLA 2.9 W typ; Nagarajan 'DSP ~ 50 %' UNVERIFIED):")
log("  nITLA 2.9 W / 16 W = %.1f %% ; / 20 W = %.1f %% (cross-vendor indicative ratio, not a measured module breakdown)" % (100 * 2.9 / 16, 100 * 2.9 / 20))
log("  Fougstedt DSP+FEC+ADC 6.42 W / 16 W = %.1f %% (22 nm RX-side study vs a 7 nm module: do NOT quote as a module share)" % (100 * 6.42 / 16))
log("  Hirokawa laser: 13 dBm = %.1f mW optical; at 20 %% wall-plug = %.0f mW electrical" % (10 ** 1.3, 10 ** 1.3 / 0.2))
open(r"<WORK_ROOT>\paper_scdc_261011_v3\review\r7_memo_261013\designflow\derived_numbers_out.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
