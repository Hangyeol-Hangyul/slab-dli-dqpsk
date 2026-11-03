# -*- coding: utf-8 -*-
"""beam_size.py (261019) -- beam is a Gaussian of 100 um intensity radius (main r21 lines 83/85/95); what changes at 5-10 um?
Paraxial Gaussian estimates only (CALC). Geometry from verify_calc.py: lam 1.55 um, n 1.444, t 592.39 um, theta 54.69 deg.
Delay arm B travels the extra slab round trip; with the grating fixing kx, its extra quadratic phase equals free propagation of
  z_x = 2t / cos^3(theta) in the plane of incidence and z_y = 2t / cos(theta) out of it (in-medium distances, beam width w measured
  in the grating plane). Field overlap of arm B with arm A: |eta|^2 = [1+(z_x/2zR)^2]^-1/2 [1+(z_y/2zR)^2]^-1/2, zR = pi w^2 n / lam.
Overlap-limited ER (equal arm powers) = ((1+|eta|)/(1-|eta|))^2. Delayed-beam radius at the output: w sqrt(1+(z/zR)^2) per axis.
Density limit if the element width a scales with the beam (a = 3w; record a = 0.30 mm at w = 100 um): strip = 2(2s+a)a - a^2."""
import math, csv, os
lam, n, t = 1.55, 1.444, 592.39
th = math.asin(lam / (n * 1.31539))
s = t * math.tan(th)
zx, zy = 2 * t / math.cos(th) ** 3, 2 * t / math.cos(th)
rows = []
for w in (5.2, 5.0, 10.0, 20.0, 50.0, 75.0, 100.0, 150.0):
    zR = math.pi * w ** 2 * n / lam
    ov = (1 + (zx / (2 * zR)) ** 2) ** -0.5 * (1 + (zy / (2 * zR)) ** 2) ** -0.5
    eta = math.sqrt(ov)
    er = ((1 + eta) / (1 - eta)) ** 2 if eta < 1 else float("inf")
    wbx, wby = w * math.sqrt(1 + (zx / zR) ** 2), w * math.sqrt(1 + (zy / zR) ** 2)
    a = 3 * w / 1000.0
    L = 2 * s / 1000.0 + a
    strip = 2 * L * a - a * a
    div = math.degrees(lam / (math.pi * n * w))
    rows.append(dict(w_um=w, zR_um=round(zR, 1), div_deg_in_glass=round(div, 3), periods_across_2w=round(2 * w / 1.31539, 1),
                     overlap=round(ov, 5), overlap_dB=round(10 * math.log10(ov), 3), ER_overlap_dB=round(10 * math.log10(er), 2),
                     armB_w_x_um=round(wbx, 1), armB_w_y_um=round(wby, 1), a_mm=round(a, 3), strip_mm2=round(strip, 4),
                     D_inf=round(200 / strip, 1)))
# smallest w with overlap loss <= 1 % and <= 0.1 %
for target in (0.99, 0.999):
    lo, hi = 1.0, 500.0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        zR = math.pi * m ** 2 * n / lam
        ov = (1 + (zx / (2 * zR)) ** 2) ** -0.5 * (1 + (zy / (2 * zR)) ** 2) ** -0.5
        lo, hi = (lo, m) if ov >= target else (m, hi)
    print("smallest w for overlap >= %.3f: %.1f um" % (target, hi))
print("theta %.3f deg, z_x %.1f um, z_y %.1f um, s %.1f um" % (math.degrees(th), zx, zy, s))
for r in rows:
    print(r)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "beam_size.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
