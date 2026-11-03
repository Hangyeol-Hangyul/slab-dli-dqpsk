import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"<WORK_ROOT>\pdfs_density_261019"
FSR = 100.0            # GHz, main line 61 / SI line 69
G_DEG_NM = 0.5474      # deg/nm registration, main line 139
G_T = 0.3877           # deg/nm slab thickness (common bias), SI line 183
DEG_PER_K = 5.6        # deg/K, SI Table S25 (estimate in record)
L_EXTRA = 2049.8       # um
LAM = 1.55             # um
S_MM = 0.8364          # mm, s = t tan(theta)
G2GHZ = FSR / 360.0

def ghz(d): return d * G2GHZ

rows = []
def add(item, inp, formula, deg, label):
    rows.append(dict(item=item, input=inp, formula=formula, phase_deg=round(deg, 5),
                     pdfs_GHz=round(ghz(deg), 5), pct_FSR=round(100 * ghz(deg) / FSR, 5), label=label))

# (a) leakage bound
for nm, p in (("x", 1.51e-4), ("y", 1.83e-4)):
    r = math.sqrt(p)
    add("leak_bound_branch_" + nm, "cross-branch leakage power %.2e (SI Table S8, 1793 harm)" % p,
        "asin(sqrt(p)), r=%.5f" % r, math.degrees(math.asin(r)), "CALC bound")
dsum = sum(math.degrees(math.asin(math.sqrt(p))) for p in (1.51e-4, 1.83e-4))
add("leak_bound_difference", "sum of both branch bounds (opposite signs)", "asin(rx)+asin(ry)", dsum, "CALC worst-case bound")
# (b) sensitivities
add("registration_1nm", "0.5474 deg/nm", "2G*(dx-dy)", G_DEG_NM * 1, "SIM/analytic (record)")
for g in (1, 5, 10):
    dt = S_MM * g
    add("thickness_grad_%gnm_per_mm" % g, "dt=s*g=%.4f nm" % dt, "0.3877 deg/nm * dt", G_T * dt, "CALC")
k_dn = 360 * L_EXTRA / LAM
for dn in (1e-7, 5e-7, 1e-6):
    add("stress_dn_%.0e" % dn, "dn=%.0e (ASSUMED)" % dn, "360*dn*L/lambda", k_dn * dn, "CALC, dn ASSUMED")
dndT, alpha, n = 1.0e-5, 0.55e-6, 1.444
own = k_dn * (dndT + n * alpha)
for dT in (0.01, 0.1):
    add("temp_diff_%gK_recorded" % dT, "5.6 deg/K (SI Table S25, estimate)", "5.6*dT", DEG_PER_K * dT, "CALC")
    add("temp_diff_%gK_own" % dT, "dn/dT=1.0e-5/K, alpha=0.55e-6/K, n=1.444 (ASSUMED) -> %.3f deg/K" % own,
        "360*(L/lam)*(dn/dT+n*alpha)*dT", own * dT, "CALC, constants ASSUMED")
with open(OUT + r"\pdfs.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# (c) tolerances
refs = [("Lize 1 % FSR", 1.0 * 3.6), ("Nasu 0.78 % FSR", 0.78 * 3.6), ("Li 0.61 % FSR", 0.61 * 3.6),
        ("Doerr2006 4.6 % FSR", 4.6 * 3.6), ("Optoplex <3 deg", 3.0), ("Kawashima 0.291 GHz", 0.291 / G2GHZ)]
tol = []
for nm, d in refs:
    tol.append((nm, d, ghz(d), d / G_DEG_NM, d / G_T, d / G_T / S_MM, d / k_dn, d / DEG_PER_K))
with open(OUT + r"\tolerance.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ref", "deg", "GHz_on_our_FSR", "dx-dy_nm", "dt_nm", "grad_nm_per_mm", "dn", "dT_K_rec5.6"])
    for t in tol: w.writerow([t[0]] + ["%.5g" % x for x in t[1:]])
for t in tol: print(t[0], ["%.4g" % x for x in t[1:]])
print("k_dn deg per dn=1:", k_dn, "own deg/K", own)

# Part 2
walk2s = 1.6727  # mm
def geom(a):
    side = walk2s + a
    return side * side, 2 * side * a - a * a
lit = [("Klamkin 2014", 33.333), ("Velha 2015 (30 Gb/s)", 42.857), ("Aamer 2012", 90.909)]
lit_all = [("Klamkin 2014", 33.333), ("Velha 2015 (30 Gb/s)", 42.857), ("Velha 2015 (20 GBd)", 57.143),
           ("Aamer 2012", 90.909), ("Doerr 2007 design", 15.735), ("Klamkin InP 107G secondary", 41.797)]
Ns = [1, 2, 4, 8, 16, 32, 64, 128]
dens = []
for a in (0.30,):
    Lb, Ls = geom(a)
    print("a", a, "Lbox", Lb, "Lstrip", Ls, "Dinf", 200 / Ls, "D1", 200 / Lb)
    for N in Ns:
        A = Lb + (N - 1) * Ls
        dens.append((a, N, A, 200 * N / A))
    dens.append((a, "inf", float("inf"), 200 / Ls))
    for nm, d in lit_all:
        Nc = d * (Lb - Ls) / (200 - d * Ls)
        print(nm, d, "N>", Nc, "ceil", math.floor(Nc) + 1)
with open(OUT + r"\density_vs_N.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["a_mm", "N", "area_mm2", "D_Gbps_per_mm2"])
    for r in dens: w.writerow(r)
for r in dens: print(r)

# figure
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(89 / 25.4, 70 / 25.4))
Nf = np.logspace(0, math.log10(1024), 200)
Lb, Ls = geom(0.30)
ax.plot(Nf, 200 * Nf / (Lb + (Nf - 1) * Ls), color="#0072B2", lw=1.2, label="This work, a = 0.30 mm")
ax.axhline(200 / Ls, color="#0072B2", lw=0.6, ls=":")
ax.text(1.1, 200 / Ls + 6, "limit 182.8", color="#0072B2", fontsize=6)
ax.plot([n for n in Ns], [200 * n / (Lb + (n - 1) * Ls) for n in Ns], "o", color="#0072B2", ms=2.5)
cols = ["#D55E00", "#009E73", "#CC79A7"]
for (nm, d), c in zip(lit, cols):
    ax.axhline(d, color=c, lw=0.9, ls="--")
    ax.text(900, d * 1.04, "%s %.1f" % (nm, d), color=c, fontsize=6, ha="right", va="bottom")
ax.set_xscale("log"); ax.set_xlim(1, 1024); ax.set_ylim(0, 205)
ax.set_xlabel("Channels N in the L-layout array"); ax.set_ylabel("Throughput density (Gb/s/mm$^2$)")
ax.legend(frameon=False, loc="lower right", fontsize=6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT + r"\fig_density_vs_N.png", dpi=300)
fig.savefig(OUT + r"\fig_density_vs_N.pdf")
