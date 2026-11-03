import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, io, json, csv, time
import numpy as np
HERE = r"<WORK_ROOT>\fig6d_y48_261020\iq_y48_261020"
SRC = r"<WORK_ROOT>\fig7_data_261017"
sys.path.insert(0, HERE)
import eye_model_g as M   # pins to cores 10-13

G2 = json.load(io.open(SRC + r"\g2_numbers.json", encoding="utf-8"))
G10 = json.load(io.open(SRC + r"\g10_numbers.json", encoding="utf-8"))
BA = {r["key"]: r["ba"] for r in G10["rows"]}["1550.0"]
ECHO = -20.59254496490972   # [261020] measured >12 Lambda zone (was G2["echo_anchor"][1] = -20.2177)
PHI0 = M.PHI_CH["1550.0"]
CASES = {
    "i_pure_quadrature": {"a": 1.0, "b": 1.0, "nu": 0.0, "W": 1.0, "phi": 0.0},
    "ii_device_1550": M.case_params(PHI0, "uniform", eps_db=ECHO, ptp_deg=M.PTP_CH["1550.0"], ba=BA),
}
SPEC = [0, 1, 2, 3, 5, 7.5, 10, 15, 20]
FINE = list(np.round(np.arange(0, 20.0001, 0.25), 4))
DELTAS = sorted(set(SPEC) | set(FINE))
t0 = time.time()
rows = []
res = {}
for cid, p in CASES.items():
    RI = M.eo_ratio("I", "bessel", chis=M.CHIS, **p)
    for dd in DELTAS:
        q = dict(p); q["phi"] = p["phi"] + np.deg2rad(dd)      # Q pair: both PDs bias moved by delta
        RQ = M.eo_ratio("Q", "bessel", chis=M.CHIS, **q)
        pj = float(M.pen(np.minimum(RI, RQ).min()))
        res[(cid, dd)] = pj
        rows.append([cid, dd, pj])
    print(cid, "done", round(time.time() - t0, 1), flush=True)
out = []
for cid, dd, pj in rows:
    out.append([cid, dd, "%.6f" % pj, "%.6f" % (pj - res[(cid, 0.0)]), 1 if dd in SPEC else 0])
with io.open(HERE + r"\iq_sweep.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "delta_deg", "penalty_dB", "incremental_dB", "requested_point"])
    w.writerows(out)

# self tests
st = {}
st["delta0_case_ii_pen"] = res[("ii_device_1550", 0.0)]
st["stored_total_dB"] = 0.671837
st["resid_dB"] = res[("ii_device_1550", 0.0)] - 0.671837
# stored summary 6 digits check
for cid, p in CASES.items():
    q = dict(p); q["phi"] = p["phi"] + np.deg2rad(90.0)
    RQ = M.eo_ratio("Q", "bessel", chis=M.CHIS, **q)
    st["delta90_%s_EO_Q_min" % cid] = float(RQ.min()); st["delta90_%s_EO_Q_max" % cid] = float(RQ.max())
# sign symmetry
p = CASES["i_pure_quadrature"]
a = M.eo_ratio("Q", "bessel", chis=M.CHIS[:1], **dict(p, phi=np.deg2rad(5.0)))[0]
b = M.eo_ratio("Q", "bessel", chis=M.CHIS[:1], **dict(p, phi=np.deg2rad(-5.0)))[0]
st["sign_sym_5deg"] = [float(a), float(b)]
st["cf_cos_minus_sin_5deg_ideal"] = float(M.cf_bias(5.0))
# tolerances
tol = {}
K = 0.5474
for cid in CASES:
    xs = np.array(FINE, float)
    inc = np.array([res[(cid, d)] - res[(cid, 0.0)] for d in FINE])
    tol[cid] = {}
    for target in (0.1, 0.2, 0.5):
        dl = float(np.interp(target, inc, xs))
        tol[cid][str(target)] = {"delta_deg": dl, "d_nm": dl / K}
json.dump({"selftest": st, "tolerance": tol, "K_deg_per_nm": K, "BA": BA, "echo_dB": ECHO, "phi_deg": PHI0}, io.open(HERE + r"\iq_numbers.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(st, indent=1)); print(json.dumps(tol, indent=1))

# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(89 / 25.4, 70 / 25.4))
cols = {"i_pure_quadrature": "#0072B2", "ii_device_1550": "#D55E00"}
labs = {"i_pure_quadrature": "No echo, equal arms", "ii_device_1550": "1550 nm device (echo, arm ratio, spread)"}
for cid in CASES:
    ax.plot(FINE, [res[(cid, d)] for d in FINE], color=cols[cid], lw=1.0, label=labs[cid])
    ax.plot(SPEC, [res[(cid, d)] for d in SPEC], "o", color=cols[cid], ms=2.5)
ax.set_xlabel("Quadrature error (deg)"); ax.set_ylabel("Eye-opening penalty (dB)")
ax.set_xlim(0, 20); ax.legend(frameon=False, loc="upper left", fontsize=7)
sec = ax.secondary_xaxis("top", functions=(lambda x: x / K, lambda x: x * K))
sec.set_xlabel("Placement error of one quadrant pattern (nm)")
for s in ("right",): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(HERE + r"\fig_iq_penalty.png", dpi=300); fig.savefig(HERE + r"\fig_iq_penalty.pdf")
print("done", round(time.time() - t0, 1))
