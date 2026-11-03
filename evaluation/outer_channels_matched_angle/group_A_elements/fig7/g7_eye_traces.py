# -*- coding: utf-8 -*-
"""G part 2, step 3 (Task B): data for the Figure 7(b) eye panel.  Uses the FIXED eye model unchanged (eye_model_g.py, a copy of fix_261015\\E_eye_fig7\\eye_fig7_model.py
whose only differences are the CPU-affinity lines and the BLAS-thread setting).  No solver, no RCWA.

FIX ROUND 261015 (defect b): the arm ratio b/a is the PER-CHANNEL value of g10_arm_ratio.py (g10_numbers.json, DERIVED from the stored P_A, P_B), not the 1550 nm constant 0.92642.
Cases (headline treatment as in eye_penalty_11ch.csv: eight atoms equally spaced over the channel's peak-to-peak range, coherent sum, no net bias; Bessel 0.75 x baud; optical dB;
echo = per-channel aperture echo, -20.66037 dB MEASURED at 1550.000 nm, -21.2279 / -19.8451 dB ESTIMATE at 1546.003 / 1554.017 nm):
  ideal                       a = b, no echo, no phase error
  device_1550.000             common DLI phase 0 (case (i) = case (ii))
  device_1546.003_fixed_bias  case (i):  bias fixed at 1550 nm, phi = +6.6143 deg
  device_1546.003_recentred   case (ii): bias re-centred per channel, phi = 0
  device_1554.017_fixed_bias  case (i):  phi = -15.6110 deg
  device_1554.017_recentred   case (ii): phi = 0

Outputs:  fig7b_eye_<case>.csv (300 representative eye windows: t_UI + one column per trace), fig7b_eye_traces_index.csv, fig7b_eye_summary.csv, fig7b_penalty_stack.csv,
          fig7b_eye_traces_full.npz (all 4096 windows, reproducibility only), fig7b_eye_preview.png (quick look, not a deliverable), g7_stdout.txt
Run: python -I -B g7_eye_traces.py   (cores 10-13 after the imports, BLAS 1 thread, < 1 GB)
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"                     # BLAS threads 1, set BEFORE numpy is imported
import csv
import io
import itertools
import json
import math
import sys
import time

HERE = r"<WORK_ROOT>\fill_A_261102\kx_chain\fig7"
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np                                                               # noqa: E402
import psutil                                                                    # noqa: E402
import eye_model_g as M                                                          # noqa: E402  (pins itself to cores 10-13, BLAS threads 1)

T0 = time.time()
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


G2 = json.load(io.open(HERE + r"\g2_numbers.json", encoding="utf-8"))
G3 = json.load(io.open(HERE + r"\g3_numbers.json", encoding="utf-8"))
G10 = json.load(io.open(HERE + r"\g10_numbers.json", encoding="utf-8"))
BA_CH = {r["key"]: r["ba"] for r in G10["rows"]}                 # b/a per channel, DERIVED (g10_arm_ratio.py)
ECHO_A = G2["echo_anchor"]                       # [1546.003, 1550.000, 1554.017] dB
ROWS = {r["key"]: r for r in G3["rows"]}
say("g7_eye_traces.py %s affinity %s RSS %.2f GB, memory available %.1f GB" % (time.strftime("%Y-%m-%d %H:%M:%S"), psutil.Process().cpu_affinity(),
                                                                              psutil.Process().memory_info().rss / 1e9, psutil.virtual_memory().available / 1e9))

CASES = [  # id, title, wavelength key, echo dB, bias case, phi key (None = 0)
    ("ideal", "Ideal", None, None, "-", None),
    ("device_1550.000", "1550 nm", "1550.0", ECHO_A[1], "(i) = (ii)", "1550.0"),
    ("device_1546.003_fixed_bias", "1546 nm", "1546.003", ECHO_A[0], "(i) fixed bias", "1546.003"),
    ("device_1546.003_recentred", "1546 nm", "1546.003", ECHO_A[0], "(ii) re-centred", None),
    ("device_1554.017_fixed_bias", "1554 nm", "1554.017", ECHO_A[2], "(i) fixed bias", "1554.017"),
    ("device_1554.017_recentred", "1554 nm", "1554.017", ECHO_A[2], "(ii) re-centred", None),
]
OS, N, PCS = M.OS, M.N, M.PCS
POS = M.POS["I"]
T_UI = (np.arange(2 * OS + 1) - OS) / OS


def params(wkey, e_db, phi_key):
    if wkey is None:
        return {"a": 1.0, "b": 1.0, "nu": 0.0, "W": 1.0, "phi": 0.0}
    phi = M.PHI_CH[phi_key] if phi_key else 0.0
    return M.case_params(phi, "uniform", eps_db=e_db, ptp_deg=M.PTP_CH[wkey], ba=BA_CH[wkey])


def joint_R(p):
    RI = M.eo_ratio("I", "bessel", chis=M.CHIS, **p)
    RQ = M.eo_ratio("Q", "bessel", chis=M.CHIS, **p)
    return RI, RQ, np.minimum(RI, RQ)


# ----------------------------------------------------------------------------------------------------------------------------------------- traces and summary
SUMMARY, FULL, INDEX, UNIQUE, UNIQUE_SRC = [], {}, [], {}, {}
say("")
say("EYE TRACES (I rail, at the joint worst echo phase chi; normalised to EO of the ideal receiver with the same total arm power)")
for cid, title, wkey, e_db, bcase, phi_key in CASES:
    p = params(wkey, e_db, phi_key)
    RI, RQ, RJ = joint_R(p)
    i = int(np.argmin(RJ))
    chi = float(M.CHIS[i])
    ne = M.NECHO if p["nu"] > 0 else 0
    L = M.level_wave("I", p["a"], p["b"], p["nu"], np.array([chi]), necho=ne, phi=p["phi"], W=p["W"])[0]
    eo_ref = M.eo_reference("I", p["a"], p["b"], "bessel", "ideal")
    flat = M.filt_wave(L).reshape(-1)
    idx = (np.arange(N)[:, None] * OS + PCS - OS + np.arange(2 * OS + 1)[None, :]) % (N * OS)
    tr = (flat[idx] / eo_ref)
    Y = M.sample(L[None, :], "bessel")[0] / eo_ref
    assert np.max(np.abs(tr[:, OS] - Y)) < 1e-12                     # window centre = decision variable
    y_hi, y_lo = float(tr[POS, OS].min()), float(tr[~POS, OS].max())
    eo = y_hi - y_lo
    assert abs(eo - RI[i]) < 1e-9, (cid, eo, RI[i])                  # bar = EO of the I rail at that chi
    pen_w = float(M.pen(RJ.min()))
    pw = M.pen(RJ)
    row = dict(case_id=cid, title=title, wavelength_nm=(wkey or ""), bias_case=bcase, chi_deg=float(np.rad2deg(chi)),
               pen_worst=pen_w, pen_best=float(pw.min()), pen_mean=float(pw.mean()), eo_joint=float(RJ.min()), eo_I_at_chi=float(RI[i]), bar_lo=y_lo, bar_hi=y_hi,
               phi_deg=(M.PHI_CH[phi_key] if phi_key else 0.0), ptp_deg=(M.PTP_CH[wkey] if wkey else 0.0), echo_dB=(e_db if e_db is not None else float("nan")),
               ba=p["b"] / p["a"], vis=abs(p["W"]))
    if wkey:                                                          # reproduce the stored 11-channel table
        key = "i" if (phi_key or wkey == "1550.0") else "ii"
        st = ROWS[wkey][key]
        assert abs(pen_w - st["pen_wc"]) < 1e-12, (cid, pen_w, st["pen_wc"])
        assert abs(row["pen_best"] - st["pen_min"]) < 1e-12 and abs(row["pen_mean"] - st["pen_mean"]) < 1e-12
        assert abs(row["chi_deg"] - st["chi_wc_deg"]) < 1e-9
        row["table_residual"] = pen_w - st["pen_wc"]
    else:
        assert abs(pen_w) < 1e-12 and abs(eo - 1.0) < 1e-12
        row["table_residual"] = 0.0
    # representative traces: K = 300 ACTUAL windows chosen by greedy farthest-first (L-infinity) selection, the two windows that define the eye bar forced in first;
    # multiplicity = number of the 4096 windows whose nearest representative it is; coverage radius = largest L-infinity distance of any window to its representative
    K = 300
    pos_idx, neg_idx = np.where(POS)[0], np.where(~POS)[0]
    sel = [int(pos_idx[np.argmin(tr[pos_idx, OS])]), int(neg_idx[np.argmax(tr[neg_idx, OS])])]
    Dm = [np.max(np.abs(tr - tr[j]), axis=1) for j in sel]
    d = np.min(np.stack(Dm), axis=0)
    while len(sel) < K and d.max() > 1e-3:                           # stop early once every window is within 0.001 (invisible)
        j = int(np.argmax(d))
        sel.append(j)
        Dm.append(np.max(np.abs(tr - tr[j]), axis=1))
        d = np.minimum(d, Dm[-1])
    cover = float(d.max())
    assign = np.argmin(np.stack(Dm), axis=0)
    cnt = np.bincount(assign, minlength=len(sel))
    lvl = POS[np.array(sel)]
    order = np.lexsort((-cnt, ~lvl))
    sel_a = np.array(sel)[order]
    rep, cnt, lvl = tr[sel_a], cnt[order], lvl[order]
    assert cnt.sum() == N and np.all(cnt >= 1)
    for u in range(len(sel_a)):                                     # a representative never stands for windows of the other level
        assert np.all(POS[assign == order[u]] == lvl[u])
    max_dev = cover
    uq = sel_a
    UNIQUE_SRC[cid] = sel_a
    for kk in ("pen_worst", "pen_best", "pen_mean"):
        row[kk] = 0.0 if abs(row[kk]) < 5e-13 else row[kk]
    row["n_unique"] = int(len(sel_a))
    row["max_dev"] = max_dev
    UNIQUE[cid] = (rep, cnt, lvl)
    FULL["tr_" + cid] = tr.astype(np.float32)
    SUMMARY.append(row)
    say("  %-28s chi %5.1f deg  penalty worst %.6f best %.6f mean %.6f dB  EO(joint) %.6f  EO(I, bar) %.6f  bar [%.4f, %.4f]  representative traces %d (coverage radius %.4f)  table residual %.1e" % (
        cid, row["chi_deg"], pen_w, row["pen_best"], row["pen_mean"], row["eo_joint"], row["eo_I_at_chi"], y_lo, y_hi, len(uq), max_dev, row["table_residual"]))
    assert max_dev <= 0.02 + 1e-9

for cid, (rep, cnt, lvl) in UNIQUE.items():
    n = rep.shape[0]
    with io.open(HERE + r"\fig7b_eye_%s.csv" % cid, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_UI"] + ["tr%03d" % (u + 1) for u in range(n)])
        for s in range(rep.shape[1]):
            w.writerow(["%.5f" % T_UI[s]] + ["%.4f" % rep[u, s] for u in range(n)])
    for u in range(n):
        INDEX.append(dict(case_id=cid, trace="tr%03d" % (u + 1), level=("1" if lvl[u] else "0"), multiplicity_of_4096=int(cnt[u]), source_window=int(UNIQUE_SRC[cid][u])))
with io.open(HERE + r"\fig7b_eye_traces_index.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case_id", "trace", "level", "multiplicity_of_4096", "source_window"])
    w.writeheader()
    for r in INDEX:
        w.writerow(r)
np.savez_compressed(HERE + r"\fig7b_eye_traces_full.npz", t_UI=T_UI, pos_level_I=POS, **FULL)

# ----------------------------------------------------------------------------------------------------------------------------------------- impairment stack
say("")
say("PENALTY STACK (worst echo phase of each subset, I and Q rails joint, optical dB).  Factors: echo, arm power ratio, metaatom phase spread, common DLI phase. HEADLINE = sequential addition in the order arm ratio, atom phase, echo, DLI phase.")
FACT = ["echo", "arm", "atom", "dli"]


def pen_subset(cid_params, S):
    wkey, e_db, phi_key = cid_params
    if wkey is None:
        return 0.0
    phi = M.PHI_CH[phi_key] if phi_key else 0.0
    ptp = M.PTP_CH[wkey]
    p = {"a": 1.0, "b": (BA_CH[wkey] if "arm" in S else 1.0), "nu": (M.nu_of(e_db) if "echo" in S else 0.0),
         "W": (M.spread_W(ptp) if "atom" in S else 1.0), "phi": (math.radians(phi) if "dli" in S else 0.0)}
    RI, RQ, RJ = joint_R(p)
    return float(M.pen(RJ.min()))


STACK = []
for (cid, title, wkey, e_db, bcase, phi_key), srow in zip(CASES, SUMMARY):
    if wkey is None:
        continue
    cp = (wkey, e_db, phi_key)
    memo = {}

    def pen_s(S):
        k = frozenset(S)
        if k not in memo:
            memo[k] = pen_subset(cp, k)
        return memo[k]

    total = pen_s(FACT)
    assert abs(total - srow["pen_worst"]) < 1e-12, (cid, total, srow["pen_worst"])
    # (1) sequential, order echo -> arm -> atom -> dli  (the order of the brief's legend)
    def sequential(order):
        cum, prev, out = [], 0.0, {}
        S = []
        for f in order:
            S.append(f)
            v = pen_s(S)
            out[f] = v - prev
            prev = v
        return out

    seq = sequential(["arm", "atom", "echo", "dli"])                 # HEADLINE
    seq_e1 = sequential(["echo", "arm", "atom", "dli"])              # echo first (the arm term becomes negative: the arm imbalance lowers the echo)
    seq_rev = sequential(["dli", "arm", "atom", "echo"])             # DLI phase first
    iso = {f: pen_s([f]) for f in FACT}
    shap = {f: 0.0 for f in FACT}
    perms = list(itertools.permutations(FACT))
    for pm in perms:
        prev, S = 0.0, []
        for f in pm:
            S.append(f)
            v = pen_s(S)
            shap[f] += (v - prev) / len(perms)
            prev = v
    res_seq = total - sum(seq.values())
    res_rev = total - sum(seq_rev.values())
    res_e1 = total - sum(seq_e1.values())
    res_shap = total - sum(shap.values())
    res_iso = total - sum(iso.values())
    assert abs(res_seq) < 1e-12 and abs(res_rev) < 1e-12 and abs(res_shap) < 1e-12 and abs(res_e1) < 1e-12
    # closed-form checks: arm alone = 10 log10((1 + r^2) / (2 r)); echo and everything else by the model
    assert abs(iso["arm"] - (-10 * math.log10(M.cf_arm(BA_CH[wkey])))) < 1e-9
    STACK.append(dict(case_id=cid, title=title, wavelength_nm=wkey, bias_case=bcase, total=total, seq=seq, seq_e1=seq_e1, seq_rev=seq_rev, res_e1=res_e1, iso=iso, shap=shap, res_seq=res_seq, res_rev=res_rev,
                      res_shap=res_shap, res_iso=res_iso, label=("DERIVED" if wkey in ("1550.0", "1546.003", "1554.017") else "ESTIMATE")))
    say("  %-28s total %.4f | sequential(arm,atom,echo,dli): %.4f %.4f %.4f %.4f residual %+.1e | echo-first (echo,arm,atom,dli): %.4f %.4f %.4f %.4f | isolated: %.4f %.4f %.4f %.4f interaction %+.4f | Shapley: %.4f %.4f %.4f %.4f" % (
        cid, total, seq["arm"], seq["atom"], seq["echo"], seq["dli"], res_seq, seq_e1["echo"], seq_e1["arm"], seq_e1["atom"], seq_e1["dli"], iso["echo"], iso["arm"], iso["atom"], iso["dli"], res_iso,
        shap["echo"], shap["arm"], shap["atom"], shap["dli"]))

hdr = ["case_id", "title", "wavelength_nm", "bias_case", "label", "total_dB",
       "seq_echo_dB", "seq_arm_ratio_dB", "seq_atom_phase_dB", "seq_dli_phase_dB", "seq_sum_dB", "seq_residual_dB",
       "echofirst_echo_dB", "echofirst_arm_ratio_dB", "echofirst_atom_phase_dB", "echofirst_dli_phase_dB", "echofirst_residual_dB",
       "iso_echo_dB", "iso_arm_ratio_dB", "iso_atom_phase_dB", "iso_dli_phase_dB", "iso_sum_dB", "iso_interaction_residual_dB",
       "shapley_echo_dB", "shapley_arm_ratio_dB", "shapley_atom_phase_dB", "shapley_dli_phase_dB", "shapley_residual_dB",
       "dlifirst_dli_dB", "dlifirst_arm_ratio_dB", "dlifirst_atom_phase_dB", "dlifirst_echo_dB", "dlifirst_residual_dB"]
with io.open(HERE + r"\fig7b_penalty_stack.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    f6 = lambda v: "%.6f" % (0.0 if abs(v) < 5e-13 else v)             # noqa: E731  (no negative zeros)
    for s in STACK:
        w.writerow([s["case_id"], s["title"], s["wavelength_nm"], s["bias_case"], s["label"], f6(s["total"]),
                    f6(s["seq"]["echo"]), f6(s["seq"]["arm"]), f6(s["seq"]["atom"]), f6(s["seq"]["dli"]), f6(sum(s["seq"].values())), "%.2e" % s["res_seq"],
                    f6(s["seq_e1"]["echo"]), f6(s["seq_e1"]["arm"]), f6(s["seq_e1"]["atom"]), f6(s["seq_e1"]["dli"]), "%.2e" % s["res_e1"],
                    f6(s["iso"]["echo"]), f6(s["iso"]["arm"]), f6(s["iso"]["atom"]), f6(s["iso"]["dli"]), f6(sum(s["iso"].values())), f6(s["res_iso"]),
                    f6(s["shap"]["echo"]), f6(s["shap"]["arm"]), f6(s["shap"]["atom"]), f6(s["shap"]["dli"]), "%.2e" % s["res_shap"],
                    f6(s["seq_rev"]["dli"]), f6(s["seq_rev"]["arm"]), f6(s["seq_rev"]["atom"]), f6(s["seq_rev"]["echo"]), "%.2e" % s["res_rev"]])

shdr = ["case_id", "title", "wavelength_nm", "bias_case", "label", "common_DLI_phase_deg", "echo_dB", "arm_ratio_b_over_a", "atom_ptp_deg", "atom_visibility",
        "worst_echo_phase_deg", "penalty_worst_dB", "penalty_best_dB", "penalty_mean_dB", "EO_over_EO_ideal_worst", "bar_I_lo_at_t0", "bar_I_hi_at_t0", "n_traces",
        "coverage_radius_Linf", "reproduction_residual_vs_eye_penalty_11ch_dB"]
with io.open(HERE + r"\fig7b_eye_summary.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(shdr)
    for r in SUMMARY:
        lab = "model (ideal)" if not r["wavelength_nm"] else ("DERIVED (echo MEASURED)" if r["wavelength_nm"] in ("1550.0", "1546.003", "1554.017") else "ESTIMATE (echo ESTIMATE; spread and phase MEASURED / DERIVED)")
        w.writerow([r["case_id"], r["title"], r["wavelength_nm"], r["bias_case"], lab, "%+.4f" % r["phi_deg"], ("%.4f" % r["echo_dB"]) if r["wavelength_nm"] else "",
                    "%.5f" % r["ba"] if r["wavelength_nm"] else "", "%.4f" % r["ptp_deg"] if r["wavelength_nm"] else "", "%.5f" % r["vis"] if r["wavelength_nm"] else "",
                    "%.1f" % r["chi_deg"], "%.4f" % r["pen_worst"], "%.4f" % r["pen_best"], "%.4f" % r["pen_mean"], "%.6f" % r["eo_joint"], "%.4f" % r["bar_lo"], "%.4f" % r["bar_hi"],
                    r["n_unique"], "%.4f" % r["max_dev"], "%.1e" % r["table_residual"]])

# ----------------------------------------------------------------------------------------------------------------------------------------- needle tests
# (1) a wrong echo (+0.1 dB) must change the 1554.017 fixed-bias penalty beyond the reproduction tolerance
pn = M.case_params(M.PHI_CH["1554.017"], "uniform", eps_db=ECHO_A[2] + 0.1, ptp_deg=M.PTP_CH["1554.017"], ba=BA_CH["1554.017"])
dn = float(M.pen(joint_R(pn)[2].min())) - [s for s in SUMMARY if s["case_id"] == "device_1554.017_fixed_bias"][0]["pen_worst"]
say("")
say("NEEDLE 1: echo + 0.1 dB at 1554.017 nm changes the worst-chi penalty by %+.4f dB (tolerance 1e-12): %s" % (dn, "caught" if abs(dn) > 1e-6 else "MISSED"))
assert abs(dn) > 1e-6
# (2) a trace file perturbed by 0.01 must break the read-back of the bar
cid0 = "device_1546.003_fixed_bias"
rep, cnt, lvl = UNIQUE[cid0]
tr0 = FULL["tr_" + cid0].astype(float)
bar_full = (tr0[POS, OS].min(), tr0[~POS, OS].max())
bar_u = (rep[lvl, OS].min(), rep[~lvl, OS].max())
say("bar from the full 4096 windows %.4f / %.4f, from the representative traces %.4f / %.4f (difference %.4f / %.4f)" % (bar_full[0], bar_full[1], bar_u[0], bar_u[1],
                                                                                                          bar_full[0] - bar_u[0], bar_full[1] - bar_u[1]))
assert abs(bar_full[0] - bar_u[0]) < 1e-6 and abs(bar_full[1] - bar_u[1]) < 1e-6      # float32 store of the full windows
# read-back of the CSV on disk
with io.open(HERE + r"\fig7b_eye_%s.csv" % cid0, encoding="utf-8", newline="") as f:
    rd = list(csv.reader(f))
arr = np.array([[float(x) for x in r] for r in rd[1:]])
assert arr.shape == (65, rep.shape[0] + 1) and abs(arr[32, 0]) < 1e-12 and abs(arr[0, 0] + 1.0) < 1e-12 and abs(arr[-1, 0] - 1.0) < 1e-12
assert np.max(np.abs(arr[:, 1:] - rep.T)) < 5.1e-5
say("CSV read-back (%s): shape %s, max |difference| to the in-memory traces %.2e (4-decimal rounding)" % (cid0, arr.shape, np.max(np.abs(arr[:, 1:] - rep.T))))
arr2 = arr.copy()
arr2[10, 3] += 0.01
assert np.max(np.abs(arr2[:, 1:] - rep.T)) > 5.1e-5
say("NEEDLE 2: a 0.01 change of one stored sample breaks the read-back: caught")

# ----------------------------------------------------------------------------------------------------------------------------------------- quick-look preview (not a deliverable)
import matplotlib                                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                  # noqa: E402
fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.6), dpi=200)
for ax, (cid, title, wkey, e_db, bcase, phi_key), srow in zip(axs.ravel(), CASES, SUMMARY):
    rep, cnt, lvl = UNIQUE[cid]
    for u in range(rep.shape[0]):
        ax.plot(T_UI, rep[u], color="#474BF3", lw=0.3, alpha=min(1.0, 0.04 + 0.5 * cnt[u] / cnt.max()))
    ax.plot([0, 0], [srow["bar_lo"], srow["bar_hi"]], color="#E2702A", lw=0.8)
    ax.set_title("%s  %s  %.3f dB" % (title, bcase, srow["pen_worst"]), fontsize=6)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.95, 1.25)
    ax.tick_params(labelsize=5)
fig.tight_layout()
fig.savefig(HERE + r"\fig7b_eye_preview.png")
say("preview written; elapsed %.1f s; RSS %.2f GB" % (time.time() - T0, psutil.Process().memory_info().rss / 1e9))
io.open(HERE + r"\g7_stdout.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
