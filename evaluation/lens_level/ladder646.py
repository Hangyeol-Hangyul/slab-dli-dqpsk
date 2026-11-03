r"""ladder646.py -- sim646 [TM] lens ladder readout (TM-2 palette, 77.5 = f8176952).  numpy only, NO solver.
Same evaluator definitions as readout_261003_1320\tm_ladder\lens_ladder_eval_sim622.py (beta = a_meas / a_pred per arm per carrier shift, the
carrier translation +-120 deg per shift removed, shifts averaged coherently; log|beta| and arg beta linear in 1/P; beta = 0 below 2 Lambda;
tail 'decay' or 'held'; L = |sum_PD conj(E_A) E_B| / sum_PD |E_0|^2 on the exact 8 x 8 um box, one quadrant).
Controls first: (C1) this code on sim622 reproduces the stored lens_ladder_eval_sim622.json bit for bit; (C2) same for sim585.
Classes of sim646 that are not complete yet are taken from sim622 and LABELLED (mixed ladder, [E]).  Run:  python -B ladder646.py
Writes ladder646_result.json beside this file; every printed value is also in stdout."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import sys, json, glob, math, cmath, time
sys.dont_write_bytecode = True
import numpy as np

HERE = r"<WORK_ROOT>\readout_261005_evening\scratch_tm646"
SC = r"<WORK_ROOT>\scdc_scratch_261001"
REV = SC + r"\wf_lpa\lens\review\review_net_f.py"
src = open(REV, encoding="utf-8").read()
ns = {}
exec(compile(src.split("rows = []")[0], "rnf", "exec"), ns)
LAMG, LAM, rho, A, K0 = ns["LAMG"], ns["LAM"], ns["rho"], ns["A"], ns["K0"]
kz2, n, npd, Mx, DT, pin = ns["kz2"], ns["n"], ns["npd"], ns["Mx"], ns["DT"], ns["pin"]

R646 = r"<DATA_ROOT>\Result\261006_scdc_tm077f817b"
R622 = r"<DATA_ROOT>\Result\261003_scdc_tmlpag62"
R585 = r"<DATA_ROOT>\Result\261001_scdc_tmlpaswap2"
CLASSES = [(6, "x06"), (9, "x09"), (12, "x12"), (15, "x15"), (18, "x18"), (24, "x24"), (36, "x36")]
NEED = {6: 3, 9: 3, 12: 3, 15: 3, 18: 3, 24: 3, 36: 1}
GATE = -0.65
T0 = time.time()


def say(s=""):
    print("[%6.1fs] %s" % (time.time() - T0, s), flush=True)


def cell_files(root, pat, c):
    return sorted(glob.glob(os.path.join(root, pat % c, "superramp_result.json")))


def load_cells(root, pat):
    """-> dict cls -> list of per-shift dicts (raw + shift-removed beta, amplitudes, echo, fid, cost)"""
    out = {}
    for M, c in CLASSES:
        rows = []
        for p in cell_files(root, pat, c):
            d = json.load(open(p, encoding="utf-8")); r = d["needle_results"]["production"]; L = r["lpa"]
            kA = [k for k in L if k.startswith("armA_s") and L[k]["order"] == r["target_order_A"]][0]
            kB = [k for k in L if k.startswith("armB_s") and L[k]["order"] == r["target_order_B"]][0]
            s = r["sc_shift"]
            aA = complex(*r["amp_tA_target"]); aB = complex(*r["amp_tB_target"])
            pA = complex(L[kA]["a_pred_re"], L[kA]["a_pred_im"]); pB = complex(L[kB]["a_pred_re"], L[kB]["a_pred_im"])
            bA = aA / pA * cmath.exp(-2j * math.pi * s / 3); bB = aB / pB * cmath.exp(+2j * math.pi * s / 3)
            rows.append(dict(path=p, M=M, cls=c, sh=s, aA=aA, aB=aB, pA=pA, pB=pB, bA=bA, bB=bB, echo_db=r["echo_p2_db"], fid=r["fid_l3"], fidm=r["fid_l3_m"],
                             wall_s=r["wall_s"], rss_mb=r["rss_after_mb"], nG=r["nG_actual"], A_atoms=r["atoms_A_deg"], n_atoms=len(r["atom_paths"]),
                             propA=r["propagates_tA_target"], propB=r["propagates_tB_target"], T_A=r["T_total_A"], T_B=r["T_total_B"]))
        out[c] = rows
    return out


def ladder_from(cells, sel=None):
    """pts = [(P, bA_mean, bB_mean, n_shifts)] for the classes in sel (default all classes that have cells)"""
    pts = []
    for M, c in CLASSES:
        if sel is not None and c not in sel:
            continue
        rs = sorted(cells.get(c, []), key=lambda r: r["sh"])
        if rs:
            pts.append((M / 3.0, sum(r["bA"] for r in rs) / len(rs), sum(r["bB"] for r in rs) / len(rs), len(rs)))
    return pts


def model(P, Pm, bm, tail):
    db = 20 * np.log10(np.abs(bm)); ar = np.unwrap(np.angle(bm))
    x = 1.0 / Pm[::-1]
    D = np.interp(1.0 / P, x, db[::-1]); F = np.interp(1.0 / P, x, ar[::-1])
    last = Pm[-1]
    if tail == "decay":
        D = np.where(P > last, db[-1] * last / P, D); F = np.where(P > last, ar[-1] * last / P, F)
    else:
        D = np.where(P > last, db[-1], D); F = np.where(P > last, ar[-1], F)
    b = 10 ** (D / 20.0) * np.exp(1j * F)
    return np.where(P < 2 * LAMG, 0.0, b)


def focal(E0, f):
    H = np.where(kz2 >= 0, np.exp(1j * np.sqrt(np.clip(kz2, 0, None)) * f), 0.0)
    Ep = np.zeros((npd, npd), dtype=complex); Ep[:n, :n] = E0
    return (Mx @ (np.fft.fft2(Ep) * H) @ Mx.T) / (npd * npd)


_CACHE = {}


def ref_field(f):
    if f not in _CACHE:
        P = LAM * np.sqrt(rho ** 2 + f ** 2) / np.maximum(rho, 1e-9)
        ph = -K0 * (np.sqrt(rho ** 2 + f ** 2) - f)
        E0 = focal(A * np.exp(1j * ph), f)
        c0 = float(np.sum(np.abs(E0) ** 2) * DT * DT) / pin
        _CACHE[f] = (P, ph, c0)
    return _CACHE[f]


def evaluate(pts, f, tail):
    """-> (ov, c0, L_dB) for the ladder pts at design focal length f"""
    P, ph, c0 = ref_field(f)
    Pm = np.array([p[0] for p in pts]) * LAMG
    EA = focal(A * model(P, Pm, np.array([p[1] for p in pts]), tail) * np.exp(1j * ph), f)
    EB = focal(A * model(P, Pm, np.array([p[2] for p in pts]), tail) * np.exp(1j * ph), f)
    ov = abs(np.sum(np.conj(EA) * EB)) * DT * DT / pin
    return float(ov), float(c0), float(10 * np.log10(ov / c0))


def all_L(pts):
    res = {}
    for f in (136, 200, 250):
        for tail in ("decay", "held"):
            res["%d_%s" % (f, tail)] = evaluate(pts, float(f), tail)
    return res


def fmtL(res, f):
    return "f%3d decay %+.4f held %+.4f" % (f, res["%d_decay" % f][2], res["%d_held" % f][2])


OUT = {}
say("loading cells")
C646 = load_cells(R646, "261005_sim646_tm077f817b_%s_s-1_sh*")
C622 = load_cells(R622, "261003_sim622_tmlpag62_%s_s-1_sh*")
C585 = load_cells(R585, "*_%s_s-1_sh*")
for nm, C in (("sim646", C646), ("sim622", C622), ("sim585", C585)):
    say("%s cells found: %s" % (nm, {c: len(C[c]) for _, c in CLASSES}))
OUT["cells_found"] = {nm: {c: len(C[c]) for _, c in CLASSES} for nm, C in (("sim646", C646), ("sim622", C622), ("sim585", C585))}

# ---------------------------------------------------------------- controls
st622 = json.load(open(SC + r"\readout_261003_1320\tm_ladder\lens_ladder_eval_sim622.json", encoding="utf-8"))
st585 = json.load(open(SC + r"\lens_purity_261005\lens_ladder_eval_sim585.json", encoding="utf-8"))
ctl = {}
for nm, C, st in (("C1 sim622", C622, st622), ("C2 sim585", C585, st585)):
    pts = ladder_from(C)
    full = [p for p in pts if p[3] == 3]
    worst = 0.0
    for f in (136, 200, 250):
        for tail in ("decay", "held"):
            for key, S in (("all", pts), ("3-s", full)):
                skey = "%d_%s_%s" % (f, key, tail)
                ov, c0, L = evaluate(S, float(f), tail)
                ref = st[skey]
                worst = max(worst, abs(ov - ref[0]), abs(c0 - ref[1]))
    ctl[nm] = worst
    say("%s: max abs diff of (ov, c0) vs stored json over 12 keys = %.3e  %s" % (nm, worst, "PASS" if worst == 0.0 else "FAIL (not bit-identical)"))
OUT["controls"] = ctl

# ---------------------------------------------------------------- per-cell table (sim646 vs sim622, same class and shift)
say("---- per cell: sim646 vs sim622 (same class, same shift; blas_threads 1 both) ----")
rows_cell = []
for M, c in CLASSES:
    d622 = {r["sh"]: r for r in C622[c]}
    for r in sorted(C646[c], key=lambda r: r["sh"]):
        q = d622.get(r["sh"])
        row = dict(cls=c, M=M, sh=r["sh"], absTA=abs(r["aA"]), absTB=abs(r["aB"]), bA_abs=abs(r["bA"]), bA_deg=math.degrees(cmath.phase(r["bA"])), bB_abs=abs(r["bB"]),
                   bB_deg=math.degrees(cmath.phase(r["bB"])), echo_db=r["echo_db"], fid=r["fid"], fidm=r["fidm"], wall_h=r["wall_s"] / 3600.0, rss_gb=r["rss_mb"] / 1000.0,
                   n77=sum(1 for a in r["A_atoms"] if abs(a - 69.4199611040033) < 1e-6), nG=r["nG"])
        if q:
            row.update(q_absTA=abs(q["aA"]), q_absTB=abs(q["aB"]), q_bA_abs=abs(q["bA"]), q_bA_deg=math.degrees(cmath.phase(q["bA"])), q_bB_abs=abs(q["bB"]),
                       q_bB_deg=math.degrees(cmath.phase(q["bB"])), q_echo_db=q["echo_db"], q_fid=q["fid"], q_fidm=q["fidm"],
                       d_bA_dB=20 * math.log10(abs(r["bA"]) / abs(q["bA"])), d_bB_dB=20 * math.log10(abs(r["bB"]) / abs(q["bB"])),
                       d_bA_deg=(math.degrees(cmath.phase(r["bA"] / q["bA"]))), d_bB_deg=(math.degrees(cmath.phase(r["bB"] / q["bB"]))),
                       d_echo_db=r["echo_db"] - q["echo_db"], d_fid=r["fid"] - q["fid"], d_fidm=r["fidm"] - q["fidm"], q_wall_h=q["wall_s"] / 3600.0, q_rss_gb=q["rss_mb"] / 1000.0)
        rows_cell.append(row)
        say("%s sh%d  |tA| %.5f |tB| %.5f  betaA %.4f@%+6.1f betaB %.4f@%+6.1f | vs622 dA %+.3f dB %+.1f deg  dB %+.3f dB %+.1f deg | echo %.2f (%+.2f) fid %.4f (%+.4f) fid_m %.4f (%+.4f) | n77 %d wall %.2f h rss %.1f GB" % (
            c, r["sh"], row["absTA"], row["absTB"], row["bA_abs"], row["bA_deg"], row["bB_abs"], row["bB_deg"], row.get("d_bA_dB", float("nan")), row.get("d_bA_deg", float("nan")),
            row.get("d_bB_dB", float("nan")), row.get("d_bB_deg", float("nan")), row["echo_db"], row.get("d_echo_db", float("nan")), row["fid"], row.get("d_fid", float("nan")),
            row["fidm"], row.get("d_fidm", float("nan")), row["n77"], row["wall_h"], row["rss_gb"]))
OUT["cells"] = rows_cell

# ---------------------------------------------------------------- per-class (3-shift coherent) beta, sim646 vs sim622 vs sim585
say("---- per class (3-shift coherent mean) ----")
pts646 = ladder_from(C646)
pts622 = ladder_from(C622)
pts585 = ladder_from(C585)
d622 = {round(p[0] * 3): p for p in pts622}
d585 = {round(p[0] * 3): p for p in pts585}
cls_rows = []
for p in pts646:
    M = round(p[0] * 3)
    q = d622[M]; z = d585[M]
    row = dict(M=M, P=p[0], n_sh646=p[3], bA=abs(p[1]), pA=math.degrees(cmath.phase(p[1])), bB=abs(p[2]), pB=math.degrees(cmath.phase(p[2])),
               ov_dB=10 * math.log10(abs(p[1]) * abs(p[2])), q_bA=abs(q[1]), q_pA=math.degrees(cmath.phase(q[1])), q_bB=abs(q[2]), q_pB=math.degrees(cmath.phase(q[2])),
               q_ov_dB=10 * math.log10(abs(q[1]) * abs(q[2])), z_bA=abs(z[1]), z_bB=abs(z[2]), z_ov_dB=10 * math.log10(abs(z[1]) * abs(z[2])),
               d_ov_dB=10 * math.log10(abs(p[1]) * abs(p[2])) - 10 * math.log10(abs(q[1]) * abs(q[2])))
    cls_rows.append(row)
    say("P %5.2f L (M %2d) n646 %d  betaA %.4f@%+6.2f betaB %.4f@%+6.2f ov %+.3f dB | sim622 %.4f@%+6.2f %.4f@%+6.2f ov %+.3f | d_ov %+.3f dB | sim585 ov %+.3f" % (
        row["P"], M, row["n_sh646"], row["bA"], row["pA"], row["bB"], row["pB"], row["ov_dB"], row["q_bA"], row["q_pA"], row["q_bB"], row["q_pB"], row["q_ov_dB"], row["d_ov_dB"], row["z_ov_dB"]))
OUT["classes"] = cls_rows

# ---------------------------------------------------------------- lens L: sim622 (control, as stored), sim646 mixed ladder, sensitivity
say("---- lens L (beta only; deficit added further down) ----")
have646 = {c for M, c in CLASSES if len(C646[c]) >= NEED[M]}
missing = [c for M, c in CLASSES if c not in have646]
say("classes complete in sim646: %s ; still missing: %s" % (sorted(have646), missing))
OUT["have646"] = sorted(have646); OUT["missing646"] = missing

mixed = []
src_of = {}
for M, c in CLASSES:
    if c in have646:
        rs = sorted(C646[c], key=lambda r: r["sh"]); src_of[c] = "sim646"
    else:
        rs = sorted(C622[c], key=lambda r: r["sh"]); src_of[c] = "sim622 (substitute)"
    mixed.append((M / 3.0, sum(r["bA"] for r in rs) / len(rs), sum(r["bB"] for r in rs) / len(rs), len(rs)))
OUT["mixed_source"] = src_of
L_622 = all_L(pts622)
L_585 = all_L(pts585)
L_mix = all_L(mixed)
OUT["L"] = dict(sim622=L_622, sim585=L_585, sim646_mixed=L_mix)
say("sim585 (TM-1 frozen) : " + " | ".join(fmtL(L_585, f) for f in (136, 200, 250)))
say("sim622 (TM-2, old 77.5): " + " | ".join(fmtL(L_622, f) for f in (136, 200, 250)))
say("sim646 mixed (%s): " % ", ".join("%s=%s" % (c, src_of[c][:6]) for _, c in CLASSES) + " | ".join(fmtL(L_mix, f) for f in (136, 200, 250)))
say("   delta mixed - sim622: " + " | ".join("f%d decay %+.4f held %+.4f" % (f, L_mix["%d_decay" % f][2] - L_622["%d_decay" % f][2], L_mix["%d_held" % f][2] - L_622["%d_held" % f][2]) for f in (136, 200, 250)))

# per-class leave-one-in (swap ONE sim646 class into the sim622 ladder) and leave-one-out (swap ONE class of the mixed ladder back to sim622)
say("---- per-class swap: sim622 ladder with ONE class taken from sim646 (delta L, f200 decay / held; f136 decay) ----")
swap = {}
for i, (M, c) in enumerate(CLASSES):
    if c not in have646:
        continue
    pts = list(pts622)
    j = [k for k, p in enumerate(pts) if round(p[0] * 3) == M][0]
    p646 = [p for p in pts646 if round(p[0] * 3) == M][0]
    pts[j] = p646
    r = all_L(pts)
    swap[c] = dict(d200_decay=r["200_decay"][2] - L_622["200_decay"][2], d200_held=r["200_held"][2] - L_622["200_held"][2], d136_decay=r["136_decay"][2] - L_622["136_decay"][2],
                   d136_held=r["136_held"][2] - L_622["136_held"][2], d250_decay=r["250_decay"][2] - L_622["250_decay"][2])
    say("  %s (P %.2f L): f200 %+.4f / %+.4f  f136 %+.4f / %+.4f  f250 %+.4f" % (c, M / 3.0, swap[c]["d200_decay"], swap[c]["d200_held"], swap[c]["d136_decay"], swap[c]["d136_held"], swap[c]["d250_decay"]))
OUT["swap_one_class"] = swap

# sensitivity of the missing points: x24 and x36 betas perturbed on the mixed ladder (both arms, amplitude in dB, phase in deg)
say("---- sensitivity of the mixed ladder to the missing points (x24 / x36), amplitude both arms +-0.30 dB and phase +-3 deg ----")
sens = {}
for c in ("x24", "x36"):
    M = {cc: m for m, cc in CLASSES}[c]
    j = [k for k, p in enumerate(mixed) if round(p[0] * 3) == M][0]
    for lab, damp_db, dph in (("amp -0.30 dB", -0.30, 0.0), ("amp +0.30 dB", +0.30, 0.0), ("phase +3 deg", 0.0, 3.0), ("phase -3 deg", 0.0, -3.0)):
        pts = list(mixed)
        p = pts[j]
        fa = 10 ** (damp_db / 20.0) * cmath.exp(1j * math.radians(dph))
        pts[j] = (p[0], p[1] * fa, p[2] * fa, p[3])
        r = all_L(pts)
        sens["%s %s" % (c, lab)] = dict(d200_decay=r["200_decay"][2] - L_mix["200_decay"][2], d200_held=r["200_held"][2] - L_mix["200_held"][2], d136_decay=r["136_decay"][2] - L_mix["136_decay"][2])
        say("  %s %-13s: f200 decay %+.4f held %+.4f  f136 decay %+.4f" % (c, lab, sens["%s %s" % (c, lab)]["d200_decay"], sens["%s %s" % (c, lab)]["d200_held"], sens["%s %s" % (c, lab)]["d136_decay"]))
OUT["sensitivity_missing"] = sens

json.dump(OUT, open(HERE + r"\ladder646_result.json", "w", encoding="utf-8"), indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else str(o))
say("wrote ladder646_result.json")
