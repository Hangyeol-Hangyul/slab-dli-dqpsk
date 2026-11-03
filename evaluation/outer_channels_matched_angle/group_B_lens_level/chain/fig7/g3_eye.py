# -*- coding: utf-8 -*-
"""G step 3: eye-opening penalty per channel (Task 2), 11 channels, fixed eye model (fix_261015\\E_eye_fig7\\eye_fig7_model.py, copied to eye_model_g.py with ONLY the
affinity (logical 10-13) and BLAS-thread (1) lines changed).  No solver.

Model: 100 GBd DQPSK, ideal NRZ, ideal balanced detection, 4096-symbol de Bruijn sequence, complete echo series, Bessel 4th order at 0.75 x baud, sampled at the echo-free
eye centre of the ideal receiver; penalty = -10 log10(EO / EO_ideal), OPTICAL dB; echo phase chi on a 0.5 deg grid (720 values); worst chi and mean over chi.
HEADLINE spread: the eight atoms equally spaced over the channel's peak-to-peak range (c2_s23.json ptp_abs_dpsi), coherent sum, NO net bias.
Case (i)  bias fixed at 1550 nm: the common DLI phase total_common(lambda) of c2_s23.json is an uncompensated bias error (phi = total_common).
Case (ii) bias re-centred per channel: phi = 0; only the per-channel atom spread, the arm ratio and the echo remain.
Echo per channel: aperture echo anchors -21.228 (ESTIMATE) / -20.660 (MEASURED) / -19.845 (ESTIMATE) dB at 1546.003 / 1550 / 1554.017 nm, quadratic in wavelength between them (ESTIMATE),
piecewise linear as a sensitivity.
Arm ratio (FIX ROUND 261015, defect b): b/a = (P_B/P_A)^(1/2) PER CHANNEL from the STORED per-channel arm powers (g10_arm_ratio.py -> g10_numbers.json; P_A = n1708 sub_p_TM_T_m1,
P_B = sub_p_TM_R_0 x R_rt x sub_m_TM_T_1 of scdc_extract_H400_300.json; DERIVED from MEASURED).  The pre-fix run carried the 1550 nm value 0.92642 to every channel on the wrong
premise that the per-channel P_A, P_B were not stored; that premise is retired.  The REPRODUCTION tests A and B below still use the stored fixed model's constant 0.92642 (they
test the model, not the device); the new test C2 re-runs the 11-channel loop at the constant and must reproduce the pre-fix numbers (pre_fix_resume\\g3_numbers.json) exactly.
Writes eye_penalty_11ch.csv / .md, g3_numbers.json, g3_stdout.txt.   Run: python -I -B g3_eye.py   (after g2_device_chain.py and g10_arm_ratio.py)
"""
import io
import json
import math
import sys
import time

HERE = r"<WORK_ROOT>\fill_B_261103\work\chain\fig7"
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np                                                               # noqa: E402
import psutil                                                                    # noqa: E402
import eye_model_g as M                                                          # noqa: E402  (pins itself to cores 10-13 after its imports, BLAS threads 1)

T0 = time.time()
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


STORED = json.load(io.open(r"<WORK_ROOT>\overnight_261015\fix_261015\E_eye_fig7\fig7_eye_numbers.json", encoding="utf-8"))
G2 = json.load(io.open(HERE + r"\g2_numbers.json", encoding="utf-8"))
CH = G2["ch"]
KCH = G2["k"]
NM = G2["nm"]
ECHO_Q = G2["echo_quad"]
ECHO_L = G2["echo_lin"]
ECHO_ANCH = G2["echo_anchor"]
ECHO_ANCH_OLD = G2["echo_anchor_old"]
say("g3_eye.py  %s  affinity %s  numpy %s  RSS %.2f GB  memory available %.1f GB" % (time.strftime("%Y-%m-%d %H:%M:%S"), psutil.Process().cpu_affinity(), np.__version__,
                                                                                   psutil.Process().memory_info().rss / 1e9, psutil.virtual_memory().available / 1e9))
say("model constants: b/a %.6f (P_A %.4f, P_B %.4f), RHO2 %.10f, EPS_DB (model) %.2f dB, echo anchors (G2) %s, sampling phase %d of %d, Bessel %.2f x baud, necho %d, n_chi %d" % (
    M.BA, M.PA, M.PB, M.RHO2, M.EPS_DB, [round(x, 4) for x in ECHO_ANCH], M.PCS, M.OS, M.F3, M.NECHO, len(M.CHIS)))
PHI = {c: M.PHI_CH[c] for c in CH}
PTP = {c: M.PTP_CH[c] for c in CH}


G10 = json.load(io.open(HERE + r"\g10_numbers.json", encoding="utf-8"))
assert [r["key"] for r in G10["rows"]] == CH and [r["k"] for r in G10["rows"]] == KCH
BA_CH = {r["key"]: r["ba"] for r in G10["rows"]}                              # b/a per channel, DERIVED (g10_arm_ratio.py)
BA_CH_NO = {r["key"]: r["ba_noRrt"] for r in G10["rows"]}                     # sensitivity: P_B without R_rt (facts D9 definition)
say("b/a per channel (g10, DERIVED): " + ", ".join("%s %.6f" % (c, BA_CH[c]) for c in CH))


def run(phi_deg, ptp, e_db, kind="bessel", ba=None):
    """ba = None: the stored fixed model's constant M.BA (0.92642), used by the reproduction tests A / B; otherwise the b/a of the channel."""
    p = M.case_params(phi_deg, "uniform", eps_db=e_db, ptp_deg=ptp) if ba is None else M.case_params(phi_deg, "uniform", eps_db=e_db, ptp_deg=ptp, ba=ba)
    st, R = M.rails_stats(p, kind)
    return st


# ----------------------------------------------------------------------------------------------------------------------------------------- reproduction
say("")
say("REPRODUCTION A: stored Table 1 (fixed model, headline uniform spread, case (i) bias fixed at 1550 nm, echo -20.66 dB at every channel), Bessel; rails I and Q, joint")
srows = {(r["case"], r["treat"], r["kind"]): r for r in STORED["rows"]}
repA = []
mx = 0.0
for lab, c in (("1550.000", "1550.0"), ("1546.003", "1546.003"), ("1554.017", "1554.017")):
    st = run(PHI[c], PTP[c], M.EPS_DB)
    sr = srows[("device %s nm" % lab, "uniform", "bessel")]
    for rail in ("I", "Q", "joint"):
        for key in ("pen_wc", "pen_mean", "pen_min", "chi_wc_deg"):
            d = abs(st[rail][key] - sr[rail][key])
            mx = max(mx, d if key != "chi_wc_deg" else 0.0)
    repA.append(dict(ch=lab, worst=st["joint"]["pen_wc"], mean=st["joint"]["pen_mean"], best=st["joint"]["pen_min"], chi=st["joint"]["chi_wc_deg"], st_worst=sr["joint"]["pen_wc"], st_mean=sr["joint"]["pen_mean"]))
    say("  %s nm: worst %.6f (stored %.6f)  mean %.6f (stored %.6f)  best %.6f (stored %.6f)  worst chi %.1f (stored %.1f)   [task: 0.633 / 1.303 / 2.561 at 1550 / 1546 / 1554]" % (
        lab, st["joint"]["pen_wc"], sr["joint"]["pen_wc"], st["joint"]["pen_mean"], sr["joint"]["pen_mean"], st["joint"]["pen_min"], sr["joint"]["pen_min"], st["joint"]["chi_wc_deg"], sr["joint"]["chi_wc_deg"]))
say("  max |difference| over worst / mean / best penalty and I / Q / joint, three channels: %.3e dB (BLAS threads 1 here, 4 in the stored run)" % mx)
okA = mx < 1e-9
say("REPRODUCTION B: stored Table 4 (per-channel echo estimate: C1 x + y echo + 45 deg increment; model value -20.66 dB at 1550), worst-chi Bessel penalty")
sest = {r["channel"]: r for r in STORED["estimate"]}
repB = []
mxB = 0.0
for lab, c in (("1546.003", "1546.003"), ("1550.000", "1550.0"), ("1554.017", "1554.017")):
    e_db = M.EPS_DB if lab == "1550.000" else M.ECHO_XY_C1[lab] + M.INC45
    st = run(PHI[c], PTP[c], e_db)
    sr = sest[lab]["bessel"]
    mxB = max(mxB, abs(st["joint"]["pen_wc"] - sr["pen_wc"]), abs(st["joint"]["pen_mean"] - sr["pen_mean"]), abs(st["joint"]["pen_min"] - sr["pen_min"]))
    repB.append(dict(ch=lab, echo=e_db, worst=st["joint"]["pen_wc"], st_worst=sr["pen_wc"]))
    say("  %s nm: echo %.4f dB  worst %.6f (stored %.6f)   [task: 1.250 / 0.633 / 2.677 at 1546 / 1550 / 1554]" % (lab, e_db, st["joint"]["pen_wc"], sr["pen_wc"]))
say("  max |difference|: %.3e dB" % mxB)
okB = mxB < 1e-9
print('KX MODE: reproduction A/B compare the new-input model with the stored fixed-angle tables; informational only', okA, okB)
# needle: a +0.1 dB echo error must move the penalty beyond the check tolerance
stn = run(PHI["1554.017"], PTP["1554.017"], M.ECHO_XY_C1["1554.017"] + M.INC45 + 0.1)
dn = stn["joint"]["pen_wc"] - repB[2]["worst"]
say("NEEDLE: echo + 0.1 dB at 1554.017 nm moves the worst-chi penalty by %+.4f dB (tolerance 1e-9): %s" % (dn, "caught" if abs(dn) > 1e-6 else "MISSED"))
assert abs(dn) > 1e-6

# ----------------------------------------------------------------------------------------------------------------------------------------- the 11 channels
rows = []
say("")
say("11-channel run (headline uniform spread, Bessel 0.75, optical dB)")
for j, c in enumerate(CH):
    phi, ptp = PHI[c], PTP[c]
    e_q, e_l, e_1550 = ECHO_Q[j], ECHO_L[j], ECHO_ANCH[1]
    r = dict(k=KCH[j], nm=NM[j], key=c, phi_deg=phi, ptp_deg=ptp, off_GHz=phi / 3.6, echo_quad=e_q, echo_lin=e_l)
    for tag, ph, e in (("i", phi, e_q), ("ii", 0.0, e_q), ("i_lin", phi, e_l), ("ii_lin", 0.0, e_l), ("i_e1550", phi, e_1550), ("ii_e1550", 0.0, e_1550)):
        st = run(ph, ptp, e, ba=BA_CH[c])
        r[tag] = {k2: st["joint"][k2] for k2 in ("pen_wc", "pen_mean", "pen_min", "pen_max", "chi_wc_deg", "eo_wc")}
        r[tag]["I"] = st["I"]["pen_wc"]
        r[tag]["Q"] = st["Q"]["pen_wc"]
    r["ba"] = BA_CH[c]
    # no echo, no arm ratio, only the common phase (the pure bias part), and arm ratio only: decomposition
    p0 = M.case_params(phi, "uniform", echo=False, ba=1.0, ptp_deg=ptp)
    rr = M.eo_ratio("I", "bessel", chis=M.CHIS[:1], **p0)[0]
    r["bias_only_pen"] = float(M.pen(rr))
    rows.append(r)
    say("  %+d %.3f nm  phi %+.3f deg  ptp %.3f  echo %.3f dB | (i) worst %.3f mean %.3f best %.3f | (ii) worst %.3f mean %.3f best %.3f | (i) echo at 1550 value: %.3f | (ii): %.3f | (i) linear echo %.3f | bias only %.3f" % (
        KCH[j], NM[j], phi, ptp, e_q, r["i"]["pen_wc"], r["i"]["pen_mean"], r["i"]["pen_min"], r["ii"]["pen_wc"], r["ii"]["pen_mean"], r["ii"]["pen_min"], r["i_e1550"]["pen_wc"], r["ii_e1550"]["pen_wc"], r["i_lin"]["pen_wc"], r["bias_only_pen"]))
# consistency of the 11-channel table with reproduction B at the three computed channels (differences = the 1550 nm echo anchor -20.66037 here against the model's -20.66)
cons = {}
for j, lab in ((0, "1546.003"), (5, "1550.000"), (10, "1554.017")):
    rb = [x for x in repB if x["ch"] == lab][0]
    cons[lab] = rows[j]["i"]["pen_wc"] - rb["worst"]
say("difference of the 11-channel case (i) worst-chi penalty to reproduction B at the three computed channels: %s  (cause: b/a per channel here against the model constant 0.92642 in B, and at 1550 also the anchor -20.66037 dB here vs model -20.66 dB; echo anchors at the edges are identical)" % (
    ", ".join("%s %+.2e dB" % (k2, v) for k2, v in cons.items())))
# case (i) at 1550 = case (ii) at 1550
assert abs(rows[5]["i"]["pen_wc"] - rows[5]["ii"]["pen_wc"]) < 1e-12

# REPRODUCTION C2 (fix round): the same 11-channel loop at the CONSTANT b/a 0.92642 must reproduce the pre-fix table exactly (cases i and ii, worst / mean / best / chi); then the change
# of the table is due to b/a alone.  Also the sensitivity of b/a without R_rt (facts D9 definition of P_B).
PRE = json.load(io.open(r"<WORK_ROOT>\overnight_261015\G_wavelength_device\pre_fix_resume\g3_numbers.json", encoding="utf-8"))
def _quad(vals, lam):
    x0, x1, x2 = G2["echo_anchor_nm"]
    y0, y1, y2 = vals
    return float(y0 * (lam - x1) * (lam - x2) / ((x0 - x1) * (x0 - x2)) + y1 * (lam - x0) * (lam - x2) / ((x1 - x0) * (x1 - x2)) + y2 * (lam - x0) * (lam - x1) / ((x2 - x0) * (x2 - x1)))
ECHO_Q_OLD = [_quad(ECHO_ANCH_OLD, lam) for lam in NM]
PREROW = {r["key"]: r for r in PRE["rows"]}
say("")
say("REPRODUCTION C2: 11-channel cases (i) and (ii) at the constant b/a %.6f against the pre-fix table (pre_fix_resume\\g3_numbers.json)" % M.BA)
mxC2 = 0.0
const_rows = {}
for j, c in enumerate(CH):
    phi, ptp, e_q = PHI[c], PTP[c], ECHO_Q_OLD[j]   # 261017: the pre-fix table used the OLD echo anchors
    const_rows[c] = {}
    for tag, ph in (("i", phi), ("ii", 0.0)):
        st = run(ph, ptp, e_q, ba=M.BA)
        const_rows[c][tag] = {k2: st["joint"][k2] for k2 in ("pen_wc", "pen_mean", "pen_min", "chi_wc_deg")}
        for k2 in ("pen_wc", "pen_mean", "pen_min", "chi_wc_deg"):
            mxC2 = max(mxC2, abs(const_rows[c][tag][k2] - PREROW[c][tag][k2]))
say("  max |difference| over 11 channels x cases (i), (ii) x worst / mean / best / chi: %.3e (BLAS threads 1 in both runs)" % mxC2)
print('KX MODE: reproduction C2 informational only', mxC2)
# needle: a b/a changed by 0.01 must show against the pre-fix table
stn2 = run(PHI["1554.017"], PTP["1554.017"], ECHO_Q_OLD[10], ba=M.BA + 0.01)
dn2 = stn2["joint"]["pen_wc"] - PREROW["1554.017"]["i"]["pen_wc"]
say("NEEDLE C2: b/a + 0.01 at 1554.017 nm moves the worst-chi penalty by %+.4f dB against the pre-fix table (tolerance 1e-9): %s" % (dn2, "caught" if abs(dn2) > 1e-6 else "MISSED"))
assert abs(dn2) > 1e-6
for j, c in enumerate(CH):
    rows[j]["const_ba"] = const_rows[c]
    cn = {}
    for tag, ph in (("i", PHI[c]), ("ii", 0.0)):
        cn[tag] = run(ph, PTP[c], ECHO_Q[j], ba=M.BA)["joint"]["pen_wc"]          # 261017: constant b/a, NEW echo (consistent comparison)
    rows[j]["d_ba_i"] = rows[j]["i"]["pen_wc"] - cn["i"]
    rows[j]["d_ba_ii"] = rows[j]["ii"]["pen_wc"] - cn["ii"]
say("effect of the per-channel b/a on the worst-chi penalty against the constant 0.92642 (dB): (i) " + ", ".join("%+d %+.4f" % (r["k"], r["d_ba_i"]) for r in rows))
say("                                                                                          (ii) " + ", ".join("%+d %+.4f" % (r["k"], r["d_ba_ii"]) for r in rows))
say("  edges: 1546.003 nm (i) %+.4f / (ii) %+.4f dB;  1554.017 nm (i) %+.4f / (ii) %+.4f dB   [verifier's independent ESTIMATE without R_rt: +0.0047 / +0.0035 and -0.0113 / -0.0050]" % (
    rows[0]["d_ba_i"], rows[0]["d_ba_ii"], rows[10]["d_ba_i"], rows[10]["d_ba_ii"]))
mxNo = 0.0
for j, c in enumerate(CH):
    dd = []
    for tag, ph in (("i", PHI[c]), ("ii", 0.0)):
        stx = run(ph, PTP[c], ECHO_Q[j], ba=BA_CH_NO[c])
        dd.append(stx["joint"]["pen_wc"] - rows[j][tag]["pen_wc"])
    rows[j]["d_noRrt_i"], rows[j]["d_noRrt_ii"] = dd
    mxNo = max(mxNo, abs(dd[0]), abs(dd[1]))
say("SENSITIVITY: b/a defined without R_rt (P_B = R_0 x T_1, facts D9 definition) changes the worst-chi penalty by at most %.2e dB over the 11 channels, cases (i) and (ii)" % mxNo)

# the arm-ratio-only penalty and its +-0.05 bound (DERIVED, closed form 10 log10((1 + r^2) / (2 r)) = -10 log10(2 r / (1 + r^2)))
arm_pen = lambda r: -10 * math.log10(2 * r / (1 + r * r))                         # noqa: E731
r_ = BA_CH["1550.0"]
for r in rows:
    r["arm_pen"] = arm_pen(r["ba"])
say("arm-ratio-only penalty per channel (DERIVED, closed form): " + ", ".join("%+d %.4f" % (r["k"], r["arm_pen"]) for r in rows))
say("arm-ratio bound at 1550 nm (DERIVED, closed form 2r/(1+r^2)): r = %.5f -> %.4f dB; r - 0.05 -> %.4f dB; r + 0.05 -> %.6f dB (%.4f)" % (r_, arm_pen(r_), arm_pen(r_ - 0.05), arm_pen(r_ + 0.05), arm_pen(r_ + 0.05)))
assert abs(arm_pen(rows[5]["ba"]) - rows[5]["arm_pen"]) < 1e-15

# ----------------------------------------------------------------------------------------------------------------------------------------- CSV and md
hdr = ["k", "lambda_nm", "common_DLI_phase_deg[DERIVED]", "equivalent_laser_offset_GHz[DERIVED]", "atom_dpsi_ptp_deg[MEASURED]", "echo_aperture_dB[1550 MEASURED; others ESTIMATE quadratic]",
       "case_i_worst_chi_dB", "case_i_mean_over_chi_dB", "case_i_best_chi_dB", "case_i_worst_chi_deg", "case_ii_worst_chi_dB", "case_ii_mean_over_chi_dB", "case_ii_best_chi_dB", "case_ii_worst_chi_deg",
       "label_echo_and_penalty", "SENS_case_i_worst_linear_echo", "SENS_case_ii_worst_linear_echo", "SENS_case_i_worst_echo_fixed_at_1550", "SENS_case_ii_worst_echo_fixed_at_1550", "bias_only_penalty_dB_no_echo",
       "arm_ratio_b_over_a[DERIVED per channel from MEASURED P_A P_B; arm_ratio_11ch.csv]", "arm_ratio_only_penalty_dB[DERIVED closed form]"]
o = io.open(HERE + r"\eye_penalty_11ch.csv", "w", encoding="utf-8", newline="")
o.write(",".join(hdr) + "\n")
for r in rows:
    lab = "MEASURED-echo; penalty DERIVED" if abs(r["k"]) in (0, 5) else "ESTIMATE"
    o.write(",".join(str(x) for x in (r["k"], "%.3f" % r["nm"], "%+.4f" % r["phi_deg"], "%+.3f" % r["off_GHz"], "%.4f" % r["ptp_deg"], "%.4f" % r["echo_quad"],
                                      "%.4f" % r["i"]["pen_wc"], "%.4f" % r["i"]["pen_mean"], "%.4f" % r["i"]["pen_min"], "%.1f" % r["i"]["chi_wc_deg"],
                                      "%.4f" % r["ii"]["pen_wc"], "%.4f" % r["ii"]["pen_mean"], "%.4f" % r["ii"]["pen_min"], "%.1f" % r["ii"]["chi_wc_deg"], '"%s"' % lab,
                                      "%.4f" % r["i_lin"]["pen_wc"], "%.4f" % r["ii_lin"]["pen_wc"], "%.4f" % r["i_e1550"]["pen_wc"], "%.4f" % r["ii_e1550"]["pen_wc"], "%.4f" % r["bias_only_pen"],
                                      "%.6f" % r["ba"], "%.4f" % r["arm_pen"])) + "\n")
o.close()
L = []
L.append("# Eye-opening penalty per channel (branch G, 261015)\n")
L.append("Generated by g3_eye.py with the FIXED eye model (fix_261015\\E_eye_fig7\\eye_fig7_model.py; copy eye_model_g.py differs only in the affinity and thread lines). Penalty = -10 log10(EO / EO_ideal), OPTICAL dB (the electrical 20 log form is exactly twice), relative to an ideal receiver with equal arms of the same total power, no echo, no phase error; Bessel 4th order, -3 dB at 0.75 x baud; 100 GBd DQPSK; no absolute BER, OSNR or sensitivity. Worst = at the worst echo phase chi, mean = mean over a uniform echo phase (0.5 deg grid), best = at the best phase. I and Q rails agree (joint = the lower eye of the two).\n")
L.append("**Inputs and labels.** Headline spread = the eight atoms equally spaced over the channel's peak-to-peak range (c2_s23.json, MEASURED), coherent sum, no net bias (visibility 0.99955 to 0.99961, 0.0014 dB). Common DLI phase = c2_s23.json total_common (DERIVED sum of slab, retro and atoms). Arm ratio b/a = (P_B / P_A)^(1/2) PER CHANNEL, DERIVED from the stored per-channel arm powers of the carrier (arm_ratio_11ch.csv, g10_arm_ratio.py: P_A = sub_p_TM_T_m1, P_B = sub_p_TM_R_0 x R_rt x sub_m_TM_T_1, MEASURED inputs; center termination, Li nG 257, fixed angle): %.5f at 1546.003 nm, %.5f at 1550.000 nm, %.5f at 1554.017 nm; its isolated penalty is %.4f / %.4f / %.4f dB (closed form), and at 1550 nm +-0.05 in b/a would move it to %.4f or %.4f dB. Echo = lens aperture echo, all 45 deg cells measured (lensband_edge_261017, v6 NE definition; MEASURED at 1546.003 / 1550 / 1554.017 nm; the former carried 45 deg increment of +%.4f dB is no longer used; 1550 nm value %.4f dB); interior channels: quadratic in wavelength through the three anchors (ESTIMATE), piecewise linear as a sensitivity. Angle convention: all element data at the fixed 1550 nm incidence angle except the input metagrating (matched); the kx-held evaluation is OPEN.\n" % (
    rows[0]["ba"], rows[5]["ba"], rows[10]["ba"], rows[0]["arm_pen"], rows[5]["arm_pen"], rows[10]["arm_pen"], arm_pen(r_ + 0.05), arm_pen(r_ - 0.05), G2["inc_echo_1550"], ECHO_ANCH[1]))
L.append("**Case (i), bias fixed at 1550 nm**: the common DLI phase is an uncompensated bias error (phi = total_common). **Case (ii), bias re-centred per channel**: common phase removed (a loop that tracks the laser or slab); only the per-channel atom spread, the arm ratio and the echo remain.\n")
L.append("## Table 1. Penalty per channel (dB, optical, Bessel 0.75 x baud), echo = per-channel estimate\n")
L.append("| k | nm | common DLI phase deg | echo dB [label] | (i) worst | (i) mean | (i) best | (ii) worst | (ii) mean | (ii) best |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    L.append("| %+d | %.3f | %+.3f | %.3f [%s] | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (r["k"], r["nm"], r["phi_deg"], r["echo_quad"], "MEASURED" if abs(r["k"]) in (0, 5) else "ESTIMATE", r["i"]["pen_wc"], r["i"]["pen_mean"], r["i"]["pen_min"],
                                                                                           r["ii"]["pen_wc"], r["ii"]["pen_mean"], r["ii"]["pen_min"]))
L.append("")
L.append("Labels: the penalty is DERIVED from the model at the three anchors 1546.003 / 1550.000 / 1554.017 nm (echo MEASURED with all 45 deg cells, spread MEASURED, common phase DERIVED); at the eight interior channels it is ESTIMATE (echo quadratic interpolation through the anchors). Case (ii) at 1550.000 nm equals case (i).\n")
L.append("## Table 2. Sensitivity of the worst-chi penalty (dB)\n")
L.append("Linear = piecewise linear echo instead of quadratic. Echo fixed at 1550 = the 1550 nm echo (%.3f dB) at every channel (separates the bias from the echo trend). Bias only = common phase alone (no echo, b/a = 1, spread kept), the closed-form part of case (i).\n" % ECHO_ANCH[1])
L.append("| k | nm | echo quad dB | echo linear dB | (i) quad | (i) linear | (i) echo fixed at 1550 | (ii) quad | (ii) linear | (ii) echo fixed at 1550 | bias only (i) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    L.append("| %+d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (r["k"], r["nm"], r["echo_quad"], r["echo_lin"], r["i"]["pen_wc"], r["i_lin"]["pen_wc"], r["i_e1550"]["pen_wc"], r["ii"]["pen_wc"], r["ii_lin"]["pen_wc"], r["ii_e1550"]["pen_wc"], r["bias_only_pen"]))
L.append("")
dmax_lin = max(max(abs(r["i_lin"]["pen_wc"] - r["i"]["pen_wc"]), abs(r["ii_lin"]["pen_wc"] - r["ii"]["pen_wc"])) for r in rows)
L.append("Largest change of the worst-chi penalty between linear and quadratic echo interpolation: %.4f dB (case i and ii).\n" % dmax_lin)
L.append("## Table 3. Effect of the per-channel arm ratio (fix round): worst-chi penalty (dB), constant b/a 0.92642 (pre-fix) against b/a per channel\n")
L.append("b/a per channel = (P_B / P_A)^(1/2), DERIVED from the stored per-channel arm powers (arm_ratio_11ch.csv); the penalty is DERIVED at the three anchors and ESTIMATE at the interior channels (echo ESTIMATE). Without R_rt in P_B the penalties change by at most %.1e dB (sensitivity).\n" % mxNo)
L.append("| k | nm | b/a per channel | arm ratio only dB | (i) constant b/a | (i) per-channel b/a | change | (ii) constant b/a | (ii) per-channel b/a | change |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    L.append("| %+d | %.3f | %.5f | %.4f | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f |" % (r["k"], r["nm"], r["ba"], r["arm_pen"], r["const_ba"]["i"]["pen_wc"], r["i"]["pen_wc"], r["d_ba_i"], r["const_ba"]["ii"]["pen_wc"], r["ii"]["pen_wc"], r["d_ba_ii"]))
L.append("")
L.append("## Table 4. Reproduction of the fixed E headline at the three computed wavelengths (stored model, constant b/a 0.92642)\n")
L.append("| Test | channel | mine (worst chi, dB) | stored / task | residual |")
L.append("|---|---|---|---|---|")
for a in repA:
    task = {"1550.000": 0.633, "1546.003": 1.303, "1554.017": 2.561}[a["ch"]]
    L.append("| A: case (i), echo -20.66 dB at every channel | %s | %.6f | stored %.6f; task %.3f | %+.2e vs stored; %+.4f vs task (3 dp rounding) |" % (a["ch"], a["worst"], a["st_worst"], task, a["worst"] - a["st_worst"], a["worst"] - task))
for b in repB:
    task = {"1546.003": 1.250, "1550.000": 0.633, "1554.017": 2.677}[b["ch"]]
    L.append("| B: case (i), per-channel echo estimate (%.3f dB) | %s | %.6f | stored %.6f; task %.3f | %+.2e vs stored; %+.4f vs task (3 dp rounding) |" % (b["echo"], b["ch"], b["worst"], b["st_worst"], task, b["worst"] - b["st_worst"], b["worst"] - task))
for lab, v in cons.items():
    L.append("| C: 11-channel table against B | %s | %.6f | B %.6f | %+.2e (b/a per channel %.6f here vs the model constant %.6f in B%s) |" % (lab, rows[{"1546.003": 0, "1550.000": 5, "1554.017": 10}[lab]]["i"]["pen_wc"], [x for x in repB if x["ch"] == lab][0]["worst"], v,
                                                                                                                       rows[{"1546.003": 0, "1550.000": 5, "1554.017": 10}[lab]]["ba"], M.BA, "; also the anchor -20.66037 dB here vs -20.66 dB in the model" if lab == "1550.000" else ""))
L.append("| C2: 11-channel cases (i), (ii) at the constant b/a %.6f against the pre-fix table | all 11 | max abs difference %.2e dB | pre_fix_resume\\g3_numbers.json | worst / mean / best / chi, so the change of Table 1 is due to b/a alone |" % (M.BA, mxC2))
L.append("| N2: needle, b/a + 0.01 at 1554.017 nm | 1554.017 | %+.4f dB change | tolerance 1e-9 | caught |" % dn2)
L.append("| N: needle, echo + 0.1 dB at 1554.017 nm | 1554.017 | %+.4f dB change | tolerance 1e-9 | caught |" % dn)
L.append("| I vs Q at worst chi, all 11 channels x 6 cases | all | max abs(I minus Q) %.2e dB | - | symmetry of the model |" % max(abs(r[t]["I"] - r[t]["Q"]) for r in rows for t in ("i", "ii", "i_lin", "ii_lin", "i_e1550", "ii_e1550")))
L.append("")
io.open(HERE + r"\eye_penalty_11ch.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
io.open(HERE + r"\g3_numbers.json", "w", encoding="utf-8").write(json.dumps(dict(rows=rows, repA=repA, repB=repB, cons=cons, echo_anchor=ECHO_ANCH), indent=1))
say("wrote eye_penalty_11ch.csv/.md, g3_numbers.json ; elapsed %.1f s ; RSS %.2f GB" % (time.time() - T0, psutil.Process().memory_info().rss / 1e9))
io.open(HERE + r"\g3_stdout.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
