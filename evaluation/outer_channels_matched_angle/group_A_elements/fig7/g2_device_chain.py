# -*- coding: utf-8 -*-
"""G step 2: device-level chain per channel (Task 1) and DLI phase per channel (Task 3).  Stored results only, no solver.

F(lambda) = (1/2) T(lambda) sqrt(R_rt(lambda)) eta(lambda) S_max C_PD(lambda)          (SI Note S12, si_v2_r12.md line 546)
Writes device_chain_11ch.csv / .md, dli_phase_11ch.csv, g2_numbers.json (used by g3_eye.py) in this folder.   Run: python -I -B g2_device_chain.py
Labels used everywhere: MEASURED (read from a stored simulation), DERIVED (arithmetic on measured values, or the scalar lens model of the evaluator),
ESTIMATE (assumption based: interpolation or the carried-over 45 deg increment), OPEN (missing input).
"""
import io
import json
import math
import os
import sys
import time

HERE = r"<WORK_ROOT>\fill_A_261102\kx_chain\fig7"
C2D = r"<WORK_ROOT>\fill_A_261102\kx_chain\c2"
BJ = r"<WORK_ROOT>\overnight_261015\B_input_channels\b_results.json"
C1J = r"<WORK_ROOT>\overnight_261015\C_lens_analyses\C1_band_xy\lensband_eval_xy_result.json"
C1F = r"<WORK_ROOT>\overnight_261015\fix_261015\C_lens_analyses\C1_band_edge_summary_fixed.json"
C2Z = r"<WORK_ROOT>\overnight_261015\C_lens_analyses\C2_zone_gamma\lens_zone_gamma_result_flips.json"
WLB = r"<DATA_ROOT>\Result\261012_scdc_wlband"
sys.dont_write_bytecode = True
import numpy as np                                                               # noqa: E402
import psutil                                                                    # noqa: E402

T0 = time.time()
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def jl(p):
    return json.load(io.open(p, encoding="utf-8"))


T = jl(C2D + r"\c2_tables.json")
S23 = jl(C2D + r"\c2_s23.json")
B = jl(BJ)
C1 = jl(C1J)
C1X = jl(C1F)
C2ZJ = jl(C2Z)
V2 = jl(C2D + r"\lensband_eval_copy\lensband_eval_v2_result.json")
G1 = jl(HERE + r"\g1_c0_reference.json")

CH = ["1546.003", "1546.801", "1547.6", "1548.399", "1549.199", "1550.0", "1550.802", "1551.604", "1552.408", "1553.212", "1554.017"]   # keys of c2_tables / c2_s23
CHN = [float(c) for c in CH]
KCH = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
TAGS = ["l1546p003", "l1546p801", "l1547p600", "l1548p399", "l1549p199", "l1550p000", "l1550p802", "l1551p604", "l1552p408", "l1553p212", "l1554p017"]
G1KEY = ["%.3f" % x for x in CHN]
LEV = list(T["atoms"].keys())
W = T["weights"]
SMAX = (4.0 / 9.0) ** 1.5
C_STAIR_1550 = 0.8243                    # SI Eq. (S20): C_PD = 0.8243 x 10^(L_tot/10), the ideal eight-level staircase at 1550 nm
ARCH1_DB = 10 * math.log10(0.5 / SMAX)   # 2.272 dB: ideal four-output receiver (1/4 per port) against the single-lossless-reciprocal-grating ceiling (1/2) S_max
ARCH2_DB = 0.3676                        # 261017: echo cost at the -20 dB criterion (FACTS_R14_additions.md R14-7: 0.3676 dB at -20 dB; 0.3850 dB at -20.5 dB); deck split 2.640 / 1.674
ARCH2_OLD_DB = -10 * math.log10(0.9152)  # 0.385 dB (-20.5 dB criterion, old split 2.657 / 1.657)
ARCH_DB = ARCH1_DB + ARCH2_DB
say("g2_device_chain.py  %s  affinity %s  numpy %s  RSS %.2f GB" % (time.strftime("%Y-%m-%d %H:%M:%S"), psutil.Process().cpu_affinity(), np.__version__, psutil.Process().memory_info().rss / 1e9))

# ----------------------------------------------------------------------------------------------------------------------------------------- factors per channel
# (a) T: input metagrating, exact-grid basis of branch B (stored five replaced by the exact-wavelength re-solves, channel 0 = stored), mean of x and y lanes, MATCHED angle
S = B["stages"]
ex = {}
for rec in S["repro"]:
    ex[int(rec["ch"])] = {q: rec["stored"][q] for q in ("eta_x", "eta_y")}
for rec in S["new"]:
    ex[int(rec["ch"])] = {q: rec[q] for q in ("eta_x", "eta_y")}
for rec in S["bonus"]:
    ex[int(rec["ch"])] = {q: rec[q] for q in ("eta_x", "eta_y")}
assert sorted(ex) == sorted(KCH), sorted(ex)
Tx = [ex[k]["eta_x"] for k in KCH]
Ty = [ex[k]["eta_y"] for k in KCH]
Tm = [0.5 * (a + b) for a, b in zip(Tx, Ty)]
# (b) R_rt: retroreflective grating R(-2), 11 channels, fixed angle
Rrt = [T["retro"][c]["R_m2"] for c in CH]
# (c) eta: usage-weighted mean of the eight atom efficiencies (percent), fixed angle
eta_w = [sum(W[lv] * T["atoms"][lv][c]["eff"] for lv in LEV) for c in CH]
eta_u = [sum(T["atoms"][lv][c]["eff"] for lv in LEV) / len(LEV) for c in CH]
# independent re-read of the 88 atom records
MD5 = {"122p5": "b59ed0ba", "032p5": "53474519", "077p5": "f8176952", "167p5": "ed750b02", "212p5": "78259586", "257p5": "accded83", "302p5": "ee3609ef", "347p5": "accf4825"}
raw_max = 0.0
raw_n = 0
for lv in LEV:
    for c, tag in zip(CH, TAGS):
        p = (WLB + r"\261012_sim783_wl_a%s_%s_%s\atom2d_opt.json" % (lv, MD5[lv], tag)) if tag == "l1550p000" else (r"<DATA_ROOT>\Result\261030_scdc_kxheld" + r"\261030_sim899_kx_a%s_%s_%s\atom2d_opt.json" % (lv, MD5[lv], tag))
        if not os.path.isfile(p):
            print("NOT FOUND: [%s]" % p)
            raise SystemExit("FAILURE: missing %s" % p)
        raw = jl(p)["carrier_eff_pct"]
        raw_max = max(raw_max, abs(raw - T["atoms"][lv][c]["eff"]))
        raw_n += 1
say("INDEPENDENT RE-READ of %d atom2d_opt.json (carrier_eff_pct) vs c2_tables.json eff: max |diff| %.3e percent points" % (raw_n, raw_max))
say("usage weights H6 sum %.12f ; eta_w per channel (%%): %s" % (sum(W.values()), ", ".join("%.4f" % e for e in eta_w)))
chk_s23 = max(abs(a - S23["S_bar"][c]["eta_w"]) for a, c in zip(eta_w, CH))
say("eta_w vs c2_s23.json S_bar eta_w: max |diff| %.3e" % chk_s23)

# (d) lens: L_tot of the evaluator at the three computed channels, decayed / held tail, x + y (45 deg cells NOT included), atom terms at the channel
ANCH = [1546.003, 1550.0, 1554.017]
AKEY = ["1546.003", "1550", "1554.017"]
Lxy = {i: C1["out"][a]["x_y"]["total"] for i, a in enumerate(AKEY)}                  # [decay, held]
Lxy_cells = {i: C1["out"][a]["cells_only_xy"]["total"] for i, a in enumerate(AKEY)}   # atom terms frozen at 1550
Exy = [C1["out"][a]["x_y"]["echo_share"] for a in AKEY]                               # x + y share-weighted aperture echo (dB)
L45_1550 = C2ZJ["full_enum"]["total"]                                                 # x + y + 45 deg at 1550 nm, MEASURED [decay, held]
E45_1550 = C2ZJ["full_enum"]["echo_share"]
INC = [L45_1550[j] - Lxy[1][j] for j in (0, 1)]                                       # 45 deg increment of the total at 1550 nm (negative)
INC_E = E45_1550 - Exy[1]                                                             # 45 deg increment of the echo at 1550 nm (dB, positive)
say("45 deg increment at 1550 nm: total decayed %+.6f / held %+.6f dB (C1 fixed json: %+.6f / %+.6f); echo %+.6f dB (C1 fixed json %+.6f)" % (
    INC[0], INC[1], C1X["increment_total_1550_dB"][0], C1X["increment_total_1550_dB"][1], INC_E, C1X["increment_echo_1550_dB"]))
assert abs(INC[0] - C1X["increment_total_1550_dB"][0]) < 1e-12 and abs(INC_E - C1X["increment_echo_1550_dB"]) < 1e-12
Lfin = {}                                                                             # final-basis L_tot at the three anchors, [decay, held]
for i in range(3):
    Lfin[i] = [Lxy[i][j] + INC[j] for j in (0, 1)]                                     # at 1550: x + y + inc = measured x + y + 45 (checked below)
chk1550 = max(abs(Lfin[1][j] - L45_1550[j]) for j in (0, 1))
say("1550 nm anchor: x + y %+.7f + increment = %+.7f vs measured x + y + 45 deg %+.7f (resid %.1e)" % (Lxy[1][0], Lfin[1][0], L45_1550[0], chk1550))
Efin_old = [Exy[0] + INC_E, E45_1550, Exy[2] + INC_E]                                  # OLD echo anchors (fig7_data_261017: kept for the change table only)
Lfin_old = {i: list(Lfin[i]) for i in range(3)}
# --- 261017 replacement: MEASURED lens read-out with all 45 deg cells at the three anchors (lensband_edge_261017, v6 definition)
NEWJ = jl(r"<WORK_ROOT>\lensband_edge_261017\lensband_edge_results_261017_1226_r1.json")["results"]
for i, a in enumerate(AKEY):
    Lfin[i] = [NEWJ[a]["total_decayed_dB"], NEWJ[a]["total_held_dB"]]
Efin = [NEWJ[a]["echo_aperture_NE_dB"] for a in AKEY]
say("NEW MEASURED anchors (lensband_edge_261017): L decayed/held %s ; echo %s" % ([["%.4f" % v for v in Lfin[i]] for i in range(3)], ["%.3f" % e for e in Efin]))
say("OLD anchors (overnight_261015): L decayed %s ; echo %s" % (["%.4f" % Lfin_old[i][0] for i in range(3)], ["%.3f" % e for e in Efin_old]))

# atom terms (deficit + purity f200) at all 11 channels (MEASURED chain, v2 evaluator, bit-controlled at the stored 1550)
terms = [V2["terms_all_channels"][t]["deficit"] + V2["terms_all_channels"][t]["purity"] for t in TAGS]
shift = [x - terms[5] for x in terms]                                                  # dB vs 1550

# interpolation of L_tot (task rule): quadratic in wavelength through the three anchors, evaluated at the channels; linear = sensitivity (piecewise between the anchors)


def quad(vals, lam):
    """three-point quadratic in wavelength (Lagrange form through the anchors, exact at the anchors)"""
    x0, x1, x2 = ANCH
    y0, y1, y2 = vals
    return float(y0 * (lam - x1) * (lam - x2) / ((x0 - x1) * (x0 - x2)) + y1 * (lam - x0) * (lam - x2) / ((x1 - x0) * (x1 - x2)) + y2 * (lam - x0) * (lam - x1) / ((x2 - x0) * (x2 - x1)))


def lin(vals, lam):
    return float(np.interp(lam, ANCH, vals))


Lq, Ll, Ls, Lh_q, Lh_l, Lh_s = [], [], [], [], [], []
for j, (c, lam) in enumerate(zip(CH, CHN)):
    Lq.append(quad([Lfin[i][0] for i in range(3)], lam))
    Ll.append(lin([Lfin[i][0] for i in range(3)], lam))
    Lh_q.append(quad([Lfin[i][1] for i in range(3)], lam))
    Lh_l.append(lin([Lfin[i][1] for i in range(3)], lam))
    # split variant (261017 redefinition): quadratic through the MEASURED anchors + (atom terms measured at the channel minus their own quadratic through the anchors)
    sh_corr = shift[j] - quad([shift[0], shift[5], shift[10]], lam)
    Ls.append(Lq[-1] + sh_corr)
    Lh_s.append(Lh_q[-1] + sh_corr)
# anchors reproduced exactly?
for idx, jj in ((0, 0), (1, 5), (2, 10)):
    assert abs(Lq[jj] - Lfin[idx][0]) < 1e-12 and abs(Ll[jj] - Lfin[idx][0]) < 1e-12, (jj, Lq[jj], Lfin[idx][0])
    assert abs(Ls[jj] - Lfin[idx][0]) < 1e-9, (jj, Ls[jj], Lfin[idx][0])
say("interpolation reproduces the three anchors (quadratic, linear: < 1e-12 dB; split: < 1e-9 dB).")

# echo
Eq = [quad(Efin, lam) for lam in CHN]
El = [lin(Efin, lam) for lam in CHN]

# ideal reference of the evaluator at each channel (g1): c0 of model fixed
c0 = [G1["c0"]["fixed"][k] for k in G1KEY]
Cref = [C_STAIR_1550 * c / c0[5] for c in c0]                                           # evaluator's own per-channel reference scaled to the SI 82.43 % at 1550
say("per-channel ideal reference C_ref = 0.8243 x c0(lambda)/c0(1550) (%%): %s" % ", ".join("%.4f" % (100 * x) for x in Cref))


def Fof(Tv, Rv, etav, Cv, L):
    return 0.5 * Tv * math.sqrt(Rv) * (etav / 100.0 * SMAX) * Cv * 10 ** (L / 10.0)


# ---------------------------------------------------------------------------------------------------------------------------------------- reproductions
say("")
say("REPRODUCTION 1: the SI / C2 1550 nm value F = 0.09258 (T = 0.9991043 from nG 1793, final lens basis -0.6008 dB decayed), from the raw sources")
Tsi_x, Tsi_y = 0.9991374, 0.9990713                                                     # SI Table S23 (nG 1793 lanes), as c2_s23.py
Cc = C_STAIR_1550 * 10 ** (-0.6008 / 10)                                                # c2_s23.py uses the rounded -0.6008
R1550 = T["retro"]["1550.0"]["R_m2"]
Fx_rep = 0.5 * Tsi_x * math.sqrt(R1550) * (eta_w[5] / 100 * SMAX) * Cc
Fy_rep = 0.5 * Tsi_y * math.sqrt(R1550) * (eta_w[5] / 100 * SMAX) * Cc
st_x, st_y = S23["final"]["nG1793|decay"]
rep1 = {"F_x": Fx_rep, "F_y": Fy_rep, "stored_F_x": st_x, "stored_F_y": st_y, "resid_x": Fx_rep - st_x, "resid_y": Fy_rep - st_y}
say("  F_x %.12f (c2_s23.json %.12f, resid %+.2e)   F_y %.12f (%.12f, resid %+.2e)   SI text: 0.09258 / 0.09257" % (Fx_rep, st_x, Fx_rep - st_x, Fy_rep, st_y, Fy_rep - st_y))
Tavg_si = 0.9991043
Fm = 0.5 * Tavg_si * math.sqrt(R1550) * (eta_w[5] / 100 * SMAX) * Cc
say("  with the lane-mean T = 0.9991043 of the SI text: F = %.8f  (%.4f %%, %.4f dB; SI: 9.26 %% / -10.335 dB)" % (Fm, 100 * Fm, 10 * math.log10(Fm)))
exc = 10 * math.log10(0.25 / Fm)
impl = exc - ARCH_DB
say("  excess against 1/4: %.4f dB (SI text 4.314), architecture %.4f = %.4f + %.4f (SI 2.657 = 2.272 + 0.385), implementation %.4f (SI 1.657), F / ((1/4) 0.5926 0.9152) = %.4f (SI 0.683)" % (
    exc, ARCH_DB, ARCH1_DB, ARCH2_DB, impl, Fm / (0.25 * 2 * SMAX * 0.9152)))
rep1.update(F_mean=Fm, F_dB=10 * math.log10(Fm), excess=exc, arch=ARCH_DB, arch1=ARCH1_DB, arch2=ARCH2_DB, impl=impl, ratio_to_arch_ceiling=Fm / (0.25 * 2 * SMAX * 0.9152))
# the SI's own factor budget
fac = {"T": Tavg_si, "sqrt_R": math.sqrt(R1550), "eta": eta_w[5] / 100, "C_PD": Cc, "Smax": SMAX, "half": 0.5}
say("  factor budget dB: " + ", ".join("%s %+.4f" % (k, 10 * math.log10(v)) for k, v in fac.items()) + "   (SI: T -0.0039, sqrtR -0.0001, eta -0.598, C_PD -1.440, Smax -5.283, 1/2 -3.010)")

say("")
say("REPRODUCTION 2: C2 table A6 (x-ramp basis, T per lane at nG 901 of the stored rounded wavelengths, constant 0.8243, decayed / held) at the three edge channels")
A6 = {}
stored_F = S23["F"]
okA6 = True
for c, k in (("1546.003", 5), ("1550.0", 0), ("1554.017", -5)):
    j = CH.index(c)
    inpk = {"1546.003": "1546.00", "1550.0": "1550.00", "1554.017": "1554.02"}[c]
    for lane in ("x", "y"):
        for tail in ("decay", "held"):
            Lx = S23["cpd"][c if c != "1550.0" else "1550.0"]["L_" + tail]
            Tl = T["input"][inpk]["eta_" + lane]
            f = 0.5 * Tl * math.sqrt(T["retro"][c]["R_m2"]) * (S23["S_bar"][c]["eta_w"] / 100 * SMAX) * C_STAIR_1550 * 10 ** (Lx / 10)
            sv = stored_F["%s|%s|%s" % (c, lane, tail)]
            A6[(c, lane, tail)] = (f, sv)
            okA6 &= abs(f - sv) < 1e-13
    say("  %s: F x decayed %.8f (c2_s23.json %.8f) held %.8f (%.8f); y decayed %.8f (%.8f)" % (
        c, A6[(c, "x", "decay")][0], A6[(c, "x", "decay")][1], A6[(c, "x", "held")][0], A6[(c, "x", "held")][1], A6[(c, "y", "decay")][0], A6[(c, "y", "decay")][1]))
say("  all 12 values equal the stored c2_s23.json F to 1e-13: %s ; C2 A6 prints (5 dp) 0.09065 / 0.09374 / 0.09495 (x decayed): %s" % (
    okA6, "yes" if all(round(A6[(c, "x", "decay")][0], 5) == v for c, v in (("1546.003", 0.09065), ("1550.0", 0.09374), ("1554.017", 0.09495))) else "NO"))
# needle: a deliberately wrong eta (+0.01 percentage point) must break the check
needle_f = 0.5 * Tsi_x * math.sqrt(R1550) * ((eta_w[5] + 0.01) / 100 * SMAX) * Cc
say("NEEDLE: eta + 0.01 pp changes F_x by %.2e (relative %.2e); the reproduction check (tolerance 1e-13) %s" % (needle_f - Fx_rep, (needle_f - Fx_rep) / Fx_rep, "catches it" if abs(needle_f - st_x) > 1e-13 else "MISSES it"))
assert abs(needle_f - st_x) > 1e-13
assert abs(Fx_rep - st_x) < 1e-13 and abs(Fy_rep - st_y) < 1e-13 and okA6

# ----------------------------------------------------------------------------------------------------------------------------------------- the chain per channel
rows = []
for j, c in enumerate(CH):
    lam = CHN[j]
    k = KCH[j]
    meas1550 = (k == 0)
    edge = (abs(k) == 5)
    Lc = Lq[j]
    Cq = Cref[j] * 10 ** (Lq[j] / 10)
    Cl = Cref[j] * 10 ** (Ll[j] / 10)
    Cs = Cref[j] * 10 ** (Ls[j] / 10)
    Ch = Cref[j] * 10 ** (Lh_q[j] / 10)
    C_const = C_STAIR_1550 * 10 ** (Lq[j] / 10)
    F = 0.5 * Tm[j] * math.sqrt(Rrt[j]) * (eta_w[j] / 100 * SMAX) * Cq
    F_lin = 0.5 * Tm[j] * math.sqrt(Rrt[j]) * (eta_w[j] / 100 * SMAX) * Cl
    F_split = 0.5 * Tm[j] * math.sqrt(Rrt[j]) * (eta_w[j] / 100 * SMAX) * Cs
    F_held = 0.5 * Tm[j] * math.sqrt(Rrt[j]) * (eta_w[j] / 100 * SMAX) * Ch
    F_const = 0.5 * Tm[j] * math.sqrt(Rrt[j]) * (eta_w[j] / 100 * SMAX) * C_const
    F_alT = F * (Tavg_si / Tm[5])
    rows.append(dict(k=k, nm=lam, nm_grid=B["meta"]["grid_nm_C2"][str(k)], Tx=Tx[j], Ty=Ty[j], T=Tm[j], R_rt=Rrt[j], eta_pct=eta_w[j], eta_unw_pct=eta_u[j], S_abs=eta_w[j] / 100 * SMAX,
                     L_quad=Lq[j], L_lin=Ll[j], L_split=Ls[j], L_held_quad=Lh_q[j], c0=c0[j], C_ref_pct=100 * Cref[j], C_PD_pct=100 * Cq, C_PD_lin_pct=100 * Cl, C_PD_split_pct=100 * Cs, C_PD_held_pct=100 * Ch,
                     F=F, F_pct=100 * F, F_dB=10 * math.log10(F), F_rel1550_dB=None, excess_dB=10 * math.log10(0.25 / F), arch_dB=ARCH_DB, impl_dB=10 * math.log10(0.25 / F) - ARCH_DB,
                     F_lin=F_lin, F_split=F_split, F_held=F_held, F_C0const=F_const, F_Talign=F_alT, echo_quad=Eq[j], echo_lin=El[j],
                     status=("DERIVED (1550 nm: every input MEASURED; T and eta are means of MEASURED values; lens total v6 read-out)" if meas1550 else ("DERIVED (edge: lens total MEASURED incl. all 45 deg cells, lensband_edge_261017; T and eta DERIVED from MEASURED)" if edge else "ESTIMATE (interior: L_tot quadratic interpolation)"))))
F1550 = rows[5]["F"]
for r in rows:
    r["F_rel1550_dB"] = 10 * math.log10(r["F"] / F1550)

# ----------------------------------------------------------------------------------------------------------------------------------------- DLI phase (Task 3)
dli = []
for j, c in enumerate(CH):
    pr = S23["phase_rows"][j]
    assert pr["ch"] == c
    # independent recomputation of the atom mean from c2_tables (usage-weighted ddpsi)
    am = sum(W[lv] * T["atoms"][lv][c]["ddpsi"] for lv in LEV)
    tot = T["slab_geom_bias_deg"][c] + T["retro"][c]["argR_rel_1550"] + am
    assert abs(am - pr["atoms_mean"]) < 1e-12 and abs(tot - pr["total_common"]) < 1e-12, (c, am, pr["atoms_mean"], tot, pr["total_common"])
    dli.append(dict(k=KCH[j], nm=CHN[j], slab=pr["geo"], retro=pr["retro"], atoms_mean=pr["atoms_mean"], total=pr["total_common"], carrier_only=pr["carrier_only"], ptp=pr["ptp_abs_dpsi"],
                    rms=pr["rms_abs_dpsi"], resid_lin=pr["resid_after_linear"], off_GHz=pr["offset_GHz_total"]))
say("")
say("DLI phase: total = slab + retro + atoms recomputed from c2_tables.json for all 11 channels: equal to c2_s23.json to 1e-12 deg (assert passed)")
# the common phase in the delay-line interferometer is a bias error of 3.6 deg per GHz -> equivalent laser offset
span = max(d["total"] for d in dli) - min(d["total"] for d in dli)

# ----------------------------------------------------------------------------------------------------------------------------------------- write CSV
hdr = ["k", "lambda_grid_nm", "lambda_label_nm", "T_x[MEASURED matched angle nG901]", "T_y[MEASURED]", "T_mean[DERIVED mean of T_x and T_y]", "R_rt[MEASURED fixed angle]", "eta_usage_weighted_pct[DERIVED usage-weighted mean of the MEASURED atom eta; fixed angle]",
       "eta_unweighted_pct[MEASURED]", "S_abs=eta*Smax[DERIVED]", "L_tot_final_decayed_dB[1550 MEASURED; others ESTIMATE]", "c0_ideal_ref[DERIVED scalar model]", "C_ref_pct[DERIVED]",
       "C_PD_pct[1550 DERIVED; others ESTIMATE]", "F_per_branch_power[see status]", "F_pct", "F_dB", "F_rel_1550_dB", "excess_vs_quarter_dB", "architecture_dB[DERIVED; wavelength independent by definition]",
       "implementation_dB[excess-architecture]", "status",
       "SENS_F_linear_interp", "SENS_F_split_interp[cells quadratic + measured atom terms]", "SENS_F_held_tail_quadratic", "SENS_F_constant_0.8243", "SENS_F_T_aligned_to_nG1793_anchor"]
out = io.open(HERE + r"\device_chain_11ch.csv", "w", encoding="utf-8", newline="")
out.write(",".join(hdr) + "\n")
for r in rows:
    out.write(",".join(str(x) for x in (r["k"], r["nm_grid"], r["nm"], "%.7f" % r["Tx"], "%.7f" % r["Ty"], "%.7f" % r["T"], "%.7f" % r["R_rt"], "%.5f" % r["eta_pct"], "%.5f" % r["eta_unw_pct"], "%.7f" % r["S_abs"],
                                          "%.6f" % r["L_quad"], "%.7f" % r["c0"], "%.5f" % r["C_ref_pct"], "%.5f" % r["C_PD_pct"], "%.7f" % r["F"], "%.5f" % r["F_pct"], "%.5f" % r["F_dB"], "%+.5f" % r["F_rel1550_dB"],
                                          "%.5f" % r["excess_dB"], "%.5f" % ARCH_DB, "%.5f" % r["impl_dB"], '"%s"' % r["status"],
                                          "%.7f" % r["F_lin"], "%.7f" % r["F_split"], "%.7f" % r["F_held"], "%.7f" % r["F_C0const"], "%.7f" % r["F_Talign"])) + "\n")
out.close()
d = io.open(HERE + r"\dli_phase_11ch.csv", "w", encoding="utf-8", newline="")
d.write("k,lambda_nm,slab_geometric_deg[DERIVED analytic Sellmeier],retro_argR(-2)_deg[MEASURED fixed angle],bottom_metagrating_with_atoms_deg[DERIVED usage-weighted mean of MEASURED dpsi; fixed angle],"
        "total_common_deg[DERIVED sum],carrier_alone_deg[MEASURED; reference only; not added],equivalent_laser_offset_GHz[DERIVED 3.6 deg per GHz],atom_dpsi_ptp_deg[MEASURED],atom_dpsi_rms_deg[DERIVED],total_minus_linear_fit_deg[DERIVED]\n")
for x in dli:
    d.write("%d,%.3f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.3f,%.4f,%.4f,%+.4f\n" % (x["k"], x["nm"], x["slab"], x["retro"], x["atoms_mean"], x["total"], x["carrier_only"], x["off_GHz"], x["ptp"], x["rms"], x["resid_lin"]))
d.close()

# ----------------------------------------------------------------------------------------------------------------------------------------- write md
M = []
M.append("# Device-level chain per channel (branch G, recomputed 261017 with the MEASURED lens values; split 2.640 / implementation = excess - 2.640)\n")
M.append("Generated by g2_device_chain.py (stored results only, no solver). F = (1/2) T sqrt(R_rt) eta S_max C_PD (SI Note S12), per unit fiber power in one branch, per photodiode. Channels: 100 GHz grid ANCHORED AT 1550.000 nm (193.4145 THz), not the ITU-T grid. "
         "Labels: MEASURED = stored simulation, DERIVED = arithmetic on measured values (or the scalar lens model), ESTIMATE = assumption based, OPEN = missing.\n")
M.append("**Angle convention.** T (input metagrating) is solved at the matched angle (MEASURED). R_rt, eta (atoms on the bottom metagrating) and the lens cells are solved at the FIXED 1550 nm incidence angle (kx/G = 1550/lambda: +0.26 % at 1546.003 nm, -0.26 % at 1554.017 nm); the kx-held (matched-angle) evaluation of these elements is OPEN.\n")
M.append("## 1. Factors per channel\n")
M.append("T: mean of the MEASURED T_x and T_y (DERIVED), nG 901, exact-grid basis (branch B). R_rt: R(-2), Li nG 257 (C2). eta: usage-weighted mean (H6 weights) of the eight MEASURED atom efficiencies (DERIVED), Li nG 1793 (sim783). C_PD = C_ref x 10^(L_tot/10) with C_ref = 0.8243 x c0(lambda)/c0(1550) (the evaluator's own ideal reference, section 3); L_tot = final lens basis (x + y + 45 deg, decayed tail): MEASURED at 1550.000 nm, ESTIMATE elsewhere (band edges: C1 x + y MEASURED plus the 1550 nm 45 deg increment %+.4f dB as ESTIMATE; the eight interior channels: quadratic in wavelength through the three points).\n" % INC[0])
M.append("| k | nm | T [DERIVED] | R_rt [MEASURED] | eta % [DERIVED] | L_tot dB [label in status] | C_PD % | F (per branch power) | F % | F dB | status |")
M.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    lab = "MEASURED" if r["k"] == 0 else "ESTIMATE"
    M.append("| %+d | %.3f | %.6f | %.7f | %.3f | %+.4f [%s] | %.3f [%s] | %.5f | %.3f | %.3f | %s |" % (r["k"], r["nm"], r["T"], r["R_rt"], r["eta_pct"], r["L_quad"], lab, r["C_PD_pct"], "DERIVED" if r["k"] == 0 else "ESTIMATE", r["F"], r["F_pct"], r["F_dB"],
                                                                                                         "DERIVED" if r["k"] == 0 else "ESTIMATE"))
M.append("")
M.append("Status legend (261017 lens update): **DERIVED** at the three anchors (1546.003, 1550.000, 1554.017 nm) = T_x, T_y, R_rt, the eight atom efficiencies and the lens total (all 45 deg cells measured, <WORK_ROOT>/lensband_edge_261017/lensband_edge_results_261017_1226_r1.json, decayed tail) are MEASURED; T and eta are means of them (DERIVED); F is arithmetic on them. **ESTIMATE, interior (k = +4 .. +1, -1 .. -4)** = T_x, T_y, R_rt and the atom efficiencies are MEASURED at the channel (T and eta DERIVED), L_tot is the quadratic interpolation through the three MEASURED anchors (smoothness assumption), so C_PD and F are ESTIMATE. R_rt, eta and the lens cells are fixed-angle data; T is matched-angle. Architecture term 2.2724 + 0.3676 = 2.6400 dB (-20 dB criterion, deck split); the old 2.6573 dB (-20.5 dB criterion) is not used.")
M.append("")
M.append("At 1550.000 nm this chain gives F = %.6f; the SI value 0.09258 uses T = 0.9991043 from nG 1793, whereas the band basis here is nG 901 (mean of x and y %.7f): the ratio is %.6f (+%.4f dB). On the nG 1793 anchor (column SENS_F_T_aligned in the CSV) the 1550 nm value is %.6f.\n" % (
    F1550, Tm[5], Tm[5] / Tavg_si, 10 * math.log10(Tm[5] / Tavg_si), rows[5]["F_Talign"]))
M.append("## 2. Excess loss against the ideal 1/4 split, architecture and implementation\n")
M.append("Excess = 10 log10((1/4)/F) [DERIVED from the F of section 1; label of the row = label of F]. Architecture = %.4f + %.4f = %.4f dB [DERIVED]: the 2.272 dB is the gap between the ideal four-output receiver (1/4 per port) and the ceiling (1/2) S_max of an interferometer with one lossless reciprocal grating, the 0.385 dB the cost of the echo criterion (best lossless reciprocal grating meeting -20.5 dB = 91.52 %% of the ceiling, SI Note S4). Both are defined at 1550 nm and are therefore **wavelength independent by definition** here (a per-channel re-derivation of the bound at the fixed angle is OPEN). Implementation = excess - architecture [DERIVED]: it contains the bottom metagrating with atoms against the echo-limited best grating, the metalens with the photodiode, the input metagrating and the retroreflective grating.\n" % (ARCH1_DB, ARCH2_DB, ARCH_DB))
M.append("| k | nm | F dB | excess dB | architecture dB | implementation dB | share of the ceiling F / ((1/4)(2 S_max)(0.9152)) | label |")
M.append("|---|---|---|---|---|---|---|---|")
for r in rows:
    M.append("| %+d | %.3f | %.3f | %.3f | %.3f | %.3f | %.4f | %s |" % (r["k"], r["nm"], r["F_dB"], r["excess_dB"], ARCH_DB, r["impl_dB"], r["F"] / (0.25 * 2 * SMAX * 0.9152), "DERIVED" if r["k"] == 0 else "ESTIMATE"))
M.append("")
M.append("Band summary: F runs from %.3f %% (%.3f dB) at %.3f nm to %.3f %% (%.3f dB) at %.3f nm over the band; excess %.3f to %.3f dB; implementation %.3f to %.3f dB (all ESTIMATE except the 1550 nm row).\n" % (
    100 * min(r["F"] for r in rows), 10 * math.log10(min(r["F"] for r in rows)), [r["nm"] for r in rows if r["F"] == min(rr["F"] for rr in rows)][0],
    100 * max(r["F"] for r in rows), 10 * math.log10(max(r["F"] for r in rows)), [r["nm"] for r in rows if r["F"] == max(rr["F"] for rr in rows)][0],
    min(r["excess_dB"] for r in rows), max(r["excess_dB"] for r in rows), min(r["impl_dB"] for r in rows), max(r["impl_dB"] for r in rows)))
M.append("## 3. The ideal eight-level capture against wavelength (evaluator reference)\n")
M.append("The evaluator (lensband_eval.py, model fixed = the lens fixed at its 1550 nm design, propagated at the channel, f = 200 um, 8 x 8 um photodiode box) computes L_gamma relative to ITS OWN ideal continuous lens at each channel, c0(lambda) (the chromatic shift of the fixed lens is inside c0). It therefore DOES have a per-channel reference; c0 is smooth and monotone (DERIVED, scalar model, controls bit-identical at 1546.003 / 1550 / 1554.017 nm, g1_c0_reference.txt). The staircase factor of the SI 82.43 %% (= 0.8243 / c0(1550) = %.6f, %.4f dB) is a fixed constant inside the evaluator's deficit term (phase-quantisation Strehl of the ideal eight-level staircase over the 1550 nm phase histogram) and is carried unchanged to all channels; its own wavelength dependence is OPEN (not computed by the evaluator).\n" % (C_STAIR_1550 / c0[5], 10 * math.log10(C_STAIR_1550 / c0[5])))
M.append("| k | nm | c0 fixed [DERIVED] | c0 vs 1550 dB | C_ref % = 0.8243 x c0/c0(1550) | c0 re-designed at the channel (not used) | fixed minus re-designed dB |")
M.append("|---|---|---|---|---|---|---|")
for j, r in enumerate(rows):
    red = G1["c0"]["fixed_redref"][G1KEY[j]]
    M.append("| %+d | %.3f | %.6f | %+.5f | %.4f | %.6f | %+.5f |" % (r["k"], r["nm"], r["c0"], 10 * math.log10(r["c0"] / c0[5]), r["C_ref_pct"], red, 10 * math.log10(r["c0"] / red)))
M.append("")
M.append("Effect on F: at most %.4f dB (the 0.8243 constant against the per-channel reference), so the capture reference does not decide anything in this chain.\n" % max(abs(10 * math.log10(r["F"] / r["F_C0const"])) for r in rows))
M.append("## 4. Sensitivity of F to the lens interpolation (the eight interior channels; the three anchors are identical in all columns)\n")
M.append("Quadratic = headline. Linear = piecewise linear between the three anchors. Split = the cell part of L_tot (cells with the atom terms frozen at 1550 nm, 3 anchors) interpolated quadratically, (261017 redefinition) = the quadratic through the MEASURED anchors plus the atom terms (deficit + purity f200) MEASURED at every channel minus their own quadratic through the anchors; it uses the structure of the atom terms (non-monotone, peak near 1548.4 nm) that a 3-point parabola cannot see. Held-tail column = headline with the held tail (quadratic).\n")
M.append("| k | nm | L_tot quad dB | L_tot linear | L_tot split | F quad | F linear | F split | F held tail | F dB quad | linear minus quad dB | split minus quad dB |")
M.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for j, r in enumerate(rows):
    M.append("| %+d | %.3f | %+.4f | %+.4f | %+.4f | %.5f | %.5f | %.5f | %.5f | %.3f | %+.4f | %+.4f |" % (r["k"], r["nm"], r["L_quad"], r["L_lin"], r["L_split"], r["F"], r["F_lin"], r["F_split"], r["F_held"], r["F_dB"],
                                                                                                         10 * math.log10(r["F_lin"] / r["F"]), 10 * math.log10(r["F_split"] / r["F"])))
M.append("")
maxd = max(max(abs(10 * math.log10(r["F_lin"] / r["F"])), abs(10 * math.log10(r["F_split"] / r["F"]))) for r in rows)
M.append("Largest difference of F between the interpolation rules: %.4f dB. Echo anchors (MEASURED at all three anchors, lensband_edge_261017; old carried increment %+.4f dB no longer used): %.3f / %.3f / %.3f dB at 1546.003 / 1550.000 / 1554.017 nm; interpolated echo is used in eye_penalty_11ch (g3_eye.py).\n" % (maxd, INC_E, Efin[0], Efin[1], Efin[2]))
M.append("## 5. Reproduction and self-tests\n")
M.append("| Test | value | stored / reference | residual |")
M.append("|---|---|---|---|")
M.append("| R1 F_x at 1550 nm, T = 0.9991374, final basis -0.6008 dB, from raw sources | %.12f | c2_s23.json %.12f | %+.2e |" % (Fx_rep, st_x, Fx_rep - st_x))
M.append("| R1 F_y at 1550 nm, T = 0.9990713 | %.12f | c2_s23.json %.12f | %+.2e |" % (Fy_rep, st_y, Fy_rep - st_y))
M.append("| R1 F with the lane-mean T = 0.9991043 | %.8f | SI text 0.09258 (rounded, 5 dp: %.5f) | %+.2e (rounding) |" % (Fm, round(Fm, 5), Fm - 0.09258))
M.append("| R1 F in dB | %.4f | SI text -10.335 | %+.4f (rounding) |" % (10 * math.log10(Fm), 10 * math.log10(Fm) + 10.335))
M.append("| R1 excess against 1/4 | %.4f dB | SI text 4.314 dB (= -10.335 + 6.021 from rounded values) | %+.4f |" % (exc, exc - 4.314))
M.append("| R1 architecture / implementation | %.4f / %.4f dB | SI text 2.657 / 1.657 dB | %+.4f / %+.4f |" % (ARCH_DB, impl, ARCH_DB - 2.657, impl - 1.657))
M.append("| R1 share of the architecture ceiling | %.4f | SI text 0.683 | %+.4f |" % (rep1["ratio_to_arch_ceiling"], rep1["ratio_to_arch_ceiling"] - 0.683))
M.append("| R2 twelve F values of C2 table A6 (3 channels x lanes x tails) | max abs diff %.2e | c2_s23.json F | %.2e |" % (max(abs(a - b) for a, b in A6.values()), max(abs(a - b) for a, b in A6.values())))
M.append("| R3 independent re-read of 88 atom2d_opt.json carrier_eff_pct | max abs diff %.2e pp | c2_tables.json eff | %.2e |" % (raw_max, raw_max))
M.append("| R4 DLI total = slab + retro + atoms (11 channels) | max abs diff 0 (assert, 1e-12) | c2_s23.json total_common | < 1e-12 |")
M.append("| N1 needle: eta + 0.01 pp | F_x changes by %.2e | tolerance 1e-13 | caught |" % (needle_f - Fx_rep))
M.append("| N2 interpolation reproduces the three anchors | quad / linear < 1e-12 dB, split < 1e-9 dB | the three L_tot anchors | pass |")
M.append("| N3 45 deg increment of the total at 1550 nm | %+.6f / %+.6f dB | C1 fixed json %+.6f / %+.6f | %.1e |" % (INC[0], INC[1], C1X["increment_total_1550_dB"][0], C1X["increment_total_1550_dB"][1], abs(INC[0] - C1X["increment_total_1550_dB"][0])))
M.append("")
io.open(HERE + r"\device_chain_11ch.md", "w", encoding="utf-8").write("\n".join(M) + "\n")

io.open(HERE + r"\g2_numbers.json", "w", encoding="utf-8").write(json.dumps(dict(
    ch=CH, k=KCH, nm=CHN, echo_quad=Eq, echo_lin=El, echo_anchor=Efin, echo_anchor_old=Efin_old, L_anchor=[Lfin[i] for i in range(3)], L_anchor_old=[Lfin_old[i] for i in range(3)], echo_anchor_nm=ANCH, inc_total_1550=INC, inc_echo_1550=INC_E, rows=rows, rep1=rep1, dli=dli,
    L_anchor_final=Lfin, atom_term_shift=shift, F1550=F1550), indent=1))
say("wrote device_chain_11ch.csv/.md, dli_phase_11ch.csv, g2_numbers.json ; elapsed %.1f s ; RSS %.2f GB" % (time.time() - T0, psutil.Process().memory_info().rss / 1e9))
io.open(HERE + r"\g2_stdout.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
