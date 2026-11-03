# -*- coding: utf-8 -*-
"""C2 (a) totals and (b) Eq. (S23) evaluation from c2_tables.json (produced by c2_extract.py) and the lens evaluator result.
Reader only; writes c2_s23.md / c2_s23.json next to this script."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "c2_tables.json"), encoding="utf-8"))
LENS = json.load(open(os.path.join(HERE, "lensband_eval_copy", "lensband_eval_v2_result.json"), encoding="utf-8"))
WLB = r"<DATA_ROOT>\Result\261012_scdc_wlband"
CH = ["1546.003", "1546.801", "1547.6", "1548.399", "1549.199", "1550.0", "1550.802", "1551.604", "1552.408", "1553.212", "1554.017"]
CHK = {"1546.003": 5, "1546.801": 4, "1547.6": 3, "1548.399": 2, "1549.199": 1, "1550.0": 0, "1550.802": -1, "1551.604": -2, "1552.408": -3, "1553.212": -4, "1554.017": -5}
W = T["weights"]
levels = list(T["atoms"].keys())
SMAX = (4.0 / 9.0) ** 1.5
L = []


def key(nm):
    # keys in c2_tables.json are str(float) of the channel nm as printed by python
    for k in T["slab_geom_bias_deg"]:
        if abs(float(k) - float(nm)) < 5e-4:
            return k
    raise KeyError(nm)


# ------------------------------------------------------------ (a) common-mode DLI phase per channel
rows = []
for c in CH:
    k = key(c)
    geo = T["slab_geom_bias_deg"][k]
    ret = T["retro"][k]["argR_rel_1550"]
    md = sum(W[lv] * T["atoms"][lv][k]["ddpsi"] for lv in levels)
    car = T["carrier"][k]["bias_rel_1550"]
    # absolute spread of dpsi over the palette about the usage-weighted mean (rms) and ptp
    dp = {lv: T["atoms"][lv][k]["dpsi"] for lv in levels}
    mean_dp = sum(W[lv] * dp[lv] for lv in levels)
    rms_abs = math.sqrt(sum(W[lv] * (dp[lv] - mean_dp) ** 2 for lv in levels))
    ptp = max(dp.values()) - min(dp.values())
    tot = geo + ret + md
    rows.append(dict(ch=c, k=CHK[c], geo=geo, retro=ret, carrier_only=car, atoms_mean=md, total_common=tot, rms_abs_dpsi=rms_abs, ptp_abs_dpsi=ptp, geo_retro_carrier=geo + ret + car))
# linear fit of total vs k
ks = [r["k"] for r in rows]
tt = [r["total_common"] for r in rows]
n = len(ks)
mk = sum(ks) / n
mt = sum(tt) / n
slope = sum((a - mk) * (b - mt) for a, b in zip(ks, tt)) / sum((a - mk) ** 2 for a in ks)
icpt = mt - slope * mk
for r in rows:
    r["lin_fit"] = icpt + slope * r["k"]
    r["resid_after_linear"] = r["total_common"] - r["lin_fit"]
    r["offset_GHz_total"] = r["total_common"] / 3.6
dtau_fs = slope / 360.0 / 100e9 * 1e15       # slope is deg per 100 GHz channel
# trim of the slab: group delay per um of slab 10.0003 ps / 592.39 um
tau_per_um_ps = 10.0003 / 592.39
trim_um = dtau_fs * 1e-3 / tau_per_um_ps
# slab-only slope from the analytic residual is zero by construction; element-only slopes (deg per channel) for reference
def fit(col):
    m = sum(r[col] for r in rows) / n
    return sum((a - mk) * (r[col] - m) for a, r in zip(ks, rows)) / sum((a - mk) ** 2 for a in ks)
sl = {c: fit(c) for c in ("retro", "carrier_only", "atoms_mean", "total_common")}
L.append("COMMON-MODE DLI PHASE per channel (deg; relative to 1550.000; convention: phase of arm B minus arm A, field exp(+i k z) as in the RCWA solvers; terms add)")
L.append("ch | k | slab geom | retro arg R(-2) | carrier-only (n1768) | atoms palette-mean (carrier incl.) | TOTAL = slab + retro + atoms | offset GHz (÷3.6) | rms|abs dpsi spread| | ptp abs dpsi | resid after linear")
for r in rows:
    L.append("%s | %+d | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.2f | %.3f | %.3f | %+.3f" % (r["ch"], r["k"], r["geo"], r["retro"], r["carrier_only"], r["atoms_mean"], r["total_common"], r["offset_GHz_total"], r["rms_abs_dpsi"], r["ptp_abs_dpsi"], r["resid_after_linear"]))
L.append("fit total vs k: slope %+.4f deg per 100 GHz channel (retro %+.4f, carrier-only %+.4f, atoms %+.4f); intercept %+.3f; equivalent extra group delay %.3f ps; slab trim to restore 100.0 GHz: %.2f um (ARITH)" % (slope, sl["retro"], sl["carrier_only"], sl["atoms_mean"], icpt, dtau_fs / 1e3, trim_um))
L.append("range of TOTAL: %.3f .. %.3f deg (peak-to-peak %.2f); max |resid after linear| %.3f" % (min(tt), max(tt), max(tt) - min(tt), max(abs(r["resid_after_linear"]) for r in rows)))

# ------------------------------------------------------------ (b) Eq. S23 factors
# S-bar: usage-weighted mean of eta over the palette (eta = S/Smax; eff in percent in the atom JSON) and unweighted mean/min/max
Sb = {}
for c in CH:
    k = key(c)
    e = {lv: T["atoms"][lv][k]["eff"] for lv in levels}
    w_mean = sum(W[lv] * e[lv] for lv in levels)
    u_mean = sum(e.values()) / len(e)
    Sb[c] = dict(eta_w=w_mean, eta_u=u_mean, eta_min=min(e.values()), eta_max=max(e.values()), S_abs_w=w_mean / 100 * SMAX)
L.append("")
L.append("S-bar (usage-weighted mean of the atom eta, H6 weights; absolute S = eta * 0.296296)")
L.append("ch | eta_w % | eta unweighted % | min | max | S_abs (weighted)")
for c in CH:
    s = Sb[c]
    L.append("%s | %.3f | %.3f | %.3f | %.3f | %.5f" % (c, s["eta_w"], s["eta_u"], s["eta_min"], s["eta_max"], s["S_abs_w"]))

# amplitude-phase correlation bound at 1550 (per arm): relative std of the arm amplitudes over the palette, and cos(delta) spread
amp = {}
for lv in levels:
    pa = WLB + r"\261012_sim783_wl_a%s_%s_l1550p000\atom2d_opt.json" % ({"122p5": "122p5", "032p5": "032p5", "077p5": "077p5", "167p5": "167p5", "212p5": "212p5", "257p5": "257p5", "302p5": "302p5", "347p5": "347p5"}[lv],
                                                                    {"122p5": "b59ed0ba", "032p5": "53474519", "077p5": "f8176952", "167p5": "ed750b02", "212p5": "78259586", "257p5": "accded83", "302p5": "ee3609ef", "347p5": "accf4825"}[lv])
    a = json.load(open(pa, encoding="utf-8"))
    amp[lv] = (a["amp_tA"], a["amp_rBtB"], a["prod_amp"])
mA = sum(W[lv] * amp[lv][0] for lv in levels)
mB = sum(W[lv] * amp[lv][1] for lv in levels)
sA = math.sqrt(sum(W[lv] * (amp[lv][0] - mA) ** 2 for lv in levels)) / mA
sB = math.sqrt(sum(W[lv] * (amp[lv][1] - mB) ** 2 for lv in levels)) / mB
mP = sum(W[lv] * amp[lv][2] for lv in levels)
# correlation bound: |delta factor| <= (rel std of a) * (std of cos(delta)/mean cos), delta up to 22.5 deg (half step of 8 levels): std of cos <= (1 - cos 22.5)/2
cos_spread = (1 - math.cos(math.radians(22.5))) / 2 / ((1 + math.cos(math.radians(22.5))) / 2)
bound_rel = max(sA, sB) * cos_spread
L.append("")
L.append("amplitude spread over the palette at 1550 (weighted): |t_A| rel std %.4f, |r_B t_B| rel std %.4f; max eta/min eta %.4f; cos(delta) rel half-range at 22.5 deg %.4f; bound on the neglected amplitude-phase correlation of the LPA: %.2e relative = %.4f dB"
         % (sA, sB, max(s["eta_max"] for s in [Sb["1550.0"]]) / Sb["1550.0"]["eta_min"], cos_spread, bound_rel, 10 * math.log10(1 + bound_rel)))

# lens C_PD at the three channels (x-only basis, rB0_complex, 'full' = atom terms at the channel)
C0 = 0.8243
cpd = {}
for c, lk in (("1546.003", "1546.003"), ("1550.0", "1550"), ("1554.017", "1554.017")):
    res = LENS["channels"][lk]["models"]["fixed"]["full"]["res"]["rB0_complex"]["total"]
    cpd[c] = dict(L_decay=res[0], L_held=res[1], C_decay=C0 * 10 ** (res[0] / 10), C_held=C0 * 10 ** (res[1] / 10),
                  echo_ap=LENS["channels"][lk]["models"]["fixed"]["full"]["echo_aperture"], echo_ap_x48=LENS["channels"][lk]["models"]["fixed"]["aperture_x48"])
# the final 1550 verdict (x + y + 45 deg, 1/P map): H20 / SI Table S23
cpd["1550.0_xy45"] = dict(L_decay=-0.6008, L_held=-0.6217, C_decay=C0 * 10 ** (-0.6008 / 10), C_held=C0 * 10 ** (-0.6217 / 10))
L.append("")
L.append("C_PD = 0.8243 x 10^(L_tot/10) (rB0 complex, atom terms at the channel; x ramps only at the three channels)")
for c, v in cpd.items():
    L.append("%s | L_tot %.4f / %.4f dB | C_PD %.4f / %.4f%s" % (c, v["L_decay"], v["L_held"], v["C_decay"], v["C_held"], "" if "echo_ap" not in v else " | echo aperture %.3f dB (x48 zone: %.3f)" % (v["echo_ap"], v["echo_ap_x48"])))

# input metagrating T at the 5 wavelengths (nG 901) and at 1550 nG 1793 / 2601
inp = T["input"]
Tx = {"1546.003": inp["1546.00"]["eta_x"], "1547.6": inp["1547.60"]["eta_x"], "1550.0": inp["1550.00"]["eta_x"], "1552.408": inp["1552.41"]["eta_x"], "1554.017": inp["1554.02"]["eta_x"]}
Ty = {"1546.003": inp["1546.00"]["eta_y"], "1547.6": inp["1547.60"]["eta_y"], "1550.0": inp["1550.00"]["eta_y"], "1552.408": inp["1552.41"]["eta_y"], "1554.017": inp["1554.02"]["eta_y"]}
RR = {c: T["retro"][key(c)]["R_m2"] for c in CH}
L.append("")
L.append("ELEMENT FACTORS per channel: T_x, T_y (input, nG 901), sqrt(R_rt) (retro Li nG 257), S_abs (weighted)")
for c in CH:
    L.append("%s | T_x %s | T_y %s | R_rt %.7f | sqrt %.7f | S_abs %.5f" % (c, "%.7f" % Tx[c] if c in Tx else "NOT COMPUTED", "%.7f" % Ty[c] if c in Ty else "NOT COMPUTED", RR[c], math.sqrt(RR[c]), Sb[c]["S_abs_w"]))

# F_l = 1/2 T_l sqrt(R) S C  (all fibre power in branch l), per unit fibre power; at the three lens channels (x-only basis) and at 1550 on the final x+y+45 basis
out = {"phase_rows": rows, "slope_deg_per_channel": slope, "dtau_fs": dtau_fs, "trim_um": trim_um, "S_bar": Sb, "cpd": cpd, "bound_rel_corr": bound_rel}
F = {}
for c in ("1546.003", "1550.0", "1554.017"):
    for lane, TT in (("x", Tx), ("y", Ty)):
        for tail in ("decay", "held"):
            Cc = cpd[c]["C_" + tail]
            f = 0.5 * TT[c] * math.sqrt(RR[c]) * Sb[c]["S_abs_w"] * Cc
            F[(c, lane, tail)] = f
ref = {(lane, tail): F[("1550.0", lane, tail)] for lane in "xy" for tail in ("decay", "held")}
L.append("")
L.append("F = (1/2) T sqrt(R_rt) S_bar C_PD (interference amplitude of one output quadrant per unit power in that branch; absolute, x-only lens basis, T at nG 901)")
L.append("ch | lane | tail | F | F/F(1550) dB | ratio to ceiling-limited 1/2*Smax*1 = %.5f" % (0.5 * SMAX))
for c in ("1546.003", "1550.0", "1554.017"):
    for lane in "xy":
        for tail in ("decay", "held"):
            f = F[(c, lane, tail)]
            L.append("%s | %s | %s | %.5f | %+.4f | %.4f (%.3f dB)" % (c, lane, tail, f, 10 * math.log10(f / ref[(lane, tail)]), f / (0.5 * SMAX), 10 * math.log10(f / (0.5 * SMAX))))
# 1550 with the final lens basis x+y+45 and T at nG 1793 (avg and lanes), nG 2601
L.append("")
L.append("1550 nm on the final lens basis (x + y + 45 deg, 1/P map: -0.6008 / -0.6217 dB), T from nG 1793 (x 0.9991374, y 0.9990713) and nG 2601 (x 0.9990709, y 0.9990147):")
final = {}
for lab, tx, ty in (("nG1793", 0.9991374, 0.9990713), ("nG2601", 0.9990709, 0.9990147)):
    for tail in ("decay", "held"):
        Cc = cpd["1550.0_xy45"]["C_" + tail]
        fx = 0.5 * tx * math.sqrt(RR["1550.0"]) * Sb["1550.0"]["S_abs_w"] * Cc
        fy = 0.5 * ty * math.sqrt(RR["1550.0"]) * Sb["1550.0"]["S_abs_w"] * Cc
        final[(lab, tail)] = (fx, fy)
        L.append("%s %s | F_x %.5f | F_y %.5f | F_x/F_y %.3e (%.5f dB) | mean %.5f = %.4f of ceiling-limited (%.3f dB)" % (lab, tail, fx, fy, fx / fy - 1, 10 * math.log10(fx / fy), 0.5 * (fx + fy), 0.5 * (fx + fy) / (0.5 * SMAX), 10 * math.log10(0.5 * (fx + fy) / (0.5 * SMAX))))
# factor budget in dB at 1550 (final basis, nG 1793 lane average, decayed tail)
Tavg = 0.9991043
fac = [("T (input branch efficiency, lane mean)", Tavg), ("sqrt(R_rt)", math.sqrt(RR["1550.0"])), ("S_bar / Smax (usage-weighted eta)", Sb["1550.0"]["eta_w"] / 100), ("C_PD (x+y+45, decayed)", cpd["1550.0_xy45"]["C_decay"]), ("Smax (ceiling)", SMAX), ("1/2 (2 x quadrant share 1/4)", 0.5)]
L.append("")
L.append("factor budget at 1550 nm (final basis, decayed tail):")
prod = 1.0
for nme, v in fac:
    prod *= v
    L.append("  %-40s %.7f  %+.4f dB" % (nme, v, 10 * math.log10(v)))
L.append("  product = %.5f  (%.3f dB re 1)" % (prod, 10 * math.log10(prod)))
# branch bias mismatch factor cos(delta/2) for equal F_l and offsets +-delta/2
for dd in (20.0, 10.0):
    L.append("  branch bias difference %.0f deg -> sum of the two branch terms at equal F: factor cos(delta/2) = %.5f = %.4f dB" % (dd, math.cos(math.radians(dd / 2)), 10 * math.log10(math.cos(math.radians(dd / 2)))))
out["F"] = {"%s|%s|%s" % k: v for k, v in F.items()}
out["final"] = {"%s|%s" % k: v for k, v in final.items()}
json.dump(out, open(os.path.join(HERE, "c2_s23.json"), "w", encoding="utf-8"), indent=1)
open(os.path.join(HERE, "c2_s23.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
