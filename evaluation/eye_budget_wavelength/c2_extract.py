# -*- coding: utf-8 -*-
"""C2 (a): per-channel tables from existing data (reader only; no solver, nothing written under <DATA_ROOT>).

Reads
  * input metagrating   Result\\261008_scdc_in2dli\\261008_sim752_in2dLi_warm_xsec_min_rf050_W1\\input2d.json  (5 wavelengths, nG 901)
  * retro grating (Li)  Result\\261011_scdc_retroli\\261011_sim778_retroli_A001_liy_b11_n257\\scdc_retro_H525_450.json (11 channels)
  * slab                analytic (calc_band.py formula, re-implemented here)
  * carrier             Result\\261012_scdc_wlband\\261012_sim783_wl_car_f100_525f74f3_b11\\scdc_extract_H400_300.json
  * atoms               Result\\261012_scdc_wlband\\261012_sim783_wl_a<lv>_<md5>_l<chan>\\atom2d_opt.json (8 levels x 11 channels)
  * delay-arm tokens    Result\\261012_scdc_wlband\\261012_sim783_wl_tok_525f74f3_<lv>_b11\\scdc_extract_H400_300.json
Writes c2_tables.json and c2_tables.md next to this script.
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
R = r"<DATA_ROOT>\Result"
WLB = R + r"\261012_scdc_wlband"
P_IN = R + r"\261008_scdc_in2dli\261008_sim752_in2dLi_warm_xsec_min_rf050_W1\input2d.json"
P_RETRO = R + r"\261011_scdc_retroli\261011_sim778_retroli_A001_liy_b11_n257\scdc_retro_H525_450.json"
P_CAR = WLB + r"\261012_sim783_wl_car_f100_525f74f3_b11\scdc_extract_H400_300.json"

CH_TAGS = ["l1546p003", "l1546p801", "l1547p600", "l1548p399", "l1549p199", "l1550p000", "l1550p802", "l1551p604", "l1552p408", "l1553p212", "l1554p017"]
CH_LAB = ["1546.0", "1546.8", "1547.6", "1548.4", "1549.2", "1550.0", "1550.8", "1551.6", "1552.4", "1553.2", "1554.0"]
CH_NM = [float(t[1:].replace("p", ".")) for t in CH_TAGS]
# ITU index k: frequency f0 + k*100 GHz, k=+5 is 1546.003 nm
CH_K = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
LEVELS = [("122p5", "b59ed0ba", "122p5"), ("032p5", "53474519", "032p5"), ("077p5", "f8176952", "077p5"), ("167p5", "ed750b02", "167p5"),
          ("212p5", "78259586", "212p5"), ("257p5", "accded83", "257acc"), ("302p5", "ee3609ef", "302p5"), ("347p5", "accf4825", "347p5")]
# palette usage over the lens sites (facts_v3 H6, source tm_focus_summary.json palette_level_usage_fraction), percent
USAGE = {"122p5": 15.23, "032p5": 9.27, "077p5": 13.88, "167p5": 11.15, "212p5": 9.85, "257p5": 12.14, "302p5": 15.82, "347p5": 12.65}
PUR_FLOOR = 0.996722
ECHO_GATE = -20.5
EFF_GATE = 85.0


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def jl(p):
    return json.load(open(p, encoding="utf-8"))


out = {}
# ---------------------------------------------------------------- slab, analytic (calc_band.py formula)
c0 = 299792458.0


def n_sell(l):
    l2 = l * l
    return math.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2) + 0.4079426 * l2 / (l2 - 0.1162414 ** 2) + 0.8974794 * l2 / (l2 - 9.896161 ** 2))


Lam = 1.31539
G = 2 * math.pi / Lam
T_SLAB = 592.39


def Phi(nu, t=T_SLAB):
    lam = c0 / nu * 1e6
    k0 = 2 * math.pi / lam
    return 2 * t * math.sqrt((n_sell(lam) * k0) ** 2 - G ** 2)


nu0 = c0 / 1.55e-6
h = 1e8
tau = (Phi(nu0 + h) - Phi(nu0 - h)) / (2 * h) / (2 * math.pi)
slab = {}
for k, nm in zip(CH_K, CH_NM):
    nu = nu0 + k * 100e9
    slab[nm] = math.degrees(Phi(nu) - Phi(nu0) - 2 * math.pi * tau * k * 100e9)
out["slab_geom_bias_deg"] = {str(nm): v for nm, v in slab.items()}
out["slab_tau_ps"] = tau * 1e12
out["channel_nm_exact_from_grid"] = {str(k): c0 / (nu0 + k * 100e9) * 1e9 for k in CH_K}
# group-delay dispersion
dd = 1e9
gdd = (Phi(nu0 + dd) - 2 * Phi(nu0) + Phi(nu0 - dd)) / (dd ** 2) / (2 * math.pi) ** 2
out["slab_gdd_fs2"] = gdd * 1e30

# ---------------------------------------------------------------- input metagrating
d_in = jl(P_IN)
inp = {}
for lab in ["1546.00", "1547.60", "1550.00", "1552.41", "1554.02"]:
    inp[lab] = {"eta_x": d_in["lam%s_eta_x" % lab], "eta_y": d_in["lam%s_eta_y" % lab], "leak_x": d_in["lam%s_leak_x" % lab], "leak_y": d_in["lam%s_leak_y" % lab],
                "refl_worst_sop": d_in["lam%s_refl_worst_sop" % lab]}
out["input"] = inp
out["input_band_nG"] = d_in["band_nG"]
out["input_phase_keys_per_wavelength"] = [k for k in d_in if re.match(r"lam\d", k) and re.search(r"arg|phase", k)]

# ---------------------------------------------------------------- retro (Li, 11 channels)
d_rt = jl(P_RETRO)
retro = {}
for lab, nm in zip(CH_LAB, CH_NM):
    retro[str(nm)] = {"R_m2": d_rt["%s_sub_p_TM_R_m2" % lab], "argR_m2": d_rt["%s_sub_p_TM_argR_m2" % lab]}
a0 = retro["1550.0"]["argR_m2"]
for nm in retro:
    retro[nm]["argR_rel_1550"] = wrap(retro[nm]["argR_m2"] - a0)
out["retro"] = retro
out["retro_slope_TM"] = d_rt["slope_retro_TM"]

# ---------------------------------------------------------------- carrier
d_car = jl(P_CAR)
ceil = (4.0 / 9.0) ** 1.5
tags = ["n1708", "n2329", "n3445", "n1616", "n1768"]
car = {}
for lab, nm in zip(CH_LAB, CH_NM):
    row = {}
    sig, ech, pur, purm = {}, {}, {}, {}
    for t in tags:
        p = t + "_" + lab + "_"
        s = math.sqrt(max(d_car[p + "sub_p_TM_T_m1"] * d_car[p + "sub_p_TM_R_0"] * d_car[p + "sub_m_TM_T_1"], 0.0))
        sig[t] = 100.0 * s / ceil
        ech[t] = 10.0 * math.log10(max(d_car[p + "sub_m_TM_R_2"], 1e-30))
        pur[t] = d_car[p + "purity"]
        purm[t] = d_car[p + "puritym"]
    b = "n1768_" + lab + "_"
    bias = wrap(d_car[b + "sub_p_TM_argR_0"] + d_car[b + "sub_m_TM_argT_1"] - d_car[b + "sub_p_TM_argT_m1"])
    # per-termination bias (spread over the five terminations at this channel)
    bt = {}
    for t in tags:
        q = t + "_" + lab + "_"
        bt[t] = wrap(d_car[q + "sub_p_TM_argR_0"] + d_car[q + "sub_m_TM_argT_1"] - d_car[q + "sub_p_TM_argT_m1"])
    row.update(sig_ref=sig["n1768"], sig_worst=min(sig.values()), echo_ref=ech["n1768"], echo_worst=max(ech.values()), pur_worst=min(pur.values()), purm_worst=min(purm.values()),
               bias_ref=bias, bias_by_term=bt)
    car[str(nm)] = row
b0 = car["1550.0"]["bias_ref"]
for nm in car:
    car[nm]["bias_rel_1550"] = wrap(car[nm]["bias_ref"] - b0)
    bt0 = car["1550.0"]["bias_by_term"]
    car[nm]["bias_rel_1550_by_term"] = {t: wrap(car[nm]["bias_by_term"][t] - bt0[t]) for t in tags}
out["carrier"] = car

# ---------------------------------------------------------------- atoms and delay-arm tokens
atoms = {}
for lv, md5, tt in LEVELS:
    tokj = jl(WLB + r"\261012_sim783_wl_tok_%s_%s_b11\scdc_extract_H400_300.json" % ("525f74f3", tt))
    for tag, lab, nm in zip(CH_TAGS, CH_LAB, CH_NM):
        pa = WLB + r"\261012_sim783_wl_a%s_%s_%s\atom2d_opt.json" % (lv, md5, tag)
        a = jl(pa)
        kk = [k for k in tokj if re.match(r"^n\d+_%s_purity$" % re.escape(lab), k)]
        assert len(kk) == 1, (lv, lab, kk)
        pre = kk[0][:-len("_purity")]
        row = dict(A=a["A_deg"] % 360.0, B=a["B_deg"], dpsi=a["dpsi_deg"], eff=a["carrier_eff_pct"], echo=a["echo_p2_db"], pur=a["purity"], pur_tok=tokj[kk[0]],
                   purm=tokj[pre + "_puritym"], absR00=a["absR00"], argR00=a["argR00_deg"], phase_err_rms=a["phase_err_rms_deg"], T_atom=a["T_atom"], prod_amp=a["prod_amp"])
        atoms.setdefault(lv, {})[str(nm)] = row
for lv in atoms:
    d0 = atoms[lv]["1550.0"]
    for nm in atoms[lv]:
        r = atoms[lv][nm]
        r["dA"] = wrap(r["A"] - d0["A"])
        r["dB"] = wrap(r["B"] - d0["B"])
        r["ddpsi"] = wrap(r["dpsi"] - d0["dpsi"])
out["atoms"] = atoms

# ---------------------------------------------------------------- derived: palette-weighted mean and spread of the DLI phase and of the lens phase A
W = {lv: USAGE[lv] / sum(USAGE.values()) for lv in USAGE}
der = {}
for nm in [str(x) for x in CH_NM]:
    md = sum(W[lv] * atoms[lv][nm]["ddpsi"] for lv in atoms)
    ma = sum(W[lv] * atoms[lv][nm]["dA"] for lv in atoms)
    rd = math.sqrt(sum(W[lv] * (atoms[lv][nm]["ddpsi"] - md) ** 2 for lv in atoms))
    ra = math.sqrt(sum(W[lv] * (atoms[lv][nm]["dA"] - ma) ** 2 for lv in atoms))
    dp = [atoms[lv][nm]["dpsi"] for lv in atoms]
    ddp = [atoms[lv][nm]["ddpsi"] for lv in atoms]
    dA = [atoms[lv][nm]["dA"] for lv in atoms]
    der[nm] = dict(mean_ddpsi=md, rms_ddpsi=rd, minmax_ddpsi=(min(ddp), max(ddp)), ptp_dpsi_abs=max(dp) - min(dp), mean_dA=ma, rms_dA=ra, minmax_dA=(min(dA), max(dA)),
                   cos_pen_dB_dA=-20 * math.log10(math.cos(math.radians(ra))), cos_pen_dB_ddpsi=-20 * math.log10(math.cos(math.radians(rd))))
    # gates per level
    gp = {}
    for lv in atoms:
        r = atoms[lv][nm]
        gp[lv] = dict(eff_ok=r["eff"] >= EFF_GATE, echo_ok=r["echo"] <= ECHO_GATE, pur_ref_ok=min(r["pur"], r["pur_tok"]) >= PUR_FLOOR, pur_delay_ok=r["purm"] >= PUR_FLOOR)
        gp[lv]["all4_ok"] = all(gp[lv].values())
    der[nm]["gates"] = gp
    der[nm]["n_all_ok"] = sum(1 for lv in gp if gp[lv]["all4_ok"])
    der[nm]["fails"] = [lv for lv in gp if not gp[lv]["all4_ok"]]
out["derived"] = der
out["weights"] = W

json.dump(out, open(os.path.join(HERE, "c2_tables.json"), "w", encoding="utf-8"), indent=1)

# ---------------------------------------------------------------- markdown
L = []
L.append("channel nm | k | slab geom (deg) | retro R(-2) | retro arg rel (deg)")
for nm, k in zip(CH_NM, CH_K):
    r = retro[str(nm)]
    L.append("%.3f | %+d | %+.4f | %.7f | %+.4f" % (nm, k, slab[nm], r["R_m2"], r["argR_rel_1550"]))
L.append("")
L.append("carrier (production JSON): nm | sig ref %ceil | sig worst5 | echo ref | echo worst5 | pur worst5 | purm worst5 | bias ref | bias rel 1550 | rel by term (min..max)")
for nm in CH_NM:
    r = car[str(nm)]
    rt = list(r["bias_rel_1550_by_term"].values())
    L.append("%.3f | %.3f | %.3f | %.3f | %.3f | %.6f | %.6f | %.3f | %+.3f | %+.3f..%+.3f" % (nm, r["sig_ref"], r["sig_worst"], r["echo_ref"], r["echo_worst"], r["pur_worst"], r["purm_worst"], r["bias_ref"], r["bias_rel_1550"], min(rt), max(rt)))
L.append("")
L.append("atoms: level | nm | A | dA | B | dpsi | ddpsi | eff | echo | pur ref(rec) | pur tok | pur delay | gates")
for lv, md5, tt in LEVELS:
    for nm in CH_NM:
        r = atoms[lv][str(nm)]
        g = der[str(nm)]["gates"][lv]
        L.append("%s | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f | %.3f | %.3f | %.6f | %.6f | %.6f | %s" % (lv, nm, r["A"], r["dA"], r["B"], r["dpsi"], r["ddpsi"], r["eff"], r["echo"], r["pur"], r["pur_tok"], r["purm"],
                                                                                                         "".join("1" if g[x] else "0" for x in ("eff_ok", "echo_ok", "pur_ref_ok", "pur_delay_ok"))))
L.append("")
L.append("derived: nm | mean ddpsi | rms ddpsi | minmax ddpsi | ptp dpsi | mean dA | rms dA | minmax dA | cospen dA dB | n all4 | fails")
for nm in CH_NM:
    d = der[str(nm)]
    L.append("%.3f | %+.3f | %.3f | %+.2f..%+.2f | %.3f | %+.3f | %.3f | %+.2f..%+.2f | %.4f | %d | %s" % (nm, d["mean_ddpsi"], d["rms_ddpsi"], d["minmax_ddpsi"][0], d["minmax_ddpsi"][1], d["ptp_dpsi_abs"], d["mean_dA"], d["rms_dA"], d["minmax_dA"][0], d["minmax_dA"][1], d["cos_pen_dB_dA"], d["n_all_ok"], ",".join(d["fails"])))
L.append("")
L.append("slab tau ps %.6f ; GDD fs2 %.2f ; input phase keys per wavelength: %s" % (out["slab_tau_ps"], out["slab_gdd_fs2"], out["input_phase_keys_per_wavelength"]))
L.append("channels exact from grid: " + ", ".join("%+d: %.4f" % (int(k), v) for k, v in out["channel_nm_exact_from_grid"].items()))
open(os.path.join(HERE, "c2_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
