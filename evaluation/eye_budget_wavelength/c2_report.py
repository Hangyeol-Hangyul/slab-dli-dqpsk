# -*- coding: utf-8 -*-
"""Compose C2_results.md body tables (numbers pulled from c2_tables.json, c2_s23.json, lens result JSON). Writes C2_tables_section.md only."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "c2_tables.json"), encoding="utf-8"))
S = json.load(open(os.path.join(HERE, "c2_s23.json"), encoding="utf-8"))
LENS = json.load(open(os.path.join(HERE, "lensband_eval_copy", "lensband_eval_v2_result.json"), encoding="utf-8"))
CH = ["1546.003", "1546.801", "1547.6", "1548.399", "1549.199", "1550.0", "1550.802", "1551.604", "1552.408", "1553.212", "1554.017"]
KK = {"1546.003": 5, "1546.801": 4, "1547.6": 3, "1548.399": 2, "1549.199": 1, "1550.0": 0, "1550.802": -1, "1551.604": -2, "1552.408": -3, "1553.212": -4, "1554.017": -5}
INMAP = {"1546.003": "1546.00", "1547.6": "1547.60", "1550.0": "1550.00", "1552.408": "1552.41", "1554.017": "1554.02"}
FL = 0.996722


def key(nm):
    for k in T["slab_geom_bias_deg"]:
        if abs(float(k) - float(nm)) < 5e-4:
            return k


o = []
o.append("### A1. Top metasurface and slab, 11 channels")
o.append("")
o.append("| ch k | wavelength (nm) | input metagrating T_x / T_y (nG 901) | retroreflective R(-2) | retro arg R(-2) change (deg) | slab geometric bias (deg) |")
o.append("|---|---|---|---|---|---|")
for c in CH:
    k = key(c)
    i = INMAP.get(c)
    ti = "%.5f / %.5f" % (T["input"][i]["eta_x"], T["input"][i]["eta_y"]) if i else "NOT COMPUTED"
    o.append("| %+d | %s | %s | %.7f | %+.3f | %+.3f |" % (KK[c], c, ti, T["retro"][k]["R_m2"], T["retro"][k]["argR_rel_1550"], T["slab_geom_bias_deg"][k]))
o.append("")
o.append("Basis and sources. Input metagrating: RCWA Li (liy, ff 1), nG 901 (889 realised), x lane / y lane branch efficiency; the five wavelengths are 1546.00, 1547.60, 1550.00, 1552.41, 1554.02 nm = channels +5, +3, 0, -3, -5 rounded to 0.01 nm (difference to the grid at most 0.003 nm); source <DATA_ROOT>\\Result\\261008_scdc_in2dli\\261008_sim752_in2dLi_warm_xsec_min_rf050_W1\\input2d.json keys lam<wl>_eta_x / _eta_y; no phase key per wavelength exists (the phase of the input metagrating acts before the split into the arms and cancels in the DLI). Leakage (at most 2.5e-3) and fiber reflection (at most 2.7e-3, worst SOP) per wavelength are in the same keys. Retroreflective grating: RCWA Li nG 257, TM, fixed incidence angle theta(1550), cell A001 (md5 79631f61); source <DATA_ROOT>\\Result\\261011_scdc_retroli\\261011_sim778_retroli_A001_liy_b11_n257\\scdc_retro_H525_450.json keys <ch>_sub_p_TM_R_m2 and <ch>_sub_p_TM_argR_m2 (arg R(-2) at 1550 = +0.5254 deg; slope -1.0746 deg per nm). Slab: analytic, SiO2 t = 592.39 um, Malitson Sellmeier, Lambda = 1315.39 nm, residual of 2 t kz after removal of the linear term at the centre (reproduces <WORK_ROOT>\\paper_scdc_261011_v3\\scratch_facts\\calc_band.py: worst 4.7503 deg, group-delay dispersion -16668.7 fs^2); computed by <WORK_ROOT>\\paper_scdc_261011_v3\\review\\tbd_analysis_261014\\c2_extract.py.")
o.append("")
o.append("### A2. Bottom metagrating (carrier f100, md5 525f74f3), 11 channels, Li, nG 257, five terminations")
o.append("")
o.append("| ch k | wavelength (nm) | eta reference termination n1768 (% of ceiling) | eta worst of 5 | echo reference (dB) | echo worst of 5 (dB) | reference-arm purity worst of 5 | delay-arm purity worst of 5 | bias change vs 1550 (deg) | bias change over the 5 terminations (deg) | four criteria |")
o.append("|---|---|---|---|---|---|---|---|---|---|---|")
for c in CH:
    r = T["carrier"][key(c)]
    rt = list(r["bias_rel_1550_by_term"].values())
    ok = r["sig_worst"] >= 85.0 and r["echo_worst"] <= -20.5 and r["pur_worst"] >= FL and r["purm_worst"] >= FL
    fails = []
    if r["echo_worst"] > -20.5:
        fails.append("echo")
    if r["pur_worst"] < FL:
        fails.append("purity")
    o.append("| %+d | %s | %.3f | %.3f | %.3f | %.3f | %.6f | %.6f | %+.3f | %+.3f to %+.3f | %s |" % (KK[c], c, r["sig_ref"], r["sig_worst"], r["echo_ref"], r["echo_worst"], r["pur_worst"], r["purm_worst"], r["bias_rel_1550"], min(rt), max(rt), "pass" if ok else "FAIL " + "+".join(fails)))
o.append("")
o.append("Source: <DATA_ROOT>\\Result\\261012_scdc_wlband\\261012_sim783_wl_car_f100_525f74f3_b11\\scdc_extract_H400_300.json (job sim783 carrier, 11 channels x 5 terminations); numbers from the copy <WORK_ROOT>\\paper_scdc_261011_v3\\review\\tbd_analysis_261014\\carrier_band_table_copy.py (output carrier_band_table_production_out.txt) and c2_extract.py, equal to the dry-run table in <WORK_ROOT>\\wlband_261012\\section6.md. eta = sqrt(T(-1)(+theta) R(0)(+theta) T(+1)(-theta)) / (4/9)^1.5; echo = R(+2) at -theta; bias = arg R(0)(+theta) + arg T(+1)(-theta) - arg T(-1)(+theta) of the n1768 termination (its value at 1550, -163.012190122 deg, equals the driver key bias_extract_arms). Gates: eta at least 85 %, echo at most -20.5 dB, purity at least 0.996722 on both arms.")
o.append("")
o.append("### A3. Atoms (palette P769, eight levels, Li, nG 1793, eval_only) at 11 channels: summary over the eight levels")
o.append("")
o.append("| ch k | wavelength (nm) | eta range (%), levels below 85 % | echo, worst level (dB), levels above -20.5 | reference-arm purity, lowest, levels below floor | delay-arm purity, lowest, levels below floor | change of A over levels (deg) | change of dpsi over levels (deg) | levels failing any of the four |")
o.append("|---|---|---|---|---|---|---|---|---|")
for c in CH:
    k = key(c)
    at = T["atoms"]
    e = [at[lv][k]["eff"] for lv in at]
    ec = [at[lv][k]["echo"] for lv in at]
    pr = {lv: min(at[lv][k]["pur"], at[lv][k]["pur_tok"]) for lv in at}
    pdl = [at[lv][k]["purm"] for lv in at]
    dA = [at[lv][k]["dA"] for lv in at]
    dd = [at[lv][k]["ddpsi"] for lv in at]
    nf = T["derived"][k]["fails"]
    o.append("| %+d | %s | %.2f to %.2f, %d | %.2f, %d | %.6f, %d | %.6f, %d | %+.2f to %+.2f | %+.2f to %+.2f | %d (%s) |" % (
        KK[c], c, min(e), max(e), sum(1 for x in e if x < 85), max(ec), sum(1 for x in ec if x > -20.5), min(pr.values()), sum(1 for x in pr.values() if x < FL), min(pdl), sum(1 for x in pdl if x < FL),
        min(dA), max(dA), min(dd), max(dd), len(nf), ", ".join(x.replace("p5", ".5") for x in nf) if nf else "none"))
o.append("")
o.append("Source: <DATA_ROOT>\\Result\\261012_scdc_wlband\\261012_sim783_wl_a<level>_<md5>_l<channel>\\atom2d_opt.json (88 jobs: carrier_eff_pct, echo_p2_db, purity, A_deg, B_deg, dpsi_deg) and the delay-arm token jobs ...\\261012_sim783_wl_tok_525f74f3_<level>_b11\\scdc_extract_H400_300.json (keys n<ns>_<ch>_purity / _puritym, carrier at the token of each atom, nG_eval 161). Reference-arm purity = the lower of the atom record and the token value; criteria as in A2 (linewidth does not depend on wavelength). Full 8 x 11 table: <WORK_ROOT>\\paper_scdc_261011_v3\\review\\tbd_analysis_261014\\c2_tables.md. Cross-check: the input map printed by lensband_eval_v2.py (A, record purity, efficiency, token values) equals these numbers for the levels compared (32.5 deg and 167.5 deg at 1554.017 nm).")
o.append("")
o.append("### A4. Atom-level lens terms (11 channels) and lens totals (3 channels, x ramps)")
o.append("")
o.append("Atom terms deficit and purity f200 on all 11 channels: output of the copy <WORK_ROOT>\\paper_scdc_261011_v3\\review\\tbd_analysis_261014\\lensband_eval_copy\\lensband_eval_v2.py --terms-all-channels (stdout run_terms_all_stdout.txt; its channel block equals the stored lensband_eval_v2_result.json):")
o.append("")
o.append("| ch k | wavelength (nm) | deficit (dB) | purity f200 (dB) | sum (dB) | shift vs 1550 (dB) |")
o.append("|---|---|---|---|---|---|")
terms = {"1546.003": (-0.059632, -0.033057, -0.092689, -0.012268), "1546.801": (-0.059963, -0.025794, -0.085757, -0.005336), "1547.6": (-0.060317, -0.019942, -0.080259, 0.000162),
         "1548.399": (-0.060693, -0.016565, -0.077258, 0.003163), "1549.199": (-0.061094, -0.016870, -0.077963, 0.002458), "1550.0": (-0.061521, -0.018901, -0.080421, 0.0),
         "1550.802": (-0.061977, -0.021864, -0.083841, -0.003420), "1551.604": (-0.062464, -0.025633, -0.088096, -0.007675), "1552.408": (-0.062979, -0.030210, -0.093188, -0.012767),
         "1553.212": (-0.063519, -0.035609, -0.099128, -0.018707), "1554.017": (-0.064075, -0.042086, -0.106161, -0.025740)}
for c in CH:
    t = terms[c]
    o.append("| %+d | %s | %.6f | %.6f | %.6f | %+.6f |" % (KK[c], c, t[0], t[1], t[2], t[3]))
o.append("")
o.append("Lens totals (the 21 supercell cells of x06 to x36 at the channel, atom terms at the channel, geometry model fixed = lens designed at 1550 nm and propagated at the channel; rB0 complex; decayed / held tail):")
o.append("")
o.append("| ch k | wavelength (nm) | L_tot (dB) | margin to -0.65 dB | C_PD (%) | aperture echo (dB), zone beyond 12 Lambda at -20.5 / measured x48 | margin to -20.5 dB |")
o.append("|---|---|---|---|---|---|---|")
for c, lk in (("1546.003", "1546.003"), ("1550.0", "1550"), ("1554.017", "1554.017")):
    f = LENS["channels"][lk]["models"]["fixed"]
    r = f["full"]["res"]["rB0_complex"]
    cp = S["cpd"][c]
    ea, e48 = f["full"]["echo_aperture"], f["aperture_x48"]
    o.append("| %+d | %s | %.4f / %.4f | %+.4f / %+.4f | %.2f / %.2f | %.3f / %.3f | %+.3f / %+.3f |" % (KK[c], c, r["total"][0], r["total"][1], r["margin"][0], r["margin"][1], 100 * cp["C_decay"], 100 * cp["C_held"], ea, e48, -20.5 - ea, -20.5 - e48))
open(os.path.join(HERE, "C2_tables_section_raw.md"), "w", encoding="utf-8").write("\n".join(o) + "\n")
print("written", len(o), "lines")
