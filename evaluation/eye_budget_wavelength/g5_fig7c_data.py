# -*- coding: utf-8 -*-
"""G part 2, step 1: assemble the data of Figure 7(c) 'Wavelength operation' (six small multiples) from STORED results only (no solver, no RCWA).

Sources (all read-only):
  device_chain_11ch.csv, eye_penalty_11ch.csv, dli_phase_11ch.csv, g2_numbers.json  (this folder, compute run of branch G)
  c2_tables.json  (paper_scdc_261011_v3\\review\\tbd_analysis_261014; carrier worst of five terminations, atoms, retro)
  fix_261015\\C_lens_analyses\\C1_band_edge_summary_fixed.json  (x + y lens totals / echo at 1546.003, 1550, 1554.017 nm)
  fix_261015\\B_input_channels\\b_table.md  (independent cross-check of T_x / T_y)
Labels (every row): MEASURED = value of a stored simulation; DERIVED = arithmetic on measured values; ESTIMATE = assumption-based (interpolation or carried-over increment);
OPEN = missing input (no OPEN rows are needed for the plotted quantities; the OPEN items are listed in the md file).
Writes fig7c_data_long.csv (one row per plotted value) and fig7c_data_wide.csv (one row per channel, value + label columns).
Run: python -I -B g5_fig7c_data.py   (pinned to logical cores 10-13, nothing heavy)
"""
import csv
import io
import json
import re
import sys

sys.dont_write_bytecode = True


HERE = r"<WORK_ROOT>\fig7_data_261017"
C2T = r"<WORK_ROOT>\paper_scdc_261011_v3\review\tbd_analysis_261014\c2_tables.json"
C1F = r"<WORK_ROOT>\overnight_261015\fix_261015\C_lens_analyses\C1_band_edge_summary_fixed.json"
BTAB = r"<WORK_ROOT>\overnight_261015\fix_261015\B_input_channels\b_table.md"

G2 = json.load(io.open(HERE + r"\g2_numbers.json", encoding="utf-8"))
CH = G2["ch"]
KCH = G2["k"]
NM = G2["nm"]
C2 = json.load(io.open(C2T, encoding="utf-8"))
C1 = json.load(io.open(C1F, encoding="utf-8"))


def read_csv(name):
    with io.open(HERE + "\\" + name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


DC = read_csv("device_chain_11ch.csv")
EY = read_csv("eye_penalty_11ch.csv")
DL = read_csv("dli_phase_11ch.csv")
assert len(DC) == len(EY) == len(DL) == 11


def col(rows, prefix):
    keys = [k for k in rows[0].keys() if k.startswith(prefix)]
    assert len(keys) == 1, (prefix, keys)
    return [float(r[keys[0]]) for r in rows]


assert [int(r["k"]) for r in DC] == KCH == [int(r["k"]) for r in EY] == [int(r["k"]) for r in DL]
LAM_EXACT = col(DC, "lambda_grid_nm")
G3 = json.load(io.open(HERE + r"\g3_numbers.json", encoding="utf-8"))
# full-precision values from the JSON files of the compute run; the CSV files (rounded to 4-7 decimals) are used as a cross-check
T_x = [r["Tx"] for r in G2["rows"]]
T_y = [r["Ty"] for r in G2["rows"]]
R_rt = [r["R_rt"] for r in G2["rows"]]
L_final = [r["L_quad"] for r in G2["rows"]]
ETA_W = [r["eta_pct"] for r in G2["rows"]]
LAM_EXACT = [r["nm_grid"] for r in G2["rows"]]
ECHO_AP = list(G2["echo_quad"])
PHASE = [r["total"] for r in G2["dli"]]
PEN_I = [r["i"]["pen_wc"] for r in G3["rows"]]
PEN_II = [r["ii"]["pen_wc"] for r in G3["rows"]]
assert [r["k"] for r in G3["rows"]] == KCH == [r["k"] for r in G2["dli"]] == [r["k"] for r in G2["rows"]]
for name, full, csvcol, tol in (("T_x", T_x, col(DC, "T_x["), 1e-7), ("T_y", T_y, col(DC, "T_y["), 1e-7), ("R_rt", R_rt, col(DC, "R_rt["), 1e-7),
                                ("L_final", L_final, col(DC, "L_tot_final_decayed_dB"), 1e-6), ("eta_w", ETA_W, col(DC, "eta_usage_weighted_pct"), 1e-5),
                                ("echo", ECHO_AP, col(EY, "echo_aperture_dB"), 1e-4), ("phase", PHASE, col(DL, "total_common_deg"), 1e-4),
                                ("pen_i", PEN_I, col(EY, "case_i_worst_chi_dB"), 1e-4), ("pen_ii", PEN_II, col(EY, "case_ii_worst_chi_dB"), 1e-4)):
    assert max(abs(a - b) for a, b in zip(full, csvcol)) < tol, (name, max(abs(a - b) for a, b in zip(full, csvcol)))

# ----------------------------------------------------------------------------------------------------------------------------------- independent cross-checks
# T_x / T_y against the b_table.md (band summary, section 4) -- six decimals
btxt = io.open(BTAB, encoding="utf-8").read()
sec4 = btxt.split("## 4. Band summary")[1].split("## 5.")[0]
rows4 = re.findall(r"^\| ([+-]\d) \| ([0-9.]+) \| ([0-9.]+) \|", sec4, re.M)
assert len(rows4) == 11, len(rows4)
for (kk, ex, ey), tx, ty, kref in zip(rows4, T_x, T_y, KCH):
    assert int(kk) == kref
    assert abs(float(ex) - tx) < 1.5e-6 and abs(float(ey) - ty) < 1.5e-6, (kk, ex, tx, ey, ty)
# R_rt, carrier, atoms against c2_tables.json
for j, c in enumerate(CH):
    r = C2["retro"][c]["R_m2"]
    assert abs(r - R_rt[j]) < 1e-7, (c, r, R_rt[j])

# ----------------------------------------------------------------------------------------------------------------------------------- series
LONG = []     # panel, series, axis, unit, k, lam_exact, lam_label, value, label, note


def add(panel, series, axis, unit, j, value, label, note=""):
    LONG.append(dict(panel=panel, series=series, axis=axis, unit=unit, k=KCH[j], lambda_grid_nm="%.4f" % LAM_EXACT[j], lambda_label_nm="%.3f" % NM[j],
                     value=repr(float(value)), label=label, note=note))


ANG = "fixed 1550 nm incidence angle (THETA_SUB)"
for j in range(11):
    # c1 input metagrating: matched angle
    add("c1", "T_x_branch_efficiency", "left", "%", j, 100.0 * T_x[j], "MEASURED", "matched angle; nG 901; exact grid wavelength; x branch")
    add("c1", "T_y_branch_efficiency", "left", "%", j, 100.0 * T_y[j], "MEASURED", "matched angle; nG 901; exact grid wavelength; y branch")
    # c2 retroreflective grating R(-2)
    add("c2", "R_m2", "left", "%", j, 100.0 * R_rt[j], "MEASURED", ANG + "; Li nG 257")
    # c3 bottom metagrating (carrier), worst of five terminations
    ca = C2["carrier"][CH[j]]
    add("c3", "carrier_S_worst5", "left", "% of ceiling", j, ca["sig_worst"], "MEASURED", ANG + "; carrier sim359 f100, worst of five terminations")
    add("c3", "carrier_echo_worst5", "right", "dB", j, ca["echo_worst"], "MEASURED", ANG + "; worst of five terminations")
    # c4 metaatoms: S range over the eight atoms, usage-weighted mean, worst echo
    effs = [C2["atoms"][a][CH[j]]["eff"] for a in C2["atoms"]]
    echos = [(C2["atoms"][a][CH[j]]["echo"], a) for a in C2["atoms"]]
    ew = max(echos)
    assert min(effs) <= ETA_W[j] <= max(effs)
    add("c4", "atoms_S_min", "left", "% of ceiling", j, min(effs), "MEASURED", ANG + "; minimum over the eight atoms")
    add("c4", "atoms_S_max", "left", "% of ceiling", j, max(effs), "MEASURED", ANG + "; maximum over the eight atoms")
    add("c4", "atoms_S_usage_weighted_mean", "left", "% of ceiling", j, ETA_W[j], "DERIVED", "usage-weighted (H6) mean of the eight MEASURED atom efficiencies")
    add("c4", "atoms_echo_worst", "right", "dB", j, ew[0], "MEASURED", ANG + "; worst (highest) echo over the eight atoms; atom %s" % ew[1])
    # c5 metalens: final basis (x + y + 45 deg), echo
    lab_L = "MEASURED" if abs(KCH[j]) in (0, 5) else "ESTIMATE"
    note_L = ("TOTAL decayed tail, all 45 deg cells measured (v6 read-out); lensband_edge_261017" if abs(KCH[j]) in (0, 5) else
              ("" if False else "quadratic in wavelength through the three anchors (1546.003, 1550, 1554.017 nm)"))
    add("c5", "lens_total_final_basis", "left", "dB", j, L_final[j], lab_L, note_L + "; " + ANG)
    note_E = ("aperture echo NE share-weighted, all 45 deg cells measured (v6 read-out); lensband_edge_261017" if abs(KCH[j]) in (0, 5) else
              ("" if False else "quadratic in wavelength through the three anchors"))
    add("c5", "lens_aperture_echo_final_basis", "right", "dB", j, ECHO_AP[j], lab_L, note_E + "; " + ANG)
    # c6 DLI phase and eye penalty
    add("c6", "common_DLI_phase", "left", "deg", j, PHASE[j], "DERIVED" if KCH[j] != 0 else "DERIVED", "slab (analytic) + retro arg R(-2) (MEASURED) + atoms mean (MEASURED), relative to 1550.000 nm; " + ANG)
    lab_P = "DERIVED" if abs(KCH[j]) in (0, 5) else "ESTIMATE"
    note_P = ("model at the anchors (1546.003, 1550, 1554.017 nm): echo, spread and phase all measured; arm ratio b/a from the stored P_A, P_B" if KCH[j] == 0 else
              "model; edges: echo and lens MEASURED (lensband_edge_261017), interior: ESTIMATE echo (quadratic); arm ratio b/a per channel DERIVED from the stored P_A, P_B (arm_ratio_11ch.csv)")
    add("c6", "eye_penalty_fixed_bias_worst_echo_phase", "right", "dB", j, PEN_I[j], lab_P, "case (i): bias fixed at 1550 nm; " + note_P)
    add("c6", "eye_penalty_recentred_worst_echo_phase", "right", "dB", j, PEN_II[j], lab_P, "case (ii): bias re-centred per channel; " + note_P)

# three-wavelength held-tail total of the lens (MEASURED, lensband_edge_261017); the old x + y-only series is dropped
JMAP = {0: 0, 1: 5, 2: 10}
for i, j in JMAP.items():
    add("c5", "lens_total_held_tail", "left", "dB", j, G2["L_anchor"][i][1], "MEASURED", "TOTAL held tail, all 45 deg cells measured (v6 read-out); lensband_edge_261017; " + ANG)
    assert abs(G2["L_anchor"][i][0] - L_final[j]) < 1e-9 and abs(G2["echo_anchor"][i] - ECHO_AP[j]) < 1e-9

# ----------------------------------------------------------------------------------------------------------------------------------- write
FIELDS = ["panel", "series", "axis", "unit", "k", "lambda_grid_nm", "lambda_label_nm", "value", "label", "note"]
with io.open(HERE + r"\fig7c_data_long.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for r in LONG:
        w.writerow(r)

series_names = []
for r in LONG:
    key = (r["panel"], r["series"])
    if key not in series_names:
        series_names.append(key)
wide_cols = ["k", "lambda_grid_nm", "lambda_label_nm"]
for p, s in series_names:
    wide_cols += ["%s_%s" % (p, s), "%s_%s_label" % (p, s)]
WIDE = {}
for r in LONG:
    row = WIDE.setdefault(int(r["k"]), {"k": r["k"], "lambda_grid_nm": r["lambda_grid_nm"], "lambda_label_nm": r["lambda_label_nm"]})
    row["%s_%s" % (r["panel"], r["series"])] = r["value"]
    row["%s_%s_label" % (r["panel"], r["series"])] = r["label"]
with io.open(HERE + r"\fig7c_data_wide.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=wide_cols, restval="")
    w.writeheader()
    for kk in sorted(WIDE, reverse=True):
        w.writerow(WIDE[kk])

print("rows long:", len(LONG), " series:", len(series_names), " channels:", len(WIDE))
cnt = {}
for r in LONG:
    cnt[(r["panel"], r["series"], r["label"])] = cnt.get((r["panel"], r["series"], r["label"]), 0) + 1
for k2, v in cnt.items():
    print("  %s %-42s %-9s %2d" % (k2[0], k2[1], k2[2], v))
print("cross-checks passed: T_x/T_y vs b_table.md (<1.5e-6), R_rt vs c2_tables.json (<1e-7), lens final basis vs C1 fixed json (<1e-9), atoms mean within range")
