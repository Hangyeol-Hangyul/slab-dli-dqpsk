"""make_c9_data.py (261018): data for the new Fig 7(c) c9 graph and the Fig 7(b) ER markers.
ER (DLI arm-balance extinction ratio) = ((1 + r)/(1 - r))^2, r = b/a = (P_B/P_A)^0.5 per channel [arm_ratio_11ch.csv, DERIVED from
MEASURED P_A, R_0, T_1, R(-2)]; atom phase spread = peak-to-peak of the arm phase difference over the eight atoms [eye_penalty_11ch.csv
atom_dpsi_ptp_deg, MEASURED]. Writes c9_er_atom_data.csv (ascending wavelength)."""
import csv, math, os
D = r"<WORK_ROOT>\fig7_data_261017"
OUT = r"<WORK_ROOT>\fig7_layout_261018\c9\c9_er_atom_data.csv"
arm = {r["lambda_nm"]: r for r in csv.DictReader(open(os.path.join(D, "arm_ratio_11ch.csv"), encoding="utf-8"))}
eye = {r["lambda_nm"]: r for r in csv.DictReader(open(os.path.join(D, "eye_penalty_11ch.csv"), encoding="utf-8"))}
assert set(arm) == set(eye) and len(arm) == 11
rows = []
for lam in sorted(arm, key=float):
    r = float(arm[lam]["b_over_a=(P_B/P_A)^0.5[DERIVED]"])
    r2 = float(eye[lam]["arm_ratio_b_over_a[DERIVED per channel from MEASURED P_A P_B; arm_ratio_11ch.csv]"])
    assert abs(r - r2) < 1e-5, (lam, r, r2)
    er = 20.0 * math.log10((1.0 + r) / (1.0 - r))
    rows.append(dict(lambda_nm=lam, b_over_a=r, ER_dB=er, ER_label="DERIVED from MEASURED arm powers",
                     atom_dpsi_ptp_deg=float(eye[lam]["atom_dpsi_ptp_deg[MEASURED]"]), atom_label="MEASURED"))
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for x in rows:
    print("%s  b/a %.6f  ER %.3f dB  atom p-p %.4f deg" % (x["lambda_nm"], x["b_over_a"], x["ER_dB"], x["atom_dpsi_ptp_deg"]))
