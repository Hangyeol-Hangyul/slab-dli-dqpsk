# -*- coding: utf-8 -*-
"""Device eye-opening penalty at 1550 nm with the measured >12 Lambda zone (aperture echo -20.593 dB instead of -20.2177 dB).
Same model and call as fig7_data_261017/g3_eye.py run(): M.case_params(phi, "uniform", eps_db, ptp_deg, ba) -> M.rails_stats(p, "bessel"),
joint worst / mean / best over the echo phase. 1550 nm: phi 0 (bias at its 1550 value, cases (i) and (ii) coincide), ptp 5.1279 deg,
b/a 0.926422 (eye_penalty_11ch.csv row k = 0). CONTROL: echo -20.2177 dB must reproduce 0.6718 / 0.5848 / 0.4214 dB (stored row)."""
import sys, json
sys.path.insert(0, r"<WORK_ROOT>\fig7_data_261017")
sys.dont_write_bytecode = True
import eye_model_g as M   # pins itself to logical cores 10-13, BLAS threads 1
import csv
row = [r for r in csv.reader(open(r"<WORK_ROOT>\fig7_data_261017\eye_penalty_11ch.csv", encoding="utf-8"))]
hdr, r0 = row[0], [r for r in row[1:] if r[0] == "0"][0]
ptp, e0 = float(r0[4]), float(r0[5])
ba = float(r0[hdr.index([h for h in hdr if h.startswith("arm_ratio_b_over_a")][0])])
out = {}
for lab, e in (("control_stored", e0), ("y48", -20.59254496490972)):
    p = M.case_params(0.0, "uniform", eps_db=e, ptp_deg=ptp, ba=ba)
    st, R = M.rails_stats(p, "bessel")
    out[lab] = dict(echo=e, worst=st["joint"]["pen_wc"], mean=st["joint"]["pen_mean"], best=st["joint"]["pen_min"], chi_wc=st["joint"]["chi_wc_deg"])
    print("%-15s echo %.4f dB : worst %.4f  mean %.4f  best %.4f  chi %.1f" % (lab, e, out[lab]["worst"], out[lab]["mean"], out[lab]["best"], out[lab]["chi_wc"]))
c = out["control_stored"]
ok = abs(c["worst"] - float(r0[6])) < 5e-5 and abs(c["mean"] - float(r0[7])) < 5e-5 and abs(c["best"] - float(r0[8])) < 5e-5
print("control vs stored row (%s / %s / %s): %s" % (r0[6], r0[7], r0[8], "PASS" if ok else "FAIL"))
out["control_pass"] = ok
out["inputs"] = dict(ptp=ptp, ba=ba)
json.dump(out, open(r"<WORK_ROOT>\fig6d_y48_261020\eye_1550_y48.json", "w", encoding="utf-8"), indent=1)
