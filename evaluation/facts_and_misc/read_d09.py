import json, os, math
R = r"<DATA_ROOT>\Result\261011_scdc_tmzone_licolp3"
names = ["261011_sim780_f2_d09_p225_li_e201", "261011_sim780_f2_d09_p225_lau_e201", "261011_sim780_f2_d09_p315_li_e201", "261011_sim780_f2_d09_p315_lau_e201",
         "261011_sim780_f1_d09_p225_li_e101", "261011_sim780_f1_d09_p225_lau_e101", "261011_sim780_f1_d09_p315_li_e101", "261011_sim780_f1_d09_p315_lau_e101"]
rows = {}
for n in names:
    p = os.path.join(R, n, "superramp_result.json")
    if not os.path.isfile(p):
        print(n, "NO superramp_result.json; files:", os.listdir(os.path.join(R, n))[:6]); continue
    j = json.load(open(p))
    pr = j["needle_results"]["production"]
    rows[n] = pr
    print(n, "rule", j.get("rcwa_rule"), "nG_ref", j.get("nG_ref"), "power_tA %.5f power_tB %.5f echo_p2_db %.3f" % (pr["power_tA_target"], pr["power_tB_target"], pr["echo_p2_db"]))
for tag in ("f2", "f1"):
    e = "e201" if tag == "f2" else "e101"
    for ps in ("p225", "p315"):
        a = rows.get("261011_sim780_%s_d09_%s_li_%s" % (tag, ps, e)); b = rows.get("261011_sim780_%s_d09_%s_lau_%s" % (tag, ps, e))
        if a and b:
            print(tag, ps, "Li - Laurent: tA %.3f dB, tB %.3f dB, echo %.3f dB" % (10 * math.log10(a["power_tA_target"] / b["power_tA_target"]), 10 * math.log10(a["power_tB_target"] / b["power_tB_target"]), a["echo_p2_db"] - b["echo_p2_db"]))
