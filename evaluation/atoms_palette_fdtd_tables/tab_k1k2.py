import sys, json, math, os
sys.path.insert(0, r"<CODE_ROOT>\Version_unified\RCWA_custom")
import fdtd_carrier2d_post as Q
R = r"<DATA_ROOT>\Result\261016_fdtd_carrier\261016_sim804_%s_k%s_%s%s\fdtd2d_result.json"
ref = json.load(open(r"<WORK_ROOT>\fdtd_carrier_261016\driver\rcwa_rounded_refs.json", encoding="utf-8"))
rr = ref["rounded"]["per_term"]
out = {}
def get(t, k, sfx=""):
    p = Q.load_result(R % (t, k, "pth", sfx)); m = Q.load_result(R % (t, k, "mth", sfx))
    c = Q.combine(p, m)
    c["energy_p"] = p["energy"]; c["energy_m"] = m["energy"]
    c["flux_p"] = p["spacer_flux_err"]; c["flux_m"] = m["spacer_flux_err"]
    c["ok_p"] = p["ok"]; c["ok_m"] = m["ok"]
    c["Rp_dir"] = c["R_go_direct_pth"]; c["Rm_dir"] = c["R_go_direct_mth"]
    return c
rows = {}
for t in ("n3445", "n1616", "n2329", "n1768", "c2"):
    for k in ("1", "2"):
        rows[(t, k)] = get(t, k)
keys = ["eta_plus", "R_go_pth", "R_go_mth", "eta_minus", "back_p1_Rm2", "echo_R_p2", "echo_db", "signal", "efficiency_pct",
        "purity_pth", "purity_mth", "recip_closure", "recip_direct", "energy_p", "energy_m", "flux_p", "flux_m"]
res = {}
for (t, k), c in rows.items():
    res["%s_k%s" % (t, k)] = {x: (c[x] if not isinstance(c[x], complex) else str(c[x])) for x in keys}
    res["%s_k%s" % (t, k)]["R_go_direct_pth"] = c["R_go_direct_pth"]
    res["%s_k%s" % (t, k)]["R_go_direct_mth"] = c["R_go_direct_mth"]
json.dump(res, open(r"<WORK_ROOT>\fdtd_carrier_261016\analysis_k1k2\tab_k1k2.json", "w", encoding="utf-8"), indent=1)
def f(x, n=4): return ("%." + str(n) + "f") % x
print("term k | ep Rgo(p/m) em | R(-2) | echo dB | S | eff%")
for t in ("n3445", "n1616", "n2329", "n1768"):
    r = rr[t]
    print(t, "RCWAr", f(r["eta_plus"]), f(r["R_go"]), f(r["eta_minus"]), f(r["back_p1_Rm2"]), f(r["echo_worst_p2_db"], 3), f(r["signal"], 5), f(100 * r["signal"] / (4 / 9) ** 1.5, 3))
    for k in "12":
        c = rows[(t, k)]
        print(t, "k" + k, f(c["eta_plus"]), f(c["R_go_pth"]) + "/" + f(c["R_go_mth"]), f(c["eta_minus"]), f(c["back_p1_Rm2"]), f(c["echo_db"], 3), f(c["signal"], 5), f(c["efficiency_pct"], 3),
              "pur %.6f %.6f" % (c["purity_pth"], c["purity_mth"]), "recip %.2e dir %.2e" % (c["recip_closure"], c["recip_direct"]), "E %.2e %.2e" % (c["energy_p"] if not isinstance(c["energy_p"], dict) else 0, c["energy_m"] if not isinstance(c["energy_m"], dict) else 0), "flux %.1e %.1e" % (c["flux_p"], c["flux_m"]))
c1, c2 = rows[("c2", "1")], rows[("c2", "2")]
print("C2 echo dB k1 %.3f k2 %.3f" % (c1["echo_db"], c2["echo_db"]))
for k in "12":
    ts = ["n3445", "n1616", "n2329", "n1768"]
    effs = {t: rows[(t, k)]["efficiency_pct"] for t in ts}
    ech = {t: rows[(t, k)]["echo_db"] for t in ts}
    print("fold k" + k, "min eff", min(effs.items(), key=lambda x: x[1]), "max echo", max(ech.items(), key=lambda x: x[1]))
