import sys, json, math
sys.path.insert(0, r"<CODE_ROOT>\Version_unified\RCWA_custom")
import fdtd_carrier2d_post as Q
R = r"<DATA_ROOT>\Result\261016_fdtd_carrier\261016_sim804_%s_k%s_%s\fdtd2d_result.json"
ref = json.load(open(r"<WORK_ROOT>\fdtd_carrier_261016\driver\rcwa_rounded_refs.json", encoding="utf-8"))
rr = ref["rounded"]["per_term"]; fo = ref["rounded"]["folds"]
EFF_REF = 100 * fo["signal_worst"] / fo["signal_ceiling"]
ECHO_REF = fo["echo_worst_p2_db"]
OUT = r"<WORK_ROOT>\fdtd_carrier_261016\analysis_k124"

def load(t, k):
    p = Q.load_result(R % (t, k, "pth")); m = Q.load_result(R % (t, k, "mth"))
    c = Q.combine(p, m)
    c["energy_p"] = p["energy"]; c["energy_m"] = m["energy"]
    c["conv_p"] = p["runs"]["st"]["logfacts"]["converged"]; c["conv_m"] = m["runs"]["st"]["logfacts"]["converged"]
    c["nt_p"] = p["runs"]["st"]["logfacts"]["nt_run"]; c["nt_m"] = m["runs"]["st"]["logfacts"]["nt_run"]
    c["flux_p"] = p["spacer_flux_err"]; c["flux_m"] = m["spacer_flux_err"]
    c["flags_p"] = [f.split(":")[0] for f in p["flags"]]; c["flags_m"] = [f.split(":")[0] for f in m["flags"]]
    return c

res = {}
for t in ("n3445", "n1616"):
    for k in ("1", "2", "4"):
        res[(t, k)] = load(t, k)

lines = []
def P(s=""):
    print(s); lines.append(s)

P("Li (rounded) references: efficiency %.3f %%, echo %.3f dB (rcwa_rounded_refs.json, MEASURED)" % (EFF_REF, ECHO_REF))
P("grid: k1 512 cells (dx 2.569 nm), k2 1024 (1.285), k4 2048 (0.642 nm)")
P("")
P("== convergence flags / energy (MEASURED) ==")
for (t, k), c in res.items():
    P("%s k%s pth conv=%s nt=%d energy=%.5f flags=%s | mth conv=%s nt=%d energy=%.5f flags=%s" % (
        t, k, c["conv_p"], c["nt_p"], c["energy_p"], c["flags_p"], c["conv_m"], c["nt_m"], c["energy_m"], c["flags_m"]))
P("")
P("== efficiency fold, n3445 (both launches converged at k4) ==")
eff = {k: res[("n3445", k)]["efficiency_pct"] for k in "124"}
for k in "124":
    P("k%s eff %.3f %%  diff to Li %+.3f pp  %s" % (k, eff[k], eff[k] - EFF_REF, "PASS" if abs(eff[k] - EFF_REF) <= 1.0 else "FAIL"))
P("== echo fold, n1616 (R(+2) at -theta = mth run, converged at k4) ==")
ech = {k: res[("n1616", k)]["echo_db"] for k in "124"}
for k in "124":
    P("k%s echo %.3f dB  diff to Li %+.3f dB  %s" % (k, ech[k], ech[k] - ECHO_REF, "PASS" if abs(ech[k] - ECHO_REF) <= 0.5 else "FAIL"))

def conv(name, v, unit, tol):
    d12 = v["2"] - v["1"]; d24 = v["4"] - v["2"]
    P("-- %s: k1->k2 %+.4f %s, k2->k4 %+.4f %s" % (name, d12, unit, d24, unit))
    mono = (d12 * d24 > 0) and abs(d24) < abs(d12)
    r = d24 / d12 if d12 != 0 else float("nan")
    out = dict(d12=d12, d24=d24, ratio=r, monotone=mono)
    if mono:
        ext = v["4"] + d24 * r / (1 - r)
        P("   monotone, ratio %.3f; extrapolated (geometric, DERIVED) %.3f %s, diff to Li %+.3f" % (r, ext, unit, ext - (EFF_REF if unit == "pp" else ECHO_REF)))
        out["extrap"] = ext
    else:
        P("   NOT monotone-converging (ratio %.3f): quote k4 with k2-k4 spread %.4f %s as uncertainty" % (r, abs(d24), unit))
        out["spread_k2k4"] = abs(d24)
    return out
P("")
P("== three-rung convergence ==")
cv_eff = conv("efficiency (n3445)", eff, "pp", 1.0)
cv_echo = conv("echo (n1616 mth)", ech, "dB", 0.5)

P("")
P("== power terms vs Li (abs diff) ==")
def terms(t, k, which):
    c = res[(t, k)]; r = rr[t]
    d = {}
    if "eta_plus" in which: d["eta_plus"] = (c["eta_plus"], r["eta_plus"])
    if "R_go" in which: d["R_go"] = (c["R_go"], r["R_go"])
    if "eta_minus" in which: d["eta_minus"] = (c["eta_minus"], r["eta_minus"])
    if "back" in which: d["back_p1_Rm2"] = (c["back_p1_Rm2"], r["back_p1_Rm2"])
    if "echo" in which: d["echo_R_p2"] = (c["echo_R_p2"], r["echo_worst_p2"])
    return {n: abs(a - b) for n, (a, b) in d.items()}
pt = {}
for k in "124":
    a = terms("n3445", k, ("eta_plus", "R_go", "eta_minus", "back"))
    mx = max(a.items(), key=lambda x: x[1])
    P("k%s n3445 four power terms: max %.2e (%s); all %s" % (k, mx[1], mx[0], {n: float("%.2e" % v) for n, v in a.items()}))
    pt["n3445_k" + k] = a
b = terms("n1616", "4", ("eta_minus", "echo"))
P("k4 n1616 mth-only terms (converged): %s" % {n: float("%.2e" % v) for n, v in b.items()})
b2 = terms("n1616", "4", ("eta_plus", "back"))
P("k4 n1616 pth-dependent terms (NON-CONVERGED, outside headline): %s" % {n: float("%.2e" % v) for n, v in b2.items()})
for k in "12":
    a = terms("n1616", k, ("eta_plus", "R_go", "eta_minus", "back", "echo"))
    P("k%s n1616 all terms: %s" % (k, {n: float("%.2e" % v) for n, v in a.items()}))
P("")
P("== uses of the non-converged n1616 pth run at k4 (excluded from headline) ==")
c = res[("n1616", "4")]
P("n1616 k4 efficiency (pth in R_go, eta_plus): %.3f %% (diff %+.3f pp vs Li 86.948 is NOT the fold; n1616 is not the efficiency worst case)" % (c["efficiency_pct"], c["efficiency_pct"] - EFF_REF))
P("n1616 k4 eta_plus %.5f, R_go_pth %.5f, back_p1_Rm2 %.5f ; reciprocity closure %.2e" % (c["eta_plus"], c["R_go_pth"], c["back_p1_Rm2"], c["recip_closure"]))
for k in "124":
    cc = res[("n1616", k)]
    P("n1616 k%s efficiency %.3f %%" % (k, cc["efficiency_pct"]))
P("")
P("== other per-term values (n3445 k4) ==")
c = res[("n3445", "4")]
for kk in ("eta_plus", "R_go_pth", "R_go_mth", "eta_minus", "back_p1_Rm2", "echo_R_p2", "echo_db", "signal", "efficiency_pct", "purity_pth", "purity_mth", "recip_closure", "recip_direct", "flux_p", "flux_m"):
    P("%s %r" % (kk, c[kk]))
c = res[("n1616", "4")]
P("n1616 k4 echo_R_p2 %r echo_db %r purity_mth %r flux_m %r" % (c["echo_R_p2"], c["echo_db"], c["purity_mth"], c["flux_m"]))
js = {"%s_k%s" % k: {x: v for x, v in c.items()} for k, c in res.items()}
json.dump(dict(rungs=js, eff_ref=EFF_REF, echo_ref=ECHO_REF, conv_eff=cv_eff, conv_echo=cv_echo, power=pt), open(OUT + r"\tab_k124.json", "w", encoding="utf-8"), indent=1, default=str)
open(OUT + r"\tab_k124.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
