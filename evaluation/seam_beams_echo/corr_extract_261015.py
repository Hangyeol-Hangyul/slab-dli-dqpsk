# -*- coding: utf-8 -*-
"""Read-only extraction of the headline residuals from the re-run cancel_result.json (for corrected_report_ko.md)."""
import io
import json
import sys

sys.dont_write_bytecode = True
d = json.load(io.open("<WORK_ROOT>/echo_cancel_check_261015/cancel_result.json", encoding="utf-8"))
tr = [r for r in d["rows"] if r["kind"] == "true"]
print("true rows:", len(tr), "(2 pairs x 12 configurations)")
for m in ("rA", "rB", "rS", "leak", "sig_dev"):
    best = max(tr, key=lambda r: r[m])
    print("max %-8s %.6e  at %s %s" % (m, best[m], best["key"], best["pair"]))
allm = max(max(r[m] for m in ("rA", "rB", "rS", "leak", "sig_dev")) for r in tr)
print("max over all five metrics: %.6e" % allm)
wp = [r for r in d["rows"] if r["kind"] == "wrong_pair"]
print("wrong pairing rA min/max: %.6f %.6f ; rB %.6f %.6f ; rS %.6f %.6f" % (
    min(r["rA"] for r in wp), max(r["rA"] for r in wp), min(r["rB"] for r in wp), max(r["rB"] for r in wp), min(r["rS"] for r in wp), max(r["rS"] for r in wp)))
for h in d["hyps"]:
    hr = [r for r in d["rows"] if r["kind"] == "hyp" and r["hyp"] == h]
    print("hyp %-42s rA %.6f..%.6f rB %.6f..%.6f (n=%d)" % (h, min(r["rA"] for r in hr), max(r["rA"] for r in hr), min(r["rB"] for r in hr), max(r["rB"] for r in hr), len(hr)))
ks = [(k, r["kappa"]["NW"]["kappa_amp"], r["kappa"]["NW"]["nu"]) for k, r in d["results"].items()]
print("kappa(amp) NW min/max: %.5f %.5f ; nu min/max: %.5f %.5f" % (min(x[1] for x in ks), max(x[1] for x in ks), min(x[2] for x in ks), max(x[2] for x in ks)))
print("configurations:", len(d["results"]), "; verdict:", {k: v for k, v in d["verdict"].items() if isinstance(v, bool)})
