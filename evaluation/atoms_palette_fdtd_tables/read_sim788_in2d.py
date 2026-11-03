# -*- coding: utf-8 -*-
"""sim788 input-metagrating readout: k0 vs the stored lossless sim752 record (ng901_* keys), then the k dependence of the branch keys."""
import io
import json
import math

R = r"<DATA_ROOT>\Result"
B = R + r"\261013_scdc_asik"
KS = ["k0", "k1e-5", "k1e-4", "k1e-3", "k1e-2", "k1e-1"]
ref = json.load(io.open(R + r"\261008_scdc_in2dli\261008_sim752_in2dLi_warm_xsec_min_rf050_W1\input2d.json", encoding="utf-8"))
res = {k: json.load(io.open(B + r"\261013_sim788_asik_in2d_%s_r\input2d.json" % k, encoding="utf-8")) for k in KS}
k0 = res["k0"]
num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
keys901 = sorted(k for k in ref if k.startswith("ng901_") and num(ref[k]))
print("stored sim752 ng901 numeric keys:", len(keys901))
pref = sorted({k.split("_", 1)[0] for k in k0 if k.startswith("ng")})
print("k0 result ng prefixes:", pref, "| status", k0.get("status"), "| eval_only", k0.get("eval_only"))
same, diff, missing = 0, [], []
for k in keys901:
    if k not in k0:
        missing.append(k)
    elif k0[k] == ref[k]:
        same += 1
    else:
        diff.append((k, ref[k], k0[k]))
print("k0 vs sim752 ng901_*: identical %d, differ %d, missing in k0 %d" % (same, len(diff), len(missing)))
for d in diff[:12]:
    print("   DIFF", d)
print("   missing sample:", missing[:8])
# the branch quantities: print every ng901 key whose name suggests efficiency / leak / reflection / echo
want = [k for k in keys901 if any(t in k for t in ("eta", "eff", "lane", "leak", "refl", "echo", "T_x", "T_y", "Tx", "Ty", "energy"))]
print("\nselected keys:", want[:30])
print("%-34s" % "key" + "".join("%13s" % k for k in KS))
for key in want[:24]:
    print("%-34s" % key[:34] + "".join("%13.6g" % res[k][key] if num(res[k].get(key)) else "%13s" % "-" for k in KS))
