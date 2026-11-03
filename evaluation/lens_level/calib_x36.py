# -*- coding: utf-8 -*-
"""Paired basis check of the x36 class on the APPROVED palette: 201-eq (sim769 x36_sh*) vs 401-eq (sim771 x36e401_sh*), per registration,
for every registration whose 401-eq cell has landed. Per cell: arm A / B node ratio amp_t/a_pred (dB, deg), rB0 ratio (dB), echo_p2_db.
When all three 401-eq cells exist it also prints the class node (lens_total_li.read_class) on both bases. Read-only, no solver."""
import cmath
import hashlib
import math
import os
import sys

sys.path.insert(0, r"<WORK_ROOT>\loop_261008\lens_li")
import lens_total_li as L                                                      # noqa: E402

R = r"<DATA_ROOT>\Result"
P201 = {sh: R + r"\261009_scdc_tmzone_li5\261009_sim769_lensli5_x36_sh%d" % sh for sh in (0, 1, 2)}
P401 = {sh: R + r"\261009_scdc_tmzone_li6\261009_sim771_lensli6_x36e401_sh%d" % sh for sh in (0, 1, 2)}


def m8(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def cell(p):
    j = L.jload(p)
    L.check_li_flags("li", j, p)
    r = j["needle_results"]["production"]
    out = {"nG_req": j.get("nG_req"), "echo": r["echo_p2_db"], "md5": [m8(a) for a in r["atom_paths"]]}
    for arm in ("A", "B"):
        q = r["lpa"]["arm%s_s%+d" % (arm, int(r["ramp_s"]))]
        out[arm] = complex(*r["amp_t%s_target" % arm]) / complex(q["a_pred_re"], q["a_pred_im"])
    q = r["lpa"]["rB0"]
    out["rB0"] = complex(*r["amp_rB_zero"]) / complex(q["a_pred_re"], q["a_pred_im"])
    return out


done = [sh for sh in (0, 1, 2) if os.path.isfile(os.path.join(P401[sh], "superramp_result.json"))]
print("401-eq registrations landed: %s" % done)
for sh in done:
    a, b = cell(P201[sh]), cell(P401[sh])
    assert a["md5"] == b["md5"], ("palette differs between bases", sh)
    print("x36 sh%d  nG_req %s -> %s  (palette md5 identical, %d slots)" % (sh, a["nG_req"], b["nG_req"], len(a["md5"])))
    for k in ("A", "B", "rB0"):
        print("   %-3s 201: %+.4f dB %+8.3f deg   401: %+.4f dB %+8.3f deg   d %+.4f dB %+.3f deg" % (
            k, L.db(a[k]), math.degrees(cmath.phase(a[k])), L.db(b[k]), math.degrees(cmath.phase(b[k])),
            L.db(b[k]) - L.db(a[k]), math.degrees(cmath.phase(b[k] / a[k]))))
    print("   echo 201: %.3f dB   401: %.3f dB   d %+.3f dB" % (a["echo"], b["echo"], b["echo"] - a["echo"]))
if len(done) == 3:
    for lab, src in (("201-eq", P201), ("401-eq", P401)):
        acc, r, e = L.read_class(src)
        print("x36 %s CLASS: node A %+.4f dB  B %+.4f dB  rB0 %+.4f dB  echo power mean %.3f dB" % (lab, L.db(acc["A"]), L.db(acc["B"]), L.db(r), L.pmean(e)))
