# -*- coding: utf-8 -*-
"""liseam_aggregate.py -- read the sim784 seam results (Main_rcwa_seam_li.py JSONs) and print the controls, the ladders and the
Li replacement of facts row G4. Read-only. Missing folders are reported, not guessed.

usage: python liseam_aggregate.py [--root <DATA_ROOT>\\Result\\261012_scdc_liseam] [--prefix 261012_sim784_seam_]

Definitions (identical to the 260929 test, calc_misc.py "loss D+" line):
  S+ / S-  = sum over the SIGNAL passes (ep, r0, em) of Re D for the +d / -d seam of the parallel half-half cell [um]
  P        = sum over the signal passes of Re D of the perpendicular cell (mean of its two seams) [um]
  loss(%)  = -(S + P) * 100 * sqrt(2/pi) / w   with w = 100 um  (0.7979 %/um): signal AMPLITUDE loss of one parallel + one
             perpendicular seam line through the centre of the beam
G4 of facts_v3.md, Laurent: 0.798 * (1.004 + 0.828) = 1.46 % (D+), 0.798 * (1.188 + 0.828) = 1.61 % (D-).
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=r"<DATA_ROOT>\Result\261012_scdc_liseam")
ap.add_argument("--prefix", default="261012_sim784_seam_")
a = ap.parse_args()
COEF = 100.0 * np.sqrt(2.0 / np.pi) / 100.0       # % per um at w = 100 um


def load(tag, kind):
    p = os.path.join(a.root, a.prefix + tag, "seam_li_%s.json" % kind)
    if not os.path.isfile(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def par(tag):
    r = load(tag, "parallel")
    return r


def fmt(r):
    return "%+.3f / %+.3f" % (r["sum_ReD_sig_plus"], r["sum_ReD_sig_minus"]) if r else "MISSING"


print("=== CONTROLS ===")
for tag in ("m1_lau", "m1_liy", "m1_liy_r"):
    r = load(tag, "m1ref")
    if r and r.get("report_cmp"):
        print("  %-10s report() vs stored record: max|diff| %.2e over %d keys (pass < 1e-9)  %s"
              % (tag, r["report_cmp"]["max_abs_diff"], len(r["report_cmp"]["pairs"]),
                 "PASS" if r["report_cmp"]["max_abs_diff"] < 1e-9 else "FAIL"))
    else:
        print("  %-10s MISSING" % tag)
for tag in ("par_null_liy_M16",):
    r = par(tag)
    print("  %-18s null (shift 0): max|Re D| over passes %s" % (tag, ("%.1e" % max(abs(v["Dp_um"][0]) + abs(v["Dm_um"][0]) for v in r["passes"].values())) if r else "MISSING"))
r = load("perp_null_liy_NP4", "perp")
print("  %-18s null (shift 0): max|Re D| over passes %s" % ("perp_null_liy_NP4", ("%.1e" % max(abs(v["D_um"][0]) for v in r["passes"].values())) if r else "MISSING"))
r = par("par_lau_M48K10")
if r:
    d1, d2 = abs(r["sum_ReD_sig_plus"] - (-1.004)), abs(r["sum_ReD_sig_minus"] - (-1.188))
    print("  par_lau_M48K10   Laurent S+ / S- = %s  vs G4 -1.004 / -1.188   diff %.4f / %.4f um  %s"
          % (fmt(r), d1, d2, "PASS" if max(d1, d2) < 0.0015 else "FAIL"))
else:
    print("  par_lau_M48K10   MISSING")
r = load("perp_lau_NP16", "perp")
if r:
    d = abs(r["sum_ReD_sig"] - (-0.828))
    print("  perp_lau_NP16    Laurent P = %+.3f  vs G4 -0.828  diff %.4f um  %s" % (r["sum_ReD_sig"], d, "PASS" if d < 0.0015 else "FAIL"))
else:
    print("  perp_lau_NP16    MISSING")

print("\n=== LI PARALLEL SEAM, M ladder (old token, K10, s64):  S+ / S-   loss+ / loss- [%] ===")
rows = []
for Mv in (24, 32, 40, 48, 56, 64):
    r = par("par_liy_M%dK10" % Mv)
    if r:
        rows.append((Mv, r))
        print("  M%-3d L %.1f um   %s   %.3f / %.3f   max loc(sig) %.2f" % (Mv, r["L_um"], fmt(r), r["loss_pct_plus"], r["loss_pct_minus"],
                                                                         min(r["passes"][k]["loc"] for k in ("ep", "r0", "em"))))
    else:
        print("  M%-3d MISSING" % Mv)
if rows:
    sp = np.array([r["sum_ReD_sig_plus"] for _, r in rows]); sm = np.array([r["sum_ReD_sig_minus"] for _, r in rows])
    print("  mean over the ladder: %+.3f / %+.3f   spread (max-min): %.3f / %.3f um   (Laurent ladder had +-15 %% scatter)" % (sp.mean(), sm.mean(), np.ptp(sp), np.ptp(sm)))

print("\n=== K ladder (Li, old token) ===")
for tag in ("par_liy_M24K10", "par_liy_M24K14", "par_liy_M24K20", "par_liy_M48K10", "par_liy_M48K20"):
    r = par(tag)
    print("  %-16s %s" % (tag, fmt(r)))
r1, r2 = par("par_liy_M48K10"), par("par_liy_M48K20")
if r1 and r2:
    print("  M48 K10 -> K20: dS+ %+.3f um (%.1f %%)  dS- %+.3f um (%.1f %%)" % (
        r2["sum_ReD_sig_plus"] - r1["sum_ReD_sig_plus"], 100 * (r2["sum_ReD_sig_plus"] / r1["sum_ReD_sig_plus"] - 1),
        r2["sum_ReD_sig_minus"] - r1["sum_ReD_sig_minus"], 100 * (r2["sum_ReD_sig_minus"] / r1["sum_ReD_sig_minus"] - 1)))

print("\n=== TERMINATION (palette-atom tokens, Li, M32 K10)  [the old token at M32 is the reference] ===")
ref = par("par_liy_M32K10")
print("  %-10s %s" % ("old token", fmt(ref)))
tk = []
for k in ("032p5", "077p5", "122p5", "167p5", "212p5", "257acc", "302p5", "347p5"):
    r = par("par_liy_M32K10_t" + k)
    print("  %-10s %s" % (k, fmt(r)))
    if r:
        tk.append(r)
if tk:
    sp = np.array([r["sum_ReD_sig_plus"] for r in tk]); sm = np.array([r["sum_ReD_sig_minus"] for r in tk])
    print("  over the 8 atoms: S+ %+.3f .. %+.3f   S- %+.3f .. %+.3f um" % (sp.min(), sp.max(), sm.min(), sm.max()))

print("\n=== LI PERPENDICULAR SEAM (old token, s128): P = sum ReD_sig [um], loss [%] ===")
perp = {}
for tag in ("perp_liy_NP12", "perp_liy_NP16", "perp_lix_NP16", "perp_liy_NP16_Kx12", "perp_liy_NP24", "perp_lau_NP16"):
    r = load(tag, "perp")
    perp[tag] = r
    print("  %-20s %s" % (tag, ("P %+.3f um  loss %.3f %%  (2nG %d, Ly %.1f um)" % (r["sum_ReD_sig"], r["loss_pct"], r["twoNG"], r["NP"] * 1.31539)) if r else "MISSING"))

print("\n=== REPLICATES (two runs, identical keys): max |difference| over every numeric field ===")
for ta, tb, kind in (("par_liy_M48K10", "par_liy_M48K10_r", "parallel"), ("par_liy_M24K10", "par_liy_M24K10_r", "parallel"),
                     ("perp_liy_NP16", "perp_liy_NP16_r", "perp"), ("m1_liy", "m1_liy_r", "m1ref")):
    ra, rb = load(ta, kind), load(tb, kind)
    if not (ra and rb):
        print("  %-18s vs %-20s MISSING" % (ta, tb))
        continue
    def flat(o, pre=""):
        out = {}
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("config", "dt", "peak_gb", "elapsed_total_sec"):
                    continue
                out.update(flat(v, pre + "/" + str(k)))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                out.update(flat(v, pre + "/%d" % i))
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            out[pre] = float(o)
        return out
    fa, fb = flat(ra), flat(rb)
    worst = max(((abs(fa[k] - fb[k]) / (abs(fa[k]) + 1e-12), k) for k in fa if k in fb), default=(np.nan, ""))
    print("  %-18s vs %-20s worst relative difference %.2e at %s" % (ta, tb, worst[0], worst[1]))

print("\n=== G4 REPLACEMENT (Li, headline M48 K10 parallel + NP16 perpendicular; w = 100 um) ===")
r48, rp = par("par_liy_M48K10"), perp.get("perp_liy_NP16")
if r48 and rp:
    P = rp["sum_ReD_sig"]
    lp, lm = -(r48["sum_ReD_sig_plus"] + P) * COEF, -(r48["sum_ReD_sig_minus"] + P) * COEF
    print("  Li   : %.2f %% (D+)  /  %.2f %% (D-)   = %.4f %%/um x (%.3f + %.3f) / (%.3f + %.3f) um"
          % (lp, lm, COEF, -r48["sum_ReD_sig_plus"], -P, -r48["sum_ReD_sig_minus"], -P))
    rl, pl = par("par_lau_M48K10"), perp.get("perp_lau_NP16")
    if rl and pl:
        print("  Laurent control through the same code: %.2f %% / %.2f %%  (G4: 1.46 / 1.61)"
              % (-(rl["sum_ReD_sig_plus"] + pl["sum_ReD_sig"]) * COEF, -(rl["sum_ReD_sig_minus"] + pl["sum_ReD_sig"]) * COEF))
    # range over the ladder / tokens / lix
    allp = [(-(r["sum_ReD_sig_plus"] + P) * COEF) for _, r in rows] + [(-(r["sum_ReD_sig_minus"] + P) * COEF) for _, r in rows]
    if allp:
        print("  range over the M ladder (both signs, NP16): %.2f .. %.2f %%" % (min(allp), max(allp)))
    if perp.get("perp_lix_NP16"):
        Pl = perp["perp_lix_NP16"]["sum_ReD_sig"]
        print("  with lix instead of liy for the perpendicular cell: %.2f / %.2f %%" % (-(r48["sum_ReD_sig_plus"] + Pl) * COEF, -(r48["sum_ReD_sig_minus"] + Pl) * COEF))
    # echo, first order (EXPLORATORY: the G4 echo sentence's own definition is not in the logs)
    def echo_db(rpar, rperp, sign):
        D = (rpar["passes"]["echo"]["Dp_um" if sign > 0 else "Dm_um"][0] + 1j * rpar["passes"]["echo"]["Dp_um" if sign > 0 else "Dm_um"][1]
             + rperp["passes"]["echo"]["D_um"][0] + 1j * rperp["passes"]["echo"]["D_um"][1])
        return 10 * np.log10(abs(1 + D * COEF / 100.0) ** 2), D
    for lab, (x, y) in (("Li", (r48, rp)), ("Laurent", (rl, pl))):
        if x and y:
            e1, D1 = echo_db(x, y, +1)
            e2, D2 = echo_db(x, y, -1)
            print("  echo (first-order |1 + D_echo*0.00798/um|^2, parallel + perp, %s): %+.2f dB (D+) / %+.2f dB (D-)   "
                  "[single-period echo P1 %.3e = %.1f dB; the G4 sentence '-30.8 -> -27 dB' is NOT reproduced by this convention]"
                  % (lab, e1, e2, x["passes"]["echo"]["P1"], 10 * np.log10(x["passes"]["echo"]["P1"])))
else:
    print("  needs par_liy_M48K10 and perp_liy_NP16")
