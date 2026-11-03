# -*- coding: utf-8 -*-
"""Read the sim784 seam JSONs and derive the numbers quoted in facts rows G4b and SI Note S5 (read-only, no solver)."""
import json
import os

import numpy as np

ROOT = r"<DATA_ROOT>\Result\261012_scdc_liseam"
PFX = "261012_sim784_seam_"
COEF = 100.0 * np.sqrt(2.0 / np.pi) / 100.0


def load(tag, kind):
    p = os.path.join(ROOT, PFX + tag, "seam_li_%s.json" % kind)
    return json.load(open(p, encoding="utf-8"))


r = load("par_liy_M48K10", "parallel")
print("keys parallel:", sorted(r.keys()))
print("M48K10 twoNG", r.get("twoNG"), "L_um", r.get("L_um"), "config", {k: r["config"][k] for k in r["config"] if k in ("M", "K", "shift_px", "out_l3_token", "rcwa_rule", "rcwa_ff", "beam_w_um", "out_l3_nm")} if "config" in r else None)
print("S+ S-", r["sum_ReD_sig_plus"], r["sum_ReD_sig_minus"], "loss+ loss-", r["loss_pct_plus"], r["loss_pct_minus"])
p = load("perp_liy_NP16", "perp")
print("keys perp:", sorted(p.keys()))
print("NP16 twoNG", p.get("twoNG"), "NP", p.get("NP"), "config", {k: p["config"][k] for k in p["config"] if k in ("NP", "Kx", "Ky", "shift_px", "out_l3_token", "rcwa_rule", "rcwa_ff")} if "config" in p else None)
print("P", p["sum_ReD_sig"], "loss", p["loss_pct"])
P = p["sum_ReD_sig"]

print("\n-- headline")
for lab, S in (("D+", r["sum_ReD_sig_plus"]), ("D-", r["sum_ReD_sig_minus"])):
    print(lab, "S %.4f P %.4f loss %.4f %%" % (S, P, -(S + P) * COEF))

print("\n-- M ladder (old token, K10)")
allv = []
for M in (24, 32, 40, 48, 56, 64):
    q = load("par_liy_M%dK10" % M, "parallel")
    lp, lm = -(q["sum_ReD_sig_plus"] + P) * COEF, -(q["sum_ReD_sig_minus"] + P) * COEF
    allv += [lp, lm]
    print("M%d L %.1f twoNG %s S %.3f/%.3f loss %.3f/%.3f" % (M, q["L_um"], q.get("twoNG"), q["sum_ReD_sig_plus"], q["sum_ReD_sig_minus"], lp, lm))
print("ladder loss range %.3f .. %.3f" % (min(allv), max(allv)))

print("\n-- termination (M32 K10): S_par and loss = COEF*(|S_par| + |P|)")
toks = ("032p5", "077p5", "122p5", "167p5", "212p5", "257acc", "302p5", "347p5")
ls = []
for k in toks:
    q = load("par_liy_M32K10_t" + k, "parallel")
    tok = q["config"].get("out_l3_token") if "config" in q else None
    lp, lm = -(q["sum_ReD_sig_plus"] + P) * COEF, -(q["sum_ReD_sig_minus"] + P) * COEF
    ls += [lp, lm]
    print("%-7s S %.3f / %.3f  loss %.3f / %.3f  token %s" % (k, q["sum_ReD_sig_plus"], q["sum_ReD_sig_minus"], lp, lm, tok))
print("termination loss range %.3f .. %.3f" % (min(ls), max(ls)))
ref = load("par_liy_M32K10", "parallel")
print("M32 old token loss %.3f / %.3f" % (-(ref["sum_ReD_sig_plus"] + P) * COEF, -(ref["sum_ReD_sig_minus"] + P) * COEF))

print("\n-- K ladder")
for tag in ("par_liy_M24K10", "par_liy_M24K14", "par_liy_M24K20", "par_liy_M48K10", "par_liy_M48K20"):
    q = load(tag, "parallel")
    print(tag, "twoNG", q.get("twoNG"), "S %.3f / %.3f" % (q["sum_ReD_sig_plus"], q["sum_ReD_sig_minus"]))
a, b = load("par_liy_M24K10", "parallel"), load("par_liy_M24K20", "parallel")
print("M24 K10->K20: %.1f %% / %.1f %%" % (100 * (b["sum_ReD_sig_plus"] / a["sum_ReD_sig_plus"] - 1), 100 * (b["sum_ReD_sig_minus"] / a["sum_ReD_sig_minus"] - 1)))
a, b = load("par_liy_M48K10", "parallel"), load("par_liy_M48K20", "parallel")
print("M48 K10->K20: %.1f %% / %.1f %%" % (100 * (b["sum_ReD_sig_plus"] / a["sum_ReD_sig_plus"] - 1), 100 * (b["sum_ReD_sig_minus"] / a["sum_ReD_sig_minus"] - 1)))

print("\n-- perp")
for tag in ("perp_liy_NP12", "perp_liy_NP16", "perp_lix_NP16", "perp_liy_NP16_Kx12", "perp_liy_NP24", "perp_lau_NP16"):
    q = load(tag, "perp")
    c = q.get("config", {})
    print(tag, "twoNG", q.get("twoNG"), "NP", c.get("NP"), "Kx", c.get("Kx"), "Ky", c.get("Ky"), "P %.3f loss %.3f" % (q["sum_ReD_sig"], q["loss_pct"]))
pl = [load(t, "perp")["sum_ReD_sig"] for t in ("perp_liy_NP12", "perp_liy_NP16", "perp_lix_NP16", "perp_liy_NP16_Kx12", "perp_liy_NP24")]
print("perp Li range P %.3f .. %.3f" % (min(pl), max(pl)))
print("headline with lix: %.3f / %.3f" % (-(r["sum_ReD_sig_plus"] + load("perp_lix_NP16", "perp")["sum_ReD_sig"]) * COEF, -(r["sum_ReD_sig_minus"] + load("perp_lix_NP16", "perp")["sum_ReD_sig"]) * COEF))
# overall extreme combination
lo = min(allv + ls)
hi = max(allv + ls)
print("overall loss range over ladder and terminations (perp fixed at NP16 liy): %.3f .. %.3f" % (lo, hi))
# analytic min / max of the termination range from the quoted sums
smin, smax = 0.689, 2.338
print("check: 0.7979*(0.689+0.923) = %.3f ; 0.7979*(2.338+0.923) = %.3f" % (0.7979 * (smin + 0.923), 0.7979 * (smax + 0.923)))
