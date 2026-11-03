# -*- coding: utf-8 -*-
"""Fix round 261015, defect 'C1: band-edge x + y margins presented without the 45 deg increment'.  Reader only (json -> md/json), no solver.
Inputs (read only): C1_band_xy\\lensband_eval_xy_result.json (x only / x + y per channel, 45 deg cells NOT included), C2_zone_gamma\\lens_zone_gamma_result_flips.json
(1550 nm x + y + 45 deg: unmodified total and full_enum echo).  Output: C1_band_edge_summary_fixed.md and .json in this folder."""
import io
import json
import math

C1 = r"<WORK_ROOT>\overnight_261015\C_lens_analyses\C1_band_xy\lensband_eval_xy_result.json"
C2 = r"<WORK_ROOT>\overnight_261015\C_lens_analyses\C2_zone_gamma\lens_zone_gamma_result_flips.json"
OUT = r"<WORK_ROOT>\overnight_261015\fix_261015\C_lens_analyses"
GATE_T, GATE_E = -0.65, -20.5
c1 = json.load(io.open(C1, encoding="utf-8"))["out"]
c2 = json.load(io.open(C2, encoding="utf-8"))
tot45 = c2["unmodified"]["total"]                      # 1550 nm, x + y + 45 deg, decayed / held
echo45 = c2["full_enum"]["echo_share"]
CH = (("1546.003", "1546.003"), ("1550", "1550"), ("1554.017", "1554.017"))
xy1550 = c1["1550"]["x_y"]["total"]
inc_t = [tot45[0] - xy1550[0], tot45[1] - xy1550[1]]   # total increment of the 45 deg ramps at 1550 nm (negative = worse)
inc_e = echo45 - c1["1550"]["x_y"]["echo_share"]       # echo increment (positive = worse)
# the 1550 nm increment is the only measured one; it is carried over to the band edges as an ESTIMATE
rows = []
for lab, key in CH:
    xy = c1[key]["x_y"]
    xo = c1[key]["x_only"]
    r = {"channel": lab, "x_only_total": xo["total"], "xy_total": xy["total"], "xy_margin": xy["margin"], "xy_echo_share": xy["echo_share"], "xy_echo_worst": xy["echo_worst"]}
    if key == "1550":
        r["xy45_total"] = tot45
        r["xy45_margin"] = [tot45[0] - GATE_T, tot45[1] - GATE_T]
        r["xy45_echo"] = echo45
        r["xy45_status"] = "MEASURED (C2 full_enum / SI Table S23)"
    else:
        r["xy45_total_est"] = [xy["total"][0] + inc_t[0], xy["total"][1] + inc_t[1]]
        r["xy45_margin_est"] = [r["xy45_total_est"][0] - GATE_T, r["xy45_total_est"][1] - GATE_T]
        r["xy45_echo_est"] = xy["echo_share"] + inc_e
        r["xy45_status"] = "OPEN: 45 deg cells not computed at this channel; the 1550 nm increment carried over is an ESTIMATE"
    rows.append(r)
json.dump({"gate_total_dB": GATE_T, "gate_echo_dB": GATE_E, "increment_total_1550_dB": inc_t, "increment_echo_1550_dB": inc_e, "rows": rows}, io.open(OUT + r"\C1_band_edge_summary_fixed.json", "w", encoding="utf-8"), indent=1)

L = []
W = L.append
W("# C1 band-edge lens summary, FIXED labelling (261015 fix round)")
W("")
W("Source numbers: C_lens_analyses\\C1_band_xy\\lensband_eval_xy_result.json (unchanged, all 60 checks PASS) and C_lens_analyses\\C2_zone_gamma\\lens_zone_gamma_result_flips.json. Nothing was recomputed; the columns are relabelled and the missing 45 deg information is stated. Gates: total >= %.2f dB, echo <= %.1f dB. Totals are decayed / held tail, dB." % (GATE_T, GATE_E))
W("")
W("**The 'x + y' column of the pre-fix C1 summary is x + y ONLY: the 45 deg (d) cells are not included at the band edges.** At 1550 nm the 45 deg ramps cost %.4f / %.4f dB in the total and %+.4f dB in the echo beyond x + y (SI line 451: 0.039 / 0.049 dB and 0.104 dB). That increment is larger than the held margin of the x + y read-out at 1546.003 nm, so the band-edge lens gate is UNDETERMINED, not passing." % (-inc_t[0], -inc_t[1], inc_e))
W("")
W("## Table C1-fixed. Lens total and aperture echo per channel")
W("")
W("| Channel (nm) | x only: total decay / held | x + y ONLY (45 deg excluded): total decay / held | margin to %.2f dB, x + y only | echo share, x + y only (dB) | x + y + 45 deg: total decay / held | margin, x + y + 45 deg | echo, x + y + 45 deg (dB) | status of the 45 deg cells |" % GATE_T)
W("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if "xy45_total" in r:
        a = "%.4f / %.4f" % tuple(r["xy45_total"])
        b = "%+.4f / %+.4f" % tuple(r["xy45_margin"])
        e = "%.3f" % r["xy45_echo"]
    else:
        a = "OPEN (estimate if the 1550 increment carried over: %.4f / %.4f)" % tuple(r["xy45_total_est"])
        b = "OPEN (estimate: %+.4f / %+.4f)" % tuple(r["xy45_margin_est"])
        e = "OPEN (estimate: %.3f)" % r["xy45_echo_est"]
    W("| %s | %.4f / %.4f | %.4f / %.4f | %+.4f / %+.4f | %.3f | %s | %s | %s | %s |" % (r["channel"], r["x_only_total"][0], r["x_only_total"][1], r["xy_total"][0], r["xy_total"][1], r["xy_margin"][0], r["xy_margin"][1],
                                                                                      r["xy_echo_share"], a, b, e, r["xy45_status"]))
W("")
W("## The 45 deg increment (measured at 1550 nm only)")
W("")
W("- x + y -> x + y + 45 deg at 1550 nm: total %.4f -> %.4f dB decayed, %.4f -> %.4f dB held, i.e. %.4f / %.4f dB; echo %.4f -> %.4f dB, i.e. %+.4f dB." % (
    xy1550[0], tot45[0], xy1550[1], tot45[1], inc_t[0], inc_t[1], c1["1550"]["x_y"]["echo_share"], echo45, inc_e))
W("- Margin at 1550 nm after the increment: %+.4f / %+.4f dB (SI: +0.0492 / +0.0283); echo margin %+.3f dB against %.1f dB." % (tot45[0] - GATE_T, tot45[1] - GATE_T, GATE_E - echo45, GATE_E))
r0 = rows[0]
W("- 1546.003 nm: x + y only margin %+.4f / %+.4f dB; the 1550 nm increment (%.4f / %.4f dB) is %s than the held margin. ESTIMATE, if it carried over: total %.4f / %.4f dB, margin %+.4f / %+.4f dB, so the held tail %s; echo %.3f dB (%s)." % (
    r0["xy_margin"][0], r0["xy_margin"][1], inc_t[0], inc_t[1], "larger" if abs(inc_t[1]) > r0["xy_margin"][1] else "smaller", r0["xy45_total_est"][0], r0["xy45_total_est"][1],
    r0["xy45_margin_est"][0], r0["xy45_margin_est"][1], "would FAIL by %.4f dB" % (-r0["xy45_margin_est"][1]) if r0["xy45_margin_est"][1] < 0 else "would pass", r0["xy45_echo_est"],
    "meets %.1f dB" % GATE_E if r0["xy45_echo_est"] <= GATE_E else "fails %.1f dB" % GATE_E))
r2 = rows[2]
W("- 1554.017 nm: x + y only margin %+.4f / %+.4f dB. ESTIMATE, if the 1550 increment carried over: total %.4f / %.4f dB, margin %+.4f / %+.4f dB; echo %.3f dB, which fails %.1f dB already without the increment (x + y only %.3f dB)." % (
    r2["xy_margin"][0], r2["xy_margin"][1], r2["xy45_total_est"][0], r2["xy45_total_est"][1], r2["xy45_margin_est"][0], r2["xy45_margin_est"][1], r2["xy45_echo_est"], GATE_E, r2["xy_echo_share"]))
W("")
W("## Verdict wording (replaces any 'x + y margin' pass reading at the band edges)")
W("")
W("- 1550 nm: x + y + 45 deg total and echo MEASURED, both pass (SI Table S23).")
W("- 1546.003 nm and 1554.017 nm: the x + y + 45 deg total and echo are OPEN. The x + y only margins above are an upper bound on the true margins IF the 45 deg ramps add loss and echo at the band edges as they do at 1550 nm (-0.039 / -0.049 dB, +0.104 dB); that is an assumption, not a measurement. The held tail at 1546.003 nm would fail if the 1550 nm increment carried over (estimate). At 1554.017 nm the echo fails -20.5 dB with the 45 deg cells excluded.")
W("- Required before any band-edge lens verdict: the 45 deg (d) cells at both band edges (the classes the 1550 nm read-out takes from the 45 deg set: d24 and d36 at orientation d225; the SI quotes d36 at about 8.6 h and 231 GB, so this is a long batch job, not a quick direct run). Not run in this fix round.")
W("")
io.open(OUT + r"\C1_band_edge_summary_fixed.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
