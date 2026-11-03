# -*- coding: utf-8 -*-
"""Per-direction aperture echoes with the measured >12 Lambda zone (recommended case = readout rule, d share -> x).
For every stored v6 case the >12L term at the -20.5 dB gate is removed and the measured term added:
  sum_{z<=12} a_z = 10^(ap/20) - w_gt12 10^(-20.5/20) ;  ap' = 20 log10( sum_{z<=12} a_z + w_gt12 10^(e_gt12/20) )
e_gt12: x only and x + 45 deg (no y) -> x48 power mean; x + y and x + y + 45 deg -> 10 log10(0.75 Px48 + 0.25 Py48) (no 45-deg cell above 12 Lambda,
its share goes to x as in the readout). Also: power-linear and zone-bin variants of x_y_d, band-edge note, and a self-check on the stored shares."""
import io, json, math, sys
sys.stdout.reconfigure(encoding="utf-8")
V6 = r"<WORK_ROOT>\loop_261008\lens45_li_261011\readout45_v6_xe201\lensdir45_verdict_1overP_v6.json"
OUT = r"<WORK_ROOT>\fig6d_y48_261020\per_direction_y48.json"
R = r"<DATA_ROOT>\Result"
ver = json.load(io.open(V6, encoding="utf-8"))
W12 = 0.16372


def jl(p):
    return json.load(io.open(p + r"\superramp_result.json", encoding="utf-8"))["needle_results"]["production"]["echo_p2_db"]


x48 = [jl(R + r"\261009_scdc_tmzone_li6\261009_sim771_lensli6_x48_sh%d" % s) for s in (0, 1, 2)]
ex = 10 * math.log10(sum(10 ** (v / 10) for v in x48) / 3)
ey = jl(R + r"\261019_scdc_tmzone_y48\261019_sim805_dir_y48_e201")
exy = 10 * math.log10(0.75 * 10 ** (ex / 10) + 0.25 * 10 ** (ey / 10))


def swap12(ap_old, e_new, e_old=-20.5):
    rest = 10 ** (ap_old / 20) - W12 * 10 ** (e_old / 20)
    return 20 * math.log10(rest + W12 * 10 ** (e_new / 20))


res = dict(x48_cells=x48, x48_pmean=ex, y48=ey, gt12_xy=exy, cases={})
GT = {"x": ex, "x_y": exy, "x_d": ex, "x_y_d": exy}
for k, d in ver["per_direction"].items():
    e = GT[k]
    r = dict(gt12=e, NE_old=d["echo_share_NE"], NE_new=swap12(d["echo_share_NE"], e),
             swapped_old=d["echo_share_swapped"], swapped_new=swap12(d["echo_share_swapped"], e))
    r["quadrant_agnostic_new"] = max(r["NE_new"], r["swapped_new"])
    res["cases"][k] = r
    print("%-6s >12L %.3f : NE %.3f -> %.3f ; swapped %.3f -> %.3f" % (k, e, r["NE_old"], r["NE_new"], r["swapped_old"], r["swapped_new"]))
# power-linear (1/P echo map in power) and zone-bin reference of the all-direction case: same substitution
for lab in ("power_linear", "zone_bin_reference"):
    o = ver["echo"][lab]["share_NE"]
    osw = ver["echo"][lab]["share_swapped"]
    res[lab] = dict(old=o, new=swap12(o, exy), swapped_old=osw, swapped_new=swap12(osw, exy))
    print("%-18s NE %.3f -> %.3f ; swapped %.3f -> %.3f" % (lab, o, res[lab]["new"], osw, res[lab]["swapped_new"]))
# worse orientation of every class (upper bound): beyond 12 Lambda the worse of x48 / y48
ew = max(ex, ey)
for k, d in ver["per_direction"].items():
    e = ex if k in ("x", "x_d") else ew
    res["cases"][k]["worst_old"] = d["echo_worst_orientation"]
    res["cases"][k]["worst_new"] = swap12(d["echo_worst_orientation"], e)
    print("worst %-6s %.3f -> %.3f (>12L %.3f)" % (k, d["echo_worst_orientation"], res["cases"][k]["worst_new"], e))
res["echo_cost_45deg_old"] = ver["per_direction"]["x_y"]["echo_share_NE"] - ver["per_direction"]["x_y_d"]["echo_share_NE"]
res["echo_cost_45deg_new"] = res["cases"]["x_y"]["NE_new"] - res["cases"]["x_y_d"]["NE_new"]
print("echo cost of the 45 deg ramps beyond x and y: %.3f -> %.3f dB" % (res["echo_cost_45deg_old"], res["echo_cost_45deg_new"]))
json.dump(res, io.open(OUT, "w", encoding="utf-8"), indent=1)
print("saved", OUT)
