# -*- coding: utf-8 -*-
"""Fig. 6(d) '>12 Lambda' zone echo with the y48 cell (sim805, 261020) -- arithmetic only, read from stored records.
Method = lensdir45_readout_v6_xe201.py echoes(): class echo e = 10 log10((1 - s_y - s_dm - s_dp) Px + s_y Py + s_dm Pdm + s_dp Pdp); a zone with
no measured d cell gives its d share to x (s_dm = s_dp = 0 in the formula). Shares of the (12, 1e9] band from oblique_class_share_stdout.txt
(+x 0, 45, +y 90, 135, -x 180, 225, -y 270, 315). Aperture = 20 log10( sum_z w_z 10^(e_z/20) ) (K1, reproduces the stored -20.2177 dB).
x48 = sim771 x48 sh0/1/2 power mean (as aperture_x48.py); y48 = sim805 (one registration, sh0)."""
import io, json, math, re, csv, sys
sys.stdout.reconfigure(encoding="utf-8")
R = r"<DATA_ROOT>\Result"
V6 = r"<WORK_ROOT>\loop_261008\lens45_li_261011\readout45_v6_xe201\lensdir45_verdict_1overP_v6.json"
SH = r"<WORK_ROOT>\te_beta_orientation_261005\oblique_class_share_stdout.txt"
OUT = r"<WORK_ROOT>\fig6d_y48_261020"


def jl(p):
    return json.load(io.open(p + r"\superramp_result.json", encoding="utf-8"))["needle_results"]["production"]


def pmean(v):
    return 10 * math.log10(sum(10 ** (x / 10) for x in v) / len(v))


x48 = [jl(R + r"\261009_scdc_tmzone_li6\261009_sim771_lensli6_x48_sh%d" % s)["echo_p2_db"] for s in (0, 1, 2)]
y48p = jl(R + r"\261019_scdc_tmzone_y48\261019_sim805_dir_y48_e201")
y48 = y48p["echo_p2_db"]
v = None
for line in io.open(SH, encoding="utf-8"):
    m = re.match(r"^\(12, 1e\+09\]\s+([0-9.]+)\s+(.+)$", line.strip())
    if m:
        v = [float(t) for t in m.group(2).split()]
assert v and len(v) == 8, v
s = dict(x=(v[0] + v[4]) / 100, y=(v[2] + v[6]) / 100, dm=(v[3] + v[5]) / 100, dp=(v[1] + v[7]) / 100)
ver = json.load(io.open(V6, encoding="utf-8"))
zones = ["2-3L", "3-4L", "4-6L", "6-12L", ">12L"]
w = [2.666, 7.462, 25.6, 47.9, 16.372]
ez = dict(ver["echo"]["zone_echo_share_NE"])


def aperture(e):
    return 20 * math.log10(sum(wi / 100 * 10 ** (e[z] / 20) for z, wi in zip(zones, w)))


def shares(e):
    a = [wi / 100 * 10 ** (e[z] / 20) for z, wi in zip(zones, w)]
    return [100 * x / sum(a) for x in a]


ex = pmean(x48)
e_gt = 10 * math.log10((1 - s["y"]) * 10 ** (ex / 10) + s["y"] * 10 ** (y48 / 10))          # readout rule: d share -> x
e_gt_alt = 10 * math.log10((s["x"] * 10 ** (ex / 10) + s["y"] * 10 ** (y48 / 10)) / (s["x"] + s["y"]))   # sensitivity: d share split x/y equally
rows = []
base = aperture(ez)
print("x48 cells %s -> power mean %.3f dB ; y48 %.3f dB (fid_l3 %.4f, nG %d)" % (" ".join("%.3f" % t for t in x48), ex, y48, y48p["fid_l3"], y48p["nG_actual"]))
print(">12 band shares: x %.3f y %.3f dm %.3f dp %.3f (sum %.3f)" % (s["x"], s["y"], s["dm"], s["dp"], sum(s.values())))
print("stored aperture %.4f ; recomputed with >12L = -20.5 : %.4f" % (ver["echo"]["share_NE"], base))
for lab, eg in (("assumed -20.5 (v6)", -20.5), ("x48 only", ex), ("x48 + y48 (readout rule, d->x)", e_gt), ("x48 + y48 (d split x/y)", e_gt_alt)):
    e2 = dict(ez); e2[">12L"] = eg
    ap = aperture(e2)
    sh = shares(e2)
    print("%-32s >12L %.3f dB -> aperture %.3f dB (vs -20 criterion margin %+.3f, vs -20.5 hinge %+.3f) ; shares %s" % (
        lab, eg, ap, -20.0 - ap, -20.5 - ap, " ".join("%.3f" % t for t in sh)))
    rows.append([lab, eg, ap] + sh)
with io.open(OUT + r"\zone_gt12_y48.csv", "w", newline="", encoding="utf-8") as f:
    c = csv.writer(f)
    c.writerow(["case", "gt12_echo_dB", "aperture_echo_dB"] + ["share_%s_pct" % z for z in zones])
    c.writerows(rows)
