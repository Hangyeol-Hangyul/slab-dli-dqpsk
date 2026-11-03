# -*- coding: utf-8 -*-
"""G fix round (261015), defect (b): arm amplitude ratio b/a PER CHANNEL from the STORED per-channel arm powers.  No solver, no RCWA.

Source (read only): <WORK_ROOT>\\wlband_261012\\scratch\\dr_car_b11\\scdc_extract_H400_300.json  (the carrier file that c2_tables.json was built from; Li nG 257, fixed 1550 nm
incidence angle, center termination n1708 = facts_v3 D9).  Per channel (label = wavelength rounded to 0.1 nm, 11 labels for the 11 grid wavelengths lams_eval_nm):
    P_A = n1708_<ch>_sub_p_TM_T_m1      forward pass, order -1 into the substrate            (arm A, MEASURED)
    R_0 = n1708_<ch>_sub_p_TM_R_0       forward pass, order 0 reflected: goes on to the retroreflector (MEASURED)
    T_1 = n1708_<ch>_sub_m_TM_T_1       returning pass, order +1 into the substrate           (MEASURED)
    R_rt = c2_tables.json retro[<ch>].R_m2   retroreflective grating R(-2), fixed angle         (MEASURED)
    P_B = R_0 x R_rt x T_1              arm B (delayed arm), DERIVED (product of MEASURED values)
    b/a = (P_B / P_A)^(1/2)             DERIVED
The variant WITHOUT R_rt (P_B = R_0 x T_1 = n1708_arm_del at 1550 nm, the facts D9 definition) is carried as a sensitivity.
Self-tests: at 1550.0 nm P_A equals n1708_arm_ref and R_0 x T_1 equals n1708_arm_del (the facts D9 keys) and both round to the printed D9 values 0.28078 / 0.24099; the label list equals
lams_eval_nm of the file and the g2 channel list; needle: the neighbouring label (1550.8) breaks the D9 reproduction.
Writes arm_ratio_11ch.csv, g10_numbers.json, g10_stdout.txt.   Run: python -I -B g10_arm_ratio.py   (cores 10-13, BLAS 1 thread)
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import io
import json
import math
import sys

sys.dont_write_bytecode = True
import psutil                                                                    # noqa: E402


HERE = r"<WORK_ROOT>\fill_B_261103\work\chain\fig7"
CAR = r"<DATA_ROOT>\Result\261030_scdc_kxheld\261030_sim899_kx_car_f100_525f74f3_b11\scdc_extract_H400_300.json"
C2T = r"<WORK_ROOT>\fill_B_261103\work\chain\c2\c2_tables.json"
TERM = "n1708"
LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


car = json.load(io.open(CAR, encoding="utf-8"))
C2 = json.load(io.open(C2T, encoding="utf-8"))
G2 = json.load(io.open(HERE + r"\g2_numbers.json", encoding="utf-8"))
CH, KCH, NM = G2["ch"], G2["k"], G2["nm"]
LAMS = car["lams_eval_nm"]
assert len(LAMS) == 11 and all(abs(a - b) < 5e-4 for a, b in zip(LAMS, NM)), (LAMS, NM)
LAB = ["%.1f" % x for x in LAMS]
assert LAB == ["1546.0", "1546.8", "1547.6", "1548.4", "1549.2", "1550.0", "1550.8", "1551.6", "1552.4", "1553.2", "1554.0"], LAB
say("g10_arm_ratio.py  cores %s  carrier file %s  termination %s  channels %s" % (psutil.Process().cpu_affinity(), CAR, TERM, ", ".join(LAB)))


def arms(lab, rrt):
    pa = car["%s_%s_sub_p_TM_T_m1" % (TERM, lab)]
    r0 = car["%s_%s_sub_p_TM_R_0" % (TERM, lab)]
    t1 = car["%s_%s_sub_m_TM_T_1" % (TERM, lab)]
    return pa, r0, t1, r0 * rrt * t1


# ------------------------------------------------------------------------------------------------------------------ self-tests (facts D9 at 1550.0)
j0 = LAB.index("1550.0")
pa0, r00, t10, pb0 = arms("1550.0", 1.0)
d_pa = abs(pa0 - car[TERM + "_arm_ref"])
d_pb = abs(r00 * t10 - car[TERM + "_arm_del"])
say("TEST 1 (facts D9, 1550.0 nm): P_A %.12f vs n1708_arm_ref %.12f (diff %.1e); R_0 x T_1 %.12f vs n1708_arm_del %.12f (diff %.1e); printed %.5f / %.5f (D9: 0.28078 / 0.24099)" % (
    pa0, car[TERM + "_arm_ref"], d_pa, r00 * t10, car[TERM + "_arm_del"], d_pb, pa0, r00 * t10))
assert d_pa < 1e-12 and d_pb < 1e-10
assert round(pa0, 5) == 0.28078 and round(r00 * t10, 5) == 0.24099
pn, _, _, _ = arms("1550.8", 1.0)
pbn = arms("1550.8", 1.0)[1] * arms("1550.8", 1.0)[2]
caught = abs(pn - car[TERM + "_arm_ref"]) > 1e-4 and abs(pbn - car[TERM + "_arm_del"]) > 1e-6
say("NEEDLE (neighbouring label 1550.8 instead of 1550.0): P_A differs by %.2e, R_0 x T_1 by %.2e: %s" % (pn - car[TERM + "_arm_ref"], pbn - car[TERM + "_arm_del"], "caught" if caught else "MISSED"))
assert caught

# ------------------------------------------------------------------------------------------------------------------ the 11 channels
rows = []
for j in range(11):
    c = CH[j]
    rrt = C2["retro"][c]["R_m2"]
    pa, r0, t1, pb = arms(LAB[j], rrt)
    ba = math.sqrt(pb / pa)
    ba_no = math.sqrt(r0 * t1 / pa)
    arm_pen = 10 * math.log10((1 + ba * ba) / (2 * ba))                        # closed form of the isolated arm-ratio penalty (optical dB)
    rows.append(dict(k=KCH[j], nm=NM[j], key=c, label=LAB[j], P_A=pa, R_0=r0, T_1=t1, R_rt=rrt, P_B=pb, ba=ba, ba_noRrt=ba_no, arm_pen_dB=arm_pen,
                     arm_pen_noRrt_dB=10 * math.log10((1 + ba_no * ba_no) / (2 * ba_no))))
    say("  k %+d  %.3f nm (%s)  P_A %.6f  R_0 %.6f  T_1 %.6f  R_rt %.7f  P_B %.6f  b/a %.6f  (without R_rt %.6f)  arm-ratio-only penalty %.4f dB (closed form)" % (
        KCH[j], NM[j], LAB[j], pa, r0, t1, rrt, pb, ba, ba_no, arm_pen))
ba_si = math.sqrt(0.2410 / 0.2808)
say("SI constant of the pre-fix run: b/a = (0.2410 / 0.2808)^(1/2) = %.6f; this table at 1550.000 nm: %.6f (difference %+.2e; R_rt = %.7f contributes %.2e); range over the band %.6f ... %.6f" % (
    ba_si, rows[j0]["ba"], rows[j0]["ba"] - ba_si, rows[j0]["R_rt"], rows[j0]["ba"] - rows[j0]["ba_noRrt"], min(r["ba"] for r in rows), max(r["ba"] for r in rows)))
assert all(rows[i]["ba"] > rows[i + 1]["ba"] for i in range(10))                    # monotone: P_A rises, P_B nearly flat towards the red
assert abs(rows[j0]["ba"] - ba_si) < 5e-5
# the closed form is DERIVED: needle on the penalty formula (a wrong b/a must move it beyond the print precision)
alt = 10 * math.log10((1 + 0.95 ** 2) / (2 * 0.95))
assert abs(alt - rows[j0]["arm_pen_dB"]) > 1e-3
say("closed form arm-ratio penalty 10 log10((1 + r^2) / (2 r)): 1550 %.4f dB (pre-fix document: 0.0127), 1546.003 %.4f dB, 1554.017 %.4f dB (pre-fix: 0.0127 at every channel)" % (
    rows[j0]["arm_pen_dB"], rows[0]["arm_pen_dB"], rows[10]["arm_pen_dB"]))

hdr = ["k", "lambda_nm", "carrier_file_label_nm", "P_A[MEASURED]", "R_0_forward_pass[MEASURED]", "T_1_return_pass[MEASURED]", "R_rt_R(-2)[MEASURED fixed angle]", "P_B=R_0*R_rt*T_1[DERIVED]",
       "b_over_a=(P_B/P_A)^0.5[DERIVED]", "SENS_b_over_a_without_R_rt[DERIVED]", "arm_ratio_only_penalty_dB[DERIVED closed form]"]
with io.open(HERE + r"\arm_ratio_11ch.csv", "w", encoding="utf-8", newline="") as f:
    f.write(",".join(hdr) + "\n")
    for r in rows:
        f.write(",".join(str(x) for x in (r["k"], "%.3f" % r["nm"], r["label"], "%.7f" % r["P_A"], "%.7f" % r["R_0"], "%.7f" % r["T_1"], "%.7f" % r["R_rt"], "%.7f" % r["P_B"],
                                           "%.7f" % r["ba"], "%.7f" % r["ba_noRrt"], "%.5f" % r["arm_pen_dB"])) + "\n")
io.open(HERE + r"\g10_numbers.json", "w", encoding="utf-8").write(json.dumps(dict(rows=rows, ba_si=ba_si, source=CAR, termination=TERM), indent=1))
say("wrote arm_ratio_11ch.csv, g10_numbers.json; RSS %.2f GB" % (psutil.Process().memory_info().rss / 1e9))
io.open(HERE + r"\g10_stdout.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
