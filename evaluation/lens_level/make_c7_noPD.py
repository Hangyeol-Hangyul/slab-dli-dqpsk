# -*- coding: utf-8 -*-
# 261018: Fig. 7(c) c7 excess loss WITHOUT the photodiode-capture term. F = (1/2) T sqrt(R_rt) eta S_max C_PD with
# C_PD = C_ref x 10^(L_tot/10), C_ref = C_8 (eight-level ideal capture); the ideal continuous-lens capture c0(lambda)
# (column c0_ideal_ref) is the PD-capture term (0.617 dB at 1550 nm). Without it: efficiency 4F / c0, excess = excess + 10 log10(c0).
# Writes c7_total_efficiency_noPD.csv beside the old file (old file kept).
import csv, math

SRC = r"<WORK_ROOT>\fig7_data_261017\device_chain_11ch.csv"
OUT = r"<WORK_ROOT>\fig7_layout_261018\c78\c7_total_efficiency_noPD.csv"
rows = list(csv.reader(open(SRC, encoding="utf-8")))
h = rows[0]
idx = lambda p: [i for i, k in enumerate(h) if k.startswith(p)][0]
iL, iF, iEx, iSt, ic0, iCref, iCPD, iLt = (idx("lambda_label_nm"), idx("F_pct"), idx("excess_vs_quarter_dB"), idx("status"), idx("c0_ideal_ref"),
                                          idx("C_ref_pct"), idx("C_PD_pct"), idx("L_tot_final"))
data = sorted(rows[1:], key=lambda r: float(r[iL]))
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["lambda_nm", "total_efficiency_pct", "excess_loss_dB", "anchored", "pd_capture_dB", "status"])
    for r in data:
        c0 = float(r[ic0]); eff = 4 * float(r[iF]); ex = float(r[iEx])
        pd = -10 * math.log10(c0)
        # consistency: C_PD / c0 must equal (C_ref / c0) x 10^(L_tot/10), i.e. eight-level + metaatom-lens terms only
        chk = float(r[iCPD]) / 100 - float(r[iCref]) / 100 * 10 ** (float(r[iLt]) / 10)
        anch = 1 if r[iSt].startswith("DERIVED") else 0
        w.writerow([r[iL], "%.4f" % (eff / c0), "%.5f" % (ex - pd), anch, "%.5f" % pd, r[iSt][:60]])
        print("%s eff %.4f -> %.4f  excess %.5f -> %.5f  pd %.5f  c0 %.6f  C_ref/c0 %.5f (%.4f dB)  CPD-chk %.1e" % (
            r[iL], eff, eff / c0, ex, ex - pd, pd, c0, float(r[iCref]) / 100 / c0, -10 * math.log10(float(r[iCref]) / 100 / c0), chk))
