# -*- coding: utf-8 -*-
"""Independent analytic check of the all-echoes eye for the generic DLI (model G), ideal sampling.
Decision variable of a rail: cos(D_k - b) + sum_{n>=1} eta^n cos(S_n - n theta - b), S_n = D_k + D_{k-1} + ... + D_{k-n}
(multiples of 90 deg, every D_{k-j}, j >= 1, free), so each echo term can independently take the worst of four
phases:  EO(theta) = 1 - sqrt(2) sum_n eta^n max(|cos u_n|, |sin u_n|),  u_n = n theta + 45 deg.
Compare max over theta with the field-level engine of echo_eye_series.py (model G, necho = 6)."""
import numpy as np
import json

HERE = r"<WORK_ROOT>\paper_scdc_261011_v3\review\r7_memo_261013\echoeye"
RHO2 = 0.9999530484
S = json.load(open(HERE + r"\echo_eye_series_numbers.json"))
th = np.deg2rad(np.arange(0.0, 360.0, 0.5))
out = []
for r in S["table"]:
    if r["eps_dB"] < -14:
        eta = np.sqrt(10 ** (r["eps_dB"] / 10) * RHO2)
        eos = []
        for beta in (45.0, -45.0):                      # I and Q rail
            tot = np.zeros_like(th)
            for n in range(1, 7):
                u = n * th + np.deg2rad(beta)
                tot += eta ** n * np.maximum(np.abs(np.cos(u)), np.abs(np.sin(u)))
            eos.append(1 - np.sqrt(2) * tot)
        eo = np.minimum(eos[0], eos[1])
        pw = -10 * np.log10(eo.min())
        line = "%-34s %7.2f dB: analytic all-echo worst penalty (G, 6 echoes) %.4f dB; field-level engine %.4f dB; diff %.1e" % (
            r["label"], r["eps_dB"], pw, r["G_ideal_wc"], abs(pw - r["G_ideal_wc"]))
        print(line)
        out.append(line)
        assert abs(pw - r["G_ideal_wc"]) < 5e-4   # the de Bruijn sequence (order 6) cannot realise the independent 6th-echo choice, eta^6 ~ 1e-6
open(HERE + r"\check_series_analytic_log.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")

# large-echo needles: the geometric series must close the eye where the first-order formula still shows it open
N2 = json.load(open(HERE + r"\echo_eye_series_numbers.json"))["needles"]
extra = []
for key, v in N2.items():
    e_db = float(key)
    eta = np.sqrt(10 ** (e_db / 10) * RHO2)
    best = 9.0
    for beta in (45.0, -45.0):
        tot = np.zeros_like(th)
        for n in range(1, 60):
            u = n * th + np.deg2rad(beta)
            tot += eta ** n * np.maximum(np.abs(np.cos(u)), np.abs(np.sin(u)))
        best = min(best, (1 - np.sqrt(2) * tot).min())
    line = "needle %s dB: analytic generic-DLI worst-case EO, 59 echoes: %.4f; engine (6 echoes, model G): %.4f; first order: %.4f" % (
        key, best, v["all_echoes_eo_wc_ideal_G"], v["first_order_eo_wc"])
    print(line)
    extra.append(line)
open(HERE + r"\check_series_analytic_log.txt", "a", encoding="utf-8").write("\n".join(extra) + "\n")
