# -*- coding: utf-8 -*-
"""Numbers table (markdown + csv) from echo_eye_series_numbers.json (all echoes) and echo_eye_numbers.json
(first order, echo-free filtered eyes). Formatting only."""
import json
import csv

HERE = r"<WORK_ROOT>\paper_scdc_261011_v3\review\r7_memo_261013\echoeye"
S = json.load(open(HERE + r"\echo_eye_series_numbers.json"))
F = json.load(open(HERE + r"\echo_eye_numbers.json"))
T = S["table"]
INF = float("inf")


def f(x, n=3):
    return "closed" if (x is None or x == INF or x != x) else ("%.*f" % (n, x))


md = []
md.append("# Eye-opening penalty of the echo: numbers (scripts echo_eye_series.py, echo_eye_model.py)\n")
md.append("Definitions. eps = R(+2) at -theta as a fraction of the power incident on the delayed pass (scd_c_grating.py:1135, :1859, :3272-3282). "
          "eta = sqrt(eps x 0.9999530) is the echo-to-arm-B AMPLITUDE ratio at the combiner (0.9999530 = power returned by the retroreflective grating, "
          "SI line 193). Eye opening EO = min('1' level) - max('0' level) of the I or Q eye at the sampling instant, divided by the echo-free opening; "
          "worst of the I and Q rails. Penalty = -10 log10(EO/EO0): optical dB (extra received power for signal-independent noise); the electrical form "
          "-20 log10 is exactly twice. 'worst' = worst over all symbol sequences and over the echo phase theta; 'mean' = mean of the penalty (dB) over theta "
          "uniform in [0, 360 deg); 'best' = best theta. 'All echoes' = the complete geometric series (the echo is reflected into R(+2) again at every return, "
          "ratio eta e^{i theta}, 6 returns kept); 'first order' = one echo only, closed form EO = 1 - sqrt(2) kappa (worst phase), 1 - 4 kappa/pi (mean EO), "
          "1 - kappa (best phase). kappa = eta b/a (device) or eta (generic DLI); b/a = (P_B/P_A)^(1/2).\n")

md.append("## Table 1. Device model D, b/a = 1 (upper bound), all echoes\n")
md.append("| Level | Echo (dB) | Echo (%) | eta (%) | Worst EO, ideal | Worst penalty, ideal (10log, dB) | Worst penalty, electrical (20log, dB) | "
          "Worst penalty, Bessel 0.75 (dB) | Mean over theta, ideal (dB) | Mean over theta, Bessel 0.75 (dB) | Best theta, ideal (dB) | First order only, worst (dB) |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
rows_csv = []
for r in T:
    eps_pct = 100 * 10 ** (r["eps_dB"] / 10)
    md.append("| %s | %.2f | %.4f | %.2f | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        r["label"], r["eps_dB"], eps_pct, 100 * r["eta"], f(r["D_ideal_eo_wc"], 4), f(r["D_ideal_wc"]), f(r["D_ideal_wc_el"]),
        f(r["D_B75_wc"]), f(r["D_ideal_mean"]), f(r["D_B75_mean"]), f(r["D_ideal_best"]), f(r["D_ideal_first_order_wc"])))
    rows_csv.append([r["label"], r["eps_dB"], eps_pct, 100 * r["eta"], r["D_ideal_eo_wc"], r["D_ideal_wc"], r["D_ideal_wc_el"], r["D_B75_wc"],
                     r["D_ideal_mean"], r["D_B75_mean"], r["D_ideal_best"], r["D_ideal_first_order_wc"]])
md.append("")
md.append("Needles. eps = 0: EO/EO0 = 1 exactly (min = max = 1.000000000000000 over theta, for ideal sampling and for all three filters), penalty 0. "
          "Large echoes (worst-case EO; first order only / all echoes, model D / all echoes, generic DLI analytic with 59 echoes):")
md.append("")
md.append("| Echo (dB) | First order, 1 - sqrt(2) eta | All echoes, model D | All echoes, model G (6 echoes) |")
md.append("|---|---|---|---|")
for key, v in S["needles"].items():
    md.append("| %.2f | %.4f | %.4f | %.4f |" % (float(key), v["first_order_eo_wc"], v["all_echoes_eo_wc_ideal"], v["all_echoes_eo_wc_ideal_G"]))
md.append("")
md.append("At -6 dB the first-order formula predicts an open eye (EO 0.2912, 5.36 dB, reproduced by the 16-pattern enumeration to 1e-16); with the echo of the echo "
          "the eye is closed (analytic generic-DLI value -0.34 with 59 echoes, check_series_analytic_log.txt). At -3.0103 dB (eta = 1/sqrt 2) the first-order "
          "worst-case EO is 0 (2.3e-5 from the retroreflection factor).\n")

md.append("## Table 2. Device arm ratio b/a = (P_B/P_A)^(1/2) of the bottom metagrating (model D, all echoes)\n")
md.append("| b/a | Level | Echo (dB) | Worst penalty, ideal (dB) | Worst penalty, Bessel 0.75 (dB) | Mean over theta, ideal (dB) | Mean over theta, Bessel 0.75 (dB) | Best theta, ideal (dB) |")
md.append("|---|---|---|---|---|---|---|---|")
for r in S["ba_rows"]:
    md.append("| %.4f (%s) | %s | %.2f | %s | %s | %s | %s | %s |" % (
        r["ba"], r["case"], r["label"], r["eps_dB"], f(r["ideal_wc"]), f(r["B75_wc"]), f(r["ideal_mean"]), f(r["B75_mean"]), f(r["ideal_best"])))
md.append("")
md.append("b/a = 0.9056 from T+ = 0.299431 and R0 T- = 0.495544^2 (si_v2_r6.md:138, optimum under the echo criterion); b/a = 0.9264 from P_A = 28.08 %, "
          "P_B = 24.10 % (si_v2_r6.md:249, center termination). The arm ratio of the atoms and of the lens is UNVERIFIED (not in the SI tables). "
          "In model D the second echo beats with arm A at amplitude eta^2 independently of b/a, which is why the b/a dependence is weaker than linear.\n")

md.append("## Table 3. Sensitivity at the gate (-20.5 dB), b/a = 1, all echoes\n")
md.append("| Case | Echo-free eye at center, EO0 / sqrt 2 | Worst penalty, model D (dB) | Worst penalty, model G (dB) | Mean over theta, D (dB) | Mean over theta, G (dB) |")
md.append("|---|---|---|---|---|---|")
g = T[0]
EF = F["eyefree"]
md.append("| Ideal sampling, no filter | 1.0000 | %s | %s | %s | %s |" % (f(g["D_ideal_wc"]), f(g["G_ideal_wc"]), f(g["D_ideal_mean"]), f(g["G_ideal_mean"])))
for tag, nm in (("B75", "4th-order Bessel, 0.75 x baud"), ("B70", "4th-order Bessel, 0.70 x baud"), ("G75", "Gaussian, 0.75 x baud")):
    md.append("| %s, sampled at the echo-free eye center | %.4f | %s | %s | %s | %s |" % (
        nm, EF[tag]["eo0_fixed"] / 2 ** 0.5, f(g["D_%s_wc" % tag]), f(g["G_%s_wc" % tag]), f(g["D_%s_mean" % tag]), f(g["G_%s_mean" % tag])))
md.append("| First order only (one echo), ideal | 1.0000 | %s | %s | %s | %s |" % (
    f(g["D_ideal_first_order_wc"]), f(g["G_ideal_first_order_wc"]), f(g["D_ideal_first_order_mean"]), f(g["G_ideal_first_order_mean"])))
bo = [r for r in F["table"] if abs(r["eps_dB"] + 20.5) < 1e-9][0]
md.append("| Bessel 0.75, first order, best sampling instant per theta | %.4f | %s | %s | %s | %s |" % (
    EF["B75"]["eo0_best"] / 2 ** 0.5, f(bo["B75_D_pen_wc_opt_phase"]), f(bo["B75_G_pen_wc_opt_phase"]), f(bo["B75_D_pen_mean_opt_phase"]), f(bo["B75_G_pen_mean_opt_phase"])))
md.append("")
md.append("The rectangular-pulse NRZ eye of the product field has no intersymbol interference at its center for the Bessel filters (EO0 / sqrt 2 = 1.0000, "
          "crossing at 13.55 / 14.55 of 32 samples); a transmitter with finite rise time would lower EO0 but leave the additive echo term unchanged.\n")

md.append("## Table 4. Echo level for a given worst-case penalty (b/a = 1, ideal sampling, optical dB)\n")
md.append("| Worst-case penalty (dB) | Echo (dB), all echoes | Echo (dB), first order |")
md.append("|---|---|---|")
for r in S["inverse"]:
    import math
    first = 20 * math.log10((1 - 10 ** (-r["pen_dB"] / 10)) / math.sqrt(2)) - 10 * math.log10(0.9999530484)
    md.append("| %.2f | %.2f | %.2f |" % (r["pen_dB"], r["eps_dB"], first))
md.append("")
open(HERE + r"\echo_eye_numbers_table.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
with open(HERE + r"\echo_eye_numbers_table.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["level", "echo_dB", "echo_pct", "eta_pct", "worst_EO_ideal_all_echoes", "worst_pen_ideal_10log_dB", "worst_pen_electrical_20log_dB",
                "worst_pen_bessel075_dB", "mean_pen_ideal_dB", "mean_pen_bessel075_dB", "best_theta_pen_ideal_dB", "first_order_worst_pen_dB"])
    for r in rows_csv:
        w.writerow(r)
print("\n".join(md[:24]))
