# -*- coding: utf-8 -*-
"""
Eye-opening penalty of the echo of the slab DLI-DQPSK receiver (SCDC paper, memo 14).

Every constant below is copied from a file; the location is in the comment next to it.
Output goes to the folder OUT only (this script never writes anywhere else).

MODEL (derivation in result.md)
  echo power ratio (the gate number) eps = R(+2) at -theta, normalized to the power INCIDENT on
  the delayed pass (scd_c_grating.py:1135 RT_Solve(normalize=1), :1859, :3272-3282).
  The echo is extracted by the same T(+1) as arm B one round trip later, after one more
  retroreflection (power RHO2), so  |echo|/|arm B| = eta = sqrt(eps*RHO2).

  Model G (generic DLI: bias phase common to the delayed arm, echo in the delayed arm):
      I = cos(D_k - b) + eta cos(D_k + D_{k-1} - th - b)
  Model D (this device: the bias is set by translating the bottom pattern, Note S5):
      A -> e^{+iGd}, B -> e^{-iGd}, C -> e^{-3iGd} (relative orders -1, +1, and +2 then +1);
      in a balanced pair the beat A-C is identical in both PDs and cancels, the beat B-C doubles
      like the signal:   I = cos(D_k - b) + kappa cos(D_{k-1} - th - b),  kappa = eta (P_B/P_A)^(1/2)
Both are exactly linear in the echo amplitude after balanced detection (no second-order term).

Eye numerics: ideal NRZ rectangular pulses, product field of the DLI (piecewise constant per symbol),
electrical 4th-order Bessel (3 dB bandwidth 0.75 / 0.70 x baud) or Gaussian filter, periodic
de Bruijn sequence of 4096 DQPSK symbols (every 6-symbol pattern once), 32 samples per symbol,
sampling at the center of the echo-free eye (mean zero-crossing time + half a symbol).

Run:  python echo_eye_model.py
"""

import json
import numpy as np
from scipy import signal as sg

OUT = r"<WORK_ROOT>\paper_scdc_261011_v3\review\r7_memo_261013\echoeye"

# ---------------------------------------------------------------- constants (with sources)
RHO2 = 0.9999530484             # retroreflected power fraction, si_v2_r6.md line 193
DESIGN = [
    ("gate", -20.50, "main_v4_r6.md:59, si_v2_r6.md:100 (criterion)"),
    ("lens, share-weighted x, y, 45 deg", -20.66, "si_v2_r6.md Table S18, main_v4_r6.md:121"),
    ("lens, x ramps", -20.82, "si_v2_r6.md:398, main_v4_r6.md:120"),
    ("worst atom (nG 1793)", -20.95, "si_v2_r6.md Table S12 (32.5 deg)"),
    ("bottom metagrating, worst of five", -21.22, "si_v2_r6.md:249 (-21.217)"),
]
CONTEXT = [("context", -15.0, "task"), ("context", -25.0, "task")]
NEEDLE = [("needle -6 dB", -6.0, "task"), ("needle -3.0103 dB", -10 * np.log10(2.0), "eta = 1/sqrt(2)")]
PA_OPT, PB_OPT = 0.299431, 0.495544 ** 2     # si_v2_r6.md:138 (T+ = 0.299431, R0 = T- = 0.495544)
PA_CEN, PB_CEN = 0.2808, 0.2410              # si_v2_r6.md:249 (28.08 %, 24.10 %)
BA_OPT = float(np.sqrt(PB_OPT / PA_OPT))
BA_CEN = float(np.sqrt(PB_CEN / PA_CEN))

BETA = {"I": np.deg2rad(45.0), "Q": np.deg2rad(-45.0)}     # main = cos(D - beta), Eq. (S5)
D_SET = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]))
OS = 32


def eta_of(eps_db):
    """echo-to-arm-B amplitude ratio at the combiner (derived: sqrt(eps*RHO2))."""
    return float(np.sqrt(10 ** (eps_db / 10.0) * RHO2))


# ---------------------------------------------------------------- field-level check of the algebra
def field_level_check(rng, n=4000):
    """Build the four quadrant fields from the translation law and form the balanced
    differences directly from |F|^2; compare with the closed forms used below."""
    Gd = np.deg2rad(np.array([22.5, 67.5, -67.5, -22.5]))        # quadrants beta = -45,-135,+135,+45
    beta_q = -2 * Gd
    worst = {"D": 0.0, "G": 0.0}
    a, b = 0.55, 0.50
    for _ in range(n):
        eta = rng.uniform(0, 0.6)
        th = rng.uniform(0, 2 * np.pi)
        sk, sk1, sk2 = np.exp(1j * rng.choice(D_SET, 3) + 1j * rng.uniform(0, 2 * np.pi))
        A, B, C = a * sk, b * sk1, b * eta * np.exp(1j * th) * sk2
        P = np.abs(A * np.exp(1j * Gd) + B * np.exp(-1j * Gd) + C * np.exp(-3j * Gd)) ** 2
        I_f = P[0] - P[2]            # P(-45) - P(+135)
        Q_f = P[3] - P[1]            # P(+45) - P(-135)
        Dk, Dk1 = np.angle(sk * np.conj(sk1)), np.angle(sk1 * np.conj(sk2))
        kap = eta * b / a
        for val, g1 in ((I_f, Gd[0]), (Q_f, Gd[3])):
            pred = 4 * a * b * (np.cos(Dk + 2 * g1) + kap * np.cos(Dk1 - th + 2 * g1))
            worst["D"] = max(worst["D"], abs(val - pred))
        PG = np.abs(A + np.exp(1j * beta_q) * (B + C)) ** 2
        I_g, Q_g = PG[0] - PG[2], PG[3] - PG[1]
        Dk2 = np.angle(sk * np.conj(sk2))
        for val, bb in ((I_g, beta_q[0]), (Q_g, beta_q[3])):
            pred = 4 * a * b * (np.cos(Dk - bb) + eta * np.cos(Dk2 - th - bb))
            worst["G"] = max(worst["G"], abs(val - pred))
    return worst


# ---------------------------------------------------------------- ideal-sampling enumeration
def echo_phase(model, Dk, Dk1, beta):
    return (Dk1 - beta) if model == "D" else (Dk + Dk1 - beta)


def ideal_eye(model, kappa, theta, rail):
    """normalized eye opening (echo-free = 1) over all 16 (D_{k-1}, D_k) pairs, ideal sampling."""
    beta = BETA[rail]
    Dk, Dk1 = np.meshgrid(D_SET, D_SET, indexing="ij")
    main = np.cos(Dk - beta)
    pos = main > 0
    th = np.atleast_1d(theta)[:, None, None]
    S = main[None] + kappa * np.cos(echo_phase(model, Dk, Dk1, beta)[None] - th)
    eo = S[:, pos].min(axis=1) - S[:, ~pos].max(axis=1)
    return eo / np.sqrt(2.0)          # echo-free opening = 2*cos(45 deg) = sqrt(2)


def analytic_eye(kappa, theta):
    """closed form, both models: 1 - sqrt(2) kappa max(|cos u|, |sin u|), u = theta + 45 deg."""
    u = np.asarray(theta) + np.deg2rad(45.0)
    return 1.0 - np.sqrt(2.0) * kappa * np.maximum(np.abs(np.cos(u)), np.abs(np.sin(u)))


def pen(eo, kind=10.0):
    eo = np.asarray(eo, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -kind * np.log10(eo)
    return np.where(eo > 0, out, np.inf)


def finite_mean(p):
    return float(np.mean(p)) if np.all(np.isfinite(p)) else float("inf")


def p95(p):
    q = float(np.percentile(np.where(np.isfinite(p), p, 1e9), 95))
    return q if q < 1e8 else float("inf")


# ---------------------------------------------------------------- filtered NRZ eye
def de_bruijn(k, n):
    a = [0] * k * n
    seq = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                seq.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1, 1)
    return np.array(seq)


SYM = de_bruijn(4, 6)                      # 4096 symbols, every 6-tuple exactly once (cyclic)
assert len(SYM) == 4 ** 6
DK = D_SET[SYM]
DK1 = np.roll(DK, 1)


def filt_H(kind, f3, n):
    f = np.fft.rfftfreq(n, d=1.0 / OS)            # in units of the baud rate
    if kind == "bessel4":
        b, a = sg.bessel(4, 2 * np.pi * f3, "low", analog=True, norm="mag")   # -3 dB at f3
        _, H = sg.freqs(b, a, worN=2 * np.pi * f)
        return H
    if kind == "gauss":
        return np.exp(-0.5 * np.log(2.0) * (f / f3) ** 2)                      # |H(f3)|^2 = 1/2
    raise ValueError(kind)


_WAVE_CACHE = {}


def rail_waves(model, rail, kind, f3):
    """filtered main wave and the two echo basis waves of one rail; arrays (N, OS)."""
    key = (model, rail, kind, f3)
    if key in _WAVE_CACHE:
        return _WAVE_CACHE[key]
    beta = BETA[rail]
    main = np.cos(DK - beta)
    ph = echo_phase(model, DK, DK1, beta)
    n = len(DK) * OS
    H = filt_H(kind, f3, n)
    out = []
    for lvl in (main, np.cos(ph), np.sin(ph)):
        x = np.repeat(lvl, OS)
        out.append(np.fft.irfft(np.fft.rfft(x) * H, n).reshape(-1, OS))
    _WAVE_CACHE[key] = (main > 0, out[0], out[1], out[2])
    return _WAVE_CACHE[key]


def eye_center(y0):
    """sampling phase [samples] at the CENTER of the echo-free eye: mean zero-crossing time of the
    filtered main wave (circular mean over the slot) plus half a symbol."""
    y = y0.reshape(-1)
    idx = np.where(y[:-1] * y[1:] < 0)[0]
    frac = y[idx] / (y[idx] - y[idx + 1])
    ang = 2 * np.pi * ((idx + frac) % OS) / OS
    t_x = (np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi) * OS) % OS
    return int(round((t_x + OS / 2.0) % OS)) % OS, float(t_x)


def eye_vs_theta(model, rail, kind, f3, kappa, thetas, phases):
    pos, y0, yc, ys = rail_waves(model, rail, kind, f3)
    neg = ~pos
    eo0 = np.array([y0[pos, p].min() - y0[neg, p].max() for p in range(OS)])
    ct, st = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
    eo = np.empty((len(thetas), len(phases)))
    for j, p in enumerate(phases):
        Sp = y0[pos, p][None, :] + kappa * (ct * yc[pos, p][None, :] + st * ys[pos, p][None, :])
        Sn = y0[neg, p][None, :] + kappa * (ct * yc[neg, p][None, :] + st * ys[neg, p][None, :])
        eo[:, j] = Sp.min(axis=1) - Sn.max(axis=1)
    return eo0, eo


def summarize_filtered(model, kind, f3, kappa, thetas=None, opt=False, rails=("I", "Q")):
    """EO(theta) normalized by the echo-free opening at the same sampling instant, sampling at the
    echo-free eye center ('fix'); optionally also with the best sampling instant per theta ('opt').
    Worst of the two rails at each theta."""
    thetas = np.deg2rad(np.arange(0.0, 360.0, 0.5)) if thetas is None else thetas
    fixs, opts, info = [], [], {}
    for r in rails:
        pc, tx = eye_center(rail_waves(model, r, kind, f3)[1])
        phases = list(range(OS)) if opt else [pc]
        eo0, eo = eye_vs_theta(model, r, kind, f3, kappa, thetas, phases)
        if opt:
            fixs.append(eo[:, pc] / eo0[pc])
            opts.append(eo.max(axis=1) / eo0.max())
        else:
            fixs.append(eo[:, 0] / eo0[pc])
        info[r] = {"pc": pc, "tx": tx, "eo0_fixed": float(eo0[pc]), "eo0_best": float(eo0.max())}
    return thetas, np.min(fixs, axis=0), (np.min(opts, axis=0) if opt else None), info


def main():
    rng = np.random.default_rng(20261013)
    log = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    w = field_level_check(rng)
    P("field-level check, four quadrant fields from the translation law vs closed form (max abs diff): "
      "D %.3e  G %.3e" % (w["D"], w["G"]))
    assert w["D"] < 1e-12 and w["G"] < 1e-12

    th = np.deg2rad(np.arange(0.0, 360.0, 0.25))
    levels = DESIGN + CONTEXT + NEEDLE
    maxdiff = 0.0
    for lab, e_db, src in levels:
        k = eta_of(e_db)
        for model in ("D", "G"):
            for rail in ("I", "Q"):
                maxdiff = max(maxdiff, float(np.abs(ideal_eye(model, k, th, rail) - analytic_eye(k, th)).max()))
    P("ideal sampling: max |16-pattern enumeration - closed form| over 9 levels x 2 models x 2 rails x 1440 theta: %.3e" % maxdiff)
    assert maxdiff < 1e-12

    z = ideal_eye("D", 0.0, th, "I")
    P("needle eps = 0 (ideal sampling): eye opening min/max over theta = %.15f / %.15f, penalty %.3e dB" %
      (z.min(), z.max(), float(pen(z.min()))))
    assert abs(z.min() - 1.0) < 1e-14 and abs(z.max() - 1.0) < 1e-14   # rounding of cos(), not physics
    k6 = eta_of(-6.0)
    P("needle -6 dB: eta = %.6f, closed form 1 - sqrt2 eta = %.6f, enumerated %.6f, penalty %.4f dB" %
      (k6, 1 - np.sqrt(2) * k6, ideal_eye("G", k6, th, "I").min(), float(pen(1 - np.sqrt(2) * k6))))
    k3 = eta_of(-10 * np.log10(2.0))
    P("needle -3.0103 dB: eta = %.6f, worst-case eye opening %.3e (the retroreflection factor RHO2 leaves 1 - sqrt2 eta = %.3e)" %
      (k3, ideal_eye("G", k3, th, "I").min(), 1 - np.sqrt(2) * k3))
    z0 = summarize_filtered("D", "bessel4", 0.75, 0.0)[1]
    P("needle eps = 0 (filtered, sampled at eye center): EO ratio min/max over theta = %.15f / %.15f, penalty %.3e dB" %
      (z0.min(), z0.max(), float(pen(z0.min()))))
    assert z0.min() == 1.0 and z0.max() == 1.0

    FILT = (("bessel4", 0.75, "B75"), ("bessel4", 0.70, "B70"), ("gauss", 0.75, "G75"))
    eyefree = {}
    for kind, f3, tag in FILT:
        info = summarize_filtered("D", kind, f3, 0.0)[3]["I"]
        eyefree[tag] = info
        P("echo-free filtered eye %s: crossing t_x = %.2f / 32 UI, center phase %d, EO at center / sqrt2 = %.4f, "
          "EO at best phase / sqrt2 = %.4f" % (tag, info["tx"], info["pc"], info["eo0_fixed"] / np.sqrt(2),
                                                info["eo0_best"] / np.sqrt(2)))

    table = []
    for lab, e_db, src in levels:
        k = eta_of(e_db)
        ea = analytic_eye(k, th)
        pa = pen(ea)
        row = {"label": lab, "source": src, "eps_dB": float(e_db), "eps_pct": float(100 * 10 ** (e_db / 10)),
               "eta": k, "eta_pct": 100 * k,
               "an_eo_wc": float(ea.min()), "an_eo_best_theta": float(ea.max()),
               "an_pen_wc_10": float(pen(ea.min())), "an_pen_wc_20": float(pen(ea.min(), 20.0)),
               "an_pen_best_theta_10": float(pen(ea.max())),
               "an_pen_mean": finite_mean(pa), "an_pen_p95": p95(pa), "an_eo_mean": float(ea.mean())}
        for kind, f3, tag in FILT:
            for model in ("D", "G"):
                full = (tag == "B75") and (e_db in [d[1] for d in DESIGN + CONTEXT])
                thetas, fix, opt, info = summarize_filtered(model, kind, f3, k, opt=full)
                pf = pen(fix)
                i = int(np.argmin(fix))
                row["%s_%s_eo_wc" % (tag, model)] = float(fix.min())
                row["%s_%s_pen_wc_10" % (tag, model)] = float(pen(fix.min()))
                row["%s_%s_pen_wc_20" % (tag, model)] = float(pen(fix.min(), 20.0))
                row["%s_%s_pen_mean" % (tag, model)] = finite_mean(pf)
                row["%s_%s_pen_p95" % (tag, model)] = p95(pf)
                row["%s_%s_pen_best_theta" % (tag, model)] = float(pen(fix.max()))
                row["%s_%s_theta_wc_deg" % (tag, model)] = float(np.rad2deg(thetas[i]))
                if opt is not None:
                    row["%s_%s_pen_wc_opt_phase" % (tag, model)] = float(pen(opt.min()))
                    row["%s_%s_pen_mean_opt_phase" % (tag, model)] = finite_mean(pen(opt))
        table.append(row)
        P("%-34s %7.2f dB eta %.4f | ideal: wc EO %.4f pen %.4f dB (20log: %.3f) best-theta %.4f mean %.4f p95 %.4f | "
          "B75-D wc %.4f mean %.4f | B75-G wc %.4f mean %.4f | B70-D wc %.4f | G75-D wc %.4f" %
          (lab, e_db, k, row["an_eo_wc"], row["an_pen_wc_10"], row["an_pen_wc_20"], row["an_pen_best_theta_10"],
           row["an_pen_mean"], row["an_pen_p95"], row["B75_D_pen_wc_10"], row["B75_D_pen_mean"],
           row["B75_G_pen_wc_10"], row["B75_G_pen_mean"], row["B70_D_pen_wc_10"], row["G75_D_pen_wc_10"]))

    ba_rows = []
    for name, ba in (("b/a = 1 (bound)", 1.0), ("b/a of the optimum under the echo criterion", BA_OPT),
                     ("b/a of the center termination", BA_CEN)):
        for lab, e_db, src in DESIGN:
            k = eta_of(e_db) * ba
            ea = analytic_eye(k, th)
            thetas, fix, opt, info = summarize_filtered("D", "bessel4", 0.75, k)
            ba_rows.append({"case": name, "ba": float(ba), "label": lab, "eps_dB": e_db, "kappa": float(k),
                            "an_pen_wc_10": float(pen(ea.min())), "an_pen_mean": finite_mean(pen(ea)),
                            "B75_pen_wc_10": float(pen(fix.min())), "B75_pen_mean": finite_mean(pen(fix))})
    for r in ba_rows:
        P("%-46s b/a %.4f %-34s kappa %.4f ideal wc %.4f mean %.4f | B75 wc %.4f mean %.4f" %
          (r["case"], r["ba"], r["label"], r["kappa"], r["an_pen_wc_10"], r["an_pen_mean"],
           r["B75_pen_wc_10"], r["B75_pen_mean"]))

    inv = []
    for pdb in (0.1, 0.25, 0.5, 1.0, 2.0):
        eo = 10 ** (-pdb / 10)
        kap = (1 - eo) / np.sqrt(2)
        inv.append({"pen_dB": pdb, "eta": float(kap), "eps_dB": float(20 * np.log10(kap) - 10 * np.log10(RHO2))})
        P("worst-case penalty %.2f dB (optical) <-> echo %.2f dB (b/a = 1)" % (pdb, inv[-1]["eps_dB"]))

    grid = np.arange(-32.0, -9.99, 0.5)
    curves = {"eps_dB": grid.tolist(), "an_wc": [], "an_mean": [], "an_best": [], "B75_wc": [], "B75_mean": []}
    for e_db in grid:
        k = eta_of(e_db)
        ea = analytic_eye(k, th)
        curves["an_wc"].append(float(pen(ea.min())))
        curves["an_best"].append(float(pen(ea.max())))
        curves["an_mean"].append(finite_mean(pen(ea)))
        thetas, fix, opt, info = summarize_filtered("D", "bessel4", 0.75, k, thetas=np.deg2rad(np.arange(0, 360, 1.0)))
        curves["B75_wc"].append(float(pen(fix.min())))
        curves["B75_mean"].append(finite_mean(pen(fix)))

    kg = eta_of(-20.5)
    thetas, fix, opt, info = summarize_filtered("D", "bessel4", 0.75, kg)
    i_w = int(np.argmin(fix))
    th_w = float(thetas[i_w])
    pos, y0, yc, ys = rail_waves("D", "I", "bessel4", 0.75)
    json.dump({"p0": info["I"]["pc"], "eo0": info["I"]["eo0_fixed"], "theta_wc_deg": float(np.rad2deg(th_w)),
               "kappa": kg, "OS": OS, "tx": info["I"]["tx"]}, open(OUT + r"\eye_meta.json", "w"))
    np.savez_compressed(OUT + r"\eye_traces.npz", pos=pos, y0=y0, yc=yc, ys=ys, p0=info["I"]["pc"], th=th_w, kappa=kg)

    json.dump({"table": table, "ba_rows": ba_rows, "inverse": inv, "curves": curves, "eyefree": eyefree,
               "BA_OPT": BA_OPT, "BA_CEN": BA_CEN, "RHO2": RHO2, "field_check": w, "ideal_maxdiff": maxdiff},
              open(OUT + r"\echo_eye_numbers.json", "w"), indent=1)
    open(OUT + r"\echo_eye_model_log.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
    P("done")


if __name__ == "__main__":
    main()
