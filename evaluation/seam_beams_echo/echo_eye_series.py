# -*- coding: utf-8 -*-
"""Eye-opening penalty with the COMPLETE echo series (echo of the echo, ...), device model D and generic model G.

The echo returns to the point where it was launched and is reflected into R(+2) again, so the field
at a PD is  F_q = A e^{iGd} + sum_{n>=0} b (eta e^{i theta})^n s_{k-1-n} e^{-i(1+2n)Gd}   (D; n = 0 is arm B)
or a common bias phase on the delayed arm (G).  The balanced difference of the two PDs of a rail is formed
directly from |F|^2 for the 4096-symbol de Bruijn sequence; the filtered sample at the echo-free eye
center is  y_k = sum_j g_j L_{k-j}  with the symbol response g_j of the electrical filter (taken from the
same FFT filter as echo_eye_model.py).  With one echo only (necho = 1) this must reproduce the first-order
results of echo_eye_model.py: that comparison is the self-test of this file.

Writes echo_eye_series_numbers.json, echo_eye_series_log.txt, eye_traces2.npz (output folder only)."""
import json
import importlib.util
import sys
import numpy as np

HERE = r"<WORK_ROOT>\paper_scdc_261011_v3\review\r7_memo_261013\echoeye"
spec = importlib.util.spec_from_file_location("eem", HERE + r"\echo_eye_model.py")
m = importlib.util.module_from_spec(spec)
sys.modules["eem"] = m
spec.loader.exec_module(m)
OS, RHO2 = m.OS, m.RHO2
N = len(m.DK)
psi = np.cumsum(m.DK)                       # Delta'_k = arg(s_k s_{k-1}^*) = psi_k - psi_{k-1}
assert abs(np.exp(1j * psi[-1]) - 1) < 1e-9  # the periodic sequence closes
S = np.exp(1j * psi)
GD = {"I": (np.deg2rad(22.5), np.deg2rad(-67.5)), "Q": (np.deg2rad(-22.5), np.deg2rad(67.5))}  # P(-45)-P(+135), P(+45)-P(-135)
NECHO = 6
FILT = (("ideal", None, 0.0), ("bessel4", 0.75, 0.0), ("bessel4", 0.70, 0.0), ("gauss", 0.75, 0.0))
log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


def level_wave(model, rail, eta, ba, th, necho):
    """balanced output of one rail: shape (n_theta, N)."""
    out = []
    for g in GD[rail]:
        F = np.broadcast_to(S * np.exp(1j * g), (len(th), N)).astype(complex)
        for n in range(necho + 1):
            amp = ba * (eta * np.exp(1j * th))[:, None] ** n
            ph = np.exp(-1j * (1 + 2 * n) * g) if model == "D" else np.exp(-1j * g)
            F = F + amp * np.roll(S, 1 + n)[None, :] * ph
        out.append(np.abs(F) ** 2)
    return out[0] - out[1]


_TAPS = {}


def taps(kind, f3):
    key = (kind, f3)
    if key in _TAPS:
        return _TAPS[key]
    if kind == "ideal":
        res = ({0: 1.0}, None)
    else:
        n = N * OS
        H = m.filt_H(kind, f3, n)
        x = np.zeros(n)
        x[:OS] = 1.0
        y = np.fft.irfft(np.fft.rfft(x) * H, n)
        pc, tx = m.eye_center(m.rail_waves("D", "I", kind, f3)[1])
        res = ({j: float(y[(j * OS + pc) % n]) for j in range(-1, 7)}, pc)
    _TAPS[key] = res
    return res


def sample(L, g):
    return sum(gj * np.roll(L, j, axis=-1) for j, gj in g.items())


def eye_ratio(L, L0, g):
    """EO(theta)/EO0 for one rail; L (n_theta, N), L0 (1, N) echo-free."""
    Y, Y0 = sample(L, g), sample(L0, g)
    pos = L0[0] > 0
    eo = Y[:, pos].min(axis=1) - Y[:, ~pos].max(axis=1)
    eo0 = Y0[0, pos].min() - Y0[0, ~pos].max()
    return eo / eo0


def run(model, eta, ba, th, necho, kinds):
    res = {}
    Ls = {r: level_wave(model, r, eta, ba, th, necho) for r in ("I", "Q")}
    L0s = {r: level_wave(model, r, 0.0, ba, th[:1], 0) for r in ("I", "Q")}
    for kind, f3, _ in kinds:
        g, pc = taps(kind, f3)
        ratio = np.min([eye_ratio(Ls[r], L0s[r], g) for r in ("I", "Q")], axis=0)
        res[(kind, f3)] = ratio
    return res


def stat(ratio):
    p = m.pen(ratio)
    return {"wc": float(m.pen(ratio.min())), "wc_el": float(m.pen(ratio.min(), 20.0)), "mean": m.finite_mean(p),
            "best": float(m.pen(ratio.max())), "eo_wc": float(ratio.min()), "p95": m.p95(p),
            "theta_wc_deg": None}


th = np.deg2rad(np.arange(0.0, 360.0, 0.5))
levels = m.DESIGN + m.CONTEXT + m.NEEDLE

# ---- self-test: one echo (first order), model D, ba = 1, gate -> compare with the basis-wave code of echo_eye_model.py
k = m.eta_of(-20.5)
r1 = run("D", k, 1.0, th, 1, FILT)
t0, fix, opt, info = m.summarize_filtered("D", "bessel4", 0.75, k)
d = abs(float(m.pen(r1[("bessel4", 0.75)].min())) - float(m.pen(fix.min())))
d2 = abs(m.finite_mean(m.pen(r1[("bessel4", 0.75)])) - m.finite_mean(m.pen(fix)))
P("self-test, one echo, Bessel 0.75, gate: series code worst %.5f mean %.5f dB; basis-wave code worst %.5f mean %.5f dB; |diff| %.2e / %.2e dB"
  % (m.pen(r1[("bessel4", 0.75)].min()), m.finite_mean(m.pen(r1[("bessel4", 0.75)])), m.pen(fix.min()), m.finite_mean(m.pen(fix)), d, d2))
assert d < 2e-3 and d2 < 2e-3
ideal1 = m.analytic_eye(k, th)
P("self-test, one echo, ideal sampling vs closed form: max |diff| = %.2e" % np.abs(r1[("ideal", None)] - ideal1).max())
assert np.abs(r1[("ideal", None)] - ideal1).max() < 1e-10   # rounding of |F|^2 differences
z = run("D", 0.0, 1.0, th, NECHO, FILT)
P("needle eps = 0, all echoes: EO ratio min/max over theta and filters: %s" %
  ", ".join("%s %.15f/%.15f" % (kk[0], v.min(), v.max()) for kk, v in z.items()))
assert all(abs(v.min() - 1) < 1e-12 and abs(v.max() - 1) < 1e-12 for v in z.values())

# ---- main table: all echoes, b/a = 1
table = []
for lab, e_db, src in levels:
    k = m.eta_of(e_db)
    row = {"label": lab, "eps_dB": float(e_db), "eta": k}
    for model in ("D", "G"):
        res = run(model, k, 1.0, th, NECHO, FILT)
        res1 = run(model, k, 1.0, th, 1, (("ideal", None, 0.0),))
        for (kind, f3), ratio in res.items():
            tag = "ideal" if kind == "ideal" else ("%s%d" % ("B" if kind == "bessel4" else "G", round(f3 * 100)))
            s = stat(ratio)
            for key, v in s.items():
                if key != "theta_wc_deg":
                    row["%s_%s_%s" % (model, tag, key)] = v
            row["%s_%s_theta_wc_deg" % (model, tag)] = float(np.rad2deg(th[int(np.argmin(ratio))]))
        s1 = stat(res1[("ideal", None)])
        row["%s_ideal_first_order_wc" % model] = s1["wc"]
        row["%s_ideal_first_order_mean" % model] = s1["mean"]
    table.append(row)
    P("%-34s %7.2f dB | D all echoes: ideal wc %.4f mean %.4f best %.4f | B75 wc %.4f mean %.4f | B70 wc %.4f | G75 wc %.4f || G ideal wc %.4f mean %.4f | first order wc %.4f" %
      (lab, e_db, row["D_ideal_wc"], row["D_ideal_mean"], row["D_ideal_best"], row["D_B75_wc"], row["D_B75_mean"],
       row["D_B70_wc"], row["D_G75_wc"], row["G_ideal_wc"], row["G_ideal_mean"], row["D_ideal_first_order_wc"]))

# ---- device b/a
ba_rows = []
for name, ba in (("b/a = 1", 1.0), ("b/a optimum", m.BA_OPT), ("b/a center termination", m.BA_CEN)):
    for lab, e_db, src in m.DESIGN:
        k = m.eta_of(e_db)
        res = run("D", k, ba, th, NECHO, FILT[:2])
        s_i, s_b = stat(res[("ideal", None)]), stat(res[("bessel4", 0.75)])
        ba_rows.append({"case": name, "ba": float(ba), "label": lab, "eps_dB": e_db, "ideal_wc": s_i["wc"], "ideal_mean": s_i["mean"],
                        "B75_wc": s_b["wc"], "B75_mean": s_b["mean"], "ideal_best": s_i["best"]})
        P("%-24s b/a %.4f %-34s ideal wc %.4f mean %.4f best %.4f | B75 wc %.4f mean %.4f" %
          (name, ba, lab, s_i["wc"], s_i["mean"], s_i["best"], s_b["wc"], s_b["mean"]))

# ---- inverse: echo level for given worst-case penalty (all echoes, ideal sampling, model D, b/a = 1); bisection
inv = []
for pdb in (0.25, 0.5, 1.0, 2.0):
    lo, hi = -50.0, -3.5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        wc = float(m.pen(run("D", m.eta_of(mid), 1.0, th[::4], NECHO, FILT[:1])[("ideal", None)].min()))
        if wc > pdb:
            hi = mid
        else:
            lo = mid
    inv.append({"pen_dB": pdb, "eps_dB": 0.5 * (lo + hi)})
    P("worst-case penalty %.2f dB <-> echo %.2f dB (all echoes, ideal sampling, D, b/a = 1); first order gave %.2f dB" %
      (pdb, 0.5 * (lo + hi), 20 * np.log10((1 - 10 ** (-pdb / 10)) / np.sqrt(2)) - 10 * np.log10(RHO2)))

# ---- curves for the figure
grid = np.arange(-32.0, -9.99, 0.5)
th1 = np.deg2rad(np.arange(0, 360, 1.0))
curves = {"eps_dB": grid.tolist(), "ideal_wc": [], "ideal_mean": [], "ideal_best": [], "B75_wc": [], "B75_mean": [], "first_wc": []}
for e_db in grid:
    k = m.eta_of(e_db)
    res = run("D", k, 1.0, th1, NECHO, FILT[:2])
    for key, kk in (("ideal", ("ideal", None)), ("B75", ("bessel4", 0.75))):
        r = res[kk]
        curves["%s_wc" % key].append(float(m.pen(r.min())))
        curves["%s_mean" % key].append(m.finite_mean(m.pen(r)))
        if key == "ideal":
            curves["ideal_best"].append(float(m.pen(r.max())))
    curves["first_wc"].append(float(m.pen(1 - np.sqrt(2) * k)))

# ---- eye traces: I rail, model D, all echoes, gate, worst theta of the Bessel 0.75 eye
k = m.eta_of(-20.5)
Ls = level_wave("D", "I", k, 1.0, th, NECHO)
L0 = level_wave("D", "I", 0.0, 1.0, th[:1], 0)
g, pc = taps("bessel4", 0.75)
ratio = eye_ratio(Ls, L0, g)
iw = int(np.argmin(ratio))
H = m.filt_H("bessel4", 0.75, N * OS)


def wave(Lrow):
    x = np.repeat(Lrow, OS)
    return np.fft.irfft(np.fft.rfft(x) * H, N * OS).reshape(N, OS)


sc = 1.0 / np.max(np.abs(L0))
np.savez_compressed(HERE + r"\eye_traces2.npz", y0=wave(L0[0]) * sc, yw=wave(Ls[iw]) * sc, pos=(L0[0] > 0), p0=pc,
                    theta=float(th[iw]), eo_ratio=float(ratio[iw]))
P("eye traces: worst theta %.1f deg, EO ratio %.4f (penalty %.4f dB), sampling phase %d" % (np.rad2deg(th[iw]), ratio[iw], m.pen(ratio[iw]), pc))

# ---- penalty versus echo phase at the gate (both rails, worst of the two), and the large-echo needles
resg = run("D", m.eta_of(-20.5), 1.0, th, NECHO, FILT[:2])
gate_theta = {"theta_deg": np.rad2deg(th).tolist(),
              "pen_ideal": [float(x) for x in m.pen(resg[("ideal", None)])],
              "pen_B75": [float(x) for x in m.pen(resg[("bessel4", 0.75)])],
              "pen_first_order_ideal": [float(x) for x in m.pen(m.analytic_eye(m.eta_of(-20.5), th))]}
needles = {}
for e_db in (-6.0, -10 * np.log10(2.0), -9.0, -12.0):
    kk = m.eta_of(e_db)
    needles[str(round(e_db, 4))] = {
        "first_order_eo_wc": float(1 - np.sqrt(2) * kk),
        "all_echoes_eo_wc_ideal": float(run("D", kk, 1.0, th, NECHO, FILT[:1])[("ideal", None)].min()),
        "all_echoes_eo_wc_ideal_G": float(run("G", kk, 1.0, th, NECHO, FILT[:1])[("ideal", None)].min())}
    P("needle %.4f dB: first-order worst-case EO %.4f; all echoes (D) %.4f, (G) %.4f" %
      (e_db, needles[str(round(e_db, 4))]["first_order_eo_wc"], needles[str(round(e_db, 4))]["all_echoes_eo_wc_ideal"],
       needles[str(round(e_db, 4))]["all_echoes_eo_wc_ideal_G"]))

json.dump({"table": table, "ba_rows": ba_rows, "inverse": inv, "curves": curves, "NECHO": NECHO,
           "gate_theta": gate_theta, "needles": needles},
          open(HERE + r"\echo_eye_series_numbers.json", "w"), indent=1)
open(HERE + r"\echo_eye_series_log.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
P("done")
