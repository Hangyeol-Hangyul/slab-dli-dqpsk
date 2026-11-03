# -*- coding: utf-8 -*-
"""Multiple-echo check (ideal sampling, field level, no closed form used).
The echo returns to the same point and is reflected again into R(+2) with the same amplitude eta:
C_n = b (eta e^{i theta})^n s_{k-1-n} e^{-i(1+2n)Gd}   (device, model D; arm B is n = 0)
or, for the generic DLI, C_n with a common bias phase.  Balanced differences are formed from the four
quadrant powers directly.  Compares the series truncated after 1 echo (first order in eta, = closed form)
with 2, 3 and 4 echoes."""
import itertools
import numpy as np

HERE = r"<WORK_ROOT>\paper_scdc_261011_v3\review\r7_memo_261013\echoeye"
RHO2 = 0.9999530484
LEVELS = [-20.5, -20.66, -20.82, -20.95, -21.22, -15.0, -25.0]
Dset = np.deg2rad([0.0, 90.0, 180.0, 270.0])
DEPTH = 6                                  # symbols s_k ... s_{k-5}
pat = np.array(list(itertools.product(range(4), repeat=DEPTH - 1)))   # Delta'_k ... Delta'_{k-4}
dphi = Dset[pat]                                                   # (n_pat, 5)
psi = np.zeros((len(pat), DEPTH))
psi[:, 1:] = -np.cumsum(dphi, axis=1)                              # psi_k = 0, psi_{k-j} = -sum of the j latest Deltas
S = np.exp(1j * psi)                                               # (n_pat, 6): s_k, s_{k-1}, ..., s_{k-5}

Gd = {"I+": np.deg2rad(22.5), "I-": np.deg2rad(-67.5), "Q+": np.deg2rad(-22.5), "Q-": np.deg2rad(67.5)}
th_grid = np.deg2rad(np.arange(0, 360, 0.5))


def field_power(model, q, eta, th, necho):
    """power at the PD of quadrant q for all patterns and all theta: shape (n_theta, n_pat)."""
    g = Gd[q]
    a = b = 1.0
    A = a * S[:, 0] * np.exp(1j * g)
    F = np.broadcast_to(A, (len(th), len(pat))).copy()
    for n in range(necho + 1):               # n = 0 is arm B
        amp = b * (eta * np.exp(1j * th))[:, None] ** n
        if model == "D":
            ph = np.exp(-1j * (1 + 2 * n) * g)
        else:                                 # generic DLI: common bias phase on the delayed arm, beta = -2Gd
            ph = np.exp(-1j * g)               # = e^{i beta} B up to the global phase e^{iGd} of the quadrant
        F = F + amp * S[:, 1 + n][None, :] * ph
    return np.abs(F) ** 2


def eye(model, eta, necho):
    out = {}
    for rail, (qp, qn) in (("I", ("I+", "I-")), ("Q", ("Q+", "Q-"))):
        diff = field_power(model, qp, eta, th_grid, necho) - field_power(model, qn, eta, th_grid, necho)
        ref = field_power(model, qp, 0.0, th_grid[:1], 0) - field_power(model, qn, 0.0, th_grid[:1], 0)
        pos = ref[0] > 0
        eo = diff[:, pos].min(axis=1) - diff[:, ~pos].max(axis=1)
        eo0 = ref[0][pos].min() - ref[0][~pos].max()
        out[rail] = eo / eo0
    return np.minimum(out["I"], out["Q"])


lines = []
for model in ("D", "G"):
    lines.append("model %s (device translation phases)" % model if model == "D" else "model G (generic DLI)")
    for e_db in LEVELS:
        eta = np.sqrt(10 ** (e_db / 10) * RHO2)
        res = []
        for n in (1, 2, 3, 4):
            eo = eye(model, eta, n)
            res.append((eo.min(), -10 * np.log10(eo.min()), -10 * np.log10(eo).mean()))
        cf = 1 - np.sqrt(2) * eta
        lines.append("  echo %6.2f dB eta %.4f closed first order EO %.4f pen %.4f | " % (e_db, eta, cf, -10 * np.log10(cf)) +
                     " | ".join("%d echoes: EO %.4f pen_wc %.4f mean %.4f" % (n, r[0], r[1], r[2]) for n, r in zip((1, 2, 3, 4), res)))
for ln in lines:
    print(ln)
open(HERE + r"\higher_order_echo_log.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
