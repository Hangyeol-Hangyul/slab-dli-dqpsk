# -*- coding: utf-8 -*-
"""Self-tests of the filtered-eye code (needle checks):
 (1) wide-bandwidth limit: a 4th-order Bessel filter with a 3 dB bandwidth of 6 x baud must
     reproduce the ideal-sampling closed form for every theta (both models, both rails);
 (2) a second, independent computation of the filtered eye: sampled responses g_j of the Bessel
     filter to one rectangular symbol are taken from the step response (scipy.signal.step), and the
     eye is enumerated over all 4^6 symbol windows without any FFT or periodic sequence;
     it must agree with the FFT / de Bruijn result to the tail truncation of the window."""
import itertools
import importlib.util
import sys
import numpy as np
from scipy import signal as sg

HERE = r"<WORK_ROOT>\paper_scdc_261011_v3\review\r7_memo_261013\echoeye"
spec = importlib.util.spec_from_file_location("eem", HERE + r"\echo_eye_model.py")
m = importlib.util.module_from_spec(spec)
sys.modules["eem"] = m
spec.loader.exec_module(m)

out = []


def P(s):
    print(s, flush=True)
    out.append(s)


# (1) wide-filter limit
kap = m.eta_of(-20.5)
for model in ("D", "G"):
    thetas, fix, opt, info = m.summarize_filtered(model, "bessel4", 6.0, kap, thetas=np.deg2rad(np.arange(0, 360, 1.0)))
    ref = m.analytic_eye(kap, thetas)
    P("wide Bessel (6 x baud), model %s: max |filtered - closed form| over 360 theta = %.3e (center phase %d, echo-free EO/sqrt2 = %.6f)"
      % (model, np.abs(fix - ref).max(), info["I"]["pc"], info["I"]["eo0_fixed"] / np.sqrt(2)))

# (2) independent enumeration with the step response of the filter
OS = m.OS
f3 = 0.75
b, a = sg.bessel(4, 2 * np.pi * f3, "low", analog=True, norm="mag")
t = np.arange(0, 12.0, 1.0 / OS)
_, s = sg.step((b, a), T=t)                      # step response, unit-time = one symbol
pc, tx = m.eye_center(m.rail_waves("D", "I", "bessel4", 0.75)[1])
# sample at time tau = pc/OS after the start of slot k.  A rectangular symbol that began j slots
# earlier (j = 0 is the current slot) contributes  s(tau + j) - s(tau + j - 1)  (the second term only for j >= 1).
tau = pc / OS


def g(j):
    def sv(x):
        return 0.0 if x < 0 else float(np.interp(x, t, s))
    return sv(tau + j) - (sv(tau + j - 1) if j >= 1 else 0.0)


J = 5
gj = np.array([g(j) for j in range(J + 1)])      # weights of slots k, k-1, ..., k-5 (k+1 never: causal filter, tau < 1)
P("sampled symbol responses g_j (j = 0..5) at tau = %.4f symbol: %s" % (tau, np.array2string(gj, precision=5)))
for model in ("D",):
    for rail in ("I", "Q"):
        beta = m.BETA[rail]
        # enumerate all 4^6 windows (D_{k-5} ... D_k); symbol k-j has weight g_j, echo basis of slot k-j uses D_{k-j-1}
        wins = np.array(list(itertools.product(range(4), repeat=J + 2)))      # D_{k-6} ... D_k
        Dwin = m.D_SET[wins]
        main = np.cos(Dwin - beta)                                              # per slot (k-6 ... k)
        # slot index i = 0..6 maps to k-6+i ; weight for slot k-j is gj[j]
        W = np.zeros(J + 2)
        for j in range(J + 1):
            W[J + 1 - j] = gj[j]                                                # slot index of k-j is J+1-j
        y_main = main @ W
        sign = main[:, -1] > 0
        for kk in (0.0, m.eta_of(-20.5)):
            ths = np.deg2rad(np.arange(0, 360, 5.0))
            eos = []
            for th in ths:
                # echo level of slot (k-j) = cos(D_{k-j-1} - th - beta) (model D) -> uses window element one earlier
                e = np.zeros(len(wins))
                for j in range(J + 1):
                    idx = J + 1 - j
                    e += gj[j] * np.cos(Dwin[:, idx - 1] - th - beta)
                S = y_main + kk * e
                eos.append(S[sign].min() - S[~sign].max())
            eos = np.array(eos)
            if kk == 0.0:
                eo0 = eos[0]
                P("independent window enumeration, model D rail %s: echo-free EO / sqrt2 = %.6f" % (rail, eo0 / np.sqrt(2)))
            else:
                th_ref, fix, opt, info = m.summarize_filtered("D", "bessel4", 0.75, kk, thetas=ths)
                P("independent window enumeration, model D rail %s, gate: worst-case penalty %.4f dB (FFT code, both rails: %.4f dB); "
                  "max |EO ratio(window) - EO ratio(FFT)| over 72 theta = %.3e" %
                  (rail, float(m.pen((eos / eo0).min())), float(m.pen(fix.min())), np.abs(eos / eo0 - fix).max()))
open(HERE + r"\check_filter_limit_log.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
