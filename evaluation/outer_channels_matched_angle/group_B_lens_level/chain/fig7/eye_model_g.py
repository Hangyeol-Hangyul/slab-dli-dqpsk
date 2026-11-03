# -*- coding: utf-8 -*-
"""
Branch E (261015): receiver eye model for Figure 7 -- eye-opening penalty relative to an IDEAL receiver
with the echo, the arm amplitude ratio, the DLI phase spread over the eight atoms and the common DLI phase
per wavelength channel.

Base: review\\r7_memo_261013\\echoeye\\ (echo_eye_model.py, echo_eye_series.py; copies in base_copy\\).
The base scripts are NOT imported (importing echo_eye_model.py would run its module-level code);
the building blocks used here (de Bruijn sequence, 4th-order Bessel FFT filter, eye-center rule, the
complete echo series of model D) are re-implemented with the same definitions, and the first self-tests
compare with the numbers STORED by the base (echo_eye_series_numbers.json).

MODEL D (this device; field level, derived in base_copy\\result.md Step 2 and checked there):
  symbols s_k = exp(i psi_k), differential phase D_k = arg(s_k s_{k-1}^*) in {0, 90, 180, 270} deg.
  Photodiode (PD) q with translation phase G*d_q ("Gd"):
      F_q[k] = a s_k e^{+i Gd} + W e^{i(phi + eps_q)} sum_{n>=0} b (nu e^{i chi})^n s_{k-1-n} e^{-i(1+2n)Gd}
  n = 0 is arm B (amplitude b, delayed by one symbol), n >= 1 the echo and the echo of the echo ... (each return
  multiplies by nu e^{i chi}, nu = sqrt(eps * RHO2)); arm A has amplitude a (reference arm).
  phi   : phase error of every arm-B-derived term against arm A (common DLI phase and/or spread of the
          arm phase difference Delta psi = phi_B - phi_A over the atoms); eps_q: extra phase on one PD only;
          W    : complex amplitude factor on the B-derived terms (coherent sum over the atoms).
  Rail I = P(Gd=+22.5) - P(Gd=-67.5)   (balanced pair at bias -45 / +135 deg)
  Rail Q = P(Gd=-22.5) - P(Gd=+67.5)   (balanced pair at bias +45 / -135 deg)
  The beat of the echo with arm A has the same phase in both PDs of a pair and cancels; the beat with arm B stays.

REFERENCE ("ideal receiver"): same total arm power a^2 + b^2 split equally (a_i = b_i), no echo, no phase error,
  same filter, same sampling instant (the echo-free eye center of the ideal receiver).  EO = min('1' levels)
  - max('0' levels) at that instant ('1' / '0' = sign of the ideal level); penalty = -10 log10(EO / EO_ideal)
  = optical dB (the electrical form is exactly twice).
  For the self-tests ref = "same_ba" reproduces the base convention (EO0 = echo-free eye of the same arms).

FIX ROUND 261015 (fix_261015\\E_eye_fig7\\; the pre-fix model is in pre_fix\\):
  (1) HEADLINE treatment of the arm-phase spread: the eight atoms equally spaced over their peak-to-peak range, coherent sum, NO net bias ("uniform").  The
      source says the spread "is absorbed into the quadrant shifts" (si_v2_r12.md line 322) and c2_s23.json total_common = geo + retro + atoms_mean already contains the
      atom mean, so adding half of the range as an extra bias error (the pre-fix headline "half") double-books the atom phase.  "half" and "full" are kept only as
      labelled BOUNDS.  New case "atoms": the real eight-atom dpsi of c2_tables.json at the channel, usage weights H6 x prod_amp, set-point = usage-weighted mean dpsi
      (the quantity inside total_common); its net bias residual arg(W) and visibility |W| are reported.
  (2) The peak-to-peak spread is read per channel from c2_s23.json phase_rows ptp_abs_dpsi (5.264 / 5.128 / 4.864 deg), not 5.13 at every channel.
  (3) Table 4 (echo estimate per channel) is rebuilt from the measured C1 x + y share echo at the channel plus the 45 deg increment measured at 1550 nm (C2 full_enum).
Run:  python -I -B eye_fig7_model.py   (pins itself to logical cores 16-19 AFTER the imports; 4 BLAS threads)
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import sys
import time
import numpy as np
from scipy import signal as sg
import psutil


ORIG = r"<WORK_ROOT>\overnight_261015\E_eye_fig7"                      # pre-fix folder, read only
OUT = r"<WORK_ROOT>\overnight_261015\fix_261015\E_eye_fig7"           # this fix folder
BASE_JSON = ORIG + r"\base_copy\echo_eye_series_numbers.json"
C2_JSON = r"<WORK_ROOT>\fill_B_261103\work\chain\c2\c2_s23.json"
C2_TABLES = r"<WORK_ROOT>\fill_B_261103\work\chain\c2\c2_tables.json"
C1_JSON = r"<WORK_ROOT>\overnight_261015\C_lens_analyses\C1_band_xy\lensband_eval_xy_result.json"
C2ZONE_JSON = r"<WORK_ROOT>\overnight_261015\C_lens_analyses\C2_zone_gamma\lens_zone_gamma_result_flips.json"

# ------------------------------------------------------------------ constants (each with its source)
RHO2 = 0.9999530484               # retroreflected power fraction R(-2): SI Table S23 / si_v2_r12.md line 213, base echo_eye_model.py:39
EPS_DB = -20.66                   # echo, metalens aperture x + y + 45 deg ramps: SI Table S23 (task)
PA, PB = 0.2808, 0.2410           # arm powers P_A, P_B at the center termination: SI Table S23 / Note S8 (task)
BA = float(np.sqrt(PB / PA))      # b/a = (P_B/P_A)^(1/2) = 0.92642
_c2 = json.load(open(C2_JSON, encoding="utf-8"))
PHI_CH = {r["ch"]: float(r["total_common"]) for r in _c2["phase_rows"]}      # deg, relative to 1550 nm
PTP_CH = {r["ch"]: float(r["ptp_abs_dpsi"]) for r in _c2["phase_rows"]}      # deg, peak-to-peak of the atom dpsi at each channel (c2_s23.json phase_rows)
PTP_DEG = PTP_CH["1550.0"]        # 5.1279 deg at 1550 nm (SI Note S9: 5.13 deg); used where a single value is needed (self-tests T3, T4)
CHANNELS = [("1550.000", PHI_CH["1550.0"]), ("1546.003", PHI_CH["1546.003"]), ("1554.017", PHI_CH["1554.017"])]
CH_KEY = {"1550.000": "1550.0", "1546.003": "1546.003", "1554.017": "1554.017"}   # label -> key in c2_s23.json / c2_tables.json
PTP_LAB = {lab: PTP_CH[k] for lab, k in CH_KEY.items()}
ECHO_AP_X = {"1546.003": _c2["cpd"]["1546.003"]["echo_ap"], "1550.000": _c2["cpd"]["1550.0"]["echo_ap"],
             "1554.017": _c2["cpd"]["1554.017"]["echo_ap"]}                  # x-ramp aperture echo per channel (dB), c2_s23.json cpd
ECHO_XY45_OFFSET = EPS_DB - ECHO_AP_X["1550.000"]                           # x+y+45 minus x at 1550 nm (dB): SUPERSEDED estimate rule of the pre-fix Table 4 (kept for the comparison column)
_c1 = json.load(open(C1_JSON, encoding="utf-8"))
ECHO_XY_C1 = {lab: float(_c1["out"][("1550" if lab == "1550.000" else lab)]["x_y"]["echo_share"]) for lab in CH_KEY}   # C1: x + y share echo per channel (45 deg cells NOT included)
ECHO_XY45_1550 = float(json.load(open(C2ZONE_JSON, encoding="utf-8"))["full_enum"]["echo_share"])                        # C2 full enumeration at 1550: x + y + 45 deg, -20.66037 dB
INC45 = ECHO_XY45_1550 - ECHO_XY_C1["1550.000"]                             # +0.1043 dB: echo increment of the 45 deg ramps at 1550 nm (SI line 451: 0.104 dB)

# ---- the real eight atoms at each channel (c2_tables.json): usage weights H6, dpsi, prod_amp
_T = json.load(open(C2_TABLES, encoding="utf-8"))


def atoms_spread(label):
    """lens-weighted coherent sum of the eight atoms at a channel, set-point = usage-weighted ARITHMETIC mean of dpsi (the quantity inside c2_s23.json total_common).
    Returns dict: W (complex factor on the arm-B-derived terms, weights = usage share x prod_amp = |t_A||r_B t_B|), mu_deg, plus the variants for the residual table."""
    k = CH_KEY[label]
    lv = list(_T["atoms"].keys())
    dp = np.array([_T["atoms"][l][k]["dpsi"] for l in lv], float)
    pa = np.array([_T["atoms"][l][k]["prod_amp"] for l in lv], float)
    w = np.array([_T["weights"][l] for l in lv], float)
    mu = float(np.sum(w * dp))
    out = {"mu_deg": mu, "ptp_deg": float(dp.max() - dp.min()), "mid_minus_mu_deg": float(0.5 * (dp.max() + dp.min()) - mu)}
    for tag, vv in (("w_prod", w * pa), ("w", w), ("equal", np.ones_like(w))):
        v = vv / vv.sum()
        Wc = complex(np.sum(v * np.exp(1j * np.deg2rad(dp - mu))))
        out[tag] = {"W": [Wc.real, Wc.imag], "abs": abs(Wc), "arg_deg": float(np.rad2deg(np.angle(Wc)))}
    out["W"] = complex(*out["w_prod"]["W"])
    out["rms_deg_w"] = float(np.sqrt(np.sum(w * (dp - mu) ** 2)))
    return out


ATOMS = {lab: atoms_spread(lab) for lab in CH_KEY}

OS = 32                           # samples per symbol (as base)
F3 = 0.75                         # Bessel -3 dB frequency in units of the baud rate
NECHO = 6                         # returns kept (as base)
GD = {"I": (np.deg2rad(22.5), np.deg2rad(-67.5)), "Q": (np.deg2rad(-22.5), np.deg2rad(67.5))}
CHIS = np.deg2rad(np.arange(0.0, 360.0, 0.5))                                # echo phase grid, 0.5 deg as base
D_SET = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]))


def de_bruijn(k, n):
    """periodic de Bruijn sequence B(k, n) (as base echo_eye_model.py)."""
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


SYM = de_bruijn(4, 6)
assert len(SYM) == 4 ** 6
DK = D_SET[SYM]
N = len(DK)
PSI = np.cumsum(DK)
assert abs(np.exp(1j * PSI[-1]) - 1) < 1e-9          # the periodic sequence closes
S = np.exp(1j * PSI)
ROLL = {n: np.roll(S, 1 + n) for n in range(0, 12)}


def nu_of(eps_db):
    """echo-to-arm-B amplitude ratio at the combiner: sqrt(eps * RHO2) (base derivation, result.md Step 1)."""
    return float(np.sqrt(10 ** (eps_db / 10.0) * RHO2))


# ------------------------------------------------------------------ the field-level rail output
def level_wave(rail, a, b, nu, chis, necho=NECHO, phi=0.0, W=1.0, eps_pd=(0.0, 0.0)):
    """balanced output of one rail for every symbol and echo phase: array (len(chis), N)."""
    chis = np.atleast_1d(np.asarray(chis, float))
    z = (nu * np.exp(1j * chis))[:, None]
    out = []
    for idx, g in enumerate(GD[rail]):
        F = np.empty((len(chis), N), complex)
        F[:] = (a * S * np.exp(1j * g))[None, :]
        pw = np.ones_like(z)
        bfac = W * np.exp(1j * (phi + eps_pd[idx]))
        for n in range(necho + 1):
            F += (b * bfac) * pw * ROLL[n][None, :] * np.exp(-1j * (1 + 2 * n) * g)
            pw = pw * z
        out.append(np.abs(F) ** 2)
    return out[0] - out[1]


# ------------------------------------------------------------------ filter and sampling
def bessel_H(f3, n):
    f = np.fft.rfftfreq(n, d=1.0 / OS)
    b, a = sg.bessel(4, 2 * np.pi * f3, "low", analog=True, norm="mag")      # -3 dB at f3 (as base)
    _, H = sg.freqs(b, a, worN=2 * np.pi * f)
    return H


HB = bessel_H(F3, N * OS)


def filt_wave(L):
    """filtered NRZ waveform of the symbol-level array L (..., N): shape (..., N, OS)."""
    x = np.repeat(L, OS, axis=-1)
    y = np.fft.irfft(np.fft.rfft(x, axis=-1) * HB, N * OS, axis=-1)
    return y.reshape(L.shape[:-1] + (N, OS))


def eye_center(y0):
    """sampling phase [samples] at the CENTER of the echo-free eye: mean zero-crossing time of the
    filtered main wave (circular mean over the slot) plus half a symbol (as base echo_eye_model.py)."""
    y = y0.reshape(-1)
    idx = np.where(y[:-1] * y[1:] < 0)[0]
    frac = y[idx] / (y[idx] - y[idx + 1])
    ang = 2 * np.pi * ((idx + frac) % OS) / OS
    t_x = (np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi) * OS) % OS
    return int(round((t_x + OS / 2.0) % OS)) % OS, float(t_x)


def ideal_levels(rail, a, b):
    return level_wave(rail, a, b, 0.0, np.array([0.0]), necho=0)[0]


# the sampling instant: echo-free eye center of the ideal receiver (equal arms), rail I; Q must agree
_L0 = {r: ideal_levels(r, 1.0, 1.0) for r in ("I", "Q")}
PC = {}
for _r in ("I", "Q"):
    PC[_r], _tx = eye_center(filt_wave(_L0[_r]) [:, :])
assert PC["I"] == PC["Q"], PC
PCS = PC["I"]
_x = np.zeros(N * OS)
_x[:OS] = 1.0
_pulse = np.fft.irfft(np.fft.rfft(_x) * HB, N * OS)
G_TAPS = np.array([_pulse[(j * OS + PCS) % (N * OS)] for j in range(N)])      # g_j = filtered pulse sampled j symbols later
GFFT = np.fft.rfft(G_TAPS)
POS = {r: _L0[r] > 0 for r in ("I", "Q")}                                    # '1' levels: sign of the ideal level


def sample(L, kind):
    """decision variable per symbol: 'ideal' (no filter, ideal sampling) or 'bessel' (4th order, 0.75 x baud,
    sampled at the echo-free eye center of the ideal receiver, exact circular convolution)."""
    if kind == "ideal":
        return L
    return np.fft.irfft(np.fft.rfft(L, axis=-1) * GFFT, N, axis=-1)


def eye_open(Y, rail):
    pos = POS[rail]
    return Y[..., pos].min(axis=-1) - Y[..., ~pos].max(axis=-1)


def eo_reference(rail, a, b, kind, ref):
    if ref == "ideal":                       # equal arms, same total power a^2 + b^2
        ai = float(np.sqrt((a * a + b * b) / 2.0))
        L = ideal_levels(rail, ai, ai)
    elif ref == "same_ba":                   # base convention: the echo-free eye of the same arms
        L = ideal_levels(rail, a, b)
    else:
        raise ValueError(ref)
    return float(eye_open(sample(L[None, :], kind)[0], rail))


def eo_ratio(rail, kind, a=1.0, b=1.0, nu=0.0, chis=CHIS, necho=NECHO, phi=0.0, W=1.0, eps_pd=(0.0, 0.0),
             ref="ideal", chunk=240):
    """EO(chi) / EO_ref for one rail; array (len(chis),)."""
    chis = np.atleast_1d(np.asarray(chis, float))
    ne = necho if nu > 0 else 0
    ref0 = eo_reference(rail, a, b, kind, ref)
    if nu == 0:                              # no echo: nothing depends on chi
        L = level_wave(rail, a, b, 0.0, chis[:1], necho=0, phi=phi, W=W, eps_pd=eps_pd)
        return np.full(len(chis), float(eye_open(sample(L, kind), rail)[0] / ref0))
    res = np.empty(len(chis))
    for i0 in range(0, len(chis), chunk):
        L = level_wave(rail, a, b, nu, chis[i0:i0 + chunk], necho=ne, phi=phi, W=W, eps_pd=eps_pd)
        res[i0:i0 + chunk] = eye_open(sample(L, kind), rail) / ref0
    return res


def pen(r, kind=10.0):
    r = np.asarray(r, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = -kind * np.log10(r)
    return np.where(r > 0, o, np.inf)


def stats(R, chis=CHIS):
    """R: EO/EO_ref over the echo phase. 'worst' = at the worst phase (= max of the penalty)."""
    p = pen(R)
    i = int(np.argmin(R))
    return {"eo_wc": float(R.min()), "pen_wc": float(pen(R.min())), "chi_wc_deg": float(np.rad2deg(chis[i])),
            "pen_min": float(p.min()), "pen_mean": float(np.mean(p)) if np.all(np.isfinite(p)) else float("inf"),
            "pen_max": float(p.max()), "eo_best": float(R.max())}


# ------------------------------------------------------------------ closed forms (derived in the self-test section of the doc)
def cf_bias(delta_deg):
    """pure rail bias error delta (both PDs of the pair shifted), equal arms, ideal sampling:
    EO/EO0 = cos(delta) - sin|delta|  = sqrt(2) cos(45 deg + |delta|)."""
    d = np.deg2rad(np.abs(delta_deg))
    return np.cos(d) - np.sin(d)


def cf_single_pd(eps_deg):
    """phase error eps on ONE PD of a pair: effective rail bias eps/2 and visibility cos(eps/2)."""
    h = np.deg2rad(eps_deg) / 2.0
    return np.cos(h) * (np.cos(h) - np.sin(np.abs(h)))


def cf_arm(ba):
    """arm imbalance at fixed total power: 2 (b/a) / (1 + (b/a)^2)."""
    return 2 * ba / (1 + ba * ba)


def cf_first_order(kappa, delta_deg, chi, ba=1.0):
    """one echo only (necho = 1), rail bias error delta, arm ratio ba, ideal sampling, reference = ideal receiver of the same total power;
    kappa = nu * ba. EO/EO_ideal = V_arm [ (cos d - sin|d|) - sqrt(2) kappa max(|cos u|, |sin u|) ], u = chi + 45 deg (rail I)."""
    u = chi + np.deg2rad(45.0)
    return cf_arm(ba) * (cf_bias(delta_deg) - np.sqrt(2.0) * kappa * np.maximum(np.abs(np.cos(u)), np.abs(np.sin(u))))


def spread_W(ptp_deg, n=8):
    """uniform spread of the eight atoms: equally spaced Delta psi over the peak-to-peak range, equal weights;
    returns the complex amplitude factor of the coherent sum, mean exp(i delta_j)."""
    dj = np.deg2rad(np.linspace(-ptp_deg / 2.0, ptp_deg / 2.0, n))
    return complex(np.mean(np.exp(1j * dj)))


# ------------------------------------------------------------------ the cases
def case_params(phi_ch_deg, treat, echo=True, ba=BA, eps_db=EPS_DB, spread=True, ptp_deg=PTP_DEG, atoms_w=None):
    """parameter dict of one device case.  treat (261015 fix round):
    'uniform' = HEADLINE: eight atoms equally spaced over ptp (per channel), coherent sum, NO net bias, visibility only (source: the spread 'is absorbed into the
                quadrant shifts', si_v2_r12.md line 322; the atom mean is already in total_common);
    'atoms'   = the real eight atoms at the channel, usage x prod_amp weights, set-point = usage-weighted mean dpsi (atoms_w = complex W from atoms_spread());
    'half'    = BOUND: ptp/2 added to the common phase as an extra bias error (worst sign);  'full' = BOUND: ptp added;  None = no spread."""
    p = {"a": 1.0, "b": ba, "nu": nu_of(eps_db) if echo else 0.0, "W": 1.0, "phi": np.deg2rad(phi_ch_deg)}
    sgn = 1.0 if phi_ch_deg >= 0 else -1.0
    if spread and treat == "half":
        p["phi"] = np.deg2rad(phi_ch_deg + sgn * ptp_deg / 2.0)
    elif spread and treat == "full":
        p["phi"] = np.deg2rad(phi_ch_deg + sgn * ptp_deg)
    elif spread and treat == "uniform":
        p["W"] = spread_W(ptp_deg)
    elif spread and treat == "atoms":
        p["W"] = complex(atoms_w)
    return p


def rails_stats(p, kind, ref="ideal", chis=CHIS):
    R = {r: eo_ratio(r, kind, chis=chis, ref=ref, **p) for r in ("I", "Q")}
    st = {r: stats(R[r], chis) for r in R}
    st["joint"] = stats(np.minimum(R["I"], R["Q"]), chis)
    return st, R


def main():
    t0 = time.time()
    log = []
    open(OUT + r"\progress.md", "w", encoding="utf-8").write(
        "STATUS: IN PROGRESS\n\nbranch E eye model, started %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    def progress(line):
        open(OUT + r"\progress.md", "a", encoding="utf-8").write(line + "\n")

    P("branch E eye model: cores", psutil.Process().cpu_affinity(), "available RAM %.1f GB" % (psutil.virtual_memory().available / 1e9))
    P("constants: eps %.2f dB, nu %.6f, b/a %.6f (P_A %.4f P_B %.4f), ptp per channel %s deg, phi_ch %s" %
      (EPS_DB, nu_of(EPS_DB), BA, PA, PB, {k: round(v, 4) for k, v in PTP_LAB.items()}, {k: round(v, 4) for k, v in CHANNELS}))
    P("atoms (c2_tables.json, usage x prod_amp weights, set-point = usage-weighted mean dpsi): " + "; ".join(
        "%s: |W| %.8f arg %+.5f deg (w only: %+.5f; equal weights: %+.5f), mid-range minus mean %+.4f deg" % (
            lab, a["w_prod"]["abs"], a["w_prod"]["arg_deg"], a["w"]["arg_deg"], a["equal"]["arg_deg"], a["mid_minus_mu_deg"]) for lab, a in ATOMS.items()))
    P("echo estimate inputs: C1 x + y share echo %s dB; 45 deg increment at 1550 nm = %.5f dB (C2 full_enum %.5f minus C1 x + y %.5f)" % (
        {k: round(v, 3) for k, v in ECHO_XY_C1.items()}, INC45, ECHO_XY45_1550, ECHO_XY_C1["1550.000"]))
    P("sampling instant: phase %d of %d samples, g_0 = %.6f, sum g = %.6f" % (PCS, OS, G_TAPS[0], G_TAPS.sum()))
    base = json.load(open(BASE_JSON))
    brow = {r["label"]: r for r in base["table"]}
    bbarow = {(r["case"], r["label"]): r for r in base["ba_rows"]}
    ST = {}                               # self-test residuals

    # ---- T1 zero impairments -> penalty 0
    res = []
    for kind in ("ideal", "bessel"):
        for rail in ("I", "Q"):
            R = eo_ratio(rail, kind, a=1.0, b=1.0, nu=0.0, chis=CHIS[:8])
            res.append(float(np.abs(R - 1.0).max()))
    # any arms of equal power split but unequal a, b with ref = ideal: not a zero-impairment case (checked in T10)
    ST["T1_zero_impairments_max|EO/EO_ideal-1|"] = max(res)
    P("T1 zero impairments (b/a = 1, no echo, no phase error), ideal + Bessel, I + Q: max |EO/EO_ideal - 1| = %.3e, penalty %.3e dB" %
      (max(res), float(pen(1 - max(res)))))

    # ---- T2 echo only reproduces the stored table (b/a = 1, -20.66 dB, all returns, joint worst of I and Q)
    nu = nu_of(EPS_DB)
    row = brow["lens, share-weighted x, y, 45 deg"]
    for kind, tag in (("ideal", "ideal"), ("bessel", "B75")):
        RI = eo_ratio("I", kind, nu=nu, ref="same_ba")
        RQ = eo_ratio("Q", kind, nu=nu, ref="same_ba")
        sj = stats(np.minimum(RI, RQ))
        for key, bk in (("pen_wc", "wc"), ("pen_mean", "mean"), ("pen_min", "best")):
            sv = row["D_%s_%s" % (tag, bk)]
            ST["T2_echo_only_%s_%s" % (tag, key)] = {"mine": sj[key], "stored": sv, "resid_dB": sj[key] - sv}
        P("T2 echo only -20.66 dB, b/a = 1, %s: worst %.6f (stored %.6f, resid %.2e) | mean %.6f (stored %.6f, resid %.2e) | best %.6f (stored %.6f, resid %.2e) | worst chi %.1f (stored %.1f)" %
          (kind, sj["pen_wc"], row["D_%s_wc" % tag], sj["pen_wc"] - row["D_%s_wc" % tag], sj["pen_mean"], row["D_%s_mean" % tag],
           sj["pen_mean"] - row["D_%s_mean" % tag], sj["pen_min"], row["D_%s_best" % tag], sj["pen_min"] - row["D_%s_best" % tag],
           sj["chi_wc_deg"], row["D_%s_theta_wc_deg" % tag]))
        P("   I vs Q at every chi: max |EO_I - EO_Q| = %.3e" % float(np.abs(RI - RQ).max()))
    ST["T6_I_vs_Q_echo_only_max|dEO|"] = float(np.abs(eo_ratio("I", "bessel", nu=nu, ref="same_ba") - eo_ratio("Q", "bessel", nu=nu, ref="same_ba")).max())

    # ---- T2b arm ratio of the center termination, base convention (EO0 of the same arms)
    for kind, tag in (("ideal", "ideal"), ("bessel", "B75")):
        RI = eo_ratio("I", kind, b=BA, nu=nu, ref="same_ba")
        RQ = eo_ratio("Q", kind, b=BA, nu=nu, ref="same_ba")
        sj = stats(np.minimum(RI, RQ))
        sv = bbarow[("b/a center termination", "lens, share-weighted x, y, 45 deg")]["%s_wc" % tag]
        ST["T2b_echo_ba_center_%s_pen_wc" % tag] = {"mine": sj["pen_wc"], "stored": sv, "resid_dB": sj["pen_wc"] - sv}
        P("T2b echo -20.66 dB with b/a = %.4f (base convention, EO0 of the same arms), %s: worst %.6f (stored %.6f, resid %.2e)" %
          (BA, kind, sj["pen_wc"], sv, sj["pen_wc"] - sv))

    # ---- T3 pure bias error: closed form EO/EO0 = cos d - sin|d|  (I and Q, both signs)
    P("T3 pure bias error (b/a = 1, no echo), closed form  EO/EO0 = cos(d) - sin|d| = sqrt2 cos(45 deg + |d|):")
    T3 = []
    for d in (1.0, 2.565, 5.13, 6.614338, 9.178, 15.610950, 18.176):
        for sgn in (+1, -1):
            for rail in ("I", "Q"):
                Ri = eo_ratio(rail, "ideal", phi=np.deg2rad(sgn * d), chis=CHIS[:1])[0]
                Rb = eo_ratio(rail, "bessel", phi=np.deg2rad(sgn * d), chis=CHIS[:1])[0]
                T3.append((d, sgn, rail, Ri, Rb))
        cf = float(cf_bias(d))
        ri = [t[3] for t in T3 if t[0] == d]
        rb = [t[4] for t in T3 if t[0] == d]
        P("   d = %7.3f deg: closed form %.15f (%.6f dB) | ideal sampling, 4 runs (+-d, I, Q): max resid %.2e | Bessel: %.6f .. %.6f (%.6f dB, vs closed form %.2e)" %
          (d, cf, float(pen(cf)), max(abs(x - cf) for x in ri), min(rb), max(rb), float(pen(min(rb))), min(rb) - cf))
    ST["T3_pure_bias_ideal_max_resid"] = max(abs(t[3] - float(cf_bias(t[0]))) for t in T3)
    ST["T3_pure_bias_bessel_max_resid"] = max(abs(t[4] - float(cf_bias(t[0]))) for t in T3)
    ST["T3_pure_bias_bessel_resid_dB_at_2.565"] = float(pen(min(t[4] for t in T3 if t[0] == 2.565)) - pen(cf_bias(2.565)))
    P("   ideal-sampling max residual over all %d runs: %.3e;  Bessel max residual: %.3e (the filter's intersymbol terms, not an error)" %
      (len(T3), ST["T3_pure_bias_ideal_max_resid"], ST["T3_pure_bias_bessel_max_resid"]))

    # ---- T4 single-PD phase error
    for e in (2.565, 5.13, 12.0):
        r = eo_ratio("I", "ideal", eps_pd=(np.deg2rad(e), 0.0), chis=CHIS[:1])[0]
        r2 = eo_ratio("I", "ideal", eps_pd=(0.0, np.deg2rad(e)), chis=CHIS[:1])[0]
        cf = float(cf_single_pd(e))
        ST["T4_single_PD_eps_%.3f" % e] = {"mine_pd1": float(r), "mine_pd2": float(r2), "closed": cf, "resid": float(max(abs(r - cf), abs(r2 - cf)))}
        P("T4 phase error %.3f deg on ONE PD of the I pair: EO/EO0 = %.12f / %.12f (PD1 / PD2); closed form cos(e/2)(cos(e/2)-sin(e/2)) = %.12f, resid %.2e; "
          "rail-level d = e/2 = %.4f deg alone would give %.12f" % (e, r, r2, cf, max(abs(r - cf), abs(r2 - cf)), e / 2, float(cf_bias(e / 2))))

    # ---- T5 uniform spread, no echo: EO/EO0 = mean cos(delta_j) (zero net bias); per channel ptp
    Wc = spread_W(PTP_DEG)
    r = eo_ratio("I", "ideal", W=Wc, chis=CHIS[:1])[0]
    ST["T5_uniform_spread_no_echo"] = {"mine": float(r), "closed_V_cos_mean": float(abs(Wc)), "resid": float(abs(r - abs(Wc))), "W": [Wc.real, Wc.imag]}
    P("T5 uniform spread of the eight atoms over %.4f deg (1550 nm), no echo: EO/EO0 = %.12f, closed form V = |mean exp(i d_j)| = %.12f, resid %.2e, penalty %.5f dB" %
      (PTP_DEG, r, abs(Wc), abs(r - abs(Wc)), float(pen(r))))
    for lab in ("1546.003", "1554.017"):
        Wl = spread_W(PTP_LAB[lab])
        rl = eo_ratio("I", "ideal", W=Wl, chis=CHIS[:1])[0]
        ST["T5_uniform_spread_no_echo_%s" % lab] = {"ptp_deg": PTP_LAB[lab], "mine": float(rl), "closed_V_cos_mean": float(abs(Wl)), "resid": float(abs(rl - abs(Wl)))}
        P("T5 same at %s nm, ptp %.4f deg: EO/EO0 = %.12f, closed form %.12f, resid %.2e, penalty %.5f dB" % (lab, PTP_LAB[lab], rl, abs(Wl), abs(rl - abs(Wl)), float(pen(rl))))

    # ---- T11 real atoms: the rail-level code with the complex W reproduces |W| (no echo, no arm imbalance) and the sampled spread stays inside the second-order bound
    for lab in ("1550.000", "1546.003", "1554.017"):
        a = ATOMS[lab]
        Wa = complex(a["W"])
        ra = eo_ratio("I", "ideal", W=Wa, chis=CHIS[:1])[0]
        closed = float(np.cos(np.angle(Wa)) - np.sin(abs(np.angle(Wa)))) * abs(Wa)      # bias residual arg(W) plus visibility |W|
        so = 1.0 - 0.5 * np.deg2rad(a["rms_deg_w"]) ** 2                                    # second-order |W| of the usage-weighted spread (weights W only, rms about the mean)
        ST["T11_atoms_%s" % lab] = {"W_abs": abs(Wa), "W_arg_deg": float(np.rad2deg(np.angle(Wa))), "eo_code": float(ra), "eo_closed": closed, "resid": float(abs(ra - closed)),
                                    "second_order_abs_w_only": float(so), "abs_w_only": a["w"]["abs"], "resid_second_order": float(abs(so - a["w"]["abs"]))}
        P("T11 real atoms at %s nm: |W| %.8f, residual bias arg(W) %+.5f deg; code EO/EO0 %.12f vs closed form (cos a - sin|a|)|W| %.12f resid %.2e; second-order |W| = 1 - rms^2/2 = %.8f vs %.8f (usage weights only)" %
          (lab, abs(Wa), np.rad2deg(np.angle(Wa)), ra, closed, abs(ra - closed), so, a["w"]["abs"]))
    ST["T12_increment_45deg_vs_SI_0.104"] = {"mine": INC45, "SI": 0.104, "resid": abs(INC45 - 0.104)}
    P("T12 45 deg echo increment at 1550 nm: %.5f dB (SI line 451: 0.104 dB), resid %.2e dB" % (INC45, abs(INC45 - 0.104)))

    # ---- T7 first-order closed form with bias error, arm ratio and echo (necho = 1), ideal sampling, rail I and Q
    worst = 0.0
    for dd in (0.0, 2.565, 9.178, 18.176, -15.61):
        for ba in (1.0, BA):
            kap = nu * ba
            for rail in ("I", "Q"):
                R = eo_ratio(rail, "ideal", b=ba, nu=nu, necho=1, phi=np.deg2rad(dd), ref="ideal")
                cf = cf_first_order(kap, dd, CHIS, ba)          # I and Q give the same EO(chi) (symmetry, T6)
                worst = max(worst, float(np.abs(R - cf).max()))
    ST["T7_first_order_closed_form_max_resid"] = worst
    P("T7 first-order closed form EO/EO_ideal = V_arm[(cos d - sin|d|) - sqrt2 kappa max(|cos u|,|sin u|)], u = chi + 45 deg, vs the field-level code with necho = 1 (d in 0, 2.565, 9.178, 18.176, -15.61 deg; b/a = 1 and %.4f; I and Q; 720 chi): max resid %.3e" % (BA, worst))

    # ---- T8 sign invariance and I-Q identity of the full model (device 1554 nm, headline uniform spread, all returns, Bessel)
    p = case_params(-15.610950, "uniform", ptp_deg=PTP_LAB["1554.017"])
    pm = dict(p)
    pm["phi"] = -p["phi"]
    sA, RA = rails_stats(p, "bessel")
    sB, RB = rails_stats(pm, "bessel")
    d1 = max(abs(sA["I"][k] - sB["I"][k]) for k in ("pen_wc", "pen_mean", "pen_min"))
    d2 = float(np.abs(RA["I"] - RA["Q"]).max())
    d3 = float(np.abs(np.sort(RA["I"]) - np.sort(RB["I"])).max())
    ST["T8_sign_invariance_max_resid_dB"] = d1
    ST["T8_I_vs_Q_full_model_max|dEO|"] = d2
    P("T8 full model, 1554.017 nm, uniform spread: phase sign +-: max resid of (worst, mean, best penalty) %.3e dB, sorted EO(chi) %.3e; I vs Q at every chi: max |dEO| = %.3e" % (d1, d3, d2))

    # ---- T9 series truncation: 6 vs 9 returns
    pp = case_params(-15.610950, "uniform", ptp_deg=PTP_LAB["1554.017"])
    s6 = stats(eo_ratio("I", "bessel", chis=CHIS, necho=6, **pp))
    s9 = stats(eo_ratio("I", "bessel", chis=CHIS, necho=9, **pp))
    ST["T9_returns_6_vs_9_resid_dB"] = max(abs(s6[k] - s9[k]) for k in ("pen_wc", "pen_mean", "pen_min"))
    P("T9 returns kept 6 vs 9 (1554.017 nm, uniform spread, Bessel, I): max resid %.3e dB" % ST["T9_returns_6_vs_9_resid_dB"])

    # ---- T10 arm ratio only
    r = eo_ratio("I", "ideal", b=BA, chis=CHIS[:1], ref="ideal")[0]
    ST["T10_arm_ratio_only"] = {"mine": float(r), "closed": float(cf_arm(BA)), "resid": float(abs(r - cf_arm(BA))), "pen_dB": float(pen(r))}
    P("T10 arm ratio only (b/a = %.5f, same total power): EO/EO_ideal %.12f, closed form 2(b/a)/(1+(b/a)^2) = %.12f, resid %.2e, penalty %.5f dB" %
      (BA, r, cf_arm(BA), abs(r - cf_arm(BA)), float(pen(r))))
    progress("self-tests done at %.1f s" % (time.time() - t0))

    # ---------------------------------------------------------------- the penalty tables
    kinds = (("bessel", "Bessel 4th order, 0.75 x baud, sampled at the eye center"), ("ideal", "ideal sampling, no filter"))
    rows = []

    def add(case, treat, tag, p, extra=None):
        for kind, _ in kinds:
            st, R = rails_stats(p, kind)
            rows.append({"case": case, "treat": treat, "tag": tag, "kind": kind, "I": st["I"], "Q": st["Q"], "joint": st["joint"],
                         "params": {"nu": p["nu"], "b_over_a": p["b"] / p["a"], "phi_eff_deg": float(np.rad2deg(p["phi"])),
                                    "W_abs": float(abs(p["W"])), "W_arg_deg": float(np.rad2deg(np.angle(p["W"])))}, "extra": extra or {}})
        return

    # reference
    add("ideal receiver", "-", "ideal", {"a": 1.0, "b": 1.0, "nu": 0.0, "W": 1.0, "phi": 0.0})
    for ch, phi in CHANNELS:
        for treat in ("uniform", "atoms", "half", "full"):
            add("device %s nm" % ch, treat, "all", case_params(phi, treat, ptp_deg=PTP_LAB[ch], atoms_w=ATOMS[ch]["W"]),
                {"phi_ch_deg": phi, "ptp_deg": PTP_LAB[ch]})
        progress("table rows done for %s at %.1f s" % (ch, time.time() - t0))

    # decomposition (Bessel + ideal) -- each impairment alone and the combination
    dec = []
    for ch, phi in CHANNELS:
        pt, aw = PTP_LAB[ch], ATOMS[ch]["W"]
        spec = [
            ("echo only (b/a = 1)", dict(a=1.0, b=1.0, nu=nu_of(EPS_DB), W=1.0, phi=0.0)),
            ("arm ratio only", dict(a=1.0, b=BA, nu=0.0, W=1.0, phi=0.0)),
            ("common phase only", dict(a=1.0, b=1.0, nu=0.0, W=1.0, phi=np.deg2rad(phi))),
            ("spread only, uniform (HEADLINE)", dict(a=1.0, b=1.0, nu=0.0, W=spread_W(pt), phi=0.0)),
            ("spread only, lens-weighted atoms", dict(a=1.0, b=1.0, nu=0.0, W=aw, phi=0.0)),
            ("spread only, half-range (BOUND)", dict(a=1.0, b=1.0, nu=0.0, W=1.0, phi=np.deg2rad(pt / 2))),
            ("spread only, full range (BOUND)", dict(a=1.0, b=1.0, nu=0.0, W=1.0, phi=np.deg2rad(pt))),
            ("echo + arm ratio", dict(a=1.0, b=BA, nu=nu_of(EPS_DB), W=1.0, phi=0.0)),
            ("echo + arm ratio + common phase", case_params(phi, None, spread=False)),
            ("all, spread uniform (HEADLINE)", case_params(phi, "uniform", ptp_deg=pt)),
            ("all, spread lens-weighted atoms", case_params(phi, "atoms", atoms_w=aw)),
            ("all, spread half-range (BOUND)", case_params(phi, "half", ptp_deg=pt)),
        ]
        for name, p in spec:
            for kind, _ in kinds:
                st, R = rails_stats(p, kind)
                dec.append({"channel": ch, "item": name, "kind": kind, "joint": st["joint"], "I": st["I"], "Q": st["Q"]})
    progress("decomposition done at %.1f s" % (time.time() - t0))

    # ESTIMATE rows (rebuilt 261015): per-channel echo = C1 measured x + y share echo at the channel + the 45 deg increment measured at 1550 nm (C2 full_enum).  At 1550 nm the
    # measured x + y + 45 deg value itself (EPS_DB) is used.  Superseded rule (pre-fix Table 4): x-ramp aperture echo + (EPS_DB - x-ramp echo at 1550).  Headline spread (uniform).
    est = []
    for ch, phi in CHANNELS:
        e_db = EPS_DB if ch == "1550.000" else ECHO_XY_C1[ch] + INC45
        p = case_params(phi, "uniform", eps_db=e_db, ptp_deg=PTP_LAB[ch])
        st, R = rails_stats(p, "bessel")
        std, Rd = rails_stats(p, "ideal")
        e_lo = EPS_DB if ch == "1550.000" else ECHO_XY_C1[ch]                         # x + y only, 45 deg cells excluded (measured at the edges)
        p_lo = case_params(phi, "uniform", eps_db=e_lo, ptp_deg=PTP_LAB[ch])
        stl, _ = rails_stats(p_lo, "bessel")
        e_old = ECHO_AP_X[ch] + ECHO_XY45_OFFSET                                       # superseded estimate
        p_old = case_params(phi, "uniform", eps_db=e_old, ptp_deg=PTP_LAB[ch])
        sto, _ = rails_stats(p_old, "bessel")
        est.append({"channel": ch, "echo_dB_ESTIMATE": e_db, "echo_xy_C1_dB": ECHO_XY_C1[ch], "inc45_dB": INC45, "echo_xy_only_dB": e_lo,
                    "echo_superseded_dB": e_old, "echo_x_ramp_dB": ECHO_AP_X[ch], "offset_dB_superseded": ECHO_XY45_OFFSET,
                    "bessel": st["joint"], "ideal": std["joint"], "bessel_xy_only": stl["joint"], "bessel_superseded": sto["joint"]})
    progress("estimate rows done at %.1f s" % (time.time() - t0))

    # eye traces for the figure: rail I, headline uniform spread, worst chi of the Bessel eye (I rail); normalized by EO_ideal(Bessel)
    traces = {}
    eo_id = eo_reference("I", 1.0, 1.0, "bessel", "ideal")
    tr_cases = [("ideal receiver", {"a": 1.0, "b": 1.0, "nu": 0.0, "W": 1.0, "phi": 0.0})]     # eye-trace panels: HEADLINE = uniform spread
    for ch, phi in CHANNELS:
        tr_cases.append(("device %s nm" % ch, case_params(phi, "uniform", ptp_deg=PTP_LAB[ch])))
    meta = {}
    arrs = {}
    for k, (name, p) in enumerate(tr_cases):
        Rb = eo_ratio("I", "bessel", chis=CHIS, **p)
        i = int(np.argmin(Rb))
        chi = CHIS[i]
        ne = NECHO if p["nu"] > 0 else 0
        L = level_wave("I", p["a"], p["b"], p["nu"], np.array([chi]), necho=ne, phi=p["phi"], W=p["W"])[0]
        ai = float(np.sqrt((p["a"] ** 2 + p["b"] ** 2) / 2.0))
        eo_ref = eo_reference("I", p["a"], p["b"], "bessel", "ideal")
        flat = filt_wave(L).reshape(-1)                       # (N * OS,) periodic
        idx = (np.arange(N)[:, None] * OS + PCS - OS + np.arange(2 * OS + 1)[None, :]) % (N * OS)
        yw = flat[idx]                                        # (N, 2 OS + 1), the sampling instant of slot k at index OS
        arrs["traces_%d" % k] = (yw / eo_ref).astype(np.float32)
        arrs["pos_%d" % k] = POS["I"]
        Y = sample(L[None, :], "bessel")[0]
        meta[name] = {"chi_wc_deg": float(np.rad2deg(chi)), "eo_ratio": float(Rb[i]), "pen_dB": float(pen(Rb[i])),
                      "pen_mean_dB": float(np.mean(pen(Rb))), "eo_check": float(eye_open(Y[None, :], "I")[0] / eo_ref)}
    np.savez_compressed(OUT + r"\eye_traces_fig7.npz", pcs=PCS, os=OS, **arrs)
    json.dump(meta, open(OUT + r"\eye_traces_fig7_meta.json", "w"), indent=1)
    P("eye traces:", json.dumps(meta))

    json.dump({"constants": {"EPS_DB": EPS_DB, "nu": nu_of(EPS_DB), "BA": BA, "PA": PA, "PB": PB, "PTP_DEG": PTP_DEG, "PTP_CH": PTP_LAB, "RHO2": RHO2,
                             "atoms": {k: {kk: vv for kk, vv in v.items() if kk != "W"} for k, v in ATOMS.items()},
                             "echo_xy_C1": ECHO_XY_C1, "echo_xy45_1550": ECHO_XY45_1550, "inc45": INC45,
                             "phi_ch": dict(CHANNELS), "echo_ap_x": ECHO_AP_X, "xy45_offset": ECHO_XY45_OFFSET, "NECHO": NECHO,
                             "OS": OS, "F3": F3, "pc": PCS, "g0": float(G_TAPS[0]), "n_chi": len(CHIS)},
               "selftests": ST, "rows": rows, "decomposition": dec, "estimate": est},
              open(OUT + r"\fig7_eye_numbers.json", "w"), indent=1)
    open(OUT + r"\selftest_log.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
    P("done %.1f s; process RSS %.2f GB" % (time.time() - t0, psutil.Process().memory_info().rss / 1e9))
    open(OUT + r"\selftest_log.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
