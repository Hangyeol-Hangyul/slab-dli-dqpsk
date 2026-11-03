# -*- coding: ascii -*-
r"""fdtd_atom_post.py -- turn FullWAVE monitor output of the sim758 atom checks into the quantities of
atom2d_opt (absR00, argR00_deg, A_deg, B_deg, argrB_deg, argtB_deg, amp_*, echo_*, ledA_* / ledB_*,
carrier_eff_pct), plus the k-ladder (Richardson with a fitted order) and the comparison with grcwa Li / Laurent.

DRAFT of 261009 (fdtdbuild role), FIXED 261009 01:xx by the fix758 role (quality flags, proxy-calibrated ladder, Tier B marking),
FIXED AGAIN 261009 by the fix2 758 role after check758b (non-finite data, overridden tolerances, verdict rule).
Standalone: numpy only, no RSoft, no grcwa. The driver Main_fdtd_atom_verify.py imports this file (install both in
code\Version_unified\RCWA_custom).
ASCII-only source and console output (cp949 consoles); every open() has encoding="utf-8".

RUN QUALITY (fix758, skeptic FAIL #14). `analyse` now FILLS `flags` and `ok` of a result requires them empty. Seven gates, tolerances in
QC_DEFAULT with the stored runs that fix them: converged (FullWAVE's own 'Converged at L' line, both runs), energy |R + T - 1|, monitor-pair
spread, empty-run |R2/R1| = 1, empty-run phase advance vs k n dz, cross-pol |Ey/Ex|, cell count. A gate that cannot be evaluated
(no log facts) is a flag too (QC_CONVERGENCE_UNKNOWN): unchecked is not passed. `analyse --qc key=val` re-judges saved orders with a
different tolerance and records it (`quality.tol_overridden`, also `tol_overridden` at the top level of the result).
NON-FINITE (fix2 758): every number read or derived must be finite. NaN / Inf in a monitor file (counted per file in `orders.nonfinite_samples`,
a Windows '1.#QNAN' token is read as NaN, not cut at the '#'), in an order, or in a derived quantity gives the flag QC_NONFINITE (ok False, exit 1);
every gate fails on a number it cannot read (`x <= tol` is False for NaN); `calibrate_order` / `extrapolate` / `_verdict` / `combine_B` return
UNRESOLVED (ok False) for any non-finite input.
OVERRIDES (fix2 758): `analyse --qc key=val` (finite positive values only) is recorded in the result; `ladder` and `combine` REFUSE a result judged
with tolerances other than QC_DEFAULT (read from `quality.tol`, so a lost declaration does not hide it) -> UNRESOLVED with the reason, unless
`--accept-overridden` is given (then listed in `notes` / `overridden_accepted`).

LADDER (fix758, skeptic FAIL #15). `ladder` takes the proxy rungs (known Li limit, plan S5), measures the FDTD convergence order per quantity,
applies it to the real-atom k1/k2/k3 rungs, and prints per quantity the extrapolated value, its error bar and LI / LAURENT / UNRESOLVED
(d arg t primary, d arg r secondary, abs r tertiary). Rungs that carry flags, non-finite values or overridden tolerances are refused.
VERDICT RULE (fix2 758): LI (LAURENT) only when the whole interval v_inf +- ERR_K * err_bar lies within gap / 3 of Li (Laurent); ERR_K = 2.5 was
calibrated on the skeptic's 48,000-ladder Monte Carlo design (20 independent sets = 960,000 ladders: 0 wrong-side verdicts; the old rule 2,701 in the
0.4-0.6-of-the-gap zone) and checked on three beyond-design stress families (tests/mc_verdict.py). err_bar = max of E_dist, E_hind, E_p, E_free,
E_pair, E_noise (see `extrapolate`).

CONVENTIONS (fixed by PLAN.md of fdtdplan 261008, sections 1-2; every one is exercised by `selftest`)
  * Launch = Ex (polarization = 0 in fullwave is TM; plan S3: |Ex| 0.5752 vs cos(th) 0.578).
  * sigma is READ from the empty run (phase advance of the launched order, R1 -> R2); never assumed. If sigma = -1
    every phasor is conjugated, as in the prototypes. sigma = +1 in all of S2/S3/S4 (no conjugation).
  * Order index m is the PHYSICAL x-Fourier index: c_m = < E exp(-i sigma m G x) > with G = 2 pi / Lx, so kx = m G
    in every basis. Tier B has Lx = Lambda: +th launch = kx = +G = m_inc +1, -th launch = m_inc -1. The record's
    RELATIVE order labels map as  m_rec = m_abs - m_inc :  +th: t(-1) = air m_abs 0, r(0) = m_abs +1 (specular),
    R(-1) = m_abs 0, R(-2) = m_abs -1;  -th: t(+1) = air m_abs 0, r(0) = m_abs -1 (specular), R(+1) = m_abs 0,
    R(+2) = m_abs +1 (the ECHO).
  * Basis: FDTD gives Ex, grcwa _amp gives H_y. Reflection r_H = -r_Ex (phase +180 deg); transmission
    t_H = t_Ex * (n_t cos_i) / (n_i cos_t) (REAL positive factor -> same phase). Powers from Ex:
    P ~ n |Ex|^2 / cos(theta): R_m = |r_Ex|^2 cos_i / cos_m ; T = |t_Ex|^2 n_t cos_i / (n_i cos_t).
  * Reference planes, as grcwa: the incident amplitude is referred to z_in_ref (Tier A: the L3 face z = 0 -- the
    plane where _r00_l3 reads b/a; Tier B: 0.5 um below the L1 bottom = the start of grcwa's substrate layer, where
    _amp reads rB), r at that same plane, t at the top face of the stack:
        r_m(z_ref) = [c_m(zR)/c_inc(zR)] * exp( +i (kz_m + kz_i) (zR - z_ref) )
        t(z_ref -> z_top) = [c_t(zT)/c_inc(zR)] * exp( -i kz_t (zT - z_top) ) * exp( +i kz_i (zR - z_ref) )
    argR00_deg = (180 + arg r_Ex) mod 360 (record: `% 360`), A / argtB / argrB in (-180, 180] (record: atan2).
  * The empty-run incident phasor is subtracted ONLY from the launched (specular) order. Non-specular orders (the
    echo) are read from the structure run alone, as the plan says; the empty run's content at those orders is
    reported as `empty_leak` (the launch artefact, ~1e-2 in amplitude at tilt).
  * NOT JUDGED (plan item 4): carrier_eff_pct (0.25 / 0.56 pp gap, needs ~1e-3 relative power) and the specular
    rB / B_deg (tilted-reflection artefact ~1e-2 in amplitude, INFRA2 260908). They are REPORTED, flagged.

Sub-commands (python -I -B fdtd_atom_post.py <cmd> ...):
  selftest                                 synthetic RSoft files: algebra, sigma = -1, power budget, ladder fit, quality gates
  extract  --manifest M --out O.json [--emp DIR --st DIR]     monitor files -> Fourier orders json
  analyse  --manifest M --orders O.json --out R.json [--result OLD_R.json | --emp-log F --st-log F] [--qc energy=1e-3 ...]   orders json -> record + quality flags
  combine  --pth R1.json --mth R2.json [--blank-pth ... --blank-mth ...] --out C.json     Tier B full record
  ladder   --atoms k1.json,k2.json,k3.json --blank B.json|tmm --rcwa S1.json --atom 032p5 --proxy-atoms P1.json,... --proxy-ref REF.json
                                           FDTD vs Li / Laurent with the proxy-calibrated order (without --proxy-* the overall verdict is UNRESOLVED)
  tmm      --manifest M                    analytic blank reference of a Tier A manifest
"""
import argparse
import cmath
import glob
import io
import json
import math
import os
import re
import sys

import numpy as np

SCHEMA = "fdtd_atom_manifest_v1"


# ------------------------------------------------------------------ small helpers
def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def deg(z):
    return math.degrees(cmath.phase(z))


def _c(v):
    return complex(v[0], v[1]) if isinstance(v, (list, tuple)) else complex(v)


def _san(o):
    if isinstance(o, dict):
        return {str(k): _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_json(path, obj):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_san(obj), ensure_ascii=True, indent=1))
    os.replace(tmp, path)


def load_json(path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fin(x):
    """True only for a real, finite number (complex: both parts). bool / str / None are not numbers here: a gate that cannot read a
    number must not pass it (fix2 758, skeptic FAIL: every `x > tol` test is False for NaN, so NaN used to pass every gate)."""
    if isinstance(x, (bool, np.bool_, str)) or x is None:
        return False
    try:
        if isinstance(x, (complex, np.complexfloating)):
            return bool(math.isfinite(x.real) and math.isfinite(x.imag))
        return bool(math.isfinite(float(x)))
    except (TypeError, ValueError, OverflowError):
        return False


def nonfinite_paths(obj, prefix="", none_is_bad=False, _out=None, _cap=40):
    """Paths of every non-finite number inside a nested dict / list / tuple / ndarray / number structure (NaN, +-Inf, complex with a bad part).
    bool and str leaves are skipped (labels); None is a bad leaf only when none_is_bad (a saved json holds None where a float was NaN,
    see _san). At most _cap paths are collected (the count of the rest is not needed: one is enough to refuse)."""
    out = [] if _out is None else _out
    if len(out) >= _cap:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            nonfinite_paths(v, "%s/%s" % (prefix, k), none_is_bad, out, _cap)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            nonfinite_paths(v, "%s[%d]" % (prefix, i), none_is_bad, out, _cap)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fc" and not bool(np.isfinite(obj).all()):
            out.append(prefix or "/")
    elif obj is None:
        if none_is_bad:
            out.append(prefix or "/")
    elif isinstance(obj, (bool, np.bool_, str, bytes)):
        pass
    elif isinstance(obj, (int, float, complex, np.number)):
        if not _fin(obj):
            out.append(prefix or "/")
    return out


# ------------------------------------------------------------------ RSoft 2-D monitor files (same as Main_fdtd_input2d_verify)
def _read_tolerant(path):
    """Data lines of a monitor file with every token float() cannot parse ('1.#QNAN', '-nan(ind)', ...) read as NaN, so it is counted and
    flagged (QC_NONFINITE) downstream. A ragged file (rows of different length) still raises ValueError."""
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()[4:]
    rows = []
    for ln in lines:
        if not ln.strip():
            continue
        row = []
        for tok in ln.split():
            try:
                row.append(float(tok))
            except ValueError:
                row.append(float("nan"))
        rows.append(row)
    if len({len(r) for r in rows}) > 1:
        raise ValueError("ragged monitor file %s: row lengths %s" % (path, sorted({len(r) for r in rows})))
    return np.atleast_2d(np.array(rows, dtype=float))


def load_field(path):
    """RSoft 2-D complex amp/phase file -> (cx[nx, ny], xg, yg). Rows = X, columns = Y."""
    with io.open(path, encoding="utf-8") as f:
        hdr = [f.readline() for _ in range(4)]
    p3, p4 = hdr[2].split(), hdr[3].split()
    nx, xmin, xmax = int(p3[0]), float(p3[1]), float(p3[2])
    ny, ymin, ymax = int(p4[0]), float(p4[1]), float(p4[2])
    try:
        # comments=None: loadtxt's default comment character '#' would cut a Windows '1.#QNAN' / '-1.#IND' token to '1.' / '-1.' and
        # read it as a NUMBER (measured 261009 with numpy 2.2.6: 1.#QNAN -> 1.0, 1.#INF -> 1.0) -- a silent corruption, not a crash
        raw = np.atleast_2d(np.loadtxt(path, skiprows=4, comments=None))
    except ValueError:
        raw = _read_tolerant(path)
    with np.errstate(invalid="ignore", over="ignore"):         # Inf x exp(i 0) = Inf + NaN i: counted as non-finite downstream, no warning noise
        cx = raw[:, 0::2] * np.exp(1j * np.radians(raw[:, 1::2]))
    return cx, np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)


def find_dft(workdir, prefix, mon, comp):
    hits = sorted(glob.glob(os.path.join(workdir, "%s_m%d_f*_%s.dat" % (prefix, mon, comp))))
    if not hits:
        hits = sorted(glob.glob(os.path.join(workdir, "%s_m%d_*_%s.dat" % (prefix, mon, comp))))
    return hits[0] if hits else None


def parse_log(path, index=-1):
    """FullWAVE log.txt -> the facts of ONE run block (default the last one: a log that was appended to holds several, one per run).
    Same regexes as Main_fdtd_input2d_verify.run_fullwave. Returns {} when the file or a block is missing. `converged` is True only
    when FullWAVE printed 'Converged at L=...' (fdtd_stop_auto 2 reached its tolerance); a run that ran to the stop cap, or a fixed-time
    run (fdtd_stop_auto 0), has converged False."""
    if not path or not os.path.isfile(path):
        return {}
    with io.open(path, encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()
    blocks = [b for b in re.split(r"SimTool: FullWAVE", txt) if "FDTD grid info" in b]
    if not blocks:
        return {}
    try:
        b = blocks[index]
    except IndexError:
        return {}
    info = {}
    m = re.search(r"NX=(\d+) NY=(\d+) NZ=(\d+), NT=(\d+)", b)
    if m:
        info.update(nx=int(m.group(1)), ny=int(m.group(2)), nz=int(m.group(3)), nt=int(m.group(4)))
    m = re.search(r"FDTD Auto Time Step = ([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", b)
    if m:
        info["dt_um"] = float(m.group(1))
    m = re.search(r"Converged at L=(\d+)\.\s+New NT=(\d+), Stop Time = ([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", b)
    if m:
        info.update(converged=True, conv_L=int(m.group(1)), nt_run=int(m.group(2)), stop_time_um=float(m.group(3)))
    elif "nt" in info and "dt_um" in info:
        info.update(converged=False, nt_run=info["nt"], stop_time_um=info["nt"] * info["dt_um"])
    else:
        info["converged"] = False
    info["finished"] = "Actual Computation Time" in b
    return info


def logfacts_from_manifest(man):
    """Log facts of both runs from the run folders the driver recorded in the manifest (offline re-analysis). When the two runs share one
    folder (stored plan smoke runs) the first block of the log is the empty run and the last one the structure run."""
    out = {}
    runs = man.get("runs", {})
    d_e, d_s = runs.get("emp", {}).get("dir"), runs.get("st", {}).get("dir")
    shared = bool(d_e) and d_e == d_s
    for run, d in (("emp", d_e), ("st", d_s)):
        if d:
            f = parse_log(os.path.join(d, "log.txt"), index=(0 if (shared and run == "emp") else -1))
            if f:
                out[run] = f
    return out


def periodic_crop(cx, xg, yg, lx, ly):
    """Drop a duplicated closing row / column when the file includes both edges of the period."""
    nx, ny = cx.shape
    dxg = (xg[-1] - xg[0]) / max(nx - 1, 1)
    dyg = (yg[-1] - yg[0]) / max(ny - 1, 1)
    if abs((xg[-1] - xg[0]) - lx) < 0.25 * dxg:
        cx, xg = cx[:-1, :], xg[:-1]
    if abs((yg[-1] - yg[0]) - ly) < 0.25 * dyg:
        cx, yg = cx[:, :-1], yg[:-1]
    return cx, xg, yg


def order_coeff(cx, xg, yg, m, n, lx, ly, sigma):
    """c_mn = <E exp(-i sigma (m Gx x + n Gy y))> over the (cropped) period."""
    kx = 2.0 * math.pi * m / lx
    ky = 2.0 * math.pi * n / ly
    ph = np.exp(-1j * sigma * (kx * xg[:, None] + ky * yg[None, :]))
    with np.errstate(invalid="ignore", over="ignore"):
        return complex(np.mean(cx * ph))


# ------------------------------------------------------------------ analytic reference (TM, exp(-i w t))
def tmm_h(n_list, d_list, kx, k0):
    """Recursive Fresnel for H_y (TM); returns (R_h, T_h) referred to the first / last interface."""
    eps = [n * n for n in n_list]
    kz = []
    for e in eps:
        q = cmath.sqrt(e * k0 * k0 - kx * kx)
        kz.append(-q if q.imag < 0 else q)
    p = [kz[j] / eps[j] for j in range(len(eps))]
    nif = len(eps) - 1
    R, T = 0j, 1 + 0j
    for j in range(nif - 1, -1, -1):
        rj = (p[j] - p[j + 1]) / (p[j] + p[j + 1])
        tj = 2 * p[j] / (p[j] + p[j + 1])
        if j == nif - 1:
            R, T = rj, tj
        else:
            d = d_list[j]
            e2, e1 = cmath.exp(2j * kz[j + 1] * d), cmath.exp(1j * kz[j + 1] * d)
            den = 1 + rj * R * e2
            T = tj * T * e1 / den
            R = (rj + R * e2) / den
    return R, T


def tmm_blank_tierA(h_lo, gap, h_up, n_sub, lam):
    """Atom-alone BLANK stack at normal incidence: SiO2 | SiO2 h_lo | SiO2 gap | air h_up | air. Returns the
    E-basis (r_E, t_E) and H-basis (r_H, t_H) referred to the L3 face (z = 0) and the top face. Same arithmetic as
    the S4 smoke (rb_e = -R_h, tb_e = T_h * n_sub), validated by S2a to 1.5e-5 / 0.01 deg."""
    k0 = 2.0 * math.pi / lam
    R_h, T_h = tmm_h([n_sub, n_sub, n_sub, 1.0, 1.0], [h_lo, gap, h_up], 0.0, k0)
    return dict(r_E=-R_h, t_E=T_h * n_sub, r_H=R_h, t_H=T_h)


# ------------------------------------------------------------------ extraction: monitor files -> Fourier orders
def _spec_orders(man):
    """Absolute x-orders to extract for the structure run (reflection side R, transmission side T) + the empty run."""
    return sorted(set(int(m) for m in man["orders_R"]) | {int(man["m_inc"])}), sorted(set(int(m) for m in man["orders_T"]))


def extract_orders(man, dirs):
    """Read the monitor files of the empty run and of the structure run and return the Fourier orders.

    man: manifest (schema fdtd_atom_manifest_v1); dirs: {"emp": dir, "st": dir}. Returns a dict with sigma, the
    launched order found, the spectrum of the empty run, and for each run / monitor / component {m: [re, im]}
    for the requested orders (Ex and Ey). Raises FileNotFoundError("NOT FOUND: ...") for a missing file."""
    lam = man["lam_um"]
    k0 = 2.0 * math.pi / lam
    out = dict(schema=man.get("schema"), emp={}, st={}, nonfinite_samples={})
    ordR, ordT = _spec_orders(man)
    # lateral cell per run
    cell = {"emp": (man["runs"]["emp"]["lx_um"], man["runs"]["emp"]["ly_um"]),
            "st": (man["runs"]["st"]["lx_um"], man["runs"]["st"]["ly_um"])}
    fields = {}
    bad_samples = {}
    for run in ("emp", "st"):
        R = man["runs"][run]
        lx, ly = cell[run]
        for mn, mon in R["mon"].items():
            for comp in ("ex", "ey"):
                f = find_dft(dirs[run], R["prefix"], int(mon), comp)
                if f is None:
                    raise FileNotFoundError("NOT FOUND: %s_m%d_f*_%s.dat in %s" % (R["prefix"], int(mon), comp, dirs[run]))
                cx_f, xg_f, yg_f = load_field(f)
                nbad = int(cx_f.size - np.count_nonzero(np.isfinite(cx_f)))      # counted on the WHOLE file, before the periodic crop
                if nbad:
                    bad_samples["%s/%s/%s" % (run, mn, comp)] = nbad
                fields[(run, mn, comp)] = periodic_crop(cx_f, xg_f, yg_f, lx, ly)
    # sigma and the launched order from the empty run (Ex), spectrum in the sigma = +1 basis first
    lx_e, ly_e = cell["emp"]
    cx1, xg1, yg1 = fields[("emp", "R1", "ex")]
    cx2, xg2, yg2 = fields[("emp", "R2", "ex")]
    spec1 = {m: abs(order_coeff(cx1, xg1, yg1, m, 0, lx_e, ly_e, 1)) for m in range(-3, 4)}
    m_s1 = max(spec1, key=spec1.get)
    a1 = order_coeff(cx1, xg1, yg1, m_s1, 0, lx_e, ly_e, 1)
    a2 = order_coeff(cx2, xg2, yg2, m_s1, 0, lx_e, ly_e, 1)
    adv = cmath.phase(a2 / a1)
    sigma = 1 if adv > 0 else -1
    spec = {m: abs(order_coeff(cx1, xg1, yg1, m, 0, lx_e, ly_e, sigma)) for m in range(-3, 4)}
    m_found = max(spec, key=spec.get)
    out["nonfinite_samples"] = bad_samples                     # {run/monitor/component: count of non-finite samples}; empty = every sample finite
    out["sigma"] = sigma
    out["m_inc_found"] = int(m_found)
    out["spectrum_R1"] = {str(m): v for m, v in spec.items()}
    out["phase_adv_deg"] = math.degrees(adv)
    out["abs_ratio_R2_R1"] = abs(a2 / a1)
    dzR = man["z"]["R2"] - man["z"]["R1"]
    kx_f = 2.0 * math.pi * m_found / lx_e
    q2 = (k0 * man["n_sub"]) ** 2 - kx_f * kx_f
    out["phase_adv_expect_deg"] = math.degrees(sigma * math.sqrt(max(q2, 0.0)) * dzR)
    for run in ("emp", "st"):
        R = man["runs"][run]
        lx, ly = cell[run]
        out[run] = {}
        for mn in R["mon"]:
            ent = {}
            for comp in ("ex", "ey"):
                cx, xg, yg = fields[(run, mn, comp)]
                ms = ordR if mn.startswith("R") else ordT
                if run == "emp":
                    ms = sorted(set(ms) | {int(m_found), int(man["m_inc"])})
                ent[comp] = {str(m): order_coeff(cx, xg, yg, m, 0, lx, ly, sigma) for m in ms}
            out[run][mn] = ent
    return out


# ------------------------------------------------------------------ analysis (pure arithmetic)
def _kz(k, kx):
    q2 = k * k - kx * kx
    return math.sqrt(q2) if q2 > 0.0 else complex(0.0, math.sqrt(-q2))


# ------------------------------------------------------------------ run quality: the gate behind `ok` (fix758, skeptic FAIL #14)
# Tolerances and the STORED runs that fix them (plan smoke runs S2 / S3 / S4, measured by the build role 261009; [E] = chosen margin):
#   energy |R + T - 1|             A 5e-4 : stored normal-incidence runs 3e-6 .. 5.8e-5 (S4 k1 4.1e-6, k2 6.4e-6; S2b full a-Si 5.8e-5), margin ~9x,
#                                           an energy error of 5e-4 moves a phase by < 0.03 deg [E];
#                                  B 1e-2 : stored Littrow slab S3r 2.0e-4 (50x margin); the default launch_pml 1 at a tilted launch is documented to leave the
#                                           empty domain non-lossless, 9.2e-3 in amplitude / 1.85e-2 in power (INFRA2 260908; S3r, with the same thin
#                                           PML, shows 10x less). PROVISIONAL: no Tier B run of the real stack exists yet; the measured values are always stored in
#                                           quality.checks, so a tolerance can be re-judged with `analyse --qc` (recorded) after the first Tier B result is read
#   spread  |x1 - x2| / max(|x|, 0.05) between the monitor pairs (R1 / R2, T1 / T2), judged orders only
#                                  A 2e-3 : stored 1.4e-5 .. 1.5e-4 (= 0.11 deg between the two planes at 2e-3; the quoted value is their mean)
#                                  B 5e-3 : S3r transmitted 6e-4 (specular reflection 3.1e-3 is NOT gated in tier B: rB is not judged, plan item 4)
#   ratio   | |R2 / R1| - 1 | of the launched order in the EMPTY run (lossless uniform medium: exactly 1)
#                                  A 1e-3 : stored 1e-16 .. 2.4e-5      B 5e-3 : S3r 1.06e-3 (tilted launch)
#   phase_adv_deg  | measured - expected | of the empty-run phase advance R1 -> R2 (k_z dz; a wrong sigma or plane distance shows tens of degrees)
#                                  A 0.05 : stored 0.0013 / 0.0019      B 0.5 : S3r 0.055
#   cross_pol  |Ey / Ex| of the launched (R1) and the transmitted (T1) order     1e-4 in both tiers: stored 3e-8; the atoms are y-mirror
#                                  symmetric to 1e-13 (plan eps_*.npy), so the physical value is 0 and a TM / TE mix-up shows up at O(1)
# Convergence is FullWAVE's own 'Converged at L' line of BOTH runs; a run cut at the stop cap (or run with fdtd_stop_auto 0) is not converged.
QC_DEFAULT = {"A": dict(energy=5e-4, spread_rel=2e-3, ratio=1e-3, phase_adv_deg=0.05, cross_pol=1e-4),
              "B": dict(energy=1e-2, spread_rel=5e-3, ratio=5e-3, phase_adv_deg=0.5, cross_pol=1e-4)}
QC_KEYS = tuple(QC_DEFAULT["A"])
SPREAD_FLOOR = 0.05


def _within(x, tol):
    """x <= tol, and False for NaN / Inf / None: a gate must FAIL on a number it cannot read (every `x > tol` is False for NaN)."""
    return bool(_fin(x) and _fin(tol) and float(x) <= float(tol))


def _res_nonfinite(res):
    """Non-finite numbers in the quantities of one analysed launch (reflection / transmission records, powers, cross-pol, the record keys)."""
    paths = []
    for key in ("refl", "trans", "record", "cross_pol"):
        if key in res:                                    # an absent cross_pol is reported by its own gate (QC_CROSS_POL), not as a non-finite value
            nonfinite_paths(res[key], key, none_is_bad=True, _out=paths)
    for key in ("energy", "R_total", "T_total", "abs_inc_R1", "abs_ratio_R2_R1", "cos_inc"):
        if key in res and not _fin(res[key]):
            paths.append(key)
    return paths


def quality(man, res, orders, logfacts=None, qc=None):
    """Run-quality gates of ONE launch -> (flags, summary). A gate that cannot be evaluated is a flag, never a pass; so is every value that is
    not a finite number (QC_NONFINITE: NaN / Inf in a monitor file, in an order, or in a derived quantity).
    logfacts: {"emp": facts, "st": facts} as returned by run_fullwave / parse_log; qc: tolerance overrides {key: value}, each a finite positive
    number (anything else is refused). `tol_overridden` lists every tolerance that differs from QC_DEFAULT after the overrides: ladder / combine
    refuse such a result unless they are told --accept-overridden."""
    tier = str(man.get("tier", "B")).upper()
    tol = dict(QC_DEFAULT["A" if tier == "A" else "B"])
    default = dict(tol)
    for src in (man.get("qc"), qc):
        for k, v in (src or {}).items():
            if k not in QC_KEYS:
                raise ValueError("unknown quality tolerance %r (known: %s)" % (k, ", ".join(QC_KEYS)))
            if not (_fin(v) and float(v) > 0.0):
                raise ValueError("quality tolerance %s = %r must be a finite positive number (a NaN / Inf / 0 tolerance would pass or fail everything)" % (k, v))
            tol[k] = float(v)
    over = [k for k in QC_KEYS if tol[k] != default[k]]
    checks, flags = {}, []
    lf = logfacts or {}
    # ---- (0) every number read or derived is finite (monitor samples, orders, results)
    bad = []
    for name, cnt in sorted((orders.get("nonfinite_samples") or {}).items()):
        bad.append("monitor file %s: %s non-finite sample(s)" % (name, cnt))
    for key in ("sigma", "m_inc_found", "phase_adv_deg", "phase_adv_expect_deg", "abs_ratio_R2_R1"):
        if not _fin(orders.get(key)):
            bad.append("orders/%s" % key)
    for run in ("emp", "st"):
        for p in nonfinite_paths(orders.get(run), "orders/" + run, none_is_bad=True):
            bad.append(p)
    bad += ["result/" + p for p in _res_nonfinite(res)]
    checks["finite"] = dict(value=len(bad), tol=0, ok=not bad, first=bad[:6])
    if bad:
        flags.append("QC_NONFINITE: %d non-finite value(s), first: %s -- a NaN / Inf run is not a measurement (every gate below fails on it too)"
                     % (len(bad), "; ".join(bad[:4])))
    # ---- (1) convergence of the time signal, both runs
    for run in ("emp", "st"):
        f = lf.get(run)
        if not f or "converged" not in f:
            checks["converged_" + run] = dict(value=None, tol=True, ok=False)
            flags.append("QC_CONVERGENCE_UNKNOWN:%s no FullWAVE log facts for the %s run -- a converged run cannot be told from one cut at the stop cap "
                         "(unchecked is not passed)" % (run, run))
        else:
            good = bool(f["converged"])
            checks["converged_" + run] = dict(value=good, tol=True, ok=good, nt_run=f.get("nt_run"), stop_time_um=f.get("stop_time_um"))
            if not good:
                flags.append("QC_NOT_CONVERGED:%s the %s run ended without 'Converged at L' (nt_run %s, stop time %s um): the DFT monitors hold an "
                             "unfinished ring-down" % (run, run, f.get("nt_run"), f.get("stop_time_um")))
    # ---- (2) cell count of both runs against the cells the driver built (the exact-raster guard)
    ce = (man.get("checks") or {}).get("cells_expected")
    if ce:
        for run, (kx, ky) in (("st", ("nx", "ny")), ("emp", ("nx_empty", "ny_empty"))):
            f = lf.get(run) or {}
            if "nx" in f and "ny" in f:
                want = [int(ce[kx]) + 3, int(ce[ky]) + 3]
                good = bool(f["nx"] == want[0] and f["ny"] == want[1])
                checks["cells_" + run] = dict(value=[f["nx"], f["ny"]], tol=want, ok=good)
                if not good:
                    flags.append("QC_CELLS:%s log NX,NY %s,%s, built %d,%d (+3 ghost): the raster is not the one the job wrote" % (run, f["nx"], f["ny"], want[0], want[1]))
    # ---- (3) energy balance
    e = abs(float(res["energy"]) - 1.0)
    checks["energy"] = dict(value=e, tol=tol["energy"], ok=_within(e, tol["energy"]))
    if not _within(e, tol["energy"]):
        flags.append("QC_ENERGY: |R + T - 1| = %.3e > %.1e (R %.5f, T %.5f)" % (e, tol["energy"], res["R_total"], res["T_total"]))
    # ---- (4) monitor-pair spread of the judged orders (tier A: r00 and t_n; tier B: transmitted + non-specular reflected, not the specular rB)
    items = []
    for m, v in res["refl"].items():
        if int(m) == int(res["m_inc"]) and tier != "A":
            continue
        items.append(("R%+d" % int(m), v["spread"], v["abs_rE"]))
    for m, v in res["trans"].items():
        items.append(("T%+d" % int(m), v["spread"], v["abs_tE"]))
    rel = {n: s / max(a, SPREAD_FLOOR) for n, s, a in items}
    worst = max(rel, key=lambda n: rel[n] if _fin(rel[n]) else float("inf")) if rel else None      # a NaN order is the worst one
    wv = (rel[worst] if _fin(rel[worst]) else float("nan")) if worst else 0.0
    checks["spread"] = dict(value=wv, tol=tol["spread_rel"], ok=_within(wv, tol["spread_rel"]), worst=worst, per_order=rel)
    if not _within(wv, tol["spread_rel"]):
        flags.append("QC_SPREAD: monitor pairs disagree by %.3e (relative, order %s) > %.1e" % (wv, worst, tol["spread_rel"]))
    # ---- (5) empty-run incident amplitude ratio
    rr = abs(float(res["abs_ratio_R2_R1"]) - 1.0)
    checks["ratio"] = dict(value=rr, tol=tol["ratio"], ok=_within(rr, tol["ratio"]))
    if not _within(rr, tol["ratio"]):
        flags.append("QC_RATIO: empty-run |R2 / R1| - 1 = %.3e > %.1e (the launched wave changes amplitude between the two monitors)" % (rr, tol["ratio"]))
    # ---- (6) empty-run phase advance against k_z dz
    pa, pe = orders.get("phase_adv_deg"), orders.get("phase_adv_expect_deg")
    if not (_fin(pa) and _fin(pe)):
        checks["phase_adv"] = dict(value=None, tol=tol["phase_adv_deg"], ok=False)
        flags.append("QC_PHASE_ADV: phase advance not available in the orders file (unchecked is not passed)")
    else:
        d = abs(wrap180(float(pa) - float(pe)))
        checks["phase_adv"] = dict(value=d, tol=tol["phase_adv_deg"], ok=_within(d, tol["phase_adv_deg"]))
        if not _within(d, tol["phase_adv_deg"]):
            flags.append("QC_PHASE_ADV: empty-run phase advance %.4f deg vs expected %.4f deg (diff %.3e > %.2g deg)" % (pa, pe, d, tol["phase_adv_deg"]))
    # ---- (7) cross-polarisation
    cp = res.get("cross_pol")
    if not cp:
        checks["cross_pol"] = dict(value=None, tol=tol["cross_pol"], ok=False)
        flags.append("QC_CROSS_POL: Ey of the launched / transmitted order not available (unchecked is not passed)")
    else:
        c_pair = (float(cp["Ey_R1_launched_over_inc"]), float(cp["Ey_T1_over_inc"]))
        c = max(c_pair) if all(_fin(x) for x in c_pair) else float("nan")       # max(0.1, nan) is 0.1 in python: never let a NaN hide
        checks["cross_pol"] = dict(value=c, tol=tol["cross_pol"], ok=_within(c, tol["cross_pol"]))
        if not _within(c, tol["cross_pol"]):
            flags.append("QC_CROSS_POL: |Ey / Ex| = %.3e > %.1e (the launch is not the pure TM / Ex wave the analysis assumes)" % (c, tol["cross_pol"]))
    return flags, dict(ok=not flags, tol=tol, tol_overridden=sorted(set(over)), checks=checks)


def analyse(man, orders, logfacts=None, qc=None):
    """Manifest + Fourier orders -> record. Reflection of every listed order, transmission of every listed order,
    both from the monitor pairs (R1, T1) and (R2, T2); the quoted value is the complex mean of the two, `spread`
    their distance (a standing-wave / ring-down indicator). The run-quality gates (`quality`) fill `flags`; a result is
    usable only when `flags` is empty (`quality_ok`). logfacts / qc: see `quality`."""
    lam = man["lam_um"]
    k0 = 2.0 * math.pi / lam
    n_in, n_out = float(man["n_sub"]), float(man["n_out"])
    lx = man["lx_um"]
    z = man["z"]
    sigma = int(orders["sigma"])
    cv = (lambda c: c) if sigma > 0 else (lambda c: c.conjugate())
    m_inc = int(man["m_inc"])
    flags = []
    nonfinite_in = (bool(orders.get("nonfinite_samples")) or bool(nonfinite_paths(orders.get("emp"), none_is_bad=True))
                    or bool(nonfinite_paths(orders.get("st"), none_is_bad=True)))
    if int(orders["m_inc_found"]) != m_inc and not nonfinite_in:
        # a +th job that launched -th (or vice versa) is a different measurement, not a noisy one: never analyse it under the wrong label
        # (not raised for non-finite data: the launched order is then read from NaN and means nothing -- the run is flagged QC_NONFINITE instead)
        raise ValueError("LAUNCH_ORDER_MISMATCH: the empty run launched x-order %d, the manifest (launch %s) expects %d -- "
                         "launch_angle sign convention differs; orders are saved, nothing was analysed"
                         % (int(orders["m_inc_found"]), man.get("launch"), m_inc))
    kx_i = 2.0 * math.pi * m_inc / lx
    kz_i = _kz(k0 * n_in, kx_i)
    if isinstance(kz_i, complex):
        raise ValueError("incident order m_inc %d is evanescent in the substrate" % m_inc)
    cos_i = kz_i / (k0 * n_in)

    def Eo(run, mon, comp, m):
        d = orders[run][mon][comp]
        key = str(int(m))
        if key not in d:
            raise KeyError("order %s not extracted for %s %s %s" % (key, run, mon, comp))
        return cv(_c(d[key]))

    inc = {j: Eo("emp", "R" + j, "ex", m_inc) for j in "12"}
    res = dict(schema="fdtd_atom_result_v1", tier=man["tier"], launch=man["launch"], atom=man.get("atom"), k=man.get("k"),
               sigma=sigma, m_inc=m_inc, theta_deg=man.get("theta_deg"), cos_inc=cos_i, flags=flags,
               abs_inc_R1=abs(inc["1"]), abs_ratio_R2_R1=abs(inc["2"] / inc["1"]),
               phase_adv_deg=orders.get("phase_adv_deg"), phase_adv_expect_deg=orders.get("phase_adv_expect_deg"),
               z_in_ref_um=z["z_in_ref"], z_top_um=z["top"],
               proxy_nc=man.get("proxy_nc"), fdtd_gpu=man.get("fdtd_gpu"), dz_um=man.get("dz_um"),
               stack=(dict(h_lo_um=man["h_lo_um"], gap_um=man["gap_um"], h_up_um=man["h_up_um"], n_sub=n_in, lam_um=lam)
                      if all(k in man for k in ("h_lo_um", "gap_um", "h_up_um")) else None))
    # ---- reflection side
    refl = {}
    for m in man["orders_R"]:
        m = int(m)
        kx = 2.0 * math.pi * m / lx
        kz = _kz(k0 * n_in, kx)
        if isinstance(kz, complex):
            continue                                  # evanescent order carries no flux
        cos_m = kz / (k0 * n_in)
        vals = {}
        for j in "12":
            tot = Eo("st", "R" + j, "ex", m)
            if m == m_inc:
                tot = tot - inc[j]                    # empty-run subtraction ONLY for the launched (specular) order
            rE = tot / inc[j] * cmath.exp(1j * (kz + kz_i) * (z["R" + j] - z["z_in_ref"]))
            vals[j] = rE
        rE_m = 0.5 * (vals["1"] + vals["2"])
        leak = abs(Eo("emp", "R1", "ex", m)) / abs(inc["1"]) if (m != m_inc and str(m) in orders["emp"]["R1"]["ex"]) else 0.0
        refl[m] = dict(r_E=rE_m, r_H=-rE_m * (cos_i / cos_m), abs_rE=abs(rE_m), arg_rE_deg=deg(rE_m),
                       arg_rH_deg=deg(-rE_m), R=abs(rE_m) ** 2 * cos_i / cos_m, spread=abs(vals["1"] - vals["2"]),
                       r_E_1=vals["1"], r_E_2=vals["2"], cos_m=cos_m, kx_um=kx, empty_leak=leak)
    res["refl"] = refl
    # ---- transmission side
    trans = {}
    for m in man["orders_T"]:
        m = int(m)
        kx = 2.0 * math.pi * m / lx
        kz_t = _kz(k0 * n_out, kx)
        if isinstance(kz_t, complex):
            raise ValueError("transmitted order %d is evanescent in the output medium" % m)
        cos_t = kz_t / (k0 * n_out)
        vals = {}
        for j in "12":
            cT = Eo("st", "T" + j, "ex", m)
            tE = (cT / inc[j] * cmath.exp(-1j * kz_t * (z["T" + j] - z["top"]))
                  * cmath.exp(1j * kz_i * (z["R" + j] - z["z_in_ref"])))
            vals[j] = tE
        tE_m = 0.5 * (vals["1"] + vals["2"])
        fac = (n_out * cos_i) / (n_in * cos_t)       # t_H = t_E * fac (real, positive)
        trans[m] = dict(t_E=tE_m, t_H=tE_m * fac, abs_tE=abs(tE_m), abs_tH=abs(tE_m) * fac, arg_t_deg=deg(tE_m),
                        T=abs(tE_m) ** 2 * fac, spread=abs(vals["1"] - vals["2"]), t_E_1=vals["1"], t_E_2=vals["2"],
                        cos_t=cos_t, kx_um=kx)
    res["trans"] = trans
    res["R_total"] = float(sum(v["R"] for v in refl.values()))
    res["T_total"] = float(sum(v["T"] for v in trans.values()))
    res["energy"] = res["R_total"] + res["T_total"]
    # cross-polarisation diagnostic (Ey of the launched order at R1, of the transmitted order at T1)
    try:
        ey_r = abs(cv(_c(orders["st"]["R1"]["ey"][str(m_inc)])))
        mt = int(man["m_t"])
        ey_t = abs(cv(_c(orders["st"]["T1"]["ey"][str(mt)])))
        res["cross_pol"] = dict(Ey_R1_launched_over_inc=ey_r / abs(inc["1"]), Ey_T1_over_inc=ey_t / abs(inc["1"]))
    except KeyError:
        pass
    res["record"] = _record_keys(man, res)
    q_flags, q = quality(man, res, orders, logfacts=logfacts, qc=qc)
    flags.extend(q_flags)                                  # `flags` is the list already stored in res
    res["quality"] = q
    res["quality_ok"] = not flags
    res["tol_overridden"] = list(q["tol_overridden"])      # also top level: a re-analysis with --qc relaxes the gates and must stay visible
    return res


def _record_keys(man, res):
    """The atom2d_opt-style keys this single launch can give. Reflection keys of the record are in the H basis."""
    tier, launch = man["tier"], man["launch"]
    refl, trans = res["refl"], res["trans"]
    mt = int(man["m_t"])
    t = trans[mt]
    rec = {}
    if tier == "A":
        r0 = refl[0]
        rec.update(absR00=r0["abs_rE"], argR00_deg=(r0["arg_rE_deg"] + 180.0) % 360.0,
                   abs_tn_E=t["abs_tE"], abs_tn_H=t["abs_tH"], arg_tn_deg=t["arg_t_deg"],
                   R_atom=r0["R"], T_atom_meas=t["T"], energy=res["energy"],
                   argR00_note="(180 + arg r_Ex) mod 360 = arg of H-basis b/a at the L3 face (_r00_l3)",
                   arg_tn_note="arg t at the top face / incident at the L3 face = the atom part of A_deg and argtB_deg")
        return rec
    if tier == "B" and launch == "pth":
        sp = refl[res["m_inc"]]                           # specular = record's r(0)
        rec.update(amp_tA=t["abs_tH"], A_deg=wrap180(t["arg_t_deg"]), amp_rB=sp["abs_rE"] * 1.0,
                   argrB_deg=wrap180(sp["arg_rH_deg"]),
                   **{"ledA_T_m-1": t["T"], "ledA_R_m+0": sp["R"], "ledA_R_m-1": refl[0]["R"],
                      "ledA_R_m-2": refl[res["m_inc"] - 2]["R"] if (res["m_inc"] - 2) in refl else 0.0})
        rec["ledA_R_total"] = res["R_total"]
        rec["ledA_T_total"] = res["T_total"]
        rec["energy_pth"] = res["energy"]
        return rec
    if tier == "B" and launch == "mth":
        sp = refl[res["m_inc"]]
        echo = refl[res["m_inc"] + 2] if (res["m_inc"] + 2) in refl else None
        rec.update(amp_tB=t["abs_tH"], argtB_deg=wrap180(t["arg_t_deg"]),
                   **{"ledB_T_m+1": t["T"], "ledB_R_m+0": sp["R"], "ledB_R_m+1": refl[0]["R"]})
        if echo is not None:
            rEh = echo["r_H"]
            rec.update(echo_R_p2=echo["R"], echo_p2_db=10.0 * math.log10(max(echo["R"], 1e-12)),
                       echo_r2_re=rEh.real, echo_r2_im=rEh.imag, echo_empty_leak=echo["empty_leak"],
                       **{"ledB_R_m+2": echo["R"]})
        rec["ledB_R_total"] = res["R_total"]
        rec["ledB_T_total"] = res["T_total"]
        rec["energy_mth"] = res["energy"]
        # specular r of this launch (reported only)
        rec["rB_mth_specular_abs"] = sp["abs_rE"]
        return rec
    return rec


# ------------------------------------------------------------------ Tier B: combine the two launches
def combine_B(pth, mth, blank_pth=None, blank_mth=None, accept_overridden=False):
    """Full atom2d_opt-style record from the +th and -th launches of ONE structure. B_deg = arg(rB * tB) as in
    _measure_arms (rB = specular reflection at +th, tB = transmission at -th). Differentials against a blank
    structure (same two launches) are added when given. carrier_eff_pct uses the ledger exactly as _measure_arms
    does (ep = ledA_T_m-1, R_go = ledA_R_m+0, em = ledB_T_m+1); NOT RESOLVABLE by FDTD (plan item 4).
    `status` is OK or UNRESOLVED (`ok` = status OK; `status_reasons` says why): UNRESOLVED when a launch (or the blank) is flagged / not ok,
    when ANY input or output number is not finite (QC_NONFINITE), or when a launch was judged with overridden quality tolerances
    (`analyse --qc`) and accept_overridden is False (CLI: --accept-overridden; an accepted override stays listed in `overridden_accepted`)."""
    a, b = pth["record"], mth["record"]
    if pth["launch"] != "pth" or mth["launch"] != "mth":
        raise ValueError("combine_B needs (pth, mth)")
    rB = pth["refl"][pth["m_inc"]]["r_H"]
    tB = mth["trans"][int(_first_t(mth))]["t_H"]
    tA = pth["trans"][int(_first_t(pth))]["t_H"]
    rBtB = rB * tB
    A_deg = wrap180(deg(tA))
    B_deg = wrap180(deg(rBtB))
    out = dict(a)
    out.update(b)
    out.update(amp_tA=abs(tA), amp_rB=abs(rB), amp_tB=abs(tB), amp_rBtB=abs(rBtB), prod_amp=abs(tA) * abs(rBtB),
               A_deg=A_deg, B_deg=B_deg, argrB_deg=wrap180(deg(rB)), argtB_deg=wrap180(deg(tB)),
               phi_c_deg=wrap180(0.5 * (A_deg + B_deg)), dpsi_deg=wrap180(B_deg - A_deg))
    ep, rgo, em = (_num(out.get(k, 0.0)) for k in ("ledA_T_m-1", "ledA_R_m+0", "ledB_T_m+1"))      # a None (saved NaN) becomes NaN, not a crash
    sig2 = ep * rgo * em
    sig = (math.sqrt(sig2) if sig2 > 0.0 else 0.0) if _fin(sig2) else float("nan")
    out["carrier_eff_pct"] = 100.0 * sig / ((4.0 / 9.0) ** 1.5)
    echo = _num(out.get("ledB_R_m+2", 0.0))
    out["carrier_echo_db"] = (10.0 * math.log10(echo) if echo > 0.0 else float("-inf")) if _fin(echo) else float("nan")
    out["eff_ok"] = bool(_within(-out["carrier_eff_pct"], -85.0))                    # carrier_eff_pct >= 85, False for NaN
    out["echo_ok"] = _within(_num(out.get("echo_p2_db", 0.0)), -20.5)
    out["not_judged"] = ["carrier_eff_pct", "amp_rB", "B_deg", "argrB_deg", "phi_c_deg", "dpsi_deg", "prod_amp"]
    out["flags"] = list(pth.get("flags", [])) + list(mth.get("flags", []))
    out["ok"] = bool(_res_ok(pth) and _res_ok(mth))
    reasons = []
    if not out["ok"]:
        reasons.append("a launch is flagged / not ok: %s" % ("; ".join(str(f)[:60] for f in out["flags"]) or "driver ok = false"))
    nf = ["pth/" + p for p in _res_nonfinite(pth)] + ["mth/" + p for p in _res_nonfinite(mth)]
    nf += ["combined/" + p for p in nonfinite_paths([A_deg, B_deg, out["amp_rBtB"], out["prod_amp"], out["carrier_eff_pct"], ep, rgo, em, echo],
                                                    "v", none_is_bad=True)]
    nf += ["records/" + p for p in nonfinite_paths([a, b], "rec", none_is_bad=True)]
    if nf:
        out["flags"].append("QC_NONFINITE:combine %d non-finite value(s), first: %s" % (len(nf), "; ".join(nf[:3])))
        reasons.append("QC_NONFINITE: %d non-finite input / output value(s), first %s" % (len(nf), "; ".join(nf[:3])))
    ov_bad, ov_acc = _override_report((("pth", pth), ("mth", mth)), accept_overridden)
    if ov_bad:
        out["flags"].append("TOL_OVERRIDDEN: " + "; ".join(ov_bad))
        reasons.append("TOL_OVERRIDDEN: " + "; ".join(ov_bad))
    if ov_acc:
        out["overridden_accepted"] = ov_acc
    # ---- what a Tier B result may and may not decide (fix758, skeptic #8 / #12; budgets = Li-Laurent gap / 3 of PLAN.md, errors [E] from the plan scaling)
    atom = str(pth.get("atom") or mth.get("atom") or "")
    k_a, k_e = int(pth.get("k") or 1), int(mth.get("k") or 1)
    judged = {}
    if atom == "032p5" and min(k_a, k_e) <= 1:
        judged["A_deg_argtB_deg"] = ("JUDGED for 032p5 only: k1 transmission-phase error ~1.1 deg [E] vs budget 1.55 / 1.00 deg (gap 4.66 / 3.00 deg / 3), "
                                     "read through vs_blank, never the absolute value")
    elif atom == "347p5" and min(k_a, k_e) <= 1:
        judged["A_deg_argtB_deg"] = ("NOT JUDGED at k1: k1 transmission-phase error ~1.1 deg [E] > budget 1.00 deg (gap 2.99 / 3.00 deg / 3); "
                                     "correct by hand with the Tier A ladder or release k2")
    else:
        judged["A_deg_argtB_deg"] = "reference / higher-k record: judged only against its own k-ladder error bar"
    if k_e <= 1:
        judged["echo_p2_db"] = ("NOT JUDGED at k1: grid error ~1 dB [E] vs budget 0.82 / 0.53 dB (gap 2.46 / 1.60 dB / 3); the echo also depends on the atom's "
                                "OBLIQUE response, which the Tier A normal-incidence ladder does not bound. Needs k2 for both atoms or an oblique-incidence atom check")
        out["not_judged"] += ["echo_p2_db", "d_echo_db"]
    else:
        judged["echo_p2_db"] = "k%d: judged only if its own ladder error bar is inside 0.82 / 0.53 dB (032p5 / 347p5)" % k_e
    out["judged"] = judged
    out["spread"] = dict(tA=pth["trans"][int(_first_t(pth))]["spread"], tB=mth["trans"][int(_first_t(mth))]["spread"],
                         rB=pth["refl"][pth["m_inc"]]["spread"])
    if blank_pth is not None and blank_mth is not None:
        bl = combine_B(blank_pth, blank_mth, accept_overridden=accept_overridden)
        out["ok"] = bool(out["ok"] and bl["ok"] and not reasons)       # a flagged blank makes every differential against it unusable
        out["blank_flags"] = list(bl["flags"])
        reasons += ["blank: " + r for r in bl["status_reasons"]]
        if bl.get("overridden_accepted"):
            out["overridden_accepted"] = dict(out.get("overridden_accepted", {}), blank=bl["overridden_accepted"])
        out["vs_blank"] = dict(
            d_A_deg=wrap180(A_deg - bl["A_deg"]), d_argtB_deg=wrap180(out["argtB_deg"] - bl["argtB_deg"]),
            d_B_deg=wrap180(B_deg - bl["B_deg"]), d_argrB_deg=wrap180(out["argrB_deg"] - bl["argrB_deg"]),
            d_echo_db=_num(out.get("echo_p2_db", float("nan"))) - _num(bl.get("echo_p2_db", float("nan"))),
            d_carrier_eff_pp=out["carrier_eff_pct"] - bl["carrier_eff_pct"],
            blank=dict(A_deg=bl["A_deg"], argtB_deg=bl["argtB_deg"], echo_p2_db=bl.get("echo_p2_db"),
                       carrier_eff_pct=bl["carrier_eff_pct"]))
    out["ok"] = bool(out["ok"] and not reasons)
    out["status_reasons"] = reasons
    out["status"] = "OK" if out["ok"] else "UNRESOLVED"
    return out


def _num(x):
    """float(x) for a finite number, NaN for anything else (None from a saved NaN, text): arithmetic on it then stays NaN instead of crashing."""
    return float(x) if _fin(x) else float("nan")


def _first_t(res):
    return sorted(res["trans"].keys(), key=lambda k: int(k))[0]


def _res_ok(r):
    """A result is usable when the driver said ok (it folds in cells_ok), else when it has no flags (post-only results)."""
    return bool(r.get("ok", not r.get("flags")))


def overridden_keys(r):
    """The quality tolerances a saved result was judged with that are NOT the QC_DEFAULT of its tier (sorted key list; [] = judged with the
    shipped gates). Reads the declared `tol_overridden` AND compares the stored `quality.tol` with QC_DEFAULT, so a result whose declaration
    was lost (or edited away) still shows the difference. A result without a `quality.tol` block cannot be verified: its keys are all reported."""
    q = r.get("quality") or {}
    keys = set(q.get("tol_overridden") or []) | set(r.get("tol_overridden") or [])
    default = QC_DEFAULT["A" if str(r.get("tier", "B")).upper() == "A" else "B"]
    tol = q.get("tol")
    if not isinstance(tol, dict):
        return sorted(QC_KEYS) if q else sorted(keys)
    for k in QC_KEYS:
        if not _fin(tol.get(k)) or float(tol[k]) != default[k]:
            keys.add(k)
    return sorted(keys)


def _override_report(named, accept):
    """(problems, accepted) over named results [(label, result)]: a result judged with overridden tolerances is a problem unless accept."""
    bad, acc = [], {}
    for name, r in named:
        ov = overridden_keys(r)
        if not ov:
            continue
        tol = (r.get("quality") or {}).get("tol") or {}
        txt = "%s judged with overridden quality tolerances %s" % (name, ", ".join("%s=%s" % (k, tol.get(k)) for k in ov))
        if accept:
            acc[name] = {k: tol.get(k) for k in ov}
        else:
            bad.append(txt + " (re-analysed with --qc; pass --accept-overridden to use it anyway)")
    return bad, acc


# ------------------------------------------------------------------ k-ladder: fitted order, Richardson, comparison with RCWA
def fit_order(hs, vs):
    """Three rungs (h1 > h2 > h3, v(h) = v_inf + C h^p). Solves p from (v1 - v2) / (v2 - v3) =
    (h1^p - h2^p) / (h2^p - h3^p) on p in [0.05, 8]. Returns dict(ok, p, v_inf, err_k3, note)."""
    h1, h2, h3 = [float(h) for h in hs]
    v1, v2, v3 = [float(v) for v in vs]
    d12, d23 = v1 - v2, v2 - v3
    if abs(d23) < 1e-12 or d12 * d23 <= 0.0:
        return dict(ok=False, p=None, v_inf=None, err_k3=None,
                    note="rungs not monotone (d12 %.4g, d23 %.4g): no fitted order" % (d12, d23))
    target = d12 / d23

    def g(p):
        return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - target

    ps = np.linspace(0.05, 8.0, 800)
    gs = [g(p) for p in ps]
    root = None
    for i in range(len(ps) - 1):
        if gs[i] == 0.0:
            root = ps[i]
            break
        if gs[i] * gs[i + 1] < 0.0:
            lo, hi = ps[i], ps[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if g(lo) * g(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            root = 0.5 * (lo + hi)
            break
    if root is None:
        return dict(ok=False, p=None, v_inf=None, err_k3=None, note="no root of the order equation in [0.05, 8]")
    C = d23 / (h2 ** root - h3 ** root)
    v_inf = v3 - C * h3 ** root
    return dict(ok=True, p=float(root), v_inf=float(v_inf), err_k3=abs(v3 - v_inf), note="")


def richardson(h_a, v_a, h_b, v_b, p):
    """Two-rung extrapolation with an ASSUMED order p (h_a > h_b)."""
    ra = (h_a / h_b) ** p
    return v_b + (v_b - v_a) / (ra - 1.0)


def rcwa_from_s1(path, atom, nG3="1793", blank="122p5"):
    """Li / Laurent atom-alone numbers (plan folder s1_out.json) as differentials vs the blank atom."""
    d = load_json(path)
    rows = {(r["nG3"], r["rule"]): r for r in d["atoms"][atom]["rows"]}
    brow = {(r["nG3"], r["rule"]): r for r in d["atoms"][blank]["rows"]}
    out = {}
    for rule, name in (("liy", "li"), ("laurent", "laurent")):
        r, b = rows[(nG3, rule)], brow[(nG3, rule)]
        out[name] = dict(abs_r=r["abs_r"], d_arg_r=wrap180(r["arg_r"] - b["arg_r"]), d_arg_t=wrap180(r["arg_t"] - b["arg_t"]),
                         arg_r=r["arg_r"], arg_t=r["arg_t"])
    return out


LAD_ROLE = {"d_arg_t": "primary", "d_arg_r": "secondary", "abs_r": "tertiary"}
LAD_QUANT = ("d_arg_t", "d_arg_r", "abs_r")
P_RANGE = (0.3, 4.0)         # an FDTD lateral order outside this range is not a discretisation order (plan smoke: 0.70 / 0.77)
P_BAND_MIN = 0.1             # order band when the proxy gives a single pair only
NOISE_FLOOR_REL = 0.5 * QC_DEFAULT["A"]["spread_rel"]      # per-rung noise the spread gate cannot exclude: planes may differ by spread_rel, the quoted mean by half of it
ERR_K = 2.5                  # half-width multiplier of the verdict interval (coverage x model-error inflation), calibrated by tests\mc_verdict.py (see _verdict)


def _sub(q, a, b):
    """Signed difference in the quantity's metric (phases wrap, |r| does not)."""
    return wrap180(a - b) if q != "abs_r" else a - b


def calibrate_order(ks, vs, limit, q, unc=0.0):
    """FDTD lateral convergence order from PROXY rungs whose limit is KNOWN (the coarse-24 structure: grcwa Li converged, PLAN section 3).
    err_k = v_k - limit; local order of an adjacent pair p = ln(|e_i| / |e_j|) / ln(h_i / h_j) with h = 1 / k (units cancel); the quoted
    order is that of the FINEST pair (nearest the asymptote); its band is the larger of the spread of the local orders, the order change
    the reference uncertainty `unc` could cause, and P_BAND_MIN when there is one pair only. Refused (ok False) when any rung / the limit /
    unc is not a finite number, when the rungs lie on both sides of the limit, do not shrink, are within 3 unc of the limit, or give an
    order outside P_RANGE."""
    pairs_k = sorted(zip([int(k) for k in ks], [float(v) for v in vs]))
    ks_s = [k for k, _ in pairs_k]
    hs = [1.0 / k for k in ks_s]
    out = dict(ok=False, p=None, p_band=None, pairs=[], errors=[], ks=ks_s, limit=limit, unc=unc, note="")
    if not (_fin(limit) and _fin(unc) and all(_fin(v) for _, v in pairs_k)):
        out["note"] = "NONFINITE: a proxy rung value, the Li limit or its uncertainty is not a finite number (rungs %s, limit %s, unc %s)" % (
            [v for _, v in pairs_k], limit, unc)
        return out
    es = [_sub(q, v, limit) for _, v in pairs_k]
    out["errors"] = es
    if len(ks_s) < 2 or len(set(ks_s)) != len(ks_s):
        out["note"] = "needs >= 2 distinct proxy rungs"
        return out
    if any(e == 0.0 for e in es):
        out["note"] = "a proxy rung sits exactly on the limit"
        return out
    if any((e > 0.0) != (es[0] > 0.0) for e in es):
        out["note"] = "proxy rungs lie on both sides of the limit (not asymptotic)"
        return out
    if any(abs(es[i + 1]) >= abs(es[i]) for i in range(len(es) - 1)):
        out["note"] = "proxy error does not shrink with k (errors %s)" % ", ".join("%.4g" % e for e in es)
        return out
    if abs(es[-1]) <= 3.0 * unc:
        out["note"] = "finest proxy error %.3g is within 3x the reference uncertainty %.3g: the order is not measurable" % (abs(es[-1]), unc)
        return out
    pairs = [math.log(abs(es[i]) / abs(es[i + 1])) / math.log(hs[i] / hs[i + 1]) for i in range(len(es) - 1)]
    p = pairs[-1]
    lr = math.log(hs[-2] / hs[-1])
    p_unc = (unc / abs(es[-2]) + unc / abs(es[-1])) / lr
    band = max(p_unc, (max(pairs) - min(pairs)) if len(pairs) > 1 else P_BAND_MIN)
    out.update(pairs=pairs, p=p, p_band=band)
    if not (P_RANGE[0] <= p <= P_RANGE[1]):
        out["note"] = "fitted order %.3f outside [%.1f, %.1f]" % (p, P_RANGE[0], P_RANGE[1])
        return out
    out["ok"] = True
    return out


def extrapolate(ks, vs, q, calib=None, sig=None):
    """Extrapolate three real-atom rungs (k ascending) to h -> 0.
    With a proxy calibration (calib.ok): Richardson on the FINEST two rungs with the calibrated order p; the k1 rung is HINDCAST (what the
    two-rung model predicts for it) as a consistency check. Without one: the free three-rung order (basis 'uncalibrated').
    err_bar = max of five terms: E_dist |v_k3 - v_inf| (the plan's stated bar), E_hind |v_k1 - hindcast|, E_p change of v_inf over the order band,
    E_free |v_inf(free order) - v_inf(calibrated order)|, E_pair |v_inf(k1, k2 pair) - v_inf(k2, k3 pair)| at the calibrated order (the two
    Richardson pairs must agree). The verdict does NOT use err_bar as a 1-sigma bar: `_verdict` widens it by ERR_K (calibrated by
    tests\\mc_verdict.py: the bar alone missed the actual error in 2.6 % of the skeptic's Monte Carlo ladders, by up to 3.5x).
    E_noise = noise amplification of the Richardson step x max(NOISE_FLOOR_REL floor, the largest measured plane disagreement `sig` of the rungs);
    `sig` = per-rung disagreement of the two monitor planes in the units of q (degrees for a phase), None = floor only.
    Any non-finite rung / calibration / sig value, or non-finite result, gives ok False (reason NONFINITE ...)."""
    pk = sorted(zip([int(k) for k in ks], [float(v) for v in vs]))
    ks_s = [k for k, _ in pk]
    vs_s = [v for _, v in pk]
    finite = all(_fin(v) for v in vs_s)
    if finite and q != "abs_r":
        vs_s = list(np.degrees(np.unwrap(np.radians(vs_s))))
    out = dict(ok=False, reason="", basis=None, ks=ks_s, rungs=vs_s, p_free=None, v_free=None, p_cal=None, p_band=None, v_inf=None,
               E_dist=None, E_hind=None, E_p=None, E_free=None, E_pair=None, E_noise=None, err_bar=None)
    if not finite:
        out["reason"] = "NONFINITE: rung value(s) %s are not all finite numbers -- no extrapolation" % ["%s" % v for v in vs_s]
        return out
    if sig is not None and not (len(sig) == len(vs_s) and all(_fin(x) and float(x) >= 0.0 for x in sig)):
        out["reason"] = "NONFINITE: the per-rung monitor-plane disagreement %s is not a list of finite non-negative numbers" % (["%s" % x for x in sig],)
        return out
    if len(ks_s) != 3:
        out["reason"] = "needs exactly 3 real-atom rungs (got %s)" % ks_s
        return out
    hs = [1.0 / k for k in ks_s]
    d12, d23 = vs_s[0] - vs_s[1], vs_s[1] - vs_s[2]
    if abs(d23) < 1e-12 or d12 * d23 <= 0.0:
        out["reason"] = "rungs not monotone (d12 %.4g, d23 %.4g): no extrapolation" % (d12, d23)
        return out
    fit = fit_order(hs, vs_s)
    out.update(p_free=fit["p"], v_free=fit["v_inf"])
    if calib is not None:
        if not calib.get("ok"):
            out["reason"] = "proxy calibration failed for this quantity: %s" % calib.get("note")
            return out
        if not (_fin(calib.get("p")) and _fin(calib.get("p_band"))):
            out["reason"] = "NONFINITE: the calibrated order / band is not a finite number (p %s, band %s)" % (calib.get("p"), calib.get("p_band"))
            return out
        p, band = float(calib["p"]), float(calib["p_band"])
        v_inf = richardson(hs[1], vs_s[1], hs[2], vs_s[2], p)
        c_coef = (vs_s[2] - vs_s[1]) / (hs[2] ** p - hs[1] ** p)
        e_hind = abs(vs_s[0] - (v_inf + c_coef * hs[0] ** p))
        v_lo = richardson(hs[1], vs_s[1], hs[2], vs_s[2], max(p - band, 0.05))
        v_hi = richardson(hs[1], vs_s[1], hs[2], vs_s[2], p + band)
        e_p = max(abs(v_lo - v_inf), abs(v_hi - v_inf))
        e_free = abs(fit["v_inf"] - v_inf) if fit["ok"] else 0.0
        e_pair = abs(richardson(hs[0], vs_s[0], hs[1], vs_s[1], p) - v_inf)
        out.update(basis="proxy-calibrated order p %.3f +- %.3f" % (p, band), p_cal=p, p_band=band)
    else:
        if not fit["ok"]:
            out["reason"] = fit["note"]
            return out
        v_inf, e_hind, e_p, e_free, e_pair = fit["v_inf"], 0.0, 0.0, 0.0, 0.0
        out["basis"] = "free three-rung order p %.3f (UNCALIBRATED)" % fit["p"]
    e_dist = abs(vs_s[2] - v_inf)
    # noise floor: the spread gate allows the two monitor planes of a rung to differ by QC spread_rel, i.e. a rung value may carry NOISE_FLOOR_REL
    # (relative; radians for a phase) of noise that no gate removes; the Richardson step v_inf = v3 + (v3 - v2) / (r - 1) amplifies it
    p_used = out["p_cal"] if out["p_cal"] is not None else fit["p"]
    r_ = (hs[1] / hs[2]) ** p_used
    amp_n = math.hypot(1.0 + 1.0 / (r_ - 1.0), 1.0 / (r_ - 1.0)) if r_ > 1.0 else float("inf")
    sig_n = math.degrees(NOISE_FLOOR_REL) if q != "abs_r" else NOISE_FLOOR_REL * max(abs(vs_s[2]), SPREAD_FLOOR)
    if sig is not None:
        sig_n = max(sig_n, max(float(x) for x in sig))
    e_noise = amp_n * sig_n
    terms = (v_inf, e_dist, e_hind, e_p, e_free, e_pair, e_noise)
    if not all(_fin(x) for x in terms):
        out["reason"] = "NONFINITE: the extrapolated value or an error term is not a finite number (v_inf, E_dist, E_hind, E_p, E_free, E_pair, E_noise = %s)" % (
            ["%s" % x for x in terms],)
        return out
    out.update(ok=True, v_inf=float(v_inf), E_dist=e_dist, E_hind=e_hind, E_p=e_p, E_free=e_free, E_pair=e_pair, E_noise=e_noise,
               err_bar=max(e_dist, e_hind, e_p, e_free, e_pair, e_noise))
    return out


def _verdict(q, ent):
    """LI / LAURENT / UNRESOLVED with a reason, from an extrapolation entry carrying li and laurent.
    Rule (fix2 758): with the half width h = ERR_K * err_bar, the verdict is the nearer of Li / Laurent S only when the WHOLE interval
    [v_inf - h, v_inf + h] lies within gap / 3 of S, i.e. |v_inf - S| + h <= gap / 3. That interval is then also closer to S than to the other
    (gap / 3 < gap / 2), so a truth inside the interval cannot be on the wrong side; the old rule (err bar < gap / 3 and distance < gap / 3,
    separately) let the truth sit up to 2 gap / 3 from the winner. UNRESOLVED when h > gap / 3 (ERR_BAR_TOO_LARGE), when the interval
    sticks out of gap / 3 (NEITHER), when there is no gap, and for ANY non-finite input (Li, Laurent, v_inf, error terms): never a vote on NaN."""
    li, la = ent["li"], ent["laurent"]
    if not (_fin(li) and _fin(la)):
        return "UNRESOLVED", "NONFINITE: the Li / Laurent reference is not a finite number (Li %s, Laurent %s)" % (li, la)
    gap = abs(_sub(q, li, la))
    ent.update(gap=gap, budget=gap / 3.0, err_k=ERR_K)
    if not ent["ok"]:
        return "UNRESOLVED", ent["reason"]
    keys = ("v_inf", "err_bar", "E_dist", "E_hind", "E_p", "E_free", "E_pair", "E_noise")
    if not all(_fin(ent.get(k)) for k in keys):
        return "UNRESOLVED", "NONFINITE: extrapolated value / error terms are not all finite (%s)" % ", ".join("%s=%s" % (k, ent.get(k)) for k in keys)
    v = ent["v_inf"]
    half = ERR_K * ent["err_bar"]
    d_li, d_la = abs(_sub(q, v, li)), abs(_sub(q, v, la))
    ent.update(dist_li=d_li, dist_laurent=d_la, half_width=half)
    if gap <= 0.0:
        return "UNRESOLVED", "no Li-Laurent gap in this quantity"
    third = gap / 3.0
    if half > third:
        return "UNRESOLVED", "ERR_BAR_TOO_LARGE: %.3g x err bar %.3g = %.3g > gap/3 %.3g (E_dist %.3g, E_hind %.3g, E_p %.3g, E_free %.3g, E_pair %.3g, E_noise %.3g)" % (
            ERR_K, ent["err_bar"], half, third, ent["E_dist"], ent["E_hind"], ent["E_p"], ent["E_free"], ent["E_pair"], ent["E_noise"])
    near, name = (d_li, "LI") if d_li < d_la else (d_la, "LAURENT")
    if near + half > third:
        return "UNRESOLVED", "NEITHER: limit %.4g +- %.3g (%.3g x err bar) reaches %.3g from %s (Li %.3g / Laurent %.3g away), beyond gap/3 %.3g" % (
            v, half, ERR_K, near + half, name, d_li, d_la, third)
    return name, "limit %.4g +- %.3g (%.3g x err bar %.3g): interval within %.3g of %s, gap/3 %.3g (gap %.3g)" % (
        v, half, ERR_K, ent["err_bar"], near + half, name, third, gap)


def _blank_rt(rec, blank, stack_fb=None):
    """(r_E, t_E) of the blank atom: a dict from tmm_blank_tierA or an FDTD blank result, else the analytic blank of the record's own stack."""
    if isinstance(blank, dict):
        if "refl" in blank:
            return blank["refl"][0]["r_E"], blank["trans"][0]["t_E"]
        return blank["r_E"], blank["t_E"]
    st = rec.get("stack") or stack_fb
    if not st:
        raise ValueError("no stack data for the analytic blank (result has no `stack`, no manifest given)")
    t = tmm_blank_tierA(st["h_lo_um"], st["gap_um"], st["h_up_um"], st["n_sub"], st["lam_um"])
    return t["r_E"], t["t_E"]


def _tierA_row(rec, blank, stack_fb=None):
    """The three ladder quantities of one Tier A rung against its blank. A rung that cannot be read (missing / non-finite refl[0], trans[0])
    gives NaN values: _rung_problems reports why, and extrapolate / _verdict refuse NaN."""
    rb, tb = _blank_rt(rec, blank, stack_fb)             # a missing stack / blank is a configuration error: it propagates
    try:
        r_e, t_e = rec["refl"][0]["r_E"], rec["trans"][0]["t_E"]
        row = dict(d_arg_r=wrap180(deg(r_e / rb)), d_arg_t=wrap180(deg(t_e / tb)), abs_r=abs(r_e))
    except (KeyError, TypeError, ZeroDivisionError, ValueError, OverflowError):
        return dict(d_arg_r=float("nan"), d_arg_t=float("nan"), abs_r=float("nan"), sig=None)
    try:                                   # disagreement of the two monitor planes, in the units of each quantity (phases: degrees)
        s_r, s_t = float(rec["refl"][0]["spread"]), float(rec["trans"][0]["spread"])
        row["sig"] = dict(d_arg_r=math.degrees(s_r / abs(r_e)), d_arg_t=math.degrees(s_t / abs(t_e)), abs_r=s_r)
    except (KeyError, TypeError, ZeroDivisionError, ValueError, OverflowError):
        row["sig"] = None                  # no spread in the record (hand-built / old result): the NOISE_FLOOR_REL floor alone applies
    return row


def _rung_problems(recs, ks, what, accept_overridden=False, notes=None):
    """Reasons a set of rung results cannot form one ladder (flagged runs, non-finite L3-order values, tolerances overridden by `analyse --qc`
    unless accept_overridden, mixed atoms / grids, missing quality gates). An accepted override is appended to `notes` (list) instead."""
    prob = []
    for k in ks:
        r = recs[k]
        if r.get("quality") is None:
            prob.append("%s k%d has no quality block (result made before the quality gates)" % (what, k))
        elif not _res_ok(r) or r.get("flags"):
            prob.append("%s k%d is flagged / not ok: %s" % (what, k, "; ".join(str(f)[:80] for f in (r.get("flags") or ["driver ok = false"]))))
        try:
            lad_vals = (r["refl"][0]["r_E"], r["trans"][0]["t_E"])
        except (KeyError, TypeError):
            prob.append("%s k%d has no refl[0] / trans[0] (the L3 order the ladder reads)" % (what, k))
        else:
            if not all(_fin(v) for v in lad_vals):
                prob.append("%s k%d: QC_NONFINITE r_E / t_E of the L3 order (%s, %s)" % (what, k, lad_vals[0], lad_vals[1]))
        ov_bad, ov_acc = _override_report((("%s k%d" % (what, k), r),), accept_overridden)
        prob += ["TOL_OVERRIDDEN: " + x for x in ov_bad]
        if ov_acc and notes is not None:
            notes.append("ACCEPTED overridden quality tolerances (--accept-overridden): %s" % "; ".join("%s %s" % (n, v) for n, v in ov_acc.items()))
    for key in ("atom", "dz_um"):
        vals = set()
        for k in ks:
            v = recs[k].get(key)
            vals.add(round(float(v), 9) if (key == "dz_um" and v is not None) else v)
        if len(vals) > 1:
            prob.append("%s rungs differ in %s: %s" % (what, key, sorted(str(v) for v in vals)))
    pnc = [int(recs[k].get("proxy_nc") or 0) for k in ks]
    if what == "real" and any(pnc):
        prob.append("real rungs must have proxy_nc 0 (got %s)" % pnc)
    if what == "proxy" and (not all(pnc) or len(set(pnc)) > 1):
        prob.append("proxy rungs must all carry the same non-zero proxy_nc (got %s)" % pnc)
    return prob


def ladder_tierA(atom_recs, blank, rcwa, ks=(1, 2, 3), proxy_recs=None, proxy_ref=None, stack_fb=None, accept_overridden=False):
    """Tier A k-ladder of one atom against Li / Laurent (fix758, skeptic FAIL #15).
    atom_recs {k: result of analyse()} real-atom rungs; blank 'tmm' (analytic blank of each record's own stack) | dict from tmm_blank_tierA |
    FDTD blank result (real rungs only); rcwa = rcwa_from_s1() output (Li / Laurent differentials of the atom);
    proxy_recs {k: result} PROXY rungs (coarse-24 structure) and proxy_ref = dict(li={q: limit}, li_unc={q: unc}) the known limit of that
    structure: the FDTD order is fitted on them and applied to the real rungs.
    Per quantity (d_arg_t primary, d_arg_r secondary, abs_r tertiary): rungs, calibration, extrapolated value, error bar terms, Li, Laurent, gap,
    verdict LI / LAURENT / UNRESOLVED + reason. `overall`: the primary decides; a resolved secondary that disagrees makes it UNRESOLVED;
    no proxy calibration supplied makes it UNRESOLVED (a verdict from the free fit alone is shown per quantity, marked UNCALIBRATED).
    Rungs judged with tolerances overridden by `analyse --qc`, or holding a non-finite L3-order value, are refused (UNRESOLVED with the reason);
    accept_overridden = True (CLI --accept-overridden) lets an overridden rung through and lists it in `notes`."""
    ks = tuple(sorted(int(k) for k in ks))
    out = dict(ks=list(ks), quantities={}, notes=[], calibrated=False)
    rung_prob = _rung_problems(atom_recs, ks, "real", accept_overridden, out["notes"])
    dev = sorted(set(str(atom_recs[k].get("fdtd_gpu")) for k in ks))
    if len(dev) > 1:
        out["notes"].append("rungs mix fdtd_gpu values %s: CPU and GPU runs are one series only if the acceptance showed CPU == GPU for this configuration" % dev)
    rows = {k: _tierA_row(atom_recs[k], blank, stack_fb) for k in ks}
    out["rung_values"] = {str(k): rows[k] for k in ks}
    calib = {q: None for q in LAD_QUANT}
    if proxy_recs:
        pks = tuple(sorted(int(k) for k in proxy_recs))
        prob_p = _rung_problems(proxy_recs, pks, "proxy", accept_overridden, out["notes"])
        prows = {k: _tierA_row(proxy_recs[k], "tmm", stack_fb) for k in pks}
        out["proxy_rung_values"] = {str(k): prows[k] for k in pks}
        for q in LAD_QUANT:
            if prob_p:
                calib[q] = dict(ok=False, note="proxy rungs refused: " + " | ".join(prob_p))
            else:
                calib[q] = calibrate_order(pks, [prows[k][q] for k in pks], proxy_ref["li"][q], q, unc=proxy_ref.get("li_unc", {}).get(q, 0.0))
        out["calibrated"] = bool(calib["d_arg_t"]["ok"] and calib["d_arg_r"]["ok"])      # the two phase quantities decide; abs r is tertiary
    else:
        out["notes"].append("no proxy rungs supplied: orders are the free three-rung fits (UNCALIBRATED), the overall verdict stays UNRESOLVED")
    for q in LAD_QUANT:
        vs = [rows[k][q] for k in ks]
        if rung_prob:
            ent = dict(ok=False, reason="real rungs refused: " + " | ".join(rung_prob), ks=list(ks), rungs=vs, err_bar=None, v_inf=None)
        else:
            sigs = [rows[k]["sig"][q] if rows[k].get("sig") else None for k in ks]
            ent = extrapolate(ks, vs, q, calib=(calib[q] if proxy_recs else None), sig=(sigs if all(x is not None for x in sigs) else None))
        ent.update(role=LAD_ROLE[q], li=rcwa["li"][q], laurent=rcwa["laurent"][q], calibration=calib[q])
        ent["verdict"], ent["reason_verdict"] = _verdict(q, ent)       # without a proxy the basis reads UNCALIBRATED
        out["quantities"][q] = ent
    prim, sec = out["quantities"]["d_arg_t"], out["quantities"]["d_arg_r"]
    if not proxy_recs:
        ov, why = "UNRESOLVED", "no proxy calibration supplied (primary d_arg_t would read %s from the free fit)" % prim["verdict"]
    elif prim["verdict"] == "UNRESOLVED":
        ov, why = "UNRESOLVED", "primary d_arg_t unresolved: %s" % prim["reason_verdict"]
    elif sec["verdict"] in ("LI", "LAURENT") and sec["verdict"] != prim["verdict"]:
        ov, why = "UNRESOLVED", "CONFLICT: d_arg_t says %s, d_arg_r says %s" % (prim["verdict"], sec["verdict"])
    else:
        ov = prim["verdict"]
        why = "d_arg_t (primary) %s; d_arg_r (secondary) %s" % (prim["verdict"], sec["verdict"] if sec["verdict"] != "UNRESOLVED" else "UNRESOLVED (%s)" % sec["reason_verdict"][:60])
    out["overall"] = dict(verdict=ov, reason=why)
    return out


# ------------------------------------------------------------------ synthetic RSoft files (selftest)
def write_rsoft_dft(path, cx, lx, ly):
    """Write a field like RSoft does: nx+1 x ny+1 samples including both edges, amp / phase(deg) pairs."""
    nx, ny = cx.shape
    xg = np.linspace(-lx / 2.0, lx / 2.0, nx + 1)
    yg = np.linspace(-ly / 2.0, ly / 2.0, ny + 1)
    full = np.empty((nx + 1, ny + 1), dtype=complex)
    full[:nx, :ny] = cx
    full[nx, :ny] = cx[0, :]
    full[:nx, ny] = cx[:, 0]
    full[nx, ny] = cx[0, 0]
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("/rn,a,b/nx0/ls1\n/r,qa,qb\n")
        fh.write("%d %.10f %.10f 0 OUTPUT_AMP_PHASE_3D Wavelength=1.55\n" % (nx + 1, xg[0], xg[-1]))
        fh.write("%d %.10f %.10f\n" % (ny + 1, yg[0], yg[-1]))
        for i in range(nx + 1):
            fh.write(" ".join("%.9E %.9E" % (abs(full[i, j]), math.degrees(cmath.phase(full[i, j]))) for j in range(ny + 1)) + "\n")


def _synth_field(lx, ly, nx, ny, comps, zs, kzs, kxs, sigma):
    """Sum of plane waves a_m exp(i sigma (kx x + s kz z)) sampled on an nx x ny periodic grid (cell nodes)."""
    xg = -lx / 2.0 + lx * np.arange(nx) / nx
    out = np.zeros((nx, ny), dtype=complex)
    for (a, kx, kz, z, s) in comps:
        ph = a * np.exp(1j * (kx * xg + s * kz * z))
        f = ph[:, None] * np.ones((1, ny))
        out += f.conjugate() if sigma < 0 else f
    return out


def _synth_manifest(tier, launch, m_inc, lx, ly, lam, n_sub, n_out, z_in_ref, top, zR1, zR2, zT1, zT2, orders_R, orders_T, m_t):
    return dict(schema=SCHEMA, tier=tier, launch=launch, lam_um=lam, n_sub=n_sub, n_out=n_out, lx_um=lx, ly_um=ly,
                m_inc=m_inc, m_t=m_t, orders_R=orders_R, orders_T=orders_T,
                z=dict(top=top, z_in_ref=z_in_ref, R1=zR1, R2=zR2, T1=zT1, T2=zT2),
                runs=dict(emp=dict(prefix="emp", lx_um=lx, ly_um=ly, mon=dict(R1=1, R2=2, T1=3, T2=4)),
                          st=dict(prefix="st", lx_um=lx, ly_um=ly, mon=dict(R1=1, R2=2, T1=3, T2=4))))


def _write_synth_case(wd, man, inc_amp, refl, trans, sigma, nx, ny):
    """refl: {m: r_E referred to z_in_ref}; trans: {m: t_E referred z_in_ref -> top}. Builds emp and st files."""
    lam, n_in, n_out, lx, ly = man["lam_um"], man["n_sub"], man["n_out"], man["lx_um"], man["ly_um"]
    k0 = 2.0 * math.pi / lam
    z = man["z"]
    m_inc = man["m_inc"]
    kx_i = 2.0 * math.pi * m_inc / lx
    kz_i = math.sqrt((k0 * n_in) ** 2 - kx_i ** 2)
    os.makedirs(wd, exist_ok=True)
    # incident a(z) = inc_amp * exp(i kz_i (z - z_in_ref)); reflected b_m(z) = r_m * inc_amp * exp(-i kz_m (z - z_in_ref))
    # transmitted c(z) = t * inc_amp * exp(i kz_t (z - top))  (t is referred z_in_ref -> top)
    for run in ("emp", "st"):
        for mn in ("R1", "R2", "T1", "T2"):
            comps = []
            zz = z[mn]
            if mn.startswith("R"):
                comps.append((inc_amp * cmath.exp(-1j * kz_i * z["z_in_ref"]), kx_i, kz_i, zz, +1))
                if run == "st":
                    for m, r in refl.items():
                        kx = 2.0 * math.pi * m / lx
                        kz = math.sqrt((k0 * n_in) ** 2 - kx ** 2)
                        # at plane zz: r_m(z_ref) exp(-i kz (zz - z_ref)) relative to the incident a(z_ref)
                        comps.append((inc_amp * r * cmath.exp(+1j * kz * z["z_in_ref"]), kx, kz, zz, -1))
                        # NB: incident term has amplitude a(z_ref) exp(i kz_i (zz - z_ref)); reflected b_m(zz) =
                        # r_m a(z_ref) exp(-i kz (zz - z_ref)); written as (amp exp(+i kz z_ref)) exp(-i kz zz)
                        # for the reflected and (amp exp(-i kz_i z_ref)) exp(+i kz_i zz) for the incident.
            else:
                if run == "st":
                    for m, t in trans.items():
                        kx = 2.0 * math.pi * m / lx
                        kz_t = math.sqrt((k0 * n_out) ** 2 - kx ** 2)
                        # c(zz) = t * a(z_ref) * exp(i kz_t (zz - top)); a(z_ref) = inc_amp
                        comps.append((inc_amp * t * cmath.exp(-1j * kz_t * z["top"]), kx, kz_t, zz, +1))
            f = _synth_field(lx, ly, nx, ny, comps, None, None, None, sigma) if comps else np.zeros((nx, ny), dtype=complex)
            for comp in ("ex", "ey"):
                fld = f if comp == "ex" else 1e-6 * f
                write_rsoft_dft(os.path.join(wd, "%s_m%d_f1_%s.dat" % (run, man["runs"][run]["mon"][mn], comp)), fld, lx, ly)


def _vname(q, ent):
    return _verdict(q, ent)[0]


def _selftest():
    import tempfile
    ok_all = True

    def chk(name, val, tol):
        nonlocal ok_all
        good = bool(abs(val) <= tol)
        ok_all = ok_all and good
        print("selftest %-62s %.3e (tol %.1e) %s" % (name, abs(val), tol, "ok" if good else "**FAIL**"))

    lam, n_in = 1.55, 1.444
    P = 1.31539
    k0 = 2.0 * math.pi / lam
    th = math.asin(lam / (P * n_in))
    rng = np.random.RandomState(7)
    tmp = tempfile.mkdtemp(prefix="fdtd_atom_post_selftest_", dir=(os.environ.get("FDTD_ATOM_SCRATCH") or None))   # set FDTD_ATOM_SCRATCH to keep the scratch out of the system temp
    lf_ok = dict(emp=dict(converged=True, nt_run=1), st=dict(converged=True, nt_run=1))
    # ---- (1) normal incidence, Tier A algebra, sigma +1 and -1
    for sigma in (1, -1):
        wd = os.path.join(tmp, "A_s%d" % sigma)
        ly = P / 3.0
        man = _synth_manifest("A", "normal", 0, ly, ly, lam, n_in, 1.0, 0.0, 0.925, -0.45, -0.25, 1.175, 1.375, [0], [0], 0)
        r_true = 0.14 * cmath.exp(1j * math.radians(37.0))
        t_true = 1.19 * cmath.exp(1j * math.radians(-101.0))
        inc_amp = 0.8 * cmath.exp(1j * 0.3)
        _write_synth_case(wd, man, inc_amp, {0: r_true}, {0: t_true}, sigma, 8, 8)
        orders = extract_orders(man, {"emp": wd, "st": wd})
        res = analyse(man, orders, logfacts=lf_ok)
        chk("A sigma %+d: sigma read" % sigma, orders["sigma"] - sigma, 0.0)
        chk("A sigma %+d: quality flags empty (good synthetic run)" % sigma, float(len(res["flags"])), 0.0)
        chk("A sigma %+d: r_E recovered (complex)" % sigma, res["refl"][0]["r_E"] - r_true, 1e-9)
        chk("A sigma %+d: t_E recovered (complex)" % sigma, res["trans"][0]["t_E"] - t_true, 1e-9)
        chk("A sigma %+d: absR00 = |r|" % sigma, res["record"]["absR00"] - abs(r_true), 1e-9)
        chk("A sigma %+d: argR00 = (180 + arg r) mod 360" % sigma,
            wrap180(res["record"]["argR00_deg"] - (math.degrees(cmath.phase(r_true)) + 180.0)), 1e-6)
        chk("A sigma %+d: T_atom power = |t|^2 / n_sub" % sigma, res["trans"][0]["T"] - abs(t_true) ** 2 / n_in, 1e-12)
        chk("A sigma %+d: |t_H| = |t_E| / n_sub" % sigma, res["trans"][0]["abs_tH"] - abs(t_true) / n_in, 1e-12)
    # ---- (2) tilted Tier B, both launches: power budget and phases through the reference planes
    ress = {}
    for launch, m_inc in (("pth", +1), ("mth", -1)):
        wd = os.path.join(tmp, "B_" + launch)
        z_in_ref, top = -0.5, 2.09
        man = _synth_manifest("B", launch, m_inc, P, P / 3.0, lam, n_in, 1.0, z_in_ref, top, -0.45, -0.25, top + 0.25, top + 0.45,
                              [-1, 0, 1], [0], 0)
        kx_i = 2.0 * math.pi * m_inc / P
        kz_i = math.sqrt((k0 * n_in) ** 2 - kx_i ** 2)
        cos_i = kz_i / (k0 * n_in)
        # prescribed POWER budget (sums to 1): specular, vertical, third order, air
        Rp = {m_inc: 0.36, 0: 0.19, -m_inc: 0.15}
        Tair = 0.30
        refl, truth = {}, {}
        for m, Rm in Rp.items():
            kx = 2.0 * math.pi * m / P
            kz = math.sqrt((k0 * n_in) ** 2 - kx ** 2)
            cos_m = kz / (k0 * n_in)
            amp = math.sqrt(Rm * cos_m / cos_i)
            refl[m] = amp * cmath.exp(1j * rng.uniform(-math.pi, math.pi))
        t_amp = math.sqrt(Tair * n_in / cos_i)          # T = |t_E|^2 n_t cos_i / (n_i cos_t), n_t = cos_t = 1 -> |t_E|^2 cos_i / n_i
        t_true = t_amp * cmath.exp(1j * rng.uniform(-math.pi, math.pi))
        _write_synth_case(wd, man, 0.7 * cmath.exp(0.2j), refl, {0: t_true}, +1, 24, 4)
        orders = extract_orders(man, {"emp": wd, "st": wd})
        res = analyse(man, orders, logfacts=lf_ok)
        ress[launch] = res
        chk("B %s: quality flags empty (good synthetic run)" % launch, float(len(res["flags"])), 0.0)
        chk("B %s: launched order found" % launch, orders["m_inc_found"] - m_inc, 0.0)
        for m, Rm in Rp.items():
            chk("B %s: R power of order %+d" % (launch, m), res["refl"][m]["R"] - Rm, 1e-9)
            chk("B %s: r_E order %+d complex" % (launch, m), res["refl"][m]["r_E"] - refl[m], 1e-9)
        chk("B %s: T power into air order 0" % launch, res["trans"][0]["T"] - Tair, 1e-9)
        chk("B %s: t_E complex through z_in_ref / top" % launch, res["trans"][0]["t_E"] - t_true, 1e-9)
        chk("B %s: energy R + T = 1" % launch, res["energy"] - 1.0, 1e-9)
        rm_h = res["refl"][m_inc]["r_H"]
        chk("B %s: r_H = -r_E (specular phase +180)" % launch, wrap180(deg(rm_h) - deg(refl[m_inc]) - 180.0), 1e-6)
        chk("B %s: t_H phase = t_E phase" % launch, wrap180(deg(res["trans"][0]["t_H"]) - deg(t_true)), 1e-6)
        if launch == "mth":
            chk("B mth: echo = |c_+1|^2 order R", res["record"]["echo_R_p2"] - Rp[-m_inc], 1e-9)
    # ---- (2b) a launch whose sign differs from the label must be refused, not analysed
    man_wrong = _synth_manifest("B", "pth", +1, P, P / 3.0, lam, n_in, 1.0, -0.5, 2.09, -0.45, -0.25, 2.34, 2.54, [-1, 0, 1], [0], 0)
    wd_mth = os.path.join(tmp, "B_mth")
    refused = False
    try:
        analyse(man_wrong, extract_orders(_synth_manifest("B", "mth", -1, P, P / 3.0, lam, n_in, 1.0, -0.5, 2.09, -0.45, -0.25, 2.34, 2.54, [-1, 0, 1], [0], 0),
                                          {"emp": wd_mth, "st": wd_mth}))
    except ValueError as e:
        refused = "LAUNCH_ORDER_MISMATCH" in str(e)
    chk("a -th launch labelled +th is refused (LAUNCH_ORDER_MISMATCH)", 0.0 if refused else 1.0, 0.0)
    # ---- (3) combine_B on the two synthetic launches
    pth, mth = ress["pth"], ress["mth"]
    cb = combine_B(pth, mth)
    chk("combine: B_deg = arg(rB tB) with rB, tB in the H basis",
        wrap180(cb["B_deg"] - deg(pth["refl"][1]["r_H"] * mth["trans"][0]["t_H"])), 1e-9)
    chk("combine: prod_amp = |tA| |rB tB|",
        cb["prod_amp"] - abs(pth["trans"][0]["t_H"]) * abs(pth["refl"][1]["r_H"] * mth["trans"][0]["t_H"]), 1e-12)
    chk("combine: A_deg = arg tA (H basis)", wrap180(cb["A_deg"] - deg(pth["trans"][0]["t_H"])), 1e-9)
    # ---- (4) amplitude check against the ledger identity T = 2.498273 |t_H|^2 of the record (n_sub / cos_i)
    chk("H-basis identity T = |t_H|^2 n_sub / cos_i (record: 2.498273)",
        pth["trans"][0]["T"] - abs(pth["trans"][0]["t_H"]) ** 2 * n_in / pth["cos_inc"], 1e-12)
    chk("n_sub / cos(th_sub) = 2.498273 (record constant)", n_in / math.cos(th) - 2.498273, 2e-6)
    # ---- (5) ladder fit on synthetic data with a known order, and the plan's two-rung p = 1 numbers
    for p_true in (0.7, 1.0, 2.0):
        hs = [1.0, 0.5, 1.0 / 3.0]
        vs = [-33.4 + 20.0 * h ** p_true for h in hs]
        f = fit_order(hs, vs)
        chk("ladder fit: p recovered (true %.2f)" % p_true, f["p"] - p_true, 1e-6)
        chk("ladder fit: v_inf recovered (true %.2f)" % p_true, f["v_inf"] + 33.4, 1e-6)
    chk("Richardson p=1 on the plan's S4 proxy rungs (d arg r -53.5656 / -45.2565 -> -36.96)",
        richardson(1.0, -53.5656, 0.5, -45.2565, 1.0) + 36.9474, 0.02)
    chk("Richardson p=1 on the plan's S4 proxy rungs (d arg t -105.3374 / -104.2 -> -103.06)",
        richardson(1.0, -105.3374, 0.5, -104.2, 1.0) + 103.0626, 0.02)
    # ---- (6) the quality gates are alive: an unknown / unconverged run and a broken power budget must flag
    wd = os.path.join(tmp, "A_s1")
    manA = _synth_manifest("A", "normal", 0, P / 3.0, P / 3.0, lam, n_in, 1.0, 0.0, 0.925, -0.45, -0.25, 1.175, 1.375, [0], [0], 0)
    ordA = extract_orders(manA, {"emp": wd, "st": wd})
    f0 = analyse(manA, ordA)["flags"]
    chk("quality: no log facts -> QC_CONVERGENCE_UNKNOWN for both runs", 0.0 if sum(f.startswith("QC_CONVERGENCE_UNKNOWN") for f in f0) == 2 else 1.0, 0.0)
    f1 = analyse(manA, ordA, logfacts=dict(emp=dict(converged=True), st=dict(converged=False, nt_run=9, stop_time_um=1.0)))["flags"]
    chk("quality: st run not converged -> QC_NOT_CONVERGED:st", 0.0 if any(f.startswith("QC_NOT_CONVERGED:st") for f in f1) else 1.0, 0.0)
    ordE = json.loads(json.dumps(_san(ordA)))
    for mn in ("T1", "T2"):
        for comp in ("ex",):
            ordE["st"][mn][comp]["0"] = [1.02 * ordE["st"][mn][comp]["0"][0], 1.02 * ordE["st"][mn][comp]["0"][1]]
    f2 = analyse(manA, ordE, logfacts=lf_ok)["flags"]
    chk("quality: transmitted order +2 percent -> QC_ENERGY", 0.0 if any(f.startswith("QC_ENERGY") for f in f2) else 1.0, 0.0)
    # ---- (6b) tier B: tolerances differ, the specular reflection is NOT gated (rB is not judged) but the echo order and the transmitted order are
    manB = _synth_manifest("B", "mth", -1, P, P / 3.0, lam, n_in, 1.0, -0.5, 2.09, -0.45, -0.25, 2.34, 2.54, [-1, 0, 1], [0], 0)
    wdB = os.path.join(tmp, "B_mth")
    ordB = extract_orders(manB, {"emp": wdB, "st": wdB})
    chk("quality tier B: good synthetic run, no flags", float(len(analyse(manB, ordB, logfacts=lf_ok)["flags"])), 0.0)

    def _pert(mon, order, factor=1.0, dphi_deg=0.0, comp="ex"):
        o = json.loads(json.dumps(_san(ordB)))
        v = _c(o["st"][mon][comp][str(order)]) * factor * cmath.exp(1j * math.radians(dphi_deg))
        o["st"][mon][comp][str(order)] = [v.real, v.imag]
        return o

    fB = analyse(manB, _pert("R2", -1, dphi_deg=4.0), logfacts=lf_ok)["flags"]            # order -1 = the specular order of the -th launch
    chk("quality tier B: specular reflection pair disagreeing by 4 deg is NOT gated (rB not judged)", float(sum(f.startswith("QC_SPREAD") for f in fB)), 0.0)
    fB = analyse(manB, _pert("R2", +1, factor=1.05), logfacts=lf_ok)["flags"]             # order +1 = the echo order of the -th launch
    chk("quality tier B: echo order pair disagreeing by 5 percent -> QC_SPREAD", 0.0 if any(f.startswith("QC_SPREAD") for f in fB) else 1.0, 0.0)
    fB = analyse(manB, _pert("T1", 0, factor=1.06), logfacts=lf_ok)["flags"]
    chk("quality tier B: transmitted order +6 percent -> QC_ENERGY", 0.0 if any(f.startswith("QC_ENERGY") for f in fB) else 1.0, 0.0)
    rov = analyse(manB, ordB, logfacts=lf_ok, qc={"energy": 1e-12})
    chk("quality: an overridden tolerance is applied (flag) and recorded (tol_overridden)",
        0.0 if (rov["quality"]["tol_overridden"] == ["energy"] and any(f.startswith("QC_ENERGY") for f in rov["flags"])) else 1.0, 0.0)
    # ---- (7) ladder tool on a synthetic proxy + real ladder with a known limit (full battery: tests\test_ladder_proxy.py)
    pro_k = (1, 2, 3, 4, 6)
    cal = calibrate_order(pro_k, [-33.4 - 20.0 * (1.0 / k) ** 0.73 for k in pro_k], -33.4, "d_arg_r", unc=0.0)
    chk("proxy calibration: fitted order (true 0.73)", cal["p"] - 0.73, 1e-9)
    ex = extrapolate((1, 2, 3), [-33.4 - 7.0 * (1.0 / k) ** 0.73 for k in (1, 2, 3)], "d_arg_r", calib=cal)
    chk("real ladder with the calibrated order: v_inf recovered (-33.4)", ex["v_inf"] + 33.4, 1e-9)
    chk("real ladder: error bar = |k3 - v_inf| (plan definition) for an exact power law", ex["err_bar"] - 7.0 * (1.0 / 3.0) ** 0.73, 1e-9)
    # ---- (8) fix2 758: non-finite data, overridden tolerances, the interval verdict rule (full batteries: tests/test_nonfinite_needles.py, test_override_needles.py)
    o_nan = json.loads(json.dumps(_san(ordB)))
    o_nan["st"]["T1"]["ex"]["-1"] = [float("nan"), float("nan")]
    f_nan = analyse(manB, o_nan, logfacts=lf_ok)
    chk("non-finite: a NaN transmitted order -> QC_NONFINITE, quality_ok False", 0.0 if (any(f.startswith("QC_NONFINITE") for f in f_nan["flags"]) and not f_nan["quality_ok"]) else 1.0, 0.0)
    ent = dict(ok=True, v_inf=float("nan"), err_bar=0.1, E_dist=0.1, E_hind=0.0, E_p=0.0, E_free=0.0, E_pair=0.0, E_noise=0.0, li=-109.4, laurent=-104.8, reason="")
    chk("non-finite: _verdict with a NaN limit -> UNRESOLVED (was LAURENT)", 0.0 if _vname("d_arg_t", ent) == "UNRESOLVED" else 1.0, 0.0)
    chk("override: --qc energy=1e-12 is listed in overridden_keys, the default result is not",
        0.0 if (overridden_keys(rov) == ["energy"] and overridden_keys(analyse(manB, ordB, logfacts=lf_ok)) == []) else 1.0, 0.0)
    gap = 4.59
    third = gap / 3.0
    for d_li, expect in ((third - 2.5 * 0.1 - 1e-6, "LI"), (third - 2.5 * 0.1 + 1e-3, "UNRESOLVED")):
        ent = dict(ok=True, v_inf=-109.4 + d_li, err_bar=0.1, E_dist=0.1, E_hind=0.0, E_p=0.0, E_free=0.0, E_pair=0.0, E_noise=0.0, li=-109.4, laurent=-109.4 + gap, reason="")
        chk("verdict rule: |v - Li| + K err = gap/3 %+.0e -> %s (K %.2f)" % (d_li + 2.5 * 0.1 - third, expect, ERR_K), 0.0 if (ERR_K == 2.5 and _vname("d_arg_t", ent) == expect) else 1.0, 0.0)
    print("selftest %s" % ("ALL OK" if ok_all else "**FAILED**"))
    print("selftest scratch (left in place, synthetic only; RSoft-style files carry 9 significant digits, so tolerances are 1e-8 / 1e-6 deg): %s" % tmp)
    return 0 if ok_all else 1


# ------------------------------------------------------------------ CLI
def _cmd_extract(a):
    man = load_json(a.manifest)
    base = os.path.dirname(os.path.abspath(a.manifest))
    dirs = {"emp": a.emp or man["runs"]["emp"].get("dir") or base, "st": a.st or man["runs"]["st"].get("dir") or base}
    out = extract_orders(man, dirs)
    save_json(a.out, out)
    print("extract: sigma %+d, launched order %d, wrote %s" % (out["sigma"], out["m_inc_found"], a.out))
    return 0


def _decode_result(d):
    """json round trip turns int keys into strings and complex numbers into [re, im]: undo it for a saved result."""
    if d is None:
        return None
    d = dict(d)
    for part in ("refl", "trans"):
        d[part] = {int(k): {kk: (_c(vv) if kk.startswith(("r_", "t_")) else vv) for kk, vv in v.items()} for k, v in d[part].items()}
    return d


def _cmd_analyse(a):
    man = load_json(a.manifest)
    lf = logfacts_from_manifest(man)
    if a.result:                                          # the driver's own result json already carries the log facts of both runs
        rj = load_json(a.result)
        for run in ("emp", "st"):
            f = (rj.get("runs", {}).get(run) or {}).get("logfacts")
            if f:
                lf[run] = f
    if a.emp_log:
        lf["emp"] = parse_log(a.emp_log)
    if a.st_log:
        lf["st"] = parse_log(a.st_log)
    qc = {}
    for kv in (a.qc or []):
        k, _, v = kv.partition("=")
        qc[k.strip()] = float(v)
    res = analyse(man, load_json(a.orders), logfacts=lf, qc=(qc or None))
    save_json(a.out, res)
    print("analyse: %s" % json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in res["record"].items()
                                      if not isinstance(v, str)}, sort_keys=True))
    print("energy R+T = %.6f  quality_ok %s  tolerances %s%s" % (res["energy"], res["quality_ok"], res["quality"]["tol"],
                                                                 ("  OVERRIDDEN %s" % res["quality"]["tol_overridden"]) if res["quality"]["tol_overridden"] else ""))
    for f in res["flags"]:
        print("FLAG %s" % f)
    if res["tol_overridden"]:
        print("NOTE tolerances %s differ from the shipped QC_DEFAULT: `ladder` / `combine` refuse this result unless run with --accept-overridden" % res["tol_overridden"])
    return 0


def _cmd_combine(a):
    def _L(p):
        return load_json(p) if p else None

    pth, mth = _decode_result(_L(a.pth)), _decode_result(_L(a.mth))
    bp, bm = _decode_result(_L(a.blank_pth)), _decode_result(_L(a.blank_mth))
    out = combine_B(pth, mth, bp, bm, accept_overridden=a.accept_overridden)
    save_json(a.out, out)
    print("combine: A %.3f argtB %.3f B %.3f echo %.3f dB eff %.2f pct  ok %s  status %s" % (out["A_deg"], out["argtB_deg"], out["B_deg"],
                                                                                          out.get("echo_p2_db", float("nan")), out["carrier_eff_pct"],
                                                                                          out["ok"], out["status"]))
    for why in out["status_reasons"]:
        print("UNRESOLVED because: %s" % why)
    for k, v in out["judged"].items():
        print("judged %-16s %s" % (k, v))
    for f in out["flags"] + out.get("blank_flags", []):
        print("FLAG %s" % f)
    return 0


def _cmd_ladder(a):
    def _load_rungs(arg, what):
        recs = {}
        for p in [p for p in (arg or "").split(",") if p]:
            d = _decode_result(load_json(p))
            k = int(d["k"])
            if k in recs:
                raise ValueError("two %s results with k = %d" % (what, k))
            recs[k] = d
        return recs

    recs = _load_rungs(a.atoms, "real-atom")
    precs = _load_rungs(a.proxy_atoms, "proxy")
    stack_fb = None
    if a.manifest:
        man = load_json(a.manifest)
        stack_fb = dict(h_lo_um=man["h_lo_um"], gap_um=man["gap_um"], h_up_um=man["h_up_um"], n_sub=man["n_sub"], lam_um=man["lam_um"])
    blank = "tmm" if a.blank == "tmm" else _decode_result(load_json(a.blank))
    pref = None
    if precs:
        if not a.proxy_ref:
            raise ValueError("--proxy-atoms needs --proxy-ref (the known Li limit of the proxy structure)")
        pref = load_json(a.proxy_ref)
    rcwa = rcwa_from_s1(a.rcwa, a.atom)
    out = ladder_tierA(recs, blank, rcwa, ks=tuple(sorted(recs)), proxy_recs=(precs or None), proxy_ref=pref, stack_fb=stack_fb,
                       accept_overridden=a.accept_overridden)
    if a.out:
        save_json(a.out, out)
    print("ladder %s: real rungs k = %s, proxy rungs k = %s, calibrated %s" % (a.atom, out["ks"], sorted(precs), out["calibrated"]))
    for n in out["notes"]:
        print("NOTE %s" % n)
    for q in LAD_QUANT:
        e = out["quantities"][q]
        c = e.get("calibration")
        print("%-8s (%s) rungs %s" % (q, e["role"], ["%.4f" % v for v in e["rungs"]]))
        if c is not None:
            print("          proxy order: %s" % (("p %.3f band %.3f (pairs %s)" % (c["p"], c["p_band"], ["%.3f" % x for x in c["pairs"]])) if c.get("ok") else "FAILED: %s" % c.get("note")))
        if e["ok"]:
            print("          %s -> v_inf %.4f, err bar %.3f (E_dist %.3f E_hind %.3f E_p %.3f E_free %.3f E_pair %.3f E_noise %.3f) x K %.2f = +- %.3f   Li %.4f  Laurent %.4f  gap %.4f  gap/3 %.4f"
                  % (e["basis"], e["v_inf"], e["err_bar"], e["E_dist"], e["E_hind"], e["E_p"], e["E_free"], e["E_pair"], e["E_noise"], ERR_K, ERR_K * e["err_bar"],
                     e["li"], e["laurent"], e["gap"], e["budget"]))
        print("          VERDICT %s -- %s" % (e["verdict"], e["reason_verdict"]))
    print("OVERALL %s -- %s" % (out["overall"]["verdict"], out["overall"]["reason"]))
    return 0


def _cmd_tmm(a):
    man = load_json(a.manifest)
    t = tmm_blank_tierA(man["h_lo_um"], man["gap_um"], man["h_up_um"], man["n_sub"], man["lam_um"])
    print("blank TMM: r_E %.6f / %+.4f deg  t_E %.6f / %+.4f deg  argR00 (H basis) %.4f"
          % (abs(t["r_E"]), deg(t["r_E"]), abs(t["t_E"]), deg(t["t_E"]), (deg(t["r_E"]) + 180.0) % 360.0))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("extract")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--emp", default="")
    p.add_argument("--st", default="")
    p = sub.add_parser("analyse")
    p.add_argument("--manifest", required=True)
    p.add_argument("--orders", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--result", default="", help="the driver's fdtd_atom_result.json (its runs.*.logfacts feed the convergence gate)")
    p.add_argument("--emp-log", default="")
    p.add_argument("--st-log", default="")
    p.add_argument("--qc", nargs="*", default=[], help="tolerance overrides key=value (%s); recorded in quality.tol_overridden" % ", ".join(QC_KEYS))
    p = sub.add_parser("combine")
    p.add_argument("--pth", required=True)
    p.add_argument("--mth", required=True)
    p.add_argument("--blank-pth", default="")
    p.add_argument("--blank-mth", default="")
    p.add_argument("--accept-overridden", action="store_true", help="use results judged with tolerances overridden by `analyse --qc` (refused / UNRESOLVED otherwise)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("ladder")
    p.add_argument("--atoms", required=True)
    p.add_argument("--blank", required=True)
    p.add_argument("--manifest", default="")
    p.add_argument("--rcwa", required=True)
    p.add_argument("--atom", required=True)
    p.add_argument("--accept-overridden", action="store_true", help="use rungs judged with tolerances overridden by `analyse --qc` (refused / UNRESOLVED otherwise)")
    p.add_argument("--proxy-atoms", default="", help="comma list of proxy-structure result jsons (k1 acceptance, k3 / k4 / k6 proxy jobs, ...)")
    p.add_argument("--proxy-ref", default="", help="json with the proxy's known Li limit: {li: {d_arg_t, d_arg_r, abs_r}, li_unc: {...}}")
    p.add_argument("--out", default="")
    p = sub.add_parser("tmm")
    p.add_argument("--manifest", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return _selftest()
    return dict(extract=_cmd_extract, analyse=_cmd_analyse, combine=_cmd_combine, ladder=_cmd_ladder, tmm=_cmd_tmm)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
