# -*- coding: ascii -*-
r"""fdtd_carrier2d_post.py -- post-processing of the 2-D carrier FDTD check (Main_fdtd_carrier2d_verify.py).   261016, review1 PASS + FIX1 261016 (empty-run stability gate, load_result).

PLAN: <WORK_ROOT>\fdtd_carrier_261016\PLAN.md items 6-7 and the QC list. Standalone: numpy + fdtd_atom_post (imported, not copied).

Reused from fdtd_atom_post (import P): wrap180, deg, _kz, tmm_h, parse_log, save_json, _fin, nonfinite_paths, quality (QC gates),
analyse (the R / T bookkeeping: empty-run subtraction ONLY for the launched order, Ex-basis powers P ~ n |Ex|^2 / cos, planes R1 / R2 / T1 / T2),
because the 1-D order coefficients are stored in the same orders-json layout. New here: 1-D line-monitor reader and projection,
the two-plane forward / backward solve in the spacer (S1, S2), the spacer flux check, energy closure R_go, reciprocity, the analytic
C0 / C1 references and the needle (deliberately wrong convention) used to prove that C1 can fail.

Conventions (identical to fdtd_atom_post): c_m = <Ex exp(-i sigma m G x)>, sigma read from the empty run (phase advance R1 -> R2),
m_abs = m_inc + m_rec, +theta launch m_inc = +1, -theta m_inc = -1. Ex-basis: B/A (Ex) = - B/A (Hy).
"""
import cmath
import io
import math
import os
import re
import sys

import numpy as np

for _p in (os.path.dirname(os.path.abspath(__file__)),):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import fdtd_atom_post as P                                    # noqa: E402

MONS = ("R1", "R2", "T1", "T2", "S1", "S2")
ORD_RANGE = range(-3, 4)
# quality tolerances of this check (PLAN section 3: tightened from the Tier B defaults of fdtd_atom_post.QC_DEFAULT["B"])
QC_C2D = dict(energy=2e-3, spread_rel=5e-3, ratio=5e-3, phase_adv_deg=0.5, cross_pol=1e-4)
SPACER_FLUX_TOL = 5e-3
TM_SIG_TOL = 1e-3          # MEASURED C0 k1: 1.6e-4 (3-D tool: 1.4e-3)
RECIP_TOL = 3e-3
EMP_STAB_TOL = 1e-3        # relative empty-run incident-phasor difference between two run lengths (MEASURED k1 90 vs 150 periods: 3.3e-4)


# ------------------------------------------------------------------ FullWAVE log of a 2-D run
def parse_log2d(path, index=-1):
    """log.txt -> facts of ONE run block. The 2-D log prints 'NX=.. NZ=.., NT=..' (no NY), which fdtd_atom_post.parse_log does not match;
    the time-step, convergence and computation-time lines are the same. converged is True only for FullWAVE's own 'Converged at L=' line."""
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
    m = re.search(r"NX=(\d+) NZ=(\d+), NT=(\d+)", b)
    if m:
        info.update(nx=int(m.group(1)), nz=int(m.group(2)), nt=int(m.group(3)))
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
    m = re.search(r"Actual Computation Time = ([0-9.]+) min", b)
    if m:
        info["compute_min"] = float(m.group(1))
    info["finished"] = "Actual Computation Time" in b
    return info


# ------------------------------------------------------------------ 1-D line monitor files
def load_line(path):
    """RSoft 1-D complex amp / phase(deg) monitor file -> (cx[nx], xg[nx], hdr). Header: a line 'N xmin xmax ...' (first line whose
    first token is an integer and has >= 3 numeric tokens), then N data lines of 'amp phase'. Non-parsable tokens are read as NaN."""
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    hi = None
    for i, ln in enumerate(lines[:12]):
        t = ln.split()
        if len(t) >= 3 and t[0].lstrip("+-").isdigit():
            try:
                float(t[1])
                float(t[2])
            except ValueError:
                continue
            hi = i
            break
    if hi is None:
        raise ValueError("no header line in %s" % path)
    h = lines[hi].split()
    n, x0, x1 = int(h[0]), float(h[1]), float(h[2])
    rows = []
    for ln in lines[hi + 1:]:
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
        raise ValueError("ragged monitor file %s" % path)
    a = np.array(rows, dtype=float)
    if a.shape[1] == 1:                                         # one number per line: amp and phase on consecutive lines
        a = a.reshape(-1, 2)
    if a.shape[0] != n and a.size == 2 * n:
        a = a.reshape(n, 2)
    with np.errstate(invalid="ignore", over="ignore"):
        cx = a[:, 0] * np.exp(1j * np.radians(a[:, 1]))
    return cx, np.linspace(x0, x1, n), dict(n=n, x0=x0, x1=x1, header=lines[:hi + 1])


def crop_period(cx, xg, lx):
    dx = (xg[-1] - xg[0]) / max(len(xg) - 1, 1)
    if abs((xg[-1] - xg[0]) - lx) < 0.25 * dx:
        return cx[:-1], xg[:-1]
    return cx, xg


def order1d(cx, xg, m, lx, sigma):
    """c_m = < E exp(-i sigma m G x) > over the (cropped) period."""
    ph = np.exp(-1j * sigma * (2.0 * math.pi * m / lx) * xg)
    with np.errstate(invalid="ignore", over="ignore"):
        return complex(np.mean(cx * ph))


def extract_orders(man, dirs, runs=("emp", "st")):
    """Monitor files of the empty and the structure run -> orders json (same layout as fdtd_atom_post.extract_orders, plus S1 / S2,
    plus the whole order spectrum -3..3 on every monitor so that a needle can use any order). Components ex, ey (and ez when present)."""
    lam = man["lam_um"]
    k0 = 2.0 * math.pi / lam
    lx = man["lx_um"]
    out = dict(schema="fdtd_carrier2d_orders_v1", emp={}, st={}, nonfinite_samples={}, files={})
    fields, bad = {}, {}
    for run in runs:
        R = man["runs"][run]
        for mn, mon in R["mon"].items():
            for comp in ("ex", "ey", "ez"):
                f = P.find_dft(dirs[run], R["prefix"], int(mon), comp)
                if f is None:
                    if comp in ("ez", "ey"):                    # TM in the 2-D tool has no Ey file (polarization 1: ex / ez / hy); ey is then zero by construction
                        continue
                    raise FileNotFoundError("NOT FOUND: %s_m%d_f*_%s.dat in %s" % (R["prefix"], int(mon), comp, dirs[run]))
                cx, xg, hdr = load_line(f)
                nb = int(cx.size - np.count_nonzero(np.isfinite(cx)))
                if nb:
                    bad["%s/%s/%s" % (run, mn, comp)] = nb
                fields[(run, mn, comp)] = crop_period(cx, xg, lx)
                out["files"]["%s/%s/%s" % (run, mn, comp)] = os.path.basename(f)
    cx1, xg1 = fields[("emp", "R1", "ex")]
    cx2, xg2 = fields[("emp", "R2", "ex")]
    spec1 = {m: abs(order1d(cx1, xg1, m, lx, 1)) for m in ORD_RANGE}
    m_s1 = max(spec1, key=spec1.get)
    a1, a2 = order1d(cx1, xg1, m_s1, lx, 1), order1d(cx2, xg2, m_s1, lx, 1)
    adv = cmath.phase(a2 / a1)
    sigma = 1 if adv > 0 else -1
    spec = {m: abs(order1d(cx1, xg1, m, lx, sigma)) for m in ORD_RANGE}
    m_found = max(spec, key=spec.get)
    out.update(nonfinite_samples=bad, sigma=sigma, m_inc_found=int(m_found), spectrum_R1={str(m): v for m, v in spec.items()},
               phase_adv_deg=math.degrees(adv), abs_ratio_R2_R1=abs(a2 / a1))
    dzR = man["z"]["R2"] - man["z"]["R1"]
    kx_f = 2.0 * math.pi * m_found / lx
    q2 = (k0 * man["n_sub"]) ** 2 - kx_f * kx_f
    out["phase_adv_expect_deg"] = math.degrees(sigma * math.sqrt(max(q2, 0.0)) * dzR)
    for run in runs:
        out[run] = {}
        for mn in man["runs"][run]["mon"]:
            ent = {}
            for comp in ("ex", "ey", "ez"):
                if (run, mn, comp) not in fields:
                    if comp == "ey":
                        ent[comp] = {str(m): 0j for m in ORD_RANGE}
                        out.setdefault("ey_absent", True)
                    continue
                cx, xg = fields[(run, mn, comp)]
                ent[comp] = {str(m): order1d(cx, xg, m, lx, sigma) for m in ORD_RANGE}
            out[run][mn] = ent
    return out


def emp_stability(orders, m_inc, run_a="emp", run_b="emp_chk"):
    """Relative difference |c_a - c_b| / |c_a| of the launched-order Ex phasor of two EMPTY runs of different length (emp_periods and
    emp_check_periods), per monitor; max_rel is the number the convergence waiver is gated on."""
    per = {}
    for mn in MONS:
        try:
            ca = P._c(orders[run_a][mn]["ex"][str(int(m_inc))])
            cb = P._c(orders[run_b][mn]["ex"][str(int(m_inc))])
        except KeyError:
            continue
        per[mn] = abs(ca - cb) / abs(ca) if abs(ca) > 0 else float("nan")
    ok = bool(per) and all(P._fin(v) for v in per.values())
    return dict(per_monitor=per, max_rel=(max(per.values()) if ok else float("nan")), tol=EMP_STAB_TOL, m_inc=int(m_inc))


_CPLX_KEYS = frozenset(("r_E", "r_H", "r_E_1", "r_E_2", "t_E", "t_H", "t_E_1", "t_E_2", "A", "B", "B_over_A"))


def _restore(o, key=None):
    if isinstance(o, dict):
        return {k: _restore(v, k) for k, v in o.items()}
    if isinstance(o, list):
        if key in _CPLX_KEYS and len(o) == 2 and all(isinstance(x, (int, float)) for x in o):
            return complex(o[0], o[1])
        return [_restore(v) for v in o]
    if key in _CPLX_KEYS and o is None:
        return complex("nan")
    return o


def load_result(path):
    """fdtd2d_result.json (save_json: int keys -> strings, complex -> [re, im]) -> the in-memory layout that combine() expects:
    int order keys in refl / trans / spacer.orders, complex values for r_E .. B_over_A."""
    import json
    with io.open(path, encoding="utf-8") as fh:
        d = _restore(json.load(fh))
    for blk in ("refl", "trans"):
        if isinstance(d.get(blk), dict):
            d[blk] = {int(m): v for m, v in d[blk].items()}
    if isinstance(d.get("spacer"), dict) and isinstance(d["spacer"].get("orders"), dict):
        d["spacer"]["orders"] = {int(m): v for m, v in d["spacer"]["orders"].items()}
    return d


# ------------------------------------------------------------------ analysis
def _cv(sigma):
    return (lambda c: c) if sigma > 0 else (lambda c: c.conjugate())


def _orders_c(d, key):
    return P._c(d[str(int(key))])


def spacer_solve(man, orders, m, sigma=None, run="st"):
    """Two-plane solve of Ex(z) = A exp(i kz (z - S1)) + B exp(-i kz (z - S1)) for x-order m (absolute). Returns dict(A, B, kz, cos_m) or None
    when the order is evanescent in the spacer. A, B are Ex amplitudes at the S1 plane (incident-phasor units of the EMPTY run at R1)."""
    sg = int(orders["sigma"]) if sigma is None else int(sigma)
    cv = _cv(sg)
    k0 = 2.0 * math.pi / man["lam_um"]
    kx = 2.0 * math.pi * m / man["lx_um"]
    kz = P._kz(k0 * man["n_sub"], kx)
    if isinstance(kz, complex):
        return None
    c1 = cv(_orders_c(orders[run]["S1"]["ex"], m))
    c2 = cv(_orders_c(orders[run]["S2"]["ex"], m))
    phi = kz * (man["z"]["S2"] - man["z"]["S1"])
    A = (c2 - c1 * cmath.exp(-1j * phi)) / (cmath.exp(1j * phi) - cmath.exp(-1j * phi))
    B = c1 - A
    return dict(A=A, B=B, kz=kz, cos_m=kz / (k0 * man["n_sub"]), kx=kx)


def spacer_report(man, orders, res):
    """Per propagating order (absolute m): forward / backward Ex amplitudes, power fractions of the incident power, B/A, and the share
    used for the purity. purity = forward share of absolute order 0 (the design channel of both launches)."""
    m_inc = int(man["m_inc"])
    sg = int(orders["sigma"])
    cv = _cv(sg)
    absinc = abs(cv(_orders_c(orders["emp"]["R1"]["ex"], m_inc)))        # incident amplitude (lossless: equal at R1 and R2, QC ratio gate)
    rep, fsum, net = {}, 0.0, 0.0
    for m in range(-3, 4):
        s = spacer_solve(man, orders, m)
        if s is None:
            continue
        F = abs(s["A"]) ** 2 * res["cos_inc"] / s["cos_m"] / absinc ** 2
        Bk = abs(s["B"]) ** 2 * res["cos_inc"] / s["cos_m"] / absinc ** 2
        rep[m] = dict(A=s["A"], B=s["B"], B_over_A=(s["B"] / s["A"] if abs(s["A"]) > 0 else complex("nan")), fwd=F, bwd=Bk, cos_m=s["cos_m"])
        fsum += F
        net += F - Bk
    shares = {m: v["fwd"] / fsum for m, v in rep.items()} if fsum > 0 else {}
    for m in rep:
        rep[m]["share"] = shares.get(m, float("nan"))
    purity = shares.get(0, float("nan"))
    return dict(orders=rep, fwd_total=fsum, net_flux=net, purity=purity, one_minus_purity=1.0 - purity, abs_inc=absinc)


def analyse(man, orders, logfacts=None):
    """fdtd_atom_post.analyse (R / T bookkeeping + QC gates, tolerances from man["qc"]) + spacer solve + closure + 2-D checks."""
    pm = dict(man)
    pm.pop("checks", None)                                       # the 3-D cell-count gate (cells_expected ny) does not apply; 2-D cells are checked here
    pm["tier"] = "B"
    pm["atom"] = man.get("term_name")
    res = P.analyse(pm, orders, logfacts=logfacts)
    flags = res["flags"]
    # ---- cell count of the two runs against the cells the driver built (log NX, NZ)
    ce = man["checks"]["cells_expected"]
    res["cells_check"] = {}
    nz_want = ce["nz"] + 2 * int(man["pml_cells"]) + 1
    for run, key in (("st", "nx"), ("emp", "nx_empty")):
        f = (logfacts or {}).get(run) or {}
        if "nx" in f:
            good = bool(f["nx"] == ce[key] + 3 and f.get("nz") == nz_want)
            res["cells_check"][run] = dict(log_nx=f["nx"], log_nz=f.get("nz"), built_nx=ce[key], built_nz=nz_want, ok=good)
            if not good:
                flags.append("QC_CELLS:%s log NX,NZ %s,%s, built %d,%d (+3 ghost in x; z incl. both PML + 1)" % (run, f["nx"], f.get("nz"), ce[key] + 3, nz_want))
    # ---- Waiver (tierB_accept.md) + PLAN: the EMPTY run need not converge when the structure run converged AND the empty
    # incident phasor is stable. The stability number (relative phasor difference between two empty-run lengths, emp_stability())
    # is stored in the result; without that number, or with it >= EMP_STAB_TOL, QC_NOT_CONVERGED:emp stays a flag.
    st_conv = bool(((logfacts or {}).get("st") or {}).get("converged"))
    stab = orders.get("emp_stability")
    res["emp_stability"] = stab
    res["waived"] = []
    stab_ok = bool(stab) and P._fin(stab.get("max_rel")) and stab["max_rel"] < EMP_STAB_TOL
    for fl in list(flags):
        if fl.startswith("QC_NOT_CONVERGED:emp"):
            if st_conv and stab_ok:
                flags.remove(fl)
                res["waived"].append(fl)
            elif st_conv:
                why = "no stability number" if not stab else "emp phasor stability %.2e >= %.1e" % (stab.get("max_rel", float("nan")), EMP_STAB_TOL)
                flags[flags.index(fl)] = fl + " [waiver refused: %s]" % why
    # ---- TM signature of the EMPTY launch: |Ex| / sqrt(|Ex|^2 + |Ez|^2) = cos(theta) (the 2-D tool's polarization key is verified from this, not from the key)
    e1 = orders["emp"]["R1"]
    if "ez" in e1:
        mi = str(int(man["m_inc"]))
        ex_, ez_ = abs(P._c(e1["ex"][mi])), abs(P._c(e1["ez"][mi]))
        tm = ex_ / math.sqrt(ex_ * ex_ + ez_ * ez_)
        dtm = abs(tm - math.cos(math.radians(man["theta_deg"])))
        res["tm_signature"] = dict(Ex_over_Eplane=tm, cos_theta=math.cos(math.radians(man["theta_deg"])), diff=dtm, tol=TM_SIG_TOL, ok=bool(P._within(dtm, TM_SIG_TOL)))
        if not res["tm_signature"]["ok"]:
            flags.append("QC_TM_SIGNATURE: Ex/|E| = %.5f vs cos(theta) %.5f (diff %.2e > %.1e): the launch is not the TM wave assumed" % (tm, math.cos(math.radians(man["theta_deg"])), dtm, TM_SIG_TOL))
    else:
        flags.append("QC_TM_SIGNATURE: no Ez file, the TM signature cannot be checked (unchecked is not passed)")
    res["kz_i"] = P._kz(2.0 * math.pi / man["lam_um"] * man["n_sub"], 2.0 * math.pi * int(man["m_inc"]) / man["lx_um"])
    sp = spacer_report(man, orders, res)
    res["spacer"] = sp
    # ---- closure: R_go from energy, compared with the direct specular
    m_inc = int(man["m_inc"])
    refl, trans = res["refl"], res["trans"]
    others = sum(v["R"] for m, v in refl.items() if m != m_inc)
    T_tot = res["T_total"]
    R_go_direct = refl[m_inc]["R"]
    R_go_closure = 1.0 - others - T_tot
    res["closure"] = dict(R_go_direct=R_go_direct, R_go_closure=R_go_closure, diff=R_go_direct - R_go_closure,
                          others_R=others, T_total=T_tot)
    # ---- spacer flux: net up-going flux in the spacer = what leaves into the air (lossless stack)
    flux_err = sp["net_flux"] - T_tot                       # independent of the R planes: spacer up-flux minus the power in the air planes
    res["spacer_flux_err"] = flux_err
    if not P._within(abs(flux_err), SPACER_FLUX_TOL):
        flags.append("QC_SPACER_FLUX: spacer net flux %.5f vs T_total %.5f (diff %.2e > %.1e)" % (sp["net_flux"], T_tot, flux_err, SPACER_FLUX_TOL))
    # ---- record additions
    rec = res["record"]
    rec["purity"] = sp["purity"]
    rec["one_minus_purity"] = sp["one_minus_purity"]
    rec["R_go_closure"] = R_go_closure
    rec["R_go_direct"] = R_go_direct
    rec["eta_air_T0"] = trans[0]["T"]
    rec["R_by_order"] = {str(m): v["R"] for m, v in sorted(refl.items())}
    res["quality_ok"] = not flags
    return res


# ------------------------------------------------------------------ combine +theta / -theta
def combine(pth, mth, recip_tol=RECIP_TOL):
    """Both launches (analysed results) -> the record quantities of PLAN section 2 and the reciprocity R0(+) vs R0(-).
    Record keys: eta_plus = T(-1)@+theta = air order 0 of pth; eta_minus = T(+1)@-theta = air order 0 of mth;
    R_go (+) / (-) = specular closure value of each launch; back_p1_Rm2 = R(-2)@+theta = abs order -1 of pth; echo = R(+2)@-theta = abs order +1 of mth;
    back_R0_rev = R(0)@-theta."""
    ep, em = pth["trans"][0]["T"], mth["trans"][0]["T"]
    Rp, Rm = pth["closure"]["R_go_closure"], mth["closure"]["R_go_closure"]
    Rpd, Rmd = pth["closure"]["R_go_direct"], mth["closure"]["R_go_direct"]
    R_go = 0.5 * (Rp + Rm)
    signal = math.sqrt(max(ep * R_go * em, 0.0))
    echo = mth["refl"][+1]["R"]
    out = dict(eta_plus=ep, eta_minus=em, R_go_pth=Rp, R_go_mth=Rm, R_go_direct_pth=Rpd, R_go_direct_mth=Rmd, R_go=R_go,
               back_p1_Rm2=pth["refl"][-1]["R"], back_R0_rev=Rm, echo_R_p2=echo, echo_db=10.0 * math.log10(max(echo, 1e-12)),
               signal=signal, efficiency_pct=100.0 * signal / (4.0 / 9.0) ** 1.5,
               purity_pth=pth["spacer"]["purity"], purity_mth=mth["spacer"]["purity"],
               one_minus_purity_pth=pth["spacer"]["one_minus_purity"], one_minus_purity_mth=mth["spacer"]["one_minus_purity"],
               recip_closure=abs(Rp - Rm), recip_direct=abs(Rpd - Rmd))
    out["recip_ok"] = bool(P._within(out["recip_closure"], recip_tol))
    return out


# ------------------------------------------------------------------ analytic C0 / C1 references
def tmm_stack(man, rho):
    """Analytic TM (H-basis recursion of fdtd_atom_post.tmm_h) of the bare stack with L1 and L2 both uniform (rho 0 -> SiO2, 1 -> a-Si),
    launched at kx = m_inc G. Returns r_E (Ex basis, referred to z = 0, the L1 bottom face) and the spacer B / A (Ex basis) at the plane z."""
    k0 = 2.0 * math.pi / man["lam_um"]
    kx = 2.0 * math.pi * int(man["m_inc"]) / man["lx_um"]
    nL = man["n_sub"] if rho == 0 else man["n_asi"]
    z = man["z"]
    term = man["term_layers"]
    n_full = [man["n_sub"], nL, nL, man["n_sub"]] + [t[0] for t in term] + [1.0]
    d_full = [z["h1"], z["h2"] - z["h1"], z["sp_top"] - z["h2"]] + [t[1] for t in term]
    R_h, _T = P.tmm_h(n_full, d_full, kx, k0)
    n_up = [man["n_sub"]] + [t[0] for t in term] + [1.0]
    d_up = [t[1] for t in term]
    Ru, _T2 = P.tmm_h(n_up, d_up, kx, k0)
    kz = P._kz(k0 * man["n_sub"], kx)
    out = dict(r_E=-R_h, kz=kz, kx=kx)
    for name in ("S1", "S2"):
        out["BA_" + name] = -(Ru * cmath.exp(2j * kz * (z["sp_top"] - z[name])))      # Ex basis; B/A at plane S_j
    return out


def _wrap(d):
    return P.wrap180(d)


def c1_observables(man, orders, m_use=None, sigma=None, basis="E"):
    """Specular r_E (mean of R1 / R2 planes, referred to z = 0, empty-run subtraction) and spacer B / A at S1 for the launched order read as
    m_use (default m_inc) with phasor convention sigma (default: the one read from the empty run). m_use / sigma / basis are the NEEDLE inputs:
    a wrong value is exactly what a convention error would be, and the same code path must then FAIL the comparison."""
    m = int(man["m_inc"]) if m_use is None else int(m_use)
    sg = int(orders["sigma"]) if sigma is None else int(sigma)
    cv = _cv(sg)
    k0 = 2.0 * math.pi / man["lam_um"]
    n = man["n_sub"]
    lx = man["lx_um"]
    kz_i = P._kz(k0 * n, 2.0 * math.pi * m / lx)
    if isinstance(kz_i, complex):
        raise ValueError("order %d evanescent" % m)
    vals = []
    for j in "12":
        tot = cv(_orders_c(orders["st"]["R" + j]["ex"], m))
        inc = cv(_orders_c(orders["emp"]["R" + j]["ex"], m))
        vals.append((tot - inc) / inc * cmath.exp(2j * kz_i * (man["z"]["R" + j] - man["z"]["z_in_ref"])))
    r = 0.5 * (vals[0] + vals[1])
    s = spacer_solve(man, orders, m, sigma=sg)
    ba = {}
    for name in ("S1", "S2"):
        c1 = cv(_orders_c(orders["st"][name]["ex"], m))
        ba[name] = c1
    A, B = s["A"], s["B"]
    BA = B / A
    if basis == "H":
        BA = -BA
        r = -r
    return dict(r_E=r, r_spread=abs(vals[0] - vals[1]), BA_S1=BA, A=A, B=B, m=m, sigma=sg)


# tolerance of PLAN section 3 control C1 (at the finest rung); the k1 / k2 rungs are judged against the same numbers and reported as they come
C1_TOL = dict(BA_amp_rel=2e-3, BA_phase_deg=0.5, r_amp_abs=2e-3, r_phase_deg=0.5, other_orders_rel=2e-3)


def c1_compare(man, orders, rho, needle=None):
    """C1: FDTD vs analytic TMM for the bare stack. needle in (None, 'm_inc_flip', 'sigma_flip', 'basis_H') selects a deliberately wrong convention.
    Returns the numbers and per-item PASS / FAIL against C1_TOL."""
    ref = tmm_stack(man, rho)
    kw = {}
    if needle == "m_inc_flip":
        kw["m_use"] = -int(man["m_inc"])
    elif needle == "sigma_flip":
        kw["sigma"] = -int(orders["sigma"])
    elif needle == "basis_H":
        kw["basis"] = "H"
    elif needle is not None:
        raise ValueError("unknown needle %r" % needle)
    ob = c1_observables(man, orders, **kw)
    ba_ref = ref["BA_S1"]
    d_ba_amp = abs(ob["BA_S1"]) / abs(ba_ref) - 1.0
    d_ba_ph = _wrap(P.deg(ob["BA_S1"] / ba_ref))
    d_r_amp = abs(ob["r_E"]) - abs(ref["r_E"])
    d_r_ph = _wrap(P.deg(ob["r_E"] / ref["r_E"]))
    # the other absolute orders must be empty in a bare stack: |A_m|, |B_m| over |A_inc| (spacer) and |c_m| over the incident in R planes
    sg = int(orders["sigma"])
    other = {}
    A_inc = abs(spacer_solve(man, orders, int(man["m_inc"]))["A"])
    for m in range(-1, 2):
        if m == int(man["m_inc"]):
            continue
        s = spacer_solve(man, orders, m)
        if s is not None:
            other[str(m)] = max(abs(s["A"]), abs(s["B"])) / A_inc
    t = C1_TOL
    items = dict(BA_amp_rel=(abs(d_ba_amp), t["BA_amp_rel"]), BA_phase_deg=(abs(d_ba_ph), t["BA_phase_deg"]),
                 r_amp_abs=(abs(d_r_amp), t["r_amp_abs"]), r_phase_deg=(abs(d_r_ph), t["r_phase_deg"]),
                 other_orders_rel=(max(other.values()) if other else 0.0, t["other_orders_rel"]))
    verdict = {k: dict(value=v[0], tol=v[1], ok=bool(P._within(v[0], v[1]))) for k, v in items.items()}
    return dict(rho=rho, needle=needle, BA_fdtd=ob["BA_S1"], BA_tmm=ba_ref, d_BA_amp_rel=d_ba_amp, d_BA_phase_deg=d_ba_ph,
                r_fdtd=ob["r_E"], r_tmm=ref["r_E"], d_r_amp=d_r_amp, d_r_phase_deg=d_r_ph, r_spread=ob["r_spread"], other_orders=other,
                items=verdict, pass_all=bool(all(v["ok"] for v in verdict.values())))


# ------------------------------------------------------------------ C0: launch
def c0_launch(man, orders):
    """C0: |Ex| / |E| = cos(theta) (needs Ez or the component ratio), kx / k0, phase advance R1 -> R2 of the launched order in the EMPTY run."""
    k0 = 2.0 * math.pi / man["lam_um"]
    theta = math.radians(man["theta_deg"])
    m = int(orders["m_inc_found"])
    out = dict(m_inc_found=m, m_inc_expected=int(man["m_inc"]), sigma=int(orders["sigma"]),
               kx_over_k0=2.0 * math.pi * m / man["lx_um"] / k0, kx_over_k0_expected=man["n_sub"] * math.sin(theta),
               cos_theta=math.cos(theta), phase_adv_deg=orders["phase_adv_deg"], phase_adv_expect_deg=orders["phase_adv_expect_deg"],
               phase_adv_err_deg=_wrap(orders["phase_adv_deg"] - orders["phase_adv_expect_deg"]),
               abs_ratio_R2_R1=orders["abs_ratio_R2_R1"])
    e = orders["emp"]["R1"]
    out["comps_present"] = sorted(e.keys())
    ex = abs(P._c(e["ex"][str(m)]))
    ey = abs(P._c(e["ey"][str(m)])) if "ey" in e else float("nan")
    ez = abs(P._c(e["ez"][str(m)])) if "ez" in e else float("nan")
    out.update(abs_ex=ex, abs_ey=ey, abs_ez=ez)
    if "ez" in e and ex > 0:
        out["Ex_over_Eplane"] = ex / math.sqrt(ex * ex + ez * ez)                      # TM: E in the x-z plane
        out["Ey_over_Ex"] = ey / ex
    return out
