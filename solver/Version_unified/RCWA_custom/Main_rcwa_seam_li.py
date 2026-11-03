# -*- coding: utf-8 -*-
"""Main_rcwa_seam_li.py -- PROPOSED DRIVER (2026-10-12), NOT INSTALLED in code\\Version_unified.

Quadrant-seam loss of the SCDC TM carrier (L1/L2 bottom grating) on the Li basis.
It is a job-file port of the 260929 scratch scripts (advchk: adv_lib.py, adv_seam.py, signres.py,
adv_perp.py) with ONE addition: the factorization rule is a job key (rcwa_rule / rcwa_ff, same keys
and same scd_c_grating.set_rcwa() as Main_rcwa_scd_c.py), and the whole job is self-contained
(no scratchpad path). Nothing in scd_c_grating.py / the vendored grcwa is changed.

INSTALL (one new file, no overwrite): copy this file to
    <CODE_ROOT>\\Version_unified\\RCWA_custom\\Main_rcwa_seam_li.py
The whole code tree is synchronised before each job, so a new file is
picked up with no restart. The script is NOT in the staging-script list: it writes
straight into Output_folder\\subfolder_name on the share (results are < 1 MB).

WHAT IT MEASURES (TM lane only; both lanes are the TM lane since 2026-10-07)
  seam_kind = parallel   groove-PARALLEL seam (grating shifted by shift_px pixels across a line
                         parallel to the grooves). 1-D supercell of M periods, half unshifted /
                         half shifted by shift_px (512 px = one period 1315.39 nm, so 64 px =
                         Lambda/8 = 164.4 nm). The two seams of the cell (+d at u = 1/2 and -d at
                         the wrap) are resolved by splitting the cell at the quarter points
                         (signres.py): D+ and D-.
  seam_kind = perp       groove-PERPENDICULAR seam. 2-D cell, x period Lambda (exact), y period
                         Ly = NP*Lambda, rectangular truncation |mx| <= Kx, |ny| <= Ky; the
                         grating is x-shifted by shift_px for y in [Ly/2, Ly) (128 px = Lambda/4).
                         Reports the MEAN of the +d / -d seam (they cannot be separated here).
  seam_kind = m1ref      single-period controls: (a) M.report() at nG_ref against a stored record
                         (ref_json) and (b) a K ladder of the four design-order powers.

  D (um) = per-pass complex amplitude deficit of one seam, relative to the local-periodic (LPA)
  field. Signal-amplitude loss of the receiver for ONE seam line through the centre of a Gaussian
  beam of 1/e^2 intensity radius w:   loss = D_sum * sqrt(2/pi) / w ,   D_sum = - sum_{ep,r0,em}
  Re D  (the three SIGNAL passes add in amplitude). For w = 100 um that is 0.7979 % per um.
  G4 of facts_v3.md: 0.798 %/um x (1.004 + 0.828) = 1.46 % (D+) / (1.188 + 0.828) = 1.61 % (D-).

JOB KEYS (all optional unless marked)
  seam_kind        parallel | perp | m1ref                         (required)
  rcwa_rule        laurent | liy | lix          default liy (Li); laurent = the 260929 basis
  rcwa_ff          0 | 1                        default 1 when rcwa_rule != laurent, else 0
  carrier_npy      (2, 512) density of the bottom grating            (required)
  H1_nm, H2_nm     400, 300        material1/2 a-Si      fill1/2 SiO2
  n_sub            1.4440          period_nm 1315.39     lam_nm 1550.0
  out_spacer_nm    465             n_out_spacer 1.4440
  out_l3_nm        875             total L3 thickness of the uniform 2-layer termination
  out_l3_token     |r|:arg_deg[:argrho_deg]   default 0.1817:217.4:90.0 (the 260929 token)
  out_l3_n_min     1.178358
  shift_px         64 (parallel) / 128 (perp); 0 is the null control (D must be 0)
  M, K             parallel: supercell periods, retained orders per period
  NP, Kx, Ky       perp: Ly = NP periods, rectangular truncation
  ref_json, nG_ref, K_list      m1ref only
  beam_w_um        100.0
  mem_cap_gb       peak working-set guard (default 12); the job stops with an error above it
  blas_threads     default 1 (set 2 for 2-core allocations)
  affinity         optional "12-15" for a DIRECT run (psutil, set after the imports); omit on the cluster
  Output_folder, subfolder_name (required)

OUTPUTS (Output_folder\\subfolder_name)
  seam_li_{seam_kind}.json   all numbers + the config
  Variable_result_information.txt   two-line summary
  run_log.txt                the same text that goes to stdout
"""
import os
import sys

sys.dont_write_bytecode = True


def _read_job(path):
    cfg = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


if len(sys.argv) < 2:
    print("[SEAM] usage: Main_rcwa_seam_li.py <job_file>")
    raise SystemExit(2)
JOB_PATH = sys.argv[1]
if not os.path.isfile(JOB_PATH):
    print("[SEAM] job file not found: %s" % JOB_PATH)
    raise SystemExit(2)
CFG = _read_job(JOB_PATH)

_bt = CFG.get("blas_threads", "").strip() or "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _bt

import json     # noqa: E402
import time     # noqa: E402
import numpy as np   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE,):
    if os.path.isfile(os.path.join(_cand, "scd_c_grating.py")):
        sys.path.insert(0, _cand)
        break
else:
    print("[SEAM] ERROR: scd_c_grating.py not found next to this file")
    raise SystemExit(2)

import scd_c_grating as M        # noqa: E402  (puts _vendor on sys.path)
import grcwa                     # noqa: E402
grcwa.set_backend("numpy")
import grcwa.rcwa as RC          # noqa: E402
import psutil                    # noqa: E402

PROC = psutil.Process()
_aff = CFG.get("affinity", "").strip()
if _aff:
    _lo, _hi = (int(v) for v in _aff.split("-"))
    PROC.cpu_affinity(list(range(_lo, _hi + 1)))     # after the imports, as the project rule demands

_LOG = []


def log(msg=""):
    print(msg, flush=True)
    _LOG.append(msg)


def num(key, default):
    v = CFG.get(key, "").strip()
    return float(v) if v else float(default)


MEM_CAP_GB = num("mem_cap_gb", 12.0)
LAM = num("lam_nm", 1550.0) * 1e-3
H1 = num("H1_nm", 400.0) * 1e-3
H2 = num("H2_nm", 300.0) * 1e-3
BEAM_W = num("beam_w_um", 100.0)
LOSS_COEF = 100.0 * np.sqrt(2.0 / np.pi) / BEAM_W            # percent per um of summed -Re D

# TM lane: (name, family, design order m) per incidence sign; the signal is ep * r0 * em
PASSES = {+1: [("ep", "T", -1), ("r0", "R", 0), ("bf", "R", -2)],
          -1: [("em", "T", +1), ("echo", "R", +2)]}
SIG = ("ep", "r0", "em")


def memcheck(tag=""):
    pk = PROC.memory_info().peak_wset / 1e9
    if pk > MEM_CAP_GB:
        raise RuntimeError("peak %.2f GB > cap %.1f GB (%s)" % (pk, MEM_CAP_GB, tag))
    return pk


def setup():
    rule = (CFG.get("rcwa_rule", "").strip().lower() or "liy")
    ffr = CFG.get("rcwa_ff", "").strip()
    ff = int(float(ffr)) if ffr else (0 if rule == "laurent" else 1)
    if rule != "laurent" or ff:
        M.set_rcwa(rule, ff)
    log("[SEAM] rcwa_rule = %s  rcwa_ff = %d  (scd_c_grating.RCWA_RULE = %s, RCWA_FF = %s)"
        % (rule, ff, M.RCWA_RULE, M.RCWA_FF))
    n_sub = num("n_sub", 1.444)
    M.set_substrate(n_sub)
    M.set_period(num("period_nm", 1315.39))
    M.LAMS = (LAM,)
    M.set_gap(0.0)
    M.set_gap_list(())
    M.set_out_spacer(num("out_spacer_nm", 465.0), num("n_out_spacer", 1.4440))
    tok = CFG.get("out_l3_token", "").strip() or "0.1817:217.4:90.0"
    f = [float(v) for v in tok.split(":")]
    rt = f[0] * np.exp(1j * np.radians(f[1]))
    t3 = num("out_l3_nm", 875.0)
    nmin = num("out_l3_n_min", 1.178358)
    if len(f) == 3:
        nb, tb, n3, res, ach, ares = M.solve_2layer_for_r_rho(rt, f[2], t3, LAM, n_min=nmin, lams_um=None, pol="TM")
        if nb is None:
            raise RuntimeError("token %s unreachable at %.0f nm" % (tok, t3))
        tail = "argrho target %.3f achieved %.3f (res %.3f deg)" % (f[2], ach, ares)
    else:
        nb, tb, n3, res = M.solve_2layer_for_r(rt, t3, LAM, n_min=nmin, lams_um=None)
        tail = "2-field token (no argrho target)"
    M.set_out_l3_list(t3, [n3])
    M.set_out_l3b_list([(nb, tb)])
    M._set_termination(0, n3)
    eb = (M.eps_of("a-Si"), M.eps_of("a-Si"))
    ef = M.eps_of("SiO2")
    d = np.load(CFG["carrier_npy"])
    if d.shape != (2, 512):
        raise RuntimeError("carrier density shape %s != (2, 512)" % (d.shape,))
    log("[SEAM setup] token %s  t3 %.0f nm: n_b %.7f t_b %.4f n3 %.9f r-res %.2e  %s | PERIOD %.5f um "
        "THETA_SUB %.4f deg | density %s uniq %s" % (tok, t3, nb, tb, n3, res, tail, M.PERIOD,
                                                      np.degrees(M.THETA_SUB), d.shape, np.unique(np.round(d, 6))[:4]))
    return dict(eb=eb, ef=ef, d=d, n3=n3, nb=nb, tb=tb, rule=rule, ff=ff, token=tok, t3=t3, res=res)


def solve(ctx, rows, Msc, K, sgn):
    """1-D supercell solve; per-order power and power-normalised complex amplitude of the
    propagating orders, keyed by the supercell index j. (adv_lib.solve, unchanged)."""
    theta = sgn * M.THETA_SUB
    M.L1 = [Msc * M.PERIOD, 0.0]
    M.NX = 512 * Msc
    try:
        t0 = time.time()
        ob, R, T = M._solve(rows[0], rows[1], H1, H2, ctx["eb"], ctx["ef"], ctx["ef"], "sub", "TM", theta, LAM,
                            2 * K * Msc + 3)
        dt = time.time() - t0
        G = np.asarray(ob.G)
        assert np.all(G[:, 1] == 0)
        j = G[:, 0].astype(int)
        assert np.abs(j).max() == K * Msc and ob.nG == 2 * K * Msc + 1, (ob.nG, np.abs(j).max())
        _a, bR = ob.GetAmplitudes(0, 0.5)
        aT, _b = ob.GetAmplitudes(ob.Layer_N - 1, 0.0)
        off = ob.nG
        bR = np.asarray(bR)[off:off + ob.nG]
        aT = np.asarray(aT)[off:off + ob.nG]
        kx = M.N_SUB * np.sin(theta) + j * LAM / (Msc * M.PERIOD)
        R = np.real(np.asarray(R))
        T = np.real(np.asarray(T))
        out = dict(j=j, kx=kx, R=R, T=T, nG=ob.nG, dt=dt, energy=float(R.sum() + T.sum()))
        for fam, P, raw, nout in (("R", R, bR, M.N_SUB), ("T", T, aT, 1.0)):
            prop = np.abs(kx) < nout
            A = np.sqrt(np.clip(P, 0, None)) * np.exp(1j * np.angle(raw))
            out[fam + "A"] = {int(jj): A[i] for i, jj in enumerate(j) if prop[i]}
        out["peak_gb"] = memcheck("solve M%d K%d" % (Msc, K))
        return out
    finally:
        M.L1 = [M.PERIOD, 0.0]
        M.NX = 512


def halfhalf_rows(d, Msc, s):
    h = Msc // 2
    return [np.concatenate([np.tile(d[l], h), np.tile(np.roll(d[l], s), Msc - h)]) for l in (0, 1)]


def seg_proj(Adict, dsc, Ks, nsamp=8192):
    ks = np.array([j - dsc for j in Adict], dtype=float)
    A = np.array([Adict[j] for j in Adict])
    u0 = (np.arange(nsamp) + 0.5) / nsamp / Ks
    g = []
    for q in range(Ks):
        uu = q / Ks + u0
        acc = 0j
        for i0 in range(0, len(uu), 2048):
            acc += (A[None, :] * np.exp(2j * np.pi * np.outer(uu[i0:i0 + 2048], ks))).sum()
        g.append(acc / len(uu))
    return np.array(g)


def ideal(a, keys, dsc, Ks, phis):
    out = {}
    for j in keys:
        k = j - dsc
        cq = 0j
        for q, ph in enumerate(phis):
            lo, hi = q / Ks, (q + 1) / Ks
            if k == 0:
                cq += np.exp(1j * ph) * (hi - lo)
            else:
                cq += np.exp(1j * ph) * (np.exp(-2j * np.pi * k * hi) - np.exp(-2j * np.pi * k * lo)) / (-2j * np.pi * k)
        out[j] = a * cq
    return out


def field(js, A, dsc, u):
    ks = js - dsc
    out = np.empty(len(u), complex)
    for i0 in range(0, len(u), 4096):
        out[i0:i0 + 4096] = (A[None, :] * np.exp(2j * np.pi * np.outer(u[i0:i0 + 4096], ks))).sum(1)
    return out


def lpa(js, a, dsc, phi):
    k = js - dsc
    c = np.zeros(len(js), complex)
    c[k == 0] = (1 + np.exp(1j * phi)) / 2
    odd = (k % 2 != 0)
    c[odd] = (1 - np.exp(1j * phi)) / (1j * np.pi * k[odd])
    return a * c


def cx(z):
    return [float(np.real(z)), float(np.imag(z))]


def run_parallel(ctx):
    Msc, K, shift = int(num("M", 48)), int(num("K", 10)), int(num("shift_px", 64))
    if Msc % 4:
        raise RuntimeError("M must be a multiple of 4 (quarter-point split of the sign-resolved deficit)")
    d = ctx["d"]
    PER = M.PERIOD
    L = Msc * PER
    rows = halfhalf_rows(d, Msc, shift)
    rec = dict(seam_kind="parallel", M=Msc, K=K, shift_px=shift, L_um=L, passes={}, energy=[])
    tt = time.time()
    n = 16384
    u = (np.arange(n) + 0.5) / n
    for sgn, lst in PASSES.items():
        F1 = solve(ctx, [d[0], d[1]], 1, K, sgn)
        Fs = solve(ctx, rows, Msc, K, sgn)
        rec["energy"].append(Fs["energy"])
        rec["twoNG"] = 2 * Fs["nG"]
        for name, fam, m in lst:
            a = F1[fam + "A"][m]
            dsc = Msc * m
            phi = -m * 2 * np.pi * shift / 512.0
            phis = [0.0, phi]
            g = seg_proj(Fs[fam + "A"], dsc, 2) * np.exp(-1j * np.array(phis))
            gi = seg_proj(ideal(a, list(Fs[fam + "A"].keys()), dsc, 2, phis), dsc, 2) * np.exp(-1j * np.array(phis))
            eta = float(np.mean(np.abs(g) ** 2) / abs(a) ** 2)
            eta0 = float(np.mean(np.abs(gi) ** 2) / abs(a) ** 2)
            w = (eta0 - eta) * L / 2
            D = np.sum(g / a - gi / a) * L / 4
            # sign-resolved deficit (signres.py): window [1/4, 3/4) holds the +d seam, the rest the -d seam
            js = np.array(sorted(Fs[fam + "A"].keys()))
            A = np.array([Fs[fam + "A"][int(jj)] for jj in js])
            E = field(js, A, dsc, u)
            E0 = field(js, lpa(js, a, dsc, phi), dsc, u)
            ph = np.where(u < 0.5, 0.0, phi)
            r = (E - E0) / (a * np.exp(1j * ph))
            plus = (u >= 0.25) & (u < 0.75)
            Dp = r[plus].sum() / n * L
            Dm = r[~plus].sum() / n * L
            dist = np.minimum(np.abs(u - 0.5), np.minimum(u, 1 - u))
            e2 = np.abs(r) ** 2
            loc = float(e2[dist < 0.125].sum() / max(e2.sum(), 1e-300))
            rec["passes"][name] = dict(P1=float(abs(a) ** 2), eta=eta, eta0=eta0, w_um=float(w), D_um=cx(D),
                                       Dp_um=cx(Dp), Dm_um=cx(Dm), loc=loc,
                                       amp=[float(abs(x)) for x in g / a],
                                       ph_deg=[float(np.degrees(np.angle(x))) for x in g / a])
    p = rec["passes"]
    sp = sum(p[k]["Dp_um"][0] for k in SIG)
    sm = sum(p[k]["Dm_um"][0] for k in SIG)
    # LOSS_COEF is percent per um: loss [%] = -sum ReD [um] * LOSS_COEF
    rec.update(sum_ReD_sig_plus=float(sp), sum_ReD_sig_minus=float(sm), sum_ReD_sig_mean=float((sp + sm) / 2),
               loss_pct_plus=float(-sp * LOSS_COEF), loss_pct_minus=float(-sm * LOSS_COEF),
               loss_coef_pct_per_um=float(LOSS_COEF), beam_w_um=BEAM_W,
               dt=time.time() - tt, peak_gb=memcheck())
    log("[parallel s%d M%d K%d %s/ff%d] 2nG %d L %.2f um  E-1 %s  (%.0f s, %.2f GB)"
        % (shift, Msc, K, ctx["rule"], ctx["ff"], rec["twoNG"], L, ["%.1e" % (e - 1) for e in rec["energy"]],
           rec["dt"], rec["peak_gb"]))
    for name, v in p.items():
        log("   %-5s D+ %+7.3f%+7.3fi   D- %+7.3f%+7.3fi   mean %+7.3f%+7.3fi  loc %.2f   eta %.5f eta0 %.5f P1 %.5f"
            % (name, v["Dp_um"][0], v["Dp_um"][1], v["Dm_um"][0], v["Dm_um"][1],
               (v["Dp_um"][0] + v["Dm_um"][0]) / 2, (v["Dp_um"][1] + v["Dm_um"][1]) / 2, v["loc"],
               v["eta"], v["eta0"], v["P1"]))
    log("   signal sum Re D:  +d seam %+.3f um   -d seam %+.3f um   mean %+.3f um  ->  loss %.3f %% (D+) / %.3f %% (D-) "
        "at w = %.0f um (%.4f %%/um)" % (sp, sm, (sp + sm) / 2, rec["loss_pct_plus"], rec["loss_pct_minus"], BEAM_W,
                                         LOSS_COEF))
    return rec


# ---------------------------------------------------------------- perpendicular seam (adv_perp.py)
_BOX = {}
_ORIG_GETG = RC.Lattice_getG


def _rect(nG, Lk1, Lk2, method=0):
    Kx, Ky = _BOX["K"]
    g1, g2 = np.meshgrid(np.arange(-Kx, Kx + 1), np.arange(-Ky, Ky + 1), indexing="ij")
    g1 = g1.ravel()
    g2 = g2.ravel()
    u = np.linalg.norm(Lk1)
    v = np.linalg.norm(Lk2)
    o = np.lexsort((g2, g1, g1 ** 2 * u ** 2 + g2 ** 2 * v ** 2))
    G = np.stack([g1[o], g2[o]], 1).astype(int)
    assert G[0, 0] == 0 and G[0, 1] == 0
    return G, len(G)


def solve2d(ctx, NP, Kx, Ky, s, sgn):
    d = ctx["d"]
    PER = M.PERIOD
    NYg = max(4 * Ky + 4, 8)
    NYg += NYg % 2
    Ly = NP * PER
    lay = []
    for l in (0, 1):
        a = np.empty((512, NYg))
        a[:, :NYg // 2] = d[l][:, None]
        a[:, NYg // 2:] = np.roll(d[l], s)[:, None]
        lay.append(a.ravel())
    save = (M.GRID_2D, M.NX, list(M.L1), list(M.L2))
    _BOX["K"] = (Kx, Ky)
    RC.Lattice_getG = _rect
    try:
        M.GRID_2D = (512, NYg)
        M.NX = 512 * NYg
        M.L1 = [PER, 0.0]
        M.L2 = [0.0, Ly]
        t0 = time.time()
        ob, R, T = M._solve(lay[0], lay[1], H1, H2, ctx["eb"], ctx["ef"], ctx["ef"], "sub", "TM", sgn * M.THETA_SUB,
                            LAM, (2 * Kx + 1) * (2 * Ky + 1))
        dt = time.time() - t0
        G = np.asarray(ob.G).astype(int)
        _a, bR = ob.GetAmplitudes(0, 0.5)
        aT, _b = ob.GetAmplitudes(ob.Layer_N - 1, 0.0)
        n = ob.nG
        off = n
        R = np.real(np.asarray(R))
        T = np.real(np.asarray(T))
        kx = M.N_SUB * np.sin(sgn * M.THETA_SUB) + G[:, 0] * LAM / PER
        ky = G[:, 1] * LAM / Ly
        out = dict(G=G, PR=R, PT=T, dt=dt, nG=n, Ly=Ly, energy=float(R.sum() + T.sum()))
        for fam, P, raw, nout in (("R", R, np.asarray(bR), M.N_SUB), ("T", T, np.asarray(aT), 1.0)):
            prop = kx ** 2 + ky ** 2 < nout ** 2
            co = raw[off:off + n]
            xp = raw[0:n]
            A = np.sqrt(np.clip(P, 0, None)) * np.exp(1j * np.angle(co))
            out[fam] = {(int(G[i, 0]), int(G[i, 1])): A[i] for i in range(n) if prop[i]}
            out[fam + "_xpfrac"] = {(int(G[i, 0]), int(G[i, 1])): float(abs(xp[i]) ** 2 / max(abs(co[i]) ** 2 + abs(xp[i]) ** 2, 1e-300))
                                    for i in range(n) if prop[i]}
        out["peak_gb"] = memcheck("perp")
        return out
    finally:
        RC.Lattice_getG = _ORIG_GETG
        M.GRID_2D, M.NX, M.L1, M.L2 = save[0], save[1], save[2], save[3]


def halves(Ad, nsamp=8192):
    ns = np.array(list(Ad.keys()), dtype=float)
    A = np.array(list(Ad.values()))
    u0 = (np.arange(nsamp) + 0.5) / nsamp / 2
    return np.array([(A[None, :] * np.exp(2j * np.pi * np.outer(lo + u0, ns))).sum(1).mean() for lo in (0.0, 0.5)])


def ideal_perp(a, keys, phi):
    o = {}
    for k in keys:
        o[k] = a * ((1 + np.exp(1j * phi)) / 2 if k == 0 else ((1 - np.exp(1j * phi)) / (1j * np.pi * k) if k % 2 else 0j))
    return o


def run_perp(ctx):
    NP, Kx, Ky, shift = int(num("NP", 16)), int(num("Kx", 8)), int(num("Ky", 32)), int(num("shift_px", 128))
    d = ctx["d"]
    rec = dict(seam_kind="perp", NP=NP, Kx=Kx, Ky=Ky, shift_px=shift, passes={}, energy=[])
    tt = time.time()
    for sgn, lst in PASSES.items():
        F1 = solve(ctx, [d[0], d[1]], 1, Kx, sgn)
        F2 = solve2d(ctx, NP, Kx, Ky, shift, sgn)
        rec["energy"].append(F2["energy"])
        rec["twoNG"] = 2 * F2["nG"]
        if shift == 0:
            offp = float(F2["PR"][F2["G"][:, 1] != 0].sum() + F2["PT"][F2["G"][:, 1] != 0].sum())
            rec.setdefault("null_off_ny_power", []).append(offp)
        for name, fam, m in lst:
            a = F1[fam + "A"][m]
            Ad = {k[1]: v for k, v in F2[fam].items() if k[0] == m}
            phi = -m * 2 * np.pi * shift / 512.0
            ph = np.array([0.0, phi])
            g = halves(Ad) * np.exp(-1j * ph) / a
            g0 = halves(ideal_perp(a, list(Ad.keys()), phi)) * np.exp(-1j * ph) / a
            eta, eta0 = float(np.mean(abs(g) ** 2)), float(np.mean(abs(g0) ** 2))
            Ly = F2["Ly"]
            D = np.sum(g - g0) * Ly / 4
            xpmax = max([F2[fam + "_xpfrac"][(m, nn)] for nn in Ad] + [0])
            rec["passes"][name] = dict(P1=float(abs(a) ** 2), eta=eta, eta0=eta0, w_um=float((eta0 - eta) * Ly / 2),
                                       D_um=cx(D), n_prop=len(Ad), xp_max=float(xpmax))
    p = rec["passes"]
    rec["w_sig_lin_um"] = float(sum(p[k]["w_um"] for k in SIG) / 2)
    rec["sum_ReD_sig"] = float(sum(p[k]["D_um"][0] for k in SIG))
    rec["loss_pct"] = float(-rec["sum_ReD_sig"] * LOSS_COEF)
    rec.update(loss_coef_pct_per_um=float(LOSS_COEF), beam_w_um=BEAM_W, dt=time.time() - tt, peak_gb=memcheck())
    log("[perp s%d NP%d Kx%d Ky%d %s/ff%d] 2nG %d Ly %.2f um E-1 %s  w_sig lin %+.3f um  sum ReD_sig %+.3f um  ->  loss %.3f %% "
        "(%.0f s, %.2f GB)" % (shift, NP, Kx, Ky, ctx["rule"], ctx["ff"], rec["twoNG"], NP * M.PERIOD,
                               ["%.1e" % (e - 1) for e in rec["energy"]], rec["w_sig_lin_um"], rec["sum_ReD_sig"],
                               rec["loss_pct"], rec["dt"], rec["peak_gb"]))
    for k, v in p.items():
        log("   %-5s P1 %.5f eta %.5f eta0 %.5f w %+8.3f um  D %+.3f%+.3fi um  nprop %d  xpol<=%.1e"
            % (k, v["P1"], v["eta"], v["eta0"], v["w_um"], v["D_um"][0], v["D_um"][1], v["n_prop"], v["xp_max"]))
    return rec


def run_m1ref(ctx):
    d = ctx["d"]
    rec = dict(seam_kind="m1ref", ladder=[], report_cmp=None)
    for K in [int(v) for v in (CFG.get("K_list", "").strip() or "10,16,20,40,128").replace(",", " ").split()]:
        row = dict(K=K, nG=2 * K + 1)
        for sgn, lst in PASSES.items():
            F1 = solve(ctx, [d[0], d[1]], 1, K, sgn)
            for name, fam, m in lst:
                row[name] = float(abs(F1[fam + "A"][m]) ** 2)
            row["energy_%+d" % sgn] = F1["energy"]
        rec["ladder"].append(row)
        log("[m1ref ladder %s/ff%d K %3d nG %3d] ep %.10f r0 %.10f bf %.10f em %.10f echo %.10f  E-1 %.1e"
            % (ctx["rule"], ctx["ff"], K, 2 * K + 1, row["ep"], row["r0"], row["bf"], row["em"], row["echo"],
               row["energy_+1"] - 1))
    ref_json = CFG.get("ref_json", "").strip()
    if ref_json:
        nref = int(num("nG_ref", 257))
        ref = json.load(open(ref_json, encoding="utf-8"))
        t0 = time.time()
        rep = M.report(d[0], d[1], H1, H2, ctx["eb"], ctx["ef"], ctx["ef"], "extract", ng=nref, pol="TM")
        pairs = []
        for k in ("1550.0_sub_p_TM_T_m1", "1550.0_sub_p_TM_R_0", "1550.0_sub_p_TM_R_m2", "1550.0_sub_m_TM_T_1",
                  "1550.0_sub_m_TM_R_2", "1550.0_sub_p_TM_argT_m1", "1550.0_sub_p_TM_argR_0"):
            if k in ref and k in rep and ref[k] is not None and rep[k] is not None:
                pairs.append((k, float(rep[k]), float(ref[k])))
        mx = max([abs(a - b) for _k, a, b in pairs] + [0.0])
        rec["report_cmp"] = dict(nG_ref=nref, ref_json=ref_json, max_abs_diff=mx, pairs=pairs, dt=time.time() - t0)
        log("[m1ref report() vs %s]  nG %d  max|diff| %.3e over %d keys (%.1f s)" % (ref_json, nref, mx, len(pairs),
                                                                                  time.time() - t0))
        for k, a, b in pairs:
            log("   %-32s driver %.12f  record %.12f  diff %+.2e" % (k, a, b, a - b))
    rec["peak_gb"] = memcheck()
    return rec


def main():
    kind = CFG.get("seam_kind", "").strip().lower()
    if kind not in ("parallel", "perp", "m1ref"):
        log("[SEAM] ERROR: seam_kind = %r (parallel | perp | m1ref)" % kind)
        return 2
    for k in ("carrier_npy", "Output_folder", "subfolder_name"):
        if not CFG.get(k, "").strip():
            log("[SEAM] ERROR: %s is required" % k)
            return 2
    out_dir = os.path.join(CFG["Output_folder"], CFG["subfolder_name"])
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    ctx = setup()
    rec = dict(run_parallel=run_parallel, run_perp=run_perp, run_m1ref=run_m1ref)["run_" + kind](ctx)
    rec["config"] = dict(CFG)
    rec["rule"] = ctx["rule"]
    rec["ff"] = ctx["ff"]
    rec["token"] = ctx["token"]
    rec["n3"] = ctx["n3"]
    rec["n_b"] = ctx["nb"]
    rec["t_b_nm"] = ctx["tb"]
    rec["elapsed_total_sec"] = time.time() - t0
    with open(os.path.join(out_dir, "seam_li_%s.json" % kind), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, default=lambda o: (float(o.real) if isinstance(o, complex) else str(o)))
    with open(os.path.join(out_dir, "Variable_result_information.txt"), "w", encoding="utf-8") as f:
        f.write("seam_kind %s  rule %s ff %d  token %s\n" % (kind, ctx["rule"], ctx["ff"], ctx["token"]))
        if kind == "parallel":
            f.write("loss_pct_plus %.4f  loss_pct_minus %.4f  sum_ReD_sig_plus %.4f  sum_ReD_sig_minus %.4f\n"
                    % (rec["loss_pct_plus"], rec["loss_pct_minus"], rec["sum_ReD_sig_plus"], rec["sum_ReD_sig_minus"]))
        elif kind == "perp":
            f.write("loss_pct %.4f  sum_ReD_sig %.4f\n" % (rec["loss_pct"], rec["sum_ReD_sig"]))
        else:
            f.write("m1ref ladder %d rows\n" % len(rec["ladder"]))
    with open(os.path.join(out_dir, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG) + "\n")
    log("[SEAM] done in %.1f s, peak %.2f GB -> %s" % (time.time() - t0, memcheck(), out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
