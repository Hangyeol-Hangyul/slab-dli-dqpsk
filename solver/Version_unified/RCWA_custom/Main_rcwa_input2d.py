# -*- coding: utf-8 -*-
"""Main_rcwa_input2d.py -- 2-D crossed-lane INPUT grating optimiser (SCDC, 261005).

Square P x P cell, fibre light from AIR at normal incidence. E_x is sent into the
substrate order (+1,0) (x-lane, 54.69 deg in the x-z plane) and E_y into (0,+1)
(y-lane, 54.69 deg in the y-z plane). Both lanes are p-polarised in their own plane
of incidence, so the y-lane is the sigma_d (x <-> y) mirror of the x-lane.

Design: <WORK_ROOT>\\input2d_261005\\design_input2d.md (read it first).
Stack (top face, shared with the retros): air / L2 Si3N4 450 nm fill air / L1 a-Si
525 nm fill SiO2 / substrate SiO2. 1550 nm only. Two independent N x N rasters
(L1, L2), each forced sigma_d-symmetric on the RAW variable before the conic filter.

Objective (maximised), two grcwa objects per iteration:
  A (air side)      : E_x and E_y on one object (the S-matrix cache survives a change of
                      excitation). eta_x = p power of E_x in (+1,0), eta_y = p power of E_y
                      in (0,+1), leak = cross-lane totals, refl = R(0,0) totals.
  B (substrate side): x-lane payer p at -theta (k_t = (-G,0)). echo = R(+2,0), relaunch
                      = R(+1,+1) (the cross-lane relaunch into the y-lane forward direction).
  J = sqrt(eta_x*eta_y) - w_leak*leak - w_refl*refl - w_echo*echo - w_rel*relaunch
      - w_lw*ramp*(P_lw(L1)+P_lw(L2))

Gates (judged on the hard binary design at the final nG rung, Section 5 of the design):
  G1 lane p power >= 0.970 each; G1-conv <= 0.005 between the last two rungs
  G2 cross-lane leak <= 0.0050 each      G3 worst-SOP fibre reflection <= 0.0080
  G4a echo R(+2,0) <= 0.0020 each lane   G4b relaunch R(+1,+1) <= 0.0020 each lane
  G5 linewidth >= 50 nm (solid and void, both layers), stack keys, support, sym_err = 0

Job keys: see KEYS below (unknown keys are an ERROR, a typo guard). Modes:
  selftest = 1  fast control set C1-C10, C12-C14 (writes selftest_input2d_<md5>.json); C14 needs init_npy
                (the S6 start) in the selftest job
  selftest = 2  adds the convergence ladder C11
  eval_only = 1 score a stored design (init_npy = (2,N,N) array) at nG_eval, no Adam
  default       Adam optimisation with ng_schedule, best-binarised tracking, ckpt/resume

ASCII-only source and console output (cp949 consoles). Every open() has encoding="utf-8".
No existing file is imported with side effects: scd_c_grating is imported (it does not
parse argv), Main_rcwa_l3_atom2d_opt is NOT imported (it parses argv at import and is an
actively edited file); its small helpers are copied verbatim and marked as such.
"""
import hashlib
import io
import json
import math
import os
import socket
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_code_dir():
    for d in (_HERE,):
        if os.path.isfile(os.path.join(d, "scd_c_grating.py")):
            return d
    return _HERE


CODE = _find_code_dir()
for _p in (CODE, os.path.join(CODE, "_vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _cfg():
    d = {}
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        for ln in io.open(sys.argv[1], encoding="utf-8", errors="replace").read().split("\n"):
            if "=" not in ln or ln.lstrip().startswith("#"):
                continue
            k, _, v = ln.partition("=")
            d[k.strip()] = v.strip()
    return d


CFG = _cfg()


# [261008 li_integrate] OPT-IN Fourier-factorization keys. Defaults = the previous behaviour, byte for byte:
#   rcwa_rule        laurent (default) | liy | lix     Li 1997 slice-consistent rule (vendored grcwa li_fmm.py)
#   rcwa_ff          0 (default) | 1                   exact pixel form factor sinc(m/N)
#   rcwa_pol_epsinv  conv1 (default) | inv             E_z rule of the Pol report only (conv1 = stock Toeplitz(1/eps))
# The Li rule / form factor apply to every NON-Pol grcwa object this driver builds; the Pol report
# (measure_air(..., fmm="pol")) keeps the Pol path and reads only rcwa_pol_epsinv.
def _li_parse(cfg):
    rule = (cfg.get("rcwa_rule", "").strip().lower() or "laurent")
    ffr = (cfg.get("rcwa_ff", "").strip() or "0")
    pinv = (cfg.get("rcwa_pol_epsinv", "").strip().lower() or "conv1")
    try:
        ff = float(ffr)
    except ValueError:
        ff = -1.0
    if rule not in ("laurent", "liy", "lix") or ff not in (0.0, 1.0) or pinv not in ("conv1", "inv"):
        print("[IN2D] ERROR: rcwa_rule=%r (laurent|liy|lix), rcwa_ff=%r (0|1), rcwa_pol_epsinv=%r (conv1|inv)"
              % (rule, ffr, pinv), flush=True)
        sys.exit(2)
    return rule, int(ff), pinv


_RCWA_RULE, _RCWA_FF, _RCWA_POLINV = _li_parse(CFG)
_LI_ON = (_RCWA_RULE != "laurent") or bool(_RCWA_FF)
_POL_ON = (_RCWA_POLINV != "conv1")
_LI_KEYS = ("rcwa_rule", "rcwa_ff", "rcwa_pol_epsinv")
# [261008 li_v2 F1] lane switch. liy is not invariant under x <-> y (lix(f) == liy(f^T) with xx <-> yy swapped), so with
# a non-Laurent rule the y-lane is NOT the sigma_d mirror of the x-lane on a sigma_d-symmetric design (measured liy+ff1:
# echo 1.95e-2 vs 1.80e-2, relaunch 1.07e-2 vs 9.1e-3). Then the substrate-side objective part_B evaluates BOTH payers
# (phi 0 and phi pi/2) and averages them. With rule laurent (also with rcwa_ff 1) the lanes are exactly equal and
# part_B keeps its x-payer-only body, byte for byte.
_LANES_BOTH = (_RCWA_RULE != "laurent")
# thread caps BEFORE numpy is imported (a numpy imported first loses the cap)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = (CFG.get("blas_threads", "").strip() or "1")

import numpy as np                                    # noqa: E402
import autograd.numpy as npa                          # noqa: E402
from autograd import value_and_grad, make_vjp         # noqa: E402
from autograd.tracer import getval                    # noqa: E402
import grcwa                                          # noqa: E402
import scd_c_grating as G                             # noqa: E402  (sets the autograd backend)

# the backend is set AFTER all imports (G sets "autograd" at import and top_grating_opt too)


def set_bk(name):
    grcwa.set_backend(name)


_DRIVER_PATH = os.path.abspath(__file__)
TAG = "[IN2D]"


def log(msg):
    print("%s %s" % (TAG, msg), flush=True)


def _md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _md5_arr(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


DRIVER_MD5 = _md5_file(_DRIVER_PATH)
_GRCWA_DIR = os.path.dirname(os.path.abspath(grcwa.__file__))

# ---------------------------------------------------------------------------
# job keys: (default, kind) ; kind "n" number, "s" string.  Unknown keys -> ERROR.
KEYS = {
    "version": ("Version_unified", "s"), "module": ("RCWA_custom", "s"),
    "script": ("Main_rcwa_input2d.py", "s"), "sweep_keys": ("[]", "s"),
    "Output_folder": ("", "s"), "subfolder_name": ("", "s"), "result_folder": ("", "s"),
    "blas_threads": ("1", "s"), "mpi_np": ("1", "s"), "mpi_np_final": ("1", "s"),
    # physics
    "period_nm": (1315.39, "n"), "n_sub": (1.4440, "n"), "lam_nm": (1550.0, "n"),
    "H1_nm": (525.0, "n"), "H2_nm": (450.0, "n"),
    "material1": ("a-Si", "s"), "material2": ("Si3N4", "s"),
    "fill1": ("SiO2", "s"), "fill2": ("air", "s"),
    # design space
    "nxy": (256, "n"), "sym": ("sigma_d", "s"), "filt_radius_nm": (70.0, "n"),
    "beta_schedule": ("0:8, 0.4:16, 0.7:32, 0.9:64", "s"), "min_feature_nm": (50.0, "n"),
    "lw_gate_nm": (57.0, "n"), "w_lw": (1.0, "n"), "lw_start_frac": (0.85, "n"),
    # optimiser
    "iters": (300, "n"), "lr": (0.03, "n"), "ng_schedule": ("", "s"),
    "best_bin_from": (0.9, "n"), "best_bin_every": (2, "n"),
    "ckpt_every": (5, "n"), "ckpt_every_s": (1800.0, "n"),
    # objective
    "w_leak": (0.5, "n"), "w_refl": (0.3, "n"), "w_echo": (1.0, "n"), "w_rel": (1.0, "n"),
    # seed
    "seed": ("xsec_min", "s"), "seed_rng": (0, "n"), "seed_src_npy": ("", "s"),
    "init_npy": ("", "s"), "init_soft": (1, "n"),
    # report
    "nG_eval": ("601, 901, 1793", "s"), "eval_only": (0, "n"), "selftest": (0, "n"),
    "selftest_c4_ng": (0, "n"), "report_band": (1, "n"), "nG_band": (901, "n"),
    "report_bias": (1, "n"), "nG_bias": (601, "n"), "report_sop": (1, "n"),
    "report_kron2": (0, "n"), "report_pol": (0, "n"), "band_flag_tol": (0.010, "n"),
    # additions to the design's key table (implementation details, see docs)
    "nG_kron2": (1793, "n"), "nG_pol": (1793, "n"), "selftest_dir": ("", "s"),
    "c1_density_npy": ("", "s"), "selftest_only": ("", "s"),
    # [261008 li_integrate] opt-in factorization keys (see _li_parse); written to the result JSON and
    # the checkpoint fingerprint ONLY when non-default
    "rcwa_rule": ("laurent", "s"), "rcwa_ff": (0, "n"), "rcwa_pol_epsinv": ("conv1", "s"),
}
# keys excluded from the checkpoint fingerprint (paths, scheduling, threads)
_FP_DROP = ("result_folder", "Output_folder", "subfolder_name", "mpi_np", "mpi_np_final",
            "blas_threads", "ckpt_every", "ckpt_every_s", "selftest_dir", "sweep_keys",
            "selftest", "selftest_c4_ng", "c1_density_npy", "selftest_only")


def cfg_get(cfg_raw, k):
    dflt, kind = KEYS[k]
    raw = cfg_raw.get(k, None)
    if raw is None or str(raw).strip() == "":
        return dflt
    if kind == "n":
        return float(str(raw).strip())
    return str(raw).strip()


def build_params(cfg_raw):
    unknown = sorted(k for k in cfg_raw if k not in KEYS)
    if unknown:
        raise ValueError("unknown job key(s): %s" % ", ".join(unknown))
    P = {k: cfg_get(cfg_raw, k) for k in KEYS}
    P["_raw"] = dict(cfg_raw)
    return P


# ---------------------------------------------------------------------------
# verbatim copies of small helpers (ATOM is not imported: it parses argv at import)
def conic_kernel_fft(n, radius_px):
    """verbatim copy of Main_rcwa_l3_atom2d_opt.conic_kernel_fft (ATOM:463-475)."""
    r = max(1, int(round(radius_px)))
    a = np.arange(n)
    d = np.minimum(a, n - a)
    DX, DY = np.meshgrid(d, d, indexing="ij")
    w = np.maximum(0.0, 1.0 - np.sqrt(DX ** 2 + DY ** 2) / float(r))
    w = w / w.sum()
    return np.fft.fft2(w)


def tanh_project(x, beta, eta=0.5):
    """verbatim copy of Main_rcwa_l3_atom2d_opt.tanh_project (ATOM:496-499)."""
    num = npa.tanh(beta * eta) + npa.tanh(beta * (x - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


_LW_ETA_E = 0.75
_LW_ETA_D = 0.25


def lw_pen(r3, KG, beta):
    """verbatim copy of Main_rcwa_l3_atom2d_opt._lw_pen (ATOM:1257-1286): area fraction of
    solid / void thinner than the structuring element (opening and closing surrogate)."""
    def _f(a):
        return npa.real(npa.fft.ifft2(npa.fft.fft2(a) * KG))
    op = tanh_project(_f(tanh_project(_f(r3), beta, _LW_ETA_E)), beta, _LW_ETA_D)
    cl = tanh_project(_f(tanh_project(_f(r3), beta, _LW_ETA_D)), beta, _LW_ETA_E)
    return (npa.mean(npa.maximum(r3 - op, 0.0)) + npa.mean(npa.maximum(cl - r3, 0.0)))


# ---------------------------------------------------------------------------
# density pipeline: sigma_d on the RAW variable, conic filter, tanh projection
def dens_one(r, KF, beta):
    s = 0.5 * (r + npa.transpose(r))
    f = npa.real(npa.fft.ifft2(npa.fft.fft2(s) * KF))
    return tanh_project(f, beta)


def dens_pair(raw, KF, beta):
    return npa.stack([dens_one(raw[0], KF, beta), dens_one(raw[1], KF, beta)])


def hard_pair(raw, KF, beta_final):
    d = np.asarray(getval(dens_pair(raw, KF, beta_final)))
    # FFT rounding leaves the filtered field asymmetric at ~1e-15, so a pixel within that distance of the
    # threshold flipped asymmetrically (sym_err 1 on the xsec_avg / multi2d_q seeds, 261006 verifier). The
    # transpose average is exactly symmetric in floating point (a + b == b + a), and so is its threshold.
    d = 0.5 * (d + np.transpose(d, (0, 2, 1)))
    return (d > 0.5).astype(float)


# ---------------------------------------------------------------------------
def _eps_keep(e):
    """[261013 opt-in] the old float(real) for a lossless material (bit-identical default path); the complex eps when k > 0."""
    return float(np.real(e)) if float(np.imag(e)) == 0.0 else complex(e)


class Phys:
    """Constants of one job (um units for lengths, eps real because every k is 0)."""

    def __init__(self, P):
        self.P = float(P["period_nm"]) * 1e-3
        self.lam = float(P["lam_nm"]) * 1e-3
        self.n_sub = float(P["n_sub"])
        self.eps_sub = self.n_sub ** 2
        self.H1 = float(P["H1_nm"]) * 1e-3
        self.H2 = float(P["H2_nm"]) * 1e-3
        self.e1s = _eps_keep(G.eps_of(P["material1"]))      # [261013] complex only for an a-Si_k* opt-in name
        self.e1v = float(np.real(G.eps_of(P["fill1"])))
        self.e2s = _eps_keep(G.eps_of(P["material2"]))
        self.e2v = float(np.real(G.eps_of(P["fill2"])))
        self.th_sub = float(np.arcsin(self.lam / (self.P * self.n_sub)))
        self.th_crit = float(np.arcsin(1.0 / self.n_sub))
        self.N = int(P["nxy"])
        self.px_nm = self.P * 1e3 / self.N


def eps_pair(rho, ph):
    e1 = ph.e1v + rho[0] * (ph.e1s - ph.e1v)
    e2 = ph.e2v + rho[1] * (ph.e2s - ph.e2v)
    return e1, e2


# ---------------------------------------------------------------------------
# grcwa object builders
def _alias_assert(ob, nx, ny):
    g = np.asarray(ob.G)
    mx, my = int(np.abs(g[:, 0]).max()), int(np.abs(g[:, 1]).max())
    if mx > nx // 4 or my > ny // 4:
        raise ValueError("2-D alias: max|mx| %d (limit %d), max|my| %d (limit %d), nG %d"
                         % (mx, nx // 4, my, ny // 4, ob.nG))
    return mx, my


def _li_apply(ob, fmm):
    """[261008 li_integrate] opt-in switches on one grcwa object; called ONLY when a Li / form-factor /
    Pol-epsinv key is non-default. Non-Pol objects get the Li rule + form factor; the Pol object gets
    only pol_epsinv."""
    if not getattr(type(ob), "LI_SUPPORT", 0):
        raise RuntimeError("the vendored grcwa in use has no Li support (obj.LI_SUPPORT missing): "
                           "refusing to run a Li job on the stock rule")
    if fmm == "pol":
        if _POL_ON:
            ob.pol_epsinv = _RCWA_POLINV
    elif _LI_ON:
        ob.rcwa_rule = _RCWA_RULE
        ob.form_factor = _RCWA_FF


def build_air(nG, e1, e2, ph, theta=0.0, lam=None, fmm=None, alias=True):
    """air | L2 | L1 | substrate (fibre light from air). e1, e2 are (Nx,Ny) eps arrays."""
    Nx, Ny = e1.shape
    lam = ph.lam if lam is None else lam
    ob = grcwa.obj(int(nG), [ph.P, 0.0], [0.0, ph.P], 1.0 / lam, theta, 0.0, verbose=0,
                   fmm_method=fmm)
    if _LI_ON or _POL_ON:
        _li_apply(ob, fmm)
    ob.Add_LayerUniform(0.5, 1.0)
    ob.Add_LayerGrid(ph.H2, Nx, Ny)
    ob.Add_LayerGrid(ph.H1, Nx, Ny)
    ob.Add_LayerUniform(0.5, ph.eps_sub)
    ob.Init_Setup()
    if alias:
        _alias_assert(ob, Nx, Ny)
    ob.GridLayer_geteps(npa.concatenate([npa.reshape(e2, (-1,)), npa.reshape(e1, (-1,))]))
    return ob


def build_sub(nG, e1, e2, ph, phi, theta=None, alias=True):
    """substrate | L1 | L2 | air at theta = -th_sub (default), azimuth phi (0 = x-payer,
    pi/2 = y-payer)."""
    Nx, Ny = e1.shape
    th = -ph.th_sub if theta is None else theta
    ob = grcwa.obj(int(nG), [ph.P, 0.0], [0.0, ph.P], 1.0 / ph.lam, th, phi, verbose=0)
    if _LI_ON:
        _li_apply(ob, None)
    ob.Add_LayerUniform(0.5, ph.eps_sub)
    ob.Add_LayerGrid(ph.H1, Nx, Ny)
    ob.Add_LayerGrid(ph.H2, Nx, Ny)
    ob.Add_LayerUniform(0.5, 1.0)
    ob.Init_Setup()
    if alias:
        _alias_assert(ob, Nx, Ny)
    ob.GridLayer_geteps(npa.concatenate([npa.reshape(e1, (-1,)), npa.reshape(e2, (-1,))]))
    return ob


def idx(ob, m, n):
    """index of order (m,n) in ob.G, looked up by VALUE (ties are ordered by argsort)."""
    g = np.asarray(ob.G)
    h = np.where((g[:, 0] == m) & (g[:, 1] == n))[0]
    if h.size != 1:
        raise KeyError("order (%d,%d) not unique/present in ob.G (nG %d)" % (m, n, ob.nG))
    return int(h[0])


def excite_Ex(ob, order=0):
    ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=order)     # y-half = +1 = E_x at phi 0


def excite_Ey(ob, order=0):
    ob.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=order)     # x-half = -1 = E_y at phi 0


def flux_halves(ob):
    """Per-order, per-half z-flux, normalised like RT_Solve(normalize=1).
    Copy of grcwa GetZPoyntingFlux (rcwa.py:897-924) WITHOUT the final x-half + y-half sum.
    Returns Rx, Ry (reflected, incidence layer), Tx, Ty (transmitted, last layer), each of
    length nG. x-half = H_x coefficient (E_y content), y-half = H_y (E_x content)."""
    aN, b0 = ob._solve_exterior()
    n = ob.nG
    nr = ob.normalization

    def _pair(ai, bi, L):
        kp, phi, q = ob.kp_list[L], ob.phi_list[L], ob.q_list[L]
        A = npa.dot(npa.dot(kp, phi), npa.diag(1.0 / ob.omega / q))
        pa, pb = npa.dot(phi, ai), npa.dot(phi, bi)
        Aa, Ab = npa.dot(A, ai), npa.dot(A, bi)
        d = npa.real(0.5 * (npa.conj(pb) * Aa - npa.conj(Ab) * pa))
        fwd = npa.real(npa.conj(Aa) * pa) + d
        bwd = -npa.real(npa.conj(Ab) * pb) + d
        return fwd, bwd

    _, bwd0 = _pair(ob.a0, b0, 0)
    fwdN, _ = _pair(aN, ob.bN, -1)
    R = -bwd0 * nr
    T = fwdN * nr
    return R[:n], R[n:], T[:n], T[n:]


# ---------------------------------------------------------------------------
# objective parts. TERMS is a side channel for the printed values (untraced floats).
TERMS = {}


def _fl(x):
    return float(getval(x))


def part_A(rho, nG, ph, w):
    e1, e2 = eps_pair(rho, ph)
    ob = build_air(nG, e1, e2, ph)
    i00, i10, i01 = idx(ob, 0, 0), idx(ob, 1, 0), idx(ob, 0, 1)
    excite_Ex(ob, i00)
    Rx, Ry, Tx, Ty = flux_halves(ob)
    eta_x = Ty[i10]
    leak_x = Tx[i01] + Ty[i01]
    refl_x = Rx[i00] + Ry[i00]
    excite_Ey(ob, i00)
    Rx2, Ry2, Tx2, Ty2 = flux_halves(ob)
    eta_y = Tx2[i01]
    leak_y = Tx2[i10] + Ty2[i10]
    refl_y = Rx2[i00] + Ry2[i00]
    leak = leak_x + leak_y
    refl = refl_x + refl_y
    f = (npa.sqrt(npa.maximum(eta_x * eta_y, 1e-12)) - w["w_leak"] * leak - w["w_refl"] * refl)
    TERMS.update(eta_x=_fl(eta_x), eta_y=_fl(eta_y), leak=_fl(leak), refl=_fl(refl),
                 leak_x=_fl(leak_x), leak_y=_fl(leak_y))
    return f


def _terms_lane_mean():
    """[261008 li_v2 F1] TERMS echo / rel = mean of the two payers (the quantities J_B is built from)."""
    TERMS.update(echo=0.5 * (TERMS["echo_x"] + TERMS["echo_y"]), rel=0.5 * (TERMS["rel_x"] + TERMS["rel_y"]))


def _part_B_lanes(rho, nG, ph, w, lane):
    """[261008 li_v2 F1] part_B for a non-Laurent rule: explicit x-payer (phi 0: echo R(+2,0), relaunch R(+1,+1)) and
    y-payer (phi pi/2: echo R(0,+2), relaunch R(+1,+1)), each its own object (measure_sub's order map), no lane
    symmetry assumed. J_B = -w_echo*mean(echo_x, echo_y) - w_rel*mean(rel_x, rel_y).
    lane None: both payers in one call. lane "x" / "y": that payer's half of J_B only (weight 0.5 each; the two halves
    add up to the full J_B), so Model.vg can keep one object's tape at a time."""
    e1, e2 = eps_pair(rho, ph)
    f = 0.0
    for ln in (("x", "y") if lane is None else (lane,)):
        ob = build_sub(nG, e1, e2, ph, 0.0 if ln == "x" else math.pi / 2)
        i00 = idx(ob, 0, 0)
        i_echo = idx(ob, 2, 0) if ln == "x" else idx(ob, 0, 2)
        i_rel = idx(ob, 1, 1)
        excite_Ex(ob, i00)
        Rx, Ry, Tx, Ty = flux_halves(ob)
        echo = Rx[i_echo] + Ry[i_echo]
        rel = Rx[i_rel] + Ry[i_rel]
        TERMS["echo_" + ln] = _fl(echo)
        TERMS["rel_" + ln] = _fl(rel)
        f = f - 0.5 * w["w_echo"] * echo - 0.5 * w["w_rel"] * rel
        del ob
    if lane is None:
        _terms_lane_mean()
    return f


def part_B(rho, nG, ph, w, lane=None):
    if _LANES_BOTH:
        return _part_B_lanes(rho, nG, ph, w, lane)
    e1, e2 = eps_pair(rho, ph)
    ob = build_sub(nG, e1, e2, ph, 0.0)
    i00, i20, i11 = idx(ob, 0, 0), idx(ob, 2, 0), idx(ob, 1, 1)
    excite_Ex(ob, i00)
    Rx, Ry, Tx, Ty = flux_halves(ob)
    echo = Rx[i20] + Ry[i20]
    rel = Rx[i11] + Ry[i11]
    TERMS.update(echo=_fl(echo), rel=_fl(rel))
    return -w["w_echo"] * echo - w["w_rel"] * rel


def part_LW(rho, KG, beta):
    return lw_pen(rho[0], KG, beta) + lw_pen(rho[1], KG, beta)


class Model:
    """Everything the objective needs: constants, filter kernels, weights."""

    def __init__(self, P):
        self.P = P
        self.ph = Phys(P)
        N = self.ph.N
        self.N = N
        self.KF = conic_kernel_fft(N, float(P["filt_radius_nm"]) / self.ph.px_nm)
        self.KG = conic_kernel_fft(N, max(1.0, float(P["lw_gate_nm"]) / self.ph.px_nm))
        self.w = {k: float(P[k]) for k in ("w_leak", "w_refl", "w_echo", "w_rel")}
        self.w_lw = float(P["w_lw"])
        self.beta_sched = parse_sched(P["beta_schedule"], "beta_schedule", False)
        self.beta_final = self.beta_sched[-1][1]

    # J on a rho (no gradient). Returns J without the linewidth term, plus terms.
    def J_rho(self, rho, nG):
        TERMS.clear()
        fa = part_A(rho, nG, self.ph, self.w)
        fb = part_B(rho, nG, self.ph, self.w)
        return float(getval(fa) + getval(fb)), dict(TERMS)

    # value, gradient wrt RAW, terms. Object-sequential VJP: one object's tape at a time.
    def vg(self, raw, nG, beta, a_lw):
        TERMS.clear()
        vjp_d, rho = make_vjp(lambda r: dens_pair(r, self.KF, beta))(raw)
        rho = np.asarray(rho)
        fa, ga = value_and_grad(lambda r: part_A(r, nG, self.ph, self.w))(rho)
        if _LANES_BOTH:
            # [261008 li_v2 F1] one payer's tape at a time, as for A and B; the two halves add up to the full J_B
            fb, gb = 0.0, 0.0
            for ln in ("x", "y"):
                f_l, g_l = value_and_grad(lambda r, ln=ln: part_B(r, nG, self.ph, self.w, ln))(rho)
                fb = fb + float(f_l)
                gb = gb + np.asarray(g_l)
            _terms_lane_mean()
        else:
            fb, gb = value_and_grad(lambda r: part_B(r, nG, self.ph, self.w))(rho)
        g_rho = np.asarray(ga) + np.asarray(gb)
        J = float(fa) + float(fb)
        P_lw = 0.0
        if self.w_lw * a_lw != 0.0:
            fl, gl = value_and_grad(lambda r: part_LW(r, self.KG, beta))(rho)
            P_lw = float(fl)
            g_rho = g_rho - self.w_lw * a_lw * np.asarray(gl)
            J -= self.w_lw * a_lw * P_lw
        else:
            P_lw = float(getval(part_LW(rho, self.KG, beta)))
        g_raw = np.asarray(vjp_d(g_rho))
        terms = dict(TERMS)
        terms["lw"] = P_lw
        return J, g_raw, terms


def parse_sched(txt, name, int_vals):
    pairs = []
    for p in str(txt).replace(";", ",").split(","):
        if not p.strip():
            continue
        a, b = p.split(":")
        pairs.append((float(a), int(float(b)) if int_vals else float(b)))
    pairs.sort()
    if not pairs or pairs[0][0] != 0.0:
        raise ValueError("%s must start with 0: (%r)" % (name, txt))
    return pairs


def sched_at(sched, frac):
    return [v for f, v in sched if frac >= f][-1]


# ---------------------------------------------------------------------------
# fast + exact medial-axis linewidth (FFT erosion). Same definition as G._medial_min_nm.
def medial_min_fast(rho, px_nm, thr=0.5):
    b = (np.asarray(rho) > thr)
    if not b.any() or b.all():
        return float("inf")
    N0, N1 = b.shape
    bf = np.fft.fft2(b.astype(float))
    D = np.zeros(b.shape, int)
    r = 1
    while r <= min(b.shape) // 2:
        ii = np.arange(-r, r + 1)
        mask = (ii[:, None] ** 2 + ii[None, :] ** 2) <= r * r
        k = np.zeros(b.shape)
        k[np.ix_(ii % N0, ii % N1)] = mask.astype(float)
        conv = np.real(np.fft.ifft2(bf * np.fft.fft2(k)))
        ero = conv > (float(mask.sum()) - 0.5)
        if not ero.any():
            break
        D[ero] = r
        r += 1
    nb = [np.roll(np.roll(D, dy, 0), dx, 1)
          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    ridge = b & np.logical_and.reduce([D >= q for q in nb])
    if not ridge.any():
        return float("inf")
    return (2 * int(D[ridge].min()) + 1) * px_nm


def mf_report(rho1, rho2, px_nm):
    out = {}
    for tag, rho in (("L1", rho1), ("L2", rho2)):
        out["mf_%s_solid_nm" % tag] = medial_min_fast(rho, px_nm)
        out["mf_%s_void_nm" % tag] = medial_min_fast(1.0 - np.asarray(rho), px_nm)
    out["min_feature_nm"] = min(out[k] for k in list(out))
    return out


def unsupported_px(rho1, rho2, fill1):
    """pixels of L2 material above a void of L1 when the L1 fill is air (floating material).
    A solid L1 fill (SiO2) planarises the surface, so nothing can float."""
    if str(fill1) not in ("air", "Air"):
        return 0
    return int(np.sum((np.asarray(rho2) > 0.5) & ~(np.asarray(rho1) > 0.5)))


# ---------------------------------------------------------------------------
# seeds (Section 6)
def rows_from_1d(d512, N):
    d = np.asarray(d512, float)
    if d.shape != (2, 512):
        raise ValueError("seed_src_npy must be a (2,512) array, got %s" % (d.shape,))
    k = 512 // N
    if k * N != 512:
        raise ValueError("N=%d does not divide 512" % N)
    f = d.reshape(2, N, k).mean(axis=2)
    return (f >= 0.5).astype(float)


def make_seed(name, N, rng_seed, src):
    """Return the RAW variable x0 of shape (2,N,N), exactly sigma_d-symmetric."""
    if name in ("xsec_min", "xsec_max", "xsec_avg"):
        if src is None:
            raise ValueError("seed %s needs seed_src_npy" % name)
        f = rows_from_1d(src, N)
        x = np.zeros((2, N, N))
        for L in (0, 1):
            fx, fy = f[L][:, None], f[L][None, :]
            if name == "xsec_min":
                rho = np.minimum(fx, fy)
            elif name == "xsec_max":
                rho = np.maximum(fx, fy)
            else:
                rho = 0.5 * (fx + fy)
            x[L] = 0.15 + 0.70 * rho              # the seed_npy softening convention
    elif name in ("multi2d_q", "multi2d"):
        u = (np.arange(N) + 0.5) / N
        phis = (np.pi / 2, -np.pi / 2) if name == "multi2d_q" else (0.0, np.pi)
        x = np.zeros((2, N, N))
        for L in (0, 1):
            s = sum(np.cos(2 * np.pi * h * u + phis[L]) for h in (1, 2))
            x[L] = 0.5 + 0.1125 * (s[:, None] + s[None, :])
    elif name == "rand":
        rng = np.random.default_rng(int(rng_seed))
        x = 0.3 + 0.4 * rng.random((2, N, N))
    else:
        raise ValueError("unknown seed %r" % name)
    x = 0.5 * (x + np.transpose(x, (0, 2, 1)))
    return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# checkpoint (copy of the ATOM pattern: atomic write, fingerprint, log truncation)
def ckpt_save(path, st, fp):
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, ".ckpt_input2d.npz.tmp_%d" % os.getpid())
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, x=np.asarray(st["x"]), m=np.asarray(st["m"]), v=np.asarray(st["v"]),
                     t=np.array([int(st["t"])], dtype=np.int64),
                     best_x=np.asarray(st["best_x"]),
                     best_J=np.array([float(st["best_J"])]),
                     best_it=np.array([int(st["best_it"])], dtype=np.int64),
                     best_ok=np.array([int(st["best_ok"])], dtype=np.int64),
                     fingerprint=np.array([str(fp)]))
        os.replace(tmp, path)
    except Exception as e:                                  # a failed save must not end the run
        log("WARN: ckpt save failed (%s) at %s -- continuing (temp file left in place)" % (e, path))
        return False
    return True


def ckpt_load(path):
    with open(path, "rb") as fh:
        z = np.load(fh)
        st = dict(x=np.array(z["x"]), m=np.array(z["m"]), v=np.array(z["v"]),
                  t=int(np.array(z["t"]).reshape(-1)[0]),
                  best_x=np.array(z["best_x"]),
                  best_J=float(np.array(z["best_J"]).reshape(-1)[0]),
                  best_it=int(np.array(z["best_it"]).reshape(-1)[0]),
                  best_ok=int(np.array(z["best_ok"]).reshape(-1)[0]))
        fp = str(np.array(z["fingerprint"]).reshape(-1)[0])
    return st, fp


def log_truncate(path, max_iter):
    if not os.path.isfile(path):
        return
    with io.open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    if not lines:
        return
    keep = [lines[0]]
    for ln in lines[1:]:
        fld = ln.split("\t", 1)
        try:
            if int(fld[0]) <= max_iter:
                keep.append(ln)
        except ValueError:
            continue
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.writelines(keep)


def fingerprint(P):
    items = sorted((k, str(P[k])) for k in KEYS if k not in _FP_DROP and k not in _LI_KEYS)
    # [261008 li_integrate] the Li keys join the fingerprint ONLY when non-default, so a Li job can
    # never resume a Laurent checkpoint while the default fingerprint is built exactly as before
    if _LI_ON or _POL_ON:
        items += sorted((k, str(P[k])) for k in _LI_KEYS)
    h = hashlib.sha256()
    h.update(repr(items).encode("utf-8"))
    h.update(DRIVER_MD5.encode("ascii"))
    h.update(_md5_file(G.__file__.replace(".pyc", ".py")).encode("ascii"))
    h.update(_md5_file(os.path.join(_GRCWA_DIR, "rcwa.py")).encode("ascii"))
    h.update(_md5_file(os.path.join(_GRCWA_DIR, "fft_funs.py")).encode("ascii"))
    if _LI_ON or _POL_ON:
        h.update(_md5_file(os.path.join(_GRCWA_DIR, "li_fmm.py")).encode("ascii"))
    for k in ("seed_src_npy", "init_npy"):
        pth = P.get(k, "")
        if pth and os.path.isfile(pth):
            h.update(_md5_file(pth).encode("ascii"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
def result_dir(P):
    if P["result_folder"]:
        return P["result_folder"]
    if not P["Output_folder"] or not P["subfolder_name"]:
        raise ValueError("Output_folder and subfolder_name (or result_folder) are required")
    return os.path.join(P["Output_folder"], P["subfolder_name"])


def _san(o):
    """JSON-safe: non-finite floats -> None, numpy scalars -> python."""
    if isinstance(o, dict):
        return {str(k): _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(path, obj):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_san(obj), ensure_ascii=True, indent=2, sort_keys=False))
    os.replace(tmp, path)


def peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", mi.rss)) / 1e9
    except Exception:
        return float("nan")


def ram_guard(est_gb, label):
    """Refuse a step that would push RAM use past the percent limit in env SIM_MAX_RAM_PCT (default: no limit)."""
    try:
        import psutil
        _lim = float(os.environ.get("SIM_MAX_RAM_PCT", "0"))
        if _lim <= 0:
            return True
        vm = psutil.virtual_memory()
        used_after = (vm.total - vm.available + est_gb * 1e9) / vm.total
        if used_after > _lim / 100.0:
            log("RAM GUARD: %s needs ~%.1f GB, RAM use would reach %.0f %% > %.0f %% -- SKIPPED"
                % (label, est_gb, 100.0 * used_after, _lim))
            return False
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# measurement of one nG rung on a hard (or any) design: objects A, B_xp(+s), B_yp
SOPS = {"H": (1.0, 0.0), "V": (0.0, 1.0), "D": (1 / math.sqrt(2), 1 / math.sqrt(2)),
        "A": (1 / math.sqrt(2), -1 / math.sqrt(2)),
        "L": (1 / math.sqrt(2), 1j / math.sqrt(2)), "R": (1 / math.sqrt(2), -1j / math.sqrt(2))}


def otag(m, n):
    return "o%+d%+d" % (m, n)


def _amp_phase_scaled(amp, power):
    """complex number with modulus sqrt(power) and the phase of amp (power-normalised amplitude)."""
    a = complex(amp)
    if abs(a) == 0.0:
        return complex(math.sqrt(max(power, 0.0)), 0.0)
    return a / abs(a) * math.sqrt(max(power, 0.0))


def measure_air(rho1, rho2, ph, nG, lam=None, theta=0.0, fmm=None, want_jones=True):
    """Forward solves from air for E_x and E_y on ONE object. numpy backend assumed.
    Returns a flat dict (no prefix) and the 2x2 Jones data."""
    e1, e2 = eps_pair(np.asarray(rho1, float), ph) if False else eps_pair(np.stack([rho1, rho2]), ph)
    ob = build_air(nG, e1, e2, ph, theta=theta, lam=lam, fmm=fmm)
    n = ob.nG
    g = np.asarray(ob.G)
    d = {"nG_realised": int(n), "K": int(max(np.abs(g[:, 0]).max(), np.abs(g[:, 1]).max()))}
    i00, i10, im10, i01, i0m1 = idx(ob, 0, 0), idx(ob, 1, 0), idx(ob, -1, 0), idx(ob, 0, 1), idx(ob, 0, -1)
    res = {}
    amps_T, amps_R = {}, {}
    for src, ex in (("Ex", excite_Ex), ("Ey", excite_Ey)):
        ex(ob, i00)
        Rx, Ry, Tx, Ty = flux_halves(ob)
        Rt, Tt = ob.RT_Solve(normalize=1, byorder=1)
        aN, b0 = ob._solve_exterior()
        d["air_%s_halves_vs_RT" % src] = float(max(np.max(np.abs(Rt - Rx - Ry)), np.max(np.abs(Tt - Tx - Ty))))
        d["air_%s_energy" % src] = float(np.sum(Rx + Ry + Tx + Ty))
        # axis-aligned orders: along x -> p = y-half, s = x-half; along y -> p = x-half, s = y-half
        for (m, nn), i in (((1, 0), i10), ((-1, 0), im10)):
            d["air_%s_T_%s_p" % (src, otag(m, nn))] = float(Ty[i])
            d["air_%s_T_%s_s" % (src, otag(m, nn))] = float(Tx[i])
        for (m, nn), i in (((0, 1), i01), ((0, -1), i0m1)):
            d["air_%s_T_%s_p" % (src, otag(m, nn))] = float(Tx[i])
            d["air_%s_T_%s_s" % (src, otag(m, nn))] = float(Ty[i])
        d["air_%s_T_o+0+0_Ex" % src] = float(Ty[i00])
        d["air_%s_T_o+0+0_Ey" % src] = float(Tx[i00])
        d["air_%s_R_o+0+0_Ex" % src] = float(Ry[i00])
        d["air_%s_R_o+0+0_Ey" % src] = float(Rx[i00])
        if want_jones:
            # reflected (0,0) amplitude, power-normalised: rows (E_x out, E_y out); sign of the rows
            # and columns is irrelevant to singular values
            bx, by = complex(b0[i00]), complex(b0[i00 + n])
            amps_R[src] = (_amp_phase_scaled(by, Ry[i00]), _amp_phase_scaled(bx, Rx[i00]))
            laneX = _amp_phase_scaled(aN[i10 + n], Ty[i10])      # (+1,0) p = y-half
            laneY = _amp_phase_scaled(aN[i01], Tx[i01])          # (0,+1) p = x-half
            amps_T[src] = (laneX, laneY)
            d["air_%s_lane_x_abs" % src] = abs(laneX)
            d["air_%s_lane_x_arg_deg" % src] = math.degrees(np.angle(laneX))
            d["air_%s_lane_y_abs" % src] = abs(laneY)
            d["air_%s_lane_y_arg_deg" % src] = math.degrees(np.angle(laneY))
    out = dict(d)
    out["eta_x"] = d["air_Ex_T_o+1+0_p"]
    out["eta_y"] = d["air_Ey_T_o+0+1_p"]
    out["leak_x"] = d["air_Ex_T_o+0+1_p"] + d["air_Ex_T_o+0+1_s"]
    out["leak_y"] = d["air_Ey_T_o+1+0_p"] + d["air_Ey_T_o+1+0_s"]
    out["leak_away_x"] = d["air_Ex_T_o+0-1_p"] + d["air_Ex_T_o+0-1_s"]
    out["leak_away_y"] = d["air_Ey_T_o-1+0_p"] + d["air_Ey_T_o-1+0_s"]
    out["ownlane_s_x"] = d["air_Ex_T_o+1+0_s"]
    out["ownlane_s_y"] = d["air_Ey_T_o+0+1_s"]
    out["refl_x"] = d["air_Ex_R_o+0+0_Ex"] + d["air_Ex_R_o+0+0_Ey"]
    out["refl_y"] = d["air_Ey_R_o+0+0_Ex"] + d["air_Ey_R_o+0+0_Ey"]
    out["vertical_x"] = d["air_Ex_T_o+0+0_Ex"] + d["air_Ex_T_o+0+0_Ey"]
    out["vertical_y"] = d["air_Ey_T_o+0+0_Ex"] + d["air_Ey_T_o+0+0_Ey"]
    if want_jones:
        # reflection Jones (rows E_x, E_y out; columns E_x, E_y in)
        Rj = np.array([[amps_R["Ex"][0], amps_R["Ey"][0]], [amps_R["Ex"][1], amps_R["Ey"][1]]], complex)
        sv = np.linalg.svd(Rj, compute_uv=False)
        out["refl_worst_sop"] = float(sv[0] ** 2)
        out["refl_best_sop"] = float(sv[-1] ** 2)
        for k, (a, b) in SOPS.items():
            v = Rj @ np.array([a, b], complex)
            out["refl_sop_%s" % k] = float(np.sum(np.abs(v) ** 2))
        # lane Jones (rows x-lane, y-lane p amplitudes; columns E_x, E_y in)
        Lj = np.array([[amps_T["Ex"][0], amps_T["Ey"][0]], [amps_T["Ex"][1], amps_T["Ey"][1]]], complex)
        lv = np.linalg.svd(Lj, compute_uv=False)
        out["lane_sv_max2"] = float(lv[0] ** 2)
        out["lane_sv_min2"] = float(lv[-1] ** 2)
        out["pdl_db"] = float(10.0 * math.log10(lv[0] ** 2 / lv[-1] ** 2)) if lv[-1] > 0 else float("inf")
        for k, (a, b) in SOPS.items():
            v = Lj @ np.array([a, b], complex)
            out["lane_power_sop_%s" % k] = float(np.sum(np.abs(v) ** 2))
        out["_Rj"] = Rj
        out["_Lj"] = Lj
    del ob
    return out


def measure_sub(rho1, rho2, ph, nG, phi, pols=("p",)):
    """Substrate-side payer at -theta, azimuth phi (0 = x-payer, pi/2 = y-payer)."""
    e1, e2 = eps_pair(np.stack([rho1, rho2]), ph)
    ob = build_sub(nG, e1, e2, ph, phi)
    n = ob.nG
    i00 = idx(ob, 0, 0)
    y = abs(phi - math.pi / 2) < 1e-9
    if not y:      # x-payer: reflected (m,n) leaves with k_t = ((m-1)G, nG)
        o_echo, o_rel, o_away, o_vert = (2, 0), (1, 1), (1, -1), (1, 0)
        o_sp = (0, 0)
    else:          # y-payer: k_t = (mG, (n-1)G)
        o_echo, o_rel, o_away, o_vert = (0, 2), (1, 1), (-1, 1), (0, 1)
        o_sp = (0, 0)
    res = {"nG_realised": int(n)}
    for pol in pols:
        pay = ("y" if y else "x") + pol
        if pol == "p":
            excite_Ex(ob, i00)          # p at phi: (1,0,0,0)
        else:
            excite_Ey(ob, i00)          # s: (0,0,1,0)
        Rx, Ry, Tx, Ty = flux_halves(ob)
        Rt, Tt = ob.RT_Solve(normalize=1, byorder=1)
        res["sub_%s_halves_vs_RT" % pay] = float(max(np.max(np.abs(Rt - Rx - Ry)), np.max(np.abs(Tt - Tx - Ty))))
        res["sub_%s_energy" % pay] = float(np.sum(Rx + Ry + Tx + Ty))

        def pick(o, along_x):
            i = idx(ob, o[0], o[1])
            # p/s labels are relative to the plane of incidence of the ORDER: along x p = y-half
            return (Ry[i], Rx[i]) if along_x else (Rx[i], Ry[i])

        if not y:
            for o, ax in ((o_sp, True), (o_echo, True), (o_rel, False), (o_away, False)):
                pp, ss = pick(o, ax)
                res["sub_%s_R_%s_p" % (pay, otag(*o))] = float(pp)
                res["sub_%s_R_%s_s" % (pay, otag(*o))] = float(ss)
        else:
            for o, ax in ((o_sp, False), (o_echo, False), (o_rel, True), (o_away, True)):
                pp, ss = pick(o, ax)
                res["sub_%s_R_%s_p" % (pay, otag(*o))] = float(pp)
                res["sub_%s_R_%s_s" % (pay, otag(*o))] = float(ss)
        iv = idx(ob, o_vert[0], o_vert[1])
        res["sub_%s_R_%s_Ex" % (pay, otag(*o_vert))] = float(Ry[iv])
        res["sub_%s_R_%s_Ey" % (pay, otag(*o_vert))] = float(Rx[iv])
        res["sub_%s_T_%s_Ex" % (pay, otag(*o_vert))] = float(Ty[iv])
        res["sub_%s_T_%s_Ey" % (pay, otag(*o_vert))] = float(Tx[iv])
        res["sub_%s_echo" % pay] = res["sub_%s_R_%s_p" % (pay, otag(*o_echo))] + res["sub_%s_R_%s_s" % (pay, otag(*o_echo))]
        res["sub_%s_relaunch" % pay] = res["sub_%s_R_%s_p" % (pay, otag(*o_rel))] + res["sub_%s_R_%s_s" % (pay, otag(*o_rel))]
        res["sub_%s_relaunch_away" % pay] = res["sub_%s_R_%s_p" % (pay, otag(*o_away))] + res["sub_%s_R_%s_s" % (pay, otag(*o_away))]
    del ob
    return res


def measure_rung(rho1, rho2, ph, nG, want_s=True, want_y=True):
    """One rung: A (E_x, E_y), x-payer p (+ s), y-payer p. Objects one at a time."""
    t0 = time.time()
    A = measure_air(rho1, rho2, ph, nG)
    Bx = measure_sub(rho1, rho2, ph, nG, 0.0, pols=("p", "s") if want_s else ("p",))
    By = measure_sub(rho1, rho2, ph, nG, math.pi / 2, pols=("p",)) if want_y else {}
    r = {}
    r.update({k: v for k, v in A.items() if not k.startswith("_")})
    r.update(Bx)
    r.update(By)
    r["echo_x"] = Bx["sub_xp_echo"]
    r["relaunch_x"] = Bx["sub_xp_relaunch"]
    r["relaunch_away_x"] = Bx["sub_xp_relaunch_away"]
    if want_y:
        r["echo_y"] = By["sub_yp_echo"]
        r["relaunch_y"] = By["sub_yp_relaunch"]
        r["relaunch_away_y"] = By["sub_yp_relaunch_away"]
    # C7 reciprocity reading: T_air(payer p) = eta_x + P_p[E_y -> (+1,0)]
    tx = Bx["sub_xp_T_o+1+0_Ex"]
    ty = Bx["sub_xp_T_o+1+0_Ey"]
    r["recip_gap_x"] = abs(tx - A["eta_x"])
    r["recip_gap_y"] = abs(ty - A["air_Ey_T_o+1+0_p"])
    r["recip_gap"] = abs(tx + ty - A["eta_x"] - A["air_Ey_T_o+1+0_p"])
    r["t_s"] = time.time() - t0
    r["peak_GB"] = peak_gb()
    r["_Rj"], r["_Lj"] = A.get("_Rj"), A.get("_Lj")
    return r


def energy_worst(r):
    es = [v for k, v in r.items() if k.endswith("_energy") and isinstance(v, float)]
    return max(abs(e - 1.0) for e in es) if es else float("nan")


# ---------------------------------------------------------------------------
# gates (Section 5)
GATES = dict(G1=0.970, G1conv=0.005, G2=0.0050, G3=0.0080, G4a=0.0020, G4b=0.0020, G5=50.0)


def eval_gates(r, r_prev, mf, stack_ok, unsup, sym_err):
    g = {}
    eta = min(r["eta_x"], r["eta_y"])
    g["G1_eta_min"] = eta
    g["G1_pass"] = bool(eta >= GATES["G1"])
    if r_prev is not None:
        dl = max(abs(r["eta_x"] - r_prev["eta_x"]), abs(r["eta_y"] - r_prev["eta_y"]))
        g["G1_conv_delta"] = dl
        g["G1_conv_flag"] = "OK" if dl <= GATES["G1conv"] else "UNCONVERGED"
    else:
        g["G1_conv_delta"] = None
        g["G1_conv_flag"] = "NO_PREVIOUS_RUNG"
    leak = max(r["leak_x"], r["leak_y"])
    g["G2_leak_max"] = leak
    g["G2_pass"] = bool(leak <= GATES["G2"])
    g["G3_refl_worst_sop"] = r["refl_worst_sop"]
    g["G3_pass"] = bool(r["refl_worst_sop"] <= GATES["G3"])
    e = max(r["echo_x"], r.get("echo_y", r["echo_x"]))
    g["G4a_echo_max"] = e
    g["G4a_pass"] = bool(e <= GATES["G4a"])
    rl = max(r["relaunch_x"], r.get("relaunch_y", r["relaunch_x"]))
    g["G4b_relaunch_max"] = rl
    g["G4b_pass"] = bool(rl <= GATES["G4b"])
    g["G5_min_feature_nm"] = mf["min_feature_nm"]
    g["G5_stack_ok"] = bool(stack_ok)
    g["G5_unsupported_px"] = int(unsup)
    g["G5_sym_err"] = float(sym_err)
    g["G5_pass"] = bool(mf["min_feature_nm"] >= GATES["G5"] and stack_ok and unsup == 0 and sym_err == 0.0)
    g["gates_passed"] = int(sum(bool(g[k]) for k in ("G1_pass", "G2_pass", "G3_pass", "G4a_pass", "G4b_pass", "G5_pass")))
    g["valid_energy_ok"] = bool(energy_worst(r) <= 1e-6)
    return g


def stack_ok_check(P):
    return bool(abs(float(P["H1_nm"]) - 525.0) < 1e-9 and abs(float(P["H2_nm"]) - 450.0) < 1e-9
                and (P["material1"] == "a-Si" or P["material1"].startswith("a-Si_k"))   # [261013] opt-in k names
                and P["material2"] == "Si3N4"
                and P["fill1"] == "SiO2" and P["fill2"] == "air"
                and P["fill1"] not in ("air", "Air"))


# ---------------------------------------------------------------------------
def morph4(b, k):
    """4-neighbour erode (k < 0) / dilate (k > 0) of a boolean raster by |k| pixels (periodic)."""
    b = np.asarray(b, bool)
    for _ in range(abs(int(k))):
        nb = [np.roll(b, 1, 0), np.roll(b, -1, 0), np.roll(b, 1, 1), np.roll(b, -1, 1)]
        if k > 0:
            b = b | nb[0] | nb[1] | nb[2] | nb[3]
        else:
            b = b & nb[0] & nb[1] & nb[2] & nb[3]
    return b


def kron2(a):
    return np.kron(np.asarray(a, float), np.ones((2, 2)))


def parse_ints(txt):
    return [int(float(v)) for v in str(txt).replace(";", ",").split(",") if v.strip()]


def full_report(rho1, rho2, P, ph, od, out, partial_path=None):
    """Ladder + gates + band + bias + kron2 + pol on the hard design. numpy backend."""
    set_bk("numpy")
    rungs = parse_ints(P["nG_eval"])
    out["rungs_requested"] = rungs
    px = ph.px_nm
    out.update(mf_report(rho1, rho2, px))
    unsup = unsupported_px(rho1, rho2, P["fill1"])
    sym_err = float(max(np.max(np.abs(rho1 - rho1.T)), np.max(np.abs(rho2 - rho2.T))))
    out["sym_err"] = sym_err
    out["fill_L1"] = float(np.mean(rho1))
    out["fill_L2"] = float(np.mean(rho2))
    prev, last = None, None
    for n in rungs:
        est = 4.6 * (n / 889.0) ** 1.95 * 0.55
        if not ram_guard(est, "rung nG %d" % n):
            out["ng%d_skipped" % n] = "RAM guard (limit SIM_MAX_RAM_PCT)"
            continue
        try:
            r = measure_rung(rho1, rho2, ph, n)
        except Exception as e:                                    # noqa: BLE001  (e.g. MemoryError at a high rung)
            import traceback
            traceback.print_exc()
            out["ng%d_error" % n] = "%s: %s" % (type(e).__name__, e)
            log("WARN: rung nG %d failed: %s" % (n, e))
            if partial_path:
                write_json(partial_path, out)
            continue
        out["ng%d_energy_worst_dev" % n] = energy_worst(r)
        for k, v in r.items():
            if k.startswith("_"):
                continue
            out["ng%d_%s" % (n, k)] = v
        out["ng%d_K" % n] = r.get("K")
        log("rung nG %d (realised %d, K %s): eta_x %.6f eta_y %.6f leak %.5f/%.5f Rworst %.5f "
            "echo %.5f/%.5f rel %.5f/%.5f  [%.0f s, peak %.2f GB]"
            % (n, r["nG_realised"], r.get("K"), r["eta_x"], r["eta_y"], r["leak_x"], r["leak_y"],
               r["refl_worst_sop"], r["echo_x"], r["echo_y"], r["relaunch_x"], r["relaunch_y"],
               r["t_s"], r["peak_GB"]))
        prev, last = last, (n, r)
        if partial_path:
            out["status"] = "IN_PROGRESS_REPORT"
            write_json(partial_path, out)
    if last is not None:
        n_fin, r_fin = last
        pr = prev[1] if prev is not None else None
        gt = eval_gates(r_fin, pr, {"min_feature_nm": out["min_feature_nm"]}, stack_ok_check(P), unsup, sym_err)
        out.update(gt)
        out["final_rung"] = n_fin
        out["G1_conv_rungs"] = [prev[0], n_fin] if prev is not None else None
        out["pdl_db"] = r_fin.get("pdl_db")
    # The optional reports below are information, not the result: each runs in its own guard so
    # that a failure after hours of optimisation is recorded (<tag>_error) and never loses the
    # design, the ladder or the gates already written.
    def _guarded(tag, fn):
        try:
            fn()
        except Exception as e:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            out["%s_error" % tag] = "%s: %s" % (type(e).__name__, e)
            log("WARN: report section %s failed: %s" % (tag, e))
        if partial_path:
            write_json(partial_path, out)

    # band (forward A only, Track A wavelengths)
    def _band():
        lams = (1.546, 1.5476, 1.550, 1.55241, 1.55402)
        nb = int(P["nG_band"])
        if ram_guard(4.6 * (nb / 889.0) ** 1.95 * 0.55, "band nG %d" % nb):
            etas = []
            etas_y = []
            for lam in lams:
                a = measure_air(rho1, rho2, ph, nb, lam=lam, want_jones=True)
                tg = "lam%.2f" % (lam * 1e3)
                for k in ("eta_x", "eta_y", "leak_x", "leak_y", "refl_x", "refl_y", "refl_worst_sop"):
                    out["%s_%s" % (tg, k)] = a[k]
                etas.append(a["eta_x"])
                etas_y.append(a["eta_y"])
            out["band_spread"] = float(max(etas) - min(etas))
            if _LANES_BOTH:             # [261008 li_v2 F1] y-lane is not the mirror of the x-lane: spread = worst of both
                out["band_spread_x"] = out["band_spread"]
                out["band_spread_y"] = float(max(etas_y) - min(etas_y))
                out["band_spread"] = max(out["band_spread_x"], out["band_spread_y"])
            out["band_flag"] = bool(out["band_spread"] > float(P["band_flag_tol"]))
            out["band_flag_text"] = "ARTEFACT SUSPECT" if out["band_flag"] else "ok"
            out["band_nG"] = nb
            log("band nG %d: eta_x spread %.5f -> %s" % (nb, out["band_spread"], out["band_flag_text"]))

    # edge bias (G6) on the kron x2 raster
    def _bias():
        nbias = int(P["nG_bias"])
        if ram_guard(4.6 * (nbias / 889.0) ** 1.95 * 0.55, "bias nG %d" % nbias):
            b1, b2 = kron2(rho1) > 0.5, kron2(rho2) > 0.5
            ph2 = Phys(dict(P, nxy=2 * ph.N))
            variants = [("nom", b1, b2),
                        ("L1_m1", morph4(b1, -1), b2), ("L1_p1", morph4(b1, 1), b2),
                        ("L2_m1", b1, morph4(b2, -1)), ("L2_p1", b1, morph4(b2, 1)),
                        ("both_m1", morph4(b1, -1), morph4(b2, -1)), ("both_p1", morph4(b1, 1), morph4(b2, 1))]
            for tag, v1, v2 in variants:
                r = measure_rung(v1.astype(float), v2.astype(float), ph2, nbias, want_s=False, want_y=True)
                for k in ("eta_x", "eta_y", "leak_x", "leak_y", "refl_worst_sop", "echo_x", "echo_y",
                          "relaunch_x", "relaunch_y"):
                    out["bias_%s_%s" % (tag, k)] = r[k]
                log("bias %-8s eta_x %.5f eta_y %.5f leak %.5f Rworst %.5f echo %.5f rel %.5f"
                    % (tag, r["eta_x"], r["eta_y"], r["leak_x"], r["refl_worst_sop"], r["echo_x"], r["relaunch_x"]))
            out["bias_nG"] = nbias

    # C9 raster refinement
    def _kron2():
        n2 = int(P["nG_kron2"])
        if ram_guard(4.6 * (n2 / 889.0) ** 1.95 * 0.55, "kron2 nG %d" % n2):
            ph2 = Phys(dict(P, nxy=2 * ph.N))
            r2 = measure_rung(kron2(rho1), kron2(rho2), ph2, n2)
            r1 = measure_rung(rho1, rho2, ph, n2)
            for k in ("eta_x", "eta_y", "leak_x", "leak_y", "refl_worst_sop", "echo_x", "echo_y", "relaunch_x", "relaunch_y"):
                out["ng%d_kron2_%s" % (n2, k)] = r2[k]
                out["ng%d_kron2_d_%s" % (n2, k)] = r2[k] - r1[k]
            log("kron2 nG %d: d_eta_x %+.2e d_leak_x %+.2e d_echo_x %+.2e"
                % (n2, r2["eta_x"] - r1["eta_x"], r2["leak_x"] - r1["leak_x"], r2["echo_x"] - r1["echo_x"]))

    # Pol FMM bracket (second opinion, forward A only)
    def _pol():
        npol = int(P["nG_pol"])
        if ram_guard(4.6 * (npol / 889.0) ** 1.95 * 0.55, "pol nG %d" % npol):
            a = measure_air(rho1, rho2, ph, npol, fmm="pol", want_jones=False)
            for k in ("eta_x", "eta_y", "leak_x", "leak_y", "refl_x", "refl_y"):
                out["ng%d_pol_%s" % (npol, k)] = a[k]
            log("Pol FMM nG %d: eta_x %.6f eta_y %.6f" % (npol, a["eta_x"], a["eta_y"]))

    if int(P["report_band"]):
        _guarded("band", _band)
    if int(P["report_bias"]):
        _guarded("bias", _bias)
    if int(P["report_kron2"]):
        _guarded("kron2", _kron2)
    if int(P["report_pol"]):
        _guarded("pol", _pol)
    return out


# ---------------------------------------------------------------------------
# Adam optimisation with ng_schedule, beta schedule, lw ramp, best-binarised tracking
def optimise(P, od, stop_after=None, quiet=False):
    set_bk("autograd")
    M = Model(P)
    ph, N = M.ph, M.N
    iters = int(P["iters"])
    lr = float(P["lr"])
    ng_sched = parse_sched(P["ng_schedule"], "ng_schedule", True)
    nG_fin = ng_sched[-1][1]
    lw_start = float(P["lw_start_frac"])
    bb_from = float(P["best_bin_from"])
    bb_every = max(1, int(P["best_bin_every"]))
    ck_every = int(P["ckpt_every"])
    ck_every_s = float(P["ckpt_every_s"])
    os.makedirs(od, exist_ok=True)
    ck_path = os.path.join(od, "ckpt_input2d.npz")
    log_path = os.path.join(od, "opt_log.dat")
    terms_path = os.path.join(od, "opt_terms.tsv")
    fp = fingerprint(P)

    src = None
    if P["seed_src_npy"]:
        src = np.load(P["seed_src_npy"])
    if P["init_npy"]:
        x0 = np.load(P["init_npy"]).astype(float)
        if x0.shape != (2, N, N):
            raise ValueError("init_npy shape %s != (2,%d,%d)" % (x0.shape, N, N))
        if int(P["init_soft"]):
            x0 = 0.15 + 0.70 * x0
        x0 = 0.5 * (x0 + np.transpose(x0, (0, 2, 1)))
        seed_label = "init_npy"
    else:
        x0 = make_seed(P["seed"], N, int(P["seed_rng"]), src)
        seed_label = P["seed"]
    seed_md5 = _md5_arr(x0)
    st = dict(x=x0.copy(), m=np.zeros_like(x0), v=np.zeros_like(x0), t=0,
              best_x=x0.copy(), best_J=-1e30, best_it=0, best_ok=0)
    resumed = False
    if ck_every and os.path.isfile(ck_path):
        try:
            st_l, fp_l = ckpt_load(ck_path)
            if fp_l == fp and st_l["x"].shape == x0.shape and st_l["t"] < iters:
                st = st_l
                resumed = True
                log("CHECKPOINT RESUME -- from iteration %d/%d" % (st["t"] + 1, iters))
            else:
                log("checkpoint discarded (fingerprint equal %s, shape equal %s, t %s < iters %d)"
                    % (fp_l == fp, st_l["x"].shape == x0.shape, st_l["t"], iters))
        except Exception as e:
            log("checkpoint unreadable (%s) -- starting from zero" % e)
    cum0 = 0.0
    hdr_log = "Iter\tJ\teta\tleak\trefl\techo\trel\tlw\tbeta\tnG\tcum_s\n"
    hdr_trm = "Iter\teta_x\teta_y\tleak\trefl\techo\trel\tP_lw\tramp\tbeta\tnG\n"
    if _LANES_BOTH:                 # [261008 li_v2 F1] echo / rel above are the lane means; both payers appended
        hdr_trm = hdr_trm.rstrip("\n") + "\techo_x\techo_y\trel_x\trel_y\n"
    if resumed:
        log_truncate(log_path, st["t"])
        log_truncate(terms_path, st["t"])
        try:
            with io.open(log_path, encoding="utf-8") as fh:
                ls = [ln for ln in fh.read().split("\n") if ln.strip()]
            if len(ls) > 1:
                cum0 = float(ls[-1].split("\t")[-1])
        except Exception:
            cum0 = 0.0
    else:
        with io.open(log_path, "w", encoding="utf-8") as fh:
            fh.write(hdr_log)
        with io.open(terms_path, "w", encoding="utf-8") as fh:
            fh.write(hdr_trm)
    if not quiet:
        log("N %d px %.3f nm  filt %.1f nm (%.1f px)  lw gate %.0f nm (KG r %.1f px)  iters %d  ng_schedule %s  beta %s"
            % (N, ph.px_nm, float(P["filt_radius_nm"]), float(P["filt_radius_nm"]) / ph.px_nm,
               float(P["lw_gate_nm"]), max(1.0, float(P["lw_gate_nm"]) / ph.px_nm), iters,
               ", ".join("%g:%d" % fn for fn in ng_sched), ", ".join("%g:%g" % fb for fb in M.beta_sched)))
        log("seed %s (md5 %s)  resumed %s" % (seed_label, seed_md5[:12], resumed))
    t_loop0 = time.time()
    t_last_ck = time.time()
    x, m, v = st["x"], st["m"], st["v"]
    t_start = st["t"] + 1
    best_x, best_J, best_it, best_ok = st["best_x"], st["best_J"], st["best_it"], st["best_ok"]
    b1, b2, eps = 0.9, 0.999, 1e-8
    stopped = False
    last_terms = {}
    n_nonfinite = 0
    last_J = None
    for t in range(t_start, iters + 1):
        frac = (t - 1) / float(iters)
        beta = sched_at(M.beta_sched, frac)
        nG = sched_at(ng_sched, frac)
        ramp = 0.0 if frac < lw_start else min(1.0, (frac - lw_start) / max(1e-9, 1.0 - lw_start))
        J, g, terms = M.vg(x, nG, beta, ramp)
        if not (math.isfinite(J) and bool(np.all(np.isfinite(g)))):
            n_nonfinite += 1
            log("WARN: it %d non-finite J or gradient (J %s) -- update skipped (%d in a row)" % (t, J, n_nonfinite))
            if n_nonfinite >= 3:
                raise RuntimeError("3 consecutive non-finite objective/gradient evaluations at iteration %d" % t)
            continue
        n_nonfinite = 0
        last_terms = terms
        last_J = J
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        x = np.clip(x + lr * mh / (np.sqrt(vh) + eps), 0.0, 1.0)
        cum = cum0 + (time.time() - t_loop0)
        eta = math.sqrt(max(terms["eta_x"] * terms["eta_y"], 0.0))
        with io.open(log_path, "a", encoding="utf-8") as fh:
            fh.write("%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%g\t%d\t%.1f\n"
                     % (t, J, eta, terms["leak"], terms["refl"], terms["echo"], terms["rel"],
                        terms["lw"], beta, nG, cum))
        with io.open(terms_path, "a", encoding="utf-8") as fh:
            ln_trm = ("%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.4f\t%g\t%d\n"
                      % (t, terms["eta_x"], terms["eta_y"], terms["leak"], terms["refl"], terms["echo"],
                         terms["rel"], terms["lw"], ramp, beta, nG))
            if _LANES_BOTH:
                ln_trm = (ln_trm.rstrip("\n") + "\t%.9f\t%.9f\t%.9f\t%.9f\n"
                          % (terms["echo_x"], terms["echo_y"], terms["rel_x"], terms["rel_y"]))
            fh.write(ln_trm)
        if not quiet:
            log("it %4d/%d J %.6f eta_x %.5f eta_y %.5f leak %.5f refl %.5f echo %.5f rel %.5f lw %.4f ramp %.2f beta %g nG %d  %.0f s"
                % (t, iters, J, terms["eta_x"], terms["eta_y"], terms["leak"], terms["refl"],
                   terms["echo"], terms["rel"], terms["lw"], ramp, beta, nG, cum))
            if _LANES_BOTH:
                log("         payers: echo x %.5f y %.5f  rel x %.5f y %.5f  (echo / rel above = lane means)"
                    % (terms["echo_x"], terms["echo_y"], terms["rel_x"], terms["rel_y"]))
        # best-binarised iterate (feasibility-first, as scd_c_grating.optimize since 260926)
        if (t / float(iters) >= bb_from) and (t % bb_every == 0 or t == iters):
            rh = hard_pair(x, M.KF, M.beta_final)
            Jb, tb = M.J_rho(rh, nG_fin)
            mfb = mf_report(rh[0], rh[1], ph.px_nm)["min_feature_nm"]
            ok = int(mfb >= float(P["min_feature_nm"]))
            if best_J < -1e29 or (ok, Jb) > (best_ok, best_J):
                best_x, best_J, best_it, best_ok = x.copy(), Jb, t, ok
            if not quiet:
                log("   best-bin check it %d: J_bin %.6f  min feature %.1f nm (ok %d)   best so far it %d J %.6f ok %d"
                    % (t, Jb, mfb, ok, best_it, best_J, best_ok))
        if ck_every and t < iters and (t % ck_every == 0 or (time.time() - t_last_ck) >= ck_every_s):
            ckpt_save(ck_path, dict(x=x, m=m, v=v, t=t, best_x=best_x, best_J=best_J,
                                    best_it=best_it, best_ok=best_ok), fp)
            t_last_ck = time.time()
        if stop_after is not None and t >= stop_after:
            stopped = True
            break
    res = dict(x=x, best_x=best_x, best_J=best_J, best_it=best_it, best_ok=best_ok, M=M, fp=fp,
               seed_label=seed_label, seed_md5=seed_md5, resumed=resumed, stopped=stopped,
               last_terms=last_terms, last_J=last_J, cum=cum0 + (time.time() - t_loop0), ng_fin=nG_fin,
               iters_run=(t if t_start <= iters else t_start - 1))
    return res


# ---------------------------------------------------------------------------
# TMM (Macleod characteristic matrix, exp(-i w t) convention, exit/decay branch Im(cos) >= 0)
def tmm_RT(layers, n_in, n_out, theta_in, pol, lam):
    s0 = n_in * math.sin(theta_in)

    def cos_t(n):
        c = np.sqrt(1.0 - (s0 / n) ** 2 + 0j)
        return -c if c.imag < 0 else c

    def eta(n):
        c = cos_t(n)
        return n * c if pol == "s" else n / c

    Mt = np.eye(2, dtype=complex)
    for n, d in layers:
        delta = 2 * math.pi / lam * n * d * cos_t(n)
        e = eta(n)
        Mj = np.array([[np.cos(delta), -1j * np.sin(delta) / e], [-1j * e * np.sin(delta), np.cos(delta)]])
        Mt = Mt @ Mj
    e0, es = eta(n_in), eta(n_out)
    B, C = Mt @ np.array([1.0, es], complex)
    r = (e0 * B - C) / (e0 * B + C)
    t = 2 * e0 / (e0 * B + C)
    R = float(abs(r) ** 2)
    T = float((es.real if hasattr(es, "real") else es) / e0.real * abs(t) ** 2)
    return R, T


class _Skip(Exception):
    """control not selected by selftest_only"""


def smooth_dir(N, rng, sig_px=6.0, layers=(0, 1)):
    """Random sigma_d-symmetric direction for the C4 finite differences (shape (2,N,N), unit L2 norm):
    white noise filtered with a Gaussian of std sig_px pixels (the conic filter passes it, so the
    directional derivative stands well above the forward-solve noise), diagonal-symmetrised."""
    k = np.fft.fftfreq(N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    gk = np.exp(-2.0 * (np.pi * sig_px) ** 2 * (KX ** 2 + KY ** 2))
    d = np.zeros((2, N, N))
    for L in layers:
        d[L] = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * gk))
    d = 0.5 * (d + np.transpose(d, (0, 2, 1)))
    return d / np.linalg.norm(d)


def pick_dir(g, rng, N, pool=24):
    """The smooth sigma_d direction with the largest |cos| to g among `pool` random candidates. Measured at
    the S1 start, nG 201 (261006): a random smooth direction has median |cos| 0.050 (10 % below 0.01, 20 %
    below 0.02, 1500 draws) while the forward-solve noise of J is 4.4e-11 (std of 8 evaluations), so a
    direction with |cos| 0.003 read rel 7.7e-5 (1.1e-4 at h 2.5e-3) for a CORRECT gradient. The best of 24
    has |cos| 0.11-0.26 (median 0.16, 62 picks): noise error <= ~1e-5, and at least 93 % of the squared
    norm of the direction is still orthogonal to g."""
    gn = float(np.linalg.norm(g))
    best, bc = None, -1.0
    for _ in range(int(pool)):
        d = smooth_dir(N, rng)
        c = abs(float(np.sum(g * d))) / max(gn, 1e-300)
        if c > bc:
            best, bc = d, c
    return best


def fd_rich(fun, x, d, h1, h2):
    """Central differences of fun along d at steps h1 and h2 (h1 > h2) and their Richardson combination
    (h1/h2)^2 D(h2) - D(h1) over (h1/h2)^2 - 1 (the h^2 error term cancels). Returns (D(h1), D(h2), Richardson)."""
    q = (h1 / h2) ** 2
    d1 = (float(fun(x + h1 * d)) - float(fun(x - h1 * d))) / (2.0 * h1)
    d2 = (float(fun(x + h2 * d)) - float(fun(x - h2 * d))) / (2.0 * h2)
    return d1, d2, (q * d2 - d1) / (q - 1.0)


def run_c4(P, nG_list, rec=None, rng=None, h_list=(2.5e-3, 1.25e-3), per_term_max_nG=201, beta=8.0, x_start=None):
    """C4: central finite difference vs the assembled AD gradient (object-sequential VJP).

    DIRECTIONS (changed 261006, verifier fix 3): smooth random sigma_d directions (white noise filtered
    with a Gaussian of 6 px, both layers, diagonal-symmetrised, unit norm; the best-overlap one of 24
    candidates per direction, see pick_dir; fixed RNG seed per nG, so the selftest and the in-job call use
    the same ones), Richardson on h_list = (2.5e-3, 1.25e-3). The old rule (FD only along the gradient's
    own direction, h 1e-3) cannot see a gradient error orthogonal to g. Pass: rel <= 1e-4 for every direction.
      FULL  : the whole J (all five optical terms and the linewidth surrogate, ramp forced to 1) at every
              nG in nG_list: 3 smooth directions + the gradient direction (kept, strongest signal) at
              nG <= 401, 2 smooth directions at larger nG (cost: 4 forward evaluations per direction)
      terms : at nG <= per_term_max_nG each term alone (eta, leak, refl, echo, relaunch, lw): its own
              gradient direction and one smooth direction picked for that term ("EVERY term active" -- a
              gated term cannot hide)
    x_start: (2,256,256) raw start to test instead of the S1 (xsec_min) default; rng is unused (kept
    for the call signature). Returns {nG: dict(...)}. rec(cid, name, value, tol, ok, note) is the
    selftest recorder; None for the in-job call (smoke job)."""
    set_bk("autograd")
    M = Model(dict(P, nxy=256))
    out = {}
    Z = dict(w_leak=0.0, w_refl=0.0, w_echo=0.0, w_rel=0.0)
    h1, h2 = float(h_list[0]), float(h_list[1])

    def W(**kw):
        w = dict(Z)
        w.update(kw)
        return w

    for nG2 in nG_list:
        if nG2 >= 901 and not ram_guard(6.0, "C4 nG %d" % nG2):
            if rec is not None:
                rec("C4@%d" % nG2, "AD vs FD", "SKIPPED (RAM guard)", "-", False)
            out[nG2] = dict(skipped=True)
            continue
        # start: S1 (xsec_min) when the 1-D density is staged -- all five optical terms are of
        # order 1e-2..1 there (eta 0.40, echo 0.29, refl 0.19, leak 0.12, rel 0.016 at beta 8)
        if x_start is not None:
            x = np.asarray(x_start, float)
        else:
            src_c4 = None
            if P["seed_src_npy"] and os.path.isfile(P["seed_src_npy"]):
                src_c4 = np.load(P["seed_src_npy"])
            if src_c4 is not None:
                x = make_seed("xsec_min", 256, 0, src_c4)
            else:
                x = np.stack([make_seed("rand", 256, 3 + L, None)[L] for L in (0, 1)])
        x = 0.5 * (x + np.transpose(x, (0, 2, 1)))

        def dens(xx):
            return dens_pair(xx, M.KF, beta)

        def f_full(xx):
            rho = dens(xx)
            return (part_A(rho, nG2, M.ph, M.w) + part_B(rho, nG2, M.ph, M.w)
                    - M.w_lw * 1.0 * part_LW(rho, M.KG, beta))

        t0 = time.time()
        J0, g0, terms0 = M.vg(x, nG2, beta, 1.0)
        t_vg = time.time() - t0
        pk = peak_gb()
        log("   C4 nG %d: value_and_grad %.1f s, peak %.2f GB, J %.6f" % (nG2, t_vg, pk, J0))
        gn = float(np.linalg.norm(g0))
        drng = np.random.default_rng(20261006 + int(nG2))
        dirs = [("smooth%d" % i, pick_dir(g0, drng, M.N)) for i in range(3 if nG2 <= 401 else 2)]
        if nG2 <= 401:
            dirs.append(("gradient", g0 / gn))
        rows = {}
        for dname, d in dirs:
            ad = float(np.sum(g0 * d))
            den = max(abs(ad), 1e-300)
            D1, D2, Dr = fd_rich(f_full, x, d, h1, h2)
            rows[dname] = dict(ad=ad, cos=ad / gn, fd_rich=Dr, rel_h1=abs(D1 - ad) / den, rel_h2=abs(D2 - ad) / den,
                               rel_rich=abs(Dr - ad) / den)
            log("   C4 nG %d FULL along %-9s g.d %+.9e  Richardson %+.9e  rel %.2e  (h %.2e: %.1e, h %.2e: %.1e)  cos(d,g) %+.3f"
                % (nG2, dname, ad, Dr, rows[dname]["rel_rich"], h1, rows[dname]["rel_h1"], h2, rows[dname]["rel_h2"], ad / gn))
        sym_g = float(np.max(np.abs(g0 - np.transpose(g0, (0, 2, 1)))))
        rel_main = max(r["rel_rich"] for r in rows.values())
        rels = {("%.2e" % h1): max(r["rel_h1"] for r in rows.values()), ("%.2e" % h2): max(r["rel_h2"] for r in rows.values())}
        res = dict(rel_full=float(rel_main), rel_by_h={k: float(v) for k, v in rels.items()},
                   rel_dirs={k: float(v["rel_rich"]) for k, v in rows.items()},
                   cos_dirs={k: float(v["cos"]) for k, v in rows.items()}, n_dirs=len(rows),
                   g_norm=gn, vg_s=t_vg, peak_GB=pk, grad_sym_err=sym_g, J=J0, eta_x=terms0["eta_x"],
                   eta_y=terms0["eta_y"], leak=terms0["leak"], refl=terms0["refl"], echo=terms0["echo"],
                   rel=terms0["rel"], lw=terms0["lw"], passed_full=bool(rel_main <= 1e-4))
        if rec is not None:
            rec("C4@%d" % nG2, "AD vs central FD, FULL J, %d directions (smooth random sigma_d%s), Richardson h %.2e/%.2e (lw ramp 1)"
                % (len(rows), " + gradient" if nG2 <= 401 else "", h1, h2), float(rel_main),
                "<=1e-4", rel_main <= 1e-4,
                "vg %.0f s, peak %.2f GB, |g| %.3e, grad sigma_d err %.1e, per direction %s, terms eta_x %.4f echo %.4f rel %.4f lw %.4f"
                % (t_vg, pk, gn, sym_g, ", ".join("%s %.1e" % (k, v["rel_rich"]) for k, v in rows.items()),
                   terms0["eta_x"], terms0["echo"], terms0["rel"], terms0["lw"]))
        if nG2 <= per_term_max_nG:
            def f_eta(xx):
                return part_A(dens(xx), nG2, M.ph, W())

            def f_leak(xx):
                rho = dens(xx)
                return part_A(rho, nG2, M.ph, W(w_leak=1.0)) - part_A(rho, nG2, M.ph, W())

            def f_refl(xx):
                rho = dens(xx)
                return part_A(rho, nG2, M.ph, W(w_refl=1.0)) - part_A(rho, nG2, M.ph, W())

            def f_echo(xx):
                return part_B(dens(xx), nG2, M.ph, W(w_echo=1.0))

            def f_rel(xx):
                return part_B(dens(xx), nG2, M.ph, W(w_rel=1.0))

            def f_lw(xx):
                return part_LW(dens(xx), M.KG, beta)

            trels, trels_own, trels_rand = {}, {}, {}
            for tname, fn in (("eta", f_eta), ("leak", f_leak), ("refl", f_refl), ("echo", f_echo),
                              ("rel", f_rel), ("lw", f_lw)):
                _v, gt = value_and_grad(fn)(x)
                gt = np.asarray(gt)
                gtn = float(np.linalg.norm(gt))
                _a, _b, fdt = fd_rich(fn, x, gt / gtn, h1, h2)
                trels_own[tname] = abs(fdt - gtn) / gtn
                dt = pick_dir(gt, drng, M.N)
                adt = float(np.sum(gt * dt))
                if abs(adt) >= 0.02 * gtn:
                    _a, _b, fdr = fd_rich(fn, x, dt, h1, h2)
                    trels_rand[tname] = abs(fdr - adt) / abs(adt)
                else:                                   # no usable overlap with this term's gradient: no denominator
                    trels_rand[tname] = float("nan")
                trels[tname] = max(v for v in (trels_own[tname], trels_rand[tname]) if v == v)
                log("   C4 nG %d term %-5s |g| %.4e  rel own-gradient %.2e  rel smooth %.2e (cos %+.3f)"
                    % (nG2, tname, gtn, trels_own[tname], trels_rand[tname], adt / gtn))
            worst_t = max(trels.values())
            res["rel_terms"] = {k: float(v) for k, v in trels.items()}
            res["rel_terms_own"] = {k: float(v) for k, v in trels_own.items()}
            res["rel_terms_smooth"] = {k: float(v) for k, v in trels_rand.items()}
            res["rel_terms_max"] = float(worst_t)
            res["passed_terms"] = bool(worst_t <= 1e-4)
            if rec is not None:
                rec("C4t@%d" % nG2, "AD vs FD per term (eta leak refl echo rel lw): own gradient + one smooth direction, Richardson",
                    float(worst_t), "<=1e-4", worst_t <= 1e-4, "per term " + ", ".join("%s %.1e" % (k, v) for k, v in trels.items()))
        res["passed"] = bool(res["passed_full"] and res.get("passed_terms", True))
        out[nG2] = res
    return out


def run_selftest(P, level, od):
    """Controls C1-C13 of the design. Writes selftest_input2d_<md5_12>.json in selftest_dir
    (default Output_folder). Returns True when every pass/fail control passes."""
    ctl = []

    def rec(cid, name, value, tol, ok, note=""):
        ctl.append(dict(id=cid, name=name, value=_san(value), tol=tol, passed=bool(ok), note=note))
        log("%-4s %-52s value %s  tol %s  -> %s %s" % (cid, name, ("%.3e" % value) if isinstance(value, float) else value,
                                                        tol, "PASS" if ok else "FAIL", note))

    ph = Phys(P)
    G.set_substrate(float(P["n_sub"]))
    G.set_period(float(P["period_nm"]))
    t_all = time.time()
    rng = np.random.default_rng(20261005)
    # selftest_only = "C9,C10,C7": run only those controls (C8, C5, C6, C3 data and C12/C13 are cheap
    # and always run). A partial run writes selftest_input2d_partial_<md5>.json, which never
    # satisfies the optimisation gate -- only a full run does.
    only = set(s.strip().upper() for s in str(P["selftest_only"]).split(",") if s.strip())

    def want(cid):
        return (not only) or cid.upper() in only

    # ---- C8 alias guard (assert on a few builds) -------------------------------------------
    try:
        # lattice-only (Init_Setup is cheap; GridLayer_geteps at nG 3409 is not -- 6 GB measured):
        # the guard is evaluated on the order list, and the builders' own guard is exercised on a
        # grid that is too coarse (raises before any matrix is built)
        set_bk("numpy")
        from grcwa.kbloch import Lattice_Reciprocate, Lattice_getG
        Lk1, Lk2 = Lattice_Reciprocate([ph.P, 0.0], [0.0, ph.P])
        ok = True
        Ks = []
        for nG in (201, 401, 601, 901, 1793, 2402, 3409):
            Gl, _n = Lattice_getG(nG, Lk1, Lk2, method=0)
            K = int(np.abs(np.asarray(Gl)).max())
            Ks.append(K)
            ok = ok and (K <= 256 // 4)
        bad = False
        try:
            build_air(3409, np.full((32, 32), 2.0), np.full((32, 32), 2.0), ph)
        except ValueError:
            bad = True
        ok_small = True
        try:
            build_air(201, np.full((64, 64), 2.0), np.full((64, 64), 2.0), ph)
        except ValueError:
            ok_small = False
        rec("C8", "alias guard: K %s <= 64 at N256; raises on N32 nG3409; passes N64 nG201" % Ks, float(bad and ok_small),
            "1.0", ok and bad and ok_small)
    except Exception as ex:
        rec("C8", "alias guard", str(ex), "-", False)

    # ---- C5 uniform stacks vs TMM ----------------------------------------------------------
    set_bk("numpy")
    worst_a = worst_b = worst_other = 0.0
    worst_c = 0.0
    nlist = {"a-Si": 3.48, "SiO2": 1.444, "Si3N4": 1.996, "air": 1.0}
    cases = [(1, 1), (1, 0), (0, 1), (0, 0)]            # (L1 solid?, L2 solid?)
    for s1, s2 in cases:
        n1 = ph.e1s ** 0.5 if s1 else ph.e1v ** 0.5
        n2 = ph.e2s ** 0.5 if s2 else ph.e2v ** 0.5
        e1 = np.full((8, 8), n1 ** 2)
        e2 = np.full((8, 8), n2 ** 2)
        # (a) air side, normal incidence
        ob = build_air(21, e1, e2, ph, alias=False)
        i00 = idx(ob, 0, 0)
        for src, ex in (("Ex", excite_Ex), ("Ey", excite_Ey)):
            ex(ob, i00)
            Rx, Ry, Tx, Ty = flux_halves(ob)
            Rg = (Ry if src == "Ex" else Rx)[i00]
            Tg = (Ty if src == "Ex" else Tx)[i00]
            Ra, Ta = tmm_RT([(n2, ph.H2), (n1, ph.H1)], 1.0, ph.n_sub, 0.0, "s", ph.lam)
            worst_a = max(worst_a, abs(Rg - Ra), abs(Tg - Ta))
            oth = np.concatenate([Rx, Ry, Tx, Ty])
            msk = np.ones(oth.size, bool)
            for q in range(4):
                msk[q * ob.nG + i00] = False
            worst_other = max(worst_other, float(np.max(np.abs(oth[msk]))))
            # the co-pol half must be the whole thing; the other half of order (0,0) zero
            cross = (Rx if src == "Ex" else Ry)[i00]
            worst_other = max(worst_other, abs(cross))
        del ob
        # (b) substrate side at 30 deg, phi 0 and 90, p and s
        th = math.radians(30.0)
        for phi in (0.0, math.pi / 2):
            ob = build_sub(21, e1, e2, ph, phi, theta=th, alias=False)
            i00 = idx(ob, 0, 0)
            for pol in ("p", "s"):
                if pol == "p":
                    excite_Ex(ob, i00)
                else:
                    excite_Ey(ob, i00)
                Rx, Ry, Tx, Ty = flux_halves(ob)
                if phi == 0.0:
                    Rco, Tco, Rcr, Tcr = (Ry, Ty, Rx, Tx) if pol == "p" else (Rx, Tx, Ry, Ty)
                else:
                    Rco, Tco, Rcr, Tcr = (Rx, Tx, Ry, Ty) if pol == "p" else (Ry, Ty, Rx, Tx)
                Rb, Tb = tmm_RT([(n1, ph.H1), (n2, ph.H2)], ph.n_sub, 1.0, th, pol, ph.lam)
                worst_b = max(worst_b, abs(Rco[i00] - Rb), abs(Tco[i00] - Tb))
                worst_other = max(worst_other, abs(Rcr[i00]), abs(Tcr[i00]))
            del ob
        # (c) substrate side at -theta_sub: TIR into air
        ob = build_sub(21, e1, e2, ph, 0.0, alias=False)
        i00 = idx(ob, 0, 0)
        excite_Ex(ob, i00)
        Rx, Ry, Tx, Ty = flux_halves(ob)
        Rtot = float(np.sum(Rx + Ry))
        Ttot = float(np.sum(Tx + Ty))
        worst_c = max(worst_c, abs(Rtot - 1.0), Ttot)
        Rb, Tb = tmm_RT([(n1, ph.H1), (n2, ph.H2)], ph.n_sub, 1.0, ph.th_sub, "p", ph.lam)
        worst_c = max(worst_c, abs(Rb - 1.0))
        del ob
    rec("C5a", "uniform stacks air side normal inc vs TMM (4 combos, Ex,Ey)", float(worst_a), "<=1e-9", worst_a <= 1e-9)
    rec("C5b", "uniform stacks substrate side 30deg phi0/90 p,s vs TMM", float(worst_b), "<=1e-9", worst_b <= 1e-9)
    rec("C5c", "substrate side at -theta_sub: R=1, T=0 (TIR)", float(worst_c), "<=1e-9", worst_c <= 1e-9)
    rec("C6", "other orders / forbidden halves on uniform stacks", float(worst_other), "<=1e-9", worst_other <= 1e-9,
        "p in y-half at phi 0, x-half at phi 90 (C5b match + forbidden half)")

    # ---- C1 extruded 1-D input vs G._solve -------------------------------------------------
    c1_path = P["c1_density_npy"] or P["seed_src_npy"] or r"<DATA_ROOT>\Result\260819_scdc_obj\260819_c26in_525_450_rf030\density_scdc_input_H525_450.npy"
    if not want("C1"):
        pass
    elif os.path.isfile(c1_path):
        D512 = np.load(c1_path)
        set_bk("numpy")
        G.set_substrate(ph.n_sub)
        G.set_period(float(P["period_nm"]))
        E_ASI, E_SIN, E_OX = ph.e1s, ph.e2s, ph.e1v
        worst1, worst_forb = 0.0, 0.0
        for nG2 in (201, 401):
            r1 = np.repeat(D512[0][:, None], 8, axis=1)
            r2 = np.repeat(D512[1][:, None], 8, axis=1)
            e1, e2 = eps_pair(np.stack([r1, r2]), ph)
            ob = build_air(nG2, e1, e2, ph, theta=G.THETA_0, alias=False)
            K = int(np.abs(np.asarray(ob.G)[:, 0]).max())
            i00, i10, im10, i01, i0m1 = idx(ob, 0, 0), idx(ob, 1, 0), idx(ob, -1, 0), idx(ob, 0, 1), idx(ob, 0, -1)
            excite_Ex(ob, i00)
            Rx, Ry, Tx, Ty = flux_halves(ob)
            ex = dict(T1=Ty[i10], T0=Ty[i00], Tm1=Ty[im10], R0=Ry[i00])
            forb = [Tx[i10], Tx[i00], Tx[im10], Rx[i00], Tx[i01] + Ty[i01], Tx[i0m1] + Ty[i0m1],
                    Rx[idx(ob, 1, 1)] + Ry[idx(ob, 1, 1)], Rx[idx(ob, 1, -1)] + Ry[idx(ob, 1, -1)]]
            excite_Ey(ob, i00)
            Rx2, Ry2, Tx2, Ty2 = flux_halves(ob)
            ey = dict(T0=Tx2[i00], T1=Tx2[i10], R0=Rx2[i00])
            forb += [Ty2[i10], Ty2[i00], Tx2[i01] + Ty2[i01]]
            del ob
            ng1 = 2 * K + 3          # trap: grcwa realises requested-2 on the quasi-1-D lattice
            obm, Rm, Tm = G._solve(D512[0], D512[1], ph.H1, ph.H2, (E_ASI, E_SIN), E_OX, 1.0, "top", "TM", G.THETA_0, 1.55, ng1)
            obe, Re, Te = G._solve(D512[0], D512[1], ph.H1, ph.H2, (E_ASI, E_SIN), E_OX, 1.0, "top", "TE", G.THETA_0, 1.55, ng1)
            obs, Rs, Ts = G._solve(D512[0], D512[1], ph.H1, ph.H2, (E_ASI, E_SIN), E_OX, 1.0, "sub", "TM", -G.THETA_SUB, 1.55, ng1)
            pm, pe, ps = G._pick(obm, Tm, (1, 0, -1)), G._pick(obe, Te, (0, 1)), G._pick(obs, Rs, (2, 0))
            rm, re_ = G._pick(obm, Rm, (0,)), G._pick(obe, Re, (0,))
            diffs = [abs(float(np.real(pm[1])) - ex["T1"]), abs(float(np.real(pm[0])) - ex["T0"]),
                     abs(float(np.real(pm[-1])) - ex["Tm1"]), abs(float(np.real(rm[0])) - ex["R0"]),
                     abs(float(np.real(pe[0])) - ey["T0"]), abs(float(np.real(pe[1])) - ey["T1"]),
                     abs(float(np.real(re_[0])) - ey["R0"])]
            # payer at -theta
            obp = build_sub(nG2, e1, e2, ph, 0.0, alias=False)
            j00, j20 = idx(obp, 0, 0), idx(obp, 2, 0)
            excite_Ex(obp, j00)
            sRx, sRy, sTx, sTy = flux_halves(obp)
            diffs += [abs(float(np.real(ps[2])) - (sRx[j20] + sRy[j20])), abs(float(np.real(ps[0])) - (sRx[j00] + sRy[j00]))]
            forb += [sRx[idx(obp, 1, 1)] + sRy[idx(obp, 1, 1)], sRx[idx(obp, 1, -1)] + sRy[idx(obp, 1, -1)]]
            del obp
            worst1 = max(worst1, max(diffs))
            worst_forb = max(worst_forb, max(abs(float(f)) for f in forb))
            log("   C1 nG_req %d K %d: max diff vs G._solve (nG_req %d) %.3e; forbidden max %.3e" % (nG2, K, ng1, max(diffs), max(abs(float(f)) for f in forb)))
        rec("C1", "2-D extruded 2129 vs 1-D G._solve at matched K (K 8, 11)", float(worst1), "<=1e-10", worst1 <= 1e-10)
        rec("C1y", "y-invariance: forbidden halves / (0,+-1) / R(+1,+-1)", float(worst_forb), "<=1e-20", worst_forb <= 1e-20,
            "")
    else:
        rec("C1", "2-D extruded vs 1-D (input density missing)", "NOT FOUND: %s" % c1_path, "<=1e-10", False)

    # ---- C2 sigma_d equalities + C3 closure / SOP -------------------------------------------
    set_bk("numpy")
    KFt = conic_kernel_fft(64, 70.0 / (ph.P * 1e3 / 64))
    raw = rng.random((2, 64, 64))
    rho_sym = np.asarray(dens_pair(raw, KFt, 8.0))
    # unsymmetrised negative control: same raw, filter and projection WITHOUT the transpose average
    rho_asym = np.stack([np.asarray(tanh_project(np.real(np.fft.ifft2(np.fft.fft2(raw[L]) * KFt)), 8.0)) for L in (0, 1)])
    seed_s1 = None
    src_for_s1 = None
    if P["seed_src_npy"] and os.path.isfile(P["seed_src_npy"]):
        src_for_s1 = np.load(P["seed_src_npy"])
    elif os.path.isfile(c1_path):
        src_for_s1 = np.load(c1_path)
    structs = [("random sigma_d N64", rho_sym[0], rho_sym[1])]
    if src_for_s1 is not None:
        xs1 = make_seed("xsec_min", 128, 0, src_for_s1)
        structs.append(("S1 xsec_min N128 (hard)", (xs1[0] > 0.5).astype(float), (xs1[1] > 0.5).astype(float)))
    c2_worst, c3_close, c3_half, c3_sop = 0.0, 0.0, 0.0, 0.0
    for name, a1, a2 in structs:
        for nG2 in ((201, 401) if want("C2") else ()):
            r = measure_rung(a1, a2, Phys(dict(P, nxy=a1.shape[0])), nG2, want_s=True, want_y=True)
            pairs = [("eta", r["eta_x"], r["eta_y"]), ("leak", r["leak_x"], r["leak_y"]),
                     ("leak_away", r["leak_away_x"], r["leak_away_y"]),
                     ("refl", r["refl_x"], r["refl_y"]),
                     ("echo", r["echo_x"], r["echo_y"]), ("relaunch", r["relaunch_x"], r["relaunch_y"]),
                     ("relaunch_away", r["relaunch_away_x"], r["relaunch_away_y"]),
                     ("Rj xx/yy", abs(r["_Rj"][0, 0]), abs(r["_Rj"][1, 1])),
                     ("Rj xy/yx", abs(r["_Rj"][0, 1]), abs(r["_Rj"][1, 0])),
                     ("T00 co-pol Ex/Ey", r["air_Ex_T_o+0+0_Ex"], r["air_Ey_T_o+0+0_Ey"]),
                     ("T00 cross-pol", r["air_Ex_T_o+0+0_Ey"], r["air_Ey_T_o+0+0_Ex"]),
                     ("ownlane s", r["ownlane_s_x"], r["ownlane_s_y"]),
                     ("payer T to air: x-pay Ex / y-pay Ey", r["sub_xp_T_o+1+0_Ex"], r["sub_yp_T_o+0+1_Ey"]),
                     ("payer T to air: x-pay Ey / y-pay Ex", r["sub_xp_T_o+1+0_Ey"], r["sub_yp_T_o+0+1_Ex"]),
                     ("payer R vertical: x-pay Ex / y-pay Ey", r["sub_xp_R_o+1+0_Ex"], r["sub_yp_R_o+0+1_Ey"]),
                     ("payer R vertical: x-pay Ey / y-pay Ex", r["sub_xp_R_o+1+0_Ey"], r["sub_yp_R_o+0+1_Ex"]),
                     ("payer R00 p", r["sub_xp_R_o+0+0_p"], r["sub_yp_R_o+0+0_p"]),
                     ("payer R00 s", r["sub_xp_R_o+0+0_s"], r["sub_yp_R_o+0+0_s"]) if "sub_yp_R_o+0+0_s" in r else None]
            pairs = [p for p in pairs if p is not None]
            d = max(abs(a - b) for _, a, b in pairs)
            if d > 1e-10:
                worst_pair = max(pairs, key=lambda p: abs(p[1] - p[2]))
                log("      C2 worst pair: %s  %.9e vs %.9e" % worst_pair)
            c2_worst = max(c2_worst, d)
            c3_close = max(c3_close, energy_worst(r))
            c3_half = max(c3_half, max(v for k, v in r.items() if k.endswith("_halves_vs_RT")))
            # six-SOP H and V equal R_tot(E_x), R_tot(E_y)
            c3_sop = max(c3_sop, abs(r["refl_sop_H"] - r["refl_x"]), abs(r["refl_sop_V"] - r["refl_y"]))
            log("   C2 %-26s nG_req %d: max |x-lane - y-lane| %.3e (eta %.6f)" % (name, nG2, d, r["eta_x"]))
    if want("C2"):                                  # C2 also carries C3 (same solves)
        rec("C2", "sigma_d: lane/leak/refl/Jones/payer equalities", float(c2_worst), "<=1e-10", c2_worst <= 1e-10)
        # negative control: same raw arrays without symmetrisation must differ
        ra = measure_rung(rho_asym[0], rho_asym[1], Phys(dict(P, nxy=64)), 201, want_s=False, want_y=True)
        neg = max(abs(ra["eta_x"] - ra["eta_y"]), abs(ra["leak_x"] - ra["leak_y"]), abs(ra["echo_x"] - ra["echo_y"]))
        rec("C2n", "NEGATIVE control (no symmetrisation) must differ", float(neg), ">=1e-4", neg >= 1e-4)
        rec("C3a", "energy closure of every solve in C2", float(c3_close), "<=1e-6", c3_close <= 1e-6)
        rec("C3b", "halves vs RT_Solve per-order totals", float(c3_half), "<=1e-12", c3_half <= 1e-12)
        rec("C3c", "six-SOP H,V reflection = R_tot(Ex), R_tot(Ey)", float(c3_sop), "<=1e-12", c3_sop <= 1e-12)

    # ---- C13 seeds: exactly sigma_d, in [0,1] ----------------------------------------------
    worst_seed = 0.0
    names = ["multi2d_q", "multi2d", "rand"] + (["xsec_min", "xsec_max", "xsec_avg"] if src_for_s1 is not None else [])
    for nm in names:
        xs = make_seed(nm, 256, 0, src_for_s1)
        worst_seed = max(worst_seed, float(np.max(np.abs(xs - np.transpose(xs, (0, 2, 1))))),
                         float(max(0.0, -xs.min(), xs.max() - 1.0)))
    rec("C13", "all seeds exactly sigma_d at N256 and inside [0,1]", float(worst_seed), "==0", worst_seed == 0.0)

    # ---- C13h hard_pair of every seed exactly sigma_d (verifier fix 4: FFT rounding at the threshold) ----
    KF13 = conic_kernel_fft(256, float(P["filt_radius_nm"]) / (ph.P * 1e3 / 256))
    beta13 = parse_sched(P["beta_schedule"], "beta_schedule", False)[-1][1]
    worst_hard = 0.0
    for nm in names + ["rand1"]:
        xs = make_seed("rand" if nm == "rand1" else nm, 256, 1 if nm == "rand1" else 0, src_for_s1)
        hp = hard_pair(xs, KF13, beta13)
        worst_hard = max(worst_hard, float(np.max(np.abs(hp - np.transpose(hp, (0, 2, 1))))))
    rec("C13h", "hard_pair (beta %g threshold) of every seed exactly sigma_d (sym_err)" % beta13, float(worst_hard), "==0",
        worst_hard == 0.0)

    # ---- C14 every start breaks the mirrors M_x and M_y (verifier fix 3) ------------------------------
    # A start that keeps M_x and M_y as well as sigma_d is C4v: both grid layers then carry exactly degenerate
    # eigenvalues and AD through the eigendecomposition is wrong (multi2d: 3 / 7 / 21 % off along smooth random
    # directions, 261006); C13 checks sigma_d only. Measured on rho at beta 8 as the smallest max|rho - mirror(rho)|
    # over every mirror axis of the raster. Positive set: every seed except the named multi2d, plus the init_npy
    # start of the job (multi2d + eps x a smooth sigma_d field) -- REQUIRED, otherwise that start is untested.
    # Negative control: the bare multi2d seed must be detected (<= 1e-9).
    def mirror_dev(xs):
        rho = np.asarray(getval(dens_pair(xs, KF13, 8.0)))
        n = rho.shape[-1]
        ii = np.arange(n)
        dx = min(float(np.max(np.abs(rho - rho[:, (s - ii) % n, :]))) for s in range(n))
        dy = min(float(np.max(np.abs(rho - rho[:, :, (s - ii) % n]))) for s in range(n))
        return min(dx, dy)

    devs14 = {}
    for nm in [n_ for n_ in names if n_ != "multi2d"] + ["rand1"]:
        devs14[nm] = mirror_dev(make_seed("rand" if nm == "rand1" else nm, 256, 1 if nm == "rand1" else 0, src_for_s1))
    init_md5 = ""
    if P["init_npy"] and os.path.isfile(P["init_npy"]):
        x0i = np.load(P["init_npy"]).astype(float)
        if x0i.shape == (2, 256, 256):
            if int(P["init_soft"]):
                x0i = 0.15 + 0.70 * x0i
            x0i = 0.5 * (x0i + np.transpose(x0i, (0, 2, 1)))
            devs14["init_npy"] = mirror_dev(x0i)
            init_md5 = _md5_file(P["init_npy"])
    dev_neg = mirror_dev(make_seed("multi2d", 256, 0, None))
    worst14 = min(devs14.values())
    ok14 = bool(worst14 >= 1e-3 and dev_neg <= 1e-9 and "init_npy" in devs14)
    rec("C14", "every start breaks M_x and M_y (min over mirror axes); negative control multi2d detected", float(worst14),
        ">=1e-3 every start; multi2d <=1e-9; init_npy tested", ok14,
        "per start %s; negative control multi2d %.1e; init_npy %s md5 %s"
        % (", ".join("%s %.2e" % kv for kv in devs14.items()), dev_neg,
           "tested" if "init_npy" in devs14 else "NOT TESTED (give init_npy in the selftest job)", init_md5 or "-"))

    # ---- C12 fast medial-axis linewidth equals G._medial_min_nm -----------------------------
    worst12 = 0.0
    cases12 = 0
    for N12, rr in ((64, 70.0), (128, 70.0), (96, 90.0)):
        for s in range(3):
            rawq = rng.random((N12, N12))
            KF12 = conic_kernel_fft(N12, rr / (ph.P * 1e3 / N12))
            b = (np.real(np.fft.ifft2(np.fft.fft2(rawq) * KF12)) > np.median(np.real(np.fft.ifft2(np.fft.fft2(rawq) * KF12))) + 0.002 * s).astype(float)
            px12 = ph.P * 1e3 / N12
            for arr in (b, 1.0 - b):
                a = medial_min_fast(arr, px12)
                ref = G._medial_min_nm(arr, px12)
                cases12 += 1
                if math.isinf(a) and math.isinf(ref):
                    continue
                worst12 = max(worst12, abs(a - ref))
    rec("C12", "fast FFT medial-axis width == G._medial_min_nm (%d arrays)" % cases12, float(worst12), "==0", worst12 == 0.0)

    # ---- C4 AD vs central FD, every term active (nG 201, 401; the smoke job adds selftest_c4_ng) --
    if want("C4"):
        run_c4(P, [201, 401] + ([int(P["selftest_c4_ng"])] if int(P["selftest_c4_ng"]) > 0 else []), rec, rng)

    # ---- C9 raster refinement (machinery; tolerances apply to final designs) -----------------
    if want("C9") and src_for_s1 is not None:
        set_bk("numpy")
        xs1 = make_seed("xsec_min", 128, 0, src_for_s1)
        h1, h2 = (xs1[0] > 0.5).astype(float), (xs1[1] > 0.5).astype(float)
        ph128, ph256 = Phys(dict(P, nxy=128)), Phys(dict(P, nxy=256))
        a = measure_rung(h1, h2, ph128, 401, want_s=False, want_y=True)
        b = measure_rung(kron2(h1), kron2(h2), ph256, 401, want_s=False, want_y=True)
        keys = ("eta_x", "leak_x", "refl_worst_sop", "echo_x", "relaunch_x")
        dif = {k: b[k] - a[k] for k in keys}
        symb = float(max(np.max(np.abs(kron2(h1) - kron2(h1).T)), np.max(np.abs(kron2(h2) - kron2(h2).T))))
        rec("C9", "kron x2 raster refinement of S1 at nG 401 (report; N128->N256)", float(max(abs(v) for v in dif.values())),
            "report (final designs: eta 2e-3, others 3e-4)", symb == 0.0,
            "d_eta %+.2e d_leak %+.2e d_R %+.2e d_echo %+.2e d_rel %+.2e; upsampled sigma_d exact %s"
            % (dif["eta_x"], dif["leak_x"], dif["refl_worst_sop"], dif["echo_x"], dif["relaunch_x"], symb == 0.0))

    # ---- C10 resume: bit-identical trajectory ----------------------------------------------
    ok10, d10 = False, float("nan")
    try:
        if not want("C10"):
            raise _Skip()
        sub = os.path.join(od, "_c10")
        Pc = dict(P, nxy=64, iters=6, ng_schedule="0:201", lw_start_frac=0.5, best_bin_from=0.5, best_bin_every=1,
                  ckpt_every=3, ckpt_every_s=1e9, seed="rand", seed_rng=1, seed_src_npy="", init_npy="",
                  result_folder="", Output_folder=sub, subfolder_name="full")
        Pc["beta_schedule"] = "0:8, 0.4:16, 0.7:32, 0.9:64"
        ra = optimise(dict(Pc, subfolder_name="full", ckpt_every=0), os.path.join(sub, "full"), quiet=True)
        Pk = dict(Pc, subfolder_name="killed")
        odk = os.path.join(sub, "killed")
        optimise(Pk, odk, stop_after=3, quiet=True)
        rb = optimise(Pk, odk, quiet=True)
        d10 = float(np.max(np.abs(ra["x"] - rb["x"])))
        dbx = float(np.max(np.abs(ra["best_x"] - rb["best_x"])))
        ok10 = (d10 == 0.0 and dbx == 0.0 and rb["resumed"] and ra["best_it"] == rb["best_it"])
        rec("C10", "resume: 6 it nG201 N64, ckpt at 3, rerun", float(d10), "==0 (x, best_x, best_it)", ok10,
            "resumed %s best_it %d/%d max|best_x diff| %.1e" % (rb["resumed"], ra["best_it"], rb["best_it"], dbx))
    except _Skip:
        pass
    except Exception as ex:
        import traceback
        traceback.print_exc()
        rec("C10", "resume", "EXC %s" % ex, "==0", False)

    # ---- C7 reciprocity reading (report) -----------------------------------------------------
    set_bk("numpy")
    if want("C7") and src_for_s1 is not None:
        xs1 = make_seed("xsec_min", 128, 0, src_for_s1)
        h1, h2 = (xs1[0] > 0.5).astype(float), (xs1[1] > 0.5).astype(float)
        gaps = []
        for nG2 in (201, 401, 601):
            r = measure_rung(h1, h2, Phys(dict(P, nxy=128)), nG2, want_s=False, want_y=False)
            gaps.append(r["recip_gap"])
        rec("C7", "reciprocity reading per rung nG 201/401/601 (S1, N128)", float(gaps[-1]), "report (must shrink with K)",
            True, "gaps %s" % ", ".join("%.2e" % g for g in gaps))

    # ---- C11 convergence ladder (selftest = 2) -----------------------------------------------
    if level >= 2 and want("C11") and src_for_s1 is not None:
        set_bk("numpy")
        xs1 = make_seed("xsec_min", 256, 0, src_for_s1)
        h1, h2 = (xs1[0] > 0.5).astype(float), (xs1[1] > 0.5).astype(float)
        rows = []
        for nG2 in (401, 601, 901, 1201):
            est = 4.6 * (nG2 / 889.0) ** 1.95 * 0.55
            if not ram_guard(est, "C11 nG %d" % nG2):
                rows.append(dict(nG=nG2, skipped="RAM guard"))
                continue
            r = measure_rung(h1, h2, ph, nG2, want_s=False, want_y=True)
            rows.append(dict(nG=nG2, nG_realised=r["nG_realised"], K=r["K"], eta_x=r["eta_x"], leak_x=r["leak_x"],
                             refl_worst_sop=r["refl_worst_sop"], echo_x=r["echo_x"], relaunch_x=r["relaunch_x"],
                             t_s=r["t_s"], peak_GB=r["peak_GB"]))
            log("   C11 nG %d (realised %d K %d): eta %.5f leak %.5f R %.5f echo %.5f rel %.5f  %.0f s peak %.2f GB"
                % (nG2, r["nG_realised"], r["K"], r["eta_x"], r["leak_x"], r["refl_worst_sop"], r["echo_x"],
                   r["relaunch_x"], r["t_s"], r["peak_GB"]))
        rec("C11", "S1 crossed-structure ladder (report)", "see rows", "report", True, json.dumps(_san(rows)))

    # a full run must have recorded every control: a skipped one (missing input file, ...) is a failure
    expected = ["C8", "C5a", "C5b", "C5c", "C6", "C1", "C1y", "C2", "C2n", "C3a", "C3b", "C3c", "C13", "C13h", "C14", "C12",
                "C4@201", "C4t@201", "C4@401", "C9", "C10", "C7"] + (["C11"] if level >= 2 else [])
    got = set(c["id"] for c in ctl)
    missing = [] if only else [e for e in expected if e not in got]
    for e in missing:
        log("MISSING control %s (not recorded) -> the run cannot pass" % e)
    allp = all(c["passed"] for c in ctl) and not missing
    outj = dict(driver_md5=DRIVER_MD5, level=level, host=socket.gethostname(), all_pass=bool(allp), missing=missing,
                n_controls=len(ctl), failed=[c["id"] for c in ctl if not c["passed"]],
                elapsed_s=time.time() - t_all, controls=ctl, scd_c_grating_md5=_md5_file(G.__file__.replace(".pyc", ".py")),
                grcwa_rcwa_md5=_md5_file(os.path.join(_GRCWA_DIR, "rcwa.py")),
                grcwa_fft_funs_md5=_md5_file(os.path.join(_GRCWA_DIR, "fft_funs.py")))
    sd = P["selftest_dir"] or P["Output_folder"] or od
    os.makedirs(sd, exist_ok=True)
    outj["selftest_only"] = sorted(only)
    pth = os.path.join(sd, ("selftest_input2d_partial_%s.json" if only else "selftest_input2d_%s.json") % DRIVER_MD5[:12])
    write_json(pth, outj)
    log("selftest level %d: %d controls, all_pass %s, failed %s -> %s" % (level, len(ctl), allp, outj["failed"], pth))
    return allp


def selftest_gate(P):
    """An optimisation job refuses to start without a passing fast-set file for THIS driver md5."""
    sd = P["selftest_dir"] or P["Output_folder"]
    pth = os.path.join(sd, "selftest_input2d_%s.json" % DRIVER_MD5[:12])
    if not os.path.isfile(pth):
        return False, "selftest file missing: %s" % pth
    try:
        with io.open(pth, encoding="utf-8") as fh:
            j = json.load(fh)
    except Exception as e:
        return False, "selftest file unreadable: %s" % e
    if j.get("driver_md5") != DRIVER_MD5:
        return False, "selftest file is for another driver md5"
    if not j.get("all_pass"):
        return False, "selftest did not pass: %s" % j.get("failed")
    return True, pth


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    P = build_params(CFG)
    G.set_substrate(float(P["n_sub"]))
    G.set_period(float(P["period_nm"]))
    if P["sym"] != "sigma_d":
        raise ValueError("sym must be sigma_d (the only value accepted)")
    if P["fill1"] in ("air", "Air"):
        raise ValueError("fill1 = air is refused (support rule: L1 must be planarised)")
    log("driver md5 %s  host %s  python %s  numpy %s" % (DRIVER_MD5[:12], socket.gethostname(), sys.version.split()[0], np.__version__))
    level = int(P["selftest"])
    eval_only = int(P["eval_only"])
    if level > 0:
        od = result_dir(P) if (P["result_folder"] or (P["Output_folder"] and P["subfolder_name"])) else (P["Output_folder"] or _HERE)
        os.makedirs(od, exist_ok=True)
        ok = run_selftest(P, level, od)
        return 0 if ok else 3
    od = result_dir(P)
    os.makedirs(od, exist_ok=True)
    ok, msg = selftest_gate(P)
    if not ok:
        log("REFUSED: %s" % msg)
        return 4
    log("selftest gate OK: %s" % msg)
    ph = Phys(P)
    out = dict(status="IN_PROGRESS", driver_md5=DRIVER_MD5,
               scd_c_grating_md5=_md5_file(G.__file__.replace(".pyc", ".py")),
               grcwa_rcwa_md5=_md5_file(os.path.join(_GRCWA_DIR, "rcwa.py")),
               grcwa_fft_funs_md5=_md5_file(os.path.join(_GRCWA_DIR, "fft_funs.py")),
               host=socket.gethostname(), N=ph.N, px_nm=ph.px_nm, lam_nm=ph.lam * 1e3, period_nm=ph.P * 1e3,
               theta_sub_deg=math.degrees(ph.th_sub), theta_crit_deg=math.degrees(ph.th_crit), sym="sigma_d")
    for k in KEYS:
        if k in _LI_KEYS and not (_LI_ON or _POL_ON):
            continue                                   # default JSON stays byte-identical
        out["key_%s" % k] = P[k]
    if _LI_ON or _POL_ON:
        out["grcwa_li_fmm_md5"] = _md5_file(os.path.join(_GRCWA_DIR, "li_fmm.py"))
    jpath = os.path.join(od, "input2d.json")
    if eval_only:
        if not P["init_npy"]:
            raise ValueError("eval_only needs init_npy (a (2,N,N) array)")
        a = np.load(P["init_npy"]).astype(float)
        if a.shape != (2, ph.N, ph.N):
            raise ValueError("init_npy shape %s != (2,%d,%d)" % (a.shape, ph.N, ph.N))
        rho = (a > 0.5).astype(float)
        out["eval_only"] = 1
        out["density_md5_in"] = _md5_arr(a)
    else:
        if not P["ng_schedule"]:
            raise ValueError("ng_schedule is required")
        if int(P["selftest_c4_ng"]) > 0:
            # smoke job: C4 (AD vs central FD, all terms) at the production nG BEFORE the loop,
            # so a gradient fault shows up in the first hour, not after the 901 stage
            c4 = run_c4(P, [int(P["selftest_c4_ng"])])
            for n_c4, d_c4 in c4.items():
                for k_c4, v_c4 in d_c4.items():
                    out["c4_ng%d_%s" % (n_c4, k_c4)] = v_c4
            write_json(jpath, out)
        res = optimise(P, od)
        M = res["M"]
        x_b = res["best_x"] if res["best_J"] > -1e29 else res["x"]
        rho = hard_pair(x_b, M.KF, M.beta_final)
        np.save(os.path.join(od, "x_raw.npy"), x_b)
        out.update(seed=res["seed_label"], seed_md5=res["seed_md5"], resumed=res["resumed"],
                   best_bin_iter=res["best_it"], best_bin_J=res["best_J"], best_bin_feasible=res["best_ok"],
                   last_J=res["last_J"], iters_run=res["iters_run"], opt_elapsed_s=res["cum"],
                   ng_final=res["ng_fin"])
        out["last_terms"] = res["last_terms"]
    np.save(os.path.join(od, "rho1.npy"), rho[0])
    np.save(os.path.join(od, "rho2.npy"), rho[1])
    np.save(os.path.join(od, "rho12.npy"), rho)
    out["density_md5_L1"] = _md5_arr(rho[0].astype(np.uint8))
    out["density_md5_L2"] = _md5_arr(rho[1].astype(np.uint8))
    out["density_md5"] = _md5_arr(rho.astype(np.uint8))
    write_json(jpath, out)
    full_report(rho[0], rho[1], P, ph, od, out, partial_path=jpath)
    out["elapsed_s"] = time.time() - t0
    out["peak_GB_process"] = peak_gb()
    out["status"] = "COMPLETE"
    write_json(jpath, out)
    # the driver never deletes: a finished run's checkpoint is renamed so that a re-submitted
    # identical job restarts from zero instead of resuming a completed run
    pth = os.path.join(od, "ckpt_input2d.npz")
    if os.path.isfile(pth):
        try:
            os.replace(pth, pth + ".done")
        except OSError as e:
            log("WARN: could not rename the finished checkpoint: %s" % e)
    log("COMPLETE  gates_passed %s  elapsed %.0f s -> %s" % (out.get("gates_passed"), out["elapsed_s"], jpath))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as _e:                                  # print and fail with a code
        import traceback
        traceback.print_exc()
        log("ERROR: %s" % _e)
        sys.exit(2)
