"""
opt_top_grcwa.py -- DESIGN DRAFT: zone (2) top outer grating of the
self-referenced DQPSK metasurface receiver (2026-08-02_133353).

STATUS: DRAFT (2026-08-02). Validation + smoke tested only; the full
optimization run has NOT been executed. Do not quote results as achieved
performance.

Problem
-------
  air (n = 1.0), normal incidence, lambda = 1550 nm
    -> binary Si grating, single etch depth H (one period, pitch fixed)
    -> Si half-space (n = 3.4777)
  Pitch  : Lambda_top = 533.8 nm = lambda / (n_Si * sin 56.617deg)  [FIXED]
  Goal   : maximize order m = +1 transmitted efficiency INTO Si
           (internal deflection 56.617deg), for BOTH polarizations.
  Blazing: +1/-1 symmetry must be broken -> asymmetric multi-ridge binary
           profile (symmetric binary pi-depth reference: ~40.5 % per order).
  Phase  : record dphi = arg(t_TM,+1) - arg(t_TE,+1)  (origin-independent).
           Cascade constraint with zone (4): dphi_top + dphi_bottom = 0.
           Zone (4) best cell measured +8.6 deg -> zone (2) target ~ -8.6 deg.
           (Recorded/reported here; joint matching is a later step.)

Method
------
  grcwa (vendored, autograd backend) topology optimization of a 1D density
  rho(x) in [0,1] on NX pixels across one period:
      rho -> circulant conic filter -> tanh projection (beta continuation)
          -> eps(x) = eps_air + rho_bar*(eps_Si - eps_air)
  FoM (ascent) = softmin( eta_TE(+1), eta_TM(+1) ),  gradient via autograd.
  Etch depth H is swept in an outer loop (NOT differentiated).
  Multi-start: random asymmetric seeds + one blaze-ramp seed.

Conventions / traps (do not repeat past errors)
-----------------------------------------------
  * ORDER SIGN TRAP: here m = +1 is simply the +x-side order of the 1D cell.
    When mapping the cell back onto concentric rings, the blaze asymmetry
    direction must be chosen so the deflection is radially INWARD (crossing
    the axis; adopted design). rcwa_multi and the analysis notes label the
    same physical escaping order with opposite signs -- always check
    propagation angle, never trust the label.
  * TE = E along the grooves (s-pol here, phi=0); TM = H along grooves.
    Cross-check V3 against rcwa_multi guards against a basis mislabel.
  * Reflected +/-1 orders are evanescent in air (lambda/Lambda = 2.90 > 1),
    so ALL reflected power sits in m = 0: R_total is a clean loss readout.
  * min feature target: keep >= ~60 nm (zone (4) optimum has 69 nm; same
    single-etch fab class). Enforced softly by the filter radius; the
    binarized result's min feature is REPORTED -- check it before adopting.

Run
---
  python opt_top_grcwa.py --validate    # V1-V4 checks (~seconds)
  python opt_top_grcwa.py --smoke      # 1 seed, 1 H, 30 iters (~1 min)
  python opt_top_grcwa.py --full       # 6 seeds x 4 H x 300 iters + JSON out

Outputs (full mode)
-------------------
  best_cell_top.json   -- same schema family as best_cell_bottom.json
  runs_log.tsv         -- per-run summary (seed, H, eta_TE, eta_TM, minfeat)
"""

import os
import sys
import json
import argparse
import numpy as np

# ----------------------------------------------------------------------------
# vendored grcwa + autograd bootstrap (no network, no site-packages needed)
# ----------------------------------------------------------------------------
# [260804] Paths resolve relative to THIS file so the module runs unchanged from
# either the shared tree or a local mirror (<CODE_ROOT>
# \Version_unified\RCWA_custom). The previous absolute <DATA_ROOT>\...
# constants went through the NAS symlink, which (a) kept every solve dependent on
# SMB for the whole run and (b) broke the LOCAL_CODE contract -- the
# code is deliberately copied to a local disk before executing it.
# rcwa_multi.py now sits next to this file, so the old SUPERCELL_DIR is gone.
_HERE  = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(_HERE, "_vendor")
for _p in (_HERE, VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import grcwa
grcwa.set_backend('autograd')
import autograd.numpy as npa
from autograd import value_and_grad

import rcwa_multi  # validated reference solver (complex t per order)

# ----------------------------------------------------------------------------
# fixed design constants (adopted operating point -- do not invent geometry)
# ----------------------------------------------------------------------------
LAM      = 1.55                 # um
FREQ     = 1.0 / LAM            # grcwa natural units (c = 1)
N_SI     = 3.4777
EPS_AIR  = 1.0
EPS_SI   = N_SI ** 2
PERIOD   = 0.5338               # um, Lambda_top (FIXED by delay condition)
THETA0   = 1e-7                 # rad; tiny tilt to dodge exact-normal basis edge case
L1       = [PERIOD, 0.0]
L2       = [0.0, 1e-4]          # quasi-1D: tiny y-period -> only Gy = 0 survives truncation
NG       = 41                   # plane-wave count (x-orders ~ -20..20)
NX       = 256                  # density pixels per period (2.09 nm / px)
PX_NM    = PERIOD * 1e3 / NX

THETA_SI_DEG = np.degrees(np.arcsin(LAM / (PERIOD * N_SI)))   # sanity: ~56.62 deg
H_PI     = LAM / (2.0 * (N_SI - 1.0))                          # ~0.3127 um binary pi depth

DPHI_TARGET_DEG = -8.6   # cascade target (zone (4) best cell: +8.6 deg) -- reported only

# Minimum fabricable feature [nm], user spec 2026-08-04 (was ~60 nm, the fab class
# of the bottom cell's 69 nm). Used two ways:
#   soft -- the conic filter radius is derived from it, so the optimizer is pushed
#           toward feasible topologies instead of being punished afterwards;
#   hard -- binarized cells whose measured min feature falls below it are REJECTED
#           (the old code only reported the number and never gated on it).
MIN_FEATURE_NM  = 40.0
FILT_RADIUS_PX  = int(round(MIN_FEATURE_NM / PX_NM))   # 40 nm / 2.085 nm-per-px = 19 px


# ----------------------------------------------------------------------------
# density pipeline (autograd-safe)
# ----------------------------------------------------------------------------
def make_filter_matrix(nx, radius_px):
    """Circulant conic (linear cone) filter matrix, periodic wrap. Plain numpy
    constant -- built once, then used inside the autograd trace via matmul."""
    x = np.arange(nx)
    d = np.minimum(np.abs(x[:, None] - x[None, :]),
                   nx - np.abs(x[:, None] - x[None, :]))
    w = np.clip(1.0 - d / float(radius_px), 0.0, None)
    return w / w.sum(axis=1, keepdims=True)


def tanh_project(rho_f, beta, eta=0.5):
    num = npa.tanh(beta * eta) + npa.tanh(beta * (rho_f - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


def softmin(a, b, k=30.0):
    return -(1.0 / k) * npa.log(npa.exp(-k * a) + npa.exp(-k * b))


# ----------------------------------------------------------------------------
# grcwa solve: one polarization, returns order-resolved transmission array + G
# ----------------------------------------------------------------------------
def _build_obj(H):
    ob = grcwa.obj(NG, L1, L2, FREQ, THETA0, 0.0, verbose=0)
    ob.Add_LayerUniform(0.5, EPS_AIR)   # incidence half-space: air (top)
    ob.Add_LayerGrid(H, NX, 1)          # binary grating layer (etched Si)
    ob.Add_LayerUniform(0.5, EPS_SI)    # transmission half-space: Si
    ob.Init_Setup()
    return ob


def _order_index(ob, mx):
    G = np.asarray(ob.G)
    hit = np.where((G[:, 0] == mx) & (G[:, 1] == 0))[0]
    if hit.size != 1:
        raise RuntimeError(f"order (mx={mx}, 0) not found in truncation (nG={ob.nG})")
    return int(hit[0])


def eta_order(rho_bar, H, pol, mx=+1):
    """Transmitted diffraction efficiency of order (mx, 0) into Si.
    rho_bar may be an autograd array -> returned value is traced."""
    ob = _build_obj(H)
    eps_grid = EPS_AIR + rho_bar * (EPS_SI - EPS_AIR)
    ob.GridLayer_geteps(eps_grid)
    if pol == "TE":     # s-pol: E along y = along the grooves
        ob.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
    elif pol == "TM":   # p-pol
        ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    else:
        raise ValueError(pol)
    Ri, Ti = ob.RT_Solve(normalize=1, byorder=1)
    return Ti[_order_index(ob, mx)], Ri, Ti, ob


def eta_scalar(rho_bar, H, pol, mx=+1):
    t, _, _, _ = eta_order(rho_bar, H, pol, mx)
    return t


# ----------------------------------------------------------------------------
# concentric-ring objective  (260804) -- replaces softmin(eta_TE, eta_TM)
# ----------------------------------------------------------------------------
def _solve_pol(rho_bar, H, pol, mx_list=(+1,)):
    """One grcwa solve -> {mx: (eta, a_t)} for every requested order.

    eta is the diffraction efficiency from RT_Solve (the validated power path).
    a_t is the COMPLEX transverse-E Fourier amplitude of the same order:
        TE -> ey   (E along the grooves, s-pol, preserved as a scalar)
        TM -> ex   (E across the grooves; at normal incidence the input p-field
                    is along x with unit amplitude and zero phase)
    Both excitations are launched with unit amplitude and zero phase, so the two
    a_t share a phase reference and their relative phase is meaningful.
    Everything stays autograd-traced.

    [260804] Extra orders are FREE: RT_Solve already returns the whole
    order-resolved array and Solve_FieldFourier the whole Fourier amplitude
    vector. Asking for mx = -1 and 0 as well costs one extra index lookup, not a
    second solve -- which is why adding the blaze term below does not double the
    optimization cost.
    """
    ob = _build_obj(H)
    eps_grid = EPS_AIR + rho_bar * (EPS_SI - EPS_AIR)
    ob.GridLayer_geteps(eps_grid)
    if pol == "TE":
        ob.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
    elif pol == "TM":
        ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    else:
        raise ValueError(pol)
    _Ri, Ti = ob.RT_Solve(normalize=1, byorder=1)
    e_f, _h_f = ob.Solve_FieldFourier(ob.Layer_N - 1, 0.0)   # [ex, ey, ez]
    out = {}
    for mx in mx_list:
        i = _order_index(ob, mx)
        out[mx] = (Ti[i], e_f[1][i] if pol == "TE" else e_f[0][i])
    return out


def _eta_m0_from(te, tm, delta_ext):
    """(eta, a) pairs for one order, both polarizations -> (eta_m0, eta_m2).

    Shared by every order so the blaze difference below compares like with like.
    See fom_m0 for the derivation of the A/B split.

    [260805 FIX] delta_ext enters with a MINUS sign because a_te and a_tm come
    from grcwa, which uses exp(-i w t), while DELTA_BOT_DEG and DPHI_TARGET_DEG
    are quoted in rcwa_multi's exp(+j w t) convention. The two report opposite
    phase differences for the same structure -- measured on an asymmetric cell,
    dphi_grcwa + dphi_multi -> 0 as truncation grows (2.13 deg at nG=41/N=25 down
    to 0.17 deg at nG=201/N=150), while the direct difference stays near 172 deg.

    With the old "+delta_ext" the cross term peaked at dphi_grcwa = -8.6 deg,
    i.e. dphi_multi = +8.6 deg -- 17.2 deg from the cascade target of -8.6 deg,
    in the wrong direction. Only objectives with delta_ext != 0 were affected
    (eta_m0 / supp / blaze); cos is even, so delta_ext = 0 hid it, which is why
    every TM-only sweep is untouched.
    """
    eta_te, a_te = te
    eta_tm, a_tm = tm
    z     = a_tm * npa.conj(a_te)
    denom = npa.abs(a_tm) * npa.abs(a_te) + 1e-30
    cos_d = npa.real(z) / denom
    sin_d = npa.imag(z) / denom
    cos_tot = cos_d * np.cos(delta_ext) + sin_d * np.sin(delta_ext)
    cross = 2.0 * npa.sqrt(npa.abs(eta_te * eta_tm)) * cos_tot
    return (eta_te + eta_tm + cross) / 4.0, (eta_te + eta_tm - cross) / 4.0, cos_tot


# Zone (4) bottom outer ring, current best cell: arg(t_TM) - arg(t_TE) = +8.6 deg
# (Lambda 503 nm, H 286 nm, 2 ridges, escape TE = TM = 11.368 %). The receiver's
# polarization objective is a CASCADE condition, Delta_top + Delta_bot = 0, so the
# top ring is designed for Delta_top = -8.6 deg, NOT for zero.
# NOTE this is a SEED value: zone (4) has not been adjoint-optimized yet, so
# re-sync DELTA_BOT_DEG whenever the bottom cell changes.
DELTA_BOT_DEG = 8.6


def fom_m0(rho_bar, H, mx=+1, delta_ext=0.0):
    """Co-polarized (m = 0) coupling efficiency of the REVOLVED device.

    delta_ext [rad] is polarization phase the light picks up ELSEWHERE in the
    cascade -- for the DQPSK receiver that is zone (4), Delta_bot = +8.6 deg.
    The objective is arg(t_TM) - arg(t_TE) + delta_ext = 0, i.e. the two ring
    surfaces cancel each other rather than each being individually matched.
    This factorizes exactly here because zone (4)'s best cell has TE and TM
    escape amplitudes that are IDENTICAL (11.368 % both): the common factor
    divides out and maximizing this quantity at zone (2) maximizes the cascade.
    Pass delta_ext = 0 to design the ring in isolation.

    Revolving this 1D cross-section about the optical axis puts the grooves along
    the azimuthal direction. At azimuth phi an x-polarized normal-incidence input
    therefore presents its radial part (E across the grooves = TM) and its
    azimuthal part (E along the grooves = TE) at the same time. With
    A = (t_TM + t_TE)/2 and B = (t_TM - t_TE)/2 the transmitted field separates
    exactly into

        E/E0 = A x_hat  +  B (cos2phi x_hat + sin2phi y_hat)

    The A term is co-polarized and azimuthally uniform -- this is what couples to
    an m = 0 mode. The B term is cross-polarized and carries exp(+-2i phi), which
    is orthogonal to m = 0. So the rotational symmetry does NOT average the TE/TM
    mismatch away: it moves it into a channel that cannot be recovered.

    |A|^2 rewards efficiency and phase alignment in one number, which is why it
    replaces softmin(eta_TE, eta_TM). softmin is blind to arg(t) even though the
    phase term is the more damaging one -- on the L2 cell a 17.2 deg phase
    difference raised the m = +-2 leakage from 3.0% to 5.1%, while a 2x amplitude
    imbalance on its own only cost 3.0%.

    Magnitudes are taken from RT_Solve and only the RELATIVE PHASE from the field
    path, so a normalization-convention error inside Solve_FieldFourier cannot
    corrupt the efficiencies -- it can only perturb cos(delta), which is bounded.

    Returns (eta_m0, eta_m2, eta_TE, eta_TM, cos_delta).
    Identity, useful as a runtime check:  eta_m0 + eta_m2 == (eta_TE + eta_TM)/2
    """
    te = _solve_pol(rho_bar, H, "TE", (mx,))[mx]
    tm = _solve_pol(rho_bar, H, "TM", (mx,))[mx]
    eta_m0, eta_m2, cos_tot = _eta_m0_from(te, tm, delta_ext)
    return eta_m0, eta_m2, te[0], tm[0], cos_tot


def fom_supp(rho_bar, H, delta_ext=0.0, mx=+1, w_t0=0.0, w_m1=0.0):
    """eta_m0(+mx) penalized by the two channels that actually eat the power.

        FoM = eta_m0(+1)  -  w_t0 * T0_avg  -  w_m1 * eta_m0(-1)

    [260805] Motivation, from the full 127-cell audit of the 2026-08-04 sweeps:
    every cell -- symmetric or blazed -- dumps 46-58 % of the input into the
    UNDEFLECTED m = 0 order, which is by a wide margin the single largest loss.
    The previous objective (fom_m0) maximized eta_m0(+1) and was completely blind
    to where the rest of the light went, so the optimizer had no reason not to
    park it in m = 0.

    The blaze experiment showed why penalizing m = -1 ALONE backfires: going from
    the best symmetric cell to the best fabricable blazed cell moved
        m = -1  -0.093   (suppressed, as asked)
        m = 0   +0.118   (where it actually went)
        m = +1  -0.037   (the order we wanted, WORSE)
    i.e. suppressing the wrong-way order just re-routes power into the straight-
    through leak. Both terms therefore have to be in the objective at once --
    that is the whole point of this function, and why w_m1 alone is not offered.

    All three quantities come from the SAME two solves (one per polarization),
    because _solve_pol returns every requested order from one RT_Solve. Adding
    the penalties costs index lookups, not runtime.

    T0 is the raw TE/TM power average, NOT put through the A/B cascade split:
    the m = 0 order is undeflected leakage that never reaches the receiver, so
    what matters is how much of it exists, not how it is polarized.

    w_t0 = w_m1 = 0 reproduces fom_m0 exactly -- keep it that way, it is the
    regression check.

    Returns (fom, eta_m0_p1, eta_m0_m1, T0_avg, eta_TE, eta_TM, cos_tot).
    """
    te = _solve_pol(rho_bar, H, "TE", (mx, -mx, 0))
    tm = _solve_pol(rho_bar, H, "TM", (mx, -mx, 0))
    eta_p, _eta_m2, cos_tot = _eta_m0_from(te[mx], tm[mx], delta_ext)
    eta_m, _e2m, _ctm = _eta_m0_from(te[-mx], tm[-mx], delta_ext)
    t0 = 0.5 * (te[0][0] + tm[0][0])
    fom = eta_p - w_t0 * t0 - w_m1 * eta_m
    return fom, eta_p, eta_m, t0, te[mx][0], tm[mx][0], cos_tot


def fom_blaze(rho_bar, H, delta_ext=0.0, mx=+1):
    """NET inward coupling:  eta_m0(+mx) - eta_m0(-mx).   [260804]

    *** TRIED AND REJECTED -- DO NOT USE AS THE OBJECTIVE. ***
    Superseded by fom_supp, which keeps the m = -1 term but adds the T0 term
    that this one was missing -- see fom_supp's docstring for why that matters.

    A full 21-point H sweep was run with this and compared against plain fom_m0
    at the same H (290 nm):

                        fom_m0      fom_blaze
        eta_m0(+1)      0.16032     0.13668     -18 %   <- the thing we want
        eta_m0(-1)      0.16032     0.06376     -60 %
        T0 undeflected  0.4708      0.5962      +27 %   <- worse
        dphi_total      +1.33 deg   +12.29 deg          <- cascade match lost

    Power removed from the -1 order did NOT move into +1; it moved into the
    undeflected m = 0 order. Summed over the two first orders: 0.320 -> 0.201.
    Subtracting eta(-1) treats a WASTED channel as a HARMFUL one and double-counts
    it, so the optimizer pays real +1 efficiency to buy asymmetry it cannot use.

    The deeper reason is that blazing needs a monotonic sawtooth phase across the
    period, which needs MULTIPLE phase levels. A single binary etch gives two
    levels, and a two-level grating has symmetric +-1 by construction. The
    optimizer did push the ratio from 0.50 to 0.80 by rearranging ridges, but only
    by giving up total deflection. That is a process limit, not a tuning problem:
    breaking it means a multi-level etch.

    The -1 order is therefore an inherent single-etch loss, and belongs in the
    loss budget and in zone (5) stray-light management -- not in the objective.

    Original rationale, retained for the record:
      fom_m0 rewards eta(+1) and says nothing about eta(-1), so a symmetric cell
      scoring 0.29/0.29 and a perfectly blazed cell scoring 0.29/0.00 receive the
      IDENTICAL score. Symmetric solutions are much easier to reach, and the H
      sweep of 2026-08-04 duly collapsed to a single-ridge symmetric cell
      (ridges = [0.95703, 1.60547]) whose T(+1) and T(-1) agreed to six decimals.
      Asymmetric seeds cannot fix that -- the objective itself has to break the
      symmetry. The README's "blazing needs asymmetry" note was about seeds and
      was not sufficient.

      In the revolved ring the +1 order goes INWARD (wanted, folds under the
      centre toward zone (4)) and -1 goes OUTWARD (lost). Subtracting the two is
      literally the net inward gain, and it has no tuning weight -- unlike a
      lambda-weighted penalty, and unlike a branching ratio which can be driven
      to 1 by a cell that deflects almost nothing at all.

      Suppressing the undeflected m = 0 order is the same problem: a mirror
      symmetric cell cannot preferentially dump power into one first order, so it
      leaves it in m = 0. Breaking the symmetry is what moves power out of m = 0.

    Cost: unchanged. Both orders come out of the same two solves (see _solve_pol).

    Returns (fom, info) where info carries the full budget for reporting:
      eta_m0_p1, eta_m0_m1, eta_m2_p1, eta_TE_p1, eta_TM_p1, eta_TE_m1,
      eta_TM_m1, cos_tot_p1, blaze_ratio
    """
    te = _solve_pol(rho_bar, H, "TE", (mx, -mx))
    tm = _solve_pol(rho_bar, H, "TM", (mx, -mx))
    m0_p, m2_p, cos_p = _eta_m0_from(te[mx], tm[mx], delta_ext)
    m0_m, _m2_m, _c_m = _eta_m0_from(te[-mx], tm[-mx], delta_ext)
    info = {
        "eta_m0_p1": m0_p, "eta_m0_m1": m0_m, "eta_m2_p1": m2_p,
        "eta_TE_p1": te[mx][0], "eta_TM_p1": tm[mx][0],
        "eta_TE_m1": te[-mx][0], "eta_TM_m1": tm[-mx][0],
        "cos_tot_p1": cos_p,
        "blaze_ratio": m0_p / (m0_p + m0_m + 1e-30),
    }
    return m0_p - m0_m, info


# ----------------------------------------------------------------------------
# binarize + ridge extraction + reference cross-check
# ----------------------------------------------------------------------------
def binarize(rho_bar):
    return (np.asarray(rho_bar) > 0.5).astype(float)


def extract_ridges(bin_rho, px_nm=None):
    """Binary pixel array -> list of (u0, u1) ridge intervals in period units
    (periodic wrap merged). Returns ridges, min_feature_nm (solid AND void).

    [260824] `px_nm` added. Without it the module-global PX_NM (0.5338/256 =
    2.08515625 nm) is used, which is WRONG for any caller with a different
    period/NX -- batch C is 1.31539/512 = 2.56912109375 nm, a factor 1.2321
    off, so every reported linewidth came out 18.9 % low. Existing callers pass
    nothing and are unaffected; `scd_c_grating` now passes its own PX_NM."""
    b = np.asarray(bin_rho).astype(int)
    n = b.size
    if b.min() == b.max():
        return ([(0.0, 1.0)] if b[0] == 1 else []), None
    # rotate so position 0 is a void->solid edge
    edges = np.where(np.diff(np.r_[b, b[0]]) != 0)[0]
    start = (edges[np.where(b[(edges + 1) % n] == 1)[0][0]] + 1) % n
    br = np.roll(b, -start)
    ridges, i = [], 0
    while i < n:
        if br[i] == 1:
            j = i
            while j < n and br[j] == 1:
                j += 1
            ridges.append(((i + start) % n / n, ((j + start) % n if j < n else start + n) / n))
            i = j
        else:
            i += 1
    # normalize wrap: represent as (u0, u1) with u1 possibly > 1 -> map back
    ridges = [(u0, u1 if u1 > u0 else u1 + 1.0) for u0, u1 in ridges]
    runs = []
    for v in (1, 0):
        bb = br if v == 1 else 1 - br
        cnt = 0
        for x in np.r_[bb, bb]:  # doubled for wrap runs
            if x == 1:
                cnt += 1
            else:
                if cnt:
                    runs.append(min(cnt, n))
                cnt = 0
        if cnt:
            runs.append(min(cnt, n))
    _px = PX_NM if px_nm is None else float(px_nm)
    minfeat_nm = min(runs) * _px if runs else None
    return ridges, minfeat_nm


def crosscheck_rcwa_multi(ridges, H):
    """Evaluate the binarized cell with the validated reference solver.
    Returns dict with eta per pol and dphi = arg(t_TM)-arg(t_TE) [deg], both for
    the order that grcwa calls mx = +1.

    [260804 FIX] The old code read res["i0"] + 1 on the claim that rcwa_multi's
    +1 label matches grcwa's mx = +1. It does not. T4c measured the pairing on an
    asymmetric cell and it is INVERTED:
        TE: |grcwa(+1) - multi(+1)| = 7.17e-03 ,  vs multi(-1) = 1.15e-07
        TM: |grcwa(+1) - multi(+1)| = 4.91e-03 ,  vs multi(-1) = 1.46e-04
    i.e. m_multi = -m_grcwa, because rcwa_multi uses kx_m = k0 n sin(theta) +
    m 2pi/P while the analysis notes use sin(theta_out) = k_t/k0 - m lambda/P.
    On a symmetric cell eta(+1) = eta(-1) hides this, which is why it survived;
    every optimized cell is asymmetric, so it does not.
    Getting this wrong corrupts BOTH the quoted efficiency AND dphi -- and dphi
    is what the cascade target Delta_top = -8.6 deg is measured against."""
    out = {}
    tt = {}
    for pol in ("TE", "TM"):
        res = rcwa_multi.solve_profile(
            wavelength=LAM, period=PERIOD, H=H, ridges=ridges,
            theta_inc=0.0, pol=pol,
            n_inc=1.0, n_out=N_SI, n_ridge=N_SI, n_groove=1.0, N=25,
        )
        i0 = res["i0"]
        i1 = i0 - 1          # = grcwa's mx = +1  (see docstring)
        im1 = i0 + 1         # = grcwa's mx = -1  (outward once revolved)
        out[f"eta_{pol}"] = float(res["DEt"][i1])
        # [260804] full order budget -- the sweep of this date showed the winner
        # was symmetric and dumping 33-61 % into the undeflected order, which the
        # old return value could not reveal.
        out[f"eta_m1_{pol}"] = float(res["DEt"][im1])
        out[f"T0_{pol}"] = float(res["DEt"][i0])
        out[f"T_total_{pol}"] = float(res["T_total"])
        out[f"R_total_{pol}"] = float(res["R_total"])
        out[f"energy_{pol}"] = float(res["energy"])
        tt[pol] = (res["t"][i1], res["t"][im1])
    out["dphi_deg"] = float(np.degrees(np.angle(tt["TM"][0]) - np.angle(tt["TE"][0])))
    out["dphi_m1_deg"] = float(np.degrees(np.angle(tt["TM"][1]) - np.angle(tt["TE"][1])))
    return out


# ----------------------------------------------------------------------------
# optimization (Adam ascent, beta continuation)
# ----------------------------------------------------------------------------
def optimize(rho0, H, iters=300, lr=0.03, beta_schedule=((0, 8.0), (0.4, 16.0), (0.7, 32.0), (0.9, 64.0)),
             filt_radius_px=FILT_RADIUS_PX, k_soft=30.0, log_every=25, tag="", objective="m0",
             delta_ext_deg=DELTA_BOT_DEG, w_t0=0.0, w_m1=0.0):
    """objective = "supp"   -> eta_m0(+1) - w_t0*T0_avg - w_m1*eta_m0(-1).  [260805]
                              The current line of work: T0 is the biggest loss in
                              every cell measured so far (46-58 %), and the blaze
                              trial proved that penalizing m = -1 on its own just
                              pushes power into T0. Both weights are job keys so
                              the Pareto front can be swept. w_t0 = w_m1 = 0
                              reduces to "m0" exactly.
       objective = "m0"     -> eta_m0(+1) alone: the co-polarized m = 0 coupling of
                              the wanted (inward) order. The 2026-08-04 default --
                              maximizing what reaches the receiver, but blind to
                              where the remaining 84 % goes.
       objective = "blaze"  -> eta_m0(+1) - eta_m0(-1). Tried 2026-08-04 and
                              MEASURED WORSE on every axis; see fom_blaze for the
                              numbers. Kept only so the comparison is reproducible.
       objective = "softmin" -> the original polarization-balance proxy, blind to
                              phase as well as to order content."""
    F = make_filter_matrix(NX, filt_radius_px)
    rho = np.clip(np.asarray(rho0, float).copy(), 0.0, 1.0)
    m = np.zeros(NX)
    v = np.zeros(NX)
    b1, b2, eps_ad = 0.9, 0.999, 1e-8

    def fom(r, beta):
        rho_bar = tanh_project(npa.dot(F, r), beta)
        if objective == "supp":
            return fom_supp(rho_bar, H, delta_ext=np.deg2rad(delta_ext_deg),
                            w_t0=w_t0, w_m1=w_m1)[0]
        if objective == "blaze":
            return fom_blaze(rho_bar, H, delta_ext=np.deg2rad(delta_ext_deg))[0]
        if objective == "m0":
            return fom_m0(rho_bar, H, delta_ext=np.deg2rad(delta_ext_deg))[0]
        # [260804] single-polarization objectives, added to test whether carrying
        # BOTH polarizations is what prevents blazing. Not for production use --
        # the revolved ring always presents both at once (radial = TM, azimuthal
        # = TE), so a cell tuned for one of them leaves the other uncontrolled.
        if objective in ("te", "TE"):
            return eta_scalar(rho_bar, H, "TE")
        if objective in ("tm", "TM"):
            return eta_scalar(rho_bar, H, "TM")
        return softmin(eta_scalar(rho_bar, H, "TE"), eta_scalar(rho_bar, H, "TM"), k_soft)

    fom_vg = value_and_grad(fom)
    hist = []
    for t in range(1, iters + 1):
        frac = t / float(iters)
        beta = [bv for fs, bv in beta_schedule if frac >= fs][-1]
        val, g = fom_vg(rho, beta)
        # Adam ASCENT (group convention: density += update)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        rho = np.clip(rho + lr * mh / (np.sqrt(vh) + eps_ad), 0.0, 1.0)
        hist.append(float(val))
        if log_every and (t % log_every == 0 or t == 1):
            print(f"  [{tag}] it {t:4d}  beta {beta:5.1f}  {objective}_eta {val:.5f}")
    beta_fin = beta_schedule[-1][1]
    rho_bar_fin = tanh_project(F @ rho, beta_fin)
    return rho, rho_bar_fin, hist


def seed_bank(n_random=5):
    seeds = []
    u = np.arange(NX) / NX
    # blaze-ramp seed (asymmetric by construction; direction sets +1 vs -1)
    seeds.append(("ramp", np.clip(1.25 - 1.5 * u, 0.0, 1.0)))
    for s in range(n_random):
        rng = np.random.RandomState(s)
        r = 0.5 + 0.12 * rng.randn(NX) + 0.15 * np.sin(2 * np.pi * (u + rng.rand()))
        seeds.append((f"rand{s}", np.clip(r, 0.0, 1.0)))
    return seeds


# ----------------------------------------------------------------------------
# validation suite
# ----------------------------------------------------------------------------
def validate():
    ok = True
    print(f"[geom] Lambda_top = {PERIOD*1e3:.1f} nm  ->  theta_Si = {THETA_SI_DEG:.3f} deg "
          f"(design 56.617)   H_pi = {H_PI*1e3:.1f} nm")

    # V1 -- Fresnel limits: uniform layer identical to a bare air/Si interface
    T_fres = 1.0 - ((N_SI - 1.0) / (N_SI + 1.0)) ** 2
    for rho_c, name in ((0.0, "all-void"), (1.0, "all-solid")):
        _, _, Ti, ob = eta_order(np.full(NX, rho_c), H_PI, "TE", mx=0)
        T0 = float(np.asarray(Ti)[_order_index(ob, 0)])
        good = abs(T0 - T_fres) < 1e-6
        ok &= good
        print(f"[V1] {name:9s} T0 = {T0:.6f}  vs Fresnel {T_fres:.6f}  "
              f"{'PASS' if good else 'FAIL'}")

    # V2 -- symmetric 50 % binary at pi depth: eta(+1) == eta(-1), energy ~ 1
    rho_sym = np.zeros(NX)
    rho_sym[NX // 4: 3 * NX // 4] = 1.0
    for pol in ("TE", "TM"):
        tp, Ri, Ti, ob = eta_order(rho_sym, H_PI, pol, mx=+1)
        tm = float(np.asarray(Ti)[_order_index(ob, -1)])
        en = float(np.sum(np.asarray(Ri)) + np.sum(np.asarray(Ti)))
        d_sym = abs(float(tp) - tm)
        good = d_sym < 1e-6 and abs(en - 1.0) < 1e-6
        ok &= good
        print(f"[V2] {pol}: eta(+1) = {float(tp):.5f}  eta(-1) = {tm:.5f}  "
              f"|d| = {d_sym:.2e}  energy = {en:.8f}  {'PASS' if good else 'FAIL'}")

    # V3 -- cross-check vs validated rcwa_multi (same symmetric cell)
    ref = crosscheck_rcwa_multi([(0.25, 0.75)], H_PI)
    for pol in ("TE", "TM"):
        tg, _, _, _ = eta_order(rho_sym, H_PI, pol, mx=+1)
        diff = abs(float(tg) - ref[f"eta_{pol}"])
        good = diff < 5e-3
        ok &= good
        print(f"[V3] {pol}: grcwa {float(tg):.5f}  rcwa_multi {ref[f'eta_{pol}']:.5f}  "
              f"|diff| = {diff:.2e}  {'PASS' if good else 'FAIL'}")
    print(f"[V3] dphi(sym cell) = {ref['dphi_deg']:+.2f} deg  (reference quantity)")

    # V4 -- autograd smoke: finite gradient, 3 ascent steps improve softmin FoM
    F = make_filter_matrix(NX, 16)
    rng = np.random.RandomState(0)
    rho = np.clip(0.5 + 0.1 * rng.randn(NX), 0, 1)

    def fom(r):
        rb = tanh_project(npa.dot(F, r), 8.0)
        return softmin(eta_scalar(rb, H_PI, "TE"), eta_scalar(rb, H_PI, "TM"))

    vg = value_and_grad(fom)
    v0, g = vg(rho)
    good = np.all(np.isfinite(g)) and np.isfinite(v0)
    v1 = v0
    for _ in range(3):
        rho = np.clip(rho + 0.05 * np.sign(g) * np.abs(g) / (np.abs(g).max() + 1e-12), 0, 1)
        v1, g = vg(rho)
    ok &= good and (v1 > v0)
    print(f"[V4] grad finite: {bool(np.all(np.isfinite(g)))}  FoM {v0:.5f} -> {v1:.5f} "
          f"after 3 steps  {'PASS' if good and v1 > v0 else 'FAIL'}")

    print(f"[validate] {'ALL PASS' if ok else 'SOME FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="zone (2) top grating draft optimizer (grcwa)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "best_cell_top.json"))
    args = ap.parse_args()

    if args.validate or not (args.smoke or args.full):
        validate()
        if not (args.smoke or args.full):
            return

    H_list = [H_PI] if args.smoke else [0.28, H_PI, 0.36, 0.40]
    seeds = seed_bank(0 if args.smoke else 5)   # smoke: ramp seed only
    iters = 30 if args.smoke else args.iters

    here = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(here, "runs_log.tsv")
    best = None
    with open(log_path, "a", encoding="utf-8") as lg:
        lg.write("# tag\tH_um\teta_TE\teta_TM\tsoftmin\teta_m0\tminfeat_nm\tdphi_deg\tverdict\n")
        for H in H_list:
            for name, rho0 in seeds:
                tag = f"H{H*1e3:.0f}_{name}"
                print(f"[run] {tag}  iters={iters}")
                rho, rho_bar, hist = optimize(rho0, H, iters=iters, tag=tag)
                b = binarize(rho_bar)
                ridges, minfeat = extract_ridges(b)
                if not ridges:
                    print(f"  [{tag}] degenerate (empty/full) -- skipped")
                    continue
                chk = crosscheck_rcwa_multi(ridges, H)
                e_te, e_tm = chk["eta_TE"], chk["eta_TM"]
                sm = min(e_te, e_tm)
                # Rank by the SAME quantity the optimizer maximizes -- the
                # cascade-aware co-polarized (m = 0) coupling. Ranking by
                # min(eta_TE, eta_TM) while optimizing eta_m0 would let a cell
                # with a bad dphi win. Evaluated on rcwa_multi's numbers so only
                # reference-solver values are ever quoted.
                _d = np.deg2rad(chk["dphi_deg"] + DELTA_BOT_DEG)
                eta_m0_x = (e_te + e_tm + 2.0 * np.sqrt(e_te * e_tm) * np.cos(_d)) / 4.0
                feasible = (minfeat is not None) and (minfeat >= MIN_FEATURE_NM)
                verdict = "ok" if feasible else f"REJECT_minfeat<{MIN_FEATURE_NM:.0f}nm"
                lg.write(f"{tag}\t{H:.4f}\t{e_te:.5f}\t{e_tm:.5f}\t{sm:.5f}\t{eta_m0_x:.5f}"
                         f"\t{minfeat if minfeat else -1:.1f}\t{chk['dphi_deg']:+.2f}\t{verdict}\n")
                print(f"  [{tag}] binarized: eta_TE {e_te:.4f}  eta_TM {e_tm:.4f}  "
                      f"eta_m0 {eta_m0_x:.4f}  minfeat {minfeat:.0f} nm  "
                      f"dphi {chk['dphi_deg']:+.1f} deg  {verdict}")
                if not feasible:
                    continue          # not fabricable -- never becomes the winner
                if best is None or eta_m0_x > best["eta_m0"]:
                    best = {
                        "created": "2026-08-02 (DRAFT -- not a final design)",
                        "zone": "top outer (2), air->Si, normal incidence",
                        "period_nm": PERIOD * 1e3,
                        "H_nm": H * 1e3,
                        "ridges_period_units": [[round(u0, 5), round(u1, 5)] for u0, u1 in ridges],
                        "ridges_nm": [[round(u0 * PERIOD * 1e3, 1), round(u1 * PERIOD * 1e3, 1)]
                                      for u0, u1 in ridges],
                        "eta_TE_plus1": e_te,
                        "eta_TM_plus1": e_tm,
                        "min_eta": sm,
                        "eta_m0": eta_m0_x,
                        "eta_m0_note": "co-pol m=0 coupling of the revolved ring, "
                                       "cascade phase included; this is the ranking key",
                        "min_feature_nm": minfeat,
                        "min_feature_spec_nm": MIN_FEATURE_NM,
                        "dphi_TM_minus_TE_deg": chk["dphi_deg"],
                        "dphi_cascade_target_deg": DPHI_TARGET_DEG,
                        "method": f"grcwa autograd topology opt, NX={NX}, nG={NG}, "
                                  f"softmin k=30, conic r=16px, beta 8->64, seed={name}",
                        "crosscheck": "binarized cell re-evaluated with rcwa_multi N=25 "
                                      "(values quoted above are the rcwa_multi numbers)",
                        "reference_notes": "symmetric binary pi-depth reference ~0.405/order; "
                                           "blaze breaks the +1/-1 tie. Order-sign trap: see docstring.",
                    }
    if best:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print(f"[out] best cell -> {args.out}")
        print(f"[out] log       -> {log_path}")


if __name__ == "__main__":
    main()
