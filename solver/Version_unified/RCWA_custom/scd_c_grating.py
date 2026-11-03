"""Layout C (folded SCD receiver, fused-silica substrate): common module for the 3 gratings.

Structure (fixed 2026-08-12, **updated 260821 -- the numbers below assume t=600 um**)
----------------------
Fused-silica **600 um** substrate. Two or more gratings on top, two gratings on the bottom.
Spacing is s = t*tan(th) = **847.12 um**.

    Input grating      (top, x=0)      air normal incidence -> substrate +1 order (54.69 deg)
                                       TE passes as order 0 to the central MS (polsplit)
    Branch grating     (bottom, x=0)   TE lane. reference arm T(0)|normal / delay arm R(+1)|normal
    Extraction grating (bottom, x=s)   TM lane. extracts part of the +1 order (reference arm) / the rest
                                       passes by specular reflection, extracts the returning -1 order (delay arm)
    Return grating     (top, x=s)      the **TE** lane arrives -> R(-2) of TE
    Return grating     (top, x=2s)     the **TM** lane arrives -> R(-2) of TM

Path difference 2t/cos(th) = **2076.2 um -> group delay 10.0002 ps**.  Net lateral shift 0.

    [260821] This paragraph was **outdated** (t=300 um / 424 um / 5.0000 ps).
    The current design has t=600 and those values were exactly half. It also missed that the
    return position is **two** places depending on polarization, which caused confusion over "x=s or x=2s".
    See the `fom_retro_te` docstring for the definition of the two positions.

    **Device width caveat.** W = 2s = 1694.24 um is **only the ray-path (group delay) dimension**
    and holds no width margin for the w0 = 100 um beam. For a margin per side of 0/50/100/154.5 um the
    clipping is -3.01/-8.00/-16.43/-30.00 dB, so aiming at -30 dB needs a total width of about
    **2003 um (+18.3 %)**. Reflect this in the die specification.

Why this layout
--------------
The shear structure sent the two arms to different points, so the beam had to be much wider than s (=796 um)
(w0 >= 1.6 mm), and the top/bottom periods had to match to 30 ppm. Folding makes the net lateral shift 0,
so the beam can shrink to 100 um, and the period tolerance relaxes to 960 ppm.
+10.4 dB over shear, based on the measured component set.

Caution: the two arms must not be made **exactly the same state** (this was once designed that way).
With the same mode the fields add as E_s + E_d and |E_s - E_d| cannot be formed, so a BPD
becomes impossible. Layout C keeps distinguishability by having the two arms arrive at the same point at
**opposite angles (+G / -G)**.

Physics check (2026-08-12)
----------------------
At Lambda = 1315.39 nm the only propagating states are substrate m = 0, +-1 and air m = 0.
  air +-1 : lambda/Lambda = 1.178 > 1  -> evanescent
  substrate +-2 : 2*lambda/(n*Lambda) = 1.632 > 1 -> evanescent
The required transitions (dm = +1 / -2 / -1 / +1 / 0) all exist, and almost no propagating channel is left
for ghost light. Cleaner than the Si structure at Lambda = 533.8 nm.

**GMR risk (a new problem with the quartz substrate).** The grating layer (n >= 1.75) is higher than
the substrate (1.444), so it forms a slab waveguide with guided-mode resonances. The cutoff thicknesses
are a-Si 25 nm, Si3N4 116 nm, Al2O3 205 nm, so all of our thicknesses (~400 nm) guide. GMR is sharp and
can give solutions extremely sensitive to wavelength/angle/thickness, so every FoM is computed as the
**3-wavelength geometric mean**. It must be flat over the 200 Gbaud signal band of 1.60 nm
(allowed Q < 967); a narrow resonant solution is killed automatically by the geometric mean.
"""
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(_HERE, "_vendor")
for _p in (_HERE, VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                          # noqa: E402
import autograd.numpy as npa                # noqa: E402
from autograd import value_and_grad         # noqa: E402

import grcwa                                # noqa: E402
grcwa.set_backend("autograd")

import top_grating_opt as T1                # noqa: E402

LAM = 1.55
N_SUB = 1.444                    # fused silica
EPS_SUB = N_SUB ** 2
EPS_AIR = 1.0

# Only what can be deposited by PECVD / LPCVD / sputtering / ALD.
# Crystalline Si cannot be deposited, so a-Si:H / poly-Si is used instead.
MATERIALS = {
    "a-Si":   (3.48,  0.0),      # PECVD a-Si:H
    # [261013 opt-in] a-Si extinction sensitivity (design stays lossless, k only as a sensitivity); a job that names
    # none of these is unchanged. Carrier: material1 / material2; atom: material3; input2d: material1 (Phys keeps complex eps).
    "a-Si_k0":    (3.48, 0.0),
    "a-Si_k1e-5": (3.48, 1.0e-5),
    "a-Si_k1e-4": (3.48, 1.0e-4),
    "a-Si_k1e-3": (3.48, 1.0e-3),
    "a-Si_k1e-2": (3.48, 1.0e-2),
    "a-Si_k1e-1": (3.48, 1.0e-1),    # needle only: must show a clear absorption deficit
    "polySi": (3.48,  0.0),      # LPCVD poly-Si
    "Si3N4":  (1.996, 0.0),      # LPCVD stoichiometric
    "SiN":    (1.95,  0.0),      # PECVD, 1.8~2.1 depending on composition
    "TiO2":   (2.40,  0.0),
    "Nb2O5":  (2.25,  0.0),
    "Ta2O5":  (2.07,  0.0),
    "HfO2":   (1.88,  0.0),
    "Al2O3":  (1.746, 0.0),
    "SiO2":   (1.444, 0.0),      # PECVD / same as the substrate
    # [260813] SU-8. A photosensitive epoxy, so it is **patterned by exposure alone, without etching**,
    # and as a filler its spin coating planarises naturally, so CMP can be dropped too.
    # n = 1.57 @1550 nm (published value 1.5706 for the SU-8 2000 family). 1550 nm lies between the C-H
    # overtone absorption bands (~1.4, ~1.7 um), so the loss is low -- k=0 is used here.
    # Submicron films (SU-8 2000.5 etc.) reach ~0.5 um, so a 600 nm layer is possible.
    "SU-8":   (1.57,  0.0),
    "air":    (1.0,   0.0),
}

PERIOD = 1.31539                 # um -- back-calculated from a 300 um slab / 5 ps
NX = 512
PX_NM = PERIOD * 1e3 / NX        # 2.57 nm/px
L1 = [PERIOD, 0.0]
L2 = list(T1.L2)
# [260907 re-correction] **The limit 257 is correct.** Earlier this spot was changed to "the actual limit is
# nG ~ 513", and that was wrong.
# What was wrong was **which quantity is measured**: `max|m|` was compared against NX/2, but the
# bound applies to the **difference of orders** `max|m_i - m_j|`.
#     grcwa/fft_funs.py:46   s_out = sfft[G[ii,0]-G[jj,0], G[ii,1]-G[jj,1]]
# The convolution matrix is indexed by the **difference** of two retained orders. The Fourier
# coefficients of an NX-point pattern are unique only for |k| <= NX/2, so the condition is `2*max|m| <= NX/2`,
# i.e. **max|m| <= NX/4 = 128 -> nG <= 257**. Above that, results do not become more accurate; they
# silently wrap into aliased coefficients and are **wrong in a different way**.
# The measurement itself was correct, so the table is kept (the difference column shows where the
# conclusion changes). L1=[1.31539,0], L2=[0,1e-4], NX=512:
#     nG input   161   241   257   321   513   521
#     pairs left 159   239   255   319   511   519
#     max|m|      79   119   127   159   255   259
#     max|m_i-m_j| 158  238   254   318   510   518   <- NX/2 = 256 is the limit
#     alias?      no    no    no   YES   YES   YES
# Measured confirmation (260907 ngladder, a3_00 own density, eval_only=1):
# nG_eval requests of 401/481/513 gave **bit-identical** results -- all were clamped to 257.
# Reading this as "converged at 321" mistakes a clamp artifact for convergence. Only 161/241/257
# are usable points, and 161 is 1.11 pp lower in efficiency than its own
# asymptotic value.
# (`Main_rcwa_scd_c.py:493-496` already states this mechanism exactly and
#  applies the clamp. Anyone doubting that comment should first read the
#  line above.)
NG = 81                          # for optimisation
NG_EVAL = 161

SIN_TH = LAM / (PERIOD * N_SUB)
THETA_SUB = float(np.arcsin(SIN_TH))
THETA_0 = T1.THETA0

# [260814] 3 wavelengths (+-0.8 nm) -> 5 wavelengths over the C-band. With the direction fixed as "single-
# polarization transmission + polarization diversity + WDM", the objective has to look at the **WDM span**
# instead of the signal band (200 Gbaud = 1.6 nm).
#
# Why 35 nm: the relative delay dispersion is (1/tau)(dtau/dlam) = tan^2(theta)/lam,
# which at theta_sub = 54.69 deg is 4.50 % per 35 nm. With a DQPSK delay tolerance of 5 %,
# 38.9 nm is the physical upper bound, so optimising the grating wider than that gains nothing as the
# delay line cannot follow. The 5 points are matched to that bound.
LAMS = (1.530, 1.540, LAM, 1.560, 1.565)

# [260814] 80 -> 70 nm. A runs_log reconstruction showed that many solutions discarded at 80
# survive at 70: best input 0.9027 -> 0.9625 (+6.6 %),
# and in the bottom grating a 65.7 % case was rejected at linewidth 70.9 nm (better than the
# current champion 63.6 %). Below 70 there is no further gain at the input, so it stops here.
MIN_FEATURE_NM = 70.0
FILT = T1.make_filter_matrix(NX, max(1, int(round(MIN_FEATURE_NM / PX_NM))))

# [260812] Interlayer spacer. When L1 is etched and the grooves are refilled by deposition (filling), the film
# fills the grooves and **also covers the top thinly**. L2 is placed on it, so a uniform
# dielectric plate forms between the two pattern layers. Unavoidable unless full planarisation
# (CMP down to the L1 top) is done; conversely it is the margin that lets CMP be skipped.
#
#   Material: same as fill1 -- it is the very deposited film that filled L1 rising up,
#         so it is not a separate material. Hence _solve passes eps_f1 on unchanged.
#   Thickness: 0 ~ 100 nm. Not a constraint but **one degree of freedom** (tunes the interlayer coupling phase).
#   Premise: if fill1 = air there is no deposition and hence no gap.
GAP_UM = 0.0
# [260922] List of gaps paired with the termination axis [nm]. If empty, GAP_UM is used unchanged.
GAP_LIST = ()
# [260924] OPT-IN 2-D IN-PLANE CARRIER ("3-D sim"), `set_grid_2d`. None = the 1-D
# path, which is untouched: every branch below tests `GRID_2D is None` first.
# When on, each layer is an (nx, ny) density stored FLATTENED (C order) in a vector
# of length NX = nx*ny, the y lattice is PERIOD/app and the conic filter is applied
# by FFT (a dense NX x NX matrix would be 6 GB). The grid is the atom driver's
# (`Main_rcwa_l3_atom2d_opt._obj`: nx = app*nxy, ly = PERIOD/app), so a 2-D carrier
# drops into it unchanged through `l12_rho1_npy` / `l12_rho2_npy`.
GRID_2D = None
APP_2D = 3
KF_2D = None
SYM_Y_2D = True
# [261008 li_integrate] OPT-IN Fourier factorization of every patterned layer built by `_build`.
# Defaults reproduce the stock Laurent rule exactly: `_build` touches the object only when one of
# these is non-default, and `_state_sig` adds them to the checkpoint fingerprint only then.
RCWA_RULE = "laurent"         # laurent | liy | lix   (vendored grcwa li_fmm.py, Li 1997)
RCWA_FF = 0                   # 0 | 1                 exact pixel form factor sinc(m/N)


def set_rcwa(rule="laurent", ff=0):
    """[261008 li_integrate] Select the factorization rule for the grid layers built from now on.
    Returns (rule, ff). Raises ValueError on an unknown value and RuntimeError when the vendored
    grcwa in use has no Li support (a stale tree must fail, not silently run Laurent)."""
    global RCWA_RULE, RCWA_FF
    rule = str(rule).strip().lower()
    if rule not in ("laurent", "liy", "lix"):
        raise ValueError("rcwa_rule %r (allowed: laurent, liy, lix)" % (rule,))
    if float(ff) not in (0.0, 1.0):
        raise ValueError("rcwa_ff %r (allowed: 0, 1)" % (ff,))
    if (rule != "laurent" or float(ff)) and not getattr(grcwa.obj, "LI_SUPPORT", 0):
        raise RuntimeError("the vendored grcwa in use has no Li support (obj.LI_SUPPORT missing)")
    RCWA_RULE, RCWA_FF = rule, int(float(ff))
    return RCWA_RULE, RCWA_FF


def set_gap(nm):
    """Interlayer spacer thickness [nm]. 0 gives the earlier model where the two layers touch."""
    global GAP_UM
    GAP_UM = max(0.0, float(nm)) * 1e-3
    return GAP_UM * 1e3


def set_gap_list(vals_nm):
    """[260922] Interlayer gap robustness fold. Same shape as `set_out_l3_list`.

    `vals_nm` must have the **same length and order** as `OUT_L3_NS` -- the fold
    loop pairs them by the same index (zip, not a Cartesian product). Length validation is
    done on the driver side in `gap_nm_list`.

    If cleared, it is released and the fold loop afterwards does not touch GAP_UM -- no extra
    solve and no numerical difference.
    """
    global GAP_LIST
    GAP_LIST = tuple(float(x) for x in vals_nm) if vals_nm else ()
    return GAP_LIST

binarize = T1.binarize
def extract_ridges(bin_rho, px_nm=None):
    """Wrapper of `T1.extract_ridges` -- passes **the pixel size of this module**.

    [260824] Previously `T1.extract_ridges` was aliased directly, but the `top_grating_opt.PX_NM`
    it uses is for the top grating (0.5338/256 = 2.085 nm), which was off by a factor **1.2321** from the
    actual pixel of layout C (1.31539/512 = 2.569 nm). Reported linewidths were
    all 18.9 % too small and the gate was tightened that much too far (setting 56.81 actually
    demanded 70.0 nm). `PX_NM` is changed by `set_period`, so it is read at call time."""
    if GRID_2D is not None:
        # [260924] 2-D: no ridge list; the min feature is the 2-D medial-axis measure.
        _m = _min_feature_2d(bin_rho, PX_NM if px_nm is None else px_nm)
        return [], (None if not np.isfinite(_m) else float(_m))
    return T1.extract_ridges(bin_rho, PX_NM if px_nm is None else px_nm)


def eps_of(name_or_n):
    if isinstance(name_or_n, str):
        if name_or_n in ("air", "Air", "", None):
            return EPS_AIR
        n, k = MATERIALS[name_or_n]
    else:
        n, k = float(name_or_n), 0.0
    return (n + 1j * k) ** 2


def set_period(period_nm):
    global PERIOD, L1, PX_NM, SIN_TH, THETA_SUB, L2
    PERIOD = float(period_nm) * 1e-3
    L1 = [PERIOD, 0.0]
    if GRID_2D is None:
        PX_NM = PERIOD * 1e3 / NX
    else:
        # [260924] 2-D: NX is the flattened count; the pixel is PERIOD / nx and is
        # square because ly = PERIOD / app and nx = app * ny.
        PX_NM = PERIOD * 1e3 / GRID_2D[0]
        L2 = [0.0, PERIOD / float(APP_2D)]
    SIN_TH = LAM / (PERIOD * N_SUB)
    THETA_SUB = float(np.arcsin(SIN_TH))
    set_min_feature(MIN_FEATURE_NM)
    return PERIOD, float(np.degrees(THETA_SUB))


def set_substrate(n_sub):
    global N_SUB, EPS_SUB, SIN_TH, THETA_SUB
    N_SUB = float(n_sub)
    EPS_SUB = N_SUB ** 2
    SIN_TH = LAM / (PERIOD * N_SUB)
    THETA_SUB = float(np.arcsin(SIN_TH))
    return N_SUB, float(np.degrees(THETA_SUB))


# [261015 kx_hold] OPT-IN matched-angle evaluation. Default OFF = previous behaviour, byte for byte.
# The input metagrating fixes the in-plane wave number kx = G = 2 pi / PERIOD, so in the device the
# substrate angle follows the wavelength: sin(theta) = lam / (PERIOD * N_SUB). THETA_SUB above is the
# value at LAM = 1.55 um (a module constant); every driver read it for EVERY channel (fixed-angle model).
#   * multi-channel drivers (Main_rcwa_scd_c.py): set_kx_hold(True); `_report_one` then solves each
#     channel of LAMS at theta_at(lam) (the optimiser keeps THETA_SUB).
#   * single-wavelength drivers (atom, lens cells): pin_theta_to(lam) makes THETA_SUB itself the
#     matched angle at lam (every `G.THETA_SUB` read happens at call time).
KX_HOLD = False


def set_kx_hold(on=True):
    global KX_HOLD
    KX_HOLD = bool(on)
    return KX_HOLD


def theta_at(lam_um):
    """Substrate angle [rad] used for channel lam_um: THETA_SUB unless KX_HOLD (then kx = G held)."""
    if not KX_HOLD:
        return THETA_SUB
    s = float(lam_um) / (PERIOD * N_SUB)
    if not (0.0 < s < 1.0):
        raise ValueError("kx_hold: lam/(PERIOD*N_SUB) = %.6f is not a propagating angle" % s)
    return float(np.arcsin(s))


def pin_theta_to(lam_um):
    """Single-wavelength drivers: THETA_SUB := arcsin(lam / (PERIOD * N_SUB)). Returns degrees.
    At lam = LAM the expression is the one of set_period / set_substrate, so it is bit-identical."""
    global SIN_TH, THETA_SUB
    SIN_TH = float(lam_um) / (PERIOD * N_SUB)
    if not (0.0 < SIN_TH < 1.0):
        raise ValueError("kx_hold: lam/(PERIOD*N_SUB) = %.6f is not a propagating angle" % SIN_TH)
    THETA_SUB = float(np.arcsin(SIN_TH))
    return float(np.degrees(THETA_SUB))


def set_min_feature(nm):
    global FILT, MIN_FEATURE_NM, KF_2D
    MIN_FEATURE_NM = float(nm)
    px = max(1, int(round(MIN_FEATURE_NM / PX_NM)))
    if GRID_2D is None:
        FILT = T1.make_filter_matrix(NX, px)
    else:
        KF_2D = _conic_kernel_fft_rect(GRID_2D[0], GRID_2D[1], px)
    return px


def _conic_kernel_fft_rect(nx, ny, radius_px):
    """[260924] Periodic 2-D conic filter (FFT), same cone as `T1.make_filter_matrix`
    (w = 1 - d/r) with d the Euclidean pixel distance. Copied from
    `Main_rcwa_l3_carrier2d.conic_kernel_fft_rect` (square pixels)."""
    r = max(1, int(round(radius_px)))
    dx = np.minimum(np.arange(nx), nx - np.arange(nx))
    dy = np.minimum(np.arange(ny), ny - np.arange(ny))
    DX, DY = np.meshgrid(dx, dy, indexing="ij")
    w = np.maximum(0.0, 1.0 - np.sqrt(DX ** 2 + DY ** 2) / float(r))
    w = w / w.sum()
    return np.fft.fft2(w)


def set_grid_2d(nxy, app=3, sym_y=True):
    """[260924] Switch this module to the 2-D in-plane carrier (call BEFORE
    `set_period` / `set_min_feature`; the driver does this for `carrier_2d = 1`).
    nx = app*nxy, ny = nxy, ly = PERIOD/app. Returns (nx, ny, PX_NM)."""
    global GRID_2D, APP_2D, NX, PX_NM, L2, FILT, SYM_Y_2D
    nxy, app = int(nxy), int(app)
    if nxy < 4 or app < 1:
        raise ValueError("set_grid_2d: nxy %d / app %d" % (nxy, app))
    GRID_2D = (app * nxy, nxy)
    APP_2D = app
    SYM_Y_2D = bool(sym_y)
    NX = GRID_2D[0] * GRID_2D[1]
    PX_NM = PERIOD * 1e3 / GRID_2D[0]
    L2 = [0.0, PERIOD / float(app)]
    FILT = None
    set_min_feature(MIN_FEATURE_NM)
    return GRID_2D[0], GRID_2D[1], PX_NM


def extrude_1d(prof, nx_out=None, ny=None):
    """[260924] A stored 1-D layer profile (any length) -> the 2-D grid, flattened.
    Area-weighted resample along x (the atom driver's `_resample`, so an edge pixel
    carries its fill fraction), then repeated along y."""
    nx_out = GRID_2D[0] if nx_out is None else int(nx_out)
    ny = GRID_2D[1] if ny is None else int(ny)
    col = np.asarray(prof, float).ravel()
    n0 = col.size
    # verbatim `Main_rcwa_l3_atom2d_opt.main._resample` (with nx -> nx_out)
    c = np.concatenate([[0.0], np.cumsum(col)])
    e = np.arange(nx_out + 1) * (n0 / float(nx_out))
    lo = np.clip(np.floor(e).astype(int), 0, n0)
    fr = e - np.floor(e)
    ci = c[lo] + fr * col[np.clip(lo, 0, n0 - 1)]
    r = (ci[1:] - ci[:-1]) / (n0 / float(nx_out))
    return np.repeat(r[:, None], ny, axis=1).reshape(-1)


def _medial_min_nm(rho, px_nm, thr=0.5):
    """Verbatim copy of `Main_rcwa_l3_atom2d_opt.min_feature_nm` (260917 medial-axis
    rule; seven controls documented there). Copied, not imported: that module
    parses sys.argv[1] as its own job file at import time."""
    b = (np.asarray(rho) > thr)
    if not b.any() or b.all():
        return float("inf")
    D = np.zeros(b.shape, int)
    r = 1
    while r <= min(b.shape) // 2:
        k = 2 * r + 1
        off = [(i - r, j - r) for i in range(k) for j in range(k)
               if (i - r) ** 2 + (j - r) ** 2 <= r * r]
        ero = np.logical_and.reduce(
            [np.roll(np.roll(b, dy, 0), dx, 1) for dy, dx in off])
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


def _min_feature_2d(b, px_nm):
    """[260924] Measured min feature of a binary 2-D layer, solid AND void, by the
    atom driver's medial-axis rule. It reports (2*D+1) px, so an EVEN-width stripe
    reads 1 px thin (needle grid 2.569 nm/px: e200's 20 px = 51.4 nm reads 48.8).
    On the production grid (4.567 nm/px) that cannot flip the 50 nm gate: 11 px =
    50.24 nm is odd and exact, and the next even width that could be under-read,
    12 px, still reads 11 px. Axis-aligned run lengths were tried and rejected: a
    disc's tip chord reads 1 px."""
    a = np.asarray(b, float).reshape(GRID_2D)
    return min(_medial_min_nm(a, px_nm), _medial_min_nm(1.0 - a, px_nm))


def unsupported(dens, fills, px_nm=None):
    """Levitation check -- catches shapes that cannot be made by a deposition stack.

    [260819] The rule itself had been documented since 260813 (`optimize` docstring), but
    **there was no gate**, so 20 cells of the 3-layer B/D stack (fill2=air, nest=0) passed with
    `feasible=True`. Example: index 2154 has 38.3 % of L3 floating over empty space
    with a maximum continuous floating width of 277 nm.

    Criteria:
      * If the filler is **solid**, that layer is mechanically continuous and supports the layer above
        (fill + planarisation process).
      * With `fill = air`, the rho = 0 sites are truly empty, so the material above them has
        nothing to rest on -- it cannot be made by deposition.
      * Below L1 is the substrate, so it is always supported.

    dens is the binary density listed **from the substrate side**, fills the filler names in the same order.
    The period boundary is cyclic, so the continuous floating width is also counted cyclically.

    Returns (maximum continuous floating width [nm], list of per-layer floating fractions).
    """
    px = PX_NM if px_nm is None else float(px_nm)
    ds = [np.asarray(d, float).ravel() > 0.5 for d in dens]
    n = ds[0].size
    below = np.ones(n, dtype=bool)                 # substrate
    fracs, span_px = [], 0
    for mat, f in zip(ds, list(fills)):
        floating = mat & ~below
        fracs.append(float(floating.sum()) / max(int(mat.sum()), 1))
        if floating.any():
            if GRID_2D is None:
                span_px = max(span_px, _max_run_circular(floating))
            else:
                # [260924] 2-D: the widest floating run along x in any y row.
                span_px = max(span_px, max(_max_run_circular(row) for row in
                                           floating.reshape(GRID_2D).T))
        below = np.ones(n, dtype=bool) if (f or "air") != "air" else mat
    return span_px * px, fracs


def _max_run_circular(mask):
    """Maximum length [px] of a True run continuing across the period boundary."""
    if mask.all():
        return int(mask.size)
    best = run = 0
    for v in np.concatenate([mask, mask]):
        run = run + 1 if v else 0
        if run > best:
            best = run
    return min(best, int(mask.size))


def density(x, beta):
    if GRID_2D is not None:
        # [260924] 2-D: y-mirror (y -> -y) keeps the TM lane polarisation-pure in
        # the m_y = 0 orders; then the periodic conic filter by FFT.
        a = npa.reshape(x, GRID_2D)
        if SYM_Y_2D:
            a = 0.5 * (a + npa.roll(a[:, ::-1], 1, axis=1))
        f = npa.real(npa.fft.ifft2(npa.fft.fft2(a) * KF_2D))
        return npa.reshape(T1.tanh_project(f, beta), (-1,))
    return T1.tanh_project(npa.dot(FILT, x), beta)


def lw_geometry(target_nm, c=None):
    """[260926] Zhou, Lazarov, Wang, Sigmund (CMAME 293, 2015, 266-282)
    minimum length-scale indicator geometry, computed once per run. Every
    quantity is in PIXEL units of the 1-D conic filter FILT actually in use
    (set by set_min_feature, i.e. filt_radius_nm, not min_feature_nm).

    R_px  -- true radius of FILT (2*R_px-1 nonzero taps per row).
    L_px  -- target linewidth (nm) rounded UP to whole pixels.
    q     -- L_px / R_px.
    eta_e -- Qian and Sigmund (2013) conic-filter threshold for a solid
             member of width >= L_px (meep python/adjoint/filters.py
             get_eta_from_conic, same piecewise formula).
    eta_d -- 1 - eta_e (the void-side threshold).
    c     -- indicator decay constant in exp(-c*g^2); default 64*R_px**2.

    Raises ValueError when q >= 2.0 -- the filter is too small to enforce
    the target at all. 1-D path only (called only when GRID_2D is None).
    """
    R_px = int((np.count_nonzero(FILT[0]) + 1) // 2)   # int: np.int64 breaks json.dump
    L_px = int(np.ceil(float(target_nm) / PX_NM))
    if L_px < 2:
        raise ValueError("lw_geometry: target %.3f nm gives L_px=%d < 2"
                         % (float(target_nm), L_px))
    if c is not None and float(c) <= 0.0:
        raise ValueError("lw_geometry: lw_c = %r must be > 0" % (c,))
    q = L_px / float(R_px)
    if q >= 2.0:
        raise ValueError(
            "lw_geometry: target %.3f nm needs L_px=%d against filter "
            "radius R_px=%d (q=%.4f >= 2.0) -- filter too small to enforce "
            "this target" % (float(target_nm), L_px, R_px, q))
    if q <= 1.0:
        eta_e = 0.5 + q * q / 4.0
    else:
        eta_e = 1.0 - (1.0 - q / 2.0) ** 2
    eta_d = 1.0 - eta_e
    cc = float(c) if c else 64.0 * R_px * R_px
    return {"R_px": R_px, "L_px": L_px, "q": q, "eta_e": eta_e,
            "eta_d": eta_d, "c": cc}


def lw_penalty(x, beta, eta_e, c):
    """[260926] Zhou et al. geometric minimum-length-scale penalty on one
    filtered layer `x` (length NX, 1-D path only). Returns a scalar >= 0,
    exactly 0 when every solid skeleton point sits at f >= eta_e and every
    void skeleton point at f <= 1 - eta_e. Traced end-to-end in autograd --
    see optimize() for how the value is scaled and gated by beta.
    """
    f = npa.dot(FILT, x)
    p = T1.tanh_project(f, beta)
    g = 0.5 * (npa.roll(f, -1) - npa.roll(f, 1))
    ind = npa.exp(-c * g * g)
    s = npa.minimum(f - eta_e, 0.0)
    v = npa.minimum((1.0 - eta_e) - f, 0.0)
    return npa.sum(ind * (p * s * s + (1.0 - p) * v * v)) / (eta_e - 0.5) ** 2


def orders(kind="sub"):
    """List of propagating orders. kind = 'sub' (substrate) or 'air'."""
    q = LAM / (PERIOD * N_SUB) if kind == "sub" else LAM / PERIOD
    return [m for m in range(-6, 7) if abs(m * q) < 1.0]


# ------------------------------------------------------------------ analysis ---
OUT_SPACER_UM = 0.0               # [260827] output-side spacer thickness [um]
OUT_SPACER_EPS = None             #          None means SiO2 (n=1.4440)
OUT_L3_UM = 0.0                   # [260901] output-side uniform L3 slab thickness [um]
OUT_L3_EPS = None                 #          None means inactive (default n=1.575)
OUT_L3_NS = ()                    # [260901] multi-termination index list -- if empty, the single termination is unchanged
OUT_L3_NS_NM = 0.0                #          shared thickness [nm] (same unit as the nm argument of set_out_l3)
OUT_L3B_UM = 0.0                  # [260914] thickness [um] of the **inner** (spacer side) layer of a 2-layer termination
OUT_L3B_EPS = None                #          None means no layer -- identical to the 1-layer path
OUT_L3B_LIST = ()                 # [260914] per-termination (n_b, t_b[nm]); if empty, the 1-layer fold is unchanged
# [260923] SiO2 gap between the L3 layers of atom2d_opt -- only tells the
# `spacer_purity`/`spacer_fid` back-calculation count that one more layer sits behind the spacer.
# At 0 (default) the `_sp_idx` expressions of the two functions below are bit-identical to before.
OUT_L3_GAP_UM = 0.0


def set_out_spacer(nm, n=1.4440):
    """[260827] **Output-side spacer (above L2).** This is where the L3 metaatom goes.

    The earlier `GAP_UM` is an interlayer spacer **between** two grating layers, so it differs from this.

    Why it is needed -- the bottom grating has so far been optimised as `L1/L2 -> air`, but the real
    stack is `L1/L2 -> SiO2 spacer -> L3 -> air`. The boundary condition differs, so
    the optimum differs too. 260827 measurement (c74, TM lane):

        direct to air (optimisation condition)   signal 0.29579
        + SiO2 200 nm            0.28866   (-0.106 dB)
        + SiO2 400 nm            0.22854   (**-1.120 dB**)

    200 nm is merely a lucky good point and is very sensitive to thickness -- because of the FP cavity
    between L1/L2 and L3 (period lam/(2 n cos th) = 928.6 nm).

    **And this makes the metaatom design easier.** With air output, the parasitic orders m=0, m=-2
    (+-54.69 deg in the spacer) are **evanescent and not seen by the FoM**. With a spacer they become
    propagating and really take power, so `signal = sqrt(eta_+ * R_0 * eta_-)` penalises them
    automatically. That is, L1/L2 are optimised to **steer power into the design channel m=-1**.
    In the 260827 measurement the design-channel share of the forward power in the spacer is only 28 % for TM;
    the rest is trapped parasitic power, which causes the breakdown of L3's local period
    and the resonances.
    """
    global OUT_SPACER_UM, OUT_SPACER_EPS
    OUT_SPACER_UM = max(0.0, float(nm)) * 1e-3
    OUT_SPACER_EPS = None if OUT_SPACER_UM <= 0 else complex(float(n)) ** 2
    return OUT_SPACER_UM * 1e3


def set_out_l3(nm, n=1.575):
    """[260901] **L3 metalens layer -- approximated as a uniform slab.**

    A further uniform layer put on top of `set_out_spacer`, closer to the air than the spacer. L3 is in
    reality a 2D metaatom array with a local period, but that period (657.695 nm) is
    sufficiently smaller than the wavelength in the spacer (lam/n_spacer = 1073.4 nm), so it **does not
    diffract the design channel** -- so it can be approximated by a uniform slab having only
    the local effective refractive index.

    The default `n=1.575` is the average effective refractive index measured in the 8-level library.

    Put the **physical H3** (e.g. 825 nm) into `OUT_L3_UM` as is -- it is not a thickness refitted to match the
    optical equivalence of a uniform slab.
    """
    global OUT_L3_UM, OUT_L3_EPS
    OUT_L3_UM = max(0.0, float(nm)) * 1e-3
    OUT_L3_EPS = None if OUT_L3_UM <= 0 else complex(float(n)) ** 2
    return OUT_L3_UM * 1e3


R_PHASE_REF = 2.711903852   # rad; grcwa - Airy, measured 260914 AT 1.550 um
R_PHASE_REF_LAM = 1.550     # wavelength at which that constant was measured


def phase_ref(lam_um=None):
    """[260914] **Wavelength dependence** of `R_PHASE_REF`. Left as a constant it is wrong at the band edges.

    This offset is a round-trip phase that arises because grcwa reads `GetAmplitudes(0, 0.0)` at the
    top of an incident half-space of thickness 0.5 um, and analytically it is
    `2*k0*n_sub*500nm - pi` -- at 1550 it gives 2.711890 rad, matching the measured constant
    2.711903852 to **1.4e-5 rad**, so the measurement confirms the physics of the slope.
    Over 1546~1554 the true offset moves by **1.731 deg**.

    The measured value is taken as the reference point and **only the difference** is added analytically --
    so as not to replace a measurement by the model. Hence at 1.550 um it is bit-identical to the old constant.
    """
    lam = (LAM if lam_um is None else float(lam_um)) * 1e3
    l0 = R_PHASE_REF_LAM * 1e3
    return R_PHASE_REF + 2.0 * (2.0 * np.pi) * N_SUB * 500.0 * (1.0 / lam - 1.0 / l0)


def r_of_slab(n3, t3_nm, lam_um=None, n_sp=None, n_out=1.0):
    """[260914] Complex reflection that the termination slab **presents to the design channel** (Airy, 3 media).

    Spacer -> slab (n3, t3) -> air, normal incidence. The reason normal incidence is exact rather than an
    approximation: because of Littrow, `kx/k0` of the TM-lane design channel is
    exactly 0 as `n_sub*sin(54.69) - lam/Lambda_x = 2.22e-16`, and the TE lane is 0 from
    the start (260914 measurement). The parasitic order (kx/k0 = 1.1784) does not enter here.

    260914 verification: compared with the stored `hole` minimum-diameter 20 points (in effect a solid a-Si slab,
    0.65 % void), the median |r| error is 0.016, and the phase is a constant reference-plane offset.
    """
    lam = (LAM if lam_um is None else float(lam_um)) * 1e3
    nsp = float(N_SUB if n_sp is None else n_sp)
    r1 = (nsp - n3) / (nsp + n3)
    r2 = (n3 - n_out) / (n3 + n_out)
    e2 = np.exp(2j * 2.0 * np.pi * n3 * t3_nm / lam)
    # [260914] The reference-plane offset depends on wavelength (`phase_ref`). At 1.550 um it is
    # bit-identical to the old constant, so no existing result moves at all.
    # [260914] **Shift to the grcwa reference plane.** Airy refers to the slab front face, while grcwa
    # reads `GetAmplitudes(0, 0.0)`, i.e. at the top of an incident half-space of thickness 0.5 um,
    # so that round-trip phase is added. The library `argR_deg` uses the grcwa reference, so to
    # impose a target the same reference is required -- without this term "unreachable" appears at all five
    # thicknesses (260914 measurement, residual 0.072~0.270). The value is **measured**, not derived, and
    # was confirmed constant to within 1e-6 deg over 6 slabs.
    return ((r1 + r2 * e2) / (1.0 + r1 * r2 * e2)) \
        * np.exp(1j * phase_ref(lam_um))


def solve_slab_for_r(r_target, t3_nm, lam_um=None, n_lo=1.02, n_hi=3.48,
                     coarse=0.002):
    """[260914] Real `n3` closest to the target complex reflection **at a fixed thickness**.

    Returns `(n3, residual)`. residual is `|r(n3, t3) - r_target|`.

    Why the thickness is fixed: the existing multi-termination fold shares one `OUT_L3_NS_NM`,
    and `fom_extract` / `fom_split` / `report` depend on that convention. Making the thickness differ
    per termination would require changing all three, so here **the job gives the thickness and only the
    refractive index is solved**. A large residual means that target is unreachable at that thickness, and the
    caller reports it loudly -- better than silently ending with a different termination.
    """
    grid = np.arange(n_lo, n_hi + 1e-9, coarse)
    err = np.abs(r_of_slab(grid, float(t3_nm), lam_um) - complex(r_target))
    j = int(np.argmin(err))
    lo = max(n_lo, grid[j] - coarse)
    hi = min(n_hi, grid[j] + coarse)
    fine = np.linspace(lo, hi, 401)
    e2 = np.abs(r_of_slab(fine, float(t3_nm), lam_um) - complex(r_target))
    k = int(np.argmin(e2))
    return float(fine[k]), float(e2[k])


def set_out_l3b(nm, n=0.0):
    """[260914] **Inner layer of a 2-layer termination.** Sits between the spacer and the L3 slab.

    With `nm <= 0` or `n <= 0` it **removes the layer** -- after that `_build` and
    `spacer_purity` take literally the same path as when this function did not exist
    (same release convention as `set_out_l3_list`). A job that does not use the new key must
    be bit-identical to the existing results, so this condition comes before any feature.

    Why 2 layers: a 1-layer slab shares its thickness among terminations (`OUT_L3_NS_NM`), so it has
    **one degree of freedom, the refractive index**, and can only draw one curve in the complex r plane.
    That curve, passing through the library cluster (|r| ~ 0.1, arg ~ +155 deg), passes only
    **below** the parasitic-order threshold `n = 1.178358` -- in a slab below the threshold the two trapped orders are
    evanescent but propagate in the real a-Si atom, so the purity channel is modelled backwards.
    Splitting the thickness as `t_b + t_3 = H3` gives three degrees of freedom and **both refractive indices**
    can be placed above the threshold. The total thickness is pinned to H3 for a separate reason -- the
    termination **front face** must stay where it is, so that `R_PHASE_REF` (grcwa reference-plane offset,
    measured at that front face) keeps applying.

    260914 measurement: with `t_b = 0` it matches `r_of_slab` to 1.7e-16, and with `n_b == n_3` a single
    slab of the combined thickness to 1.9e-15 (two independent reductions).
    """
    global OUT_L3B_UM, OUT_L3B_EPS
    _n = float(n)
    OUT_L3B_UM = max(0.0, float(nm)) * 1e-3
    if OUT_L3B_UM <= 0 or _n <= 0:
        OUT_L3B_UM, OUT_L3B_EPS = 0.0, None
    else:
        OUT_L3B_EPS = complex(_n) ** 2
    return OUT_L3B_UM * 1e3


def set_out_l3b_list(pairs):
    """[260914] Inner layer `(n_b, t_b[nm])` for each termination.

    Must have the **same length and order** as `OUT_L3_NS` -- the fold loop pairs them by index.
    If empty, the 1-layer fold is unchanged.
    """
    global OUT_L3B_LIST
    if not pairs:
        OUT_L3B_LIST = ()
        return OUT_L3B_LIST
    OUT_L3B_LIST = tuple((float(a), float(b)) for a, b in pairs)
    if OUT_L3_NS and len(OUT_L3B_LIST) != len(OUT_L3_NS):
        raise ValueError(
            "set_out_l3b_list: %d inner layers differ from %d terminations -- "
            "the fold loop pairs by index, so a silent mismatch must not happen"
            % (len(OUT_L3B_LIST), len(OUT_L3_NS)))
    return OUT_L3B_LIST


def _set_termination(i, n3):
    """[260914] Sets up the termination for one point of the fold loop.

    For 1 layer it is the single line `set_out_l3(OUT_L3_NS_NM, n3)` -- exactly as before the fold loop called
    this function. For 2 layers the inner layer is set first and the outer slab thickness is reduced to
    `OUT_L3_NS_NM - t_b` so that the **total thickness stays at H3**. If the total thickness varies,
    the termination front face moves and `R_PHASE_REF` is no longer that phase reference,
    so a different r is solved than the target r that was set.
    """
    if OUT_L3B_LIST:
        _nb, _tb = OUT_L3B_LIST[i]
        set_out_l3b(_tb, _nb)
        set_out_l3(OUT_L3_NS_NM - _tb, n3)
    else:
        set_out_l3(OUT_L3_NS_NM, n3)


def r_of_2layer(n_b, t_b_nm, n_3, t_3_nm, lam_um=None, n_sp=None, n_out=1.0):
    """[260914] Complex reflection of `spacer - inner layer (n_b,t_b) - slab (n_3,t_3) - air`.

    Recursive Airy from the air side inward. It carries the **same** reference-plane offset as `r_of_slab`,
    so it can be compared directly with the library `argR_deg` (as long as the total thickness is
    pinned to H3 the termination front face is at the same z).
    """
    lam = (LAM if lam_um is None else float(lam_um)) * 1e3
    nsp = float(N_SUB if n_sp is None else n_sp)
    nb = np.asarray(n_b, dtype=complex)
    n3 = np.asarray(n_3, dtype=complex)
    r23 = (n3 - n_out) / (n3 + n_out)
    r12 = (nb - n3) / (nb + n3)
    r01 = (nsp - nb) / (nsp + nb)
    e3 = np.exp(2j * 2.0 * np.pi * n3 * t_3_nm / lam)
    R1 = (r12 + r23 * e3) / (1.0 + r12 * r23 * e3)
    eb = np.exp(2j * 2.0 * np.pi * nb * t_b_nm / lam)
    R0 = (r01 + R1 * eb) / (1.0 + r01 * R1 * eb)
    return R0 * np.exp(1j * phase_ref(lam_um))


def solve_2layer_for_r(r_target, h3_nm, lam_um=None, n_min=1.178358, n_max=3.48,
                       coarse_n=0.01, coarse_t=5.0, refine=3, lams_um=None):
    """[260914] `(n_b, t_b, n_3)` matching the target complex reflection while keeping the total thickness `h3_nm`.

    Returns `(n_b, t_b, n_3, residual)`. Restricting **both refractive indices to `n_min` or above**
    is the reason this function exists -- the default 1.178358 is the parasitic-order propagation threshold, and a
    termination below it makes the trapped orders evanescent and models purity backwards.

    Grid search followed by local refinement. A large residual means that target is unreachable **above the
    threshold** at this `h3_nm`, and the caller must report it loudly (same convention as `solve_slab_for_r` --
    better than silently ending with a different termination).

    [260914] Given `lams_um`, it minimises the **worst residual over all those wavelengths**.
    Why: the termination is solved at one wavelength, but the objective folds three wavelengths. Measured -- the
    stage-1 termination currently used (n_b 3.1942 / t_b 460.5 / n_3 2.1810) moves by
    **0.0283** over 1546~1554. This is **10.2 times** the median shift (0.0028) of the 966 library atoms
    that have all three wavelengths at H3 875, and half the palette radius (0.0570).
    So stage 1 was buying a band that shakes ten times more than the thing it stands in for, and
    it did not show anywhere because it matched to 3.42e-06 at 1550.
    Solving over the band gives a worst residual of 0.0080 (n_min 1.178) / 0.0087 (n_min 1.35),
    at least 3.2 times better. The 1.178 solution lands **exactly on the parasitic threshold**, so the
    option with margin (1.35, cost 9 %) is recommended. The search is a grid plus local refinement on a
    multi-extremum landscape, so these residuals are **upper bounds** (n_min 1.25 coming out worse than 1.35 is the evidence).

    With `lams_um = None` it is exactly identical to the existing call.
    """
    rt = complex(r_target)
    _ls = (None,) if not lams_um else tuple(float(v) for v in lams_um)

    def _err(NB, t, N3, t3):
        e = None
        for _l in _ls:
            v = np.abs(r_of_2layer(NB, t, N3, t3, _l if _l else lam_um) - rt)
            e = v if e is None else np.maximum(e, v)
        return e
    nb = np.arange(n_min, n_max, coarse_n)
    n3 = np.arange(n_min, n_max, coarse_n)
    best = None
    for t in np.arange(0.0, float(h3_nm) + 1e-9, coarse_t):
        NB, N3 = np.meshgrid(nb, n3, indexing="ij")
        e = _err(NB, t, N3, float(h3_nm) - t)
        k = int(np.argmin(e))
        i, j = k // e.shape[1], k % e.shape[1]
        if best is None or e[i, j] < best[3]:
            best = (float(nb[i]), float(t), float(n3[j]), float(e[i, j]))
    cn, ct = coarse_n, coarse_t
    for _ in range(int(refine)):
        b_nb, b_t, b_n3 = best[0], best[1], best[2]
        nbf = np.linspace(max(n_min, b_nb - cn), min(n_max, b_nb + cn), 41)
        n3f = np.linspace(max(n_min, b_n3 - cn), min(n_max, b_n3 + cn), 41)
        for t in np.linspace(max(0.0, b_t - ct), min(float(h3_nm), b_t + ct), 41):
            NB, N3 = np.meshgrid(nbf, n3f, indexing="ij")
            e = _err(NB, t, N3, float(h3_nm) - t)
            k = int(np.argmin(e))
            i, j = k // e.shape[1], k % e.shape[1]
            if e[i, j] < best[3]:
                best = (float(nbf[i]), float(t), float(n3f[j]), float(e[i, j]))
        cn *= 0.25
        ct *= 0.25
    return best


def _rho_trapped(n_b, t_b_nm, n3, t3_nm, lam_um=None, n_sp=None, nG=21,
                 app=3, pol="TM"):
    """[260923] `rho` of the TRAPPED +-THETA_SUB parasitic channel for a
    UNIFORM 2-layer termination, seen from the SPACER side, referenced at
    the L3 FACE (the spacer/termination interface) -- same convention
    `spacer_purity`/`spacer_fid` already use (`GetAmplitudes(_sp_idx,
    OUT_SPACER_UM)`, this file :887/:904, read at the END of the incident
    spacer layer, not its start). `|rho| = 1` by unitarity (needle -- the
    channel can neither diffract at Lambda/3 nor escape past the critical
    angle into air); only `arg rho` is physical.

    [260923 review fix] The first version of this function read
    `GetAmplitudes(0, 0.0)` -- the PATCH at the far (top) edge of the
    0.5 um incident half-space, 0.5 um IN FRONT of the L3 face -- which
    adds a spurious round-trip phase of `2*kz*0.5um` (193.85 deg at TM,
    theta=54.69deg, n=1.444, 1550nm) on top of the physical `arg rho`.
    Reading at `dz = T0` (the end of the incident layer, i.e. the L3 face)
    removes it -- same fix, same reasoning, as Track 1's `_measure_rho`
    (`Main_rcwa_l3_atom2d_opt.py`, `measure_rho_face.diff`); this file's
    `argrho_termination_face.diff` is its Track-2 twin.

    [260923 review fix] Three more corrections, all found by the same
    review:
      * `pol` is now a real parameter, threaded from the driver's lane
        (`extract` -> TM, `split` -> TE) instead of hard-coded TM. TE uses
        `MakeExcitationPlanewave(0,0,1,0)` with amplitude offset 0; TM uses
        `(1,0,0,0)` with offset `ob.nG` -- same convention as `_solve`
        elsewhere in this file.
      * The spacer index `n_sp` now defaults to the job's ACTUAL output
        spacer (`sqrt(real(OUT_SPACER_EPS))`, i.e. `n_out_spacer`) when one
        is set via `set_out_spacer`, falling back to `N_SUB` only when no
        out-spacer is defined -- so a caller that never sets an out-spacer
        keeps the old default and old numbers.
      * The incidence angle is now derived from the SAME `lam`/`n_sp` used
        by this call (`theta = arcsin((lam/PERIOD)/n_sp)`), not the fixed
        module constant `THETA_SUB`. `THETA_SUB` is the SUBSTRATE-side
        Littrow angle (`arcsin(LAM/(PERIOD*N_SUB))`) and only equals the
        angle the trapped order actually travels at in THIS spacer when
        `n_sp == N_SUB` and `lam == LAM` -- true for every job so far
        (default `n_out_spacer = 1.4440 == N_SUB`), which is why the bug
        was silent, but not true in general (e.g. the `n_out_spacer=1.08`
        low-index carrier on record elsewhere in this project). `np.arcsin`
        of an out-of-domain ratio (spacer below the propagation floor)
        returns `nan` rather than raising, so a job with too low an
        `n_sp` degrades to `rho = nan` instead of a wrong angle.

    Needle-verified 260923 (Track 2 Step A): for a UNIFORM stack this is
    **nG-independent** to machine precision (nG=21/81/201 bit-identical,
    0.002 s / 34 s / 116 s) -- unlike the atom's own Add_LayerGrid probe,
    where TM + 2D grid + TIR drifts with nG. `nG=21` is used everywhere
    below for that reason; it is not a truncation, it is the measurement.

    [260923] **Forces the `numpy` grcwa backend for this call, then
    restores whatever backend was active.** `scd_c_grating.py` sets
    `grcwa.set_backend("autograd")` at import (module-global, monkeypatched
    class) for the optimizer's `value_and_grad`. No gradient is needed here
    -- this is a closed-form forward evaluation used only to MEASURE
    `arg rho` -- but `grcwa`'s `autograd` backend routes `eig`/`inv` through
    autograd's custom VJP primitives regardless, which is far slower per
    call than plain `numpy.linalg.eig`. Measured 260923: a
    `solve_2layer_for_r_rho` family sweep (~800 accepted `_rho_trapped`
    calls) that took ~1 s total under `numpy` ran for **>8 minutes without
    finishing** under the driver's inherited `autograd` backend (killed,
    not a hang -- CPU time was climbing) before this fix.
    """
    lam = LAM if lam_um is None else float(lam_um)
    if n_sp is not None:
        nsp = float(n_sp)
    elif OUT_SPACER_UM > 0 and OUT_SPACER_EPS is not None:
        nsp = float(np.sqrt(np.real(OUT_SPACER_EPS)))
    else:
        nsp = N_SUB
    ly = PERIOD / float(app)
    th = float(np.arcsin((lam / PERIOD) / nsp))
    _prev = grcwa.backend.__class__.__name__
    grcwa.set_backend("numpy")
    try:
        ob = grcwa.obj(int(nG), [PERIOD, 0.0], [0.0, ly], 1.0 / lam,
                       th, 0.0, verbose=0)
        T0 = 0.5
        ob.Add_LayerUniform(T0, complex(nsp) ** 2)      # spacer (incident)
        if t_b_nm > 0:
            ob.Add_LayerUniform(t_b_nm * 1e-3, complex(n_b) ** 2)
        ob.Add_LayerUniform(t3_nm * 1e-3, complex(n3) ** 2)
        ob.Add_LayerUniform(0.5, EPS_AIR)                 # air (output)
        ob.Init_Setup()
        idx = None
        for i, g in enumerate(ob.G):
            if int(g[0]) == 0 and int(g[1]) == 0:
                idx = i
                break
        _p = (pol or "TM").upper()
        if _p == "TE":
            ob.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
            off = 0
        else:
            ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
            off = ob.nG
        a, b = ob.GetAmplitudes(0, T0)
        rho = complex(b[idx + off]) / (complex(a[idx + off]) + 1e-30)
    finally:
        grcwa.set_backend("numpy" if _prev == "NumpyBackend" else "autograd")
    return abs(rho), float(np.degrees(np.angle(rho)))


def _wrap180(deg):
    return ((deg + 180.0) % 360.0) - 180.0


def solve_2layer_for_r_rho(r_target, argrho_target_deg, h3_nm, lam_um=None,
                           n_min=1.178358, n_max=3.48, coarse_n=0.01,
                           coarse_t=1.0, refine=3, lams_um=None, rho_nG=21,
                           pol="TM"):
    """[260923] Like `solve_2layer_for_r`, but ALSO targets `arg rho` of the
    trapped +-THETA_SUB channel (Track 2, "DEMAND"), on the L3-FACE
    reference and the caller's LANE (`pol`) -- see `_rho_trapped` for both
    fixes. `pol` and the spacer index are threaded straight through to
    every `_rho_trapped` call below (family sweep AND local refine).

    Why a family exists: total thickness `h3_nm` is fixed and `r_target`
    (design channel, normal incidence) is 2 real numbers, leaving 1 degree
    of freedom among the 3 unknowns (n_b, t_b, n_3). `arg rho` (trapped
    channel, THETA_SUB) is generally FREE along that family, but NOT the
    full circle -- reachable arcs measured on the FIXED (face-reference)
    code, H3=875 nm, on the solver's own 1 nm `t_b` grid (largest
    UNREACHABLE gap, and bins filled of 24; `|rho|=1` unitarity holds
    everywhere sampled, max `||rho|-1|` 1.1e-13):
        r00=0.16@172deg  reachable span 234.24 deg (gap 125.76 deg,
                          152.35 to 278.10)   17/24 bins filled
        r00=0.20@170deg  reachable span 225.06 deg (gap 134.94 deg,
                          166.93 to 301.87)   16/24 bins filled
        r00=0.03@172deg  reachable span 342.71 deg (gap  17.29 deg,
                          244.94 to 262.23)   24/24 bins filled
    The reachable span SHRINKS as `|r00|` grows. The earlier claim here --
    "353.922 deg, essentially the whole circle" -- was measured on the
    UNFIXED (patch-reference, hard-coded-TM, fixed-`THETA_SUB`) code and is
    superseded by the table above; full per-point data in
    `argrho_termination_face.diff`'s accompanying `report.md`.

    Method: sample the family densely on a `t_b` grid (same coarse+refine
    (n_b, n3) solve as `solve_2layer_for_r`, fixed `t_b` per sample -- i.e.
    the same numerics `solve_2layer_for_r` uses internally, just not letting
    `t_b` float), measure `arg rho` at each accepted sample (`_rho_trapped`,
    nG=21, ~2 ms/point -- confirmed nG-independent so this is cheap and
    exact, not a truncation), then keep the single sample closest to
    `argrho_target_deg` (circular distance) and locally re-scan a narrow
    window around it at 1/4 the step for a final polish.

    [260923] **`n_min` is a genuine wall, not a search failure.** At the
    default floor 1.178358 (the parasitic-propagation threshold -- see
    `solve_2layer_for_r`), the reachable arc can be narrower than the full
    circle by pinning `n_3` (or `n_b`) exactly AT `n_min` -- do not read a
    large `argrho_residual_deg` here as a bug; check whether `n_b`/`n_3`
    sit at `n_min` in the returned solution first.

    Returns `(n_b, t_b, n_3, r_residual, argrho_achieved_deg,
    argrho_residual_deg)`. `argrho_residual_deg` is the circular distance to
    `argrho_target_deg`; the caller decides what counts as "reached".
    `(None,)*6` if no family point meets `r_residual <= 0.005` at valid n.
    """
    rt = complex(r_target)
    _ls = (None,) if not lams_um else tuple(float(v) for v in lams_um)

    def _err(NB, t, N3, t3):
        e = None
        for _l in _ls:
            v = np.abs(r_of_2layer(NB, t, N3, t3, _l if _l else lam_um) - rt)
            e = v if e is None else np.maximum(e, v)
        return e

    def _solve_at_t(t, nb0, n30):
        t3 = float(h3_nm) - t
        NB, N3 = np.meshgrid(nb0, n30, indexing="ij")
        e = _err(NB, t, N3, t3)
        k = int(np.argmin(e))
        i, j = k // e.shape[1], k % e.shape[1]
        nb, n3, be = float(nb0[i]), float(n30[j]), float(e[i, j])
        cn = coarse_n
        for _ in range(refine):
            nbf = np.linspace(max(n_min, nb - cn), min(n_max, nb + cn), 41)
            n3f = np.linspace(max(n_min, n3 - cn), min(n_max, n3 + cn), 41)
            NB, N3 = np.meshgrid(nbf, n3f, indexing="ij")
            e2 = _err(NB, t, N3, t3)
            k = int(np.argmin(e2))
            i, j = k // e2.shape[1], k % e2.shape[1]
            if e2[i, j] < be:
                nb, n3, be = float(nbf[i]), float(n3f[j]), float(e2[i, j])
            cn *= 0.25
        return nb, n3, be, t3

    nb0 = np.arange(n_min, n_max, coarse_n)
    n30 = np.arange(n_min, n_max, coarse_n)

    fam = []
    for t in np.arange(0.0, float(h3_nm) + 1e-9, coarse_t):
        nb, n3, be, t3 = _solve_at_t(t, nb0, n30)
        if be <= 0.005 and nb >= n_min - 1e-9 and n3 >= n_min - 1e-9:
            _absr, _argr = _rho_trapped(nb, t, n3, t3, lam_um, nG=rho_nG,
                                        pol=pol)
            fam.append((t, nb, n3, be, _argr))
    if not fam:
        return None, None, None, None, None, None

    dists = [abs(_wrap180(f[4] - argrho_target_deg)) for f in fam]
    k0 = int(np.argmin(dists))
    best = fam[k0] + (dists[k0],)          # (t, nb, n3, be, argr, dist)

    lo = max(0.0, best[0] - coarse_t)
    hi = min(float(h3_nm), best[0] + coarse_t)
    for _ in range(refine):
        for t in np.linspace(lo, hi, 21):
            nb, n3, be, t3 = _solve_at_t(t, nb0, n30)
            if be > 0.005 or nb < n_min - 1e-9 or n3 < n_min - 1e-9:
                continue
            _absr, _argr = _rho_trapped(nb, t, n3, t3, lam_um, nG=rho_nG,
                                        pol=pol)
            d = abs(_wrap180(_argr - argrho_target_deg))
            if d < best[5]:
                best = (t, nb, n3, be, _argr, d)
        span = (hi - lo) * 0.3
        lo, hi = max(0.0, best[0] - span), min(float(h3_nm), best[0] + span)

    return best[1], best[0], best[2], best[3], best[4], best[5]


def set_out_l3_list(nm, ns):
    """Evaluates the L3 equivalent layer at several refractive indices at once (worst-case optimisation).

    [260901 new] `set_out_l3` fixes the L3 uniform slab with a single n, but the real L3 library
    spans n = 1.071~1.967, and within it the response is **non-monotonic** (because it is a
    Fabry-Perot of an H3=825 nm slab) -- in the measurements of the two champions, changing
    n moves signal by up to 1.4 dB and echo by up to 8.3 dB. The gate is defined at the **worst**
    condition of that range, so the objective must also sweep the same range with several points and
    take the worst value -- so that objective and gate measure the same thing
    (see the [260901] section of `fom_extract` / `fom_split`).

    `nm` is the thickness shared by all terminations -- L3 is already fixed to a single physical
    thickness (H3), and within the library only the local effective refractive index differs per level.
    `ns` is the list of those indices.

    If `ns` is empty or None it is released -- after that `fom_extract` / `fom_split`
    return to exactly the same single-termination path as before this function was called
    (no extra solve, no numerical difference).
    """
    global OUT_L3_NS, OUT_L3_NS_NM
    ns = tuple(float(x) for x in ns) if ns else ()
    OUT_L3_NS = ns
    OUT_L3_NS_NM = float(nm) if ns else 0.0
    return OUT_L3_NS, OUT_L3_NS_NM


def _build(H1, H2, eps_in, eps_out, first_layer, theta, lam, ng, eps_gap=None):
    """first_layer = 'L1' means layer1 is met first on the incident side.

    If GAP_UM > 0 a uniform spacer is inserted between the two grating layers (see set_gap).
    The physical stack is 'substrate - L1 - spacer - L2 - air' regardless of the illumination direction,
    so even if the incidence order is reversed the spacer always goes **between** the two grating layers.

    [260827] If `OUT_SPACER_UM > 0` one more uniform spacer is put on the **output side** (behind the
    grating, before eps_out) -- the place for the L3 metaatom (set_out_spacer).
    If the incidence direction is 'top' (air -> substrate), that place is optically behind the grating,
    not in front, so it is always attached on the L2 side by the physical stack.

    [260901] If `OUT_L3_UM > 0` the L3 metalens uniform slab (set_out_l3) is put one more layer
    toward the air than that spacer -- the physical stack is '... - L2 - spacer
    - L3 - air'. If incidence is flipped to 'top', illumination starts on the air side, so in
    incidence order L3 is added **before** the spacer (closer to the incident side).
    """
    ob = grcwa.obj(NG if ng is None else int(ng), L1, L2, 1.0 / lam,
                   theta, 0.0, verbose=0)
    if RCWA_RULE != "laurent" or RCWA_FF:
        # [261008 li_integrate] opt-in only (see set_rcwa)
        ob.rcwa_rule = RCWA_RULE
        ob.form_factor = RCWA_FF
    ob.Add_LayerUniform(0.5, eps_in)
    if first_layer == "L1":
        ob.Add_LayerGrid(H1, *_grid())
        if GAP_UM > 0:
            ob.Add_LayerUniform(GAP_UM, EPS_AIR if eps_gap is None else eps_gap)
        ob.Add_LayerGrid(H2, *_grid())
        if OUT_SPACER_UM > 0:
            ob.Add_LayerUniform(OUT_SPACER_UM, OUT_SPACER_EPS)
        # [260914] Inner layer of a 2-layer termination (set_out_l3b). When off, no layer is attached and
        # the stack is literally the same as before -- the layer offset of `spacer_purity` is counted by the
        # **same expression** as this condition, so the two cannot get out of step.
        if OUT_L3B_UM > 0:
            ob.Add_LayerUniform(OUT_L3B_UM, OUT_L3B_EPS)
        if OUT_L3_UM > 0:
            ob.Add_LayerUniform(OUT_L3_UM, OUT_L3_EPS)
    else:
        # 'top' illumination: air -> L2 -> L1 -> substrate. The spacer is **outside** L2, so it
        # comes on the incident side.
        if OUT_L3_UM > 0:
            ob.Add_LayerUniform(OUT_L3_UM, OUT_L3_EPS)
        if OUT_L3B_UM > 0:
            ob.Add_LayerUniform(OUT_L3B_UM, OUT_L3B_EPS)
        if OUT_SPACER_UM > 0:
            ob.Add_LayerUniform(OUT_SPACER_UM, OUT_SPACER_EPS)
        ob.Add_LayerGrid(H2, *_grid())
        if GAP_UM > 0:
            ob.Add_LayerUniform(GAP_UM, EPS_AIR if eps_gap is None else eps_gap)
        ob.Add_LayerGrid(H1, *_grid())
    ob.Add_LayerUniform(0.5, eps_out)
    ob.Init_Setup()
    if GRID_2D is not None:
        # [260924] alias ceiling per axis (the 1-D rule, see NG above): the
        # convolution reads order DIFFERENCES, so max|m| <= n/4 on each axis.
        _G = np.abs(np.asarray(ob.G))
        if _G[:, 0].max() > GRID_2D[0] // 4 or _G[:, 1].max() > GRID_2D[1] // 4:
            raise ValueError("2-D alias: nG %d keeps |mx| %d / |my| %d > %d / %d"
                             % (ob.nG, _G[:, 0].max(), _G[:, 1].max(),
                                GRID_2D[0] // 4, GRID_2D[1] // 4))
    return ob


def _grid():
    """(nx, ny) for Add_LayerGrid -- (NX, 1) on the 1-D path."""
    return (NX, 1) if GRID_2D is None else GRID_2D


def _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2, side, pol, theta,
           lam=None, ng=None):
    """side = 'top'  : air -> (L2, L1) -> substrate   (input grating geometry)
       side = 'sub'  : substrate -> (L1, L2) -> air   (extraction/return geometry)
    Returns (R_by_order, T_by_order) -- both dicts, in relative orders.
    """
    lam = LAM if lam is None else float(lam)
    e1 = eps_f1 + rho1 * (eps_b[0] - eps_f1)
    e2 = eps_f2 + rho2 * (eps_b[1] - eps_f2)
    # The spacer material is fill1 -- the deposited film that filled L1 rising up as it is.
    if side == "top":
        ob = _build(H1, H2, EPS_AIR, EPS_SUB, "L2", theta, lam, ng, eps_f1)
        ob.GridLayer_geteps(npa.concatenate([e2, e1]))
    else:
        ob = _build(H1, H2, EPS_SUB, EPS_AIR, "L1", theta, lam, ng, eps_f1)
        ob.GridLayer_geteps(npa.concatenate([e1, e2]))
    if pol == "TE":
        ob.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
    else:
        ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    Ri, Ti = ob.RT_Solve(normalize=1, byorder=1)
    return ob, Ri, Ti


def _pick(ob, arr, ms):
    return {m: arr[T1._order_index(ob, m)] for m in ms}


def _args(ob, pol, ms, amp_floor=1e-6, with_abs=False):
    """**Phase** (deg) of the complex amplitude per order. Both reflection (b0) and transmission (aN).

    [260816] New for the phase library. Running several seeds at the same (material, H1, H2)
    gives solutions of similar efficiency but different phase difference, and to pick the solution that
    cancels the bias phase accumulated in the system, that phase must be known.

    Implementation notes
    ---------
    * The amplitude vector of `GetAmplitudes` has length 2*nG and the **first half is the x component,
      the second half the y component**. In a 1D classical mount only one is alive per polarization --
      measured: TM is y, TE is x, and the other is exactly 0.
    * |amplitude|^2 is not equal to power (measured ratio 2.4983 / 1.4440). That factor comes from kz/n
      and is a **real positive number**, so it does not affect the phase.
    * The reference plane is **fixed** at z_offset=0 of the incident-side uniform layer. The absolute phase
      depends on the reference plane, so attach no meaning to it; use **only differences between designs**.
    * Orders with amplitude near 0 (non-propagating/evanescent) have a meaningless phase, so None is returned.

    [260908] With `with_abs=True` the amplitude (|c|, no amp_floor gate) is returned together with the phase.
    For an evanescent order the power is zero by definition, so the amplitude is the only measure left in that
    place -- without it there is no way at all to measure what that order contains.
    The default (False) still returns just the two values (outR, outT) -- the existing callers in
    `scd_c_grating3.py` and `Main_rcwa_l3_hsweep.py` depend on this number of return values.
    """
    n = ob.nG
    _a0, b0 = ob.GetAmplitudes(0, 0.0)
    aN, _bN = ob.GetAmplitudes(ob.Layer_N - 1, 0.0)
    b0, aN = np.asarray(b0), np.asarray(aN)
    off = n if pol == "TM" else 0
    outR, outT = {}, {}
    absR, absT = {}, {}
    for m in ms:
        i = T1._order_index(ob, m)
        for arr, dst, dst_abs in ((b0, outR, absR), (aN, outT, absT)):
            c = arr[i + off]
            dst[m] = (float(np.degrees(np.angle(c)))
                      if abs(c) > amp_floor else None)
            dst_abs[m] = float(abs(c))
    if with_abs:
        return outR, outT, absR, absT
    return outR, outT




def _untraced(x):
    """[260918] Strips the autograd box and returns a plain Python value.

    To use the spacer thickness as a **design variable**, `sp` has to be an autograd box, and the
    structure-check `float(...)` raises `TypeError` on a box (measured 260918).
    The check itself is not differentiated, so only the value is stripped for comparison -- if it is not a box
    the input is returned as it is, so the existing path is **bit-identical**.
    """
    while hasattr(x, "_value"):
        x = x._value
    return x

def spacer_purity(ob, pol, design_m=-1, theta=None, shares=None):
    """[260827] **Fraction of power in the design channel at the plane where L3 sits.**

    Why it is needed -- `eta_+ = T(design_m)` is measured in **air**, and the only propagating order
    going out into air is `m = -1` (kx/k0: m=0 is 1.1784, m=-2 is
    -1.1784, both |kx| > 1 and so evanescent). So the parasitic orders m=0, m=-2 **do not pass through
    the FoM** -- even with the output spacer on. Inside the spacer (n=1.4440) those orders
    **propagate** at +-54.69 deg but cannot leave into air, so they are
    trapped between L1/L2 and L3.

    The trapped parasitics are L3's problem -- they cause the FP cavity (period lam/(2 n cos th) = 928.6 nm)
    resonance and the breakdown of the local period, and they move the illumination seen by the L3 atom
    away from a clean normal plane wave. 260827 measurement (c74, TM): the design-channel share of the
    forward power in the spacer is **only 28 %**, with m=0 at 38 % and m=-2 at 34 %.

    So to make it "safe to put a metalens on top" this ratio has to go **explicitly** into the
    objective. It reads the forward amplitude of the spacer layer (just below the air), sums only the
    propagating orders with z-Poynting weights, and returns the design-channel share.

    **[260827 two bug fixes -- both were wrong when first introduced]**

    (1) **The weight differs with polarization.** What `GetAmplitudes` gives is the H-field
        Fourier components (first half hx = TE, second half hy = TM), and the z-Poynting of a plane wave is

            TE(hx):  S_z = (1/2) * omega*mu0 * |hx|^2 / kz     -> **1/kz**
            TM(hy):  S_z = (1/2) * kz * |hy|^2 / (omega*eps)   -> **kz**

        The first code used `kz` for both polarizations. TM is right and **only TE was wrong.**
        Measured by comparing per order against grcwa `GetZPoyntingFlux(byorder=1)`:
            TM  ∝kz  max error 1.11e-16 / ∝1/kz 6.22e-04   -> kz is right
            TE  ∝kz  max error 1.34e-02 / ∝1/kz 1.73e-18   -> 1/kz is right
        The design channel always has kx~0 so kz is maximal (1.4440) and the parasitic is 0.8346,
        so the wrong weight **overestimates** the design channel -- i.e. the penalty is weakened.
        The true value of the recorded TE 0.6616 is **0.3951**, penalty 0.338 -> 0.605 (factor 1.79).

    (2) **The diffraction term used the module constant `LAM`, not the loop wavelength.** It was right only at
        1550 nm out of 5 wavelengths and the other 4 points were computed with a wrong kx (measured error
        at 1530 nm: 0.0152 for m=-1, 0.0304 for m=-2). Now the wavelength is recovered
        directly from `ob` -- no caller needs fixing and none can get out of step
        (confirmed that `2*pi/ob.omega` reproduces the lam that was put in to machine precision).

    Returns: 0~1 (1 means the illumination is a pure normal plane wave).
    """
    if OUT_SPACER_UM <= 0:
        # [260904] **Make the silent switch-off visible.** Without a spacer the purity is not
        # defined and 1.0 is returned, but `fom_extract`/`fom_split` call this function unconditionally
        # when `w_pure` is nonzero (:959, :1163).
        # So if a job omits `out_spacer_nm`, **the purity term is always 1.0 =
        # penalty 0** and `w_pure` becomes ineffective without any signal
        # (zero omissions currently in all 801 cases, but the structure remains).
        # The return value is not changed -- changing it would break reproduction of past jobs.
        if not _WARNED["purity"]:
            _WARNED["purity"] = True
            print("[SCDC] **WARNING: out_spacer_nm is missing, so the purity is always 1.0 "
                  "-- the w_pure term is ineffective. Put out_spacer_nm (e.g. 200) into the "
                  "job.**", flush=True)
        return 1.0                      # undefined without a spacer
    # [260901] If an L3 slab (OUT_L3_UM) exists, one more layer comes behind the spacer,
    # so the offset grows accordingly -- the fixed -2 would read the L3 layer instead of the
    # spacer after L3 was introduced.
    # [260914] If the 2-layer termination (set_out_l3b) is on, the layers behind the spacer increase by one more.
    # Counted by the **same condition expression** as `_build` -- if the two drift apart it silently
    # reads the wrong layer. The thickness check below is still the last safety net.
    _sp_idx = ob.Layer_N - (2 + (1 if OUT_L3_UM > 0 else 0)
                              + (1 if OUT_L3B_UM > 0 else 0)
                              + (1 if OUT_L3_GAP_UM > 0 else 0))
    # [260901] The offset above is **for the substrate-side illumination ('L1') geometry only**. With 'top'
    # illumination the spacer comes first so it is wrong -- the earlier fixed -2 was the same.
    # The current callers fom_extract / fom_split / report all use side='sub' so it does not trigger,
    # but stopping here beats silently reading the wrong layer.
    if abs(float(_untraced(ob.thickness_list[_sp_idx]))
           - float(_untraced(OUT_SPACER_UM))) > 1e-12:
        raise ValueError(
            "spacer_purity: layer %d has thickness %.6f um, which differs from the spacer %.6f um -- "
            "this may not be the substrate-side illumination geometry"
            % (_sp_idx, float(ob.thickness_list[_sp_idx]), OUT_SPACER_UM))
    a, _b = ob.GetAmplitudes(_sp_idx, 0.0)
    off = ob.nG if pol == "TM" else 0
    n_sp = npa.sqrt(npa.real(OUT_SPACER_EPS))
    # Incident transverse wavenumber. The TM lane is +THETA_SUB in the substrate, the TE lane (branch) is normal.
    th = THETA_SUB if theta is None else float(theta)
    kx0 = N_SUB * npa.sin(th)
    # Actual wavelength of this solve. grcwa uses omega = 2*pi*freq, freq = 1/lam.
    lam = 2.0 * npa.pi / npa.real(ob.omega)
    num = den = 0.0
    _acc = {} if shares is not None else None
    for i, g in enumerate(ob.G):
        if int(g[1]) != 0:
            continue
        m = int(g[0])
        # kx/k0 is obtained directly from the order (1D classical mount)
        kxn = kx0 + m * (lam / PERIOD)
        c2 = n_sp ** 2 - kxn ** 2
        if c2 <= 0:                     # evanescent in the spacer -- not illumination
            continue
        kz = npa.sqrt(c2)
        # z-Poynting weight per polarization. The overall scale cancels in the ratio, so only relative ratios matter.
        w = (kz / n_sp) if pol == "TM" else (n_sp / kz)
        p = npa.abs(a[i + off]) ** 2 * w
        den = den + p
        if m == design_m:
            num = num + p
        if shares is not None:
            _acc[m] = p
    # [260907] **Record the per-order breakdown additionally -- the return value is unchanged.**
    # `w_pure` uses this return value in the objective and 899 cells of the corpus sit on the
    # current definition, so the meaning must not change. The problem is that only the design channel's
    # share remains, and apart from one cell (c74 TM, 260827: design 28 % /
    # m=0 38 % / m=-2 34 %) nobody knows **which parasitic order dominates**. m=0 and m=-2 leave
    # inside the spacer at opposite-sign angles (+-54.69 deg), so which of them is
    # larger changes what has to be suppressed.
    #
    # Filled only when `shares` is given. This function is on the **autograd path**
    # (fom_extract:1016 / fom_split:1222), and during tracing `p` is a box so float()
    # fails -- so a conversion failure is silently skipped, and the objective value and
    # gradient do not change by a single character.
    if shares is not None:
        _tot = 0.0
        _num = {}
        for m, p in _acc.items():
            try:
                _num[m] = float(p)
            except (TypeError, ValueError):
                return num / (den + 1e-12)      # during tracing -- not recorded
            _tot += _num[m]
        if _tot > 0:
            for m, v in _num.items():
                shares[m] = v / _tot
    return num / (den + 1e-12)


def spacer_fid(ob, pol, design_m=-1, theta=None):
    """[260922] **Illumination fidelity at the L3 plane -- evanescent orders are not discarded.**

    `spacer_purity` discards evanescent orders wholesale with `if c2 <= 0: continue` and
    reads at the L2-side plane (`dz = 0`). When the spacer refractive index falls below the propagation threshold
    `kx/k0 = N_SUB * sin(THETA_SUB) = 1.1784`, both parasitics become
    evanescent, so **purity becomes 1.000 from the first iteration and the gradient of `w_pure` is
    exactly 0**. The optimisation can then grow that channel without any constraint.

    Measured (260922): the `n_out_spacer = 1.08` carrier passes all four gates
    (efficiency 86.3 %, purity 1.00000, echo -24.53 dB, linewidth 71.9 nm), but with a real
    atom on top the best `fid_l3` is **0.6814** -- much worse than **0.9836** for the n = 1.444 carrier --
    and `evan_frac_l3` is **30.4 times** larger (median 0.972 vs 0.032),
    which tracks fid cell by cell (0.684 -> 0.681, 1.229 -> 0.398).

    So this function reads at the **plane where the atom actually sits** (`dz = OUT_SPACER_UM`) and
    counts **all** orders with `g[1] == 0`. The definition is the same as `_fid_l3` of the atom driver,
    so the carrier and the atom finally optimise the same quantity.

    The z-Poynting weight is not used -- because it is undefined for evanescent orders, and
    this too is the same convention as `_fid_l3`.
    """
    if OUT_SPACER_UM <= 0:
        return 1.0                      # undefined without a spacer
    _sp_idx = ob.Layer_N - (2 + (1 if OUT_L3_UM > 0 else 0)
                              + (1 if OUT_L3B_UM > 0 else 0)
                              + (1 if OUT_L3_GAP_UM > 0 else 0))
    if abs(float(_untraced(ob.thickness_list[_sp_idx]))
           - float(_untraced(OUT_SPACER_UM))) > 1e-12:
        raise ValueError(
            "spacer_fid: layer %d has thickness %.6f um, which differs from the spacer %.6f um -- "
            "this may not be the substrate-side illumination geometry"
            % (_sp_idx, float(ob.thickness_list[_sp_idx]), OUT_SPACER_UM))
    a, _b = ob.GetAmplitudes(_sp_idx, OUT_SPACER_UM)
    off = ob.nG if pol == "TM" else 0
    num = den = 0.0
    for i, g in enumerate(ob.G):
        if int(g[1]) != 0:
            continue
        p = npa.abs(a[i + off]) ** 2
        den = den + p
        if int(g[0]) == design_m:
            num = num + p
    return num / (den + 1e-30)


def _tamp(ob, pol, m):
    """Transmission complex amplitude -- **autograd path**.

    [260820] Same convention as `_args` but without the `np.asarray` cast, so differentiation continues
    (`_args` is for report only, so the tape breaks there). This path is needed to put a
    phase target into the FoM.

    The amplitude vector has length 2*nG; the first half is the x and the second half the y component.
    In a 1D classical mount only one is alive per polarization.
    """
    aN, _bN = ob.GetAmplitudes(ob.Layer_N - 1, 0.0)
    return aN[T1._order_index(ob, m) + (ob.nG if pol == "TM" else 0)]


def _ramp(ob, pol, m):
    """**Reflection** complex amplitude -- autograd path (reflection counterpart of `_tamp`, added 260820).

    Same convention as `_tamp`. Returns `b0` (downward amplitude of the incident-side uniform layer)
    used by `_args` as it is without a cast, so the differentiation tape continues.
    """
    _a0, b0 = ob.GetAmplitudes(0, 0.0)
    return b0[T1._order_index(ob, m) + (ob.nG if pol == "TM" else 0)]


def _dphi(a, b):
    """Difference of two phases (deg, -180~180). None if either is None."""
    if a is None or b is None:
        return None
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(d)


def _gmean(vals):
    """Geometric mean -- kills resonant solutions that are good at only one wavelength."""
    out = 1.0
    for v in vals:
        out = out * npa.maximum(v, 1e-9)
    return out ** (1.0 / len(vals))


def _wmin(vals):
    """Band worst value (fixed 2026-08-14: all domain FoMs unified on the worst value).

    The geometric mean lets values in the band compensate each other -- if one wavelength collapses but the others
    are high the score holds, so for a broadband objective it picks the wrong solution. A measured
    case is index 1505: 0.9353 at 1550 but halved to 0.4336 at 1530,
    and the 3-wavelength (+-0.8 nm) geometric mean did not penalise it at all.

    npa.minimum is used folded -- it is an autograd-wrapped function so differentiation passes through
    (a minimax subgradient where the gradient flows to the wavelength that attains the minimum).
    """
    out = vals[0]
    for v in vals[1:]:
        out = npa.minimum(out, v)
    return out


def _wmax(vals):
    """Counterpart of `_wmin` -- for metrics (echo etc.) whose band/condition **worst** is the maximum.

    [260901 new] Needed because the echo hinge (`_echo_cap_pen`) was changed to take the band worst (maximum)
    instead of the band mean. The reason is symmetric to `_wmin` -- folding npa.maximum
    gives a minimax subgradient where the gradient flows to the sample that attains the maximum.
    max is associative and order independent, so whether the list holds band values only or
    termination x wavelength values, it gives the overall maximum as it is.
    """
    out = vals[0]
    for v in vals[1:]:
        out = npa.maximum(out, v)
    return out


# ------------------------------------------------------------- objectives ---
# [260904] Flag to warn only once about a silently disabled setting. Both places
# are called every iteration, so printing every time would make the log meaningless.
_WARNED = {"cap": False, "purity": False}


def _echo_cap_pen(echo2s, w_cap, cap_db):
    """**Epsilon-constraint (hinge)** term on echo. At 0 it does nothing.

    [260831 new] Why a constraint and not a weight
    ---------------------------------------
    In the 33-cell measurement the echo-efficiency front split into **two clusters**:

        echo -4~-10 dB : 29 cells, best efficiency 72.7 %
        echo -10~-13   :  2 cells, best 52.7 %   (Pareto bound 98.1 %)
        echo -13~-17   :  **0 cells**            (bound 94.4 %)
        echo -17~-30   :  2 cells, best 27.0 %   (bound 88~92 %)

    Correlation coefficient r = +0.781.

    **[260831 19:38 correction, n=47] "non-convex, so nothing can stand in the middle" was an over-diagnosis.**
    The -13~-17 dB range that was empty with 33 cells was **too few samples**, and it was filled
    once thick stacks were reported (H600_425_e200, -14.76 dB / 40.5 %). The real shape is not a discontinuous
    jump but a **smooth but excessively steep decline** -- the achieved fraction of the Pareto bound
    collapses systematically, 59 % -> 43 % -> 29 % -> 25 %.

    **The hinge is still right -- only the reason differs.** A linear weight keeps pushing
    echo down even beyond the cap and sells efficiency without limit. The over-suppression above is exactly that symptom.
    The hinge **cuts the gradient to 0** once the target is met so that nothing more is sold.

    The hinge is different. While `echo <= cap` the **gradient is exactly 0**, so the optimiser
    raises only efficiency freely inside it -- i.e. it imitates "maximise under constraint"
    and can also enter the non-convex region.

    Why it is measured in dB: the gate is defined in dB, and measured in linear power the values near
    cap are too small to set the weight scale. The dB hinge is
    "how many dB the gate is exceeded by", so `w_cap ~ 0.01~0.05` gives the same size as
    signal (~0.28).

    [260901] **The gate is defined on the band/condition worst value -- the hinge must also
    take the worst value.** Previously `sum(echo2s)/n` (band mean) was fed to the hinge,
    an approximation that held only while the L1/L2-only band spread was a narrow 0.14 dB (mean = worst).
    Once the L3 equivalent slab (n=1.071~1.967, `set_out_l3_list`) is added,
    echo spreads by up to 8.3 dB between terminations, so the mean no longer
    stands for the gate -- the list passed in as `echo2s` is folded to its **maximum**
    (`_wmax`). In a single-termination call `echo2s` is the list of band values, and in a
    multi-termination call (the termination loop of `fom_extract`/`fom_split`) it is a
    list of termination x wavelength combined, so this one line covers both cases.
    The `n` argument is no longer used and was dropped from the signature. This correction changes the
    single-termination result (with echo_cap on) -- an intended change.
    """
    if not w_cap or cap_db >= 0.0:
        # [260904] **Make the silent switch-off visible.** The default of `echo_cap_db` is 0.0,
        # so if a job gives `w_echo_cap` but omits `echo_cap_db`, the hinge becomes ineffective
        # **without any signal**. Running with the gate-reference term off is
        # the worst silent failure (zero omissions currently in all 339 cases, but the structure remains).
        # **The default is not changed** -- changing it would break reproduction of past jobs.
        # Only one warning is printed (called every iteration, so limited to once).
        if w_cap and cap_db >= 0.0 and not _WARNED["cap"]:
            _WARNED["cap"] = True
            print("[SCDC] **WARNING: w_echo_cap = %g but echo_cap_db = %g "
                  "-- the echo hinge runs switched off. Put echo_cap_db (e.g. -20.5) "
                  "into the job.**" % (w_cap, cap_db), flush=True)
        return 0.0
    e = _wmax(echo2s)
    e_db = 10.0 * npa.log10(npa.maximum(e, 1e-12))
    return w_cap * npa.maximum(0.0, e_db - cap_db)


def fom_input(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
              w_te=1.0, w_echo=0.0, w_refl=0.0, w_back=0.0, ng=None, lams=None):
    """Input grating: air normal incidence -> substrate +1 order (TM). TE passes as order 0.

    w_refl penalises the reflection returning to air for **both polarizations**. On the air side
    lambda/Lambda = 1.178 > 1, so only m=0 propagates and the reflection is R_0 alone,
    which is exactly the light returning to the input fibre. For TM it is signal loss and back-reflection to the source;
    TE has to pass as order 0 to the central MS, and if it is reflected that share is lost.
    Compensating only T0_TE leaves reflected TE unpenalised, so this term is needed separately.

    w_echo suppresses the path (dm=+2) where 'the returned -1 order bounces back into the +1 order'.
    If that path is alive, the 10 ps second-order echo returns to the detector. The return grating
    must maximise exactly that transition, so the requirements are opposite, but the two gratings are 847 um
    apart and do not conflict.

    w_back [new 260819] suppresses the **fibre back-coupling path**. The R(-2) 10.4 % that the extraction
    grating could not kill rises at -theta inside the substrate and lands exactly at this grating position (x=0)
    (not a coincidence, given the folded layout with 2t*tan(theta) spacing); leaving here as relative order
    +1 gives k_x = -G + G = 0, i.e. **normal emission into air**. The only air-side propagating order is
    m=0, so no other direction is physically possible, and the fibre is attached normally at exactly that
    spot, so the mode overlap is worst.

    Measured (260819): input 0.9845 x R(-2) 0.1040 x T(+1) 0.9847 = **0.1008**, i.e. 10.1 % of the input
    power. That is an order of magnitude above the input-grating immediate reflection (R_both = 0.0114, 19.4 dB),
    and at 100 % overlap the return loss falls to 9.5 dB.

    Turning this term on sends that share into R instead of T and it stays in the substrate -- a trade that
    converts fibre backflow into scattering inside the slab. Which is better is decided by the system.
    """
    lams = LAMS if lams is None else lams
    p1s, te0s, rtm, rte = [], [], [], []
    for lm in lams:
        ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                          "top", "TM", THETA_0, lm, ng)
        p1s.append(_pick(ob, T, (+1,))[+1])
        rtm.append(_pick(ob, R, (0,))[0])
        ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                          "top", "TE", THETA_0, lm, ng)
        te0s.append(_pick(ob, T, (0,))[0])
        rte.append(_pick(ob, R, (0,))[0])
    # [260814] Worst value + **balance** of the two polarizations. With single-polarization transmission +
    # polarization diversity the TM and TE arms are summed in photocurrent, so their gain matching dominates performance.
    # That is why the square root of the product is used, not a weighted sum -- a sum allows solutions that discard one side
    # and grow the other (measured index 1262: eta 0.9938 / T0_TE 0.0005).
    # With a product, if one side dies the score dies with it.
    if w_te:
        val = _wmin([npa.sqrt(npa.maximum(p * t, 1e-12))
                     for p, t in zip(p1s, te0s)])
    else:
        val = _wmin(p1s)
    if w_refl:
        n = float(len(lams))
        val = val - w_refl * (sum(rtm) + sum(rte)) / n
    if w_echo or w_back:
        # Fate of the beam that rose at -theta inside the substrate. The two terms use the same solve.
        ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                          "sub", "TM", -THETA_SUB, LAM, ng)
        if w_echo:
            val = val - w_echo * _pick(ob, R, (+2,))[+2]
        if w_back:
            # relative order +1 -> k_x = 0 -> normal emission into air = fibre back-coupling
            val = val - w_back * _pick(ob, T, (+1,))[+1]
    return val, _wmin(p1s), _wmin(te0s)


def fom_retro(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
              w_leak=1.0, w_te=0.0, w_slope=0.0, ng=None, lams=None):
    """Return grating: substrate +theta incidence -> relative order -2 reflection (k -> -k, Littrow retro).

    Competing channels are R_0 (specular reflection to +G), R_-1 (normal downward), T_-1 (leakage into air).
    Since the unitary sum is 1, R_-2 = 1 is possible in principle.

    [260814] w_te added -- **TE also uses the same return grating**.
    With the direction fixed as single-polarization transmission + polarization diversity, the TE delay arm
    also goes through the k_x reversal at the top return grating. But this FoM was solving only TM,
    and as a result in 44 measured cells the TE return was left at a median 0.448 / minimum 0.0001
    (champion index 1530 is TM 0.999998 but TE 0.2327, matching 6.33 dB).
    The TE arm dies by that much, so it must be included together.

    With w_te > 0 the **square root of the product** of the two polarizations is taken at the worst wavelength --
    for the same reason as for the input grating. With a sum, the solution that discards one side wins.
    """
    lams = LAMS if lams is None else lams
    r2s, leaks, te2s, ramps = [], [], [], []
    for lm in lams:
        ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                          "sub", "TM", THETA_SUB, lm, ng)
        r2s.append(_pick(ob, R, (-2,))[-2])
        leaks.append(_pick(ob, T, (-1,))[-1])
        if w_slope:
            # [260820] Both-direction TM lane dpsi = (extraction_TM + returnTM) - (branch_TE +
            # returnTE). The slope at this spot (x=2s) enters directly into that combination --
            # C53 (retro_te) is at the x=s spot, so it does not touch this one.
            ramps.append(_ramp(ob, "TM", -2))
        if w_te:
            obt, Rt, _Tt = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                                  "sub", "TE", THETA_SUB, lm, ng)
            te2s.append(_pick(obt, Rt, (-2,))[-2])
    _sl = 0.0
    if w_slope and len(ramps) >= 2:
        # The geometric term is an even function, so a slope (an odd function) cannot cancel it -> the target is 0.
        _B = ramps[-1] * npa.conj(ramps[0])
        _sl = w_slope * (1.0 - npa.real(_B) / (npa.abs(_B) + 1e-12))
    if w_te:
        both = [npa.sqrt(npa.maximum(a * b, 1e-12))
                for a, b in zip(r2s, te2s)]
        val = _wmin(both) - w_leak * (sum(leaks) / len(leaks)) - _sl
        return val, _wmin(both), sum(leaks) / len(leaks)
    val = _wmin(r2s) - w_leak * (sum(leaks) / len(leaks)) - _sl
    return val, _wmin(r2s), sum(leaks) / len(leaks)


def fom_retro_te(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                 w_leak=1.0, w_passn=0.0, w_slope=0.0, ng=None, lams=None):
    """**TE-only** return grating (top x=s spot of the coplanar layout).

    [260820 new] In the coplanar layout the two return spots receive different polarizations:

        top x=s   the TE lane arrives  -> R(-2) of TE is needed
        top x=2s  the TM lane arrives  -> R(-2) of TM is needed

    The two spots are **different gratings**, so their patterns may differ (they are made on one face
    as long as the layer stack is the same). So there is no reason to satisfy both polarizations with one design --
    `fom_retro(w_te=1)` looks at the square root of the product, so it is **over-constrained**, and here
    only TE is maximised and all the degrees of freedom are used.

    Why needed: `retro` has `w_te=0` by default and had been solving only TM, and the confirmed champion
    index 2312 (`260817_c15are_525_450`) picked that way has TM R(-2) 0.99998 but
    **TE R(-2) 0.0641**. Used as is at both spots, the TE branch dies by 11.93 dB.

    The competing channels are the same as for TM: R_0 (specular), R_-1 (normal downward), T_-1 (air leakage).
    `w_leak` penalises only the air leakage among them.
    """
    lams = LAMS if lams is None else lams
    r2s, leaks, passn, ramps = [], [], [], []
    for lm in lams:
        ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                          "sub", "TE", THETA_SUB, lm, ng)
        r2s.append(_pick(ob, R, (-2,))[-2])
        leaks.append(_pick(ob, T, (-1,))[-1])
        if w_slope:
            # [260820] Bias phase slope. Since the delay arm passes the return, arg R(-2)
            # enters the system bias directly -- measured -1.8218 deg/nm, the largest of the three
            # devices, yet it was in no FoM.
            ramps.append(_ramp(ob, "TE", -2))
        if w_passn:
            # [260820] **Normal-incidence TM transmission.** At this spot (top x=s) the normal-emission
            # channel of the TM-lane extraction grating lands -- if it cannot be taken out into air,
            # it is reflected back, re-enters the extraction and becomes new stray light. The propagating
            # orders are only substrate m=0,+-1 / air m=0, so T(0) alone decides it.
            # Cost: solves per iteration double (cost ~1900 -> ~3800 s/seed).
            ob2, R2, T2 = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                                 "sub", "TM", 0.0, lm, ng)
            passn.append(_pick(ob2, T2, (0,))[0])
    val = _wmin(r2s) - w_leak * (sum(leaks) / len(leaks))
    if w_passn:
        val = val + w_passn * _wmin(passn)
    if w_slope and len(ramps) >= 2:
        # Penalises the phase difference between the complex amplitudes at the two band edges as 1 - cos(dphi).
        # Same form as `w_phase` -- differentiable and sign independent (only the magnitude is seen).
        _P = ramps[-1] * npa.conj(ramps[0])
        val = val - w_slope * (1.0 - npa.real(_P) / (npa.abs(_P) + 1e-12))
    return val, _wmin(r2s), sum(leaks) / len(leaks)


def fom_extract(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                w_te=0.0, w_ratio=0.0, w_stray=0.0, w_back=0.0, w_bal=0.0,
                w_echo2=0.0, w_slope=0.0, w_pure=0.0, w_fid=0.0,
                w_echo_cap=0.0, echo_cap_db=0.0, ng=None, lams=None,
                w_dump=0.0, dump_target=1.0 / 9.0, w_r2=0.0, fid_minus=0,
                w_rgo=0.0, rgo_target=0.5, pure_minus=0):
    """Extraction grating: extracts part at substrate +theta (reference arm) + passes by specular reflection,
    extracts the returned -theta (delay arm).

        reference arm P_ref = eta_+          (dm=-1, air order 0)
        pass-through  P_go  = R_0            (specular reflection keeping +G)
        delay arm     P_del = P_go * rho * eta_-   (rho is the return-grating share, excluded here)

        signal ~ sqrt(eta_+ * R_0 * eta_-)

    By unitarity eta_+ + eta_- <= 1 (same air m=0 row) and eta_+ + R_0 <= 1, so
    the optimum is **eta_+ = R_0 = eta_- = 4/9**, with the bound (4/9)^1.5 = 0.296296
    ([260828 correction]. The discarded value 1/3 . 2/3 -> 0.3849 is a model using only energy conservation,
    and grating reciprocity is S = P S^T P, not S = S^T -- the same correction as :926-927, :1134-1146,
    and signal_ceiling(:1877) is already 4/9).
    'Take out 50 % on entry and 100 % on return' is impossible.

    w_ratio pulls that split toward a **target form**: (ep-4/9)^2 + (em-4/9)^2
    (implementation :985. Corrected from 1/3 . 2/3 on 260829).
    A difference form ((em-2*ep)^2 etc.) must not be used -- when both channels are small it is cheaper to reduce
    the larger one, so it cuts efficiency just to fix the ratio (measured in Round P).

    w_stray penalises the share of the **part not extracted in the first pass (5/9 for the ceiling split 4/9)
    that scatters instead of going cleanly into R_0** (R_-1, R_-2). R_0 itself is not penalised --
    it is not thrown away but the pass-through channel that becomes the delay arm, and penalising it removes the delay line.

    [260906 caution] **This term ties two channels of different nature to one weight.**
    R(-1)|+th has k_x = 0 (normal), so it is the only kind of channel that can leave the slab
    (the exit), and the only fibre backflow to the x=0 input grating is R(-2)|+th.
    [260907 measured] At w_stray = 0.05, R(-2) fell from 0.3087 to 0.0451 but
    R(-1) stayed at 0.2016 -- the whole reduction budget goes to the backflow order and the penalised **sum**
    barely moves. Moreover every solution with w_stray > 0 **fell out at the linewidth gate**
    (minf2 89.9 -> 30.8 nm, gate 50.0). Backflow will be handled by `w_back`,
    and the exit has to be handled separately by a **bounded target term** aiming at 1/9.

    w_dump [260908] is that bounded target term. The designated exit R(-1)|+th (k_x = 0,
    the only channel that can leave the slab) has a measured median of 20.95 %, which is excessive against the ideal
    1/9 = 11.11 %, and neither of the two objectives reads this value.
    It is a **bounded squared** term of the form (rs[-1] - dump_target)^2 -- not an unbounded linear reward term.
    How the unbounded linear form collapses was already shown by `w_exit`
    (signal 0.296 -> 0.002, the exit took 98.6 %).

    w_back [260819] penalises the **total returning to the fibre**. In the folded layout all grating
    spacings are multiples of t*tan(theta), so the two below **land exactly on the x=0 input grating**:

        path 1  R(-2) @ +theta                     (scatter of the first pass)
        path 2  R(0) @ +theta  *  R(0) @ -theta    (specular reflection of the delay-arm residue)

        B = R(-2)|_+theta + R_0|_+theta * R_0|_-theta

    **Path 2 is 1.8 times path 1 and until now was in no FoM** (`w_stray` sees only
    +theta incidence, and the input grating's `w_refl` sees only immediate reflection). This is the main
    culprit of the measured system return loss of 5.1 dB.

    The Pareto of the 4-port unitary + reciprocity model said **signal 0.296096 is possible at B <= 0.05**,
    but **[260906 retracted] that point is unreachable.** The error was
    putting R(0)|+th and R(0)|-th as two free variables (0.5806 and 0.0129, a factor of 45 apart),
    and **reciprocity ties the two to the same number.**

    At the Littrow condition `SIN_TH = LAM/(PERIOD*N_SUB)`, k_x(+th) = +G, so writing
    the substrate-side modes as (k_x, direction), A=(+G,down) A'=(+G,up)
    B=(-G,down) B'=(-G,up), and by reciprocity S_ij = S_(j-bar)(i-bar)
    the reverse of A' is B and that of A is B'. That is, **the reverse partner of the +th specular channel
    is exactly the -th specular channel, so R_0|+th = R_0|-th holds identically.** No symmetry assumption
    enters, so **it cannot be broken by structure search.** Measured: 1410 pairs of the 2-layer code
    (two blocks) max deviation 1.8e-9. In a counterexample check ep = T(-1)|+th and em = T(+1)|-th
    differ by up to 0.68 -- because their reverse partners are not each other.

    Hence B = R(-2)|+th + R_0^2 and at the ceiling split R_0 = 4/9
    B >= (4/9)^2 = 0.1975 (value assuming the ceiling split), four times the 0.05 that the Pareto
    gave. **The effective lower bound is 0.1838 = 7.36 dB below, from the measured R_0**
    ([260907] in the pilot run back_total bottomed out at 0.1894 = 7.23 dB,
    and at that point R_0^2 = 0.1319 is 70 % of the remainder -- the bound was confirmed by measurement).
    Even killing R(-2) completely
    **the floor is B_min = R_0^2 = 0.1838, i.e. 7.36 dB**, and above it
    **0.25 dB/dB** of signal is given up (analytic 0.25, C34 measured 0.257; at w_back 0.4
    return loss 8.94 dB with a sensitivity cost of 0.855 dB).

    -> `w_back` is a real term that was in no FoM, but **it is not free.**
       The sweep table and the derivation are in the "[corrected, night of 260819] C34 measurement" section
       of the 260819 back-reflection path analysis.
       `w_stray` is useless here -- R(-2) 0.1118 -> 0.1063 but R_0^2
       rises 0.1838 -> 0.1848, so the total B is unchanged (squeezing a balloon).

    w_bal [260819] matches the **power of the two arms 1:1**.

        P_ref = eta_+ ,  P_del = R_0 * R_retro * eta_-   (R_retro = 0.99998 ~ 1)
        penalty = (eta_+ - R_0 * eta_-)^2

    Until now this term did not exist. `signal = sqrt(eta_+ * R_0 * eta_-)` **sees only the product, not the
    ratio** -- (0.44, 0.20) and (0.30, 0.30) score the same, so the optimisation had no reason to
    seek the balance point, and it actually ended at 2.24:1 (ER 14.0 dB).

    `w_ratio` aims at a predetermined split with `(ep-4/9)^2 + (em-4/9)^2` (:985), and
    neither `R_0` nor the return enters. When turned on in C21 it actually got worse.
    [260906] After the target was corrected to 4/9 . 4/9 it aims at ep = em, so the earlier argument that it "is not a
    balance term" no longer holds -- the ground for turning this term off is the C21 measurement,
    not an asymmetry of the target.

    [260827 correction] **Do not turn `w_bal` on.** The above "nearly free (-0.055 dB)" was wrong
    in two ways -- (1) the premise 'if the total is fixed' does not hold (the optimisation changes the total too),
    (2) the bound 0.334717 itself is a value of the discarded `S=S^T` model (the correct grating-reciprocity ceiling is
    (4/9)^1.5 = 0.296296 from `S = P S^T P`). In the 260821 correction the cost is **0.738 dB**.

    The interference term that serves as the criterion, `B = 2*sqrt(P_ref*P_del) = 2*signal`, is **already a
    product**, so maximising `signal` is maximising B. `w_bal` only **constrains** the solution to 1:1, so it can
    only lower the maximum or leave it. Measured: with `w_bal` on, the delivered power rose 64.5 -> 69.0 %
    but **B fell 58.7 -> 48.8 %** (-0.80 dB).

    The "ER 14.0 dB -> infinite" said to improve is **fringe visibility**, and in this structure the differential pair
    (bias 180 deg apart) cancels the common term `P_ref + P_del` exactly, so it is not a criterion.
    `w_ratio` is also turned off.

    [260901] **Band aggregation from mean to worst value.** The echo hinge (`w_echo_cap`) and
    `w_pure` now take the band **worst value**, not the band **mean** -- because the gate is
    defined at the worst condition (see `_echo_cap_pen`). `w_echo2` (linear
    soft weight) and `signal` (always `_wmin`) are left as they are. This correction
    changes the single-termination result slightly -- an intended change.

    [260901] **If `OUT_L3_NS` is not empty, the L3 equivalent refractive index is evaluated at several points
    at once and the worst value is optimised** (see `set_out_l3_list`). The single path of this function is
    solved again for each termination; signal/purity are combined as the minimum over terminations, the
    echo hinge as the maximum over all termination x wavelength, and the other terms as the
    termination mean. If `OUT_L3_NS` is empty, the cost and the result are
    exactly the same as before.

    w_rgo (ASCII note, 260926): optional bound target term on r0 = R(0)|+th,
    same shape as fom_split's w_r1/r1_target. Off by default; w_rgo=0.0 is a
    no-op and leaves every stored result bit-identical.

    pure_minus (ASCII note, 261002): when nonzero AND w_pure != 0, the w_pure
    term also reads the delay-arm (-theta) spacer power purity, design order
    m=+1 (same pass/order as fid_minus), and folds min over both passes.
    Default 0 leaves the objective and the checkpoint fingerprint unchanged.
    """
    lams = LAMS if lams is None else lams

    def _one_pass():
        """Solves the whole band at one termination (the current OUT_L3_UM/OUT_L3_EPS state).

        [260901 new] The body of the existing single loop was extracted as it is for reuse in the
        `OUT_L3_NS` termination loop -- the return values are the same as the local variables of the earlier version.
        """
        sigs, tes, rats, strays, backs, bals = [], [], [], [], [], []
        pures = []                      # [260827] design-channel purity at the L3 plane
        fids = []                       # [260922] same plane, including evanescent orders
        echo2s, bamps = [], []
        dumps = []                      # [260908] bounded target term for the designated exit R(-1)|+th
        bflows = []                     # [260923] fibre backflow R(-2)|+th **alone**
        rgos = []                       # [260926] w_rgo -- r0 = R(0)|+th target term
        for lm in lams:
            ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                              "sub", "TM", THETA_SUB, lm, ng)
            ep = _pick(ob, T, (-1,))[-1]          # +G -> air 0
            r0 = _pick(ob, R, (0,))[0]            # reflection keeping +G (delay arm pass-through)
            rs = _pick(ob, R, (-1, -2))           # neither arm
            # [260906] rs[-1] = R(-1)|+th is the k_x=0 normal **exit**, rs[-2] =
            # R(-2)|+th is the **fibre backflow** to the x=0 input grating. They must not be
            # lumped under one name and called "scatter" (strays sums the two).
            ob2, R2, T2 = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                                 "sub", "TM", -THETA_SUB, lm, ng)
            em = _pick(ob2, T2, (+1,))[+1]        # -G -> air 0
            if w_slope:
                # extraction bias = argR(0)|+th + argT(+1)|-th - argT(-1)|+th
                bamps.append(_ramp(ob, "TM", 0) * _tamp(ob2, "TM", 1)
                             * npa.conj(_tamp(ob, "TM", -1)))
            mr0 = _pick(ob2, R2, (0,))[0]         # reflection keeping -G -> x=0 input grating
            # [260820] Second-order echo seed. The m=+2 of -th incidence has k_x = +G, so it
            # **rises at +th and re-enters the return grating** (a 2-symbol delayed copy).
            # It shares the same residual share as the exit R(+1)|-th, so penalising this term can
            # move it to the exit without touching the signal (T(+1), R(0)).
            echo2s.append(_pick(ob2, R2, (+2,))[+2])
            sigs.append(npa.sqrt(npa.maximum(ep * r0 * em, 1e-12)))
            # [260829] Target corrected from 1/3 . 2/3 to **4/9**. The old value used
            # only energy conservation; the optimum including reciprocity `S = P S^T P` is
            # A = P = B = 4/9 and the ceiling is (4/9)^1.5 = 0.296296.
            rats.append((ep - 4.0 / 9.0) ** 2 + (em - 4.0 / 9.0) ** 2)
            strays.append(rs[-1] + rs[-2])
            backs.append(rs[-2] + r0 * mr0)
            bals.append((ep - r0 * em) ** 2)
            # [260908] Bounded squared term aiming the exit (rs[-1] = R(-1)|+th) at the ideal 1/9
            # -- not tied to the backflow (rs[-2]) as `w_stray` does.
            dumps.append((rs[-1] - dump_target) ** 2)
            # [260923] **Fibre backflow R(-2)|+th alone.** None of the existing three terms
            # sees only this quantity:
            #   `w_stray` = rs[-1] + rs[-2]  -> also penalises the exit that should stay open
            #   `w_back`  = rs[-2] + r0*mr0  -> `r0 = R_go` is an essential
            #                                   factor of signal, so pressing it reduces the signal directly.
            #                                   The record "w_back is a closed axis" meant
            #                                   that the **quantity was wrong**, not that the
            #                                   axis was closed.
            #   `w_dump`  = exit only         -> does not see backflow at all
            # Measured 260923 (SiO2 713 cells): backflow median **0.1913**, ideal 0. And the shortfall of
            # `eta_+ + R_go` against the ideal (8/9) is **exactly 0.1913**, so the whole
            # efficiency gap is this one term. The distribution spans 0.0039~0.7353, a factor 190,
            # and top-efficiency cells (93~95 % of the bound) hold 0.05~0.09 --
            # i.e. it is a quantity that can be reduced.
            # The ideal is 0, so no target form like `w_dump` is needed and it is **linear**.
            # Being a penalty, its runaway direction is opposite to that of the unbounded **reward** that
            # collapsed `w_exit`.
            bflows.append(rs[-2])
            # [260926] w_rgo -- delay-arm target term on r0 = R(0)|+th,
            # mirrors fom_split's w_r1/r1_target (same two-sided square).
            rgos.append((r0 - rgo_target) ** 2)
            if w_pure:
                # Fraction of power the design channel m=-1 takes at the plane where L3 sits (the spacer).
                # eta_+ is measured in air, so it cannot see the trapped parasitics m=0,-2 -- this term
                # fills that blind spot. **The term that makes it safe to put a metalens on top.**
                pures.append(spacer_purity(ob, "TM", -1, THETA_SUB))
                if pure_minus:
                    # [261002] delay-arm (-theta) pass crosses the same L3
                    # face (design order m=+1, same as the fid_minus call
                    # below). Default 0 -- when on, the existing min-fold
                    # over pures sees both passes.
                    pures.append(spacer_purity(ob2, "TM", +1, -THETA_SUB))
            if w_fid:
                # [260922] Also counts the evanescent orders that purity cannot see, at the same plane.
                # If the spacer refractive index is below 1.1784, purity is structurally 1.000,
                # so the gradient is 0, and through that gap the parasitic amplitude grows 30 times.
                fids.append(spacer_fid(ob, "TM", -1, THETA_SUB))
                if fid_minus:
                    # [260924] The -theta extraction pass also crosses the same L3 plane (delay arm,
                    # design order m=+1). Default 0 -- when on, the fid term sees the worse of the two directions.
                    fids.append(spacer_fid(ob2, "TM", +1, -THETA_SUB))
            if w_te:
                obt, _Rt, Tt = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                                      "sub", "TE", THETA_SUB, lm, ng)
                tes.append(_pick(obt, Tt, (-1,))[-1])
        return (sigs, tes, rats, strays, backs, bals, pures, fids, echo2s,
                bamps, dumps, bflows, rgos)

    if not OUT_L3_NS:
        (sigs, tes, rats, strays, backs, bals, pures, fids, echo2s, bamps,
         dumps, bflows, rgos) = _one_pass()
        n = float(len(lams))
        val = _wmin(sigs)
        if w_ratio:
            val = val - w_ratio * sum(rats) / n
        if w_stray:
            val = val - w_stray * sum(strays) / n
        if w_dump:
            val = val - w_dump * sum(dumps) / n
        if w_r2:
            val = val - w_r2 * sum(bflows) / n
        if w_rgo:
            val = val - w_rgo * sum(rgos) / n
        if w_back:
            val = val - w_back * sum(backs) / n
        if w_echo2:
            # [260903 T10b] The single-termination path also from band mean -> band worst.
            # The gate is the band worst **regardless of the number of terminations**, so the axis of this path
            # was also misaligned. Now every job gives `out_l3_ns` and nobody takes this road,
            # but if even one job that left out that key came in it would silently return to the
            # old behaviour -- the split convention itself is the danger.
            val = val - w_echo2 * _wmax(echo2s)
        val = val - _echo_cap_pen(echo2s, w_echo_cap, echo_cap_db)
        if w_pure and pures:
            # [260901] Band mean -> band worst (minimum). Because the gate is defined at the worst condition
            # -- for the detailed reason see the [260901] section of `_echo_cap_pen`.
            val = val - w_pure * (1.0 - _wmin(pures))
        if w_fid and fids:
            val = val - w_fid * (1.0 - _wmin(fids))
        if w_slope and len(bamps) >= 2:
            # [260820] Bias phase **slope**. The geometric term is an even function (ch^2), so a slope, an odd function,
            # cannot cancel it and only raises the maximum -> it is the thing to minimise.
            _B = bamps[-1] * npa.conj(bamps[0])
            val = val - w_slope * (1.0 - npa.real(_B) / (npa.abs(_B) + 1e-12))
        if w_bal:
            val = val - w_bal * sum(bals) / n
        if w_te and tes:
            val = val - w_te * (sum(tes) / len(tes))
        return val, _wmin(sigs), (sum(tes) / len(tes) if tes else 0.0)

    # [260901] Multi-termination -- iterates over OUT_L3_NS and reruns
    # _one_pass() for each termination; signal/purity are combined as the minimum over terminations, the echo
    # hinge as the maximum over all termination x wavelength, and the other terms as the termination mean.
    # OUT_L3_UM/OUT_L3_EPS are overwritten by set_out_l3 for every termination, and
    # even if solve raises midway, they are restored to the original values in finally so
    # the globals do not leak.
    global OUT_L3_UM, OUT_L3_EPS, OUT_L3B_UM, OUT_L3B_EPS, GAP_UM
    _orig_um, _orig_eps = OUT_L3_UM, OUT_L3_EPS
    _orig_bum, _orig_beps = OUT_L3B_UM, OUT_L3B_EPS
    _orig_gap = GAP_UM          # [260922] gap robustness fold -- see GAP_LIST
    sig_terms, pure_terms, echo_pool, te_terms = [], [], [], []
    fid_terms = []                      # [260922] fidelity including evanescent orders
    rat_terms, stray_terms, back_terms = [], [], []
    echo2_terms, bal_terms, slope_terms = [], [], []
    dump_terms = []                     # [260908] w_dump, folded the same way as w_stray
    bflow_terms = []                    # [260923] w_r2 -- backflow R(-2) alone
    rgo_terms = []                      # [260926] w_rgo -- r0 target term
    try:
        for _i3, _n3 in enumerate(OUT_L3_NS):
            _set_termination(_i3, _n3)
            if GAP_LIST:
                set_gap(GAP_LIST[_i3])
            (sigs, tes, rats, strays, backs, bals, pures, fids, echo2s,
             bamps, dumps, bflows, rgos) = _one_pass()
            n = float(len(lams))
            sig_terms.append(_wmin(sigs))
            echo_pool.extend(echo2s)          # the hinge uses the maximum over all termination x wavelength
            if w_pure and pures:
                pure_terms.append(_wmin(pures))
            if w_fid and fids:
                fid_terms.append(_wmin(fids))
            if w_ratio:
                rat_terms.append(sum(rats) / n)
            if w_stray:
                stray_terms.append(sum(strays) / n)
            if w_dump:
                dump_terms.append(sum(dumps) / n)
            if w_r2:
                bflow_terms.append(sum(bflows) / n)
            if w_rgo:
                rgo_terms.append(sum(rgos) / n)
            if w_back:
                back_terms.append(sum(backs) / n)
            if w_echo2:
                # [260903 correction (B) wavelength axis] Band mean -> band worst.
                # The gate is "band worst", so the objective must measure the same thing.
                # echo is a term **subtracted** from val, so worse means larger -> worst = maximum.
                echo2_terms.append(_wmax(echo2s))
            if w_bal:
                bal_terms.append(sum(bals) / n)
            if w_te and tes:
                te_terms.append(sum(tes) / len(tes))
            if w_slope and len(bamps) >= 2:
                _B = bamps[-1] * npa.conj(bamps[0])
                slope_terms.append(1.0 - npa.real(_B) / (npa.abs(_B) + 1e-12))
    finally:
        OUT_L3_UM, OUT_L3_EPS = _orig_um, _orig_eps
        OUT_L3B_UM, OUT_L3B_EPS = _orig_bum, _orig_beps
        GAP_UM = _orig_gap

    nt = float(len(OUT_L3_NS))
    val = _wmin(sig_terms)
    if w_ratio:
        val = val - w_ratio * sum(rat_terms) / nt
    if w_stray:
        val = val - w_stray * sum(stray_terms) / nt
    if w_dump:
        val = val - w_dump * sum(dump_terms) / nt
    if w_r2:
        val = val - w_r2 * sum(bflow_terms) / nt
    if w_rgo:
        val = val - w_rgo * sum(rgo_terms) / nt
    if w_back:
        val = val - w_back * sum(back_terms) / nt
    if w_echo2:
        # [260903 correction (A) termination axis] Termination mean -> termination worst.
        # **This is a defect proven by measurement.** The cheapest way to reduce the mean of the three
        # termination echoes is to dig one point deep, and the worst value stays as it is.
        # 260903 measurement (4 cells): a -19~-21 dB notch only at n=1.575, -10~-14 dB at both end points.
        # Against the gate (worst -20 dB) it fell short by 8.3~9.7 dB.
        # signal is _wmin and purity is also _wmin but only echo was a mean, so the axes were misaligned.
        val = val - w_echo2 * _wmax(echo2_terms)
    val = val - _echo_cap_pen(echo_pool, w_echo_cap, echo_cap_db)
    if w_pure and pure_terms:
        val = val - w_pure * (1.0 - _wmin(pure_terms))
    if w_fid and fid_terms:
        val = val - w_fid * (1.0 - _wmin(fid_terms))
    if w_slope and slope_terms:
        val = val - w_slope * sum(slope_terms) / nt
    if w_bal:
        val = val - w_bal * sum(bal_terms) / nt
    if w_te and te_terms:
        val = val - w_te * sum(te_terms) / nt
    return val, _wmin(sig_terms), (sum(te_terms) / nt if te_terms else 0.0)


def fom_split(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
              w_sym=0.0, w_ratio=0.0, w_stray=0.0, w_bal=0.0, w_cross=0.0,
              w_echo2=0.0, w_r1=0.0, r1_target=4.0 / 9.0, w_slope=0.0,
              pol="TE", w_pure=0.0, w_fid=0.0,
              w_echo_cap=0.0, echo_cap_db=0.0, ng=None, lams=None,
              fid_minus=0, pure_minus=0):
    """[260813] Bottom branch grating -- one device serves as both branch and extraction.

    [260819] `pol` added. **If the rotated layout is adopted this device becomes TM.**
    If the propagation plane of the TE branch is rotated 90 deg about y, the beam (E // y) that passed the input
    grating as order 0 meets a y-period grating (grooves // x), so E is **perpendicular** to the grooves = TM for it.
    Rotating the grating and the beam path together is a coordinate rotation, so the error is 0.
    The return grating can reuse the TM design as it is for the same reason (R(-2)=0.999983),
    and only the branch grating has to be redesigned for TM -- the TE-optimised one gives normal-incidence
    TM T(0) of 0.84~0.92, far from the required 1/3 (form birefringence).

    Layout: if the input grating passes TE as order 0 (normal), that light goes straight down the slab
    and hits this grating on the x=0 bottom face at **normal incidence**. There

        T(0)  -> emitted into air immediately    = reference arm
        R(+1) -> reflected into the substrate at +54.69 deg   = enters the delay arm
        R(-1) -> the other side (a loss if unused)
        R(0)  -> goes straight back up (backflow to the input grating)

    The delay arm has k_x reversed at the TE return grating at top x=423.5, comes back to the same spot at
    -54.69 deg, and this grating emits it again as T(+1) **normal emission** = extraction.
    The round-trip path is 2t/cos(theta), so the delay is exactly the same 5 ps as the TM arm.

        signal = sqrt( T0 * R(+1) * T_ext )

    The bound is the **same** as the extraction grating: T(0) and T_ext both leave as air m=0,
    sharing the same row, so T0 + T_ext <= 1, and in the normal-incidence row T0 + R(+1) <= 1.

    !! [260828 correction] The 1/3 . 2/3 model below was **discarded.** That value used only
       energy conservation, and grating reciprocity is **`S = P S^T P`**, not `S = S^T`.
       The actual ceiling including that constraint is

           optimum  T0 = R(+1) = T_ext = **4/9 = 0.44444**
           ceiling  signal <= **(4/9)^1.5 = 0.296296**   (not 0.38490)

       Confirmed 260816, achieved 260821 with extraction 99.979 % / branch 99.883 %.
       The `signal_ceiling` computation on 260828 and the `w_ratio` / `r1_target` defaults on
       260829 were **both corrected to 4/9** (:1207 signature `r1_target=4/9`, :1317
       `rats` target 4/9, stored-cell measurement `r1_target = 0.4444444444444444`).
       [260908 correction] The statement that used to be here, "the default (1/3) still keeps the old model",
       was **a stale statement from one day, 260828** -- the code is already 4/9.
       4/9 is not an individual bound but the **symmetric point where the product bound is reached** --
       one arm exceeding 4/9 is not a violation (measured P 0.4787 / B 0.5115).
       (discarded values: optimum T0 = 1/3, R(+1) = T_ext = 2/3 -> 0.38490)

    w_sym   : R(-1) suppression -- normal incidence is left-right symmetric, so if left alone R(+-1) splits
              evenly. To use only one side the symmetry must be broken by structure (the same means the input grating
              already uses with T(+1)/T(-1) 78:1).
    w_ratio : target form -- (T0-1/3)^2 + (R1-2/3)^2. A difference form wastes efficiency, as when both are small
              it is cheaper to reduce the larger one (measured in the extraction grating).
              **[260828] The target values 1/3 . 2/3 belong to a discarded model (see above).
              To turn it on, change to 4/9 first. Currently no job has it on.**
    w_stray : **normal-incidence** R(0) suppression -- the backflow to the input grating (:1213,
              taken from the theta = 0 solution). It is a different row from the **exit R(0)|-th of this lane** (:1192)
              and that one is not penalised -- easy to confuse because the names are the same.
              [260828] `R(+1)@-theta` (47.8 %) also goes to the same x=0 input grating, but
              it **must not be penalised** -- by `S = P S^T P` `R(+1)@-theta`
              = `R(+1)@normal` = `P` (delay-arm launch) and they agree to 8 digits at the design wavelength.
              Penalising it directly cuts the signal factor.
              (the driver Main_rcwa_scd_c.py:517-520 states the same point)

    [260901] **Band aggregation from mean to worst value + multi-termination.**
    The reason `w_echo_cap`/`w_pure` take the band worst value instead of the band mean, and
    the combination rule of the `OUT_L3_NS` multi-termination, are the same as in the
    same section of `fom_extract`. Terms that exist only here (`w_sym`, `w_r1`, `w_cross`)
    also follow the same rule (termination mean) in multi-termination.
    """
    lams = LAMS if lams is None else lams

    def _one_pass():
        """Solves the whole band at one termination (same structure as `fom_extract._one_pass`)."""
        sigs, syms, rats, strays, bals, crosses = [], [], [], [], [], []
        pures_sp = []                    # [260827] design-channel purity at the L3 plane
        fids_sp = []                     # [260922] L3-plane fidelity (including evanescent orders)
        echo2s, r1s, bamps = [], [], []
        for lm in lams:
            # (1) normal incidence -- reference-arm branch
            ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                              "sub", pol, 0.0, lm, ng)
            t0 = _pick(ob, T, (0,))[0]
            rr = _pick(ob, R, (-1, 0, 1))
            r1, rm1, r0 = rr[1], rr[-1], rr[0]
            # (2) returned beam (-theta) -- extraction
            ob2, R2, T2 = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                                  "sub", pol, -THETA_SUB, lm, ng)
            text = _pick(ob2, T2, (+1,))[+1]
            if w_slope:
                # branch bias = argR(+1)|normal + argT(+1)|-th - argT(0)|normal
                bamps.append(_ramp(ob, pol, 1) * _tamp(ob2, pol, 1)
                             * npa.conj(_tamp(ob, pol, 0)))
            # [260820] Second-order echo seed. The m=+2 of -th has k_x = +G, so it rises at +th
            # and re-enters the return grating (a coherent copy delayed by 2 symbols).
            # It shares the same residual share as the exit R(0)|-th, so penalising this term can
            # move it to the exit without touching the signal (T(+1), R(+1)).
            echo2s.append(_pick(ob2, R2, (+2,))[+2])
            if w_pure:
                # The design channel of the branch lane is normal-incidence m=0 (T(0) is the reference arm).
                pures_sp.append(spacer_purity(ob, pol, 0, 0.0))
                if pure_minus:
                    # [261002] delay-arm (-theta) pass crosses the same L3
                    # face (design order m=+1, same as the fid_minus call
                    # below). Default 0 -- when on, the existing min-fold
                    # over pures_sp sees both passes.
                    pures_sp.append(spacer_purity(ob2, pol, +1, -THETA_SUB))
            if w_fid:
                # [260922] The same channel at the L3 plane, without discarding evanescent orders.
                # Below the threshold purity becomes 1.000 and the gradient is 0.
                fids_sp.append(spacer_fid(ob, pol, 0, 0.0))
                if fid_minus:
                    # [260926] The -theta extraction pass also crosses the same L3 plane (delay arm,
                    # design order m=+1). Default 0 -- when on, the fid term sees the worse of the two directions
                    # (same convention as the same option of fom_extract).
                    fids_sp.append(spacer_fid(ob2, pol, +1, -THETA_SUB))
            sigs.append(npa.sqrt(npa.maximum(t0 * r1 * text, 1e-12)))
            syms.append(rm1)
            # [260819] Arm balance: P_ref = T0, P_del = R(+1)*R_retro*T_ext (R_retro ~ 1)
            bals.append((t0 - r1 * text) ** 2)
            if w_cross:
                # [260819] Cross-polarization blocking -- this grating must respond only to its own polarization.
                # If the opposite polarization leaked by the input grating reaches here, it becomes inter-channel crosstalk.
                obx, _Rx, Tx = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                                      "sub", _OTHER[pol], 0.0, lm, ng)
                crosses.append(_pick(obx, Tx, (0,))[0])
            # [260820] R(+1) target term. R1 is a factor of signal and, by reciprocity, also the upper bound of
            # Text -> an interior optimum at 4/9 exists. [260908 correction] 1/3 is a value of the discarded model.
            r1s.append((r1 - r1_target) ** 2)
            # [260829] Same correction as above -- the target is 4/9.
            rats.append((t0 - 4.0 / 9.0) ** 2 + (r1 - 4.0 / 9.0) ** 2)
            strays.append(r0)
        return (sigs, syms, rats, strays, bals, crosses, pures_sp,
                fids_sp, echo2s, r1s, bamps)

    if not OUT_L3_NS:
        (sigs, syms, rats, strays, bals, crosses, pures_sp, fids_sp,
         echo2s, r1s, bamps) = _one_pass()
        n = float(len(lams))
        val = _wmin(sigs)
        if w_sym:
            val = val - w_sym * sum(syms) / n
        if w_ratio:
            val = val - w_ratio * sum(rats) / n
        if w_stray:
            val = val - w_stray * sum(strays) / n
        if w_echo2:
            # [260903 T10b] The single-termination path also from band mean -> band worst.
            # For the grounds see the same section of `fom_extract` (the gate is the band worst regardless of
            # the number of terminations, and a job that omits out_l3_ns silently takes the old behaviour).
            val = val - w_echo2 * _wmax(echo2s)
        val = val - _echo_cap_pen(echo2s, w_echo_cap, echo_cap_db)
        if w_pure and pures_sp:
            # The design channel of the TE (branch) lane is normal-incidence m=0 (T(0) is the reference arm).
            # [260901] Band mean -> band worst (minimum) -- see fom_extract for the reason.
            val = val - w_pure * (1.0 - _wmin(pures_sp))
        if w_fid and fids_sp:
            val = val - w_fid * (1.0 - _wmin(fids_sp))
        if w_r1:
            val = val - w_r1 * sum(r1s) / n
        if w_slope and len(bamps) >= 2:
            # [260820] Bias phase **slope**. The geometric term is an even function (ch^2), so a slope, an odd function,
            # cannot cancel it and only raises the maximum -> it is the thing to minimise.
            _B = bamps[-1] * npa.conj(bamps[0])
            val = val - w_slope * (1.0 - npa.real(_B) / (npa.abs(_B) + 1e-12))
        if w_bal:
            val = val - w_bal * sum(bals) / n
        if w_cross and crosses:
            val = val - w_cross * sum(crosses) / n
        return val, _wmin(sigs), sum(syms) / n

    # [260901] Multi-termination -- same structure as fom_extract. sym_terms is
    # filled for every termination regardless of w_sym -- because the 3rd return value is always
    # sum(syms)/n (same convention as the original code).
    global OUT_L3_UM, OUT_L3_EPS, OUT_L3B_UM, OUT_L3B_EPS, GAP_UM
    _orig_um, _orig_eps = OUT_L3_UM, OUT_L3_EPS
    _orig_bum, _orig_beps = OUT_L3B_UM, OUT_L3B_EPS
    _orig_gap = GAP_UM          # [260922] gap robustness fold -- see GAP_LIST
    sig_terms, pure_terms, echo_pool, sym_terms = [], [], [], []
    fid_terms = []                       # [260922] worst value on the termination axis
    rat_terms, stray_terms, echo2_terms = [], [], []
    r1_terms, bal_terms, cross_terms, slope_terms = [], [], [], []
    try:
        for _i3, _n3 in enumerate(OUT_L3_NS):
            _set_termination(_i3, _n3)
            if GAP_LIST:
                set_gap(GAP_LIST[_i3])
            (sigs, syms, rats, strays, bals, crosses, pures_sp, fids_sp,
             echo2s, r1s, bamps) = _one_pass()
            n = float(len(lams))
            sig_terms.append(_wmin(sigs))
            echo_pool.extend(echo2s)
            sym_terms.append(sum(syms) / n)
            if w_pure and pures_sp:
                pure_terms.append(_wmin(pures_sp))
            if w_fid and fids_sp:
                fid_terms.append(_wmin(fids_sp))
            if w_ratio:
                rat_terms.append(sum(rats) / n)
            if w_stray:
                stray_terms.append(sum(strays) / n)
            if w_echo2:
                # [260903 correction (B) wavelength axis] Band mean -> band worst.
                # Matches the convention of the same section of fom_extract. echo is a subtracted term, so
                # worse means larger -> worst = maximum (`_wmax`).
                echo2_terms.append(_wmax(echo2s))
            if w_r1:
                r1_terms.append(sum(r1s) / n)
            if w_bal:
                bal_terms.append(sum(bals) / n)
            if w_cross and crosses:
                cross_terms.append(sum(crosses) / n)
            if w_slope and len(bamps) >= 2:
                _B = bamps[-1] * npa.conj(bamps[0])
                slope_terms.append(1.0 - npa.real(_B) / (npa.abs(_B) + 1e-12))
    finally:
        OUT_L3_UM, OUT_L3_EPS = _orig_um, _orig_eps
        OUT_L3B_UM, OUT_L3B_EPS = _orig_bum, _orig_beps
        GAP_UM = _orig_gap

    nt = float(len(OUT_L3_NS))
    val = _wmin(sig_terms)
    if w_sym:
        val = val - w_sym * sum(sym_terms) / nt
    if w_ratio:
        val = val - w_ratio * sum(rat_terms) / nt
    if w_stray:
        val = val - w_stray * sum(stray_terms) / nt
    if w_echo2:
        # [260903 correction (A) termination axis] Termination mean -> termination worst.
        # It was the same defect as in fom_extract. For the measured basis see the comment there
        # (all 4 cells have a single notch at n=1.575, 8.3~9.7 dB short of the worst-case criterion).
        val = val - w_echo2 * _wmax(echo2_terms)
    val = val - _echo_cap_pen(echo_pool, w_echo_cap, echo_cap_db)
    if w_pure and pure_terms:
        val = val - w_pure * (1.0 - _wmin(pure_terms))
    if w_fid and fid_terms:
        val = val - w_fid * (1.0 - _wmin(fid_terms))
    if w_r1:
        val = val - w_r1 * sum(r1_terms) / nt
    if w_slope and slope_terms:
        val = val - w_slope * sum(slope_terms) / nt
    if w_bal:
        val = val - w_bal * sum(bal_terms) / nt
    if w_cross and cross_terms:
        val = val - w_cross * sum(cross_terms) / nt
    return val, _wmin(sig_terms), sum(sym_terms) / nt


def fom_shared(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
               w_sym=0.0, w_ratio=0.0, w_stray=0.0, ng=None, lams=None):
    """[260814] Bottom output grating -- **one structure** serves as both TM extraction and TE branch.

    Fixed 2026-08-14: the bottom output grating must have identical TE/TM parameters. So the two polarizations
    cannot be designed separately, and a single (material, H1, H2, rho) must work both as the extraction
    grating and as the branch grating.

    The score is min(extraction, branch). A product or a sum must not be used -- a solution that discards one
    side and grows the other wins. In polarization diversity the two arms are summed in photocurrent and
    the loss at the worst SOP is |10log10(g_TM/g_TE)|, so physically too
    the worst value is the right metric.

    Prior basis (round SX, 2026-08-13): measuring both metrics on the same landscape the sharing
    cost was only 0.6 %p (individual optimum 76.7 % -> shared 76.1 %, index 1617/1621).
    At such a cost it is better to save one mask and one alignment degree of freedom.

    Cost: extraction (2 solves) + branch (2 solves) = 4 solves per wavelength, twice the single kind.
    """
    ve, se, _te = fom_extract(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                              w_ratio=w_ratio, w_stray=w_stray,
                              ng=ng, lams=lams)
    vs, ss, sym = fom_split(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                            w_sym=w_sym, w_ratio=w_ratio, w_stray=w_stray,
                            ng=ng, lams=lams)
    return npa.minimum(ve, vs), npa.minimum(se, ss), sym


def fom_input_pi(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                 w_refl=0.0, ng=None, lams=None):
    """Polarization-independent input grating -- sends TE and TM into the **same +1 order**.

    [260819 new] For the **single track** that gives up polarization diversity and makes the device
    itself polarization independent. The existing `input` **splits** TM into the +1 order and TE into
    the 0 order to make two branches; here both go to the +1 order so only one branch is made.
    Then branch gain matching / branch bias matching / reference-arm skew all disappear.

    The score is the worst-wavelength **geometric mean** of T(+1) of the two polarizations. A weighted sum allows
    solutions that discard one side and grow the other (same reason as the w_te comment of fom_input).

    **Why there is no inter-polarization phase-difference term:** the input grating is common to the reference
    and delay arms, so it cancels exactly in the bias. Phase matching is the job of the extraction/return.

    Measured note (no cell has been optimised for this target yet): index 1262, which maximised only TM
    (`w_te=0`), happens to give TM 0.9938 / TE 0.8349, geometric mean 0.9109.
    For a blazed grating **sending both polarizations to the same order is the natural state**,
    and the present polarization-splitting behaviour is in fact made by imposing constraints.
    """
    lams = LAMS if lams is None else lams
    tm1s, te1s, refl = [], [], []
    for lm in lams:
        ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                          "top", "TM", THETA_0, lm, ng)
        tm1s.append(_pick(ob, T, (+1,))[+1])
        refl.append(_pick(ob, R, (0,))[0])
        ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                          "top", "TE", THETA_0, lm, ng)
        te1s.append(_pick(ob, T, (+1,))[+1])
        refl.append(_pick(ob, R, (0,))[0])
    val = _wmin([npa.sqrt(npa.maximum(a * b, 1e-12))
                 for a, b in zip(tm1s, te1s)])
    if w_refl:
        val = val - w_refl * sum(refl) / float(len(refl))
    return val, _wmin(tm1s), _wmin(te1s)


def fom_extract_pi(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                   w_stray=0.0, w_back=0.0, ng=None, lams=None):
    """Polarization-independent extraction grating -- makes the same delay interferometer for both TM and TE.

    [260819 new] For the single track. For each polarization separately

        signal_p = sqrt(eta_+ * R_0 * eta_-)

    is computed and the **geometric mean of the two** is used as the score. If one side dies, the score dies too.

    Cost: solves per wavelength rise from 2 to 4, so the optimisation time is about 2 times.

    **The inter-polarization phase difference is not penalised here.** The return grating, which exists only on the
    delay arm, gives +-100 deg through Delta(arg R_-2) with no efficiency loss (measured index 2171 =
    -101.7 deg / 2177 = -10.9 / 2176 = +112.6, all three with both polarizations at 0.9996 or above).
    Leaving the extraction-grating phase free and matching it with the return uses fewer degrees of freedom.
    The value is recorded by report as `dbeta_extract`, so look at it when selecting cells.
    """
    lams = LAMS if lams is None else lams
    sigs, strays, backs = [], [], []
    for lm in lams:
        per = []
        for pol in ("TM", "TE"):
            ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                              "sub", pol, THETA_SUB, lm, ng)
            ep = _pick(ob, T, (-1,))[-1]
            r0 = _pick(ob, R, (0,))[0]
            rs = _pick(ob, R, (-1, -2))
            ob2, R2, T2 = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                                 "sub", pol, -THETA_SUB, lm, ng)
            em = _pick(ob2, T2, (+1,))[+1]
            mr0 = _pick(ob2, R2, (0,))[0]
            per.append(npa.sqrt(npa.maximum(ep * r0 * em, 1e-12)))
            if pol == "TM":
                # Scatter/backflow is penalised on a TM basis only. The two polarizations use the same structure,
                # so pushing one down pushes the other down too.
                strays.append(rs[-1] + rs[-2])
                backs.append(rs[-2] + r0 * mr0)
        sigs.append(npa.sqrt(npa.maximum(per[0] * per[1], 1e-12)))
    n = float(len(lams))
    val = _wmin(sigs)
    if w_stray:
        val = val - w_stray * sum(strays) / n
    if w_back:
        val = val - w_back * sum(backs) / n
    return val, _wmin(sigs), 0.0


def fom_split_pi(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                 w_sym=0.0, w_stray=0.0, w_echo2=0.0, w_slope=0.0,
                 w_dpsi=0.0, dpsi_target_deg=0.0,
                 ng=None, lams=None):
    """Polarization-independent **input/output grating** -- the only active device of the 2-grating TE-geometry single track.

    [260820 new] Layout: the beam passes vertically through the top face (a window without grating) and goes straight
    down the slab and hits this grating at bottom x=0 at **normal incidence**. One device serves three roles.

        normal incidence   T(0)  -> normal emission into air        = reference-arm output
                           R(+1) -> reflected into the substrate at +54.69 deg = enters the delay arm
                           R(0)  -> straight up = **fibre backflow** (the window has no grating, so
                                    it leaves as it is without re-diffraction -> there is no echo path at all)
        -theta incidence T(+1) -> normal emission into air        = delay-arm output (coaxial with the reference arm)

        signal_p = sqrt( T0 * R(+1) * T_ext )

    is computed for each polarization and the **geometric mean of the two** is used as the score (same
    convention as `fom_extract_pi`). A weighted sum allows solutions that discard one side, so it is forbidden.

    The bound is the same as `fom_split`: T0 and T_ext share the air m=0 row
    (T0 + T_ext <= 1), and in the normal-incidence row T0 + R(+1) <= 1, so
    the optimum is T0 = R(+1) = T_ext = **4/9** -> (4/9)^1.5 = 0.296296.
    (discarded: 1/3 . 2/3 -> 0.38490. The same correction as `fom_split`:1134-1146.)

    **Difference from `fom_extract_pi`:** scatter (w_sym) and backflow (w_stray) are penalised for **both
    polarizations**, not only TM. extract_pi applies only TM for the reason that "pushing one down pushes
    the other down too", but in this structure R(0) is the main path going **directly** to the fibre
    (it does not pass through one more diffraction order as in the TM geometry), so there is a value to
    watch separately for each polarization. There is no additional solve -- both polarizations are already solved.

    Cost: solves per wavelength rise from 2 to 4, about 2 times that of `fom_split`.
    """
    lams = LAMS if lams is None else lams
    sigs, syms, strays, echo2s, bamps = [], [], [], [], []
    btm, bte = [], []
    for lm in lams:
        per, sy, st, e2 = [], [], [], []
        for pol in ("TM", "TE"):
            ob, R, T = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                              "sub", pol, 0.0, lm, ng)
            t0 = _pick(ob, T, (0,))[0]
            rr = _pick(ob, R, (-1, 0, 1))
            r1, rm1, r0 = rr[1], rr[-1], rr[0]
            ob2, R2, T2 = _solve(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                                 "sub", pol, -THETA_SUB, lm, ng)
            text = _pick(ob2, T2, (+1,))[+1]
            # [260820] Second-order echo seed. The m=+2 of -th incidence has k_x = +G, so it
            # **rises at +th and re-enters the return grating**. If it makes one more round trip and
            # exits as T(+1), it is a coherent copy delayed by 2 symbols = real ISI.
            # Measured eye closure 2.4~2.9 dB, larger than the backflow. No additional solve.
            e2.append(_pick(ob2, R2, (+2,))[+2])
            if (w_slope and pol == "TE") or w_dpsi:
                # Based on the main polarization TE. bias = argR(+1)|normal + argT(+1)|-th - argT(0)
                # [260821] When `w_dpsi` is on, **both polarizations** are held -- to make the inter-polarization
                # bias difference the TM one is needed too. The order is (TM, TE).
                _b = (_ramp(ob, pol, 1) * _tamp(ob2, pol, 1)
                      * npa.conj(_tamp(ob, pol, 0)))
                if w_dpsi:
                    (btm if pol == "TM" else bte).append(_b)
                if w_slope and pol == "TE":
                    bamps.append(_b)
            per.append(npa.sqrt(npa.maximum(t0 * r1 * text, 1e-12)))
            sy.append(rm1)
            st.append(r0)
        sigs.append(npa.sqrt(npa.maximum(per[0] * per[1], 1e-12)))
        syms.append(0.5 * (sy[0] + sy[1]))
        strays.append(0.5 * (st[0] + st[1]))
        echo2s.append(0.5 * (e2[0] + e2[1]))
    n = float(len(lams))
    val = _wmin(sigs)
    if w_sym:
        val = val - w_sym * sum(syms) / n
    if w_stray:
        val = val - w_stray * sum(strays) / n
    if w_echo2:
        # [260903] The other foms unified the echo axis to `_wmax` (worst), but this function
        # is held back because its **gate convention is unconfirmed**. Whether the gate of the
        # polarization-independent device is the band worst has not been confirmed, and what is unconfirmed is not
        # changed for the sake of unification. **Check before use.**
        val = val - w_echo2 * sum(echo2s) / n
    if w_slope and len(bamps) >= 2:
        _B = bamps[-1] * npa.conj(bamps[0])
        val = val - w_slope * (1.0 - npa.real(_B) / (npa.abs(_B) + 1e-12))
    if w_dpsi and btm and bte:
        # [260821] **Inter-polarization bias difference `dpsi`.** In the single track the two polarizations
        # share one device and cannot be removed by a translation (the shift moves both equally by 2Gd).
        # The summed loss is -20log10|cos(dpsi/2)|, so it is pulled to 0.
        _th = dpsi_target_deg * npa.pi / 180.0
        _acc = 0.0
        for _t, _m in zip(bte, btm):
            _Dv = _t * npa.conj(_m)
            _acc = _acc + (1.0 - (npa.real(_Dv) * npa.cos(_th)
                                  + npa.imag(_Dv) * npa.sin(_th))
                           / (npa.abs(_Dv) + 1e-12))
        val = val - w_dpsi * _acc / float(len(bte))
    return val, _wmin(sigs), sum(syms) / n


FOMS = {"input": fom_input, "retro": fom_retro, "extract": fom_extract,
        "split": fom_split, "shared": fom_shared,
        # [260819] Polarization-independent single track. For the return, the existing fom_retro is used with w_te=1
        # as it is -- it already looks at the square root of the product of the two polarizations.
        "input_pi": fom_input_pi, "extract_pi": fom_extract_pi,
        # [260820] Input/output grating of the 2-grating TE geometry (bottom x=0).
        "split_pi": fom_split_pi,
        # [260820] TE return of the coplanar layout (top x=s). The two return spots are
        # different gratings so the patterns may differ, hence only TE is maximised.
        "retro_te": fom_retro_te}


_OTHER = {"TE": "TM", "TM": "TE"}


def _report_one(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2, kind, ng=None,
                pol="TE"):
    """Plain numpy summary -- all propagating orders, two polarizations, 3 wavelengths.

    **Single termination only.** It solves only at the one currently set `OUT_L3_UM/OUT_L3_EPS`.
    For the worst of 3 terminations call `report()` -- it calls this function per termination
    and folds them. (Split off in 260903 T13. The earlier `report` was this function.)
    """
    ng = NG_EVAL if ng is None else int(ng)
    r1, r2 = np.asarray(rho1, float), np.asarray(rho2, float)
    out = {"kind": kind, "nG": ng, "period_nm": PERIOD * 1e3,
           "n_sub": N_SUB, "theta_sub_deg": float(np.degrees(THETA_SUB)),
           "orders_sub": orders("sub"), "orders_air": orders("air")}
    if KX_HOLD:
        out["kx_hold"] = 1
    cases = {"input": [("air_TM", "top", "TM", THETA_0),
                       ("air_TE", "top", "TE", THETA_0),
                       ("sub_m_TM", "sub", "TM", -THETA_SUB)],
             # [260820] sub_n (normal-incidence TM) added. The top x=s spot is where the TM-lane
             # **normal-emission channel of the extraction grating** (extraction R(+1)|-th) lands.
             # If that share cannot be taken out into air it is reflected back and re-enters the extraction.
             # The normal-incidence propagating orders are only substrate m=0,+-1 / air m=0 (G/k0=1.17836),
             # so the decision key is a single T(0).
             "retro": [("sub_p_TM", "sub", "TM", THETA_SUB),
                       ("sub_p_TE", "sub", "TE", THETA_SUB),
                       ("sub_n_TM", "sub", "TM", 0.0)],
             "retro_te": [("sub_p_TE", "sub", "TE", THETA_SUB),
                          ("sub_p_TM", "sub", "TM", THETA_SUB),
                          ("sub_n_TM", "sub", "TM", 0.0)],
             "extract": [("sub_p_TM", "sub", "TM", THETA_SUB),
                         ("sub_m_TM", "sub", "TM", -THETA_SUB),
                         ("sub_p_TE", "sub", "TE", THETA_SUB)],
             # [260813] TE bottom branch/extraction. Normal incidence (branch) and -theta (extraction) are
             # the main conditions, and TM does not come to this spot but is measured to check contamination.
             # [260819] pol swaps the main/secondary polarization. In the rotated layout the branch grating becomes a
             # TM device, so with pol="TM" the main polarization is TM and the contamination monitor is TE.
             "split": [("sub_n_%s" % pol, "sub", pol, 0.0),
                       ("sub_m_%s" % pol, "sub", pol, -THETA_SUB),
                       ("sub_n_%s" % _OTHER[pol], "sub", _OTHER[pol], 0.0)],
             # [260814] Shared bottom grating -- solves both the extraction (TM) and branch (TE) conditions.
             "shared": [("sub_p_TM", "sub", "TM", THETA_SUB),
                        ("sub_m_TM", "sub", "TM", -THETA_SUB),
                        ("sub_n_TE", "sub", "TE", 0.0),
                        ("sub_m_TE", "sub", "TE", -THETA_SUB)],
             # [260819] Polarization-independent single track. input_pi measures the same conditions as input and
             # only the interpretation differs (TE also looks at the +1 order). extract_pi needs the TE
             # **-theta** condition additionally -- the existing extract measured TE only at +theta
             # for contamination monitoring, so eta_-(TE) was unknown.
             "input_pi": [("air_TM", "top", "TM", THETA_0),
                          ("air_TE", "top", "TE", THETA_0),
                          ("sub_m_TM", "sub", "TM", -THETA_SUB)],
             "extract_pi": [("sub_p_TM", "sub", "TM", THETA_SUB),
                            ("sub_m_TM", "sub", "TM", -THETA_SUB),
                            ("sub_p_TE", "sub", "TE", THETA_SUB),
                            ("sub_m_TE", "sub", "TE", -THETA_SUB)],
             # [260820] Input/output grating of the 2-grating TE geometry. Normal incidence (branch) and
             # -theta (extraction) are measured for **both polarizations** -- 4 solves/lam.
             "split_pi": [("sub_n_TM", "sub", "TM", 0.0),
                          ("sub_m_TM", "sub", "TM", -THETA_SUB),
                          ("sub_n_TE", "sub", "TE", 0.0),
                          ("sub_m_TE", "sub", "TE", -THETA_SUB)]}[kind]
    ms = tuple(range(-3, 4))
    for lam in LAMS:
        tag = "%.1f" % (lam * 1e3)
        # [261015 kx_hold] angle of this channel: THETA_SUB (ratio exactly 1.0) unless KX_HOLD. Only the
        # +-THETA_SUB entries of `cases` scale (0.0 stays 0.0); the air side (THETA_0) is untouched.
        _ths = theta_at(lam) / THETA_SUB
        if KX_HOLD:
            out["%s_theta_sub_deg" % tag] = float(np.degrees(THETA_SUB * _ths))
        # [260819 bug fix] If the loop variable is `pol` it **overwrites the function argument pol.**
        # In 260819 the opposite polarization for contamination monitoring (`_OTHER[pol]`) was added as the
        # 3rd of the split case, so after the loop pol changed to that opposite polarization and the split
        # branch below **read the key of a polarization that does not exist** and died with KeyError.
        # (Before that both cases had the same polarization, so the values coincided by accident and it did not show.)
        for nm, side, _pl, th in cases:
            ob, R, T = _solve(r1, r2, H1, H2, eps_b, eps_f1, eps_f2,
                              side, _pl, (th * _ths if side == "sub" else th),
                              lam, ng)
            Rd, Td = _pick(ob, np.real(R), ms), _pick(ob, np.real(T), ms)
            # [260816] The phase is extracted too -- ob is already held, so there is no additional
            # solve. Key names are argR_/argT_ to pair with the power keys.
            aR, aT, bR, bT = _args(ob, _pl, ms, with_abs=True)
            for m in ms:
                key = "m%d" % -m if m < 0 else "%d" % m
                out["%s_%s_R_%s" % (tag, nm, key)] = float(Rd[m])
                out["%s_%s_T_%s" % (tag, nm, key)] = float(Td[m])
                out["%s_%s_argR_%s" % (tag, nm, key)] = aR[m]
                out["%s_%s_argT_%s" % (tag, nm, key)] = aT[m]
                # [260908] For an evanescent order the power is zero by definition, so the amplitude is the only
                # measurable content of that order.
                out["%s_%s_absR_%s" % (tag, nm, key)] = bR[m]
                out["%s_%s_absT_%s" % (tag, nm, key)] = bT[m]
            out["%s_%s_energy" % (tag, nm)] = float(np.real(R).sum()
                                                    + np.real(T).sum())
            # [260827] **Keep the L3-plane illumination purity in the result.** It was used in the
            # optimisation through `w_pure` but not kept in report -- then to see the second axis of the decision
            # all stored densities have to be re-evaluated
            # (the same thing happened with 13 cells on 260823).
            # Measured only when a spacer layer exists: `spacer_purity` reads the spacer
            # layer (`Layer_N - 2` or `Layer_N - 3` depending on whether L3 exists), so
            # with no such layer it points at the wrong layer.
            if OUT_SPACER_UM > 0:
                _dm = None
                if kind == "split" and nm == "sub_n_%s" % pol:
                    _dm, _th = 0, 0.0
                elif kind == "extract" and nm == "sub_p_TM":
                    _dm, _th = -1, THETA_SUB * _ths
                if _dm is not None:
                    try:
                        # [260907] The per-order breakdown is kept too. The `w_pure` call sites
                        # (fom_extract:1043 / fom_split:1249) are on the autograd tracing
                        # path and have no result dict, so recording is done **here**
                        # in the report path -- the two lines pointed to earlier
                        # are inside the objective and have nowhere to store.
                        # Only three orders propagate inside the spacer:
                        # the design channel and the parasitic m=0 / m=-2 (extract) or
                        # m=+-1 (split, normal incidence). Which parasitic dominates
                        # changes what has to be suppressed, yet until now
                        # only the design-channel share was kept.
                        _sh = {}
                        out["%s_purity" % tag] = float(
                            spacer_purity(ob, _pl, _dm, _th, shares=_sh))
                        for _m, _v in _sh.items():
                            out["%s_purity_m%+d" % (tag, _m)] = float(_v)
                        # [260922] **The fidelity including evanescent orders is kept too.**
                        # `spacer_purity`, used by `w_pure`, discards evanescent orders, so
                        # below the propagation threshold 1.1784 it is structurally
                        # 1.000, and carriers below it could not be told apart from the result json alone --
                        # to read the `w_fid` ladder one had to put on the atom
                        # and wait 34 minutes. Measured (260922): the same 6 atoms
                        # give fid 0.5144 on a 465 nm n=1.08 carrier, 0.8640 at 900 nm,
                        # and 0.9301 at 1300 nm, while purity is 1.00000 in all three
                        # cases.
                        out["%s_fid" % tag] = float(
                            spacer_fid(ob, _pl, _dm, _th))
                    except Exception as _e:      # purity is auxiliary information --
                        out["%s_purity" % tag] = None   # it must not die here
                        out["purity_error"] = repr(_e)
                # [260924] fid of the -theta extraction pass (delay arm, design order m=+1).
                # Report only, with a separate suffix `_fidm` -- with `_fid` it would mix into
                # the `fid_worst` fold below and change the meaning of the +theta gate.
                if kind == "extract" and nm == "sub_m_TM":
                    try:
                        out["%s_fidm" % tag] = float(
                            spacer_fid(ob, _pl, +1, -THETA_SUB * _ths))
                    except Exception as _e:
                        out["%s_fidm" % tag] = None
                        out["fidm_error"] = repr(_e)
                # [260926] fid of the -theta extraction pass of the branch lane (delay arm, design order
                # m=+1) -- same convention as the same option of extract (`_fidm`).
                if kind == "split" and nm == "sub_m_%s" % pol:
                    try:
                        out["%s_fidm" % tag] = float(
                            spacer_fid(ob, _pl, +1, -THETA_SUB * _ths))
                    except Exception as _e:
                        out["%s_fidm" % tag] = None
                        out["fidm_error"] = repr(_e)
                # [261002] delay-arm (-theta) spacer POWER purity -- the same
                # spacer_purity() as the reference pass above, on the -theta
                # solve `ob` already in hand (no extra solve), design order
                # m=+1 (kx = 0 at -theta), both lanes. Always reported; the
                # suffix is `_puritym` (not `_purity`) so it stays out of the
                # `purity_worst` fold, exactly like `_fidm` vs `_fid`.
                if ((kind == "extract" and nm == "sub_m_TM")
                        or (kind == "split" and nm == "sub_m_%s" % pol)):
                    try:
                        out["%s_puritym" % tag] = float(
                            spacer_purity(ob, _pl, +1, -THETA_SUB * _ths))
                    except Exception as _e:
                        out["%s_puritym" % tag] = None
                        out["puritym_error"] = repr(_e)
    c = "%.1f" % (LAM * 1e3)
    # [260827] Band worst purity. Same convention as `signal_worst` (band worst value).
    _pv = [out[k] for k in out
           if k.endswith("_purity") and isinstance(out.get(k), float)]
    if _pv:
        out["purity_worst"] = float(min(_pv))
        out["purity_mean"] = float(sum(_pv) / len(_pv))
    # [260922] Fidelity is folded with the same band-worst convention. Kept next to `purity_worst`
    # so that the point where the two diverge (below the threshold) is directly visible in the result.
    _fv = [out[k] for k in out
           if k.endswith("_fid") and isinstance(out.get(k), float)]
    if _fv:
        out["fid_worst"] = float(min(_fv))
        out["fid_mean"] = float(sum(_fv) / len(_fv))
    # [260924] Band worst of the -theta pass fid (the `_fidm` keys above).
    _fmv = [out[k] for k in out
            if k.endswith("_fidm") and isinstance(out.get(k), float)]
    if _fmv:
        out["fidm_worst"] = float(min(_fmv))
    # [261002] -theta pass power purity, band worst (see `_puritym` above).
    _pmv = [out[k] for k in out
            if k.endswith("_puritym") and isinstance(out.get(k), float)]
    if _pmv:
        out["purity_m_worst"] = float(min(_pmv))
    out["gap_nm"] = float(GAP_UM * 1e3)
    if kind == "input":
        out["eta_TM_p1"] = out["%s_air_TM_T_1" % c]
        out["T0_TE"] = out["%s_air_TE_T_0" % c]
        out["echo_R_p2"] = out["%s_sub_m_TM_R_2" % c]
        # Only m=0 propagates on the air side, so the reflection is R_0 alone -- that is the fibre backflow.
        out["R_TM"] = out["%s_air_TM_R_0" % c]
        out["R_TE"] = out["%s_air_TE_R_0" % c]
        out["R_both"] = out["R_TM"] + out["R_TE"]
        # [260816] Headline for the phase library -- relative phase of the two arms.
        # Phase difference between TM (+1 order, going to the delay arm) and TE (0 order, reference arm).
        out["arg_TM_p1"] = out["%s_air_TM_argT_1" % c]
        out["arg_TE_0"] = out["%s_air_TE_argT_0" % c]
        out["dphi_TM_TE"] = _dphi(out["arg_TM_p1"], out["arg_TE_0"])
    elif kind == "input_pi":
        # [260819] Both polarizations into the same +1 order. The score is the square root of the product.
        out["eta_TM_p1"] = out["%s_air_TM_T_1" % c]
        out["eta_TE_p1"] = out["%s_air_TE_T_1" % c]
        out["bal_pi"] = float(np.sqrt(max(out["eta_TM_p1"]
                                          * out["eta_TE_p1"], 0.0)))
        out["pi_mismatch_dB"] = float(abs(10.0 * np.log10(
            max(out["eta_TM_p1"], 1e-12) / max(out["eta_TE_p1"], 1e-12))))
        out["R_TM"] = out["%s_air_TM_R_0" % c]
        out["R_TE"] = out["%s_air_TE_R_0" % c]
        out["R_both"] = out["R_TM"] + out["R_TE"]
        out["echo_R_p2"] = out["%s_sub_m_TM_R_2" % c]
        # The input grating is common to both arms, so it cancels in the bias. Recorded only.
        out["dphi_pi_in"] = _dphi(out["%s_air_TM_argT_1" % c],
                                  out["%s_air_TE_argT_1" % c])
    elif kind == "extract_pi":
        # [260819] signal is computed per polarization and the geometric mean is used as the score.
        for _p in ("TM", "TE"):
            _ep = out["%s_sub_p_%s_T_m1" % (c, _p)]
            _r0 = out["%s_sub_p_%s_R_0" % (c, _p)]
            _em = out["%s_sub_m_%s_T_1" % (c, _p)]
            out["eta_plus_%s" % _p] = _ep
            out["R_go_%s" % _p] = _r0
            out["eta_minus_%s" % _p] = _em
            out["signal_%s" % _p] = float(np.sqrt(max(_ep * _r0 * _em, 0.0)))
        # Plain keys for workbook compatibility (TM values). Per-polarization values are in the suffixed keys above.
        out["eta_plus"] = out["eta_plus_TM"]
        out["R_go"] = out["R_go_TM"]
        out["eta_minus"] = out["eta_minus_TM"]
        out["signal"] = float(np.sqrt(max(out["signal_TM"]
                                          * out["signal_TE"], 0.0)))
        out["signal_ceiling"] = float((4.0 / 9.0) ** 1.5)
        out["pi_mismatch_dB"] = float(abs(10.0 * np.log10(
            max(out["signal_TM"], 1e-12) / max(out["signal_TE"], 1e-12))))
        # **The extraction-grating share of the inter-polarization bias phase difference.** The return grating's
        # Delta(arg R_-2) must cancel this for the single track to be SOP independent.
        #   beta = arg R_0 + arg T_+1 - arg T_-1 + arg R_-2(retro)
        # and the input grating and the propagation 2t*k_z are polarization independent, so they drop out.
        _d = lambda t, k: _dphi(out["%s_%s_TM_%s" % (c, t, k)],
                                out["%s_%s_TE_%s" % (c, t, k)])
        out["dbeta_extract"] = float(
            (_d("sub_p", "argR_0") + _d("sub_m", "argT_1")
             - _d("sub_p", "argT_m1") + 180.0) % 360.0 - 180.0)
    elif kind == "split_pi":
        # [260820] Per polarization signal = sqrt(T0 * R(+1) * T_ext), geometric mean is the score.
        for _p in ("TM", "TE"):
            _t0 = out["%s_sub_n_%s_T_0" % (c, _p)]
            _r1 = out["%s_sub_n_%s_R_1" % (c, _p)]
            _tx = out["%s_sub_m_%s_T_1" % (c, _p)]
            out["split_T0_%s" % _p] = _t0
            out["split_R1_%s" % _p] = _r1
            out["split_Text_%s" % _p] = _tx
            out["signal_%s" % _p] = float(np.sqrt(max(_t0 * _r1 * _tx, 0.0)))
            # R(0) = backflow directly to the fibre. The window has no grating, so it leaves without re-diffraction.
            out["split_R0_%s" % _p] = out["%s_sub_n_%s_R_0" % (c, _p)]
            out["split_Rm1_%s" % _p] = out["%s_sub_n_%s_R_m1" % (c, _p)]
        out["signal"] = float(np.sqrt(max(out["signal_TM"]
                                          * out["signal_TE"], 0.0)))
        out["signal_ceiling"] = float((4.0 / 9.0) ** 1.5)
        out["pi_mismatch_dB"] = float(abs(10.0 * np.log10(
            max(out["signal_TM"], 1e-12) / max(out["signal_TE"], 1e-12))))
        # Backflow is taken as the worst of the two polarizations -- the aim is to guarantee SOP independence.
        out["back_R0_worst"] = float(max(out["split_R0_TM"],
                                         out["split_R0_TE"]))
        # Plain keys for workbook compatibility
        out["split_T0"] = out["split_T0_TM"]
        out["split_R1"] = out["split_R1_TM"]
        out["split_Text"] = out["split_Text_TM"]
        # This grating's share of the inter-polarization bias phase difference; the return must cancel it (SOP independence).
        _ds = lambda t, k: _dphi(out["%s_%s_TM_%s" % (c, t, k)],
                                 out["%s_%s_TE_%s" % (c, t, k)])
        # [260823] None guard. `_args` gives None for orders with amplitude below 1e-6 and
        # `_dphi` propagates it as it is. Without the guard a TypeError occurs at the report stage and
        # **the whole result is lost even after all seeds have run** (measured 260823 06:18:
        # c62pi_we080_lr03 ran 8.2 hours and died). Same measure as the 3-layer case.
        _b1, _b2, _b3 = (_ds("sub_n", "argR_1"), _ds("sub_m", "argT_1"),
                         _ds("sub_n", "argT_0"))
        if None not in (_b1, _b2, _b3):
            out["dbeta_split"] = float(
                (_b1 + _b2 - _b3 + 180.0) % 360.0 - 180.0)
    elif kind == "retro_te":
        # [260820] TE is the main polarization. TM is measured too to keep the matching.
        out["retro_R_m2_TE"] = out["%s_sub_p_TE_R_m2" % c]
        out["retro_R_m2"] = out["%s_sub_p_TM_R_m2" % c]
        out["leak_T_m1_TE"] = out["%s_sub_p_TE_T_m1" % c]
        out["retro_gain_mismatch_dB"] = float(abs(10.0 * np.log10(
            max(out["retro_R_m2"], 1e-12)
            / max(out["retro_R_m2_TE"], 1e-12))))
        # Phase library -- the phase of the TE return enters the TE branch bias directly.
        out["arg_retro_R_m2_TE"] = out["%s_sub_p_TE_argR_m2" % c]
        out["arg_retro_R_m2"] = out["%s_sub_p_TM_argR_m2" % c]
        # [260820] TM-lane exit. This spot has to let the light pass for the normal channel of the extraction
        # grating to open. R(0) is the share that is reflected back and re-enters the extraction.
        out["pass_T0_n_TM"] = out["%s_sub_n_TM_T_0" % c]
        out["backn_R0_TM"] = out["%s_sub_n_TM_R_0" % c]
        out["pass_T0_n_worst"] = float(min(
            out["%.1f_sub_n_TM_T_0" % (lm * 1e3)] for lm in LAMS))
        # [260820] Bias phase slope [deg/nm]. Since the delay arm passes the return,
        # arg R(-2) enters the system bias directly -- the largest of the three devices, yet
        # until now it was in no key (measured TE -1.8218 deg/nm).
        _ls = [lm * 1e3 for lm in LAMS]
        for _p in ("TE", "TM"):
            _a = [out["%.1f_sub_p_%s_argR_m2" % (x, _p)] for x in _ls]
            _u = np.degrees(np.unwrap(np.radians(_a)))
            out["slope_retro_%s" % _p] = float(np.polyfit(_ls, _u, 1)[0])
    elif kind == "retro":
        # [260821] The slope record that existed only in the retro_te branch is put here too.
        # C56 (w_slope, 4 cells) ended without a slope key and could not be evaluated -- it had
        # to be recomputed from the stored density.
        _ls = [lm * 1e3 for lm in LAMS]
        for _p in ("TE", "TM"):
            try:
                _a = [out["%.1f_sub_p_%s_argR_m2" % (x, _p)] for x in _ls]
            except KeyError:
                continue
            _u = np.degrees(np.unwrap(np.radians(_a)))
            out["slope_retro_%s" % _p] = float(np.polyfit(_ls, _u, 1)[0])
        out["retro_R_m2"] = out["%s_sub_p_TM_R_m2" % c]
        out["leak_T_m1"] = out["%s_sub_p_TM_T_m1" % c]
        # [260814] TE also uses the same grating. Raised as a named key so that it shows in the ranking
        # -- because this was not visible, a TE 0.23 one remained as champion.
        out["retro_R_m2_TE"] = out["%s_sub_p_TE_R_m2" % c]
        out["retro_gain_mismatch_dB"] = float(abs(10.0 * np.log10(
            max(out["retro_R_m2"], 1e-12)
            / max(out["retro_R_m2_TE"], 1e-12))))
        # [260816] For the phase library. The return efficiency is already in the 0.9999 range, so the purpose of
        # adding seeds is to **secure phase degrees of freedom** -- how much phase the returned beam
        # carries out is itself the means of correction.
        out["arg_retro_R_m2"] = out["%s_sub_p_TM_argR_m2" % c]
        # [260820] Same quantities as retro_te. No normal channel lands at this spot (x=2s),
        # but the two spots are made with the same stack, so they are kept together for comparison.
        out["pass_T0_n_TM"] = out["%s_sub_n_TM_T_0" % c]
        out["backn_R0_TM"] = out["%s_sub_n_TM_R_0" % c]
        out["pass_T0_n_worst"] = float(min(
            out["%.1f_sub_n_TM_T_0" % (lm * 1e3)] for lm in LAMS))
    elif kind == "split":
        _P, _Q = pol, _OTHER[pol]
        t0 = out["%s_sub_n_%s_T_0" % (c, _P)]    # reference arm (normal emission)
        r1 = out["%s_sub_n_%s_R_1" % (c, _P)]    # enters the delay arm
        rm1 = out["%s_sub_n_%s_R_m1" % (c, _P)]  # other side (unused)
        r0 = out["%s_sub_n_%s_R_0" % (c, _P)]    # backflow upward
        text = out["%s_sub_m_%s_T_1" % (c, _P)]  # extraction of the returned beam
        out["split_pol"] = _P
        out.update(split_T0=t0, split_R1=r1, split_Rm1=rm1, split_R0=r0,
                   split_Text=text,
                   signal=float(np.sqrt(max(t0 * r1 * text, 0.0))),
                   signal_ceiling=float((4.0 / 9.0) ** 1.5))
        # Left-right asymmetry -- 1 means it goes completely to one side only.
        out["split_asym"] = float((r1 - rm1) / max(r1 + rm1, 1e-12))
        # [260908] **Unify the split vocabulary of the two lanes.** split and extract have
        # used the same name for different physical quantities -- `arm_ref` in extract duplicates
        # `eta_plus` (=ep), and in split it is `t0`. With different names the lookup would have failed and
        # shown it, but **when the same name holds different things, an aggregate pooling the two lanes
        # gives a plausible value and no error occurs.**
        # Same form as the `R_fs`/`R_go` merge, and it is already in all stored results.
        #
        # The two models have the same structure:
        #   extract  signal = sqrt(eta_plus * R_go * eta_minus), optimum 4/9 each
        #   split    signal = sqrt(T0 * R(+1) * T_ext),          optimum 4/9 each
        # So t0 -> eta_plus, r1 -> R_go, text -> eta_minus correspond.
        # `split_*` and `arm_*` are **left as they are** -- stored results and analysis
        # scripts all read those names, so renaming would break the corpus retroactively.
        #
        # **`stray_R` is not created.** In extract `ep + R_go + stray_R = 1`
        # is an **exact identity** in that lane, but split has not an identity but two
        # inequalities -- `T0 + T_ext <= 1` in the air m=0 row, `T0 + R(+1) <= 1` in the normal-incidence
        # row. Putting `stray_R` here would produce an identity that does not
        # hold. The saturation is already held by `unitary_air_row` /
        # `unitary_norm_row`.
        out["eta_plus"] = float(t0)
        out["R_go"] = float(r1)
        out["eta_minus"] = float(text)
        # [260829 correction] The target is not 1:2 but **T0=R1=Text=4/9**.
        out["split_ratio_R1_over_T0"] = float(r1 / max(t0, 1e-12))
        # Saturation of the two unitary rows -- see which one ties up the budget.
        out["unitary_norm_row"] = float(t0 + r1 + rm1 + r0)   # normal-incidence row
        out["unitary_air_row"] = float(t0 + text)             # air m=0 row
        # [260819] Arm balance (return R(-2) approximated as 1)
        # [260908] **`arm_ref` holds something different in each lane.** Here (split)
        # `t0` = reference-arm normal emission, and in extract it is `ep` = a duplicate of `eta_plus`.
        # Do not read the lanes mixed -- the unified names are `eta_plus`/`R_go`/`eta_minus` above.
        out["arm_ref"] = float(t0)
        out["arm_del"] = float(r1 * text)
        out["arm_ratio"] = float(t0 / max(r1 * text, 1e-12))
        out["arm_imbalance"] = float(abs(t0 - r1 * text))
        # TM contamination -- TM should not come to this spot, but it can arrive by scattering.
        out["split_TM_leak_T0"] = out["%s_sub_n_%s_T_0" % (c, _Q)]
        # [260816] For the phase library -- phase difference between the reference arm (T0) and the delay-arm extraction (T_ext).
        # The two arms meet and interfere at the detector, so this difference is the operating point.
        out["arg_split_T0"] = out["%s_sub_n_%s_argT_0" % (c, _P)]
        out["arg_split_Text"] = out["%s_sub_m_%s_argT_1" % (c, _P)]
        out["dphi_split_arms"] = _dphi(out["arg_split_Text"],
                                       out["arg_split_T0"])
        # [260820 new] **True arm bias.** The delay arm passes the branch grating twice:
        #   (0,down) -R(+1)@normal-> (832,up) -R(-2)-> (832-832,down) = the same grating
        #   -> emitted as T(+1)@-th
        # So sum m = +1 +1 = 2 and the shift sensitivity is 2G (same as extraction).
        # `dphi_split_arms` above lacks the entry R(+1), so sum m = 1 -- **half**, and
        # used as a trimming target the actual bias comes out at 2 times the indicated value.
        out["arg_split_R1"] = out["%s_sub_n_%s_argR_1" % (c, _P)]
        # [260823] None guard -- same reason as dbeta_split above.
        if None not in (out["arg_split_R1"], out["arg_split_Text"],
                        out["arg_split_T0"]):
            out["bias_split_arms"] = float(
                (out["arg_split_R1"] + out["arg_split_Text"]
                 - out["arg_split_T0"] + 180.0) % 360.0 - 180.0)
    elif kind == "shared":
        # [260814] One structure serves as both TM extraction and TE branch. Both metrics are recorded and
        # signal is set to their **minimum** -- because the decision criterion is min.
        ep = out["%s_sub_p_TM_T_m1" % c]
        rgo = out["%s_sub_p_TM_R_0" % c]
        em = out["%s_sub_m_TM_T_1" % c]
        t0 = out["%s_sub_n_TE_T_0" % c]
        r1 = out["%s_sub_n_TE_R_1" % c]
        rm1 = out["%s_sub_n_TE_R_m1" % c]
        text = out["%s_sub_m_TE_T_1" % c]
        s_ex = float(np.sqrt(max(ep * rgo * em, 0.0)))
        s_sp = float(np.sqrt(max(t0 * r1 * text, 0.0)))
        out.update(eta_plus=ep, R_go=rgo, eta_minus=em,
                   split_T0=t0, split_R1=r1, split_Rm1=rm1, split_Text=text,
                   signal_extract=s_ex, signal_split=s_sp,
                   signal=min(s_ex, s_sp),
                   signal_ceiling=float((4.0 / 9.0) ** 1.5))
        out["split_asym"] = float((r1 - rm1) / max(r1 + rm1, 1e-12))
        # In polarization diversity the worst-SOP loss = |10log10(g_TM/g_TE)| dB.
        # This must be small so that no fading arises when the two arms are summed in photocurrent.
        out["gain_mismatch_dB"] = float(abs(10.0 * np.log10(
            max(s_ex, 1e-12) / max(s_sp, 1e-12))))
    else:
        ep = out["%s_sub_p_TM_T_m1" % c]
        r0 = out["%s_sub_p_TM_R_0" % c]
        em = out["%s_sub_m_TM_T_1" % c]
        stray = out["%s_sub_p_TM_R_m1" % c] + out["%s_sub_p_TM_R_m2" % c]
        out.update(eta_plus=ep, R_go=r0, eta_minus=em, stray_R=stray,
                   signal=float(np.sqrt(max(ep * r0 * em, 0.0))),
                   signal_ceiling=float((4.0 / 9.0) ** 1.5))
        # [260906 correction] **The target ratio is 1.000.** The 1:2 split (eta_-/eta_+ = 2)
        # is a discarded model using only energy conservation, and the optimum including reciprocity S = P S^T P is
        # ep = r0 = em = 4/9, so eta_-/eta_+ = 1. The 260829 correction reached only the objective
        # (:985 rats) and not this report key, so **a cell that exactly hit 4/9 was printed with
        # ratio_err 50 %**. It is a display-only key not used in scoring/gates. Old JSON is
        # distinguished by the presence of `ratio_target` -- if absent
        # it is the |ratio-2|/2 definition.
        out["ratio_minus_over_plus"] = float(em / max(ep, 1e-12))
        out["ratio_target"] = 1.0
        out["ratio_err"] = float(abs(out["ratio_minus_over_plus"] - 1.0))
        # Energy budget -- a table to check by eye 'does the total energy come out that way'.
        # First pass (+G incidence): extraction + pass-through + scatter + rest = 1
        out["budget_fwd_extract"] = ep
        out["budget_fwd_pass"] = r0
        out["budget_fwd_stray"] = stray
        out["budget_fwd_rest"] = float(1.0 - ep - r0 - stray)
        # Return (-G incidence): extraction + rest = 1
        out["budget_rev_extract"] = em
        out["budget_rev_rest"] = float(1.0 - em)
        # How saturated the two unitary constraints are (1.0 is the bound)
        out["unitary_air_row"] = float(ep + em)
        out["unitary_fwd_row"] = float(ep + r0)
        # [260819] Fibre backflow -- both paths land on the x=0 input grating.
        #   path 1 R(-2)@+th / path 2 R(0)@+th * R(0)@-th  (path 2 is 1.8 times larger)
        mr0 = out["%s_sub_m_TM_R_0" % c]
        out["back_p1_Rm2"] = out["%s_sub_p_TM_R_m2" % c]
        out["back_R0_rev"] = mr0
        out["back_p2"] = float(r0 * mr0)
        out["back_total"] = float(out["back_p1_Rm2"] + r0 * mr0)
        # [260819] Arm balance. The return R(-2)=0.99998, so it is approximated as 1.
        # [260908] **`arm_ref` here is a complete duplicate of `eta_plus` (=ep)** --
        # it is not the delay-arm reflection. Put side by side in a table, `arm_ref` and `eta_plus` show
        # the same column twice, and that duplication is itself the discriminating signal. Delay-arm reflection: `R_go`.
        # And this name **differs per lane** -- in split it is `t0`.
        out["arm_ref"] = float(ep)
        out["arm_del"] = float(r0 * em)
        out["arm_ratio"] = float(ep / max(r0 * em, 1e-12))
        out["arm_imbalance"] = float(abs(ep - r0 * em))
        # [260816] For the phase library -- phase difference of reference arm (eta_+) and delay-arm extraction (eta_-).
        # The two arms meet at the detector, so this difference sets the operating point.
        out["arg_eta_plus"] = out["%s_sub_p_TM_argT_m1" % c]
        out["arg_eta_minus"] = out["%s_sub_m_TM_argT_1" % c]
        out["dphi_extract_arms"] = _dphi(out["arg_eta_minus"],
                                         out["arg_eta_plus"])
        # [260831 new] **Complete arm bias.** Extraction counterpart of the correction split got in 260820
        # -- back then extraction was passed over with "sum m = 2 so it happened to be right", but
        # what was right was **only the slope (|dm|=2)** and **the absolute phase was wrong.**
        #
        # The actual path of the delay arm is R(0)@+th -> return -> T(+1)@-th, but
        # `dphi_extract_arms` omits the entry reflection `arg R(0)@+th`.
        # Being m=0 it does not enter the shift sensitivity (deg/nm), but **it does enter the operating point.**
        # Measured (55 cells, 1550 nm): median |arg R(0)@+th| **100.7 deg**
        # (min 11.9 / max 176.5). The 4-zone I/Q needs 90 deg steps, so
        # **the omitted term is larger than the thing to be controlled.**
        #
        # `dphi_extract_arms` is left as it is for slope use (both keys listed side by side as in split).
        # The phase library/trimming must use this key.
        out["arg_extract_R0"] = out["%s_sub_p_TM_argR_0" % c]
        # [same None guard as 260823] An order with amplitude below 1e-6 has phase None, so
        # adding it as it is dies in the final report (after all 8 seeds have run).
        if None not in (out["arg_extract_R0"], out["arg_eta_minus"],
                        out["arg_eta_plus"]):
            out["bias_extract_arms"] = float(
                (out["arg_extract_R0"] + out["arg_eta_minus"]
                 - out["arg_eta_plus"] + 180.0) % 360.0 - 180.0)
    # Bandwidth: spread between wavelengths. Solutions narrowed by GMR show up here.
    #
    # [260814] Also puts out **the same quantity as the FoM** as the score key. Before that the FoM was
    # min_lam sqrt(eta*T0_TE) but score_key was eta_TM_p1, so a seed with only high eta was adopted
    # regardless of the balance value the optimiser raised. In 44 measured cells the
    # median was 0 % but up to 35.9 % was being missed.
    if kind == "input":
        v = [out["%.1f_air_TM_T_1" % (lm * 1e3)] for lm in LAMS]
        out["eta_TM_p1_spread"] = float((max(v) - min(v)) / max(max(v), 1e-9))
        b = [float(np.sqrt(max(out["%.1f_air_TM_T_1" % (lm * 1e3)]
                               * out["%.1f_air_TE_T_0" % (lm * 1e3)], 0.0)))
             for lm in LAMS]
        out["bal_worst"] = float(min(b))
        out["bal_worst_spread"] = float((max(b) - min(b)) / max(max(b), 1e-9))
    elif kind == "retro":
        v = [out["%.1f_sub_p_TM_R_m2" % (lm * 1e3)] for lm in LAMS]
        out["retro_R_m2_spread"] = float((max(v) - min(v)) / max(max(v), 1e-9))
        b = [float(np.sqrt(max(out["%.1f_sub_p_TM_R_m2" % (lm * 1e3)]
                               * out["%.1f_sub_p_TE_R_m2" % (lm * 1e3)], 0.0)))
             for lm in LAMS]
        out["bal_worst"] = float(min(b))
        out["bal_worst_spread"] = float((max(b) - min(b)) / max(max(b), 1e-9))
    elif kind == "split":
        # split also has score key 'signal' but a different channel -- using the key of extract
        # as it is dies with KeyError (because the TM condition is not solved).
        # [260819 bug fix] The polarization was hard-coded to TE. A pol="TM" (rotated layout) job
        # does not solve the sub_m_TE condition at all, so it dies here with KeyError.
        _P = out.get("split_pol") or pol
        v = [float(np.sqrt(max(out["%.1f_sub_n_%s_T_0" % (lm * 1e3, _P)]
                               * out["%.1f_sub_n_%s_R_1" % (lm * 1e3, _P)]
                               * out["%.1f_sub_m_%s_T_1" % (lm * 1e3, _P)], 0.0)))
             for lm in LAMS]
        out["signal_spread"] = float((max(v) - min(v)) / max(max(v), 1e-9))
        # [260814] Same quantity as the FoM -- band worst value. score_key points to this.
        out["signal_worst"] = float(min(v))
        out["signal_worst_spread"] = out["signal_spread"]
    elif kind == "shared":
        # The worse of the two polarizations is the score at that wavelength -- the spread is measured on the same basis.
        v = []
        for lm in LAMS:
            t = "%.1f" % (lm * 1e3)
            v.append(min(
                float(np.sqrt(max(out["%s_sub_p_TM_T_m1" % t]
                                  * out["%s_sub_p_TM_R_0" % t]
                                  * out["%s_sub_m_TM_T_1" % t], 0.0))),
                float(np.sqrt(max(out["%s_sub_n_TE_T_0" % t]
                                  * out["%s_sub_n_TE_R_1" % t]
                                  * out["%s_sub_m_TE_T_1" % t], 0.0)))))
        out["signal_spread"] = float((max(v) - min(v)) / max(max(v), 1e-9))
        # [260814] Same quantity as the FoM -- band worst value. score_key points to this.
        out["signal_worst"] = float(min(v))
        out["signal_worst_spread"] = out["signal_spread"]
    elif kind == "retro_te":
        # [260820] Same quantity as the FoM: band worst value of TE R(-2).
        v = [out["%.1f_sub_p_TE_R_m2" % (lm * 1e3)] for lm in LAMS]
        out["retro_TE_worst"] = float(min(v))
        out["retro_TE_worst_spread"] = float((max(v) - min(v)) / max(max(v), 1e-9))
        w = [out["%.1f_sub_p_TM_R_m2" % (lm * 1e3)] for lm in LAMS]
        out["retro_R_m2_worst"] = float(min(w))
    elif kind == "input_pi":
        # [260819] Square root of the product of the two polarizations T(+1), band worst value. Same quantity as the FoM.
        b = [float(np.sqrt(max(out["%.1f_air_TM_T_1" % (lm * 1e3)]
                               * out["%.1f_air_TE_T_1" % (lm * 1e3)], 0.0)))
             for lm in LAMS]
        out["bal_pi"] = float(min(b))
        out["bal_pi_spread"] = float((max(b) - min(b)) / max(max(b), 1e-9))
        # Workbook/driver compatibility -- score_key is bal_pi but the spread key name is unified.
        out["bal_worst"] = out["bal_pi"]
        out["bal_worst_spread"] = out["bal_pi_spread"]
    elif kind == "split_pi":
        # [260820] Same convention as extract_pi -- band worst value of the geometric mean of per-polarization
        # signal. If this is left out it leaks into else and dies reading the extraction-grating key.
        v = []
        for lm in LAMS:
            t = "%.1f" % (lm * 1e3)
            per = [float(np.sqrt(max(out["%s_sub_n_%s_T_0" % (t, p)]
                                     * out["%s_sub_n_%s_R_1" % (t, p)]
                                     * out["%s_sub_m_%s_T_1" % (t, p)], 0.0)))
                   for p in ("TM", "TE")]
            v.append(float(np.sqrt(max(per[0] * per[1], 0.0))))
        out["signal_spread"] = float((max(v) - min(v)) / max(max(v), 1e-9))
        out["signal_worst"] = float(min(v))
        out["signal_worst_spread"] = out["signal_spread"]
    elif kind == "extract_pi":
        # [260819] Geometric mean of per-polarization signal, band worst value. Same quantity as the FoM.
        # If this is left out it leaks into else and **TM-only signal** becomes the score,
        # and seed selection is done on a different basis from the FoM (same accident as 260814).
        v = []
        for lm in LAMS:
            t = "%.1f" % (lm * 1e3)
            per = [float(np.sqrt(max(out["%s_sub_p_%s_T_m1" % (t, p)]
                                     * out["%s_sub_p_%s_R_0" % (t, p)]
                                     * out["%s_sub_m_%s_T_1" % (t, p)], 0.0)))
                   for p in ("TM", "TE")]
            v.append(float(np.sqrt(max(per[0] * per[1], 0.0))))
        out["signal_spread"] = float((max(v) - min(v)) / max(max(v), 1e-9))
        out["signal_worst"] = float(min(v))
        out["signal_worst_spread"] = out["signal_spread"]
    else:
        v = [float(np.sqrt(max(out["%.1f_sub_p_TM_T_m1" % (lm * 1e3)]
                               * out["%.1f_sub_p_TM_R_0" % (lm * 1e3)]
                               * out["%.1f_sub_m_TM_T_1" % (lm * 1e3)], 0.0)))
             for lm in LAMS]
        out["signal_spread"] = float((max(v) - min(v)) / max(max(v), 1e-9))
        # [260814] Same quantity as the FoM -- band worst value. score_key points to this.
        out["signal_worst"] = float(min(v))
        out["signal_worst_spread"] = out["signal_spread"]
    return out


# -- [260903 T13] report() over 3 terminations --------------------------
# **Keys folded to the worst.** Only what the gate and seed selection look at is folded -- if keys are
# folded by guessing from their type, one misclassification silently gives a wrong "worst".
#
# `_FOLD_MIN` is for quantities where "larger is better", so worst = minimum.
#   signal_worst / bal_worst / bal_pi / retro_R_m2 are the **score_key** of `KINDS`
#   (after Main_rcwa_scd_c.py:244; retro changes to retro_R_m2 at :686).
#   The seed winner is chosen by `sc = rep[score_key]`, so if these are not folded
#   **the 3-termination worst is optimised but the winning seed is chosen at a single point** --
#   that is the reason for T13.
# [260923] **`fid_worst` / `fid_mean` are added.** fid is one of the four gates fixed as
#   "fid is the gate", but it was missing here, so it was reported **alone,
#   unfolded** -- the base dict is `dict(per[n_ref])`, so
#   the reported value was the fid of the single termination `n_ref`.
#   Measured 260923, 493 cells with per-termination fid keys: `folded_min_keys` mentions fid in
#   **0** cells, and the difference between the reported value and the true worst is **median 0.08749**
#   (p90 0.41063, max 0.93798). That is **27 times** the gate budget `1 - 0.996722 = 0.00328`,
#   and in **398 of 493 cells** the gap exceeds the whole budget.
#   The gate decision flips in only one cell (most miss fid anyway), but ranking and
#   margin were all wrong values.
#   `_FOLD_MIN` enters neither `_ckpt_sig` nor `_state_sig`, so
#   **this is a report-only change and does not discard the checkpoints of running jobs** (checked).
_FOLD_MIN = ("signal", "signal_worst", "bal_worst", "bal_pi", "retro_R_m2",
             "purity_worst", "purity_mean", "fid_worst", "fid_mean",
             "fidm_worst", "purity_m_worst")
# [261002] `purity_m_worst` = delay-arm (-theta) power purity, band worst
#   (report-only key, `_puritym` per wavelength). Folded as a MIN over
#   terminations like `purity_worst`. Adding it makes `folded_min_keys` list
#   it too -- that list is derived from this tuple, same as for `fidm_worst`.

_N_REF = 1.575          # the termination that the existing key names point to (backward-compatibility reference)


def _n_tag(n):
    """1.071 -> 'n1071'. No decimal point goes into a key -- a column name containing
    `.` is awkward to handle in TSV/reading scripts."""
    return "n%04d" % int(round(float(n) * 1000))


def report(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2, kind, ng=None,
           pol="TE"):
    """Calls `_report_one` **per termination** and folds the gate metrics to the worst.

    [260903 T13] Why it is needed
    ------------------------
    The optimisation looks at the **3-termination worst** via `_wmin(sig_terms)` in `fom_*`
    (`optimize` -> `neg()`), but seed-winner selection
    is done by `Main_rcwa_scd_c.py:765-797` with the `score_key` of the `report()` result, and
    `report()` had no termination loop. That is, **the termination axes of optimisation and selection
    were misaligned** -- optimised for the 3-termination worst, winner picked at the single point n=1.575.
    `infeasible_best` selection uses the same value.

    Return convention
    ---------
    - If `OUT_L3_NS` is empty it is **exactly identical to before** (single-termination path,
      `_report_one` is called once and returned as it is). Backward compatibility is not broken.
    - If not empty:
        * The base dict is that of the **`n_ref = 1.575`** termination -- the values the existing key names
          point to do not change, so past reading scripts do not break.
          (If `OUT_L3_NS` has no 1.575 the termination at the **median** of the list is used as
          the reference and its value is written in `n_ref`.)
        * The keys of `_FOLD_MIN` are overwritten with the **minimum over all terminations**.
        * echo does not overwrite the raw order keys but is given as **new keys** --
          `echo_worst_p2` (linear), `echo_worst_p2_db` (dB).
          It is the maximum over all termination x wavelength (= the worst). Overwriting the raw keys
          would break the meaning "this termination's value at this wavelength".
        * The full per-termination results are kept **in full** with the `n1071_...` / `n1575_...` / `n1967_...`
          prefixes. So no recomputation is needed
          when another metric is later folded to the worst again.
        * Meta keys: `report_schema` / `out_l3_ns_used` / `termination_fold` /
          `n_ref`. Analysis code uses these to tell old JSON (single n=1.575) apart.

    The cost is proportional to the number of terminations (3 terminations means 3 times the solves).
    """
    global OUT_L3_UM, OUT_L3_EPS, OUT_L3B_UM, OUT_L3B_EPS, GAP_UM
    ns = tuple(OUT_L3_NS)
    if not ns:
        return _report_one(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2, kind,
                           ng=ng, pol=pol)

    # [260914] `per` is **keyed by the n3 value** -- the `n_ref` selection below and the
    # `n{int(n*1000):04d}` prefix convention all hang on that value. In a 2-layer termination
    # different (n_b, t_b) can land on the same n3 and the keys can collide, so
    # the driver checks the uniqueness of the values **right after solving** and stops -- overwriting silently
    # here would produce a result in which a whole termination has vanished.
    if OUT_L3B_LIST and len(set(ns)) != len(ns):
        raise ValueError(
            "report: n3 values of the 2-layer terminations coincide (%s) -- the termination axis would "
            "vanish from the result" % ", ".join("%.6f" % v for v in ns))
    _orig_um, _orig_eps = OUT_L3_UM, OUT_L3_EPS
    _orig_bum, _orig_beps = OUT_L3B_UM, OUT_L3B_EPS
    _orig_gap = GAP_UM          # [260922] gap robustness fold -- see GAP_LIST
    per = {}
    try:
        for _i3, n3 in enumerate(ns):
            _set_termination(_i3, n3)
            # Reports with the same (termination, gap) pairs that the objective saw.
            # If the lengths mismatch (out_l3_ns_eval combination) it does not silently use only the front part
            # but stops here -- the driver also blocks it at submission time.
            if GAP_LIST:
                if len(GAP_LIST) != len(ns):
                    raise ValueError(
                        "report: GAP_LIST %d entries vs %d terminations -- "
                        "they are paired by index, so the lengths must be equal"
                        % (len(GAP_LIST), len(ns)))
                set_gap(GAP_LIST[_i3])
            per[n3] = _report_one(rho1, rho2, H1, H2, eps_b, eps_f1, eps_f2,
                                  kind, ng=ng, pol=pol)
    finally:
        OUT_L3_UM, OUT_L3_EPS = _orig_um, _orig_eps
        OUT_L3B_UM, OUT_L3B_EPS = _orig_bum, _orig_beps
        GAP_UM = _orig_gap

    # Reference termination -- 1.575 if present, otherwise the median of the list.
    n_ref = _N_REF if _N_REF in per else sorted(ns)[len(ns) // 2]
    out = dict(per[n_ref])

    # (1) score_key / purity as the minimum over all terminations
    for k in _FOLD_MIN:
        vals = [per[n][k] for n in ns
                if isinstance(per[n].get(k), float)]
        if len(vals) == len(ns) and vals:
            out[k] = float(min(vals))

    # (2) echo -- scans the raw order keys for the maximum over termination x wavelength. Keys are not assembled but
    #     **looked up**: `_report_one` prints the wavelength with "%.1f", so appending the LAMS value
    #     as it is mismatches 1552.41 -> 1552.4 (caught by the 260903 measurement).
    _sfx = "_sub_m_%s_R_2" % ("TM" if kind in ("extract", "input") else pol)
    _e = [float(v) for n in ns for k, v in per[n].items()
          if k.endswith(_sfx) and isinstance(v, (int, float))]
    if _e:
        _w = max(max(_e), 0.0)
        out["echo_worst_p2"] = _w
        out["echo_worst_p2_db"] = (10.0 * float(npa.log10(_w))
                                   if _w > 0 else float("-inf"))

    # (3) Keep the full per-termination results under prefixes
    for n3 in ns:
        tag = _n_tag(n3)
        for k, v in per[n3].items():
            out["%s_%s" % (tag, k)] = v
        # [260907] **Per-termination echo is kept too.** The above (2) is only a single maximum over all
        # termination x wavelength, so "is the thickness ranking invariant to the L3 index" could not be
        # asked on the echo axis -- efficiency/purity were confirmed with `n{XXXX}_signal_worst` /
        # `n{XXXX}_purity_worst` (Spearman +0.905/+0.860) but only echo had no
        # corresponding key. A thickness picked at one termination may fail to pass echo at another,
        # and there was no way to know.
        # `_sfx` is **reused** -- assembling keys from LAMS fails every time because of %.1f rounding
        # (1552.41 -> 1552.4) (see the comment of (2) above).
        # The existing `echo_worst_p2*` and the raw order keys are not touched.
        _e1 = [float(v) for k, v in per[n3].items()
               if k.endswith(_sfx) and isinstance(v, (int, float))]
        if _e1:
            _w1 = max(max(_e1), 0.0)
            out["%s_echo_worst_p2" % tag] = _w1
            out["%s_echo_worst_p2_db" % tag] = (
                10.0 * float(npa.log10(_w1)) if _w1 > 0 else float("-inf"))

    # [260906] This value is **not the number of terminations but a version label of the fold convention**. It is
    # printed as it is even for a 5-termination job. For the count/list see `out_l3_ns_used` below.
    # The string is deliberately not changed -- already stored JSON and reading scripts tell
    # generations apart by this value, so changing it would break comparisons.
    out["report_schema"] = "3term_worst_260903"
    out["out_l3_ns_used"] = [float(x) for x in ns]
    # [260922] Which gaps were folded -- same reason as `out_l3_ns_used`.
    # `gap_nm` is the value restored by finally, so it cannot tell whether folding was used.
    out["gap_nm_list_used"] = [float(x) for x in GAP_LIST]
    out["termination_fold"] = "worst"
    out["n_ref"] = float(n_ref)
    out["folded_min_keys"] = [k for k in _FOLD_MIN if k in out]
    return out


def seed_bank(n_random=3, rng=None, harmonics=(1, 2)):
    """Use only seeds that carry the Fourier component of the target order.

    A uniform density (rho=0.5) is not a grating: S = 0 for all m != 0 and
    eta = |S|^2, so d(eta)/dx = 0 -- a stationary point (confirmed by measurement 260812).
    """
    rng = rng or np.random.default_rng(0)
    u = (np.arange(NX) + 0.5) / NX
    a = 0.45 / len(harmonics)

    def multi(ph):
        return np.clip(0.5 + a * sum(np.cos(2 * np.pi * h * u + ph)
                                     for h in harmonics), 0.0, 1.0)

    out = [("multi", multi(0.0), multi(np.pi)),
           ("multi_q", multi(np.pi / 2), multi(-np.pi / 2)),
           ("blaze", np.clip(0.5 + 0.4 * np.sin(2 * np.pi * u)
                             + 0.2 * np.sin(4 * np.pi * u), 0, 1),
            np.clip(0.5 - 0.3 * np.sin(2 * np.pi * u + 1.0), 0, 1))]
    for k in range(max(0, n_random)):
        out.append(("rand%d" % k, rng.random(NX) * 0.4 + 0.3,
                    rng.random(NX) * 0.4 + 0.3))
    return out


# ---- Checkpoint (260831) ---------------------------------------------------
# Even if a job is cut midway (cancel / power outage / process kill) the optimisation does not restart from
# scratch. The Adam state is only (xy, m, v, t), so one save is 24 KB at NX=512.
#
# Caution: on Windows `proc.terminate()` is TerminateProcess and **cannot be intercepted.**
# So a "receive a termination signal and save then" approach cannot work in principle.
# Saving **in advance** periodically is the only way.
_CKPT_VER = 3      # [260905] 2: returns the best-binarised iterate. A checkpoint stored as 1
                   # is the last-iteration policy, so resuming would mix the two policies
                   # in one run. Raising the version discards it automatically.
                   # 3: [260908] includes `_state_sig()` (module global state) in the fingerprint.
_BEST_BIN_FROM = 0.9   # [260905] From this progress fraction the binarised FoM is measured each iteration
                       # and the best is tracked. 0.9 equals the start of the last stage of beta_schedule
                       # (beta 64) -- the coarsest, and the stretch where the result is
                       # actually decided. Only values are computed, so about
                       # +3 % per seed (measured 11.6 s x 30 times).



def _state_sig():
    """[260908] Job-settable module state, folded into the checkpoint
    fingerprint.

    `_ckpt_sig` used to digest only its own arguments, so everything set
    through a module setter -- `set_out_l3_list` (out_l3_ns),
    `set_out_l3`, `set_out_spacer`, `set_gap`, `set_period`,
    `set_substrate`, `set_min_feature`, and the driver's direct
    `LAMS = ...` -- was invisible to it. A checkpoint written under one
    termination basis could then be resumed under another with no warning.

    Looked up through `globals().get` on purpose: a name that disappears
    from the module must not raise inside the fingerprint and kill the run.
    """
    g = globals()
    names = ("LAMS", "N_SUB", "PERIOD", "NX", "NG", "NG_EVAL", "GAP_UM",
             "MIN_FEATURE_NM", "PX_NM", "SIN_TH", "THETA_SUB",
             "OUT_SPACER_UM", "OUT_SPACER_EPS", "OUT_L3_UM", "OUT_L3_EPS",
             "OUT_L3_NS", "OUT_L3_NS_NM")
    base = tuple((n, repr(g.get(n, "<missing>"))) for n in names)
    # [260922] `GAP_LIST` is added **only when non-empty**. If simply appended to `names`, the fingerprints of
    # every job that does not use gaps would change, and running jobs would discard the seed in progress entirely
    # on stop-resume. The switch cost is 0 only if the fingerprint is exactly
    # the same as before when it is empty.
    if g.get("GAP_LIST"):
        base = base + (("GAP_LIST", repr(g["GAP_LIST"])),)
    # [260924] 2-D carrier grid -- only when on, same rule as GAP_LIST.
    if g.get("GRID_2D") is not None:
        base = base + (("GRID_2D", repr(g["GRID_2D"])),
                       ("APP_2D", repr(g.get("APP_2D"))),
                       ("SYM_Y_2D", repr(g.get("SYM_Y_2D"))),
                       ("L2", repr(g.get("L2"))))
    # [261008 li_integrate] Li switches -- only when non-default (same rule as GAP_LIST / GRID_2D),
    # so a Li job can never resume a Laurent checkpoint and default fingerprints are unchanged.
    if g.get("RCWA_RULE", "laurent") != "laurent" or g.get("RCWA_FF", 0):
        base = base + (("RCWA_RULE", repr(g.get("RCWA_RULE"))),
                       ("RCWA_FF", repr(g.get("RCWA_FF"))))
    return repr(base)


def _ckpt_sig(kind, iters, lr, beta_schedule, nest, H1, H2,
              eps_b, eps_f1, eps_f2, kw, n):
    """Fingerprint of all values that determine the trajectory.

    If any one changes the checkpoint is **discarded.** Without this, after editing `iters` or the weights and
    resubmitting, it would resume from the old state and silently optimise the wrong thing.
    """
    raw = repr((_CKPT_VER, kind, int(iters), float(lr), tuple(beta_schedule),
                bool(nest), float(H1), float(H2),
                repr(eps_b), repr(eps_f1), repr(eps_f2),
                sorted((k, repr(v)) for k, v in kw.items()), int(n),
                _state_sig()))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _ckpt_save(path, sig, xy, m, v, t, hist):
    """Atomic save -- even if it dies midway, the previous checkpoint remains intact."""
    tmp = path + ".tmp"
    try:
        # Pass a file object. If a path string is given numpy appends .npz.
        with open(tmp, "wb") as fh:
            np.savez(fh, sig=np.array(sig), xy=xy, m=m, v=v,
                     t=np.array(int(t)), hist=np.array(hist[-8:], dtype=float))
        os.replace(tmp, path)
    except Exception as e:                     # a save failure must not kill the run
        print("  [ckpt] save failed (ignored): %r" % (e,), flush=True)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _ckpt_load(path, sig):
    """(xy, m, v, t, hist) if a checkpoint exists and its fingerprint matches, otherwise None."""
    if not path or not os.path.isfile(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["sig"]) != sig:
            print("  [ckpt] settings changed -- checkpoint discarded", flush=True)
            return None
        t = int(z["t"])
        if t < 1:
            return None
        return (z["xy"].copy(), z["m"].copy(), z["v"].copy(),
                t, [float(x) for x in z["hist"]])
    except Exception as e:
        print("  [ckpt] read failed (ignored): %r" % (e,), flush=True)
        return None


def ckpt_clear(path):
    """Deletes the checkpoint of a seed that finished normally (scratch data, so it is not passed on to a shared location)."""
    for p in (path, path + ".tmp"):
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def optimize(kind, x1_0, x2_0, H1, H2, eps_b, eps_f1, eps_f2, iters=300,
             lr=0.03,
             beta_schedule=((0, 8.0), (0.4, 16.0), (0.7, 32.0), (0.9, 64.0)),
             log_every=25, tag="", nest=False,
             ckpt_path=None, ckpt_every=25,
             top_k=1, top_k_gap=10, ng_schedule=None,
             w_lw=0.0, lw_target_nm=None, lw_beta_min=16.0, lw_c=None,
             lw_gate_nm=None, **kw):
    """nest = rho2 <= rho1 (L2 solid only on top of L1 solid).

    [260813 correction] The earlier comment said "if the materials of the two layers differ it is a deposition
    stack, so nest has no meaning", but that was **wrong.** Even with different materials the subtractive
    process of a deposition stack forces the same condition:
        deposit L1/L2 as blankets -> etch L2 with mask A (stop at L1)
        -> etch L1 with mask B. The remaining L2 becomes a hard mask and the
           L1 beneath it cannot be etched.  L1 solid = A | B^c >= A = L2 solid.
    So nest is a **support condition independent of the number of materials**. With it off, a shape appears in
    which L2 material floats over places where L1 is air, and that cannot be made (260813 audit: fill1=air
    97 of 129 cells were floating -- champion 1260 had 40 % of L2, width 110~123 nm).

    The only case where it holds even with nest off is when fill1 is solid -- the 'air'
    places of L1 are actually filled so L2 can sit on them (fill + planarisation process).
    """
    f = FOMS[kind]

    # [260923] nG CONTINUATION (`ng_schedule`). A tuple of (frac, nG) pairs,
    # resolved per iteration by t/iters exactly like `beta_schedule`. The
    # driver clamps every nG to the same alias ceiling as `nG` before it gets
    # here. None (the default) makes every FoM call below equivalent to
    # `f(..., **kw)` -- `_ngkw()` returns {} -- so the path, the numbers and
    # the checkpoint fingerprint are unchanged when the key is absent.
    # Only `ng` is forwarded to the FoM; `ng_schedule` itself is a separate
    # parameter and never enters `kw`, so it cannot reach the FoM signature
    # (TypeError) nor the result JSON's `**kw` merge in the driver.
    if ng_schedule:
        ng_schedule = tuple((float(fs), int(g)) for fs, g in ng_schedule)
        _fr = [fs for fs, _g in ng_schedule]
        if _fr[0] != 0.0 or any(b <= a for a, b in zip(_fr, _fr[1:])):
            raise ValueError("ng_schedule must start at frac 0 and increase "
                             "strictly: %r" % (ng_schedule,))
        if "ng" in kw:
            raise ValueError("ng_schedule and an explicit ng in kw conflict")
        _ng_final = ng_schedule[-1][1]
    else:
        ng_schedule = None
        _ng_final = None

    def _ngkw(t):
        if ng_schedule is None:
            return {}
        return {"ng": [g for fs, g in ng_schedule if t / float(iters) >= fs][-1]}

    # Final-stage scoring (best-binarised, top-K, last-iterate comparison) is
    # always judged at the LAST scheduled nG, so every candidate in that pool
    # is on one basis even if a switch falls inside the final stage.
    _ngkw_final = {} if ng_schedule is None else {"ng": _ng_final}

    def _pair(xy, beta):
        r1 = density(xy[:NX], beta)
        r2 = density(xy[NX:], beta)
        return (r1, r1 * r2) if nest else (r1, r2)

    def neg(xy, beta, ngkw):
        r1, r2 = _pair(xy, beta)
        return -f(r1, r2, H1, H2, eps_b, eps_f1, eps_f2, **kw, **ngkw)[0]

    vg = value_and_grad(neg)

    # [260926] Zhou-type geometric minimum-linewidth penalty (`w_lw`), 1-D
    # path only. Off by default (`w_lw = 0.0`): `_lw_on` stays False, `vg_lw`
    # is never built, and the loop below calls the unmodified `vg`/`neg`
    # exactly as before -- value and gradient are the same floating-point
    # operation sequence, not "+ 0*P".
    _lw_on = float(w_lw) > 0.0
    if _lw_on:
        if GRID_2D is not None:
            raise ValueError("w_lw: the 1-D Zhou geometric penalty does not "
                             "support carrier_2d = 1 (2-D in-plane carrier)")
        if nest:
            raise ValueError("w_lw: nest = 1 is not supported (the penalty "
                             "acts on x2's own filtered field, not the "
                             "nested product r1*r2)")
        _max_sched_beta = max(bv for _fs, bv in beta_schedule)
        if float(lw_beta_min) > _max_sched_beta:
            raise ValueError("w_lw: lw_beta_min %.3f exceeds the max "
                             "scheduled beta %.3f -- the penalty could "
                             "never turn on" % (lw_beta_min, _max_sched_beta))
        _lwg = lw_geometry(lw_target_nm, lw_c)
        _lw_gate_nm = (float(lw_gate_nm) if lw_gate_nm is not None
                       else float(lw_target_nm))

        def neg_lw(xy, beta, ngkw, lw_scale):
            r1, r2 = _pair(xy, beta)
            val = f(r1, r2, H1, H2, eps_b, eps_f1, eps_f2, **kw, **ngkw)[0]
            if lw_scale:
                _P = (lw_penalty(xy[:NX], beta, _lwg["eta_e"], _lwg["c"]) +
                      lw_penalty(xy[NX:], beta, _lwg["eta_e"], _lwg["c"]))
                val = val - lw_scale * _P
            return -val

        vg_lw = value_and_grad(neg_lw)

    xy = np.clip(np.concatenate([np.asarray(x1_0, float),
                                 np.asarray(x2_0, float)]), 0.0, 1.0)
    m = np.zeros_like(xy)
    v = np.zeros_like(xy)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    t_done = 0
    best = (None, None, None)   # (binarised fom, xy, it)
    best_ok = False             # [260926] feasibility of `best`, w_lw only
    # [260915] top-K retention, same binarised FoM as `best`, gap-separated
    # iterations. Empty and untouched when top_k <= 1 -- see the loop body
    # below for why this stays off by default.
    topk_list = []   # [(binarised fom, rb1, rb2, it), ...] len <= top_k
    meta = None
    # [260923] `ng_schedule` joins the fingerprint ONLY when set (same rule
    # as `w_r2` / `GAP_LIST`): absent -> `kw` is hashed exactly as before.
    _sig_kw = kw if ng_schedule is None else dict(kw, ng_schedule=ng_schedule)
    # [260926] `w_lw` joins the fingerprint ONLY when on (same rule).
    if _lw_on:
        _sig_kw = dict(_sig_kw, lw_penalty=(
            float(w_lw), int(_lwg["L_px"]), int(_lwg["R_px"]),
            float(_lwg["eta_e"]), float(_lwg["c"]), float(lw_beta_min),
            float(_lw_gate_nm)))
    sig = _ckpt_sig(kind, iters, lr, beta_schedule, nest, H1, H2,
                    eps_b, eps_f1, eps_f2, _sig_kw, xy.size)
    if ckpt_path:
        _st = _ckpt_load(ckpt_path, sig)
        if _st is not None:
            xy, m, v, t_done, hist = _st
            print("  [%s] checkpoint resumed -- from it %d/%d"
                  % (tag, t_done + 1, iters), flush=True)
    for t in range(t_done + 1, iters + 1):
        beta = [bv for fs, bv in beta_schedule if t / float(iters) >= fs][-1]
        # Adam (m, v, t) is NOT reset at an nG switch: xy does not depend on
        # nG, the beta switches above already change the FoM mid-run with no
        # reset, and a reset with t kept would divide by a stale bias
        # correction (1 - b2**t) -> a full-lr step on every variable.
        _ngk = _ngkw(t)
        if _ngk and (t == t_done + 1 or _ngk != _ngkw(t - 1)):
            print("  [%s] it %4d  nG -> %d" % (tag, t, _ngk["ng"]), flush=True)
        if _lw_on:
            _lw_scale = float(w_lw) if beta >= lw_beta_min else 0.0
            val, g = vg_lw(xy, beta, _ngk, _lw_scale)
        else:
            val, g = vg(xy, beta, _ngk)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        xy = np.clip(xy - lr * (m / (1 - b1 ** t)) /
                     (np.sqrt(v / (1 - b2 ** t)) + eps), 0.0, 1.0)
        hist.append(-float(val))
        # The last it is not saved -- the result is returned below anyway, and
        # the caller saves the seed and then deletes it with ckpt_clear().
        if ckpt_path and ckpt_every and t % ckpt_every == 0 and t < iters:
            _ckpt_save(ckpt_path, sig, xy, m, v, t, hist)
        if log_every and (t % log_every == 0 or t == 1):
            r1, r2 = _pair(xy, beta)
            _v, a1, a2 = f(r1, r2, H1, H2, eps_b, eps_f1, eps_f2, **kw, **_ngk)
            print("  [%s] it %4d  beta %5.1f  fom %.6f   main %.4f  aux %.4f"
                  % (tag, t, beta, -float(val), float(a1), float(a2)),
                  flush=True)
            if _lw_on:
                _pen = float(
                    lw_penalty(xy[:NX], beta, _lwg["eta_e"], _lwg["c"]) +
                    lw_penalty(xy[NX:], beta, _lwg["eta_e"], _lwg["c"]))
                print("  [%s] it %4d  lw_pen %.6e  gate %s (beta %.1f >= %.1f)"
                      % (tag, t, _pen,
                         "OPEN" if beta >= lw_beta_min else "CLOSED",
                         beta, lw_beta_min), flush=True)

        # [260905] BEST-BINARISED-ITERATE TRACKING.
        #
        # Two defects this closes, both measured on the 260905 thickness scan:
        #   (a) the loop returned the LAST iterate and tracked no best. `hist`
        #       holds FoM(xy_0..xy_iters-1) -- computed BEFORE each step -- so
        #       the returned xy_iters was never evaluated at all. Measured gap
        #       best-of-last-5 minus last: 13 of 24 cells nonzero, max 0.23106,
        #       which is 26.3 % of the whole FoM spread across thicknesses.
        #   (b) `hist` is the SOFT FoM at `_pair(xy, beta)`, while the artifact
        #       the caller keeps is `binarize()`d (Main_rcwa_scd_c.py:854) and
        #       linewidth/feasibility are judged on that hard version. So the
        #       recorded number never described the saved structure.
        #
        # Both are fixed by scoring the **binarised** density directly, which
        # also removes the need for a "is it binary enough" threshold -- every
        # candidate is binary by construction.
        #
        # Cost is why this runs only in the final beta stage: a value-only FoM
        # is ~11.6 s against ~34.7 s for value+grad (measured, seeds=2 / 5.78 h
        # / 300 it). Every iteration would add ~33 %; the last stage adds ~3 %.
        # The final stage is also the only place it matters -- that is where
        # beta is highest and the walk is roughest.
        if t / float(iters) >= _BEST_BIN_FROM:
            _rb1, _rb2 = _pair(xy, beta)
            _rb1, _rb2 = binarize(_rb1), binarize(_rb2)
            if nest:
                _rb2 = _rb2 * _rb1
            _fb = float(f(_rb1, _rb2, H1, H2, eps_b, eps_f1, eps_f2,
                          **kw, **_ngkw_final)[0])
            if _lw_on:
                # [260926] Feasibility-first selection: a candidate that
                # clears the linewidth gate always beats one that does not,
                # regardless of FoM. Off path (`_lw_on` False) keeps the
                # exact existing `_fb > best[0]` comparison below.
                _mf = min(m if m is not None else float("inf")
                          for m in (extract_ridges(_rb1)[1],
                                    extract_ridges(_rb2)[1]))
                _ok = _mf >= _lw_gate_nm
                if best[0] is None or (_ok, _fb) > (best_ok, best[0]):
                    best = (_fb, np.array(xy, copy=True), int(t))
                    best_ok = _ok
            elif best[0] is None or _fb > best[0]:
                best = (_fb, np.array(xy, copy=True), int(t))
            if top_k > 1:
                # Greedy online top-K on the same binarised FoM as `best`.
                # A candidate within top_k_gap iterations of a kept entry
                # replaces it only if better (keeps slots genuinely distinct);
                # otherwise it is appended if there is room, or swapped in for
                # the single worst kept entry if it beats it. Reuses the
                # already-binarised, nest-resolved _rb1/_rb2 above -- no
                # extra FoM evaluation.
                _near = [i for i, e in enumerate(topk_list)
                         if abs(e[3] - t) < top_k_gap]
                if _near:
                    if _fb > max(topk_list[i][0] for i in _near):
                        topk_list = [e for i, e in enumerate(topk_list)
                                     if i not in _near]
                        topk_list.append((_fb, _rb1.copy(), _rb2.copy(), int(t)))
                elif len(topk_list) < top_k:
                    topk_list.append((_fb, _rb1.copy(), _rb2.copy(), int(t)))
                elif _fb > min(e[0] for e in topk_list):
                    topk_list.remove(min(topk_list, key=lambda e: e[0]))
                    topk_list.append((_fb, _rb1.copy(), _rb2.copy(), int(t)))

    if best[0] is not None:
        _last = _pair(xy, beta_schedule[-1][1])
        _lb1, _lb2 = binarize(_last[0]), binarize(_last[1])
        if nest:
            _lb2 = _lb2 * _lb1
        _fl = float(f(_lb1, _lb2, H1, H2, eps_b, eps_f1, eps_f2,
                      **kw, **_ngkw_final)[0])
        print("  [%s] best-binarised it %d fom %.6f  vs last it %d fom %.6f"
              "  (gain %+.6f)" % (tag, best[2], best[0], iters, _fl,
                                  best[0] - _fl), flush=True)
        # [260908] This tracking is also kept in the result JSON so that what it actually did is visible.
        # Until now it went out only via print, so there was no way to know whether it fired, when, or the gain.
        meta = {"best_bin_it": int(best[2]),
                "best_bin_fom": float(best[0]),
                "last_bin_fom": float(_fl),
                "best_bin_gain": float(best[0] - _fl),
                "best_bin_from": float(_BEST_BIN_FROM)}
        if _lw_on:
            # [260926] additive: only when the penalty ran. Penalty at the
            # final xy, final (last-scheduled) beta -- computed BEFORE `xy`
            # is reassigned to `best[1]` below.
            meta["lw_best_ok"] = bool(best_ok)
            meta["lw_pen_last"] = float(
                lw_penalty(xy[:NX], beta_schedule[-1][1],
                          _lwg["eta_e"], _lwg["c"]) +
                lw_penalty(xy[NX:], beta_schedule[-1][1],
                          _lwg["eta_e"], _lwg["c"]))
        if ng_schedule is not None:
            # [260923] additive: only when a schedule ran.
            meta["best_bin_ng"] = int(_ng_final)
        if top_k > 1 and topk_list:
            # [260915] Additive only: at top_k <= 1 topk_list is always empty
            # (see the loop body above), so this block never runs and `meta`
            # is exactly the dict built above -- unchanged from today.
            topk_list.sort(key=lambda e: e[0], reverse=True)
            meta["top_k_its"] = [int(e[3]) for e in topk_list]
            meta["top_k_foms"] = [float(e[0]) for e in topk_list]
            meta["_top_k_densities"] = [(e[1], e[2]) for e in topk_list]
            print("  [%s] top_k=%d retained its %s foms %s"
                  % (tag, top_k, meta["top_k_its"],
                     ["%.6f" % v for v in meta["top_k_foms"]]), flush=True)
        xy = best[1]
    r1, r2 = _pair(xy, beta_schedule[-1][1])
    return xy, r1, r2, hist, meta
