"""Driver for scd_c_grating.py -- Batch C (folded SCD, fused silica), shared by the 3 grating kinds.

One job = one grating kind (kind) x one (H1, H2) pair. The three kinds use the same module and the
same geometry, so there is a single driver -- period, substrate refractive index, incidence angle and
the 3 wavelength points are shared; only kind selects the objective function.

    kind = input     air normal incidence -> substrate +1 order (TM). TE passes through in the 0 order.
    kind = retro     substrate +theta -> relative order -2 reflection (k -> -k, Littrow retro-reflection)
    kind = extract   substrate +theta partial extraction + specular pass, extract the returning -theta

ceiling
----
    input    eta(+1) <= 1      (if asymmetry is allowed; a left-right symmetric design is stuck at 0.5)
    retro    R(-2)   <= 1
    extract  sqrt(eta_+ * R_0 * eta_-) <= (4/9)^1.5 = 0.296296
             (discarded value sqrt(1/3*(2/3)^2) = 0.3849 -- it missed the reciprocity S = P S^T P)
             -- eta_+ + eta_- <= 1 (same air m=0 row) and eta_+ + R_0 <= 1 both apply
             at once. 'drop 50 % on the way in and 100 % on the way back' is impossible.

Reference measurements from other conditions (Lambda 533.8 nm, Si substrate): input 0.7405 (#977),
retro 0.85921 (#1160), extract 0.4533 (#666, but that metric did not multiply by R_0).
Fused silica + Lambda 1315.4 nm has different conditions, so these cannot be compared directly.

GMR
---
The grating layer (n >= 1.75) is higher than the substrate (1.444), so it forms a slab waveguide and
guided-mode resonances appear. Every FoM is computed as the geometric mean over 3 wavelengths
(1549.2 / 1550.0 / 1550.8 nm) to kill narrow resonant solutions. The `*_spread` in the result JSON is
that spread; above 0.05 it means the response is not flat over the 200 Gbaud band.

JOB KEYS
--------
  kind               input | retro | extract        (required)
  H1_nm, H2_nm       layer thickness (nm)                    (required)
  material1          bulk material of the substrate-side layer (default a-Si)
  material2          bulk material of the air-side layer (default a-Si)
  fill1, fill2       void material of each layer (default air)
  n_sub              substrate refractive index (default 1.444, fused silica)
  period_nm          grating period (default 1315.39 -- back-calculated from slab 300 um / 5 ps)
  nG / nG_eval       truncation for optimisation / for reporting (default 81 / 161)
  lams_nm            wavelength set in nm (comma separated). The current Track A
                     1546.00, 1547.60, 1550.00, 1552.41, 1554.02.
                     If unspecified, the 35 nm band discarded in 260816 is used, so
                     **always specify it**. (Added to the key list in 260904, where it was missing)
  lams_eval_nm       [260904] **evaluation-only** wavelength set. If given, only `report()` runs on this
                     band and the optimisation loop uses `lams_nm` -- 3-point
                     optimisation + 5-point judgement. If absent, behaviour is exactly as before.
  seeds, iters       number of seeds / Adam iterations
  lr                 Adam step (default 0.03)
  min_feature_nm     hard gate (default = M.MIN_FEATURE_NM = **70.0**, :505).
                     The required value is 50.0, so **always set it in the job** -- the seed
                     filter (:865) judges with the job's own value, so setting 56.81 makes
                     every 50-56.81 nm seed fail.
  filt_radius_nm     conic filter radius (default = min_feature_nm)
  float_tol_nm       floating-material tolerance in nm (default = PX_NM = 1 px).
                     If a non-top layer has fill=air, the material of the layer above must exist
                     only on top of the material of the layer below (same condition as nest). The
                     void of the top layer has nothing above to support, so it is always allowed.
  nest               rho2 <= rho1, single-film 2-step etch (default 0).
                     Meaningless if material1 != material2 (then it is a deposited stack).
  w_lw               [260926] Zhou-type minimum-linewidth penalty weight,
                     1-D path only (default 0.0 = off). Refused with
                     carrier_2d=1 or nest=1. See scd_c_grating.lw_geometry
                     / lw_penalty for the formula.
  lw_target_nm       target min solid/void width in nm, L1 and L2
                     (default = the job's min_feature_nm).
  lw_beta_min        beta gate: the penalty acts with full weight once the
                     schedule's beta >= this value (default 16.0).
  lw_c               advanced override of the indicator decay constant
                     (default 64*R_px**2). Leave unset.
  harmonics          seed Fourier components, e.g. "1,2" (default depends on kind)
  w_te               input: TE 0-order pass compensation (default 1.0)
                     extract: TE leakage penalty (default 0)
  w_echo             input: penalty on the path where the returning -theta bounces into the +2 order (default 0)
  w_back             [260819] input: **fibre back-coupling** penalty (default 0).
                     Path where the extraction grating R(-2) returns to x=0 and exits normally into
                     air (T(+1) of sub -theta TM). Measured at 10.1 % of the input power,
                     an order of magnitude above the immediate reflection (1.1 %).
  w_rgo              [260926] extract: delay-arm target term on
                     r0 = R(0)|+th, (r0 - rgo_target)^2, mirrors split's
                     w_r1/r1_target (default 0 = off).
  rgo_target         target for w_rgo above (default 0.5).
  w_leak             retro: penalty on T_-1 leaking into air (default 1.0)
  seed_npy           (2, 512) density warm start
  seed_rng           random seed (default 0)
  blas_threads       default 1 -- by design a 1-core job
  phase_target_deg   [260818] shift the final design sideways to match this phase.
                     Phase key per kind: input dphi_TM_TE / retro arg_retro_R_m2
                     / extract dphi_extract_arms / split dphi_split_arms.
                     **Efficiency is preserved exactly** (translation theorem, numerically verified
                     in 260818 -- variation 1e-12). Hence it is a design output, not an optimisation variable.
                     Recorded in the JSON: offset_px / offset_nm / phase_before_deg /
                     phase_after_deg / phase_err_deg / phase_slope_deg_per_nm.

OUTPUTS (in Output_folder/subfolder_name)
  scdc_{kind}_H{H1}_{H2}.json         best cell + full order table (3 wavelengths x polarization)
  density_scdc_{kind}_H{H1}_{H2}.npy  (2, 512) binarised density
  runs_log_scdc_{kind}_H{H1}_{H2}.tsv one line per seed
  Variable_result_information.txt
"""
import os
import sys


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
    print("[SCDC] usage: Main_rcwa_scd_c.py <job_file>")
    raise SystemExit(2)

JOB_PATH = sys.argv[1]
if not os.path.isfile(JOB_PATH):
    print("[SCDC] job file not found: %s" % JOB_PATH)
    raise SystemExit(2)

CFG = _read_job(JOB_PATH)


# [261008 li_integrate] OPT-IN Fourier-factorization keys. Defaults = the previous behaviour, byte for byte:
#   rcwa_rule  laurent (default) | liy | lix    Li 1997 slice-consistent rule (vendored grcwa li_fmm.py)
#   rcwa_ff    0 (default) | 1                  exact pixel form factor sinc(m/N)
# `rcwa_pol_epsinv` belongs to the Pol path, which this driver does not have: non-default is an ERROR.
def _li_parse(cfg):
    rule = (cfg.get("rcwa_rule", "").strip().lower() or "laurent")
    ffr = (cfg.get("rcwa_ff", "").strip() or "0")
    pinv = (cfg.get("rcwa_pol_epsinv", "").strip().lower() or "conv1")
    try:
        ff = float(ffr)
    except ValueError:
        ff = -1.0
    if rule not in ("laurent", "liy", "lix") or ff not in (0.0, 1.0) or pinv != "conv1":
        print("[SCDC] ERROR: rcwa_rule=%r (laurent|liy|lix), rcwa_ff=%r (0|1), rcwa_pol_epsinv=%r "
              "(this driver has no Pol path: only conv1)" % (rule, ffr, pinv), flush=True)
        raise SystemExit(2)
    return rule, int(ff)


_RCWA_RULE, _RCWA_FF = _li_parse(CFG)
_LI_ON = (_RCWA_RULE != "laurent") or bool(_RCWA_FF)

# BLAS threads must be set before importing numpy. On a 258x258 eigenproblem 6 threads gave only
# 1.41x (efficiency 24 %), so running many 1-core jobs is faster.
_bt = CFG.get("blas_threads", "").strip() or "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _bt

import json                                     # noqa: E402
import time                                     # noqa: E402
import numpy as np                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Per kind: (score key, auxiliary keys, ceiling, reference value measured under other conditions)
# [260814] Make score_key **the same quantity as the FoM**.
# Previously the FoM was min_lam sqrt(eta*T0_TE) while score_key was eta_TM_p1, so
# seed selection used a different criterion from the FoM -- it ignored the balance value raised by
# the optimiser and chose seeds with only high eta (measured: over 44 cells median 0 %, max 35.9 % loss).
# The ceiling is changed to match: the ceiling of the balance value sqrt(eta*T0_TE) stays 1.0,
# the ceiling of signal_worst is (4/9)^1.5 = 0.296296 (both are just the worst wavelength --
# 0.38490 is a model value discarded in 260828).
# [260816] Phase library -- per kind, the phase columns to put in runs_log.
# The purpose of running several seeds at the same parameters is **to secure phase freedom**
# (collect solutions with similar efficiency but different phase differences, then pick the ones that
# cancel the accumulated bias phase), so the per-seed phase must be a first-class column of the log.
PHASE_KEYS = {
    "input":   ("dphi_TM_TE", "arg_TM_p1", "arg_TE_0"),
    "retro":   ("arg_retro_R_m2",),
    # [260831 fix] For extract too, **bias_extract_arms** is the operating point.
    # In 260820 it was passed over as "extract has sum m = 2, so it happened to match", but only
    # **the slope** matched. The delay arm enters through R(0)@+th and `dphi_extract_arms`
    # misses that phase -- being m=0 it does not enter deg/nm, but it does enter the operating point.
    # Measured over 55 cells: median of the missing term |arg R(0)@+th| **100.7 deg** (max 176.5).
    # This is larger than the 90 deg step of the 4-zone I/Q, so trimming with this key collapses I/Q.
    "extract": ("bias_extract_arms", "dphi_extract_arms",
                "arg_extract_R0", "arg_eta_plus", "arg_eta_minus"),
    # [260820 fix] The arm bias is **dbeta_split** (sum m = 2).
    # The delay arm passes the splitting grating twice -- entry R(+1)@normal and exit T(+1)@-th.
    #   (0,lower) -R(+1)-> (832,upper) -R(-2)-> (832-832,lower) = the same grating it started from
    # `dphi_split_arms` is argT_1-argT_0, i.e. sum m = 1, so it is **half**.
    # Trimming with this key gives an actual bias of 2X deg when X deg is requested, and putting a
    # Lambda/4 step in the mask gives 180 deg instead of 90 deg, so the 4-zone
    # I/Q collapses. (The extraction grating has sum m = 2, so it happened to match.)
    "split":   ("bias_split_arms", "dphi_split_arms", "arg_split_T0",
                "arg_split_Text"),
    # [260819] For the single track the **inter-polarization** phase difference is judged. The extraction
    # grating's share must be cancelled by the return grating's Delta(arg R_-2) to be SOP independent.
    "retro_te":   ("arg_retro_R_m2_TE", "retro_R_m2_TE", "retro_R_m2"),
    "input_pi":   ("dphi_pi_in", "eta_TM_p1", "eta_TE_p1"),
    "extract_pi": ("dbeta_extract", "signal_TM", "signal_TE"),
    "split_pi":   ("dbeta_split", "signal_TM", "signal_TE"),
}


# [260823] **Auxiliary keys may be missing or None.**
# In 260822 a None guard was added to the bias phase arithmetic of `report`; when an order with floor
# amplitude occurs, the `dbeta_split` / `bias_split_arms` keys are **not inserted at all**.
# But this driver accesses `rep[k]` with brackets, so the TypeError only turned into a KeyError --
# the rand4 seed of 260823 18:43 `c62pi_we080_lr03` wasted 3650 s on
# `KeyError('dbeta_split')` (the job survived thanks to seed isolation).
# The echo suppression round aims to push an order to 0, so it keeps producing this case.
def _fk(d, k, fmt="%.4f", dash="-"):
    """Safe format for auxiliary keys. Return a dash if the key is missing or None."""
    v = d.get(k)
    if v is None:
        return dash
    try:
        return fmt % v
    except (TypeError, ValueError):
        return dash


# [260904] **Separation of optimisation band and evaluation band.** If the job key `lams_eval_nm` is
# present, only `report()` runs on that band, and the optimisation loop uses `lams_nm` unchanged.
#
# Why split: over 56 cells, the 3-point (1546.00/1550.00/1554.02) **evaluation** reproduced the 5-point
# within 0.010 %p efficiency / 0.009 dB echo, with rank correlation 1.0000 and 0 gate-decision
# flips. But that only verified the **evaluation**. If one **optimises** with 3 points, the
# optimiser can wreck the unmonitored 1547.6 / 1552.4 for free --
# using the unmonitored axes is the nature of an optimiser. report runs once per seed so
# its cost is negligible, and the very drift that the reduced band invites is caught here.
#
# If the key is absent, `_LAMS_EVAL` is None and `_report_eval` calls `M.report(...)`
# unchanged -- the behaviour of running jobs and past jobs does not change by a single character.
_LAMS_EVAL = None
_LAMS_EVAL_LOGGED = False

# [260909] **Optimisation/evaluation separation of the L3 multi-termination axis.** It carries the
# exact same mechanism as `lams_eval_nm` over to the termination axis -- if the job key
# `out_l3_ns_eval` is present, only `M.OUT_L3_NS` at the time `report()` is called is swapped for that
# list, and the `M.OUT_L3_NS` read by the optimisation loop (`fom_extract`/`fom_split`)
# stays `out_l3_ns`. `OUT_L3_NS`, like `LAMS`, is a module global, so
# saving/replacing/restoring it inside `_report_eval` is the only way to change the band
# (see `_report_eval` below).
# `OUT_L3_NS_NM` (shared thickness) is not touched -- the thickness is not different per
# termination but a value shared by the whole stack.
#
# If the key is absent, `_NS_EVAL` is None and `_report_eval` leaves `M.OUT_L3_NS`
# unchanged -- the behaviour of running jobs and past jobs does not change by a single character.
_NS_EVAL = None
_NS_EVAL_LOGGED = False

# [260923 "1-termination optimisation / 5-termination evaluation"] **Optimisation/evaluation separation of
# the 2-layer complex-reflection termination.** It carries the same mechanism as `out_l3_ns_eval` over to
# the `out_l3_r_list` (2-layer complex-reflection termination) axis -- if the job key `out_l3_r_list_eval`
# is present, `M.OUT_L3_NS` and `M.OUT_L3B_LIST` are swapped for that list only at the time `report()`
# is called, and the two globals read by the optimisation loop (`fom_extract`/`fom_split`)
# stay at the values resolved from `out_l3_r_list`. `out_l3_ns_eval` is a bare refractive-index list and
# cannot carry the 2-layer termination (including the inner layer), so this key is used instead -- hence the two
# cannot be used together (the following value already enforces that: when `out_l3_2layer` is on,
# `out_l3_ns_eval` is already an ERROR).
# `OUT_L3_NS_NM` (shared thickness) is not touched -- the evaluation list was resolved with the same
# `out_l3_nm`, so the thickness is already the same.
#
# If the key is absent, `_RLIST_EVAL_NS` is None and `_report_eval` leaves the two globals
# unchanged -- the behaviour of running jobs and past jobs does not change by a single character.
_RLIST_EVAL_NS = None
_RLIST_EVAL_PAIRS = None
_RLIST_EVAL_RAW = None
_RLIST_EVAL_LOGGED = False


def _report_eval(M, *a, **kw):
    """Call `report()` on the evaluation band.

    `M.LAMS` is a module global read by `_report_one` at call time
    (`scd_c_grating.py:1583` `for lam in LAMS`), so save/replace/restore is the only way to switch
    the band. **Restore is done in `finally`** -- if report throws an exception, the next seed
    inherits the evaluation band and silently starts optimising with 5 points,
    which is recorded nowhere in the log.

    [260909] `M.OUT_L3_NS` (L3 multi-termination list) is also swapped here for the same reason
    -- `report()` also reads the module global at call time
    (`ns = tuple(OUT_L3_NS)` in `report()` of `scd_c_grating.py`). The two axes are
    independent (one can switch on only `lams_eval_nm`, or only `out_l3_ns_eval`), so
    the early return happens **only when both are off**, and the swap and log work separately
    only when each axis is on. `OUT_L3_NS_NM` (shared thickness) is not
    touched.

    [260923] `out_l3_r_list_eval` (2-layer complex-reflection termination) is also swapped
    at the same place -- not only `M.OUT_L3_NS` but also `M.OUT_L3B_LIST` (inner layer (n_b,t_b))
    is a global that **completely** defines the 2-layer termination, so both must be swapped.
    When `out_l3_2layer` is on, `out_l3_ns_eval` is already an ERROR, so
    `_NS_EVAL` and `_RLIST_EVAL_NS` cannot apply at the same time -- the two switches are
    placed side by side but treated as mutually exclusive (even if both are on, the later does not
    win; the combination is blocked at the job parsing stage in the first place).
    """
    if not _LAMS_EVAL and not _NS_EVAL and not _RLIST_EVAL_NS:
        return M.report(*a, **kw)
    global _LAMS_EVAL_LOGGED, _NS_EVAL_LOGGED, _RLIST_EVAL_LOGGED
    _saved = M.LAMS
    _saved_ns = M.OUT_L3_NS
    _saved_blist = M.OUT_L3B_LIST
    # [260923] Put the swap and the confirmation print inside the try too -- even if the print raises,
    # finally restores the three globals, so the next seed's optimisation does not
    # inherit the evaluation list.
    try:
        if _LAMS_EVAL:
            M.LAMS = _LAMS_EVAL
        if _NS_EVAL:
            M.OUT_L3_NS = _NS_EVAL
        if _RLIST_EVAL_NS:
            M.OUT_L3_NS = _RLIST_EVAL_NS
            M.OUT_L3B_LIST = _RLIST_EVAL_PAIRS
        if _LAMS_EVAL and not _LAMS_EVAL_LOGGED:
            # Rule out the failure mode where the key is only parsed but never actually applied. Once only.
            _LAMS_EVAL_LOGGED = True
            print("[SCDC] evaluation band applied: report uses %s nm (%d points), "
                  "optimisation uses %s nm (%d points)"
                  % (", ".join("%.2f" % (v * 1e3) for v in _LAMS_EVAL),
                     len(_LAMS_EVAL),
                     ", ".join("%.2f" % (v * 1e3) for v in _saved),
                     len(_saved)),
                  flush=True)
        if _NS_EVAL and not _NS_EVAL_LOGGED:
            # Likewise once only -- confirm that the termination list swap actually applies.
            _NS_EVAL_LOGGED = True
            print("[SCDC] evaluation termination applied: report uses n = %s (%d points), "
                  "optimisation uses n = %s (%d points)"
                  % (", ".join("%.4f" % v for v in _NS_EVAL), len(_NS_EVAL),
                     ", ".join("%.4f" % v for v in _saved_ns), len(_saved_ns)),
                  flush=True)
        if _RLIST_EVAL_NS and not _RLIST_EVAL_LOGGED:
            # Likewise once only -- confirm that the 2-layer termination swap actually applies.
            _RLIST_EVAL_LOGGED = True
            print("[SCDC] evaluation r_list termination applied: report uses n3 = %s (%d points), "
                  "optimisation uses n3 = %s (%d points)"
                  % (", ".join("%.4f" % v for v in _RLIST_EVAL_NS),
                     len(_RLIST_EVAL_NS),
                     ", ".join("%.4f" % v for v in _saved_ns), len(_saved_ns)),
                  flush=True)
        return M.report(*a, **kw)
    finally:
        M.LAMS = _saved
        M.OUT_L3_NS = _saved_ns
        M.OUT_L3B_LIST = _saved_blist


def apply_phase_offset(dens, target_deg, kind, H1, H2, eps_b, e_f1, e_f2,
                       ng, M, np, pol="TE"):
    """Shift the final design sideways to match the **target phase**. Efficiency does not change.

    [260818] Numerical verification of the translation theorem: shifting the grating by d rotates only
    the phase of the m-th order amplitude by -mGd, and the magnitude is exactly preserved (measured
    efficiency variation 1e-12 = numerical noise). Confirmed on all four devices:

        input   dphi_TM_TE       |dm|=1   -0.2737 deg/nm
        return  arg_retro_R_m2   m=-2     +0.5474 deg/nm
        extract dphi_extract_arms |dm|=2  -0.5474 deg/nm
        split   dphi_split_arms  |dm|=1   -0.2737 deg/nm

    The sign is set by **the sign of the order** (phi_m -> phi_m - m*G*d). Only the return is + not
    because it is a reflection but because its order m=-2 is negative -- the translation theorem
    applies equally to reflection and transmission (confirmed by the 260818 adversarial cross-check).

    So the offset **cannot be an optimisation variable** (the efficiency gradient is identically 0).
    It is properly a **design output** that is shifted to the desired phase after optimisation ends.

    The sign of the slope differs per device and is entangled with the reflection/transmission convention,
    so it is not hard-coded in a table but **obtained by measuring one point at runtime** -- one
    report takes about 1 s, and sign mistakes are blocked at the source. The measuring point is NX/8
    (about 164 nm), so |phase change| <= 90 deg and there is no wrap-around ambiguity.
    """
    pkey = PHASE_KEYS.get(kind, (None,))[0]
    if pkey is None:
        return dens, None

    def phase_of(a):
        # [260904] Use the evaluation band. The phase trim is a **design output** after optimisation
        # ends, so it does not enter the gradient computation, and above all it must be on the
        # **same band** as the final report right after the trim -- otherwise the `phase_after_deg`
        # recorded here silently disagrees with the phase in the JSON.
        o = _report_eval(M, a[0], a[1], H1, H2, eps_b, e_f1, e_f2, kind, ng=ng,
                         pol=pol)
        return o.get(pkey)

    def wrap(d):
        return (d + 180.0) % 360.0 - 180.0

    p0 = phase_of(dens)
    if p0 is None:
        print("[SCDC] phase offset: cannot read %s, skipping." % pkey, flush=True)
        return dens, None
    k_probe = max(1, M.NX // 8)
    px_nm = M.PERIOD * 1e3 / M.NX
    p1 = phase_of(np.vstack([np.roll(dens[0], k_probe), np.roll(dens[1], k_probe)]))
    slope = wrap(p1 - p0) / (k_probe * px_nm)          # deg/nm
    if abs(slope) < 1e-6:
        print("[SCDC] phase offset: slope is 0, cannot control.", flush=True)
        return dens, None
    need_nm = wrap(target_deg - p0) / slope
    k = int(round(need_nm / px_nm)) % M.NX
    moved = np.vstack([np.roll(dens[0], k), np.roll(dens[1], k)])
    p2 = phase_of(moved)
    info = {"phase_key": pkey, "phase_target_deg": float(target_deg),
            "phase_before_deg": float(p0), "phase_after_deg": float(p2),
            "phase_err_deg": float(wrap(p2 - target_deg)),
            "offset_px": int(k), "offset_nm": float(k * px_nm),
            "phase_slope_deg_per_nm": float(slope),
            "period_nm": float(M.PERIOD * 1e3)}
    print("[SCDC] phase offset: %s %.2f -> %.2f deg (target %.2f, error %+.2f) "
          "| %d px = %.2f nm | slope %+.4f deg/nm"
          % (pkey, p0, p2, target_deg, info["phase_err_deg"], k,
             info["offset_nm"], slope), flush=True)
    return moved, info


def stem_of(kind, h1, h2):
    """Common stem of the density/log file names."""
    return "%s_H%.0f_%.0f" % (kind, h1, h2)


KINDS = {
    # [260831 correction] The ceiling is set to **(4/9)^1.5 = 0.296296**.
    # The earlier `sqrt(1/3*(2/3)^2) = 0.38490` is a **discarded model** using only energy conservation
    # (`scd_c_grating.py:886-898` states so), and the `signal_ceiling` of `report()` was
    # already fixed in 260828, so **the two ceilings in the same code
    # contradicted each other**. The ceiling is display-only (lines 882/910) so it does not
    # affect the optimisation, but the reported %-of-ceiling came out 1.299 times too low.

    "input":   ("bal_worst", ("eta_TM_p1", "T0_TE", "echo_R_p2"), 1.0,
                0.7405),
    # [260814] Promote TE to an auxiliary key. In polarization diversity the TE delay arm uses the same
    # return grating, but this was not visible in the ranking, so a TE 0.23 one (index 1530)
    # remained as the champion.
    "retro":   ("bal_worst", ("retro_R_m2", "retro_R_m2_TE",
                              "retro_gain_mismatch_dB", "leak_T_m1",
                              "pass_T0_n_worst"),
                1.0, 0.85921),
    # [260910] Put purity_worst / echo_worst_p2_db into the auxiliary keys. Both are
    # already computed by report() for every seed but were discarded because they were not in runs_log
    # (measured: of 1,613 runs_log_scdc_*.tsv files, 0 files had a purity column).
    # As a result a seed that passed all four gates was not visible because it was not the
    # winner (260909_tenb_H325_325_n1700 seed rand1).
    "extract": ("signal_worst", ("signal", "eta_plus", "R_go", "eta_minus",
                                 "purity_worst", "echo_worst_p2_db"),
                float((4.0 / 9.0) ** 1.5), None),
    # [260819] Polarization-independent single track. Drop polarization diversity and make the device
    # itself polarization independent -- with only one branch, branch gain matching / branch bias
    # matching / reference-arm skew all disappear. The return uses the existing retro with w_te=1
    # unchanged, so there is no new kind.
    # [260820] Dedicated to the coplanar-layout TE return.
    "retro_te": ("retro_TE_worst", ("retro_R_m2_TE", "retro_R_m2",
                                    "retro_gain_mismatch_dB", "leak_T_m1_TE",
                                    "pass_T0_n_worst", "backn_R0_TM",
                                    "slope_retro_TE", "slope_retro_TM"),
                 1.0, None),
    "input_pi": ("bal_pi", ("eta_TM_p1", "eta_TE_p1", "pi_mismatch_dB",
                            "R_both"), 1.0, None),
    "extract_pi": ("signal_worst", ("signal", "signal_TM", "signal_TE",
                                    "pi_mismatch_dB", "dbeta_extract"),
                   float((4.0 / 9.0) ** 1.5), None),
    # [260820] **Input/output grating** of the 2-grating TE geometry (bottom x=0). The top x=0 is a window
    # without a grating, so R(0) goes straight to the fibre without re-diffraction -> back_R0_worst
    # is always viewed together as an auxiliary key.
    "split_pi": ("signal_worst", ("signal", "signal_TM", "signal_TE",
                                  "pi_mismatch_dB", "back_R0_worst",
                                  "dbeta_split"),
                 float((4.0 / 9.0) ** 1.5), None),
    # [260813] TE-only lower split/extract. The ceiling is the same as extract -- T(0) and T_ext
    # both leave into air m=0 and share the same row (T0+Text<=1), and at the normal-incidence row
    # T0+R(+1)<=1, so the optimum T0=1/3, R1=Text=2/3 gives the same 1:2 structure.
    "split":   ("signal_worst", ("signal", "split_T0", "split_R1",
                                 "split_Text", "split_asym",
                                 "purity_worst", "echo_worst_p2_db"),
                float((4.0 / 9.0) ** 1.5), None),
    # [260814] Confirmed: lower output grating shares the TE/TM parameters. One structure serves as
    # extract (TM) and split (TE), and the score is the minimum of the two. gain_mismatch_dB is the
    # worst-SOP loss in polarization diversity, so it is always viewed together as an auxiliary key.
    "shared":  ("signal_worst", ("signal", "signal_extract", "signal_split",
                                 "gain_mismatch_dB", "split_asym"),
                float((4.0 / 9.0) ** 1.5), None),
}
# Seed Fourier components: they must carry the order change of the target transition to leave the stationary point.
DEFAULT_HARMONICS = {"input": (1, 2), "retro": (1, 2), "extract": (1, 2),
                     "split": (1, 2), "shared": (1, 2),
                     "input_pi": (1, 2), "extract_pi": (1, 2),
                     "retro_te": (1, 2), "split_pi": (1, 2)}


def _num(key, default):
    raw = CFG.get(key, "")
    try:
        return float(raw) if raw != "" else float(default)
    except ValueError:
        print("[SCDC] bad %s=%r -- using %s" % (key, raw, default))
        return float(default)


def main():
    global _LAMS_EVAL, _LAMS_EVAL_LOGGED, _NS_EVAL, _NS_EVAL_LOGGED
    global _RLIST_EVAL_NS, _RLIST_EVAL_PAIRS, _RLIST_EVAL_RAW, _RLIST_EVAL_LOGGED
    kind = CFG.get("kind", "").strip().lower()
    if kind not in KINDS:
        print("[SCDC] ERROR: kind = %r. Allowed values: %s"
              % (kind, ", ".join(sorted(KINDS))))
        return 2
    for k in ("H1_nm", "H2_nm"):
        if not CFG.get(k, "").strip():
            print("[SCDC] ERROR: %s is required (one job = one (H1, H2) pair)" % k)
            return 2
    H1_nm, H2_nm = float(CFG["H1_nm"]), float(CFG["H2_nm"])
    H1, H2 = H1_nm * 1e-3, H2_nm * 1e-3

    import scd_c_grating as M                    # noqa: E402
    if _LI_ON:
        # [261008 li_integrate] opt-in only; a stale scd_c_grating / grcwa fails loudly here
        if not hasattr(M, "set_rcwa"):
            print("[SCDC] ERROR: scd_c_grating has no set_rcwa -- refusing to run a Li job on the stock rule")
            return 2
        try:
            M.set_rcwa(_RCWA_RULE, _RCWA_FF)
        except (ValueError, RuntimeError) as _e_li:
            print("[SCDC] ERROR: %s" % _e_li)
            return 2
        print("[SCDC] rcwa_rule = %s  rcwa_ff = %d (opt-in Li factorization)" % (_RCWA_RULE, _RCWA_FF),
              flush=True)

    # [260924] OPT-IN 2-D in-plane carrier ("3-D sim"), `carrier_2d = 1`.
    # Absent / 0 -> nothing here runs and the 1-D path is byte-identical.
    # Grid = the atom driver's: nx = atoms_per_period * NXY, ny = NXY,
    # ly = PERIOD / atoms_per_period; densities are stored flattened (2, nx*ny)
    # and the best one is also written as rho1_2d.npy / rho2_2d.npy (nx, ny)
    # for `Main_rcwa_l3_atom2d_opt` (`l12_rho1_npy` / `l12_rho2_npy`).
    _c2d = int(_num("carrier_2d", 0))
    if _c2d:
        _g2 = M.set_grid_2d(int(_num("NXY", 96)),
                            int(_num("atoms_per_period", 3)),
                            bool(int(_num("sym_y", 1))))

    period_nm = _num("period_nm", M.PERIOD * 1e3)
    n_sub = _num("n_sub", M.N_SUB)
    M.set_substrate(n_sub)
    _p, th_deg = M.set_period(period_nm)
    if not np.isfinite(th_deg):
        print("[SCDC] ERROR: lambda/(Lambda*n_sub) = %.4f >= 1 -- the +1 order does not "
              "propagate in the substrate. Increase the period or raise n_sub."
              % (M.LAM / (M.PERIOD * M.N_SUB)))
        return 2
    # [261015 kx_hold] OPT-IN matched-angle evaluation (see scd_c_grating.set_kx_hold). Absent / 0 -> nothing
    # here runs and the job is byte-identical to the previous behaviour. An evaluation switch only: the
    # optimiser keeps the fixed 1550 nm THETA_SUB, so a job that optimises is refused.
    _kx_hold = int(_num("kx_hold", 0))
    if _kx_hold:
        if not hasattr(M, "set_kx_hold"):
            print("[SCDC] ERROR: scd_c_grating has no set_kx_hold -- refusing to run a kx_hold job at the "
                  "fixed angle", flush=True)
            return 2
        if not int(_num("eval_only", 0)):
            print("[SCDC] ERROR: kx_hold = 1 needs eval_only = 1", flush=True)
            return 2
        M.set_kx_hold(True)
        print("[SCDC] kx_hold = 1: report() solves each channel at sin(theta) = lam/(Lambda*n_sub) "
              "(kx = G held); THETA_SUB(1550) = %.5f deg is kept for everything else" % th_deg, flush=True)
    # [260814] Wavelength set override. The default is the C-band 5 points (broadband), but a control
    # run with a single 1550 optimisation must be possible -- how much efficiency a broadband FoM trades for
    # bandwidth is known only by comparing with the single-wavelength solution of the same landscape.
    # Values in nm, separated by commas or spaces. E.g. lams_nm = 1550
    _lams = CFG.get("lams_nm", "").strip()
    if _lams:
        try:
            M.LAMS = tuple(float(v) * 1e-3
                           for v in _lams.replace(",", " ").split())
        except ValueError:
            print("[SCDC] ERROR: bad lams_nm = %r" % _lams)
            return 2
        if not M.LAMS:
            print("[SCDC] ERROR: lams_nm is empty")
            return 2
    else:
        # [260824] **Prevent silent use of the discarded band.** The module default LAMS is
        # (1530, 1540, 1550, 1560, 1565) = the 35 nm band discarded in 260816.
        # 19 v2sh jobs of 260814 did not write this key and ran on that band,
        # and it was shown only in one log line, so nobody noticed. The current one is Track A
        # (1546.00~1554.02, ITU 100 GHz +-5 channels).
        print("[SCDC] ** WARNING: lams_nm not specified -- using the module default. "
              "The default is the 35 nm band discarded in 260816, so to use the current Track A "
              "put the following in the job: "
              "lams_nm = 1546.00, 1547.60, 1550.00, 1552.41, 1554.02 **")
    # [260904] Evaluation-only band (see the `_report_eval` comment). If absent, the old behaviour.
    # Initialise explicitly for every job -- if main() is called twice in one process,
    # the band of the previous job must not be inherited.
    _LAMS_EVAL = None
    _LAMS_EVAL_LOGGED = False
    _lev = CFG.get("lams_eval_nm", "").strip()
    if _lev:
        try:
            _LAMS_EVAL = tuple(float(v) * 1e-3
                               for v in _lev.replace(",", " ").split())
        except ValueError:
            print("[SCDC] ERROR: bad lams_eval_nm = %r" % _lev)
            return 2
        if not _LAMS_EVAL:
            print("[SCDC] ERROR: lams_eval_nm is empty")
            return 2
    print("[SCDC] LAMS = %s nm  (%d points)"
          % (", ".join("%.1f" % (v * 1e3) for v in M.LAMS), len(M.LAMS)))
    if _LAMS_EVAL:
        print("[SCDC] LAMS_EVAL = %s nm  (%d points)  -- optimisation uses the %d points above, "
              "report only this band"
              % (", ".join("%.2f" % (v * 1e3) for v in _LAMS_EVAL),
                 len(_LAMS_EVAL), len(M.LAMS)))
    th_c = float(np.degrees(np.arcsin(1.0 / M.N_SUB)))
    print("[SCDC] kind = %s" % kind)
    print("[SCDC] Lambda = %.2f nm   NX = %d (%.3f nm/px)   n_sub = %.4f"
          % (M.PERIOD * 1e3, M.NX, M.PX_NM, M.N_SUB))
    if _c2d:
        print("[SCDC] carrier_2d = 1 -- grid %d x %d, ly = %.3f nm (PERIOD/%d), "
              "pixel %.3f nm, y-mirror %s"
              % (M.GRID_2D[0], M.GRID_2D[1], M.L2[1] * 1e3, M.APP_2D, M.PX_NM,
                 "ON" if M.SYM_Y_2D else "OFF"), flush=True)
    print("[SCDC] theta_sub = %.4f deg   critical angle %.3f deg   margin %+.2f deg"
          % (th_deg, th_c, th_deg - th_c))
    if th_deg <= th_c:
        # [260814] If theta <= critical angle, the air-side propagating orders open. The default is to
        # reject (light leaks through the face without a grating), but there are cases where this region
        # is measured **on purpose**: with 2 or more air channels, forward and return extraction sit in
        # different rows of the S matrix, so the eta+ + eta- <= 1 constraint is lifted, and 100 % return
        # extraction and a 1:1 split (ceiling 0.385 -> 0.5) become possible in principle.
        # The price is more leakage channels, so whether the return grating withstands it is the criterion.
        # Only jobs with allow_air_orders = 1 open this region.
        if not int(_num("allow_air_orders", 0)):
            print("[SCDC] ERROR: theta_sub is at or below the critical angle -- light leaks out "
                  "through the face without a grating. Reduce the period or specify "
                  "allow_air_orders=1.")
            return 2
        print("[SCDC] WARN: theta_sub %.3f <= critical angle %.3f -- proceeding with the air orders "
              "open (allow_air_orders=1)." % (th_deg, th_c))
    print("[SCDC] propagating orders  substrate %s   air %s"
          % (M.orders("sub"), M.orders("air")))
    print("[SCDC] wavelengths %s nm (3-point geometric mean)"
          % ", ".join("%.1f" % (v * 1e3) for v in M.LAMS))

    def _mat(key, default):
        v = CFG.get(key, "").strip() or default
        if v in M.MATERIALS:
            return M.eps_of(v), v
        try:
            return M.eps_of(float(v)), "n=%s" % v
        except ValueError:
            print("[SCDC] ERROR: %s = %r unknown. Known: %s"
                  % (key, v, ", ".join(sorted(M.MATERIALS))))
            return None, None

    e_b1, m1_name = _mat("material1", "a-Si")
    e_b2, m2_name = _mat("material2", "a-Si")
    e_f1, f1_name = _mat("fill1", "air")
    e_f2, f2_name = _mat("fill2", "air")
    if None in (m1_name, m2_name, f1_name, f2_name):
        return 2
    eps_b = (e_b1, e_b2)

    # grcwa reads sfft[m_i - m_j] from the NX-point FFT, so if nG exceeds NX//2+1 it silently
    # wraps around into aliased coefficients -- it does not become more accurate but wrong
    # in a different way (the energy sum is still 1, so it goes unnoticed).
    ng_max = M.NX // 2 + 1
    ng = min(int(_num("nG", M.NG)), ng_max)
    ng_eval = min(int(_num("nG_eval", M.NG_EVAL)), ng_max)
    if int(_num("nG", M.NG)) > ng_max or int(_num("nG_eval", M.NG_EVAL)) > ng_max:
        print("[SCDC] nG exceeded the alias limit %d of NX = %d and was lowered."
              % (M.NX, ng_max))
    M.NG, M.NG_EVAL = max(3, ng), max(3, ng_eval)

    # [260923] nG CONTINUATION. `ng_schedule = frac:nG, frac:nG, ...`
    # (e.g. "0:41, 0.4:61, 0.7:81"), resolved per iteration by t/iters exactly
    # like beta_schedule. Each nG gets the SAME clamp as `nG` above. Absent ->
    # `_ngs` is None, nothing is passed to optimize(), and M.NG / the
    # checkpoint fingerprint / the JSON are exactly as before.
    # When set, the job's `nG` is not used for optimisation: M.NG is set to
    # the LAST scheduled nG, so the JSON `nG` field names the basis the final
    # structure was judged at. The report stays at nG_eval.
    _ngs = None
    _ngs_raw = CFG.get("ng_schedule", "").strip()
    if _ngs_raw:
        try:
            _ngs = []
            for _tok in _ngs_raw.replace(";", ",").split(","):
                _tok = _tok.strip()
                if not _tok:
                    continue
                _fs, _g = _tok.split(":")
                _ngs.append((float(_fs), int(float(_g))))
        except ValueError:
            print("[SCDC] ERROR: bad ng_schedule = %r (want 'frac:nG, ...')"
                  % _ngs_raw)
            return 2
        _fr = [fs for fs, _g in _ngs]
        if (not _ngs or _fr[0] != 0.0 or _fr[-1] >= 1.0
                or any(b <= a for a, b in zip(_fr, _fr[1:]))):
            print("[SCDC] ERROR: ng_schedule = %r -- fractions must start at 0, "
                  "increase strictly and stay < 1" % _ngs_raw)
            return 2
        if any(g > ng_max for _fs, g in _ngs):
            print("[SCDC] nG of ng_schedule exceeded the alias limit %d and was lowered."
                  % ng_max)
        _ngs = tuple((fs, max(3, min(g, ng_max))) for fs, g in _ngs)
        M.NG = _ngs[-1][1]

    n_seeds = max(1, int(_num("seeds", 4)))
    # [260907] **Score the stored structure unchanged.** If 1, neither warm relaxation nor optimisation
    # is done; `seed_npy` is read and only `report()` is run.
    #
    # Why needed: the corpus-best TM cell (260903_a3_00, ceiling ratio 87.21 %) was produced before the
    # fold-record machinery was introduced, so `out_l3_ns_used` / `report_schema` /
    # `echo_worst_p2_db` are all missing -- i.e. it fits none of the four criteria.
    # But re-scoring was impossible: if `seed_npy` is present the relaxation (0.15 + 0.70*rho)
    # is applied **unconditionally**, and `iters = 0` is a ZeroDivisionError in `scd_c_grating.py`,
    # so the minimum is 1, and even that 1 takes one Adam step. So the 5 runs 260907_rf_* became
    # **warm re-optimisations**, not re-scoring.
    #
    # If 0 or absent, the behaviour does not differ from before by a single character.
    eval_only = int(_num("eval_only", 0))
    iters = max(1, int(_num("iters", 300)))
    lr = _num("lr", 0.03)
    # [260831] Checkpoint period. 0 turns it off. At 24 KB per write, 25 is practically free,
    # and the computation lost on interruption is limited to at most 25 it (about 4 min).
    ckpt_every = max(0, int(_num("ckpt_every", 25)))
    # [260923] Leave the per-seed FoM trajectory (parallel decomposition -- when one seed
    # becomes one job there was no way to inspect the trajectory outside the job file). If 0/absent,
    # it does not differ from before by a single character.
    save_hist = bool(int(_num("save_hist", 0)))
    # [260915] top-K retained iterates from optimize()'s final beta stage
    # (see scd_c_grating.py optimize()). Default 1 reproduces today's
    # single-best-iterate behaviour exactly -- no new files, no new JSON
    # keys, nothing threaded into M.optimize() beyond the parameter itself
    # (which defaults to this same 1).
    top_k = max(1, int(_num("top_k", 1)))
    min_feat = _num("min_feature_nm", M.MIN_FEATURE_NM)
    float_tol = _num("float_tol_nm", M.PX_NM)
    nest = bool(int(_num("nest", 0)))
    # [260926] w_lw -- Zhou-type geometric minimum-linewidth penalty, 1-D
    # path only. Refused up front, before any termination solve or seed
    # optimisation. Absent/0 leaves every later branch untouched.
    _wlw = _num("w_lw", 0.0)
    if _wlw < 0:
        print("[SCDC] ERROR: w_lw = %.4g -- must be >= 0" % _wlw)
        return 2
    if _wlw > 0 and _c2d:
        print("[SCDC] ERROR: w_lw (Zhou 1-D geometric linewidth penalty) is "
              "1-D only; carrier_2d = 1 is not supported")
        return 2
    if _wlw > 0 and nest:
        print("[SCDC] ERROR: w_lw is not supported with nest = 1 (the "
              "penalty acts on x2's own filtered field, not the nested "
              "product r1*r2)")
        return 2
    if _wlw > 0 and _num("lw_beta_min", 16.0) > 64.0:
        print("[SCDC] ERROR: lw_beta_min = %.3f exceeds 64.0 -- the default "
              "beta_schedule never reaches it, so the penalty could never "
              "turn on" % _num("lw_beta_min", 16.0))
        return 2
    if _wlw > 0 and "lw_start_frac" in CFG:
        print("[SCDC] WARN: lw_start_frac is present but this driver's "
              "w_lw uses lw_beta_min (a beta gate), not lw_start_frac's "
              "progress ramp -- it is ignored here.", flush=True)
    if f1_name == "air" and not nest:
        print("[SCDC] WARNING: fill1 = air but nest = 0. If the L2 material floats over the "
              "void of L1 it cannot be made by deposition and fails the gate "
              "(set nest = 1 or change fill1 to a solid).", flush=True)
    if nest and m1_name != m2_name:
        print("[SCDC] WARNING: nest=1 but material1(%s) != material2(%s). "
              "This is a deposited stack, not a single-film 2-step etch, so nest is "
              "meaningless process-wise -- check that this is intended." % (m1_name, m2_name))

    # [260812] Interlayer spacer. If L1 is filled by deposition, that film also covers the top and a
    # uniform plate forms between the two pattern layers. If fill1 = air there is no deposition, so no gap either.
    gap_nm = _num("gap_nm", 0.0)
    if gap_nm > 0 and f1_name == "air":
        print("[SCDC] WARNING: gap_nm = %.0f but fill1 = air. The spacer is the deposited "
              "film that filled L1 and covered the top, so it does not exist "
              "without filling -- check that this is intended." % gap_nm)
    M.set_gap(gap_nm)
    # [260827] **Exit-side spacer** -- where the L3 metaatom will sit. Different from `gap_nm` (between
    # layers). The lower grating has so far been optimised assuming it radiates into air, but the actual
    # stack is `L1/L2 -> SiO2 -> L3 -> air`. Measured (c74 TM): air 0.29579 ->
    # SiO2 200 nm 0.28866 (-0.106 dB) -> 400 nm 0.22854 (-1.120 dB).
    # Turning this on makes the parasitic orders m=0,-2 **propagating waves** inside the spacer, so the
    # FoM penalises them automatically (for air exit they were evanescent and went uncaught). As a result
    # L1/L2 push the light into the design channel m=-1, which in turn makes the illumination seen by L3
    # a clean normal plane wave and makes the metaatom design easier.
    _osp = _num("out_spacer_nm", 0.0)
    # [260830] `_osn` is also recorded in the result JSON, so define it **outside the block**.
    # If defined only inside, a job with `out_spacer_nm = 0` gets a NameError at the recording point.
    _osn = _num("n_out_spacer", 1.4440)
    if _osp > 0:
        M.set_out_spacer(_osp, _osn)
        print("[SCDC] exit-side spacer %.0f nm (n=%.4f) -- L3 position" % (_osp, _osn),
              flush=True)
    # [260901] The L1/L2 optimisation has so far terminated the exit side with air,
    # but in the actual stack the L3 metalens sits on the spacer. `out_l3_nm` is set to the
    # physical H3 (e.g. 825), and `n_out_l3` to the effective refractive index (measured mean 1.575,
    # library range 1.071~1.967).
    _ol3 = _num("out_l3_nm", 0.0)
    _ol3n = _num("n_out_l3", 1.575)
    if _ol3 > 0:
        M.set_out_l3(_ol3, _ol3n)
        print("[SCDC] exit-side L3 equivalent layer %.0f nm (n=%.4f)" % (_ol3, _ol3n),
              flush=True)
    # [260901] If `out_l3_ns` is given, those refractive indices are evaluated **simultaneously** and
    # the worst value is optimised (the gate is defined by the worst condition, so the objective must be too).
    # Measured: changing n moves signal by up to 1.4 dB and echo by up to 8.3 dB, and
    # the n giving the worst differs per metric (TM signal at the interior value 1.575, TE echo at
    # the endpoint 1.967). If empty, the single-termination path above is used unchanged -- no extra solve.
    # The cost is proportional to the number of terminations.
    # [260914] **Specify the termination by complex reflection** -- step 1 (b).
    # `out_l3_r_list = |r|:arg_deg; |r|:arg_deg; ...`
    # What the library measures is the complex reflection of the atom (`absR`, `argR_deg`), so instead of
    # converting it to an "effective refractive index", discarding the phase and recovering it, it is
    # applied **as is**. The thickness is given by `out_l3_nm` and only the refractive index is solved for.
    # [260914] **Clear the 2-layer termination state explicitly for every job.** If main() is called twice
    # in one process and the inner layer of the previous job is inherited, the stack silently differs
    # (same reason as the `_NS_EVAL` re-initialisation). Only the job that turns it on below sets it again.
    M.set_out_l3b_list(())
    M.set_out_l3b(0.0)
    # [260922] The gap-robustness fold is also cleared for every job -- only a job with
    # `gap_nm_list` sets it up again below (after out_l3_ns is fixed).
    M.set_gap_list(())
    # [260923] The r_list evaluation-only termination is also cleared for every job -- same reason as
    # the `_NS_EVAL` / `_RLIST_EVAL_*` re-initialisation (if main() is called twice in one
    # process, the state of the previous job must not be inherited).
    _RLIST_EVAL_NS = None
    _RLIST_EVAL_PAIRS = None
    _RLIST_EVAL_RAW = None
    _RLIST_EVAL_LOGGED = False
    # [260923] Track 2 (DEMAND) -- achieved `arg rho` per termination, only
    # populated when a token actually carries the 3rd field. `None` entries
    # (2-field tokens) are dropped before anything is stored in `_band`, so
    # a job with no 3rd field anywhere produces byte-identical JSON to
    # before this change.
    _ARGRHO_OPT = None
    _ARGRHO_EVAL = None
    # [260923 review fix] Track 2 needs the LANE's polarisation, not a
    # hard-coded TM. `extract` never reads `pol` (always TM); `split`
    # defaults to TE but honours the job's own `pol` key -- same default
    # as the `kind == "split"` block further down, which validates/reads
    # `pol` again for the FoM weights; this is evaluated here because
    # `_solve_r_terms` runs earlier in `main()`.
    if kind == "split":
        _rho_pol = (CFG.get("pol", "") or "TE").strip().upper()
        if _rho_pol not in ("TE", "TM"):
            _rho_pol = "TE"
    else:
        _rho_pol = "TM"

    def _solve_r_terms(_raw, _t3v, _twov, _nminv, _bandv, _luv,
                        label="out_l3_r_list", pol=_rho_pol):
        """Resolve the `out_l3_r_list` tokens and return `(ns, pairs, bad, argrho)`.

        [260923] Split so that `out_l3_r_list` and `out_l3_r_list_eval` take the **same code
        path** (same solver, same argument set) -- the code guarantees that
        if thickness / n_min / band / design wavelength are the same, the two lists give the same
        solution. The result for the optimisation list is bit-identical to before the refactor
        (only the operation order / arguments were moved unchanged).

        [260923] **If the token carries a 3rd field** (`|r|:arg_deg:argrho_deg`)
        the trapped +-THETA_SUB parasitic's reflection (`arg rho`, seen
        from the spacer side, referenced at the L3 FACE -- see
        `_rho_trapped`'s 260923 review-fix docstring) is targeted too.
        `r00` (normal incidence, design channel) fixes only 2 of the
        2-layer termination's 3 unknowns, so the 1 remaining degree of
        freedom goes to `arg rho`. `pol` is the caller's lane (`_rho_pol`,
        computed above: `extract` -> TM always, `split` -> TE unless the
        job's own `pol` key says TM).

        [260923 review fix] **The earlier claim here -- "arg rho reaches a
        353.922 deg span, essentially the whole circle" -- was measured on
        the UNFIXED code** (patch-referenced `GetAmplitudes(0, 0.0)`,
        hard-coded TM, fixed `THETA_SUB`) and is superseded. Re-measured on
        the FIXED (face-referenced) code, on the solver's own 1 nm `t_b`
        grid, H3=875 nm (`argrho_termination_face.diff`'s `report.md` has
        the full table; headline numbers, largest UNREACHABLE gap and bins
        filled of 24):
            r00=0.16@172deg  reachable span 234.24 deg (gap 125.76 deg,
                              152.35 to 278.10)   17/24 bins filled
            r00=0.20@170deg  reachable span 225.06 deg (gap 134.94 deg,
                              166.93 to 301.87)   16/24 bins filled
            r00=0.03@172deg  reachable span 342.71 deg (gap  17.29 deg,
                              244.94 to 262.23)   24/24 bins filled
        `|rho|=1` (unitarity) still holds everywhere sampled (max
        `||rho|-1|` 1.1e-13 across all three). The reachable span SHRINKS
        as `|r00|` grows -- do not assume "near the full circle" at any
        contract without re-measuring it.

        If the 3rd field is **absent**, it is completely identical to the existing 2-field path -- no extra
        solve and no JSON change (next needle: rerunning an existing job file is
        byte-identical with eval_only=1).

        If token parsing fails, `(None, None, None, None)`.
        """
        _ns_l, _pairs_l, _bad_l = [], [], 0
        _argrho_l = []
        print("[SCDC]   %-16s %-9s %-8s %-9s %-10s %s"
              % ("target r", "n_b", "t_b", "n3", "residual", "verdict"), flush=True)
        for _tok in _raw.replace(",", ";").split(";"):
            _tok = _tok.strip()
            if not _tok:
                continue
            _fields = _tok.split(":")
            if len(_fields) not in (2, 3):
                print("[SCDC] ERROR: bad %s token %r (expected |r|:arg_deg or "
                      "|r|:arg_deg:argrho_deg)" % (label, _tok), flush=True)
                return None, None, None, None
            try:
                _rt = float(_fields[0]) * np.exp(1j * np.radians(float(_fields[1])))
                _argrho_target = (float(_fields[2]) if len(_fields) == 3
                                  else None)
            except Exception:
                print("[SCDC] ERROR: bad %s token %r" % (label, _tok),
                      flush=True)
                return None, None, None, None
            if _argrho_target is not None and not _twov:
                print("[SCDC] ERROR: the 3rd field (argrho_deg) of %s is supported only "
                      "with out_l3_2layer = 1 (a 1-layer has a single degree of freedom, "
                      "n3, so it can aim at only one r00): %r"
                      % (label, _tok), flush=True)
                return None, None, None, None
            if _twov:
                # [260914] If `out_l3_band = 1`, minimise the worst residual over all of
                # `lams_nm`. Off by default -- if not turned on it is
                # exactly the same as before.
                if _argrho_target is None:
                    _nb, _tb, _n3, _res = M.solve_2layer_for_r(
                        _rt, _t3v, _luv, n_min=_nminv,
                        lams_um=(tuple(M.LAMS) if _bandv else None))
                    _argrho_l.append(None)
                else:
                    # [260923] band (`out_l3_band=1`) is passed to the argrho path as is, like
                    # `solve_2layer_for_r` -- the r00 residual computation is
                    # shared, and arg rho itself is always measured at the single design
                    # wavelength (`_luv`) (the trap channel is not subject to the band fold).
                    _nb, _tb, _n3, _res, _argach, _argres = \
                        M.solve_2layer_for_r_rho(
                            _rt, _argrho_target, _t3v, _luv, n_min=_nminv,
                            lams_um=(tuple(M.LAMS) if _bandv else None),
                            pol=pol)
                    if _nb is None:
                        print("[SCDC] ERROR: %s no r00 solution at this thickness can aim "
                              "at the argrho target: %r" % (label, _tok),
                              flush=True)
                        return None, None, None, None
                    _argrho_l.append(dict(target_deg=_argrho_target,
                                          achieved_deg=_argach,
                                          residual_deg=_argres,
                                          pol=pol,
                                          ref="L3 face"))
                _pairs_l.append((_nb, _tb))
            else:
                _nb, _tb = 0.0, 0.0
                _n3, _res = M.solve_slab_for_r(_rt, _t3v, _luv)
                _argrho_l.append(None)
            _argrho_ok = (_argrho_target is None
                         or _argrho_l[-1]["residual_deg"] <= 2.0)
            _ok = (_res <= 0.005) and _argrho_ok
            _bad_l += 0 if _ok else 1
            _tag = ("ok" if _ok else
                   ("** unreachable at this thickness **" if _res > 0.005 else
                    "** argrho outside 2 deg (r00 ok) **"))
            print("[SCDC]   %-16s %-9s %-8s %-9.4f %-10.2e %s"
                  % ("%.4f@%+.0f" % (abs(_rt), np.degrees(np.angle(_rt))),
                     ("%.4f" % _nb) if _twov else "-",
                     ("%.1f" % _tb) if _twov else "-",
                     _n3, _res, _tag), flush=True)
            if _argrho_target is not None:
                print("[SCDC]     arg rho target %+7.2f deg -> achieved %+7.2f deg "
                      "(residual %.2f deg, pol=%s, ref=L3 face)"
                      % (_argrho_target, _argrho_l[-1]["achieved_deg"],
                         _argrho_l[-1]["residual_deg"], pol), flush=True)
            _ns_l.append(_n3)
        return _ns_l, _pairs_l, _bad_l, _argrho_l

    _ol3r_raw = CFG.get("out_l3_r_list", "").strip()
    _rle_raw = CFG.get("out_l3_r_list_eval", "").strip()
    if not _ol3r_raw and _rle_raw:
        # [260923] A job with only an evaluation list and no out_l3_r_list -- SPEC 2.
        print("[SCDC] ERROR: using out_l3_r_list_eval requires out_l3_2layer = 1 "
              "and out_l3_r_list", flush=True)
        return 2
    if _ol3r_raw:
        _t3 = _num("out_l3_nm", 0.0)
        if _t3 <= 0:
            print("[SCDC] ERROR: using out_l3_r_list requires out_l3_nm (thickness)",
                  flush=True)
            return 2
        # [260914] If `out_l3_2layer = 1`, solve with a **2-layer termination**.
        # Why: a 1-layer slab has a thickness shared among terminations, so the only degree of freedom
        # is the refractive index, and only one curve can be drawn in the complex r plane. Measured (260914) -- at H3 = 875 nm
        # the (|r|, arg) grid reached by 1 layer is **8.9 %**, and of the 6 representative targets of the
        # library cluster **0** are reachable above the parasitic threshold `n = 1.178358`, with residuals
        # of 1.4e-2~7.5e-2. 2 layers (`t_b + t_3 = H3`) cover **100 %** of the same grid and
        # **6/6** of the same targets, solving both refractive indices above the threshold with residual <= 1.8e-5.
        # A termination below the threshold turns the trapped parasitic orders into evanescent waves and
        # models purity backwards, so it becomes a termination that is "right but wrong".
        _two = _num("out_l3_2layer", 0.0) > 0
        _nmin = _num("out_l3_n_min", 1.178358)
        _band = _num("out_l3_band", 0.0) > 0
        # [260914] **The wavelength at which the termination is solved never read this key.**
        # `_num("lams_nm", 1550.0)` applies `float()` to a comma list, which always
        # raises ValueError, and silently fell back **to the default 1550** -- only one log line.
        # All current jobs are centred at 1550, so the used and intended values agreed, but a job whose
        # band centre is not 1550 solves the termination at the wrong wavelength.  `M.LAMS` is
        # already parsed correctly at :450, so its **median** (= band centre) is
        # used, and it can be overridden with `out_l3_lam_nm`.
        _lu = _num("out_l3_lam_nm", 0.0) * 1e-3
        if _lu <= 0.0:
            _sl = sorted(M.LAMS)
            _lu = _sl[len(_sl) // 2]
        print("[SCDC] complex-reflection termination, total thickness %.1f nm, %s (n >= %.4f), %s:"
              % (_t3, "2 layers" if _two else "1 layer", _nmin if _two else 1.02,
                 ("band match %s nm"
                  % ", ".join("%.2f" % (v * 1e3) for v in M.LAMS)) if _band
                 else "design wavelength %.2f nm" % (_lu * 1e3)), flush=True)
        _ns, _pairs, _bad, _argrho = _solve_r_terms(_ol3r_raw, _t3, _two,
                                                    _nmin, _band, _lu)
        if _ns is None:
            return 2
        if any(v is not None for v in _argrho):
            _ARGRHO_OPT = _argrho
        if _bad:
            print("[SCDC] **%d targets are unreachable at total thickness %.1f nm -- change out_l3_nm, "
                  "or turn on out_l3_2layer = 1 if it is 1 layer, or reduce the targets. "
                  "Stopping.**" % (_bad, _t3), flush=True)
            return 2
        if _two:
            # The result dict keys the termination by its **n3 value** (`n{int(n*1000):04d}`
            # prefix, `n_ref` selection). For 2 layers, different (n_b, t_b) can
            # land on the same n3, so block it here -- stopping before the computation is better than
            # one termination silently vanishing from the result.
            if len(set("%.6f" % v for v in _ns)) != len(_ns):
                print("[SCDC] ERROR: n3 values of the 2-layer solutions coincide (%s) -- the termination axis "
                      "vanishes from the result. Spread the targets or change out_l3_nm."
                      % ", ".join("%.4f" % v for v in _ns), flush=True)
                return 2
            # `out_l3_ns_eval` is a **bare refractive-index** list, so it has no meaning for 2 layers.
            # If the lengths differ, the fold loop pairs the inner layer by index incorrectly.
            if CFG.get("out_l3_ns_eval", "").strip():
                print("[SCDC] ERROR: out_l3_2layer and out_l3_ns_eval cannot be used together "
                      "-- the evaluation termination must also be given as complex r (key not implemented)", flush=True)
                return 2
        M.set_out_l3_list(_t3, _ns)
        if _two:
            M.set_out_l3b_list(_pairs)
            print("[SCDC] %d 2-layer terminations -- inner layer (n_b, t_b) = %s, outer n3 = %s, "
                  "total thickness fixed at %.1f nm"
                  % (len(_pairs),
                     ", ".join("(%.3f, %.0f)" % p for p in _pairs),
                     ", ".join("%.4f" % v for v in _ns), _t3), flush=True)
        else:
            M.set_out_l3b_list(())
            print("[SCDC] out_l3_ns <- %s (back-calculated from %d complex targets)"
                  % (", ".join("%.4f" % v for v in _ns), len(_ns)), flush=True)
        # [260923 "1-termination optimisation / 5-termination evaluation"] **Report-only 2-layer
        # termination list.** Solved with exactly the same solver/arguments (thickness /
        # n_min / band / design wavelength) as `out_l3_r_list`, but it never enters
        # the optimisation objective -- `_report_eval` swaps it in only at the time `report()` is called.
        if not _two and _rle_raw:
            # SPEC 2: reject immediately unless out_l3_2layer = 1.
            print("[SCDC] ERROR: out_l3_r_list_eval requires out_l3_2layer = 1 and "
                  "out_l3_r_list (1-layer termination is not supported)",
                  flush=True)
            return 2
        if _two and _rle_raw:
            if CFG.get("out_l3_ns_eval", "").strip():
                # The [260914] check above already blocks this combination (same reason:
                # `out_l3_2layer` and `out_l3_ns_eval` cannot be used together),
                # but the message states it once more with this key name.
                print("[SCDC] ERROR: out_l3_r_list_eval and out_l3_ns_eval "
                      "cannot be used together -- both swap M.OUT_L3_NS at report() time "
                      "and would overwrite each other.", flush=True)
                return 2
            _ens, _epairs, _ebad, _eargrho = _solve_r_terms(
                _rle_raw, _t3, _two, _nmin, _band, _lu,
                label="out_l3_r_list_eval")
            if _ens is None:
                return 2
            if any(v is not None for v in _eargrho):
                _ARGRHO_EVAL = _eargrho
            if _ebad:
                print("[SCDC] **out_l3_r_list_eval: %d targets are unreachable at total thickness %.1f nm "
                      "-- change out_l3_nm or reduce the targets. "
                      "Stopping.**" % (_ebad, _t3), flush=True)
                return 2
            if len(set("%.6f" % v for v in _ens)) != len(_ens):
                print("[SCDC] ERROR: n3 values of the out_l3_r_list_eval 2-layer solutions "
                      "coincide (%s) -- the termination axis vanishes from the result. Spread the targets "
                      "or change out_l3_nm."
                      % ", ".join("%.4f" % v for v in _ens), flush=True)
                return 2
            _RLIST_EVAL_NS = _ens
            _RLIST_EVAL_PAIRS = _epairs
            _RLIST_EVAL_RAW = _rle_raw
            print("[SCDC] out_l3_r_list_eval <- %s (report-only, %d "
                  "terminations); objective sees %d"
                  % (_rle_raw, len(_ens), len(_ns)), flush=True)
    _ol3ns_raw = CFG.get("out_l3_ns", "").strip()
    _ol3ns = []
    if _ol3ns_raw:
        try:
            _ol3ns = [float(v) for v in _ol3ns_raw.replace(",", " ").split()]
        except ValueError:
            print("[SCDC] ERROR: bad out_l3_ns = %r" % _ol3ns_raw, flush=True)
            return 2
    if _ol3 > 0 and _ol3ns:
        M.set_out_l3_list(_ol3, _ol3ns)
        print("[SCDC] L3 multi-termination worst-case optimisation -- n = %s (%d points, %.0f nm)"
              % (", ".join("%.4f" % v for v in _ol3ns), len(_ol3ns), _ol3),
              flush=True)
    # [260922] **Interlayer-gap robustness fold.** The SiO2 deposition that fills L1 also covers the top,
    # creating an unintended uniform plate (0~100 nm) between L1/L2 -- measured (champion
    # density, spacer 200): efficiency 85.8 % (gap 0) -> 50.7 % (5 nm) -> 7.7 % (25 nm).
    # A different gap is applied to each termination with the **same length and the same order** as the
    # termination list, so the fold loop pairs them by index (zip, not a Cartesian product) -- the number of
    # solves per iteration stays the same.
    # **Why placed here**: terminations enter through two paths, out_l3_r_list (complex reflection) and
    # out_l3_ns (refractive-index list), and both resolve to M.OUT_L3_NS. If the validation is put
    # on the raw variable of one path, jobs of the other path are silently rejected -- the needle
    # caught exactly that.
    _gapl_raw = CFG.get("gap_nm_list", "").strip()
    if _gapl_raw:
        try:
            _gapl = [float(v) for v in
                     _gapl_raw.replace(";", " ").replace(",", " ").split()]
        except ValueError:
            print("[SCDC] ERROR: bad gap_nm_list = %r" % _gapl_raw, flush=True)
            return 2
        _nterm = len(M.OUT_L3_NS)
        if _nterm == 0:
            print("[SCDC] ERROR: gap_nm_list is meaningful only with a termination fold "
                  "-- both out_l3_r_list and out_l3_ns are empty.",
                  flush=True)
            return 2
        if len(_gapl) != _nterm:
            print("[SCDC] ERROR: gap_nm_list has %d entries but there are %d terminations -- "
                  "they are paired by index so the lengths must be equal (a shorter list does not "
                  "silently use only the first part)." % (len(_gapl), _nterm), flush=True)
            return 2
        _nse_chk = CFG.get("out_l3_ns_eval", "").strip()
        if _nse_chk:
            _nn = len(_nse_chk.replace(";", " ").replace(",", " ").split())
            if _nn != len(_gapl):
                print("[SCDC] ERROR: gap_nm_list has %d entries but out_l3_ns_eval has "
                      "%d -- the gap index overflows while report runs on the evaluation list. "
                      "Blocked at submission time." % (len(_gapl), _nn),
                      flush=True)
                return 2
        # [260923] `out_l3_r_list_eval` gets the same check mirrored, for the same reason
        # -- `_RLIST_EVAL_NS` is already resolved above, so its length is used directly
        # (the raw string is not counted again).
        if _RLIST_EVAL_NS is not None:
            if len(_RLIST_EVAL_NS) != len(_gapl):
                print("[SCDC] ERROR: gap_nm_list has %d entries but "
                      "out_l3_r_list_eval has %d -- the gap index overflows while report runs on the "
                      "evaluation list. Blocked at submission time."
                      % (len(_gapl), len(_RLIST_EVAL_NS)), flush=True)
                return 2
        M.set_gap_list(_gapl)
        print("[SCDC] interlayer-gap robustness fold -- gap = %s nm (%d points, with the termination "
              "zip)" % (", ".join("%.1f" % v for v in _gapl), len(_gapl)),
              flush=True)
    # [260909] **Evaluation-only termination list.** Same mechanism as `lams_eval_nm` --
    # `M.OUT_L3_NS` is replaced by this list only at the time `_report_eval` is called, and
    # the optimisation loop (`fom_extract`/`fom_split`) keeps using the `_ol3ns` above
    # (see `_report_eval`). Initialise explicitly for every job -- if
    # main() is called twice in one process, the termination list of the previous job must not be
    # inherited (same reason as the lams_eval_nm re-initialisation).
    _NS_EVAL = None
    _NS_EVAL_LOGGED = False
    _nsev_raw = CFG.get("out_l3_ns_eval", "").strip()
    if _nsev_raw:
        # Without the L3 thickness the termination list has no effect at all -- unless blocked here,
        # an experiment different from what the job file says would silently run.
        if _ol3 <= 0:
            print("[SCDC] ERROR: out_l3_ns_eval is given but out_l3_nm <= 0 "
                  "(without the L3 thickness the termination list has no effect)",
                  flush=True)
            return 2
        try:
            _NS_EVAL = [float(v) for v in _nsev_raw.replace(",", " ").split()]
        except ValueError:
            print("[SCDC] ERROR: bad out_l3_ns_eval = %r" % _nsev_raw,
                  flush=True)
            return 2
        if not _NS_EVAL:
            print("[SCDC] ERROR: out_l3_ns_eval is empty", flush=True)
            return 2
        print("[SCDC] L3 termination evaluation-only list -- report uses n = %s (%d points), "
              "optimisation uses n = %s (%d points)"
              % (", ".join("%.4f" % v for v in _NS_EVAL), len(_NS_EVAL),
                 ", ".join("%.4f" % v for v in _ol3ns), len(_ol3ns)),
              flush=True)

    kw = {}
    if kind == "input":
        kw["w_te"] = _num("w_te", 1.0)
        kw["w_echo"] = _num("w_echo", 0.0)
        kw["w_refl"] = _num("w_refl", 0.0)
        kw["w_back"] = _num("w_back", 0.0)
        # [260824 **bug fix**] w_bal(260819) / w_slope(260820) /
        # w_echo2(260820) were put in here, but the `fom_input` signature is only
        # `(w_te, w_echo, w_refl, w_back, ng, lams)`, so it **died at once with a TypeError**.
        # The last successful input job was 260817 (c17ain), and it was never
        # run again after that, so this stayed hidden. It was only by luck that
        # the upper design being fixed in 260817 masked it.
        # To really use the three terms, they must first be implemented in `fom_input`.
    elif kind == "input_pi":
        # Polarization-independent input grating -- the only weight is w_refl. Since TE is not sent
        # to the 0 order, w_te / w_echo / w_bal lose their meaning.
        kw["w_refl"] = _num("w_refl", 0.0)
    elif kind == "split_pi":
        # [260820] Polarization-independent input/output grating. w_ratio/w_bal/w_cross of fom_split are
        # not included -- tying down more degrees of freedom while matching both polarizations at once
        # (same judgement as extract_pi). w_cross is meaningless anyway: this structure is meant to
        # accept both polarizations, so there is no 'opposite polarization'.
        kw["w_sym"] = _num("w_sym", 0.0)
        kw["w_stray"] = _num("w_stray", 0.0)
        # [260820] 2nd-order echo suppression -- R(+2)|-th (return re-entry). Default 0.
        kw["w_echo2"] = _num("w_echo2", 0.0)
        kw["w_slope"] = _num("w_slope", 0.0)
        # [260821] **Inter-polarization bias difference `dpsi`.** In the single track TE/TM share one device,
        # so it cannot be removed by translation (a shift moves both equally by 2Gd).
        # The total loss is -20log10|cos(dpsi/2)|, yet the measured values were scattered
        # over 0.41~15.94 dB -- the signal_worst ranking had been ignoring this term.
        kw["w_dpsi"] = _num("w_dpsi", 0.0)
        kw["dpsi_target_deg"] = _num("dpsi_target_deg", 0.0)
    elif kind == "extract_pi":
        # Polarization-independent extraction grating. w_ratio / w_bal are not included since they would tie down
        # more degrees of freedom while matching both polarizations at once.
        kw["w_stray"] = _num("w_stray", 0.0)
        kw["w_back"] = _num("w_back", 0.0)
    elif kind == "retro_te":
        # Only TE is maximised, so there is no w_te.
        kw["w_leak"] = _num("w_leak", 1.0)
        # [260820] Normal-incidence TM transmission compensation. The outlet of the TM-lane extraction grating
        # exits at this position, so this term is needed to reduce the share reflected back (R(0)|normal).
        # Default 0 -- turning it on doubles the solves per iteration.
        kw["w_passn"] = _num("w_passn", 0.0)
        # [260820] Bias phase slope suppression. The delay arm passes the return grating, so
        # arg R(-2) enters the system bias directly (measured -1.82 deg/nm).
        kw["w_slope"] = _num("w_slope", 0.0)
    elif kind == "retro":
        kw["w_leak"] = _num("w_leak", 1.0)
        # [260820] Bias phase slope. One axis of the dpsi of the two-way TM lane.
        kw["w_slope"] = _num("w_slope", 0.0)
        # [260814] If w_te=1, the square root of the TM/TE product is taken on the worst-wavelength basis.
        kw["w_te"] = _num("w_te", 0.0)
    elif kind in ("split", "shared"):
        # [260814] shared and split share the signature in **only the first three**.
        # Leaking into the else branch (extract) passes w_te and raises a TypeError.
        # w_sym suppresses R(-1) -- normal incidence is left-right symmetric, so without it the split is half and half.
        #
        # [260824 **bug fix**] The 4 split-only keys below (w_echo2 / w_r1 /
        # r1_target / w_slope) were also passed to shared, and `fom_shared()` **died at once
        # with a TypeError**. The `fom_shared` signature is only
        # `(w_sym, w_ratio, w_stray, ng, lams)`.
        # When the keys were added to the split side in 260820 this branch was not reviewed
        # together, and `kw[k] = _num(k, 0.0)` is a structure that **assigns unconditionally even if
        # the value is 0**, so it blew up even when the job did not use those keys.
        # Symptom: 19 v2sh jobs of 260814 ended with `seed failed -- skipped [0 s]` for all seeds,
        # the job ended normally with `[result summary] no data`. It was quiet, so nobody noticed.
        kw["w_sym"] = _num("w_sym", 0.0)
        kw["w_ratio"] = _num("w_ratio", 0.0)
        kw["w_stray"] = _num("w_stray", 0.0)
        # [260820] 2nd-order echo suppression -- R(+2)|-th (return re-entry). Default 0.
        # The outlet of the split geometry is R(0)|-th (-th upper left -> x-832 upper bare), and
        # the fibre channel R(+1)|-th is tied to the delay-arm launch by reciprocity, so it cannot be touched.
        # [260906 correction] **"Leaves the slab by total internal reflection" was wrong -- TIR is a
        # mechanism that traps light inside.** The critical angle arcsin(1/1.4440) = 43.83 deg while theta = 54.69 deg,
        # so everything reaching the bare face at +-theta is totally reflected and **stays in the slab**.
        # Only the k_x = 0 normal channel can leave. So the penalised
        # share moves to the outlet channel, but that channel is not a free sink --
        # the measured median R(0)|-th of 10.75 % (458 cells) is not even read by fom_split.
        # [260824] From here on it is **split-only** -- fom_shared does not accept it.
        if kind == "split":
            kw["w_echo2"] = _num("w_echo2", 0.0)
            # [260820] R(+1) target term. Default target 1/3 (optimum of the zero-stray-light limit).
            # With stray light the optimum moves lower, so r1_target is left open.
            kw["w_r1"] = _num("w_r1", 0.0)
            # [260829 correction] Default 1/3 -> **4/9**. 1/3 is the discarded model using only
            # energy conservation, and the optimum including reciprocity `S = P S^T P` is
            # A = P = B = 4/9, with ceiling (4/9)^1.5 = 0.296296.
            kw["r1_target"] = _num("r1_target", 4.0 / 9.0)
            # [260820] Bias phase slope suppression. The geometric term (ch^2) cannot be removed, but
            # the grating slope is pushed negative and enlarges the asymmetry, so it is a minimisation target.
            kw["w_slope"] = _num("w_slope", 0.0)
        # [260819] If the rotated layout is chosen, the split grating becomes a TM device.
        # pol=TE is the default, so an unspecified job behaves exactly as before.
        if kind == "split":
            _pl = (CFG.get("pol", "") or "TE").strip().upper()
            if _pl not in ("TE", "TM"):
                print("[SCDC] ERROR: pol = %r (only TE | TM supported)" % _pl)
                return 2
            kw["pol"] = _pl
            # [260824 bug fix] w_bal / w_cross are also **split-only**.
            # They were outside the `if kind == "split"` block and leaked into shared.
            kw["w_bal"] = _num("w_bal", 0.0)
            kw["w_cross"] = _num("w_cross", 0.0)
            # [260827] L3 plane illumination purity. The design channel of the split lane is normal incidence m=0.
            kw["w_pure"] = _num("w_pure", 0.0)
            # [260922] Fidelity term that also counts evanescent orders in the same channel.
            # `spacer_purity` is exactly 1.000 from the first iteration below the propagation threshold
            # `kx/k0 = 1.1784`, so its gradient is exactly 0 -- TM lost 403 cells
            # through that hole. To keep TE from walking the same path, it is wired in
            # before the lane starts. Default 0, so previous jobs are
            # bit-identical.
            kw["w_fid"] = _num("w_fid", 0.0)
            # [260926] Include the -theta extraction-pass fid in the fid term (split lane, used
            # together with `w_fid`). Same convention as the extract-side option (:1327-1330) -- put in kw
            # only when nonzero (so as not to touch the checkpoint fingerprint,
            # same reason as w_r2 above).
            if _num("fid_minus", 0.0):
                kw["fid_minus"] = 1
            # [261002] -theta delay-arm spacer POWER purity into the w_pure
            # term (split lane; use with w_pure > 0). Same rule as fid_minus:
            # inserted into kw only when non-zero, so `_ckpt_sig` and every
            # stored result stay bit-identical when the key is absent.
            if _num("pure_minus", 0.0):
                kw["pure_minus"] = 1
            # [260831] echo epsilon constraint (hinge). See the extract-side comment below.
            kw["w_echo_cap"] = _num("w_echo_cap", 0.0)
            kw["echo_cap_db"] = _num("echo_cap_db", 0.0)
    else:
        kw["w_te"] = _num("w_te", 0.0)
        kw["w_ratio"] = _num("w_ratio", 0.0)
        kw["w_stray"] = _num("w_stray", 0.0)
        # [260819] Total fibre back-flow B = R(-2)|+th + R_0|+th * R_0|-th.
        # It overlaps with w_stray (both contain R(-2)), so do not turn both on at once.
        kw["w_back"] = _num("w_back", 0.0)
        # [260827 **wiring omission fix**] The `fom_extract` (line 608) signature is
        # `(w_te, w_ratio, w_stray, w_back, w_bal, w_echo2, w_slope, ng, lams)`
        # but this branch was passing only the first four. The last three were **silently ignored** even if
        # written in the job -- since there was no TypeError, it stayed hidden.
        #
        # In particular `w_echo2` is a term that routes the 2nd-order echo to the fourth arm (outlet) that does not use it.
        # [260906 correction] **This is not the top-priority shortfall item.** In the completed 29 cells
        # (seeds=8, out_l3_ns=[1.1..1.9]) efficiency passes 0/29 (best H475_375 =
        # 70.68 % of the ceiling, 14.3 %p short of the 85 % gate), echo 2/29 (1/29 on the -20.5 basis),
        # purity 1/29, linewidth 29/29. **The axis that fails in all cells is efficiency, and echo is an
        # axis with passing cases.** Moreover the two are tied in a single trade-off (below,
        # signal_worst decreases monotonically from w_echo2 0.2 -> 4.0), so raising w_echo2 is
        # selling the 0/29 axis to buy the 2/29 axis. -12.9 dB is a single-condition (no-L3) number
        # from before the 5-termination fold.
        # In the 3-layer driver `Main_rcwa_scd_c3.py` (lines 164/168/188) it is wired, so
        # the C50 echo Pareto came out (signal_worst decreases monotonically from w_echo2 0.2 -> 4.0,
        # 0.291864 -> 0.255552, it really works), but **the 2-layer track
        # did not have it**. c74 (then the best TM cell) was also optimised without these three.
        # [260906] Do not call c74 the "current champion" -- in the 5-termination fold TM has
        # 0/29 for 4 gates, 0 for 3 gates too, and the c74-family numbers are no-L3 single-condition.
        kw["w_bal"] = _num("w_bal", 0.0)
        kw["w_echo2"] = _num("w_echo2", 0.0)
        kw["w_slope"] = _num("w_slope", 0.0)
        # [260827] **L3 plane illumination purity.** Used together with `out_spacer_nm`.
        # eta_+ is measured in air, but the only propagating order into air is m=-1, so
        # it cannot see the parasitic m=0, m=-2 (+-54.69 deg in the spacer, evanescent in air).
        # Those parasitics get trapped in the spacer and create FP resonance of L3 and local-period breakdown.
        # Measured (c74 TM): the design-channel share of the forward power in the spacer is only 28 %.
        # This term directly targets a "base that causes no problem when a metalens is put on top".
        kw["w_pure"] = _num("w_pure", 0.0)
        # [260922] **Fidelity term that also counts evanescent orders.** The `spacer_purity` used by
        # `w_pure` discards evanescent orders by `if c2 <= 0: continue`, and
        # if the spacer refractive index is below the propagation threshold `kx/k0 = 1.1784`, both parasitics
        # become evanescent, so **purity is 1.000 from the first iteration and the gradient is 0**.
        # The parasitic amplitude grows through that gap -- measured (260922): an n = 1.08 carrier
        # passes all four gates, but the best `fid_l3` of the atom above it is 0.6814,
        # worse than 0.9836 at n = 1.444, and `evan_frac_l3` is 30.4 times.
        # `spacer_fid` counts all orders at the face where the atom sits, so it
        # closes that hole. Default 0, so previous jobs are bit-identical (needle diff 0.000e+00).
        # [260922 update] `fom_split` now also accepts this argument -- the per-lane
        # design order is (m=-1, THETA_SUB) for extract and (m=0, 0.0) for split.
        kw["w_fid"] = _num("w_fid", 0.0)
        # [260831] **echo epsilon constraint (hinge).** It is meant to open the region
        # that `w_echo2` (weighted sum) cannot reach. In a 33-cell measurement, echo -13~-17 dB was **0 cells**
        # at all three weights (0.5/2.0/3.0), and the echo-efficiency correlation r = +0.781 was
        # a double cluster -- a typical symptom of using a weighted sum on a non-convex front.
        # Inside `echo <= cap` the gradient is 0, so only the efficiency is raised freely.
        # Default 0 = off, so existing jobs behave exactly the same.
        #   w_echo_cap   hinge weight (multiplies the excess in dB). 0.01~0.05 recommended
        #   echo_cap_db  target upper limit, given as negative (e.g. -20.0)
        kw["w_echo_cap"] = _num("w_echo_cap", 0.0)
        kw["echo_cap_db"] = _num("echo_cap_db", 0.0)
        # [260908] **Designated-outlet bounded target term.** rs[-1] = R(-1)|+th has k_x = 0, so it is
        # the only channel that can really leave the slab, i.e. a legitimate outlet. The measured
        # median of 20.95 % uses nearly twice the ideal 1/9 = 11.11 %, yet neither
        # objective function reads this value. It is a bounded squared term of the form (rs[-1] - dump_target)^2,
        # and an unbounded linear reward is not used -- `w_exit` collapsed that way
        # (signal 0.296 -> 0.002, the outlet took 98.6 %).
        # Default 0 = off, so existing jobs behave exactly the same.
        # The `dump_target` default must be **the same 1/9** as the `fom_extract` signature --
        # using a different value silently overrides the signature default even if the job does not write the key.
        #
        # **Do not turn on together with w_stray.** `strays = rs[-1] + rs[-2]`, so both
        # grab rs[-1], and the two terms pull the same order in opposite directions.
        # (Judgement 260908: the `strays` split is not done now -- it would change the meaning of
        #  `w_stray` in all stored results. Obtain the answer by running the w_dump ladder at
        #  `w_stray = 0`, and handle the split then as a separate change with a basis.)
        kw["w_dump"] = _num("w_dump", 0.0)
        kw["dump_target"] = _num("dump_target", 1.0 / 9.0)
        # [260923] **Linear penalty on the fibre back-flow R(-2)|+th alone.**
        # Of the existing three terms, none looked at only this quantity:
        #   `w_stray` = rs[-1] + rs[-2] -- also penalises the outlet that must be kept open
        #   `w_back`  = rs[-2] + r0*mr0 -- `r0 = R_go` is a required factor of signal, so
        #                                  pressing it directly reduces the signal. The record's
        #                                  "w_back is a closed axis" was not about the axis but
        #                                  **the quantity was wrong**.
        #   `w_dump`  = outlet only     -- does not look at back-flow at all
        # Measured 260923 (SiO2, 713 cells): the median back-flow is 0.1913 and the shortfall of `eta_+ + R_go`
        # against the ideal 8/9 is **exactly the same 0.1913** -- the whole efficiency gap is
        # this one term. The distribution is 0.0039~0.7353 (190x) and cells at 93~95 % of the ceiling
        # hold 0.05~0.09, so it can be reduced.
        #
        # **Do not assign 0 to `kw`.** `_ckpt_sig` hashes `sorted(kw.items())`
        # as a whole, so if the key appears even with value 0 the fingerprint changes and **on
        # resubmission the completed seeds and Adam checkpoints are all discarded**. The 53 jobs
        # stopped in the 260923 overhaul are exactly that case. Same reason and
        # same handling as `GAP_LIST` (260922). `w_dump` is assigned unconditionally, so it already
        # paid that cost once in 260908 -- not repeated.
        _wr2 = _num("w_r2", 0.0)
        if _wr2:
            kw["w_r2"] = _wr2
        # [260926] **delay-arm target term on r0 = R(0)|+th** (extract only,
        # mirrors split's w_r1/r1_target). Same rule as w_r2 above -- insert
        # into kw only when non-zero, so `_ckpt_sig` and every stored result
        # stay bit-identical when this key is absent from the job file.
        _wrgo = _num("w_rgo", 0.0)
        if _wrgo:
            kw["w_rgo"] = _wrgo
            kw["rgo_target"] = _num("rgo_target", 0.5)
        # [260924] **Include the -theta extraction-pass fid in the fid term** (used together with `w_fid`).
        # Put in kw only when nonzero -- same reason as `w_r2` above (checkpoint fingerprint).
        if _num("fid_minus", 0.0):
            kw["fid_minus"] = 1
        # [261002] -theta delay-arm spacer POWER purity into the w_pure term
        # (extract lane; use with w_pure > 0). Same rule as fid_minus above.
        if _num("pure_minus", 0.0):
            kw["pure_minus"] = 1

    raw_h = CFG.get("harmonics", "").strip()
    if raw_h:
        try:
            harm = tuple(int(v) for v in raw_h.replace(",", " ").split())
        except ValueError:
            print("[SCDC] bad harmonics=%r -- using default" % raw_h)
            harm = DEFAULT_HARMONICS[kind]
    else:
        harm = DEFAULT_HARMONICS[kind]

    out_dir = CFG.get("result_folder", "").strip()
    if not out_dir:
        root = CFG.get("Output_folder", r"<DATA_ROOT>\Result").strip()
        out_dir = os.path.join(root, CFG.get("subfolder_name", "scd_c"))
    os.makedirs(out_dir, exist_ok=True)

    print("[SCDC] H1 = %.1f nm  %s in %s   |   H2 = %.1f nm  %s in %s"
          % (H1_nm, m1_name, f1_name, H2_nm, m2_name, f2_name))
    print("[SCDC]   (L1 touches the substrate and L2 touches the air)")
    if gap_nm > 0:
        print("[SCDC] interlayer spacer %.1f nm (%s, uniform) -- stack: substrate/L1/spacer/L2/air"
              % (gap_nm, f1_name))
    print("[SCDC] nG = %d (opt) / %d (eval)   seeds = %d   iters = %d   lr = %.3f"
          % (M.NG, M.NG_EVAL, n_seeds, iters, lr))
    if _ngs:
        print("[SCDC] ng_schedule = %s  (frac:nG, t/iters; final-stage "
              "scoring at nG %d)"
              % (", ".join("%g:%d" % (fs, g) for fs, g in _ngs), _ngs[-1][1]))
    print("[SCDC] %s   nest = %s   harmonics = %s"
          % ("  ".join("%s = %s" % (k, ("%.3f" % v) if isinstance(v, (int, float))
                                  else v)
                    for k, v in sorted(kw.items())),
             "ON" if nest else "OFF", list(harm)))

    filt_nm = _num("filt_radius_nm", min_feat)
    px = M.set_min_feature(filt_nm)
    print("[SCDC] min feature = %.1f nm (gate)   conic filter = %.1f nm (%d px)"
          % (min_feat, filt_nm, px))
    # [260926] w_lw geometry -- computed once the real filter radius (px,
    # i.e. filt_radius_nm) is known. Absent/0/eval_only -> _lw_kw stays {}
    # and M.optimize() below is called exactly as before.
    _lw_kw = {}
    if _wlw > 0 and not eval_only:
        try:
            _lwg = M.lw_geometry(_num("lw_target_nm", min_feat),
                                 (_num("lw_c", 0.0) or None))
        except ValueError as e:
            print("[SCDC] ERROR: %s" % e, flush=True)
            return 2
        _lw_kw = dict(w_lw=_wlw, lw_target_nm=_num("lw_target_nm", min_feat),
                      lw_beta_min=_num("lw_beta_min", 16.0),
                      lw_c=_lwg["c"], lw_gate_nm=min_feat)
        print("[SCDC] w_lw = %.4g  target %.1f nm -> L_px %d  R_px %d  "
              "eta_e %.5f  eta_d %.5f  c %.6g  gate opens at beta >= %.1f"
              % (_wlw, _num("lw_target_nm", min_feat), _lwg["L_px"],
                 _lwg["R_px"], _lwg["eta_e"], _lwg["eta_d"], _lwg["c"],
                 _num("lw_beta_min", 16.0)), flush=True)
    elif _wlw > 0 and eval_only:
        print("[SCDC] w_lw ignored (eval_only)", flush=True)
    print("[SCDC] out = %s" % out_dir, flush=True)

    score_key, aux_keys, ceiling, ref = KINDS[kind]
    # [260816] In the return, if w_te=0 TE is not an optimisation target (the TE arm uses a copy of the TM
    # structure rotated by 90 degrees). But score_key is fixed to
    # bal_worst, so seeds were chosen by **TE, which was not even optimised**.
    # Measured: c12are_525_500_r1 saved multi_q (TM 0.996864) while rand4 of the same run
    # passed the gate with TM 0.999984. The reason bal_worst was put in in 260814
    # (TE not visible in the ranking) is valid only when w_te=1.
    if kind == "retro" and kw.get("w_te", 0.0) == 0.0:
        score_key = "retro_R_m2"
        aux_keys = tuple(k for k in aux_keys if k != score_key) + ("bal_worst",)

    # [260904] Build the column names of the runs_log TSV **only once**, so that the header writing and
    # the error-row filling use the same list.
    #
    # Previously the error row put the verdict by position with `_row[-2]`, but the header is
    # `... verdict, sec, <PHASE_KEYS...>`, so `-2` was the **second phase column from the end**.
    # It happened to be right only for kinds with empty PHASE_KEYS, and every kind actually run
    # has phase columns. Measured (260904, runs_log of the entire Result, 6874 rows):
    # ERROR rows that went into verdict **0**, those that went into a phase column **51**
    # (arg_eta_plus 38 / arg_split_T0 12 / signal_TM 1).
    # There were two consequences -- a failed seed was **not visible** to whoever reads the verdict,
    # and a string went into a phase column that should be numeric (a column the phase library reads as float).
    #
    # A correction such as `-2 - len(PHASE_KEYS)` is not used -- if the header changes again it would go
    # out of step silently in the same way. **Look up by name.** And since the header is kept
    # in only one place here, the log writer and the error row cannot structurally diverge.
    _HDR = (["tag", score_key, "spread"] + list(aux_keys)
            + ["minf1_nm", "minf2_nm", "verdict", "sec"]
            + list(PHASE_KEYS.get(kind, ())))
    _VERDICT_I = _HDR.index("verdict")

    seed_rng = int(_num("seed_rng", 0))
    rng = np.random.default_rng(seed_rng)
    seeds = M.seed_bank(max(0, n_seeds - 3), rng=rng, harmonics=harm)[:n_seeds]

    warm = CFG.get("seed_npy", "").strip()
    if _c2d and not warm:
        # [260924] the seed bank is 1-D; a 2-D run starts from a stored carrier.
        print("[SCDC] ERROR: carrier_2d = 1 needs seed_npy (a stored 1-D or 2-D "
              "carrier density)")
        return 2
    if warm:
        if not os.path.isfile(warm):
            print("[SCDC] ERROR: seed_npy not found -- %s" % warm)
            return 2
        w = np.load(warm)
        if _c2d:
            # [260924] (2, nx, ny) -> flattened; a stored 1-D carrier (2, N) is
            # area-resampled to nx and extruded along y (the atom driver's rule).
            w = np.asarray(w, float)
            if w.ndim == 3:
                w = w.reshape(w.shape[0], -1)
            elif w.ndim == 2 and w.shape[1] != M.NX:
                print("[SCDC] carrier_2d: 1-D seed %s -> extruded (2, %d x %d)"
                      % (w.shape, M.GRID_2D[0], M.GRID_2D[1]), flush=True)
                w = np.vstack([M.extrude_1d(w[0]), M.extrude_1d(w[1])])
            # [260924] SYMMETRY BREAKING. At a y-invariant density the gradient
            # along every zero-y-mean perturbation is exactly 0 (such a
            # perturbation couples m_y = 0 to m_y = +-1 only, so the m_y = 0
            # response moves at second order) -- an extruded 1-D seed would stay
            # 1-D forever. `seed_ynoise` adds a smooth zero-y-mean field of that
            # peak amplitude (conic-filtered, per layer); optimisation only.
            _yn = float(_num("seed_ynoise", 0.0))
            if _yn > 0 and not eval_only and w.shape == (2, M.NX):
                _rg = np.random.default_rng(int(_num("seed_ynoise_rng", 1)))
                w = np.array(w, float)
                for _k in (0, 1):
                    _z = _rg.standard_normal(M.GRID_2D)
                    _z = np.real(np.fft.ifft2(np.fft.fft2(_z) * M.KF_2D))
                    _z = _z - _z.mean(axis=1, keepdims=True)
                    _z = _z / (np.abs(_z).max() + 1e-12)
                    w[_k] = np.clip(w[_k] + _yn * _z.reshape(-1), 0.0, 1.0)
                print("[SCDC] carrier_2d: seed_ynoise %.3f (rng %d) added"
                      % (_yn, int(_num("seed_ynoise_rng", 1))), flush=True)
        if w.shape != (2, M.NX):
            print("[SCDC] ERROR: seed_npy shape %s, expected (2, %d)"
                  % (w.shape, M.NX))
            return 2
        if eval_only:
            # [260907] **eval_only -- score the stored structure as is.**
            # Relaxing here would make the scored object differ from the stored structure.
            # The warm relaxation below (0.15 + 0.70*rho) is an **optimisation preparation** to prevent
            # the gradient from dying at points where the tanh projection is saturated, and
            # on a path that does no optimisation it has no purpose. There is also only one seed --
            # mixing in random seeds could change the winner by competing with them.
            seeds = [("eval", w[0], w[1])]
            print("[SCDC] eval_only = 1 -- scoring the stored density as is "
                  "(no relaxation, no optimisation): %s" % warm)
        else:
            # The stored density is a {0,1} binary value. Put in as is, it starts from a point where the
            # tanh projection is saturated and the gradient almost dies -- so it is put in
            # after being compressed to the amplitude used by structural seeds.
            w = 0.15 + 0.70 * np.clip(w, 0.0, 1.0)
            seeds = [("warm", w[0], w[1])] + seeds[:max(0, n_seeds - 1)]
            print("[SCDC] warm start = %s" % warm)
    elif eval_only:
        # Without seed_npy there is no structure to score. Silently optimising a random seed
        # would be the worst outcome, so stop here.
        print("[SCDC] ERROR: eval_only = 1 but there is no seed_npy -- "
              "specify the stored density to score.")
        return 2

    # [260923] **Parallel decomposition** -- the seed bank drawn by rng is left as is
    # (filtering happens only after the seed_bank call above + warm/eval_only handling are done),
    # and only the names to run are chosen. This is for submitting one seed as one job in parallel, so
    # the remaining seeds of each job must start from **exactly the same point**.
    # Give parallel jobs a different subfolder_name per seed -- cell-level files such as scdc_*.json /
    # runs_log are overwritten with "w".
    _seed_only = CFG.get("seed_only", "").strip()
    if _seed_only:
        if eval_only:
            print("[SCDC] ERROR: seed_only cannot be used together with eval_only "
                  "(eval_only already has only one seed)")
            return 2
        _bank_names = [s[0] for s in seeds]
        _want = [w.strip() for w in _seed_only.split(",") if w.strip()]
        if not _want:
            print("[SCDC] ERROR: seed_only is empty -- possible names: %s"
                  % ", ".join(_bank_names))
            return 2
        _unknown = [w for w in _want if w not in _bank_names]
        if _unknown:
            print("[SCDC] ERROR: name %s in seed_only does not exist -- possible names: %s"
                  % (", ".join(_unknown), ", ".join(_bank_names)))
            return 2
        seeds = [s for s in seeds if s[0] in _want]
        print("[SCDC] seed_only -> %s (of %d in the bank)"
              % (", ".join(s[0] for s in seeds), len(_bank_names)))

    rows, best, best_any = [], None, None
    # [260904 P2] Names of seeds that died with an exception. **Excluded** from the checkpoint cleanup below.
    # The ckpt of the seed that died with an IndexError in 260904 (w1_g43w0065_c5000 / multi, 8681 s)
    # was deleted together in the cleanup loop, so the failed density could not in principle be reproduced
    # (only the npy of the successful warm remained). Without the failing input, offline reproduction is
    # impossible and one has to wait for the next occurrence by eye again.
    _failed_seeds = set()
    t_start = time.time()
    for name, x1, x2 in seeds:
        tag = "%s_H%.0f_%.0f_%s" % (kind, H1_nm, H2_nm, name)
        t0 = time.time()
        # [260822] The exception of one seed does not lose the whole job.
        # In 260821 a crash during the 8th seed discarded the 7 seeds that
        # had already finished (same measure as the 3-layer driver).
        # [260831] **Stop/resume.** If an interrupted job is resubmitted with the same job file, it continues.
        #   1) seed that already has a density  -> skip the optimisation, re-evaluate only
        #   2) seed that was in progress        -> continue from that iteration in the ckpt
        # out_dir is determined by subfolder_name, so a resubmission sees
        # the same folder. Results already written there are not deleted, so they remain.
        _seed_stem = stem_of(kind, H1_nm, H2_nm)
        _npy_path = os.path.join(out_dir, "density_scdc_%s_%s.npy"
                                 % (_seed_stem, name))
        _ckpt_path = os.path.join(out_dir, "ckpt_scdc_%s_%s.npz"
                                  % (_seed_stem, name))
        try:
            _done = None
            if eval_only:
                # [260907] **Skip** the optimisation. The "completed seed" path below is
                # already exactly that behaviour (run only report and leave hist empty), so
                # use the same branch -- a separate path would let the two places diverge.
                # `iters` / `lr` are ignored (not made an error): the purpose of this key is to
                # reuse an existing job file as is for scoring.
                _done = (x1, x2)
            elif os.path.isfile(_npy_path):
                try:
                    _w = np.load(_npy_path)
                    if _w.shape == (2, M.NX):
                        _done = (_w[0], _w[1])
                    else:
                        print("  [%s] existing density shape %s -- ignored, running again"
                              % (tag, _w.shape), flush=True)
                except Exception as _e:
                    print("  [%s] failed to read existing density -- running again: %r"
                          % (tag, _e), flush=True)
            if _done is not None:
                # The stored density is the only truth. Instead of trusting the sidecar, evaluate
                # again -- report takes tens of seconds, optimisation tens of minutes.
                d1, d2 = _done
                hist, dt = [], 0.0
                _bbmeta = None
                M.ckpt_clear(_ckpt_path)
                print("  [%s] %s" % (tag, "eval_only -- scoring the stored structure as is"
                                     if eval_only else
                                     "completed seed -- re-evaluation only"), flush=True)
            else:
                # [260923] `ng_schedule` is passed only when set, so the
                # absent-key call (`**{}`) is equivalent to the old one.
                _ngs_kw = {"ng_schedule": _ngs} if _ngs else {}
                _xy, r1, r2, hist, _bbmeta = M.optimize(
                    kind, x1, x2, H1, H2, eps_b, e_f1, e_f2, iters=iters, lr=lr,
                    tag=tag, nest=nest, log_every=max(1, iters // 6),
                    ckpt_path=_ckpt_path, ckpt_every=ckpt_every,
                    top_k=top_k, **_ngs_kw, **_lw_kw, **kw)
                dt = time.time() - t0

                d1, d2 = M.binarize(r1), M.binarize(r2)
                if nest:
                    d2 = d2 * d1
                if save_hist:
                    # [260923] In the same place as the density file, with the same stem name.
                    # Column 0 = iteration, column 1 = FoM. For a seed continued from a checkpoint
                    # hist has only the tail (previous 8 + after), so the iteration is written too --
                    # the tail is contiguous and ends at iters.
                    try:
                        _h = np.asarray(hist, dtype=float)
                        _it = np.arange(iters - len(_h) + 1, iters + 1,
                                        dtype=float)
                        np.save(os.path.join(
                            out_dir, "hist_scdc_%s_%s.npy"
                            % (_seed_stem, name)),
                            np.column_stack([_it, _h]))
                    except Exception as e:           # a save failure must not kill the run
                        print("  [%s] hist save failed (ignored): %r" % (tag, e),
                              flush=True)
            mf = []
            for d in (d1, d2):
                if d.min() == d.max():
                    mf.append(float(M.PERIOD * 1e3))       # a uniform layer has no linewidth constraint
                else:
                    _r, m = M.extract_ridges(d)
                    mf.append(float(m or 0.0))
            float_nm, _ffr = M.unsupported((d1, d2), (f1_name, f2_name))
            feasible = min(mf) >= min_feat and float_nm <= float_tol
            verdict = ("ok" if feasible else
                       "INFEASIBLE(float %.0fnm)" % float_nm
                       if float_nm > float_tol else "INFEASIBLE(linewidth)")

            # [260904] The seed winner selection is also done on the evaluation band. The `score_key`
            # produced here is the selection criterion, so if picked with 3 points, a seed that
            # wrecked the unmonitored 1547.6 / 1552.4 could win.
            rep = _report_eval(M, d1, d2, H1, H2, eps_b, e_f1, e_f2, kind,
                               ng=M.NG_EVAL, pol=kw.get("pol", "TE"))
            rep["float_span_nm"] = float_nm
            sc = rep[score_key]
            aux = "  ".join("%s %s" % (k, _fk(rep, k)) for k in aux_keys)
            spread = rep.get("%s_spread" % score_key, float("nan"))

            # [260816] **Keep a density for every seed.** Previously only the best 1 (+ the best rejected
            # 1) was saved and the structures of the other seeds were lost. The purpose of running
            # several seeds at the same parameters is not an efficiency search but **securing phase freedom**
            # (build a library of solutions with similar efficiency but different phase differences,
            # then pick the solution that cancels the bias phase accumulated in the system), so all are needed.
            # One file is tens of KB, so even 8 seeds is no burden.
            if _done is None:
                try:
                    np.save(_npy_path, np.vstack([d1, d2]))
                    # Discard the checkpoint only after the seed has been safely saved.
                    M.ckpt_clear(_ckpt_path)
                except Exception as e:                 # a save failure must not kill the run
                    print("  [%s] density save failed (ignored): %r" % (tag, e), flush=True)
                # [260915] top_k > 1: pop the raw per-rank density arrays out
                # of _bbmeta before it is ever merged into a JSON-bound dict
                # (below, and again where a seed becomes best_any/best) --
                # those merges use **_bbmeta and a numpy array would break
                # json.dump. This is placed after the block above on purpose
                # so _npy_path is unaffected either way.
                _topk_dens = (_bbmeta or {}).pop("_top_k_densities", ())
                for _rk, (_td1, _td2) in enumerate(_topk_dens, 1):
                    try:
                        np.save(os.path.join(
                            out_dir, "density_scdc_%s_%s_top%d.npy"
                            % (_seed_stem, name, _rk)),
                            np.vstack([_td1, _td2]))
                    except Exception as e:
                        print("  [%s] top_k density save failed (ignored): %r"
                              % (tag, e), flush=True)

            rows.append([tag, "%.6f" % sc, "%.4f" % spread]
                        + [_fk(rep, k, "%.6f", "") for k in aux_keys]
                        + ["%.1f" % mf[0], "%.1f" % mf[1],
                           verdict, "%.1f" % dt]
                        # Phase is the key axis of the library, so put it directly in the log.
                        + [_fk(rep, k, "%.2f")
                           for k in PHASE_KEYS.get(kind, ())])
            print("  [%s] %s %.6f (spread %.3f)  %s  minf %.1f/%.1f nm  %s  [%.0f s]"
                  % (tag, score_key, sc, spread, aux, mf[0], mf[1],
                     verdict, dt), flush=True)

            if best_any is None or sc > best_any[score_key]:
                best_any = dict(rep, seed=name, feasible=feasible,
                                min_feature_1_nm=mf[0], min_feature_2_nm=mf[1],
                                # [260908] Also leave the best-binarised-iterate observations
                                # in the result JSON (until now only print).
                                **(_bbmeta or {}),
                                _density=np.vstack([d1, d2]))
            if not feasible:
                continue
            if best is None or sc > best[score_key]:
                best = dict(
                    rep, seed=name, H1_nm=H1_nm, H2_nm=H2_nm,
                    material1=m1_name, material2=m2_name,
                    fill1=f1_name, fill2=f2_name, nest=nest, gap_nm=gap_nm,
                    # [260830] **Record the exit-side spacer.** Previously the result
                    # JSON had only `gap_nm` (between layers) and not `out_spacer_nm` (above L2,
                    # where the L3 atom will sit), so which spacer the density was optimised at
                    # remained **only in the job file**. Now that the spacer has become the outermost variable
                    # (260830) this causes axis contamination -- in fact the combination of the sp200 density
                    # evaluated at sp175 was once reported as the best (1.545 dB).
                    out_spacer_nm=_osp, n_out_spacer=_osn,
                    out_l3_nm=_ol3, n_out_l3=_ol3n,
                    out_l3_ns=list(_ol3ns),
                    n_sub=float(M.N_SUB), period_nm=float(M.PERIOD * 1e3),
                    theta_sub_deg=th_deg, theta_crit_deg=th_c,
                    seeds=n_seeds, iters=iters, lr=lr, seed_rng=seed_rng,
                    harmonics=list(harm), nG=M.NG, nG_eval=M.NG_EVAL,
                    min_feature_1_nm=mf[0], min_feature_2_nm=mf[1],
                    min_feature_nm=min(mf), min_feature_nm_job=min_feat,
                    filt_radius_nm=filt_nm, feasible=True, score_key=score_key,
                    ceiling=ceiling, ref_other_conditions=ref,
                    fom_history_tail=[float(v) for v in hist[-5:]],
                    # [260908] Also leave the best-binarised-iterate observations (triggered or not / when / gain)
                    # in the result JSON -- **placed before kw so that if names collide
                    # (in practice they do not) **kw wins.
                    **(_bbmeta or {}),
                    # [260813] Without this line the saved structure came out as that of the **last seed**
                    # (the fallback of the saving part below used the loop-leftover d1/d2).
                    # The JSON score is that of the best seed, so the two were different structures.
                    _density=np.vstack([d1, d2]), **kw)

        except Exception as _e:
            import traceback
            dt = time.time() - t0
            print("  [%s] **seed failed -- skipped** [%.0f s]  %r"
                  % (tag, dt, _e), flush=True)
            traceback.print_exc()
            # [260904 P2] Keep the ckpt of this seed -- the failed input is the evidence.
            _failed_seeds.add(name)
            print("  [%s] checkpoint kept (failed seed) -- "
                  "reproducible offline from ckpt_scdc_%s_%s.npz"
                  % (tag, stem_of(kind, H1_nm, H2_nm), name), flush=True)
            if rows:
                # The TSV width differs per kind, so fill it to match the preceding successful row.
                _row = [""] * len(rows[0])
                _row[0] = tag
                # [260904] Look up by **name**, not by position (-2). See `_HDR` above.
                _row[_VERDICT_I] = "ERROR(%s)" % type(_e).__name__
                rows.append(_row)
    elapsed = time.time() - t_start
    # [260831] All seeds are finished, so the checkpoints are scratch files.
    # They are cleaned up below.
    # [260904 P2] **But those of failed seeds are kept.** Successful seeds are
    # cleaned up exactly as before, so nothing extra is added to the output folder -- only when there was a failure
    # does the one ckpt of that seed (about 24 KB) stay with the results. It is the input needed for reproduction.
    for _name, _x1, _x2 in seeds:
        if _name in _failed_seeds:
            continue
        M.ckpt_clear(os.path.join(out_dir, "ckpt_scdc_%s_%s.npz"
                                  % (stem_of(kind, H1_nm, H2_nm), _name)))
    if _failed_seeds:
        print("[SCDC] kept the checkpoints of %d failed seeds: %s"
              % (len(_failed_seeds), ", ".join(sorted(_failed_seeds))),
              flush=True)
    stem = "%s_H%.0f_%.0f" % (kind, H1_nm, H2_nm)
    # The wavelength set was not left in any JSON. To compare per band (35 nm / Track A 8.01 nm /
    # Track B 0.80 nm) one had to re-read the job file in the result folder or rely on the folder
    # naming convention -- both fragile. It is written directly into the result.
    lams_nm = [round(v * 1e3, 4) for v in M.LAMS]
    # [260904] `lams_nm` is the **optimisation** band -- `_report_eval` restores it in finally,
    # so `M.LAMS` at this point is always the optimisation band. The evaluation band is added
    # separately only when the key is present. If absent, the JSON does not differ from before by a single character.
    _band = {"lams_nm": lams_nm}
    if _c2d:
        # [260924] only when on -> 1-D JSONs unchanged.
        _band["carrier_2d"] = dict(nx=int(M.GRID_2D[0]), ny=int(M.GRID_2D[1]),
                                   atoms_per_period=int(M.APP_2D),
                                   ly_nm=float(M.L2[1] * 1e3),
                                   px_nm=float(M.PX_NM), sym_y=bool(M.SYM_Y_2D))
    # [260907] Record in the result that this was a scoring-only run. If absent or 0, the key is not put in,
    # so the bytes equal those of past JSONs. Without this mark one could not later tell
    # whether "this 87.21 % is a re-optimisation result or the stored structure as is".
    if eval_only:
        _band["eval_only"] = 1
    if _LI_ON:
        # [261008 li_integrate] only when non-default -> default JSON byte-identical
        _band["rcwa_rule"] = _RCWA_RULE
        _band["rcwa_ff"] = int(_RCWA_FF)
    if _seed_only:
        # [260923] Record in the result that this job ran only a subset of the bank --
        # without it one could not later tell "why is this seed missing".
        _band["seed_only"] = _seed_only
    if top_k > 1:
        _band["top_k"] = top_k
    if _ngs and not eval_only:
        # [260923] only when a schedule was set; absent -> JSON unchanged.
        _band["ng_schedule_used"] = [[float(fs), int(g)] for fs, g in _ngs]
    if _lw_kw and not eval_only:
        # [260926] only when w_lw > 0 and not eval_only (same rule as
        # ng_schedule_used above); absent -> JSON is byte-identical.
        _band["lw_penalty"] = dict(
            w_lw=_lw_kw["w_lw"], lw_target_nm=_lw_kw["lw_target_nm"],
            L_px=_lwg["L_px"], R_px=_lwg["R_px"],
            filt_radius_eff_nm=_lwg["R_px"] * M.PX_NM,
            eta_e=_lwg["eta_e"], eta_d=_lwg["eta_d"], c=_lwg["c"],
            lw_beta_min=_lw_kw["lw_beta_min"], method="zhou2015_geometric_1d")
    if _LAMS_EVAL:
        _band["lams_eval_nm"] = [round(v * 1e3, 4) for v in _LAMS_EVAL]
    if _NS_EVAL:
        # [260909] report() is called with the swap in place, so `out_l3_ns_used`
        # becomes the **evaluation** list -- the optimisation list would remain nowhere in the JSON,
        # and a reader following the "branch on out_l3_ns_used" (project convention) would
        # take this job to have the optimisation list as is. Both are kept.
        _band["out_l3_ns_eval"] = list(_NS_EVAL)
        _band["out_l3_ns_opt"] = list(_ol3ns)
    if _RLIST_EVAL_NS:
        # [260923] Same reason -- `out_l3_ns_used` records `M.OUT_L3_NS` as read at the time of
        # report(), so in this job, where report() runs with the swap in place, it
        # becomes the **evaluation** (r_list_eval) list. The list the optimisation
        # actually saw (out_l3_r_list) remains nowhere in the JSON, so both are
        # kept.
        _band["out_l3_r_list_eval"] = _RLIST_EVAL_RAW
        _band["out_l3_r_list_opt"] = _ol3r_raw
    # [260923] Track 2 (DEMAND) -- achieved `arg rho` per termination, only
    # when the job's `out_l3_r_list`/`out_l3_r_list_eval` actually used the
    # 3rd token field. Absent otherwise -> JSON byte-identical to before.
    if _ARGRHO_OPT is not None:
        _band["out_l3_argrho_opt"] = _ARGRHO_OPT
    if _ARGRHO_EVAL is not None:
        _band["out_l3_argrho_eval"] = _ARGRHO_EVAL
    if _ARGRHO_OPT is not None or _ARGRHO_EVAL is not None:
        _band["out_l3_argrho_ref"] = "L3 face"
    with open(os.path.join(out_dir, "runs_log_scdc_%s.tsv" % stem), "w",
              encoding="utf-8") as f:
        # [260904] There is only one header, `_HDR` -- the error row finds the verdict position
        # from the same list, so the two cannot diverge. The string is the same as before.
        f.write("\t".join(_HDR) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    if best_any is not None and not best_any["feasible"]:
        inf_d = best_any.pop("_density")
        np.save(os.path.join(out_dir, "density_scdc_%s_infeasible.npy" % stem),
                inf_d)
        if _c2d:
            for _k in (0, 1):
                np.save(os.path.join(out_dir, "rho%d_2d_infeasible.npy" % (_k + 1)),
                        np.asarray(inf_d[_k]).reshape(M.GRID_2D))
        with open(os.path.join(out_dir, "infeasible_best_scdc_%s.json" % stem),
                  "w", encoding="utf-8") as f:
            json.dump(dict(best_any, H1_nm=H1_nm, H2_nm=H2_nm, kind=kind,
                           material1=m1_name, material2=m2_name,
                           fill1=f1_name, fill2=f2_name,
                           nG=M.NG, nG_eval=M.NG_EVAL, **_band,
                           min_feature_nm_job=min_feat), f, indent=2)
        print("[SCDC] the best seed fails the linewidth gate (minf %.1f/%.1f vs %.1f) "
              "-- the density is kept for repair."
              % (best_any["min_feature_1_nm"], best_any["min_feature_2_nm"],
                 min_feat), flush=True)

    if best is None:
        print("[SCDC] no usable cell  kind=%s H1=%.1f H2=%.1f  [%.0f s]"
              % (kind, H1_nm, H2_nm, elapsed))
        with open(os.path.join(out_dir, "scdc_%s.json" % stem), "w",
                  encoding="utf-8") as f:
            json.dump(dict(kind=kind, H1_nm=H1_nm, H2_nm=H2_nm,
                           material1=m1_name, material2=m2_name,
                           fill1=f1_name, fill2=f2_name, feasible=False,
                           nG=M.NG, nG_eval=M.NG_EVAL, **_band,
                           min_feature_nm_job=min_feat,
                           elapsed_total_sec=elapsed), f, indent=2)
        with open(os.path.join(out_dir, "Variable_result_information.txt"), "w",
                  encoding="utf-8") as f:
            f.write("scdc %s H1=%.1f H2=%.1f : no feasible cell\n"
                    % (kind, H1_nm, H2_nm))
        return 0

    # No default fallback is provided -- if missing, it must show up immediately as a KeyError instead of
    # silently saving a wrong structure.
    dens = best.pop("_density")
    # [260818] If a target phase is given, the final design is shifted sideways to match it. Efficiency is
    # preserved exactly, so the optimisation result is not damaged (see apply_phase_offset).
    _pt = CFG.get("phase_target_deg", "").strip()
    if _pt and _c2d:
        print("[SCDC] carrier_2d: phase_target_deg is 1-D only -- ignored",
              flush=True)
        _pt = ""
    if _pt:
        dens, _pinfo = apply_phase_offset(dens, float(_pt), kind, H1, H2,
                                          eps_b, e_f1, e_f2, M.NG_EVAL, M, np,
                                          pol=kw.get("pol", "TE"))
        if _pinfo:
            best.update(_pinfo)
            best.update(_report_eval(M, dens[0], dens[1], H1, H2, eps_b,
                                     e_f1, e_f2, kind, ng=M.NG_EVAL,
                                     pol=kw.get("pol", "TE")))
    np.save(os.path.join(out_dir, "density_scdc_%s.npy" % stem), dens)
    if _c2d:
        for _k in (0, 1):
            np.save(os.path.join(out_dir, "rho%d_2d.npy" % (_k + 1)),
                    np.asarray(dens[_k]).reshape(M.GRID_2D))
    best["elapsed_total_sec"] = elapsed
    best.update(_band)
    with open(os.path.join(out_dir, "scdc_%s.json" % stem), "w",
              encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    sc = best[score_key]
    sp = best.get("%s_spread" % score_key, float("nan"))
    print("[SCDC] BEST  %s = %.6f   ceiling %.4f (%.1f %%)"
          % (score_key, sc, ceiling, 100 * sc / ceiling))
    if ref is not None:
        print("[SCDC]       reference: measured under other conditions (Lambda 533.8, Si) %.5f" % ref)
    print("[SCDC]       3-wavelength spread %.4f  %s"
          % (sp, "OK" if sp < 0.05 else "** resonance suspected -- not flat over the band"))
    print("[SCDC]       %s" % "  ".join("%s %s" % (k, _fk(best, k))
                                        for k in aux_keys))
    if kind == "extract":
        # Check by eye 'whether the total energy comes out that way'. From the headline alone one cannot
        # tell whether the split is right or where the share that did not leave went.
        print("[SCDC]       energy budget  first pass (+G): extract %.4f + pass %.4f + "
              "scatter %.4f + rest %.4f"
              % (best["budget_fwd_extract"], best["budget_fwd_pass"],
                 best["budget_fwd_stray"], best["budget_fwd_rest"]))
        print("[SCDC]                    return (-G): extract %.4f + rest %.4f"
              % (best["budget_rev_extract"], best["budget_rev_rest"]))
        print("[SCDC]       split       eta_-/eta_+ = %.3f (target 1.000, error %.1f %%)"
              % (best["ratio_minus_over_plus"], 100 * best["ratio_err"]))
        print("[SCDC]       unitarity   air row %.4f / first-pass row %.4f  (each ceiling 1.0)"
              % (best["unitary_air_row"], best["unitary_fwd_row"]))
    elif kind == "input":
        print("[SCDC]       reflection (fibre back-flow)  TM %.4f + TE %.4f = %.4f"
              % (best["R_TM"], best["R_TE"], best["R_both"]))
    print("[SCDC]       min_feature %.1f nm" % best["min_feature_nm"])
    with open(os.path.join(out_dir, "Variable_result_information.txt"), "w",
              encoding="utf-8") as f:
        f.write("kind = %s\n" % kind)
        f.write("%s = %.6f  (ceiling %.4f, %.1f %%)\n"
                % (score_key, sc, ceiling, 100 * sc / ceiling))
        f.write("spread_3lambda = %.4f\n" % sp)
        for k in aux_keys:
            f.write("%s = %s\n" % (k, _fk(best, k, "%.6f", "None")))
        f.write("min_feature = %.1f nm\n" % best["min_feature_nm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
