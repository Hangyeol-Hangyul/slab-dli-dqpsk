# -*- coding: utf-8 -*-
r"""L3 2D atom **free-form** adjoint optimisation (added 260825).

Why this is needed
-----------
At present the L3 atom is a disk (1 knob) or an ellipse (2 knobs). But a single library
element must satisfy two targets:

    phi_c = (A+B)/2   lens phase -- a **different** value `360*j/N` for each element
    dpsi  =  B - A    differential -- the **same** value for all elements

Two knobs for two targets works in principle (260825 Jacobian condition number 1.3~3.2),
but if a residual remains there is no way to push further. The free form has `NXY^2` degrees of freedom (48 -> 2304,
64 -> 4096), so it **can meet both targets exactly for every element.**

Cost (260825 measured, `bench_autograd`)
------------------------------------
Backpropagation multiplier **3.0~3.4** (the same level as 1D even at 4608 degrees of freedom).
nG scaling exponent of `fwd+bwd` **2.69**.

    nG=201  5.6 s/solve  -> 2 solves including the arm = 11.2 s/iter -> 1200 iter 3.7 h
    nG=301 16.6                            33.2          -> 11.1 h
    nG=601 106                            213            -> 71 h

**The bottleneck is nG, not autograd.** 1D needs only nG=81 for 512 degrees of freedom, but
2D has two-dimensional G vectors, so nG must be much larger. So, as in 1D,
**the optimisation nG and the evaluation nG are separate** (`nG` / `nG_eval`).

What is not seen
--------------
**The breakdown (LPA violation) does not enter this objective.** It assumes an infinite array of a single atom,
so neighbouring-cell coupling is not seen. As of 260825, 38 % of the total budget is that term,
so a shape obtained here must be measured separately with `Main_rcwa_l3_atom2d_lpa.py`.

job keys
    l12_density      L1/L2 density npy (fixed -- treated as freeze12)
    H1_nm H2_nm H3_nm  spacer_nm  n_spacer
    phi_target_deg   lens phase target of this element
    dpsi_target_deg  differential target common to all elements
    w_amp w_phi w_dpsi w_r0   objective-function weights
    prod_target      **target of the product `|t_A| * |r_B t_B|`.** This is the signal itself
                     (differential interference term = 2|A||B|). If set, it replaces amp_target.
                     Measured for a disk (H3=850/t=200/1550, nG=201):
                     TM median 0.1092 / TE median 0.0232.
                     When using it set `w_r0 = 0` (it is subsumed in the product target).
                     **Raise `w_amp` as well** -- the phase term is `_ph` (0~2) x 2, so
                     at most 4, and an amplitude term smaller than that is ignored.
    amp_target       (old) target of `|t_A|` only. Do not use, since the reference arm collapses.
    init_ellipse     "dx,dy" -- start from an ellipse (disk seed if absent)
    init_npy         start from a stored density (`rho3_opt.npy`) -- phase continuation.
                     Takes **priority** over `init_ellipse`. Error if the grid differs.
    init_npy_soft    margin to relax that density (default 0.05, same as the ellipse seed)
    w_eff eff_gate_pct  [260929] efficiency hinge (fom(), near the
                     amp/prod terms). w_eff = 0 (default) -> off,
                     bit-identical to before. w_eff > 0:
                     _eff = 843.167 * |tA| * |rB*tB| (already in pct
                     of the (4/9)^1.5 ceiling); pen = w_eff *
                     max(eff_gate_pct - _eff, 0) -- linear one-sided
                     hinge, same shape as the echo hinge. Uses the
                     TRACED tA/rB/tB, not _untraced. eff_gate_pct
                     default 86.0. Report gains carrier_eff_pct,
                     carrier_echo_db, eff_ok (>= 85), eff_id_err
                     (WARN printed if > 1e-5).
    echo_gate_db     [260929] dB-based alternative to echo_a2_gate,
                     must be <= -20.5. echo_a2_gate =
                     10**(echo_gate_db/10) * (1.0 TM / 0.334083 TE).
                     Giving both echo_gate_db and echo_a2_gate is an
                     error (rc 2).
    es_window        [261001] Adam-loop convergence early stop, FINAL
                     nG/beta stage only (see ng_schedule/beta_schedule).
                     int, default 0 = OFF (driver stays bit-identical).
                     > 0 sets the window size W: the last W+1 iterations
                     EXECUTED BY THIS PROCESS must all sit in the final
                     nG stage and the final beta before the check runs.
                     A resumed run starts with an empty window, so it
                     may stop later than an uninterrupted run, never
                     earlier.
    es_dA_deg        default 0.3    max wrapped |A_deg(j) - A_deg(t)|
                     over the window, degrees (A_deg is the captured
                     opt_terms.tsv column).
    es_dfom_rel      default 0.001  |FoM(t) - FoM(t-W)| <= es_dfom_rel *
                     |FoM(t)|. FoM is the same value written to
                     opt_log.dat.
    es_dprod_rel     default 0.001  |P(t) - P(t-W)| <= es_dprod_rel *
                     P(t). P = amp_tA * amp_rBtB (both captured, always
                     computed regardless of weights).
    es_fid_gate      default 0.994 if pol == TE else 0.996722 (the
                     standing four-gate purity floor).
    es_fid_margin    default 0.001  for fid_l3 and, if also captured,
                     fid_l3_m (needs w_fid / w_fid_m > 0): if the window
                     minimum is below es_fid_gate + es_fid_margin,
                     additionally require max |q(j) - q(t)| <= es_dfid
                     over the window. Skipped for a quantity this job
                     never captures.
    es_dfid          default 1e-4
    es_echo_margin_db default 2.0   needs w_echo2 > 0 and an echo gate
                     (echo_gate_db or echo_a2_gate) set: if the window
                     maximum of the captured echo_a2_db (the same
                     quantity the objective's own echo hinge tests in
                     fom(), ~line 1058-1069) is above
                     (10*log10(w["echo_a2_gate"]) - es_echo_margin_db --
                     w["echo_a2_gate"] already carries the per-lane scale
                     baked in at parse time, so no further lane scaling
                     belongs in this guard), additionally require
                     max |e(j) - e(t)| <= es_decho_db over the window.
                     Skipped if the echo term is not in play.
    es_decho_db      default 0.3
                     On a stop, the current (post-update) x is measured
                     with the SAME linewidth/fill gate the post-loop
                     code uses (_min_feature_all/_fill_frac_all/
                     _gate_with_fill at beta 64.0); the stop is taken
                     only if that gate passes and a live layer exists,
                     else the loop continues (the normal linewidth ramp
                     runs). With es_window > 0, atom2d_opt.json gains
                     es_window, es_dA_deg, es_dfom_rel, es_dprod_rel,
                     es_fid_gate, es_fid_margin, es_dfid,
                     es_echo_margin_db, es_decho_db, es_stop_iter (int
                     or null) and iters_run (last iteration executed).
                     With es_window = 0 none of these keys are added.
    edge_margin_px   [260917] hard edge mask, px of void frozen at EACH side
                     of the NXY x NXY cell on both axes (0 = off).
    edge_margin_x_px edge_margin_y_px   [261002] per-axis margins; -1
                     (default) = use edge_margin_px for that axis, 0 = no
                     mask on that axis. Named by PHYSICAL axis: x = array
                     axis 0 = the plane-of-incidence axis (the cell is tiled
                     ATOMS_PER_PERIOD times along it, `_obj`), y = array
                     axis 1 (the sym_y mirror axis).
    edge_margin_mode [261002] 'pre' (default) multiplies the RAW variable
                     before the conic filter (the filter refills the strip:
                     void = 2*margin px only for margin >= 6 px at a 14 px
                     kernel, 4 -> 2 px, 5 -> 8 px; the start-up log prints
                     the guaranteed void); 'post' multiplies the
                     FILTERED field before the tanh projection, so the
                     projected density is exactly 0 on the strip and the
                     periodic void between neighbouring cells is exactly
                     2*margin px on that axis. An eval_only job must carry
                     the same three keys as the job that made the density.
                     All three keys unset -> numerically bit-identical to
                     before. [261002 consumers] atom2d_opt.json now ALWAYS
                     records the realised mask (edge_margin_x_px,
                     edge_margin_y_px = effective margins, -1 resolved;
                     edge_margin_mode; edge_margin_px stays). Consumers
                     rebuild it with `edge_mask_from_record(json, nxy)`.
    distinct_atoms   [261003] 0 (default) = the original single atom tiled
                     atoms_per_period times along x (driver bit-identical).
                     1 = a K = 1 cell with atoms_per_period (3) DISTINCT atoms
                     side by side along x inside the Lambda cell (sub-atom j
                     occupies x in [j*ly, (j+1)*ly), ly = Lambda/atoms_per_
                     period). Needs atoms_per_period >= 2 (else rc 2). The
                     design vector is atom-major: [atom 0: layer 0, layer 1],
                     [atom 1: ...]; every sub-atom gets its OWN conic filter
                     pass, tanh projection, sym_y and edge mask (the same
                     per-axis mask keys for all) -- the filter is periodic
                     inside each sub-atom, not across the seams. The
                     differentiable linewidth penalty (w_lw) and the final
                     linewidth gate are measured on the FULL cell (periodic
                     in x over Lambda), so seam slivers are seen. Every other
                     term (tA, rB, tB, echo, fid, r00, eff ...) is read from
                     the whole-cell orders exactly as before.
                     init_npy       ';'-separated rho3 .npy paths, one per
                                    sub-atom (old single-atom format), or ONE
                                    path replicated, or ONE rho3_cell_opt.npy
                                    cell file (shape atoms x layers x NXY x NXY).
                     rho3_npy (eval_only) same three forms.
                     OUTPUT NAMES DIFFER ON PURPOSE: the cell is saved as
                     rho3_cell_opt.npy (4-D) and atom2d_cell_opt.json, never
                     as rho3_opt.npy / atom2d_opt.json, so superramp
                     _load_atom, eval_atom2d_opt.py, standalone_eval,
                     Main_fdtd_l3_verify and the census scripts cannot pick
                     up a 3-atom cell as a single atom (a 4-D rho3_npy given
                     to a distinct_atoms = 0 job is rc 1, SystemExit).
                     The JSON adds distinct_atoms, atoms_in_cell, cell_kind,
                     fill_frac_atoms, min_feature_atoms_nm.
    iters lr nG nG_eval NXY  min_feature_nm
    ckpt_every       periodic Adam-state checkpoint for resume (default
                     10, 0 disables). File: ckpt_atom2d.npz in the result
                     dir. The shared copy is written/read/deleted ONLY
                     when the job file itself sets BOTH Output_folder and
                     subfolder_name, so a job redirected by result_folder
                     keeps its shared ckpt at
                     Output_folder/subfolder_name. Resume requires an
                     identical fingerprint: job keys minus
                     result_folder/Output_folder/subfolder_name/
                     mpi_np/mpi_np_final/blas_threads/ckpt_every, plus
                     this driver's own md5, the imported
                     scd_c_grating's md5, and the md5 of the CONTENTS of
                     every hashed input file. A resume from
                     the shared copy writes opt_log.dat/opt_terms.tsv
                     from the resume point only, not the full trajectory.
"""
import cmath
import glob
import hashlib
import io
import json
import math
import os
import sys
import time
import weakref

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "_vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _cfg():
    d = {}
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        for ln in io.open(sys.argv[1], encoding="utf-8",
                          errors="replace").read().split("\n"):
            if "=" not in ln or ln.lstrip().startswith("#"):
                continue
            k, _, v = ln.partition("=")
            d[k.strip()] = v.strip()
    return d


CFG = _cfg()


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
        print("[OPT2D] ERROR: rcwa_rule=%r (laurent|liy|lix), rcwa_ff=%r (0|1), rcwa_pol_epsinv=%r "
              "(this driver has no Pol path: only conv1)" % (rule, ffr, pinv), flush=True)
        sys.exit(2)
    return rule, int(ff)


_RCWA_RULE, _RCWA_FF = _li_parse(CFG)
_LI_ON = (_RCWA_RULE != "laurent") or bool(_RCWA_FF)
_LI_KEYS = ("rcwa_rule", "rcwa_ff", "rcwa_pol_epsinv")
# [261008 F2/D2] DEFAULT-DENY FOR IMPORTERS. This module applies the Li rule inside `_obj` / `_measure_*` for ANY
# importer, so an importer that builds its own grcwa.obj (superramp5, superopt, carrier2d, standalone_eval,
# fdtd_l3_verify, ffwin, eval_atom2d_opt, calib_circle_product, ...) would mix Laurent and Li inside one job.
# A non-default rcwa_rule / rcwa_ff is therefore honoured only when this file IS the process entry
# (`__name__ == "__main__"`), or when the PROCESS ENTRY script declared `LI_AWARE_ENTRY = True` at module level
# BEFORE importing this module (Main_rcwa_l3_superramp.py does). Anything else -> ERROR + exit 2. The entry (not the
# immediate importer) is what counts: superopt imports superramp, which declares, but superopt builds its own
# grcwa.obj. Default keys: this block is a no-op.
if _LI_ON and __name__ != "__main__":
    _entry = sys.modules.get("__main__")
    if not getattr(_entry, "LI_AWARE_ENTRY", False):
        print("[OPT2D] ERROR: rcwa_rule=%s rcwa_ff=%d but %s is IMPORTED by the process entry %r, which has not "
              "declared LI_AWARE_ENTRY = True -- an importer that builds its own grcwa.obj would mix Laurent and "
              "Li inside one job; refusing (exit 2)"
              % (_RCWA_RULE, int(_RCWA_FF), __name__,
                 getattr(_entry, "__file__", sys.argv[0] if sys.argv else "?")), flush=True)
        sys.exit(2)
# result-JSON extras: present ONLY when non-default (the default JSON stays byte-identical)
_li_extra = ({"rcwa_rule": _RCWA_RULE, "rcwa_ff": int(_RCWA_FF)} if _LI_ON else {})
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = (CFG.get("blas_threads", "").strip() or "1")

import numpy as np                                    # noqa: E402
import autograd.numpy as npa                          # noqa: E402
from autograd import value_and_grad                   # noqa: E402
import grcwa                                          # noqa: E402
# [260909] eval_only needs no gradient -- gated strictly on that flag so
# every other job (eval_only unset or 0) keeps the "autograd" backend
# byte-for-byte as before. Parsed inline (not via `_num`, not defined yet).
try:
    _eval_only_backend = (
        float(str(CFG.get("eval_only", "0")).strip() or "0") == 1.0)
except ValueError:
    _eval_only_backend = False
grcwa.set_backend("numpy" if _eval_only_backend else "autograd")  # noqa: E402
import scd_c_grating as G                             # noqa: E402


def _num(k, d):
    try:
        return float(str(CFG.get(k, "")).strip())
    except (TypeError, ValueError):
        return float(d)


def _txt(k, d=""):
    return (CFG.get(k, "") or d).strip()


def _li_apply(ob):
    """[261008 li_integrate] opt-in switches on one grcwa object; called ONLY when rcwa_rule / rcwa_ff
    is non-default."""
    if not getattr(type(ob), "LI_SUPPORT", 0):
        raise RuntimeError("the vendored grcwa in use has no Li support (obj.LI_SUPPORT missing): "
                           "refusing to run a Li job on the stock rule")
    ob.rcwa_rule = _RCWA_RULE
    ob.form_factor = _RCWA_FF


# ------------------------------------------------------------ ckpt (260929)
# [ckpt] **Iteration checkpoint / resume for the Adam loop.** `ckpt_every`
# (job key, default 10, 0 disables) periodically saves (x, m, v, t) plus a
# config fingerprint to `ckpt_atom2d.npz` in `od` (and best-effort to the
# shared Output_folder/subfolder_name when staging makes `od` a different,
# local dir) so a killed job resumes instead of restarting from iteration 1.
# Touches only the optimisation path (never `eval_only`); default behaviour
# with no ckpt file present is bit-identical to before this feature.
#
# [ckpt-r2] **Shared copy is written/read/deleted only when the JOB FILE
# ITSELF sets BOTH `Output_folder` and `subfolder_name`** (checked with this
# driver's own `_txt`, no defaults) -- a job that stages via `result_folder`
# alone, without either key, has no known shared destination, and treating
# the un-set defaults ("." / "opt") as one wrote a bogus `./opt/` next to
# the current working directory. A job redirected by `result_folder` that
# DOES also carry both keys keeps its shared ckpt at
# `Output_folder/subfolder_name` exactly as before.
#
# [ckpt-r2] A resume from the shared checkpoint never ships `opt_log.dat` / `opt_terms.tsv`
# with the shared checkpoint (only the `.npz` -- see `_ckpt_save`), so a job
# resumed from the SHARED copy where the local `od` has no history of
# its own reports `opt_log.dat` / `opt_terms.tsv` starting from the resume
# point only, not the full trajectory (T3).
_CKPT_DROP_KEYS = frozenset((
    "result_folder", "Output_folder", "subfolder_name",
    "mpi_np", "mpi_np_final", "blas_threads", "ckpt_every",
    # [261001] es_* keys only decide WHEN a converged run stops, never
    # the arithmetic that produced the trajectory up to a stop -- so a
    # resume must not be invalidated by changing them.
    "es_window", "es_dA_deg", "es_dfom_rel", "es_dprod_rel",
    "es_fid_gate", "es_fid_margin", "es_dfid", "es_echo_margin_db",
    "es_decho_db",
))
# [ckpt-r2] Input files whose CONTENTS (not just the path string already in
# CFG) are part of what a resume must match -- a job resubmitted after its
# `l12_density`/`l12_rho1_npy`/`l12_rho2_npy`/`init_npy` file was silently
# overwritten at the same path must not resume against the old content.
# Skipped when the job does not set that key, or the named file is missing.
_CKPT_HASH_INPUT_KEYS = ("l12_density", "l12_rho1_npy", "l12_rho2_npy",
                         "init_npy")


def _md5_of_file(path):
    h = hashlib.md5()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ckpt_fingerprint():
    """[ckpt] sha256 over CFG (comments/blank lines already dropped by
    `_cfg()`) minus the output/`ckpt_every` keys injected at launch or
    that do not describe the computation, keys sorted so job-file line
    order and duplicate-key overwrites do not change it, plus:
    [ckpt-r2] (1) this driver file's own md5 (`__file__`), (2) the md5 of
    the `scd_c_grating` module actually imported (`G.__file__` -- a job can
    only resume against the SAME code, not just the same config), and
    (3) the md5 of the CONTENTS of every file named by
    `_CKPT_HASH_INPUT_KEYS` that the job sets and that exists on disk.
    Returns (hexdigest, key_names, file_names) -- the caller prints both
    lists once at start so a resume or a mismatch is auditable from the log
    alone without re-deriving what went into the hash."""
    items = sorted((k, v) for k, v in CFG.items()
                   if k not in _CKPT_DROP_KEYS and k not in _LI_KEYS)
    # [261008 li_integrate] the Li keys join the fingerprint ONLY when non-default (normalised), so a
    # Li job can never resume a Laurent checkpoint while the default fingerprint is built as before
    if _LI_ON:
        items = sorted(items + [("rcwa_rule", _RCWA_RULE), ("rcwa_ff", str(int(_RCWA_FF)))])
    key_names = [k for k, _ in items]
    lines = ["%s=%s" % (k, v) for k, v in items]
    file_names = []
    _drv = os.path.abspath(__file__)
    lines.append("__driver_md5__=%s" % _md5_of_file(_drv))
    file_names.append(_drv)
    _g = os.path.abspath(G.__file__)
    lines.append("__scd_c_grating_md5__=%s" % _md5_of_file(_g))
    file_names.append(_g)
    if _LI_ON:
        _lf = os.path.join(os.path.dirname(os.path.abspath(grcwa.__file__)), "li_fmm.py")
        if os.path.isfile(_lf):
            lines.append("__li_fmm_md5__=%s" % _md5_of_file(_lf))
            file_names.append(_lf)
    for _key in _CKPT_HASH_INPUT_KEYS:
        _p = CFG.get(_key, "").strip()
        if _p and os.path.isfile(_p):
            lines.append("__%s_md5__=%s" % (_key, _md5_of_file(_p)))
            file_names.append(_p)
        elif _p and ";" in _p:
            # [261003 distinct_atoms] a ';'-separated seed list: hash the
            # CONTENTS of every existing part, in order (the CFG string itself,
            # which carries `distinct_atoms` and the list, is already in `items`).
            for _i, _pp in enumerate(z.strip() for z in _p.split(";")):
                if _pp and os.path.isfile(_pp):
                    lines.append("__%s_md5_%d__=%s"
                                 % (_key, _i, _md5_of_file(_pp)))
                    file_names.append(_pp)
    txt_all = "\n".join(lines)
    return (hashlib.sha256(txt_all.encode("utf-8")).hexdigest(),
            key_names, file_names)


def _ckpt_save(path, x, m, v, t, fp):
    """[ckpt] Atomic write: build the npz in a temp file next to `path`,
    then `os.replace` it into place -- a kill mid-write leaves either the
    old checkpoint (rename never happened) or the new one, never a
    half-written `path`. A save failure (locked file, full disk, missing
    share) must never end the run -- caught here, tmp swept, False
    returned -- same rule `scd_c_grating._ckpt_save` already follows: a
    failed save must not kill the run.
    [ckpt-r2] `x`/`m`/`v` keep their NATIVE dtype (`np.asarray` with no
    `dtype=` override) -- forcing float64 here silently upgraded a
    float32-seeded run (`init_npy` of that dtype) on its first resume,
    diverging the trajectory from an uninterrupted run bit-for-bit."""
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, ".ckpt_atom2d.npz.tmp_%d" % os.getpid())
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, x=np.asarray(x), m=np.asarray(m), v=np.asarray(v),
                     t=np.array([int(t)], dtype=np.int64),
                     fingerprint=np.array([str(fp)]))
        os.replace(tmp, path)
    except Exception as _e:
        print("[OPT2D] WARN: ckpt save failed (%s) at %s -- continuing"
              % (_e, path), flush=True)
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False
    return True


def _ckpt_load(path):
    """[ckpt] Inverse of `_ckpt_save` -- returns (x, m, v, t, fingerprint).
    [ckpt-r2] `x`/`m`/`v` are read back with NO `dtype=` override, so a
    float32 checkpoint stays float32 -- see the matching note in
    `_ckpt_save`."""
    with open(path, "rb") as fh:
        z = np.load(fh)
        x = np.array(z["x"])
        m = np.array(z["m"])
        v = np.array(z["v"])
        t = int(np.array(z["t"]).reshape(-1)[0])
        fp = str(np.array(z["fingerprint"]).reshape(-1)[0])
    return x, m, v, t, fp


def _ckpt_truncate_log(path, max_iter):
    """[ckpt] Keep the header line plus data rows whose first
    tab-separated field (the iteration number) is <= `max_iter`, dropping
    any row written after the last checkpoint but before the kill --
    without this, a resumed log would carry a few rows the resumed loop is
    about to write again. Report-only files, never read by the training
    path itself."""
    if not os.path.isfile(path):
        return
    with io.open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    if not lines:
        return
    keep = [lines[0]]
    for ln in lines[1:]:
        fld = ln.split("\t", 1)
        if not fld or not fld[0]:
            continue
        try:
            it = int(fld[0])
        except ValueError:
            continue
        if it <= max_iter:
            keep.append(ln)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.writelines(keep)


# [ckpt-r3] **Cross-process record of which shared directory levels this
# job's OWN checkpoint machinery created.** The `_shared_made` walk (in
# `main()`, at the first shared save) only sees what exists at the moment
# IT runs -- correct for a single process, but a killed-and-resumed job
# spans two OS processes, and by the time the resuming process starts,
# `_shared_od` already exists (the first process created it before dying).
# Recomputing the walk there would find everything already present and
# conclude (wrongly) that nothing is this job's to clean up. This sidecar,
# written once inside `_shared_od` at the same moment `_shared_made` is
# first computed, lets a LATER process recover that same list instead of
# re-deriving it from directory state that has since changed. Named with
# "ckpt" so a failure to remove it (mirrored in the same cleanup loop as
# the `.npz`) still trips the existing "no ckpt/tmp left" hygiene check.
_SHARED_DIRS_MARKER = ".ckpt_atom2d.shared_dirs.json"


def _shared_dirs_read(shared_od):
    """[ckpt-r3] Best-effort read of the sidecar `_shared_made` list from
    an EXISTING `shared_od`. Returns None on any failure (no marker, an
    older shared dir written before this field existed, or a corrupt
    file) -- the caller falls back to computing fresh, exactly as if no
    marker had ever been written."""
    p = os.path.join(shared_od, _SHARED_DIRS_MARKER)
    try:
        with io.open(p, encoding="utf-8") as fh:
            v = json.loads(fh.read())
        return [str(z) for z in v] if isinstance(v, list) else None
    except Exception:
        return None


def _shared_dirs_write(shared_od, made):
    """[ckpt-r3] Inverse of `_shared_dirs_read` -- best-effort, a failure
    here must not end the run (same rule as `_ckpt_save`)."""
    p = os.path.join(shared_od, _SHARED_DIRS_MARKER)
    try:
        with io.open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(made))
    except OSError as _e:
        print("[OPT2D] WARN: shared dirs marker write failed (%s) at %s"
              % (_e, p), flush=True)


# ------------------------------------------------------------ density pipeline
def conic_kernel_fft(n, radius_px):
    """FFT of the periodic-boundary 2D conic filter. It is a constant, so build it only once.

    This is the 2D version of the 1D track (`top_grating_opt.make_filter_matrix`). Its purpose is to
    enforce the minimum linewidth; without it the optimisation diverges into pixel-level noise.
    """
    r = max(1, int(round(radius_px)))
    a = np.arange(n)
    d = np.minimum(a, n - a)                       # periodic distance
    DX, DY = np.meshgrid(d, d, indexing="ij")
    w = np.maximum(0.0, 1.0 - np.sqrt(DX ** 2 + DY ** 2) / float(r))
    w = w / w.sum()
    return np.fft.fft2(w)


def conic_kernel_fft_rect(nx, ny, radius_px):
    """[261003 distinct_atoms] Rectangular (nx, ny) periodic conic filter FFT.

    For nx == ny it is the same array `conic_kernel_fft(nx, r)` returns (same
    distances, same weights, same normalisation). Used only for the linewidth
    penalty of a K = 1 cell with distinct atoms, whose projected density is the
    full (atoms*NXY, NXY) cell, periodic in both axes."""
    r = max(1, int(round(radius_px)))
    ax = np.arange(nx)
    ay = np.arange(ny)
    dx = np.minimum(ax, nx - ax)
    dy = np.minimum(ay, ny - ay)
    DX, DY = np.meshgrid(dx, dy, indexing="ij")
    w = np.maximum(0.0, 1.0 - np.sqrt(DX ** 2 + DY ** 2) / float(r))
    w = w / w.sum()
    return np.fft.fft2(w)


def tanh_project(x, beta, eta=0.5):
    num = npa.tanh(beta * eta) + npa.tanh(beta * (x - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


def density(rho, KF, beta):
    """rho -> conic filter -> tanh projection. `beta` is raised on a continuation schedule."""
    f = npa.real(npa.fft.ifft2(npa.fft.fft2(rho) * KF))
    return tanh_project(f, beta)


# ------------------------------------------------------------ RCWA
def sym_y(a):
    """Enforce y mirror symmetry -- `a` is (nxy, nxy), axis 1 is y.

    [260907] **Why only y.** `_obj` gives phi = 0 through `grcwa.obj(..., theta, 0.0)`,
    so **the plane of incidence is the x-z plane**. The condition forbidding TM <-> TE conversion is
    that the structure is symmetric about that plane of incidence, i.e. invariant under `y -> -y`, which in the
    stored array is axis 1. **Asymmetry in x (axis 0) lies inside the plane of incidence and does not mix polarizations** --
    on the contrary it is a degree of freedom that oblique incidence physically requires, so constraining it only loses performance.

    Measured on 5 pilots of 260907: axis 1 asymmetry 1e-13 ~ 1e-10 (numerical zero),
    axis 0 asymmetry 0.019 ~ 1.067. So it is already satisfied -- but only as an **emergent property**
    (result of a symmetric-ellipse initial value + y-symmetric dynamics). The code does not enforce it,
    so it would break silently if an initial value or a term changed. Here it is turned into a structural guarantee.
    The current results are already symmetric, so **the values do not change**.

    Put **before** `density()` -- the filter is isotropic and preserves symmetry, so changing the order
    gives the same result, but placing it first keeps the optimisation variable itself in the symmetric subspace
    and the gradient symmetric too (with after-the-fact symmetrisation the gradient does not know the symmetry).
    """
    return 0.5 * (a + a[:, ::-1])


# [260921] **Atoms per period.** The job key `atoms_per_period` overrides
# this value in `main()`.
#
# **The default was changed from 2 to 3 (260921).** Because a job that omits the key
# silently running at Lambda/2 is far more dangerous -- at Lambda/2 the
# L3-alone echo `R(+2)` is **0.4438** (1.5e-25 for Lambda/3), which makes the design
# entirely invalid while the result looks plausible. The safe side is made the default.
# If the previous behaviour is needed, **state** `atoms_per_period = 2` in the job.
# All 32 jobs alive at the time of the switch stated the key explicitly (checked).
#
# The super-period is always fixed to `Lambda` (`app * (PERIOD/app)`), and the only thing that changes
# is the **atom pitch** `ly = PERIOD / app` -- so the order numbers
# keep matching the carrier. The atom cell stays a square `ly x ly`.
#
# Why 3 is needed (260921 measured): the condition for no diffraction at oblique incidence is
# `p < lam / (n * (1 + sin(theta))) = 591.1 nm`, and Lambda/2 = 657.7
# exceeds it by 11 %. Lambda/3 = 438.5 passes, and as a result the L3-alone
# `R(+2)` closes from 0.4438 -> 1.5e-25 and the oblique-incidence channel becomes `|rho| = 1`.
ATOMS_PER_PERIOD = 3
# [261003] K = 1 cell with `ATOMS_PER_PERIOD` DISTINCT atoms (job key
# `distinct_atoms`). 0 (default) = the original single atom tiled along x.
# 1 = the design vector holds ATOMS_PER_PERIOD independent atoms, projected
# one by one and laid side by side along x (see `_project_cell`, `_tile_x`,
# `_layer_densities`). Set by main() before the first solve.
DISTINCT_ATOMS = 0
# [260921] L3 sub-layer count. 1 = the original single extruded atom.
# 2 = two stacked cross-sections, which is what unlocks `arg r'` (see the
# module docstring). Both default to the old path, bit-identical.
L3_LAYERS = 1
L3_SPLIT = 0.5          # fraction of H3 in the LOWER sub-layer
L3_ETCH_STACK = 0       # 1 = upper sub-layer only on top of the lower one
L3_FILL_N = 1.0         # background index of the LOWER sub-layer (1.0 = air)
# [260923] Interlayer SiO2 gap (etch-stop) between the two L3 sub-layers.
# Only meaningful when L3_LAYERS == 2. 0 (default) -> no gap layer added,
# bit-identical to the pre-gap path.
L3_GAP_NM = 0.0
N_L3_GAP = 1.4440
# [261002] Where `emask` enters the density pipeline (job key
# `edge_margin_mode`). 0 = 'pre' (default, the 260917 behaviour): multiply the
# RAW variable before the conic filter -- the filter refills the strip, so the
# void is 2*margin px only for margin >= 6 px at a 14 px kernel (measured
# 261002: 4 -> 2 px, 5 -> 8 px, 6 -> 12 px, 7 -> 14 px). 1 = 'post': multiply
# the FILTERED field before the tanh projection. tanh_project(0) is exactly
# 0 for any beta (tanh(beta*eta) + tanh(-beta*eta) cancels), so the projected
# density is exactly 0 on the strip and the periodic void between neighbouring
# cells is exactly 2*margin px. The mask is a 0/1 (nxy, nxy) array, array axis 0
# = x (plane of incidence, tiled ATOMS_PER_PERIOD times in `_obj`), axis 1 = y.
#
# [261002 consumers] The mode used to be the module global `EDGE_MASK_POST`,
# set only inside main(), so any other process that rebuilt the density (the
# LPA superramp, the eval tools) had no way to know it. The mode now travels
# WITH the mask: `EdgeMask` holds the 0/1 array and the order flag, and
# `build_edge_mask` / `edge_mask_spec` / `edge_mask_from_record` are the one
# place a mask is made, so a consumer rebuilds exactly what the optimiser used
# from the keys the driver wrote into atom2d_opt.json (edge_margin_x_px,
# edge_margin_y_px, edge_margin_mode; edge_margin_px stays for old readers).
class EdgeMask(object):
    """0/1 (nxy, nxy) edge mask + the order it is applied in.

    arr   float (nxy, nxy), 1 = usable, 0 = frozen strip. Axis 0 = x, 1 = y.
    post  False ('pre'): multiply the RAW variable before the conic filter.
          True ('post'): multiply the FILTERED field before the projection.
    mx my margin in px at EACH side of axis 0 / axis 1 (>= 1 on at least one).
    Not an ndarray on purpose: a bare ndarray passed as `emask` (every caller
    written before 261002) is read as mode 'pre', the only mode that existed."""
    __slots__ = ("arr", "post", "mx", "my")

    def __init__(self, arr, post, mx, my):
        self.arr = arr
        self.post = bool(post)
        self.mx = int(mx)
        self.my = int(my)

    @property
    def mode(self):
        return "post" if self.post else "pre"

    def mean(self):
        return float(self.arr.mean())


def build_edge_mask(nxy, mx, my, mode="pre"):
    """Edge mask for an `nxy` x `nxy` cell: `mx` px of void at EACH side of
    axis 0 (x), `my` px at each side of axis 1 (y). Returns None when both
    margins are 0 (no mask), so the unmasked path stays the original one.
    Same array the 260917 code built for mx == my: `a[m:n-m, m:n-m] = 1`.
    ValueError on a bad mode or a margin outside 0 <= 2*m < nxy."""
    _mode = (str(mode).strip().lower() or "pre")
    if _mode not in ("pre", "post"):
        raise ValueError("edge_margin_mode must be pre|post (got %r)" % mode)
    mx, my = int(mx), int(my)
    if mx < 0 or my < 0 or 2 * mx >= nxy or 2 * my >= nxy:
        raise ValueError("edge margin x=%d y=%d invalid for NXY %d "
                         "(0 <= 2*margin < NXY)" % (mx, my, nxy))
    if mx == 0 and my == 0:
        return None
    _a = np.zeros((nxy, nxy))
    _a[mx:nxy - mx, my:nxy - my] = 1.0
    return EdgeMask(_a, _mode == "post", mx, my)


_EDGE_KEYS = ("edge_margin_x_px", "edge_margin_y_px", "edge_margin_mode")


def _edge_int(d, k, dflt):
    try:
        return int(round(float(str(d.get(k, "")).strip())))
    except (TypeError, ValueError):
        return int(dflt)


def edge_mask_spec(rec, job=None):
    """(mx, my, mode, source) for ONE atom, from the keys it recorded.

    rec  the atom's atom2d_opt.json (dict). job  optional dict of the job
    file's key = value strings, consulted ONLY when rec carries none of
    edge_margin_px / the three new keys (a hand-made or very old record):
    a json that has edge_margin_px but no new keys was written by a driver
    that ignored them, so a job's new keys must not override it.
    The driver writes the EFFECTIVE margins (a -1 job key already resolved),
    so for a new record the three keys are the answer as they stand.
    source is 'json' (new keys), 'json_legacy' (edge_margin_px only),
    'job' or 'none'. Nothing recorded -> (0, 0, 'pre', 'none'), which is
    `build_edge_mask` None, i.e. no mask."""
    rec = rec or {}
    if any(k in rec for k in _EDGE_KEYS):
        _g = _edge_int(rec, "edge_margin_px", 0)
        _mx = _edge_int(rec, "edge_margin_x_px", -1)
        _my = _edge_int(rec, "edge_margin_y_px", -1)
        _md = str(rec.get("edge_margin_mode", "pre")).strip().lower() or "pre"
        return (_g if _mx < 0 else _mx, _g if _my < 0 else _my, _md, "json")
    if "edge_margin_px" in rec:
        _g = _edge_int(rec, "edge_margin_px", 0)
        return (_g, _g, "pre", "json_legacy")
    if job and any(k in job for k in ("edge_margin_px",) + _EDGE_KEYS):
        _g = _edge_int(job, "edge_margin_px", 0)
        _mx = _edge_int(job, "edge_margin_x_px", -1)
        _my = _edge_int(job, "edge_margin_y_px", -1)
        _md = str(job.get("edge_margin_mode", "pre")).strip().lower() or "pre"
        return (_g if _mx < 0 else _mx, _g if _my < 0 else _my, _md, "job")
    return (0, 0, "pre", "none")


def edge_mask_from_record(rec, nxy, job=None):
    """The EdgeMask (or None) the atom described by `rec` was optimised with.
    See `edge_mask_spec` for where the keys come from."""
    _mx, _my, _md, _src = edge_mask_spec(rec, job)
    return build_edge_mask(nxy, _mx, _my, _md)


def _masked_density(_r, emask, KF, beta):
    """`_r` (already sym_y'd raw variable) -> projected density with the mask.

    emask None -> `density(_r)`. pre: `density(_r * emask)` (the original two
    lines, so the default path is bit-identical). post: filter, then mask,
    then project. Shared by `_project3` (autograd path) and
    `_layer_densities` (reporting) so the two cannot drift apart.
    `emask` is an `EdgeMask` (mode travels with it) or, for callers older than
    261002, a bare 0/1 array, which is read as 'pre'."""
    if emask is None:
        return density(_r, KF, beta)
    if isinstance(emask, EdgeMask):
        _post, _m = emask.post, emask.arr
    else:
        _post, _m = False, emask
    if _post:
        _f = npa.real(npa.fft.ifft2(npa.fft.fft2(_r) * KF))
        return tanh_project(_f * _m, beta)
    return density(_r * _m, KF, beta)


def _edge_void_px(d, axis):
    """Shortest PERIODIC void run (px) through the cell boundary of the 0/1
    density `d` along array `axis` (0 = x, 1 = y), over every line of `d`.
    All-solid-raw worst case in, so this is the guaranteed void: the raw
    variable is clipped to [0, 1] and filter + projection are monotone."""
    _l = np.moveaxis(np.asarray(d) < 0.5, axis, 0)       # (n, lines)
    n = _l.shape[0]
    _o = []
    for j in range(_l.shape[1]):
        v = _l[:, j]
        a = 0
        while a < n and v[a]:
            a += 1
        b = 0
        while b < n and v[n - 1 - b]:
            b += 1
        _o.append(n if a == n else a + b)
    return min(_o)


def _layer_densities(x, nxy, emask, KF, beta):
    """[261003 distinct_atoms] Dispatcher. DISTINCT_ATOMS = 0 (default): the
    original single-atom body, `_layer_densities_atom`, called once on `x`
    (identical result). DISTINCT_ATOMS = 1: `x` is the atom-major vector of
    ATOMS_PER_PERIOD atoms; each atom is projected on its own and the per-layer
    arrays are laid side by side along x, giving full-cell (atoms*nxy, nxy)
    layers (the reporting helpers then measure them periodic over Lambda)."""
    if not DISTINCT_ATOMS:
        return _layer_densities_atom(x, nxy, emask, KF, beta)
    flat = np.asarray(x).reshape(-1)
    app = int(ATOMS_PER_PERIOD)
    per = max(1, int(L3_LAYERS)) * int(nxy) * int(nxy)
    if flat.size != app * per:
        raise ValueError("distinct_atoms: design vector size %d != %d atoms "
                         "x %d" % (flat.size, app, per))
    parts = [_layer_densities_atom(flat[j * per:(j + 1) * per], nxy, emask,
                                   KF, beta) for j in range(app)]
    return [np.concatenate([p[k] for p in parts], axis=0)
            for k in range(len(parts[0]))]


def _layer_densities_atom(x, nxy, emask, KF, beta):
    """Projected density of every L3 sub-layer, AS SIMULATED.

    [260922] The reporting helpers used to project each sub-layer independently
    from the raw variable, so with `l3_etch_stack` on they measured a structure
    that is never simulated -- the smoke printed `fill_frac_each` = 0.036 /
    0.162, an upper layer larger than the lower one, which the constraint
    forbids.  That matters for the gate: what gets built is the intersection,
    and its smallest feature can be narrower than either layer's, so the
    linewidth reading would be optimistic.

    Applies the same containment `_project3` applies.  `sym_y` is deliberately
    NOT applied here, exactly as before, so the flag-off path is unchanged.
    """
    n2 = int(nxy) * int(nxy)
    flat = np.asarray(x).reshape(-1)
    out = []
    for k in range(max(1, int(L3_LAYERS))):
        _rm = flat[k * n2:(k + 1) * n2].reshape(nxy, nxy)
        out.append(np.asarray(_masked_density(_rm, emask, KF, beta)))
    if int(L3_ETCH_STACK) and len(out) > 1:
        out[1] = out[1] * out[0]
    return out


def _min_feature_all(x, nxy, emask, KF, beta, px_nm):
    """Worst measured linewidth over every L3 sub-layer.

    The etch sees each sub-layer separately, so the binding number is the
    WORST of them, not the bottom one.  Single-layer returns exactly the
    old value.
    """
    _d = _layer_densities(x, nxy, emask, KF, beta)
    out = [min_feature_nm(z, px_nm) for z in _d]
    return min(out), out


def _fill_frac_all(x, nxy, emask, KF, beta):
    """Projected fill fraction of every L3 sub-layer.

    [260922] `min_feature_nm` returns `inf` when the projected density has no
    edge at all, and `inf >= gate_nm` PASSES -- so an atom that vanished during
    optimisation was reported as a linewidth pass.  Measured on
    `260921_scdc_l3bi8`: 6 of 50 cells collapsed to all-void and reported the
    bare spacer's numbers (`absR00` 0.18167, `argR00` exactly 180.0) with
    `gate_ok = True`; four shared one `arg tau` and inflated the palette.

    Same preprocessing order as `_min_feature_all` (emask, then `density`) so
    the two measurements cannot drift apart.
    """
    return [float(np.mean(z))
            for z in _layer_densities(x, nxy, emask, KF, beta)]


def _gate_with_fill(mf, gate_nm, ff, lo=0.005, hi=0.995):
    """Linewidth verdict that an empty structure cannot pass.

    A sub-layer whose fill is outside (lo, hi) carries no structure.  If EVERY
    sub-layer is like that there is no atom, and no linewidth claim can be made
    about it.  One empty sub-layer is allowed -- that is a shorter atom, and the
    etch genuinely sees no feature in the empty half.
    """
    _live = [f for f in ff if lo < f < hi]
    return bool(mf >= gate_nm) and bool(_live), _live


def _project3(rho3, nxy, symy, emask, KF, beta):
    """Raw design variable -> projected L3 density (or list of two).

    The single-layer branch is the ORIGINAL three lines verbatim, so the
    default path stays bit-identical; the two-layer branch applies the
    same sym_y / emask / density order to each half of the variable.
    [261002] The emask + density step now lives in `_masked_density`
    (same operations, same order in the default 'pre' mode; 'post' mode
    masks after the filter). sym_y still runs first in both modes and the
    mask is mirror-symmetric in y, so the symmetry survives either order.
    """
    if int(L3_LAYERS) < 2:
        _r = npa.reshape(rho3, (nxy, nxy))
        if symy:
            _r = sym_y(_r)
        return _masked_density(_r, emask, KF, beta)
    n2 = int(nxy) * int(nxy)
    flat = npa.reshape(rho3, (-1,))
    out = []
    for k in (0, 1):
        _r = npa.reshape(flat[k * n2:(k + 1) * n2], (nxy, nxy))
        if symy:
            _r = sym_y(_r)
        out.append(_masked_density(_r, emask, KF, beta))
    # [260922] **2-step etch constraint.** The two sublayers were independent design variables until now
    # and the background is air (`1.0 + rho*(eps3-1)`), so solutions appear where the upper layer
    # puts material over a place where the lower layer is empty. Measured (260921_scdc_l3bi8): of 23 true 2-layer cells,
    # only 5 have `L2` inside `L1` (protrusion < 2 %), median 28.4 %, max
    # 69.9 %, and **6 cells have only the upper layer** on top (a slab floating over a void).
    #
    # In a 2-step etch with 1 deposition + 2 masks, only the places that survived both etches remain as the upper layer,
    # so `L2` is necessarily a subset of `L1`, and then the current model with air background
    # is exact. With 2 depositions the upper layer can be put anywhere, but planarising
    # SiO2 is needed in between, so the background becomes 1.4440 and the model is wrong.
    # That is, **for the air model not to be self-contradictory, the inclusion relation is required.**
    #
    # The product is an intersection of projected densities, so it is differentiable and exact
    # after beta has binarised. With the default 0 this line is not taken, so it is bit-identical to before.
    if int(L3_ETCH_STACK):
        out[1] = out[1] * out[0]
    return out


def _project_cell(x, nxy, symy, emask, KF, beta):
    """[261003 distinct_atoms] Raw design vector -> projected L3 density of the
    WHOLE Lambda cell, in the form `_obj_build` consumes.

    DISTINCT_ATOMS = 0 (default): exactly `_project3` (one atom (nxy, nxy) or a
    list of two sub-layer arrays; `_obj_build` tiles it ATOMS_PER_PERIOD times).
    DISTINCT_ATOMS = 1: `x` is the atom-major vector, atom j = x[j*per:(j+1)*per]
    with per = layers*nxy*nxy. Each atom goes through `_project3` on its own
    (own conic filter, projection, sym_y, edge mask, etch-stack containment) and
    the per-layer results are laid side by side along x (axis 0). The return is
    an (atoms*nxy, nxy) array (1 layer) or a list of two such arrays, which
    `_tile_x` passes through untouched. autograd-safe (slices and concatenate)."""
    if not DISTINCT_ATOMS:
        return _project3(x, nxy, symy, emask, KF, beta)
    app = int(ATOMS_PER_PERIOD)
    nl = max(1, int(L3_LAYERS))
    per = nl * int(nxy) * int(nxy)
    flat = npa.reshape(x, (-1,))
    if flat.shape[0] != app * per:
        raise ValueError("distinct_atoms: design vector size %d != %d atoms "
                         "x %d" % (flat.shape[0], app, per))
    outs = [_project3(flat[j * per:(j + 1) * per], nxy, symy, emask, KF, beta)
            for j in range(app)]
    if nl == 1:
        return npa.concatenate(outs, axis=0)
    return [npa.concatenate([o[k] for o in outs], axis=0) for k in range(nl)]


def _tile_x(_rr, app):
    """[261003 distinct_atoms] One atom (nxy, nxy) tiled `app` times along x --
    the original `npa.concatenate([_rr] * app, axis=0)` -- or, with
    DISTINCT_ATOMS, an already full (app*nxy, nxy) cell used as given."""
    if DISTINCT_ATOMS:
        if _rr.shape[0] != app * _rr.shape[1]:
            raise ValueError("distinct_atoms: layer array %s is not a full "
                             "(%d*nxy, nxy) cell" % (str(_rr.shape), app))
        return _rr
    return npa.concatenate([_rr] * app, axis=0)


# ------------------------------------------------------------ record rebuild (261002 consumers)
# Everything a consumer needs to rebuild what the optimiser simulated from the
# atom's OWN atom2d_opt.json: the module globals `main()` sets from job keys
# (stack geometry) and the projected density (filter, sym_y, edge mask).
_RECORD_GLOBALS = (("atoms_per_period", "ATOMS_PER_PERIOD", int),
                   ("l3_layers", "L3_LAYERS", int),
                   ("l3_split", "L3_SPLIT", float),
                   ("l3_etch_stack", "L3_ETCH_STACK", int),
                   ("l3_fill_n", "L3_FILL_N", float),
                   ("l3_gap_nm", "L3_GAP_NM", float),
                   ("n_l3_gap", "N_L3_GAP", float))


def apply_record_globals(rec, only=None):
    """Set the stack-geometry module globals (`ATOMS_PER_PERIOD`, `L3_*`,
    `N_L3_GAP`) from one atom's record, for the keys the record carries (a
    missing key leaves the global alone). `only` limits it to those global
    names. Returns {global name: previous value}; pass it to
    `restore_record_globals` to undo."""
    prev = {}
    for _k, _g, _cast in _RECORD_GLOBALS:
        if only is not None and _g not in only:
            continue
        if rec.get(_k) is not None:
            prev[_g] = globals()[_g]
            globals()[_g] = _cast(rec[_k])
    return prev


def restore_record_globals(prev):
    globals().update(prev)


def project_recorded(x, rec, beta=64.0, job=None, default_app=None):
    """The projected L3 density the driver realised for the stored raw
    variable `x` (rho3_opt.npy), rebuilt from the atom's own record `rec`
    (atom2d_opt.json): NXY, filt_radius_nm (already gate-raised) -> filter at
    the record's atom pitch, sym_y, l3_layers / l3_etch_stack, and the edge
    mask from `edge_mask_from_record` (`job` = fallback key dict, see
    `edge_mask_spec`). `G.PERIOD` must already be set. `default_app` is the
    atom pitch to assume when the record has no `atoms_per_period` (None =
    the module value; the pre-260921 tools pass 2). Returns one array, or
    a list of two for a 2-layer atom -- the same shape `_project3` returns.
    The stack globals are set only for the call and restored after."""
    nxy = int(rec["NXY"])
    x = np.asarray(x).reshape(-1)
    nl = int(rec.get("l3_layers") or (2 if x.size == 2 * nxy * nxy else 1))
    if x.size != nl * nxy * nxy:
        raise ValueError("rho3 size %d does not match NXY %d x %d layer(s)"
                         % (x.size, nxy, nl))
    app = int(rec.get("atoms_per_period",
                      ATOMS_PER_PERIOD if default_app is None
                      else default_app))
    px_nm = G.PERIOD * 1e3 / float(app) / nxy
    KF = conic_kernel_fft(nxy, max(1.0, float(rec["filt_radius_nm"]) / px_nm))
    symy = bool(int(rec.get("sym_y", 1)))
    emask = edge_mask_from_record(rec, nxy, job)
    prev = {"L3_LAYERS": L3_LAYERS, "L3_ETCH_STACK": L3_ETCH_STACK}
    globals()["L3_LAYERS"] = nl
    globals()["L3_ETCH_STACK"] = int(rec.get("l3_etch_stack", 0) or 0)
    try:
        r3 = _project3(x, nxy, symy, emask, KF, beta)
    finally:
        restore_record_globals(prev)
    return ([np.asarray(z) for z in r3] if isinstance(r3, (list, tuple))
            else np.asarray(r3))


def use_record_projection(rec, setting="auto"):
    """Should an eval tool rebuild this atom from its record
    (`project_recorded`) instead of its own pre-261002 code path?
    'auto' (default): yes iff the record carries one of the keys the driver
    writes since 261002 (edge_margin_x_px / edge_margin_y_px /
    edge_margin_mode) -- an older record keeps the tool's old behaviour
    byte for byte. 'record': always. 'legacy': never."""
    _s = (str(setting or "auto")).strip().lower()
    if _s == "legacy":
        return False
    if _s == "record":
        return True
    if _s != "auto":
        raise ValueError("edge_mask_source must be auto|record|legacy "
                         "(got %r)" % setting)
    return any(k in rec for k in _EDGE_KEYS)


# ------------------------------------------------------------ eval_only solve memo (261002)
# [dedupe] An `eval_only` job used to solve every full-stack object TWICE:
# `fom()` builds them through `_obj`, then `_measure_arms()` builds the SAME
# objects again (same projected density, carrier, angle, polarisation), and
# inside one object the interior solve (`_solve_interior`: one inverse plus
# one matmul at size 2*nG) and the `RT_Solve` flux were repeated per reader
# (`_fid_l3`, `_r00_l3`, `spacer_purity`, `_spacer_probe`, `_ledger`, echo).
# Measured 261001 (sim588 S302, nG 1793): `fom()` alone 4505 s of ~10007 s.
#
# While `_OBJ_MEMO` is a dict, `_obj` returns the already-built object for an
# identical call, and every object it builds gets a per-instance cache for
# `_solve_interior` and `RT_Solve`. Both are pure functions of the built
# object, so a cached value IS the value a recomputation returns; arrays are
# handed out as copies, so no caller can alias the cache.
# `_OBJ_MEMO` is None everywhere else -- in particular during the whole
# optimisation (autograd) path -- and `_obj` then forwards to `_obj_build`
# untouched. `main()` switches the memo on only around the eval_only
# `fom()` + `_measure_arms()` pair and clears it right after, so the objects
# are freed before `_measure_rho` builds its own. A call whose arrays are not
# plain ndarrays (autograd boxes) is never memoised.
_OBJ_MEMO = None


def _obj_memo_begin():
    global _OBJ_MEMO
    _OBJ_MEMO = {}


def _obj_memo_end():
    global _OBJ_MEMO
    if _OBJ_MEMO is not None:
        _OBJ_MEMO.clear()
    _OBJ_MEMO = None


def _memo_digest(h, a):
    """Feed an ndarray (or a list/tuple of them) into hash `h` -- dtype, shape
    and raw bytes. Returns False if anything is not a plain ndarray."""
    if isinstance(a, (list, tuple)):
        h.update(("L%d;" % len(a)).encode("ascii"))
        for z in a:
            if not _memo_digest(h, z):
                return False
        return True
    if type(a) is not np.ndarray or a.dtype == object:
        return False
    h.update(("%s|%s;" % (a.dtype.str, a.shape)).encode("ascii"))
    h.update(np.ascontiguousarray(a).tobytes())
    return True


def _obj_memo_key(nG, rho3, rho12, H, sp, n_sp, eps3, lam, nxy, theta, pol):
    """Key = every input `_obj_build` reads: its arguments (arrays by exact
    byte hash) and the module/scd_c globals it consults. None = do not memoise."""
    try:
        h = hashlib.sha256()
        if not (_memo_digest(h, rho3) and _memo_digest(h, rho12)):
            return None
        scal = (int(nG), tuple(float(z) for z in H), float(sp),
                complex(n_sp), complex(eps3), float(lam), int(nxy),
                float(theta), str(pol), int(ATOMS_PER_PERIOD),
                int(DISTINCT_ATOMS),
                float(L3_SPLIT), float(L3_GAP_NM), float(L3_FILL_N),
                complex(N_L3_GAP), float(G.PERIOD), complex(G.EPS_SUB),
                complex(G.EPS_AIR))
        return (scal, h.hexdigest())
    except (TypeError, ValueError):
        return None


def _memo_wrap_ob(ob):
    """Per-instance cache for the two repeated read-only solves of ONE built
    object. Shadows the methods on the instance only (the vendored class is
    untouched); the first call computes exactly as before. The closures hold
    the object by WEAK reference: a strong one (e.g. a bound method) would make
    a reference cycle and keep the multi-GB object alive until the cyclic gc
    happens to run, which raises the peak memory."""
    _w = weakref.ref(ob)
    _f_int = type(ob)._solve_interior
    _f_rt = type(ob).RT_Solve
    _ic = {}
    _rc = {}

    def _solve_interior(which_layer):
        k = int(which_layer)
        v = _ic.get(k)
        if v is None:
            ai, bi = _f_int(_w(), which_layer)
            if type(ai) is not np.ndarray or type(bi) is not np.ndarray:
                return ai, bi
            _ic[k] = (ai.copy(), bi.copy())
            return ai, bi
        return v[0].copy(), v[1].copy()

    def RT_Solve(normalize=0, byorder=0):
        k = (int(normalize), int(byorder))
        v = _rc.get(k)
        if v is None:
            R, T = _f_rt(_w(), normalize=normalize, byorder=byorder)
            if not (isinstance(R, (np.ndarray, np.generic, float, complex))
                    and isinstance(T, (np.ndarray, np.generic, float, complex))):
                return R, T                      # never cache anything traced
            _rc[k] = (R.copy() if isinstance(R, np.ndarray) else R,
                      T.copy() if isinstance(T, np.ndarray) else T)
            return R, T
        return (v[0].copy() if isinstance(v[0], np.ndarray) else v[0],
                v[1].copy() if isinstance(v[1], np.ndarray) else v[1])

    ob._solve_interior = _solve_interior
    ob.RT_Solve = RT_Solve


def _obj_build(nG, rho3, rho12, H, sp, n_sp, eps3, lam, nxy, theta, pol="TM"):
    # [260825] The cell is `Lambda_x` x `Lambda_y` and `Lambda_y = Lambda_x / app`.
    # `app` atoms fit along x, so `Lx = PERIOD` (= app * ly) is correct.
    # (The first version had a bug writing `2 * PERIOD` -- the cell became twice as wide and the order
    #  numbers were entirely shifted. The FoM still rises, so a smoke test does not catch it.)
    # [260921] `Lx` is **always `G.PERIOD`** regardless of `app`. The constant is used
    # as is instead of a multiplication so that there is no 1 ulp error even at app = 3.
    app = int(ATOMS_PER_PERIOD)
    nx = app * nxy
    ly = G.PERIOD / float(app)
    ob = grcwa.obj(int(nG), [G.PERIOD, 0.0],
                   [0.0, ly], 1.0 / lam, theta, 0.0, verbose=0)
    if _LI_ON:
        _li_apply(ob)
    ob.Add_LayerUniform(0.5, G.EPS_SUB)
    ob.Add_LayerGrid(H[0], nx, nxy)
    ob.Add_LayerGrid(H[1], nx, nxy)
    if sp > 0:
        ob.Add_LayerUniform(sp, complex(n_sp) ** 2)
    # [260921] L3 may be ONE extruded layer or TWO stacked cross-sections.
    # The spacer stays layer index 3 either way, so `GetAmplitudes(3, sp)`
    # in `_fid_l3` / `_r00_l3` is unaffected.
    _r3 = rho3 if isinstance(rho3, (list, tuple)) else [rho3]
    if len(_r3) == 2:
        _f = float(L3_SPLIT)
        _h3 = [H[2] * _f, H[2] * (1.0 - _f)]
    else:
        _h3 = [H[2]]
    # [260923] SiO2 etch-stop gap between the two L3 sub-layers (process:
    # a-Si / SiO2 etch-stop / a-Si -- mask B etches the top a-Si only and
    # stops on SiO2; mask A then etches SiO2 + the lower a-Si together).
    # The gap therefore SURVIVES exactly where sub-layer A survives, so it
    # is a GRID layer patterned by rhoA (`_r3[0]`), not a new design
    # variable -- its eps grid is built in the `es` loop below.
    # Only inserted when both sub-layers exist and the gap is > 0 --
    # otherwise the loop is exactly the original one, bit-identical.
    _gap_um = float(L3_GAP_NM) * 1e-3
    _has_gap = (len(_h3) == 2 and _gap_um > 0.0)
    for _k3, _hh in enumerate(_h3):
        if _k3 == 1 and _has_gap:
            ob.Add_LayerGrid(_gap_um, nx, nxy)
        ob.Add_LayerGrid(_hh, nx, nxy)
    ob.Add_LayerUniform(0.5, G.EPS_AIR)
    ob.Init_Setup()
    es = []
    for i in (0, 1):
        # [260918] L1/L2 may be **2-dimensional**. If 1-dimensional, spread along
        # y as before (bit-identical); if it is already `(nx, nxy)`, use it as is.
        # The 2D carrier is made by `Main_rcwa_l3_carrier2d.py` -- measured
        # (260918): y freedom raises TM spacer purity from 0.960885 (fail, 0 of 9 in the
        # corpus) -> **0.999947 (pass)**, and for TE improves purity and product at the same time.
        # This branch is needed to use that carrier in the palette optimisation.
        _e12 = rho12[i]
        if npa.ndim(_e12) == 1:
            _e12 = _e12[:, None] * npa.ones((1, nxy))
        es.append(1.444 ** 2 + _e12 * (3.48 ** 2 - 1.444 ** 2))
    # `app` atoms -- the same element `app` times along x (period Lambda_y = ly)
    # [260922] **Planarising background.** The 2-deposition process must pattern the lower layer, then cover it with SiO2
    # and planarise by CMP before the upper layer can be placed freely. Then the voids around the
    # lower layer are **SiO2, not air**. The etch-OFF cells so far modelled it as air,
    # so the upper layer was a structure floating over vacuum -- the 8/8 palettes reached by that
    # route were never computed with their own physics.
    # The upper-layer background is left as air (nothing is deposited on it). A single-layer L3
    # has no planarisation step, so it is left alone. With the default 1.0 it is bit-identical to before.
    _fill_eps = float(L3_FILL_N) ** 2
    for _k, _rr in enumerate(_r3):
        _eb = _fill_eps if (_k == 0 and len(_r3) > 1) else 1.0
        if _k == 1 and _has_gap:
            # [260923] Gap eps grid -- patterned by rhoA (`_r3[0]`, the SAME
            # array used to build sub-layer A above), background `_fill_eps`
            # (the same L3 background sub-layer A itself sits on). No new
            # design variable.
            _gape = complex(N_L3_GAP) ** 2
            es.append(_fill_eps + _tile_x(_r3[0], app)
                      * (_gape - _fill_eps))
        es.append(_eb + _tile_x(_rr, app) * (eps3 - _eb))
    ob.GridLayer_geteps(npa.concatenate([npa.reshape(e, (-1,)) for e in es]))
    # [260826] Polarization. Same convention as scd_c_grating._solve:
    #   TM -> (p, s) = (1, 0),  TE -> (0, 1)
    if pol == "TE":
        ob.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
    else:
        ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    return ob


def _obj(nG, rho3, rho12, H, sp, n_sp, eps3, lam, nxy, theta, pol="TM"):
    # [dedupe 261002] see the memo block above `_obj_build`. With the memo off
    # (`_OBJ_MEMO is None`, every path except eval_only) this is a pure forward.
    memo = _OBJ_MEMO
    if memo is None:
        return _obj_build(nG, rho3, rho12, H, sp, n_sp, eps3, lam, nxy,
                          theta, pol)
    key = _obj_memo_key(nG, rho3, rho12, H, sp, n_sp, eps3, lam, nxy,
                        theta, pol)
    if key is None:
        return _obj_build(nG, rho3, rho12, H, sp, n_sp, eps3, lam, nxy,
                          theta, pol)
    ob = memo.get(key)
    if ob is None:
        ob = _obj_build(nG, rho3, rho12, H, sp, n_sp, eps3, lam, nxy,
                        theta, pol)
        _memo_wrap_ob(ob)
        memo[key] = ob
    return ob


def _amp(ob, order, refl=False, pol="TM"):
    """Complex amplitude of order `order` (y order 0).

    [260826] The amplitude vector has length 2*nG, with the **first half the x component and the second half the y component**.
    In the 1D classical mount only y survives for TM and only x for TE (see the scd_c_grating._args
    comment). Previously `v[i + ob.nG]` **hard-coded TM**, so the TE lane could
    not be computed at all.
    """
    if refl:
        _a, v = ob.GetAmplitudes(0, 0.0)
    else:
        v, _b = ob.GetAmplitudes(ob.Layer_N - 1, 0.0)
    off = 0 if pol == "TE" else ob.nG
    for i, g in enumerate(ob.G):
        if int(g[0]) == order and int(g[1]) == 0:
            return v[i + off]
    return 0.0 * v[0]


def _Ttot(ob):
    """Total transmission. It must pass through autograd, so grcwa's `RT_Solve` is used as is."""
    _R, T = ob.RT_Solve(normalize=1)
    return npa.real(T)


def _ord_power(ob, arr, order):
    """**Normalised power** of order `order` (y order 0). Same indexing rule as `_amp`.

    [260917] Taken from the array given by `RT_Solve(normalize=1, byorder=1)` in `scd_c_grating:660`.
    **It is not `|amplitude|^2`** -- the ratio of the two is measured as 2.4983/1.4440
    and the factor comes from `kz/n` (`scd_c_grating:680`).
    """
    for i, g in enumerate(ob.G):
        if int(g[0]) == order and int(g[1]) == 0:
            return float(npa.real(arr[i]))
    return 0.0


def _ledger(ob, tag):
    """Normalised R/T per order and the totals. Holds only propagating orders (substrate 3, air 1).

    [260917] What must be read on the output side is not the "purity" but **where the power goes** --
    air has **only one** propagating order, `m = -1` (lam/Lambda = 1.1784 > 1), so the air
    purity is identically 1 and measures nothing. A trapped parasitic does not accumulate in the lossless steady
    state but **leaks down into the substrate**, so measured (260917, nG 401)
    **89.8 %** of the reflection 0.956 leaves at **54.69 deg**.
    """
    R, T = ob.RT_Solve(normalize=1, byorder=1)
    out = {"%s_R_total" % tag: float(npa.real(npa.sum(R))),
           "%s_T_total" % tag: float(npa.real(npa.sum(T)))}
    for m in (-3, -2, -1, 0, 1, 2, 3):
        for nm, arr in (("R", R), ("T", T)):
            v = _ord_power(ob, arr, m)
            if v > 1e-9:
                out["%s_%s_m%+d" % (tag, nm, m)] = v
    return out


_LW_ETA_E = 0.75      # erosion threshold (> 0.5 -> erodes the solid)
_LW_ETA_D = 0.25      # dilation threshold (< 0.5 -> thickens the solid)


def _lw_pen(r3, KG, beta):
    """[260917] **Differentiable** surrogate of the linewidth gate.

    It is a direct port of `Main_rcwa_l3_atom2d_ffwin._lw_pen`. Why this file needs it too:
    if the border is blocked by `edge_margin_px`, the optimisation pushes material inward
    and creates fine structure, while `min_feature_nm()` is binarisation + morphology,
    which breaks the autograd tape so it cannot be called inside the objective, and is used only for reporting.
    Measured (260917, mask 4): passes of the true minimum linewidth fell from 6/8 -> 5/8.

    The conic filter `KG` is used as the structuring element, and raising the eta of `tanh_project` above 0.5 gives
    erosion, lowering it below gives dilation:

        op  = P(KG*P(KG*r3, eta_e), eta_d)   solids thinner than the gate disappear
        cl  = P(KG*P(KG*r3, eta_d), eta_e)   voids thinner than the gate are filled
        pen = <relu(r3 - op)> + <relu(cl - r3)>

    that is, the **area fraction of solid/void thinner than the gate**. Measured calibration of ffwin
    (NXY 96, pixel 6.851 nm, gate 57 nm, beta 64): 0.0593 for a 54.8 nm strip,
    0.00004 at 61.7 nm -- the knee stands within one pixel of the gate. A clean
    disk gives 0.0066~0.0081 (perimeter residual) and noise shapes 0.195~1.000, separated by 25~130 times.
    Voids are also looked at because measuring only the solid would miss thin gaps.
    """
    def _f(a):
        return npa.real(npa.fft.ifft2(npa.fft.fft2(a) * KG))
    op = tanh_project(_f(tanh_project(_f(r3), beta, _LW_ETA_E)),
                      beta, _LW_ETA_D)
    cl = tanh_project(_f(tanh_project(_f(r3), beta, _LW_ETA_D)),
                      beta, _LW_ETA_E)
    return (npa.mean(npa.maximum(r3 - op, 0.0))
            + npa.mean(npa.maximum(cl - r3, 0.0)))


def _ph(t, target_deg):
    """Phase target term. A form without a branch point (avoids the `angle()` discontinuity).

    It is 0 when `arg t = target` and 2 when opposite.
    """
    th = target_deg * math.pi / 180.0
    return 1.0 - (npa.real(t) * npa.cos(th)
                  + npa.imag(t) * npa.sin(th)) / (npa.abs(t) + 1e-12)


def _ph_esc(t, target_deg):
    """[260920] Antipode escape term. Uses the same projection as `_ph` and has no branch.

    By subtracting `_ph = 1 - cos(d)` it maximises `cos(d)`, but its derivative `-sin(d)` is
    0 not only at `d = 0` but **also at `d = 180`**, so the antipode is an exact stationary point.
    Measured 260920: at spacer 465, target 145~165, all 6 independent starts
    landed at target+185 deg. A smooth 2pi-periodic penalty must have two or more critical
    points, so the antipode itself cannot be removed -- giving a finite gradient is all that can be done.

        g(d) = sin(d) * (1 - cos(d)) / 2
        d = 0   : g = 0, g' = 0   -> the target remains an unbiased minimum
        d = 180 : g = 0, g' = -1  -> a finite escape force of constant sign

    A plain `sin(d)` also escapes but pushes the minimum by atan(eps) and violates the target.

    **The usable range is measured by `w_phesc / w_phi`, with a measured upper limit of 1.60.** Below that
    the target remains the exact global minimum, and at 2.00 the minimum jumps to -90 deg (260920 scan).
    The local analysis -- the second critical point enters at 4/(3*pi) = 0.424 -- is an **underestimate**:
    that point is only a local minimum for a while, and the point where it becomes the global minimum is 1.60. Use the measured
    value. With a margin of 3x, `w_phesc <= 0.5 * w_phi` is recommended (100 if w_phi=200).
    The escape force only has to be nonzero, so it is better to start small.

    **[260920] Measured result -- this term does not solve the antipodal landing. Do not use it.**
    `260919_scdc_phesc`, spacer 465, target 150 (cell that collapsed to 333):

        w_phesc   ratio  landing phi_c
        none      --       -26.0
        2       0.01       -26.0
        10      0.05       -26.2
        50      0.25       -26.8
        100     0.50       -27.5

    The landing moves by 1.5 deg over a 50x range. The diagnosis was incomplete: the cell is at Delta = 183 deg,
    not exactly 180, and there the slope of `_ph` itself is already -sin(183) = +0.052.
    What holds it is not the stationarity of the phase term but the **basin of the whole objective** made together by
    `w_rmag` / `w_rarg` / `w_pure` / `w_echo2` /
    `w_fid`. What actually succeeded in escaping at target 150 was `w_fid = 5000`, and that is not a phase term
    (`260919_scdc_wallpush`: both starts landed at 148.1 / 148.4).

    The term itself is kept because it was proven harmless: 52 result values are bit-identical at the default 0.
    It can be used for other purposes, but **do not retry it as a solution to the antipodal problem.**
    """
    th = target_deg * math.pi / 180.0
    n = npa.abs(t) + 1e-12
    c = (npa.real(t) * npa.cos(th) + npa.imag(t) * npa.sin(th)) / n
    s = (npa.imag(t) * npa.cos(th) - npa.real(t) * npa.sin(th)) / n
    return s * (1.0 - c) * 0.5


def _wrap180(d):
    """[260926] Wrap an angle in degrees to (-180, 180]. Report-only helper
    for the `opt_terms.tsv` per-iteration capture below -- not on the
    autograd path."""
    return (d + 180.0) % 360.0 - 180.0


# [261001] **Early-stop window test.** See the `es_window` docstring block
# near the top of this file. `buf` is a list of the last W+1 ELIGIBLE
# per-iteration snapshots (oldest first, current iteration last), each a
# small dict built by the training loop from `_ITER_CAPTURE`. Returns True
# iff every criterion this job actually tracks is satisfied -- the caller
# still has to measure the linewidth/fill gate on the current x before
# actually stopping (this function never touches x).
def _es_ready(buf, dA_deg, dfom_rel, dprod_rel, fid_gate, fid_margin,
              dfid, echo_gate_db, echo_margin_db, decho_db):
    cur, old = buf[-1], buf[0]
    if max(abs(_wrap180(b["A_deg"] - cur["A_deg"])) for b in buf) > dA_deg:
        return False
    if abs(cur["fom_total"] - old["fom_total"]) > dfom_rel * abs(cur["fom_total"]):
        return False
    if abs(cur["prod"] - old["prod"]) > dprod_rel * abs(cur["prod"]):
        return False
    for _key in ("fid_l3", "fid_l3_m"):
        if _key not in cur:
            continue                                    # not tracked by this job
        if min(b[_key] for b in buf) < fid_gate + fid_margin:
            if max(abs(b[_key] - cur[_key]) for b in buf) > dfid:
                return False
    if echo_gate_db is not None and "echo_a2_db" in cur:
        if max(b["echo_a2_db"] for b in buf) > echo_gate_db - echo_margin_db:
            if max(abs(b["echo_a2_db"] - cur["echo_a2_db"])
                   for b in buf) > decho_db:
                return False
    return True


# [260926] **Per-iteration term capture for `opt_terms.tsv`.** `fom()` fills
# this dict from quantities it already computes -- no extra solve, no change
# to `val` or its gradient (see the capture lines inside `fom()`). The
# training loop reads it right after each `value_and_grad` call, which is
# exactly one `fom()` call per iteration, so the dict always reflects the
# iteration that just ran. `_OPT_TERMS_COLS` is the fixed column order;
# `_OPT_TERMS_INT_COLS` are written as ints, everything else as float
# (`nan` when a schedule/weight-gated term did not run this iteration).
_ITER_CAPTURE = {}
_OPT_TERMS_COLS = (
    "iter", "beta", "nG_stage", "fom_total", "A_deg", "dA_to_target",
    "B_deg", "dpsi_deg", "amp_tA", "amp_rBtB", "absR00", "argR00_deg",
    "echo_a2_db", "fid_l3", "fid_l3_m", "phi", "dpsi", "echo2", "fid",
    "fid_m", "r00mag", "r00arg", "r00win", "lw", "amp", "eff",
)
_OPT_TERMS_INT_COLS = ("iter", "nG_stage")


def _write_opt_terms_row(fh, it, first=False):
    """[260926] Append one row of `_ITER_CAPTURE` to `opt_terms.tsv`, next
    to (but never touching) `opt_log.dat`. Writes the header on the first
    call. Report-only -- does not read from or affect the training path.
    [ckpt] `first=True` forces the header even when `it != 1` -- needed
    only when a resume rebuilds this file from empty (the local
    iteration history was not shipped with the checkpoint, see
    `_ckpt_save`). Default False keeps every existing call byte-identical."""
    if first or it == 1:
        fh.write("\t".join(_OPT_TERMS_COLS) + "\n")
    row = dict(_ITER_CAPTURE)
    row["iter"] = it
    fh.write("\t".join(
        ("%d" % int(row[c])) if c in _OPT_TERMS_INT_COLS
        else ("%.6f" % float(row.get(c, float("nan"))))
        for c in _OPT_TERMS_COLS) + "\n")
    fh.flush()


def make_fom(nG, rho12, H, sp, n_sp, eps3, lam, nxy, KF, w, pol="TM",
             symy=True, emask=None, KG=None):
    """`(rho3, beta) -> FoM`. **Solve only once per angle and read transmission and reflection together.**

    [260826] **Per-lane arm definition** (same as scd_c_grating3 / fullstack2d):

        TM lane (fom_extract3)  -- the side diffracted by the input grating. 54.69 deg in the slab
            A = arg t(-1) @ +th      B = arg r(0) @ +th + arg t(+1) @ -th
        TE lane (fom_split3)    -- the side transmitted through the input grating. Normal in the slab
            A = arg t(0)  @ 0        B = arg r(+1) @ 0  + arg t(+1) @ -th

    **Both the order and the angle differ.** The previous version had only the TM arm and `_amp` also
    hard-coded TM, so TE could not be computed at all (a job labelled TE was
    made in that state and then withdrawn).

    [260826] **Amplitude target.** If `amp_target > 0`, |t| is **matched** to that value instead of being
    **maximised**. Library elements having amplitudes that all differ
    causes aperture-illumination non-uniformity (measured TM 0.0033 / TE 0.0299 dB) and makes elements incomparable,
    so it is better to gather all elements at the same amplitude. If 0, maximise as before.
    """
    def fom(rho3, beta, lw_ramp=0.0):
        # [260907] y mirror symmetry -- being symmetry about the plane of incidence (x-z), it makes the TM <-> TE
        # conversion structurally 0. See the `sym_y` docstring. x is not touched.
        # [260917] EDGE MARGIN -- a hard mask that confines the atom to an island. It must be multiplied after the
        # symmetriser and **before** `density()`, so that the filter and the projection both see the border set to 0
        # (if multiplied after, the filter could push material back to the edge). The reason for reducing
        # the feasible set itself rather than adding a penalty follows the measured evidence in the `ffwin` comment.
        # If `emask is None` (edge_margin_px = 0) it is bit-identical to the previous path.
        r3 = _project_cell(rho3, nxy, symy, emask, KF, beta)
        if pol == "TE":
            ob0 = _obj(nG, r3, rho12, H, sp, n_sp, eps3, lam, nxy, 0.0, "TE")
            tA = _amp(ob0, 0, pol="TE")                   # t(0) @ 0
            rB = _amp(ob0, +1, refl=True, pol="TE")       # r(+1) @ 0
            obA = ob0
            obm = _obj(nG, r3, rho12, H, sp, n_sp, eps3, lam, nxy,
                       -G.THETA_SUB, "TE")
            tB = _amp(obm, +1, pol="TE")                  # t(+1) @ -th
        else:
            obp = _obj(nG, r3, rho12, H, sp, n_sp, eps3, lam, nxy,
                       +G.THETA_SUB, "TM")
            tA = _amp(obp, -1, pol="TM")                  # t(-1) @ +th
            rB = _amp(obp, 0, refl=True, pol="TM")        # r(0)  @ +th
            obA = obp
            obm = _obj(nG, r3, rho12, H, sp, n_sp, eps3, lam, nxy,
                       -G.THETA_SUB, "TM")
            tB = _amp(obm, +1, pol="TM")                  # t(+1) @ -th
        # Total transmission at the A-arm angle. Not computed unless `T_floor` is used (cost per iteration).
        Ttot = _Ttot(obA) if w.get("T_floor", 0.0) > 0 else 0.0
        # A = arg tA, B = arg rB + arg tB
        # phi_c = (A+B)/2, dpsi = B - A -> the two targets are placed on tA and rB*tB
        tgt_A = w["phi"] - 0.5 * w["dpsi_t"]
        tgt_B = w["phi"] + 0.5 * w["dpsi_t"]
        # [260926] **opt_terms.tsv capture -- measured arm state.** `tA`,
        # `rB`, `tB` are already computed above for every call (no gating),
        # so this is report-only postprocessing, not an extra solve.
        _row = {}
        _tAc = complex(G._untraced(tA))
        _rBtBc = complex(G._untraced(rB)) * complex(G._untraced(tB))
        _row["A_deg"] = math.degrees(math.atan2(_tAc.imag, _tAc.real))
        _row["B_deg"] = math.degrees(math.atan2(_rBtBc.imag, _rBtBc.real))
        _row["dA_to_target"] = _wrap180(_row["A_deg"] - tgt_A)
        _row["dpsi_deg"] = _wrap180(_row["B_deg"] - _row["A_deg"])
        _row["amp_tA"] = abs(_tAc)
        _row["amp_rBtB"] = abs(_rBtBc)
        at = w.get("amp_t", 0.0)
        pt = w.get("prod_t", 0.0)
        if pt > 0:
            # [260826] **The product is the signal.** The differential interference term amplitude is `2|A||B|`, so if one
            # arm collapses there is no signal however well the phase is matched. The previous form, which targeted only
            # `|t_A|` and weakly compensated `|r_B|` with `w_r0`,
            # lets the optimiser sell off the reference arm -- measured TE arm ratio 55:1,
            # product 0.0068, i.e. **1/3.4 of the disk median 0.0232**. TM was also 0.0667,
            # short of the disk median 0.1092. Both lanes were worse than the disk.
            #
            # **Normalise by relative error.** The absolute squared error is at most about 0.01 since the product is on the
            # 0.02~0.11 scale, whereas the phase term is `_ph` 0~2 with weight 2, i.e. 0~4.
            # With a difference of several hundred times the amplitude is practically ignored -- that was the core of the defect.
            # The relative error is 1 when the product is 0 and 1 when it is twice the target, so it has
            # the same order of magnitude as the phase term.
            val = -w["amp"] * (npa.abs(tA) * npa.abs(rB * tB) / pt - 1.0) ** 2
        elif at > 0:
            val = -w["amp"] * (npa.abs(tA) - at) ** 2     # gather at the target amplitude
        else:
            val = w["amp"] * npa.abs(tA) ** 2             # previous: maximise
        # [260926] opt_terms.tsv capture -- this branch always runs (not
        # weight-gated), so "amp" is never nan.
        _row["amp"] = float(G._untraced(val))
        # If `prod_t` is used, this term is subsumed in the product target, so set it to 0 in the job.
        val = val + w["r0"] * npa.abs(rB) ** 2
        # [260917] **Reflection target -- gather all elements at the same complex r.**
        #
        # Why needed. `prod_target` constrains only the product `|tA| * |rB tB|`, so `|tB|`
        # trades off against `|rB|`. Measured (260917, o2d8 8 levels): the product was kept but the bare-atom
        # reflection scattered over **0.014 ~ 0.383 (28x)**, and when that palette was applied as the termination
        # to re-score the carrier, the efficiency collapsed from **85.0 -> 54.7 %**
        # (the worst element has `|r| = 0.0137`, so `R_go` is practically 0). 4 gates 3/4 ->
        # 1/4. `w_r0` is a term that **raises** `|rB|`, so it cannot prevent this.
        #
        # Physically this is also the requirement: in `signal = sqrt(ep * R_go * em)`
        # `R_go` is set by the reflection of that element, so if `rB` differs per element, **the delay-arm
        # amplitude differs per zone** -- aperture-illumination non-uniformity and interference visibility loss.
        # This is the reason the docstring of this file already says "it is better to gather all elements
        # at the same amplitude", and the product alone does not guarantee it.
        #
        # The formula is the `w_rsplit > 0` branch of `Main_rcwa_l3_atom2d_ffwin.py:474` moved over
        # as is -- magnitude and phase are penalised **separately**, and the phase is measured by `_ph` (range 0~2,
        # no branch point). If magnitude/phase are lumped as `|r - r_t|^2`,
        # a factor `2|r||r_t|` (= 0.049) attaches to the angle gradient and the strengths of the two axes
        # differ by 20x -- a trap ffwin met in 260916, so here they are split from the start.
        #
        # **At the default 0 neither term is computed, so it is bit-identical to the previous path.**
        if w.get("w_rmag", 0.0) > 0.0:
            val = val - w["w_rmag"] * (npa.abs(rB) - w["absR_t"]) ** 2
        if w.get("w_rarg", 0.0) > 0.0:
            val = val - w["w_rarg"] * _ph(rB, w["argR_t"])
        # [260929] efficiency hinge -- see driver docstring
        # (w_eff/eff_gate_pct). Uses the TRACED tA/rB/tB (the same
        # values already used by the amp/prod term above), NOT the
        # _untraced copies -- so this term stays inside autograd.
        # Same shape as the echo2 hinge below: a linear one-sided
        # hinge in percentage points, exactly 0 once the gate is met.
        # Default w_eff = 0 skips the block entirely -- old path
        # bit-identical.
        if w.get("w_eff", 0.0) > 0.0:
            _eff = 843.167 * npa.abs(tA) * npa.abs(rB * tB)
            _pen_eff = w["w_eff"] * npa.maximum(
                w["eff_gate_pct"] - _eff, 0.0)
            val = val - _pen_eff
            _row["eff"] = float(G._untraced(_pen_eff))
        # [260918] **echo term -- decision.** With the ported instrumentation, measuring the echo on the actual L3
        # grating for the first time gave a median over 231 wscreen_TE cells of
        # **-13.47 dB**, with only **6.1 %** passing the gate -20.5. The objective had none of
        # the four gate terms, so that is the floor.
        #
        # **Why `_amp`.** The echo is `R(+2)` of the `-th` solution and `obm` is
        # already solved -- the cost is only reading one more order. `RT_Solve` is an
        # all-order RT and would charge the same per-iteration cost as `_Ttot`.
        #
        # **Why a hinge.** It penalises only the amount beyond the gate and leaves everything below free
        # -- the ceiling version of what `T_floor` does for the floor. The moment the gate
        # is satisfied the term is exactly 0 and does not fight the product target.
        # A linear term pushing indefinitely toward 0 is the very form in which `w_exit` collapsed.
        #
        # **Why dB.** A linear excess has too large a dynamic range -- measured
        # (260918, own run of the driver): TE fs8f_T045 is 3.3 times the gate,
        # TM fs9_T090 is **24.1 times**. With a relative quadratic hinge the penalties are
        # 5.3 and 533, so one weight cannot hold both ends -- on the far side
        # it overwhelms all other terms and near the gate it vanishes. It is the same form as the order-of-magnitude
        # defect the product target suffered in 260826. Measured in dB, 3.3 times is
        # 5.2 dB and 24.1 times is 13.8 dB, so **the dynamic range shrinks to 2.7x**,
        # and since the gate itself is already defined in dB the units also agree.
        # The shape is the same one-sided hinge as `T_floor` -- 0 when the gate is satisfied.
        #
        # **At the default 0 the computation itself is skipped, so it is bit-identical to the previous path.**
        if w.get("w_echo2", 0.0) > 0.0 and w.get("echo_a2_gate", 0.0) > 0:
            _r2 = _amp(obm, +2, refl=True, pol=pol)
            _e2 = npa.real(_r2 * npa.conj(_r2))
            _ex = npa.maximum(10.0 * npa.log10(
                (_e2 + 1e-12) / w["echo_a2_gate"]), 0.0)
            _pen_echo2 = w["w_echo2"] * _ex
            val = val - _pen_echo2
            # [260926] opt_terms.tsv capture -- only reachable when this
            # block runs (schedule/weight-gated), matching the spec's nan
            # rule for the other iterations.
            _row["echo2"] = float(G._untraced(_pen_echo2))
            _row["echo_a2_db"] = 10.0 * math.log10(
                float(G._untraced(_e2)) + 1e-12)
        # [260917] **Linewidth penalty -- turned on only near the end.** For the same reason as ffwin, the early and
        # middle search is left free and raised linearly from `lw_start_frac` on. Before that
        # `lw_ramp` is exactly 0, so **the computation itself is skipped** (with the default
        # `w_lw = 0` it is always 0 -> bit-identical to the previous path).
        if w.get("w_lw", 0.0) > 0.0 and lw_ramp > 0.0 and KG is not None:
            # [260922] With 2 layers `r3` is a list and `_lw_pen` accepts only an array
            # (`npa.fft.fft2` rejects a list -- this actually
            # killed 15 cells). The penalty is an area fraction, so it is measured per layer and averaged.
            # The 1-layer branch is left alone so that the default path is bit-identical.
            if isinstance(r3, (list, tuple)):
                _lwp = sum(_lw_pen(_z, KG, beta) for _z in r3)
                _lwp = _lwp / float(len(r3))
            else:
                _lwp = _lw_pen(r3, KG, beta)
            _pen_lw = w["w_lw"] * lw_ramp * _lwp
            val = val - _pen_lw
            # [260926] opt_terms.tsv capture -- gated by `lw_ramp > 0`
            # (`lw_start_frac`), so most iterations leave "lw" nan.
            _row["lw"] = float(G._untraced(_pen_lw))
        # [260826] **Breakdown avoidance -- move away from the resonance.**
        # The breakdown (LPA departure) is defined only with neighbouring atoms, so it cannot be
        # computed directly in a single-atom optimisation. But a 38-row measurement of the r1 batch gave a proxy:
        #
        #     Spearman |breakdown| vs T_total  **-0.822**   (vs gap it is -0.703)
        #     T_total < 0.60  -> |breakdown| median 3.490 dB (max 17.05)
        #     T_total >= 0.85 -> |breakdown| median 0.223 dB (max  0.82)
        #
        # `T_total` is a **quantity computed from a single atom**, so it enters the FoM directly.
        # It penalises only the amount below the floor (`T_floor`) -- above it is left free
        # (the product target already holds the amplitude, so there is no reason to push twice).
        tf = w.get("T_floor", 0.0)
        if tf > 0:
            short = npa.maximum(0.0, tf - Ttot) / tf
            val = val - w.get("w_tmin", 0.0) * short ** 2
        # [260918] **purity -- illumination purity at the face where L3 will sit.**
        #
        # Why add it now. Measured on 18 `spw2` cells (260918): re-scoring with L3 **fixed**
        # at the same thickness passes purity 27/27, but **re-optimising** at the same thickness
        # collapses it to **2/18**. It is not the thickness -- this
        # objective has no purity term, so the optimiser sells purity for the product / `|rB|` target.
        # Then the spacer ranking of that block means
        # nothing.
        #
        # `spacer_purity` returns a **traced value** (in 260918 its internal
        # `float(...)` safety check was changed to `_untraced` to pass the box, and
        # d(purity)/dt was measured to agree with the central difference to 4e-11).
        #
        # Uses the **same global save/restore** as `_measure_arms`. If the two places diverge,
        # the layer numbers go out of step and the wrong face is silently read.
        #
        # At the default 0 there is neither a solve nor a call, so it is **bit-identical** to the previous path.
        if w.get("w_pure", 0.0) > 0.0 and sp > 0:
            _kp = (G.OUT_SPACER_UM, G.OUT_SPACER_EPS, G.OUT_L3_UM,
                   G.OUT_L3B_UM, G.OUT_L3_GAP_UM)
            try:
                G.OUT_SPACER_UM = float(G._untraced(sp))
                G.OUT_SPACER_EPS = complex(n_sp) ** 2
                # [260921] With a 2-layer atom there are two layers behind the spacer. `spacer_purity`
                # counts layer numbers from the top, so both must be told (the same
                # phase structure is already used by the `set_out_l3b` 2-layer termination).
                # [260923] If there is a gap, there are **three** layers behind the spacer, so that must
                # be told too -- otherwise the back-calculated layer number of `spacer_purity` is
                # off by one and a ValueError is raised in the thickness cross-check.
                if int(L3_LAYERS) == 2:
                    G.OUT_L3_UM = float(H[2]) * float(L3_SPLIT)
                    G.OUT_L3B_UM = float(H[2]) * (1.0 - float(L3_SPLIT))
                    G.OUT_L3_GAP_UM = (float(L3_GAP_NM) * 1e-3
                                        if float(L3_GAP_NM) > 0.0 else 0.0)
                else:
                    G.OUT_L3_UM = float(H[2])
                    G.OUT_L3B_UM = 0.0
                    G.OUT_L3_GAP_UM = 0.0
                _dm, _th = (0, 0.0) if pol == "TE" else (-1, G.THETA_SUB)
                _pur = G.spacer_purity(obA, pol, design_m=_dm, theta=_th)
            finally:
                (G.OUT_SPACER_UM, G.OUT_SPACER_EPS, G.OUT_L3_UM,
             G.OUT_L3B_UM, G.OUT_L3_GAP_UM) = _kp
            val = val - w["w_pure"] * (1.0 - _pur)
        # [260918] **Fidelity term -- put the reinforced purity into the objective.**
        #
        # `w_pure` uses `spacer_purity`, so it discards evanescent orders and reads at the L2 face.
        # `w_fid` uses the design-channel share of the L3-face field from the same solution but
        # includes evanescent orders -- the two terms are **different quantities** so they can be on together.
        #
        # There is no additional solve (`obA` is used as is). At the default 0 it is not even called,
        # so it is **bit-identical** to the previous path.
        if w.get("w_fid", 0.0) > 0.0 and sp > 0:
            _dmf = 0 if pol == "TE" else -1
            _fid = _fid_l3(obA, pol, sp, n_sp, _dmf)
            _pen_fid = w["w_fid"] * (1.0 - _fid)
            val = val - _pen_fid
            # [260926] opt_terms.tsv capture -- gated on `w_fid > 0`.
            _row["fid"] = float(G._untraced(_pen_fid))
            _row["fid_l3"] = float(G._untraced(_fid))
        # [260924] **-theta pass fid term** (the same quantity as the report key `fid_l3_m`). The t(+1)@-th solution `obm`
        # of delay arm B was already solved above, so there is no additional solve. This quantity depends monotonically on the atom's arg rho
        # (260924 measured, nG 601: arg rho 62-71 -> 0.93, 135-162 -> 0.88-0.92, 205-217 -> 0.66-0.74),
        # so it carries the arg rho **forbidden zone** agreed in 260923 over into the objective. The design order is m = +1 in both lanes.
        # At the default 0 it is not even called, so it is bit-identical to the previous path.
        if w.get("w_fid_m", 0.0) > 0.0 and sp > 0:
            _fidm = _fid_l3(obm, pol, sp, n_sp, +1)
            _pen_fidm = w["w_fid_m"] * (1.0 - _fidm)
            val = val - _pen_fidm
            # [260926] opt_terms.tsv capture -- gated on `w_fid_m > 0`.
            _row["fid_m"] = float(G._untraced(_pen_fidm))
            _row["fid_l3_m"] = float(G._untraced(_fidm))
        # [260921] **`r00` target -- gather the design-channel complex reflection across the whole palette.**
        #
        # Why `rB` is not enough: the `rB` tied by `w_rmag`/`w_rarg` is the oblique-incidence
        # channel, and what the carrier termination must reproduce is the **design channel** `r00`.
        # Measured (260921): unifying `|r00|` reduces the `|rB|` spread from 0.427 -> 0.067
        # along with it (Spearman -0.72) -- the converse does not hold.
        #
        # No additional solve (`obA` is used as is). Both weights default to 0, so the
        # term is not even computed and it is **bit-identical** to the previous path.
        if (w.get("w_r00mag", 0.0) > 0.0 or w.get("w_r00arg", 0.0) > 0.0) \
                and sp > 0:
            _dm0 = 0 if pol == "TE" else -1
            _r00 = _r00_l3(obA, pol, sp, _dm0)
            # [260926] opt_terms.tsv capture -- `_r00` is already solved
            # above whenever this block runs, so recording it costs
            # nothing extra. If the `r00win` block below also runs it
            # will not overwrite absR00/argR00_deg (same quantity).
            _r00c = complex(G._untraced(_r00))
            _row["absR00"] = abs(_r00c)
            _row["argR00_deg"] = math.degrees(
                math.atan2(_r00c.imag, _r00c.real)) % 360.0
            if w.get("w_r00mag", 0.0) > 0.0:
                _pen_r00mag = w["w_r00mag"] * (npa.abs(_r00)
                                               - w["absR00_t"]) ** 2
                val = val - _pen_r00mag
                _row["r00mag"] = float(G._untraced(_pen_r00mag))
            if w.get("w_r00arg", 0.0) > 0.0:
                _pen_r00arg = w["w_r00arg"] * _ph(_r00, w["argR00_t"])
                val = val - _pen_r00arg
                _row["r00arg"] = float(G._untraced(_pen_r00arg))
        # [260926] **r00 WINDOW -- keep the design-channel r00 inside a
        # region, not a single point.** Decision 260926: the carrier
        # r00 map defines a pass region per lane, and each atom is steered
        # to its own phase level while staying inside it. `w_r00mag`/
        # `w_r00arg` above pull `r00` to one target point; this term instead
        # only penalises leaving the region (smooth squared hinge, zero
        # strictly inside).
        #
        # No extra solve (`obA` already solved above). Default
        # (`r00win_rows` empty, i.e. `r00win` job key absent, or
        # `w_r00win` unset/0) skips the block entirely -- old path
        # bit-identical.
        if w.get("w_r00win", 0.0) > 0.0 and w.get("r00win_rows") and sp > 0:
            _dmw = 0 if pol == "TE" else -1
            _r00w = _r00_l3(obA, pol, sp, _dmw)
            _pen_r00win = w["w_r00win"] * _r00win_pen(_r00w, w["r00win_rows"])
            val = val - _pen_r00win
            # [260926] opt_terms.tsv capture. absR00/argR00_deg may already
            # be set by the w_r00mag/w_r00arg block above (same quantity,
            # only recorded once).
            _row["r00win"] = float(G._untraced(_pen_r00win))
            if "absR00" not in _row:
                _r00wc = complex(G._untraced(_r00w))
                _row["absR00"] = abs(_r00wc)
                _row["argR00_deg"] = math.degrees(
                    math.atan2(_r00wc.imag, _r00wc.real)) % 360.0
        _pen_phi = w["w_phi"] * _ph(tA, tgt_A)
        val = val - _pen_phi
        _row["phi"] = float(G._untraced(_pen_phi))
        _pen_dpsi = w["w_dpsi"] * _ph(rB * tB, tgt_B)
        val = val - _pen_dpsi
        _row["dpsi"] = float(G._untraced(_pen_dpsi))
        # [260920] Antipode escape. If 0 (default) it is bit-identical to the previous path.
        if w.get("w_phesc", 0.0) != 0.0:
            val = val - w["w_phesc"] * (_ph_esc(tA, tgt_A)
                                        + _ph_esc(rB * tB, tgt_B))
        # [260926] opt_terms.tsv capture -- finalise and publish this
        # call's row. One `fom()` call per training iteration (the
        # `value_and_grad` forward pass), so this always reflects the
        # iteration that just ran; never mutates `val`.
        _row["beta"] = float(G._untraced(beta))
        _row["nG_stage"] = int(nG)
        _row["fom_total"] = float(G._untraced(val))
        _ITER_CAPTURE.clear()
        _ITER_CAPTURE.update(_row)
        return val
    return fom


def min_feature_nm(rho, px_nm, thr=0.5):
    """**Measured** minimum linewidth (nm). Measured by morphological **opening** after binarisation.

    [260826] The filter radius (60 nm) only *derives* a length scale and does **not guarantee** it.
    So when the optimisation ends it is actually measured, and a shortfall against the gate is flagged loudly.
    It must be measured wrapping the periodic boundary -- an atom may straddle the cell edge.

    [260917] **Fix.** The previous version cut off with `ero.any()` after erosion. That asks "does a disk of radius
    r fit **somewhere**", which returns the **largest inscribed circle**
    -- the exact opposite of the quantity to measure. Now it is measured as the ridge minimum of the distance transform:
    `D(p)` = the largest radius of a disk centred at p that fits entirely (built by repeating
    periodic erosion), ridge = solid pixel where `D` is a local maximum over the 8 neighbours, linewidth =
    `2*min(D over ridge) + 1`.
    An exact `opening(b, r) == b` match was also tried, but the staircase of the rasterised curved boundary
    always broke the match and under-reported the disk from 25 -> 9 pixels, and a scheme
    with a tolerance on the removed-pixel area was **incompatible** (a disk requires >13.7 %, a 3-pixel bridge
    <1.18 %). A scheme eroding the removal set by 1 pixel **cannot see a 1-pixel line
    at all** (it returned 650.8 nm). Only the ridge method passed the seven controls.

    Controls (96x96, pixel 6.851 nm) -- true value / this version, in pixels:
        two r=16 disks + 3-pixel bridge     3 / 3      (previous version 31 = 212.4 nm, 10x)
        clean disk r=12                    23 / 23
        9-pixel strip                       9 / 9
        **1-pixel line                      1 / 1**    (the most important worst case)
        two separated disks r=10           19 / 19
        all solid / all void              inf / inf
    The two that miss are in the **conservative** direction: ellipse (minor axis 13) gives 9, ring (wall 6) gives 5 --
    an ellipse really does get thinner toward the ends, so 9 is in fact the right side.

    **Resolution.** The returned value is `(2*best + 1) * px_nm`, so it is quantised by pixels. At NXY 96
    one step is 6.851 nm, so the 50 nm rule can be enforced only as `>= 8 px = 54.8 nm`
    (7 px = 48.0 nm is below the rule).

    **This function is used only for reporting and `gate_ok`** -- the linewidth term inside the objective is
    a separate differentiable surrogate, `Main_rcwa_l3_atom2d_ffwin._lw_pen`,
    and that already imitates opening/closing correctly. So this fix does not change any optimisation
    trajectory, and **re-judges the gates recorded as passing until now.**
    """
    b = (np.asarray(rho) > thr)
    if not b.any() or b.all():
        return float("inf")
    # D(p) = largest r whose disc, centred at p, fits entirely inside b.
    # Built by successive periodic erosions -- numpy only, no scipy.
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
    # The medial axis is where D is a local maximum; the thinnest point of the
    # shape is the smallest D on it. A plain min over all solid pixels would
    # return the boundary (D = 0) for every shape.
    nb = [np.roll(np.roll(D, dy, 0), dx, 1)
          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    ridge = b & np.logical_and.reduce([D >= q for q in nb])
    if not ridge.any():
        return float("inf")
    return (2 * int(D[ridge].min()) + 1) * px_nm


# ------------------------------------------------------------ eval_only support (260909)
def _resample_rho3_2d(arr, n_out):
    """Nearest-neighbour resample of a stored rho3 to an (n_out, n_out) grid.

    Fallback only -- used when a stored `rho3_npy` was optimised at a
    different NXY than the job now configures. `arr` is assumed square,
    which is how this file always saves it (one atom per (nxy, nxy) cell).
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        n0 = int(round(math.sqrt(arr.size)))
        arr = arr.reshape(n0, n0)
    n_in = arr.shape[0]
    idx = np.clip((np.arange(n_out) * n_in) // n_out, 0, n_in - 1)
    return arr[idx][:, idx]


def _resample_rho3_all(arr, n_out):
    """Resample a stored rho3 -- one sub-layer or two -- to `n_out` per layer.

    [260922] `eval_only` could not read a bilayer density at all: the stored
    vector is `2*nxy*nxy` flat, the call site tested `shape != (nxy, nxy)`, and
    `_resample_rho3_2d` then tried `reshape(136, 136)` on 18432 elements.  So
    no bilayer cell could be re-scored, which is the standard way to check that
    a reported gate is converged in `nG`.
    """
    a = np.asarray(arr).reshape(-1)
    nl = max(1, int(L3_LAYERS))
    if a.size % nl:
        raise ValueError("rho3 size %d is not divisible by %d sub-layers"
                         % (a.size, nl))
    n2 = a.size // nl
    n_in = int(round(math.sqrt(n2)))
    if n_in * n_in != n2:
        raise ValueError("rho3 sub-layer size %d is not square" % n2)
    out = [_resample_rho3_2d(a[k * n2:(k + 1) * n2].reshape(n_in, n_in), n_out)
           for k in range(nl)]
    if nl == 1:
        return out[0]
    return np.concatenate([np.asarray(z).reshape(-1) for z in out])


def _fid_l3(ob, pol, sp, n_sp, design_m):
    """[260918] **Illumination fidelity at the face where L3 sits -- differentiable.**

    Two of the things `spacer_purity` cannot see are closed here (260918:
    "change the objective to match the reinforced purity"):

      (1) **Evanescent orders are not discarded.** `spacer_purity` discards evanescent orders
          wholesale by `if c2 <= 0: continue`. Measured (260918, 87 cells): `evan_frac_l3` is
          **0.181~0.246** at a spacer of 200 nm -- 1/5 of the L3-face amplitude was
          outside the metric. The L3 atom is a real structure and couples to the near field.
      (2) **The reference plane is on the L3 side.** `spacer_purity` reads at `dz = 0` (the L2 face).
          Here it is read at `dz = sp`, using the face the atom actually sees.

    Left out: the backward wave `b`. In LPA the atom's own reflection is already included in the response, so
    counting it as contamination would double count -- this needs judgement, so it was not included.
    (`_spacer_probe` still **measures** it as `fb_ratio_l3`: median 0.192,
     max 0.527.)

    No weighting is used. The z-Poynting weighting is undefined for evanescent orders, so
    it returns **the ratio of squared amplitudes**, not a power ratio. So this value is
    not the same quantity as `spacer_purity` -- do not compare them, use them separately.

    Returns: 0~1. At 1 the L3-face illumination is a pure design-channel plane wave.
    """
    a, _b = ob.GetAmplitudes(3, sp)          # 3 = spacer, dz = sp -> L3 face
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


def _r00_l3(ob, pol, sp, design_m):
    """[260921] **Design-channel complex reflection `r00`, without an additional solve.**

    This is exactly the value modelled by the carrier termination (`out_l3_r_list`), yet the driver
    had no term for it -- `w_rmag`/`w_rarg` act on `rB` (the oblique-incidence channel), which is a
    **different quantity**.

    The `GetAmplitudes(3, sp)` already called by `_fid_l3` gives both the forward `a` and the backward `b`.
    `_fid_l3` discards `b`. At the L3 face `b/a` of the design order
    is the atom's own reflection.

    Measured verification (260921, 14 atoms): against the direct measurement of the atom alone, the magnitude has a mean error of
    **0.00070** (Spearman +1.000), and the phase a **constant +25.04 deg, sd 0.47 deg**.
    The constant offset is a reference-plane convention and is absorbed by the target value -- it is the same offset
    as the 24.619 deg shown by the analytic TMM.
    """
    a, b = ob.GetAmplitudes(3, sp)           # 3 = spacer, dz = sp -> L3 face
    off = ob.nG if pol == "TM" else 0
    for i, g in enumerate(ob.G):
        if int(g[1]) == 0 and int(g[0]) == design_m:
            _a = a[i + off]
            return b[i + off] / (_a + 1e-30)
    return 0.0 * a[0]


def _r00win_lohi(m, rows):
    """[260926] Piecewise-linear (lo, hi) window boundary (degrees) at
    `|r00| = m`, clamped to the end rows. `rows` is a static (non-traced),
    mag-ascending list of `(mag, lo_deg, hi_deg)` python floats parsed from
    the `r00win` job key; `m` may be an autograd-traced scalar.

    Written as a running sum of clipped ramps (`npa.clip`) rather than
    `npa.interp`, which is not differentiable in its `x` argument under
    autograd:

        y(m) = y[0] + sum_i  slope_i * clip(mc - mag[i-1], 0, mag[i] - mag[i-1])

    where `mc` is `m` clamped into `[mag[0], mag[-1]]` first, so a query
    outside the row span holds the nearest end row's window flat (spec:
    job-key docs in `main()`).
    """
    m0, m1 = rows[0][0], rows[-1][0]
    mc = npa.clip(m, m0, m1)
    lo = rows[0][1]
    hi = rows[0][2]
    for i in range(1, len(rows)):
        pm, plo, phi = rows[i - 1]
        nm, nlo, nhi = rows[i]
        seg = npa.clip(mc - pm, 0.0, nm - pm)
        lo = lo + (nlo - plo) / (nm - pm) * seg
        hi = hi + (nhi - phi) / (nm - pm) * seg
    return lo, hi


def _r00win_pen(r00, rows):
    """[260926] Smooth squared-hinge penalty keeping `r00` inside a
    per-`|r00|` window (`rows`: ascending `(mag, lo_deg, hi_deg)`, see
    `_r00win_lohi`). Zero strictly inside the window; grows outside it.

        a   = arg r00, mapped to [0, 360)
        m   = |r00|
        mid = (lo+hi)/2, half = (hi-lo)/2, d = wrap180(a - mid)
        pen_arg = max(0, |d| - half) / 15.0
        pen_mag = max(0, mag_lo - m, m - mag_hi) / 0.03
        return pen_arg**2 + pen_mag**2

    Used by `make_fom`'s `fom()` when `w_r00win > 0` and `r00win` is given
    (spec: 260926).
    """
    m = npa.abs(r00)
    a = npa.mod(npa.degrees(npa.arctan2(npa.imag(r00), npa.real(r00))),
                360.0)
    lo, hi = _r00win_lohi(m, rows)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    d = npa.mod(a - mid + 180.0, 360.0) - 180.0
    pen_arg = npa.maximum(0.0, npa.abs(d) - half) / 15.0
    mmin, mmax = rows[0][0], rows[-1][0]
    pen_mag = npa.maximum(0.0, npa.maximum(mmin - m, m - mmax)) / 0.03
    return pen_arg ** 2 + pen_mag ** 2


def _r00win_inside(absr00, argdeg, rows):
    """[260926] Report-only, plain-float twin of `_r00win_pen`: is the
    FINAL atom's `(|r00|, arg r00)` inside the `r00win` region? Runs once
    after optimisation (or once for an `eval_only` score), so it is not
    autograd-traced -- plain python `min`/`max` stand in for
    `npa.clip`/`npa.maximum`.
    """
    m0, m1 = rows[0][0], rows[-1][0]
    mc = min(max(absr00, m0), m1)
    lo, hi = rows[0][1], rows[0][2]
    for i in range(1, len(rows)):
        pm, plo, phi = rows[i - 1]
        nm, nlo, nhi = rows[i]
        seg = min(max(mc - pm, 0.0), nm - pm)
        lo += (nlo - plo) / (nm - pm) * seg
        hi += (nhi - phi) / (nm - pm) * seg
    d = ((argdeg - 0.5 * (lo + hi) + 180.0) % 360.0) - 180.0
    return bool(m0 <= absr00 <= m1 and abs(d) <= 0.5 * (hi - lo))


def _spacer_probe(ob, pol, sp, n_sp, design_m, theta):
    """[260918] **Four things purity cannot express, from the same solution.**

    Purity has three structural blind spots (the first two were triple-verified in 260910):

      (1) **Blind to phase** -- `scd_c_grating:571` is `npa.abs(...)**2`, so
          the complex argument is discarded. Hence purity is an **upper bound**, not a value:
          for the contaminated field `E = 1 + r e^{i d}`, `r = sqrt(p/(1-p))`, the actual error takes
          the full range **from 0 up to arcsin(r)** depending on `d`.
      (2) **Blind to evanescent orders** -- `if c2 <= 0: continue`. At a spacer of 200 nm
          22 % of the amplitude of the first evanescent order survives and reaches L3 (1/e 132.5 nm).
      (3) **Adds two coherent terms as powers** -- the two parasitics are at `+-G`, so the beat
          period is `Lambda_x/2 = 657.7 nm`, **exactly the atom pitch**. The power sum is
          optimistic by up to +3.01 dB.

    **Reference plane.** `spacer_purity` reads with `GetAmplitudes(idx, 0.0)` and
    since `TranslateAmplitudes` is `ai*exp(1j*q*dz)`, `dz = 0` is the **L2-side
    face**. Propagating orders do not change in magnitude, so the purity value itself is harmless, but evanescent
    orders and relative phases differ from face to face -- here they are
    read at the **L3 face (`dz = sp`)**. The `b` at that face is the downward wave jointly reflected back by L3.

    **Handling of the rule of not duplicating the physics.** The order bookkeeping and polarization weighting
    use the same convention as `spacer_purity`, but this function produces quantities that the other function
    **does not compute at all**. To prevent divergence, purity is
    recomputed in its own way and exported as `purity_chk` -- the caller compares it with
    the value of `G.spacer_purity`, so if the conventions split it shows up on the spot.

    All returned keys are report-only (outside the autograd path).
    """
    idx = 3                                  # 0 substrate | 1 L1 | 2 L2 | 3 spacer
    a, b = ob.GetAmplitudes(idx, float(sp))  # L3 face
    off = ob.nG if pol == "TM" else 0
    kx0 = G.N_SUB * np.sin(float(theta))
    lam_s = 2.0 * np.pi / np.real(ob.omega)
    des = None
    par, evan, prop2, evan2, b2 = [], 0.0, 0.0, 0.0, 0.0
    for i, g in enumerate(ob.G):
        if int(g[1]) != 0:
            continue
        m = int(g[0])
        kxn = kx0 + m * (lam_s / G.PERIOD)
        c2 = n_sp ** 2 - kxn ** 2
        amp = complex(a[i + off])
        if c2 <= 0:                          # the part purity discards
            evan2 += abs(amp) ** 2
            continue
        kz = float(np.sqrt(c2))
        w = (kz / n_sp) if pol == "TM" else (n_sp / kz)
        prop2 += abs(amp) ** 2
        b2 += abs(complex(b[i + off])) ** 2
        if m == design_m:
            des = (amp, w)
        else:
            par.append((amp, w, kxn))
    if des is None or des[0] == 0:
        return {}
    num = abs(des[0]) ** 2 * des[1]
    den = num + sum(abs(p[0]) ** 2 * p[1] for p in par)
    # (3) coherent worst case: add the two parasitics **by magnitude** and then square
    coh = (sum(abs(p[0]) * math.sqrt(p[1]) for p in par)) ** 2
    # (1) Actual phase error. At the cell centre, and the rms/maximum deviation over the cell -- how much the phase of the
    #     illumination seen by the atom wobbles laterally is the LPA assumption itself.
    errs = []
    for k in range(64):
        x = (k / 64.0) * G.PERIOD
        e = 1.0 + sum((p[0] / des[0])
                      * cmath.exp(2j * np.pi * p[2] * x / lam_s) for p in par)
        errs.append(math.degrees(cmath.phase(e)))
    e0 = 1.0 + sum(p[0] / des[0] for p in par)
    r = math.sqrt(max(1.0 - num / den, 0.0) / max(num / den, 1e-12))
    return {
        "purity_chk": float(num / den),
        "purity_coh": float(num / (num + coh)),
        "phase_err_deg": float(math.degrees(cmath.phase(e0))),
        "phase_err_rms_deg": float(math.sqrt(sum(e * e for e in errs)
                                             / len(errs))),
        "phase_err_ptp_deg": float(max(errs) - min(errs)),
        "phase_err_sup_deg": float(math.degrees(math.asin(min(r, 1.0)))),
        "evan_frac_l3": float(math.sqrt(evan2 / prop2)) if prop2 > 0 else None,
        "fb_ratio_l3": float(math.sqrt(b2 / prop2)) if prop2 > 0 else None,
    }


def _measure_arms(nG, rho12, H, sp, n_sp, eps3, lam, nxy, KF, rho3, beta,
                  pol="TM", symy=True, emask=None):
    """Achieved amplitude/phase of arms A and B, for reporting only.

    Mirrors the branch inside `make_fom`'s inner `fom()` exactly -- same
    `_obj`/`_amp` calls, same argument order, same `sym_y`/`density`
    preprocessing -- so this reading cannot drift from the optimising run's
    convention. Does not touch `make_fom`/`fom`; kept as a separate
    read-only measurement so the training path stays untouched.
    """
    # [260921] same projection as `fom`, one or two sub-layers.
    r3 = _project_cell(rho3, nxy, symy, emask, KF, beta)
    if pol == "TE":
        ob0 = _obj(nG, r3, rho12, H, sp, n_sp, eps3, lam, nxy, 0.0, "TE")
        obA = ob0
        tA = _amp(ob0, 0, pol="TE")                     # t(0) @ 0
        rB = _amp(ob0, +1, refl=True, pol="TE")         # r(+1) @ 0
        obm = _obj(nG, r3, rho12, H, sp, n_sp, eps3, lam, nxy,
                   -G.THETA_SUB, "TE")
        tB = _amp(obm, +1, pol="TE")                    # t(+1) @ -th
    else:
        obp = _obj(nG, r3, rho12, H, sp, n_sp, eps3, lam, nxy,
                   +G.THETA_SUB, "TM")
        obA = obp
        tA = _amp(obp, -1, pol="TM")                    # t(-1) @ +th
        rB = _amp(obp, 0, refl=True, pol="TM")          # r(0)  @ +th
        obm = _obj(nG, r3, rho12, H, sp, n_sp, eps3, lam, nxy,
                   -G.THETA_SUB, "TM")
        tB = _amp(obm, +1, pol="TM")                    # t(+1) @ -th
    tA_c, rB_c, tB_c = complex(tA), complex(rB), complex(tB)
    rBtB = rB_c * tB_c

    def wrap(d):
        return (d + 180.0) % 360.0 - 180.0

    A_deg = math.degrees(math.atan2(tA_c.imag, tA_c.real))
    B_deg = math.degrees(math.atan2(rBtB.imag, rBtB.real))
    # [260917] **Output arg rB and arg tB separately.** B_deg is the phase of the product, so
    # the same value comes out even if the two components cancel each other. But the B arm propagates
    # back and forth 2051 um in the slab between the two events (s=836.35 up to the retro
    # grating, round trip 2t/cos th), so if `arg rB` differs per element a real lens
    # is applied to the delay arm -- the product target cannot see this. To catch it with `w_rarg` it must first
    # be readable. It is report-only, so unrelated to the optimisation path.
    argrB_deg = math.degrees(math.atan2(rB_c.imag, rB_c.real))
    argtB_deg = math.degrees(math.atan2(tB_c.imag, tB_c.real))
    # [260917] **echo, from the already solved solution.** The reflections propagating in the substrate at `-th` incidence
    # are only three: m = 0(-th), +1(normal), +2(+th), and **only m=+2 flips k_x from -G to +G,
    # goes up at +th and re-enters the return grating** (a copy delayed by 2 symbols and
    # coherent). It is exactly the quantity gated by `scd_c_grating:1309`.
    #
    # Why it must be measured anew: every recorded echo came from `scd_c`, where L3 is a
    # **uniform slab**. The actual L3 array has period Lambda_x/2, so the reciprocal-lattice vector is
    # **exactly 2G**, and the echo transition is exactly `delta k_x = 2G` -- i.e. the array supplies that
    # transition at **first order**, and that path is in none of the recorded values.
    _Rm, _Tm = obm.RT_Solve(normalize=1, byorder=1)
    echo_R_p2 = _ord_power(obm, _Rm, +2)
    # [260918] Also read the same echo with `_amp`. The objective cannot call `RT_Solve`
    # every iteration (all-order RT solve, same cost as `_Ttot`), so
    # it uses the amplitude path, and then **the reported value and the penalty are in different units**.
    # Emit both so that every run measures the ratio -- it is not estimated.
    echo_a2 = float(abs(_amp(obm, +2, refl=True, pol=pol))) ** 2
    # [261007] report only: the same R(+2) @ -th amplitude WITH its phase, for the lens-level
    # echo overlap model (<WORK_ROOT>\echo_overlap_261007). Not used by any objective.
    echo_r2_c = complex(_amp(obm, +2, refl=True, pol=pol))
    # [260918] **Spacer purity -- priority item.** Measured for the first time with L3 actually
    # on top. All purity values so far came from the uniform
    # slab termination of `scd_c`, and the effect of the L3 grating closing the spacer into an FP cavity
    # was missing entirely (260918 measured: wobble 0.026297 at 20 termination points
    # = 13.0 times the 300/300 gate margin).
    #
    # The physics is **not duplicated** -- `G.spacer_purity` is called as is.
    # That function counts the layer number as `Layer_N - (2 + (OUT_L3_UM>0) + (OUT_L3B_UM>0))`
    # and the stack of this driver is
    #     0 substrate | 1 L1 | 2 L2 | 3 spacer | 4 L3 | 5 air
    # so `Layer_N = 6` and with `OUT_L3_UM > 0` it points exactly to **3 = spacer**.
    # That L3 is a grating and not a slab does not change the layer **count**,
    # so this is not a hack -- and the thickness cross-check of that function itself
    # proves that the index is right (ValueError if wrong).
    #
    # Why it is not copied: the original contains two fixes, the 1/kz z-Poynting weighting of TE and
    # recovering the wavelength from `ob.omega`, and when a third comes
    # it will not reach the copy -- the recorded failure 'fixes do not reach the sibling file'.
    #
    # The design channel and the illumination angle differ per lane:
    #     TM  design_m = -1, theta = +THETA_SUB   (A arm is the +th solution)
    #     TE  design_m =  0, theta = 0            (A arm is the normal solution)
    purity = None
    _probe = {}
    if sp > 0:
        _keep = (G.OUT_SPACER_UM, G.OUT_SPACER_EPS, G.OUT_L3_UM,
               G.OUT_L3B_UM, G.OUT_L3_GAP_UM)
        try:
            G.OUT_SPACER_UM = float(G._untraced(sp))
            G.OUT_SPACER_EPS = complex(n_sp) ** 2
            # [260921] With a 2-layer atom there are two layers behind the spacer. `spacer_purity`
            # counts layer numbers from the top, so both must be told (the same
            # phase structure is already used by the `set_out_l3b` 2-layer termination).
            # [260923] If there is a gap, there are **three** layers behind the spacer -- likewise
            # they must be told so that the back-calculated layer number of `spacer_purity` is right.
            if int(L3_LAYERS) == 2:
                G.OUT_L3_UM = float(H[2]) * float(L3_SPLIT)
                G.OUT_L3B_UM = float(H[2]) * (1.0 - float(L3_SPLIT))
                G.OUT_L3_GAP_UM = (float(L3_GAP_NM) * 1e-3
                                    if float(L3_GAP_NM) > 0.0 else 0.0)
            else:
                G.OUT_L3_UM = float(H[2])
                G.OUT_L3B_UM = 0.0
                G.OUT_L3_GAP_UM = 0.0
            _shares = {}
            _dm, _th = (0, 0.0) if pol == 'TE' else (-1, G.THETA_SUB)
            purity = float(G.spacer_purity(obA, pol, design_m=_dm,
                                           theta=_th, shares=_shares))
            # [260918] The four things purity cannot see. **Includes a self-check** --
            # `purity_chk` is the purity independently recounted by this function, and
            # if it disagrees with `spacer_purity`, the order bookkeeping or polarization weighting convention
            # has split, which shows up right here (rather than being silently wrong).
            _probe = _spacer_probe(obA, pol, float(sp), float(n_sp),
                                   _dm, float(_th))
            _probe["fid_l3"] = float(_fid_l3(obA, pol, float(sp),
                                             float(n_sp), _dm))
            # [260924] **fid of the -theta pass (report-only).** The second event t(+1)@-th of delay arm B
            # also passes the same L3 face, yet fid had been measured only at the +theta (A arm) solution -- same reason as
            # `fidm` of the carrier. `obm` was already solved above, so there is no additional solve, and
            # the objective (`make_fom`) is not touched. The design order is m = +1 in both lanes.
            _probe["fid_l3_m"] = float(_fid_l3(obm, pol, float(sp),
                                               float(n_sp), +1))
            # [260921] Always record `r00` -- even if the weight is not used, one must know what the palette
            # presents to the carrier. It is the same solve, so it is free.
            _r00v = complex(_r00_l3(obA, pol, float(sp), _dm))
            _probe["absR00"] = abs(_r00v)
            _probe["argR00_deg"] = math.degrees(
                math.atan2(_r00v.imag, _r00v.real)) % 360.0
            _probe["T_atom"] = 1.0 - abs(_r00v) ** 2
            if _probe:
                _probe["purity_chk_rel"] = abs(
                    _probe["purity_chk"] - purity) / max(purity, 1e-12)
                if _probe["purity_chk_rel"] > 1e-9:
                    print("[OPT2D] **WARNING: purity %.9f of _spacer_probe differs from "
                          "spacer_purity %.9f (relative %.2e) -- the conventions "
                          "have split**" % (_probe["purity_chk"], purity,
                                          _probe["purity_chk_rel"]), flush=True)
        finally:
            (G.OUT_SPACER_UM, G.OUT_SPACER_EPS, G.OUT_L3_UM,
         G.OUT_L3B_UM, G.OUT_L3_GAP_UM) = _keep
    info = dict(amp_tA=abs(tA_c), amp_rB=abs(rB_c), amp_tB=abs(tB_c),
                amp_rBtB=abs(rBtB), prod_amp=abs(tA_c) * abs(rBtB),
                A_deg=A_deg, B_deg=B_deg,
                argrB_deg=argrB_deg, argtB_deg=argtB_deg,
                phi_c_deg=wrap(0.5 * (A_deg + B_deg)),
                dpsi_deg=wrap(B_deg - A_deg),
                echo_R_p2=echo_R_p2,
                echo_a2=echo_a2,
                echo_r2_re=echo_r2_c.real,
                echo_r2_im=echo_r2_c.imag,
                purity=purity,
                purity_gate=0.996722,
                purity_ok=(None if purity is None
                           else bool(purity >= 0.996722)),
                echo_p2_db=10.0 * math.log10(max(echo_R_p2, 1e-12)),
                echo_gate_db=-20.5,
                echo_ok=bool(10.0 * math.log10(max(echo_R_p2, 1e-12)) <= -20.5))
    info.update(_probe)
    info.update(_ledger(obA, "ledA"))
    info.update(_ledger(obm, "ledB"))
    # [260929] carrier_eff_pct / carrier_echo_db / eff_ok / eff_id_err
    # -- ledger-based efficiency prediction, same lane mapping as
    # ledger/atom_to_carrier.py (260929 review, wf_f3a28e8d-37f). No
    # extra solve: reuses the ledA_*/ledB_* keys flattened in just
    # above by info.update(_ledger(...)).
    _egl = lambda k: float(info.get(k, 0.0))
    if pol == "TM":
        _eff_ep = _egl("ledA_T_m-1")
        _eff_rgo = _egl("ledA_R_m+0")
    else:
        _eff_ep = _egl("ledA_T_m+0")
        _eff_rgo = _egl("ledA_R_m+1")
    _eff_em = _egl("ledB_T_m+1")
    _eff_echo = _egl("ledB_R_m+2")
    _eff_sig2 = _eff_ep * _eff_rgo * _eff_em
    _eff_sig = math.sqrt(_eff_sig2) if _eff_sig2 > 0.0 else 0.0
    info["carrier_eff_pct"] = 100.0 * _eff_sig / ((4.0 / 9.0) ** 1.5)
    info["carrier_echo_db"] = (10.0 * math.log10(_eff_echo)
                               if _eff_echo > 0.0 else float("-inf"))
    info["eff_ok"] = bool(info["carrier_eff_pct"] >= 85.0)
    if info["carrier_eff_pct"] > 0.0:
        info["eff_id_err"] = abs(843.167 * info["prod_amp"]
                                 - info["carrier_eff_pct"]
                                 ) / info["carrier_eff_pct"]
        if info["eff_id_err"] > 1e-5:
            print("[OPT2D] WARN: eff_id_err %.3e > 1e-5 "
                  "(843.167*prod_amp=%.6f carrier_eff_pct=%.6f)"
                  % (info["eff_id_err"], 843.167 * info["prod_amp"],
                     info["carrier_eff_pct"]), flush=True)
    else:
        info["eff_id_err"] = None
    return info


# ------------------------------------------------------------ rho (260923)
def _measure_rho(nG, r3, H, sp, n_sp, eps3, lam, nxy):
    """Atom-alone rho -- the trapped +-theta_sub parasitic's reflection as
    seen from the spacer side, referenced at the L3 FACE (the spacer/L3
    interface). Matches the convention `spacer_purity`/`spacer_fid` already
    use (`GetAmplitudes(_sp_idx, OUT_SPACER_UM)`, `scd_c_grating.py:887,
    904`) -- read at the END of the incident spacer layer, not its start.

    [260923 review fix] The original version of this function read
    `GetAmplitudes(0, 0.0)`, i.e. the PATCH at the far (top) edge of the
    0.5 um incident half-space -- 0.5 um IN FRONT of the L3 face. That
    adds a spurious round-trip phase of `2*kz*0.5um` (193.85 deg at TM,
    theta=54.69deg, n=1.444, 1550nm) on top of the physical `arg rho`.
    Reading at `dz = T0` (the end of the incident layer, i.e. the L3 face)
    removes it -- confirmed against the analytic Fresnel arg for a bare
    spacer/air interface and against Track 2's `_rho_trapped`, which now
    reads the same plane (`argrho_termination_face.diff`).

    Stack is semi-infinite spacer -> L3 partial layer(s) -> air (same as
    `argrho_probe.py`'s `measure()`), and the incidence angle uses the same
    formula (`arcsin((lam/PERIOD)/n_sp)`). The one difference from a fresh
    probe script: instead of re-reading `rho3_opt.npy` and re-projecting,
    this receives the density the driver JUST SCORED (`r3`, the output of
    `_project3`) directly, so a structure that was never simulated is
    never measured.

    **Only +theta_sub is solved.** `rho(+theta) == rho(-theta)` is a
    RECIPROCITY identity (`scd_c_grating.py:886-910`: "no symmetry
    assumption, so no structure search can break it") -- it holds for ANY
    atom shape, x-mirror-symmetric or not, so solving both directions
    would measure the same number twice at 2x the cost. [Superseded
    260923: an earlier draft of this docstring argued the two directions
    needed independent solves because a freeform atom generally is not
    x -> -x symmetric -- that conflated reciprocity (an exact identity,
    unconditional on structure) with x-mirror symmetry (a structural
    property this driver does not enforce). Only the first is why
    rho(+th) = rho(-th) here; the atom's own asymmetry is irrelevant to
    that equality.] Per-lane label (extract/TM geometry): the two trapped
    orders are **m=0 at +theta and m=-2 at -theta** -- both are read off
    this ONE +theta solve via the reciprocity identity above, not from two
    separate solves.

    Both polarisations TM/TE come out of the one +theta solve -- the
    eigenmode decomposition (`Init_Setup`) does not depend on polarisation,
    so it is solved once; the second polarisation is a cheap re-excitation
    (linear read of the same modes, no extra eigensolve).

    **Propagation floor.** If `n_sp < lam/PERIOD` the trapped order would
    be evanescent even inside the spacer itself -- `arcsin` has no real
    solution. Write NaN for every rho key instead of clamping theta to
    grazing incidence (90 deg), which would silently report a number for a
    channel that never propagates.
    """
    hs = r3 if isinstance(r3, (list, tuple)) else [r3]
    app = int(ATOMS_PER_PERIOD)
    ly = G.PERIOD / float(app)
    _T0 = 0.5   # spacer (incident) layer thickness -- must match Add_LayerUniform below
    _sin_th = (lam / G.PERIOD) / n_sp
    info = dict(rho_ref="L3 face", rho_incidence="+theta")
    if _sin_th >= 1.0:
        # [260923] below the propagation floor -- write NaN, do not clamp.
        info["rho_theta_deg"] = float("nan")
        for p in ("TM", "TE"):
            info["abs_rho_%s" % p] = float("nan")
            info["arg_rho_%s_deg" % p] = float("nan")
        return info
    th = float(np.arcsin(_sin_th))
    if len(hs) == 2:
        _f = float(L3_SPLIT)
        _h3 = [H[2] * _f, H[2] * (1.0 - _f)]
    else:
        _h3 = [H[2]]
    _fill_eps = float(L3_FILL_N) ** 2
    # [260923] Interlayer SiO2 gap -- same grid layer `_obj` inserts between
    # the two L3 sub-layers (patterned by sub-layer A's density, `hs[0]`,
    # the SAME array `_obj` calls `_r3[0]`). Only when both sub-layers
    # exist and the gap is > 0; otherwise this is exactly the original
    # loop, bit-identical.
    _gap_um = float(L3_GAP_NM) * 1e-3
    _has_gap = (len(_h3) == 2 and _gap_um > 0.0)

    ob = grcwa.obj(int(nG), [G.PERIOD, 0.0], [0.0, ly], 1.0 / lam,
                   float(th), 0.0, verbose=0)
    if _LI_ON:
        _li_apply(ob)
    ob.Add_LayerUniform(_T0, complex(n_sp) ** 2)          # spacer (incident)
    for _k3, hh in enumerate(_h3):
        if _k3 == 1 and _has_gap:
            ob.Add_LayerGrid(_gap_um, app * nxy, nxy)
        ob.Add_LayerGrid(hh, app * nxy, nxy)
    ob.Add_LayerUniform(0.5, G.EPS_AIR)
    ob.Init_Setup()
    eg = []
    for k, rr in enumerate(hs):
        eb = _fill_eps if (k == 0 and len(hs) > 1) else 1.0
        if k == 1 and _has_gap:
            _gape = complex(N_L3_GAP) ** 2
            eg.append(_fill_eps + _tile_x(hs[0], app)
                      * (_gape - _fill_eps))
        eg.append(eb + _tile_x(rr, app) * (eps3 - eb))
    ob.GridLayer_geteps(npa.concatenate([npa.reshape(e, (-1,)) for e in eg]))
    idx = None
    for i, g in enumerate(ob.G):
        if int(g[0]) == 0 and int(g[1]) == 0:
            idx = i
            break
    info["rho_theta_deg"] = float(math.degrees(th))
    for p, exc in (("TM", (1.0, 0.0, 0.0, 0.0)),
                   ("TE", (0.0, 0.0, 1.0, 0.0))):
        ob.MakeExcitationPlanewave(*exc, order=0)
        a, b = ob.GetAmplitudes(0, _T0)
        off = ob.nG if p == "TM" else 0
        rho = complex(b[idx + off]) / (complex(a[idx + off]) + 1e-30)
        info["abs_rho_%s" % p] = float(abs(rho))
        info["arg_rho_%s_deg" % p] = float(math.degrees(cmath.phase(rho)))
    return info


# ------------------------------------------------------------ distinct_atoms (261003)
def _distinct_atom_vec(path, nxy, nl, resample):
    """One OLD-format single-atom file (rho3_opt.npy: (nxy, nxy) or a flat
    layers*nxy*nxy vector) -> flat float vector. A 4-D file is a cell file and
    is refused here. `resample` (eval_only only) nearest-resamples a stored
    grid of another NXY, exactly like the single-atom eval_only path."""
    a = np.load(path)
    if a.ndim == 4:
        raise SystemExit("[OPT2D] distinct_atoms: %s is a 4-D cell file where "
                         "a single-atom list entry is expected" % path)
    if a.size != nl * nxy * nxy:
        if not resample:
            raise SystemExit(
                "[OPT2D] distinct_atoms: %s has %d values, the grid wants %d "
                "(%d layer(s) x %dx%d); not resampled for a seed -- it would "
                "be a different structure" % (path, a.size, nl * nxy * nxy,
                                              nl, nxy, nxy))
        a = _resample_rho3_all(a, nxy)
    return np.asarray(a, dtype=float).reshape(-1)


def _distinct_load(spec, nxy, app, nl, resample):
    """`spec` (a job key value) -> flat atom-major vector of app*nl*nxy*nxy.
    Three forms: ';'-separated list of `app` single-atom files (one per
    sub-atom, in x order), ONE single-atom file (replicated `app` times), or
    ONE 4-D cell file (shape (app, nl, nxy, nxy), as saved by this driver)."""
    parts = [p.strip() for p in str(spec).split(";") if p.strip()]
    if not parts:
        raise SystemExit("[OPT2D] distinct_atoms: empty atom path list")
    for p in parts:
        if not os.path.isfile(p):
            raise SystemExit("[OPT2D] distinct_atoms: not found: %s" % p)
    if len(parts) == 1:
        a = np.load(parts[0])
        if a.ndim == 4:
            if tuple(a.shape) != (app, nl, nxy, nxy):
                raise SystemExit(
                    "[OPT2D] distinct_atoms: cell file %s has shape %s, "
                    "expected %s" % (parts[0], tuple(a.shape),
                                     (app, nl, nxy, nxy)))
            return np.asarray(a, dtype=float).reshape(-1)
        return np.tile(_distinct_atom_vec(parts[0], nxy, nl, resample), app)
    if len(parts) != app:
        raise SystemExit("[OPT2D] distinct_atoms: %d paths given, atoms_per_"
                         "period is %d (give 1, or exactly %d)"
                         % (len(parts), app, app))
    return np.concatenate([_distinct_atom_vec(p, nxy, nl, resample)
                           for p in parts])


def _distinct_report(x, nxy, emask, KF, beta, px_nm):
    """Report-only keys of a distinct-atoms cell: per sub-atom projected fill
    fraction and min feature (each sub-atom measured periodic INSIDE its own
    cell; the cell-level, seam-aware figure is `min_feature_meas_nm`)."""
    flat = np.asarray(x).reshape(-1)
    app = int(ATOMS_PER_PERIOD)
    per = max(1, int(L3_LAYERS)) * int(nxy) * int(nxy)
    ff, mf = [], []
    for j in range(app):
        ds = _layer_densities_atom(flat[j * per:(j + 1) * per], nxy, emask,
                                   KF, beta)
        ff.append([float(np.mean(z)) for z in ds])
        mf.append([float(min_feature_nm(z, px_nm)) for z in ds])
    return {"distinct_atoms": 1, "atoms_in_cell": app,
            "cell_kind": "distinct_atoms_cell",
            "fill_frac_atoms": ff, "min_feature_atoms_nm": mf}


def main():
    G.set_period(_num("period_nm", 1315.39))
    G.set_substrate(_num("n_sub", 1.4440))
    lam = _num("lams_nm", 1550.0) * 1e-3
    # [261015 kx_hold] OPT-IN matched angle: THETA_SUB := arcsin(lam / (PERIOD * n_sub)) so that the arm keys
    # (A, B, dpsi, eff, echo, purity) and `_measure_rho` (which already uses that angle, :2424) share one
    # angle. Absent / 0 -> byte-identical. Evaluation only.
    if int(_num("kx_hold", 0)):
        if not hasattr(G, "pin_theta_to"):
            print("[OPT2D] ERROR: scd_c_grating has no pin_theta_to -- refusing to run a kx_hold job at the "
                  "fixed angle", flush=True)
            return 2
        if not int(_num("eval_only", 0)):
            print("[OPT2D] ERROR: kx_hold = 1 needs eval_only = 1", flush=True)
            return 2
        _th_kx = G.pin_theta_to(lam)
        _li_extra.update(kx_hold=1, theta_sub_deg_used=_th_kx)
        print("[OPT2D] kx_hold = 1: THETA_SUB = %.5f deg at %.3f nm (kx = G held)" % (_th_kx, lam * 1e3),
              flush=True)
    # [260928] Carrier-TOKEN-axis offset. The carrier's r00 token
    # (scd_c_grating.py:701, r_of_2layer * exp(1j*phase_ref(lam))) and the
    # atom's FACE r00 (this driver's `_r00_l3`, read via GetAmplitudes(3,
    # sp)) are the same physical quantity on two reference planes; measured
    # token = face - 24.619 deg (equivalently face = token + 24.619 deg) at
    # 1550 nm / n_spacer 1.444 (driver check 260928). `lam` here is already
    # in um, matching `phase_ref`'s argument.
    _r00_off_deg = 180.0 - math.degrees(G.phase_ref(lam))
    nxy = int(_num("NXY", 48))
    # [260921] **Atoms per period** -- overwrites the module global. `_obj` reads this,
    # so it must come before the first call (here is that place).
    global ATOMS_PER_PERIOD
    ATOMS_PER_PERIOD = int(_num("atoms_per_period", 3))
    if ATOMS_PER_PERIOD < 1:
        print("[OPT2D] ERROR: atoms_per_period must be 1 or more (got %d)"
              % ATOMS_PER_PERIOD, flush=True)
        return 2
    # [261003] K = 1 cell with DISTINCT atoms. Default 0 leaves every path
    # below the original one (see the module docstring and `_project_cell`).
    global DISTINCT_ATOMS
    DISTINCT_ATOMS = int(_num("distinct_atoms", 0))
    if DISTINCT_ATOMS not in (0, 1):
        print("[OPT2D] ERROR: distinct_atoms must be 0 or 1 (got %d)"
              % DISTINCT_ATOMS, flush=True)
        return 2
    if DISTINCT_ATOMS and ATOMS_PER_PERIOD < 2:
        print("[OPT2D] ERROR: distinct_atoms = 1 needs atoms_per_period >= 2 "
              "(got %d)" % ATOMS_PER_PERIOD, flush=True)
        return 2
    _json_name = "atom2d_cell_opt.json" if DISTINCT_ATOMS else "atom2d_opt.json"
    _npy_name = "rho3_cell_opt.npy" if DISTINCT_ATOMS else "rho3_opt.npy"
    # [260921] **L3 sub-layer count.** The single extruded atom is a
    # lossless 2-port, so `arg tau = (arg r00 + arg r')/2 -+ 90` and its
    # `arg r'` reaches a hard 154.8 deg -- which caps ONE unified
    # `arg r00` at N=2 (measured on 516 atoms, three independent routes).
    # Two stacked cross-sections reach 332.5 deg of `arg r'` inside the
    # palette window (n=99 of 420), past the 300 needed for N=8 at one
    # `arg r00`.  `arg r'` is the AIR-side reflection: no downstream
    # quantity reads it, so it is a free internal variable and the job
    # contract is unchanged.
    global L3_LAYERS, L3_SPLIT, L3_ETCH_STACK, L3_FILL_N, L3_GAP_NM, N_L3_GAP
    L3_LAYERS = int(_num("l3_layers", 1))
    L3_SPLIT = float(_num("l3_split", 0.5))
    L3_ETCH_STACK = int(_num("l3_etch_stack", 0))
    L3_FILL_N = float(_num("l3_fill_n", 1.0))
    L3_GAP_NM = _num("l3_gap_nm", 0.0)
    N_L3_GAP = _num("n_l3_gap", 1.4440)
    if L3_LAYERS not in (1, 2):
        print("[OPT2D] ERROR: l3_layers must be 1 or 2 (got %d)"
              % L3_LAYERS, flush=True)
        return 2
    if not (0.05 <= L3_SPLIT <= 0.95):
        print("[OPT2D] ERROR: l3_split must be 0.05~0.95 (got %.3f)"
              % L3_SPLIT, flush=True)
        return 2
    # [260923] Interlayer gap only means something with two sub-layers.
    if L3_GAP_NM > 0.0 and L3_LAYERS != 2:
        print("[OPT2D] ERROR: l3_gap_nm > 0 is valid only with l3_layers = 2 "
              "(got l3_layers=%d, l3_gap_nm=%.2f)"
              % (L3_LAYERS, L3_GAP_NM), flush=True)
        return 2
    if L3_GAP_NM > 0.0:
        print("[OPT2D] L3 interlayer SiO2 gap: %.1f nm (n=%.4f, follows the A-layer pattern)"
              % (L3_GAP_NM, N_L3_GAP), flush=True)
    # [260923] The key is put in the result JSON only when a gap is used -- because for the default path (gap 0)
    # the JSON must be byte-identical.
    _gap_extra = ({"l3_gap_nm": float(L3_GAP_NM), "n_l3_gap": float(N_L3_GAP)}
                  if L3_GAP_NM > 0.0 else {})
    _cell_nm = G.PERIOD * 1e3 / float(ATOMS_PER_PERIOD)
    nG = int(_num("nG", 201))
    nG_eval = int(_num("nG_eval", 601))
    # [260924] `ng_schedule` -- raise nG in the later part of the iterations (ported from the same key of the carrier driver).
    # Format "0:201, 0.75:401" (iteration fraction:nG, starting with 0:). If absent, [(0.0, nG)] and bit-identical to before.
    # The final report (_measure_arms / _measure_rho) is measured at the **last-stage nG** -- measured in 260924 (sim330)
    # the nG 201 values had not converged (against nG 601: A -9~-28 deg, echo up to 6 dB, arg r00 up to 55 deg).
    # The eval_only path does not use this key (`nG` as before).
    _ngs_raw = _txt("ng_schedule", "").strip()
    if _ngs_raw:
        try:
            ng_sched = sorted((float(_a), int(_b)) for _a, _b in
                              (_p.split(":") for _p in _ngs_raw.replace(";", ",").split(",")
                               if _p.strip()))
        except ValueError:
            print("[OPT2D] ERROR: ng_schedule format error %r (e.g. 0:201, 0.75:401)" % _ngs_raw,
                  flush=True)
            return 2
        if not ng_sched or ng_sched[0][0] != 0.0:
            print("[OPT2D] ERROR: ng_schedule must start with 0: (%r)" % _ngs_raw, flush=True)
            return 2
    else:
        ng_sched = [(0.0, nG)]
    nG_fin = ng_sched[-1][1]
    H = [_num("H1_nm", 500.0) * 1e-3, _num("H2_nm", 350.0) * 1e-3,
         _num("H3_nm", 775.0) * 1e-3]
    sp = _num("spacer_nm", 200.0) * 1e-3
    n_sp = _num("n_spacer", 1.4440)
    # [260928] `phase_ref`'s analytic term is written against `G.N_SUB`
    # (the substrate index, job key `n_sub`), not this driver's separate
    # `n_spacer` -- a job that sets them differently makes `_r00_off_deg`
    # only approximate for this spacer. Report keys are written regardless.
    if abs(float(n_sp) - G.N_SUB) > 1e-9:
        print("[OPT2D] WARN: n_spacer=%.6f != G.N_SUB=%.6f -- phase_ref "
              "assumes N_SUB, so r00_token_offset_deg is approximate here."
              % (float(n_sp), G.N_SUB), flush=True)
    eps3 = G.eps_of(_txt("material3", "a-Si"))
    iters = int(_num("iters", 400))
    lr = _num("lr", 0.05)
    # [260909] Pure-evaluation path -- score a stored rho3 as-is, no
    # softening, no optimisation, no value_and_grad. `iters`/`lr` above are
    # parsed as before but ignored in this branch, not treated as errors.
    eval_only = int(_num("eval_only", 0))
    rho3_npy = _txt("rho3_npy", "")
    # [260923] **rho -- atom-alone reflection of the trapped +theta_sub
    # parasitic, seen from the spacer.** 0 (default) means it is switched
    # off -- no extra solve, result JSON bit-identical to
    # before. Switching it on makes `main()` solve one extra RCWA at the
    # density it just scored (`_measure_rho`, one +theta solve, both
    # polarisations reuse the eigenmode) and writes it into atom2d_opt.json
    # -- both the eval_only path and the optimisation path.
    measure_rho = int(_num("measure_rho", 0))

    l12 = _txt("l12_density")
    if not l12 or not os.path.isfile(l12):
        print("[OPT2D] ERROR: l12_density not found -> %s" % l12, flush=True)
        return 2
    a = np.load(l12)
    nx = ATOMS_PER_PERIOD * nxy

    def _resample(col):
        n0 = col.size
        c = np.concatenate([[0.0], np.cumsum(col)])
        e = np.arange(nx + 1) * (n0 / float(nx))
        lo = np.clip(np.floor(e).astype(int), 0, n0)
        fr = e - np.floor(e)
        ci = c[lo] + fr * col[np.clip(lo, 0, n0 - 1)]
        return (ci[1:] - ci[:-1]) / (n0 / float(nx))

    rho12 = [_resample(a[0]), _resample(a[1])]
    # [260918] **2D carrier.** If `l12_rho1_npy` / `l12_rho2_npy` are both
    # given and read, they replace the 1D resampling above. Those files are
    # the `rho1_2d.npy` / `rho2_2d.npy` stored by `carrier2d`, and are physical densities that have already been
    # through filter + projection. If either is missing nothing changes, so
    # previous jobs are bit-identical. `l12_density` is kept
    # required in 2D mode too -- it is the carrier provenance recorded in the result json, and the 2D
    # array itself was made with it as the seed.
    _p1, _p2 = _txt("l12_rho1_npy"), _txt("l12_rho2_npy")
    if _p1 and _p2:
        if not (os.path.isfile(_p1) and os.path.isfile(_p2)):
            print("[OPT2D] ERROR: cannot read l12_rho1_npy / l12_rho2_npy",
                  flush=True)
            return 2
        _r1, _r2 = np.load(_p1), np.load(_p2)
        if _r1.shape != (nx, nxy) or _r2.shape != (nx, nxy):
            print("[OPT2D] ERROR: 2D carrier shapes %s / %s differ from (%d, %d)"
                  "" % (_r1.shape, _r2.shape, nx, nxy), flush=True)
            return 2
        rho12 = [_r1, _r2]
        print("[OPT2D] **2D carrier** %dx%d -- fill L1 %.3f / L2 %.3f"
              % (nx, nxy, float(_r1.mean()), float(_r2.mean())), flush=True)

    pol = (_txt("pol", "TM").strip().upper() or "TM")
    if pol not in ("TM", "TE"):
        print("[OPT2D] ERROR: pol must be TM|TE (got %r)" % pol, flush=True)
        return 2
    gate_nm = _num("min_feature_nm", 60.0)
    px_nm = _cell_nm / nxy                         # y-direction pixel (= same in x)
    # [260826] **Enforce the gate lower bound.** The conic filter radius derives the length scale,
    # so a radius below the gate cannot be satisfied in the first place. Even if a smaller value is given
    # it is raised to the gate -- better than silently producing a violating shape.
    _req = _num("filt_radius_nm", 60.0)
    if _req < gate_nm:
        print("[OPT2D] filt_radius_nm %.0f < gate %.0f -> raised to %.0f"
              % (_req, gate_nm, gate_nm), flush=True)
        _req = gate_nm
    rad = max(1.0, _req / px_nm)
    KF = conic_kernel_fft(nxy, rad)
    # [260917] Structuring element of the linewidth penalty. Built at the **gate**
    # scale, not the filter radius -- because what is measured is "is it thinner than the gate".
    KG = conic_kernel_fft(nxy, max(1.0, gate_nm / px_nm))
    if DISTINCT_ATOMS:
        # [261003] the linewidth penalty sees the FULL cell (atoms*nxy, nxy),
        # periodic over Lambda in x, so seam slivers count.
        KG = conic_kernel_fft_rect(ATOMS_PER_PERIOD * nxy, nxy,
                                   max(1.0, gate_nm / px_nm))
    print("[OPT2D] atoms %d/period -- cell %.2f nm / pixel %.2f nm / filter radius "
          "%.1f px (%.0f nm)"
          % (ATOMS_PER_PERIOD, _cell_nm, px_nm, rad, rad * px_nm), flush=True)
    if DISTINCT_ATOMS:
        print("[OPT2D] distinct_atoms = 1: %d INDEPENDENT atoms per period "
              "(x order), design variables %d, output %s / %s"
              % (ATOMS_PER_PERIOD,
                 ATOMS_PER_PERIOD * max(1, L3_LAYERS) * nxy * nxy,
                 _npy_name, _json_name), flush=True)

    # Initial value -- starting from an ellipse starts in a good basin
    u = (np.arange(nxy) + 0.5) / nxy - 0.5
    X, Y = np.meshgrid(u, u, indexing="ij")
    ie = _txt("init_ellipse")
    if ie:
        dx, dy = [float(v) for v in ie.split(",")[:2]]
    else:
        dx = dy = _num("init_diam_nm", 360.0)
    # [260921] When cloning a Lambda/2 job to Lambda/3, leaving `init_ellipse` as is
    # makes the ellipse overflow the cell and become a **uniform slab** -- it silently becomes meaningless, so
    # warn (do not abort; there are cases where filling is intended).
    if max(dx, dy) > _cell_nm:
        print("[OPT2D] WARNING: init_ellipse %.0f,%.0f exceeds the cell %.1f nm "
              "-- fills the cell completely" % (dx, dy, _cell_nm), flush=True)
    ax = dx / _cell_nm / 2.0
    ay = dy / _cell_nm / 2.0
    rho0 = (((X / ax) ** 2 + (Y / ay) ** 2) <= 1.0).astype(float)
    rho0 = 0.5 + 0.45 * (2.0 * rho0 - 1.0)         # relaxed to 0.05~0.95
    # [260921] With 2 layers the upper-layer ellipse is given separately. If `init_ellipse_b` is absent
    # it starts from the same ellipse as the lower layer -- in that case the two layers have the same cross-section, so
    # the initial structure is the same as the single layer (apart from the thickness split), and the optimisation splits them.
    if L3_LAYERS == 2:
        _ib = _txt("init_ellipse_b")
        if _ib:
            _bx, _by = [float(v) for v in _ib.split(",")[:2]]
        else:
            _bx, _by = dx, dy
        if max(_bx, _by) > _cell_nm:
            print("[OPT2D] WARNING: init_ellipse_b %.0f,%.0f exceeds the cell %.1f nm "
                  "-- the upper layer fills the cell completely"
                  % (_bx, _by, _cell_nm), flush=True)
        _bax, _bay = _bx / _cell_nm / 2.0, _by / _cell_nm / 2.0
        _rb = (((X / _bax) ** 2 + (Y / _bay) ** 2) <= 1.0).astype(float)
        _rb = 0.5 + 0.45 * (2.0 * _rb - 1.0)
        rho0 = np.concatenate([rho0.reshape(-1), _rb.reshape(-1)])
        print("[OPT2D] L3 2 layers: split %.2f (%.1f + %.1f nm), upper-layer ellipse "
              "%.0f,%.0f -- design variables %d"
              % (L3_SPLIT, H[2] * 1e3 * L3_SPLIT,
                 H[2] * 1e3 * (1.0 - L3_SPLIT), _bx, _by, rho0.size),
              flush=True)

    # [260918] **Start from a stored density (phase continuation).**
    #
    # The derivative of the phase term `_ph = 1 - cos(arg t - th)` with respect to `arg t` is `sin(arg t - th)`,
    # so it is **exactly 0 at the antipode**. A cell that landed 180 deg away from the target
    # sits at the stationary point of the very term that should pull it, and indeed in the 943
    # phase-target cells of 260918 the residual is **bimodal** -- 404 within 30 deg,
    # **189** at 150 deg or more. Doing many cold starts solves this trap by luck,
    # while starting from an already converged neighbouring phase (45 deg apart) avoids
    # it structurally.
    #
    # If not given, the ellipse path above is unchanged, so it is **bit-identical** to before.
    inp = _txt("init_npy")
    if inp and not DISTINCT_ATOMS:
        _a = np.load(inp)
        if _a.ndim == 4:
            raise SystemExit(
                "[OPT2D] init_npy %s is a 4-D distinct-atoms cell file but "
                "distinct_atoms = 0 -- it would be read as a single atom. "
                "Set distinct_atoms = 1 to warm-start a cell." % inp)
        # [260924] **2-layer warm start.** With L3_LAYERS == 2 the design variable is a
        # flat 2*nxy*nxy vector made by concatenate (see above). An array that matches its element count
        # -- flat or (2, nxy, nxy) -- has the same layer order as stored by rho3_opt.npy
        # so it is accepted as is (flattened with reshape(-1) both
        # forms become [layer 0 flat, layer 1 flat]). Any other
        # size is still an error as for a disk: no resampling -- because it
        # would become a different structure under the same name.
        _n2 = nxy * nxy
        _is_bilayer = (L3_LAYERS == 2 and _a.size == 2 * _n2)
        if not _is_bilayer and _a.shape != (nxy, nxy):
            raise SystemExit(
                "[OPT2D] init_npy grid mismatch: %s is %s but nxy=%d "
                "(l3_layers=%d -> (%d,%d) or %d elements allowed). "
                "No resampling -- because it would become a different structure "
                "under the same name." % (inp, _a.shape, nxy, L3_LAYERS, nxy, nxy,
                              2 * _n2 if L3_LAYERS == 2 else _n2))
        # The stored array has **converged**, so it sticks to 0 and 1. A variable stuck on the boundary
        # also has no gradient, so it is brought back inside 0.05~0.95,
        # the same as the ellipse seed.
        _lo = _num("init_npy_soft", 0.05)
        _flat = _a.reshape(-1) if _is_bilayer else _a
        rho0 = _lo + (1.0 - 2.0 * _lo) * np.clip(_flat, 0.0, 1.0)
        print("[OPT2D] starting from init_npy%s: %s (mean %.4f -> %.4f, margin %.3f)"
              % (" 2-layer" if _is_bilayer else "", inp, float(_a.mean()),
                 float(rho0.mean()), _lo), flush=True)
    if DISTINCT_ATOMS:
        # [261003] Atom-major start vector of the K = 1 cell. With init_npy:
        # a ';'-separated list (one old-format atom file per sub-atom), ONE
        # file (replicated), or ONE 4-D cell file; softened exactly like the
        # single-atom warm start. Without init_npy: the ellipse start
        # replicated (identical sub-atoms; they separate through the carrier's
        # x dependence only -- give a list for a real start).
        if inp:
            _lo = _num("init_npy_soft", 0.05)
            _d0 = _distinct_load(inp, nxy, ATOMS_PER_PERIOD, max(1, L3_LAYERS),
                                 False)
            rho0 = _lo + (1.0 - 2.0 * _lo) * np.clip(_d0, 0.0, 1.0)
            print("[OPT2D] distinct_atoms init_npy: %d atom(s) from %s "
                  "(mean %.4f -> %.4f, soft %.3f)"
                  % (ATOMS_PER_PERIOD, inp, float(_d0.mean()),
                     float(rho0.mean()), _lo), flush=True)
        else:
            rho0 = np.tile(np.asarray(rho0).reshape(-1), ATOMS_PER_PERIOD)

    # [260926] **`r00win` -- a per-|r00| pass REGION, not one target point.**
    # Decision 260926: the carrier r00 map defines a pass region per
    # lane; atoms must stay inside it while each is steered to its own
    # phase level. Complements `w_r00mag`/`w_r00arg` (which pull `r00` to a
    # single point) rather than replacing them -- a job may use either or
    # both.
    #
    # Format: "mag:lo:hi; mag:lo:hi; ..." -- at `|r00| = mag` the allowed
    # `arg r00` interval is `[lo, hi]` degrees (0-360, `lo < hi`, no wrap
    # needed). Between rows, `lo`/`hi` interpolate linearly in `|r00|`;
    # outside the row span the nearest end row's window is held (clamped).
    # `w_r00win` = 0 (default) or `r00win` absent -> the term is never
    # computed, old path bit-identical.
    _r00win_raw = _txt("r00win", "").strip()
    _w_r00win = _num("w_r00win", 0.0)
    # [260928] `r00win_axis` -- carrier maps are measured on the r00 TOKEN
    # axis (`scd_c_grating.r_of_2layer`) but this driver's atoms are scored
    # on the FACE axis (`_r00_l3`). Default 'face' is today's literal
    # behaviour, rows used exactly as given, unconverted.
    _r00win_axis = _txt("r00win_axis", "face").strip().lower()
    if _r00win_axis not in ("face", "token"):
        print("[OPT2D] ERROR: r00win_axis must be 'face' or 'token' "
              "(got %r)" % _r00win_axis, flush=True)
        return 2
    r00win_rows = []
    if _r00win_raw:
        try:
            for _row in _r00win_raw.replace(";", ",").split(","):
                _row = _row.strip()
                if not _row:
                    continue
                _rm, _rlo, _rhi = (float(_x) for _x in _row.split(":"))
                r00win_rows.append((_rm, _rlo, _rhi))
        except ValueError:
            print("[OPT2D] ERROR: r00win format error %r "
                  "(e.g. 0.06:165:240; 0.12:165:240)" % _r00win_raw,
                  flush=True)
            return 2
        if not r00win_rows:
            print("[OPT2D] ERROR: r00win has no valid row (%r)"
                  % _r00win_raw, flush=True)
            return 2
        # [260928] Shift TOKEN-axis rows onto the FACE axis atoms are
        # actually scored on, before any validation below reads lo/hi --
        # face = token + off (`_r00_off_deg`, set near `lam` above).
        # 'face' (default) leaves rows exactly as parsed, byte-identical
        # to the pre-260928 path.
        if _r00win_axis == "token":
            r00win_rows = [(_rm, _rlo + _r00_off_deg, _rhi + _r00_off_deg)
                           for _rm, _rlo, _rhi in r00win_rows]
            print("[OPT2D] r00win_axis=token -> face rows (+%.3f deg): %s"
                  % (_r00_off_deg,
                     "; ".join("%.4f:%.3f:%.3f" % _r for _r in r00win_rows)),
                  flush=True)
        for _rm, _rlo, _rhi in r00win_rows:
            if not (_rlo < _rhi):
                print("[OPT2D] ERROR: an r00win row must have lo < hi "
                      "(%.4f:%.2f:%.2f)" % (_rm, _rlo, _rhi), flush=True)
                return 2
        _r00win_mags = [_rm for _rm, _rlo, _rhi in r00win_rows]
        # [260926] D1 fix: `mags != sorted(mags)` PASSES on duplicate mags
        # (a duplicate is already sorted), and `_r00win_lohi`'s segment
        # slope divides by `nm - pm` -- a duplicate makes that 0 and raises
        # ZeroDivisionError later, mid-optimisation. Require STRICTLY
        # ascending mags so duplicates are rejected here instead.
        if any(b <= a for a, b in zip(_r00win_mags, _r00win_mags[1:])):
            print("[OPT2D] ERROR: r00win rows must be in ascending mag order with no duplicates "
                  "(%r)" % _r00win_raw, flush=True)
            return 2
        # [260926] D2 fix: `absR00`/`argR00_deg` are filled into `_probe`
        # by `_measure_arms` ONLY inside its `if sp > 0:` branch (see the
        # `_probe["absR00"] = ...` block below) -- with `sp <= 0` `_probe`
        # stays `{}` and `info` never gets those keys, so the later
        # `_r00win_extra` read (`info["absR00"]`) dies with KeyError once
        # the run reaches that point, and the r00win term is silently
        # skipped up to then. Reject at parse time instead.
        if r00win_rows and sp <= 0:
            print("[OPT2D] ERROR: r00win is valid only when spacer_nm > 0 "
                  "(absR00/argR00_deg are recorded only for sp > 0, spacer_nm=%r)"
                  % _num("spacer_nm", 200.0), flush=True)
            return 2
    if _w_r00win > 0.0 and not r00win_rows:
        print("[OPT2D] ERROR: w_r00win > 0 but r00win is missing.",
              flush=True)
        return 2
    # [260929] echo_gate_db -- dB-based alternative to echo_a2_gate,
    # with per-lane power scaling (see docstring). Giving both keys is
    # an error. Giving neither leaves echo_a2_gate's old default path
    # (0.0, term off) byte-identical.
    _echo_gate_db_raw = _txt("echo_gate_db", "").strip()
    _echo_a2_gate_raw = _txt("echo_a2_gate", "").strip()
    _echo_a2_gate_from_db = None
    if _echo_gate_db_raw and _echo_a2_gate_raw:
        print("[OPT2D] ERROR: echo_gate_db and echo_a2_gate cannot "
              "both be given.", flush=True)
        return 2
    if _echo_gate_db_raw:
        _echo_gate_db_val = _num("echo_gate_db", 0.0)
        if _echo_gate_db_val > -20.5:
            print("[OPT2D] ERROR: echo_gate_db must be <= -20.5 dB "
                  "(got %.3f)" % _echo_gate_db_val, flush=True)
            return 2
        _echo_lane_scale = 1.0 if pol == "TM" else 0.334083
        _echo_a2_gate_from_db = (10.0 ** (_echo_gate_db_val / 10.0)
                                  * _echo_lane_scale)
    w = dict(amp=_num("w_amp", 1.0), r0=_num("w_r0", 0.3),
             w_phi=_num("w_phi", 2.0), w_dpsi=_num("w_dpsi", 2.0),
             # [260918] **purity weight.** If 0 (default) the term is off and it is bit-identical
             # to before. Shows what happens without `spw2`
             # (purity 2/18 on re-optimisation) -- see the comment inside fom().
             w_pure=_num("w_pure", 0.0),
             # [260918] Fidelity weight (L3 face, including evanescent orders, coherent ratio).
             # If 0 (default) the term is off and it is bit-identical to before.
             w_fid=_num("w_fid", 0.0),
             w_fid_m=_num("w_fid_m", 0.0),
             # [260917] **Reflection target.** Gathers all elements at the same complex `rB` --
             # because with the product target alone `|tB|` trades off against `|rB|`
             # (see the comment inside fom()). Magnitude and phase are penalised separately, and both default to
             # 0, so if not given it is bit-identical to the previous path.
             # [260920] Antipode escape weight. If 0 (default) it is bit-identical.
             w_phesc=_num("w_phesc", 0.0),
             w_rmag=_num("w_rmag", 0.0), w_rarg=_num("w_rarg", 0.0),
             # [260917] Linewidth penalty. If 0 (default) it is bit-identical to before.
             w_lw=_num("w_lw", 0.0),
             # [260918] **echo penalty.** `echo_a2_gate` is a gate in units of the `_amp` squared
             # amplitude. The conversion ratio to the normalised-power gate (-20.5 dB = 0.008913)
             # differs per lane, so **measure it from `echo_a2` / `echo_R_p2` of the
             # result json and put it in the job.**
             # If either of the two is 0 the term is off and it is bit-identical to before.
             w_echo2=_num("w_echo2", 0.0),
             echo_a2_gate=(_echo_a2_gate_from_db
                           if _echo_a2_gate_from_db is not None
                           else _num("echo_a2_gate", 0.0)),
             absR_t=_num("absR_target", 0.0),
             argR_t=_num("argR_target_deg", 0.0),
             # [260921] Design-channel `r00` target. If both are 0 (default) it is bit-identical.
             # `argR00_target_deg` is given in the convention measured by this file (**+25.04 deg**
             # relative to the direct measurement of the atom alone) -- it is on the same axis as `argR00_deg` of the result json,
             # so read it from there and put it in the job.
             w_r00mag=_num("w_r00mag", 0.0), w_r00arg=_num("w_r00arg", 0.0),
             absR00_t=_num("absR00_target", 0.0),
             argR00_t=_num("argR00_target_deg", 0.0),
             # [260926] `r00win` pass-region term (see the block above `w =
             # dict(...)`). `w_r00win` = 0 (default) or `r00win_rows` empty
             # -> never computed, old path bit-identical.
             w_r00win=_w_r00win, r00win_rows=r00win_rows,
             r00win_axis=_r00win_axis,
             phi=_num("phi_target_deg", 0.0),
             dpsi_t=_num("dpsi_target_deg", 130.0),
             # [260826] Amplitude target. If >0, gather at that value instead of maximising.
             amp_t=_num("amp_target", 0.0),
             # [260826] **Product target** `|t_A| * |r_B t_B|`. If >0 it replaces amp_target.
             # Measured for a disk (H3=850/t=200/1550, nG=201): TM median
             # 0.1092 (max 0.1177) / TE median 0.0232 (max 0.0283).
             prod_t=_num("prod_target", 0.0),
             # [260826] **Breakdown avoidance.** Penalises the amount by which `T_total` falls below this floor.
             # r1 measured: T>=0.85 -> breakdown median 0.223 dB /
             # T<0.60 -> 3.490 dB. 0 means off (default).
             T_floor=_num("T_floor", 0.0),
             w_tmin=_num("w_tmin", 4.0),
             # [260929] efficiency hinge weight/gate -- see docstring
             # (w_eff/eff_gate_pct). Spec: ledger/atom_to_carrier.py,
             # ledger/judge_atoms.py (260929 review, wf_f3a28e8d-37f).
             w_eff=_num("w_eff", 0.0),
             eff_gate_pct=_num("eff_gate_pct", 86.0))
    print("[OPT2D] lane=%s  phi*=%.1f  dpsi*=%.1f  %s  gate %.0f nm%s"
          % (pol, w["phi"], w["dpsi_t"],
             ("prod*=%.4f" % w["prod_t"]) if w["prod_t"] > 0 else
             ("amp*=%.3f" % w["amp_t"]) if w["amp_t"] > 0 else "amp*=(maximise)",
             gate_nm,
             ("  T_floor=%.2f (w %.1f)" % (w["T_floor"], w["w_tmin"]))
             if w["T_floor"] > 0 else ""), flush=True)
    # [260926] When r00win is used, record the window in the log. By default (off) nothing is printed.
    if w["w_r00win"] > 0.0 and w["r00win_rows"]:
        print("[OPT2D] r00win (w=%.1f): %s"
              % (w["w_r00win"],
                 "; ".join("%.4f:%.1f:%.1f" % r for r in w["r00win_rows"])),
              flush=True)
    # [260907] `sym_y` -- default 1 (enforced). If 0, bit-identical to the previous path.
    # The codebase convention is "the default reproduces the old behaviour", but here the default
    # is deliberately on: the off path has no guarantee of blocking polarization conversion, so it can yield atoms unusable in this
    # architecture, and the on path was measured not to change current results.
    _symy = bool(int(_num("sym_y", 1)))
    print("[OPT2D] sym_y=%d (y mirror enforced -- x-z plane-of-incidence symmetry, TM<->TE conversion blocked)"
          % int(_symy), flush=True)
    # [260917] EDGE MARGIN -- ported from the same-named key of `ffwin` and with the same meaning:
    # **fix** the N edge pixels to air so that the atom cannot touch its own periodic image. The local-period
    # approximation erases exactly that contact by assumption.
    # 260917 measured (74 pairs excluding self pairs): island/island 9.9 % vs the rest 28.8 %, and
    # only island/island is flat against the phase step (9.5 -> 10.4 % vs 11.1 -> 30.5 %).
    # If the mask is narrower than the filter radius, the filter pushes the material back out, so at least
    # the radius (in pixels) must be given. If 0 (default), bit-identical to the previous path.
    _emg = int(_num("edge_margin_px", 0.0))
    if _emg < 0 or 2 * _emg >= nxy:
        print("[OPT2D] ERROR: edge_margin_px = %d is not possible at NXY %d "
              "(0 <= 2*margin < NXY)" % (_emg, nxy), flush=True)
        return 2
    # [261002] **Per-axis margins + 'pre'/'post' mode** (`edge_margin_x_px`,
    # `edge_margin_y_px`, `edge_margin_mode`; see the module docstring). x =
    # array axis 0 = plane of incidence, y = array axis 1 (the sym_y axis).
    # -1 = fall back to `edge_margin_px` for that axis; 0 = no mask on it.
    # None of the three keys set -> `_em_new` False, `_emx_eff == _emy_eff ==
    # _emg`, so the mask array and the log lines are the old ones (the JSON
    # only gains the three recording keys, see `_emask_extra`).
    _emx = int(_num("edge_margin_x_px", -1.0))
    _emy = int(_num("edge_margin_y_px", -1.0))
    _emode = (_txt("edge_margin_mode", "pre").strip().lower() or "pre")
    if _emode not in ("pre", "post"):
        print("[OPT2D] ERROR: edge_margin_mode must be pre|post (got %r)"
              % _emode, flush=True)
        return 2
    _emx_eff = _emg if _emx < 0 else _emx
    _emy_eff = _emg if _emy < 0 else _emy
    if min(_emx, _emy) < -1 or 2 * _emx_eff >= nxy or 2 * _emy_eff >= nxy:
        print("[OPT2D] ERROR: edge_margin_x_px = %d / edge_margin_y_px = %d "
              "invalid for NXY %d (-1, or 0 <= 2*margin < NXY)"
              % (_emx, _emy, nxy), flush=True)
        return 2
    _em_new = (_emx >= 0 or _emy >= 0 or _emode != "pre")
    # [261002 consumers] One constructor, mode carried by the mask object (no
    # module global): `edge_mask_from_record` rebuilds the same object from
    # the keys written to atom2d_opt.json below.
    _emask = build_edge_mask(nxy, _emx_eff, _emy_eff, _emode)
    if _emask is not None:
        if not _em_new:
            print("[OPT2D] edge_margin_px = %d (%.1f nm) -- edge fixed, "
                  "usable %dx%d (%.1f %%), filter radius %.1f px"
                  % (_emg, _emg * px_nm, nxy - 2 * _emg, nxy - 2 * _emg,
                     100.0 * _emask.mean(), rad), flush=True)
            if _emg < rad:
                print("[OPT2D] WARNING: edge_margin_px %d < filter radius %.1f px -- "
                      "the filter can push material back to the edge" % (_emg, rad),
                      flush=True)
        else:
            print("[OPT2D] edge mask mode=%s  margin x=%d px (%.1f nm) "
                  "y=%d px (%.1f nm) -- usable %.1f %%, filter radius "
                  "%.1f px, gate %.0f nm, min margin for void >= gate: "
                  "%d px" % (_emode, _emx_eff, _emx_eff * px_nm, _emy_eff,
                             _emy_eff * px_nm, 100.0 * _emask.mean(), rad,
                             gate_nm,
                             int(math.ceil(gate_nm / (2.0 * px_nm)))),
                  flush=True)
            # Measured, not assumed: the all-solid raw variable through the
            # real mask/filter/projection is the WORST case (raw in [0, 1],
            # monotone chain), so this void is guaranteed for every design.
            # 'pre' realises 2*margin only for margin >= 6 px at this
            # kernel (measured 261002: 4 -> 2 px, 5 -> 8, 6 -> 12, 7 -> 14).
            _wc = np.asarray(_masked_density(np.ones((nxy, nxy)), _emask,
                                             KF, 64.0))
            for _ax, _m, _dim in (("x", _emx_eff, 0), ("y", _emy_eff, 1)):
                if _m <= 0:
                    continue
                _vp = _edge_void_px(_wc, _dim)
                print("[OPT2D] edge mask axis %s: margin %d px, guaranteed "
                      "periodic void %d px = %.1f nm (nominal 2*margin = %d "
                      "px), gate %.0f nm" % (_ax, _m, _vp, _vp * px_nm,
                                              2 * _m, gate_nm), flush=True)
                if _vp * px_nm < gate_nm:
                    print("[OPT2D] WARN: axis %s guaranteed void %.1f nm < "
                          "gate %.0f nm" % (_ax, _vp * px_nm, gate_nm),
                          flush=True)
            if _emode == "post":
                print("[OPT2D] WARN: post mode cuts the solid at the strip "
                      "edge; solid left within gate %.0f nm of the strip is a "
                      "sliver -- only w_lw (%.3g) and the final min_feature_nm "
                      "gate see it" % (gate_nm, w["w_lw"]), flush=True)
    # [261002 consumers] The REALISED mask is always recorded (effective
    # margins, -1 already resolved), not only when a new key was set, so a
    # consumer never has to guess from edge_margin_px alone. A json without
    # these keys came from an older driver: edge_margin_px both axes, 'pre'.
    _emask_extra = {"edge_margin_x_px": int(_emx_eff),
                    "edge_margin_y_px": int(_emy_eff),
                    "edge_margin_mode": _emode}
    # [260917] Point at which the penalty switches on (it/iters). Same default as ffwin -- the early and middle
    # search is left free and raised linearly over the last 15 %.
    lw_start_frac = min(1.0, max(0.0, _num("lw_start_frac", 0.85)))
    if w["w_lw"] > 0.0:
        print("[OPT2D] w_lw = %.3g, lw_start_frac = %.2f -- gate %.0f nm "
              "structuring element %.1f px" % (w["w_lw"], lw_start_frac, gate_nm,
                                    max(1.0, gate_nm / px_nm)), flush=True)
    fom = make_fom(nG, rho12, H, sp, n_sp, eps3, lam, nxy, KF, w, pol=pol,
                   symy=_symy, emask=_emask, KG=KG)
    vg = value_and_grad(lambda r, b, lr_: -fom(r, b, lr_))
    _vg_by = {nG: vg}
    for _f_s, _n_s in ng_sched:
        if _n_s not in _vg_by:
            _fo_s = make_fom(_n_s, rho12, H, sp, n_sp, eps3, lam, nxy, KF, w, pol=pol,
                             symy=_symy, emask=_emask, KG=KG)
            _vg_by[_n_s] = value_and_grad(
                lambda r, b, lr_, _fo=_fo_s: -_fo(r, b, lr_))
    if _ngs_raw:
        print("[OPT2D] ng_schedule %s -- final report nG %d"
              % (", ".join("%g:%d" % fn for fn in ng_sched), nG_fin), flush=True)

    sched = ((0.0, 8.0), (0.4, 16.0), (0.7, 32.0), (0.9, 64.0))
    # [260924] `beta_schedule` -- projection beta schedule (format "0:64" or "0:8, 0.4:16, 0.7:32, 0.9:64",
    # starting with 0:). If absent the default above is used as is (bit-identical). Purpose: init_npy + init_npy_soft 0 + ng_schedule
    # 0:601 running only a few iterations as a **continuation polish** -- restarting from beta 8 blurs the structure again
    # (in 260924 sim322, 3 of 4 warm arms lost the linewidth grating). The final measurement is beta 64 as before.
    _bs_raw = _txt("beta_schedule", "").strip()
    if _bs_raw:
        try:
            sched = tuple(sorted((float(_a), float(_b)) for _a, _b in
                                 (_p.split(":") for _p in _bs_raw.replace(";", ",").split(",")
                                  if _p.strip())))
        except ValueError:
            print("[OPT2D] ERROR: beta_schedule format error %r (e.g. 0:64)" % _bs_raw, flush=True)
            return 2
        if not sched or sched[0][0] != 0.0:
            print("[OPT2D] ERROR: beta_schedule must start with 0: (%r)" % _bs_raw, flush=True)
            return 2
        print("[OPT2D] beta_schedule %s" % ", ".join("%g:%g" % fb for fb in sched), flush=True)

    if eval_only:
        # [260909] **Pure-evaluation path.** Score `rho3_npy` as-is at the
        # final scheduled beta -- no Adam loop, no `value_and_grad` call
        # (the one above is built but never invoked with these inputs).
        # [261003] distinct_atoms: rho3_npy may be a ';'-separated list.
        _rho3_paths = ([_z.strip() for _z in rho3_npy.split(";") if _z.strip()]
                       if DISTINCT_ATOMS else [rho3_npy])
        if (not rho3_npy or not _rho3_paths
                or not all(os.path.isfile(_z) for _z in _rho3_paths)):
            print("[OPT2D] ERROR: eval_only = 1 but there is no rho3_npy -- "
                  "specify the stored density to score (rho3_npy=%r)."
                  % rho3_npy, flush=True)
            return 2
        if DISTINCT_ATOMS:
            x_eval = _distinct_load(rho3_npy, nxy, ATOMS_PER_PERIOD,
                                    max(1, int(L3_LAYERS)), True)
        else:
            x_eval = np.load(rho3_npy)
            if x_eval.ndim == 4:
                raise SystemExit(
                    "[OPT2D] rho3_npy %s is a 4-D distinct-atoms cell file "
                    "but distinct_atoms = 0 -- it would be scored as a single "
                    "atom. Set distinct_atoms = 1." % rho3_npy)
        # [260922] Judge by size. A 2-layer density is a `2*nxy*nxy` flat vector, so
        # `shape != (nxy, nxy)` was always true and it went into the resampler, where
        # the square reshape fails, so **re-scoring a 2-layer cell was impossible**.
        _n2 = int(nxy) * int(nxy)
        _want = _n2 * max(1, int(L3_LAYERS))
        if DISTINCT_ATOMS:
            pass                      # [261003] already the flat atom-major vector
        elif x_eval.size != _want:
            print("[OPT2D] rho3_npy size %d != configured grid %d (%d layers x %dx%d) -- "
                  "resampling." % (x_eval.size, _want,
                                 max(1, int(L3_LAYERS)), nxy, nxy), flush=True)
            x_eval = _resample_rho3_all(x_eval, nxy)
        elif int(L3_LAYERS) < 2:
            x_eval = np.asarray(x_eval).reshape(nxy, nxy)
        beta_final = sched[-1][1]
        t0e = time.time()
        # [dedupe 261002] `fom()` and `_measure_arms()` solve the SAME
        # full-stack objects; build each once. Memo on only here, cleared in
        # `finally` (objects freed before `_measure_rho`). See the memo block
        # above `_obj_build`.
        _obj_memo_begin()
        try:
            val = fom(x_eval, beta_final)
            fwd_s = time.time() - t0e
            info = _measure_arms(nG, rho12, H, sp, n_sp, eps3, lam, nxy, KF,
                                 x_eval, beta_final, pol=pol, symy=_symy,
                                 emask=_emask)
        finally:
            _obj_memo_end()
        # [260926] `r00win` report keys -- only added when the term is in
        # play (`w_r00win` > 0 or `r00win` given), so the default JSON is
        # unchanged. `info["absR00"]`/`info["argR00_deg"]` are recorded by
        # `_measure_arms` only inside its `sp > 0` branch -- safe to read
        # here with no extra solve because the parse-time D2 check above
        # already rejects `r00win` with `sp <= 0` before this point runs.
        _r00win_extra = (
            {"r00win": [list(_r) for _r in w["r00win_rows"]],
             "w_r00win": float(w["w_r00win"]),
             "r00win_inside": _r00win_inside(
                 info["absR00"], info["argR00_deg"], w["r00win_rows"]),
             "r00win_axis": w["r00win_axis"],
             "r00win_rows_face": [list(_r) for _r in w["r00win_rows"]]}
            if (w["w_r00win"] > 0.0 or w["r00win_rows"]) else {})
        # [260928] Carrier-token-axis mirror of argR00_deg -- see the
        # `_r00_off_deg` comment near `lam` in main(). Present only when
        # argR00_deg itself is (sp > 0), same guard `_r00win_extra` needs.
        _r00tok_extra = (
            {"argR00_token_deg": (info["argR00_deg"] - _r00_off_deg) % 360.0,
             "r00_token_offset_deg": _r00_off_deg}
            if "argR00_deg" in info else {})
        _rho_info = {}
        # [261003] distinct_atoms report keys (empty dict when off -> JSON unchanged).
        _distinct_extra = (_distinct_report(x_eval, nxy, _emask, KF, beta_final,
                                            px_nm) if DISTINCT_ATOMS else {})
        if measure_rho:
            _r3e = _project_cell(x_eval, nxy, _symy, _emask, KF, beta_final)
            _rho_info = _measure_rho(nG, _r3e, H, sp, n_sp, eps3, lam, nxy)
            print("[OPT2D] rho (ref=%s, theta=%.2f deg)  |rho_TM|=%.6f "
                  "arg=%.1f deg  |rho_TE|=%.6f arg=%.1f deg"
                  % (_rho_info["rho_ref"], _rho_info["rho_theta_deg"],
                     _rho_info["abs_rho_TM"], _rho_info["arg_rho_TM_deg"],
                     _rho_info["abs_rho_TE"], _rho_info["arg_rho_TE_deg"]),
                  flush=True)
        od = _txt("result_folder", "") or os.path.join(
            _txt("Output_folder", "."), _txt("subfolder_name", "opt"))
        os.makedirs(od, exist_ok=True)
        lg = io.open(os.path.join(od, "opt_log.dat"), "w", encoding="utf-8")
        lg.write("Iter\tFoM\tbeta\tFwd_s\n")
        lg.write("%d\t%.6f\t%.1f\t%.1f\n"
                 % (1, -float(val), beta_final, fwd_s))
        lg.close()
        # [260928] eval_only must score linewidth/fill-fraction against the
        # SAME masked density the EM solve above and the optimisation
        # path's own final report use (`_emask`, not None) -- passing None
        # silently read the raw, un-masked design variable whenever a job
        # sets `edge_margin_px > 0`.
        mf, _ = _min_feature_all(x_eval, nxy, _emask, KF, beta_final, px_nm)
        _ff = _fill_frac_all(x_eval, nxy, _emask, KF, beta_final)
        _ok, _live = _gate_with_fill(mf, gate_nm, _ff)
        print("[OPT2D] fill per layer %s -- layers with structure %d/%d"
              % (", ".join("%.4f" % z for z in _ff), len(_live), len(_ff)),
              flush=True)
        print("[OPT2D] eval_only  fom %.6f  beta %.1f  |t_A| %.6f  "
              "|r_B t_B| %.6f  A %.2f deg  B %.2f deg  [%.1f s]"
              % (-float(val), beta_final, info["amp_tA"], info["amp_rBtB"],
                 info["A_deg"], info["B_deg"], fwd_s), flush=True)
        print("[OPT2D] measured minimum linewidth %.1f nm (gate %.0f) -> %s"
              % (mf, gate_nm, "pass" if _ok else "**fail**"), flush=True)
        io.open(os.path.join(od, _json_name), "w",
                encoding="utf-8").write(json.dumps(
                    dict(pol=pol, min_feature_meas_nm=float(mf),
                         min_feature_gate_nm=float(gate_nm), gate_ok=bool(_ok),
                         fill_frac_each=[float(z) for z in _ff],
                         n_live_layers=len(_live),
                         filt_radius_nm=float(rad * px_nm),
                         sym_y=int(_symy), edge_margin_px=_emg,
                         **_emask_extra,
                         w_lw=w["w_lw"], lw_start_frac=lw_start_frac,
                         l3_layers=int(L3_LAYERS), l3_split=float(L3_SPLIT),
                         atoms_per_period=int(ATOMS_PER_PERIOD),
                         l3_fill_n=float(L3_FILL_N),
                         l3_etch_stack=int(L3_ETCH_STACK),
                         nG=nG, nG_eval=nG_eval, NXY=nxy,
                         iters=iters, lr=lr,
                         H_nm=[h * 1e3 for h in H], spacer_nm=sp * 1e3,
                         weights=w, elapsed_s=fwd_s,
                         eval_only=1,
                         rho3_npy=(rho3_npy if DISTINCT_ATOMS
                                   else os.path.abspath(rho3_npy)),
                         beta_eval=beta_final, fom_eval=-float(val),
                         **_gap_extra, **_li_extra, **info, **_rho_info,
                         **_r00win_extra, **_r00tok_extra, **_distinct_extra),
                    ensure_ascii=False, indent=2))
        return 0

    x = rho0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    # [260830] **Output-folder override.** If `result_folder` is given in the job, the output is
    # recorded there instead, and copied to the final folder when the job succeeds.
    # Same convention as `Main_rcwa_scd_c.py:586` -- if absent, Output_folder as before.
    od = _txt("result_folder", "") or os.path.join(
        _txt("Output_folder", "."), _txt("subfolder_name", "opt"))
    os.makedirs(od, exist_ok=True)
    # [ckpt] `ckpt_every` (default 10, 0 disables) -- see the helper block
    # above `main()` (`_ckpt_fingerprint`/`_ckpt_save`/`_ckpt_load`/
    # `_ckpt_truncate_log`). `_shared_od` is the eventual shared
    # destination (`Output_folder/subfolder_name`); when `result_folder`
    # staging makes `od` a different, local dir, the checkpoint also gets a
    # best-effort copy there so a resubmitted job (after
    # its first run died) can resume from it.
    # [ckpt-r2] The shared copy is only in play when the JOB FILE ITSELF
    # sets BOTH `Output_folder` and `subfolder_name` (read with no default
    # via this driver's own `_txt`) -- a `result_folder`-only staged job
    # has no known shared destination, and the old code's `_txt(..., ".")`
    # / `_txt(..., "opt")` defaults synthesised a meaningless `./opt/`
    # whenever those two keys were absent. A job that sets both keeps its
    # shared ckpt at exactly `Output_folder/subfolder_name`, staged or not.
    ckpt_every = int(_num("ckpt_every", 10))
    _out_folder = _txt("Output_folder")
    _subfolder = _txt("subfolder_name")
    _has_shared_target = bool(_out_folder) and bool(_subfolder)
    if _has_shared_target:
        _shared_od = os.path.join(_out_folder, _subfolder)
        _shared_differs = (os.path.abspath(_shared_od) != os.path.abspath(od))
    else:
        _shared_od = None
        _shared_differs = False
    _ckpt_local = os.path.join(od, "ckpt_atom2d.npz")
    _ckpt_shared = (os.path.join(_shared_od, "ckpt_atom2d.npz")
                    if _shared_od else None)
    _fp, _fp_key_names, _fp_file_names = _ckpt_fingerprint()
    print("[OPT2D] ckpt fingerprint sha256=%s (%d keys, ckpt_every=%d)"
          % (_fp, len(_fp_key_names), ckpt_every), flush=True)
    print("[OPT2D] ckpt fingerprint key names: %s"
          % (" ".join(_fp_key_names) if _fp_key_names else "(none)"),
          flush=True)
    print("[OPT2D] ckpt fingerprint files hashed: %s"
          % (", ".join(_fp_file_names) if _fp_file_names else "(none)"),
          flush=True)
    t_start = 1
    if ckpt_every > 0:
        _src = None
        if os.path.isfile(_ckpt_local):
            _src = _ckpt_local
        elif _shared_differs and os.path.isfile(_ckpt_shared):
            _src = _ckpt_shared
        if _src is not None:
            try:
                _cx, _cm, _cv, _ct, _cfp = _ckpt_load(_src)
            except Exception as _e:
                print("[OPT2D] WARN: ckpt read failed (%s) at %s -- "
                      "starting fresh" % (_e, _src), flush=True)
                _cx = None
            if _cx is not None:
                if _cfp != _fp:
                    print("[OPT2D] ckpt ignored: fingerprint mismatch (%s)"
                          % _src, flush=True)
                elif _cx.shape != x.shape:
                    print("[OPT2D] ckpt ignored: shape %s != %s despite "
                          "matching fingerprint (%s)"
                          % (_cx.shape, x.shape, _src), flush=True)
                elif _ct >= iters:
                    print("[OPT2D] ckpt ignored: saved iter %d >= iters %d "
                          "(%s)" % (_ct, iters, _src), flush=True)
                else:
                    x, m, v, t_start = _cx, _cm, _cv, _ct + 1
                    print("[OPT2D] resumed from ckpt at iter %d (%s)"
                          % (_ct, _src), flush=True)
    # [ckpt] The shared dir is created lazily, only when a shared checkpoint
    # is actually about to be written (inside the `ckpt_every > 0` save
    # block below) -- with `ckpt_every = 0` this driver must leave the file
    # system exactly as before the ckpt feature (no early folder at the
    # normal Result destination for a job that later fails).
    # [ckpt] A resumed run truncates the existing logs to iteration
    # `t_start - 1` and appends from there, so the finished files read
    # exactly as an uninterrupted run's would (T2). When the local `od`
    # was wiped between the crash and this resubmission (resume from the shared copy,
    # T3) there is nothing local to truncate -- `_local_log_ok` is False,
    # both files restart at "w" with a fresh header, and the row history
    # for iterations before `t_start` is not reconstructed (only the shared
    # `ckpt_atom2d.npz` was carried by the shared copy, not the per-iteration logs;
    # `rho3_opt.npy`/`atom2d_opt.json` stay bit-exact regardless, since
    # those come from `x`/`m`/`v` alone).
    _local_log_ok = (os.path.isfile(os.path.join(od, "opt_log.dat")) and
                     os.path.isfile(os.path.join(od, "opt_terms.tsv")))
    if t_start > 1 and _local_log_ok:
        _ckpt_truncate_log(os.path.join(od, "opt_log.dat"), t_start - 1)
        _ckpt_truncate_log(os.path.join(od, "opt_terms.tsv"), t_start - 1)
        _log_mode = "a"
    else:
        _log_mode = "w"
    lg = io.open(os.path.join(od, "opt_log.dat"), _log_mode, encoding="utf-8")
    if _log_mode == "w":
        lg.write("Iter\tFoM\tbeta\tFwd_s\n")
    # [260926] per-iteration term logging (new file only -- opt_log.dat
    # above is unchanged). See `_write_opt_terms_row`/`_ITER_CAPTURE`.
    tg = io.open(os.path.join(od, "opt_terms.tsv"), _log_mode, encoding="utf-8")
    _force_terms_header = (_log_mode == "w" and t_start > 1)
    # [ckpt-r3] Directories THIS JOB's checkpoint machinery created for the
    # shared copy, deepest first. `None` means "not yet known"; `[]` (once
    # known) means every level already existed independently of this job.
    # Only entries in this list are ever rmdir'd at cleanup, so a directory
    # that pre-dates the job (e.g. an `Output_folder` shared by many jobs,
    # or an empty `subfolder_name` left by an earlier, unrelated run) is
    # never touched. Recovered here from the `_shared_dirs_read` sidecar
    # when `_shared_od` already exists -- i.e. a killed-and-resumed job
    # whose earlier process created it and is not this one -- else left
    # `None` to be computed fresh (and the sidecar written) at this
    # process's own first shared save below.
    _shared_made = (_shared_dirs_read(_shared_od)
                    if _shared_od and os.path.isdir(_shared_od) else None)
    # ------------------------------------------------------------ early stop (261001)
    # [es] Adam-loop convergence early stop -- see the `es_window` docstring
    # block near the top of this file. Default 0 leaves every line below
    # inert and the driver bit-identical to before this feature.
    es_window = int(_num("es_window", 0))
    es_dA_deg = _num("es_dA_deg", 0.3)
    es_dfom_rel = _num("es_dfom_rel", 0.001)
    es_dprod_rel = _num("es_dprod_rel", 0.001)
    es_fid_gate = _num("es_fid_gate", 0.994 if pol == "TE" else 0.996722)
    es_fid_margin = _num("es_fid_margin", 0.001)
    es_dfid = _num("es_dfid", 1e-4)
    es_echo_margin_db = _num("es_echo_margin_db", 2.0)
    es_decho_db = _num("es_decho_db", 0.3)
    # [es] `w["echo_a2_gate"]` is already the linear-power gate the fom()
    # echo hinge itself uses (from either `echo_gate_db` or `echo_a2_gate`,
    # or 0.0 if the echo term is off), with the per-lane scale ALREADY
    # baked in at parse time (see `_echo_lane_scale` above, in the job-key
    # parsing section). The hinge switches on exactly at
    # `echo_a2_db > 10*log10(w["echo_a2_gate"])` (fom(), ~line 1058-1061)
    # -- so this guard must compare against that same quantity, with NO
    # further lane scaling. `None` means no echo gate is in play, so the
    # echo criterion is skipped.
    _es_echo_gate_db = (10.0 * math.log10(w["echo_a2_gate"])
                        if w["echo_a2_gate"] > 0.0 else None)
    _es_buf = []
    _es_stop_iter = None
    if es_window > 0:
        print("[OPT2D] early stop armed: window=%d dA<=%.3f deg "
              "dfom_rel<=%.4g dprod_rel<=%.4g fid_gate=%.6f "
              "(margin %.4g, dfid %.4g) echo_margin=%.2f dB (decho %.2f dB)"
              % (es_window, es_dA_deg, es_dfom_rel, es_dprod_rel,
                 es_fid_gate, es_fid_margin, es_dfid,
                 es_echo_margin_db, es_decho_db), flush=True)
    t0 = time.time()
    # [es] `_iters_run` must exist even when the loop body never executes
    # (`iters = 0`, no `eval_only` -- 88 stored job files use this, e.g.
    # `Result\260919_scdc_carprobe`). Initialised to `t_start - 1` (the last
    # iteration already done, 0 on a fresh run) before the loop, then
    # updated to the just-executed `t` as the loop's first statement --
    # never read from the bare `for` variable after the loop, which is
    # unbound when `range(t_start, iters + 1)` is empty.
    _iters_run = t_start - 1
    for t in range(t_start, iters + 1):
        _iters_run = t                                  # [es] last t executed
        beta = [b for f, b in sched if (t - 1) / float(iters) >= f][-1]
        _frac = (t - 1) / float(iters)
        _ramp = (0.0 if _frac < lw_start_frac else
                 (_frac - lw_start_frac) / max(1e-9, 1.0 - lw_start_frac))
        _ng_t = [_n for _f, _n in ng_sched if _frac >= _f][-1]
        val, g = _vg_by[_ng_t](x, beta, _ramp)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        x = np.clip(x - lr * (m / (1 - 0.9 ** t))
                    / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), 0.0, 1.0)
        lg.write("%d\t%.6f\t%.1f\t%.1f\n" % (t, -float(val), beta,
                                             time.time() - t0))
        lg.flush()
        _write_opt_terms_row(tg, t, first=(_force_terms_header and
                                           t == t_start))
        if t % 10 == 0 or t == 1:
            print("[OPT2D] it %4d  beta %5.1f  fom %.6f  %.0f s"
                  % (t, beta, -float(val), time.time() - t0), flush=True)
        # [es] Early stop window check -- eligible only in the FINAL nG
        # stage at the FINAL beta, tested by SCHEDULE FRACTION (`_frac`,
        # the same value the loop already uses to pick `_ng_t`/`beta`),
        # NOT by the nG/beta VALUE. A non-monotone schedule can revisit the
        # final stage's VALUE earlier (e.g. `0:25, 0.2:41, 0.4:25, 0.7:41`
        # revisits nG=41 at frac 0.2-0.4, before the real final stage at
        # frac >= 0.7) -- a value-only check would count those iterations
        # as eligible too. `_frac >= ng_sched[-1][0] and _frac >= sched[-1][0]`
        # is monotone in `t`, so once eligible this iteration stays
        # eligible for the rest of the run; the buffer is cleared on every
        # ineligible iteration so no stale entry from an earlier stage can
        # survive into the real final stage. Uses the PRE-update capture
        # (`_ITER_CAPTURE`, filled by the `fom()` call above) for the
        # stability quantities, per the docstring; the linewidth/fill gate
        # below is measured on the POST-update `x`, exactly like the
        # post-loop code, and does not mutate anything. The stop is only
        # ever evaluated at `t < iters` -- a criterion that first holds at
        # the LAST iteration records `es_stop_iter = None` and lets the
        # loop end normally (there is no later iteration to skip to).
        if es_window > 0:
            if _frac >= ng_sched[-1][0] and _frac >= sched[-1][0]:
                _es_row = dict(A_deg=_ITER_CAPTURE["A_deg"],
                              fom_total=-float(val),
                              prod=(_ITER_CAPTURE["amp_tA"]
                                    * _ITER_CAPTURE["amp_rBtB"]))
                for _k in ("fid_l3", "fid_l3_m", "echo_a2_db"):
                    if _k in _ITER_CAPTURE:
                        _es_row[_k] = _ITER_CAPTURE[_k]
                _es_buf.append(_es_row)
                if len(_es_buf) > es_window + 1:
                    del _es_buf[0]
                if (t < iters and len(_es_buf) == es_window + 1
                        and _es_ready(
                            _es_buf, es_dA_deg, es_dfom_rel, es_dprod_rel,
                            es_fid_gate, es_fid_margin, es_dfid,
                            _es_echo_gate_db, es_echo_margin_db, es_decho_db)):
                    _mf_es, _ = _min_feature_all(x, nxy, _emask, KF, 64.0, px_nm)
                    _ff_es = _fill_frac_all(x, nxy, _emask, KF, 64.0)
                    _ok_es, _live_es = _gate_with_fill(_mf_es, gate_nm, _ff_es)
                    if _ok_es:
                        _cur_es, _old_es = _es_buf[-1], _es_buf[0]
                        _dA_es = max(abs(_wrap180(b["A_deg"] - _cur_es["A_deg"]))
                                    for b in _es_buf)
                        print("[OPT2D] early stop at iter %d of %d (window %d): "
                              "dA %.3f deg dFoM %.2e dprod %.2e lw %.1f nm"
                              % (t, iters, es_window, _dA_es,
                                 abs(_cur_es["fom_total"] - _old_es["fom_total"]),
                                 abs(_cur_es["prod"] - _old_es["prod"]),
                                 _mf_es), flush=True)
                        _es_stop_iter = t
                        break
            else:
                _es_buf = []
        if ckpt_every > 0 and t % ckpt_every == 0 and t != iters:
            _saved_ok = _ckpt_save(_ckpt_local, x, m, v, t, _fp)
            if _shared_differs:
                _shared_made_fresh = False
                if _shared_made is None:
                    # [ckpt-r3] First shared save THIS PROCESS knows of --
                    # record, deepest first, every directory level from
                    # `_shared_od` upward that does not exist YET (i.e.
                    # before `os.makedirs` below creates it). Stops at the
                    # first level that already exists, since everything
                    # above that already exists too. (`_shared_made` was
                    # already set, above the loop, when a sidecar from an
                    # EARLIER process's own such save was found first.)
                    _shared_made = []
                    _walk = os.path.abspath(_shared_od)
                    while _walk and not os.path.isdir(_walk):
                        _shared_made.append(_walk)
                        _parent = os.path.dirname(_walk)
                        if _parent == _walk:
                            break
                        _walk = _parent
                    _shared_made_fresh = True
                try:
                    os.makedirs(_shared_od, exist_ok=True)
                except OSError as _e:
                    print("[OPT2D] WARN: shared ckpt dir failed (%s) -- "
                          "local ckpt is intact" % _e, flush=True)
                else:
                    if _shared_made_fresh:
                        # [ckpt-r3] Persist the list just computed so a
                        # LATER process (this job killed and resumed again)
                        # can recover it -- see `_shared_dirs_read` above
                        # `main()`.
                        _shared_dirs_write(_shared_od, _shared_made)
                    _ckpt_save(_ckpt_shared, x, m, v, t, _fp)
            if _saved_ok:
                print("[OPT2D] ckpt saved at iter %d -> %s" % (t, _ckpt_local),
                      flush=True)
    lg.close()
    tg.close()
    if DISTINCT_ATOMS:
        # [261003] 4-D (atoms, layers, NXY, NXY) under a DIFFERENT name: no
        # existing consumer globs rho3_cell_opt.npy, none can read it as an atom.
        np.save(os.path.join(od, _npy_name),
                np.asarray(x).reshape(ATOMS_PER_PERIOD, max(1, int(L3_LAYERS)),
                                      nxy, nxy))
    else:
        np.save(os.path.join(od, _npy_name), x)
    # [ckpt-r2] Checkpoint cleanup moved to AFTER `atom2d_opt.json` is
    # written (see the bottom of this function) -- a failure in the final
    # `nG_fin` measurement below (`_measure_arms`/`_measure_rho`) must keep
    # the resume point, not discard it along with the half-finished report.
    # [260826] **Gate measurement.** The filter radius only derives the scale and does not guarantee it.
    # [260917] The stored `x` is the raw variable, so values remain outside the mask too. The physical shape is
    # the one multiplied by the mask, so it must be multiplied in the same order here for the measurement to be right.
    mf, _mf_each = _min_feature_all(x, nxy, _emask, KF, 64.0, px_nm)
    _ff = _fill_frac_all(x, nxy, _emask, KF, 64.0)
    if len(_mf_each) > 1:
        print("[OPT2D] minimum linewidth per layer %s nm -- the worse one is used"
              % ", ".join("%.1f" % z for z in _mf_each), flush=True)
    _ok, _live = _gate_with_fill(mf, gate_nm, _ff)
    print("[OPT2D] fill per layer %s -- layers with structure %d/%d"
          % (", ".join("%.4f" % z for z in _ff), len(_live), len(_ff)),
          flush=True)
    if not _live:
        print("[OPT2D] **atom is empty** -- the linewidth %.1f arises from having no corners, "
              "so the gate is not passed" % mf, flush=True)
    print("[OPT2D] measured minimum linewidth %.1f nm (gate %.0f) -> %s"
          % (mf, gate_nm, "pass" if _ok else "**fail**"), flush=True)
    # [260917] **Record the achieved values.** Previously this path left only the gate and settings,
    # so to know how well a 31-minute optimisation met its own targets (`absR_target` / `prod_target` /
    # `phi_target_deg`) one had to run an `eval_only` job once more.
    # The `eval_only` branch already builds this at `:700`.
    # `_measure_arms` is report-only, so it does not touch `make_fom`/`fom`.
    # The cost is 2 solves at `nG` (about 11 s = 0.5 % for a 2079 s job).
    _info = _measure_arms(nG_fin, rho12, H, sp, n_sp, eps3, lam, nxy, KF,
                          x, 64.0, pol=pol, symy=_symy, emask=_emask)
    print("[OPT2D] achieved  |r_B| %.4f (target %.4f)  arg r_B %.2f deg  "
          "product %.4f (target %.4f)  A %.2f  B %.2f"
          % (_info["amp_rB"], w["absR_t"], _info["argrB_deg"],
             _info["prod_amp"], w["prod_t"], _info["A_deg"], _info["B_deg"]),
          flush=True)
    print("[OPT2D] done. rho3_opt.npy saved (nG_eval=%d verification is a separate job)"
          % nG_eval, flush=True)
    # [260926] `r00win` report keys -- see the matching comment in the
    # eval_only branch above; only added when the term is in play.
    _r00win_extra = (
        {"r00win": [list(_r) for _r in w["r00win_rows"]],
         "w_r00win": float(w["w_r00win"]),
         "r00win_inside": _r00win_inside(
             _info["absR00"], _info["argR00_deg"], w["r00win_rows"]),
         "r00win_axis": w["r00win_axis"],
         "r00win_rows_face": [list(_r) for _r in w["r00win_rows"]]}
        if (w["w_r00win"] > 0.0 or w["r00win_rows"]) else {})
    # [260928] Carrier-token-axis mirror of argR00_deg -- see the
    # `_r00_off_deg` comment near `lam` in main(). Present only when
    # argR00_deg itself is (sp > 0), same guard `_r00win_extra` needs.
    _r00tok_extra = (
        {"argR00_token_deg": (_info["argR00_deg"] - _r00_off_deg) % 360.0,
         "r00_token_offset_deg": _r00_off_deg}
        if "argR00_deg" in _info else {})
    # [261001] Early-stop report keys -- only added when the feature is
    # armed (`es_window > 0`); default JSON is unchanged.
    _es_extra = (
        {"es_window": es_window, "es_dA_deg": es_dA_deg,
         "es_dfom_rel": es_dfom_rel, "es_dprod_rel": es_dprod_rel,
         "es_fid_gate": es_fid_gate, "es_fid_margin": es_fid_margin,
         "es_dfid": es_dfid, "es_echo_margin_db": es_echo_margin_db,
         "es_decho_db": es_decho_db, "es_stop_iter": _es_stop_iter,
         "iters_run": _iters_run}
        if es_window > 0 else {})
    _rho_info = {}
    # [261003] distinct_atoms report keys (empty dict when off -> JSON unchanged).
    _distinct_extra = (_distinct_report(x, nxy, _emask, KF, 64.0, px_nm)
                       if DISTINCT_ATOMS else {})
    if measure_rho:
        _r3o = _project_cell(x, nxy, _symy, _emask, KF, 64.0)
        _rho_info = _measure_rho(nG_fin, _r3o, H, sp, n_sp, eps3, lam, nxy)
        print("[OPT2D] rho (ref=%s, theta=%.2f deg)  |rho_TM|=%.6f "
              "arg=%.1f deg  |rho_TE|=%.6f arg=%.1f deg"
              % (_rho_info["rho_ref"], _rho_info["rho_theta_deg"],
                 _rho_info["abs_rho_TM"], _rho_info["arg_rho_TM_deg"],
                 _rho_info["abs_rho_TE"], _rho_info["arg_rho_TE_deg"]),
              flush=True)
    io.open(os.path.join(od, _json_name), "w",
            encoding="utf-8").write(json.dumps(
                dict(pol=pol, min_feature_meas_nm=float(mf),
                     min_feature_gate_nm=float(gate_nm), gate_ok=bool(_ok),
                     fill_frac_each=[float(z) for z in _ff],
                     n_live_layers=len(_live),
                     filt_radius_nm=float(rad * px_nm),
                     sym_y=int(_symy), edge_margin_px=_emg,
                     **_emask_extra,
                         w_lw=w["w_lw"], lw_start_frac=lw_start_frac,
                     l3_layers=int(L3_LAYERS), l3_split=float(L3_SPLIT),
                     # [260923] **Record the pitch and the background in the result.** `atoms_per_period`
                     #   changes the physics entirely (at Lambda/2 the atoms supply the echo order
                     #   2G at first order, at Lambda/3 all of that is evanescent), yet it was
                     #   not in the result JSON, so which pitch a stored cell ran at could not be
                     #   known from the JSON alone. To check 322 cells in 260923 the job copies in the
                     #   folders had to be read one by one -- that route is capped at 100 entries,
                     #   so it breaks at older blocks.
                     #   `l3_fill_n` / `l3_etch_stack` are the same: whether the lower layer
                     #   background of a 2-layer atom is air or SiO2 changes the structure.
                     atoms_per_period=int(ATOMS_PER_PERIOD),
                     l3_fill_n=float(L3_FILL_N),
                     l3_etch_stack=int(L3_ETCH_STACK),
                     nG=nG, nG_eval=nG_eval, NXY=nxy, iters=iters, lr=lr,
                     H_nm=[h * 1e3 for h in H], spacer_nm=sp * 1e3,
                     weights=w, elapsed_s=time.time() - t0,
                     **({"ng_schedule_used": [[f, n] for f, n in ng_sched],
                         "nG_report": nG_fin} if _ngs_raw else {}),
                     **_gap_extra, **_li_extra, **_info, **_rho_info,
                     **_r00win_extra, **_r00tok_extra, **_es_extra, **_distinct_extra),
                ensure_ascii=False, indent=2))
    # [ckpt-r2] Success -- `atom2d_opt.json` above is the last thing this
    # run needs to produce, so the checkpoint's job is done. Remove it
    # (local and shared) so a later resubmission of this same job file
    # starts fresh rather than reading a stale/completed state. A removal
    # failure is reported, not swallowed -- a stale ckpt left in the final
    # Result folder would otherwise resume a later resubmission from it
    # silently. Leftover `.tmp_*` files from a save that lost the
    # `os.replace` race (or crashed mid-write) are swept the same way,
    # best-effort. Moved here (was right after `rho3_opt.npy` above) so a
    # failure in the `nG_fin` measurement/`atom2d_opt.json` write keeps the
    # checkpoint intact -- see the note above `_measure_arms` (T8).
    # [ckpt-r3] The `_shared_dirs_read`/`_shared_dirs_write` sidecar (see
    # above `main()`) lives beside `_ckpt_shared` and is cleaned up in the
    # same pass, for the same reason -- it is scratch state for this job's
    # own checkpoint machinery, not a result file.
    _shared_marker = (os.path.join(_shared_od, _SHARED_DIRS_MARKER)
                      if _shared_od else None)
    for _p in (_ckpt_local, _ckpt_shared, _shared_marker):
        if _p and os.path.isfile(_p):
            try:
                os.remove(_p)
            except OSError as _e:
                print("[OPT2D] WARN: ckpt cleanup failed (%s) at %s"
                      % (_e, _p), flush=True)
    for _d in {os.path.dirname(_p) for _p in (_ckpt_local, _ckpt_shared)
               if _p and os.path.isdir(os.path.dirname(_p) or ".")}:
        try:
            for _tmp in glob.glob(os.path.join(_d or ".",
                                                ".ckpt_atom2d.npz.tmp_*")):
                try:
                    os.remove(_tmp)
                except OSError:
                    pass
        except OSError:
            pass
    # [ckpt-r3] If this run itself created directory levels for the shared
    # staging path (`_shared_made`, deepest first -- see the walk at the
    # first shared save above) purely to hold the checkpoint just removed,
    # leaving them behind as empty directories is a stray artefact at the
    # eventual copy destination. Remove each one iff it is now
    # empty, deepest first so a parent only goes empty once its child is
    # gone -- and ONLY directories this run created. Never touch a
    # directory that pre-dates the run: not `Output_folder` (shared across
    # many jobs), and not a `subfolder_name` some earlier run already left
    # behind (empty or not) -- that one is simply absent from
    # `_shared_made`.
    for _d in (_shared_made or []):
        try:
            if os.path.isdir(_d) and not os.listdir(_d):
                os.rmdir(_d)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
