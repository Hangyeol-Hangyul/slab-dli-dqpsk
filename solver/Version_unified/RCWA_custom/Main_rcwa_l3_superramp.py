# -*- coding: utf-8 -*-
"""L3 supercell phase-ramp (blazed grating) LPA-error probe. v4, 260927.

v4 adds the TE lane D-C left out of scope (this is the "D-C, done
properly" fix), plus one enabling generalisation (D-E). Everything else
below (F1-F7, M2, M8, D1, D-A/D-B/D-D, the caveats) is UNCHANGED from v3
-- the TM path is bit-identical to v3 on every job that does not set
`atoms_meta` (verified, see the t4_te regression needles).

    D-C (v4) pol=TE is now supported. v3's hard error diagnosed the exact
        defect: this file always read rB (the delay-arm's non-diffracted
        reflection) at order 0 @ +th, which is the TM small-cell design
        order (Main_rcwa_l3_atom2d_opt.py:1278, `rB = _amp(obp, 0,
        refl=True)`) but WRONG for TE, whose own convention
        (:1265-1272 -- `_measure_arms`'s TE branch, confirmed by reading,
        not by the line numbers alone) reads rB at order **+1 @ theta=0**
        (`rB = _amp(ob0, +1, refl=True, pol="TE")`, off the SAME solve
        that gives tA -- TE arm A illuminates at normal incidence, not
        +THETA_SUB). `_measure_super`'s `thA/thB`/`gA0/gB0` selection by
        pol was ALREADY generic in v3 (:730-735) -- only the rB order was
        hardcoded to (0,0). Fixed via `_rb_small_order(pol)` (0 for TM,
        +1 for TE), applied everywhere rB's order is read or reported:
        `_measure_super`'s rB0 scan, `_lpa_ref_from_solve`'s rB reference,
        the N1x/N1y needle's independent single-cell rB check, and
        `translation_phase_deg`'s `rB_zero` entry (which used a bare
        `_tphase_deg(0)` -- now `_tphase_deg(rb_base)`, matching the same
        x-ramp K-scaling every other named order already gets via `mk`).
        Every other TE-relevant piece (`_obj_super`'s excitation,
        `_full_order_scan`'s amplitude offset, `_fid_l3_super`'s offset)
        was ALREADY pol-generic in v3 and needed no change -- confirmed by
        reading each, not assumed. `main()`'s hard error is narrowed to
        `pol not in ("TM", "TE")`; F7's existing `pol` cross-check
        (job pol vs atoms' own pol, already in v3) is unchanged and still
        applies to both lanes.
    D-E (v4) `atoms_meta` -- an optional job key, ';'-separated
        directories in the same order as `atoms`, giving each atom's
        `atom2d_opt.json` (+ job_*.txt fallback) directory EXPLICITLY when
        it differs from the atom's own npy directory. v3's `_load_atom`
        assumed both files live next to the npy -- true for every TM
        palette atom (optimised in place), but the TE palette's atoms are
        SCORED rho3 arrays pre-rotated by an external step
        (`_rotated_rho3\\rot90_*.npy`, no companion json there at all --
        the json + job_*.txt for that same rotated array live in a
        SEPARATE eval_only result folder). Omitting `atoms_meta` (or
        giving `""` for one atom's slot) reproduces v3's exact behaviour
        (meta dir = npy's own dir) -- bit-identical default, never forced.

v3 fixes the 4 blocking defects (D-A..D-D) and 3 caveats the 260926
Verify-A pass found in v2. Each is tagged at the point it lands:

    D-A the `lpa` block's per-`s` lookup was keyed by the LITERAL loop
        index (`{-1: ghost, 0: zero, 1: target}` always), correct only
        for `ramp_s=+1`; at the default `ramp_s=-1` target and ghost were
        read SWAPPED. Fixed: keyed by order relative to `ramp_s`
        (`{ramp_s: target, -ramp_s: ghost, 0: zero}`), in `_measure_super`.
    D-B arm B compared measured tB alone against a predicted rB*tB
        (identical-atom control gave ratio 1/|rB|^2, not 1). Fixed:
        `_lpa_ref_from_solve`/`_lpa_ref_from_jsons` store tB and rB
        SEPARATELY; `_lpa_predict_v2` predicts the DFT of tB alone for
        arm B; rB(0) (not diffracted by the ramp) is compared separately
        against the plain mean of the per-atom rB (`lpa["rB0"]`).
    D-C pol=TE is out of scope for this campaign and this driver's TE
        convention is known-wrong -- hard error, exit 2, in `main()`
        right after `pol` is parsed. Not attempted to be fixed.
    D-D `nG_ref` was defaulted to the bare `nG` (a routine harmonic
        mismatch, measured 15.6% error). Fixed: `_nG_ref_check` (wrapping
        `_matched_single_nG`, the inverse of `_matched_super_nG`) requires
        `nG_ref` to be harmonic-matched to `nG` -- explicit or defaulted
        to the matched value -- and records the requested/actual (single,
        super) harmonic counts in the output JSON; exit 2 on mismatch.
        Applied in both the production branch and the N4 needle.

    Caveats fixed:
    - `wall_s` (job-level, wraps the WHOLE measurement incl. the F5
      reference solves) is now written alongside `elapsed_s` (which stays
      narrowly scoped to the two supercell builds, now documented as such
      at its own assignment).
    - F7's hard-error check is extended to n_spacer/n_sub/lams_nm/pol/
      material3/l3_fill_n/n_l3_gap. These are not all in the eval_only
      json (n_spacer/n_sub/lams_nm/material3 never are) -- `_load_atom`
      now falls back to the atom's own companion job_*.txt for them, the
      same pattern already used for l3_split; a key present in NEITHER
      source raises a clean ValueError (`_consistent`'s KeyError is now
      caught and re-raised as one) -- exit 2, never a KeyError crash.
    - The sibling-import path order now puts the driver's OWN folder
      ahead of the share ground truth (ground truth is the fallback for a
      sibling missing locally, not the default) -- see the `sys.path`
      setup below.

Everything else below is unchanged from v2 (F1-F7, M2, M8, D1 as
originally landed) except where a D-A..D-D fix required touching the
same function.

EVAL ONLY -- no optimisation. Builds a supercell containing M atoms (M
distinct stored rho3 densities) next to the real carrier (L1/L2), either
tiled along the carrier's grating direction (ramp_axis=x) or stacked along
the atom's second in-plane direction (ramp_axis=y), and measures how much
power the resulting phase ramp sends into the intended deflected order vs
how the plain local-periodicity-approximation (LPA) single-atom prediction
says it should.

v2 fixes the defects found by the 260926 verifier report and the scoping
review (see the READ FIRST paths in the task that produced this file).
Each fix is tagged Fn / Dn / Mn at the point it lands so the mapping to
that report is traceable. Summary:

    F1  needle=none (default) is now a PRODUCTION measurement on the job's
        own atoms/ramp_axis/ramp_s/sc_shift/nG -- the old driver only ever
        ran the hard-coded needle set and a `needle=none` job silently
        wrote an empty result.
    F2  BLAS thread env vars are set from `blas_threads` BEFORE numpy (or
        anything that imports it) is touched anywhere in this file, and
        the grcwa backend is forced to "numpy" (this driver never takes a
        gradient). The old file did `import numpy as np` at module scope
        BEFORE importing the atom driver (whose import is what used to set
        the env vars) -- numpy/grcwa read the *inherited* environment in
        that window, in which the thread count is unbounded.
    F3  the driver no longer pins its own CPU affinity. CPU affinity is
        set externally right after launch; a
        second pin from inside the driver overrides it.
    F4  rB is read at r(0) @ +th (matches Main_rcwa_l3_atom2d_opt.py:1278,
        `rB = _amp(obp, 0, refl=True)`) -- the old file read the WRONG
        order (it reused `base_gB*K`, i.e. r(+K), which the design never
        intends to be reflective and is often evanescent).
        Echo is read at R(+2*K) (matches :1298-1313, carrier-frame order
        is always +2 regardless of ramp_s -- it is a property of the
        RETRO grating's re-entry, not of the L3 ramp) -- the old file read
        R((gB0+2)*K) = R(3*K), one order off. Echo (and every propagating
        order) is now read as RT POWER via `_full_order_scan`
        (`ob.RT_Solve(byorder=1)`, which itself already sums BOTH
        polarisation halves per order -- see `GetZPoyntingFlux`,
        `forward = forward_xy[:n] + forward_xy[n:]` in
        _vendor/grcwa/rcwa.py:918) -- amplitude-squared is reported
        alongside, separately labelled, never substituted for power.
        The two side orders 2K-1 and 2K+1 are also reported.
    F5  the LPA reference is built either from single-atom solves on the
        SAME carrier the supercell run uses (`ref_mode=solve`, default --
        this also fixes the old file's defect of scoring 4/8 palette
        atoms against a DIFFERENT carrier than the supercell), or from
        supplied reference JSONs (`ref_mode=jsons`, `ref_jsons=...`).
        s = -1, 0, +1 are predicted on an amplitude basis AND on a
        kz-weighted power basis (the conversion factor is read from the
        SAME solved supercell at the corresponding order -- see
        `_measure_super`'s `lpa` block). A target order that this
        driver's own solve shows is non-propagating in air (exact test via
        the layer's eigenvalue `ob.q_list`, not a power threshold) is
        marked `propagates=False` and its power-basis ratio is omitted;
        the s=0 (zero-order) comparison is always reported.
    F6  the old N1 needle compared x M=3 against itself (K=1, so it tested
        nothing) and only checked tA. It is replaced by matched-harmonic
        identical-atom checks at x M=6 and y M=2 (the supercell nG is
        searched so its retained-harmonic sublattice reproduces the
        single-cell's set EXACTLY -- `_matched_super_nG`, ported from the
        verifier's vcommon.py), comparing EVERY arm key (tA, rB, tB, echo
        power, fid, fid_m, r00) to <= 1e-9 relative. N3 is now the y-mirror
        test only (y -> -y is the one EXACT symmetry available here -- the
        old file's x-reversal "N3" was against oblique incidence and the
        report already flagged its 1.95% agreement as not a real test).
    F7  a mismatch between the job's H1_nm/H2_nm/spacer_nm (always
        supplied) or H3_nm/l3_split/l3_gap_nm (optional job keys, checked
        only when given) and the atoms' own json values is now a hard
        error, exit 2 -- the old file only printed a WARNING for H1/H2/
        spacer and never checked H3/split/gap against the job at all.

Review corrections (scoping-review defects D1/D2/M1/M2/M8, applied to
THIS driver, not to the higher-level supercell-vs-lens-echo question that
review was actually about):
    D1  every propagating-order quantity reported here is RT-solved POWER
        (`_full_order_scan`, already both-polarisation-half-summed per
        grcwa's own convention), never |amplitude|^2 substituted for it.
        Amplitude-squared is kept too, explicitly labelled, for phase/
        coherent-averaging use (F5, sc_shift below) -- never confused with
        power.
    M2  `ramp_s` is a job key in {-1, +1}; x-ramp's PRIMARY direction is
        -1 (default), +1 is available as a control.
    M8  `sc_shift` in {0, 1, 2} (x-ramp only) cyclically re-registers the
        tiled carrier under the FIXED atom sequence by that many atom-
        pitches (see `_obj_super`) -- physically a rigid translation of
        the whole M*a_pitch-periodic assembly by an integer number of
        atom-pitches, so supercell order m's amplitude picks up the exact
        Bloch-shift phase factor exp(-i*2*pi*m*sc_shift/M). That phase is
        reported per key (`translation_phase_deg`) so it can be divided
        back out and average the complex target amplitude coherently over
        the 3 registrations (and over independent grating offsets more
        generally) instead of averaging power.
    (fid) `_fid_l3_super` below is a small, LOCAL, axis-symmetric
        generalisation of the ground-truth `_fid_l3` (its denominator sums
        every retained (g0,g1), not only g1==0) so the reported quantity
        means the same thing on the x-ramp and the y-ramp axis. It is
        kept STRICTLY as a diagnostic here, exactly as it already was --
        this driver never gates on it.

Reuses the ground-truth atom driver's stack-building and measurement code
by IMPORTING it (import order matters for F2 -- see below). Cited source
lines are in that file unless noted:

    _obj            (stack builder for ONE tiled atom)          :308-389
    _amp            (single-order complex amplitude)            :392-408
    _ord_power      (single-order RT power, NOT |amp|^2)         :417-427
    _ledger         (fixed -3..+3 order R/T ledger)              :430-447
    _fid_l3         (L3-face illumination fidelity)              :1031-1065
    _r00_l3         (design-channel atom-alone reflection)       :1068-1090
    _project3       (raw design var -> projected density)        :265-305
    _measure_arms   (report-only arm A/B measurement, the        :1253-1412
                     convention every order/index choice below
                     is copied from)
    main() eval_only branch (npy load / resample / rescoring)    :1969-2060

This file's `_obj_super` mirrors `_obj` but places M DIFFERENT projected
atoms into an enlarged supercell instead of tiling ONE atom `app` times;
`_full_order_scan` / `_scan_order` generalise `_amp` / `_ledger` to match an
arbitrary (g0, g1) pair instead of a fixed small range with g1 pinned to 0.
`_fid_l3` / `_r00_l3` (the r00 one) are reused UNMODIFIED (they only do
integer order matching, no lattice-constant arithmetic, so they generalise
exactly given the right `design_m`) -- `_fid_l3` itself is superseded by
the local `_fid_l3_super` above for REPORTING (same reason: axis symmetry),
but `_r00_l3` needs no such fix (its g[1]==0 restriction is correct on
both axes -- r00 is a property of the atom-alone design channel, which by
definition sits at g1==0 regardless of which axis carries the ramp).
`_spacer_probe` / `G.spacer_purity` do NOT generalise (their order->kx
arithmetic hardcodes `G.PERIOD` as the fundamental period) and are
therefore not used here at all (they were only ever exercised inside the
old N1 needle at M=3, which is retired by F6).

job keys
    l12_density       carrier L1/L2 density npy (2, Ncol), same file the
                      atom driver's `l12_density` key takes.
    H1_nm H2_nm       carrier heights (required).
    H3_nm             OPTIONAL -- if given, hard-checked (F7) against the
                      atoms' own json; if absent, taken from the atoms.
    l3_split l3_gap_nm  OPTIONAL -- same hard-check-if-given rule as H3_nm.
    spacer_nm n_spacer n_sub period_nm material3 lams_nm pol
    NXY               nominal per-atom grid (cross-checked against each
                      atom's own json NXY, via `_consistent`).
    ramp_axis         "x" or "y"
    ramp_s            -1 or +1 (default -1 -- M2, x-ramp primary).
    sc_shift          0, 1 or 2 (default 0; x-ramp only -- M8).
    atoms             ';'-separated rho3_opt.npy paths, in ramp order.
                      Each file's directory must also hold that atom's
                      atom2d_opt.json (written next to it by the atom
                      driver -- see :2040-2060 / :2150-2160) UNLESS
                      `atoms_meta` (below, v4) gives that atom's meta
                      directory explicitly.
    atoms_meta        OPTIONAL (v4, D-E) -- ';'-separated directories,
                      1:1 with `atoms`, each holding that atom's
                      atom2d_opt.json (+ job_*.txt fallback) when it is
                      NOT the npy's own directory (an empty slot, e.g.
                      "path1;;path3", falls back to that atom's own npy
                      directory -- v3's only behaviour). Needed for the
                      TE palette, whose rotated rho3 arrays and their
                      eval_only score jsons live in separate folders.
                      [261002] Each atom's edge mask is rebuilt from ITS
                      OWN record (`ATOM.edge_mask_spec`): edge_margin_x_px /
                      edge_margin_y_px / edge_margin_mode in its json, else
                      edge_margin_px (both axes, 'pre' -- the old rule),
                      else its companion job_*.txt. For a rotated TE atom
                      the json read here must be the one that scored the
                      ROTATED array (a rot90 swaps the x and y margins).
    nG                nG for the measurement / needle runs (reduced --
                      this driver is not meant to reach nG_eval=601-per-
                      atom equivalent in one shot).
    nG_eval           reference "per-atom" nG, carried through to the
                      output only (labelling, no computation depends on
                      it).
    needle            "none" (DEFAULT, F1 production mode), or one of
                      "N1x", "N1y", "N2", "N3", "N4", "all" (runs the
                      validation needles instead of a production
                      measurement).
    ref_mode          "solve" (DEFAULT, F5) or "jsons".
    nG_ref            nG for the F5 ref_mode=solve single-atom solves
                      (default = `nG`).
    ref_jsons         F5 ref_mode=jsons only -- ';'-separated
                      atom2d_opt.json paths, SAME ORDER as `atoms`.
    blas_threads
    result_folder / Output_folder + subfolder_name   output location
"""
import io
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# sys.path order: the driver's own folder first, its _vendor folder second.
for _p in (os.path.join(_HERE, "_vendor"), _HERE):
    while _p in sys.path:
        sys.path.remove(_p)
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
# The atom driver imported below parses the SAME job file with the same rule, so ATOM._obj (the LPA
# reference solves) and `_obj_super` always use the same factorization; a non-default job refuses to
# run when the imported atom driver is a stale copy that cannot do it.
def _li_parse(cfg):
    rule = (cfg.get("rcwa_rule", "").strip().lower() or "laurent")
    ffr = (cfg.get("rcwa_ff", "").strip() or "0")
    pinv = (cfg.get("rcwa_pol_epsinv", "").strip().lower() or "conv1")
    try:
        ff = float(ffr)
    except ValueError:
        ff = -1.0
    if rule not in ("laurent", "liy", "lix") or ff not in (0.0, 1.0) or pinv != "conv1":
        print("[SUPERRAMP] ERROR: rcwa_rule=%r (laurent|liy|lix), rcwa_ff=%r (0|1), rcwa_pol_epsinv=%r "
              "(this driver has no Pol path: only conv1)" % (rule, ffr, pinv), flush=True)
        sys.exit(2)
    return rule, int(ff)


_RCWA_RULE, _RCWA_FF = _li_parse(CFG)
_LI_ON = (_RCWA_RULE != "laurent") or bool(_RCWA_FF)
# result-JSON extras: present ONLY when non-default (the default JSON stays byte-identical)
_li_extra = ({"rcwa_rule": _RCWA_RULE, "rcwa_ff": int(_RCWA_FF)} if _LI_ON else {})
# [261012 lam_eval] OPT-IN band evaluation of a FIXED lens: lam_eval_override = 1 lets the job lams_nm differ from the
# atoms' design lams_nm (F7 skips only that one check; the LPA references are solved at the job wavelength anyway, F5).
# The result JSON then records lam_eval_override / lam_eval_nm / atom_lams_nm. Default 0 = previous behaviour, byte for byte.
_LAM_OVERRIDE = (CFG.get("lam_eval_override", "").strip() or "0") in ("1", "1.0")

# ---- F2: BLAS thread env vars set BEFORE numpy is imported anywhere -----
# (this must happen before `import numpy`, `import grcwa` and `import
# Main_rcwa_l3_atom2d_opt` -- all three end up loading numpy/BLAS, and the
# defect measured 260926 was exactly this driver importing numpy at
# module scope BEFORE the atom-driver import that used to set these vars).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = (CFG.get("blas_threads", "").strip() or "1")

import numpy as np  # noqa: E402  -- AFTER the env vars above (F2)

import grcwa  # noqa: E402
grcwa.set_backend("numpy")  # noqa: E402  -- F2: this driver is EVAL ONLY,
# never differentiates, so the backend is forced regardless of what the
# job file's `eval_only` key says (defensive: do not depend on the atom
# driver's own CFG-gated choice, which this file also inherits below).

# [261008 F2/D2] This driver is Li-aware (it parses rcwa_rule / rcwa_ff itself, builds `_obj_super` with them and
# checks below that ATOM carries the same setting), so as the PROCESS ENTRY it declares that to the atom driver,
# which otherwise refuses a non-default rcwa_rule / rcwa_ff for any importer. Must stay ABOVE the import.
LI_AWARE_ENTRY = True

import Main_rcwa_l3_atom2d_opt as ATOM  # noqa: E402  (side effects: see
# module docstring above -- re-parses the SAME sys.argv[1] job file and
# re-applies the identical BLAS env vars, a harmless no-op at that point;
# does not run main() unless invoked as __main__)

# ---- F2 (continued): re-force numpy AFTER the ATOM import chain. -------
# `Main_rcwa_l3_atom2d_opt` imports `scd_c_grating as G` (its own :99),
# and `scd_c_grating.py:70` does an UNCONDITIONAL module-level
# `grcwa.set_backend("autograd")` (for its own optimizer's
# `value_and_grad`) -- that import runs strictly AFTER the `set_backend
# ("numpy")` call above, so it silently undoes it every time (measured:
# `grcwa.backend.__class__.__name__` reads back "AutogradBackend" right
# after the ATOM import completes, even with `eval_only=1`). This driver
# never differentiates, and the autograd backend's `eig`/`inv` route
# through autograd's VJP primitives regardless of whether a gradient is
# ever taken, which is far slower per call (scd_c_grating.py:820-826
# documents the identical trap for its own `_rho_trapped`-style probe:
# "~1s under numpy ... >8 minutes ... under autograd"). Force it back,
# AFTER the import chain has had its say, so this call is the last word.
grcwa.set_backend("numpy")

G = ATOM.G

if _LI_ON:
    # [261008 li_integrate] opt-in only: the atom driver must carry the identical setting
    if (getattr(ATOM, "_RCWA_RULE", None), getattr(ATOM, "_RCWA_FF", None)) != (_RCWA_RULE, _RCWA_FF):
        print("[SUPERRAMP] ERROR: the imported Main_rcwa_l3_atom2d_opt does not carry rcwa_rule=%s "
              "rcwa_ff=%d (stale atom driver?) -- refusing to run a Li job on the stock rule"
              % (_RCWA_RULE, _RCWA_FF), flush=True)
        sys.exit(2)
    if not getattr(grcwa.obj, "LI_SUPPORT", 0):
        print("[SUPERRAMP] ERROR: the vendored grcwa in use has no Li support (obj.LI_SUPPORT missing)",
              flush=True)
        sys.exit(2)

# ---- F3: NO cpu_affinity pin here. The old file pinned to logical cores
# >= 10 at import time, which overrides an affinity set externally
# right after launch. Any affinity
# pinning for a manual/interactive run belongs in the TEST SCRIPT that
# invokes this driver, never inside the driver itself.


def _num(k, d):
    try:
        return float(str(CFG.get(k, "")).strip())
    except (TypeError, ValueError):
        return float(d)


def _txt(k, d=""):
    return (CFG.get(k, "") or d).strip()


# ------------------------------------------------------------ atom loading
def _companion_job_kv(meta_d):
    """key -> value strings of the first job_*.txt in `meta_d` ({} if none),
    read the same way `_load_atom`'s other job fallbacks read it. Used only
    for the edge-mask fallback (`ATOM.edge_mask_spec`)."""
    kv = {}
    for fn in os.listdir(meta_d):
        if fn.startswith("job_") and fn.endswith(".txt"):
            for ln in io.open(os.path.join(meta_d, fn), encoding="utf-8",
                              errors="replace").read().split("\n"):
                if "=" not in ln or ln.lstrip().startswith("#"):
                    continue
                k, _, v = ln.partition("=")
                kv[k.strip()] = v.strip()
            break
    return kv


def _load_atom(npy_path, meta_dir=None):
    """One palette atom: (r3_projected, meta_dict). `r3_projected` is a
    single (nxy, nxy) array or a 2-list for a bilayer atom -- output of
    `ATOM._project3`, exactly as `_measure_arms` consumes it (:1264).

    [v4, D-E] `meta_dir`, when given (non-empty), is where this atom's
    atom2d_opt.json (+ job_*.txt fallback) lives, when that is NOT the
    npy's own directory -- the TE palette's case (rotated rho3 arrays
    under `_rotated_rho3\\`, no companion files there at all). Default
    (None or "") reproduces v3 exactly: meta dir = npy's own directory."""
    d = os.path.dirname(os.path.abspath(npy_path))
    meta_d = os.path.abspath(meta_dir) if meta_dir else d
    jf = os.path.join(meta_d, "atom2d_opt.json")
    if not os.path.isfile(jf):
        raise ValueError("no atom2d_opt.json next to %s" % npy_path)
    meta = json.load(io.open(jf, encoding="utf-8"))
    # [260927 FIX, defect 5] The (npy_path, meta_dir) PAIRING itself was
    # never checked when an EXPLICIT `meta_dir` is given (the v4 D-E
    # feature, the TE palette's case: rotated npy under `_rotated_rho3\\`,
    # its atom2d_opt.json in a separate eval_only result folder) -- a
    # caller can swap two `atoms_meta` entries relative to `atoms` (the
    # npy list) and the driver would load atom i's npy with atom j's json
    # silently: exit 0, a wrong but plausible-looking `A_deg`/gates, no
    # error at all. Every atom2d_opt json carries `rho3_npy` (the npy it
    # was itself produced against); confirmed on the live te_palette.json
    # (all 7 levels) that this matches the actual rotated npy path when
    # the pairing is correct. Scoped to `meta_dir` truthy ONLY: v3's own
    # default path (meta_dir=None/"", json and npy co-located) leaves
    # `rho3_npy` as the atom's ORIGINAL production path, which is legitimately
    # different from wherever that npy has since been copied/staged to
    # (confirmed against a real staged TM level, e.g.
    # 260926_scdc_lpastep1\_palette\level_077p5\) -- checking it there
    # would false-positive on every ordinary staged TM job and break the
    # TM bit-identity requirement.
    x = np.load(npy_path).reshape(-1)
    if meta_dir and "rho3_npy" in meta:
        _npy_abs = os.path.normcase(os.path.abspath(npy_path))
        _meta_npy_path = meta["rho3_npy"]
        _meta_npy = os.path.normcase(os.path.abspath(_meta_npy_path))
        if _npy_abs != _meta_npy:
            # [260927 FIX r2, defect 5] A path mismatch alone used to be a
            # hard reject -- but the declared `rho3_npy` can legitimately
            # be a staged/moved COPY of the same array (a different path,
            # identical contents), which this rejected as a false-positive
            # pairing mismatch. Fall back to a CONTENT comparison: accept
            # if the declared file exists and its array is bit-identical
            # to the one actually being loaded; only reject if the arrays
            # genuinely differ (a real swapped-entry mismatch) or the
            # declared file is missing entirely.
            if not os.path.isfile(_meta_npy_path):
                raise ValueError(
                    "atoms_meta pairing mismatch: %s's atom2d_opt.json "
                    "(meta_dir=%s) declares rho3_npy=%s, which does not "
                    "match the npy actually being loaded (%s), and the "
                    "declared file does not exist on disk either"
                    % (jf, meta_d, meta["rho3_npy"], npy_path))
            _decl = np.load(_meta_npy_path).reshape(-1)
            if not np.array_equal(_decl, x):
                raise ValueError(
                    "atoms_meta pairing mismatch: %s's atom2d_opt.json "
                    "(meta_dir=%s) declares rho3_npy=%s, whose CONTENTS "
                    "differ from the npy actually being loaded (%s) -- "
                    "paths differ but this is not a relocated copy"
                    % (jf, meta_d, meta["rho3_npy"], npy_path))
    nxy = int(meta["NXY"])
    l3_layers = 2 if x.size == 2 * nxy * nxy else 1
    if x.size != l3_layers * nxy * nxy:
        raise ValueError("rho3 size %d does not match NXY %d (%d layers)"
                         % (x.size, nxy, l3_layers))
    meta.setdefault("l3_layers", l3_layers)
    # [260909 in ATOM] the eval_only json write never includes `l3_split`
    # (only the optimisation-completion path does, :2159) -- fall back to
    # the job file sitting next to this result (job_*.txt, plain
    # key=value text) so every atom's meta has it regardless of which
    # write path produced its json.
    if "l3_split" not in meta:
        jobtxt = None
        for fn in os.listdir(meta_d):
            if fn.startswith("job_") and fn.endswith(".txt"):
                jobtxt = os.path.join(meta_d, fn)
                break
        val = 0.5
        if jobtxt:
            for ln in io.open(jobtxt, encoding="utf-8",
                              errors="replace").read().split("\n"):
                if "=" not in ln or ln.lstrip().startswith("#"):
                    continue
                k, _, v = ln.partition("=")
                if k.strip() == "l3_split":
                    val = float(v.strip())
                    break
        meta["l3_split"] = val
    # [v3 caveat fix] same fallback pattern as l3_split just above, for
    # process parameters the eval_only json write never included AT ALL
    # (n_spacer/n_sub/lams_nm/material3 -- confirmed absent from every
    # real atom2d_opt.json checked; pol/l3_fill_n/n_l3_gap normally ARE
    # already in the json but fall back the same way in case an atom came
    # from an older code path that omitted them). Read from the atom's own
    # companion job_*.txt (plain key=value text, same file the l3_split
    # fallback above already locates). A key absent from BOTH the json and
    # the job txt is left OUT of `meta` entirely -- the extended F7 check
    # in `_cross_check_atoms`/`_consistent` below turns that into a clean
    # ValueError -> exit 2, never a KeyError crash.
    _f7_extra_keys = ("n_spacer", "n_sub", "lams_nm", "material3", "pol",
                      "l3_fill_n", "n_l3_gap")
    if any(_k not in meta for _k in _f7_extra_keys):
        _jobtxt2 = None
        for fn in os.listdir(meta_d):
            if fn.startswith("job_") and fn.endswith(".txt"):
                _jobtxt2 = os.path.join(meta_d, fn)
                break
        _jkv = {}
        if _jobtxt2:
            for ln in io.open(_jobtxt2, encoding="utf-8",
                              errors="replace").read().split("\n"):
                if "=" not in ln or ln.lstrip().startswith("#"):
                    continue
                _k, _, _v = ln.partition("=")
                _jkv[_k.strip()] = _v.strip()
        for _k in _f7_extra_keys:
            if _k in meta or _k not in _jkv:
                continue
            if _k in ("material3", "pol"):
                meta[_k] = _jkv[_k]
            else:
                try:
                    meta[_k] = float(_jkv[_k])
                except ValueError:
                    pass
    # per-atom filter kernel / edge mask, reconstructed EXACTLY as the atom
    # driver built them for THIS atom (json already stores the GATE-RAISED
    # filt_radius_nm, see Main_rcwa_l3_atom2d_opt.py:2046) -- :1670-1683,
    # :1912-1920.
    cell_nm = G.PERIOD * 1e3 / 3.0
    px_nm = cell_nm / nxy
    rad_px = max(1.0, float(meta["filt_radius_nm"]) / px_nm)
    KF = ATOM.conic_kernel_fft(nxy, rad_px)
    # [261002 consumers] The edge mask is rebuilt from THIS atom's own record
    # (`ATOM.edge_mask_from_record`): edge_margin_x_px / edge_margin_y_px /
    # edge_margin_mode from its atom2d_opt.json, edge_margin_px for a json
    # written before those keys existed (both axes, 'pre' -- exactly the mask
    # this file built before), and the companion job_*.txt only when the json
    # carries no edge-margin key at all. The old code rebuilt every atom from
    # edge_margin_px alone, both axes, 'pre', so a per-axis or 'post' atom was
    # evaluated as a different structure than the one that was optimised.
    _espec = ATOM.edge_mask_spec(meta)
    if _espec[3] == "none":
        _espec = ATOM.edge_mask_spec(meta, _companion_job_kv(meta_d))
    emask = ATOM.build_edge_mask(nxy, _espec[0], _espec[1], _espec[2])
    if emask is not None and _espec[3] != "json_legacy":
        print("[SR] edge mask %s: x=%d y=%d px, mode %s (from %s)"
              % (os.path.basename(d), _espec[0], _espec[1], _espec[2],
                 _espec[3]), flush=True)
    symy = bool(int(meta.get("sym_y", 1)))
    # eval_only path stores "beta_eval"; the optimisation-completion path
    # (Main_rcwa_l3_atom2d_opt.py:2124,2142) always reports at the fixed
    # final beta 64.0 with no key of its own -- default to that.
    beta = float(meta.get("beta_eval", 64.0))
    ATOM.L3_LAYERS = l3_layers
    ATOM.L3_ETCH_STACK = int(meta.get("l3_etch_stack", 0) or 0)
    r3 = ATOM._project3(x, nxy, symy, emask, KF, beta)
    r3 = [np.asarray(z) for z in r3] if isinstance(r3, (list, tuple)) \
        else np.asarray(r3)
    return r3, meta


def _consistent(atoms_meta, key, label, tol=1e-9):
    # [v3 caveat fix] a KEY missing from one or more atoms' meta (json +
    # job-txt fallback, see `_load_atom`) used to raise an uncaught
    # KeyError here -- a hard crash (implicit exit 1, no message) instead
    # of the clean "FATAL (F7)" / exit 2 every other cross-atom mismatch
    # gets. Convert it to the same ValueError the caller (main()'s F7
    # try/except) already handles.
    try:
        vals = [m[key] for m in atoms_meta]
    except KeyError:
        raise ValueError("%s missing from one or more atoms' json/job "
                         "records" % label)
    v0 = vals[0]
    for v in vals[1:]:
        if isinstance(v0, (int, float)) and isinstance(v, (int, float)):
            if abs(v - v0) > tol * max(1.0, abs(v0)):
                raise ValueError("%s differs across atoms: %r vs %r"
                                 % (label, v0, v))
        elif v != v0:
            raise ValueError("%s differs across atoms: %r vs %r"
                             % (label, v0, v))
    return v0


# ------------------------------------------------------------ carrier
def _resample_col(col, n_out):
    """Cumulative-sum resample -- copy of Main_rcwa_l3_atom2d_opt.py:1633-1640
    (`_resample`, module-local closure there, not importable as-is)."""
    n0 = col.size
    c = np.concatenate([[0.0], np.cumsum(col)])
    e = np.arange(n_out + 1) * (n0 / float(n_out))
    lo = np.clip(np.floor(e).astype(int), 0, n0)
    fr = e - np.floor(e)
    ci = c[lo] + fr * col[np.clip(lo, 0, n0 - 1)]
    return (ci[1:] - ci[:-1]) / (n0 / float(n_out))


# ------------------------------------------------------------ supercell object
def _obj_super(nG, atoms_r3, rho12_1d, H, sp, n_sp, eps3, lam, nxy, theta,
               pol, ramp_axis, M, l3_split, l3_gap_nm, l3_fill_n, n_l3_gap,
               sc_shift=0):
    """Generalises `_obj` (Main_rcwa_l3_atom2d_opt.py:308-389) to M distinct
    atoms. Returns (ob, Lx, Ly, K) where K is the carrier-order multiplier
    for x-ramp (physical carrier order m -> supercell g0 = m*K) and 1 for
    y-ramp (carrier order unaffected; the ramp instead lives on g1).

    `sc_shift` (M8, x-ramp only): cyclically rolls the ONE-PERIOD carrier
    column by `sc_shift` atom-pitches (`sc_shift * nxy` samples) before
    tiling it K times. `rho12_1d` is exactly one carrier period (length
    3*nxy, matching `_obj`'s own nx=app*nxy convention at app=3), so this
    is a legitimate cyclic re-registration of the fixed atom sequence
    against the carrier's own periodic pattern, not an approximation --
    the carrier is periodic in Lambda=3*a_pitch by construction, and
    Lambda divides Lx=M*a_pitch exactly (Lx/Lambda=K). Left at 0 (no-op)
    for y-ramp, where there is no such registration freedom (Lx stays
    G.PERIOD, unaffected by the ramp)."""
    a_pitch = G.PERIOD / 3.0
    nlayer = len(atoms_r3[0]) if isinstance(atoms_r3[0], list) else 1
    if ramp_axis == "x":
        if M % 3 != 0:
            raise ValueError("x-ramp needs M multiple of 3 (got %d)" % M)
        K = M // 3
        Lx, Ly = M * a_pitch, a_pitch
        nx, ny = M * nxy, nxy
        _shift_n = int(sc_shift) * nxy
        if _shift_n:
            rho12_grid = [np.tile(np.roll(np.asarray(col), _shift_n), K)
                          for col in rho12_1d]
        else:
            rho12_grid = [np.tile(col, K) for col in rho12_1d]
        if nlayer == 2:
            combined = [np.concatenate([atoms_r3[j][k] for j in range(M)],
                                        axis=0) for k in range(2)]
        else:
            combined = [np.concatenate(list(atoms_r3), axis=0)]
    elif ramp_axis == "y":
        K = 1
        Lx, Ly = G.PERIOD, M * a_pitch
        nx, ny = 3 * nxy, M * nxy
        rho12_grid = [np.asarray(c) for c in rho12_1d]
        if nlayer == 2:
            combined = [np.concatenate(
                [np.concatenate([atoms_r3[j][k]] * 3, axis=0)
                 for j in range(M)], axis=1) for k in range(2)]
        else:
            combined = [np.concatenate(
                [np.concatenate([atoms_r3[j]] * 3, axis=0)
                 for j in range(M)], axis=1)]
    else:
        raise ValueError("ramp_axis must be x or y (got %r)" % ramp_axis)

    ob = grcwa.obj(int(nG), [Lx, 0.0], [0.0, Ly], 1.0 / lam, theta, 0.0,
                   verbose=0)
    if _LI_ON:
        # [261008 li_integrate] opt-in only
        ob.rcwa_rule = _RCWA_RULE
        ob.form_factor = _RCWA_FF
    ob.Add_LayerUniform(0.5, G.EPS_SUB)
    ob.Add_LayerGrid(H[0], nx, ny)
    ob.Add_LayerGrid(H[1], nx, ny)
    if sp > 0:
        ob.Add_LayerUniform(sp, complex(n_sp) ** 2)
    _gap_um = float(l3_gap_nm) * 1e-3
    _has_gap = (len(combined) == 2 and _gap_um > 0.0)
    if len(combined) == 2:
        _f = float(l3_split)
        _h3 = [H[2] * _f, H[2] * (1.0 - _f)]
    else:
        _h3 = [H[2]]
    for _k3, _hh in enumerate(_h3):
        if _k3 == 1 and _has_gap:
            ob.Add_LayerGrid(_gap_um, nx, ny)
        ob.Add_LayerGrid(_hh, nx, ny)
    ob.Add_LayerUniform(0.5, G.EPS_AIR)
    ob.Init_Setup()
    es = []
    for i in (0, 1):
        _e12 = rho12_grid[i]
        _e12_2d = _e12[:, None] * np.ones((1, ny))
        es.append(1.444 ** 2 + _e12_2d * (3.48 ** 2 - 1.444 ** 2))
    _fill_eps = float(l3_fill_n) ** 2
    for _k, _rr in enumerate(combined):
        _eb = _fill_eps if (_k == 0 and len(combined) > 1) else 1.0
        if _k == 1 and _has_gap:
            _gape = complex(n_l3_gap) ** 2
            es.append(_fill_eps + combined[0] * (_gape - _fill_eps))
        es.append(_eb + _rr * (eps3 - _eb))
    ob.GridLayer_geteps(
        np.concatenate([np.reshape(np.asarray(e), (-1,)) for e in es]))
    if pol == "TE":
        ob.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
    else:
        ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    return ob, Lx, Ly, K


def _full_order_scan(ob, pol):
    """Generalises `_ledger` (:430-447) to EVERY (g0, g1), both sides,
    power AND complex amplitude. `R`/`T` from `ob.RT_Solve(byorder=1)`
    already sum BOTH polarisation halves per order (grcwa's own
    `GetZPoyntingFlux`: `forward = forward_xy[:n] + forward_xy[n:]`,
    _vendor/grcwa/rcwa.py:918) -- so `R_power`/`T_power` below are the
    correct total order power (D1), never |amplitude|^2.

    Returns a dict with a JSON-safe `orders`/`R_total`/`T_total` (for
    embedding directly in the result) PLUS the raw solved arrays under
    underscore keys, for exact by-order lookup via `_scan_order` below --
    those underscore keys must never be written to the output JSON."""
    R, T = ob.RT_Solve(normalize=1, byorder=1)
    # `GetAmplitudes` returns (forward, backward) at the requested layer
    # (rcwa.py:577-603). At layer 0 (substrate/incidence side) the FORWARD
    # wave is the fixed INCIDENT field -- the REFLECTED field is the
    # BACKWARD one, `b_r`. At the last layer (air/output side) it is the
    # other way around: the OUTGOING/transmitted field is the FORWARD
    # one, `a_t` (there is nothing incident from beyond air, so the
    # backward component there is always ~0). `_amp` (:392-408) and
    # `_amp` follow exactly this convention; matched here.
    _a_inc, b_r = ob.GetAmplitudes(0, 0.0)
    a_t, _b_zero = ob.GetAmplitudes(ob.Layer_N - 1, 0.0)
    off = 0 if pol == "TE" else ob.nG
    out = {}
    for i, g in enumerate(ob.G):
        g0, g1 = int(g[0]), int(g[1])
        rp = float(np.real(R[i]))
        tp = float(np.real(T[i]))
        if rp > 1e-9 or tp > 1e-9:
            ra = complex(b_r[i + off])
            ta = complex(a_t[i + off])
            out["%+d,%+d" % (g0, g1)] = dict(
                g0=g0, g1=g1, R_power=rp, T_power=tp,
                R_amp=[ra.real, ra.imag], T_amp=[ta.real, ta.imag])
    return dict(orders=out, R_total=float(np.sum(np.real(R))),
               T_total=float(np.sum(np.real(T))),
               _R=R, _T=T, _a_r=b_r, _v_t=a_t, _off=off, _G=ob.G, _ob=ob)


def _scan_order(scan, g0, g1, want="T"):
    """Exact (power, complex amplitude) for one order from a
    `_full_order_scan` result, regardless of the 1e-9 reporting threshold
    used for `orders` -- needed for named orders (echo, echo+-1, target,
    ghost, zero) that must be read even when genuinely (near-)zero, so
    their absence from `orders` is never mistaken for "not computed"."""
    R = scan["_R"] if want == "R" else scan["_T"]
    amp_arr = scan["_a_r"] if want == "R" else scan["_v_t"]
    off = scan["_off"]
    for i, g in enumerate(scan["_G"]):
        if int(g[0]) == g0 and int(g[1]) == g1:
            return float(np.real(R[i])), complex(amp_arr[i + off])
    return 0.0, 0j


def _propagates(scan, g0, g1, side="air"):
    """True if order (g0,g1) is a propagating (real-q) channel on the
    given side ('air' = last layer / transmission side, 'sub' = layer 0 /
    reflection side). Exact test via grcwa's own per-order eigenvalue
    (q ~ kz for a uniform layer, real if propagating / purely imaginary if
    evanescent in this lossless model) -- independent of how much power
    actually landed there, unlike a power threshold, which cannot tell
    apart 'cannot propagate' from 'propagates but happens to be dark'."""
    ob = scan["_ob"]
    q = ob.q_list[-1] if side == "air" else ob.q_list[0]
    for i, g in enumerate(scan["_G"]):
        if int(g[0]) == g0 and int(g[1]) == g1:
            qi = complex(q[i])
            return abs(qi.imag) < 1e-6 * max(abs(qi.real), 1e-30)
    return False


def _fid_l3_super(ob, pol, sp, design_g0):
    """Axis-symmetric diagnostic, LOCAL to this file -- generalises
    ATOM._fid_l3 (:1031-1065) by summing the denominator over EVERY
    retained (g0,g1) instead of only g1==0, so the reported quantity means
    the SAME thing on the y-ramp axis as on the x-ramp axis (on x-ramp the
    two formulas agree to numerical noise anyway, since Ly=a_pitch there
    carries no real g1 diffraction). Diagnostic only, exactly like the
    original -- never gates anything in this driver."""
    a, _b = ob.GetAmplitudes(3, sp)
    off = ob.nG if pol == "TM" else 0
    num = den = 0.0
    for i, g in enumerate(ob.G):
        p = abs(complex(a[i + off])) ** 2
        den += p
        if int(g[0]) == design_g0 and int(g[1]) == 0:
            num += p
    return num / (den + 1e-30)


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------ LPA reference (F5)
def _lpa_ref_from_solve(atoms_r3, atoms_meta, carrier_1period, H, sp, n_sp,
                        eps3, lam, nxy, pol, nG_ref):
    """F5 ref_mode=solve: per-atom (tA, tB, rB) from single-atom solves on
    the SAME carrier/process the supercell run uses (`ATOM._obj`, the
    ordinary app=3 small cell) -- this is the actual LPA premise being
    tested, and unlike a stored atom2d_opt.json it cannot be scored on a
    different carrier than the supercell run (defect 6: 4/8 palette atoms
    in the old tm_palette.json were scored on a different carrier).

    D-B fix: tB and rB are stored SEPARATELY (not pre-multiplied into
    `rBtB`) -- arm B's prediction is the DFT of tB alone, compared against
    the measured supercell tB; rB(0) is compared separately (predicted
    plain mean of rB vs measured rB(0), never diffracted by the ramp -- see
    the `rB0` entry `_measure_super` adds to its `lpa` block below). The
    old `rBtB` product gave an identical-atom-control ratio of 1/|rB|^2
    instead of 1, because the measured side was tB alone while the
    predicted side carried an extra factor of rB.

    [v4, D-C] `obp`'s incidence angle is now pol-dependent (0.0 for TE,
    +THETA_SUB for TM), matching `_measure_arms`'s own TE branch
    (:1265-1272 -- arm A illuminates at NORMAL incidence for TE, not
    +THETA_SUB) and `_measure_super`'s already-generic `thA` selection.
    v3 always solved `obp` at +th regardless of pol, which is silently
    wrong for TE (arm A's design order 0 is a normal-incidence quantity,
    not the +th quantity that order happens to also be nonzero at) --
    masked in v3 by the pol=TE hard error, so this path was never
    actually exercised there. `rB`'s order is `_rb_small_order(pol)`
    (also read off `obp`, same solve as `tA` -- both lanes)."""
    out = []
    thA = 0.0 if pol == "TE" else G.THETA_SUB
    thB = -G.THETA_SUB
    rb_order = _rb_small_order(pol)
    for r3, meta in zip(atoms_r3, atoms_meta):
        ATOM.ATOMS_PER_PERIOD = 3
        ATOM.L3_LAYERS = 2 if isinstance(r3, list) else 1
        ATOM.L3_SPLIT = float(meta.get("l3_split", 0.5))
        ATOM.L3_GAP_NM = float(meta.get("l3_gap_nm", 0.0))
        ATOM.L3_FILL_N = float(meta.get("l3_fill_n", 1.0))
        ATOM.N_L3_GAP = float(meta.get("n_l3_gap", 1.4440))
        ATOM.L3_ETCH_STACK = int(meta.get("l3_etch_stack", 0) or 0)
        obp = ATOM._obj(nG_ref, r3, carrier_1period, H, sp, n_sp, eps3, lam,
                        nxy, thA, pol)
        obm = ATOM._obj(nG_ref, r3, carrier_1period, H, sp, n_sp, eps3, lam,
                        nxy, thB, pol)
        gA0, gB0 = (0, +1) if pol == "TE" else (-1, +1)
        tA = complex(ATOM._amp(obp, gA0, pol=pol))
        rB = complex(ATOM._amp(obp, rb_order, refl=True, pol=pol))
        tB = complex(ATOM._amp(obm, gB0, pol=pol))
        out.append(dict(tA=tA, tB=tB, rB=rB, nG_actual=int(obp.nG)))
    return out


def _lpa_ref_from_jsons(ref_json_paths):
    """F5 ref_mode=jsons: per-atom (tA, tB, rB) straight from stored
    atom2d_opt.json amp_tA/A_deg, amp_tB/argtB_deg and amp_rB/argrB_deg
    (D-B: read tB and rB SEPARATELY -- both keys are always present, see
    `_measure_arms`'s own return dict, Main_rcwa_l3_atom2d_opt.py:1394-1399
    -- never derived from the combined amp_rBtB/B_deg, which loses the
    tB-vs-rB split D-B needs). Whatever carrier those jsons were evaluated
    on is on the CALLER to have matched; this driver does not (cannot)
    verify that from the json alone."""
    out = []
    for jf in ref_json_paths:
        m = json.load(io.open(jf, encoding="utf-8"))
        a_deg = math.radians(float(m["A_deg"]))
        tA = float(m["amp_tA"]) * complex(math.cos(a_deg), math.sin(a_deg))
        rb_deg = math.radians(float(m["argrB_deg"]))
        rB = float(m["amp_rB"]) * complex(math.cos(rb_deg), math.sin(rb_deg))
        tb_deg = math.radians(float(m["argtB_deg"]))
        tB = float(m["amp_tB"]) * complex(math.cos(tb_deg), math.sin(tb_deg))
        out.append(dict(tA=tA, tB=tB, rB=rB, source=jf))
    return out


def _lpa_predict_v2(refs, M, s, arm):
    """a_pred(s) = (1/M) sum_j x_j exp(-i 2 pi s j / M), x_j = tA_j for arm
    A, tB_j for arm B (D-B: tB alone, not rB*tB -- both already complex,
    from either F5 ref builder above)."""
    key = "tA" if arm == "A" else "tB"
    acc = 0j
    for j, r in enumerate(refs):
        acc += r[key] * complex(math.cos(-2.0 * math.pi * s * j / M),
                                math.sin(-2.0 * math.pi * s * j / M))
    a_pred = acc / M
    return a_pred, abs(a_pred) ** 2


def _rb_small_order(pol):
    """[v4, D-C/D-E] rB's small-cell physical order (y=0 always): 0 for TM
    (`_measure_arms`'s TM branch, :1278, `rB = _amp(obp, 0, refl=True)`),
    +1 for TE (:1269, `rB = _amp(ob0, +1, refl=True, pol="TE")` -- off the
    SAME normal-incidence solve that gives tA, not a separate +th one).
    v3 hardcoded 0 everywhere rB's order was read/reported, which is the
    TM value -- correct there, silently wrong for TE (the exact defect
    v3's D-C hard error flagged instead of fixing)."""
    return 1 if pol == "TE" else 0


# ------------------------------------------------------------ core measurement
def _measure_super(nG, atoms_r3, rho12_1d, H, sp, n_sp, eps3, lam, nxy,
                   pol, ramp_axis, M, l3_split, l3_gap_nm, l3_fill_n,
                   n_l3_gap, ramp_s, sc_shift=0, lpa_refs=None):
    """One full arm-A/arm-B measurement on the supercell, mirroring
    `_measure_arms` (:1253-1412) but on `_obj_super`, with every order
    index fixed per F4 and every propagating-order power read via
    `_full_order_scan`/`_scan_order` (D1). If `lpa_refs` is given (a list
    of M per-atom dicts from `_lpa_ref_from_solve`/`_lpa_ref_from_jsons`),
    also fills in the F5 LPA-vs-measured comparison at s=-1,0,+1."""
    t0 = time.time()
    if pol == "TE":
        thA, thB = 0.0, -G.THETA_SUB
        gA0, gB0 = 0, +1     # small-cell design orders, TE lane (`fom`
                             # :614-621; `_measure_arms` :1265-1272)
                             # [260927 FIX, defect 7] was ":589-594", stale
    else:
        thA, thB = +G.THETA_SUB, -G.THETA_SUB
        gA0, gB0 = -1, +1    # small-cell design orders, TM lane (:591-593)
    obA, Lx, Ly, K = _obj_super(nG, atoms_r3, rho12_1d, H, sp, n_sp, eps3,
                                lam, nxy, thA, pol, ramp_axis, M,
                                l3_split, l3_gap_nm, l3_fill_n, n_l3_gap,
                                sc_shift)
    obB, _, _, _ = _obj_super(nG, atoms_r3, rho12_1d, H, sp, n_sp, eps3,
                              lam, nxy, thB, pol, ramp_axis, M,
                              l3_split, l3_gap_nm, l3_fill_n, n_l3_gap,
                              sc_shift)
    # NOTE (v3 caveat): `elapsed` times ONLY these two `_obj_super` builds
    # (obA/obB) -- NOT the order scans below, the LPA reference solves
    # (`_lpa_ref_from_solve`, which run BEFORE this function is even
    # called), or file I/O. Measured 260926: 8.98 s here against 54.4 s of
    # job-level `wall_s` (added in `main()`, wraps the WHOLE per-job
    # measurement including the reference solves) for the same run -- use
    # `wall_s` for real wall-clock timing; `elapsed_s` stays for its
    # original, narrower purpose (isolating the two solves' own cost).
    elapsed = time.time() - t0

    def mk(base, s):
        return (base + s, 0) if ramp_axis == "x" else (base, s)

    base_gA, base_gB = (gA0 * K, gB0 * K) if ramp_axis == "x" else (gA0, gB0)
    target_gA, ghost_gA, zero_gA = mk(base_gA, ramp_s), mk(base_gA, -ramp_s), mk(base_gA, 0)
    target_gB, ghost_gB, zero_gB = mk(base_gB, ramp_s), mk(base_gB, -ramp_s), mk(base_gB, 0)

    scanA = _full_order_scan(obA, pol)
    scanB = _full_order_scan(obB, pol)

    # F4: rB is the SMALL-CELL order `_rb_small_order(pol)` @ obA (TM: 0
    # @ +th; TE, v4: +1 @ 0 -- see that function). Scaled to the supercell
    # frame the same way every other named order is (`mk` at s=0 -- rB is
    # NOT diffracted by the ramp, so it always sits at the zero-shift slot)
    # -- matches Main_rcwa_l3_atom2d_opt.py:1278 (TM) / :1269 (TE) exactly.
    # Never ramp-order-dependent (defect F4: the old file read r(+K) here
    # via `base_gB*K`; v3 then hardcoded the TM order 0 for both lanes,
    # the exact defect its own D-C note flagged instead of fixing).
    rb_smallcell = _rb_small_order(pol)
    rb_base = rb_smallcell * K if ramp_axis == "x" else rb_smallcell
    zero_rB = mk(rb_base, 0)
    rB0_power, rB0_amp = _scan_order(scanA, zero_rB[0], zero_rB[1], "R")

    tA_power, tA_amp = _scan_order(scanA, target_gA[0], target_gA[1], "T")
    tA_ghost_power, tA_ghost_amp = _scan_order(scanA, ghost_gA[0], ghost_gA[1], "T")
    tA_zero_power, tA_zero_amp = _scan_order(scanA, zero_gA[0], zero_gA[1], "T")

    tB_power, tB_amp = _scan_order(scanB, target_gB[0], target_gB[1], "T")
    tB_ghost_power, tB_ghost_amp = _scan_order(scanB, ghost_gB[0], ghost_gB[1], "T")
    tB_zero_power, tB_zero_amp = _scan_order(scanB, zero_gB[0], zero_gB[1], "T")

    # F4: echo = R(+2*K) @ -th (obB), carrier-frame order ALWAYS +2 (a
    # property of the retro grating's re-entry, not of ramp_s) -- matches
    # :1298-1313 `echo_R_p2 = _ord_power(obm, _Rm, +2)`. The old file read
    # R((gB0+2)*K) = R(3*K), one order off (defect 5). Reported as RT
    # POWER (correct unit) with amplitude-squared alongside, separately
    # labelled (D1) -- plus the two side orders 2K-1 / 2K+1 (F4 "plus").
    echo_gB = (2 * K, 0)
    echo_power, echo_amp = _scan_order(scanB, echo_gB[0], echo_gB[1], "R")
    echo_side_m = ((2 * K - 1, 0), (2 * K + 1, 0))
    echo_side = []
    for _o in echo_side_m:
        _rp, _ra = _scan_order(scanB, _o[0], _o[1], "R")
        _tp, _ta = _scan_order(scanB, _o[0], _o[1], "T")
        echo_side.append(dict(order=list(_o), R_power=_rp, T_power=_tp,
                              R_amp=[_ra.real, _ra.imag],
                              T_amp=[_ta.real, _ta.imag]))

    fid_l3 = _fid_l3_super(obA, pol, sp, base_gA)
    fid_l3_m = _fid_l3_super(obB, pol, sp, base_gB)
    r00v = complex(ATOM._r00_l3(obA, pol, sp, base_gA))

    prop_tA_target = _propagates(scanA, *target_gA, side="air")
    prop_tA_ghost = _propagates(scanA, *ghost_gA, side="air")
    prop_tB_target = _propagates(scanB, *target_gB, side="air")
    prop_tB_ghost = _propagates(scanB, *ghost_gB, side="air")

    # M8: IDEALISED reference translation phase for sc_shift, per named
    # order. Derivation: treating the sc_shift registration change as a
    # RIGID translation of the WHOLE M*a_pitch-periodic assembly by one
    # atom-pitch per shift multiplies supercell order m's amplitude by
    # exp(-i*2*pi*m*sc_shift/M) (standard Bloch shift theorem, already in
    # degrees below -- NOT run through `math.degrees` again).
    #
    # CAVEAT, measured directly (M=6 x-ramp, 6 DISTINCT palette atoms,
    # sc_shift 0 vs 1): the rigid-translation picture is only EXACT when
    # the atom SEQUENCE is itself periodic at the shift distance (true for
    # the identical-atom N1x/N1y needles, verified there to reproduce this
    # formula to numerical precision) -- what `sc_shift` actually rolls is
    # the CARRIER under a FIXED atom sequence, and for genuinely distinct
    # neighbouring atoms that is NOT equivalent to shifting the whole
    # assembly (the atom pattern does not repeat at one atom-pitch, so a
    # rigid shift of "everything" would move the atoms too, which
    # `sc_shift` deliberately does not do). Measured mismatch: this
    # formula predicts a pure +-180deg sign flip between sc_shift=0 and
    # sc_shift=1 for M=6/K=2 at ramp_s=-1, target order -3; the actual
    # measured amp_tA_target ratio has |ratio|=1.011 (not 1) and phase
    # 142.7deg (not 180). So this field is a LEADING-ORDER REFERENCE VALUE
    # for a first phase-alignment guess, not an exact correction -- the
    # reported complex `amp_*_target/ghost/zero` fields ARE the ground
    # truth, and any coherent average over shifts/grating-offsets should
    # be built from those directly, using this phase only as a sanity
    # check or a starting point.
    def _tphase_deg(m):
        if ramp_axis != "x" or M == 0 or sc_shift == 0:
            return 0.0
        return (-360.0 * m * sc_shift / M) % 360.0

    info = dict(
        elapsed_s=elapsed, K=K, M=M, ramp_axis=ramp_axis, ramp_s=ramp_s,
        sc_shift=sc_shift,
        base_gA=base_gA, base_gB=base_gB,
        rb_smallcell_order=rb_smallcell, rb_base=rb_base,
        zero_order_rB=list(zero_rB),
        target_order_A=list(target_gA), target_order_B=list(target_gB),
        zero_order_A=list(zero_gA), zero_order_B=list(zero_gB),
        ghost_order_A=list(ghost_gA), ghost_order_B=list(ghost_gB),
        amp_tA_target=[tA_amp.real, tA_amp.imag], power_tA_target=tA_power,
        propagates_tA_target=prop_tA_target,
        amp_tA_zero=[tA_zero_amp.real, tA_zero_amp.imag],
        power_tA_zero=tA_zero_power,
        amp_tA_ghost=[tA_ghost_amp.real, tA_ghost_amp.imag],
        power_tA_ghost=tA_ghost_power, propagates_tA_ghost=prop_tA_ghost,
        amp_rB_zero=[rB0_amp.real, rB0_amp.imag], power_rB_zero=rB0_power,
        amp_tB_target=[tB_amp.real, tB_amp.imag], power_tB_target=tB_power,
        propagates_tB_target=prop_tB_target,
        amp_tB_zero=[tB_zero_amp.real, tB_zero_amp.imag],
        power_tB_zero=tB_zero_power,
        amp_tB_ghost=[tB_ghost_amp.real, tB_ghost_amp.imag],
        power_tB_ghost=tB_ghost_power, propagates_tB_ghost=prop_tB_ghost,
        echo_order=list(echo_gB), echo_R_power=echo_power,
        echo_amp2=abs(echo_amp) ** 2,
        echo_p2_db=10.0 * math.log10(max(echo_power, 1e-12)),
        echo_amp2_db=10.0 * math.log10(max(abs(echo_amp) ** 2, 1e-12)),
        echo_side_orders=echo_side,
        translation_phase_deg=dict(
            tA_target=_tphase_deg(target_gA[0]),
            tA_ghost=_tphase_deg(ghost_gA[0]),
            tB_target=_tphase_deg(target_gB[0]),
            tB_ghost=_tphase_deg(ghost_gB[0]),
            rB_zero=_tphase_deg(rb_base), echo=_tphase_deg(echo_gB[0]),
            note=("IDEALISED rigid-translation reference (Bloch shift "
                 "theorem), EXACT only if the atom sequence repeats at "
                 "the shift distance (e.g. the identical-atom N1 "
                 "needles); for distinct-atom ramps it is a leading-order "
                 "phase-alignment guess, not an exact correction -- "
                 "measured mismatch |ratio|=1.011/phase 142.7deg vs this "
                 "formula's 1.0/180deg for one M=6 sc_shift 0-vs-1 case. "
                 "Build any coherent shift-average from the reported "
                 "complex amp_*_target/ghost/zero fields directly.")),
        fid_l3=fid_l3, fid_l3_m=fid_l3_m,
        absR00=abs(r00v), argR00_deg=math.degrees(math.atan2(
            r00v.imag, r00v.real)) % 360.0,
        R_total_A=scanA["R_total"], T_total_A=scanA["T_total"],
        energy_total_A=scanA["R_total"] + scanA["T_total"],
        R_total_B=scanB["R_total"], T_total_B=scanB["T_total"],
        energy_total_B=scanB["R_total"] + scanB["T_total"],
        ordersA=scanA["orders"], ordersB=scanB["orders"],
        Lx=Lx, Ly=Ly, nG_actual=int(obA.nG),
        arm_note=("target/zero/ghost orders are for the TRANSMITTED "
                  "design channel (arm A tA, arm B tB); rB is reported "
                  "only at its small-cell order (`rb_smallcell_order` --  "
                  "0 for TM, +1 for TE, v4 D-C), scaled to this supercell "
                  "the same way every other named order is, matching the "
                  "atom driver's own convention -- it is NOT ramp-order "
                  "dependent (F4)."),
    )

    if lpa_refs is not None:
        lpa = {}
        # D-A fix: key the measured (power, amp, propagates) triple by the
        # ORDER relative to `ramp_s`, not by the literal loop `s`. `mk`
        # places the target at `base + ramp_s`, the ghost at
        # `base - ramp_s`, the zero at `base + 0` -- so the measured value
        # that belongs to loop-index `s` is whichever of
        # target/ghost/zero sits at `base + s`, and that mapping SWAPS
        # target<->ghost when `ramp_s == -1` (the default, primary mode).
        # The old code hardcoded {-1: ghost, 0: zero, 1: target}
        # regardless of `ramp_s`, which is only correct for `ramp_s=+1`;
        # at the default `ramp_s=-1` it read the ghost's power under the
        # "target" label and vice versa (measured: ratio 0.908/11.04
        # instead of the correct values, both arms swapped the same way).
        mapA = {ramp_s: (tA_power, tA_amp, prop_tA_target),
                -ramp_s: (tA_ghost_power, tA_ghost_amp, prop_tA_ghost),
                0: (tA_zero_power, tA_zero_amp, True)}
        mapB = {ramp_s: (tB_power, tB_amp, prop_tB_target),
                -ramp_s: (tB_ghost_power, tB_ghost_amp, prop_tB_ghost),
                0: (tB_zero_power, tB_zero_amp, True)}
        for s in (-1, 0, +1):
            for arm, base, mm in (("A", base_gA, mapA), ("B", base_gB, mapB)):
                power, amp, prop = mm[s]
                amp2 = abs(amp) ** 2
                order = mk(base, s)
                # D-B: arm B now predicts the DFT of tB alone (see
                # `_lpa_predict_v2`'s "tB" key, fixed above) and compares
                # it against the measured tB directly -- NOT against a
                # predicted rB*tB compared to measured tB alone, which is
                # what the old code did (identical-atom control used to
                # give ratio 1.886271 = 1/|rB|^2 at s=0 instead of 1,
                # because the predicted side carried an extra factor of
                # rB that the measured side never had).
                a_pred, p_pred_amp = _lpa_predict_v2(lpa_refs, M, s, arm)
                kappa = (power / amp2) if amp2 > 1e-30 else None
                p_pred_power = (kappa * p_pred_amp
                               if (kappa is not None and prop) else None)
                ratio_amp = (amp2 / p_pred_amp) if p_pred_amp > 0 else None
                ratio_power = (power / p_pred_power
                              if (p_pred_power is not None
                                  and p_pred_power > 0) else None)
                lpa["arm%s_s%+d" % (arm, s)] = dict(
                    order=list(order), propagates=bool(prop),
                    a_pred_re=a_pred.real, a_pred_im=a_pred.imag,
                    P_pred_amp_basis=p_pred_amp,
                    P_pred_power_basis=p_pred_power,
                    P_meas_amp2=amp2, P_meas_power=power,
                    kappa_power_over_amp2=kappa,
                    ratio_meas_over_pred_amp_basis=ratio_amp,
                    ratio_meas_over_pred_power_basis=ratio_power,
                    note=(None if prop else
                         "non-propagating in air (exact q-eigenvalue "
                         "test) -- power-basis ratio withheld; only the "
                         "zero-order (s=0) comparison should be trusted "
                         "for this cell (F5)."))
        # D-B (continued): rB(0) is NOT diffracted by the ramp -- it sits
        # at the fixed small-cell order (0,0) on obA regardless of which
        # atom occupies which supercell position, so its LPA prediction is
        # the plain (non-DFT) mean of the per-atom rB references, compared
        # against the measured supercell rB(0,0) (`rB0_power`/`rB0_amp`,
        # already read above via F4's fixed convention).
        rB_pred = sum(r["rB"] for r in lpa_refs) / M
        rB_pred_amp2 = abs(rB_pred) ** 2
        rB_meas_amp2 = abs(rB0_amp) ** 2
        lpa["rB0"] = dict(
            # [260927 FIX, defect 4] was hardcoded [0, 0] (the TM order)
            # regardless of pol/axis; the actually-measured order is
            # `zero_rB` (same value already reported as `zero_order_rB`
            # above) -- TM order-0, but TE's small-cell order is
            # `rb_smallcell` (+1, see `_rb_small_order`), so TE's zero_rB
            # is [2, 0] on x M=6 / [1, 0] on y, not [0, 0].
            order=list(zero_rB), a_pred_re=rB_pred.real,
            a_pred_im=rB_pred.imag,
            P_pred_amp_basis=rB_pred_amp2, P_meas_amp2=rB_meas_amp2,
            P_meas_power=rB0_power,
            ratio_meas_over_pred_amp_basis=(
                (rB_meas_amp2 / rB_pred_amp2) if rB_pred_amp2 > 0 else None),
            note=("rB is the small-cell order `rb_smallcell` (TM: order-0 "
                 "reflection; TE, v4: order +1 -- see `_rb_small_order`) "
                 "-- not diffracted by the ramp -- so its LPA reference "
                 "is the plain mean of the per-atom rB, not a DFT (D-B)."))
        info["lpa"] = lpa
    return info


# ------------------------------------------------------------ needle-only utils
def _single_sets(nG):
    """G-vector set grcwa selects for the ordinary atom-driver cell
    (app=3) at request nG."""
    Lx, Ly = G.PERIOD, G.PERIOD / 3.0
    Lk1, Lk2 = grcwa.kbloch.Lattice_Reciprocate([Lx, 0.0], [0.0, Ly])
    Gs, _n = grcwa.kbloch.Lattice_getG(int(nG), Lk1, Lk2)
    return set((int(a), int(b)) for a, b in Gs)


def _super_sets(nG, axis, M):
    a = G.PERIOD / 3.0
    Lx, Ly = (M * a, a) if axis == "x" else (G.PERIOD, M * a)
    Lk1, Lk2 = grcwa.kbloch.Lattice_Reciprocate([Lx, 0.0], [0.0, Ly])
    Gs, _n = grcwa.kbloch.Lattice_getG(int(nG), Lk1, Lk2)
    return set((int(p), int(q)) for p, q in Gs)


def _matched_super_nG(nG_single, axis, M, search_limit=4000):
    """Smallest supercell request nG whose retained-harmonic sublattice
    (g0 % K == 0 for x, g1 % M == 0 for y) reproduces the single-cell's
    set EXACTLY. None if not found within `search_limit`. (F6 -- ported
    from the verifier's vcommon.py, which is how the matched-harmonic
    needles that report reproduces below were built.)"""
    ss = _single_sets(nG_single)
    K = M // 3 if axis == "x" else M
    for n in range(nG_single, search_limit):
        st = _super_sets(n, axis, M)
        if axis == "x":
            sub = set((p // K, q) for p, q in st if p % K == 0)
        else:
            sub = set((p, q // K) for p, q in st if q % K == 0)
        if sub == ss:
            return n
        if len(sub) > len(ss):
            return None
    return None


def _matched_single_nG(nG_super, axis, M, search_limit=4000):
    """D-D: the INVERSE of `_matched_super_nG` above -- given a supercell
    REQUEST nG (the one the production/needle run actually solves at),
    return the single-cell nG whose retained G-set is EXACTLY that
    supercell's own retained-harmonic sublattice (same g0 % K == 0 / g1 %
    M == 0 filter). None if not found within `search_limit`."""
    st = _super_sets(nG_super, axis, M)
    K = M // 3 if axis == "x" else M
    if axis == "x":
        sub = set((p // K, q) for p, q in st if p % K == 0)
    else:
        sub = set((p, q // K) for p, q in st if q % K == 0)
    for n in range(1, search_limit):
        if _single_sets(n) == sub:
            return n
    return None


def _nG_ref_check(nG, nG_ref_raw, ramp_axis, M):
    """D-D: harmonic-match `nG_ref` (the F5 single-atom reference solve)
    to `nG` (the supercell request). `nG_ref_raw` is None when the job did
    not set `nG_ref` explicitly -- default it to the INVERSE of
    `_matched_super_nG` (`_matched_single_nG`, above), i.e. the single-cell
    nG whose retained set reproduces THIS supercell's own sublattice
    exactly. When `nG_ref_raw` IS given, use it as requested and instead
    VERIFY the match. Either way returns
    (nG_ref, nG_ref_actual, nG_super_sub_actual, ok); ok=False means the
    caller must hard-error exit 2 -- a mismatched pair measured 15.6%
    error in the 260926 verifier report (defect D-D)."""
    K = M // 3 if ramp_axis == "x" else M
    st = _super_sets(nG, ramp_axis, M)
    if ramp_axis == "x":
        sub = set((p // K, q) for p, q in st if p % K == 0)
    else:
        sub = set((p, q // K) for p, q in st if q % K == 0)
    if nG_ref_raw is None:
        mm = _matched_single_nG(nG, ramp_axis, M)
        if mm is None:
            return None, None, len(sub), False
        nG_ref = mm
    else:
        nG_ref = int(nG_ref_raw)
    single_set = _single_sets(nG_ref)
    return nG_ref, len(single_set), len(sub), (single_set == sub)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ------------------------------------------------------------ needle driver
def _peak_rss_mb():
    """Windows tracks a true peak working set (`peak_wset`) per process --
    prefer it over sampling current RSS at two points, which can miss a
    spike that happened and freed between samples."""
    try:
        import psutil
        mi = psutil.Process().memory_full_info()
        return getattr(mi, "peak_wset", mi.rss) / 1e6
    except Exception:
        return None


def _load_atoms(atom_paths, meta_dirs=None):
    """[v4, D-E] `meta_dirs`, when given, is 1:1 with `atom_paths` (an
    empty string in a slot falls back to that atom's own npy directory --
    v3's only behaviour, and this function's own default when `meta_dirs`
    is None)."""
    atoms_r3, atoms_meta = [], []
    md = meta_dirs if meta_dirs is not None else [None] * len(atom_paths)
    for p, m in zip(atom_paths, md):
        r3, meta = _load_atom(p, meta_dir=(m or None))
        atoms_r3.append(r3)
        atoms_meta.append(meta)
    return atoms_r3, atoms_meta


def _cross_check_atoms(atoms_meta):
    """Cross-atom consistency (raises ValueError on mismatch -- caught by
    the caller and turned into the F7 hard error). Returns a DICT (v3: not
    a positional tuple any more -- extending a fixed-size tuple risks
    silently misaligning a caller's unpack; every caller below was updated
    to use the named keys):
    nxy, l3_split, l3_gap_nm, l3_fill_n, n_l3_gap, h1_nm, h2_nm, h3_nm,
    spacer_nm, n_spacer, n_sub, lams_nm, material3, pol (the last five are
    the [v3 caveat fix] extension -- see the `_load_atom` job-txt fallback
    above, which is what makes them present in `atoms_meta` at all)."""
    for m in atoms_meta:
        m["_H1_nm"], m["_H2_nm"], m["_H3_nm"] = (
            m["H_nm"][0], m["H_nm"][1], m["H_nm"][2])
    nxy = int(_consistent(atoms_meta, "NXY", "NXY"))
    l3_split = (_consistent(atoms_meta, "l3_split", "l3_split")
               if "l3_split" in atoms_meta[0] else 0.5)
    l3_gap_nm = (_consistent(atoms_meta, "l3_gap_nm", "l3_gap_nm")
                if "l3_gap_nm" in atoms_meta[0] else 0.0)
    l3_fill_n = _consistent(atoms_meta, "l3_fill_n", "l3_fill_n")
    # [v3 caveat fix] n_l3_gap used to be `atoms_meta[0].get(...)` -- the
    # FIRST atom's value taken silently, with no check that every other
    # atom in the ramp agrees. Upgraded to the same `_consistent()`
    # cross-atom check every other process parameter gets (also now a
    # hard ValueError/exit 2 if some atom lacks the key at all, via the
    # `_consistent` fix above).
    n_l3_gap = _consistent(atoms_meta, "n_l3_gap", "n_l3_gap")
    h1 = _consistent(atoms_meta, "_H1_nm", "H1_nm (across atoms)")
    h2 = _consistent(atoms_meta, "_H2_nm", "H2_nm (across atoms)")
    h3 = _consistent(atoms_meta, "_H3_nm", "H3_nm (across atoms)")
    spacer = _consistent(atoms_meta, "spacer_nm", "spacer_nm (across atoms)")
    # [v3 caveat fix] F7 extended to these process parameters. They are
    # never in the eval_only json for n_spacer/n_sub/lams_nm/material3 --
    # present here only via the `_load_atom` job-txt fallback -- so a
    # cross-atom mismatch OR a total absence both raise ValueError here
    # (never a KeyError crash).
    n_spacer = _consistent(atoms_meta, "n_spacer", "n_spacer (across atoms)")
    n_sub = _consistent(atoms_meta, "n_sub", "n_sub (across atoms)")
    lams_nm = _consistent(atoms_meta, "lams_nm", "lams_nm (across atoms)")
    material3 = _consistent(atoms_meta, "material3",
                            "material3 (across atoms)")
    pol_atoms = _consistent(atoms_meta, "pol", "pol (across atoms)")
    return dict(nxy=nxy, l3_split=l3_split, l3_gap_nm=l3_gap_nm,
               l3_fill_n=l3_fill_n, n_l3_gap=n_l3_gap, h1_nm=h1, h2_nm=h2,
               h3_nm=h3, spacer_nm=spacer, n_spacer=n_spacer, n_sub=n_sub,
               lams_nm=lams_nm, material3=material3, pol=pol_atoms)


def _mirror_atom_y(r3):
    """[260927 FIX, defect 3] Exact per-atom y-mirror of a loaded/projected
    density -- axis 1 of the (nxy, nxy) array (or of each sub-layer, for a
    bilayer atom's 2-list), matching the atom driver's own y-symmetry
    convention (`sym_y`) and the verifier's independent needle. Reversing
    the ATOM ORDER along a ramp is an exact y-mirror of the supercell only
    when every individual atom is ALSO y-mirror-symmetric on its own
    (`sym_y=1`, true for TM's own palette atoms). The TE palette's atoms
    are the TM palette rotated 90 deg, which turns that y-symmetry into
    x-symmetry (`sym_y=0`) -- so for those, order-reversal alone is NOT a
    y-mirror and must be paired with mirroring each atom's own pattern."""
    if isinstance(r3, (list, tuple)):
        return [np.ascontiguousarray(np.asarray(z)[:, ::-1]) for z in r3]
    return np.ascontiguousarray(np.asarray(r3)[:, ::-1])


def run_one(atom_paths, ramp_axis, M, nG, carrier_1d, H, sp, n_sp, eps3,
           lam, pol, ramp_s, sc_shift, tag, lpa_refs=None, atom_meta_dirs=None,
           mirror_y=False):
    atoms_r3, atoms_meta = _load_atoms(atom_paths, meta_dirs=atom_meta_dirs)
    if mirror_y:
        # [260927 FIX, defect 3] see `_mirror_atom_y` above. Applied AFTER
        # loading (never changes which atom/meta pair is used, only the
        # in-plane pattern each atom presents), so every other measurement
        # and cross-check below is unaffected.
        # [260927 FIX r2, defect 4] Mirror only the atoms whose OWN density
        # is NOT already y-mirror-symmetric (sym_y==0, the TE/rotated
        # palette's case -- see `_mirror_atom_y`'s docstring). A sym_y==1
        # atom (TM's own convention) is exactly y-symmetric already, so
        # mirroring it is a PHYSICAL no-op but not a numerical one (the
        # array flip changes memory layout / FFT summation order) --
        # applying it unconditionally to every atom silently broke the TM
        # N3 needle's bit-identity against the installed v3 driver, which
        # has no mirror_y concept at all (measured: 104-109 differing
        # numeric leaves, all in the N3 block, max_abs ~9e-13). Flipping
        # only the atoms that actually need it (sym_y==0) restores TM's
        # exact match while keeping the TE mirror this needle needs.
        atoms_r3 = [
            _mirror_atom_y(z) if int(m.get("sym_y", 1)) == 0 else z
            for z, m in zip(atoms_r3, atoms_meta)]
    cc = _cross_check_atoms(atoms_meta)
    nxy, l3_split, l3_gap_nm, l3_fill_n, n_l3_gap = (
        cc["nxy"], cc["l3_split"], cc["l3_gap_nm"], cc["l3_fill_n"],
        cc["n_l3_gap"])
    rss0 = _peak_rss_mb()
    t0 = time.time()
    info = _measure_super(nG, atoms_r3, carrier_1d, H, sp, n_sp, eps3, lam,
                          nxy, pol, ramp_axis, M, l3_split, l3_gap_nm,
                          l3_fill_n, n_l3_gap, ramp_s, sc_shift, lpa_refs)
    wall = time.time() - t0
    rss1 = _peak_rss_mb()
    info["tag"] = tag
    info["wall_s"] = wall
    info["rss_before_mb"] = rss0
    info["rss_after_mb"] = rss1
    info["nG_req"] = nG
    info["atom_paths"] = atom_paths
    info["atoms_A_deg"] = [float(m["A_deg"]) for m in atoms_meta]
    info["atoms_amp_tA"] = [float(m["amp_tA"]) for m in atoms_meta]
    print("[SUPERRAMP] %s: ramp_axis=%s M=%d nG=%d ramp_s=%+d sc_shift=%d "
          "wall=%.1fs rss=%.0fMB"
          % (tag, ramp_axis, M, nG, ramp_s, sc_shift, wall, (rss1 or 0)),
          flush=True)
    return info, atoms_meta, atoms_r3


def main():
    G.set_period(_num("period_nm", 1315.39))
    G.set_substrate(_num("n_sub", 1.4440))
    lam = _num("lams_nm", 1550.0) * 1e-3
    # [261015 kx_hold] OPT-IN matched angle for the supercell solves (thA / thB / th all read G.THETA_SUB at
    # call time). Absent / 0 -> byte-identical. Recorded in the result JSON through _li_extra.
    if int(_num("kx_hold", 0)):
        if not hasattr(G, "pin_theta_to"):
            print("[SUPERRAMP] ERROR: scd_c_grating has no pin_theta_to -- refusing to run a kx_hold job at "
                  "the fixed angle", flush=True)
            return 2
        _th_kx = G.pin_theta_to(lam)
        _li_extra.update(kx_hold=1, theta_sub_deg_used=_th_kx)
        print("[SUPERRAMP] kx_hold = 1: THETA_SUB = %.5f deg at %.3f nm (kx = G held)" % (_th_kx, lam * 1e3),
              flush=True)
    H = [_num("H1_nm", 400.0) * 1e-3, _num("H2_nm", 300.0) * 1e-3, None]
    sp = _num("spacer_nm", 465.0) * 1e-3
    n_sp = _num("n_spacer", 1.4440)
    eps3 = G.eps_of(_txt("material3", "a-Si"))
    pol = (_txt("pol", "TM").strip().upper() or "TM")
    # D-C (v4): pol=TE is now supported -- see the module docstring's D-C
    # entry for what was fixed (rB's order, everywhere it is read/reported)
    # and why nothing else needed to change. Still refuse anything else.
    if pol not in ("TM", "TE"):
        print("[SUPERRAMP] FATAL: pol=%r is not supported by this driver "
              "-- only TM and TE are implemented. Refusing to run." % pol,
              flush=True)
        return 2
    nG = int(_num("nG", 41))
    nG_eval_ref = int(_num("nG_eval", 601))

    l12 = _txt("l12_density")
    if not l12 or not os.path.isfile(l12):
        print("[SUPERRAMP] ERROR: l12_density not found -> %s" % l12,
              flush=True)
        return 2
    araw = np.load(l12)

    def carrier_1d(nx0):
        return [_resample_col(araw[0], nx0), _resample_col(araw[1], nx0)]

    atoms_raw = _txt("atoms")
    atom_list = [a.strip() for a in atoms_raw.split(";") if a.strip()]
    # [v4, D-E] OPTIONAL, 1:1 with atom_list; an empty slot (or the whole
    # key absent) falls back to that atom's own npy directory -- v3's
    # only behaviour, so a job that never sets this key is unaffected.
    _atoms_meta_raw = _txt("atoms_meta")
    if _atoms_meta_raw:
        atom_meta_dirs = [a.strip() for a in _atoms_meta_raw.split(";")]
        if len(atom_meta_dirs) != len(atom_list):
            print("[SUPERRAMP] FATAL: atoms_meta has %d entries, atoms has "
                  "%d -- must match 1:1 in order (empty slots allowed)"
                  % (len(atom_meta_dirs), len(atom_list)), flush=True)
            return 2
    else:
        atom_meta_dirs = [None] * len(atom_list)
    ramp_axis = _txt("ramp_axis", "x").strip().lower()
    # M2: x-ramp's PRIMARY sign is -1 (default); +1 is the control.
    ramp_s = int(_num("ramp_s", -1))
    if ramp_s not in (-1, 1):
        print("[SUPERRAMP] FATAL: ramp_s must be -1 or +1 (got %r)"
              % ramp_s, flush=True)
        return 2
    # M8: cyclic carrier registration, x-ramp only.
    sc_shift = int(_num("sc_shift", 0))
    if ramp_axis == "x" and sc_shift not in (0, 1, 2):
        print("[SUPERRAMP] FATAL: sc_shift must be 0, 1 or 2 (got %r)"
              % sc_shift, flush=True)
        return 2
    needle = _txt("needle", "none").strip()  # F1: default is PRODUCTION
    ref_mode = _txt("ref_mode", "solve").strip().lower()
    # D-D: `nG_ref` is resolved (and harmonic-match-checked against `nG`)
    # per branch below, via `_nG_ref_check` -- NOT defaulted to the bare
    # `nG` here (that old default is exactly the mismatch the 260926
    # verifier measured 15.6% error from). `_nG_ref_raw` is None when the
    # job did not set the key at all. `nG_ref` itself starts as this raw
    # value purely so branches that never call `_nG_ref_check` (N1x/N1y/
    # N3/prop -- none of which use a reference solve) still have SOME
    # value to put in the output JSON; it is overwritten with the
    # validated value in every branch that actually uses it.
    _nG_ref_raw = CFG.get("nG_ref")
    nG_ref = (int(float(_nG_ref_raw)) if _nG_ref_raw not in (None, "")
             else None)

    if not atom_list:
        print("[SUPERRAMP] ERROR: atoms= is empty", flush=True)
        return 2

    # ---- F7: load every listed atom's json up front and hard-check ------
    # [260927 FIX, defect 5] the loading list comprehension itself used to
    # sit OUTSIDE this try/except, so a `_load_atom` ValueError (e.g. the
    # new atoms_meta pairing check) was an uncaught crash, not the clean
    # "FATAL (F7)"/exit 2 every other cross-atom mismatch gets. Both steps
    # now share one try/except.
    try:
        atoms_meta0 = [_load_atom(p, meta_dir=m)[1]
                      for p, m in zip(atom_list, atom_meta_dirs)]
        cc0 = _cross_check_atoms(atoms_meta0)
    except ValueError as e:
        print("[SUPERRAMP] FATAL (F7, cross-atom mismatch): %s" % e,
              flush=True)
        return 2
    _job_h3 = CFG.get("H3_nm")
    _job_split = CFG.get("l3_split")
    _job_gap = CFG.get("l3_gap_nm")
    for _name, _jv, _av, _kind in (
            ("H1_nm", H[0] * 1e3, cc0["h1_nm"], "num"),
            ("H2_nm", H[1] * 1e3, cc0["h2_nm"], "num"),
            ("spacer_nm", sp * 1e3, cc0["spacer_nm"], "num"),
            ("H3_nm", (float(_job_h3) if _job_h3 else None),
             cc0["h3_nm"], "num"),
            ("l3_split", (float(_job_split) if _job_split else None),
             cc0["l3_split"], "num"),
            ("l3_gap_nm", (float(_job_gap) if _job_gap else None),
             cc0["l3_gap_nm"], "num"),
            # [v3 caveat fix] F7 extended to the process parameters every
            # atom in the ramp must ALSO share with the superramp job
            # itself (checked "if given" like H3_nm/l3_split/l3_gap_nm
            # above -- these five always resolve to SOME value in this
            # driver via a default, so "given" means the job file itself
            # set the key, not merely that a default applied).
            ("n_spacer", (float(CFG["n_spacer"]) if "n_spacer" in CFG
                         else None), cc0["n_spacer"], "num"),
            ("n_sub", (float(CFG["n_sub"]) if "n_sub" in CFG else None),
             cc0["n_sub"], "num"),
            ("lams_nm", (float(CFG["lams_nm"]) if ("lams_nm" in CFG and not _LAM_OVERRIDE)
                        else None), cc0["lams_nm"], "num"),
            ("pol", (CFG["pol"].strip().upper() if "pol" in CFG else None),
             cc0["pol"], "str"),
            ("material3", (CFG["material3"].strip() if "material3" in CFG
                          else None), cc0["material3"], "str")):
        if _jv is None:
            continue
        if _kind == "str":
            if str(_jv).strip().upper() != str(_av).strip().upper():
                print("[SUPERRAMP] FATAL (F7): job %s=%r does not match "
                      "atom(s) %s=%r -- refusing to run"
                      % (_name, _jv, _name, _av), flush=True)
                return 2
        elif abs(float(_jv) - float(_av)) > 1e-6 * max(1.0, abs(float(_av))):
            print("[SUPERRAMP] FATAL (F7): job %s=%.6g does not match "
                  "atom(s) %s=%.6g -- refusing to run"
                  % (_name, _jv, _name, _av), flush=True)
            return 2
    if _LAM_OVERRIDE:
        _li_extra.update(lam_eval_override=1, lam_eval_nm=float(CFG.get("lams_nm", "1550.0")),
                         atom_lams_nm=float(cc0["lams_nm"]))
        print("[SUPERRAMP] lam_eval_override=1: job lams_nm=%s nm, atoms designed at %s nm -- F7 lams_nm check skipped "
              "(band evaluation of a fixed lens; LPA references solved at the job wavelength)"
              % (CFG.get("lams_nm", "1550.0"), cc0["lams_nm"]), flush=True)
    H[2] = cc0["h3_nm"] * 1e-3

    od = _txt("result_folder", "") or os.path.join(
        _txt("Output_folder", "."), _txt("subfolder_name", "superramp"))
    os.makedirs(od, exist_ok=True)

    results = {}
    lpa_block = {}

    def get_carrier(nx0):
        return carrier_1d(nx0)

    # =====================================================================
    # F1: PRODUCTION MODE (needle == "none", the default) -- ONE
    # measurement on the job's own atoms / ramp_axis / ramp_s / sc_shift /
    # nG, plus the F5 LPA reference/comparison, written to
    # superramp_result.json. This is the branch the old driver did not
    # have at all (a needle=none job used to write an empty result).
    # =====================================================================
    if needle in ("none", ""):
        _t_whole0 = time.time()  # [v3 caveat fix] wraps the WHOLE
        # measurement -- ref solves included -- unlike `elapsed_s` inside
        # `_measure_super`, which times only the two supercell builds.
        M = len(atom_list)
        atoms_r3, atoms_meta = _load_atoms(atom_list, meta_dirs=atom_meta_dirs)
        nxy = int(atoms_meta[0]["NXY"])
        _nGref_report = dict(nG_ref_requested=None, nG_ref_actual=None,
                             nG_super_sublattice_actual=None,
                             nG_ref_matched=None)
        if ref_mode == "jsons":
            ref_raw = _txt("ref_jsons")
            ref_paths = [a.strip() for a in ref_raw.split(";") if a.strip()]
            if len(ref_paths) != M:
                print("[SUPERRAMP] FATAL: ref_jsons has %d entries, atoms "
                      "has %d -- must match 1:1 in order"
                      % (len(ref_paths), M), flush=True)
                return 2
            lpa_refs = _lpa_ref_from_jsons(ref_paths)
        else:
            # D-D: nG_ref must be harmonic-matched to nG -- required
            # explicitly (checked) OR defaulted to the matched value
            # (`_nG_ref_check`, which wraps `_matched_single_nG`, the
            # inverse of `_matched_super_nG`). The old unconditional
            # `nG_ref = nG` default was itself a routine mismatch (the
            # 260926 verifier measured 15.6% error from exactly this).
            _nGref, _nGref_act, _nGsup_sub, _nGref_ok = _nG_ref_check(
                nG, nG_ref, ramp_axis, M)
            if not _nGref_ok:
                print("[SUPERRAMP] FATAL (D-D): nG_ref=%r is not "
                      "harmonic-matched to nG=%d (%s-ramp M=%d) -- "
                      "nG_ref_actual=%r, supercell matched-sublattice "
                      "harmonics=%r -- refusing to run"
                      % (_nGref, nG, ramp_axis, M, _nGref_act, _nGsup_sub),
                      flush=True)
                return 2
            nG_ref = _nGref
            _nGref_report = dict(
                nG_ref_requested=nG_ref, nG_ref_actual=_nGref_act,
                nG_super_sublattice_actual=_nGsup_sub, nG_ref_matched=True)
            lpa_refs = _lpa_ref_from_solve(
                atoms_r3, atoms_meta, get_carrier(3 * nxy), H, sp, n_sp,
                eps3, lam, nxy, pol, nG_ref)
        info, _meta, _r3 = run_one(
            atom_list, ramp_axis, M, nG, get_carrier(3 * nxy), H, sp, n_sp,
            eps3, lam, pol, ramp_s, sc_shift, tag="production",
            lpa_refs=lpa_refs, atom_meta_dirs=atom_meta_dirs)
        results["production"] = info
        out = dict(mode="production", needle_results=results,
                  timing_points=[], nG_req=nG, nG_eval_ref=nG_eval_ref,
                  nG_ref=nG_ref, ref_mode=ref_mode, pol=pol,
                  ramp_axis_requested=ramp_axis, ramp_s=ramp_s,
                  sc_shift=sc_shift, atoms_requested=atom_list,
                  wall_s=time.time() - _t_whole0, **_nGref_report, **_li_extra)
        outpath = os.path.join(od, "superramp_result.json")
        io.open(outpath, "w", encoding="utf-8").write(
            json.dumps(out, indent=1))
        print("[SUPERRAMP] wrote %s" % outpath, flush=True)
        return 0

    # =====================================================================
    # NEEDLES (on request only, F1) -- validation, not production.
    # =====================================================================
    _t_needle0 = time.time()  # [v3 caveat fix] wraps the whole needle run
    # (every needle block below, in "all" mode) for the same reason the
    # production branch's `wall_s` does -- `elapsed_s` inside
    # `_measure_super` only ever times the two supercell builds.
    _nG_ref_ok_report = dict(nG_ref_requested=None, nG_ref_actual=None,
                             nG_super_sublattice_actual=None,
                             nG_ref_matched=None)
    # ---------------- F6: N1x / N1y matched-harmonic identical-atom check
    if needle in ("N1x", "N1", "all"):
        p = atom_list[0]
        p_meta = atom_meta_dirs[0]  # [v4, D-E]
        r, meta = _load_atom(p, meta_dir=p_meta)
        nxy = int(meta["NXY"])
        c1 = get_carrier(3 * nxy)
        nGm = _matched_super_nG(nG, "x", 6)
        if nGm is None:
            print("[SUPERRAMP] N1x: no matched nG found below search "
                  "limit -- needle SKIPPED", flush=True)
        else:
            info, _m, _r3 = run_one([p] * 6, "x", 6, nGm, c1, H, sp, n_sp,
                                    eps3, lam, pol, ramp_s=+1, sc_shift=0,
                                    tag="N1x_M6_matched",
                                    atom_meta_dirs=[p_meta] * 6)
            ATOM.ATOMS_PER_PERIOD = 3
            ATOM.L3_LAYERS = 2 if isinstance(r, list) else 1
            ATOM.L3_SPLIT = float(meta.get("l3_split", 0.5))
            ATOM.L3_GAP_NM = float(meta.get("l3_gap_nm", 0.0))
            ATOM.L3_FILL_N = float(meta.get("l3_fill_n", 1.0))
            ATOM.N_L3_GAP = float(meta.get("n_l3_gap", 1.4440))
            ATOM.L3_ETCH_STACK = int(meta.get("l3_etch_stack", 0) or 0)
            # [v4, D-C] thA/rB order are pol-dependent -- see
            # `_rb_small_order` and `_lpa_ref_from_solve`'s own note.
            thA = 0.0 if pol == "TE" else G.THETA_SUB
            th = G.THETA_SUB
            obp = ATOM._obj(nG, r, c1, H, sp, n_sp, eps3, lam, nxy, thA, pol)
            obm = ATOM._obj(nG, r, c1, H, sp, n_sp, eps3, lam, nxy, -th, pol)
            gA0, gB0 = (0, +1) if pol == "TE" else (-1, +1)
            rb_order = _rb_small_order(pol)
            ref = dict(
                tA=complex(ATOM._amp(obp, gA0, pol=pol)),
                rB=complex(ATOM._amp(obp, rb_order, refl=True, pol=pol)),
                tB=complex(ATOM._amp(obm, gB0, pol=pol)),
                echoP=float(ATOM._ord_power(
                    obm, obm.RT_Solve(normalize=1, byorder=1)[0], +2)),
                fid=float(ATOM._fid_l3(obp, pol, sp, n_sp, gA0)),
                fidm=float(ATOM._fid_l3(obm, pol, sp, n_sp, gB0)),
                r00=complex(ATOM._r00_l3(obp, pol, sp, gA0)),
                nG_actual=int(obp.nG))
            row = dict(nG_req=nG, nG_matched=nGm,
                      nG_actual_super=info["nG_actual"],
                      nG_actual_single=ref["nG_actual"])
            row["rel_tA"] = _rel(complex(*info["amp_tA_zero"]), ref["tA"])
            row["rel_rB"] = _rel(complex(*info["amp_rB_zero"]), ref["rB"])
            row["rel_tB"] = _rel(complex(*info["amp_tB_zero"]), ref["tB"])
            row["rel_echo_power"] = _rel(info["echo_R_power"], ref["echoP"])
            row["rel_fid"] = _rel(info["fid_l3"], ref["fid"])
            row["rel_fidm"] = _rel(info["fid_l3_m"], ref["fidm"])
            row["rel_r00"] = _rel(info["absR00"], abs(ref["r00"]))
            row["pass_all_1e9"] = bool(max(
                row["rel_tA"], row["rel_rB"], row["rel_tB"],
                row["rel_echo_power"], row["rel_fid"], row["rel_fidm"],
                row["rel_r00"]) <= 1e-9)
            row["measure"] = info
            results["N1x"] = row
            print("[SUPERRAMP] N1x pass_all_1e9=%s max_rel=%.3e"
                  % (row["pass_all_1e9"], max(
                      row["rel_tA"], row["rel_rB"], row["rel_tB"],
                      row["rel_echo_power"], row["rel_fid"],
                      row["rel_fidm"], row["rel_r00"])), flush=True)

    if needle in ("N1y", "N1", "all"):
        p = atom_list[0]
        p_meta = atom_meta_dirs[0]  # [v4, D-E]
        r, meta = _load_atom(p, meta_dir=p_meta)
        nxy = int(meta["NXY"])
        c1 = get_carrier(3 * nxy)
        nGm = _matched_super_nG(nG, "y", 2)
        if nGm is None:
            print("[SUPERRAMP] N1y: no matched nG found below search "
                  "limit -- needle SKIPPED", flush=True)
        else:
            info, _m, _r3 = run_one([p, p], "y", 2, nGm, c1, H, sp, n_sp,
                                    eps3, lam, pol, ramp_s=+1, sc_shift=0,
                                    tag="N1y_M2_matched",
                                    atom_meta_dirs=[p_meta, p_meta])
            ATOM.ATOMS_PER_PERIOD = 3
            ATOM.L3_LAYERS = 2 if isinstance(r, list) else 1
            ATOM.L3_SPLIT = float(meta.get("l3_split", 0.5))
            ATOM.L3_GAP_NM = float(meta.get("l3_gap_nm", 0.0))
            ATOM.L3_FILL_N = float(meta.get("l3_fill_n", 1.0))
            ATOM.N_L3_GAP = float(meta.get("n_l3_gap", 1.4440))
            ATOM.L3_ETCH_STACK = int(meta.get("l3_etch_stack", 0) or 0)
            # [v4, D-C] thA/rB order are pol-dependent -- see
            # `_rb_small_order` and `_lpa_ref_from_solve`'s own note.
            thA = 0.0 if pol == "TE" else G.THETA_SUB
            th = G.THETA_SUB
            obp = ATOM._obj(nG, r, c1, H, sp, n_sp, eps3, lam, nxy, thA, pol)
            obm = ATOM._obj(nG, r, c1, H, sp, n_sp, eps3, lam, nxy, -th, pol)
            gA0, gB0 = (0, +1) if pol == "TE" else (-1, +1)
            rb_order = _rb_small_order(pol)
            ref = dict(
                tA=complex(ATOM._amp(obp, gA0, pol=pol)),
                rB=complex(ATOM._amp(obp, rb_order, refl=True, pol=pol)),
                tB=complex(ATOM._amp(obm, gB0, pol=pol)),
                echoP=float(ATOM._ord_power(
                    obm, obm.RT_Solve(normalize=1, byorder=1)[0], +2)),
                fid=float(ATOM._fid_l3(obp, pol, sp, n_sp, gA0)),
                fidm=float(ATOM._fid_l3(obm, pol, sp, n_sp, gB0)),
                r00=complex(ATOM._r00_l3(obp, pol, sp, gA0)),
                nG_actual=int(obp.nG))
            row = dict(nG_req=nG, nG_matched=nGm,
                      nG_actual_super=info["nG_actual"],
                      nG_actual_single=ref["nG_actual"])
            row["rel_tA"] = _rel(complex(*info["amp_tA_zero"]), ref["tA"])
            row["rel_rB"] = _rel(complex(*info["amp_rB_zero"]), ref["rB"])
            row["rel_tB"] = _rel(complex(*info["amp_tB_zero"]), ref["tB"])
            row["rel_echo_power"] = _rel(info["echo_R_power"], ref["echoP"])
            row["rel_fid"] = _rel(info["fid_l3"], ref["fid"])
            row["rel_fidm"] = _rel(info["fid_l3_m"], ref["fidm"])
            row["rel_r00"] = _rel(info["absR00"], abs(ref["r00"]))
            row["pass_all_1e9"] = bool(max(
                row["rel_tA"], row["rel_rB"], row["rel_tB"],
                row["rel_echo_power"], row["rel_fid"], row["rel_fidm"],
                row["rel_r00"]) <= 1e-9)
            row["measure"] = info
            results["N1y"] = row
            print("[SUPERRAMP] N1y pass_all_1e9=%s max_rel=%.3e"
                  % (row["pass_all_1e9"], max(
                      row["rel_tA"], row["rel_rB"], row["rel_tB"],
                      row["rel_echo_power"], row["rel_fid"],
                      row["rel_fidm"], row["rel_r00"])), flush=True)

    # ---------------- N3: y-mirror EXACT symmetry (F6) --------------------
    # (x-reversal is NOT an exact symmetry -- oblique incidence + a carrier
    # that is not itself x-symmetric -- so it is retired as a needle.)
    if needle in ("N3", "all"):
        if len(atom_list) >= 4:
            picks, picks_meta = atom_list[:4], atom_meta_dirs[:4]
        else:
            _rep3 = 4 // max(1, len(atom_list)) + 1
            picks = (atom_list * _rep3)[:4]
            picks_meta = (atom_meta_dirs * _rep3)[:4]
        nxy = int(_load_atom(picks[0], meta_dir=picks_meta[0])[1]["NXY"])
        c1 = get_carrier(3 * nxy)
        fwd, _mf, _rf = run_one(picks, "y", 4, nG, c1, H, sp, n_sp, eps3,
                                lam, pol, ramp_s=+1, sc_shift=0,
                                tag="N3_fwd", atom_meta_dirs=picks_meta)
        # [260927 FIX, defect 3] Reversing the atom ORDER alone is an exact
        # y-mirror of the supercell only when every individual atom is also
        # y-mirror-symmetric on its own (sym_y=1, true for TM's palette).
        # The TE palette's atoms are the TM palette rotated 90 deg
        # (sym_y=0) -- for those, order-reversal alone leaves each atom's
        # own pattern un-mirrored, so this needle used to report
        # pass_1e9=False for every correct TE run. Fixed: also mirror each
        # atom's own density in y (`mirror_y=True`, see `_mirror_atom_y`).
        # For TM (sym_y=1) atoms this mirror is a no-op up to floating-
        # point noise, so the needle stays exact there too.
        rev, _mr, _rr = run_one(list(reversed(picks)), "y", 4, nG, c1, H,
                                sp, n_sp, eps3, lam, pol, ramp_s=-1,
                                sc_shift=0, tag="N3_rev",
                                atom_meta_dirs=list(reversed(picks_meta)),
                                mirror_y=True)
        # y -> -y with the atom order reversed AND each atom mirrored in y
        # is an EXACT mirror: forward s=+1 must equal reverse s=-1 bit for
        # bit (up to solver noise).
        results["N3"] = dict(
            forward=fwd, reverse=rev,
            rel_target_power=_rel(fwd["power_tA_target"],
                                  rev["power_tA_target"]),
            rel_target_amp=_rel(complex(*fwd["amp_tA_target"]),
                                complex(*rev["amp_tA_target"])),
            rel_zero_power=_rel(fwd["power_tA_zero"], rev["power_tA_zero"]),
            pass_1e9=bool(max(
                _rel(fwd["power_tA_target"], rev["power_tA_target"]),
                _rel(fwd["power_tA_zero"], rev["power_tA_zero"])) <= 1e-9))
        print("[SUPERRAMP] N3 pass_1e9=%s rel_target_power=%.3e"
              % (results["N3"]["pass_1e9"],
                 results["N3"]["rel_target_power"]), flush=True)

    # ---------------- N4 smoke (also serves as N2 energy check) ----------
    if needle in ("N4", "N2", "all"):
        M4 = min(6, max(3, (len(atom_list) // 3) * 3)) if len(atom_list) >= 6 \
            else 6
        _rep4 = (M4 // max(1, len(atom_list))) + 1
        picks = (atom_list * _rep4)[:M4]
        picks_meta = (atom_meta_dirs * _rep4)[:M4]
        atoms_r3, atoms_meta = _load_atoms(picks, meta_dirs=picks_meta)
        nxy = int(atoms_meta[0]["NXY"])
        c1 = get_carrier(3 * nxy)
        # D-D applies here too -- N4 also builds LPA refs via nG_ref.
        _nGref4, _nGref4_act, _nGsup4_sub, _nGref4_ok = _nG_ref_check(
            nG, nG_ref, "x", M4)
        if not _nGref4_ok:
            print("[SUPERRAMP] FATAL (D-D, N4): nG_ref=%r is not "
                  "harmonic-matched to nG=%d (x-ramp M=%d) -- "
                  "nG_ref_actual=%r, supercell matched-sublattice "
                  "harmonics=%r -- refusing to run"
                  % (_nGref4, nG, M4, _nGref4_act, _nGsup4_sub), flush=True)
            return 2
        nG_ref = _nGref4
        _nG_ref_ok_report = dict(
            nG_ref_requested=nG_ref, nG_ref_actual=_nGref4_act,
            nG_super_sublattice_actual=_nGsup4_sub, nG_ref_matched=True)
        lpa_refs = _lpa_ref_from_solve(atoms_r3, atoms_meta, c1, H, sp,
                                       n_sp, eps3, lam, nxy, pol, nG_ref)
        info, _m, _r3 = run_one(picks, "x", M4, nG, c1, H, sp, n_sp, eps3,
                                lam, pol, ramp_s=ramp_s, sc_shift=sc_shift,
                                tag="N4", lpa_refs=lpa_refs,
                                atom_meta_dirs=picks_meta)
        info["energy_A_err"] = abs(info["energy_total_A"] - 1.0)
        info["energy_B_err"] = abs(info["energy_total_B"] - 1.0)
        results["N4"] = info
        results["N2"] = dict(
            energy_total_A=info["energy_total_A"],
            energy_total_B=info["energy_total_B"],
            energy_A_err=info["energy_A_err"],
            energy_B_err=info["energy_B_err"], source_tag="N4")
        lpa_block = info.get("lpa", {})

    out = dict(mode="needle", needle_results=results, lpa_prediction=lpa_block,
              nG_req=nG, nG_eval_ref=nG_eval_ref, nG_ref=nG_ref,
              ref_mode=ref_mode, pol=pol, ramp_axis_requested=ramp_axis,
              ramp_s=ramp_s, sc_shift=sc_shift, atoms_requested=atom_list,
              wall_s=time.time() - _t_needle0, **_nG_ref_ok_report, **_li_extra)
    outpath = os.path.join(od, "superramp_result.json")
    io.open(outpath, "w", encoding="utf-8").write(
        json.dumps(out, indent=1))  # F: no default=None -- fail LOUDLY on
    # anything unserialisable rather than silently writing null.
    print("[SUPERRAMP] wrote %s" % outpath, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
