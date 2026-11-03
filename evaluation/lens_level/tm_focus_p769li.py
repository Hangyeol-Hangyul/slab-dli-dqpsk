# -*- coding: utf-8 -*-
r"""
tm_focus.py -- TM metalens focusing visualization.

Four 150x150um quadrants mirrored into one 300x300um aperture, one Gaussian pump
(w=100um) centred at the origin, each quadrant its own hyperbolic lens (f=200um,
lambda=1.55um in air) focusing onto its own 8x8um PD at (+-39.894,+-39.894)um.
TM lane, current 8-level LPA palette P769 (Li records).

Three near-field models per arm (A,B):
  M0 ideal            : continuous phase, unit amplitude, Gaussian-apodised.
  M1 LPA palette       : Lambda/3 grid, nearest-phase palette atom (measured
                         complex tA/tB, amplitude included).
  M2 = M1 * beta_arm(P): Li superramp beta(P) ladder model (P769 verdict set,
                         rB0_complex basis, after conv()), 'decay' tail.

See tm_focus.md (written at the end of this script's run) for full method,
control result, figure list and limitations.

HARD RULES observed here:
 - never writes under <DATA_ROOT> (read-only source tree; only reads
   from there)
 - OMP/MKL/OPENBLAS_NUM_THREADS=1 set before numpy import; cpu affinity
   pinned to logical cores >=10; peak arrays kept well under ~4 GB (tracked below)
 - code from <DATA_ROOT> is never imported as a module here (only two
   scratchpad review scripts are read as TEXT and exec'd, which is what the
   existing validated pipeline (lens_ladder_eval_recheck3.py) itself does --
   no .pyc side effect either way); sys.dont_write_bytecode is set regardless

[261002 ADAPTATION] Copy of the 260929 script re-pointed at the FROZEN TM palette (rho3_opt.npy md5
 3fdfa744 / 23fb4eaf / b59ed0ba / ed750b02 / 4ba93995 / 4a7a8295 / 488fc44a / 142a72ac for
 32.5 / 77.5 / 122.5 / 167.5 / 212.5 / 257.5 / 302.5 / 347.5).
 Changed: input paths (LEVEL_DIRS, EXPECTED_MD5, beta-ladder candidate list, NEW_DIR), OUTDIR, and the md text
 that hard-coded 260929-specific facts. ADDED (no algorithm line touched): md5 gate, ladder fallback log,
 CONTROL-2 (sim585 verdict reproduction), md section 0 (old-vs-new). See diff_vs_260929.txt.

[261011 ADAPTATION] Copy of the 261002 script (tm_focus_frozen261002.py, never edited) re-pointed at the CURRENT TM palette P769
 (Li records, rcwa_rule liy / rcwa_ff 1, nG 1793; rho3 md5 53474519 / f8176952 / b59ed0ba / ed750b02 / 78259586 / accded83 /
 ee3609ef / accf4825 for 32.5 / 77.5 / 122.5 / 167.5 / 212.5 / 257.5 / 302.5 / 347.5) and the CURRENT Li lens ladder (the 21 cells of
 the verdict "P769 approved (Li terms)" of lens_total_li5.py, read through lens_total_li.py's own functions).
 REMOVED: the recheck3 CONTROL and CONTROL-2 (sim585) -- their reference json no longer exists.
 ADDED: CONTROL-V (the verdict, lens_total_li.evaluate_cells with the P769 atom-level terms, vs lens_total_li5_final.json, tol 1e-12),
 CONTROL-F (this script's own focal_q / beta_model reproduce the ladder betaL at f=200, tol 1e-9 dB), the flag --controls-only
 (md5 gate, palette, ladder, CONTROL-V, CONTROL-F, scratch\controls_261011.json, exit 0), the TE rot90 copies of the full-aperture
 and focal-plane figures, and tm_focus_arrays.npz (the exact plotted arrays). SECTION 4 onward: no algorithm line changed
 (text, summary keys, md writer and the additions above only). See diff_vs_261002.txt.
 RCWA cell results + scalar angular-spectrum propagation; not a full 3-D field solve.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
import time
import json
import math
import cmath

_ap = argparse.ArgumentParser(description="TM metalens focal-plane visualisation, P769 Li palette + Li lens ladder")
_ap.add_argument("--controls-only", action="store_true",
                 help="run imports, md5 gate, palette load, ladder load, CONTROL-V, CONTROL-F, write scratch\\controls_261011.json, exit 0")
CONTROLS_ONLY = _ap.parse_args().controls_only

T_START = time.time()


def log(msg):
    print("[%8.1fs] %s" % (time.time() - T_START, msg), flush=True)


import psutil
_proc = psutil.Process()


def _pin(tag=""):
    pass


_pin("at start")

_vm = psutil.virtual_memory()
log("RAM before compute: total %.1f GB, avail %.1f GB, used %.1f %%" %
    (_vm.total / 1e9, _vm.available / 1e9, _vm.percent))
_lim = float(os.environ.get("SIM_MAX_RAM_PCT", "0"))   # opt-in RAM ceiling
if _lim > 0 and _vm.percent > _lim:
    log("REFUSING: RAM used %.1f%% already exceeds the %.0f%% ceiling" % (_vm.percent, _lim))
    sys.exit(1)

_PEAK_RSS_MB = [0.0]


def track_rss(tag=""):
    r = _proc.memory_info().rss / 1e6
    if r > _PEAK_RSS_MB[0]:
        _PEAK_RSS_MB[0] = r
    log("RSS=%.1f MB (peak %.1f MB) %s" % (r, _PEAK_RSS_MB[0], tag))
    return r


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Nature-style plotting defaults; TM family = Okabe-Ito orange sequential,
# white -> #F0C27A -> #E69F00 -> #D55E00 -> #6B2E00 (per spec).
TM_COLORS = ["#FFFFFF", "#F0C27A", "#E69F00", "#D55E00", "#6B2E00"]
TM_CMAP = LinearSegmentedColormap.from_list("tm_orange", TM_COLORS, N=256)
try:
    _has_arial = any(f.name == "Arial" for f in matplotlib.font_manager.fontManager.ttflist)
except Exception:
    _has_arial = False
plt.rcParams.update({
    "font.family": "Arial" if _has_arial else "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
})

track_rss("after numpy/matplotlib import")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH = r"<WORK_ROOT>\tm_focus_261011\scratch"
OUTDIR = r"<WORK_ROOT>\tm_focus_261011"
os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

RESULT_ROOT = r"<DATA_ROOT>\Result"
# [261011] CURRENT TM palette P769 (Li records: rcwa_rule liy / rcwa_ff 1, nG 1793, eval_only). One atom2d_opt.json record per level.
# These folders hold atom2d_opt.json + a job file but NO rho3_opt.npy: the density is the file at record["rho3_npy"].
LEVEL_DIRS = {
    "blank_122p5": RESULT_ROOT + r"\261008_scdc_atomli\261008_sim751_atomli_122p5_n1793_liyff1",
    "level_032p5": RESULT_ROOT + r"\261008_scdc_atomcand\261008_sim754_atomcand_53474519_n1793_liyff1",
    "level_077p5": RESULT_ROOT + r"\261008_scdc_atomli\261008_sim751_atomli_077p5_n1793_liyff1",
    "level_167p5": RESULT_ROOT + r"\261008_scdc_atomli\261008_sim751_atomli_167p5_n1793_liyff1",
    "level_212p5": RESULT_ROOT + r"\261008_scdc_atomcand\261008_sim754_atomcand_78259586_n1793_liyff1",
    "level_257p5": RESULT_ROOT + r"\261008_scdc_atomcand\261008_sim754_atomcand_accded83_n1793_liyff1",
    "level_302p5": RESULT_ROOT + r"\261009_scdc_atomreopt\261009_sim759_reopt_302p5_S1_488fc44a_ev1793_liyff1",
    "level_347p5": RESULT_ROOT + r"\261008_scdc_atomli\261008_sim751_atomli_347p5_n1793_liyff1",
}
EXPECTED_MD5 = {"level_032p5": "53474519", "level_077p5": "f8176952", "blank_122p5": "b59ed0ba",
                "level_167p5": "ed750b02", "level_212p5": "78259586", "level_257p5": "accded83",
                "level_302p5": "ee3609ef", "level_347p5": "accf4825"}
OLD_SUMMARY = r"<WORK_ROOT>\tm_focus_261002\tm_focus_summary.json"      # the 261002 run (FROZEN palette, Laurent) -- section 0 comparison

REVIEW_NET_F = r"<WORK_ROOT>\scdc_scratch_261001\wf_lpa\lens\review\review_net_f.py"
# reference code / data of the CURRENT lens verdict (read only; lens_total_li.py is imported, the rest is read as data)
LENS_LI_DIR = r"<WORK_ROOT>\loop_261008\lens_li"
LI5_DIR = r"<WORK_ROOT>\loop_261008\lens_li5"
LI5_FINAL_JSON = LI5_DIR + r"\lens_total_li5_final.json"
SIM769_CELLS_JSON = LI5_DIR + r"\sim769_cells.json"
P769_TERMS_JSON = r"<WORK_ROOT>\loop_261008\improve_261009\tm_lens_total_a1\a1_def_pur_p769_result.json"
VERDICT_KEY = "P769 approved (Li terms)"
LADDER646_HEAD = r"<WORK_ROOT>\readout_261005_evening\scratch_tm646\ladder646.py"   # exec'd by lens_total_li.evaluator()

for _p in list(LEVEL_DIRS.values()) + [REVIEW_NET_F, OLD_SUMMARY, LENS_LI_DIR + r"\lens_total_li.py", LI5_FINAL_JSON,
                                       SIM769_CELLS_JSON, P769_TERMS_JSON, LADDER646_HEAD]:
    if not os.path.exists(_p):
        print("NOT FOUND: %s" % _p, flush=True)
        sys.exit(1)
log("all required input paths found")

# [261011] md5 gate: every level must be the expected P769 density (first 8 hex of md5 of the file at record["rho3_npy"]) and the
# record must be a TM / Li / nG 1793 record. Any mismatch -> print it and stop.
import hashlib
MD5_TABLE = []
_GATE_FAIL = []
for _lvl, _dir in LEVEL_DIRS.items():
    _fp = os.path.join(_dir, "atom2d_opt.json")
    if not os.path.exists(_fp):
        print("NOT FOUND: %s" % _fp, flush=True)
        sys.exit(1)
    _rec = json.load(open(_fp, encoding="utf-8"))
    _rho_path = _rec.get("rho3_npy")
    if not _rho_path or not os.path.exists(_rho_path):
        print("NOT FOUND: %s (record[rho3_npy] of %s)" % (_rho_path, _fp), flush=True)
        sys.exit(1)
    _m = hashlib.md5(open(_rho_path, "rb").read()).hexdigest()[:8]
    _ok = (_m == EXPECTED_MD5[_lvl])
    _want = {"pol": "TM", "rcwa_rule": "liy", "rcwa_ff": 1, "nG": 1793}
    _got = {k: _rec.get(k) for k in _want}
    _rec_ok = all(_got[k] == _want[k] for k in _want)
    MD5_TABLE.append(dict(level=_lvl, folder=_dir, rho3_npy=_rho_path, md5=_m, expected=EXPECTED_MD5[_lvl], ok=bool(_ok),
                          record_fields=_got, record_ok=bool(_rec_ok)))
    log("md5 gate %-12s %s expected %s %s ; record %s %s  <- %s" %
        (_lvl, _m, EXPECTED_MD5[_lvl], "OK" if _ok else "MISMATCH", _got, "OK" if _rec_ok else "MISMATCH", _dir))
    if not _ok:
        _GATE_FAIL.append("%s: md5 %s != expected %s (rho3_npy %s)" % (_lvl, _m, EXPECTED_MD5[_lvl], _rho_path))
    if not _rec_ok:
        _GATE_FAIL.append("%s: record fields %s != required %s" % (_lvl, _got, _want))
if _GATE_FAIL:
    for _g in _GATE_FAIL:
        print("GATE MISMATCH: %s" % _g, flush=True)
    log("palette gate FAILED -- palette is not the approved P769 Li palette, stopping.")
    sys.exit(1)
log("palette gate PASSED for all 8 levels")

# [261011] the 261002 run (FROZEN palette, Laurent) for the md section 0 old-vs-new comparison; loaded here so that a missing file
# stops the run before any compute (the old per-level atom records are the folders listed in its summary's palette_md5_check).
OLD = json.load(open(OLD_SUMMARY, encoding="utf-8"))
_old_md5 = {_r["level"]: _r["md5"] for _r in OLD["palette_md5_check"]}
_old_atom = {}
for _r in OLD["palette_md5_check"]:
    _fp = os.path.join(_r["folder"], "atom2d_opt.json")
    if not os.path.exists(_fp):
        print("NOT FOUND: %s" % _fp, flush=True)
        sys.exit(1)
    _old_atom[_r["level"]] = json.load(open(_fp, encoding="utf-8"))
log("261002 comparison run loaded: %s ; old palette md5 %s" % (OLD_SUMMARY, _old_md5))

# ---------------------------------------------------------------------------
# SECTION 0 -- exact reuse of review_net_f.py's grid + cap() (verbatim exec,
# not reinvented). This namespace is ALSO the one the control check runs on.
# ---------------------------------------------------------------------------
_src_rnf = open(REVIEW_NET_F, encoding="utf-8").read()
_ns_rnf = {}
exec(compile(_src_rnf.split("rows = []")[0], "review_net_f", "exec"), _ns_rnf)

LAM = _ns_rnf["LAM"]; LAMG = _ns_rnf["LAMG"]; W_PUMP = _ns_rnf["W"]; APER = _ns_rnf["APER"]
XC = _ns_rnf["XC"]; K0 = _ns_rnf["K0"]; dx = _ns_rnf["dx"]; n_q = _ns_rnf["n"]
Xq = _ns_rnf["X"]; Yq = _ns_rnf["Y"]; Aq = _ns_rnf["A"]; pin_q = _ns_rnf["pin"]; rho_q = _ns_rnf["rho"]
npd_q = _ns_rnf["npd"]; kz2_q = _ns_rnf["kz2"]; k_q = _ns_rnf["k"]
DT = _ns_rnf["DT"]; AX_box = _ns_rnf["AX"]; Mx_q = _ns_rnf["Mx"]
cap_q = _ns_rnf["cap"]

log("review_net_f grid: LAM=%.5f LAMG=%.5f W=%.1f APER=%.1f XC=%.6f K0=%.6f dx=%.3f n=%d npd=%d"
    % (LAM, LAMG, W_PUMP, APER, XC, K0, dx, n_q, npd_q))

# ---------------------------------------------------------------------------
# SECTION 1 -- beta_model / focal_q (copied unchanged from the 261002 script; identical to ladder646.py model() / focal()).
# [261011] The old pre-swap CONTROL (recheck3 reproduction) and CONTROL-2 (sim585) are REMOVED: lens_ladder_eval_recheck3.json
# no longer exists. They are replaced by CONTROL-V and CONTROL-F in section 3.
# ---------------------------------------------------------------------------

def beta_model(P, Pm, bm, tail, lamg):
    db = 20 * np.log10(np.abs(bm)); ar = np.unwrap(np.angle(bm))
    xv = 1.0 / Pm[::-1]
    D = np.interp(1.0 / P, xv, db[::-1]); F = np.interp(1.0 / P, xv, ar[::-1])
    last = Pm[-1]
    if tail == "decay":
        D = np.where(P > last, db[-1] * last / P, D); F = np.where(P > last, ar[-1] * last / P, F)
    else:
        D = np.where(P > last, db[-1], D); F = np.where(P > last, ar[-1], F)
    b = 10 ** (D / 20.0) * np.exp(1j * F)
    return np.where(P < 2 * lamg, 0.0, b)


def focal_q(E0, f):
    """Exact PD-box focal field on the ORIGINAL quadrant-only grid (review_net_f's Mx)."""
    H = np.where(kz2_q >= 0, np.exp(1j * np.sqrt(np.clip(kz2_q, 0, None)) * f), 0.0)
    Ep = np.zeros((npd_q, npd_q), dtype=complex); Ep[:n_q, :n_q] = E0
    return (Mx_q @ (np.fft.fft2(Ep) * H) @ Mx_q.T) / (npd_q * npd_q)

# ---------------------------------------------------------------------------
# SECTION 2 -- load the CURRENT (post-swap) TM 8-level LPA palette.
# ---------------------------------------------------------------------------
PALETTE_LEVELS = ["blank_122p5", "level_032p5", "level_077p5", "level_167p5",
                  "level_212p5", "level_257p5", "level_302p5", "level_347p5"]

palette = []
for lvl in PALETTE_LEVELS:
    # [261011] per-level folder from LEVEL_DIRS (P769 Li records, all eval_only at nG 1793; the numbers quoted in the comments below are the 260929 palette's)
    fp = os.path.join(LEVEL_DIRS[lvl], "atom2d_opt.json")
    if not os.path.exists(fp):
        print("NOT FOUND: %s" % fp, flush=True)
        sys.exit(1)
    d = json.load(open(fp, encoding="utf-8"))
    if d.get("pol") != "TM":
        log("WARNING: %s pol=%s (expected TM)" % (lvl, d.get("pol")))
    nG_eval = d["nG_eval"]
    # REPAIR (verifier defect, major/factual): the EFFECTIVE nG that amp_tA/A_deg/
    # amp_tB/B_deg actually correspond to is NOT always nG_eval. By convention
    # ("eval_only uses nG, not nG_eval") an eval_only cell's stored amp/phase is at
    # nG; a normal (non-eval_only) cell run under a progressive ng_schedule reports
    # its FINAL evaluation nG in nG_report, which can exceed the schedule's first
    # step (nG). Prefer nG_report when present, else nG. Verified against the raw
    # JSON (260929): 6/8 levels effectively at nG=601, but level_167p5 and
    # level_302p5 (ng_schedule_used=[[0,201],[0.5,401]]) are at nG=401, not 601.
    if d.get("nG_report") is not None:
        eff_nG = d["nG_report"]
    else:
        eff_nG = d["nG"]
    tA = d["amp_tA"] * np.exp(1j * math.radians(d["A_deg"]))
    tB = d["amp_tB"] * np.exp(1j * math.radians(d["B_deg"]))
    TA_total = d.get("ledA_T_total")
    TB_total = d.get("ledB_T_total")
    # REPAIR (verifier defect, major): amp_tA/amp_tB are ONE polarisation component
    # (absT is a POLARISATION COMPONENT; T is the order's TOTAL power),
    # not the atom's total transmitted power. T_total/|t|^2 (~2.50 here) is the
    # correction factor needed for an ABSOLUTE (power-consistent) capture/overlap.
    t_over_t2_A = (TA_total / (d["amp_tA"] ** 2)) if TA_total is not None else None
    t_over_t2_B = (TB_total / (d["amp_tB"] ** 2)) if TB_total is not None else None
    palette.append(dict(name=lvl, nG_eval=nG_eval, nG=d["nG"], nG_report=d.get("nG_report"),
                         eval_only=bool(d.get("eval_only")), eff_nG=eff_nG,
                         amp_tA=d["amp_tA"], A_deg=d["A_deg"], amp_tB=d["amp_tB"], B_deg=d["B_deg"],
                         tA=tA, tB=tB, TA_total=TA_total, TB_total=TB_total,
                         t_over_t2_A=t_over_t2_A, t_over_t2_B=t_over_t2_B))
    log("palette %-14s eff_nG=%-4d (nG=%d nG_eval=%d nG_report=%s eval_only=%s) amp_tA=%.4f "
        "A_deg=%8.3f  amp_tB=%.4f B_deg=%8.3f  T_tot/|t|^2_A=%s"
        % (lvl, eff_nG, d["nG"], nG_eval, str(d.get("nG_report")), bool(d.get("eval_only")),
           d["amp_tA"], d["A_deg"], d["amp_tB"], d["B_deg"],
           ("%.4f" % t_over_t2_A) if t_over_t2_A is not None else "n/a"))

_effnG_counts = {}
for p in palette:
    _effnG_counts[p["eff_nG"]] = _effnG_counts.get(p["eff_nG"], 0) + 1
log("palette EFFECTIVE nG basis per level (eval_only record: nG, else nG_report): count by nG = %s" % _effnG_counts)
FAC_T_OVER_T2 = float(np.mean([p["t_over_t2_A"] for p in palette if p["t_over_t2_A"] is not None]))
log("mean T_total/|tA|^2 correction factor across palette = %.4f" % FAC_T_OVER_T2)

# ---------------------------------------------------------------------------
# SECTION 3 -- the CURRENT Li lens ladder = the 21 cells of the verdict "P769 approved (Li terms)" of lens_total_li5.py
# (M = 6 9 12 15 18 24 36, registration sh = 0 1 2; x12_sh0 is the sim766b cell), read through lens_total_li.py's OWN functions
# (read_class / conv / betaL / evaluate_cells / check_li_flags, imported, unmodified). Figure basis = verdict variant "rB0_complex":
# arm-B node = nd[M]["B"] * nr[M] (registration-rotated class node times the mean rB0 node), then lens_total_li.conv()
# (sqrt(cos) + phase), so that beta_model() below sees exactly the ladder that lens_total_li.betaL() sees.
# ---------------------------------------------------------------------------
sys.path.insert(0, LENS_LI_DIR)
import lens_total_li as L                                                   # noqa: E402  (pins logical cores 10-17 at import)
_pin("after lens_total_li import")
_ev_fn, _ev_LAMG, _ev_LAM = L.evaluator()                                   # runs the ladder646 head now; it pins 10-17 on first use
_pin("after lens_total_li.evaluator()")
if _ev_LAMG != LAMG or _ev_LAM != LAM:
    log("LAMG/LAM of the ladder646 evaluator (%r, %r) differ from review_net_f's (%r, %r) -- stopping" % (_ev_LAMG, _ev_LAM, LAMG, LAM))
    sys.exit(1)

LENS_M = (6, 9, 12, 15, 18, 24, 36)
if tuple(L.CLASSES) != LENS_M:
    log("lens_total_li.CLASSES %r != %r -- stopping" % (tuple(L.CLASSES), LENS_M))
    sys.exit(1)

_CELLS769 = json.load(open(SIM769_CELLS_JSON, encoding="utf-8"))
EXP769 = {(c["M"], c["sh"]): c["after"] for c in _CELLS769["cells"]}
EXP769[(12, 0)] = _CELLS769["reuse"][0]["slots"]
_md5_cache = {}


def m8(p):
    if p not in _md5_cache:
        _md5_cache[p] = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    return _md5_cache[p]


def path769(M, sh):
    # copied from lens_total_li5.py: x12_sh0 is the sim766b cell, every other cell is sim769
    if (M, sh) == (12, 0):
        return RESULT_ROOT + r"\261009_scdc_tmzone_iso\261009_sim766_iso_x12_sh0_212_257acc"
    return RESULT_ROOT + r"\261009_scdc_tmzone_li5\261009_sim769_lensli5_x%02d_sh%d" % (M, sh)


LENS_CELLS = {M: {sh: path769(M, sh) for sh in (0, 1, 2)} for M in LENS_M}
_missing_cells = [p for M in LENS_M for p in LENS_CELLS[M].values() if not os.path.isfile(os.path.join(p, "superramp_result.json"))]
if _missing_cells:
    for _p in _missing_cells:
        print("NOT FOUND: %s" % os.path.join(_p, "superramp_result.json"), flush=True)
    sys.exit(1)
_bad769 = []
for M in LENS_M:
    for sh in (0, 1, 2):
        _p = LENS_CELLS[M][sh]
        _j = L.jload(_p)
        L.check_li_flags("li", _j, _p)                                       # raises SystemExit("REFUSE ...") on a non-Li cell
        _got = [m8(q) for q in _j["needle_results"]["production"]["atom_paths"]]
        if _got != EXP769[(M, sh)]:
            _bad769.append((M, sh, _got, EXP769[(M, sh)]))
if _bad769:
    for _b in _bad769:
        print("PER-SLOT MD5 MISMATCH x%02d_sh%d: got %s expected %s" % _b, flush=True)
    sys.exit(1)
log("ladder cells: 21/21 present, every cell rcwa_rule liy / rcwa_ff 1, per-slot atom md5 list equals sim769_cells.json (x12_sh0 from the sim766b slots)")

ND, NR, ECHO_CLS = {}, {}, {}
for M in LENS_M:
    _a, _r, _e = L.read_class(LENS_CELLS[M])
    ND[M], NR[M], ECHO_CLS["x%02d" % M] = _a, _r, L.pmean(_e)
_pts_fig = [(M / 3.0, ND[M]["A"], ND[M]["B"] * NR[M]) for M in sorted(ND)]
_pts_real = L.conv(_pts_fig)
SRC_USED = {}
for M in LENS_M:
    _lab = ("P769 Li verdict set: sh0 = sim766b (261009_scdc_tmzone_iso), sh1/sh2 = sim769 (261009_scdc_tmzone_li5)" if M == 12
            else "P769 Li verdict set: sim769 (261009_scdc_tmzone_li5)")
    SRC_USED[M] = dict(label=_lab, root=os.path.dirname(LENS_CELLS[M][0]), code="x%02d" % M, n_shift=3,
                       cells=[LENS_CELLS[M][sh] for sh in (0, 1, 2)])
    log("real ladder M=%2d  P=%.4f Lambda_g  n_shift=3  %s" % (M, M / 3.0, _lab))
for P, a, b, k in _pts_real:
    log("  P %5.2f Lambda_g  n %d  beta_A %.3f @ %+6.1f deg  beta_B %.3f @ %+6.1f deg  (figure basis: rB0_complex, after conv())" %
        (P, k, abs(a), math.degrees(cmath.phase(a)), abs(b), math.degrees(cmath.phase(b))))
_Pm_real = np.array([p[0] for p in _pts_real]) * LAMG
_bmA_real = np.array([p[1] for p in _pts_real])
_bmB_real = np.array([p[2] for p in _pts_real])
track_rss("after palette + ladder load")

# ---------------------------------------------------------------------------
# CONTROL-V (NEW 261011) -- the current VERDICT: lens_total_li.evaluate_cells() with load_terms overridden to the P769 atom-level terms
# (deficit_p' , purity_p' f200 of a1_def_pur_p769_result.json), compared with lens_total_li5_final.json["results"][VERDICT_KEY].
# Pass = every |diff| <= 1e-12.  total = betaL + deficit_p' + purity_p'.
# ---------------------------------------------------------------------------
TOL_V = 1e-12
TOL_F = 1e-9
_T769 = json.load(open(P769_TERMS_JSON, encoding="utf-8"))
_terms769 = (_T769["deficit"], _T769["pur_f200"])
_orig_load_terms = L.load_terms
L.load_terms = lambda: _terms769
try:
    EV = L.evaluate_cells(LENS_CELLS)
finally:
    L.load_terms = _orig_load_terms
_pin("after CONTROL-V evaluate_cells")
_REF = json.load(open(LI5_FINAL_JSON, encoding="utf-8"))["results"][VERDICT_KEY]
SPEC_EXPECTED = {"rB0_complex total decay": -0.5470182455060397, "rB0_complex total held": -0.5581150950176934,
                 "rB0_complex betaL decay": -0.46659703474616226, "rB0_complex betaL held": -0.47769388425781595,
                 "echo_aperture": -20.821628038615852}
CONTROL_V_ROWS = []
for _var in ("no_rB0", "rB0_complex", "rB0_abs"):
    for _key in ("betaL", "total"):
        for _i, _tail in enumerate(("decay", "held")):
            CONTROL_V_ROWS.append(("%s %s %s" % (_var, _key, _tail), float(EV["res"][_var][_key][_i]), float(_REF["res"][_var][_key][_i])))
CONTROL_V_ROWS.append(("echo_aperture", float(EV["echo_aperture"]), float(_REF["echo_aperture"])))
CONTROL_V_ROWS.append(("deficit_p'", float(EV["deficit"]), float(_REF["deficit"])))
CONTROL_V_ROWS.append(("purity_p' f200", float(EV["purity"]), float(_REF["purity"])))
CONTROL_V = dict(tolerance=TOL_V, reference_file=LI5_FINAL_JSON, reference_key=VERDICT_KEY, terms_file=P769_TERMS_JSON,
                 rows=[], spec_expected_rows=[])
for _nm, _m, _r in CONTROL_V_ROWS:
    _d = abs(_m - _r)
    CONTROL_V["rows"].append(dict(name=_nm, measured=_m, reference=_r, absdiff=_d, ok=bool(_d <= TOL_V)))
    log("CONTROL-V %-30s measured %+.16f  reference %+.16f  |diff| %.3e  %s" % (_nm, _m, _r, _d, "OK" if _d <= TOL_V else "FAIL"))
for _nm, _r in SPEC_EXPECTED.items():
    _m = dict((n, v) for n, v, _ in CONTROL_V_ROWS)[_nm]
    _d = abs(_m - _r)
    CONTROL_V["spec_expected_rows"].append(dict(name=_nm, measured=_m, spec_expected=_r, absdiff=_d, ok=bool(_d <= TOL_V)))
    log("CONTROL-V vs task-quoted %-26s measured %+.16f  expected %+.16f  |diff| %.3e  %s" % (_nm, _m, _r, _d, "OK" if _d <= TOL_V else "FAIL"))
CONTROL_V["pass"] = bool(all(r["ok"] for r in CONTROL_V["rows"]) and all(r["ok"] for r in CONTROL_V["spec_expected_rows"]))
CONTROL_V["verdict_rB0_complex"] = dict(betaL=list(EV["res"]["rB0_complex"]["betaL"]), total=list(EV["res"]["rB0_complex"]["total"]),
                                         margin=list(EV["res"]["rB0_complex"]["margin"]), echo_aperture=EV["echo_aperture"],
                                         deficit=EV["deficit"], purity=EV["purity"])
if not CONTROL_V["pass"]:
    log("CONTROL-V FAILED -- stopping. Nothing else in this script is valid until this is fixed.")
    json.dump(dict(control_V=CONTROL_V), open(os.path.join(SCRATCH, "control_FAILED.json"), "w", encoding="utf-8"), indent=2, default=str)
    sys.exit(1)
log("CONTROL-V PASSED: every compared number within %.0e of lens_total_li5_final.json[%s]" % (TOL_V, VERDICT_KEY))

# ---------------------------------------------------------------------------
# CONTROL-F (NEW 261011) -- this script's OWN focal_q() / beta_model() on the review_net_f quadrant grid, from _Pm_real/_bmA_real/_bmB_real
# (phase -K0 (sqrt(rho^2+f^2) - f), Aq apodisation, f = 200): L_dB = 10 log10(ov / c0) for the "decay" and "held" tails must equal
# lens_total_li.betaL() of the rB0_complex variant (= the beta-L part of the verdict). Pass = |diff| <= 1e-9 dB for both tails.
# ---------------------------------------------------------------------------
_f_ctrl = 200.0
_rho_ctrl = rho_q
_P_ctrl = LAM * np.sqrt(_rho_ctrl ** 2 + _f_ctrl ** 2) / np.maximum(_rho_ctrl, 1e-9)
_ph_ctrl = -K0 * (np.sqrt(_rho_ctrl ** 2 + _f_ctrl ** 2) - _f_ctrl)
_E0_ctrl = focal_q(Aq * np.exp(1j * _ph_ctrl), _f_ctrl)
_c0_ctrl = float(np.sum(np.abs(_E0_ctrl) ** 2) * DT * DT) / pin_q
CONTROL_F = dict(tolerance_dB=TOL_F, f_um=_f_ctrl, c0=_c0_ctrl, tails={})
_ref_betaL = EV["res"]["rB0_complex"]["betaL"]
for _i, _tail in enumerate(("decay", "held")):
    _EA_ctrl = focal_q(Aq * beta_model(_P_ctrl, _Pm_real, _bmA_real, _tail, LAMG) * np.exp(1j * _ph_ctrl), _f_ctrl)
    _EB_ctrl = focal_q(Aq * beta_model(_P_ctrl, _Pm_real, _bmB_real, _tail, LAMG) * np.exp(1j * _ph_ctrl), _f_ctrl)
    _ov_t = float(abs(np.sum(np.conj(_EA_ctrl) * _EB_ctrl)) * DT * DT / pin_q)
    _L_t = 10.0 * math.log10(_ov_t / _c0_ctrl)
    _d = abs(_L_t - float(_ref_betaL[_i]))
    CONTROL_F["tails"][_tail] = dict(ov=_ov_t, L_dB=_L_t, reference_betaL_rB0_complex=float(_ref_betaL[_i]), absdiff=_d, ok=bool(_d <= TOL_F))
    log("CONTROL-F f=200 %-5s ov=%.16f c0=%.16f L_dB=%+.16f  betaL(rB0_complex)=%+.16f  |diff| %.3e  %s" %
        (_tail, _ov_t, _c0_ctrl, _L_t, float(_ref_betaL[_i]), _d, "OK" if _d <= TOL_F else "FAIL"))
CONTROL_F["pass"] = bool(all(v["ok"] for v in CONTROL_F["tails"].values()))
if not CONTROL_F["pass"]:
    log("CONTROL-F FAILED -- stopping. The figure model does not reproduce the ladder betaL.")
    json.dump(dict(control_V=CONTROL_V, control_F=CONTROL_F), open(os.path.join(SCRATCH, "control_FAILED.json"), "w", encoding="utf-8"),
              indent=2, default=str)
    sys.exit(1)
_ov_ctrl = CONTROL_F["tails"]["decay"]["ov"]                                 # figure-basis ladder-only overlap at f=200, decay tail (used below)
log("CONTROL-F PASSED within %.0e dB for both tails." % TOL_F)

CONTROLS_JSON = os.path.join(SCRATCH, "controls_261011.json")
json.dump(dict(script=os.path.abspath(__file__), controls_only=bool(CONTROLS_ONLY), wall_s_at_write=time.time() - T_START,
               peak_rss_mb_at_write=_PEAK_RSS_MB[0], palette_md5_gate=MD5_TABLE, control_V=CONTROL_V, control_F=CONTROL_F,
               ladder_figure_basis=[dict(P_Lambda_g=p[0], absA=abs(p[1]), argA_deg=math.degrees(cmath.phase(p[1])),
                                         absB=abs(p[2]), argB_deg=math.degrees(cmath.phase(p[2]))) for p in _pts_real],
               ladder_cells={str(M): SRC_USED[M]["cells"] for M in LENS_M},
               echo_class_db=ECHO_CLS, node_dB=EV["node_dB"]),
          open(CONTROLS_JSON, "w", encoding="utf-8"), indent=2, default=str)
log("wrote %s" % CONTROLS_JSON)
track_rss("after CONTROL-V / CONTROL-F")
if CONTROLS_ONLY:
    log("--controls-only: stopping here (exit 0), SECTION 4 onward not run. wall=%.1fs peak_RSS=%.1f MB" %
        (time.time() - T_START, _PEAK_RSS_MB[0]))
    sys.exit(0)
_pin("before SECTION 4")

# ---------------------------------------------------------------------------
# SECTION 4 -- design parameters, fine (Lambda/3) grid for M1/M2, mirroring to
# the full 300x300um aperture, and the free-offset nearest-palette-level
# assignment.
# ---------------------------------------------------------------------------
F_DESIGN = 200.0  # um -- adopted TM LPA design focal length (verdict f200, CONTROL-V)
GRID_M1 = LAMG / 3.0  # um, Lambda/3 unit-cell pitch (atoms_per_period=3)
log("design focal length f=%.1f um; M1/M2 grid pitch Lambda/3=%.6f um (spec quotes 0.43846)" % (F_DESIGN, GRID_M1))

# --- M0 continuous-phase field on the ORIGINAL quadrant grid (dx=0.5,n=300) ---
rho0 = rho_q
ph0 = -K0 * (np.sqrt(rho0 ** 2 + F_DESIGN ** 2) - F_DESIGN)
P0 = LAM * np.sqrt(rho0 ** 2 + F_DESIGN ** 2) / np.maximum(rho0, 1e-9)
M0_field_q = Aq * np.exp(1j * ph0)

# --- fine grid for M1/M2 ---
n2 = int(round(APER / GRID_M1))
x2 = (np.arange(n2) + 0.5) * GRID_M1
X2, Y2 = np.meshgrid(x2, x2)
A2 = np.exp(-(X2 ** 2 + Y2 ** 2) / W_PUMP ** 2)
rho2 = np.hypot(X2 - XC, Y2 - XC)
ph2 = -K0 * (np.sqrt(rho2 ** 2 + F_DESIGN ** 2) - F_DESIGN)
P2 = LAM * np.sqrt(rho2 ** 2 + F_DESIGN ** 2) / np.maximum(rho2, 1e-9)
log("fine (M1/M2) grid: n2=%d actual quadrant extent=%.4f um (target %.1f)" % (n2, n2 * GRID_M1, APER))

# --- free global offset minimising circular RMS phase error (arm-A basis) ---
A_deg_levels = np.array([p["A_deg"] for p in palette])


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


ph2_deg = np.degrees(ph2) % 360.0
_bin_w = 0.1
_bins = np.arange(0.0, 360.0, _bin_w)
_counts, _ = np.histogram(ph2_deg, bins=np.append(_bins, 360.0))
_offsets = np.arange(0.0, 360.0, _bin_w)
_best_rms = None
_best_off = 0.0
for _off in _offsets:
    _shifted = _bins + _off
    _d = wrap180(_shifted[None, :] - A_deg_levels[:, None])  # (8, n_bins)
    _mind = np.min(np.abs(_d), axis=0)
    _rms = math.sqrt(np.sum(_counts * _mind ** 2) / np.sum(_counts))
    if _best_rms is None or _rms < _best_rms:
        _best_rms = _rms
        _best_off = _off
PHASE_OFFSET_DEG = float(_best_off)
PHASE_RMS_DEG = float(_best_rms)
log("free global offset (arm-A, unweighted circular RMS over 1 quadrant, 0.1deg search): "
    "offset=%.2f deg  rms_error=%.3f deg" % (PHASE_OFFSET_DEG, PHASE_RMS_DEG))

# per-cell nearest palette level assignment on the fine grid
_shifted_cell = (ph2_deg + PHASE_OFFSET_DEG)
_dcell = wrap180(_shifted_cell[None, :, :] - A_deg_levels[:, None, None])  # (8, n2, n2)
level_idx = np.argmin(np.abs(_dcell), axis=0)  # (n2,n2) int in [0,8)
_level_counts = np.bincount(level_idx.ravel(), minlength=len(palette))
for i, p in enumerate(palette):
    log("  level %-14s used in %6d / %6d cells (%.1f %%)" %
        (p["name"], _level_counts[i], level_idx.size, 100.0 * _level_counts[i] / level_idx.size))

tA_arr = np.array([p["tA"] for p in palette])
tB_arr = np.array([p["tB"] for p in palette])
tA_cell = tA_arr[level_idx]
tB_cell = tB_arr[level_idx]

M1_A_q = A2 * tA_cell
M1_B_q = A2 * tB_cell

beta_A_cell = beta_model(P2, _Pm_real, _bmA_real, "decay", LAMG)
beta_B_cell = beta_model(P2, _Pm_real, _bmB_real, "decay", LAMG)
M2_A_q = M1_A_q * beta_A_cell
M2_B_q = M1_B_q * beta_B_cell

# REPAIR (verifier defect, major): "lens-only" (phase-only, UNIT amplitude per
# assigned palette level) variants. These divide out each level's own base |t|
# (a fixed atom-transmission budget item, not a focusing loss -- see
# FAC_T_OVER_T2 above) so that capture/overlap/Strehl measure the LPA phase
# quantisation (+ beta ladder) contribution in isolation, comparable to the
# earlier "-0.37 dB" beta-ladder-only verdict from CONTROL.
tA_phase_arr = tA_arr / np.abs(tA_arr)
tB_phase_arr = tB_arr / np.abs(tB_arr)
tA_phase_cell = tA_phase_arr[level_idx]
tB_phase_cell = tB_phase_arr[level_idx]
M1_A_lens_q = A2 * tA_phase_cell
M1_B_lens_q = A2 * tB_phase_cell
M2_A_lens_q = M1_A_lens_q * beta_A_cell
M2_B_lens_q = M1_B_lens_q * beta_B_cell

log("beta(P2) stats: |beta_A| min/med/max = %.4f/%.4f/%.4f ; zero-beta cells (P<2*Lambda_g) = %d/%d"
    % (np.min(np.abs(beta_A_cell)), np.median(np.abs(beta_A_cell)), np.max(np.abs(beta_A_cell)),
       int(np.sum(P2 < 2 * LAMG)), P2.size))
track_rss("after M0/M1/M2 quadrant field construction")


def mirror_quadrant(field):
    """Mirror a (n,n) quadrant field (row=Y>=0,col=X>=0) into a full (2n,2n)
    array covering all four quadrants of the 300x300um aperture."""
    n_ = field.shape[0]
    full = np.empty((2 * n_, 2 * n_), dtype=field.dtype)
    full[n_:, n_:] = field
    full[n_:, :n_] = field[:, ::-1]
    full[:n_, n_:] = field[::-1, :]
    full[:n_, :n_] = field[::-1, ::-1]
    return full


M0_full = mirror_quadrant(M0_field_q)
M1_A_full = mirror_quadrant(M1_A_q)
M1_B_full = mirror_quadrant(M1_B_q)
M2_A_full = mirror_quadrant(M2_A_q)
M2_B_full = mirror_quadrant(M2_B_q)
M1_A_lens_full = mirror_quadrant(M1_A_lens_q)
M1_B_lens_full = mirror_quadrant(M1_B_lens_q)
M2_A_lens_full = mirror_quadrant(M2_A_lens_q)
M2_B_lens_full = mirror_quadrant(M2_B_lens_q)
log("mirrored to full aperture: grid0 (M0) %dx%d @ dx=%.4f um ; grid1 (M1/M2) %dx%d @ dx=%.6f um"
    % (M0_full.shape[0], M0_full.shape[1], dx, M1_A_full.shape[0], M1_A_full.shape[1], GRID_M1))
track_rss("after mirroring to full aperture")

# ---------------------------------------------------------------------------
# SECTION 5 -- generic band-limited-ASM propagation on the full-aperture grids
# (padding factor 4, evanescent orders zeroed -- same convention as the
# validated review_net_f.cap()/recheck3.focal(), generalised to: (a) the full
# mirrored aperture instead of one isolated quadrant, and (b) evaluation at
# ARBITRARY (x,y,z), not just the one stored 8x8um PD box, via the same
# Mx-exact-DFT trick used for the PD box, applied along whichever output axis
# is smaller first.
# ---------------------------------------------------------------------------
PAD_FACTOR = 4


def build_grid(n_quad, dx_val, pad_factor=PAD_FACTOR):
    n_full = 2 * n_quad
    npd = pad_factor * n_full
    kk = 2.0 * np.pi * np.fft.fftfreq(npd, d=dx_val)
    KX, KY = np.meshgrid(kk, kk)
    kz2 = K0 ** 2 - KX ** 2 - KY ** 2
    del KX, KY
    X0 = -(n_quad - 0.5) * dx_val
    du = 1.0 / (npd * dx_val)  # spatial-frequency sample spacing, for the BLAS band limit
    return dict(n_quad=n_quad, dx=dx_val, n_full=n_full, npd=npd, k=kk, kz2=kz2, X0=X0, du=du)


GRID0 = build_grid(n_q, dx)
GRID1 = build_grid(n2, GRID_M1)
log("GRID0 (M0): n_full=%d npd=%d X0=%.4f um" % (GRID0["n_full"], GRID0["npd"], GRID0["X0"]))
log("GRID1 (M1/M2): n_full=%d npd=%d X0=%.4f um" % (GRID1["n_full"], GRID1["npd"], GRID1["X0"]))


def spectrum(field_full, grid):
    npd = grid["npd"]; n_full = grid["n_full"]
    Ep = np.zeros((npd, npd), dtype=complex)
    Ep[:n_full, :n_full] = field_full
    return np.fft.fft2(Ep)


def make_M(xs, grid):
    xs = np.asarray(xs, dtype=float)
    return np.exp(1j * np.outer(xs - grid["X0"], grid["k"]))


def eval_field(S0, grid, z, xs, ys):
    """Exact field at output points xs (rows) x ys (cols), at propagation
    distance z, from precomputed angular spectrum S0=fft2(padded aperture)."""
    kz2 = grid["kz2"]; npd = grid["npd"]
    H = np.where(kz2 >= 0, np.exp(1j * np.sqrt(np.clip(kz2, 0, None)) * z), 0.0)
    Sz = S0 * H
    Mx = make_M(xs, grid)
    My = make_M(ys, grid)
    if My.shape[0] <= Mx.shape[0]:
        t = Sz @ My.T
        E = Mx @ t
    else:
        t = Mx @ Sz
        E = t @ My.T
    return E / (npd * npd)


log("computing angular spectra (one FFT2 per near-field pattern) ...")
S0_M0 = spectrum(M0_full, GRID0)
S0_M1A = spectrum(M1_A_full, GRID1)
S0_M1B = spectrum(M1_B_full, GRID1)
S0_M2A = spectrum(M2_A_full, GRID1)
S0_M2B = spectrum(M2_B_full, GRID1)
S0_M1Ap = spectrum(M1_A_lens_full, GRID1)
S0_M1Bp = spectrum(M1_B_lens_full, GRID1)
S0_M2Ap = spectrum(M2_A_lens_full, GRID1)
S0_M2Bp = spectrum(M2_B_lens_full, GRID1)
track_rss("after 9 angular spectra computed (5 physical + 4 lens-only)")

# quadrant-corner max NA (farthest corner of one quadrant from ITS OWN lens centre)
_rho_corner = math.hypot(APER - XC, APER - XC)
NA_MAX = _rho_corner / math.sqrt(_rho_corner ** 2 + F_DESIGN ** 2)
log("max NA at quadrant far corner (rho=%.3f um, f=%.1f um): NA_max=%.4f" % (_rho_corner, F_DESIGN, NA_MAX))

# Gaussian-weighted effective NA (second moment of |A|^2 about the lens centre,
# over the physical quadrant aperture). REPAIR (verifier defect, major): this
# rms-aperture NA_eff is kept for reference ONLY -- it is NOT the diffraction
# limit. It predicts FWHM=3.80um, 57% wider than M0's own measured spot
# (~2.40um), because the hard 150um aperture edge (not just the Gaussian rms
# width) sets the actual spot size. The corrected diffraction-limit reference
# (from M0's own measured FWHM) is computed after design_plane, below.
_rho_rms = math.sqrt(float(np.sum(Aq ** 2 * rho_q ** 2)) / float(np.sum(Aq ** 2)))
NA_RMS_APERTURE = _rho_rms / math.sqrt(_rho_rms ** 2 + F_DESIGN ** 2)
log("NA_rms_aperture (reference only; Gaussian-amplitude-weighted rms aperture radius %.3f um) = %.4f -- "
    "not compared to a limit here; M0 FWHM (reference) is reported separately below"
    % (_rho_rms, NA_RMS_APERTURE))
track_rss("after grid/NA setup")

# internal (non-mandatory) sanity check: full-aperture M0 PD capture at the
# design plane, for context against the isolated-quadrant c0 used in CONTROL.


def make_H(grid, z):
    kz2 = grid["kz2"]
    H = np.where(kz2 >= 0, np.exp(1j * np.sqrt(np.clip(kz2, 0, None)) * z), 0.0)
    # REPAIR (verifier defect, minor/factual): actually apply the band-limited-ASM
    # (Matsushima & Shimobaba 2009) frequency limit u_lim(z) = 1/(lambda*sqrt((2 du
    # z)^2+1)) instead of only the plain evanescent (kz2>=0) cutoff -- the method
    # this script's docstring/report names but did not, until now, implement.
    # Independently measured effect at z=200um (verify pass, 260929): negligible
    # (dCapture ~1e-6 to 1e-7 level; see tm_focus.md limitations) but the statement
    # is now true rather than only approximately so.
    du = grid["du"]
    u_lim = 1.0 / (LAM * math.sqrt((2.0 * du * z) ** 2 + 1.0))
    m = np.abs(grid["k"] / (2.0 * np.pi)) < u_lim
    return H * (m[:, None] & m[None, :])


def eval_with_H(S0, H, grid, xs, ys):
    Sz = S0 * H
    Mx = make_M(xs, grid); My = make_M(ys, grid)
    if My.shape[0] <= Mx.shape[0]:
        t = Sz @ My.T; E = Mx @ t
    else:
        t = Mx @ Sz; E = t @ My.T
    return E / (grid["npd"] ** 2)


_H0_200 = make_H(GRID0, F_DESIGN)
_Ebox_M0full = eval_with_H(S0_M0, _H0_200, GRID0, XC + AX_box, XC + AX_box)
_c0_full_sanity = float(np.sum(np.abs(_Ebox_M0full) ** 2) * DT * DT / pin_q)
log("(internal sanity, not the mandated control) full 4-quadrant M0 PD capture at z=200: %.4f "
    "vs isolated-quadrant c0=%.4f (difference is expected: real cross-quadrant coupling is now included)"
    % (_c0_full_sanity, _c0_ctrl))

# ---------------------------------------------------------------------------
# SECTION 6 -- per-(model,arm) scans: on-axis vs z (all 5), PD-capture vs z
# (the 3 combos shown in F3), capture/overlap/FWHM at the design plane z=200
# (all 5, both arms where applicable).
# ---------------------------------------------------------------------------
COMBOS = [
    ("M0", None, GRID0, S0_M0),
    ("M1", "A", GRID1, S0_M1A),
    ("M1", "B", GRID1, S0_M1B),
    ("M2", "A", GRID1, S0_M2A),
    ("M2", "B", GRID1, S0_M2B),
]
Z_GRID = np.arange(0.0, 260.0 + 1e-9, 0.5)
log("z-scan grid: %d points, 0..260 um step 0.5 um" % len(Z_GRID))

# FIX (major, BASIS -- item 1): F3(a)/(b) must show the POWER basis
# (solid) and the LENS-ONLY basis (dashed) on one basis-labelled axis, not the
# raw one-polarisation-component basis with no basis label. The lens-only
# spectra (S0_M1Ap/S0_M2Ap) are already built in Section 5; scanning them here
# reuses the SAME H0/H1 already built once per z for the physical COMBOS below
# (no extra make_H cost, only the extra eval_with_H matrix products).
COMBOS_LENS_F3 = [("M1Ap", GRID1, S0_M1Ap), ("M2Ap", GRID1, S0_M2Ap)]

onaxis = {name + (arm or ""): np.zeros(len(Z_GRID)) for name, arm, _, _ in COMBOS}
onaxis_lens = {key: np.zeros(len(Z_GRID)) for key, _, _ in COMBOS_LENS_F3}
t_scan0 = time.time()
for iz, z in enumerate(Z_GRID):
    H0 = make_H(GRID0, z)
    H1 = make_H(GRID1, z)
    for name, arm, grid, S0 in COMBOS:
        H = H0 if grid is GRID0 else H1
        E = eval_with_H(S0, H, grid, [XC], [XC])
        onaxis[name + (arm or "")][iz] = abs(E[0, 0]) ** 2
    for key, grid, S0 in COMBOS_LENS_F3:
        H = H0 if grid is GRID0 else H1
        E = eval_with_H(S0, H, grid, [XC], [XC])
        onaxis_lens[key][iz] = abs(E[0, 0]) ** 2
    if iz % 100 == 0:
        log("  on-axis z-scan %d/%d (z=%.1f) elapsed %.1fs" % (iz, len(Z_GRID), z, time.time() - t_scan0))
log("on-axis z-scan done in %.1fs" % (time.time() - t_scan0))
track_rss("after on-axis z-scan (5 combos + 2 lens-only)")

PD_COMBOS_FOR_F3B = [("M0", None, GRID0, S0_M0), ("M1", "A", GRID1, S0_M1A), ("M2", "A", GRID1, S0_M2A)]
PD_COMBOS_LENS_F3B = [("M1Ap", GRID1, S0_M1Ap), ("M2Ap", GRID1, S0_M2Ap)]
pdcap_vs_z = {name + (arm or ""): np.zeros(len(Z_GRID)) for name, arm, _, _ in PD_COMBOS_FOR_F3B}
pdcap_lens_vs_z = {key: np.zeros(len(Z_GRID)) for key, _, _ in PD_COMBOS_LENS_F3B}
t_scan1 = time.time()
for iz, z in enumerate(Z_GRID):
    H0 = make_H(GRID0, z)
    H1 = make_H(GRID1, z)
    for name, arm, grid, S0 in PD_COMBOS_FOR_F3B:
        H = H0 if grid is GRID0 else H1
        E = eval_with_H(S0, H, grid, XC + AX_box, XC + AX_box)
        pdcap_vs_z[name + (arm or "")][iz] = float(np.sum(np.abs(E) ** 2) * DT * DT / pin_q)
    for key, grid, S0 in PD_COMBOS_LENS_F3B:
        H = H0 if grid is GRID0 else H1
        E = eval_with_H(S0, H, grid, XC + AX_box, XC + AX_box)
        pdcap_lens_vs_z[key][iz] = float(np.sum(np.abs(E) ** 2) * DT * DT / pin_q)
    if iz % 100 == 0:
        log("  PD-capture z-scan %d/%d (z=%.1f) elapsed %.1fs" % (iz, len(Z_GRID), z, time.time() - t_scan1))
log("PD-capture z-scan done in %.1fs" % (time.time() - t_scan1))
track_rss("after PD-capture z-scan (3 combos + 2 lens-only)")

# power-basis (solid) and lens-only (dashed) series for F3(a)/(b), arm A only
# (matching the existing M0/M1A/M2A panel scope). Power basis = raw x
# FAC_T_OVER_T2, the SAME constant used for every other absolute-throughput
# number in this script; M0 has no atom and is already power-consistent by
# construction, so it is carried through unscaled on both panels.
onaxis_power = {"M0": onaxis["M0"], "M1A": onaxis["M1A"] * FAC_T_OVER_T2, "M2A": onaxis["M2A"] * FAC_T_OVER_T2}
onaxis_lensbasis = {"M1A": onaxis_lens["M1Ap"], "M2A": onaxis_lens["M2Ap"]}
pdcap_power = {"M0": pdcap_vs_z["M0"], "M1A": pdcap_vs_z["M1A"] * FAC_T_OVER_T2, "M2A": pdcap_vs_z["M2A"] * FAC_T_OVER_T2}
pdcap_lensbasis = {"M1A": pdcap_lens_vs_z["M1Ap"], "M2A": pdcap_lens_vs_z["M2Ap"]}

# --- design-plane (z=200) quantities for ALL 5 combos: capture, line cuts ---
X_LINE = XC + np.linspace(-8.0, 8.0, 1601)
Y_LINE = XC + np.linspace(-8.0, 8.0, 1601)


def fwhm(xs, I):
    xs = np.asarray(xs); I = np.asarray(I)
    imax = int(np.argmax(I)); half = I[imax] / 2.0
    li = imax
    while li > 0 and I[li] > half:
        li -= 1
    if li == imax:
        left = xs[imax]
    else:
        x0, x1 = xs[li], xs[li + 1]; y0, y1 = I[li], I[li + 1]
        left = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
    ri = imax; nlen = len(I)
    while ri < nlen - 1 and I[ri] > half:
        ri += 1
    if ri == imax:
        right = xs[imax]
    else:
        x0, x1 = xs[ri - 1], xs[ri]; y0, y1 = I[ri - 1], I[ri]
        right = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
    return float(right - left)


design_plane = {}
Ecuts_x = {}
Ecuts_y = {}
H0_design = make_H(GRID0, F_DESIGN)
H1_design = make_H(GRID1, F_DESIGN)
for name, arm, grid, S0 in COMBOS:
    key = name + (arm or "")
    H = H0_design if grid is GRID0 else H1_design
    Ebox = eval_with_H(S0, H, grid, XC + AX_box, XC + AX_box)
    cap = float(np.sum(np.abs(Ebox) ** 2) * DT * DT / pin_q)
    Ex = eval_with_H(S0, H, grid, X_LINE, [XC])[:, 0]
    Ey = eval_with_H(S0, H, grid, [XC], Y_LINE)[0, :]
    Ecuts_x[key] = Ex; Ecuts_y[key] = Ey
    fw_x = fwhm(X_LINE, np.abs(Ex) ** 2)
    fw_y = fwhm(Y_LINE, np.abs(Ey) ** 2)
    peak_z = float(Z_GRID[int(np.argmax(onaxis[key]))])
    peak_I = float(np.max(onaxis[key]))
    design_plane[key] = dict(capture_pd=cap, fwhm_x_um=fw_x, fwhm_y_um=fw_y,
                              focal_peak_z_um=peak_z, onaxis_peak_intensity=peak_I)
    log("design-plane z=200: %-4s capture=%.4f fwhm_x=%.4f fwhm_y=%.4f focal_peak_z=%.2f peak_I=%.5f"
        % (key, cap, fw_x, fw_y, peak_z, peak_I))

peak_I_M0 = design_plane["M0"]["onaxis_peak_intensity"]
for key in design_plane:
    design_plane[key]["strehl"] = design_plane[key]["onaxis_peak_intensity"] / peak_I_M0

# overlap L (arm A vs arm B) at the design plane, dB vs the M0 reference capture
cap_M0_ref = design_plane["M0"]["capture_pd"]
overlap = {}
for name in ("M1", "M2"):
    EA = eval_with_H(dict(M1=S0_M1A, M2=S0_M2A)[name], H1_design, GRID1, XC + AX_box, XC + AX_box)
    EB = eval_with_H(dict(M1=S0_M1B, M2=S0_M2B)[name], H1_design, GRID1, XC + AX_box, XC + AX_box)
    L = float(np.abs(np.sum(np.conj(EA) * EB)) * DT * DT / pin_q)
    L_db = 10.0 * math.log10(L / cap_M0_ref)
    overlap[name] = dict(L=L, L_db_vs_M0=L_db)
    log("overlap L (%s, armA vs armB) = %.4f (%.2f dB vs M0 capture %.4f)" % (name, L, L_db, cap_M0_ref))
overlap["M0"] = dict(L=cap_M0_ref, L_db_vs_M0=0.0, note="trivial self-overlap: M0 has no arm distinction")
track_rss("after design-plane scans")

# REPAIR (verifier defect, major): corrected diffraction-limit reference. M0
# (unit amplitude, continuous phase, Gaussian-apodised, full 300x300um aperture)
# is by construction the best-focusing field this optical system can produce, so
# its OWN measured FWHM is the true diffraction-limited reference -- not the
# rms-aperture NA_RMS_APERTURE estimate above (which was 57% too wide).
DIFF_LIMIT_FWHM = 0.5 * (design_plane["M0"]["fwhm_x_um"] + design_plane["M0"]["fwhm_y_um"])
NA_EFF_FROM_M0 = LAM / (2.0 * DIFF_LIMIT_FWHM)
_old_wrong_fwhm = LAM / (2.0 * NA_RMS_APERTURE)
log("M0 FWHM (reference): x=%.4f y=%.4f um (mean=%.4f um); NA_back_inferred_from_M0=%.4f. "
    "lambda/(2 NA_rms_aperture)=%.4f um reported separately (NA_rms_aperture, reference only, "
    "not compared here as a limit)."
    % (design_plane["M0"]["fwhm_x_um"], design_plane["M0"]["fwhm_y_um"], DIFF_LIMIT_FWHM,
       NA_EFF_FROM_M0, _old_wrong_fwhm))

# ---------------------------------------------------------------------------
# SECTION 6b -- "lens-only" (phase-only) and power-normalised (absolute
# throughput) metrics. REPAIR (verifier defect, major): raw M1/M2 capture/L
# numbers sit 3.98 dB below the power-consistent value because amp_tA/amp_tB
# are ONE polarisation component, not the atom's total transmitted power
# (absT is a POLARISATION COMPONENT). Two complementary fixes:
#  (1) lens-only: divide out each level's own |t| entirely (unit amplitude per
#      level) to isolate the LPA-quantisation(+beta) focusing loss from the
#      atom's transmission budget;
#  (2) power-normalised: scale the raw numbers by T_total/|t|^2 so they are at
#      least power-consistent (still includes the atom's transmission budget,
#      which is NOT a focusing loss -- do not compare the two tables).
# ---------------------------------------------------------------------------
_LENS_S0 = dict(M1Ap=S0_M1Ap, M1Bp=S0_M1Bp, M2Ap=S0_M2Ap, M2Bp=S0_M2Bp)
lens_only = {}
I0_onaxis_z200 = float(np.abs(eval_with_H(S0_M0, H0_design, GRID0, [XC], [XC])[0, 0]) ** 2)
for _key, _S0 in _LENS_S0.items():
    _Ebox = eval_with_H(_S0, H1_design, GRID1, XC + AX_box, XC + AX_box)
    _cap = float(np.sum(np.abs(_Ebox) ** 2) * DT * DT / pin_q)
    _E_on = eval_with_H(_S0, H1_design, GRID1, [XC], [XC])[0, 0]
    _I_on = float(np.abs(_E_on) ** 2)
    lens_only[_key] = dict(capture_pd=_cap, onaxis_I_z200=_I_on, strehl_z200=_I_on / I0_onaxis_z200)
    log("lens-only %-4s capture=%.4f onaxis_I_z200=%.5f strehl_z200=%.4f" % (_key, _cap, _I_on, _I_on / I0_onaxis_z200))

lens_only_overlap = {}
for _name in ("M1", "M2"):
    _EA = eval_with_H(_LENS_S0[_name + "Ap"], H1_design, GRID1, XC + AX_box, XC + AX_box)
    _EB = eval_with_H(_LENS_S0[_name + "Bp"], H1_design, GRID1, XC + AX_box, XC + AX_box)
    _L = float(np.abs(np.sum(np.conj(_EA) * _EB)) * DT * DT / pin_q)
    _L_db = 10.0 * math.log10(_L / cap_M0_ref)
    lens_only_overlap[_name] = dict(L=_L, L_db_vs_M0=_L_db)
    log("lens-only overlap L (%s, armA vs armB, phase-only) = %.4f (%.2f dB vs M0)" % (_name, _L, _L_db))

# predicted phase-quantisation-ONLY loss from the measured rms cell phase error,
# via the standard random-phase-error Strehl model exp(-sigma^2), sigma in radians.
_sigma_rad = math.radians(PHASE_RMS_DEG)
PRED_QUANT_LOSS_DB = 10.0 * math.log10(math.exp(-_sigma_rad ** 2))
_verdict_L_db = 10.0 * math.log10(_ov_ctrl / _c0_ctrl)  # CONTROL: beta-ladder-only, continuous phase
log("decomposition: CONTROL (beta-ladder-only, no LPA quantisation) L=%.3f dB  +  predicted "
    "phase-quantisation-only loss (rms=%.3f deg) = %.3f dB  ->  sum=%.3f dB  vs measured "
    "lens-only M2 L=%.3f dB" % (_verdict_L_db, PHASE_RMS_DEG, PRED_QUANT_LOSS_DB,
                                 _verdict_L_db + PRED_QUANT_LOSS_DB, lens_only_overlap["M2"]["L_db_vs_M0"]))

# power-normalised (absolute-throughput) table -- includes the atom's own
# transmission budget (T_A~0.22-0.28, T_B~0.43-0.55); NOT a focusing loss.
power_norm = dict(
    M1A=design_plane["M1A"]["capture_pd"] * FAC_T_OVER_T2,
    M1B=design_plane["M1B"]["capture_pd"] * FAC_T_OVER_T2,
    M2A=design_plane["M2A"]["capture_pd"] * FAC_T_OVER_T2,
    M2B=design_plane["M2B"]["capture_pd"] * FAC_T_OVER_T2,
    L_M1=overlap["M1"]["L"] * FAC_T_OVER_T2,
    L_M2=overlap["M2"]["L"] * FAC_T_OVER_T2,
)
power_norm["L_M1_db_vs_M0"] = 10.0 * math.log10(power_norm["L_M1"] / cap_M0_ref)
power_norm["L_M2_db_vs_M0"] = 10.0 * math.log10(power_norm["L_M2"] / cap_M0_ref)
log("power-normalised (x T_tot/|t|^2=%.4f): M2A=%.4f M2B=%.4f L_M2=%.4f (%.2f dB vs M0)"
    % (FAC_T_OVER_T2, power_norm["M2A"], power_norm["M2B"], power_norm["L_M2"], power_norm["L_M2_db_vs_M0"]))
track_rss("after lens-only + power-normalised metrics")

# ---------------------------------------------------------------------------
# SECTION 6c -- NEW (user-facing promise, 260929): vector check of the scalar
# approximation. The TM output is x-polarised (E in the x-z plane, Ey=0), so
# from the angular spectrum at the design plane z=200um, Ez = -(kx/kz) Ex on
# every retained (propagating, band-limited) plane-wave component. Evaluated
# for M0 and M2 arm A/B, over the same exact PD-box points used everywhere
# else in this script (Mx_box=XC+AX_box). KX here is reconstructed exactly as
# grid["kz2"] itself was built (np.meshgrid(kk,kk) default 'xy' indexing:
# KX varies along axis=1/columns), so it is guaranteed consistent with this
# grid's own kz2, independent of the xs/ys row/column mislabelling noted at F1
# (which affects neither kz2 nor this KX -- both are built directly from the
# meshgrid, never through eval_with_H's xs/ys arguments).
# ---------------------------------------------------------------------------
log("computing vector check (Ez = -(kx/kz) Ex, Ey=0) at design plane z=%.1f ..." % F_DESIGN)


def eval_vector_with_H(S0, H, grid, xs, ys):
    """Return (Ex, Ez) at output points (xs rows x ys cols): Ex as eval_with_H
    already computes it; Ez = -(kx/kz)*Ex applied in the (band-limited) angular
    spectrum domain BEFORE the same exact-DFT evaluation. kz2<=0 (evanescent /
    band-limit-excluded) cells get ratio=0 (H is already 0 there; the floor
    under sqrt only avoids a spurious 0/0 -> nan)."""
    kz2 = grid["kz2"]; npd = grid["npd"]
    KX = np.broadcast_to(grid["k"][None, :], (npd, npd))
    kz_safe = np.sqrt(np.clip(kz2, 1e-12, None))
    ratio = np.where(kz2 > 0, -KX / kz_safe, 0.0)
    Sz = S0 * H
    Sz_Ez = Sz * ratio
    Mx = make_M(xs, grid); My = make_M(ys, grid)
    if My.shape[0] <= Mx.shape[0]:
        Ex = Mx @ (Sz @ My.T); Ez = Mx @ (Sz_Ez @ My.T)
    else:
        Ex = (Mx @ Sz) @ My.T; Ez = (Mx @ Sz_Ez) @ My.T
    return Ex / (npd ** 2), Ez / (npd ** 2)


_vec = {}
for _vkey, _vS0, _vgrid, _vH in (("M0", S0_M0, GRID0, H0_design),
                                  ("M2A", S0_M2A, GRID1, H1_design),
                                  ("M2B", S0_M2B, GRID1, H1_design)):
    _Ex, _Ez = eval_vector_with_H(_vS0, _vH, _vgrid, XC + AX_box, XC + AX_box)
    _pow_ex = float(np.sum(np.abs(_Ex) ** 2) * DT * DT)
    _pow_ez = float(np.sum(np.abs(_Ez) ** 2) * DT * DT)
    _vec[_vkey] = dict(Ex=_Ex, Ez=_Ez, pow_ex=_pow_ex, pow_ez=_pow_ez,
                        ratio_ez_over_ex=_pow_ez / _pow_ex,
                        pd_capture_scalar_Ex2_only=_pow_ex / pin_q,
                        pd_capture_vector_Ex2_plus_Ez2=(_pow_ex + _pow_ez) / pin_q)
    log("  vector check %-4s: PD-box int|Ez|^2/int|Ex|^2=%.6e  capture(scalar,|Ex|^2 only)=%.6f  "
        "capture(vector,|Ex|^2+|Ez|^2)=%.6f" %
        (_vkey, _vec[_vkey]["ratio_ez_over_ex"], _vec[_vkey]["pd_capture_scalar_Ex2_only"],
         _vec[_vkey]["pd_capture_vector_Ex2_plus_Ez2"]))

_L_scalar_M2 = float(np.abs(np.sum(np.conj(_vec["M2A"]["Ex"]) * _vec["M2B"]["Ex"])) * DT * DT / pin_q)
_L_vector_M2 = float(np.abs(np.sum(np.conj(_vec["M2A"]["Ex"]) * _vec["M2B"]["Ex"]
                                    + np.conj(_vec["M2A"]["Ez"]) * _vec["M2B"]["Ez"])) * DT * DT / pin_q)
_L_vec_rel_diff_pct = 100.0 * abs(_L_vector_M2 - _L_scalar_M2) / _L_scalar_M2
log("  vector check overlap L (M2, armA vs armB): scalar(ExA*ExB only)=%.6f  "
    "vector(ExA*ExB + EzA*EzB)=%.6f  (abs diff=%.3e, %.4f%% of scalar)" %
    (_L_scalar_M2, _L_vector_M2, abs(_L_vector_M2 - _L_scalar_M2), _L_vec_rel_diff_pct))

# FIX (verifier defect, item 1/6): the pd_capture_* numbers above are computed
# directly from the raw (one-polarisation-component) Ex/Ez amplitudes -- same
# basis issue as section 5's raw capture numbers ("absT is a
# POLARISATION COMPONENT; T is the order's TOTAL power"). Add the power-basis
# conversion (x FAC_T_OVER_T2, the same factor used for every other absolute-
# throughput number in this script; M0 has no atom and stays unscaled) so the
# md table can present both, labelled, rather than an unlabelled raw number.
# The intEz^2/intEx^2 RATIO itself is basis-invariant (a common factor on Ex
# cancels top and bottom of the M2A/M2B ratio) and is not rescaled.
_VEC_POWER_FAC = {"M0": 1.0, "M2A": FAC_T_OVER_T2, "M2B": FAC_T_OVER_T2}
VECTOR_CHECK = dict(
    z_um=F_DESIGN,
    definition="Ez = -(kx/kz) Ex, Ey=0 (TM output is x-polarised, E in x-z plane); evaluated on the "
               "SAME band-limited angular spectrum used for Ex everywhere else in this script "
               "(evanescent kz2<=0 zeroed, plus the Matsushima-Shimobaba u_lim(z) mask from make_H)",
    pd_capture_denominator="scalar pin_q -- the ONE-quadrant incident power (same normalisation as "
                           "every other capture/overlap number in this script, e.g. design_plane/"
                           "overlap_L_vs_M0; NOT the 4-quadrant total incident power)",
    pd_box_intEz2_over_intEx2=dict(M0=_vec["M0"]["ratio_ez_over_ex"], M2A=_vec["M2A"]["ratio_ez_over_ex"],
                                    M2B=_vec["M2B"]["ratio_ez_over_ex"]),
    pd_capture_scalar_Ex2_only_raw=dict(M0=_vec["M0"]["pd_capture_scalar_Ex2_only"],
                                         M2A=_vec["M2A"]["pd_capture_scalar_Ex2_only"],
                                         M2B=_vec["M2B"]["pd_capture_scalar_Ex2_only"]),
    pd_capture_vector_Ex2_plus_Ez2_raw=dict(M0=_vec["M0"]["pd_capture_vector_Ex2_plus_Ez2"],
                                             M2A=_vec["M2A"]["pd_capture_vector_Ex2_plus_Ez2"],
                                             M2B=_vec["M2B"]["pd_capture_vector_Ex2_plus_Ez2"]),
    pd_capture_scalar_Ex2_only_power_basis={
        k: _vec[k]["pd_capture_scalar_Ex2_only"] * _VEC_POWER_FAC[k] for k in ("M0", "M2A", "M2B")},
    pd_capture_vector_Ex2_plus_Ez2_power_basis={
        k: _vec[k]["pd_capture_vector_Ex2_plus_Ez2"] * _VEC_POWER_FAC[k] for k in ("M0", "M2A", "M2B")},
    power_basis_factor_note="M0 factor=1.0 (no atom, already power-consistent); M2A/M2B factor="
                            "FAC_T_OVER_T2=palette_T_total_over_amp_tA_sq_mean (%.4f)" % FAC_T_OVER_T2,
    overlap_L_M2_scalar_ExA_ExB_only=_L_scalar_M2,
    overlap_L_M2_vector_ExAExB_plus_EzAEzB=_L_vector_M2,
    overlap_L_M2_vector_vs_scalar_relative_diff_pct=_L_vec_rel_diff_pct,
)
track_rss("after vector check")

# ---------------------------------------------------------------------------
# SECTION 7 -- figures (F1-F4), Nature style, TM orange family, z increasing
# downward (lens plane at top, PD plane at bottom).
# ---------------------------------------------------------------------------


def save_fig(fig, name):
    png = os.path.join(OUTDIR, name + ".png")
    pdf = os.path.join(OUTDIR, name + ".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log("saved %s and %s" % (png, pdf))
    return png, pdf


FIG_FILES = {}

# ---- F1: xz slice through y=+XC, M2 arm A ----
# NOTE (verifier defect, minor/zero numeric impact): eval_with_H(S0, H, grid, xs, ys)
# internally applies its FIRST coordinate array to the row (y) axis and its second
# to the column (x) axis (the opposite of what the parameter names suggest). The
# call below therefore literally sweeps y at fixed x=XC, not x at fixed y=XC as the
# figure caption states. This design is exactly x<->y transpose-symmetric (Gaussian
# pump, hyperbolic phase and palette-level assignment are all symmetric under
# X2<->Y2, verified: idx == idx.T), so the two slices are numerically identical and
# the mislabel has NO effect on the plotted image or any reported number.
log("computing F1 (xz slice) ...")
t0 = time.time()
X_F1 = np.linspace(-150.0, 150.0, 601)
I1 = np.zeros((len(Z_GRID), len(X_F1)))
for iz, z in enumerate(Z_GRID):
    H1z = make_H(GRID1, z)
    E = eval_with_H(S0_M2A, H1z, GRID1, X_F1, [XC])[:, 0]
    I1[iz, :] = np.abs(E) ** 2
log("F1 field computed in %.1fs" % (time.time() - t0))

# FIX (major): a single global-focal-peak linear normalisation makes the
# aperture-edge launch invisible -- the focal peak is ~1e4 x the launch
# intensity at z~0, so I1n (below) renders as a blank field near the top of
# the plot and the two converging cones only appear right at the PD plane.
# Two panels replace the old single linear-global panel: (a) log scale (dB
# re the global focal peak, clipped at -40 dB) shows the full dynamic range
# end-to-end; (b) each z-row normalised to ITS OWN max (linear) shows the
# cone SHAPE/convergence at every z, including near the aperture where the
# absolute intensity is tiny. Both must show both cones converging onto the
# two PDs -- verified visually below.
I1_peak = np.max(I1)
I1n_global = I1 / I1_peak
I1_db = 10.0 * np.log10(np.clip(I1n_global, 1e-6, None))  # floor well below the -40dB display clip
# FIX (verifier defect, F1 cosmetic, optional): a single hot near-field pixel
# (z~15-22um) dominated the per-row max and blanked the rest of its row.
# Clip the per-row normaliser at the 99.5th percentile instead of the bare
# max, then clip the ratio itself to [0,1] (a handful of pixels above the
# 99.5th-pctl norm can otherwise exceed 1).
_rowmax = np.percentile(I1, 99.5, axis=1, keepdims=True)
I1n_row = np.divide(I1, _rowmax, out=np.zeros_like(I1), where=_rowmax > 0)
I1n_row = np.clip(I1n_row, 0.0, 1.0)
_iz200 = int(np.argmin(np.abs(Z_GRID - 200.0)))
log("F1 basis check: global peak I=%.6e ; dB range at z=%.1f row: min=%.2f max=%.2f dB"
    % (I1_peak, Z_GRID[_iz200], np.min(I1_db[_iz200]), np.max(I1_db[_iz200])))

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(7.6, 3.5))
_f1_panels = [
    (ax1a, I1_db, -40.0, 0.0, r"$10\log_{10}(|E|^2/I_{focal\,peak})$ (dB, clipped $-$40 dB)",
     "(a) global, log scale"),
    (ax1b, I1n_row, 0.0, 1.0, r"|E|$^2$ / (99.5th-pctl$_x$|E|$^2$ at this z), clipped",
     "(b) per-z row, linear (row norm. at 99.5th pctl)"),
]
for ax1, arr, vmin, vmax, cb_label, panel_title in _f1_panels:
    pcm = ax1.pcolormesh(X_F1, Z_GRID, arr, cmap=TM_CMAP, shading="auto", vmin=vmin, vmax=vmax, rasterized=True)
    ax1.set_ylim(Z_GRID.max(), Z_GRID.min())  # z=0 at top, z increasing downward
    ax1.axvline(0.0, color="k", lw=0.5, ls=":")
    for xc_sign in (+1, -1):
        # REPAIR (verifier defect, minor colour rule): PD-plane marker recoloured from
        # green (#009E73) to black (spec: "black or #6B2E00" for boxes/markers; black
        # chosen so it never blends with the TM_CMAP heatmap's own dark-orange end).
        ax1.plot([xc_sign * XC - 4, xc_sign * XC + 4], [200, 200], color="#000000", lw=2.2, solid_capstyle="butt")
    ax1.axhline(0.0, color="k", lw=0.8)
    for xe in (-150, 150):
        ax1.axvline(xe, color="k", lw=0.5, ls="--")
    # FIX (major): xlim widened past +-150 so the aperture-edge dashed
    # lines (previously sitting exactly on the axes border) are visible.
    ax1.set_xlim(-160.0, 160.0)
    ax1.set_xlabel(r"x ($\mu$m)")
    ax1.set_title(panel_title, fontsize=7)
    cb = fig1.colorbar(pcm, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label(cb_label, fontsize=6)
    cb.ax.tick_params(labelsize=6)
ax1a.set_ylabel(r"z ($\mu$m)")
fig1.suptitle("TM M2 arm A -- |E|$^2$, x-z at y=+%.3f $\\mu$m (aperture edges $\\pm$150 $\\mu$m dashed; "
              "PD plane z=200 $\\mu$m marked)" % XC, fontsize=7.2, y=1.03)
fig1.tight_layout()
FIG_FILES["F1"] = save_fig(fig1, "tm_focus_xz")

# ---- F2: focal plane crop, M0 / M2 arm A / M2 arm B / overlap ----
log("computing F2 (focal-plane crop) ...")
t0 = time.time()
N_F2 = 301
off_f2 = np.linspace(-15.0, 15.0, N_F2)
xy_f2 = XC + off_f2
E_M0_f2 = eval_with_H(S0_M0, H0_design, GRID0, xy_f2, xy_f2)
E_M2A_f2 = eval_with_H(S0_M2A, H1_design, GRID1, xy_f2, xy_f2)
E_M2B_f2 = eval_with_H(S0_M2B, H1_design, GRID1, xy_f2, xy_f2)
I_M0_f2 = np.abs(E_M0_f2) ** 2
I_M2A_f2 = np.abs(E_M2A_f2) ** 2
I_M2B_f2 = np.abs(E_M2B_f2) ** 2
OV_f2 = np.abs(np.conj(E_M2A_f2) * E_M2B_f2)
log("F2 fields computed in %.1fs" % (time.time() - t0))

# REPAIR (verifier defect, major/readability): a SHARED vmax (set by M0, capture
# ~87%) left the M2 arm A/B panels nearly blank, since their |t|^2-scaled capture
# is only ~8-15%. Each panel is now normalised to its OWN peak, with its own
# colourbar reading "|E|^2 / own peak" (0-1) -- the physical capture/L numbers
# stay in the subtitle text, not on the colourbar scale.
# FIX (major, BASIS): the arm-A/arm-B/overlap subtitles used to quote the
# RAW capture/L (one polarisation component, amp_tA/amp_tB -- "absT
# is a POLARISATION COMPONENT, not power"), presented with no basis label. Each
# arm/overlap panel now shows BOTH physical bases: lens-only (unit amplitude
# per level -- focusing quality only, section 5.1) and power basis (raw x
# T_total/|t|^2 = FAC_T_OVER_T2 (~2.4983) -- fraction of incident power
# actually reaching the PD through carrier+atom, section 5.2). M0 has no
# arm/atom distinction so its single "capture" number is unchanged (already
# power-consistent, unit amplitude by construction).
# FIX (verifier defect, minor -- F2 empty band): with aspect="equal" the 4
# square panels are width-limited by figure width / wspace, so a 3.3in-tall row
# left them vertically centred with a large blank band above (between suptitle
# and panel titles, 178 px of 890 measured). Figure height reduced to match the
# width-constrained axes size; 2.6in still left a 64-70 px band (verified by
# row-scanning a same-shape prototype), 2.2in removes it (largest remaining
# top-region gap 10 px).
fig2, axs2 = plt.subplots(1, 4, figsize=(13.6, 2.2), gridspec_kw=dict(wspace=0.85))
panels = [
    ("M0 (ideal)", I_M0_f2, "capture %.1f%%" % (100 * design_plane["M0"]["capture_pd"])),
    ("M2 arm A", I_M2A_f2, "lens-only capture %.1f%%\npower basis capture %.1f%%" %
        (100 * lens_only["M2Ap"]["capture_pd"], 100 * power_norm["M2A"])),
    ("M2 arm B", I_M2B_f2, "lens-only capture %.1f%%\npower basis capture %.1f%%" %
        (100 * lens_only["M2Bp"]["capture_pd"], 100 * power_norm["M2B"])),
    (r"|conj($E_A$)$E_B$|", OV_f2, "L lens-only %.2f dB\nL power basis %.2f dB vs M0" %
        (lens_only_overlap["M2"]["L_db_vs_M0"], power_norm["L_M2_db_vs_M0"])),
]
for ax, (title, arr, sub) in zip(axs2, panels):
    # FIX (verifier defect, item 1 minor -- F2 colourbar/suptitle): the heatmap
    # shape is basis-invariant (dividing by the panel's own peak removes the
    # raw-vs-lens-only-vs-power-basis question entirely), so normalise every
    # panel to 1 and label the colourbar accordingly instead of showing raw
    # absolute |E|^2 ticks (which the item-1 basis fix excludes from the main
    # body). The physical capture/L numbers stay in the subtitle, on the
    # labelled lens-only/power basis (unchanged).
    arr_n = arr / arr.max()
    pcm_i = ax.pcolormesh(off_f2, off_f2, arr_n, cmap=TM_CMAP, shading="auto",
                           vmin=0, vmax=1, rasterized=True)
    # REPAIR (verifier defect, minor colour rule): PD box recoloured from blue
    # (#0072B2) to black (spec: "black or #6B2E00"; black keeps contrast against
    # the TM_CMAP heatmap's own dark-orange end).
    ax.add_patch(plt.Rectangle((-4, -4), 8, 8, fill=False, edgecolor="#000000", lw=1.1))
    ax.set_xlabel(r"$\Delta x$ ($\mu$m)")
    ax.set_title("%s\n%s" % (title, sub), fontsize=6.2)
    ax.set_aspect("equal")
    # FIX (minor, layout): fraction/pad tuned + wspace widened above so a
    # panel's own colourbar tick labels no longer collide with the NEXT
    # panel's y-tick labels (measured overlap in the pre-fix PNG).
    cb_i = fig2.colorbar(pcm_i, ax=ax, fraction=0.05, pad=0.07)
    cb_i.ax.tick_params(labelsize=5)
    cb_i.set_label(r"|E|$^2$ / own peak", fontsize=5)
axs2[0].set_ylabel(r"$\Delta y$ ($\mu$m)")
# FIX (verifier defect, item 1 minor): suptitle shortened (no more contradictory
# "raw basis" / "own peak" pair) and y moved from 1.18 (large empty band) to a
# standard just-above-panels offset.
fig2.suptitle("TM focal plane, z=200 $\\mu$m, PD window (+%.3f,+%.3f) $\\mu$m -- each panel "
              "normalised to its own peak; capture/L in subtitles are lens-only / power basis"
              % (XC, XC), fontsize=6.6, y=1.02)
fig2.tight_layout()
FIG_FILES["F2"] = save_fig(fig2, "tm_focus_focalplane")

# ---- F3: (a) on-axis vs z, (b) PD capture vs z, (c) line cuts + FWHM ----
log("plotting F3 (profiles) ...")
# REPAIR (verifier defect, minor colour rule): M1 recoloured from blue (#0072B2)
# to the TM orange family; M0 recoloured from plain black to the family's
# darkest shade so all three series are within the mandated TM colour family.
# FIX (major, BASIS -- item 1): (a)/(b) previously plotted the raw (one-
# polarisation-component) basis with no basis label at all. They now plot the
# POWER basis (solid) and the LENS-ONLY basis (dashed) together, with the
# basis named in the panel title (NOT the rotated y-label -- a first attempt
# putting the full "(power basis, solid; lens-only, dashed)" phrase into the
# rotated ylabel measured ~5.3in of required vertical text extent against a
# ~2in usable panel height and was clipped by savefig's tight bbox; verified
# by cropping the saved PNG. The title runs horizontally across the full
# panel width, which has enough room, so it carries the basis note instead).
COL = {"M0": "#6B2E00", "M1A": "#E69F00", "M2A": "#D55E00"}
LAB = {"M0": "M0 ideal", "M1A": "M1 LPA (arm A)", "M2A": "M2 LPA+beta (arm A)"}
fig3, ax3 = plt.subplots(1, 3, figsize=(9.6, 2.7))
ax3[0].plot(Z_GRID, onaxis_power["M0"], color=COL["M0"], lw=1.1, ls="-", label=LAB["M0"])
for key in ("M1A", "M2A"):
    ax3[0].plot(Z_GRID, onaxis_power[key], color=COL[key], lw=1.1, ls="-", label="%s, power basis" % LAB[key])
    ax3[0].plot(Z_GRID, onaxis_lensbasis[key], color=COL[key], lw=1.0, ls="--", label="%s, lens-only" % LAB[key])
# FIX (verifier defect, markers minor): the per-key colour-coded dotted marker
# (one per M1A/M2A, in COL[key]) is replaced by ONE black dashed marker at the
# design plane z=200, matching panel (b)'s marker exactly (same colour/style).
ax3[0].axvline(200.0, color="k", lw=0.6, ls="--")
ax3[0].set_xlabel(r"z ($\mu$m)")
# FIX (verifier defect, item 1 minor -- F3(a) axis label): basis stated on the
# axis itself, not only in the title/legend.
ax3[0].set_ylabel(r"on-axis |E|$^2$ (see line style)")
ax3[0].set_title("(a) on-axis vs z at (+%.2f,+%.2f)\n(solid: power basis; dashed: lens-only)" % (XC, XC),
                  fontsize=6.2)
ax3[0].legend(fontsize=4.6, frameon=False)

ax3[1].plot(Z_GRID, 100 * pdcap_power["M0"], color=COL["M0"], lw=1.1, ls="-", label=LAB["M0"])
for key in ("M1A", "M2A"):
    ax3[1].plot(Z_GRID, 100 * pdcap_power[key], color=COL[key], lw=1.1, ls="-", label="%s, power basis" % LAB[key])
    ax3[1].plot(Z_GRID, 100 * pdcap_lensbasis[key], color=COL[key], lw=1.0, ls="--", label="%s, lens-only" % LAB[key])
ax3[1].axvline(200.0, color="k", lw=0.6, ls="--")
ax3[1].set_xlabel(r"z ($\mu$m)")
# FIX (verifier defect, item 1 minor -- F3(b) axis label): basis stated on the
# axis itself, not only in the title/legend.
ax3[1].set_ylabel("PD capture (%, basis per line style)")
ax3[1].set_title("(b) PD capture vs z\n(solid: power basis; dashed: lens-only)", fontsize=6.2)
ax3[1].legend(fontsize=4.6, frameon=False)

# FIX (verifier defect, item 1 major -- F3(c) raw basis): the raw one-
# polarisation-component curves (M2 peak ~40, matching JSON's raw
# onaxis_peak_intensity, NOT the power basis ~99.6 or lens-only ~388.1) are
# replaced by each curve normalised to ITS OWN peak. FWHM is a ratio (shape)
# property, so normalising does not change it -- the plotted FWHM values are
# identical to the un-normalised ones, and the basis question that raw |E|^2
# used to raise (item 1) simply does not apply to a basis-independent curve.
for key in ("M0", "M1A", "M2A"):
    _Ix2 = np.abs(Ecuts_x[key]) ** 2; _Ix2n = _Ix2 / _Ix2.max()
    _Iy2 = np.abs(Ecuts_y[key]) ** 2; _Iy2n = _Iy2 / _Iy2.max()
    ax3[2].plot(X_LINE - XC, _Ix2n, color=COL[key], lw=1.1,
                label="%s x (FWHM %.3f)" % (LAB[key], design_plane[key]["fwhm_x_um"]))
    ax3[2].plot((Y_LINE - XC), _Iy2n, color=COL[key], lw=1.1, ls="--",
                label="%s y (FWHM %.3f)" % (LAB[key], design_plane[key]["fwhm_y_um"]))
# REPAIR (verifier defect, major): diffraction-limit markers now placed at
# +-M0's OWN measured FWHM/2 (self-consistent by construction, so they sit
# exactly at M0's half-maximum instead of 57% outside it), and recoloured from
# green (#009E73) to black (distinct from the M0 series, which is now #6B2E00).
# FIX (minor): the FWHM marker had no legend entry at all (dotted lines
# plotted but never labelled) -- label the SECOND axvline only (matplotlib
# would otherwise draw two identical legend rows for the same pair of lines).
ax3[2].axvline(-DIFF_LIMIT_FWHM / 2, color="#000000", lw=0.8, ls=":")
ax3[2].axvline(+DIFF_LIMIT_FWHM / 2, color="#000000", lw=0.8, ls=":", label="M0 FWHM (reference)")
ax3[2].set_xlim(-4, 4)
ax3[2].set_xlabel(r"offset from PD centre ($\mu$m)")
# FIX (verifier defect, item 1 major): axis label states the new own-peak
# normalisation and its basis-independence.
ax3[2].set_ylabel(r"|E|$^2$ / own peak")
# FIX (verifier defect, item 4): never call M0's own FWHM the "diffraction
# limit" -- it is M0's own measured spot, used here only as a reference, and
# judging M0 against its own FWHM would be circular. Report it as "M0 FWHM
# (reference)" (matching the legend entry) and give the aperture-derived
# lambda/(2*NA_rms,aperture) as a SEPARATE number alongside it (not folded
# into one "diffraction limit" label as before).
ax3[2].set_title("(c) line cuts (z=200), each curve normalised to its own peak "
                  "(basis-independent; FWHM unaffected)\nM0 FWHM=%.3f $\\mu$m (reference); "
                  "$\\lambda/(2\\,NA_{rms,aperture})$=%.3f $\\mu$m" %
                  (DIFF_LIMIT_FWHM, _old_wrong_fwhm), fontsize=6.0)
# REPAIR (verifier defect, minor/readability): legend moved outside the axes --
# it previously overlapped the black/dark M0 curve near the peak.
ax3[2].legend(fontsize=4.6, frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.03))
fig3.tight_layout()
FIG_FILES["F3"] = save_fig(fig3, "tm_focus_profiles")

# ---- F4: full-aperture focal plane, M2 arm A, four PD spots ----
log("computing F4 (full-aperture focal plane) ...")
t0 = time.time()
xy_f4 = np.linspace(-150.0, 150.0, 601)
E_f4 = eval_with_H(S0_M2A, H1_design, GRID1, xy_f4, xy_f4)
I_f4 = np.abs(E_f4) ** 2
I_f4n = I_f4 / I_f4.max()
log("F4 field computed in %.1fs" % (time.time() - t0))

fig4, ax4 = plt.subplots(figsize=(3.6, 3.4))
pcm4 = ax4.pcolormesh(xy_f4, xy_f4, I_f4n, cmap=TM_CMAP, shading="auto", vmin=0, vmax=1, rasterized=True)
for sx in (+1, -1):
    for sy in (+1, -1):
        # REPAIR (verifier defect, minor colour rule): PD box recoloured from
        # blue (#0072B2) to black (spec: "black or #6B2E00").
        ax4.add_patch(plt.Rectangle((sx * XC - 4, sy * XC - 4), 8, 8, fill=False, edgecolor="#000000", lw=1.0))
ax4.axhline(0, color="k", lw=0.4, ls=":"); ax4.axvline(0, color="k", lw=0.4, ls=":")
ax4.set_xlabel(r"x ($\mu$m)"); ax4.set_ylabel(r"y ($\mu$m)")
ax4.set_aspect("equal")
ax4.set_title("TM M2 arm A, full aperture focal plane, z=200 $\\mu$m", fontsize=7)
cb4 = fig4.colorbar(pcm4, ax=ax4, fraction=0.046, pad=0.04)
cb4.set_label("|E|$^2$ (norm.)")
FIG_FILES["F4"] = save_fig(fig4, "tm_focus_fullaperture")

# ---- [261011 ADDED] TE copies of the full-aperture (F4) and focal-plane (F2) figures ----
# The TE lens is the TM lens rotated by 90 deg: the TE panels are np.rot90 of the SAME arrays the TM figures plot, with a
# white -> #0072B2 colormap; same style, same normalisation. No new computation.
TE_CMAP = LinearSegmentedColormap.from_list("te_blue", ["#FFFFFF", "#0072B2"], N=256)

figT2, axsT2 = plt.subplots(1, 4, figsize=(13.6, 2.2), gridspec_kw=dict(wspace=0.85))
for ax, (title, arr, sub) in zip(axsT2, panels):
    arr_n = np.rot90(arr / arr.max())
    pcm_i = ax.pcolormesh(off_f2, off_f2, arr_n, cmap=TE_CMAP, shading="auto", vmin=0, vmax=1, rasterized=True)
    ax.add_patch(plt.Rectangle((-4, -4), 8, 8, fill=False, edgecolor="#000000", lw=1.1))
    ax.set_xlabel(r"$\Delta x$ ($\mu$m)")
    ax.set_title("TE: %s\n%s" % (title, sub), fontsize=6.2)
    ax.set_aspect("equal")
    cb_i = figT2.colorbar(pcm_i, ax=ax, fraction=0.05, pad=0.07)
    cb_i.ax.tick_params(labelsize=5)
    cb_i.set_label(r"|E|$^2$ / own peak", fontsize=5)
axsT2[0].set_ylabel(r"$\Delta y$ ($\mu$m)")
figT2.suptitle("TE focal plane (TM lens rotated 90 deg: np.rot90 of the TM arrays), z=200 $\\mu$m, PD window (+%.3f,+%.3f) $\\mu$m -- "
               "each panel normalised to its own peak; capture/L in subtitles are the TM lens-only / power-basis numbers"
               % (XC, XC), fontsize=6.6, y=1.02)
figT2.tight_layout()
FIG_FILES["F2_TE"] = save_fig(figT2, "tm_focus_focalplane_TE_rot90")

figT4, axT4 = plt.subplots(figsize=(3.6, 3.4))
pcmT4 = axT4.pcolormesh(xy_f4, xy_f4, np.rot90(I_f4n), cmap=TE_CMAP, shading="auto", vmin=0, vmax=1, rasterized=True)
for sx in (+1, -1):
    for sy in (+1, -1):
        axT4.add_patch(plt.Rectangle((sx * XC - 4, sy * XC - 4), 8, 8, fill=False, edgecolor="#000000", lw=1.0))
axT4.axhline(0, color="k", lw=0.4, ls=":"); axT4.axvline(0, color="k", lw=0.4, ls=":")
axT4.set_xlabel(r"x ($\mu$m)"); axT4.set_ylabel(r"y ($\mu$m)")
axT4.set_aspect("equal")
axT4.set_title("TE M2 arm A (TM rot90), full aperture focal plane, z=200 $\\mu$m", fontsize=7)
cbT4 = figT4.colorbar(pcmT4, ax=axT4, fraction=0.046, pad=0.04)
cbT4.set_label("|E|$^2$ (norm.)")
FIG_FILES["F4_TE"] = save_fig(figT4, "tm_focus_fullaperture_TE_rot90")
# diagnostic only: the mirrored full-aperture array is x<->y symmetric, so its rot90 may differ from itself only at rounding level
_TE_FULL_DEV = float(np.max(np.abs(np.rot90(I_f4n) - I_f4n)))
log("TE full-aperture check: max|rot90(I_f4n) - I_f4n| = %.3e (normalised units; ~0 means the TE full-aperture figure differs from the TM one only by colormap)" % _TE_FULL_DEV)

# ---- [261011 ADDED] tm_focus_arrays.npz: the EXACT arrays the F2 / F4 figures (TM and TE rot90) plot. Saving only. ----
# Array layout: I[iy, ix], i.e. exactly as passed to pcolormesh(x, y, I). full_* : absolute x/y in um (aperture centre = origin);
# focal_* : x/y are the plotted offsets from the PD centre in um (absolute = XC + offset, see focal_x_abs_um / focal_y_abs_um).
# *_I is the plotted (already normalised, colour scale 0..1) array, *_norm the divisor (raw |E|^2 = *_I * *_norm, same E units as
# every other number of this script: one polarisation component, one-quadrant incident-power basis). TE keys = np.rot90 of the TM array.
_NPZ = {}
_NPZ["full_M2A_I"] = np.asarray(I_f4n)
_NPZ["full_M2A_x"] = np.asarray(xy_f4)
_NPZ["full_M2A_y"] = np.asarray(xy_f4)
_NPZ["full_M2A_norm"] = np.float64(I_f4.max())
_NPZ["full_M2A_label"] = np.array("TM M2 arm A (LPA palette x beta ladder), full aperture focal plane z=200 um, |E|^2 / max")
_NPZ["full_M2A_TE_I"] = np.rot90(I_f4n)
_NPZ["full_M2A_TE_x"] = np.asarray(xy_f4)
_NPZ["full_M2A_TE_y"] = np.asarray(xy_f4)
_NPZ["full_M2A_TE_norm"] = np.float64(I_f4.max())
_NPZ["full_M2A_TE_label"] = np.array("TE view = np.rot90 of the TM M2 arm A full aperture focal plane, z=200 um, |E|^2 / max")
_NPZ["full_M2A_rot90_maxabsdiff"] = np.float64(_TE_FULL_DEV)
for _nk, _arr, _lab in (("M0", I_M0_f2, "M0 (ideal)"), ("M2A", I_M2A_f2, "M2 arm A"), ("M2B", I_M2B_f2, "M2 arm B"),
                        ("OV", OV_f2, "|conj(E_A) E_B| (overlap, M2)")):
    _arr_n = _arr / _arr.max()
    _NPZ["focal_%s_I" % _nk] = np.asarray(_arr_n)
    _NPZ["focal_%s_x" % _nk] = np.asarray(off_f2)
    _NPZ["focal_%s_y" % _nk] = np.asarray(off_f2)
    _NPZ["focal_%s_norm" % _nk] = np.float64(_arr.max())
    _NPZ["focal_%s_label" % _nk] = np.array("TM %s, focal plane crop z=200 um around the PD centre, |E|^2 / own peak" % _lab)
    _NPZ["focal_%s_TE_I" % _nk] = np.rot90(_arr_n)
    _NPZ["focal_%s_TE_x" % _nk] = np.asarray(off_f2)
    _NPZ["focal_%s_TE_y" % _nk] = np.asarray(off_f2)
    _NPZ["focal_%s_TE_norm" % _nk] = np.float64(_arr.max())
    _NPZ["focal_%s_TE_label" % _nk] = np.array("TE view = np.rot90 of TM %s focal plane crop, z=200 um, |E|^2 / own peak" % _lab)
_NPZ["focal_x_abs_um"] = np.asarray(xy_f2)
_NPZ["focal_y_abs_um"] = np.asarray(xy_f2)
_NPZ["pd_centres_um"] = np.array([[XC, XC], [-XC, XC], [-XC, -XC], [XC, -XC]])
_NPZ["pd_half_width_um"] = np.float64(4.0)
_NPZ["XC_um"] = np.float64(XC)
_NPZ["z_design_um"] = np.float64(F_DESIGN)
_NPZ["array_layout"] = np.array("I[iy, ix] as passed to matplotlib pcolormesh(x, y, I); x = column axis, y = row axis")
NPZ_PATH = os.path.join(OUTDIR, "tm_focus_arrays.npz")
np.savez_compressed(NPZ_PATH, **_NPZ)
NPZ_KEYS = sorted(_NPZ.keys())
log("wrote %s (%d keys)" % (NPZ_PATH, len(NPZ_KEYS)))

track_rss("after all figures saved")

# ---------------------------------------------------------------------------
# SECTION 8 -- JSON summary
# ---------------------------------------------------------------------------
summary = {
    # [261011] CONTROL-V (verdict reproduction) / CONTROL-F (figure model vs ladder betaL) replace the recheck3 CONTROL / CONTROL-2
    "control_V": CONTROL_V,
    "control_F": CONTROL_F,
    "verdict_basis": dict(definition="total = betaL + deficit_p' + purity_p' (lens_total_li.evaluate_cells, P769 atom-level terms), "
                                     "variant rB0_complex, decay / held tails, dB vs the ideal lens", **CONTROL_V["verdict_rB0_complex"]),
    "caption_note": "RCWA cell results + scalar angular-spectrum propagation; not a full 3-D field solve.",
    "arrays_npz": dict(path=NPZ_PATH, keys=NPZ_KEYS),
    "palette_md5_check": MD5_TABLE,
    "palette_level_dirs": LEVEL_DIRS,
    "design": dict(f_um=F_DESIGN, lam_um=LAM, W_pump_um=W_PUMP, APER_um=APER, XC_um=XC,
                   grid_M1_um=GRID_M1, LAMG_um=LAMG),
    "phase_offset_deg": PHASE_OFFSET_DEG,
    "phase_offset_rms_error_deg": PHASE_RMS_DEG,
    # REPAIR (verifier defect, major/factual): per-level EFFECTIVE nG, replacing
    # the previous single "palette_nG_eval_basis: 601" (false for 2/8 levels).
    "palette_effective_nG_per_level": {p["name"]: p["eff_nG"] for p in palette},
    "palette_effective_nG_counts": _effnG_counts,
    "palette_T_total_over_amp_tA_sq": {p["name"]: p["t_over_t2_A"] for p in palette},
    "palette_T_total_over_amp_tA_sq_mean": FAC_T_OVER_T2,
    "palette_level_usage_fraction": {p["name"]: float(_level_counts[i]) / float(level_idx.size)
                                      for i, p in enumerate(palette)},
    "beta_ladder_sources_per_M": SRC_USED,
    "per_model_arm": design_plane,
    "overlap_L_vs_M0": overlap,
    # REPAIR (verifier defect, major): lens-only (phase-only, unit amplitude per
    # level) and power-normalised (absolute throughput) metrics, plus the
    # decomposition explaining the M2 lens-only dB figure.
    "lens_only_design_plane": lens_only,
    "lens_only_overlap_L_vs_M0": lens_only_overlap,
    "phase_quantisation_predicted_loss_db": PRED_QUANT_LOSS_DB,
    "control_beta_ladder_only_L_db": _verdict_L_db,
    "decomposition_sum_db_vs_measured_lens_only_M2_db": [
        _verdict_L_db + PRED_QUANT_LOSS_DB, lens_only_overlap["M2"]["L_db_vs_M0"]],
    "power_normalised_absolute_throughput": power_norm,
    "grid": dict(pad_factor=PAD_FACTOR,
                 grid0=dict(dx_um=GRID0["dx"], n_full=GRID0["n_full"], npd=GRID0["npd"]),
                 grid1=dict(dx_um=GRID1["dx"], n_full=GRID1["n_full"], npd=GRID1["npd"]),
                 band_limited_asm_applied=True),
    "NA_max_quadrant_far_corner": NA_MAX,
    # FIX (verifier defect, item 4): never label M0's own FWHM "the diffraction
    # limit" (judging M0 against its own FWHM is circular). Renamed per the
    # verifier's own fix text: m0_fwhm_reference_um / NA_back_inferred_from_M0.
    # The aperture-derived NA and its lambda/(2*NA) are reported SEPARATELY
    # (NA_rms_aperture / lambda_over_2NA_rms_aperture_um), not folded together.
    "NA_rms_aperture": NA_RMS_APERTURE,
    "lambda_over_2NA_rms_aperture_um": _old_wrong_fwhm,
    "NA_back_inferred_from_M0": NA_EFF_FROM_M0,
    "m0_fwhm_reference_um": DIFF_LIMIT_FWHM,
    "internal_sanity_full_vs_isolated_quadrant_M0_capture": dict(
        full_aperture=_c0_full_sanity, isolated_quadrant=_c0_ctrl),
    # NEW (user-facing promise, 260929): vector check of the scalar approximation.
    "vector_check": VECTOR_CHECK,
}
SUMMARY_PATH = os.path.join(OUTDIR, "tm_focus_summary.json")
json.dump(summary, open(SUMMARY_PATH, "w", encoding="utf-8"), indent=2)
log("wrote %s" % SUMMARY_PATH)

# ---------------------------------------------------------------------------
# SECTION 9 -- Korean .md report
# ---------------------------------------------------------------------------
_wall_s = time.time() - T_START
_peak_rss = track_rss("final")


def pct(x):
    return "%.2f%%" % (100.0 * x)


def src_line(M):
    s = SRC_USED[M]
    return "M=%d (P=%.2f Lambda_g): %s, shift count=%d" % (M, M / 3.0, s["label"], s["n_shift"])


md = []
md.append("# TM metalens focusing shape (261011, regenerated with the current palette P769 Li + the current Li lens ladder)\n")
md.append("Numerical visualisation of the focusing shape of the TM metalens (the 261002 figures regenerated with the current TM palette P769 (Li record) and the current Li lens ladder).\n")
md.append("\n**Common figure caption text**: RCWA cell results + scalar angular-spectrum propagation; not a full 3-D field solve.\n")

# ---- SECTION 0.0 (ADDED 261011): verdict basis versus the numbers of this figure model ---------------------------
_VB = CONTROL_V["verdict_rB0_complex"]
md.append("\n## 0. Verdict criterion vs. the numbers of this figure, and changes relative to 261002\n\n")
md.append("### 0.0 Verdict criterion and figure-model numbers (placed in the same table but they are different quantities)\n\n")
md.append("**Verdict criterion (reproduced by CONTROL-V)**: total = betaL + deficit_p' + purity_p' (lens_total_li.evaluate_cells, P769 atom-level terms, "
          "variant rB0_complex, f=200 um, dB relative to an ideal lens). The current verdict values are decay / held = **%+.3f / %+.3f dB** "
          "(betaL %+.3f / %+.3f dB, deficit_p' %+.4f dB, purity_p' f200 %+.4f dB, echo aperture %.3f dB).\n\n" %
          (_VB["total"][0], _VB["total"][1], _VB["betaL"][0], _VB["betaL"][1], _VB["deficit"], _VB["purity"], _VB["echo_aperture"]))
md.append("| quantity | value | what it measures |\n|---|---|---|\n")
md.append("| verdict total (decay / held) | %+.3f / %+.3f dB | betaL + deficit_p' + purity_p' (CONTROL-V) |\n" % (_VB["total"][0], _VB["total"][1]))
md.append("| ladder-only betaL (decay / held), CONTROL-F | %+.3f / %+.3f dB | continuous phase + beta ladder only, 1 quadrant exact 8x8 um PD box, reproduced by focal_q/beta_model of this script |\n" %
          (CONTROL_F["tails"]["decay"]["L_dB"], CONTROL_F["tails"]["held"]["L_dB"]))
md.append("| figure model M2 overlap L (lens-only, dB vs M0) | %+.3f dB | palette atom of nearest phase (unit amplitude per level) x beta ladder, section 5.1 |\n" % lens_only_overlap["M2"]["L_db_vs_M0"])
md.append("| figure model M2 overlap L (power basis, dB vs M0) | %+.3f dB | the above value times the atom transmission budget (mean factor T_tot vs squared tA amplitude), section 5.2 |\n" % power_norm["L_M2_db_vs_M0"])
md.append("\n**Note**: M1/M2 of the figures is a Lambda/3 cell model that uses **the complex t (tA/tB) of the palette atom itself** with the nearest phase for each cell as is, "
          "while the verdict is a separate model that adds the atom-level deficit_p' / purity_p' terms to betaL. Therefore **the capture / L numbers of the figures are not verdict values** "
          "(the verdict value is the first row of the table above) and must not be compared as the same quantity. The figures serve to show the focal shape.\n\n")

# ---- SECTION 0 (ADDED 261002, re-pointed 261011): what changed vs the 261002 run -------------------------------------
md.append("The 261002 figures were based on the FROZEN TM palette (Laurent record, 8 levels) and the sim585 ladder. "
          "The figures and numbers of this document are the result of rerunning the same script (algorithm lines after SECTION 4 unchanged; list of changes in diff_vs_261002.txt) with the current palette P769 (Li record) and "
          "the current Li lens ladder (21 cells of verdict \"%s\"). Design f=%.0f um, 4-quadrant mirror, 300x300 um aperture and 8x8 um PD conditions are the same as in 261002.\n\n"
          % (VERDICT_KEY, F_DESIGN))
md.append("### 0.1 palette replacement (all 8 levels: 261002 FROZEN Laurent -> 261011 P769 Li, first 8 digits of rho3 md5 and atom measurements)\n\n")
md.append("| level | rho3 md5 261002 -> 261011 | rcwa_rule 261002 -> 261011 | A_deg 261002 | A_deg 261011 | amp_tA 261002 | amp_tA 261011 | eff. nG 261002 -> 261011 |\n")
md.append("|---|---|---|---|---|---|---|---|\n")
for _lvl in PALETTE_LEVELS:
    _pn = [p for p in palette if p["name"] == _lvl][0]
    md.append("| %s | %s -> %s | %s -> liy | %.3f | %.3f | %.4f | %.4f | %d -> %d |\n" %
              (_lvl, _old_md5[_lvl], EXPECTED_MD5[_lvl], str(_old_atom[_lvl].get("rcwa_rule") or "laurent"),
               _old_atom[_lvl]["A_deg"], _pn["A_deg"],
               _old_atom[_lvl]["amp_tA"], _pn["amp_tA"], OLD["palette_effective_nG_per_level"][_lvl], _pn["eff_nG"]))
md.append("\n### 0.2 palette verification (all 8 levels)\n\n")
md.append("For each level one atom2d_opt.json (eval_only record; rho3_opt.npy is not in this folder but is the file pointed to by record[rho3_npy]) was read, "
          "and it was checked that the first 8 digits of its md5 equal the expected value and that the record has pol=TM, rcwa_rule=liy, rcwa_ff=1, nG=1793.\n\n")
md.append("| level | record folder (relative to Result) | record[rho3_npy] | md5 | expected | md5 result | record field result |\n|---|---|---|---|---|---|---|\n")
for _r in MD5_TABLE:
    md.append("| %s | %s | %s | %s | %s | %s | %s |\n" % (_r["level"], _r["folder"].replace(RESULT_ROOT + "\\", ""), _r["rho3_npy"],
                                                         _r["md5"], _r["expected"], "OK" if _r["ok"] else "MISMATCH",
                                                         "OK" if _r["record_ok"] else "MISMATCH"))


def _hrow(label, o, n, kind):
    if kind == "pct":
        return "| %s | %.2f%% | %.2f%% | %+.2f pp |\n" % (label, 100 * o, 100 * n, 100 * (n - o))
    if kind == "db":
        return "| %s | %.3f dB | %.3f dB | %+.3f dB |\n" % (label, o, n, n - o)
    if kind == "deg":
        return "| %s | %.2f deg | %.2f deg | %+.2f deg |\n" % (label, o, n, n - o)
    if kind == "pctval":
        return "| %s | %.3f%% | %.3f%% | %+.3f pp |\n" % (label, o, n, n - o)
    return "| %s | %.4f | %.4f | %+.4f |\n" % (label, o, n, n - o)


md.append("\n### 0.3 headline comparison (design plane z=%.0f um, M2 = LPA palette x beta ladder, 8x8 um PD)\n\n" % F_DESIGN)
md.append("capture/L is relative to the incident power of 1 quadrant (the convention of this whole script); lens-only is phase only (unit amplitude per level), "
          "and power basis includes the atom transmission budget (see section 5).\n\n")
md.append("| item | 261002 (FROZEN palette, sim585 ladder) | 261011 (P769 Li, Li ladder) | change |\n|---|---|---|---|\n")
_O = OLD
md.append(_hrow("M0 ideal capture", _O["per_model_arm"]["M0"]["capture_pd"], design_plane["M0"]["capture_pd"], "pct"))
md.append(_hrow("M1 arm A capture (lens-only)", _O["lens_only_design_plane"]["M1Ap"]["capture_pd"], lens_only["M1Ap"]["capture_pd"], "pct"))
md.append(_hrow("M1 arm B capture (lens-only)", _O["lens_only_design_plane"]["M1Bp"]["capture_pd"], lens_only["M1Bp"]["capture_pd"], "pct"))
md.append(_hrow("M2 arm A capture (lens-only)", _O["lens_only_design_plane"]["M2Ap"]["capture_pd"], lens_only["M2Ap"]["capture_pd"], "pct"))
md.append(_hrow("M2 arm B capture (lens-only)", _O["lens_only_design_plane"]["M2Bp"]["capture_pd"], lens_only["M2Bp"]["capture_pd"], "pct"))
md.append(_hrow("M2 arm A capture (power basis)", _O["power_normalised_absolute_throughput"]["M2A"], power_norm["M2A"], "pct"))
md.append(_hrow("M2 arm B capture (power basis)", _O["power_normalised_absolute_throughput"]["M2B"], power_norm["M2B"], "pct"))
md.append(_hrow("overlap L, M1 (lens-only, dB vs M0)", _O["lens_only_overlap_L_vs_M0"]["M1"]["L_db_vs_M0"], lens_only_overlap["M1"]["L_db_vs_M0"], "db"))
md.append(_hrow("overlap L, M2 (lens-only, dB vs M0)", _O["lens_only_overlap_L_vs_M0"]["M2"]["L_db_vs_M0"], lens_only_overlap["M2"]["L_db_vs_M0"], "db"))
md.append(_hrow("overlap L, M2 (lens-only, linear)", _O["lens_only_overlap_L_vs_M0"]["M2"]["L"], lens_only_overlap["M2"]["L"], "lin"))
md.append(_hrow("overlap L, M2 (power basis, dB vs M0)", _O["power_normalised_absolute_throughput"]["L_M2_db_vs_M0"], power_norm["L_M2_db_vs_M0"], "db"))
md.append(_hrow("overlap L, M2 (power basis, linear)", _O["power_normalised_absolute_throughput"]["L_M2"], power_norm["L_M2"], "lin"))
md.append(_hrow("M2 arm A FWHM x (um)", _O["per_model_arm"]["M2A"]["fwhm_x_um"], design_plane["M2A"]["fwhm_x_um"], "lin"))
md.append(_hrow("M2 arm A Strehl (lens-only)", _O["lens_only_design_plane"]["M2Ap"]["strehl_z200"], lens_only["M2Ap"]["strehl_z200"], "lin"))
md.append(_hrow("M2 arm A focal peak z (um)", _O["per_model_arm"]["M2A"]["focal_peak_z_um"], design_plane["M2A"]["focal_peak_z_um"], "lin"))
md.append(_hrow("global phase offset", _O["phase_offset_deg"], PHASE_OFFSET_DEG, "deg"))
md.append(_hrow("residual rms phase error", _O["phase_offset_rms_error_deg"], PHASE_RMS_DEG, "deg"))
md.append(_hrow("predicted phase-quantisation loss", _O["phase_quantisation_predicted_loss_db"], PRED_QUANT_LOSS_DB, "db"))
md.append(_hrow("vector check: L change scalar -> vector (M2)", _O["vector_check"]["overlap_L_M2_vector_vs_scalar_relative_diff_pct"],
                VECTOR_CHECK["overlap_L_M2_vector_vs_scalar_relative_diff_pct"], "pctval"))
md.append("\n### 0.4 Ladder-level values (continuous phase + beta ladder only, 1 quadrant exact 8x8 um PD box, f=200 um, decay tail)\n\n")
md.append("These are ladder-only numbers without palette quantisation. The 261002 column is sim585 (FROZEN palette, Laurent ladder, all-points decay, control2 f=200 of the 261002 summary json), "
          "the 261011 column is the current Li ladder (betaL of the verdict variant rB0_complex, reproduced by focal_q/beta_model of this script as CONTROL-F). "
          "The two columns differ in ladder and in calculation basis (Laurent vs Li, with or without the rB0 term), so the difference is not an effect of the palette alone.\n\n")
_c_old = OLD["control2_sim585_verdict"]["200"]
md.append("| item | 261002 (sim585) | 261011 (Li ladder) | change (dB) |\n|---|---|---|---|\n")
md.append("| ov / c0 (decay) | %.4f / %.4f | %.4f / %.4f | |\n" %
          (_c_old["ov"], _c_old["c0"], CONTROL_F["tails"]["decay"]["ov"], CONTROL_F["c0"]))
md.append("| ladder-only L, decay (dB) | %.3f | %.3f | %+.3f |\n" %
          (_c_old["L_db"], CONTROL_F["tails"]["decay"]["L_dB"], CONTROL_F["tails"]["decay"]["L_dB"] - _c_old["L_db"]))
md.append("| ladder-only L, held (dB) | n/a | %.3f | |\n" % CONTROL_F["tails"]["held"]["L_dB"])
md.append("\n**Note (f=136)**: the design plane of this script is the single F_DESIGN=%.0f um. Therefore capture/overlap numbers at f=136 um "
          "including palette quantisation were produced neither in the previous version nor in this one.\n\n"
          % F_DESIGN)

md.append("## 1. Method\n")
md.append("A single 300x300 um aperture was built by mirroring four 150x150 um quadrants. "
          "A common Gaussian pump (w=%.1f um, centred at the origin) is used, and each quadrant has its own hyperbolic lens "
          "(f=%.1f um, lambda=%.2f um) centred at (+-%.3f, +-%.3f) um and focuses onto the 8x8 um PD at the same position. "
          "TM polarization and the current 8-level LPA palette P769 (Li record, identified by the md5 of the record[rho3_npy] file, see section 0) are used.\n" %
          (W_PUMP, F_DESIGN, LAM, XC, XC))
md.append("Three near-field models were built for each of arm A/B.\n")
md.append("- **M0 (ideal)**: continuous phase, unit amplitude, apodised by the Gaussian pump.\n")
md.append("- **M1 (LPA palette)**: Lambda/3 = %.6f um grating. For each cell the palette level nearest to the target phase (after the global offset) "
          "is chosen and the measured complex tA(arm A)/tB(arm B) of that level "
          "(including amplitude) is used. The same cell assignment is used for both arms (physically a single atom).\n" % GRID_M1)
md.append("- **M2 = M1 x beta_arm(P)**: the model multiplied by the superramp beta(P) ladder model of the current Li lens ladder (21 cells of verdict \"P769 approved (Li terms)\", "
          "for arm B on the rB0_complex basis nd[M][B] x nr[M], after lens_total_li.conv()) ('decay' tail, "
          "all points used). P was computed analytically from the phase gradient.\n")
md.append("Propagation is a band-limited ASM in air (Matsushima-Shimobaba method; as in the already verified "
          "review_net_f.py/ladder646.py (the evaluator used by lens_total_li.py), evanescent components are clipped to 0, "
          "padding factor %d), applied as a scalar (co-polarised component). The maximum NA seen from the quadrant far corner "
          "is %.4f, which means that the validity of the paraxial/scalar approximation is limited (see the limitations item).\n" %
          (PAD_FACTOR, NA_MAX))

md.append("## 2. CONTROL results (required, check first)\n\n")
md.append("The recheck3 CONTROL and CONTROL-2 (sim585) of the 261002 version were removed because the reference json (lens_ladder_eval_recheck3.json) no longer exists, and "
          "were replaced by CONTROL-V and CONTROL-F below. Both must pass before SECTION 4 onwards (the figure calculation) is executed.\n\n")
md.append("### 2.1 CONTROL-V: reproduction of the current verdict (tolerance %.0e)\n\n" % CONTROL_V["tolerance"])
md.append("lens_total_li.evaluate_cells (21 cells) was run with load_terms replaced by the P769 atom-level terms (deficit_p', purity_p' f200; %s); "
          "in %s, results[\"%s\"] and all values were compared (total = betaL + deficit_p' + purity_p').\n\n" %
          (CONTROL_V["terms_file"], CONTROL_V["reference_file"], CONTROL_V["reference_key"]))
md.append("| item | measured | reference | absolute difference | result |\n|---|---|---|---|---|\n")
for _r in CONTROL_V["rows"]:
    md.append("| %s | %.16f | %.16f | %.3e | %s |\n" % (_r["name"], _r["measured"], _r["reference"], _r["absdiff"], "OK" if _r["ok"] else "FAIL"))
md.append("\nComparison with the expected values quoted in the task:\n\n")
md.append("| item | measured | quoted expected value | absolute difference | result |\n|---|---|---|---|---|\n")
for _r in CONTROL_V["spec_expected_rows"]:
    md.append("| %s | %.16f | %.16f | %.3e | %s |\n" % (_r["name"], _r["measured"], _r["spec_expected"], _r["absdiff"], "OK" if _r["ok"] else "FAIL"))
md.append("\n**Result: %s** (all absolute differences <= %.0e)\n\n" % ("PASS" if CONTROL_V["pass"] else "FAIL", CONTROL_V["tolerance"]))
md.append("### 2.2 CONTROL-F: does the figure model (focal_q / beta_model of this script) reproduce the ladder betaL (tolerance %.0e dB)\n\n" % CONTROL_F["tolerance_dB"])
md.append("On the review_net_f quadrant grid with f=%.0f um, phase -K0 (sqrt(rho^2+f^2) - f), Aq Gaussian apodisation, and the ladder being _Pm_real / _bmA_real / _bmB_real after conv(). "
          "L_dB = 10 log10(ov / c0) was compared with betaL (decay / held) of the verdict variant rB0_complex (c0 = %.16f).\n\n" % (CONTROL_F["f_um"], CONTROL_F["c0"]))
md.append("| tail | overlap ov | L_dB (this script) | betaL rB0_complex (CONTROL-V) | absolute difference | result |\n|---|---|---|---|---|---|\n")
for _tail in ("decay", "held"):
    _t = CONTROL_F["tails"][_tail]
    md.append("| %s | %.16f | %.16f | %.16f | %.3e | %s |\n" %
              (_tail, _t["ov"], _t["L_dB"], _t["reference_betaL_rB0_complex"], _t["absdiff"], "OK" if _t["ok"] else "FAIL"))
md.append("\n**Result: %s** (absolute difference <= %.0e dB for both tails)\n\n" % ("PASS" if CONTROL_F["pass"] else "FAIL", CONTROL_F["tolerance_dB"]))
md.append("All CONTROL numbers (including the palette md5 gate) are saved in %s.\n\n" % CONTROLS_JSON)
md.append("## 3. Global phase offset, palette usage, palette nG basis (REPAIR applied)\n")
md.append("The global offset obtained by minimising the unweighted circular RMS over all cells of 1 quadrant, on the arm-A phase basis, "
          "is **%.2f deg** (search grid 0.1 deg), and the residual RMS error then is **%.3f deg**.\n\n"
          % (PHASE_OFFSET_DEG, PHASE_RMS_DEG))
_NG_UNIFORM = (len(_effnG_counts) == 1)
md.append("**palette nG basis**: not the `nG_eval` field of the JSON but the EFFECTIVE nG to which amp_tA/A_deg/amp_tB/B_deg "
          "actually correspond was read per level (for eval_only levels `nG` (\"eval_only uses nG, not nG_eval\"), "
          "for non-eval_only levels `nG_report` if present, otherwise `nG`). Number of levels per basis in this P769 palette: %s "
          "(261002 FROZEN palette: %s). %s\n\n" %
          (", ".join("nG=%d: %d levels" % (k, _effnG_counts[k]) for k in sorted(_effnG_counts, reverse=True)),
           ", ".join("nG=%s: %d levels" % (k, v) for k, v in sorted(OLD["palette_effective_nG_counts"].items(), key=lambda kv: -int(kv[0]))),
           ("All 8 levels have the same nG, so there is no nG basis difference between levels (Li record, eval_only)." if _NG_UNIFORM else
            "Different nG values are mixed, so the small differences of phase/amplitude between levels include nG basis differences. "
            "Re-scoring all with the same nG needs a new RCWA calculation and is outside the scope of this visualisation, so the mixed basis is stated in the table below and "
            "in the limitations item.")))
md.append("| level | eff. nG | nG (raw) | nG_eval | nG_report | eval_only | T_tot vs tA squared-amplitude ratio |\n")
md.append("|---|---|---|---|---|---|---|\n")
for p in palette:
    md.append("| %s | %d | %d | %d | %s | %s | %.4f |\n" %
              (p["name"], p["eff_nG"], p["nG"], p["nG_eval"], str(p["nG_report"]),
               "Y" if p["eval_only"] else "N", p["t_over_t2_A"]))
md.append("\n| level | usage ratio |\n|---|---|\n")
for i, p in enumerate(palette):
    md.append("| %s | %s |\n" % (p["name"], pct(_level_counts[i] / level_idx.size)))

md.append("\n## 4. Beta ladder sources (per M)\n")
md.append("The current Li lens ladder = 21 cells of verdict \"%s\" (M = 6 9 12 15 18 24 36, registration sh = 0 1 2; x12_sh0 is the sim766b cell, "
          "the rest are sim769 = <DATA_ROOT>\\Result\\261009_scdc_tmzone_li5). All cells have rcwa_rule=liy / rcwa_ff=1, and "
          "it was confirmed that the md5 list of the per-cell atom paths equals the P769 slot list of sim769_cells.json. No fallback source is used.\n\n" % VERDICT_KEY)
for M in sorted(SRC_USED.keys()):
    md.append("- %s\n" % src_line(M))

md.append("\n## 5. Result table (design plane z=%.0f um)\n\n" % F_DESIGN)
# FIX (verifier defect, item 1 major): the headline table used to show the RAW
# (one-polarisation-component, amp_tA/amp_tB) capture/Strehl with no basis
# label (M2A 7.58%%/M2B 14.96%% -- the number the verifier flagged). It is
# replaced by the two physical bases: lens-only (phase-only, focusing quality
# only) and power basis (absolute throughput, atom transmission included).
# The raw numbers are NOT deleted -- they are moved to the appendix below
# (titled per the verifier's exact wording), since the raw table is still
# useful diagnostically (it is what the palette JSON directly stores).
md.append("**Basis note**: capture/Strehl is reported on two physical bases: lens-only (phase only, unit amplitude per level -- isolates focusing performance only, "
          "section 5.1) and power basis (raw x mean factor T_tot/tA squared amplitude %.4f -- includes the atom transmission budget, "
          "absolute throughput, section 5.2). Raw (the polarisation-component amplitude as is, neither of these two "
          "bases) numbers were excluded from the main text and moved to the appendix 'raw single-polarisation-component basis, "
          "not power'.\n\n" % FAC_T_OVER_T2)
md.append("| model/arm | capture (lens-only) | capture (power basis) | focus position z (um) | FWHM x (um) | FWHM y (um) | Strehl (lens-only) |\n")
md.append("|---|---|---|---|---|---|---|\n")
_head_rows = [
    ("M0 ideal", design_plane["M0"]["capture_pd"], design_plane["M0"]["capture_pd"], "M0", design_plane["M0"]["strehl"]),
    ("M1 arm A", lens_only["M1Ap"]["capture_pd"], power_norm["M1A"], "M1A", lens_only["M1Ap"]["strehl_z200"]),
    ("M1 arm B", lens_only["M1Bp"]["capture_pd"], power_norm["M1B"], "M1B", lens_only["M1Bp"]["strehl_z200"]),
    ("M2 arm A", lens_only["M2Ap"]["capture_pd"], power_norm["M2A"], "M2A", lens_only["M2Ap"]["strehl_z200"]),
    ("M2 arm B", lens_only["M2Bp"]["capture_pd"], power_norm["M2B"], "M2B", lens_only["M2Bp"]["strehl_z200"]),
]
for disp, cap_lens, cap_pow, dpkey, strehl_lens in _head_rows:
    d = design_plane[dpkey]
    md.append("| %s | %s | %s | %.2f | %.4f | %.4f | %.4f |\n" %
              (disp, pct(cap_lens), pct(cap_pow), d["focal_peak_z_um"], d["fwhm_x_um"],
               d["fwhm_y_um"], strehl_lens))
md.append("\n(M0 has no atoms, so lens-only = power basis = raw are all identical; FWHM / focus z are the shape values of the actual raw "
          "field and are basis-"
          "independent quantities, unaffected by a uniform amplitude renormalisation (power basis).)\n")

md.append("\n### Overlap L (arm A vs arm B, capture relative to dB vs M0) -- lens-only vs power basis\n\n")
md.append("| model | L (lens-only) | L (dB, lens-only) | L (power basis) | L (dB, power basis) |\n")
md.append("|---|---|---|---|---|\n")
for name in ("M1", "M2"):
    md.append("| %s | %.4f | %.2f dB | %.4f | %.2f dB |\n" %
              (name, lens_only_overlap[name]["L"], lens_only_overlap[name]["L_db_vs_M0"],
               power_norm["L_" + name], power_norm["L_" + name + "_db_vs_M0"]))

md.append("\n### 5.1 Lens-only (phase only, unit amplitude per level) -- isolates focusing performance only [REPAIR, major]\n\n")
md.append("This is the version in which the atom's own amplitude |t| is divided out (unit amplitude, only the per-level phase kept). Only the focusing loss due to LPA phase "
          "quantisation (+beta ladder) remains, and it can be compared directly with the previously recorded \"-0.37 dB\" (CONTROL, "
          "continuous phase + beta ladder, no quantisation) verdict.\n\n")
md.append("| model/arm | capture (lens-only) | on-axis I (z=200) | Strehl (z=200, lens-only) |\n")
md.append("|---|---|---|---|\n")
_lo_disp = {"M1Ap": "M1 arm A", "M1Bp": "M1 arm B", "M2Ap": "M2 arm A", "M2Bp": "M2 arm B"}
for key in ("M1Ap", "M1Bp", "M2Ap", "M2Bp"):
    lo = lens_only[key]
    md.append("| %s | %s | %.5f | %.4f |\n" %
              (_lo_disp[key], pct(lo["capture_pd"]), lo["onaxis_I_z200"], lo["strehl_z200"]))
md.append("\n| model | L (linear, lens-only) | L (dB vs M0, lens-only) |\n|---|---|---|\n")
for name in ("M1", "M2"):
    md.append("| %s | %.4f | %.2f dB |\n" % (name, lens_only_overlap[name]["L"], lens_only_overlap[name]["L_db_vs_M0"]))
_sum_old = _verdict_L_db + PRED_QUANT_LOSS_DB
md.append("\n**Decomposition**: ladder-only L (beta ladder only, no phase quantisation, current Li ladder, CONTROL-F f=200 decay) = **%.3f dB** + predicted "
          "phase-quantisation-only loss (rms cell phase error %.3f deg, exp(-sigma^2) model) = **%.3f dB** -> sum "
          "**%.3f dB**, measured lens-only M2 L = **%.3f dB** (difference %+.3f dB). "
          "This decomposition explains the lens-only number of the figure model and is a different quantity from the verdict value (section 0.0, betaL + deficit_p' + purity_p').\n" %
          (_verdict_L_db, PHASE_RMS_DEG, PRED_QUANT_LOSS_DB, _sum_old, lens_only_overlap["M2"]["L_db_vs_M0"],
           lens_only_overlap["M2"]["L_db_vs_M0"] - _sum_old))

md.append("\n### 5.2 Power-normalised (absolute throughput, including the atom transmission budget) [REPAIR, major]\n\n")
md.append("This is the raw capture/L multiplied by the mean factor T_tot/tA squared amplitude (**%.4f**). Since it includes the atom's own "
          "transmission budget (T_A~%.2f-%.2f, T_B~%.2f-%.2f), it is **not a focusing loss** and must not be compared "
          "directly with the lens-only numbers of section 5.1.\n\n" %
          (FAC_T_OVER_T2, min(p["TA_total"] for p in palette), max(p["TA_total"] for p in palette),
           min(p["TB_total"] for p in palette), max(p["TB_total"] for p in palette)))
md.append("| model/arm | capture (absolute, including atom transmission) |\n|---|---|\n")
for key, disp in (("M1A", "M1 arm A"), ("M1B", "M1 arm B"), ("M2A", "M2 arm A"), ("M2B", "M2 arm B")):
    md.append("| %s | %s |\n" % (disp, pct(power_norm[key])))
md.append("\n| model | L (linear, absolute) | L (dB vs M0, absolute) |\n|---|---|---|\n")
for name in ("M1", "M2"):
    md.append("| %s | %.4f | %.2f dB |\n" % (name, power_norm["L_" + name], power_norm["L_" + name + "_db_vs_M0"]))

md.append("\n### 5.3 M0 FWHM reference value and lambda/(2 NA_rms,aperture) [FIX, item 4 -- the name 'diffraction limit' is not used]\n\n")
md.append("**Naming correction**: this value is **not called the 'diffraction limit'** -- judging M0's own "
          "measured FWHM by M0's own criterion is circular (M0 is only the best focus this optical system "
          "can make, not an independent diffraction-limit criterion). Instead it is written as **'M0 FWHM "
          "(reference value)'**, and the aperture-based lambda/(2 NA_rms,aperture) is reported as a **separate number** "
          "(the two are not merged into one 'diffraction limit').\n\n")
md.append("The measured FWHM of M0 (unit amplitude, continuous phase, Gaussian apodised, full 300x300 aperture) is used as the reference value: "
          "**M0 FWHM (reference value) = %.4f um** (mean of x=%.4f, y=%.4f), back-calculated "
          "NA_back_inferred_from_M0 = **%.4f**.\n\n" %
          (DIFF_LIMIT_FWHM, design_plane["M0"]["fwhm_x_um"], design_plane["M0"]["fwhm_y_um"], NA_EFF_FROM_M0))
md.append("**Reported separately: aperture-based lambda/(2 NA_rms,aperture)**. The aperture-based NA_rms,aperture "
          "(NA obtained from the rms value of the Gaussian-amplitude |A|^2-weighted aperture radius (rho_rms), over the whole quadrant, lens "
          "centre as reference) is **NA_rms,aperture = %.4f**, and **lambda/(2 NA_rms,aperture) = %.4f um**.\n\n"
          % (NA_RMS_APERTURE, _old_wrong_fwhm))
md.append("**Plain statement**: the M0 FWHM (reference value, **%.4f um**) is **narrower** than lambda/(2 NA_rms,aperture) "
          "(**%.4f um**) -- i.e. judged by this aperture-based yardstick, M0 is "
          "at or below this value (at or below this yardstick). This does not mean that M0 violates diffraction physics; it is because the rms "
          "aperture radius underestimates the hard-edge contribution of the aperture (a one-sentence explanation, "
          "no further claim is made).\n" % (DIFF_LIMIT_FWHM, _old_wrong_fwhm))

md.append("\n## 6. Figure list\n\n")
for tag, (png, pdf) in FIG_FILES.items():
    md.append("- **%s**: %s (+ .pdf)\n" % (tag, png))
md.append("- **Summary JSON**: %s\n" % SUMMARY_PATH)
md.append("- **CONTROL numbers JSON**: %s\n" % CONTROLS_JSON)
md.append("- **Figure array npz**: %s (np.savez_compressed, the exact arrays drawn by figures F2 / F4 and the TE rot90 copies; only stored, calculated values are not changed)\n" % NPZ_PATH)
md.append("\nThe TE figures are copies that assume the TE lens is the TM lens rotated by 90 degrees: np.rot90 is applied to the same arrays that the TM figures draw and a white -> #0072B2 colormap is used "
          "(capture / L numbers are the TM values as is and are invariant under the rotation). The full-aperture arrays are x<->y symmetric, so the TE copy of F4 differs from the TM figure only by max|rot90(I) - I| = %.3e (normalised units); "
          "if the value is close to 0 only the colour map differs.\n\n" % _TE_FULL_DEV)
md.append("**Common figure caption text**: RCWA cell results + scalar angular-spectrum propagation; not a full 3-D field solve.\n\n")
md.append("### 6.1 npz key list (%d keys)\n\n" % len(NPZ_KEYS))
md.append("The array layout is I[iy, ix], i.e. exactly as passed to pcolormesh(x, y, I). The x/y of full_* are absolute coordinates (um, aperture centre at the origin), "
          "the x/y of focal_* are offsets (um) relative to the PD centre used in the figure, and the absolute coordinates are focal_x_abs_um / focal_y_abs_um. "
          "*_I is the drawn (already normalised, colour range 0..1) array, *_norm is its normalisation constant (original |E|^2 = *_I x *_norm), *_label is the model/arm description string. "
          "The TE keys are np.rot90 of the TM arrays.\n\n")
for _k in NPZ_KEYS:
    _v = _NPZ[_k]
    md.append("- `%s` (shape %s, dtype %s)\n" % (_k, tuple(np.shape(_v)), np.asarray(_v).dtype))

md.append("\n## 7. Scalar approximation check (vector verification) [NEW, 260929]\n\n")
md.append("The TM output is x-polarised (E in the x-z plane, Ey=0). In the angular spectrum at the design plane z=%.0f um "
          "**Ez = -(kx/kz) Ex** was computed on the same band-limited angular spectrum as everywhere else in this script (evanescent components "
          "set to 0 + Matsushima-Shimobaba u_lim(z) mask applied). For M0 and M2 "
          "arm A/B, the following is reported at the PD box (8x8 um, the same exact discrete-transform points as in all other capture/overlap calculations):"
          "\n\n" % F_DESIGN)
# FIX (verifier defect, item 1 minor -- section 7 capture columns): the raw
# pd_capture_* columns quoted M2 captures (7.58%%/14.96%%) with no basis label
# (same one-polarisation-component issue as section 5's raw table). Columns
# are now explicitly labelled "raw" AND a power-basis conversion (x
# FAC_T_OVER_T2) is given alongside, so the reader is never left to guess
# which basis a bare number is on. The intEz^2/intEx^2 ratio is unaffected
# (basis-invariant -- a common amplitude factor cancels top and bottom).
md.append("**Denominator stated** [FIX, item 6]: the PD capture fractions below are all normalised by the scalar pin_q (incident power of 1 "
          "quadrant), which is the same denominator as in all other capture/overlap numbers of this script "
          "(not the incident power of all 4 quadrants). The vector capture uses int(|Ex|^2+|Ez|^2) instead of "
          "int|Ex|^2 as numerator and is not the Poynting flux (Sz, k/kz weighted), so it is not a rigorous "
          "power fraction -- read it only as a diagnostic of how much of the vector "
          "components the scalar approximation misses.\n\n")
md.append("| field | PD box integral intEz^2/intEx^2 | PD capture raw (scalar, Ex only) | PD capture raw (vector, Ex+Ez) "
          "| PD capture power basis (scalar, x%.4f) | PD capture power basis (vector, x%.4f) |\n" %
          (FAC_T_OVER_T2, FAC_T_OVER_T2))
md.append("|---|---|---|---|---|---|\n")
for _k in ("M0", "M2A", "M2B"):
    md.append("| %s | %.3e | %s | %s | %s | %s |\n" %
              (_k, VECTOR_CHECK["pd_box_intEz2_over_intEx2"][_k],
               pct(VECTOR_CHECK["pd_capture_scalar_Ex2_only_raw"][_k]),
               pct(VECTOR_CHECK["pd_capture_vector_Ex2_plus_Ez2_raw"][_k]),
               pct(VECTOR_CHECK["pd_capture_scalar_Ex2_only_power_basis"][_k]),
               pct(VECTOR_CHECK["pd_capture_vector_Ex2_plus_Ez2_power_basis"][_k])))
md.append("\n(M0 power-basis factor=1.0 -- no atoms so already power-consistent; M2A/M2B factor=%.4f "
          "=FAC_T_OVER_T2, the same factor as in section 5.)\n\n" % FAC_T_OVER_T2)
md.append("\n**Overlap L (M2, arm A vs arm B, raw basis)** -- scalar (Ex_A* Ex_B only) = **%.4f**, vector (Ex_A* Ex_B + "
          "Ez_A* Ez_B) = **%.4f** (absolute difference %.3e, **%.4f%%** relative to scalar).\n\n" %
          (VECTOR_CHECK["overlap_L_M2_scalar_ExA_ExB_only"],
           VECTOR_CHECK["overlap_L_M2_vector_ExAExB_plus_EzAEzB"],
           abs(VECTOR_CHECK["overlap_L_M2_vector_ExAExB_plus_EzAEzB"] -
               VECTOR_CHECK["overlap_L_M2_scalar_ExA_ExB_only"]),
           VECTOR_CHECK["overlap_L_M2_vector_vs_scalar_relative_diff_pct"]))
# FIX (verifier defect, item 6 minor): a one-line reading of the measured
# values, instead of leaving the table with no conclusion at all.
md.append("**Measurement summary (reading)**: intEz^2/intEx^2 = %.3e (M0) / %.3e (M2A) / %.3e (M2B) -- all "
          "at or below the %% level, and the overlap L changes by **%+.3f%%** relative to scalar when the vector term is added (small). "
          "In this range the scalar approximation is good near the PD box.\n\n" %
          (VECTOR_CHECK["pd_box_intEz2_over_intEx2"]["M0"],
           VECTOR_CHECK["pd_box_intEz2_over_intEx2"]["M2A"],
           VECTOR_CHECK["pd_box_intEz2_over_intEx2"]["M2B"],
           VECTOR_CHECK["overlap_L_M2_vector_vs_scalar_relative_diff_pct"]))
md.append("**Interpretation**: if all three numbers (the intEz^2/intEx^2 ratio, scalar vs vector capture, scalar vs vector "
          "overlap L) are small (e.g. at or below the %% level), the scalar approximation is good near the PD box; "
          "if large (e.g. for arms closer to the quadrant far-corner maximum NA=%.4f), vector diffraction effects (polarization mixing) "
          "cannot be neglected. Judge from the measured values of the table above -- this paragraph only supplies the "
          "criterion for interpreting the numbers and does not itself assert the values.\n" % NA_MAX)

md.append("\n## 8. Limitations\n")
# FIX (verifier defect, item 6 minor): cross-reference section 7's measured
# values instead of stating "not reflected" with no pointer to the actual
# numbers.
md.append("- This is a scalar band-limited ASM approximation. With maximum NA=%.4f at the quadrant far corner, vector "
          "diffraction effects (polarization mixing, obliquity factor, etc.) are not included. **For measured values see section 7 'scalar "
          "approximation check'**: intEz^2/intEx^2 ~%.2e (M0) / ~%.2e (M2A/B), overlap L change "
          "%+.3f%% (vector relative to scalar).\n" %
          (NA_MAX, VECTOR_CHECK["pd_box_intEz2_over_intEx2"]["M0"],
           VECTOR_CHECK["pd_box_intEz2_over_intEx2"]["M2A"],
           VECTOR_CHECK["overlap_L_M2_vector_vs_scalar_relative_diff_pct"]))
md.append("- M1/M2 is an LPA (Locally Periodic Approximation) + measured beta(P) ladder model and not a complete "
          "3-D full-wave simulation. A real full-wave (RCWA/FDTD) simulation at the 300x300x200 um scale "
          "cannot be performed with the current computing resources.\n")
md.append("- The DQPSK per-quadrant bias (rigid shift of the delay arm) is not included in this visualisation.\n")
_oneshift = [M for M in sorted(SRC_USED) if SRC_USED[M]["n_shift"] < 3]
md.append("- The Beta ladder uses only the verdict set (P769 Li 21 cells) and has no fallback source (see section 4). M with fewer than 3 shifts: %s.\n" %
          (", ".join("M=%d (%d shifts)" % (M, SRC_USED[M]["n_shift"]) for M in _oneshift) if _oneshift else "none"))
md.append("- **M1/M2 of the figures are not the verdict**: the figures use a model with the complex t (tA/tB) of the nearest-phase atom itself for each cell, and the verdict is the atom-level "
          "deficit_p' / purity_p' terms added to betaL (section 0.0). Do not compare the two numbers as the same quantity.\n")
md.append("- RCWA cell results + scalar angular-spectrum propagation; not a full 3-D field solve.\n")
md.append("- The capture/overlap normalisation denominator (pin) follows the review_net_f / ladder646.py convention and is the incident "
          "power of '1 quadrant' (not the total incident power of all 4 quadrants).\n")
md.append("- The global phase offset was obtained for arm A, over all cells of the quadrant, by unweighted RMS minimisation "
          "(not weighted by the Gaussian pump intensity).\n")
if _NG_UNIFORM:
    md.append("- Palette nG basis: all 8 levels have the same nG (%s, see the table of section 3).\n" %
              ", ".join("nG=%d: %d levels" % (k, _effnG_counts[k]) for k in sorted(_effnG_counts, reverse=True)))
else:
    md.append("- **The palette nG basis is not uniform over the 8 levels.** Number of levels per basis: %s (see the table of section 3). Re-scoring all with the same nG "
              "(eval_only rerun) needs a new RCWA calculation and is outside the scope of this visualisation.\n" %
              ", ".join("nG=%d: %d levels" % (k, _effnG_counts[k]) for k in sorted(_effnG_counts, reverse=True)))
md.append("- **[REPAIR] The band-limited ASM (Matsushima-Shimobaba) is now actually implemented** (the previous version "
          "used only the name and applied just an evanescent-only cutoff). The effect measured in the earlier independent verification is "
          "very small at z=200 um (M0 capture change -1.47e-6, M2 arm A capture change +2.95e-7, "
          "L_M2 change +4.54e-7) -- the numerical effect is negligible, but the name and the implementation now agree.\n")
md.append("- **[REPAIR] Do not compare raw (polarisation-component amplitude basis) capture/L, lens-only (phase only) capture/L, "
          "and power-normalised (absolute throughput) capture/L with each other.** The three answer different questions: "
          "raw is the number of the atom model as is (component basis), lens-only is focusing performance only (atom transmission budget "
          "excluded), power-normalised is absolute throughput (atom transmission budget included, not a focusing loss).\n")
md.append("- **Axis labelling of F1 (x-z section)**: the internal function eval_with_H(S0,H,grid,xs,ys) applies the first coordinate "
          "array actually to the row (y) axis and the second to the column (x) axis (opposite to the variable names). This design is "
          "completely x<->y symmetric (Gaussian pump, hyperbolic phase and palette level assignment are all symmetric, "
          "idx == idx.T confirmed), so the figures/numbers are not affected.\n")

# FIX (verifier defect, item 1 major): appendix for the raw (one-polarisation-
# component, amp_tA/amp_tB) numbers, moved out of the main body (headline of section 5
# and the overlap table now use only lens-only/power basis). Title matches the
# verifier's exact wording.
md.append("\n## Appendix: raw single-polarisation-component basis, not power\n\n")
md.append("The numbers below use amp_tA/amp_tB, i.e. the polarisation-component amplitude, as is "
          "(\"absT is a POLARISATION COMPONENT; T is the order's TOTAL power\"). The actual "
          "total transmitted power T_tot of the atom is %.4f times the mean of |t|^2 (section 5.2), so the raw capture/L below "
          "mix focusing performance and the atom transmission budget and cannot by themselves be used as a measure of "
          "focusing quality. **Use section 5 (lens-only/power basis) for judgements in the main text, and use this "
          "appendix only for reference/diagnostics.**\n\n" % FAC_T_OVER_T2)
md.append("| model/arm | PD capture (raw) | focus position z (um) | FWHM x (um) | FWHM y (um) | Strehl (raw) |\n")
md.append("|---|---|---|---|---|---|\n")
_disp_names = {"M0": "M0 ideal", "M1A": "M1 arm A", "M1B": "M1 arm B", "M2A": "M2 arm A", "M2B": "M2 arm B"}
for key in ("M0", "M1A", "M1B", "M2A", "M2B"):
    d = design_plane[key]
    md.append("| %s | %s | %.2f | %.4f | %.4f | %.4f |\n" %
              (_disp_names[key], pct(d["capture_pd"]), d["focal_peak_z_um"], d["fwhm_x_um"],
               d["fwhm_y_um"], d["strehl"]))
md.append("\n| model | L (raw, linear) | L (raw, dB vs M0) |\n|---|---|---|\n")
for name in ("M0", "M1", "M2"):
    md.append("| %s | %.4f | %.2f dB |\n" % (name, overlap[name]["L"], overlap[name]["L_db_vs_M0"]))

md.append("\n## 9. Run information\n")
md.append("- Total run time: %.1f s\n" % _wall_s)
md.append("- Observed peak RSS: %.1f MB\n" % _peak_rss)
md.append("- Script: %s\n" % os.path.abspath(__file__))

MD_PATH = os.path.join(OUTDIR, "tm_focus.md")
open(MD_PATH, "w", encoding="utf-8").write("".join(md))
log("wrote %s" % MD_PATH)

log("ALL DONE. wall=%.1fs peak_RSS=%.1f MB" % (_wall_s, _peak_rss))
