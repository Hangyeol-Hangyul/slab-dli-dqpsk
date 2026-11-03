# -*- coding: utf-8 -*-
r"""lensdir45_readout_v6_xe201.py -- COPY of readout45_v5_sim800_d15\lensdir45_readout_v5_sim800_d15.py (md5 49ad66b5d2ef058a58fb27f7728b1f1a, untouched) with MEASURED x e201 reference nodes below P = 4 Lambda.
v6 (2026-10-16, task readout45_v6_xe201; the diff is in v5_to_v6.diff).  v5 normalised each d cell by the x e201 node interpolated in 1/P over XREF_M = (12, 15, 18, 24, 36); below the first node (P < 4 Lambda) interp_c / interp_r CLAMP, so d12 (P 2.828)
  and d15 (P 3.536) were normalised by the x12 node and the x12 echo offset held constant.  Changes against v5 (arithmetic, enumeration, shares, gates and echo formulas are v5 verbatim):
  (Y1) OUT = readout45_v6_xe201\ (the only write location); V5_DIR points at the v5 folder (read-only, only referenced).
  (Y2) XREF_M = (6, 9, 12, 15, 18, 24, 36): x06 (P = 2 Lambda) and x09 (P = 3 Lambda) e201 reference nodes are MEASURED; d12 (2.828) lies between x06 and x09, d15 (3.536) between x09 and x12, so the clamp is gone.  d18 / d24 / d36 (P >= 4.243) and every y anchor use the same
       interval of the same nodes as in v5 and must stay bit-identical.
  (Y3) xref(): M = 6 -> Result\261011_scdc_tmzone_licolp3\261011_sim780_g_x06_sh{0,1,2}_li_e201 (sim780, nG 390), M = 9 -> Result\261016_scdc_xe201_ref\261016_sim802_x09e201_sh{0,1,2}_l1550p000 (sim802, nG 582); M = 12..36 paths as in v5.  The tuple assert
       ("liy", 1, -1, "x", 201) holds for every reference cell; for M = 6, 9 also: wavelength 1550 nm (superramp_result.json stores NO wavelength key, so it is the lams_nm of the job_*.txt in the cell folder, no lam_eval_override), sc_shift and the
       atom list equal to the sim769 x%02d sh%d eq-401 cell (the ONLY tolerated difference: the blank_122p5 slot with its archive / re-pointed location), M atoms.
  (Y4) text: the XREF_M comment, the (V1) selftest check wording and the d-direction say() no longer describe x06 / x09 as reference-less.
  (Y5) output names carry v6 (lensdir45_readout_v6_result / lensdir45_verdict_1overP_v6 / lensdir45_readout_v6_xe201_selftest.json), provenance 'stdout' text.
v5 header follows (v5 file names in it are v5 names).
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
lensdir45_readout_v5_sim800_d15.py -- COPY of readout45_v4_sim800\lensdir45_readout_v4_sim800.py (md5 1e4c9bb0cf86a9aa178a4bb6682fef28, untouched) for the run WITH d15 (no --exclude-d15).
v5 (2026-10-15 21:5x, update of Figure 6(d)(a) with the sim800 d12 AND d15 anchors; task readout45_v5_sim800_d15).  Changes against v4 (the diff is in v4_to_v5.diff):
  (X1) OUT = readout45_v5_sim800_d15\ (the only write location); V4_DIR points at the v4 folder (read-only, only referenced).
  (X2) output names carry v5: lensdir45_readout_v5_result(.json) / lensdir45_verdict_1overP_v5(.json) / lensdir45_readout_v5_sim800_d15_selftest.json.
  (X3) (W8) 'nearest' reference map: carried (pending) anchors are skipped, and when several measured anchors of a psi have the same 1/P-nearest x class (d15 and d18 are both nearest to x12)
       the anchor closest to that class in 1/P is taken (|1/P - 3/M| minimal).  With d12 / d15 forced pending this is the v3 rule (one candidate per class, bit-identical to the stored v3
       nearest map); in v4 the same defect (a carried d15 shadowing d18 in first-match order) was repaired in the external nearest_ref_fixed.py.  Reference map only; the verdict does not use it.
  (X4) selftest: the d_cands check also accepts the sim800 d12 / d15 folder names (nothing else changed).
  (X5) provenance 'stdout' text.
  d12 AND d15 are read from LD\261015_sim800_d%02d_p%d_li_e201 when superramp_result.json exists (it does since 20:50 / 20:53); --exclude-d15 / --force-pending still exist for the regression mode.
v4 header follows (v4 file names in it are v4 names).
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
lensdir45_readout_v4_sim800.py -- COPY of readout45\lensdir45_readout_v3.py (md5 bdfc538f882a94b38dad0ccb2a89d778, untouched) extended by the sim800 45-degree cells d12 / d15.   json only + the lens evaluator; NO solver.
v4 (2026-10-15, update of Figure 6(d)(a) with the newest anchors; task readout45_v4_sim800).  Changes against v3 (every other line is v3 verbatim; the diff is in v3_to_v4.diff):
  (W1) D_CLASSES = (12, 15, 18, 24, 36); d cell folders: sim781 d18 / d24 / d36 as in v3 (explicit candidates first), sim800 d12 / d15 = LD\261015_sim800_d%02d_p%d_li_e201 (+ any <base>* re-run).
  (W2) read_d_cell: the atom-list reference cell of class M is path769(M, 0) as in v3, EXCEPT x12 where the sim800 d12 jobs were built from the sim769 x12 sh1 list (x12 sh0 is the sim766 iso cell with another list); the reference used is printed.
  (W3) the search for job files of unfinished cells is restricted to the result folders; an unfinished d class is carried at ratio 1 exactly as in v3.
  (W4) flags: --force-pending 12,15 (treat those d classes as pending even if a result exists: the REGRESSION mode that must reproduce the stored v3 verdict bit for bit),
       --pending-ok 15 (a pending class in this list does not refuse the run and does not make the verdict INTERIM; it is listed in the verdict as not included),
       --exclude-d15 (alias: --force-pending 15 --pending-ok 15), --anchors-only (read the anchors, print the table, write <out>_anchors.json, no enumeration).
  (W5) output names carry v4 (outputs go to readout45_v4_sim800\ only); the verdict json gets a 'sources' list (path, md5, role of every superramp_result.json that was opened) and 'other_inputs'.
  (W6) processes are pinned to logical cores 14-15 AFTER every import / evaluator load; BLAS / OpenMP threads = 1 (environment set before numpy is imported).
  (W7) selftest: the count checks that were written for three d classes are written for len(D_CLASSES) (nothing else changed).
v3 header follows (v3 file names in it are v3 names).
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
lensdir45_readout_v3.py -- [TM] lens-direction read-out WITH the 45-degree (d) cells of sim781.  json only + the lens evaluator; NO solver.
v3 (2026-10-12, 10:5x: the 45-degree verdict is judged on the 1/P map): copy of lensdir45_readout_v2.py (md5 4dc60e32, untouched).  The 45-degree verdict is judged on the GLOBAL 1/P map:
  (V5) PRIMARY map = global 1/P for BOTH the TOTAL (rmap="global", unchanged arithmetic of v2) AND the share-weighted aperture echo (new emap="ip_db": the d echo of class M is
       the measured d anchors of the psi interpolated in 1/P at P = M/3 in dB (primary) or in power ('ip_pow', printed beside it), held beyond the end anchors, only when the zone of M holds
       a measured anchor -- exactly the construction of VERIFY_FINAL.md / vfy_final.py "ip_db" / "ip_pow").  v2 computed the echo of EVERY map on the zone bins (emap="zone"), which is the
       optimistic echo map (VERIFY_FINAL F1).
  (V6) the zone map (TOTAL -0.6453 / -0.6664, zone-bin echo) and the nearest-class map are printed as REFERENCES only; the zone map is the pessimistic TOTAL bound (bin edge at P = 6).
  (V7) one VERDICT line: "VERDICT (1/P map, 2026-10-12): TOTAL <decay> / <held> dB vs -0.65 -> PASS/FAIL ; echo <NE> dB (quadrant-agnostic <max>) vs -20.5 -> PASS/FAIL".
  (V8) the sensitivity of VERIFY_FINAL.md section 4 is computed here (d36 magnitude +-0.1 dB, d36 rB0 -0.1 dB on the 1/P map; echo under a d echo step) and printed under the verdict.
  (V9) outputs: lensdir45_readout_v3_result.json (full), lensdir45_verdict_1overP.json (verdict), lensdir45_readout_v3_selftest.json; files are created exclusively (never overwritten).
  evaluate(..., emap="zone") keeps the v2 default, so every v2 self-test check is unchanged.
python -I -B lensdir45_readout_v3.py [--selftest] [--partial] [--out NAME] [--no-nearest] [--no-zone] [--no-sens]
--- v2 header follows (v2 file names in it are v3 names now) ---
v2 (verifier, 2026-10-12, VERIFY.md): copy of lensdir45_readout.py (md5 3288a278, untouched) with four changes, arithmetic of x / y / d / zone / echo unchanged:
  (V1) TOTAL variant 'global' (rmap="global"): a class whose zone holds at least one MEASURED d anchor of that psi gets the d ratio interpolated in 1/P over ALL
       measured anchors of that psi (across zones; log|r|, arg r; held beyond the outer anchors) -- the same 1/P rule the lens evaluator uses for beta(P) and the
       one LENSDIR_PLAN.md:201 planned.  Printed beside 'zone' (primary, conservative) and 'nearest' (lower bound).  --no-global skips it.
  (V2) breakeven_d36 plants ONLY the d36 psi that is still pending (v1 replaced a measured d36 psi by the planted one if only one psi was pending).
  (V3) d cell discovery: after the explicit candidates, every folder LD\<base>* holding a readable superramp_result.json is a candidate (a re-run of the
       memory-tripped d36 psi315 under a new suffix is found); the folder used is printed, and a WARNING is printed when more than one folder has a result.
  (V4) output names carry _v2 (selftest json, default result json) so the implementer's files are never overwritten.
Copy of gap_plan_261010\lensdir\lensdir_readout.py (x + y, frozen) extended by the d direction; the original is untouched.

Directions and what the frozen rule does with them (memory project_scdc_lpa_lens_criterion_260926: "B(P): worst orientation for every angle"):
  x    adopted -x nodes (sim769, eq-401 for x06..x24, x36 eq-201, with the -x rB0 node) = the base.
  y    sim779 y12..y24 (e201) vs the paired sim779 x e201 cells, class by class (M identical, P identical).        [unchanged from lensdir_readout.py]
  d    sim781 d18 / d24 / d36 at psi 225 (diag_sign +1) and psi 315 (diag_sign -1), e201, licol; psi 225 == psi 135 (d-), psi 315 == psi 45 (d+)
       (y-mirror equivalence, all 8 P769 atoms carry sym_y = 1).  THE d CELL IS NOT AT AN x PERIOD, so it is read against the x e201 nodes
       INTERPOLATED IN 1/P (power_centre basis, K0_TM both arms; log|beta| and arg linear in 1/P between the x classes x12 x15 x18 x24 x36).
Local period (units of Lambda = 1.31539 um): the d cell has L1 = (0, n a), L2 = (Lambda, -Lambda ds), a = Lambda/3; the ramp order (ramp_s, 0) has the
  wave vector ramp_s * 2 pi Lk1, Lk1 = (L2y, -L2x)/det = (ds, 1)/(n a), so  P_d = 1/|Lk1| = n a / sqrt(2) = n / (3 sqrt 2) Lambda  and
  psi = atan2(ramp_s Lk1y, ramp_s Lk1x).  It is derived from the lattice vectors of the result JSON (or of the job file for a pending cell) and checked against
  the formula.  d18 / d24 / d36 = 4.243 / 5.657 / 8.485 Lambda.
Zone of a P (the class-to-zone map of lens_total_li.ZMAP written as intervals): [2,3) 2-3L | [3,4) 3-4L | [4,6) 4-6L | [6,12] 6-12L | >12 -> checked equal to ZMAP
  for every x class.  d18, d24 -> 4-6L; d36 -> 6-12L (d24 is NOT in the zone of x18).
TOTAL (worst orientation, decided on the TOTAL itself as in F4 of REVIEW.md): per class M the options are x (the adopted node), y (if a y anchor exists), d225, d315
  (if a d anchor exists in the zone of the class).  d option of class M = the anchors of that zone for that psi: one anchor -> its ratio; two or more -> ratio
  interpolated in 1/P (log|r|, arg r) at P = M/3, held beyond the outer anchors.  Ratio r = beta_d(P_d) / beta_x,e201(P_d) per arm, plus the rB0 ratio rR; node_d(M) = adopted node(M) * r
  (the same transfer the y readout does: the ratio is read on one basis and put on the adopted -x node).  Every combination of per-class choices is evaluated
  (L.betaL, decay and held tail), the minimum per TOTAL component is reported, ties keep x.  A class with no option keeps its -x node.
  Variant 'nearest' (sensitivity only): every d anchor replaces ONLY the 1/P-nearest x class (d18 -> x12, d24 -> x18, d36 -> x24).
ECHO (261011: share-weighted over orientations): e_c = 10 log10( s_x Px + s_y Py + s_dm Pdm + s_dp Pdp ) per class, class echoes -> L.aperture (zone power mean, ZONE_W, >12L at the gate).
  s_* = shares of the zone from te_beta_orientation_261005\oblique_class_share_stdout.txt (NE quadrant, f 200): s_y = +y + -y, s_dm = 135 + 225 (psi 225 cell), s_dp = 45 + 315 (psi 315 cell),
  s_x = 1 - s_y - s_dm - s_dp (= -x + +x, assigned the adopted x echo).  Py = y cell echo + (e_x401 - e_x201) of the paired class (as the y readout), Pdm / Pdp = power mean of the d anchors of the zone
  (echo + the same 201 -> 401 offset interpolated in 1/P), a zone without a measured d anchor keeps the class x echo.  d- and d+ swap in the NW / SW quadrants (psi -> 180 - psi), so 'share_swap'
  evaluates the other assignment; 'worst' = max over x, y, d225, d315 per class (conservative).  PENDING d cells (--partial only) are carried at their x values: ratio 1, echo = x.
Modes: --selftest   d switched off must reproduce the stored x (lens_total_li5_final.json) and y (lensdir_readout_result.json) results BIT FOR BIT; d carried / d identity; needles.
       (default)    real read-out; REFUSES unless every d cell has superramp_result.json (and the Li needles of sim779 / sim780 passed).
       --partial    pending d / y cells are carried at x; loud INTERIM banner; NOT a verdict.      --out NAME  base name of the result json (default lensdir45_readout_result).
(v2 usage line, superseded by the v3 line at the top: --no-global does not exist in v3, the global map is the primary one.)"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):    # (W6) one BLAS thread, set before numpy is imported
    os.environ[_k] = "1"
import cmath
import datetime
import hashlib
import io
import itertools
import json
import math
import re
import struct
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, r"<WORK_ROOT>\loop_261008\lens_li")
import lens_total_li as L                                                     # noqa: E402  (pins 10-17)
sys.path.insert(0, r"<WORK_ROOT>\te_beta_orientation_261005\power_basis_261006")
import pb2_common as C                                                        # noqa: E402  (pins >= 10)
import psutil                                                                 # noqa: E402


def _pin():
    pass


_pin()

R = r"<DATA_ROOT>\Result"
SIM = r"<DATA_ROOT>"
NEW = R + r"\261011_scdc_tmzone_dir"                                          # sim779: y cells + x e201 pairs
LD = R + r"\261011_scdc_tmzone_licoldir"                                      # sim781: d cells
P3 = R + r"\261011_scdc_tmzone_licolp3"                                       # sim780: Phase 3 needles
OUT = r"<WORK_ROOT>\loop_261008\lens45_li_261011\readout45_v6_xe201"                 # (Y1) the only write location
V5_DIR = r"<WORK_ROOT>\loop_261008\lens45_li_261011\readout45_v5_sim800_d15"           # (Y1) read-only (v5 result, reference only)
V4_DIR = r"<WORK_ROOT>\loop_261008\lens45_li_261011\readout45_v4_sim800"             # read-only (v4 result, reference only)
V3_DIR = r"<WORK_ROOT>\loop_261008\lens45_li_261011\readout45"                   # read-only (stored v3 verdict / result)
Y_STORED = r"<WORK_ROOT>\loop_261008\gap_plan_261010\lensdir\lensdir_readout_result.json"
X_STORED = r"<WORK_ROOT>\loop_261008\lens_li5\lens_total_li5_final.json"
SHARE_FILE = r"<WORK_ROOT>\te_beta_orientation_261005\oblique_class_share_stdout.txt"
T769_PATH = r"<WORK_ROOT>\loop_261008\improve_261009\tm_lens_total_a1\a1_def_pur_p769_result.json"
T769 = json.load(io.open(T769_PATH, encoding="utf-8"))
DFC, PRT = T769["deficit"], T769["pur_f200"]
K0 = C.K0_TM
Y_SHARE = {"2-3L": 0.301, "3-4L": 0.301, "4-6L": 0.145, "6-12L": 0.249}          # as lensdir_readout.py (y = +y + -y); re-derived from SHARE_FILE below and cross-checked
SELFTEST = "--selftest" in sys.argv
PSIS = (225, 315)
DS_OF = {225: +1, 315: -1}
D_CLASSES = (12, 15, 18, 24, 36)                                              # (W1) sim800 d12 / d15 added to sim781 d18 / d24 / d36
XREF_M = (6, 9, 12, 15, 18, 24, 36)                                           # (Y2) x classes that have MEASURED e201 cells (sim780 x06, sim802 x09, sim779 x12..x24, sim769 x36)
ORDER = ("y", "d225", "d315")


def _argv_list(flag):
    if flag in sys.argv:
        return [int(t) for t in sys.argv[sys.argv.index(flag) + 1].split(",") if t.strip()]
    return []


FORCE_PENDING = set(_argv_list("--force-pending")) | ({15} if "--exclude-d15" in sys.argv else set())          # (W4) d classes treated as pending even if a result exists
PENDING_OK = set(_argv_list("--pending-ok")) | ({15} if "--exclude-d15" in sys.argv else set())                  # (W4) pending classes that neither refuse the run nor make it INTERIM
USED = {}                                                                      # (W5) path -> sorted roles, every superramp_result.json that was opened


def _use(p, role):
    USED.setdefault(os.path.normpath(p), set()).add(role)


def path769(M, sh):
    if (M, sh) == (12, 0):
        return R + r"\261009_scdc_tmzone_iso\261009_sim766_iso_x12_sh0_212_257acc"
    return R + r"\261009_scdc_tmzone_li5\261009_sim769_lensli5_x%02d_sh%d" % (M, sh)


def jl(p):
    _use(p, "json")
    return json.load(io.open(os.path.join(p, "superramp_result.json"), encoding="utf-8"))


def beta_pc(p):
    _use(p, "beta_pc")
    c = C.read_cell(p, +1)
    out = []
    for a in ("A", "B"):
        b = c[a]["b_ac"] * (cmath.exp(-2j * math.pi * c["sc"] / 3.0) if a == "A" else 1.0)
        out.append(C.convert(b, c[a]["kappa"], a, c["G_delta"], "power_centre", k0=K0))
    return out, c


def node_pc(paths):
    bs = [beta_pc(p)[0] for p in paths]
    return [sum(b[k] for b in bs) / len(bs) for k in (0, 1)]


def pmean_e(paths):
    return L.pmean([jl(p)["needle_results"]["production"]["echo_p2_db"] for p in paths])


def r0_of(jj):
    q = jj["needle_results"]["production"]
    return complex(*q["amp_rB_zero"]) / complex(q["lpa"]["rB0"]["a_pred_re"], q["lpa"]["rB0"]["a_pred_im"])


def say(s=""):
    print(s, flush=True)


# ---------------------------------------------------------------- adopted -x nodes (amp_corner, lens_total_li basis) and echoes ----------------------------------------------
ND, NR, EX = {}, {}, {}
for _M in L.CLASSES:
    _a, _r, _e = L.read_class({sh: path769(_M, sh) for sh in (0, 1, 2)})
    ND[_M], NR[_M], EX[_M] = _a, _r, L.pmean(_e)
    for _sh in (0, 1, 2):
        _use(path769(_M, _sh), "adopted_x_node (L.read_class)")
L.evaluator()                                                                 # load the ladder646 head now (it re-pins to 10-17), then pin 10-19 again
_pin()
ZCLASS = {c: z for z, cl in L.ZMAP.items() for c in cl}                       # "x12" -> "4-6L"


def zone_of(P):
    """Class-to-zone map as intervals in P (units of Lambda); equals L.ZMAP for every x class (checked below)."""
    if P < 3.0:
        return "2-3L"
    if P < 4.0:
        return "3-4L"
    if P < 6.0:
        return "4-6L"
    if P <= 12.0:
        return "6-12L"
    return ">12L"


assert all(zone_of(M / 3.0) == ZCLASS["x%02d" % M] for M in L.CLASSES), "zone_of disagrees with L.ZMAP"


# ---------------------------------------------------------------- shares (NE quadrant) ------------------------------------------------------------------------------------------
def read_shares():
    rows = {}
    for line in io.open(SHARE_FILE, encoding="utf-8"):
        m = re.match(r"^\(([0-9.e+]+), ([0-9.e+]+)\]\s+([0-9.]+)\s+(.+)$", line.strip())
        if m:
            v = [float(t) for t in m.group(4).split()]
            if len(v) == 8:
                rows[(m.group(1), m.group(2))] = v                            # +x(0) 45 +y(90) 135 -x(180) 225 -y(270) 315, % of the band
    band = {"2-3L": ("2", "4"), "3-4L": ("2", "4"), "4-6L": ("4", "6"), "6-12L": ("6", "12")}
    sh = {}
    for z, k in band.items():
        v = rows[k]
        sh[z] = dict(x=(v[0] + v[4]) / 100.0, y=(v[2] + v[6]) / 100.0, dm=(v[3] + v[5]) / 100.0, dp=(v[1] + v[7]) / 100.0, band=k, raw=v)
        assert abs(sh[z]["y"] - Y_SHARE[z]) < 1e-9, (z, sh[z]["y"], Y_SHARE[z])      # same y share as the frozen y readout
    return sh


SHARES = read_shares()
D_SHARE = {z: (SHARES[z]["dm"], SHARES[z]["dp"]) for z in SHARES}


# ---------------------------------------------------------------- interpolation in 1/P (log|z| and arg linear, held beyond the ends) ----------------------------------------------
def _t(P, P0, P1):
    return (1.0 / P0 - 1.0 / P) / (1.0 / P0 - 1.0 / P1)


def interp_r(P, pts):
    pts = sorted(pts, key=lambda a: a[0])
    if P <= pts[0][0]:
        return pts[0][1]
    if P >= pts[-1][0]:
        return pts[-1][1]
    for (P0, v0), (P1, v1) in zip(pts, pts[1:]):
        if P0 <= P <= P1:
            return v0 + _t(P, P0, P1) * (v1 - v0)


def interp_c(P, pts):
    pts = sorted(pts, key=lambda a: a[0])
    if P <= pts[0][0]:
        return pts[0][1]
    if P >= pts[-1][0]:
        return pts[-1][1]
    for (P0, z0), (P1, z1) in zip(pts, pts[1:]):
        if P0 <= P <= P1:
            t = _t(P, P0, P1)
            lm = math.log(abs(z0)) + t * (math.log(abs(z1)) - math.log(abs(z0)))
            ph = cmath.phase(z0) + t * cmath.phase(z1 / z0)
            return cmath.rect(math.exp(lm), ph)


# ---------------------------------------------------------------- lens total with a cache on the node values -------------------------------------------------------------------
_BC = {}


def _bl(nd):
    key = tuple((M, nd[M]["A"], nd[M]["B"]) for M in sorted(nd))
    v = _BC.get(key)
    if v is None:
        v = L.betaL(nd)
        _BC[key] = v
    return v


def total(nodes, nr=None):
    nr = NR if nr is None else nr
    b = _bl({M: {"A": nodes[M]["A"], "B": nodes[M]["B"] * nr[M]} for M in L.CLASSES})
    return b[0] + DFC + PRT, b[1] + DFC + PRT


def _build(ch, opts):
    nodes, nr = {}, {}
    for M in L.CLASSES:
        o = ch.get(M)
        if o is None:
            nodes[M], nr[M] = dict(ND[M]), NR[M]
        else:
            rA, rB, rR = opts[M][o]
            nodes[M] = {"A": ND[M]["A"] * rA, "B": ND[M]["B"] * rB}
            nr[M] = NR[M] * rR
    return nodes, nr


def nearest_class(P):
    return min(L.CLASSES, key=lambda M: abs(1.0 / P - 3.0 / M))


def ratio_for_class(M, psi, dan, rmap="zone"):
    """(rA, rB, rR) of the d option of class M for psi, or None.  zone: anchors of the class's zone, 1 -> as is, >= 2 -> 1/P interpolation at M/3 (held beyond); nearest: only the 1/P-nearest class."""
    ans = dan.get(psi, [])
    if rmap == "zone":
        z = ZCLASS["x%02d" % M]
        a = [x for x in ans if x["zone"] == z]
        if not a:
            return None
        if len(a) == 1:
            return (a[0]["r"][0], a[0]["r"][1], a[0]["rR"])
        P = M / 3.0
        return (interp_c(P, [(x["P"], x["r"][0]) for x in a]), interp_c(P, [(x["P"], x["r"][1]) for x in a]), interp_c(P, [(x["P"], x["rR"]) for x in a]))
    if rmap == "global":                                                       # (V1) zone must hold a measured anchor; interpolation over all measured anchors of this psi
        meas = [x for x in ans if not x["carried"]]
        z = ZCLASS["x%02d" % M]
        if not any(x["zone"] == z for x in meas):
            return None
        P = M / 3.0
        return (interp_c(P, [(x["P"], x["r"][0]) for x in meas]), interp_c(P, [(x["P"], x["r"][1]) for x in meas]), interp_c(P, [(x["P"], x["rR"]) for x in meas]))
    cand = [x for x in ans if not x["carried"] and nearest_class(x["P"]) == M]       # (X3 / W8) measured anchors whose 1/P-nearest x class is M
    if not cand:
        return None
    x = min(cand, key=lambda a: abs(1.0 / a["P"] - 3.0 / M))                      # several candidates (d15 and d18 -> x12): the one closest to the class in 1/P
    return (x["r"][0], x["r"][1], x["rR"])


def zone_d_echo(dan, psi, z):
    v = [x["e"] for x in dan.get(psi, []) if x["zone"] == z and not x["carried"] and x["e"] is not None]
    return L.pmean(v) if v else None


def ip_d_echo(dan, psi, M, mode):
    """(V5) d echo of class M on the global 1/P map: the MEASURED d anchors of this psi (all zones, echo basis-corrected) interpolated linearly in 1/P at P = M/3, held beyond the end anchors.
    mode 'db' = linear in dB (primary), 'pow' = linear in power.  None when the zone of M holds no measured anchor (same condition as ratio_for_class(rmap='global'))."""
    meas = [x for x in dan.get(psi, []) if not x["carried"] and x["e"] is not None]
    if not any(x["zone"] == ZCLASS["x%02d" % M] for x in meas):
        return None
    P = M / 3.0
    if mode == "db":
        return interp_r(P, [(x["P"], x["e"]) for x in meas])
    return 10 * math.log10(interp_r(P, [(x["P"], 10 ** (x["e"] / 10.0)) for x in meas]))


def make_opts(anchors, dan, rmap="zone"):
    opts = {}
    for M in L.CLASSES:
        o = {}
        if M in anchors:
            rA, rB = anchors[M]["r"]
            o["y"] = (rA, rB, anchors[M].get("rR", 1.0))
        for psi in PSIS:
            q = ratio_for_class(M, psi, dan, rmap)
            if q is not None and not (q[0] == 1 and q[1] == 1 and q[2] == 1):          # a ratio of exactly 1 is the x node itself: no option
                o["d%d" % psi] = q
        if o:
            opts[M] = o
    return opts


EMAPS = ("zone", "ip_db", "ip_pow")                                            # (V5) echo maps: zone bins (v2) | 1/P in dB (primary) | 1/P in power


def echoes(anchors, dan, swap=False, emap="zone"):
    """class echoes (worst, share) for the aperture; with no d anchor measured in the zone of a class the formula reduces to the y readout's expression (bit-identical).
    emap 'zone' = v2 (power mean of the d anchors inside the zone bin); 'ip_db' / 'ip_pow' (V5) = d anchors interpolated in 1/P at P = M/3 (ip_d_echo)."""
    assert emap in EMAPS, emap
    ec_w, ec_s = {}, {}
    for M in L.CLASSES:
        c = "x%02d" % M
        ex = EX[M]
        z = ZCLASS[c]
        ey = anchors[M]["ey"] if (M in anchors and anchors[M].get("ey") is not None) else None
        if emap == "zone":
            ed = {psi: zone_d_echo(dan, psi, z) for psi in PSIS}
        else:
            ed = {psi: ip_d_echo(dan, psi, M, "db" if emap == "ip_db" else "pow") for psi in PSIS}
        has_d = any(v is not None for v in ed.values())
        if ey is None and not has_d:
            ec_w[c] = ec_s[c] = ex
            continue
        s_y = Y_SHARE[z]
        s_dm, s_dp = D_SHARE[z] if has_d else (0.0, 0.0)
        if swap:
            s_dm, s_dp = s_dp, s_dm
        ey_ = ex if ey is None else ey
        P_ = (1 - s_y - s_dm - s_dp) * 10 ** (ex / 10) + s_y * 10 ** (ey_ / 10)
        if has_d:
            edm = ex if ed[225] is None else ed[225]
            edp = ex if ed[315] is None else ed[315]
            P_ = P_ + s_dm * 10 ** (edm / 10) + s_dp * 10 ** (edp / 10)
        ec_s[c] = 10 * math.log10(P_)
        ec_w[c] = max([ex] + ([ey] if ey is not None else []) + [v for v in ed.values() if v is not None])
    return ec_w, ec_s


def evaluate(anchors, dan=None, rmap="zone", verbose=False, emap="zone", opts_filter=None):
    """anchors: y anchors {M: dict(r=[rA, rB], rR, ey)}; dan: {psi: [d anchor dicts]}.  Worst orientation decided on the TOTAL itself: every per-class choice, minimum per component (ties keep x).
    emap (V5): echo map of the aperture ('zone' = v2 default, 'ip_db' / 'ip_pow' = 1/P); it never touches the TOTAL.
    opts_filter (V8, sensitivity only): function opts -> reduced opts (the 54-combination set of VERIFY_FINAL.md section 4); None = every option (v2)."""
    dan = dan or {225: [], 315: []}
    opts = make_opts(anchors, dan, rmap)
    if opts_filter is not None:
        opts = opts_filter(opts)
    keys = sorted(opts)
    names = {M: [n for n in ORDER if n in opts[M]] for M in keys}
    ncomb = sum(math.prod(len(names[M]) for M in use) for n in range(len(keys) + 1) for use in itertools.combinations(keys, n))
    best = [None, None]
    k = 0
    t0 = time.time()
    for n in range(len(keys) + 1):
        for use in itertools.combinations(keys, n):
            for nm in itertools.product(*[names[M] for M in use]):
                ch = dict(zip(use, nm))
                t = total(*_build(ch, opts))
                for i in (0, 1):
                    if best[i] is None or t[i] < best[i][0] - 1e-15:
                        best[i] = (t[i], ch)
                k += 1
                if verbose and ncomb >= 100 and k % 100 == 0:
                    say("    ... %d / %d combinations (%.0f s)" % (k, ncomb, time.time() - t0))
    tot = (best[0][0], best[1][0])
    used = {}
    for M in keys:
        u = {}
        for nm, (rA, rB, rR) in opts[M].items():
            u[nm] = dict(A_dB=20 * math.log10(abs(rA)), B_dB=20 * math.log10(abs(rB)), dphase_deg=[math.degrees(cmath.phase(rA)), math.degrees(cmath.phase(rB))],
                         rB0_dB=20 * math.log10(abs(rR)), rB0_deg=math.degrees(cmath.phase(rR)))
        used[M] = dict(options=u, chosen=[best[0][1].get(M, "x"), best[1][1].get(M, "x")])
        if M in anchors:                                                       # the y readout's own keys (kept for the bit-for-bit comparison)
            rA, rB = anchors[M]["r"]
            used[M].update(y_minus_x_dB=[20 * math.log10(abs(rA)), 20 * math.log10(abs(rB))], dphase_deg=[math.degrees(cmath.phase(rA)), math.degrees(cmath.phase(rB))],
                           rB0_y_minus_x_dB=20 * math.log10(abs(anchors[M].get("rR", 1.0))), y_worse=bool(best[0][1].get(M) == "y" or best[1][1].get(M) == "y"))
    ec_w, ec_s = echoes(anchors, dan, emap=emap)
    ec_w_sw, ec_sw = echoes(anchors, dan, swap=True, emap=emap)
    return dict(total=tot, margin=[t - L.GATE_TOTAL for t in tot], used=used, echo_worst=L.aperture(ec_w), echo_share=L.aperture(ec_s), echo_share_swap=L.aperture(ec_sw),
                class_echo_worst=ec_w, class_echo_share=ec_s, class_echo_share_swap=ec_sw, n_combinations=ncomb, emap=emap)


def echo_summary(anchors, dan, emap):
    """(V5) echo side only (no TOTAL enumeration): aperture NE / swapped / quadrant-agnostic (max, the worse of the two) / worst-orientation, class echoes, zone echoes (power mean of the zone classes)."""
    ec_w, ec_s = echoes(anchors, dan, emap=emap)
    ec_w2, ec_sw = echoes(anchors, dan, swap=True, emap=emap)
    ne, sw = L.aperture(ec_s), L.aperture(ec_sw)
    zone_e = {z: (L.GATE_ECHO if z == ">12L" else L.pmean([ec_s[c] for c in L.ZMAP[z]])) for z in L.ZONE_W}
    return dict(emap=emap, echo_share=ne, echo_share_swap=sw, echo_quadrant_agnostic=max(ne, sw), echo_worst=L.aperture(ec_w), class_echo_share=ec_s, class_echo_share_swap=ec_sw,
                class_echo_worst=ec_w, zone_echo_share=zone_e)


def single_options(anchors, dan, rmap="zone"):
    """TOTAL when ONLY one class takes one option (all other classes keep x) -- shows which anchor binds."""
    opts = make_opts(anchors, dan, rmap)
    rows = []
    for M in sorted(opts):
        for nm in [n for n in ORDER if n in opts[M]]:
            t = total(*_build({M: nm}, opts))
            rows.append(dict(cls=M, option=nm, total=list(t), delta=[t[0] - base_total()[0], t[1] - base_total()[1]]))
    return rows


_BASE = {}


def base_total():
    if "t" not in _BASE:
        _BASE["t"] = total(ND, NR)
    return _BASE["t"]


# ---------------------------------------------------------------- bit-for-bit helpers ------------------------------------------------------------------------------------------
def _bits(a, b):
    """exact equality of two json-like structures (floats by their 8-byte pattern); returns the list of differing paths"""
    bad = []

    def rec(x, y, p):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in x:
                if k not in y:
                    bad.append(p + "/" + str(k) + " missing in stored")
                else:
                    rec(x[k], y[k], p + "/" + str(k))
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                bad.append(p + " length %d vs %d" % (len(x), len(y)))
            else:
                for i, (u, v) in enumerate(zip(x, y)):
                    rec(u, v, p + "[%d]" % i)
        elif isinstance(x, float) and isinstance(y, float):
            if struct.pack("<d", x) != struct.pack("<d", y):
                bad.append("%s %r vs %r" % (p, x, y))
        elif isinstance(x, bool) or isinstance(y, bool) or isinstance(x, str) or isinstance(y, str):
            if x != y:
                bad.append("%s %r vs %r" % (p, x, y))
        elif isinstance(x, int) and isinstance(y, int):
            if x != y:
                bad.append("%s %r vs %r" % (p, x, y))
        else:
            bad.append("%s type %s vs %s" % (p, type(x).__name__, type(y).__name__))
    rec(a, b, "")
    return bad


def jsonable(o):
    return json.loads(json.dumps(o, default=lambda z: [z.real, z.imag] if isinstance(z, complex) else (sorted(z) if isinstance(z, set) else str(z))))


# ---------------------------------------------------------------- y anchors (code of lensdir_readout.py, arithmetic unchanged) -----------------------------------------------------
def y_anchors(verbose=True):
    anchors, missing = {}, []
    for M in (9, 12, 15, 18, 24, 36):
        yp = os.path.join(NEW, "261011_sim779_dir_y%02d_e201" % M)
        xp = [os.path.join(NEW, "261011_sim779_dir_x%02de201_sh%d" % (M, sh)) for sh in (0, 1, 2)] if M != 36 else [path769(36, sh) for sh in (0, 1, 2)]
        have = os.path.isfile(os.path.join(yp, "superramp_result.json")) and all(os.path.isfile(os.path.join(p, "superramp_result.json")) for p in xp)
        if not have:
            missing.append(M)
            continue
        by, cy = beta_pc(yp)
        bx = node_pc(xp)
        j = jl(yp)
        assert (j.get("rcwa_rule"), j.get("rcwa_ff")) == ("liy", 1) and j["ramp_axis_requested"] == "y", yp
        for p in xp:
            jx = jl(p)
            assert (jx.get("rcwa_rule"), jx.get("rcwa_ff"), jx.get("ramp_s"), jx.get("ramp_axis_requested")) == ("liy", 1, -1, "x"), p
            assert M == 36 or jx.get("nG_ref") == j.get("nG_ref") == 201, (p, jx.get("nG_ref"), j.get("nG_ref"))
        rR = r0_of(j) / (sum(r0_of(jl(p)) for p in xp) / len(xp))
        ky = [cy[a]["kappa"] / (K0 / cy["cos_t"]) for a in ("A", "B")]
        ey = j["needle_results"]["production"]["echo_p2_db"] + (EX[M] - pmean_e(xp))
        anchors[M] = dict(r=[by[0] / bx[0], by[1] / bx[1]], rR=rR, ey=ey, kappa_y_over_closed=ky, x201_vs_x401_echo=EX[M] - pmean_e(xp))
        if verbose:
            say("  y%02d: y-x A %+.3f B %+.3f dB, dphase %+.1f / %+.1f deg, rB0 y-x %+.3f dB, kappa_y/closed %.4f / %.4f, echo y %.2f (basis-corrected) vs x %.2f" % (
                M, 20 * math.log10(abs(anchors[M]["r"][0])), 20 * math.log10(abs(anchors[M]["r"][1])), math.degrees(cmath.phase(anchors[M]["r"][0])),
                math.degrees(cmath.phase(anchors[M]["r"][1])), 20 * math.log10(abs(rR)), ky[0], ky[1], ey, EX[M]))
    return anchors, missing


# ---------------------------------------------------------------- x e201 references for the d anchors ---------------------------------------------------------------------------
_XR = {}


XREF_X06 = P3 + "\\261011_sim780_g_x06_sh%d_li_e201"                                 # (Y3) sim780 x06 e201 (P = 2 Lambda, nG 390)
XREF_X09 = R + "\\261016_scdc_xe201_ref\\261016_sim802_x09e201_sh%d_l1550p000"       # (Y3) sim802 x09 e201 (P = 3 Lambda, nG 582)
BLANK_SLOT = "\\_palette\\blank_122p5\\rho3_opt.npy"                                  # (Y3) the one atom path that was re-pointed (archive -> byte-identical copy) in some sim802 x06 jobs


def check_new_xref(M, sh, p):
    """(Y3) measured x06 / x09 e201 reference cell: wavelength 1550 nm, same registration and atom list as the sim769 x%02d sh%d eq-401 cell.  superramp_result.json stores no wavelength key,
    so the wavelength is the lams_nm of the (single) job_*.txt in the cell folder.  Returns the list of atom slots that differ (only the blank_122p5 slot may)."""
    import glob
    jf = sorted(glob.glob(os.path.join(p, "job_*.txt")))
    assert len(jf) == 1, ("expected exactly one job_*.txt in the reference cell", p, jf)
    kv = parse_job(jf[0])
    assert abs(float(kv["lams_nm"]) - 1550.0) < 1e-9 and "lam_eval_override" not in kv, ("reference cell is not the 1550.00 nm design wavelength", p, kv.get("lams_nm"), kv.get("lam_eval_override"))
    j, j769 = jl(p), jl(path769(M, sh))
    a, b = j["atoms_requested"], j769["atoms_requested"]
    assert len(a) == len(b) == M, ("atom count", p, len(a), len(b), M)
    assert int(j.get("sc_shift", 0)) == int(j769.get("sc_shift", 0)) == sh, ("registration differs from the sim769 cell", p, j.get("sc_shift"), j769.get("sc_shift"), sh)
    diff = [i for i in range(M) if a[i] != b[i]]
    for i in diff:
        assert a[i].endswith(BLANK_SLOT) and b[i].endswith(BLANK_SLOT), ("atom list differs from sim769 x%02d sh%d beyond the blank_122p5 location" % (M, sh), p, i, a[i], b[i])
    return diff


def xref():
    if not _XR:
        for M in XREF_M:
            if M == 6:                                                         # (Y3)
                xp = [XREF_X06 % sh for sh in (0, 1, 2)]
            elif M == 9:                                                       # (Y3)
                xp = [XREF_X09 % sh for sh in (0, 1, 2)]
            else:
                xp = [os.path.join(NEW, "261011_sim779_dir_x%02de201_sh%d" % (M, sh)) for sh in (0, 1, 2)] if M != 36 else [path769(36, sh) for sh in (0, 1, 2)]
            for sh, p in enumerate(xp):
                jx = jl(p)
                assert (jx.get("rcwa_rule"), jx.get("rcwa_ff"), jx.get("ramp_s"), jx.get("ramp_axis_requested"), jx.get("nG_ref")) == ("liy", 1, -1, "x", 201), p
                if M in (6, 9):                                                # (Y3)
                    check_new_xref(M, sh, p)
            _XR[M] = dict(P=M / 3.0, node=node_pc(xp), r0=sum(r0_of(jl(p)) for p in xp) / len(xp), e201=pmean_e(xp), off=EX[M] - pmean_e(xp), paths=xp)
    return _XR


# ---------------------------------------------------------------- d geometry and d anchors --------------------------------------------------------------------------------------
def lattice_formula(n, lam, ds):
    a = lam / 3.0
    return (0.0, n * a), (lam, -lam * ds)


def ramp_geometry(L1, L2, ramp_s):
    """period of the ramp order along its direction (um) and its direction psi (deg): Lk1 = (L2y, -L2x)/det as kbloch.py (no 2 pi)"""
    det = L1[0] * L2[1] - L1[1] * L2[0]
    k1 = (L2[1] / det, -L2[0] / det)
    return 1.0 / math.hypot(k1[0], k1[1]), math.degrees(math.atan2(ramp_s * k1[1], ramp_s * k1[0])) % 360.0


def d_cands(M, psi):
    import glob
    if M in (12, 15):                                                          # (W1) sim800: no suffix
        base = "261015_sim800_d%02d_p%d_li_e201" % (M, psi)
        sfx = ("",)
    else:                                                                      # sim781 as in v3
        base = "261011_sim781_d%02d_p%d_li_e201" % (M, psi)
        sfx = {18: ("_bt12_r", "_bt12"), 24: ("_bt12",), 36: ("_bt14",)}[M]
    explicit = [LD + "\\" + base + s for s in sfx]
    extra = [p for p in sorted(glob.glob(os.path.join(LD, base + "*"))) if os.path.isdir(p) and p not in explicit]      # (V3) re-runs under a new suffix
    return explicit + extra


def parse_job(path):
    kv = {}
    for line in io.open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def pending_job_file(M, psi):
    """(W3) result folders only -- queue folders are never read.  An unfinished class without a job file in its result folder gets its geometry from the class label (pending_anchor)."""
    import glob
    for c in d_cands(M, psi):
        hit = sorted(glob.glob(os.path.join(c, "job_*.txt")))
        if hit:
            return hit[-1]
    return None


def read_d_cell(path, M, psi, XR):
    j = jl(path)
    q = j["needle_results"]["production"]
    ds = DS_OF[psi]
    assert j["ramp_axis_requested"] == "d" and q["ramp_axis"] == "d" and j.get("pol") == "TM", path
    assert int(j["diag_sign"]) == ds and int(q["diag_sign"]) == ds, (path, j["diag_sign"], ds)
    assert (j.get("rcwa_rule"), j.get("rcwa_ff")) == ("liy", 1), path
    assert j.get("ramp_s") == -1 and int(j.get("sc_shift", 0)) == 0, path
    assert j.get("li_raster") == [[0, 1], [1, -ds]] and j.get("li_inner_axis") == 0, (path, j.get("li_raster"))
    assert j.get("nG_ref") == 201 and j.get("nG_ref_matched") is True, (path, j.get("nG_ref"), j.get("nG_ref_matched"))
    assert int(q["M"]) == M == len(j["atoms_requested"]), path
    ref_sh = 1 if M == 12 else 0                                               # (W2) sim800 d12 was built from the sim769 x12 sh1 list; x12 sh0 is the sim766 iso cell with another list
    jx = jl(path769(M, ref_sh))
    assert j["atoms_requested"] == jx["atoms_requested"] and q["atom_paths"] == jx["needle_results"]["production"]["atom_paths"], "atom list differs from the sim769 x%02d sh%d list" % (M, ref_sh)
    L1, L2, lam = q["lattice_L1_um"], q["lattice_L2_um"], q["Lx"]
    f1, f2 = lattice_formula(M, lam, ds)
    assert max(abs(L1[0] - f1[0]), abs(L1[1] - f1[1]), abs(L2[0] - f2[0]), abs(L2[1] - f2[1])) < 1e-9, ("lattice differs from the v5 formula", path, L1, L2, f1, f2)
    P_um, psi_g = ramp_geometry(L1, L2, -1)
    P = P_um / lam
    assert abs(psi_g - psi) < 1e-6, ("ramp direction", psi_g, psi)
    b, c = beta_pc(path)
    assert abs(c["P_L"] - P) < 1e-9 and abs(P - M / (3.0 * math.sqrt(2.0))) < 1e-9, (c["P_L"], P)
    r = [b[k] / interp_c(P, [(XR[m]["P"], XR[m]["node"][k]) for m in XREF_M]) for k in (0, 1)]
    r0x = interp_c(P, [(XR[m]["P"], XR[m]["r0"]) for m in XREF_M])
    rR = r0_of(j) / r0x
    off = interp_r(P, [(XR[m]["P"], XR[m]["off"]) for m in XREF_M])
    return dict(cls=M, psi=psi, P=P, zone=zone_of(P), nearest_x="x%02d" % nearest_class(P), r=r, rR=rR, e_raw=q["echo_p2_db"], off=off, e=q["echo_p2_db"] + off, carried=False,
                src=path, psi_from_lattice=psi_g, kappa_over_K0=[c["A"]["kappa"] / K0, c["B"]["kappa"] / K0], nG_actual=q.get("nG_actual"), wall_s=j.get("wall_s"))


def pending_anchor(M, psi):
    jf = pending_job_file(M, psi)
    ds = DS_OF[psi]
    if jf:
        kv = parse_job(jf)
        n = len([a for a in kv["atoms"].split(";") if a.strip()])
        lam = float(kv["period_nm"]) / 1000.0
        assert kv.get("ramp_axis") == "d" and int(kv.get("diag_sign", "0")) == ds and n == M, (jf, kv.get("ramp_axis"), kv.get("diag_sign"), n)
        why = "geometry from the job file %s" % jf
    else:
        n, lam, why = M, C.LAMG, "geometry from the class label (no job file found)"
    L1, L2 = lattice_formula(n, lam, ds)
    P_um, psi_g = ramp_geometry(L1, L2, -1)
    assert abs(psi_g - psi) < 1e-6
    P = P_um / lam
    return dict(cls=M, psi=psi, P=P, zone=zone_of(P), nearest_x="x%02d" % nearest_class(P), r=[1 + 0j, 1 + 0j], rR=1 + 0j, e_raw=None, off=None, e=None, carried=True, src=why, psi_from_lattice=psi_g)


def d_anchors(partial, cands=None, verbose=True):
    """returns (dan, pending [(M, psi)]); a (class, psi) without superramp_result.json in any candidate folder is pending (carried at x only with --partial)"""
    XR = xref()
    dan = {225: [], 315: []}
    pending = []
    for M in D_CLASSES:
        for psi in PSIS:
            cl = (cands or d_cands)(M, psi)
            hit = []
            for p in cl:
                if M in FORCE_PENDING:                                         # (W4) regression / exclusion mode: a result that exists is deliberately not read
                    if verbose and os.path.isfile(os.path.join(p, "superramp_result.json")):
                        say("  d%02d psi%d: result exists in %s but class %d is FORCED PENDING (--force-pending / --exclude-d15) -> not read" % (M, psi, os.path.basename(p), M))
                    continue
                if os.path.isfile(os.path.join(p, "superramp_result.json")):
                    try:
                        jl(p)
                        hit.append(p)
                    except Exception as ex_:                                   # a file that is still being copied to the share is NOT a result yet
                        if verbose:
                            say("  d%02d psi%d: %s exists but is not readable yet (%s) -> treated as pending" % (M, psi, p, type(ex_).__name__))
            if len(hit) > 1 and verbose:
                say("  WARNING d%02d psi%d: %d folders hold a result (%s) -- using the first, %s" % (M, psi, len(hit), "; ".join(os.path.basename(h) for h in hit), os.path.basename(hit[0])))
            if hit:
                a = read_d_cell(hit[0], M, psi, XR)
                _use(hit[0], "d_anchor d%02d psi%d" % (M, psi))
                if verbose:
                    say("  d%02d psi%d [MEASURED] P %.4f L (%s; 1/P-nearest x class %s): d/x_ref(P) A %+.3f B %+.3f dB, dphase %+.1f / %+.1f deg, rB0 d/x_ref %+.3f dB, echo d %.2f (%+.3f offset -> %.2f), kappa/K0 %.4f" % (
                        M, psi, a["P"], a["zone"], a["nearest_x"], 20 * math.log10(abs(a["r"][0])), 20 * math.log10(abs(a["r"][1])), math.degrees(cmath.phase(a["r"][0])),
                        math.degrees(cmath.phase(a["r"][1])), 20 * math.log10(abs(a["rR"])), a["e_raw"], a["off"], a["e"], a["kappa_over_K0"][0]))
                    say("      cell: %s" % a["src"])
            else:
                a = pending_anchor(M, psi)
                pending.append((M, psi))
                if verbose:
                    say("  d%02d psi%d [PENDING -> carried at x: ratio 1, echo = x] P %.4f L (%s; 1/P-nearest x class %s); %s" % (M, psi, a["P"], a["zone"], a["nearest_x"], a["src"]))
            dan[psi].append(a)
    return dan, pending


def refuse_code(pending, partial):
    """exit code main() uses when d cells are pending: 3 (REFUSE) unless --partial.  (W4) classes in --pending-ok do not count."""
    pend = [mp for mp in pending if mp[0] not in PENDING_OK]
    return 3 if (pend and not partial) else 0


def banner_interim(pending):
    say("!" * 118)
    say("!! INTERIM (--partial): d cells still PENDING -> carried at their x values (ratio 1, echo = x): %s" % ", ".join("d%02d psi%d" % mp for mp in pending))
    say("!! THIS IS NOT A VERDICT.  The 6-12 Lambda zone (47.9 % of the lens power, decides both gates) carries NO measured d information while d36 is pending.")
    say("!" * 118)


def show(name, v, ref=None):
    say("  %-34s TOTAL %+.4f / %+.4f dB (margin %+.4f / %+.4f to %.2f) | echo worst %.3f  share(NE) %.3f  share(swapped d+/d-) %.3f dB (gate %.1f)" % (
        name, v["total"][0], v["total"][1], v["margin"][0], v["margin"][1], L.GATE_TOTAL, v["echo_worst"], v["echo_share"], v["echo_share_swap"], L.GATE_ECHO))


# ---------------------------------------------------------------- (V7) verdict, (V8) sensitivity, (V9) exclusive output ---------------------------------------------------------------
def excl_open(path):
    """(V9) create a NEW text file; an existing file is never overwritten (FileExistsError)."""
    return io.open(path, "x", encoding="utf-8")


def verdict_text(total, echo_ne, echo_agn):
    """(V7) the one verdict line.  TOTAL PASS = BOTH tails (decayed and held) at or above the gate; echo PASS = the NE value AND the quadrant-agnostic value (the worse of NE / swapped) at or below the gate."""
    tp = all(t - L.GATE_TOTAL >= 0 for t in total)
    ep = echo_ne <= L.GATE_ECHO and echo_agn <= L.GATE_ECHO
    line = "VERDICT (1/P map, 2026-10-12): TOTAL %.4f / %.4f dB vs %.2f -> %s ; echo %.4f dB (quadrant-agnostic %.4f) vs %.1f -> %s" % (
        total[0], total[1], L.GATE_TOTAL, "PASS" if tp else "FAIL", echo_ne, echo_agn, L.GATE_ECHO, "PASS" if ep else "FAIL")
    return line, tp, ep


def _red(opts):
    """(V8) the reduced option set of VERIFY_FINAL.md section 4 (54 combinations on the final anchors): x12 keeps only y, x15 is dropped, x18 / x24 / x36 keep only their d options
    (the y options of those classes and the d options of x12 / x15 raise the TOTAL when taken alone and never enter the argmin).  Classes left without an option are dropped."""
    out = {}
    for M, v in opts.items():
        if M == 15:
            continue
        w = ({"y": v["y"]} if "y" in v else {}) if M == 12 else {k: x for k, x in v.items() if k != "y"}
        if w:
            out[M] = w
    return out


def perturb_d(dan, cls=(36,), dB=0.0, dph_deg=0.0, dR_dB=0.0, dE_dB=0.0):
    """(V8) copy of the d anchors in which the classes `cls` are shifted: magnitude dB on both arms' ratio, phase +dph on arm A and -dph on arm B, rB0 ratio dR_dB, echo dE_dB.  The input is not modified."""
    g, gr = 10 ** (dB / 20.0), 10 ** (dR_dB / 20.0)
    pa, pb = cmath.exp(1j * math.radians(dph_deg)), cmath.exp(-1j * math.radians(dph_deg))
    out = {}
    for psi in PSIS:
        out[psi] = []
        for a in dan[psi]:
            if a["cls"] in cls:
                a = dict(a, r=[a["r"][0] * g * pa, a["r"][1] * g * pb], rR=a["rR"] * gr, e=(None if a["e"] is None else a["e"] + dE_dB))
            out[psi].append(a)
    return out


def sensitivity(anchors, dan, base_total):
    """(V8) VERIFY_FINAL.md section 4 on the 1/P map, computed here.  TOTAL under d36 node errors with the reduced option set (checked against the full 768-combination set at the base and for
    the adverse magnitude case), the d36 deficit at which each tail reaches the gate, and the echo under a d echo step.  Same rules as the verification (eq-201 convergence size 0.1 dB, phase 1.3 deg)."""
    out = {}

    def red(d_):
        v = evaluate(anchors, d_, rmap="global", emap="ip_db", opts_filter=_red)
        return v["total"], v["n_combinations"]
    say("== SENSITIVITY on the 1/P map (reduced option set of VERIFY_FINAL.md section 4: x12 {x, y}, x15 x, x18 / x24 / x36 {x, d225, d315})")
    t0, n0 = red(dan)
    out["reduced_base"] = dict(total=list(t0), n=n0, diff_vs_full=[t0[i] - base_total[i] for i in (0, 1)])
    say("  reduced-set base TOTAL %.10f / %.10f (%d combinations); full-set TOTAL %.10f / %.10f; difference %.1e / %.1e" % (t0[0], t0[1], n0, base_total[0], base_total[1], t0[0] - base_total[0], t0[1] - base_total[1]))
    cases = (("d36 magnitude -0.1 dB (both arms)", dict(dB=-0.1)), ("d36 magnitude +0.1 dB (both arms)", dict(dB=+0.1)), ("d36 rB0 -0.1 dB", dict(dR_dB=-0.1)),
             ("d36 phase A +1.3 / B -1.3 deg", dict(dph_deg=1.3)))
    out["cases"] = []
    for nm, kw in cases:
        t, n = red(perturb_d(dan, **kw))
        row = dict(case=nm, total=list(t), shift=[t[i] - t0[i] for i in (0, 1)], margin=[t[i] - L.GATE_TOTAL for i in (0, 1)], n=n, gate_pass=[bool(t[i] - L.GATE_TOTAL >= 0) for i in (0, 1)])
        out["cases"].append(row)
        say("  %-36s TOTAL %.4f / %.4f dB (shift %+.4f / %+.4f), margin %+.4f / %+.4f -> decay %s, held %s" % (nm, t[0], t[1], row["shift"][0], row["shift"][1], row["margin"][0], row["margin"][1],
            "PASS" if row["gate_pass"][0] else "FAIL", "PASS" if row["gate_pass"][1] else "FAIL"))
    t_full = evaluate(anchors, perturb_d(dan, dB=-0.1), rmap="global", emap="ip_db")["total"]
    out["full_check_d36_minus0p1"] = dict(total=list(t_full), reduced=out["cases"][0]["total"], diff=[t_full[i] - out["cases"][0]["total"][i] for i in (0, 1)])
    say("  full 768-combination check of 'd36 magnitude -0.1 dB': TOTAL %.10f / %.10f ; reduced %.10f / %.10f ; difference %.1e / %.1e" % (
        t_full[0], t_full[1], out["cases"][0]["total"][0], out["cases"][0]["total"][1], out["full_check_d36_minus0p1"]["diff"][0], out["full_check_d36_minus0p1"]["diff"][1]))
    # d36 deficit at which each tail reaches the gate: linear interpolation base -> -0.1 dB, then a direct evaluation at that deficit [E]
    adv = out["cases"][0]
    be = []
    for i in (0, 1):
        m0 = t0[i] - L.GATE_TOTAL
        be.append(0.1 * m0 / (-adv["shift"][i]) if adv["shift"][i] < 0 else None)
    chk = []
    for i in (0, 1):
        if be[i] is None:
            chk.append(None)
            continue
        tt, _ = red(perturb_d(dan, dB=-be[i]))
        chk.append(list(tt))
    out["breakeven_d36_deficit_dB"] = dict(decay=be[0], held=be[1], check_total_at_decay_deficit=chk[0], check_total_at_held_deficit=chk[1])
    say("  d36 magnitude deficit at which TOTAL reaches %.2f dB: decayed tail %s dB, held tail %s dB [E: linear interpolation, direct evaluation at the held deficit: %s]" % (
        L.GATE_TOTAL, "%.3f" % be[0] if be[0] is not None else "n/a", "%.3f" % be[1] if be[1] is not None else "n/a", ("%.4f / %.4f" % tuple(chk[1])) if chk[1] else "n/a"))
    # echo under a d echo step (quadrant-agnostic aperture = worse of NE / swapped)
    steps = (0.1, 0.3, 0.63, 0.81)
    tab = []
    for st in steps:
        row = dict(step_dB=st)
        for lab, cl in (("d36_only", (36,)), ("all_d", (18, 24, 36))):
            dd = perturb_d(dan, cls=cl, dE_dB=st)
            s1, s0 = echo_summary(anchors, dd, "ip_db"), echo_summary(anchors, dd, "zone")
            row[lab] = dict(ip_db_NE=s1["echo_share"], ip_db_agnostic=s1["echo_quadrant_agnostic"], zone_NE=s0["echo_share"], zone_agnostic=s0["echo_quadrant_agnostic"])
        tab.append(row)
        say("  d echo step %+.2f dB: 1/P map d36 only NE %.3f / all d NE %.3f (agnostic %.3f) ; zone map d36 only %.3f / all d %.3f" % (
            st, row["d36_only"]["ip_db_NE"], row["all_d"]["ip_db_NE"], row["all_d"]["ip_db_agnostic"], row["d36_only"]["zone_NE"], row["all_d"]["zone_NE"]))
    out["echo_steps"] = tab

    def breakeven(cl, emap):
        f = lambda st: echo_summary(anchors, perturb_d(dan, cls=cl, dE_dB=st), emap)["echo_quadrant_agnostic"]
        lo, hi = 0.0, 10.0
        if f(hi) <= L.GATE_ECHO:
            return None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(mid) > L.GATE_ECHO:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    out["echo_breakeven_step_dB"] = dict(ip_db_all_d=breakeven((18, 24, 36), "ip_db"), ip_db_d36_only=breakeven((36,), "ip_db"), zone_all_d=breakeven((18, 24, 36), "zone"))
    say("  uniform d echo rise that brings the quadrant-agnostic aperture to %.1f dB: 1/P map all d %s dB, d36 only %s dB ; zone map all d %s dB (bisection)" % (
        L.GATE_ECHO, *["%.3f" % v if v is not None else ">10" for v in (out["echo_breakeven_step_dB"]["ip_db_all_d"], out["echo_breakeven_step_dB"]["ip_db_d36_only"], out["echo_breakeven_step_dB"]["zone_all_d"])]))
    return out


def SOURCES_FN():
    """(W5) every superramp_result.json the script opened: path, md5, roles"""
    out = []
    for pth in sorted(USED):
        f = os.path.join(pth, "superramp_result.json")
        out.append(dict(path=f, md5=hashlib.md5(io.open(f, "rb").read()).hexdigest(), bytes=os.path.getsize(f), roles=sorted(USED[pth])))
    return out


def OTHER_FN():
    out = []
    for f in (T769_PATH, Y_STORED, X_STORED, SHARE_FILE, os.path.join(V3_DIR, "lensdir45_verdict_1overP.json"),
              r"<WORK_ROOT>\loop_261008\lens_li\lens_total_li.py", r"<WORK_ROOT>\readout_261005_evening\scratch_tm646\ladder646.py"):
        if os.path.isfile(f):
            out.append(dict(path=f, md5=hashlib.md5(io.open(f, "rb").read()).hexdigest(), bytes=os.path.getsize(f)))
    return out


def d_summary(a):
    return dict(cls=a["cls"], psi=a["psi"], P=a["P"], zone=a["zone"], nearest_x=a["nearest_x"], carried=a["carried"], A_dB=20 * math.log10(abs(a["r"][0])), A_deg=math.degrees(cmath.phase(a["r"][0])),
                B_dB=20 * math.log10(abs(a["r"][1])), B_deg=math.degrees(cmath.phase(a["r"][1])), rB0_dB=20 * math.log10(abs(a["rR"])), rB0_deg=math.degrees(cmath.phase(a["rR"])),
                echo_raw_dB=a["e_raw"], echo_offset_dB=a["off"], echo_used_dB=a["e"], kappa_over_K0=a.get("kappa_over_K0"), nG_actual=a.get("nG_actual"), wall_s=a.get("wall_s"), src=a["src"])


# ================================================================ SELFTEST =====================================================================================================
def mk_anchor(M, psi, rdB, edB=None, carried=False, rRdB=0.0, ph_deg=0.0):
    P = M / (3.0 * math.sqrt(2.0))
    g = 10 ** (rdB / 20.0)
    if ph_deg:
        gz = g * cmath.exp(1j * math.radians(ph_deg))
        r = [gz, gz]
    else:
        r = [g, g]
    return dict(cls=M, psi=psi, P=P, zone=zone_of(P), nearest_x="x%02d" % nearest_class(P), r=r if not carried else [1 + 0j, 1 + 0j],
                rR=(10 ** (rRdB / 20.0) if rRdB else 1.0) if not carried else 1 + 0j, e=edB, e_raw=edB, off=0.0, carried=carried, src="planted")


def selftest():
    _sp = os.path.join(OUT, "lensdir45_readout_v6_xe201_selftest.json")             # (Y5)
    if os.path.exists(_sp):                                                    # (V9) never overwrite
        say("REFUSE: output exists, not overwritten: %s" % _sp)
        return 5
    res, ok = {}, True
    t0 = time.time()

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        say("SELFTEST %-62s %s %s" % (name, "OK" if cond else "FAIL", detail))
        res[name] = bool(cond)

    # ---- geometry / zone checks
    check("zone_of reproduces L.ZMAP for all 7 x classes", all(zone_of(M / 3.0) == ZCLASS["x%02d" % M] for M in L.CLASSES))
    for M, ds in ((18, 1), (24, -1), (36, 1)):
        L1, L2 = lattice_formula(M, C.LAMG, ds)
        P_um, psi_g = ramp_geometry(L1, L2, -1)
        check("d%d ds %+d: P = n/(3 sqrt 2) = %.4f, psi %.1f, zone %s" % (M, ds, P_um / C.LAMG, psi_g, zone_of(P_um / C.LAMG)),
              abs(P_um / C.LAMG - M / (3 * math.sqrt(2))) < 1e-12 and abs(psi_g - (225 if ds > 0 else 315)) < 1e-9)
    check("d zones: d12 2-3L, d15 3-4L, d18 4-6L, d24 4-6L, d36 6-12L", [zone_of(M / (3 * math.sqrt(2))) for M in D_CLASSES] == ["2-3L", "3-4L", "4-6L", "4-6L", "6-12L"])
    check("shares: y equals the frozen Y_SHARE (asserted at import), the 4 shares sum to 1 within 0.004 (eight 1-decimal percentages)", all(abs(SHARES[z]["x"] + SHARES[z]["y"] + SHARES[z]["dm"] + SHARES[z]["dp"] - 1.0) < 4e-3 for z in SHARES),
          "(" + ", ".join("%s x%.3f y%.3f d-%.3f d+%.3f" % (z, SHARES[z]["x"], SHARES[z]["y"], SHARES[z]["dm"], SHARES[z]["dp"]) for z in SHARES) + ")")
    pts = [(4.0, 1.0 + 0j), (5.0, 0.5 * cmath.exp(0.4j)), (6.0, 0.2 * cmath.exp(-0.3j))]
    check("interp_c: returns the node at a node, holds beyond the ends, mid-value in 1/P",
          interp_c(5.0, pts) == pts[1][1] and interp_c(3.0, pts) == pts[0][1] and interp_c(9.0, pts) == pts[2][1]
          and abs(abs(interp_c(2.0 / (1 / 4.0 + 1 / 5.0), pts)) - math.sqrt(1.0 * 0.5)) < 1e-12)

    # ---- x: adopted base must be the stored adopted result bit for bit
    stored_x = json.load(io.open(X_STORED, encoding="utf-8"))["results"]["P769 approved (Li terms)"]
    base = evaluate({}, None)
    bad = _bits(dict(total=list(base["total"]), echo=base["echo_worst"], cls=base["class_echo_worst"]),
                dict(total=stored_x["res"]["rB0_complex"]["total"], echo=stored_x["echo_aperture"], cls={"x%02d" % M: stored_x["echo_class"]["x%02d" % M] for M in L.CLASSES}))
    check("x only == stored adopted result (lens_total_li5_final.json), bit for bit", not bad, "%.6f / %.6f echo %.4f" % (base["total"][0], base["total"][1], base["echo_worst"]) + (" DIFF " + str(bad[:3]) if bad else ""))

    # ---- y: stored result bit for bit, from the cells and from the stored anchors
    sy = json.load(io.open(Y_STORED, encoding="utf-8"))
    an, miss = y_anchors(verbose=False)
    bad = _bits(jsonable(dict(anchors=an, missing=miss)), dict(anchors=sy["anchors"], missing=sy["missing"]))
    check("y anchors re-read from the cells == stored lensdir_readout_result.json anchors, bit for bit", not bad, (" DIFF " + str(bad[:3])) if bad else "")
    off_ = {225: [], 315: []}
    v = evaluate(an, off_)
    r0_ = sy["result"]
    bad = _bits(jsonable(dict(total=list(v["total"]), margin=v["margin"], echo_worst=v["echo_worst"], echo_share=v["echo_share"], class_echo_worst=v["class_echo_worst"],
                              class_echo_share=v["class_echo_share"], used={str(M): {k: v["used"][M][k] for k in r0_["used"][str(M)]} for M in an})),
                jsonable(dict(total=r0_["total"], margin=r0_["margin"], echo_worst=r0_["echo_worst"], echo_share=r0_["echo_share"], class_echo_worst=r0_["class_echo_worst"],
                              class_echo_share=r0_["class_echo_share"], used=r0_["used"])))
    check("x + y, d OFF == stored lensdir_readout_result.json result, bit for bit", not bad,
          "TOTAL %.6f / %.6f echo worst %.4f share %.4f" % (v["total"][0], v["total"][1], v["echo_worst"], v["echo_share"]) + ((" DIFF " + str(bad[:3])) if bad else ""))
    # stored anchors fed through evaluate (separates the reading path from the evaluation path)
    an_s = {int(M): dict(r=[complex(*a["r"][0]), complex(*a["r"][1])], rR=complex(*a["rR"]), ey=a["ey"]) for M, a in sy["anchors"].items()}
    v_s = evaluate(an_s, off_)
    check("x + y from the STORED anchors == stored result, bit for bit", not _bits(jsonable(dict(total=list(v_s["total"]), e=v_s["echo_worst"], s=v_s["echo_share"])),
                                                                                  jsonable(dict(total=r0_["total"], e=r0_["echo_worst"], s=r0_["echo_share"]))))

    # ---- d carried everywhere == d off, bit for bit (with and without y)
    carried = {psi: [mk_anchor(M, psi, 0.0, carried=True) for M in D_CLASSES] for psi in PSIS}
    for lab, a_ in (("x", {}), ("x + y", an)):
        v0, v1 = evaluate(a_, off_), evaluate(a_, carried)
        bad = _bits(jsonable(dict(t=list(v0["total"]), w=v0["echo_worst"], s=v0["echo_share"], sw=v0["echo_share_swap"], cw=v0["class_echo_worst"], cs=v0["class_echo_share"])),
                    jsonable(dict(t=list(v1["total"]), w=v1["echo_worst"], s=v1["echo_share"], sw=v1["echo_share_swap"], cw=v1["class_echo_worst"], cs=v1["class_echo_share"])))
        check("%s: d carried at x (all d anchors) == d off, bit for bit" % lab, not bad, (" DIFF " + str(bad[:3])) if bad else "")
    # measured-type d anchors with ratio exactly 1 and echo = x-zone mean: TOTAL bit-identical (options pruned), echo to 1e-12
    ident = {psi: [] for psi in PSIS}
    for psi in PSIS:
        for M in D_CLASSES:
            a_ = mk_anchor(M, psi, 0.0, carried=False)
            a_["r"], a_["rR"] = [1 + 0j, 1 + 0j], 1 + 0j
            a_["e"] = a_["e_raw"] = L.pmean([EX[m] for m in L.CLASSES if ZCLASS["x%02d" % m] == a_["zone"]])
            ident[psi].append(a_)
    v1 = evaluate({}, ident)
    check("measured-type d anchors == x (ratio 1, echo = zone mean): TOTAL bit-identical", not _bits(list(v1["total"]), list(base["total"])))
    check("  ... echo share to 1e-12 dB (zone-mean regrouping), worst unchanged within the x echoes",
          abs(v1["echo_share"] - base["echo_worst"]) < 1e-12 and abs(v1["echo_share_swap"] - base["echo_worst"]) < 1e-12, "share %.12f swap %.12f base %.12f" % (v1["echo_share"], v1["echo_share_swap"], base["echo_worst"]))

    # ---- needles (y off, so the search is small and the oracle is a direct betaL call)
    def oracle(classes, g):
        nodes = {M: {"A": ND[M]["A"] * (g if M in classes else 1.0), "B": ND[M]["B"] * (g if M in classes else 1.0) * NR[M]} for M in L.CLASSES}
        b = L.betaL(nodes)
        return b[0] + DFC + PRT, b[1] + DFC + PRT

    g1 = 10 ** (-1.0 / 20)
    for lab, anc, classes, want in (("d36 psi225 -1 dB -> x18 x24 x36 (zone 6-12L)", {225: [mk_anchor(36, 225, -1.0)], 315: []}, (18, 24, 36), "d225"),
                                    ("d36 psi315 -1 dB -> x18 x24 x36, chooses d315", {225: [], 315: [mk_anchor(36, 315, -1.0)]}, (18, 24, 36), "d315"),
                                    ("d18 psi225 -1 dB alone -> x12 x15 only (zone 4-6L)", {225: [mk_anchor(18, 225, -1.0)], 315: []}, (12, 15), "d225")):
        v = evaluate({}, anc)
        o = oracle(classes, g1)
        ch = [v["used"][M]["chosen"] for M in classes]
        untouched = [M for M in v["used"] if M not in classes]
        check("needle " + lab, abs(v["total"][0] - o[0]) < 1e-12 and abs(v["total"][1] - o[1]) < 1e-12 and all(c == [want, want] for c in ch) and not untouched,
              "dTOTAL %+.4f / %+.4f (oracle %+.4f / %+.4f)" % (v["total"][0] - base["total"][0], v["total"][1] - base["total"][1], o[0] - base["total"][0], o[1] - base["total"][1]))
    def brute(classes, rA, rB, rR):
        """independent oracle: every subset of `classes` substituted by (rA, rB, rR), direct L.betaL, minimum per TOTAL component"""
        best_ = [None, None]
        for n in range(len(classes) + 1):
            for sub in itertools.combinations(classes, n):
                nodes = {M: {"A": ND[M]["A"] * (rA if M in sub else 1.0), "B": ND[M]["B"] * (rB if M in sub else 1.0) * (NR[M] * rR if M in sub else NR[M])} for M in L.CLASSES}
                b = L.betaL(nodes)
                for i in (0, 1):
                    if best_[i] is None or b[i] + DFC + PRT < best_[i]:
                        best_[i] = b[i] + DFC + PRT
        return best_

    for lab, kw, ra, rb, rr in (("rB0-only -1 dB (|r| = 1, rR 0.891)", dict(rdB=0.0, rRdB=-1.0), 1.0, 1.0, 10 ** (-1.0 / 20)),
                                ("phase-only +25 deg (|r| = 1)", dict(rdB=0.0, ph_deg=25.0), cmath.exp(1j * math.radians(25.0)), cmath.exp(1j * math.radians(25.0)), 1.0),
                                ("phase-only -25 deg (|r| = 1)", dict(rdB=0.0, ph_deg=-25.0), cmath.exp(-1j * math.radians(25.0)), cmath.exp(-1j * math.radians(25.0)), 1.0)):
        a_ = mk_anchor(36, 225, **kw)
        v = evaluate({}, {225: [a_], 315: []})
        bb = brute((18, 24, 36), a_["r"][0], a_["r"][1], a_["rR"])
        moved = (v["total"][0] < base["total"][0] - 1e-9) or (v["total"][1] < base["total"][1] - 1e-9)
        check("needle d36 " + lab + ": == brute force over all 8 class subsets", abs(v["total"][0] - bb[0]) < 1e-12 and abs(v["total"][1] - bb[1]) < 1e-12,
              "dTOTAL %+.4f / %+.4f %s" % (v["total"][0] - base["total"][0], v["total"][1] - base["total"][1], "(moves TOTAL)" if moved else "(no move: this sign is better than x)"))
    _a = [evaluate({}, {225: [mk_anchor(36, 225, 0.0, ph_deg=sg * 25.0)], 315: []}) for sg in (+1, -1)]
    check("phase-only needles: at least one sign lowers TOTAL (phase enters the worst-orientation test, REVIEW F4)", any(x["total"][0] < base["total"][0] - 1e-9 for x in _a))
    _b = evaluate({}, {225: [mk_anchor(36, 225, 0.0, rRdB=-1.0)], 315: []})
    check("rB0-only needle lowers TOTAL (rB0 enters, REVIEW F3)", _b["total"][0] < base["total"][0] - 1e-9)
    v = evaluate({}, {225: [mk_anchor(36, 225, +1.0, edB=EX[36] - 1.0)], 315: []})
    check("needle d36 +1 dB (better than x): TOTAL unchanged bit for bit, echo worst unchanged",
          not _bits(list(v["total"]), list(base["total"])) and abs(v["echo_worst"] - base["echo_worst"]) < 1e-12 and all(v["used"][M]["chosen"] == ["x", "x"] for M in (18, 24, 36)),
          "echo share %.4f (x %.4f)" % (v["echo_share"], base["echo_worst"]))
    # two anchors in one zone (4-6L): class P is interpolated in 1/P between them; x12 (P 4 < 4.243) holds d18, x15 gets the 1/P blend
    a18, a24 = mk_anchor(18, 225, -2.0), mk_anchor(24, 225, -0.5)
    rr = {M: ratio_for_class(M, 225, {225: [a18, a24], 315: []}) for M in (12, 15, 18, 24)}
    want15 = interp_c(5.0, [(a18["P"], a18["r"][0]), (a24["P"], a24["r"][0])])
    check("zone 4-6L, two anchors: x12 holds d18, x15 = 1/P blend, x18 / x24 (zone 6-12L) have no option",
          rr[12][0] == a18["r"][0] and rr[15][0] == want15 and abs(20 * math.log10(abs(rr[15][0])) + 2.0) > 0 and rr[18] is None and rr[24] is None,
          "x15 ratio %+.4f dB (d18 %+.2f, d24 %+.2f)" % (20 * math.log10(abs(rr[15][0])), -2.0, -0.5))
    # echo oracle: psi225 +3 dB and psi315 -3 dB above / below the zone-6-12 x echoes, NE vs swapped assignment, independent recomputation
    zc = [m for m in L.CLASSES if ZCLASS["x%02d" % m] == "6-12L"]
    ex_z = L.pmean([EX[m] for m in zc])
    e225, e315 = ex_z + 3.0, ex_z - 3.0
    v = evaluate({}, {225: [mk_anchor(36, 225, 0.0, edB=e225)], 315: [mk_anchor(36, 315, 0.0, edB=e315)]})
    sdm, sdp = D_SHARE["6-12L"]

    def aper(sdm_, sdp_):
        amp = 0.0
        for z, w in L.ZONE_W.items():
            if z == ">12L":
                e = L.GATE_ECHO
            elif z == "6-12L":
                p = sum(((1 - sdm_ - sdp_) * 10 ** (EX[m] / 10) + sdm_ * 10 ** (e225 / 10) + sdp_ * 10 ** (e315 / 10)) for m in zc) / len(zc)
                e = 10 * math.log10(p)
            else:
                e = L.pmean([EX[int(c[1:])] for c in L.ZMAP[z]])
            amp += (w / 100.0) * math.sqrt(10 ** (e / 10.0))
        return 10 * math.log10(amp * amp)

    check("echo needle (psi225 +3 dB, psi315 -3 dB in 6-12L): share(NE) and share(swapped) == independent zone formula",
          abs(v["echo_share"] - aper(sdm, sdp)) < 1e-9 and abs(v["echo_share_swap"] - aper(sdp, sdm)) < 1e-9 and abs(v["echo_share"] - v["echo_share_swap"]) > 1e-4,
          "NE %.4f (oracle %.4f) swapped %.4f (oracle %.4f)" % (v["echo_share"], aper(sdm, sdp), v["echo_share_swap"], aper(sdp, sdm)))
    check("echo needle: worst = max(x, d225, d315) in 6-12L classes", all(abs(v["class_echo_worst"]["x%02d" % m] - max(EX[m], e225, e315)) < 1e-12 for m in zc))
    # zone gating: a planted anchor in 6-12L must not move zone 4-6L classes' echo and vice versa
    v = evaluate({}, {225: [mk_anchor(18, 225, 0.0, edB=EX[12] + 5.0)], 315: []})
    check("echo needle: a 4-6L anchor leaves the 6-12L class echoes at x", all(v["class_echo_share"]["x%02d" % m] == EX[m] for m in (18, 24, 36, 6, 9)))
    # refusal logic
    pend = d_anchors(True, cands=lambda M, psi: [LD + r"\_no_such_folder_%d_%d" % (M, psi)], verbose=False)
    check("pending d cells: all %d (class, psi) reported pending, carried at ratio 1; REFUSE (exit 3) without --partial, none with it" % (2 * len(D_CLASSES)),
          len(pend[1]) == 2 * len(D_CLASSES) and all(a["carried"] and a["r"] == [1 + 0j, 1 + 0j] for psi in PSIS for a in pend[0][psi]) and refuse_code(pend[1], False) == 3 and refuse_code(pend[1], True) == 0
          and refuse_code([], False) == 0)
    # ---- (V1) global-1/P map: x18 / x24 / x36 interpolate d24 -> d36 in 1/P, x12 holds d18, x15 blends d18 -> d24; a carried d36 gives the 6-12L classes no option
    g18, g24, g36 = mk_anchor(18, 225, -2.0, ph_deg=4.0), mk_anchor(24, 225, -0.5, ph_deg=-3.0), mk_anchor(36, 225, -1.5, rRdB=-0.4, ph_deg=7.0)
    dg = {225: [g18, g24, g36], 315: []}
    rg = {M: ratio_for_class(M, 225, dg, "global") for M in L.CLASSES}
    want = {M: interp_c(M / 3.0, [(a["P"], a["r"][0]) for a in (g18, g24, g36)]) for M in (12, 15, 18, 24, 36)}
    wantR = interp_c(6.0, [(a["P"], a["rR"]) for a in (g18, g24, g36)])
    c36 = mk_anchor(36, 225, 0.0, carried=True)
    rg_c = {M: ratio_for_class(M, 225, {225: [g18, g24, c36], 315: []}, "global") for M in (18, 24, 36)}
    check("(V1) global map: classes 12-36 = 1/P interpolation over d18 / d24 / d36 (x18 between d24 and d36), x06 / x09 no d option (no planted anchor in 2-3L / 3-4L); carried d36 -> 6-12L none",
          all(abs(rg[M][0] - want[M]) < 1e-15 for M in want) and abs(rg[18][2] - wantR) < 1e-15 and rg[6] is None and rg[9] is None and rg[12][0] == g18["r"][0]
          and rg[36][0] == g36["r"][0] and all(v is None for v in rg_c.values()),
          "x18 %+.4f dB (d24 -0.5, d36 -1.5), x24 %+.4f dB" % (20 * math.log10(abs(rg[18][0])), 20 * math.log10(abs(rg[24][0]))))
    vg = evaluate({}, dg, rmap="global")
    best_ = [None, None]                                                       # per-class oracle (brute() above substitutes ONE ratio for all classes)
    for n in range(6):
        for sub in itertools.combinations((12, 15, 18, 24, 36), n):
            nodes = {M: {"A": ND[M]["A"] * (rg[M][0] if M in sub else 1.0), "B": ND[M]["B"] * (rg[M][1] if M in sub else 1.0) * NR[M] * (rg[M][2] if M in sub else 1.0)} for M in L.CLASSES}
            b = L.betaL(nodes)
            for i in (0, 1):
                if best_[i] is None or b[i] + DFC + PRT < best_[i]:
                    best_[i] = b[i] + DFC + PRT
    check("(V1) global map TOTAL == direct per-class oracle over all 32 subsets", abs(vg["total"][0] - best_[0]) < 1e-12 and abs(vg["total"][1] - best_[1]) < 1e-12,
          "%.6f / %.6f (oracle %.6f / %.6f)" % (vg["total"][0], vg["total"][1], best_[0], best_[1]))
    # ---- (V2) breakeven_d36 with d36 psi225 MEASURED (ratio 1, echo planted) and psi315 pending: psi225 kept, only psi315 planted; echo bisection reaches the gate on the zone formula
    e225m = ex_z + 3.0
    m225 = mk_anchor(36, 225, 0.0, edB=e225m)
    m225["r"], m225["rR"] = [1 + 0j, 1 + 0j], 1 + 0j
    dV2 = {225: [mk_anchor(18, 225, 0.0, carried=True), mk_anchor(24, 225, 0.0, carried=True), m225], 315: [mk_anchor(M, 315, 0.0, carried=True) for M in D_CLASSES]}
    bV2 = breakeven_d36({}, dV2)

    def aper2(e315):
        amp = 0.0
        for z, w in L.ZONE_W.items():
            if z == ">12L":
                e = L.GATE_ECHO
            elif z == "6-12L":
                p = sum(((1 - sdm - sdp) * 10 ** (EX[m] / 10) + sdm * 10 ** (e225m / 10) + sdp * 10 ** (e315 / 10)) for m in zc) / len(zc)
                e = 10 * math.log10(p)
            else:
                e = L.pmean([EX[int(c[1:])] for c in L.ZMAP[z]])
            amp += (w / 100.0) * math.sqrt(10 ** (e / 10.0))
        return 10 * math.log10(amp * amp)
    check("(V2) breakeven_d36: one d36 psi measured -> only the pending psi planted, the measured echo kept (zone formula at the gate)",
          bV2["planted_psi"] == [315] and bV2["scan_psi"] == 315 and abs(aper2(bV2["echo_breakeven_dB"]) - L.GATE_ECHO) < 1e-9,
          "planted %s, e* psi315 %.4f dB, zone formula %.6f" % (bV2["planted_psi"], bV2["echo_breakeven_dB"], aper2(bV2["echo_breakeven_dB"])))
    # ---- (V3) candidate discovery keeps the explicit order first and appends other <base>* folders
    c18, c36 = d_cands(18, 225), d_cands(36, 315)
    c12, c15 = d_cands(12, 225), d_cands(15, 315)
    check("(X4) d_cands: sim800 d12 / d15 folders are the explicit candidates (suffix-less)", c12[0].endswith("261015_sim800_d12_p225_li_e201") and c15[0].endswith("261015_sim800_d15_p315_li_e201"), "%s %s" % (os.path.basename(c12[0]), os.path.basename(c15[0])))
    check("(V3) d_cands: explicit candidates first (d18 _bt12_r, _bt12; d36 _bt14), other <base>* folders appended",
          c18[0].endswith("_bt12_r") and c18[1].endswith("_bt12") and c36[0].endswith("_bt14") and all(os.path.basename(p).startswith("261011_sim781_d36_p315_li_e201") for p in c36),
          "d36 psi315 candidates: %s" % [os.path.basename(p) for p in c36])
    # ---- (V5) 1/P echo map: independent np.interp hand calculation on planted anchors (d18 / d24 in 4-6L, d36 in 6-12L, psi 225 only)
    import numpy as np
    dge = {225: [mk_anchor(18, 225, -2.0, edB=-19.0), mk_anchor(24, 225, -0.5, edB=-21.0), mk_anchor(36, 225, -1.5, edB=-23.0)], 315: []}
    xs_ = np.array([1.0 / dge[225][2]["P"], 1.0 / dge[225][1]["P"], 1.0 / dge[225][0]["P"]])                 # ascending in 1/P: d36, d24, d18
    es_ = np.array([-23.0, -21.0, -19.0])

    def hand_ce(mode, swap):
        ce = {}
        for M in L.CLASSES:
            z = ZCLASS["x%02d" % M]
            ex_ = EX[M]
            if z in ("4-6L", "6-12L"):                                                                      # the zones that hold a planted anchor
                ed_ = float(np.interp(3.0 / M, xs_, es_)) if mode == "db" else 10 * math.log10(float(np.interp(3.0 / M, xs_, 10 ** (es_ / 10.0))))
                sdm_, sdp_ = D_SHARE[z]
                if swap:
                    sdm_, sdp_ = sdp_, sdm_
                pw = (1 - sdm_ - sdp_ - Y_SHARE[z]) * 10 ** (ex_ / 10) + Y_SHARE[z] * 10 ** (ex_ / 10) + sdm_ * 10 ** (ed_ / 10) + sdp_ * 10 ** (ex_ / 10)     # no y anchor, no psi315 anchor -> x echo
                ce[M] = 10 * math.log10(pw)
            else:
                ce[M] = ex_
        return ce

    def hand_ap(ce):
        amp = 0.0
        for z, w in L.ZONE_W.items():
            e = L.GATE_ECHO if z == ">12L" else 10 * math.log10(sum(10 ** (ce[int(c[1:])] / 10.0) for c in L.ZMAP[z]) / len(L.ZMAP[z]))
            amp += (w / 100.0) * math.sqrt(10 ** (e / 10.0))
        return 10 * math.log10(amp * amp)
    for mode, em in (("db", "ip_db"), ("pow", "ip_pow")):
        s_ = echo_summary({}, dge, em)
        hn, hs = hand_ap(hand_ce(mode, False)), hand_ap(hand_ce(mode, True))
        check("(V5) echo map '%s' == independent np.interp hand calculation (7 class echoes, NE and swapped aperture)" % em,
              abs(s_["echo_share"] - hn) < 1e-9 and abs(s_["echo_share_swap"] - hs) < 1e-9 and all(abs(s_["class_echo_share"]["x%02d" % M] - hand_ce(mode, False)[M]) < 1e-9 for M in L.CLASSES)
              and abs(s_["echo_quadrant_agnostic"] - max(hn, hs)) < 1e-9, "NE %.6f (hand %.6f) swapped %.6f (hand %.6f)" % (s_["echo_share"], hn, s_["echo_share_swap"], hs))
    s_db, s_pw, s_zn = echo_summary({}, dge, "ip_db"), echo_summary({}, dge, "ip_pow"), echo_summary({}, dge, "zone")
    check("(V5) power-linear class echoes >= dB-linear ones (convexity); the zone-bin map differs from the 1/P map on the planted anchors",
          all(s_pw["class_echo_share"][c] >= s_db["class_echo_share"][c] - 1e-12 for c in s_db["class_echo_share"]) and abs(s_zn["echo_share"] - s_db["echo_share"]) > 1e-4,
          "NE dB %.4f power %.4f zone %.4f" % (s_db["echo_share"], s_pw["echo_share"], s_zn["echo_share"]))
    d1 = {225: [mk_anchor(36, 225, 0.0, edB=-22.0)], 315: [mk_anchor(36, 315, 0.0, edB=-22.5)]}
    sa, sb = echo_summary({}, d1, "ip_db"), echo_summary({}, d1, "zone")
    check("(V5) one d anchor per psi (d36 only): the 1/P echo (held) == the zone-bin echo, class by class and in the aperture",
          all(abs(sa["class_echo_share"][c] - sb["class_echo_share"][c]) < 1e-12 for c in sa["class_echo_share"]) and abs(sa["echo_share"] - sb["echo_share"]) < 1e-12 and abs(sa["echo_share_swap"] - sb["echo_share_swap"]) < 1e-12,
          "NE %.6f vs %.6f" % (sa["echo_share"], sb["echo_share"]))
    vg2 = evaluate({}, dg, rmap="global", emap="ip_db")
    check("(V5) emap never touches the TOTAL: evaluate(rmap='global', emap='ip_db') == evaluate(rmap='global') TOTAL bit for bit; default emap is 'zone'",
          not _bits(list(vg2["total"]), list(vg["total"])) and vg2["emap"] == "ip_db" and vg["emap"] == "zone" and abs(vg2["echo_share"] - echo_summary({}, dg, "ip_db")["echo_share"]) < 1e-12)
    # ---- (V7) verdict line and its PASS / FAIL logic
    l1, t1, e1_ = verdict_text((-0.6008293318, -0.6217217020), -20.6604, -20.6604)
    l2, t2, e2_ = verdict_text((-0.6452836190, -0.6663746284), -20.7989, -20.7989)
    l3, t3, e3_ = verdict_text((-0.60, -0.62), -20.60, -20.40)
    check("(V7) verdict_text: 1/P numbers -> TOTAL PASS + echo PASS; zone numbers -> TOTAL FAIL (held tail); a worse quadrant-agnostic echo -> echo FAIL",
          t1 and e1_ and (not t2) and e2_ and t3 and (not e3_)
          and l1 == "VERDICT (1/P map, 2026-10-12): TOTAL -0.6008 / -0.6217 dB vs -0.65 -> PASS ; echo -20.6604 dB (quadrant-agnostic -20.6604) vs -20.5 -> PASS", l1)
    # ---- (V8) reduced option set and anchor perturbation
    ro = _red({12: {"y": 1, "d225": 2, "d315": 3}, 15: {"y": 1, "d225": 2}, 18: {"y": 1, "d225": 2, "d315": 3}, 24: {"y": 1, "d225": 2, "d315": 3}, 36: {"d225": 2, "d315": 3}, 9: {"y": 1}})
    check("(V8) _red: x12 {y}, x15 dropped, x18 / x24 / x36 their d options only, a y-only class dropped; 54 combinations",
          ro == {12: {"y": 1}, 18: {"d225": 2, "d315": 3}, 24: {"d225": 2, "d315": 3}, 36: {"d225": 2, "d315": 3}} and math.prod(1 + len(v) for v in ro.values()) == 54)
    pd_ = perturb_d(dge, cls=(36,), dB=-1.0, dph_deg=10.0, dR_dB=-0.5, dE_dB=0.2)
    a_n, a_o = pd_[225][2], dge[225][2]
    check("(V8) perturb_d: magnitude -1 dB, phase +10 / -10 deg on A / B, rB0 -0.5 dB, echo +0.2 dB on d36 only; other anchors and the input untouched",
          abs(abs(a_n["r"][0]) / abs(a_o["r"][0]) - 10 ** (-1.0 / 20)) < 1e-12 and abs(cmath.phase(a_n["r"][0] / a_o["r"][0]) - math.radians(10.0)) < 1e-12
          and abs(cmath.phase(a_n["r"][1] / a_o["r"][1]) + math.radians(10.0)) < 1e-12 and abs(abs(a_n["rR"] / a_o["rR"]) - 10 ** (-0.5 / 20)) < 1e-12 and abs(a_n["e"] - a_o["e"] - 0.2) < 1e-12
          and pd_[225][0] is dge[225][0] and pd_[225][1] is dge[225][1] and dge[225][2]["e"] == -23.0)
    say("SELFTEST %s  (%.0f s, %d betaL evaluations cached)" % ("PASS" if ok else "FAIL", time.time() - t0, len(_BC)))
    with excl_open(_sp) as fh:                                                 # (V4 / V9)
        json.dump(dict(passed=ok, checks=res), fh, indent=1)
    return 0 if ok else 1


def breakeven_d36(anchors, dan):
    """what the pending d36 has to show.  Echo: bisection on the d36 echo (both psi equal, measured-type anchors, ratio 1) until the share-weighted aperture is at the gate.  TOTAL: a planted
    magnitude-only d36 deficit (both arms, rR 1, no phase) in the zone 6-12L, y12 kept only (the other y options are better than x and never the worst), interpolated to the gate."""
    pend36 = [psi for psi in PSIS if any(a["cls"] == 36 and a["carried"] for a in dan[psi])]                # (V2) plant only the pending d36 psi
    d2 = {psi: ([a for a in dan[psi] if a["cls"] != 36] + [dict(mk_anchor(36, psi, 0.0, edB=-20.0), carried=False)]) if psi in pend36 else list(dan[psi]) for psi in PSIS}

    def ap(e):
        for psi in pend36:
            d2[psi][-1]["e"] = e
        return L.aperture(echoes(anchors, d2)[1])
    lo, hi = -40.0, 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ap(mid) > L.GATE_ECHO:
            hi = mid
        else:
            lo = mid
    e_star = 0.5 * (lo + hi)
    etab = [(e, ap(e)) for e in (-23.0, -22.0, -21.5, -21.0, -20.5, -20.0)]
    a12 = {12: anchors[12]} if 12 in anchors else {}
    psp = pend36[0] if pend36 else 225                                         # (V2) planted psi = a pending one; a measured d36 psi stays in
    keep = {psi: [a for a in dan[psi] if a["cls"] == 36 and not a["carried"]] for psi in PSIS}

    def dset(rdB):
        d = {psi: list(keep[psi]) for psi in PSIS}
        if rdB is not None:
            d[psp] = d[psp] + [mk_anchor(36, psp, rdB)]
        return d
    scan = []
    for rdB in (-0.05, -0.10, -0.15, -0.20, -0.30):
        v = evaluate(a12, dset(rdB))
        scan.append(dict(rdB=rdB, total=list(v["total"])))
    be = []
    for i in (0, 1):
        pts = [(0.0, evaluate(a12, dset(None))["total"][i])] + [(r["rdB"], r["total"][i]) for r in scan]
        for (x0, t0_), (x1, t1_) in zip(pts, pts[1:]):
            if t0_ >= L.GATE_TOTAL >= t1_:
                be.append(x0 + (L.GATE_TOTAL - t0_) * (x1 - x0) / (t1_ - t0_))
                break
        else:
            be.append(None)
    chk = evaluate(a12, dset(min(b for b in be if b is not None)))["total"] if any(b is not None for b in be) else [None, None]
    return dict(echo_breakeven_dB=e_star, echo_table=etab, scan=scan, breakeven_dB=[-(b) if b is not None else None for b in be], check_total=list(chk),
                planted_psi=pend36, scan_psi=psp)


# ================================================================ REAL READ-OUT ===============================================================================================
def main():
    partial = "--partial" in sys.argv
    name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else ("lensdir45_readout_v6_result_partial" if partial else "lensdir45_readout_v6_result")   # (V4 / V9 / W5 / X2 / Y5)
    vname = (name.replace("readout_v6_result", "verdict_1overP_v6") if "readout_v6_result" in name else name + "_verdict")   # (W5 / X2 / Y5) default: lensdir45_verdict_1overP_v6 (+ _partial)
    for _p in (os.path.join(OUT, name + ".json"), os.path.join(OUT, vname + ".json")):                     # (V9) never overwrite
        if os.path.exists(_p):
            say("REFUSE: output exists, not overwritten: %s" % _p)
            return 5
    t0 = time.time()
    need = os.path.join(NEW, "261011_sim779_dir_needle_li_all")
    if not os.path.isfile(os.path.join(need, "superramp_result.json")):
        say("REFUSE: needle cell missing: %s" % need)
        return 3
    nr_ = jl(need)["needle_results"]
    okn = bool(nr_.get("N1y", {}).get("pass_all_1e9")) and bool(nr_.get("N3", {}).get("pass_1e9")) and bool(nr_.get("N1x", {}).get("pass_all_1e9"))
    say("needle sim779 (Li y): N1x %s N1y %s N3 %s" % (nr_.get("N1x", {}).get("pass_all_1e9"), nr_.get("N1y", {}).get("pass_all_1e9"), nr_.get("N3", {}).get("pass_1e9")))
    if not okn:
        say("REFUSE: the Li needle did not pass -- the y cells are not trusted")
        return 4
    n3 = {}
    for nm, key in (("261011_sim780_b1_nd3_li", "Nd3"), ("261011_sim780_b2_n1d_li", "N1d")):
        p = os.path.join(P3, nm)
        if not os.path.isfile(os.path.join(p, "superramp_result.json")):
            say("REFUSE: licol needle cell missing: %s" % p)
            return 3
        j = jl(p)
        n3[key] = bool(j["needle_results"][key].get("pass_1e9")) and (j.get("rcwa_rule"), j.get("rcwa_ff")) == ("liy", 1)
    say("needle sim780 (licol identity): Nd3 %s N1d %s" % (n3["Nd3"], n3["N1d"]))
    if not all(n3.values()):
        say("REFUSE: a licol identity needle did not pass -- the d cells are not trusted")
        return 4
    say("== y direction (sim779, e201 pairs; code and arithmetic of lensdir_readout.py)")
    anchors, missing = y_anchors()
    for M in (12, 15, 18):
        a, b = os.path.join(NEW, "261011_sim779_dir_x%02de201_sh0" % M), os.path.join(NEW, "261011_sim779_dir_x%02de201_sh0_rep" % M)
        if os.path.isfile(os.path.join(a, "superramp_result.json")) and os.path.isfile(os.path.join(b, "superramp_result.json")):
            ba, bb = beta_pc(a)[0], beta_pc(b)[0]
            say("  host replicate x%02d sh0: |d beta| A %.2e B %.2e, d echo %.2e dB" % (M, abs(ba[0] - bb[0]), abs(ba[1] - bb[1]),
                jl(a)["needle_results"]["production"]["echo_p2_db"] - jl(b)["needle_results"]["production"]["echo_p2_db"]))
    ymiss = [M for M in (12, 15, 18, 24) if M in missing]
    if ymiss and not partial:
        say("REFUSE: T1 y / x pairs missing for M %s" % ymiss)
        return 3
    say("== d direction (sim781 + sim800, licol, e201; read against the MEASURED x e201 nodes x06 (sim780) / x09 (sim802) / x12..x36 interpolated in 1/P, power_centre, K0_TM; no clamp below P = 4)")
    xr = xref()
    say("  x e201 reference nodes (power_centre, 3-registration coherent): " + "; ".join("x%02d P %.0f A %+.3f dB %+.1f deg" % (M, xr[M]["P"], 20 * math.log10(abs(xr[M]["node"][0])), math.degrees(cmath.phase(xr[M]["node"][0]))) for M in XREF_M))
    say("  201 -> 401 echo offset of the x classes (e_x401 - e_x201, dB): " + "; ".join("x%02d %+.3f" % (M, xr[M]["off"]) for M in XREF_M))
    dan, pending = d_anchors(partial)
    if "--anchors-only" in sys.argv:                                           # (W4) no enumeration: anchors table + json only
        ao = os.path.join(OUT, name + "_anchors.json")
        if os.path.exists(ao):
            say("REFUSE: output exists, not overwritten: %s" % ao)
            return 5
        with excl_open(ao) as fh:
            json.dump(dict(pending=[list(p) for p in pending], d_anchors=[d_summary(a) for psi in PSIS for a in dan[psi]], x_ref_nodes={str(M): dict(P=xr[M]["P"], A_dB=20 * math.log10(abs(xr[M]["node"][0])),
                      B_dB=20 * math.log10(abs(xr[M]["node"][1])), echo_offset_dB=xr[M]["off"]) for M in XREF_M}, force_pending=sorted(FORCE_PENDING), pending_ok=sorted(PENDING_OK)), fh, indent=1, default=lambda o: str(o))
        say("wrote %s" % ao)
        return 0
    if refuse_code(pending, partial):
        say("REFUSE: d cells pending (no superramp_result.json): %s -- re-run when they have landed, or use --partial for an INTERIM read-out" % ", ".join("d%02d psi%d" % mp for mp in pending))
        return 3
    interim = bool([mp for mp in pending if mp[0] not in PENDING_OK] or ymiss)
    pending_not_included = sorted({mp[0] for mp in pending if mp[0] in PENDING_OK})
    if pending_not_included:
        say("== NOTE (--pending-ok): d%s still PENDING and NOT included in this read-out (carried at x = ratio 1, no option, no echo); the verdict below is for the measured anchors only" % ", d".join("%02d" % c for c in pending_not_included))
    if interim:
        banner_interim(pending)
        if ymiss:
            say("!! y cells missing for M %s -- those classes keep -x" % ymiss)
    say("== period -> zone mapping (this is what decides which x classes a d anchor speaks for)")
    for psi in (225,):
        for a in dan[psi]:
            say("  d%02d: P_d = n a / sqrt 2 = %d x %.5f / 3 / sqrt 2 = %.4f Lambda -> zone %s ; 1/P-nearest x class %s (the 'x12-ish / x18-ish / x24-ish' label); psi from the lattice %.1f" % (
                a["cls"], a["cls"], C.LAMG, a["P"], a["zone"], a["nearest_x"], a["psi_from_lattice"]))
    say("  zone shares (NE quadrant, f200): " + "; ".join("%s -x/+x %.3f y %.3f d-(psi225) %.3f d+(psi315) %.3f" % (z, s["x"], s["y"], s["dm"], s["dp"]) for z, s in SHARES.items()) + "  (2-3L and 3-4L use the (2,4] band as in the y readout)")
    say("== results on the GLOBAL 1/P MAP (2026-10-12: PRIMARY for the TOTAL and for the echo); every per-class choice among x / y / d225 / d315 evaluated; minimum per TOTAL component; echo share-weighted")
    cfg = {}
    off_ = {225: [], 315: []}
    for key, lab, a_, d_ in (("x", "x only (adopted)", {}, off_), ("x_y", "x + y (frozen y read-out)", anchors, off_), ("x_d", "x + d (y off), 1/P map", {}, dan), ("x_y_d", "x + y + d (COMBINED), 1/P map", anchors, dan)):
        say("  [%s] evaluating ..." % lab)
        cfg[key] = evaluate(a_, d_, rmap="global", verbose=True, emap="ip_db")
        show(lab, cfg[key])
    if not partial and not ymiss:
        sy = json.load(io.open(Y_STORED, encoding="utf-8"))["result"]
        same = not _bits(jsonable(dict(t=list(cfg["x_y"]["total"]), w=cfg["x_y"]["echo_worst"], s=cfg["x_y"]["echo_share"])), jsonable(dict(t=sy["total"], w=sy["echo_worst"], s=sy["echo_share"])))
        say("  x + y reproduces the stored y read-out bit for bit: %s" % same)
    refs = {}
    if "--no-zone" not in sys.argv:
        say("  [REFERENCE: x + y + d, zone map (TOTAL and echo on the zone bins; pessimistic TOTAL bound, bin edge at P = 6)] evaluating ...")
        refs["zone"] = evaluate(anchors, dan, rmap="zone", verbose=True, emap="zone")
        show("REF zone map (zone-bin echo)", refs["zone"])
    if "--no-nearest" not in sys.argv:
        say("  [REFERENCE: x + y + d, nearest-class map (each d anchor replaces only the 1/P-nearest class)] evaluating ...")
        refs["nearest"] = evaluate(anchors, dan, rmap="nearest", verbose=True, emap="zone")
        show("REF nearest-class map (zone-bin echo)", refs["nearest"])
    c = cfg["x_y_d"]
    es = {em: echo_summary(anchors, dan, em) for em in EMAPS}
    e1, ep_, ez = es["ip_db"], es["ip_pow"], es["zone"]
    assert abs(e1["echo_share"] - c["echo_share"]) < 1e-12 and abs(e1["echo_share_swap"] - c["echo_share_swap"]) < 1e-12 and abs(e1["echo_worst"] - c["echo_worst"]) < 1e-12
    say("")
    say("== ECHO on the 1/P map (primary): the d echo of class M = the measured d anchors interpolated in 1/P at P = M/3 (dB-linear), held beyond d18 / d36; a zone without a measured d anchor keeps the x echo")
    say("  aperture, dB-linear (PRIMARY): NE %.4f  swapped d+/d- %.4f  quadrant-agnostic (worse of the two) %.4f  worst-orientation %.4f dB  (gate %.1f)" % (
        e1["echo_share"], e1["echo_share_swap"], e1["echo_quadrant_agnostic"], e1["echo_worst"], L.GATE_ECHO))
    say("  aperture, power-linear       : NE %.4f  swapped %.4f  quadrant-agnostic %.4f dB" % (ep_["echo_share"], ep_["echo_share_swap"], ep_["echo_quadrant_agnostic"]))
    say("  REFERENCE zone-bin echo (v2 basis, optimistic, VERIFY_FINAL F1): NE %.4f  swapped %.4f  quadrant-agnostic %.4f dB" % (ez["echo_share"], ez["echo_share_swap"], ez["echo_quadrant_agnostic"]))
    say("  class echoes, 1/P dB-linear (NE): " + "; ".join("%s %.3f" % (k, v) for k, v in e1["class_echo_share"].items()))
    say("  zone echoes, 1/P dB-linear (NE): " + "; ".join("%s %.3f" % (k, v) for k, v in e1["zone_echo_share"].items()))
    say("")
    say("== TOTAL by map (decay / held): PRIMARY global 1/P %.4f / %.4f ; REFERENCE zone %s ; REFERENCE nearest %s" % (
        c["total"][0], c["total"][1], ("%.4f / %.4f" % tuple(refs["zone"]["total"])) if "zone" in refs else "(skipped)", ("%.4f / %.4f" % tuple(refs["nearest"]["total"])) if "nearest" in refs else "(skipped)"))
    echo_q = e1["echo_quadrant_agnostic"]
    say("COMBINED (x + y + d, worst orientation on TOTAL, GLOBAL 1/P MAP): TOTAL %.4f / %.4f dB, margin %+.4f / %+.4f to %.2f ; echo share(NE) %.4f, quadrant-agnostic (max of NE / swapped) %.4f, worst-orientation %.4f dB (gate %.1f)" % (
        c["total"][0], c["total"][1], c["margin"][0], c["margin"][1], L.GATE_TOTAL, e1["echo_share"], echo_q, e1["echo_worst"], L.GATE_ECHO))
    say("  d effect on the combined read-out (x + y + d minus x + y): dTOTAL %+.4f / %+.4f dB, d echo share(NE) %+.3f dB" % (
        c["total"][0] - cfg["x_y"]["total"][0], c["total"][1] - cfg["x_y"]["total"][1], c["echo_share"] - cfg["x_y"]["echo_share"]))
    say("  chosen orientation per class (decay / held): " + "; ".join("x%02d %s/%s" % (M, u["chosen"][0], u["chosen"][1]) for M, u in sorted(c["used"].items())))
    be = None
    if any(M == 36 for M, _ in pending):
        be = breakeven_d36(anchors, dan)
        say("== what the pending d36 must show (planted into the zone 6-12L for the PENDING psi %s, a measured d36 psi kept)" % be["planted_psi"])
        say("  d36 echo breakeven %.2f dB (zone-bin formula); TOTAL deficit breakeven %s dB (decay / held) [E: linear interpolation]" % (be["echo_breakeven_dB"], be["breakeven_dB"]))
    so = single_options(anchors, dan, rmap="global")
    say("  one class at a time on the 1/P map (others keep x), dTOTAL vs the adopted base (decay / held):")
    for r in so:
        say("    x%02d -> %-5s dTOTAL %+.4f / %+.4f" % (r["cls"], r["option"], r["delta"][0], r["delta"][1]))
    sens = None
    if "--no-sens" not in sys.argv and not interim:
        say("")
        sens = sensitivity(anchors, dan, c["total"])
    if interim:
        banner_interim(pending)
    vline, tp, ep = verdict_text(c["total"], e1["echo_share"], echo_q)
    say("")
    say("=" * 118)
    if pending_not_included:
        vline = vline + "   [d%s PENDING: not included]" % "/d".join("%02d" % c for c in pending_not_included)
    say(vline + ("   [INTERIM: d cells pending, NOT a verdict]" if interim else ""))
    if sens:
        ad = sens["cases"][0]
        up = sens["cases"][1]
        sh_lo, sh_hi = min(abs(ad["shift"][0]), abs(ad["shift"][1])), max(abs(ad["shift"][0]), abs(ad["shift"][1]))
        rng = ("%.2f-%.2f" % (sh_lo, sh_hi)) if round(sh_lo, 2) != round(sh_hi, 2) else ("%.2f" % sh_hi)
        inside = min(c["margin"]) <= sh_hi                                     # the smaller TOTAL margin does not exceed the shift caused by a 0.1 dB d36 magnitude error
        sens["total_margin_inside_0p1dB_d36_shift"] = bool(inside)
        sens["adverse_shift_range_dB"] = [sh_lo, sh_hi]
        say("SENSITIVITY (1/P map): a 0.1 dB d36 magnitude error moves TOTAL by about %s dB on this map (computed, adverse sign, decay / held: d36 -0.1 dB -> %+.4f / %+.4f giving %.4f / %.4f dB; "
            "favourable sign d36 +0.1 dB -> %+.4f / %+.4f); the TOTAL margin %+.4f / %+.4f dB is %s the d-cell convergence uncertainty (about 0.1 dB on the eq-201 node, sim780 READOUT.md line 144)" % (
                rng, ad["shift"][0], ad["shift"][1], ad["total"][0], ad["total"][1], up["shift"][0], up["shift"][1], c["margin"][0], c["margin"][1], "INSIDE" if inside else "OUTSIDE"))
        bd = sens["breakeven_d36_deficit_dB"]
        eb = sens["echo_breakeven_step_dB"]
        nz = lambda v: "%.3f" % v if v is not None else "n/a"
        say("SENSITIVITY (gate break-even; TOTAL deficit [E] linear interpolation checked by a direct evaluation, echo step by bisection): the TOTAL gate holds while the d36 magnitude is not more than %s dB (held tail) / %s dB (decayed tail) below its measured value; "
            "the echo gate holds against a uniform d echo rise of up to %s dB (all d) / %s dB (d36 only) on the 1/P map (zone map, all d: %s dB)" % (nz(bd["held"]), nz(bd["decay"]), nz(eb["ip_db_all_d"]), nz(eb["ip_db_d36_only"]), nz(eb["zone_all_d"])))
    say("REFERENCES (printed only, not the verdict): zone map TOTAL %s, zone-bin echo NE %.4f ; nearest-class map TOTAL %s ; 1/P echo power-linear NE %.4f" % (
        ("%.4f / %.4f" % tuple(refs["zone"]["total"])) if "zone" in refs else "(skipped)", ez["echo_share"], ("%.4f / %.4f" % tuple(refs["nearest"]["total"])) if "nearest" in refs else "(skipped)", ep_["echo_share"]))
    say("=" * 118)
    mi = psutil.Process().memory_info()
    rss = getattr(mi, "peak_wset", mi.rss) / 1e9
    elapsed = time.time() - t0
    say("footprint: python peak working set %.2f GB (numpy only, no solver), affinity %s, %.0f s" % (rss, psutil.Process().cpu_affinity(), elapsed))
    jd = lambda o: [o.real, o.imag] if isinstance(o, complex) else str(o)
    pr = lambda v: dict(total=list(v["total"]), margin=list(v["margin"]), echo_share_NE=v["echo_share"], echo_share_swapped=v["echo_share_swap"], echo_worst_orientation=v["echo_worst"],
                        echo_quadrant_agnostic=max(v["echo_share"], v["echo_share_swap"]), echo_map=v["emap"], chosen={str(M): u["chosen"] for M, u in sorted(v["used"].items())}, n_combinations=v["n_combinations"])
    verdict = dict(
        verdict_line=vline, interim=interim, d_classes_pending_not_included=pending_not_included, force_pending=sorted(FORCE_PENDING), basis="decision 2026-10-12 10:5x: the 45-degree lens verdict is judged on the global 1/P map (TOTAL and echo); zone map and nearest-class map are references",
        map="global 1/P (ratio of beta_d / beta_x,e201 interpolated in 1/P over all measured d anchors of the psi, held beyond d18 / d36; d echoes interpolated in 1/P in dB)",
        total=dict(decay=c["total"][0], held=c["total"][1], gate=L.GATE_TOTAL, margin_decay=c["margin"][0], margin_held=c["margin"][1], passed=bool(tp)),
        echo=dict(share_NE=e1["echo_share"], share_swapped=e1["echo_share_swap"], quadrant_agnostic=echo_q, gate=L.GATE_ECHO, margin_NE=L.GATE_ECHO - e1["echo_share"], margin_quadrant_agnostic=L.GATE_ECHO - echo_q,
                  passed=bool(ep), worst_orientation=e1["echo_worst"], power_linear=dict(share_NE=ep_["echo_share"], share_swapped=ep_["echo_share_swap"], quadrant_agnostic=ep_["echo_quadrant_agnostic"]),
                  zone_bin_reference=dict(share_NE=ez["echo_share"], share_swapped=ez["echo_share_swap"], quadrant_agnostic=ez["echo_quadrant_agnostic"]),
                  class_echo_share_NE=e1["class_echo_share"], class_echo_share_swapped=e1["class_echo_share_swap"], zone_echo_share_NE=e1["zone_echo_share"]),
        per_direction={"x": pr(cfg["x"]), "x_y": pr(cfg["x_y"]), "x_d": pr(cfg["x_d"]), "x_y_d": pr(cfg["x_y_d"])},
        per_direction_note="x_d and x_y_d on the global 1/P map with the dB-linear 1/P echo; x and x_y have no d anchor (map independent)",
        reference_maps={k: pr(v) for k, v in refs.items()}, reference_maps_note="zone: anchors of the class's zone bin only, echo on the zone bins (pessimistic TOTAL bound; x18 at P = 6.000 sits on the bin edge); nearest: each d anchor replaces only the 1/P-nearest x class, echo on the zone bins",
        d_anchors=[d_summary(a) for psi in PSIS for a in dan[psi]], y_anchors={str(M): dict(y_minus_x_A_dB=20 * math.log10(abs(a["r"][0])), y_minus_x_B_dB=20 * math.log10(abs(a["r"][1])),
                                                                                      dphase_deg=[math.degrees(cmath.phase(a["r"][0])), math.degrees(cmath.phase(a["r"][1]))], rB0_dB=20 * math.log10(abs(a["rR"])), echo_basis_corrected_dB=a["ey"]) for M, a in sorted(anchors.items())},
        single_options_1overP=so, sensitivity=sens, shares=SHARES,
        provenance=dict(script=os.path.abspath(__file__), script_md5=hashlib.md5(io.open(os.path.abspath(__file__), "rb").read()).hexdigest(), python=sys.version.split()[0], affinity=psutil.Process().cpu_affinity(),
                        written=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), seconds=elapsed, peak_working_set_gb=rss, stdout="v6 stdout log in the output folder"),
        sources=SOURCES_FN(), other_inputs=OTHER_FN())
    with excl_open(os.path.join(OUT, name + ".json")) as fh:                                   # (V9) exclusive: never overwrites
        json.dump(dict(interim=interim, pending=[list(p) for p in pending], y_missing=missing, anchors_y=anchors, anchors_d=dan, shares=SHARES, configs=cfg, reference_configs=refs, echo_maps=es, single_options=so,
                       sensitivity=sens, quadrant_agnostic_echo=echo_q, breakeven_d36=be, verdict=verdict, rss_gb=rss), fh, indent=1, default=jd)
    with excl_open(os.path.join(OUT, vname + ".json")) as fh:
        json.dump(verdict, fh, indent=1, default=jd)
    say("wrote %s" % os.path.join(OUT, name + ".json"))
    say("wrote %s" % os.path.join(OUT, vname + ".json"))
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if SELFTEST else main())
