# -*- coding: utf-8 -*-
r"""lensband_orient_v3.py -- [TM] lens-level verdict at the band-edge channels (1546.003 / 1550 / 1554.017 nm) INCLUDING the y-ramp and 45-degree orientation,
with the same worst-orientation rules the 1550 nm verdict (lensdir45_readout_v3.py, 1/P map) uses.  json + the lens evaluator only; NO solver.

WHAT IT COMPUTES, per channel, three variants of TOTAL (decayed / held tail, dB) and of the share-weighted aperture ECHO:
  V-x    x only                       = lensband_eval_v2 (class nodes of the sim785 band-edge cells, atom terms of sim783 at the channel).
  V-xy   worst orientation x / y      y option of class M (M = 12 15 18 24) = ratio of the y cell to the PAIRED x e201 node at the same truncation,
                                      power_centre basis, K0_TM (cancels in the ratio), put on the adopted x node (class node at the channel).  Arithmetic copied from
                                      lensdir45_readout_v3.y_anchors (v3 arithmetic, unchanged): x partners = the 3 registrations coherent mean (node_pc), y = the single y cell
                                      (sc_shift 0), rR = r0(y) / mean r0(x partners), y echo = y cell echo + (EX[M] - pmean e201 of the partners).
                                      1550 column: sim779 y12..y24 + sim779 x e201 partners, as v3.  Band-edge columns: sim789 y12..y24 + sim789 x e201 sh0..sh2 of the SAME channel.
  V-xyd  worst orientation x / y / d  as V-xy plus the 45-degree options d225 / d315 with the 1550 d/x ratios (stored anchors_d of lensdir45_readout_v3_result.json)
                                      TRANSFERRED UNCHANGED to the band-edge channels (a stated assumption: no band-edge 45-degree cell exists), global 1/P map, ratio of
                                      class M = d anchors interpolated in 1/P at P = M/3 (log|r|, arg r linear, held beyond d18 / d36), only when the zone of M holds a measured anchor.
  TOTAL  = every combination of per-class choices (x / y / d225 / d315) evaluated, minimum per total component (decayed / held), ties keep x (strict < by 1e-15), enumeration order of v3.
  ECHO   = share-weighted over orientations (zone shares of oblique_class_share_stdout.txt, NE quadrant; s_y = +y + -y, s_dm = 135 + 225, s_dp = 45 + 315), class echoes -> V1.aperture
           (zone power mean, ZONE_W, >12L at the gate).  d echo (1/P dB-linear map of v3).  quadrant-agnostic = max(NE, swapped d+/d-).
D-ECHO TRANSFER (the one open reading of the task): primary 'C' = the d cell echo at e201 (e_raw of the 1550 anchor) UNCHANGED + the channel's own 201->401 offset by the v3 rule
  (off = EX[m] - e201[m] of the x classes at the channel, interpolated in 1/P, m = 12 15 18 24 36);  sensitivities printed beside it: 'A' = the 1550 d echo e unchanged,
  'B' = d echo tracks the x echo (e_raw + [x e201 echo(P_d) at the channel - at 1550] + channel offset).  At 1550 'C' == 'A' == stored (checked).
ATOM TERMS (deficit_p', purity_p' f200) per channel = those of lensband_eval_v2_result.json (cross-checked against run_all_261013_1708.txt), added the same way in all variants.
LENS MODEL: lensband_eval.Geo('fixed', channel) = physical lens fixed at its 1550 design, propagated / obliquity at the channel (the model lensband_eval_v2 prints).

CONTROLS (all must PASS before any band-edge number is printed as a result; every control prints numbers):
  (a) V-x reproduces lensband_eval_v2 at all three channels (TOTAL decay / held, echo aperture)             |diff| <= 1e-12 (bit equality reported)
  (b) 1550: V-xy == frozen y read-out (lensdir_readout_result.json) and == v3 x_y config; V-xyd == v3 verdict (-0.6008 / -0.6217, echo -20.660) incl. chosen orientations,
      the y anchors re-derived from the sim779 cells == stored anchors, d-echo offset rule re-derived == stored d echoes, shares re-read == stored shares
  (c) needles: x partners planted as the y cells -> y ratio 1 and V-xy == V-x; a y ratio of -0.1 dB on one class must change the total (and equal an independent betaL oracle);
      a wrong-channel y cell must be refused
Rules: nothing is deleted, nothing under <DATA_ROOT> is written, no solver, the existing scripts are only imported.  Outputs: lensband_orient_v3_stdout_<stamp>.txt
(STATUS: IN PROGRESS ... STATUS: COMPLETE) and lensband_orient_v3_result_<stamp>.json next to this file, created exclusively (never overwritten).
python -I -B lensband_orient_v3.py [--workers 6] [--tag NAME] [--no-needles]"""
import os
import sys

sys.dont_write_bytecode = True
EVD = r"<WORK_ROOT>\wlband_261012\lensband_eval"
PB = r"<WORK_ROOT>\te_beta_orientation_261005\power_basis_261006"
WD = r"<WORK_ROOT>\wlband_261012\lensband_orient_v3"
sys.path.insert(0, EVD)
import argparse                                                                  # noqa: E402
import cmath                                                                     # noqa: E402
import concurrent.futures as cf                                                  # noqa: E402
import datetime                                                                  # noqa: E402
import hashlib                                                                   # noqa: E402
import io                                                                        # noqa: E402
import itertools                                                                 # noqa: E402
import json                                                                      # noqa: E402
import math                                                                      # noqa: E402
import re                                                                        # noqa: E402
import struct                                                                    # noqa: E402
import time                                                                      # noqa: E402
import lensband_eval as V1                                                       # noqa: E402  (v1, unmodified; pins cores 10-17, BLAS threads 1)
import psutil                                                                    # noqa: E402

R = V1.R
R779 = R + r"\261011_scdc_tmzone_dir"                                            # sim779: y12..y24 e201 + paired x e201 (1550)
R789 = R + r"\261013_scdc_lensband_y"                                            # sim789: band-edge y12..y24 e201 + paired x e201 sh0..sh2
V3DIR = r"<WORK_ROOT>\loop_261008\lens45_li_261011\readout45"
V3_RESULT = V3DIR + r"\lensdir45_readout_v3_result.json"
V3_VERDICT = V3DIR + r"\lensdir45_verdict_1overP.json"
Y_STORED = r"<WORK_ROOT>\loop_261008\gap_plan_261010\lensdir\lensdir_readout_result.json"
V2_RESULT = EVD + r"\lensband_eval_v2_result.json"
V2_STDOUT = EVD + r"\run_all_261013_1708.txt"
SHARE_FILE = r"<WORK_ROOT>\te_beta_orientation_261005\oblique_class_share_stdout.txt"

CLASSES = V1.CLASSES                                                             # (6, 9, 12, 15, 18, 24, 36)
GATE_TOTAL, GATE_ECHO = V1.GATE_TOTAL, V1.GATE_ECHO
CHS = ("1546.003", "1550", "1554.017")
PSIS = (225, 315)
ORDER = ("y", "d225", "d315")
XREF_M = (12, 15, 18, 24, 36)                                                    # x classes with e201 echoes (v3)
YM = (12, 15, 18, 24)                                                            # y classes (sim779 / sim789)
ZCLASS = {c: z for z, cl in V1.ZMAP.items() for c in cl}                         # "x12" -> "4-6L"
TOL = 1e-12


# ====================================================================== log =====================================================================================================
class Log:
    def __init__(self, path):
        self.path = path
        self.fh = io.open(path, "x", encoding="utf-8")                           # exclusive: an earlier run's stdout is never overwritten
        self.fh.write("STATUS: IN PROGRESS\n")
        self.fh.flush()

    def say(self, s=""):
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()

    def close(self, status="COMPLETE"):
        self.fh.write("STATUS: %s\n" % status)
        self.fh.close()


LG = None


def say(s=""):
    if LG is not None:
        LG.say(s)
    else:
        print(s, flush=True)


# ====================================================================== copies of the v3 helpers (arithmetic verbatim) ============================================================
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


Y_SHARE = {"2-3L": 0.301, "3-4L": 0.301, "4-6L": 0.145, "6-12L": 0.249}


def read_shares():
    """lensdir45_readout_v3.read_shares verbatim (NE quadrant shares of the zone)"""
    rows = {}
    for line in io.open(SHARE_FILE, encoding="utf-8"):
        m = re.match(r"^\(([0-9.e+]+), ([0-9.e+]+)\]\s+([0-9.]+)\s+(.+)$", line.strip())
        if m:
            v = [float(t) for t in m.group(4).split()]
            if len(v) == 8:
                rows[(m.group(1), m.group(2))] = v
    band = {"2-3L": ("2", "4"), "3-4L": ("2", "4"), "4-6L": ("4", "6"), "6-12L": ("6", "12")}
    sh = {}
    for z, k in band.items():
        v = rows[k]
        sh[z] = dict(x=(v[0] + v[4]) / 100.0, y=(v[2] + v[6]) / 100.0, dm=(v[3] + v[5]) / 100.0, dp=(v[1] + v[7]) / 100.0, band=k, raw=v)
        assert abs(sh[z]["y"] - Y_SHARE[z]) < 1e-9, (z, sh[z]["y"], Y_SHARE[z])
    return sh


SHARES = read_shares()
D_SHARE = {z: (SHARES[z]["dm"], SHARES[z]["dp"]) for z in SHARES}


def ratio_for_class(M, psi, dan):
    """v3 ratio_for_class(rmap='global'): the zone of M must hold a MEASURED anchor of this psi; ratio interpolated in 1/P over all measured anchors at P = M/3, held beyond the ends"""
    meas = [x for x in dan.get(psi, []) if not x["carried"]]
    z = ZCLASS["x%02d" % M]
    if not any(x["zone"] == z for x in meas):
        return None
    P = M / 3.0
    return (interp_c(P, [(x["P"], x["r"][0]) for x in meas]), interp_c(P, [(x["P"], x["r"][1]) for x in meas]), interp_c(P, [(x["P"], x["rR"]) for x in meas]))


def ip_d_echo(dan, psi, M):
    """v3 ip_d_echo(mode 'db'): measured d anchors of this psi interpolated in 1/P at P = M/3 in dB, held beyond the ends; None when the zone of M holds no measured anchor"""
    meas = [x for x in dan.get(psi, []) if not x["carried"] and x["e"] is not None]
    if not any(x["zone"] == ZCLASS["x%02d" % M] for x in meas):
        return None
    return interp_r(M / 3.0, [(x["P"], x["e"]) for x in meas])


def make_opts(anchors, dan):
    """v3 make_opts(rmap='global')"""
    opts = {}
    for M in CLASSES:
        o = {}
        if M in anchors:
            rA, rB = anchors[M]["r"]
            o["y"] = (rA, rB, anchors[M].get("rR", 1.0))
        for psi in PSIS:
            q = ratio_for_class(M, psi, dan)
            if q is not None and not (q[0] == 1 and q[1] == 1 and q[2] == 1):
                o["d%d" % psi] = q
        if o:
            opts[M] = o
    return opts


def enumerate_combos(opts):
    """v3 evaluate() enumeration order: number of non-x classes ascending, class subsets in combinations order, option product"""
    keys = sorted(opts)
    names = {M: [n for n in ORDER if n in opts[M]] for M in keys}
    out = []
    for n in range(len(keys) + 1):
        for use in itertools.combinations(keys, n):
            for nm in itertools.product(*[names[M] for M in use]):
                out.append(dict(zip(use, nm)))
    return out


def echoes(EX, anchors, dan, swap=False):
    """v3 echoes(emap='ip_db') -> class echoes (share-weighted); EX = class x echoes of the channel"""
    ec_s = {}
    for M in CLASSES:
        c = "x%02d" % M
        ex = EX[M]
        z = ZCLASS[c]
        ey = anchors[M]["ey"] if (M in anchors and anchors[M].get("ey") is not None) else None
        ed = {psi: ip_d_echo(dan, psi, M) for psi in PSIS}
        has_d = any(v is not None for v in ed.values())
        if ey is None and not has_d:
            ec_s[c] = ex
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
    return ec_s


def aperture_of(ec):
    return V1.aperture({c: ec[c] for c in V1.CL})


def echo_summary(EX, anchors, dan):
    ne = aperture_of(echoes(EX, anchors, dan))
    sw = aperture_of(echoes(EX, anchors, dan, swap=True))
    return dict(NE=ne, swapped=sw, agnostic=max(ne, sw))


# ====================================================================== cell readers (v3 arithmetic) =============================================================================
_C = {}


def pb2():
    """pb2_common (v3's reader / converter), imported lazily"""
    if "m" not in _C:
        sys.path.insert(0, PB)
        import pb2_common as C                                                   # noqa: E402
        _C["m"] = C
    return _C["m"]


jl = V1.jload


def beta_pc(p):
    C = pb2()
    c = C.read_cell(p, +1)
    out = []
    for a in ("A", "B"):
        b = c[a]["b_ac"] * (cmath.exp(-2j * math.pi * c["sc"] / 3.0) if a == "A" else 1.0)
        out.append(C.convert(b, c[a]["kappa"], a, c["G_delta"], "power_centre", k0=C.K0_TM))
    return out, c


def node_pc(paths):
    bs = [beta_pc(p)[0] for p in paths]
    return [sum(b[k] for b in bs) / len(bs) for k in (0, 1)]


def pmean_e(paths):
    return V1.pmean([jl(p)["needle_results"]["production"]["echo_p2_db"] for p in paths])


def r0_of(jj):
    q = jj["needle_results"]["production"]
    return complex(*q["amp_rB_zero"]) / complex(q["lpa"]["rB0"]["a_pred_re"], q["lpa"]["rB0"]["a_pred_im"])


def y_anchor(ypaths, xpaths, EXM):
    """lensdir45_readout_v3.y_anchors arithmetic for one class.  ypaths: one real y cell, or several cells (planting needle: node of the x partners as the 'y' cell)."""
    if len(ypaths) == 1:
        by = beta_pc(ypaths[0])[0]
        j = jl(ypaths[0])
        echo_y = j["needle_results"]["production"]["echo_p2_db"]
        r0y = r0_of(j)
    else:
        by = node_pc(ypaths)
        echo_y = pmean_e(ypaths)
        r0y = sum(r0_of(jl(p)) for p in ypaths) / len(ypaths)
    bx = node_pc(xpaths)
    rR = r0y / (sum(r0_of(jl(p)) for p in xpaths) / len(xpaths))
    off = EXM - pmean_e(xpaths)
    ey = echo_y + off
    return dict(r=[by[0] / bx[0], by[1] / bx[1]], rR=rR, ey=ey, x201_vs_x401_echo=off, y_echo_raw=echo_y)


def pair_paths(ch, M):
    if ch == "1550":
        yp = R779 + r"\261011_sim779_dir_y%02d_e201" % M
        xp = [R779 + r"\261011_sim779_dir_x%02de201_sh%d" % (M, sh) for sh in (0, 1, 2)]
    else:
        tag = V1.CHANNELS[ch]["tag"]
        yp = R789 + r"\261013_sim789_y%02de201_%s" % (M, tag)
        xp = [R789 + r"\261013_sim789_x%02de201_sh%d_%s" % (M, sh, tag) for sh in (0, 1, 2)]
    return yp, xp


def x36_paths(ch):
    return [V1.stored_dir(36, sh) if ch == "1550" else V1.band_dir(36, sh, V1.CHANNELS[ch]["tag"]) for sh in (0, 1, 2)]


def validate_pair(ch, M, yp, xps, twin_md5):
    """refusal rules for one (y cell, 3 x partner cells) set at channel ch; returns the list of problems (empty = usable).  Nothing is ever filled in."""
    why = []
    nom = V1.CHANNELS[ch]["lam_nm"]
    info = {}

    def chk(p, axis, sh, tagname):
        if not os.path.isfile(os.path.join(p, "superramp_result.json")):
            why.append("%s: no superramp_result.json (%s)" % (tagname, p))
            return None
        j = jl(p)
        r = j["needle_results"]["production"]
        if (j.get("rcwa_rule"), j.get("rcwa_ff")) != ("liy", 1):
            why.append("%s: not a Li cell (rcwa_rule %r rcwa_ff %r)" % (tagname, j.get("rcwa_rule"), j.get("rcwa_ff")))
        if j.get("pol") != "TM":
            why.append("%s: pol %r != TM" % (tagname, j.get("pol")))
        if j.get("ramp_axis_requested") != axis or r.get("ramp_axis") != axis:
            why.append("%s: ramp axis %r/%r != %s" % (tagname, j.get("ramp_axis_requested"), r.get("ramp_axis"), axis))
        if j.get("ramp_s") != -1 or int(r["ramp_s"]) != -1:
            why.append("%s: ramp_s %r != -1" % (tagname, j.get("ramp_s")))
        if int(j.get("sc_shift", -9)) != sh or int(r["sc_shift"]) != sh:
            why.append("%s: sc_shift %r/%r != %d" % (tagname, j.get("sc_shift"), r["sc_shift"], sh))
        if int(r["M"]) != M:
            why.append("%s: M %r != %d" % (tagname, r["M"], M))
        if j.get("nG_ref") != 201 or j.get("nG_ref_matched") is not True:
            why.append("%s: nG_ref %r matched %r (expected 201 / True)" % (tagname, j.get("nG_ref"), j.get("nG_ref_matched")))
        jlam = V1.job_lams(p)
        if ch == "1550":
            if j.get("lam_eval_override") is not None or j.get("lam_eval_nm") is not None:
                why.append("%s: 1550 cell carries lam_eval keys" % tagname)
            if jlam is not None and abs(jlam - 1550.0) > 1e-9:
                why.append("%s: job lams_nm %r != 1550" % (tagname, jlam))
        else:
            if j.get("lam_eval_override") != 1:
                why.append("%s: lam_eval_override %r != 1" % (tagname, j.get("lam_eval_override")))
            if j.get("lam_eval_nm") is None or abs(float(j["lam_eval_nm"]) - nom) > 1e-9:
                why.append("%s: lam_eval_nm %r != %s" % (tagname, j.get("lam_eval_nm"), nom))
            if j.get("atom_lams_nm") is None or abs(float(j["atom_lams_nm"]) - 1550.0) > 1e-9:
                why.append("%s: atom_lams_nm %r != 1550" % (tagname, j.get("atom_lams_nm")))
            if jlam is None or abs(jlam - nom) > 1e-9:
                why.append("%s: job lams_nm %r != %s" % (tagname, jlam, nom))
        md5 = [V1.m8(a) for a in r["atom_paths"]]
        info[tagname] = dict(j=j, md5=md5)
        return j

    jy = chk(yp, "y", 0, "y")
    jxs = [chk(p, "x", sh, "x_sh%d" % sh) for sh, p in enumerate(xps)]
    if jy is not None and all(x is not None for x in jxs):
        for k in ("pol", "ref_mode", "nG_eval_ref", "nG_ref", "ramp_s"):
            for sh, jx in enumerate(jxs):
                if jy.get(k) != jx.get(k):
                    why.append("y vs x_sh%d differ in %s (%r vs %r)" % (sh, k, jy.get(k), jx.get(k)))
        if info["y"]["md5"] != info["x_sh0"]["md5"]:
            why.append("y atom md5 list != x_sh0 atom md5 list")
        for sh in (0, 1, 2):
            if info["x_sh%d" % sh]["md5"] != twin_md5[sh]:
                why.append("x_sh%d atom md5 list != the adopted twin cell at this channel" % sh)
    return why


def host_bt(p):
    """(pin-host, blas_threads) of a cell's job file (informational)"""
    host, bt = "?", "?"
    try:
        for fn in os.listdir(p):
            if fn.startswith("job_") and fn.endswith(".txt"):
                txt = io.open(os.path.join(p, fn), encoding="utf-8", errors="replace").read()
                m = re.search(r"pin-host\]\s*([A-Za-z0-9-]+)", txt)
                host = m.group(1) if m else "?"
                m = re.search(r"^blas_threads\s*=\s*(\d+)", txt, re.M)
                bt = m.group(1) if m else "?"
                break
    except OSError:
        pass
    return host, bt


# ====================================================================== parallel betaL ===========================================================================================
_W = {}


def _init_worker(ctx_small):
    _W["ctx"] = ctx_small
    _W["geo"] = {}


def _geo(ch):
    if ch not in _W["geo"]:
        _W["geo"][ch] = V1.Geo("fixed", _W["ctx"][ch][0])
    return _W["geo"][ch]


def _eval_chunk(args):
    ch, items = args
    lam_um, DFC, PRT = _W["ctx"][ch]
    geo = _geo(ch)
    out = []
    for idx, nd in items:
        b = geo.betaL(nd)
        out.append((idx, b[0] + DFC + PRT, b[1] + DFC + PRT))                     # same order as lens_from_nodes: (b0 + DFC) + PRT
    return out


class Eval:
    def __init__(self, ctxs, workers):
        self.ctx_small = {ch: (c["lam_um"], c["DFC"], c["PRT"]) for ch, c in ctxs.items()}
        self.workers = workers
        self.ex = None
        self.n = 0
        if workers > 0:
            self.ex = cf.ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(self.ctx_small,))
        else:
            _init_worker(self.ctx_small)

    def totals(self, ch, nds, chunk=6):
        """[(decay, held)] for the list of node dicts {M: {'A', 'B'}} (B already multiplied by the rB0 node), order preserved"""
        self.n += len(nds)
        items = list(enumerate(nds))
        res = [None] * len(nds)
        if self.ex is None:
            for idx, d, h in _eval_chunk((ch, items)):
                res[idx] = (d, h)
            return res
        tasks = [(ch, items[i:i + chunk]) for i in range(0, len(items), chunk)]
        for part in self.ex.map(_eval_chunk, tasks):
            for idx, d, h in part:
                res[idx] = (d, h)
        return res

    def close(self):
        if self.ex is not None:
            self.ex.shutdown()


def build_nodes(ctx, choice, opts):
    """nodes {M: {'A', 'B'}} of one combination (v3 _build + total: B = (ND_B * rB) * (NR * rR))"""
    nd = {}
    for M in CLASSES:
        o = choice.get(M)
        if o is None:
            nd[M] = {"A": ctx["ND"][M]["A"], "B": ctx["ND"][M]["B"] * ctx["NR"][M]}
        else:
            rA, rB, rR = opts[M][o]
            nd[M] = {"A": ctx["ND"][M]["A"] * rA, "B": ctx["ND"][M]["B"] * rB * (ctx["NR"][M] * rR)}
    return nd


def search(EV, ctx, anchors, dan):
    opts = make_opts(anchors, dan)
    combos = enumerate_combos(opts)
    tot = EV.totals(ctx["ch"], [build_nodes(ctx, c, opts) for c in combos])
    return dict(opts=opts, combos=combos, totals=tot)


def pick(S, allowed=None):
    """worst orientation per TOTAL component; ties keep x (strict < by 1e-15, v3).  allowed: set of option names admitted (None = all)."""
    best = [None, None]
    for c, t in zip(S["combos"], S["totals"]):
        if allowed is not None and any(v not in allowed for v in c.values()):
            continue
        for i in (0, 1):
            if best[i] is None or t[i] < best[i][0] - 1e-15:
                best[i] = (t[i], c)
    return dict(total=(best[0][0], best[1][0]), choice=(dict(best[0][1]), dict(best[1][1])))


def chosen_str(res, opts):
    return "; ".join("x%02d %s/%s" % (M, res["choice"][0].get(M, "x"), res["choice"][1].get(M, "x")) for M in sorted(opts))


def single_options(S):
    base = S["totals"][0]
    idx = {tuple(sorted(c.items())): i for i, c in enumerate(S["combos"])}
    rows = []
    for M in sorted(S["opts"]):
        for nm in [n for n in ORDER if n in S["opts"][M]]:
            t = S["totals"][idx[((M, nm),)]]
            rows.append((M, nm, t[0] - base[0], t[1] - base[1]))
    return rows


def dbdeg(z):
    return 20 * math.log10(abs(z)), math.degrees(cmath.phase(z))


def bits_equal(a, b):
    return struct.pack("<d", a) == struct.pack("<d", b)


def jd(o):
    return [o.real, o.imag] if isinstance(o, complex) else str(o)


# ====================================================================== d-echo transfer ==========================================================================================
def xref_echo(ch, EXch, partners):
    """x e201 echo per class and the 201->401 offset (v3 xref(): off = EX[m] - pmean e201); x36 = its own (eq-201) cells -> offset 0"""
    out = {}
    for M in XREF_M:
        if M == 36:
            xp = x36_paths(ch)
            for p in xp:
                assert jl(p).get("nG_ref") == 201, ("x36 cell is not eq-201", p)
        else:
            xp = partners[M]
        e201 = pmean_e(xp)
        out[M] = dict(e201=e201, off=EXch[M] - e201)
    return out


def transfer_d(dan0, XRch, XR1550, mode):
    """d anchors with ratios UNCHANGED and the echo e re-derived for the channel.  C: e_raw + channel offset (v3 rule); A: the 1550 e as is; B: d tracks the x echo (see header)."""
    out = {}
    for psi in PSIS:
        out[psi] = []
        for a in dan0[psi]:
            P = a["P"]
            off_ch = interp_r(P, [(m / 3.0, XRch[m]["off"]) for m in XREF_M])
            if mode == "A":
                e = a["e"]
            elif mode == "C":
                e = a["e_raw"] + off_ch
            elif mode == "B":
                dx = interp_r(P, [(m / 3.0, XRch[m]["e201"]) for m in XREF_M]) - interp_r(P, [(m / 3.0, XR1550[m]["e201"]) for m in XREF_M])
                e = a["e_raw"] + dx + off_ch
            else:
                raise ValueError(mode)
            out[psi].append(dict(a, e=e))
    return out


def load_d_anchors(j3):
    dan = {}
    for k, lst in j3["anchors_d"].items():
        dan[int(k)] = []
        for a in lst:
            b = dict(a)
            b["r"] = [complex(*a["r"][0]), complex(*a["r"][1])]
            b["rR"] = complex(*a["rR"])
            dan[int(k)].append(b)
    return dan


# ====================================================================== main ====================================================================================================
def main():
    global LG
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-needles", action="store_true")
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M") + (("_" + a.tag) if a.tag else "")
    LG = Log(os.path.join(WD, "lensband_orient_v3_stdout_%s.txt" % stamp))
    T0 = time.time()
    say("lensband_orient_v3.py  %s  affinity %s  workers %d  numpy-only, NO solver" % (time.strftime("%Y-%m-%d %H:%M:%S"), psutil.Process().cpu_affinity(), a.workers))
    mem0 = psutil.virtual_memory()
    say("memory at start: %.1f GB available of %.1f GB (%.0f %% used); each pool process is numpy-only, measured peak RSS reported at the end" % (mem0.available / 1e9, mem0.total / 1e9, mem0.percent))
    controls = []                                                                # (name, ok, text)

    def ctl(name, ok, text):
        controls.append((name, bool(ok), text))
        say("CONTROL %-34s %s  %s" % (name, "PASS" if ok else "FAIL", text))

    # ------------------------------------------------------------------ stored references
    j3 = json.load(io.open(V3_RESULT, encoding="utf-8"))
    v3v = json.load(io.open(V3_VERDICT, encoding="utf-8"))
    ys = json.load(io.open(Y_STORED, encoding="utf-8"))
    v2 = json.load(io.open(V2_RESULT, encoding="utf-8"))
    dan0 = load_d_anchors(j3)
    assert not j3["interim"] and not j3["pending"] and all(not a_["carried"] for psi in PSIS for a_ in dan0[psi]), "v3 stored result is not a complete (non-interim) one"
    say("stored references: v3 result %s (interim %s, pending %s) ; y read-out %s ; lensband_eval_v2 %s" % (V3_RESULT, j3["interim"], j3["pending"], Y_STORED, V2_RESULT))

    # ------------------------------------------------------------------ channel cells (lensband_eval gather) and atom terms (lensband_eval_v2 stored)
    g1550 = V1.gather("1550")
    G = {"1550": g1550}
    twins = dict(g1550["cells"])
    for ch in ("1546.003", "1554.017"):
        G[ch] = V1.gather(ch, twins)
    for ch in CHS:
        miss, inval, lens_missing = V1.inventory(G[ch])
        if miss or inval or lens_missing:
            say("REFUSED: channel %s adopted cells not all usable: missing %s invalid %s classes %s" % (ch, miss, inval, lens_missing))
            LG.close("FAILED")
            return 3
    stdout_terms = {}
    for ln in io.open(V2_STDOUT, encoding="utf-8"):
        m = re.match(r"^ATOM TERMS at channel ([0-9.]+): deficit (\S+) purity f200 (\S+)\s*$", ln)
        if m:
            stdout_terms[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    TERMS = {"1550": (V1.DFC769, V1.PRT769)}
    for ch in ("1546.003", "1554.017"):
        at = v2["channels"][ch]["atom_terms"]
        TERMS[ch] = (at["deficit"], at["purity"])
        assert stdout_terms.get(ch) == TERMS[ch], ("atom terms of the v2 json differ from run_all_261013_1708.txt", ch, stdout_terms.get(ch), TERMS[ch])
    say("atom terms (deficit, purity f200) per channel, lensband_eval_v2 stored == run_all_261013_1708.txt: " + " | ".join("%s %+.9f %+.9f" % (ch, *TERMS[ch]) for ch in CHS))
    CTX = {}
    for ch in CHS:
        nodes = G[ch]["nodes"]
        CTX[ch] = dict(ch=ch, lam_um=V1.CHANNELS[ch]["lam_um"], DFC=TERMS[ch][0], PRT=TERMS[ch][1],
                       ND={M: {"A": nodes[M]["A"], "B": nodes[M]["B"]} for M in CLASSES}, NR={M: nodes[M]["rB0"] for M in CLASSES}, EX={M: nodes[M]["echo"] for M in CLASSES})

    # ------------------------------------------------------------------ y anchors + validation (refuse, never fill)
    ANCH, PAIR = {}, {}
    for ch in CHS:
        ANCH[ch], PAIR[ch] = {}, {}
        for M in YM:
            yp, xps = pair_paths(ch, M)
            why = validate_pair(ch, M, yp, xps, {sh: G[ch]["cells"][(M, sh)]["md5"] for sh in (0, 1, 2)})
            if why:
                say("REFUSED: %s y%02d: %s" % (ch, M, " | ".join(why)))
                LG.close("FAILED")
                return 3
            PAIR[ch][M] = (yp, xps)
            ANCH[ch][M] = y_anchor([yp], xps, CTX[ch]["EX"][M])
    say("input validation: all 12 (channel, class) y / x-partner sets valid (Li, TM, axis, ramp_s -1, sc_shift, nG_ref 201 matched, lam_eval keys, job lams_nm, atom md5 lists == the adopted twin cells)")
    say("per-cell pin / blas_threads (informational; the rule 'blas_threads must match across a split' is about the mechanism):")
    for ch in CHS:
        for M in YM:
            yp, xps = PAIR[ch][M]
            hy = host_bt(yp)
            hx = sorted(set(host_bt(p) for p in xps))
            say("   %-8s y%02d host %s bt %s | x partners %s" % (ch, M, hy[0], hy[1], "; ".join("%s bt %s" % h for h in hx)))
    XR = {}
    for ch in CHS:
        XR[ch] = xref_echo(ch, CTX[ch]["EX"], {M: PAIR[ch][M][1] for M in YM})
    say("echo offsets (EX - pmean e201, dB) per class x12 x15 x18 x24 x36: " + " | ".join("%s " % ch + " ".join("%+.3f" % XR[ch][M]["off"] for M in XREF_M) for ch in CHS))

    EV = Eval(CTX, a.workers)
    say("== computing (pool of %d processes): full 768-combination x/y/d enumeration + standalone x/y enumeration per channel ..." % a.workers)
    S, SXY, DAN = {}, {}, {}
    for ch in CHS:
        DAN[ch] = transfer_d(dan0, XR[ch], XR["1550"], "C")
        t = time.time()
        S[ch] = search(EV, CTX[ch], ANCH[ch], DAN[ch])
        SXY[ch] = search(EV, CTX[ch], ANCH[ch], {225: [], 315: []})
        say("   %-8s %d + %d combinations evaluated in %.0f s (elapsed %.0f s)" % (ch, len(S[ch]["combos"]), len(SXY[ch]["combos"]), time.time() - t, time.time() - T0))

    # variants
    VAR = {}
    off_d = {225: [], 315: []}
    for ch in CHS:
        EXc = CTX[ch]["EX"]
        vx = dict(total=S[ch]["totals"][0], choice=({}, {}))
        vxy = pick(S[ch], allowed={"y"})
        vxyd = pick(S[ch])
        ap_x = aperture_of({"x%02d" % M: EXc[M] for M in CLASSES})
        e_xy = echo_summary(EXc, ANCH[ch], off_d)
        e_xyd = echo_summary(EXc, ANCH[ch], DAN[ch])
        e_alt = {m: echo_summary(EXc, ANCH[ch], transfer_d(dan0, XR[ch], XR["1550"], m)) for m in ("A", "B")}
        VAR[ch] = dict(x=dict(res=vx, echo=dict(NE=ap_x, swapped=ap_x, agnostic=ap_x)), xy=dict(res=vxy, echo=e_xy), xyd=dict(res=vxyd, echo=e_xyd), echo_alt=e_alt)

    # ------------------------------------------------------------------ CONTROLS
    say("=" * 140)
    say("CONTROLS (every one must PASS before any band-edge number is printed as a result)")
    # (a) V-x == lensband_eval_v2
    for ch in CHS:
        m = v2["channels"][ch]["models"]["fixed"]
        st_t, st_e = m["full"]["res"]["rB0_complex"]["total"], m["cells_only"]["echo_aperture"]
        mine_t, mine_e = VAR[ch]["x"]["res"]["total"], VAR[ch]["x"]["echo"]["NE"]
        d = [abs(mine_t[0] - st_t[0]), abs(mine_t[1] - st_t[1]), abs(mine_e - st_e)]
        be = bits_equal(mine_t[0], st_t[0]) and bits_equal(mine_t[1], st_t[1]) and bits_equal(mine_e, st_e)
        txt = "V-x vs lensband_eval_v2 stored: max |diff| total decay/held %.2e / %.2e dB, echo %.2e dB (bit-identical: %s)" % (d[0], d[1], d[2], be)
        if ch == "1550":
            txt += " ; values %+.10f / %+.10f echo %.6f" % (mine_t[0], mine_t[1], mine_e)
        ctl("(a) V-x == lensband_eval_v2 @%s" % ch, max(d) <= TOL, txt)
    # pool == main process, one combination (bit)
    ch0 = "1550"
    chk_nd = build_nodes(CTX[ch0], {}, S[ch0]["opts"])
    geo_m = V1.Geo("fixed", CTX[ch0]["lam_um"])
    bm = geo_m.betaL(chk_nd)
    tm = (bm[0] + CTX[ch0]["DFC"] + CTX[ch0]["PRT"], bm[1] + CTX[ch0]["DFC"] + CTX[ch0]["PRT"])
    ctl("(e) pool == main process", bits_equal(tm[0], S[ch0]["totals"][0][0]) and bits_equal(tm[1], S[ch0]["totals"][0][1]), "x-only total at 1550 computed in the main process %+.12f / %+.12f vs pool %+.12f / %+.12f" % (
        tm[0], tm[1], S[ch0]["totals"][0][0], S[ch0]["totals"][0][1]))
    for ch in CHS:
        sub = pick(SXY[ch])
        same = bits_equal(sub["total"][0], VAR[ch]["xy"]["res"]["total"][0]) and bits_equal(sub["total"][1], VAR[ch]["xy"]["res"]["total"][1]) and sub["choice"] == VAR[ch]["xy"]["res"]["choice"]
        ctl("(e) V-xy standalone == subset @%s" % ch, same, "standalone x/y enumeration (%d combos) == y-only subset of the x/y/d table: %s" % (len(SXY[ch]["combos"]), same))
    # (b) 1550 == v3
    ch = "1550"
    a_stored = {int(k): v for k, v in j3["anchors_y"].items()}
    bad = []
    for M in YM:
        s_, m_ = a_stored[M], ANCH[ch][M]
        for lab, x, y in (("rA", complex(*s_["r"][0]), m_["r"][0]), ("rB", complex(*s_["r"][1]), m_["r"][1]), ("rR", complex(*s_["rR"]), m_["rR"])):
            if not (bits_equal(x.real, y.real) and bits_equal(x.imag, y.imag)):
                bad.append("y%02d %s %r vs %r" % (M, lab, x, y))
        for lab, x, y in (("ey", s_["ey"], m_["ey"]), ("off", s_["x201_vs_x401_echo"], m_["x201_vs_x401_echo"])):
            if not bits_equal(x, y):
                bad.append("y%02d %s %r vs %r" % (M, lab, x, y))
    ys_anch = {int(k): v for k, v in ys["anchors"].items()}
    bad2 = [("y%02d" % M) for M in YM if not (complex(*ys_anch[M]["r"][0]) == ANCH[ch][M]["r"][0] and complex(*ys_anch[M]["r"][1]) == ANCH[ch][M]["r"][1] and complex(*ys_anch[M]["rR"]) == ANCH[ch][M]["rR"]
                                          and ys_anch[M]["ey"] == ANCH[ch][M]["ey"])]
    ctl("(b) y anchors 1550 == stored", not bad and not bad2, "sim779 cells re-read: %d y anchors (r A, r B, rR, ey, offset) bit-identical to lensdir45_readout_v3_result.json anchors_y (%d diffs) and to lensdir_readout_result.json (%d diffs)" % (len(YM), len(bad), len(bad2)))
    sh_diff = [z for z in SHARES if not all(SHARES[z][k] == j3["shares"][z][k] for k in ("x", "y", "dm", "dp")) or list(SHARES[z]["raw"]) != list(j3["shares"][z]["raw"])]
    ctl("(b) zone shares == stored", not sh_diff, "shares re-read from oblique_class_share_stdout.txt: " + " ".join("%s x%.3f y%.3f d-%.3f d+%.3f" % (z, s["x"], s["y"], s["dm"], s["dp"]) for z, s in SHARES.items()))
    # d echo offset rule re-derived at 1550 == stored e
    bade = []
    for psi in PSIS:
        for a0, a1 in zip(dan0[psi], DAN["1550"][psi]):
            if not bits_equal(a0["e"], a1["e"]):
                bade.append("d%d psi%d e %r vs %r" % (a0["cls"], psi, a0["e"], a1["e"]))
    ctl("(b) d-echo offset rule @1550 == stored", not bade, "e_raw + off (off from the x e201 partners re-read here, offsets %s) == stored d echoes of all 6 anchors (%d diffs)" % (
        " ".join("x%d %+.4f" % (M, XR["1550"][M]["off"]) for M in XREF_M), len(bade)))
    # V-xy == frozen y read-out and v3 x_y
    r_y = ys["result"]
    xy = VAR[ch]["xy"]
    ec_xy = echoes(CTX[ch]["EX"], ANCH[ch], off_d)
    d_tot = max(abs(xy["res"]["total"][0] - r_y["total"][0]), abs(xy["res"]["total"][1] - r_y["total"][1]))
    d_echo = abs(xy["echo"]["NE"] - r_y["echo_share"])
    d_cls = max(abs(ec_xy[c] - r_y["class_echo_share"][c]) for c in r_y["class_echo_share"])
    cx_ = j3["configs"]["x_y"]
    d_tot3 = max(abs(xy["res"]["total"][0] - cx_["total"][0]), abs(xy["res"]["total"][1] - cx_["total"][1]))
    d_echo3 = abs(xy["echo"]["NE"] - cx_["echo_share"])
    ch_ok = all([xy["res"]["choice"][0].get(int(M), "x"), xy["res"]["choice"][1].get(int(M), "x")] == u["chosen"] for M, u in cx_["used"].items())
    yw_ok = all((bool(r_y["used"][str(M)]["y_worse"]) == (xy["res"]["choice"][0].get(M) == "y" or xy["res"]["choice"][1].get(M) == "y")) for M in YM)
    ctl("(b) V-xy @1550 == stored x+y", max(d_tot, d_echo, d_cls, d_tot3, d_echo3) <= TOL and ch_ok and yw_ok,
        "V-xy %+.10f / %+.10f echo %.10f ; frozen y read-out %+.10f / %+.10f echo %.10f ; max |diff| total %.2e echo %.2e class echoes %.2e ; vs v3 x_y config total %.2e echo %.2e ; chosen orientations equal %s, y_worse flags equal %s ; bit-identical total %s" % (
            xy["res"]["total"][0], xy["res"]["total"][1], xy["echo"]["NE"], r_y["total"][0], r_y["total"][1], r_y["echo_share"], d_tot, d_echo, d_cls, d_tot3, d_echo3, ch_ok, yw_ok,
            bits_equal(xy["res"]["total"][0], r_y["total"][0]) and bits_equal(xy["res"]["total"][1], r_y["total"][1])))
    # V-xyd == v3 verdict
    xd = VAR[ch]["xyd"]
    cfg = j3["configs"]["x_y_d"]
    d_tot = max(abs(xd["res"]["total"][0] - cfg["total"][0]), abs(xd["res"]["total"][1] - cfg["total"][1]))
    d_ne, d_sw = abs(xd["echo"]["NE"] - cfg["echo_share"]), abs(xd["echo"]["swapped"] - cfg["echo_share_swap"])
    ecc = echoes(CTX[ch]["EX"], ANCH[ch], DAN[ch])
    d_cls = max(abs(ecc[c] - cfg["class_echo_share"][c]) for c in cfg["class_echo_share"])
    ch_ok = all([xd["res"]["choice"][0].get(int(M), "x"), xd["res"]["choice"][1].get(int(M), "x")] == u["chosen"] for M, u in cfg["used"].items())
    nc_ok = len(S[ch]["combos"]) == cfg["n_combinations"]
    vline = "VERDICT (1/P map, 2026-10-12): TOTAL %.4f / %.4f dB vs %.2f -> %s ; echo %.4f dB (quadrant-agnostic %.4f) vs %.1f -> %s" % (
        xd["res"]["total"][0], xd["res"]["total"][1], GATE_TOTAL, "PASS" if all(t - GATE_TOTAL >= 0 for t in xd["res"]["total"]) else "FAIL", xd["echo"]["NE"], xd["echo"]["agnostic"], GATE_ECHO,
        "PASS" if (xd["echo"]["NE"] <= GATE_ECHO and xd["echo"]["agnostic"] <= GATE_ECHO) else "FAIL")
    # stored outputs may carry an older label variant: compare after normalising the "VERDICT (1/P map, " prefix
    vl_ok = (re.sub(r"^VERDICT \(1/P map, (?:[A-Z]+ )?", "VERDICT (1/P map, ", vline) == re.sub(r"^VERDICT \(1/P map, (?:[A-Z]+ )?", "VERDICT (1/P map, ", v3v["verdict_line"]))
    ctl("(b) V-xyd @1550 == v3 verdict", max(d_tot, d_ne, d_sw, d_cls) <= TOL and ch_ok and nc_ok and vl_ok,
        "V-xyd %+.10f / %+.10f echo NE %.8f swapped %.8f agnostic %.8f ; v3 %+.10f / %+.10f echo NE %.8f swapped %.8f ; max |diff| total %.2e NE %.2e swapped %.2e class echoes %.2e ; combinations %d == %d ; chosen equal %s ; verdict line identical %s ; bit-identical total %s" % (
            xd["res"]["total"][0], xd["res"]["total"][1], xd["echo"]["NE"], xd["echo"]["swapped"], xd["echo"]["agnostic"], cfg["total"][0], cfg["total"][1], cfg["echo_share"], cfg["echo_share_swap"],
            d_tot, d_ne, d_sw, d_cls, len(S[ch]["combos"]), cfg["n_combinations"], ch_ok, vl_ok, bits_equal(xd["res"]["total"][0], cfg["total"][0]) and bits_equal(xd["res"]["total"][1], cfg["total"][1])))
    say("            1550 verdict line reproduced: " + vline)
    say("            v3 stored verdict line:       " + v3v["verdict_line"])

    # (c) needles
    if not a.no_needles:
        g = 10 ** (-0.1 / 20.0)
        for ch in CHS:
            planted = {}
            for M in YM:
                _, xps = PAIR[ch][M]
                planted[M] = y_anchor(list(xps), xps, CTX[ch]["EX"][M])            # the x partners themselves as the y 'cells'
            worst = max(max(abs(planted[M]["r"][0] - 1), abs(planted[M]["r"][1] - 1), abs(planted[M]["rR"] - 1)) for M in YM)
            exact = all(planted[M]["r"][0] == 1 and planted[M]["r"][1] == 1 and planted[M]["rR"] == 1 for M in YM)
            Sp = search(EV, CTX[ch], planted, {225: [], 315: []})
            vp = pick(Sp)
            e_p = echo_summary(CTX[ch]["EX"], planted, off_d)["NE"]
            vx = VAR[ch]["x"]
            dt = max(abs(vp["total"][0] - vx["res"]["total"][0]), abs(vp["total"][1] - vx["res"]["total"][1]))
            de = abs(e_p - vx["echo"]["NE"])
            ctl("(c) needle partners-as-y @%s" % ch, worst <= 1e-15 and dt <= TOL and de <= 1e-9,
                "y ratio max |r - 1| %.2e (exactly 1+0j: %s), V-xy vs V-x |diff| total %.2e dB, echo %.2e dB" % (worst, exact, dt, de))
            for M in (12, 24):
                pl = {M: dict(planted[M], r=[g, g], rR=1.0 + 0j)}
                Sm = search(EV, CTX[ch], pl, {225: [], 315: []})
                vm = pick(Sm)
                nd = {k: {"A": CTX[ch]["ND"][k]["A"] * (g if k == M else 1.0), "B": CTX[ch]["ND"][k]["B"] * (g if k == M else 1.0) * CTX[ch]["NR"][k]} for k in CLASSES}
                orc = EV.totals(ch, [nd])[0]
                moved = [vx["res"]["total"][i] - vm["total"][i] for i in (0, 1)]
                ctl("(c) needle y -0.1 dB on x%02d @%s" % (M, ch), all(m_ > 1e-6 for m_ in moved) and abs(vm["total"][0] - orc[0]) <= TOL and abs(vm["total"][1] - orc[1]) <= TOL
                    and vm["choice"][0].get(M) == "y" and vm["choice"][1].get(M) == "y",
                    "total moves by %+.6f / %+.6f dB (V-x minus planted), independent betaL oracle |diff| %.2e / %.2e, chosen %s/%s" % (
                        -moved[0], -moved[1], abs(vm["total"][0] - orc[0]), abs(vm["total"][1] - orc[1]), vm["choice"][0].get(M, "x"), vm["choice"][1].get(M, "x")))
        # refusal needle: the 1546 y12 / x12 cells presented as the 1554 set must be refused
        yp, xps = pair_paths("1546.003", 12)
        why = validate_pair("1554.017", 12, yp, xps, {sh: G["1554.017"]["cells"][(12, sh)]["md5"] for sh in (0, 1, 2)})
        ctl("(c) refusal needle wrong channel", any("lam_eval_nm" in w for w in why) and len(why) >= 4, "1546.003 y12 + x12 partners presented as a 1554.017 set: %d problems, e.g. %s" % (len(why), why[0][:90] if why else "-"))
        yp55, xps55 = pair_paths("1550", 12)
        why = validate_pair("1550", 12, yp55, xps55, {sh: G["1550"]["cells"][(12, sh)]["md5"] for sh in (0, 1, 2)})
        ctl("(c) refusal positive control 1550", not why, "sim779 y12 + x12 partners at 1550: problems %s" % (why or "none"))
    else:
        say("(needles skipped: --no-needles)")

    allok = all(ok for _, ok, _ in controls)
    say("CONTROL SUMMARY: %d controls, %d PASS, %d FAIL  ->  %s" % (len(controls), sum(1 for _, ok, _ in controls if ok), sum(1 for _, ok, _ in controls if not ok), "ALL PASS" if allok else "FAILURES: " + ", ".join(n for n, ok, _ in controls if not ok)))
    rname = os.path.join(WD, "lensband_orient_v3_result_%s.json" % stamp)
    out = dict(stamp=stamp, controls=[dict(name=n, passed=ok, text=t) for n, ok, t in controls], all_controls_pass=allok)
    if not allok:
        say("RESULTS WITHHELD: a control failed -- no band-edge number is printed as a result.")
        with io.open(rname, "x", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=jd)
        EV.close()
        LG.close("FAILED")
        return 1

    # ------------------------------------------------------------------ RESULTS
    say("=" * 140)
    say("RESULTS  (TOTAL = rB0 complex, decayed / held tail, dB, margin to %.2f; echo = share-weighted aperture dB, margin to %.1f; PASS needs both tails / (NE and quadrant-agnostic echo))" % (GATE_TOTAL, GATE_ECHO))
    say("lens model 'fixed' (1550-designed lens, propagation / obliquity at the channel); atom terms of sim783 at the channel; d options = the 1550 d/x ratios TRANSFERRED unchanged (assumption, no band-edge 45-degree cell);")
    say("d echo = 'C' (e_raw of the 1550 d cell + the channel's 201->401 offset, v3 rule); y: channel's own sim789 y / x e201 pair (1550: sim779).")
    res_json = {}
    for ch in CHS:
        say("-" * 140)
        say("CHANNEL %s nm   atom terms deficit %+.6f purity %+.6f" % (ch, *TERMS[ch]))
        say("  variant   TOTAL decayed / held   margin decayed / held    gate  | echo NE / swapped / agnostic   margin(agn)  gate | chosen orientation per class (decayed / held)")
        res_json[ch] = {}
        for key, lab in (("x", "V-x  "), ("xy", "V-xy "), ("xyd", "V-xyd")):
            v = VAR[ch][key]
            t = v["res"]["total"]
            tp = all(x - GATE_TOTAL >= 0 for x in t)
            e = v["echo"]
            ep = e["NE"] <= GATE_ECHO and e["agnostic"] <= GATE_ECHO
            chs = "x only" if key == "x" else chosen_str(v["res"], S[ch]["opts"] if key == "xyd" else {M: S[ch]["opts"][M] for M in YM})
            say("  %s  %+.4f / %+.4f      %+.4f / %+.4f         %s  | %8.3f / %8.3f / %8.3f   %+7.3f     %s | %s" % (
                lab, t[0], t[1], t[0] - GATE_TOTAL, t[1] - GATE_TOTAL, "PASS" if tp else "FAIL", e["NE"], e["swapped"], e["agnostic"], GATE_ECHO - e["agnostic"], "PASS" if ep else "FAIL", chs))
            res_json[ch][key] = dict(total=list(t), margin=[x - GATE_TOTAL for x in t], total_pass=bool(tp), echo=dict(e), echo_margin_agnostic=GATE_ECHO - e["agnostic"], echo_pass=bool(ep),
                                     chosen=[{str(M): o for M, o in v["res"]["choice"][i].items()} for i in (0, 1)])
        ea = VAR[ch]["echo_alt"]
        say("  V-xyd echo, d-echo transfer sensitivity: C (primary) NE %.3f agnostic %.3f | A (1550 d echo as is) NE %.3f agnostic %.3f | B (d tracks the x echo) NE %.3f agnostic %.3f dB (gate %.1f)" % (
            VAR[ch]["xyd"]["echo"]["NE"], VAR[ch]["xyd"]["echo"]["agnostic"], ea["A"]["NE"], ea["A"]["agnostic"], ea["B"]["NE"], ea["B"]["agnostic"], GATE_ECHO))
        res_json[ch]["echo_alt"] = ea
        so = single_options(S[ch])
        say("  one class at a time (others keep x), dTOTAL vs V-x decayed / held: " + "; ".join("x%02d %s %+.4f/%+.4f" % r for r in so))
        res_json[ch]["single_options"] = so
        say("  x-only class echoes (dB): " + " ".join("x%02d %.2f" % (M, CTX[ch]["EX"][M]) for M in CLASSES))
    say("=" * 140)
    say("y / x e201 RATIOS per class and channel (power_centre, y cell / 3-registration x partner node; arm A, arm B, rB0; dB and deg); 1550 = sim779 (the stored v3 values)")
    say("  class channel     A dB    A deg |   B dB    B deg | rB0 dB  rB0 deg | y echo basis-corrected / x echo (dB) | y raw echo")
    for M in YM:
        for ch in ("1550", "1546.003", "1554.017"):
            an = ANCH[ch][M]
            aA, pA = dbdeg(an["r"][0])
            aB, pB = dbdeg(an["r"][1])
            aR, pR = dbdeg(an["rR"])
            say("  x%02d  %-8s %+7.3f %+8.2f | %+7.3f %+8.2f | %+7.3f %+8.2f | %8.3f / %8.3f | %8.3f" % (M, ch, aA, pA, aB, pB, aR, pR, an["ey"], CTX[ch]["EX"][M], an["y_echo_raw"]))
    say("  (the 1550 rows equal the v3 console lines 'y12: y-x A +0.110 B +0.117 dB, dphase -12.1 / -12.3 deg, rB0 y-x -0.560 dB ...')")
    say("D ratios used (1550 anchors, unchanged at every channel; class option = 1/P interpolation at P = M/3): " + "; ".join(
        "x%02d %s A %+.3f dB %+.1f deg B %+.3f dB rB0 %+.3f dB" % ((M, o) + (lambda q: (dbdeg(q[0])[0], dbdeg(q[0])[1], dbdeg(q[1])[0], dbdeg(q[2])[0]))(S["1550"]["opts"][M][o]))
        for M in sorted(S["1550"]["opts"]) for o in S["1550"]["opts"][M] if o != "y"))
    mi = psutil.Process().memory_info()
    say("main process peak working set %.2f GB ; workers %d (numpy only, no solver, affinity 10-17) ; betaL evaluations dispatched %d ; elapsed %.0f s" % (getattr(mi, "peak_wset", mi.rss) / 1e9, a.workers, EV.n, time.time() - T0))
    out.update(results=res_json, y_anchors={ch: {str(M): ANCH[ch][M] for M in YM} for ch in CHS}, atom_terms=TERMS, d_transfer="C primary (e_raw + channel offset); A/B sensitivities in results[ch].echo_alt",
               provenance=dict(script=os.path.abspath(__file__), script_md5=hashlib.md5(io.open(os.path.abspath(__file__), "rb").read()).hexdigest(), python=sys.version.split()[0], elapsed_s=time.time() - T0))
    with io.open(rname, "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=jd)
    say("wrote %s" % rname)
    EV.close()
    LG.close("COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
