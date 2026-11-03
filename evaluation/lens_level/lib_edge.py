# -*- coding: utf-8 -*-
r"""lensband_orient_v4_dmeas.py -- [TM] lens-level verdict at the band-edge channels (1546.003 / 1550 / 1554.017 nm) with the MEASURED 45-degree band-edge cells.
Task A of finished_sims_261016 (2026-10-16).  json + the lens evaluator only; NO solver.  Derived from <WORK_ROOT>\wlband_261012\lensband_orient_v3\lensband_orient_v3.py
(md5 dcc927ae79d025ac5bc9c92fd006d742; the verbatim copy is lensband_orient_v3_orig_copy.py beside this file) by make_v4.py: the v3 helper code (lines 33-561) is carried over with the
replacements listed in make_v4.py only (code folder -> the patched evaluator copies in ..\src_copies, pins 10-17 -> 10-15, WD -> this folder); everything below the marker "[v4]" is new.

WHAT CHANGED against v3 (v3 TRANSFERRED the six 1550 d anchors unchanged to the band edges; v4 reads the band-edge 45-degree cells that now exist):
  d cells at the edges: sim792 d18 / d24 (psi 225 / 315) = Result\261014_scdc_lensband_d\261014_sim792_d{18,24}p{225,315}_l{1546p003,1554p017} and
                        sim801 d12 / d15 (psi 225 / 315) = Result\261014_scdc_lensband_d\261015_sim801_d{12,15}p{225,315}_l{1546p003,1554p017}.   sim792 d36 is NOT run (held behind sim796 R3).
  ratio of a d cell = beta_pc(d) / [x e201 reference interpolated in 1/P at P_d] per arm (lensdir45_readout_v5 read_d_cell arithmetic, verbatim); the x e201 reference is the CHANNEL'S OWN:
                        x12..x24 = sim789 x{M}e201_sh0..2 of the channel, x36 = sim785 x36 sh0..2 (e201) of the channel; P < 4 (d12, d15): reference HELD at the x12 node (no x e201 below x12).
  d echo = d cell echo + the channel's own 201->401 offset interpolated in 1/P (v5 rule).   rB0 ratio r0(d) / interpolated r0(x e201) (v5 rule).
  1550 column: the v5 reader on the sim781 / sim800 cells with the sim779 / sim769 references, compared BIT FOR BIT with lensdir45_readout_v5_primary.json anchors_d (control).
HANDLING OF THE MISSING d36 (PROVISIONAL, three readings, never mixed):
  'pending'   d36 carried exactly as the 1550 v5 read-out carries a pending class: ratio 1, no measured anchor in the 6-12L zone -> x18 / x24 / x36 get NO d option and no d echo share (--pending-ok rule).  Pessimistic for TOTAL.
  'transfer'  d36 ratios of the 1550 v5 anchors TRANSFERRED unchanged (the v3 assumption), echo = 1550 e_raw + the channel's own offset (v3 mode C).
  (the v3 'transfer ALL' numbers of lensband_orient_v3 are quoted from its stdout for comparison, not recomputed.)
REFERENCE NOTE of the v5 verify: 'fixed' = x e201 reference HELD at x12 for P < 4 (the stored rule); 'corrected' = ratio / [x_pc(P)/x_pc(4)] on the adopted x06 x09 x12 power_centre nodes
  (shape_alt_local_v5.py after its 2026-10-16 fix; rB0 shape from the adopted rB0 nodes).  The corrected values are ESTIMATES by coordinate descent (steepest, both tails, two starts), method-checked against
  the full enumeration of the uncorrected variant.
TOTAL = every combination of per-class choices (x / y / d225 / d315) evaluated, minimum per tail, ties keep x (strict < by 1e-15), v3 enumeration order.  ECHO = share-weighted (v3 ip_db), NE and swapped.
Rules: nothing deleted, nothing under <DATA_ROOT> written, no solver; direct run: BLAS 1 thread, cores 10-15 only, pool processes numpy-only.
python -X utf8 -I -B lensband_orient_v4_dmeas.py --stage tables|enumerate [--workers 6] [--tag NAME] [--full1550]"""

import os
import sys

sys.dont_write_bytecode = True
EVD = r"<WORK_ROOT>\lensband_edge_261017\src"                 # [v4] patched evaluator copies (code)
EVD_ORIG = r"<WORK_ROOT>\wlband_261012\lensband_eval"                                # [v4] original folder, READ-ONLY (stored lensband_eval_v2 results)
PB = r"<WORK_ROOT>\te_beta_orientation_261005\power_basis_261006"
WD = r"<WORK_ROOT>\lensband_edge_261017"
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
V2_RESULT = EVD_ORIG + r"\lensband_eval_v2_result.json"
V2_STDOUT = EVD_ORIG + r"\run_all_261013_1708.txt"
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




# =====================================================================================================================================================================================
# [v4] NEW CODE (everything above this marker is the v3 helper code carried over by make_v4.py)
# =====================================================================================================================================================================================
import pickle                                                                    # noqa: E402

LDB = R + r"\261014_scdc_lensband_d"                                             # sim792 (d18 / d24 / d36 edges) + sim801 (d12 / d15 edges)
LD1550 = R + r"\261011_scdc_tmzone_licoldir"                                     # sim781 (d18 / d24 / d36) + sim800 (d12 / d15), 1550
V5DIR = r"<WORK_ROOT>\loop_261008\lens45_li_261011\readout45_v5_sim800_d15"
V5_PRIMARY = V5DIR + r"\lensdir45_readout_v5_primary.json"
V5_VERDICT = V5DIR + r"\lensdir45_verdict_1overP_v5.json"
V5_SHAPE_JSON = V5DIR + r"\shape_alt_local_v5.json"
EVD_ORIG_V3 = r"<WORK_ROOT>\wlband_261012\lensband_orient_v3\lensband_orient_v3_stdout_261014_0550_dev1.txt"
DS_OF = {225: +1, 315: -1}
D_CLASSES = (12, 15, 18, 24, 36)
PAIR_KEYS_D = ("nG_req", "nG_ref", "nG_eval_ref", "ramp_s", "sc_shift", "pol", "ramp_axis_requested", "ref_mode", "diag_sign", "li_raster", "li_inner_axis", "rcwa_rule", "rcwa_ff")
V5_FIXED = (-0.6390877001304132, -0.6601830620722883)                            # v5 primary TOTAL (decay, held) at 1550, fixed reference (x12 held)
V5_ECHO = -20.211601530947977
V5_CORR = (-0.6082828152, -0.6292225201)                                         # v5 verify [E] power_centre corrected reference (10 digits as printed)


def zone_of(P):
    """lensdir45_readout_v5.zone_of verbatim"""
    if P < 3.0:
        return "2-3L"
    if P < 4.0:
        return "3-4L"
    if P < 6.0:
        return "4-6L"
    if P <= 12.0:
        return "6-12L"
    return ">12L"


assert all(zone_of(M / 3.0) == ZCLASS["x%02d" % M] for M in CLASSES), "zone_of disagrees with the evaluator ZMAP"


def nearest_class(P):
    return min(CLASSES, key=lambda M: abs(1.0 / P - 3.0 / M))


def lattice_formula(n, lam, ds):
    a = lam / 3.0
    return (0.0, n * a), (lam, -lam * ds)


def ramp_geometry(L1, L2, ramp_s):
    det = L1[0] * L2[1] - L1[1] * L2[0]
    k1 = (L2[1] / det, -L2[0] / det)
    return 1.0 / math.hypot(k1[0], k1[1]), math.degrees(math.atan2(ramp_s * k1[1], ramp_s * k1[0])) % 360.0


def path769(M, sh):
    return V1.stored_dir(M, sh)


def d_dirs(ch, M, psi):
    """candidate result folders of the d cell (M, psi) at channel ch, first one with a result is used (v5 d_cands order at 1550)"""
    if ch == "1550":
        if M in (12, 15):
            return [LD1550 + r"\261015_sim800_d%02d_p%d_li_e201" % (M, psi)]
        sfx = {18: ("_bt12_r", "_bt12"), 24: ("_bt12",), 36: ("_bt14",)}[M]
        return [LD1550 + "\\" + "261011_sim781_d%02d_p%d_li_e201" % (M, psi) + s for s in sfx]
    tag = V1.CHANNELS[ch]["tag"]
    if M in (12, 15):
        return [LDB + r"\261015_sim801_d%02dp%d_%s" % (M, psi, tag)]
    return [LDB + r"\261014_sim792_d%02dp%d_%s" % (M, psi, tag)]


def d_result_path(ch, M, psi):
    for p in d_dirs(ch, M, psi):
        if os.path.isfile(os.path.join(p, "superramp_result.json")):
            try:
                jl(p)
                return p
            except Exception:                                                    # a file still being copied is not a result yet
                continue
    return None


def xref_ch(ch, EXch):
    """x e201 reference per class at channel ch: node (power_centre, registration-coherent), r0, e201 echo and the 201->401 offset (v5 xref arithmetic).  x36 = its own eq-201 adopted cells (offset 0)."""
    XR = {}
    for M in XREF_M:
        xp = x36_paths(ch) if M == 36 else pair_paths(ch, M)[1]
        for p in xp:
            jx = jl(p)
            assert (jx.get("rcwa_rule"), jx.get("rcwa_ff"), jx.get("ramp_s"), jx.get("ramp_axis_requested"), jx.get("nG_ref")) == ("liy", 1, -1, "x", 201), p
        e201 = pmean_e(xp)
        XR[M] = dict(P=M / 3.0, node=node_pc(xp), r0=sum(r0_of(jl(p)) for p in xp) / len(xp), e201=e201, off=EXch[M] - e201, paths=list(xp))
    return XR


def read_d_cell_ch(path, M, psi, ch, XR):
    """lensdir45_readout_v5.read_d_cell arithmetic (verbatim) at channel ch with the channel's own x e201 reference XR, plus the channel / twin checks"""
    j = jl(path)
    q = j["needle_results"]["production"]
    ds = DS_OF[psi]
    C = pb2()
    assert j["ramp_axis_requested"] == "d" and q["ramp_axis"] == "d" and j.get("pol") == "TM", path
    assert int(j["diag_sign"]) == ds and int(q["diag_sign"]) == ds, (path, j["diag_sign"], ds)
    assert (j.get("rcwa_rule"), j.get("rcwa_ff")) == ("liy", 1), path
    assert j.get("ramp_s") == -1 and int(j.get("sc_shift", 0)) == 0, path
    assert j.get("li_raster") == [[0, 1], [1, -ds]] and j.get("li_inner_axis") == 0, (path, j.get("li_raster"))
    assert j.get("nG_ref") == 201 and j.get("nG_ref_matched") is True, (path, j.get("nG_ref"), j.get("nG_ref_matched"))
    assert int(q["M"]) == M == len(j["atoms_requested"]), path
    ref_sh = 1 if M == 12 else 0
    jx = jl(path769(M, ref_sh))
    assert j["atoms_requested"] == jx["atoms_requested"] and q["atom_paths"] == jx["needle_results"]["production"]["atom_paths"], "atom list differs from the sim769 x%02d sh%d list" % (M, ref_sh)
    nom = V1.CHANNELS[ch]["lam_nm"]
    jlam = V1.job_lams(path)
    if ch == "1550":
        assert j.get("lam_eval_override") is None and j.get("lam_eval_nm") is None, ("1550 cell carries lam_eval keys", path)
        assert jlam is None or abs(jlam - 1550.0) < 1e-9, (path, jlam)
    else:
        assert j.get("lam_eval_override") == 1, (path, j.get("lam_eval_override"))
        assert j.get("lam_eval_nm") is not None and abs(float(j["lam_eval_nm"]) - nom) < 1e-9, (path, j.get("lam_eval_nm"), nom)
        assert j.get("atom_lams_nm") is not None and abs(float(j["atom_lams_nm"]) - 1550.0) < 1e-9, (path, j.get("atom_lams_nm"))
        assert jlam is not None and abs(jlam - nom) < 1e-9, (path, jlam, nom)
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
    host, bt = host_bt(path)
    return dict(cls=M, psi=psi, P=P, zone=zone_of(P), nearest_x="x%02d" % nearest_class(P), r=r, rR=rR, e_raw=q["echo_p2_db"], off=off, e=q["echo_p2_db"] + off, carried=False,
                src=path, psi_from_lattice=psi_g, kappa_over_K0=[c["A"]["kappa"] / pb2().K0_TM, c["B"]["kappa"] / pb2().K0_TM], nG_actual=q.get("nG_actual"), wall_s=j.get("wall_s"),
                b_ac=[c["A"]["b_ac"], c["B"]["b_ac"]], r0_cell=r0_of(j), host=host, blas_threads=bt, nG_req=j.get("nG_req"))


def pending_anchor_ch(M, psi):
    """lensdir45_readout_v5.pending_anchor semantics: carried at x, ratio 1, no echo; geometry from the class label"""
    P = M / (3.0 * math.sqrt(2.0))
    return dict(cls=M, psi=psi, P=P, zone=zone_of(P), nearest_x="x%02d" % nearest_class(P), r=[1 + 0j, 1 + 0j], rR=1 + 0j, e_raw=None, off=None, e=None, carried=True, src="PENDING (no result)", psi_from_lattice=float(psi))


def d_anchors_ch(ch, XR):
    """-> ({psi: [anchor ...]} for the classes that have a result, [(M, psi) missing])"""
    dan = {225: [], 315: []}
    missing = []
    for M in D_CLASSES:
        for psi in PSIS:
            p = d_result_path(ch, M, psi)
            if p is None:
                missing.append((M, psi))
                continue
            dan[psi].append(read_d_cell_ch(p, M, psi, ch, XR))
    return dan, missing


def twin_diff(path_edge, path_1550):
    """job-key differences between a band-edge d cell and its 1550 twin (PAIR_KEYS_D + atom md5 list)"""
    je, j0 = jl(path_edge), jl(path_1550)
    bad = [k for k in PAIR_KEYS_D if je.get(k) != j0.get(k)]
    me = [V1.m8(a) for a in je["needle_results"]["production"]["atom_paths"]]
    m0 = [V1.m8(a) for a in j0["needle_results"]["production"]["atom_paths"]]
    if me != m0:
        bad.append("atom md5 list")
    return bad


def alt_shape(dan, NDPC, NRc):
    """shape_alt_local_v5.alt(basis 'pc') arithmetic: anchors with P < 4 get ratio / [x_pc(P) / x_pc(4)] (x06 x09 x12 adopted nodes, 1/P rule), rB0 ratio / the same shape of the rB0 nodes"""
    out = {225: [], 315: []}
    for psi in PSIS:
        for a in dan[psi]:
            if a["carried"] or a["P"] >= 4.0:
                out[psi].append(a)
                continue
            sh = [interp_c(a["P"], [(m / 3.0, NDPC[m][arm]) for m in (6, 9, 12)]) / NDPC[12][arm] for arm in ("A", "B")]
            shR = interp_c(a["P"], [(m / 3.0, NRc[m]) for m in (6, 9, 12)]) / NRc[12]
            out[psi].append(dict(a, r=[a["r"][0] / sh[0], a["r"][1] / sh[1]], rR=a["rR"] / shR, shape=[dbdeg(sh[0]), dbdeg(sh[1])], shapeR=dbdeg(shR)))
    return out


def with_d36(dan_meas, mode, D1550, XRch):
    """d anchors of the channel with the d36 pair handled as 'pending' | 'transfer' (1550 ratios unchanged, v3 mode-C echo) | 'measured' (nothing to add)"""
    out = {psi: [a for a in dan_meas[psi] if a["cls"] != 36] for psi in PSIS}
    for psi in PSIS:
        have = [a for a in dan_meas[psi] if a["cls"] == 36]
        if have:
            out[psi].extend(have)
        elif mode == "pending":
            out[psi].append(pending_anchor_ch(36, psi))
        elif mode == "transfer":
            a0 = [a for a in D1550[psi] if a["cls"] == 36][0]
            off_ch = interp_r(a0["P"], [(m / 3.0, XRch[m]["off"]) for m in XREF_M])
            out[psi].append(dict(a0, e=a0["e_raw"] + off_ch, off=off_ch, transferred=True))
        else:
            raise ValueError(mode)
    for psi in PSIS:
        out[psi].sort(key=lambda a: a["cls"])
    return out


# ---------------------------------------------------------------------- evaluation with a persistent cache ------------------------------------------------------------------------
def combo_key(ch, choice, opts):
    return (ch,) + tuple((M, nm) + tuple((z.real, z.imag) for z in opts[M][nm]) for M, nm in sorted(choice.items()))


class Eval2(Eval):
    def totals(self, ch, nds, chunk=6):
        n = len(nds)
        self.n += n
        items = list(enumerate(nds))
        res = [None] * n
        if self.ex is None:
            for idx, d, h in _eval_chunk((ch, items)):
                res[idx] = (d, h)
            return res
        tasks = [(ch, items[i:i + chunk]) for i in range(0, n, chunk)]
        done, t0, nxt = 0, time.time(), 300
        for part in self.ex.map(_eval_chunk, tasks):
            for idx, d, h in part:
                res[idx] = (d, h)
            done += len(part)
            if n >= 300 and done >= nxt:
                el = time.time() - t0
                say("      ... %s %d / %d evaluated, %.0f s elapsed, ETA %.0f s" % (ch, done, n, el, el * (n - done) / max(done, 1)))
                nxt += 300
        return res


class Cache:
    def __init__(self, path):
        self.path = path
        self.d = {}
        if path and os.path.isfile(path):
            self.d = pickle.load(open(path, "rb"))
            say("evaluation cache loaded: %d entries from %s" % (len(self.d), path))

    def save(self):
        if self.path:
            tmp = self.path + ".tmp"
            pickle.dump(self.d, open(tmp, "wb"))
            os.replace(tmp, self.path)


def totals_for(EV, ctx, opts, choices, cache):
    keys = [combo_key(ctx["ch"], c, opts) for c in choices]
    uniq = {}
    for i, k in enumerate(keys):
        if k not in cache.d and k not in uniq:
            uniq[k] = i
    if uniq:
        nds = [build_nodes(ctx, choices[i], opts) for i in uniq.values()]
        tot = EV.totals(ctx["ch"], nds)
        for k, t in zip(uniq.keys(), tot):
            cache.d[k] = t
        cache.save()
    return [cache.d[k] for k in keys], len(uniq)


def search_c(EV, ctx, anchors, dan, cache):
    opts = make_opts(anchors, dan)
    combos = enumerate_combos(opts)
    tot, nnew = totals_for(EV, ctx, opts, combos, cache)
    return dict(opts=opts, combos=combos, totals=tot, n_new=nnew)


def descend(EV, ctx, anchors, dan, start, tail, cache):
    """steepest descent over single-class changes for one tail (0 decay / 1 held); returns (total, choice, evaluations requested)"""
    opts = make_opts(anchors, dan)
    names = {M: ["x"] + [n for n in ORDER if n in opts[M]] for M in opts}
    ch = {M: o for M, o in start.items() if o != "x" and M in opts}
    cur = totals_for(EV, ctx, opts, [ch], cache)[0][0][tail]
    n_req = 1
    while True:
        nb = []
        for M in sorted(names):
            for o in names[M]:
                if o == ch.get(M, "x"):
                    continue
                c2 = dict(ch)
                if o == "x":
                    c2.pop(M, None)
                else:
                    c2[M] = o
                nb.append(c2)
        if not nb:
            break
        tt, _ = totals_for(EV, ctx, opts, nb, cache)
        n_req += len(nb)
        k = min(range(len(nb)), key=lambda i: tt[i][tail])
        if tt[k][tail] < cur - 1e-12:
            ch, cur = nb[k], tt[k][tail]
        else:
            break
    return cur, ch, n_req


def descent_both(EV, ctx, anchors, dan, extra_starts, cache):
    """per tail: best of the descents from all-x and from each extra start; returns dict(total=(d,h), choice=(cd, ch), n_req)"""
    best = [None, None]
    n_tot = 0
    for tail in (0, 1):
        for st in [{}] + list(extra_starts[tail] if extra_starts else []):
            t, c, n = descend(EV, ctx, anchors, dan, st, tail, cache)
            n_tot += n
            if best[tail] is None or t < best[tail][0] - 1e-12:
                best[tail] = (t, c)
    return dict(total=(best[0][0], best[1][0]), choice=(dict(best[0][1]), dict(best[1][1])), n_req=n_tot)


def verdict_txt(tot, echo):
    t_ok = all(t - GATE_TOTAL >= 0 for t in tot)
    e_ok = echo["NE"] <= GATE_ECHO and echo["agnostic"] <= GATE_ECHO
    return "TOTAL %.4f / %.4f dB vs %.2f -> %s ; echo %.4f dB (swapped %.4f, agnostic %.4f) vs %.1f -> %s" % (
        tot[0], tot[1], GATE_TOTAL, "PASS" if t_ok else "FAIL", echo["NE"], echo["swapped"], echo["agnostic"], GATE_ECHO, "PASS" if e_ok else "FAIL")


def cstr(res):
    return "decay " + " ".join("x%02d:%s" % (M, res["choice"][0].get(M, "x")) for M in CLASSES) + " | held " + " ".join("x%02d:%s" % (M, res["choice"][1].get(M, "x")) for M in CLASSES)


def deltaz(z1, z0):
    """ratio z1 / z0 -> (dB, deg)"""
    return dbdeg(z1 / z0)


def anchor_row(a, ch, a1550):
    """report row of one d anchor, with the change against the 1550 anchor of the same class and psi (ratio of ratios)"""
    row = dict(ch=ch, cls=a["cls"], psi=a["psi"], P=a["P"], zone=a["zone"], nearest_x=a["nearest_x"], src=a["src"], host=a.get("host"), blas_threads=a.get("blas_threads"), wall_s=a.get("wall_s"),
               nG_actual=a.get("nG_actual"), nG_req=a.get("nG_req"), kappa_over_K0=a.get("kappa_over_K0"))
    for k, arm in ((0, "A"), (1, "B")):
        row["bac_%s" % arm] = dbdeg(a["b_ac"][k]) if "b_ac" in a else None
        row["r_%s" % arm] = dbdeg(a["r"][k])
    row["rB0"] = dbdeg(a["rR"])
    row["r0_cell"] = dbdeg(a["r0_cell"]) if "r0_cell" in a else None
    row["echo_raw"], row["echo_off"], row["echo_used"] = a["e_raw"], a["off"], a["e"]
    if a1550 is not None and ch != "1550":
        row["vs1550"] = dict(r_A=deltaz(a["r"][0], a1550["r"][0]), r_B=deltaz(a["r"][1], a1550["r"][1]), rB0=deltaz(a["rR"], a1550["rR"]), echo_used=a["e"] - a1550["e"], echo_raw=a["e_raw"] - a1550["e_raw"],
                             bac_A=deltaz(a["b_ac"][0], a1550["b_ac"][0]), bac_B=deltaz(a["b_ac"][1], a1550["b_ac"][1]))
    return row


def j2(o):
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, dict):
        return {str(k): j2(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [j2(v) for v in o]
    return o


# ====================================================================== prepare (shared by both stages) ===============================================================================
def prepare():
    S_ = {}
    j3 = json.load(io.open(V3_RESULT, encoding="utf-8"))
    v2 = json.load(io.open(V2_RESULT, encoding="utf-8"))
    S_["j3"], S_["v2"] = j3, v2
    S_["v5"] = json.load(io.open(V5_PRIMARY, encoding="utf-8"))
    S_["v5v"] = json.load(io.open(V5_VERDICT, encoding="utf-8"))
    g1550 = V1.gather("1550")
    G = {"1550": g1550}
    twins = dict(g1550["cells"])
    for ch in ("1546.003", "1554.017"):
        G[ch] = V1.gather(ch, twins)
    for ch in CHS:
        miss, inval, lens_missing = V1.inventory(G[ch])
        if miss or inval or lens_missing:
            raise RuntimeError("channel %s adopted cells not all usable: missing %s invalid %s classes %s" % (ch, miss, inval, lens_missing))
    stdout_terms = {}
    for ln in io.open(V2_STDOUT, encoding="utf-8"):
        m = re.match(r"^ATOM TERMS at channel ([0-9.]+): deficit (\S+) purity f200 (\S+)\s*$", ln)
        if m:
            stdout_terms[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    TERMS = {"1550": (V1.DFC769, V1.PRT769)}
    for ch in ("1546.003", "1554.017"):
        at = v2["channels"][ch]["atom_terms"]
        TERMS[ch] = (at["deficit"], at["purity"])
        assert stdout_terms.get(ch) == TERMS[ch], ("atom terms differ", ch)
    CTX = {}
    for ch in CHS:
        nodes = G[ch]["nodes"]
        CTX[ch] = dict(ch=ch, lam_um=V1.CHANNELS[ch]["lam_um"], DFC=TERMS[ch][0], PRT=TERMS[ch][1],
                       ND={M: {"A": nodes[M]["A"], "B": nodes[M]["B"]} for M in CLASSES}, NR={M: nodes[M]["rB0"] for M in CLASSES}, EX={M: nodes[M]["echo"] for M in CLASSES})
    ANCH, PAIR = {}, {}
    for ch in CHS:
        ANCH[ch], PAIR[ch] = {}, {}
        for M in YM:
            yp, xps = pair_paths(ch, M)
            why = validate_pair(ch, M, yp, xps, {sh: G[ch]["cells"][(M, sh)]["md5"] for sh in (0, 1, 2)})
            if why:
                raise RuntimeError("REFUSED %s y%02d: %s" % (ch, M, " | ".join(why)))
            PAIR[ch][M] = (yp, xps)
            ANCH[ch][M] = y_anchor([yp], xps, CTX[ch]["EX"][M])
    XR = {ch: xref_ch(ch, CTX[ch]["EX"]) for ch in CHS}
    NDPC = {}
    for ch in CHS:
        NDPC[ch] = {}
        for M in CLASSES:
            n_ = node_pc([G[ch]["cells"][(M, sh)]["path"] for sh in (0, 1, 2)])
            NDPC[ch][M] = {"A": n_[0], "B": n_[1]}
    S_.update(G=G, CTX=CTX, TERMS=TERMS, ANCH=ANCH, PAIR=PAIR, XR=XR, NDPC=NDPC)
    # d anchors per channel
    DAN, MISS = {}, {}
    for ch in CHS:
        DAN[ch], MISS[ch] = d_anchors_ch(ch, XR[ch])
    S_.update(DAN=DAN, MISS=MISS)
    return S_


# ====================================================================== main ====================================================================================================
def main():
    global LG
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tables", "enumerate"], default="tables")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--full1550", action="store_true", help="also enumerate all 6912 combinations at 1550 (control: argmin of the v5 primary run)")
    ap.add_argument("--cache", default=os.path.join(WD, "cache_v4.pkl"))
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M") + (("_" + a.tag) if a.tag else "")
    LG = Log(os.path.join(WD, "lensband_orient_v4_%s_stdout_%s.txt" % (a.stage, stamp)))
    T0 = time.time()
    say("lensband_orient_v4_dmeas.py  stage %s  %s  affinity %s  workers %d  numpy-only, NO solver" % (a.stage, time.strftime("%Y-%m-%d %H:%M:%S"), psutil.Process().cpu_affinity(), a.workers))
    mem0 = psutil.virtual_memory()
    say("memory at start: %.1f GB available of %.1f GB (%.0f %% used)" % (mem0.available / 1e9, mem0.total / 1e9, mem0.percent))
    controls = []

    def ctl(name, ok, text):
        controls.append((name, bool(ok), text))
        say("CONTROL %-44s %s  %s" % (name, "PASS" if ok else "FAIL", text))

    try:
        P = prepare()
    except Exception as ex_:
        say("PREPARE FAILED: %s: %s" % (type(ex_).__name__, ex_))
        LG.close("FAILED")
        return 3
    G, CTX, ANCH, XR, DAN, MISS, NDPC = P["G"], P["CTX"], P["ANCH"], P["XR"], P["DAN"], P["MISS"], P["NDPC"]
    v5 = P["v5"]
    say("inputs read: channels %s ; x e201 reference nodes (x12 x15 x18 x24 x36) and d cells per channel" % ", ".join(CHS))
    for ch in CHS:
        say("   %-8s d cells MEASURED: %s ; MISSING (no result): %s" % (ch, ", ".join("d%02d/psi%d" % (a_["cls"], psi) for psi in PSIS for a_ in DAN[ch][psi]) or "none",
                                                                         ", ".join("d%02d/psi%d" % mp for mp in MISS[ch]) or "none"))
    say("   echo offsets (EX - pmean e201, dB) x12 x15 x18 x24 x36: " + " | ".join("%s " % ch + " ".join("%+.4f" % XR[ch][M]["off"] for M in XREF_M) for ch in CHS))

    # ------------------------------------------------------------------ controls (cheap)
    # (1) 1550 d anchors == v5 primary json anchors_d, bit for bit
    bad = []
    n_cmp = 0
    for psi in PSIS:
        s5 = {x["cls"]: x for x in v5["anchors_d"][str(psi)]}
        for a_ in DAN["1550"][psi]:
            s_ = s5[a_["cls"]]
            n_cmp += 1
            for lab, x, y in (("rA", complex(*s_["r"][0]), a_["r"][0]), ("rB", complex(*s_["r"][1]), a_["r"][1]), ("rR", complex(*s_["rR"]), a_["rR"])):
                if not (bits_equal(x.real, y.real) and bits_equal(x.imag, y.imag)):
                    bad.append("d%02d psi%d %s %r vs %r" % (a_["cls"], psi, lab, x, y))
            for lab in ("e_raw", "off", "e", "P"):
                if not bits_equal(s_[lab], a_[lab]):
                    bad.append("d%02d psi%d %s %r vs %r" % (a_["cls"], psi, lab, s_[lab], a_[lab]))
    ctl("(1) 1550 d anchors == v5 primary json", not bad and n_cmp == 10, "%d anchors (r A, r B, rB0, e_raw, offset, e, P) bit-identical to lensdir45_readout_v5_primary.json anchors_d: %s %s" % (n_cmp, "yes" if not bad else "NO", "; ".join(bad[:4])))
    # (2) V-x at three channels == lensband_eval_v2 stored
    EVx = Eval2(CTX, a.workers if a.stage == "enumerate" else 0)
    EV = EVx
    for ch in CHS:
        m = P["v2"]["channels"][ch]["models"]["fixed"]
        st_t, st_e = m["full"]["res"]["rB0_complex"]["total"], m["cells_only"]["echo_aperture"]
        nd = build_nodes(CTX[ch], {}, {})
        tt = EVx.totals(ch, [nd])[0]
        ap_x = aperture_of({"x%02d" % M: CTX[ch]["EX"][M] for M in CLASSES})
        d = [abs(tt[0] - st_t[0]), abs(tt[1] - st_t[1]), abs(ap_x - st_e)]
        ctl("(2) V-x == lensband_eval_v2 @%s" % ch, max(d) <= TOL, "V-x %+.10f / %+.10f echo %.6f ; stored %+.10f / %+.10f echo %.6f ; max |diff| %.2e ; bit-identical %s" % (
            tt[0], tt[1], ap_x, st_t[0], st_t[1], st_e, max(d), bits_equal(tt[0], st_t[0]) and bits_equal(tt[1], st_t[1]) and bits_equal(ap_x, st_e)))
    # (3) sim792 control cell (1550, lam_eval_override 1) vs the stored sim781 d18 psi225 cell
    pc_ = LDB + r"\261014_sim792_ctl_d18p225_l1550p000"
    p0_ = d_result_path("1550", 18, 225)
    if os.path.isfile(os.path.join(pc_, "superramp_result.json")) and p0_:
        bc, cc = beta_pc(pc_)
        b0, c0 = beta_pc(p0_)
        qa, qb = jl(pc_)["needle_results"]["production"], jl(p0_)["needle_results"]["production"]
        dd = [abs(bc[k] / b0[k] - 1) for k in (0, 1)]
        de = abs(qa["echo_p2_db"] - qb["echo_p2_db"])
        jc = jl(pc_)
        ctl("(3) sim792 ctl d18p225 @1550 == sim781 d18p225", max(dd) < 1e-6 and de < 1e-6,
            "beta_pc ratio |ctl/stored - 1| A %.2e B %.2e, echo |diff| %.2e dB (lam_eval_override %r, lam_eval_nm %r; hosts ctl %s stored %s)" % (dd[0], dd[1], de, jc.get("lam_eval_override"), jc.get("lam_eval_nm"), host_bt(pc_), host_bt(p0_)))
    else:
        ctl("(3) sim792 ctl d18p225 @1550 == sim781 d18p225", False, "NOT FOUND: %s or the 1550 d18 psi225 cell" % pc_)
    # (4) twin checks: every band-edge d cell has the job keys / atom md5 list of its 1550 twin
    tw_bad = []
    n_tw = 0
    for ch in ("1546.003", "1554.017"):
        for psi in PSIS:
            for a_ in DAN[ch][psi]:
                twin = [b for b in DAN["1550"][psi] if b["cls"] == a_["cls"]][0]
                w = twin_diff(a_["src"], twin["src"])
                n_tw += 1
                if w:
                    tw_bad.append("%s d%02d psi%d: %s" % (ch, a_["cls"], psi, w))
    ctl("(4) band-edge d cell == its 1550 twin (keys, atoms)", not tw_bad, "%d cells compared on %s + atom md5 list: %s" % (n_tw, ",".join(PAIR_KEYS_D), "all equal" if not tw_bad else "; ".join(tw_bad[:3])))
    # (5) refusal needle: a 1554.017 d cell presented as a 1546.003 cell must be refused
    try:
        p_wrong = d_result_path("1554.017", 18, 225)
        read_d_cell_ch(p_wrong, 18, 225, "1546.003", XR["1546.003"])
        ctl("(5) refusal needle wrong channel", False, "the 1554.017 d18 psi225 cell was ACCEPTED as a 1546.003 cell")
    except AssertionError as ex_:
        ctl("(5) refusal needle wrong channel", True, "the 1554.017 d18 psi225 cell presented as a 1546.003 cell is refused (%s)" % str(ex_)[:110])
    # (6) 1550 anchor-count: the zones
    ctl("(6) 1550 has all 10 d anchors, edges 8 (d12 d15 d18 d24) x 2 psi", len(MISS["1550"]) == 0 and all(sorted(MISS[ch]) == [(36, 225), (36, 315)] for ch in ("1546.003", "1554.017")),
        "missing: " + " | ".join("%s %s" % (ch, MISS[ch]) for ch in CHS))

    # ------------------------------------------------------------------ the cell table (both stages)
    ROWS = []
    for ch in CHS:
        for psi in PSIS:
            for a_ in DAN[ch][psi]:
                a0 = [b for b in DAN["1550"][psi] if b["cls"] == a_["cls"]][0]
                ROWS.append(anchor_row(a_, ch, a0))
    say("=" * 160)
    say("D CELL TABLE (ratio = beta_pc(d) / x e201 reference at P_d of the CHANNEL; A / B arm in dB and deg; rB0; echo raw + offset = used; change vs the 1550 anchor of the same class and psi = ratio of ratios)")
    for r_ in ROWS:
        v = r_.get("vs1550")
        say("  %-8s d%02d psi%d P %.4f %-5s | r A %+.4f dB %+7.2f deg  B %+.4f dB %+7.2f deg | rB0 %+.4f dB %+6.2f deg | echo raw %.3f off %+.3f used %.3f | host %s bt %s wall %.0f s%s" % (
            r_["ch"], r_["cls"], r_["psi"], r_["P"], r_["zone"], r_["r_A"][0], r_["r_A"][1], r_["r_B"][0], r_["r_B"][1], r_["rB0"][0], r_["rB0"][1], r_["echo_raw"], r_["echo_off"], r_["echo_used"],
            r_["host"], r_["blas_threads"], r_["wall_s"] or 0.0,
            "" if v is None else "\n               vs 1550: dr A %+.4f dB %+6.2f deg  B %+.4f dB %+6.2f deg | drB0 %+.4f dB %+6.2f deg | decho used %+.3f dB | dbac A %+.4f dB %+6.2f deg B %+.4f dB %+6.2f deg" % (
                v["r_A"][0], v["r_A"][1], v["r_B"][0], v["r_B"][1], v["rB0"][0], v["rB0"][1], v["echo_used"], v["bac_A"][0], v["bac_A"][1], v["bac_B"][0], v["bac_B"][1])))
    allok = all(ok for _, ok, _ in controls)
    say("CONTROL SUMMARY (cheap controls): %d, %d PASS, %d FAIL" % (len(controls), sum(1 for _, ok, _ in controls if ok), sum(1 for _, ok, _ in controls if not ok)))
    out = dict(stamp=stamp, stage=a.stage, controls=[dict(name=n, passed=ok, text=t) for n, ok, t in controls], d_rows=j2(ROWS),
               xref=j2({ch: {str(M): dict(P=XR[ch][M]["P"], node_dB=[dbdeg(XR[ch][M]["node"][0]), dbdeg(XR[ch][M]["node"][1])], r0=dbdeg(XR[ch][M]["r0"]), e201=XR[ch][M]["e201"], off=XR[ch][M]["off"]) for M in XREF_M} for ch in CHS}))
    if a.stage == "tables":
        rname = os.path.join(WD, "lensband_orient_v4_tables_%s.json" % stamp)
        json.dump(out, io.open(rname, "x", encoding="utf-8"), indent=1)
        say("wrote %s" % rname)
        EVx.close()
        LG.close("COMPLETE" if allok else "COMPLETE_WITH_FAILED_CONTROLS")
        return 0 if allok else 1

    # ================================================================== stage enumerate ==================================================================================
    cache = Cache(a.cache)
    D1550 = DAN["1550"]
    RES = {}
    # ------------------------------------------------------------------ 1550 control block: the v5 chosen set and the descent
    ch = "1550"
    ctx = CTX[ch]
    chosen5 = v5["configs"]["x_y_d"]["used"]
    ch_dec = {int(M): u["chosen"][0] for M, u in chosen5.items() if u["chosen"][0] != "x"}
    ch_hel = {int(M): u["chosen"][1] for M, u in chosen5.items() if u["chosen"][1] != "x"}
    opts1550 = make_opts(ANCH[ch], D1550)
    t_dec = totals_for(EV, ctx, opts1550, [ch_dec], cache)[0][0]
    t_hel = totals_for(EV, ctx, opts1550, [ch_hel], cache)[0][0]
    ok_b = bits_equal(t_dec[0], v5["configs"]["x_y_d"]["total"][0]) and bits_equal(t_hel[1], v5["configs"]["x_y_d"]["total"][1])
    ctl("(7) 1550 v5 chosen sets re-evaluated == v5 TOTAL", ok_b, "decay set -> %+.10f (v5 %+.10f), held set -> %+.10f (v5 %+.10f), bit-identical %s" % (
        t_dec[0], v5["configs"]["x_y_d"]["total"][0], t_hel[1], v5["configs"]["x_y_d"]["total"][1], ok_b))
    starts5 = ({0: None}, )
    d1550 = descent_both(EV, ctx, ANCH[ch], D1550, ([ch_dec], [ch_hel]), cache)
    ok_d = bits_equal(d1550["total"][0], v5["configs"]["x_y_d"]["total"][0]) and bits_equal(d1550["total"][1], v5["configs"]["x_y_d"]["total"][1])
    ctl("(8) 1550 descent from all-x and from the v5 set", ok_d, "descent %+.10f / %+.10f vs v5 full enumeration %+.10f / %+.10f (%d evaluations requested), same chosen sets: %s" % (
        d1550["total"][0], d1550["total"][1], v5["configs"]["x_y_d"]["total"][0], v5["configs"]["x_y_d"]["total"][1], d1550["n_req"],
        d1550["choice"][0] == ch_dec and d1550["choice"][1] == ch_hel))
    echo1550 = echo_summary(ctx["EX"], ANCH[ch], D1550)
    ctl("(9) 1550 echo == v5", abs(echo1550["NE"] - V5_ECHO) < 1e-9, "echo NE %.10f (v5 %.10f), swapped %.10f, agnostic %.10f" % (echo1550["NE"], V5_ECHO, echo1550["swapped"], echo1550["agnostic"]))
    RES["1550"] = dict(measured=dict(total=(t_dec[0], t_hel[1]), choice=(ch_dec, ch_hel), echo=echo1550, n_combinations=6912, note="v5 chosen sets re-evaluated; full enumeration only with --full1550"))
    if a.full1550:
        t = time.time()
        S1550 = search_c(EV, ctx, ANCH[ch], D1550, cache)
        pk = pick(S1550)
        okf = bits_equal(pk["total"][0], v5["configs"]["x_y_d"]["total"][0]) and bits_equal(pk["total"][1], v5["configs"]["x_y_d"]["total"][1]) and len(S1550["combos"]) == 6912
        ctl("(10) 1550 full enumeration == v5 primary", okf, "%d combinations in %.0f s: %+.10f / %+.10f vs v5 %+.10f / %+.10f ; chosen equal v5: %s" % (
            len(S1550["combos"]), time.time() - t, pk["total"][0], pk["total"][1], v5["configs"]["x_y_d"]["total"][0], v5["configs"]["x_y_d"]["total"][1],
            pk["choice"][0] == ch_dec and pk["choice"][1] == ch_hel))
        RES["1550"]["full"] = dict(total=pk["total"], choice=pk["choice"])
    # [E] corrected reference at 1550 (control of alt_shape): descent
    D1550_alt = alt_shape(D1550, NDPC["1550"], CTX["1550"]["NR"])
    e1550 = descent_both(EV, ctx, ANCH[ch], D1550_alt, ([ch_dec], [ch_hel]), cache)
    ctl("(11) 1550 corrected reference == v5 verify [E]", abs(e1550["total"][0] - V5_CORR[0]) < 5e-10 and abs(e1550["total"][1] - V5_CORR[1]) < 5e-10,
        "descent %+.10f / %+.10f vs v5 verify %+.10f / %+.10f ; chosen decay %s held %s" % (e1550["total"][0], e1550["total"][1], V5_CORR[0], V5_CORR[1],
                                                                                          json.dumps({M: o for M, o in sorted(e1550["choice"][0].items())}), json.dumps({M: o for M, o in sorted(e1550["choice"][1].items())})))
    RES["1550"]["corrected"] = dict(total=e1550["total"], choice=e1550["choice"], echo=echo1550)

    # ------------------------------------------------------------------ band-edge channels
    for ch in ("1546.003", "1554.017"):
        ctx = CTX[ch]
        RES[ch] = {}
        EXc = ctx["EX"]
        # V-x / V-xy
        vx_tot = totals_for(EV, ctx, {}, [{}], cache)[0][0]
        ap_x = aperture_of({"x%02d" % M: EXc[M] for M in CLASSES})
        RES[ch]["V-x"] = dict(total=vx_tot, choice=({}, {}), echo=dict(NE=ap_x, swapped=ap_x, agnostic=ap_x))
        Sxy = search_c(EV, ctx, ANCH[ch], {225: [], 315: []}, cache)
        pxy = pick(Sxy)
        RES[ch]["V-xy"] = dict(total=pxy["total"], choice=pxy["choice"], echo=echo_summary(EXc, ANCH[ch], {225: [], 315: []}))
        for mode in ("pending", "transfer"):
            dan_u = with_d36(DAN[ch], mode, D1550, XR[ch])
            t = time.time()
            Sx = search_c(EV, ctx, ANCH[ch], dan_u, cache)
            pk = pick(Sx)
            echo = echo_summary(EXc, ANCH[ch], dan_u)
            say("== %s d36 %s (fixed reference): %d combinations (%d newly evaluated) in %.0f s" % (ch, mode, len(Sx["combos"]), Sx["n_new"], time.time() - t))
            say("   " + verdict_txt(pk["total"], echo))
            say("   chosen " + cstr(pk))
            RES[ch]["M4+" + mode] = dict(total=pk["total"], choice=pk["choice"], echo=echo, n_combinations=len(Sx["combos"]), opts_classes={str(M): sorted(o) for M, o in Sx["opts"].items()})
            # descent method check on the uncorrected variant
            dm = descent_both(EV, ctx, ANCH[ch], dan_u, ([pk["choice"][0]], [pk["choice"][1]]), cache)
            dm0 = descent_both(EV, ctx, ANCH[ch], dan_u, None, cache)
            okm = bits_equal(dm0["total"][0], pk["total"][0]) and bits_equal(dm0["total"][1], pk["total"][1])
            ctl("(12) %s %s descent from all-x == full enumeration" % (ch, mode), okm, "descent from all-x %+.10f / %+.10f vs enumeration %+.10f / %+.10f (%d evaluations requested)" % (
                dm0["total"][0], dm0["total"][1], pk["total"][0], pk["total"][1], dm0["n_req"]))
            # corrected reference (estimate by descent from all-x and from the uncorrected argmin)
            dan_c = alt_shape(dan_u, NDPC[ch], ctx["NR"])
            dc = descent_both(EV, ctx, ANCH[ch], dan_c, ([pk["choice"][0]], [pk["choice"][1]]), cache)
            say("   corrected reference (descent): TOTAL %.4f / %.4f dB (fixed %.4f / %.4f ; change %+.4f / %+.4f) chosen decay %s held %s" % (
                dc["total"][0], dc["total"][1], pk["total"][0], pk["total"][1], dc["total"][0] - pk["total"][0], dc["total"][1] - pk["total"][1],
                json.dumps({M: o for M, o in sorted(dc["choice"][0].items())}), json.dumps({M: o for M, o in sorted(dc["choice"][1].items())})))
            RES[ch]["M4+" + mode + "|corrected"] = dict(total=dc["total"], choice=dc["choice"], echo=echo, estimate=True)
            RES[ch]["M4+" + mode]["opts_ratio_dB_deg"] = j2({str(M): {nm: [dbdeg(z) for z in v] for nm, v in Sx["opts"][M].items()} for M in sorted(Sx["opts"])})
    # ------------------------------------------------------------------ summary
    say("=" * 160)
    say("SUMMARY (TOTAL decay / held in dB, gate %.2f ; echo NE / swapped, gate %.1f ; PROVISIONAL = d36 not measured at the edges)" % (GATE_TOTAL, GATE_ECHO))
    v3tab = {"1546.003": (-0.6497, -0.6739, -20.910), "1550": (-0.6008, -0.6217, -20.660), "1554.017": (-0.6191, -0.6390, -20.245)}
    for ch in CHS:
        for k, v in RES[ch].items():
            e = v.get("echo")
            say("   %-9s %-26s %+.4f / %+.4f  margin %+.4f / %+.4f  %s" % (ch, k, v["total"][0], v["total"][1], v["total"][0] - GATE_TOTAL, v["total"][1] - GATE_TOTAL,
                                                                           ("echo %.3f / %.3f (margin %+.3f)" % (e["NE"], e["swapped"], GATE_ECHO - e["NE"])) if e else ""))
        say("   %-9s %-26s %+.4f / %+.4f  echo %.3f   (lensband_orient_v3 stdout, all 6 v3 anchors, 1550 d12/d15 absent)" % (ch, "v3 V-xyd (quoted)", v3tab[ch][0], v3tab[ch][1], v3tab[ch][2]))
    say("evaluations: %d betaL evaluations in the pool (+ cache hits); elapsed %.0f s" % (EV.n, time.time() - T0))
    allok = all(ok for _, ok, _ in controls)
    say("CONTROL SUMMARY: %d, %d PASS, %d FAIL" % (len(controls), sum(1 for _, ok, _ in controls if ok), sum(1 for _, ok, _ in controls if not ok)))
    out.update(controls=[dict(name=n, passed=ok, text=t) for n, ok, t in controls], results=j2(RES), all_controls_pass=allok, elapsed_s=time.time() - T0)
    rname = os.path.join(WD, "lensband_orient_v4_enumerate_%s.json" % stamp)
    json.dump(out, io.open(rname, "x", encoding="utf-8"), indent=1)
    say("wrote %s" % rname)
    EV.close()
    LG.close("COMPLETE" if allok else "COMPLETE_WITH_FAILED_CONTROLS")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
