# -*- coding: utf-8 -*-
r"""lensband_eval.py -- [TM] x-only lens total + aperture echo of the FINAL Li lens at the band-edge channels (1546.003 / 1550 / 1554.017 nm).

WHAT IT IS.  A wavelength-aware, self-contained copy of the arithmetic of lens_total_li5.py (-> lens_total_li.py -> ladder646.py head ->
review_net_f.py head); no module of that chain is imported or exec'd in normal runs (--control cross-checks against the originals).
Every formula is copied verbatim; the only change is that three constants that the originals hard-code as LAM = 1.55 um are parameters:
    lam_design  period map P(rho) = lam*sqrt(rho^2+f^2)/rho  and design phase ph = -K0*(sqrt(rho^2+f^2)-f)          [ladder646.ref_field]
    lam_prop    free-space propagation kz^2 = K0^2-kx^2-ky^2 (focal()) and therefore the capture c0 of the ideal reference [review_net_f / ladder646.focal]
    lam_c       obliquity factor c = sqrt(1-(lam/(P*LAMG))^2) of the ramp order in conv()                                    [lens_total_li.conv]
Lens models (--models):
    fixed         (DEFAULT, "physical lens fixed at its 1550 design"): lam_design = 1.55, lam_prop = lam_c = channel; ideal reference = the SAME
                  1550-designed ideal lens (beta = 1) propagated at the channel, i.e. its own chromatic shift is not charged to the lens.
    fixed_redref  numerator as `fixed`; ideal reference = a lens RE-DESIGNED for the channel (ph with the channel K0), so the chromatic defocus of the
                  fixed lens IS charged.
    redesign      everything at the channel (P map, ph, kz, reference, obliquity): the 1550 evaluation applied to the channel cells.
    cells_only    everything stays at 1550 (lam_design = lam_prop = lam_c = 1.55): ONLY the cell data (beta nodes, echo) change.
    fixed_noobl   as `fixed` but the obliquity factor c of conv() keeps the 1550 value (isolates that one term).
At 1550 nm all five models are arithmetically identical (checked in --control).

CELLS.  1550: stored cells of the final lens (sim769 x06..x36, x12_sh0 = sim766 iso; x48 = sim771, echo/node information only).
          channel: Result\261012_scdc_lensband\261012_sim785_x{MM}_sh{k}_l{1546p003|1554p017}\superramp_result.json.
A class node needs all 3 registrations; a missing or invalid cell makes its class unavailable and the LENS TOTAL / APERTURE REFUSED -- nothing is
ever substituted or estimated.  Each cell is validated (Li flags, sc_shift, job lams_nm, lam_eval keys, atom md5 list equal to the 1550 twin).
ATOM-LEVEL TERMS deficit_p' / purity_p' f200 (-0.061521 / -0.018901) are the 1550 values of a1_def_pur_p769_result.json (as in lens_total_li5.py);
they are NOT re-evaluated at the channel (the sim783 atom data would be needed) -- the printed channel totals are 'cells + 1550 atom terms'.

RUN (one command, no solver):   python -I -B <WORK_ROOT>\wlband_261012\lensband_eval\lensband_eval.py
   options: --channels 1546.003,1550,1554.017  --models fixed|all|<list>  --control (control + needle + original-code cross-check only)  --no-control
Writes lensband_eval_result.json and lensband_eval_stdout.txt (--control: lensband_eval_control_stdout.txt) next to this file.  Exit code 0 = control passed (or skipped) and run finished."""
import os
import sys

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import psutil                                                                    # noqa: E402
import argparse                                                                  # noqa: E402
import cmath                                                                     # noqa: E402
import hashlib                                                                   # noqa: E402
import io                                                                        # noqa: E402
import json                                                                      # noqa: E402
import math                                                                      # noqa: E402
import re                                                                        # noqa: E402
import time                                                                      # noqa: E402
import numpy as np                                                               # noqa: E402

R = r"<DATA_ROOT>\Result"
WD = r"<WORK_ROOT>\finished_sims_261016\A_results\mid_channels"
FINAL = r"<WORK_ROOT>\loop_261008\lens_li5\lens_total_li5_final.json"
T769F = r"<WORK_ROOT>\loop_261008\improve_261009\tm_lens_total_a1\a1_def_pur_p769_result.json"
CELLS769 = r"<WORK_ROOT>\loop_261008\lens_li5\sim769_cells.json"
STORED_KEY = "P769 approved (Li terms)"
CLASSES = (6, 9, 12, 15, 18, 24, 36)                                             # lens_total_li.CLASSES
CL = ["x%02d" % M for M in CLASSES]
GATE_TOTAL = -0.65                                                               # lens_total_li.GATE_TOTAL
GATE_ECHO = -20.5                                                                # lens_total_li.GATE_ECHO
LAM_D = 1.55                                                                     # review_net_f.LAM (design / 1550 nm), um
LAMG = 1.31539                                                                   # review_net_f.LAMG (Lambda of the ramp unit), um
CHANNELS = {"1546.003": dict(tag="l1546p003", lam_nm=1546.003, lam_um=1546.003 / 1000.0),
            "1550": dict(tag=None, lam_nm=1550.0, lam_um=LAM_D),                 # literal 1.55, bit-identical to the original LAM
            "1554.017": dict(tag="l1554p017", lam_nm=1554.017, lam_um=1554.017 / 1000.0),
            "1551.604": dict(tag="l1551p604", lam_nm=1551.604, lam_um=1551.604 / 1000.0),      # [A-copy] sim793 interior channel
            "1552.408": dict(tag="l1552p408", lam_nm=1552.408, lam_um=1552.408 / 1000.0)}      # [A-copy] sim793 interior channel
MODELS = {  # which wavelength each constant takes: D = design 1.55, C = channel
    "fixed": dict(design="D", ref="D", prop="C", c="C"),
    "fixed_redref": dict(design="D", ref="C", prop="C", c="C"),
    "redesign": dict(design="C", ref="C", prop="C", c="C"),
    "cells_only": dict(design="D", ref="D", prop="D", c="D"),
    "fixed_noobl": dict(design="D", ref="D", prop="C", c="D"),                   # as `fixed` but the obliquity factor of conv() stays at 1550 (isolates that term)
}
PAIR_KEYS = ("nG_req", "nG_ref", "nG_eval_ref", "ramp_s", "sc_shift", "pol", "ramp_axis_requested", "ref_mode")   # lens_total_li.PAIR_KEYS minus atoms_requested (paths differ for archived inputs; md5 compared instead)
T0 = time.time()
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ======================================================================= geometry (review_net_f.py 18-38, verbatim) ===========================================================
W = 100.0                                                                        # Gaussian 1/e field radius, um (review_net_f.py:20)
APER = 150.0                                                                     # quadrant 150 x 150 um (:21)
XC = W / np.sqrt(2.0 * np.pi)                                                    # (:22) quadrant centroid / PD centre
dx = 0.5                                                                         # (:24)
n = int(round(APER / dx))                                                        # (:25)
_x = (np.arange(n) + 0.5) * dx                                                   # (:26)
X, Y = np.meshgrid(_x, _x)                                                       # (:27)
A = np.exp(-(X ** 2 + Y ** 2) / W ** 2)                                          # (:28)
pin = float(np.sum(A ** 2) * dx * dx)                                            # (:29)
rho = np.hypot(X - XC, Y - XC)                                                   # (:30)
npd = 4 * n                                                                      # (:31)
_k = 2.0 * np.pi * np.fft.fftfreq(npd, d=dx)                                     # (:32)
KX, KY = np.meshgrid(_k, _k)                                                     # (:33)
DT = 0.1                                                                         # (:36)
AX = DT * (np.arange(80) + 0.5) - 4.0                                            # (:37) 80 x 0.1 um = 8 um PD box
Mx = np.exp(1j * np.outer(XC + AX - dx / 2.0, _k))                               # (:38)
_KZ2, _H, _DES = {}, {}, {}


def kz2_of(lam):
    """review_net_f.py:23,34 with LAM -> lam"""
    if lam not in _KZ2:
        K0 = 2.0 * np.pi / lam
        _KZ2[lam] = K0 ** 2 - KX ** 2 - KY ** 2
    return _KZ2[lam]


def focal(E0, f, lam):
    """ladder646.py:87-90 with the propagation wavelength lam"""
    if (lam, f) not in _H:
        kz2 = kz2_of(lam)
        _H[(lam, f)] = np.where(kz2 >= 0, np.exp(1j * np.sqrt(np.clip(kz2, 0, None)) * f), 0.0)
    H = _H[(lam, f)]
    Ep = np.zeros((npd, npd), dtype=complex)
    Ep[:n, :n] = E0
    return (Mx @ (np.fft.fft2(Ep) * H) @ Mx.T) / (npd * npd)


def design(f, lam):
    """ladder646.py:98-99 with LAM -> lam: local period map P (um) and hyperbolic design phase"""
    if (f, lam) not in _DES:
        K0 = 2.0 * np.pi / lam
        P = lam * np.sqrt(rho ** 2 + f ** 2) / np.maximum(rho, 1e-9)
        ph = -K0 * (np.sqrt(rho ** 2 + f ** 2) - f)
        _DES[(f, lam)] = (P, ph)
    return _DES[(f, lam)]


def model(P, Pm, bm, tail):
    """ladder646.py:74-84 verbatim (LAMG is the physical ramp unit)"""
    db_ = 20 * np.log10(np.abs(bm))
    ar = np.unwrap(np.angle(bm))
    x = 1.0 / Pm[::-1]
    D = np.interp(1.0 / P, x, db_[::-1])
    F = np.interp(1.0 / P, x, ar[::-1])
    last = Pm[-1]
    if tail == "decay":
        D = np.where(P > last, db_[-1] * last / P, D)
        F = np.where(P > last, ar[-1] * last / P, F)
    else:
        D = np.where(P > last, db_[-1], D)
        F = np.where(P > last, ar[-1], F)
    b = 10 ** (D / 20.0) * np.exp(1j * F)
    return np.where(P < 2 * LAMG, 0.0, b)


class Geo:
    """one lens model at one channel: wavelengths (um) of the design, the ideal reference, the propagation and the obliquity factor"""

    def __init__(self, mname, lam_ch):
        m = MODELS[mname]
        pick = lambda c: LAM_D if c == "D" else lam_ch                           # noqa: E731
        self.name, self.lam_design, self.lam_ref, self.lam_prop, self.lam_c = mname, pick(m["design"]), pick(m["ref"]), pick(m["prop"]), pick(m["c"])
        self._c0 = {}

    def c0(self, f):
        if f not in self._c0:
            ph_ref = design(f, self.lam_ref)[1]
            E0 = focal(A * np.exp(1j * ph_ref), f, self.lam_prop)
            self._c0[f] = float(np.sum(np.abs(E0) ** 2) * DT * DT) / pin
        return self._c0[f]

    def evaluate(self, pts, f, tail):
        """ladder646.py:106-113 -> (ov, c0, L_dB)"""
        P, ph = design(f, self.lam_design)
        c0 = self.c0(f)
        Pm = np.array([p[0] for p in pts]) * LAMG
        EA = focal(A * model(P, Pm, np.array([p[1] for p in pts]), tail) * np.exp(1j * ph), f, self.lam_prop)
        EB = focal(A * model(P, Pm, np.array([p[2] for p in pts]), tail) * np.exp(1j * ph), f, self.lam_prop)
        ov = abs(np.sum(np.conj(EA) * EB)) * DT * DT / pin
        return float(ov), float(c0), float(10 * np.log10(ov / c0))

    def conv(self, pts):
        """lens_total_li.py:83-91 with LAM -> lam_c"""
        out = []
        for (P, bA, bB) in pts:
            Mat = int(round(P * 3))
            c = math.sqrt(1.0 - (self.lam_c / (P * LAMG)) ** 2)
            ph = cmath.exp(-1j * math.pi * (96 - 1) / (Mat * 96))
            out.append((P, bA * math.sqrt(c) * ph, bB * math.sqrt(c) * ph, 3))
        return out

    def betaL(self, nd, f=200.0):
        """lens_total_li.py:94-97"""
        q = self.conv([(M / 3.0, nd[M]["A"], nd[M]["B"]) for M in sorted(nd)])
        return self.evaluate(q, f, "decay")[2], self.evaluate(q, f, "held")[2]


# ======================================================================= zone / echo (lens_total_li.py:101-121) =====================================================================
ZONE_W = {"2-3L": 2.666, "3-4L": 7.462, "4-6L": 25.6, "6-12L": 47.9}
ZONE_W[">12L"] = 100.0 - sum(ZONE_W.values())
ZMAP = {"2-3L": ["x06"], "3-4L": ["x09"], "4-6L": ["x12", "x15"], "6-12L": ["x18", "x24", "x36"]}


def db(z):
    return 20 * math.log10(abs(z))


def pmean(v):
    return 10 * math.log10(sum(10 ** (x / 10.0) for x in v) / len(v))


def aperture(md, e12=GATE_ECHO):
    """lens_total_li.aperture; DIFFERENCE: returns None if ANY class of a zone is missing (the original would silently average only the present ones).
    e12 = echo of the >12 Lambda zone (default the gate value -20.5 dB as in the final lens; pass the measured x48 power mean for the x48 variant)."""
    amp = 0.0
    for z, w in ZONE_W.items():
        if z == ">12L":
            e = e12
        else:
            vals = [md.get(c) for c in ZMAP[z]]
            if any(v is None for v in vals):
                return None
            e = pmean(vals)
        amp += (w / 100.0) * math.sqrt(10 ** (e / 10.0))
    return 10 * math.log10(amp * amp)


# ======================================================================= cells ======================================================================================================
_MD5 = {}


ARCH = r"<DATA_ROOT>\Result"                                                 # archive of finished result folders, read-only


def m8(p):
    """first 8 hex digits of the md5 of an atom density file; an input that has been archived since (moved out of Result\\) is read from ARCH (same
    relative path), as build_sim785_786.resolve() did when it byte-copied archived inputs"""
    if p not in _MD5:
        q = p
        if not os.path.isfile(q) and p.startswith(R + "\\"):
            alt = os.path.join(ARCH, p[len(R) + 1:])
            if os.path.isfile(alt):
                q = alt
        _MD5[p] = hashlib.md5(open(q, "rb").read()).hexdigest()[:8]
    return _MD5[p]


def jload(p):
    return json.load(io.open(os.path.join(p, "superramp_result.json"), encoding="utf-8"))


def stored_dir(M, sh):
    if M == 48:
        return R + r"\261009_scdc_tmzone_li6\261009_sim771_lensli6_x48_sh%d" % sh
    if (M, sh) == (12, 0):
        return R + r"\261009_scdc_tmzone_iso\261009_sim766_iso_x12_sh0_212_257acc"
    return R + r"\261009_scdc_tmzone_li5\261009_sim769_lensli5_x%02d_sh%d" % (M, sh)


DEFAULT_BAND_ROOT = R + r"\261012_scdc_lensband"
BAND_ROOT = DEFAULT_BAND_ROOT                                          # folder of the sim785 band-edge cells (--band-root overrides, used by the self-test)


def band_dir(M, sh, tag):
    if tag in ("l1551p604", "l1552p408"):                                  # [A-copy] sim793 interior channels live in their own result folder
        return R + r"\261013_scdc_lensband_mid\261013_sim793_x%02d_sh%d_%s" % (M, sh, tag)
    return BAND_ROOT + r"\261012_sim785_x%02d_sh%d_%s" % (M, sh, tag)


_C769 = json.load(io.open(CELLS769, encoding="utf-8"))
EXP = {(c["M"], c["sh"]): c["after"] for c in _C769["cells"]}
EXP[(12, 0)] = _C769["reuse"][0]["slots"]


def job_lams(p):
    try:
        for fn in os.listdir(p):
            if fn.startswith("job_") and fn.endswith(".txt"):
                m = re.search(r"^lams_nm\s*=\s*([0-9.]+)", io.open(os.path.join(p, fn), encoding="utf-8").read(), re.M)
                if m:
                    return float(m.group(1))
    except OSError:
        pass
    return None


def load_cell(ch, M, sh, twin=None, path=None):
    """-> dict(status OK|MISSING|INVALID, why, ...).  Raw per-registration terms follow lens_total_li.read_class (:125-143) exactly."""
    p = path or (stored_dir(M, sh) if ch == "1550" else band_dir(M, sh, CHANNELS[ch]["tag"]))
    c = dict(M=M, sh=sh, ch=ch, path=p, status="MISSING", why="no superramp_result.json")
    if not os.path.isfile(os.path.join(p, "superramp_result.json")):
        return c
    why = []
    try:
        j = jload(p)
        r = j["needle_results"]["production"]
        if (j.get("rcwa_rule"), j.get("rcwa_ff")) != ("liy", 1):
            why.append("not a Li cell (rcwa_rule %r rcwa_ff %r)" % (j.get("rcwa_rule"), j.get("rcwa_ff")))
        if int(r["sc_shift"]) != sh or int(j["sc_shift"]) != sh:
            why.append("sc_shift %r/%r != registration %d" % (r["sc_shift"], j["sc_shift"], sh))
        jl = job_lams(p)
        nom = CHANNELS[ch]["lam_nm"]
        if ch == "1550":
            if j.get("lam_eval_override") is not None or j.get("lam_eval_nm") is not None:
                why.append("stored 1550 cell carries lam_eval keys")
            if jl is not None and abs(jl - 1550.0) > 1e-9:
                why.append("job lams_nm %r != 1550" % jl)
        else:
            if j.get("lam_eval_override") != 1:
                why.append("lam_eval_override %r != 1" % j.get("lam_eval_override"))
            if j.get("lam_eval_nm") is None or abs(float(j["lam_eval_nm"]) - nom) > 1e-9:
                why.append("lam_eval_nm %r != %s" % (j.get("lam_eval_nm"), nom))
            if j.get("atom_lams_nm") is None or abs(float(j["atom_lams_nm"]) - 1550.0) > 1e-9:
                why.append("atom_lams_nm %r != 1550" % j.get("atom_lams_nm"))
            if jl is not None and abs(jl - nom) > 1e-9:
                why.append("job lams_nm %r != %s" % (jl, nom))
        md5 = [m8(a) for a in r["atom_paths"]]
        if ch == "1550" and M != 48:
            if md5 != EXP[(M, sh)]:
                why.append("per-slot atom md5 list != expected P769 list")
        if twin is not None and twin["status"] == "OK":
            if md5 != twin["md5"]:
                why.append("atom md5 list != the 1550 twin")
            bad = [k for k in PAIR_KEYS if j.get(k) != twin["j"].get(k)]
            if bad:
                why.append("differs from the 1550 twin in %s" % bad)
        L = r["lpa"]["arm%s_s%+d" % ("A", int(r["ramp_s"]))]
        LB = r["lpa"]["arm%s_s%+d" % ("B", int(r["ramp_s"]))]
        L0 = r["lpa"]["rB0"]
        c.update(j=j, md5=md5, lam_job=jl,
                 ampA=complex(*r["amp_tA_target"]), ampB=complex(*r["amp_tB_target"]),
                 predA=complex(L["a_pred_re"], L["a_pred_im"]), predB=complex(LB["a_pred_re"], LB["a_pred_im"]),
                 amprB=complex(*r["amp_rB_zero"]), predrB=complex(L0["a_pred_re"], L0["a_pred_im"]), echo=r["echo_p2_db"],
                 nG_req=j.get("nG_req"), nG_ref=j.get("nG_ref"))
    except Exception as e:                                                       # noqa: BLE001 -- any unreadable / incomplete json is INVALID, never used
        why.append("unreadable: %s %s" % (type(e).__name__, e))
    c["status"] = "OK" if not why else "INVALID"
    c["why"] = "; ".join(why)
    return c


def class_node(cells):
    """lens_total_li.read_class (:125-143) on already loaded raw terms; cells {sh: cell} with status OK -> (acc{A,B}, rB0 node, [echo per registration])"""
    acc = {"A": 0j, "B": 0j}
    rs, ech = [], []
    lp = None
    for sh in (0, 1, 2):
        c = cells[sh]
        for arm, sg in (("A", -1), ("B", +1)):
            acc[arm] += c["amp" + arm] / c["pred" + arm] * cmath.exp(sg * 2j * math.pi * sh / 3.0) / 3
        p = c["predrB"]
        if lp is not None and not abs(p - lp) < 1e-12:
            raise ValueError("rB0 LPA reference differs between registrations (%r vs %r)" % (p, lp))
        lp = p
        rs.append(c["amprB"] / p)
        ech.append(c["echo"])
    return acc, sum(rs) / 3, ech


def gather(ch, twins=None):
    """load every cell of channel ch for classes x06..x36 and x48; classes with 3 OK registrations get a node"""
    cells = {}
    for M in CLASSES + (48,):
        for sh in (0, 1, 2):
            cells[(M, sh)] = load_cell(ch, M, sh, None if twins is None else twins[(M, sh)])
    return dict(ch=ch, cells=cells, nodes=build_nodes(cells))


def build_nodes(cells):
    """class nodes for every class (x06..x36, x48) whose 3 registrations are all OK; a class never gets a node from fewer than 3 cells"""
    nodes = {}
    for M in CLASSES + (48,):
        if all(cells[(M, sh)]["status"] == "OK" for sh in (0, 1, 2)):
            try:
                a, r_, e = class_node({sh: cells[(M, sh)] for sh in (0, 1, 2)})
                nodes[M] = dict(A=a["A"], B=a["B"], rB0=r_, echo_list=e, echo=pmean(e))
            except ValueError as ex:
                for sh in (0, 1, 2):
                    cells[(M, sh)]["status"], cells[(M, sh)]["why"] = "INVALID", str(ex)
    return nodes


def inventory(g):
    miss, inval = [], []
    for (M, sh), c in sorted(g["cells"].items()):
        if c["status"] == "MISSING":
            miss.append("x%02d_sh%d" % (M, sh))
        elif c["status"] == "INVALID":
            inval.append("x%02d_sh%d (%s)" % (M, sh, c["why"]))
    lens_missing = [x for x in CLASSES if x not in g["nodes"]]
    return miss, inval, lens_missing


# ======================================================================= lens evaluation (lens_total_li.evaluate_cells :177-191) ===================================================
def lens_from_nodes(nodes, geo, DFC, PRT):
    """nodes {M: dict(A,B,rB0,echo)} for ALL of CLASSES -> same dict as lens_total_li.evaluate_cells"""
    nd = {M: {"A": nodes[M]["A"], "B": nodes[M]["B"]} for M in CLASSES}
    nr = {M: nodes[M]["rB0"] for M in CLASSES}
    echo = {"x%02d" % M: nodes[M]["echo"] for M in CLASSES}
    res = {}
    for var, mk in (("no_rB0", lambda M: nd[M]["B"]), ("rB0_complex", lambda M: nd[M]["B"] * nr[M]), ("rB0_abs", lambda M: nd[M]["B"] * abs(nr[M]))):
        b = geo.betaL({M: {"A": nd[M]["A"], "B": mk(M)} for M in CLASSES})
        tot = (b[0] + DFC + PRT, b[1] + DFC + PRT)
        res[var] = dict(betaL=b, total=tot, margin=(tot[0] - GATE_TOTAL, tot[1] - GATE_TOTAL))
    ap = aperture({c: echo[c] for c in CL})
    return dict(res=res, echo_class=echo, echo_aperture=ap, deficit=DFC, purity=PRT,
                node_dB={"x%02d" % M: dict(A=db(nd[M]["A"]), B=db(nd[M]["B"]), rB0=db(nr[M]), rB0_deg=math.degrees(cmath.phase(nr[M]))) for M in CLASSES})


_T = json.load(io.open(T769F, encoding="utf-8"))
DFC769, PRT769 = _T["deficit"], _T["pur_f200"]                                   # lens_total_li5.py:23,64 (P769 Li terms, 1550)
STORED = json.load(io.open(FINAL, encoding="utf-8"))["results"][STORED_KEY]


def flat(d, pre=""):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += flat(v, pre + str(k) + ".")
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out += flat(v, pre + "[%d]." % i)
    else:
        out.append((pre[:-1], d))
    return out


def compare_stored(ev, label=""):
    a, b = dict(flat(STORED)), dict(flat(ev))
    keys = [k for k, _ in flat(STORED)]
    rows = []
    for k in keys:
        rows.append((k, a[k], b.get(k), (b.get(k) == a[k])))
    return rows


def run_control(g1550):
    say("=" * 150)
    say("CONTROL 1: the 1550 cells reproduce lens_total_li5_final.json ['%s'] (x-only) -- every stored number, compared with ==, repr to the last digit" % STORED_KEY)
    say("   terms: deficit %r  purity %r  (a1_def_pur_p769_result.json) ; stored: %r / %r" % (DFC769, PRT769, STORED["deficit"], STORED["purity"]))
    geo = Geo("fixed", CHANNELS["1550"]["lam_um"])
    if any(M not in g1550["nodes"] for M in CLASSES):
        say("   CONTROL FAIL: 1550 cells incomplete")
        return False, None
    ev = lens_from_nodes(g1550["nodes"], geo, DFC769, PRT769)
    rows = compare_stored(ev)
    bad = [r for r in rows if not r[3]]
    worst = max((abs(float(r[1]) - float(r[2])) for r in rows), default=0.0)
    for k, s, v, ok in rows:
        say("   %-34s stored %-24r recomputed %-24r %s" % (k, s, v, "EXACT" if ok else "DIFF %.3e" % abs(float(s) - float(v))))
    say("   -> %d stored numbers compared, %d not exactly equal, max |diff| = %.3e : %s" % (len(rows), len(bad), worst, "PASS (bit-identical)" if not bad else "FAIL"))
    ok1 = not bad
    # models collapse at 1550
    say("CONTROL 2: the five lens models are arithmetically identical at 1550 nm (betaL decay/held of rB0_complex, compared with ==)")
    ref = ev["res"]["rB0_complex"]["betaL"]
    ok2 = True
    for mn in MODELS:
        gm = Geo(mn, CHANNELS["1550"]["lam_um"])
        nd = {M: {"A": g1550["nodes"][M]["A"], "B": g1550["nodes"][M]["B"] * g1550["nodes"][M]["rB0"]} for M in CLASSES}
        b = gm.betaL(nd)
        same = (b[0] == ref[0] and b[1] == ref[1])
        ok2 = ok2 and same
        say("   %-13s betaL decay %r held %r  %s" % (mn, b[0], b[1], "identical" if same else "DIFFERENT"))
    # needle: perturb one 1550 node by 0.01 dB
    say("NEEDLE: one 1550 node perturbed by +0.01 dB must move the total and must be caught by the comparator")
    ok3 = True
    f001 = 10 ** (0.01 / 20.0)
    for cls, arm in ((24, "A"), (15, "B"), (6, "A")):
        nodes2 = {M: dict(v) for M, v in g1550["nodes"].items()}
        nodes2[cls][arm] = nodes2[cls][arm] * f001
        ev2 = lens_from_nodes(nodes2, geo, DFC769, PRT769)
        d0 = ev2["res"]["rB0_complex"]["total"][0] - ev["res"]["rB0_complex"]["total"][0]
        d1 = ev2["res"]["rB0_complex"]["total"][1] - ev["res"]["rB0_complex"]["total"][1]
        nbad = len([r for r in compare_stored(ev2) if not r[3]])
        ok = (d0 != 0.0 and d1 != 0.0 and nbad > 0)
        ok3 = ok3 and ok
        say("   x%02d node %s +0.01 dB: total(rB0 complex) decay %+.6f dB held %+.6f dB ; comparator flags %d stored numbers as DIFFERENT  -> %s" % (
            cls, arm, d0, d1, nbad, "needle detected" if ok else "NEEDLE NOT DETECTED"))
    return ok1 and ok2 and ok3, ev


def run_control_original(g1550):
    """cross-check against the original code path (imports lens_total_li.py unmodified; execs ladder646 head): read_class and betaL on the same 1550 cells"""
    say("CONTROL 4: this script vs the ORIGINAL lens_total_li.read_class / betaL on the same 1550 cells (original modules imported unmodified)")
    sys.path.insert(0, r"<WORK_ROOT>\loop_261008\lens_li")
    import lens_total_li as L                                                    # noqa: E402
    worst, ok = 0.0, True
    nd = {}
    for M in CLASSES:
        a, r_, e = L.read_class({sh: stored_dir(M, sh) for sh in (0, 1, 2)})
        nn = g1550["nodes"][M]
        d = max(abs(a["A"] - nn["A"]), abs(a["B"] - nn["B"]), abs(r_ - nn["rB0"]), max(abs(x - y) for x, y in zip(e, nn["echo_list"])))
        worst = max(worst, d)
        nd[M] = dict(A=a["A"], B=a["B"] * r_)
    b_orig = L.betaL(nd)
    geo = Geo("fixed", CHANNELS["1550"]["lam_um"])
    b_mine = geo.betaL({M: {"A": g1550["nodes"][M]["A"], "B": g1550["nodes"][M]["B"] * g1550["nodes"][M]["rB0"]} for M in CLASSES})
    ok = (worst == 0.0 and b_orig[0] == b_mine[0] and b_orig[1] == b_mine[1])
    say("   max |node diff| (A, B, rB0, echo of 7 classes) = %.3e ; betaL(rB0 complex) original decay %r held %r ; this script decay %r held %r  -> %s" % (
        worst, b_orig[0], b_orig[1], b_mine[0], b_mine[1], "PASS (bit-identical)" if ok else "FAIL"))
    ap_o = L.aperture({c: g1550["nodes"][int(c[1:])]["echo"] for c in CL})
    ap_m = aperture({c: g1550["nodes"][int(c[1:])]["echo"] for c in CL})
    say("   aperture echo: original %r ; this script %r -> %s" % (ap_o, ap_m, "PASS" if ap_o == ap_m else "FAIL"))
    ok_ch = True
    for ch in ("1546.003", "1554.017"):                                           # channel cells on disk right now: the loader of this script vs the original read_class
        gch = gather(ch, dict(g1550["cells"]))
        done = [M for M in CLASSES + (48,) if M in gch["nodes"]]
        wc = 0.0
        for M in done:
            a, r_, e = L.read_class({sh: band_dir(M, sh, CHANNELS[ch]["tag"]) for sh in (0, 1, 2)})
            nn = gch["nodes"][M]
            wc = max(wc, abs(a["A"] - nn["A"]), abs(a["B"] - nn["B"]), abs(r_ - nn["rB0"]), max(abs(x - y) for x, y in zip(e, nn["echo_list"])))
        ok_ch = ok_ch and wc == 0.0
        say("   channel %s: classes with 3 cells now %s -> max |node diff| vs original read_class on the channel cells = %.3e %s" % (
            ch, " ".join("x%02d" % M for M in done) or "none", wc, "PASS" if wc == 0.0 else "FAIL"))
    return ok and ap_o == ap_m and ok_ch


def run_control_ctl():
    """the three verbatim 1550 re-runs of campaign sim785 (ctl) against the stored cells they re-ran"""
    say("CONTROL 3: sim785 verbatim 1550 re-runs (controls) vs the stored 1550 cells they clone: raw terms used by the lens model, compared with ==")
    pairs = [("ctl_x06_sh0_l1550p000", 6, 0), ("ctl_x12_sh1_l1550p000", 12, 1), ("ctl_x12_sh1_apol_l1550p000", 12, 1)]
    ok = True
    for name, M, sh in pairs:
        p = BAND_ROOT + r"\261012_sim785_" + name
        if not os.path.isfile(os.path.join(p, "superramp_result.json")):
            say("   %s: NOT FOUND (skipped)" % name)
            continue
        r = jload(p)["needle_results"]["production"]
        s = jload(stored_dir(M, sh))["needle_results"]["production"]
        keys = ("amp_tA_target", "amp_tB_target", "amp_rB_zero", "echo_p2_db", "sc_shift")
        same = all(r[k] == s[k] for k in keys)
        la, lb = r["lpa"], s["lpa"]
        same_l = all(la[k]["a_pred_re"] == lb[k]["a_pred_re"] and la[k]["a_pred_im"] == lb[k]["a_pred_im"] for k in la)
        ok = ok and same and same_l
        say("   %-28s vs stored x%02d_sh%d: amp_tA/amp_tB/amp_rB_zero/echo_p2_db/sc_shift %s ; all LPA a_pred (re, im) of %d blocks %s" % (
            name, M, sh, "EXACT" if same else "DIFFER", len(la), "EXACT" if same_l else "DIFFER"))
    return ok


def run_control_gating(g1550):
    """needles for the refusal logic: a missing cell, a twin mismatch and a wrong-wavelength cell must each be refused"""
    say("CONTROL 5: refusal needles (a missing / mismatching / wrong-wavelength cell must never reach a class node)")
    cells = {k: dict(v) for k, v in g1550["cells"].items()}
    cells[(15, 1)] = dict(cells[(15, 1)], status="MISSING", why="needle")
    nodes = build_nodes({k: dict(v) for k, v in cells.items()})
    lens_missing = [M for M in CLASSES if M not in nodes]
    ok_a = (lens_missing == [15])
    say("   (a) 1550 lens with x15_sh1 marked MISSING: classes without node %s -> lens would be REFUSED: %s" % (lens_missing, "needle detected" if ok_a else "NEEDLE NOT DETECTED"))
    t = load_cell("1550", 6, 0, twin=g1550["cells"][(9, 0)])
    ok_b = (t["status"] == "INVALID" and "md5" in t["why"])
    say("   (b) stored x06_sh0 checked against the x09_sh0 cell as 'twin' (different atom list): status %s (%s) -> %s" % (t["status"], t["why"][:60], "needle detected" if ok_b else "NEEDLE NOT DETECTED"))
    pth = band_dir(6, 0, "l1546p003")
    if os.path.isfile(os.path.join(pth, "superramp_result.json")):
        w = load_cell("1554.017", 6, 0, twin=g1550["cells"][(6, 0)], path=pth)
        ok_c = (w["status"] == "INVALID" and "lam_eval_nm" in w["why"])
        say("   (c) the 1546.003 x06_sh0 cell presented as a 1554.017 cell: status %s (%s) -> %s" % (w["status"], w["why"][:70], "needle detected" if ok_c else "NEEDLE NOT DETECTED"))
        w = load_cell("1546.003", 6, 0, twin=g1550["cells"][(6, 0)])
        ok_d = (w["status"] == "OK")
        say("   (d) positive control: the 1546.003 x06_sh0 cell checked against its own 1550 twin: status %s -> %s" % (w["status"], "OK as expected" if ok_d else "UNEXPECTED: " + w["why"]))
    else:
        ok_c = ok_d = True
        say("   (c)(d) skipped (1546.003 x06_sh0 cell not found)")
    return ok_a and ok_b and ok_c and ok_d


ATOM_LEVELS = [("032p5", "53474519"), ("077p5", "f8176952"), ("122p5", "b59ed0ba"), ("167p5", "ed750b02"), ("212p5", "78259586"), ("257p5", "accded83"), ("302p5", "ee3609ef"), ("347p5", "accf4825")]
ATOM_TAGS = (("1546.003", "l1546p003"), ("1550", "l1550p000"), ("1554.017", "l1554p017"))


def atom_context():
    """DATA ONLY (no re-evaluation of the deficit / purity terms): the eight palette atoms at the three channels from the sim783 records"""
    say("=" * 150)
    say("ATOM CONTEXT (sim783 atom2d_opt.json records, nG 1793, Li; data only -- the deficit_p' / purity_p' terms of the totals above are NOT recomputed from them):")
    say("   level      md5       A deg @1546.003  @1550.000  @1554.017 | dA(1546)  dA(1554) | purity @1546 @1550 @1554 | eff % @1546 @1550 @1554 | echo dB @1546 @1550 @1554")
    out = {}
    wrp = lambda d: (d + 180.0) % 360.0 - 180.0                                  # noqa: E731
    for lv, md in ATOM_LEVELS:
        row = {}
        for ch, tag in ATOM_TAGS:
            fp = R + "\\261012_scdc_wlband\\261012_sim783_wl_a%s_%s_%s\\atom2d_opt.json" % (lv, md, tag)
            try:
                j = json.load(io.open(fp, encoding="utf-8"))
                row[ch] = dict(A=j["A_deg"] % 360.0, pur=j["purity"], eff=j["carrier_eff_pct"], echo=j.get("echo_p2_db"))
            except Exception:                                                    # noqa: BLE001
                row[ch] = None
        out[lv] = row
        if all(row[c] is not None for c in row):
            say("   %-9s %s  %8.3f %8.3f %8.3f | %+7.3f %+8.3f | %.6f %.6f %.6f | %.2f %.2f %.2f | %.2f %.2f %.2f" % (
                lv, md, row["1546.003"]["A"], row["1550"]["A"], row["1554.017"]["A"], wrp(row["1546.003"]["A"] - row["1550"]["A"]), wrp(row["1554.017"]["A"] - row["1550"]["A"]),
                row["1546.003"]["pur"], row["1550"]["pur"], row["1554.017"]["pur"], row["1546.003"]["eff"], row["1550"]["eff"], row["1554.017"]["eff"],
                row["1546.003"]["echo"], row["1550"]["echo"], row["1554.017"]["echo"]))
        else:
            say("   %-9s %s  record(s) NOT FOUND at %s" % (lv, md, [c for c in row if row[c] is None]))
    return out


# ======================================================================= channel run ================================================================================================
def fmt_cell_table(g, g1550):
    lines = ["   cell table (registration-resolved, rotated terms as in the class node; delta = channel minus the same registration at 1550):",
             "     cell      status   |A| dB   arg A   |B| dB   arg B   |rB0| dB  arg rB0  echo dB | dA dB  dArgA  dB dB  dArgB  drB0 dB dArgrB0 decho dB"]
    for (M, sh), c in sorted(g["cells"].items()):
        if c["status"] != "OK":
            lines.append("     x%02d_sh%d  %-8s %s" % (M, sh, c["status"], c["why"] if c["status"] == "INVALID" else ""))
            continue
        o = g1550["cells"].get((M, sh))
        tA = c["ampA"] / c["predA"] * cmath.exp(-2j * math.pi * sh / 3.0)
        tB = c["ampB"] / c["predB"] * cmath.exp(+2j * math.pi * sh / 3.0)
        rr = c["amprB"] / c["predrB"]
        row = "     x%02d_sh%d  OK       %+7.3f %+7.2f %+7.3f %+7.2f %+8.3f %+8.2f %8.3f |" % (
            M, sh, db(tA), math.degrees(cmath.phase(tA)), db(tB), math.degrees(cmath.phase(tB)), db(rr), math.degrees(cmath.phase(rr)), c["echo"])
        if o is not None and o["status"] == "OK":
            oA = o["ampA"] / o["predA"] * cmath.exp(-2j * math.pi * sh / 3.0)
            oB = o["ampB"] / o["predB"] * cmath.exp(+2j * math.pi * sh / 3.0)
            orr = o["amprB"] / o["predrB"]
            wr = lambda z: math.degrees(cmath.phase(z))                         # noqa: E731
            row += " %+6.3f %+6.2f %+6.3f %+6.2f %+7.3f %+7.2f %+7.3f" % (db(tA) - db(oA), wr(tA / oA), db(tB) - db(oB), wr(tB / oB), db(rr) - db(orr), wr(rr / orr), c["echo"] - o["echo"])
        lines.append(row)
    return lines


def run_channel(ch, g1550, model_names, twins):
    CH = CHANNELS[ch]
    g = g1550 if ch == "1550" else gather(ch, twins)
    miss, inval, lens_missing = inventory(g)
    say("=" * 150)
    say("CHANNEL %s nm  (lam_prop %.6f um ; design 1550 = %.6f um)  cells found %d of %d (x06..x36 : 21 ; x48 : 3 more)" % (
        ch, CH["lam_um"], LAM_D, sum(1 for c in g["cells"].values() if c["status"] == "OK"), len(g["cells"])))
    say("   MISSING cells (%d): %s" % (len(miss), " ".join(miss) if miss else "none"))
    if inval:
        say("   INVALID cells (%d, NOT used): %s" % (len(inval), " | ".join(inval)))
    say("   classes complete (3 of 3 registrations): %s ; classes UNAVAILABLE for the lens: %s" % (
        " ".join("x%02d" % M for M in CLASSES if M in g["nodes"]) or "none", " ".join("x%02d" % M for M in lens_missing) or "none"))
    say("   x48 (information only; not part of the stored lens total): %s" % ("complete" if 48 in g["nodes"] else "NOT complete"))
    say("   class nodes (3-registration coherent mean, dB) and delta against 1550:")
    say("     class   A dB    B dB    rB0 dB  rB0 deg  echo dB  | dA      dB      drB0    drB0deg  decho")
    for M in CLASSES + (48,):
        if M not in g["nodes"]:
            say("     x%02d    -- UNAVAILABLE (cells missing/invalid) --" % M)
            continue
        nn = g["nodes"][M]
        row = "     x%02d  %+7.4f %+7.4f %+7.4f %+7.2f %8.3f " % (M, db(nn["A"]), db(nn["B"]), db(nn["rB0"]), math.degrees(cmath.phase(nn["rB0"])), nn["echo"])
        if ch != "1550" and M in g1550["nodes"]:
            o = g1550["nodes"][M]
            dph = math.degrees(cmath.phase(nn["rB0"] / o["rB0"]))
            row += "| %+7.4f %+7.4f %+7.4f %+8.2f %+7.3f" % (db(nn["A"]) - db(o["A"]), db(nn["B"]) - db(o["B"]), db(nn["rB0"]) - db(o["rB0"]), dph, nn["echo"] - o["echo"])
        say(row + ("   [x48: info only]" if M == 48 else ""))
    for ln in fmt_cell_table(g, g1550) if ch != "1550" else []:
        say(ln)
    out = dict(channel=ch, missing=miss, invalid=inval, classes_unavailable=["x%02d" % M for M in lens_missing],
               nodes={"x%02d" % M: dict(A_dB=db(v["A"]), B_dB=db(v["B"]), rB0_dB=db(v["rB0"]), rB0_deg=math.degrees(cmath.phase(v["rB0"])), echo_dB=v["echo"]) for M, v in g["nodes"].items()},
               models={})
    if ch != "1550":
        ev0 = lens_from_nodes(g1550["nodes"], Geo("fixed", LAM_D), DFC769, PRT769)
        t0 = ev0["res"]["rB0_complex"]["total"]
        say("   DECOMPOSITION (a) GEOMETRY ONLY: the 1550 cells in the lens model of this channel (no channel cell used), TOTAL rB0 complex decay / held, delta vs 1550 (%+.4f / %+.4f):" % t0)
        out["geometry_only"] = {}
        for mn in model_names:
            ev = lens_from_nodes(g1550["nodes"], Geo(mn, CH["lam_um"]), DFC769, PRT769)
            t = ev["res"]["rB0_complex"]["total"]
            out["geometry_only"][mn] = dict(total=t, delta=(t[0] - t0[0], t[1] - t0[1]))
            say("        [%-12s] TOTAL %+.5f / %+.5f   delta %+.5f / %+.5f dB" % (mn, t[0], t[1], t[0] - t0[0], t[1] - t0[1]))
        have = [M for M in CLASSES if M in g["nodes"]]
        say("   DECOMPOSITION (b) ONE-CLASS SWAP (cells_only geometry; NOT a channel total -- the 1550 lens with ONLY that class's node replaced by its channel node; sensitivity, nothing is filled):")
        out["swap"] = {}
        for M in have:
            nodes2 = {k: dict(v) for k, v in g1550["nodes"].items()}
            nodes2[M] = g["nodes"][M]
            ev = lens_from_nodes(nodes2, Geo("cells_only", LAM_D), DFC769, PRT769)
            t = ev["res"]["rB0_complex"]["total"]
            out["swap"]["x%02d" % M] = dict(total=t, delta=(t[0] - t0[0], t[1] - t0[1]), aperture=ev["echo_aperture"])
            say("        x%02d only: TOTAL delta %+.5f / %+.5f dB ; echo aperture %.3f dB (1550: %.3f)" % (M, t[0] - t0[0], t[1] - t0[1], ev["echo_aperture"], ev0["echo_aperture"]))
        if not have:
            say("        (no class complete at this channel yet)")
    # x48 aperture variant (measured x48 for the > 12 Lambda zone), available when all 7 classes + x48 are complete
    for mn in model_names:
        if ch == "1550" and mn != model_names[0]:
            out["models"][mn] = "identical to %s at 1550 by construction (CONTROL 2)" % model_names[0]
            continue
        if lens_missing:
            out["models"][mn] = "REFUSED: classes unavailable " + " ".join("x%02d" % M for M in lens_missing)
            say("   [%s] LENS TOTAL / APERTURE REFUSED: class(es) %s unavailable (missing cells above) -- nothing is filled in" % (mn, " ".join("x%02d" % M for M in lens_missing)))
            continue
        geo = Geo(mn, CH["lam_um"])
        ev = lens_from_nodes(g["nodes"], geo, DFC769, PRT769)
        r_ = ev["res"]
        ap48 = aperture({c: ev["echo_class"][c] for c in CL}, e12=g["nodes"][48]["echo"]) if 48 in g["nodes"] else None
        say("   [%-12s] c0 (ideal ref) %.6f | TOTAL rB0 complex decay %+.4f held %+.4f | no_rB0 %+.4f / %+.4f | |rB0| %+.4f / %+.4f | betaL(rB0 cplx) %+.4f / %+.4f | echo aperture %.3f dB%s" % (
            mn, geo.c0(200.0), r_["rB0_complex"]["total"][0], r_["rB0_complex"]["total"][1], r_["no_rB0"]["total"][0], r_["no_rB0"]["total"][1],
            r_["rB0_abs"]["total"][0], r_["rB0_abs"]["total"][1], r_["rB0_complex"]["betaL"][0], r_["rB0_complex"]["betaL"][1], ev["echo_aperture"],
            ("   [>12L = measured x48: %.3f dB]" % ap48) if ap48 is not None else ""))
        out["models"][mn] = dict(ev=ev, aperture_x48=ap48, c0_f200=geo.c0(200.0), lam=dict(design=geo.lam_design, ref=geo.lam_ref, prop=geo.lam_prop, c=geo.lam_c))
    return out, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="1546.003,1550,1554.017")
    ap.add_argument("--models", default="all", help="fixed | all | comma list of %s" % ",".join(MODELS))
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--band-root", default=None, help="folder holding the 261012_sim785_x..._l..., default Result\\261012_scdc_lensband")
    a = ap.parse_args()
    global BAND_ROOT
    if a.band_root:
        BAND_ROOT = a.band_root
    chs = [c.strip() for c in a.channels.split(",")]
    for c in chs:
        assert c in CHANNELS, c
    mnames = list(MODELS) if a.models == "all" else [m.strip() for m in a.models.split(",")]
    for m in mnames:
        assert m in MODELS, m
    say("lensband_eval.py  %s  cores %s  numpy %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), psutil.Process().cpu_affinity(), np.__version__))
    say("band root: %s%s" % (BAND_ROOT, "" if BAND_ROOT == DEFAULT_BAND_ROOT else "   <-- NON-DEFAULT (synthetic / test cells): outputs go to *_altroot files"))
    g1550 = gather("1550")
    miss, inval, lens_missing = inventory(g1550)
    if miss or inval or lens_missing:
        say("ABORT: the stored 1550 cells are not all usable: missing %s invalid %s" % (miss, inval))
        return 3
    twins = {k: v for k, v in g1550["cells"].items()}
    rc = 0
    result = dict(stamp=time.strftime("%Y-%m-%d %H:%M:%S"), channels={})
    if not a.no_control:
        ok, ev1550 = run_control(g1550)
        ok_c = run_control_ctl()
        ok_g = run_control_gating(g1550)
        okx = True
        if a.control:
            okx = run_control_original(g1550)
        result["control"] = dict(stored_exact=ok, ctl_cells=ok_c, gating_needles=ok_g, original_code=okx if a.control else "not run (use --control)")
        say("CONTROL SUMMARY: stored-values/models/needle %s ; sim785 ctl cells %s ; refusal needles %s ; original code %s" % (
            "PASS" if ok else "FAIL", "PASS" if ok_c else "FAIL", "PASS" if ok_g else "FAIL", ("PASS" if okx else "FAIL") if a.control else "not run (use --control)"))
        rc = 0 if (ok and ok_c and ok_g and okx) else 1
        if a.control:
            return rc
    summary = []
    for ch in chs:
        out, g = run_channel(ch, g1550, mnames, twins)
        result["channels"][ch] = out
    say("=" * 150)
    say("SUMMARY (models: %s; TOTAL = rB0 complex, decay / held, dB; cells + 1550 atom-level terms; gate %.2f dB) -- 'REFUSED' = classes missing, never filled" % (",".join(mnames), GATE_TOTAL))
    for ch in chs:
        for mn in mnames:
            m = result["channels"][ch]["models"].get(mn)
            if isinstance(m, dict):
                e = m["ev"]
                tt = e["res"]["rB0_complex"]["total"]
                say("   %-9s %-13s TOTAL %+.4f / %+.4f (margin to %.2f: %+.4f / %+.4f)   echo aperture %.3f dB (margin to %.1f: %+.3f)%s" % (
                    ch, mn, tt[0], tt[1], GATE_TOTAL, tt[0] - GATE_TOTAL, tt[1] - GATE_TOTAL, e["echo_aperture"], GATE_ECHO, GATE_ECHO - e["echo_aperture"],
                    ("   with measured x48 in the >12L zone %.3f dB" % m["aperture_x48"]) if m["aperture_x48"] is not None else ""))
            else:
                say("   %-9s %-13s %s" % (ch, mn, m))
    result["atom_context"] = atom_context()
    result["elapsed_s"] = time.time() - T0
    result["band_root"] = BAND_ROOT
    jn = "lensband_eval_result.json" if BAND_ROOT == DEFAULT_BAND_ROOT else "lensband_eval_result_altroot.json"
    json.dump(result, io.open(os.path.join(WD, jn), "w", encoding="utf-8"), indent=1,
              default=lambda o: [o.real, o.imag] if isinstance(o, complex) else str(o))
    say("wrote %s (%.1f s)" % (os.path.join(WD, jn), time.time() - T0))
    return rc


if __name__ == "__main__":
    _rc = main()
    _name = "lensband_eval_control_stdout.txt" if "--control" in sys.argv else ("lensband_eval_stdout.txt" if BAND_ROOT == DEFAULT_BAND_ROOT else "lensband_eval_stdout_altroot.txt")
    io.open(os.path.join(WD, _name), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    sys.exit(_rc)
