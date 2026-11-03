# -*- coding: utf-8 -*-
r"""conv45_readout_v4.py -- [TM] convergence check of the 45-degree (d) lens cells of sim787.  json only; NO solver; read-only on <DATA_ROOT>/Result.

WHAT IT JUDGES (adversary finding, binding): the READOUT RATIO itself at two truncations.
  r = beta_d(P_d) / beta_x(P_d) per arm (arm A, arm B) plus the rB0 ratio rR, exactly as <WORK_ROOT>/loop_261008/lens45_li_261011/readout45/lensdir45_readout_v3.py
  read_d_cell() (lines 545-571) computes it: beta on the power_centre basis (pb2_common.read_cell / convert, K0_TM), registrations sh0..sh2 averaged COHERENTLY AFTER the
  conversion (node_pc), x nodes interpolated in 1/P between the bracketing x classes (log|z| and arg z linear in 1/P, interp_c), rB0 = amp_rB_zero / a_pred of the rB0 lpa entry.
  v3 itself cannot read the new cells (it asserts nG_ref 201 at lines 494 / 554 and its module level loads the lens evaluator), so its small functions are COPIED here verbatim
  (jl, beta_pc, node_pc, r0_of, _t, interp_c, lattice_formula, ramp_geometry) and pb2_common is imported as v3 does; v3 is not modified.
  For each rung:  r_lo = all cells at the 201-equivalent truncation (what v3 uses);  r_hi = the d cell at the higher truncation / the x nodes AT THE SAME higher truncation.
      dR = r_hi / r_lo per arm A, arm B and rB0.
  dB CONVENTION (the one v3 uses for its ratios, A_dB / B_dB / rB0_dB): dR_dB = 20 log10(|r_hi| / |r_lo|), computed as 20 (log10|r_hi| - log10|r_lo|); degrees = arg(r_hi / r_lo)
      computed as the wrapped difference of the two arguments.  AMPLITUDE dB (20 log10), not power.
  ACCEPTANCE: |dR| <= 0.02 dB for arm A, arm B and rB0 at BOTH rungs -> TRANSFER ACCEPTED; any failing quantity at any available rung -> R3 NEEDED; a rung whose hi cell is absent
      or unreadable is PENDING (no verdict); the overall verdict is PENDING until both rungs are present and pass.
  AUXILIARY only: D(M) = [beta_d(hi) - beta_d(lo)] - [beta_x_samelist(hi) - beta_x_samelist(lo)], same 20 log10 convention on the power_centre beta (d18 -> x18, d24 -> x24),
      plus the identity dR = D(M) + gap, gap = same-list x shift - interpolated x shift (the quantity adv_review_261013/dshift.py calls "gap").

RUNGS (all superramp_result.json under <DATA_ROOT>/Result):
  R1  d18 psi 225, P_d = 4.243 L, bracket x12 / x15, nG_ref 401.  d lo = 261011_scdc_tmzone_licoldir/261011_sim781_d18_p225_li_e201_bt12_r ; d hi = 261013_scdc_lens45conv/261013_sim787_d18p225_e401
      x lo = 261011_scdc_tmzone_dir/261011_sim779_dir_x{12,15,18,24}e201_sh{0,1,2} ; x hi (e401) = 261009_scdc_tmzone_li5/261009_sim769_lensli5_x{12,15,18,24}_sh{0,1,2},
      EXCEPT x12 sh0 e401 = 261009_scdc_tmzone_iso/261009_sim766_iso_x12_sh0_212_257acc (as in adv_review_261013/dshift.py).
  R2  d24 psi 225, P_d = 5.657 L, bracket x15 / x18, nG_ref 301.  d lo = 261011_scdc_tmzone_licoldir/261011_sim781_d24_p225_li_e201_bt12 ; d hi = 261013_scdc_lens45conv/261013_sim787_d24p225_n301
      x hi (n301) = 261013_scdc_lens45conv/261013_sim787_x{15,18,24}n301_sh{0,1,2}.
  Per cell the json keys nG_ref / nG_ref_requested / nG_ref_matched are asserted against the rung (201 / 401 / 301) and nG_ref_actual must be IDENTICAL for the d cell and every x node of a rung.

SELF-TESTS (all must PASS, else the real read-out is skipped and the exit code is 1):
  (a) r_lo of d18 / d24 psi 225 reproduces the ratios stored in lensdir45_readout_v3_result.json (anchors_d["225"]) bit for bit / <= 1e-12 relative.
  (b) adv_review_261013/dshift.py is run (read-only) and its x e201 -> e401 node shifts per class and its same-list vs interpolated gaps are reproduced with this script's own
      functions to 1e-4 dB (power-ratio convention of dshift.py: 10 log10 of lpa ratio_meas_over_pred_amp_basis), plus the four quoted headline gaps; a perturbation needle must be rejected.
  (c) identity needle: lo d cell fed as hi d cell with the lo x nodes fed as hi x nodes -> dR exactly 0; lo d cell as hi with the e401 / n301 x nodes -> dR equals minus the
      1/P-interpolated x node shift (non-zero), computed independently from the node values.
  (d) the acceptance rule rejects 0.025 dB and accepts 0.015 dB; the overall-verdict combinations (logic needle).
  (e) end-to-end needle of the real read-out branch with real cells: lo d cell as hi at the lo truncation -> ACCEPT with dR exactly 0; tolerance forced to -1 -> FAIL; absent hi folder -> PENDING;
      a d cell of the wrong class as hi -> ERROR (the branch that reads a hi cell has no real hi cell to run on yet, so this is the only exercise of it).
Exit code: 0 after the self-tests (also when a rung is PENDING), 1 if a self-test fails, 2 if a rung hit an assertion on a hi cell.
Usage: python -I -B conv45_readout_v4.py        (the stdout of every run is also written to conv45_readout_v4_stdout_<YYMMDD_HHMMSS>.txt beside the script, created exclusively)
"""
import cmath
import datetime
import hashlib
import io
import json
import math
import os
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, r"<WORK_ROOT>\te_beta_orientation_261005\power_basis_261006")
import pb2_common as C                                                        # noqa: E402  (as v3: read_cell / convert / K0_TM; pins cores >= 10)
import psutil                                                                 # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
STAMP = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
OUTTXT = os.path.join(HERE, "conv45_readout_v5_r3_stdout_%s.txt" % STAMP)


class Tee(object):
    def __init__(self, a, f):
        self.a, self.f = a, f

    def write(self, s):
        self.a.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.a.flush()
        self.f.flush()


_F = io.open(OUTTXT, "x", encoding="utf-8")                                   # created exclusively: a previous run's stdout is never overwritten
sys.stdout = Tee(sys.stdout, _F)
sys.stderr = Tee(sys.stderr, _F)


def say(s=""):
    print(s, flush=True)


R = "<DATA_ROOT>/Result"
NEW = R + "/261011_scdc_tmzone_dir"                                           # sim779: x e201 nodes
LD = R + "/261011_scdc_tmzone_licoldir"                                       # sim781: d e201 cells
CONV = R + "/261013_scdc_lens45conv"                                          # sim787: d hi cells + n301 x nodes
V3_SCRIPT = "<WORK_ROOT>/loop_261008/lens45_li_261011/readout45/lensdir45_readout_v3.py"
V3_RESULT = "<WORK_ROOT>/loop_261008/lens45_li_261011/readout45/lensdir45_readout_v3_result.json"
DSHIFT = "<WORK_ROOT>/adv_review_261013/dshift.py"
PB2 = "<WORK_ROOT>/te_beta_orientation_261005/power_basis_261006/pb2_common.py"
MD5_AT_BUILD = {V3_SCRIPT: "bdfc538f882a94b38dad0ccb2a89d778", V3_RESULT: "6733234df3a124161a1cc49c59153f17", DSHIFT: "1919ee3a96ba7803c2a5954f0c40c25b", PB2: "a7a871dbfefcce8c461d3a2f0b8f5cfd"}
K0 = C.K0_TM
TOL_DB = 0.02
DS_OF = {225: +1, 315: -1}
KEYS = ("A", "B", "rB0")
KEYNAME = {"A": "arm A", "B": "arm B", "rB0": "rB0  "}


def x_lo_path(M, sh):
    if M == 36:                                                               # v5: x36 e201 = the sim769 cells (as v3 XREF_M / lines 460, 491)
        return path769(36, sh)
    return NEW + "/261011_sim779_dir_x%02de201_sh%d" % (M, sh)


def path769(M, sh):                                                           # v3 path769(): the e401 x nodes (x12 sh0 = the sim766 iso cell)
    if (M, sh) == (12, 0):
        return R + "/261009_scdc_tmzone_iso/261009_sim766_iso_x12_sh0_212_257acc"
    return R + "/261009_scdc_tmzone_li5/261009_sim769_lensli5_x%02d_sh%d" % (M, sh)


def x_n301_path(M, sh):
    return CONV + "/261013_sim787_x%02dn301_sh%d" % (M, sh)


R3D = R + "/261014_scdc_lens45conv_r3"                                        # sim796: R3 d36 hi cell + x24 / x36 n238 partners


def x_n238_path(M, sh):
    return R3D + "/261014_sim796_x%02dn238_sh%d" % (M, sh)


RUNGS = {
    "R1": dict(M=18, psi=225, rung=401, brk=(12, 15), classes=(12, 15, 18, 24), same=18, x_hi=path769,
               d_lo=LD + "/261011_sim781_d18_p225_li_e201_bt12_r", d_hi=CONV + "/261013_sim787_d18p225_e401"),
    "R2": dict(M=24, psi=225, rung=301, brk=(15, 18), classes=(15, 18, 24), same=24, x_hi=x_n301_path,
               d_lo=LD + "/261011_sim781_d24_p225_li_e201_bt12", d_hi=CONV + "/261013_sim787_d24p225_n301"),
    "R3": dict(M=36, psi=225, rung=238, brk=(24, 36), classes=(24, 36), same=24, x_hi=x_n238_path,
               d_lo=LD + "/261011_sim781_d36_p225_li_e201_bt14", d_hi=R3D + "/261014_sim796_d36p225_n238"),
}


# ---------------------------------------------------------------- functions copied verbatim from lensdir45_readout_v3.py (arithmetic unchanged) ----------------------------------
def jl(p):                                                                    # v3 line 106
    return json.load(io.open(os.path.join(p, "superramp_result.json"), encoding="utf-8"))


def beta_pc(p):                                                               # v3 lines 110-116
    c = C.read_cell(p, +1)
    out = []
    for a in ("A", "B"):
        b = c[a]["b_ac"] * (cmath.exp(-2j * math.pi * c["sc"] / 3.0) if a == "A" else 1.0)
        out.append(C.convert(b, c[a]["kappa"], a, c["G_delta"], "power_centre", k0=K0))
    return out, c


def node_pc(paths):                                                           # v3 lines 119-121
    bs = [beta_pc(p)[0] for p in paths]
    return [sum(b[k] for b in bs) / len(bs) for k in (0, 1)]


def r0_of(jj):                                                                # v3 lines 128-130
    q = jj["needle_results"]["production"]
    return complex(*q["amp_rB_zero"]) / complex(q["lpa"]["rB0"]["a_pred_re"], q["lpa"]["rB0"]["a_pred_im"])


def _t(P, P0, P1):                                                            # v3 lines 186-187
    return (1.0 / P0 - 1.0 / P) / (1.0 / P0 - 1.0 / P1)


def interp_c(P, pts):                                                         # v3 lines 201-212
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


def lattice_formula(n, lam, ds):                                              # v3 lines 500-502
    a = lam / 3.0
    return (0.0, n * a), (lam, -lam * ds)


def ramp_geometry(L1, L2, ramp_s):                                            # v3 lines 505-509
    det = L1[0] * L2[1] - L1[1] * L2[0]
    k1 = (L2[1] / det, -L2[0] / det)
    return 1.0 / math.hypot(k1[0], k1[1]), math.degrees(math.atan2(ramp_s * k1[1], ramp_s * k1[0])) % 360.0


# ---------------------------------------------------------------- new in v4: truncation assertion, rung-aware loaders, dR ------------------------------------------------------------
def readable(p):
    """True when superramp_result.json of folder p exists and parses (a file still being copied to the share is NOT a result yet)."""
    f = os.path.join(p, "superramp_result.json")
    if not os.path.isfile(f):
        return False
    try:
        jl(p)
        return True
    except Exception:
        return False


def check_nG(j, rung, path):
    """assert the requested / realised truncation keys against the rung (v3 asserted a fixed 201); returns nG_ref_actual"""
    ref, req, act, mat = j.get("nG_ref"), j.get("nG_ref_requested"), j.get("nG_ref_actual"), j.get("nG_ref_matched")
    assert ref == rung and req == rung and mat is True, ("nG_ref / nG_ref_requested / nG_ref_matched differ from the rung", path, ref, req, mat, rung)
    assert isinstance(act, int) and rung - 4 <= act <= rung, ("nG_ref_actual outside [rung-4, rung]", path, act, rung)
    return act


def load_x_nodes(path_fn, classes, rung):
    """x nodes of one truncation: {M: dict(P, node [cA, cB], r0, paths, rung, nG_actual)}; the three registrations are averaged exactly as v3 xref() does"""
    XN = {}
    for M in classes:
        xp = [path_fn(M, sh) for sh in (0, 1, 2)]
        acts = []
        for p in xp:
            jx = jl(p)
            assert (jx.get("rcwa_rule"), jx.get("rcwa_ff"), jx.get("ramp_s"), jx.get("ramp_axis_requested"), jx.get("pol")) == ("liy", 1, -1, "x", "TM"), p
            acts.append(check_nG(jx, rung, p))
        assert len(set(acts)) == 1, ("nG_ref_actual differs between the registrations of x%02d" % M, xp, acts)
        XN[M] = dict(P=M / 3.0, node=node_pc(xp), r0=sum(r0_of(jl(p)) for p in xp) / len(xp), paths=xp, rung=rung, nG_actual=acts[0])
    return XN


def read_d(dpath, M, psi, XN, rung, brk, same_trunc=True):
    """readout ratio of one d cell against the x nodes XN (v3 read_d_cell(), nG assertion parameterised).  same_trunc=False only for the needles."""
    j = jl(dpath)
    q = j["needle_results"]["production"]
    ds = DS_OF[psi]
    assert j["ramp_axis_requested"] == "d" and q["ramp_axis"] == "d" and j.get("pol") == "TM", dpath
    assert int(j["diag_sign"]) == ds and int(q["diag_sign"]) == ds, (dpath, j["diag_sign"], ds)
    assert (j.get("rcwa_rule"), j.get("rcwa_ff")) == ("liy", 1), dpath
    assert j.get("ramp_s") == -1 and int(j.get("sc_shift", 0)) == 0, dpath
    assert j.get("li_raster") == [[0, 1], [1, -ds]] and j.get("li_inner_axis") == 0, (dpath, j.get("li_raster"))
    act = check_nG(j, rung, dpath)
    assert int(q["M"]) == M == len(j["atoms_requested"]), dpath
    jx = jl(path769(M, 0))
    assert j["atoms_requested"] == jx["atoms_requested"] and q["atom_paths"] == jx["needle_results"]["production"]["atom_paths"], "atom list differs from the sim769 x%02d list" % M
    L1, L2, lam = q["lattice_L1_um"], q["lattice_L2_um"], q["Lx"]
    f1, f2 = lattice_formula(M, lam, ds)
    assert max(abs(L1[0] - f1[0]), abs(L1[1] - f1[1]), abs(L2[0] - f2[0]), abs(L2[1] - f2[1])) < 1e-9, ("lattice differs from the v5 formula", dpath, L1, L2, f1, f2)
    P_um, psi_g = ramp_geometry(L1, L2, -1)
    P = P_um / lam
    assert abs(psi_g - psi) < 1e-6, ("ramp direction", psi_g, psi)
    b, c = beta_pc(dpath)
    assert abs(c["P_L"] - P) < 1e-9 and abs(P - M / (3.0 * math.sqrt(2.0))) < 1e-9, (c["P_L"], P)
    below = max(m for m in XN if XN[m]["P"] <= P)
    above = min(m for m in XN if XN[m]["P"] >= P)
    assert (below, above) == tuple(brk), ("x classes bracketing P_d differ from the declared bracket", P, below, above, brk)
    if same_trunc:
        assert all(XN[m]["rung"] == rung and XN[m]["nG_actual"] == act for m in XN), ("d cell and x nodes are not at the same truncation", dpath, rung, act, {m: XN[m]["nG_actual"] for m in XN})
    xi = [interp_c(P, [(XN[m]["P"], XN[m]["node"][k]) for m in XN]) for k in (0, 1)]
    r = [b[k] / xi[k] for k in (0, 1)]
    r0x = interp_c(P, [(XN[m]["P"], XN[m]["r0"]) for m in XN])
    r0d = r0_of(j)
    return dict(M=M, psi=psi, P=P, t=_t(P, brk[0] / 3.0, brk[1] / 3.0), brk=tuple(brk), r=r, rR=r0d / r0x, beta=b, r0d=r0d, xi=xi, r0x=r0x, nG_actual=act, src=dpath, j=j)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def dlog(zh, zl):
    """(amplitude dB = 20 (log10|zh| - log10|zl|), degrees = wrapped arg difference) of zh / zl"""
    return 20.0 * (math.log10(abs(zh)) - math.log10(abs(zl))), math.degrees(wrap(cmath.phase(zh) - cmath.phase(zl)))


def ratios(res):
    return dict(A=res["r"][0], B=res["r"][1], rB0=res["rR"])


def delta_R(hi, lo):
    """dR = r_hi / r_lo per quantity: {key: (dB, deg)}"""
    rh, rl = ratios(hi), ratios(lo)
    return {k: dlog(rh[k], rl[k]) for k in KEYS}


def accepts(dr, tol=TOL_DB):
    return all(abs(dr[k][0]) <= tol for k in KEYS)


def same_structure(jlo, jhi):
    """the hi d cell must be the SAME structure as the lo d cell (only the truncation differs)"""
    for key in ("atoms_requested", "diag_sign", "li_raster", "li_inner_axis", "ramp_s", "sc_shift", "pol", "rcwa_rule", "rcwa_ff", "ramp_axis_requested"):
        assert jlo[key] == jhi[key], ("d lo / hi differ in %s" % key, jlo[key], jhi[key])
    ql, qh = jlo["needle_results"]["production"], jhi["needle_results"]["production"]
    for key in ("atom_paths", "lattice_L1_um", "lattice_L2_um", "M", "Lx", "Ly", "diag_sign", "ramp_axis", "sc_shift"):
        assert ql[key] == qh[key], ("d lo / hi differ in production.%s" % key, ql[key], qh[key])


def betas_d(res):
    """beta_d per quantity (complex): arm A, arm B (power_centre), rB0 (r0_of)"""
    return dict(A=res["beta"][0], B=res["beta"][1], rB0=res["r0d"])


def x_same(XN, M):
    n = XN[M]
    return dict(A=n["node"][0], B=n["node"][1], rB0=n["r0"])


def aux_DM(hi, lo, XNh, XNl, same):
    """AUXILIARY D(M) = d shift - same-list x shift; gap = same-list x shift - interpolated x shift; identity dR = D + gap"""
    bh, bl = betas_d(hi), betas_d(lo)
    xh, xl = x_same(XNh, same), x_same(XNl, same)
    xih = dict(A=hi["xi"][0], B=hi["xi"][1], rB0=hi["r0x"])
    xil = dict(A=lo["xi"][0], B=lo["xi"][1], rB0=lo["r0x"])
    out = {}
    for k in KEYS:
        dsh, xsh, ish = dlog(bh[k], bl[k]), dlog(xh[k], xl[k]), dlog(xih[k], xil[k])
        out[k] = dict(d_shift=dsh, x_same_shift=xsh, x_interp_shift=ish, D=(dsh[0] - xsh[0], math.degrees(wrap(math.radians(dsh[1] - xsh[1])))),
                      gap=(xsh[0] - ish[0], math.degrees(wrap(math.radians(xsh[1] - ish[1])))))
    return out


def fmt_dr(dr):
    return "  ".join("%s %+.4f dB %+.3f deg" % (k, dr[k][0], dr[k][1]) for k in KEYS)


# ---------------------------------------------------------------- self-test (b) helpers: dshift.py convention (power ratio dB of lpa ratio_meas_over_pred_amp_basis) ---------------
ARMS_B = ("armA_s-1", "armB_s-1", "rB0")


def lpa_db(p, arm):
    return 10.0 * math.log10(jl(p)["needle_results"]["production"]["lpa"][arm]["ratio_meas_over_pred_amp_basis"])


def class_shift(M, arm):
    d = [lpa_db(path769(M, s), arm) - lpa_db(x_lo_path(M, s), arm) for s in (0, 1, 2)]
    return sum(d) / 3.0, d


def own_dshift_items():
    """{label: value} of everything dshift.py prints, recomputed here"""
    items, sh = {}, {}
    for M in (12, 15, 18, 24):
        for arm in ARMS_B:
            m, d = class_shift(M, arm)
            sh[(M, arm)] = m
            items["x%02d %s mean" % (M, arm)] = m
            for i, v in enumerate(d):
                items["x%02d %s sh%d" % (M, arm, i)] = v
    for name, M, lo, hi, same in (("d18", 18, 12, 15, 18), ("d24", 24, 15, 18, 24)):
        P = M / (3.0 * math.sqrt(2.0))
        w = _t(P, lo / 3.0, hi / 3.0)
        for arm in ARMS_B:
            ip = sh[(lo, arm)] + w * (sh[(hi, arm)] - sh[(lo, arm)])
            items["%s %s interp" % (name, arm)] = ip
            items["%s %s same-list" % (name, arm)] = sh[(same, arm)]
            items["%s %s gap" % (name, arm)] = sh[(same, arm)] - ip
    return items


RX_SHIFT = re.compile(r"^x(\d\d) shift e401-e201 \(dB\): (.*)$")
RX_ARM = re.compile(r"(armA_s-1|armB_s-1|rB0) mean ([+-]\d+\.\d+) per-sh \[([^\]]*)\]")
RX_GAP = re.compile(r"^d(\d\d) P ([\d.]+) t ([\d.]+) (armA_s-1|armB_s-1|rB0)\s+interp\(x(\d+),x(\d+)\) ([+-]\d+\.\d+)\s+same-list x(\d+) ([+-]\d+\.\d+)\s+gap ([+-]\d+\.\d+) dB\s*$")


def parse_dshift(text):
    items = {}
    for line in text.splitlines():
        m = RX_SHIFT.match(line)
        if m:
            M = int(m.group(1))
            for am in RX_ARM.finditer(m.group(2)):
                items["x%02d %s mean" % (M, am.group(1))] = float(am.group(2))
                for i, v in enumerate(re.findall(r"[+-]\d+\.\d+", am.group(3))):
                    items["x%02d %s sh%d" % (M, am.group(1), i)] = float(v)
            continue
        m = RX_GAP.match(line)
        if m:
            nm, arm = "d%s" % m.group(1), m.group(4)
            items["%s %s interp" % (nm, arm)] = float(m.group(7))
            items["%s %s same-list" % (nm, arm)] = float(m.group(9))
            items["%s %s gap" % (nm, arm)] = float(m.group(10))
    return items


def compare_items(mine, ref, tol):
    """(number of reference items, list of (label, mine, ref, |diff|) beyond tol, max |diff|); a reference item missing in `mine` counts as beyond tol"""
    bad, mx = [], 0.0
    for k, v in ref.items():
        if k not in mine:
            bad.append((k, None, v, float("inf")))
            continue
        d = abs(mine[k] - v)
        mx = max(mx, d)
        if d > tol:
            bad.append((k, mine[k], v, d))
    return len(ref), bad, mx


def md5(p):
    h = hashlib.md5()
    with io.open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def eval_rung(rn, cfg, xnl, lo, emit, tol=TOL_DB):
    """one rung of the real read-out -> (status, dR or None); status PENDING (hi cell absent / unreadable, no verdict), ACCEPT, FAIL (R3 NEEDED for this rung) or ERROR (assertion on a hi cell).
    emit(line) receives the printed lines (say for the real run, a list's append in the needles)."""
    M, rung = cfg["M"], cfg["rung"]
    dhf = os.path.normpath(os.path.join(cfg["d_hi"], "superramp_result.json"))
    xfiles = [os.path.join(cfg["x_hi"](m, sh), "superramp_result.json") for m in cfg["classes"] for sh in (0, 1, 2)]
    emit("%s  d%02d psi225  nG_ref %d  P_d %.4f L  bracket x%02d/x%02d (t %.4f)" % (rn, M, rung, lo["P"], cfg["brk"][0], cfg["brk"][1], lo["t"]))
    emit("   hi d cell file: %s" % dhf)
    emit("   exists at run time: %s  (folder exists: %s, entries in folder: %s)" % (
        os.path.isfile(dhf), os.path.isdir(cfg["d_hi"]), len(os.listdir(cfg["d_hi"])) if os.path.isdir(cfg["d_hi"]) else "n/a"))
    emit("   lo d cell: %s" % os.path.normpath(cfg["d_lo"]))
    emit("   r_lo (nG_ref 201, actual %d):  %s" % (lo["nG_actual"], "  ".join("%s %+.4f dB %+.3f deg" % (k, 20 * math.log10(abs(ratios(lo)[k])), math.degrees(cmath.phase(ratios(lo)[k]))) for k in KEYS)))
    pend = []
    if not os.path.isfile(dhf):
        pend.append("hi d cell has no superramp_result.json")
    elif not readable(cfg["d_hi"]):
        pend.append("hi d cell superramp_result.json exists but is not readable yet")
    miss = [f for f in xfiles if not readable(os.path.dirname(f))]
    if miss:
        pend.append("%d hi x node file(s) missing/unreadable, e.g. %s" % (len(miss), os.path.normpath(miss[0])))
    if pend:
        emit("   %s: PENDING (no verdict) -- %s" % (rn, "; ".join(pend)))
        return "PENDING", None
    try:
        XNH = load_x_nodes(cfg["x_hi"], cfg["classes"], rung)
        hi = read_d(cfg["d_hi"], M, cfg["psi"], XNH, rung, cfg["brk"])
        same_structure(lo["j"], hi["j"])
        emit("   nG_ref_actual: d hi %d, x hi nodes %s (identical required); d lo %d, x lo nodes %s" % (
            hi["nG_actual"], sorted(set(XNH[m]["nG_actual"] for m in XNH)), lo["nG_actual"], sorted(set(xnl[m]["nG_actual"] for m in xnl))))
        emit("   r_hi (nG_ref %d, actual %d): %s" % (rung, hi["nG_actual"], "  ".join("%s %+.4f dB %+.3f deg" % (k, 20 * math.log10(abs(ratios(hi)[k])), math.degrees(cmath.phase(ratios(hi)[k]))) for k in KEYS)))
        dr = delta_R(hi, lo)
        emit("   dR table (r_hi / r_lo), limit |dR| <= %.2f dB:" % tol)
        for k in KEYS:
            emit("      %s   dR %+.4f dB  %+.3f deg   %s" % (KEYNAME[k], dr[k][0], dr[k][1], "ok" if abs(dr[k][0]) <= tol else "EXCEEDS"))
        aux = aux_DM(hi, lo, XNH, xnl, cfg["same"])
        emit("   AUXILIARY D(M) = d shift - same-list x%02d shift (20 log10 on power_centre beta; NOT a criterion):" % cfg["same"])
        for k in KEYS:
            a_ = aux[k]
            emit("      %s  d shift %+.4f dB %+.3f deg | x%02d shift %+.4f dB %+.3f deg | D(M) %+.4f dB %+.3f deg | interp-x shift %+.4f dB | gap %+.4f dB | D+gap-dR %.1e dB" % (
                KEYNAME[k], a_["d_shift"][0], a_["d_shift"][1], cfg["same"], a_["x_same_shift"][0], a_["x_same_shift"][1], a_["D"][0], a_["D"][1],
                a_["x_interp_shift"][0], a_["gap"][0], a_["D"][0] + a_["gap"][0] - dr[k][0]))
        st = "ACCEPT" if accepts(dr, tol) else "FAIL"
        emit("   %s: %s (%s)" % (rn, "within %.2f dB on A, B and rB0" % tol if st == "ACCEPT" else "R3 NEEDED for this rung", st))
        return st, dr
    except AssertionError as ex:
        emit("   %s: ERROR (assertion on a hi cell) -- %s" % (rn, ex))
        return "ERROR", None


def overall_verdict(status):
    """status {rung: PENDING / ACCEPT / FAIL / ERROR} -> verdict text"""
    if "FAIL" in status.values():
        return "R3 NEEDED (rung %s fails the 0.02 dB criterion)" % ", ".join(rn for rn, s in status.items() if s == "FAIL")
    if all(s == "ACCEPT" for s in status.values()):
        return "TRANSFER ACCEPTED (|dR| <= 0.02 dB for arm A, arm B and rB0 at both rungs)"
    return "PENDING (no verdict): " + ", ".join("%s %s" % (rn, s) for rn, s in status.items() if s != "ACCEPT")


# ================================================================ main ======================================================================================================================
def main():
    say("conv45_readout_v4.py  run %s  python %s  affinity %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sys.version.split()[0], psutil.Process().cpu_affinity()))
    say("stdout copy: %s" % OUTTXT)
    say("dB convention: dR_dB = 20 log10(|r_hi| / |r_lo|) on the complex readout ratio r = beta_d(P_d) / beta_x(P_d) (AMPLITUDE dB, the convention v3 prints as A_dB / B_dB / rB0_dB); degrees = arg(r_hi / r_lo).")
    say("acceptance: |dR| <= %.2f dB for arm A, arm B and rB0 at both rungs -> TRANSFER ACCEPTED; otherwise R3 NEEDED." % TOL_DB)
    for p, h in MD5_AT_BUILD.items():
        cur = md5(p)
        say("  input md5 %s  %s  (build-time %s: %s)" % (cur, p, h[:8], "MATCH" if cur == h else "DIFFERS"))
    say()
    results = {}                                                              # self-test name -> bool

    def report(name, ok, detail):
        results[name] = ok
        say("%-5s %s" % ("PASS" if ok else "FAIL", name))
        for line in detail:
            say("        " + line)

    XNL, XNH_NEEDLE = {}, {}
    for rn, cfg in RUNGS.items():
        XNL[rn] = load_x_nodes(x_lo_path, cfg["classes"], 201)
        XNH_NEEDLE[rn] = load_x_nodes(cfg["x_hi"], cfg["classes"], cfg["rung"])
    DLO = {rn: read_d(cfg["d_lo"], cfg["M"], cfg["psi"], XNL[rn], 201, cfg["brk"]) for rn, cfg in RUNGS.items()}

    # ---- (a) r_lo reproduces the ratios stored by v3 ----------------------------------------------------------------------------------------------------------------------------
    say("SELF-TESTS")
    stored = {a["cls"]: a for a in json.load(io.open(V3_RESULT, encoding="utf-8"))["anchors_d"]["225"]}
    det, ok = [], True
    for rn, cfg in RUNGS.items():
        st = stored[cfg["M"]]
        same_src = os.path.normpath(st["src"]) == os.path.normpath(cfg["d_lo"])
        ref = dict(A=complex(*st["r"][0]), B=complex(*st["r"][1]), rB0=complex(*st["rR"]))
        mine = ratios(DLO[rn])
        rels, bits = {}, {}
        for k in KEYS:
            rels[k] = abs(mine[k] - ref[k]) / abs(ref[k])
            bits[k] = struct.pack("<2d", mine[k].real, mine[k].imag) == struct.pack("<2d", ref[k].real, ref[k].imag)
        good = same_src and all(v <= 1e-12 for v in rels.values())
        ok = ok and good
        det.append("%s d%02d psi225: stored src == this cell: %s; bit-identical A/B/rB0: %s/%s/%s; max rel err %.2e (limit 1e-12)" % (
            rn, cfg["M"], same_src, bits["A"], bits["B"], bits["rB0"], max(rels.values())))
        det.append("    r_lo  %s" % "  ".join("%s %+.4f dB %+.3f deg" % (k, 20 * math.log10(abs(mine[k])), math.degrees(cmath.phase(mine[k]))) for k in KEYS))
    report("(a) r_lo reproduces lensdir45_readout_v3_result.json anchors_d[225] (d18, d24)", ok, det)

    # ---- (b) dshift.py reproduced ------------------------------------------------------------------------------------------------------------------------------------------------
    pr = subprocess.run([sys.executable, "-I", "-B", DSHIFT], capture_output=True, text=True, encoding="utf-8")
    ref_items = parse_dshift(pr.stdout)
    mine_items = own_dshift_items()
    n_ref, bad, mx = compare_items(mine_items, ref_items, 1e-4)
    quoted = {"d18 armA_s-1 gap": +0.0181, "d18 armB_s-1 gap": +0.0108, "d24 armA_s-1 gap": -0.0228, "d24 armB_s-1 gap": -0.0172}
    n_q, bad_q, mx_q = compare_items(mine_items, quoted, 1e-4)
    expected_n = 4 * 3 * 4 + 2 * 3 * 3                                        # 4 classes x 3 arms x (mean + 3 registrations), 2 d cells x 3 arms x (interp, same-list, gap)
    needle = dict(mine_items)
    needle["x15 armB_s-1 mean"] += 2e-4                                       # a perturbation the comparator must reject
    n_bad_needle = len(compare_items(needle, ref_items, 1e-4)[1])
    needle2 = dict(mine_items)
    needle2["d18 armA_s-1 gap"], needle2["d18 armB_s-1 gap"] = mine_items["d18 armB_s-1 gap"], mine_items["d18 armA_s-1 gap"]      # swapped arms (0.0073 dB apart)
    n_bad_needle2 = len(compare_items(needle2, ref_items, 1e-4)[1])
    ok = pr.returncode == 0 and len(ref_items) == expected_n and not bad and not bad_q and n_bad_needle >= 1 and n_bad_needle2 >= 1
    det = ["dshift.py exit %d, %d items parsed from its stdout (expected %d); compared to this script's own functions: max |diff| %.2e dB (limit 1e-4; dshift.py prints 4 decimals), items beyond limit: %d" % (
        pr.returncode, len(ref_items), expected_n, mx, len(bad))]
    for nm in ("d18", "d24"):
        for arm in ARMS_B:
            det.append("%s %-8s node-shift interp %+.4f  same-list %+.4f  gap %+.4f dB  (dshift.py: %+.4f %+.4f %+.4f)" % (
                nm, arm, mine_items["%s %s interp" % (nm, arm)], mine_items["%s %s same-list" % (nm, arm)], mine_items["%s %s gap" % (nm, arm)],
                ref_items.get("%s %s interp" % (nm, arm), float("nan")), ref_items.get("%s %s same-list" % (nm, arm), float("nan")), ref_items.get("%s %s gap" % (nm, arm), float("nan"))))
    for M in (12, 15, 18, 24):
        det.append("x%02d e401-e201 mean shift (dB): %s" % (M, "  ".join("%s %+.4f" % (a, mine_items["x%02d %s mean" % (M, a)]) for a in ARMS_B)))
    det.append("quoted headline gaps d18 A +0.0181 / B +0.0108, d24 A -0.0228 / B -0.0172: max |diff| %.2e dB, beyond limit: %d" % (mx_q, len(bad_q)))
    det.append("needles: +2e-4 dB perturbation of x15 armB mean -> %d item(s) rejected; swapped d18 arm A/B gaps -> %d item(s) rejected (both must be >= 1)" % (n_bad_needle, n_bad_needle2))
    for b_ in bad[:5]:
        det.append("BEYOND LIMIT: %s mine %s ref %s diff %s" % b_)
    report("(b) dshift.py x e201->e401 node shifts and same-list vs interpolated gaps reproduced to 1e-4 dB", ok, det)

    # ---- (c) identity needle ------------------------------------------------------------------------------------------------------------------------------------------------------
    det, ok = [], True
    for rn, cfg in RUNGS.items():
        fake = read_d(cfg["d_lo"], cfg["M"], cfg["psi"], XNL[rn], 201, cfg["brk"])
        dr0 = delta_R(fake, DLO[rn])
        exact0 = all(dr0[k][0] == 0.0 and dr0[k][1] == 0.0 for k in KEYS)
        # lo d cell as hi, e401 / n301 x nodes as hi: the d cell is NOT at the x truncation on purpose (needle only)
        fake2 = read_d(cfg["d_lo"], cfg["M"], cfg["psi"], XNH_NEEDLE[rn], 201, cfg["brk"], same_trunc=False)
        dr1 = delta_R(fake2, DLO[rn])
        m0, m1 = cfg["brk"]
        t = _t(DLO[rn]["P"], m0 / 3.0, m1 / 3.0)
        exp1, mxe = {}, 0.0
        for k in KEYS:
            xl, xh = x_same(XNL[rn], m0), x_same(XNH_NEEDLE[rn], m0)
            xl1, xh1 = x_same(XNL[rn], m1), x_same(XNH_NEEDLE[rn], m1)
            s0, s1 = dlog(xh[k], xl[k]), dlog(xh1[k], xl1[k])
            exp1[k] = (-((1 - t) * s0[0] + t * s1[0]), -((1 - t) * s0[1] + t * s1[1]))
            mxe = max(mxe, abs(dr1[k][0] - exp1[k][0]), abs(dr1[k][1] - exp1[k][1]))
        nonzero = max(abs(dr1[k][0]) for k in KEYS) > 1e-6
        aux1 = aux_DM(fake2, DLO[rn], XNH_NEEDLE[rn], XNL[rn], cfg["same"])
        mxi = max(max(abs(aux1[k]["D"][0] + aux1[k]["gap"][0] - dr1[k][0]), abs(math.degrees(wrap(math.radians(aux1[k]["D"][1] + aux1[k]["gap"][1] - dr1[k][1]))))) for k in KEYS)
        good = exact0 and mxe < 1e-9 and nonzero and mxi < 1e-9
        ok = ok and good
        det.append("%s lo-as-hi with lo x nodes: dR exactly 0 (all A/B/rB0, dB and deg): %s" % (rn, exact0))
        det.append("%s lo-as-hi with the nG%d x nodes (t %.4f): dR = %s" % (rn, cfg["rung"], t, fmt_dr(dr1)))
        det.append("    minus the interpolated x node shift, computed from node values: %s ; max |diff| %.2e (limit 1e-9); non-zero: %s" % (
            "  ".join("%s %+.4f dB %+.3f deg" % (k, exp1[k][0], exp1[k][1]) for k in KEYS), mxe, nonzero))
        det.append("    (a d cell that did not move would show this dR; it %s the %.2f dB acceptance)" % ("ACCEPTS" if accepts(dr1) else "FAILS", TOL_DB))
        det.append("    aux identity dR = D(M) + gap on these non-zero numbers (same-list x%02d): D(M) %s ; max |D+gap-dR| %.2e (limit 1e-9)" % (
            cfg["same"], "  ".join("%s %+.4f dB" % (k, aux1[k]["D"][0]) for k in KEYS), mxi))
    report("(c) identity needle (lo as hi -> 0; lo d as hi with higher-truncation x nodes -> minus the interpolated x shift)", ok, det)

    # ---- (d) acceptance logic needle ------------------------------------------------------------------------------------------------------------------------------------------
    t_hi = dict(A=(0.025, 0.0), B=(0.0, 0.0), rB0=(0.0, 0.0))
    t_lo = dict(A=(0.015, 0.0), B=(-0.015, 0.0), rB0=(0.0, 0.0))
    t_neg = dict(A=(0.0, 0.0), B=(0.0, 0.0), rB0=(-0.0201, 0.0))
    ov = [overall_verdict(dict(R1="ACCEPT", R2="ACCEPT")), overall_verdict(dict(R1="ACCEPT", R2="FAIL")), overall_verdict(dict(R1="PENDING", R2="ACCEPT")),
          overall_verdict(dict(R1="FAIL", R2="PENDING")), overall_verdict(dict(R1="PENDING", R2="PENDING")), overall_verdict(dict(R1="ERROR", R2="ACCEPT"))]
    ok = ((not accepts(t_hi)) and accepts(t_lo) and (not accepts(t_neg)) and ov[0].startswith("TRANSFER ACCEPTED") and ov[1].startswith("R3 NEEDED") and ov[2].startswith("PENDING")
          and ov[3].startswith("R3 NEEDED") and ov[4].startswith("PENDING") and ov[5].startswith("PENDING"))
    report("(d) decision logic: 0.025 dB rejected, 0.015 dB accepted, -0.0201 dB (rB0) rejected; overall verdict combinations", ok, [
        "accepts(0.025)=%s accepts(0.015)=%s accepts(-0.0201 on rB0)=%s" % (accepts(t_hi), accepts(t_lo), accepts(t_neg)),
        "overall: ACCEPT+ACCEPT -> %s | ACCEPT+FAIL -> %s | PENDING+ACCEPT -> %s | FAIL+PENDING -> %s | PENDING+PENDING -> %s | ERROR+ACCEPT -> %s" % tuple(v.split(" (")[0] for v in ov)])

    # ---- (e) end-to-end needle of the real read-out branch (eval_rung) with real cells: lo as hi at the lo truncation ----------------------------------------------------------------------
    det, ok = [], True
    for rn, cfg in RUNGS.items():
        ncfg = dict(cfg, rung=201, x_hi=x_lo_path, d_hi=cfg["d_lo"])
        buf = []
        st, dr = eval_rung(rn, ncfg, XNL[rn], DLO[rn], buf.append)
        exact0 = dr is not None and all(dr[k] == (0.0, 0.0) for k in KEYS)
        st_neg, _ = eval_rung(rn, ncfg, XNL[rn], DLO[rn], [].append, tol=-1.0)
        st_pend, _ = eval_rung(rn, dict(cfg, d_hi=R + "/__needle_absent_folder__"), XNL[rn], DLO[rn], [].append)
        other = [c for r2, c in RUNGS.items() if r2 != rn][0]
        st_err, _ = eval_rung(rn, dict(ncfg, d_hi=other["d_lo"]), XNL[rn], DLO[rn], [].append)
        good = st == "ACCEPT" and exact0 and st_neg == "FAIL" and st_pend == "PENDING" and st_err == "ERROR"
        ok = ok and good
        det.append("%s eval_rung: lo as hi -> %s (dR exactly 0: %s); tolerance forced to -1 dB -> %s; absent hi d folder -> %s; wrong-class d cell as hi -> %s" % (rn, st, exact0, st_neg, st_pend, st_err))
    report("(e) end-to-end needle of the verdict branch (eval_rung): ACCEPT / FAIL / PENDING / ERROR all reachable", ok, det)

    say()
    if not all(results.values()):
        say("SELF-TESTS: FAILED (%s) -- the real read-out is SKIPPED, no verdict." % ", ".join(k for k, v in results.items() if not v))
        return 1
    say("SELF-TESTS: ALL PASS (%d of %d)" % (sum(results.values()), len(results)))
    say()

    # ---- real read-out -------------------------------------------------------------------------------------------------------------------------------------------------------------
    say("READ-OUT (r_hi / r_lo)")
    status = {}
    for rn, cfg in RUNGS.items():
        status[rn], _dr = eval_rung(rn, cfg, XNL[rn], DLO[rn], say)
        say()

    say("SUMMARY")
    for rn, cfg in RUNGS.items():
        say("  %s (d%02d psi225, nG_ref %d): %s" % (rn, cfg["M"], cfg["rung"], status[rn]))
    verdict = overall_verdict(status)
    say("  OVERALL: %s" % verdict)
    say("  stdout copy: %s" % OUTTXT)
    return 2 if "ERROR" in status.values() else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.exit(code)
