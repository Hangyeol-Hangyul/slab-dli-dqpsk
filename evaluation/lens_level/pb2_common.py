# -*- coding: utf-8 -*-
r"""pb2_common.py -- shared readers / converters for the power-basis re-analysis (power_basis_261006, second pass, 261006 09:xx).  numpy / json only, NO solver.
Sets BLAS threads 2 BEFORE numpy.  Nothing under <DATA_ROOT> is written, nothing is deleted.

Bases (names used everywhere):
    amp_corner   : beta = amp_target / a_pred exactly as the driver stores it (arm B * exp(+2j pi sc/3) carrier translation removed)  -> today's verdict basis
    power_corner : beta * sqrt(kappa / K0_arm)                     kappa = power_target / abs(amp_target)^2 of THE CELL, K0 = single-cell reference kappa
    amp_centre   : beta * exp(+i G.delta)                          phase referred to the SITE CENTRE
    power_centre : both  (the corrected basis)
G.delta = ramp_s * ((ds + 1) if d else 1) * pi * (nxy - 1) / (M * nxy)      nxy = 96   (the verifier's bound_d09.py / pb_readout_rule.py rule)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"
import sys
sys.dont_write_bytecode = True
import json
import math
import cmath
import glob
import numpy as np

RES = r"<DATA_ROOT>\Result"
WD = r"<WORK_ROOT>\te_beta_orientation_261005\power_basis_261006"
LAM, LAMG, NSUB, THSUB, NXY = 1.55, 1.31539, 1.4440, math.radians(54.690050), 96
K0 = {"A": NSUB, "B": NSUB / math.cos(THSUB)}            # TE single-cell reference kappa (vertical exit): 1.44400 / 2.49827
K0_TM = NSUB / math.cos(THSUB)                           # TM: both arms 2.49827
BASES = ("amp_corner", "power_corner", "amp_centre", "power_centre")


def theta_cos(P_L):
    s = LAM / (P_L * LAMG)
    return (math.degrees(math.asin(s)), math.sqrt(1.0 - s * s)) if s < 1.0 else (float("nan"), float("nan"))


def read_cell(path_dir, kB_translation=+1):
    """one superramp cell (directory) -> dict; kB_translation = k in the arm-B carrier translation factor exp(+2j pi k sc / 3) (TE rule k = +1)."""
    d = json.load(open(os.path.join(path_dir, "superramp_result.json"), encoding="utf-8"))
    r = d["needle_results"]["production"]
    axis, M, rs = r["ramp_axis"], int(r["M"]), int(r["ramp_s"])
    ds = int(r.get("diag_sign", 1)) if axis == "d" else 1
    sc = int(r.get("sc_shift", 0) or 0)
    P_L = M / 3.0 if axis in ("x", "y") else M / (3.0 * math.sqrt(2.0))
    th, ct = theta_cos(P_L)
    gd = rs * ((ds + 1) if axis == "d" else 1) * math.pi * (NXY - 1) / (M * NXY)
    out = dict(path=path_dir, cell=os.path.basename(path_dir), pol=d.get("pol"), axis=axis, M=M, ramp_s=rs, ds=ds, sc=sc, P_L=P_L, theta_deg=th, cos_t=ct,
               G_delta=gd, nG_ref=d.get("nG_ref"), nG_req=d.get("nG_req"), atoms_A=r.get("atoms_A_deg"))
    for arm in ("A", "B"):
        L = r["lpa"]["arm%s_s%+d" % (arm, rs)]
        amp = complex(*r["amp_t%s_target" % arm])
        ap = complex(L["a_pred_re"], L["a_pred_im"])
        pw = r["power_t%s_target" % arm]
        kap = pw / abs(amp) ** 2 if abs(amp) > 0 else float("nan")
        tr = cmath.exp(2j * math.pi * kB_translation * sc / 3.0) if arm == "B" else 1.0
        b_ac = amp / ap * tr
        cf = {"x": K0[arm] / ct, "y": K0[arm] * ct, "d": K0[arm] * 0.5 * (1.0 / ct + ct)}[axis] if ct == ct else float("nan")
        out[arm] = dict(amp=amp, a_pred=ap, power=pw, kappa=kap, kappa_closed=cf, b_ac=b_ac, prop=bool(r.get("propagates_t%s_target" % arm)),
                        P_pred_amp=L["P_pred_amp_basis"], tr=tr)
    return out


def convert(b, kappa, arm, gd, basis, k0=None):
    """amp_corner complex beta -> requested basis (kappa of the cell, reference K0 of the arm)"""
    k0 = (K0[arm] if k0 is None else k0)
    z = b
    if basis.startswith("power"):
        z = z * math.sqrt(kappa / k0)
    if basis.endswith("centre"):
        z = z * cmath.exp(1j * gd)
    return z


def cell_beta(c, basis):
    return convert(c["A"]["b_ac"], c["A"]["kappa"], "A", c["G_delta"], basis), convert(c["B"]["b_ac"], c["B"]["kappa"], "B", c["G_delta"], basis)


def node_from_cells(cells, basis):
    """registration-coherent mean AFTER the conversion (kappa and G.delta are registration independent)"""
    bs = [cell_beta(c, basis) for c in cells]
    return sum(b[0] for b in bs) / len(bs), sum(b[1] for b in bs) / len(bs), len(bs)


def s0_cells(ramp_s, M, eq):
    """sim655 cells of one class, sorted by sc_shift"""
    pat = RES + r"\261005_scdc_tebetaori\261005_sim655_tebori_x%02d_s%+d_sh?_e%d\superramp_result.json" % (M, ramp_s, eq)
    return [read_cell(os.path.dirname(p)) for p in sorted(glob.glob(pat))]


def say(s=""):
    print(s, flush=True)


def deg(z):
    return math.degrees(cmath.phase(z))


def db_amp(z):
    return 20.0 * math.log10(abs(z))
