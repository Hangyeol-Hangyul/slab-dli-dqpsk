# -*- coding: utf-8 -*-
"""[261015] Shared loader/mapper for the bottom-metagrating chain snapshot figure (sim799 re-runs).
Uses the driver's own functions (scd_c_grating.density / seed_bank / binarize / set_period / set_min_feature).
Read-only on every input.  Imported by draw_figure.py and selfcheck.py."""
import os
import sys
import io
import json
import hashlib

sys.dont_write_bytecode = True
import numpy as np

RC = r"<CODE_ROOT>\Version_unified\RCWA_custom"
sys.path.insert(0, RC)
import scd_c_grating as M            # noqa: E402


RES = r"<DATA_ROOT>\Result\261015_scdc_botsnap"
SNAP = os.path.join(RES, "_snapshots")
REV = r"<WORK_ROOT>\paper_scdc_261011_v3\review\si_optprocess_261014"
OUT = os.path.join(REV, "botchain_snap_261015")
TABLE = os.path.join(REV, "botchain_table.tsv")

ITERS = 400
PERIOD_NM = 1315.39
NX = 512
BETA_SCHEDULE = ((0, 8.0), (0.4, 16.0), (0.7, 32.0), (0.9, 64.0))   # optimize() default; Main_rcwa_scd_c.py passes none


def beta_of(t, iters=ITERS, sched=BETA_SCHEDULE):
    """beta of iteration t (1-based), exactly the expression in optimize()."""
    return [bv for fs, bv in sched if t / float(iters) >= fs][-1]


def md5_file(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def read_table():
    rows = []
    L = io.open(TABLE, encoding="utf-8").read().splitlines()
    hdr = L[0].split("\t")
    for ln in L[1:]:
        if ln.strip():
            rows.append(dict(zip(hdr, ln.split("\t"))))
    return rows


def parse_job(p):
    d = {}
    for ln in io.open(p, encoding="utf-8"):
        if ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        d[k.strip()] = v.strip()
    return d


def stage_info():
    """list of dicts, stage 1..9 (index 0..8)."""
    tab = read_table()
    out = []
    for r in tab:
        s = int(r["stage"])
        sim = r["sim"]
        sub = "261015_sim799_botsnap_s%d_%s" % (s, sim)
        sdir = os.path.join(SNAP, sub)
        rdir = os.path.join(RES, sub)
        job = parse_job([os.path.join(rdir, f) for f in os.listdir(rdir) if f.startswith("job_")][0])
        seed = "multi_q" if s == 1 else "warm"
        fin_names = sorted(f for f in os.listdir(sdir) if f.startswith("final_density_"))
        # adopted density = final_density_scdc_extract_H400_300_<seed>.npy (the per-seed file); the unsuffixed / infeasible
        # copies are checked in selfcheck
        fin = os.path.join(sdir, "final_density_scdc_extract_H400_300_%s.npy" % seed)
        hist = os.path.join(sdir, "final_hist_scdc_extract_H400_300_%s.npy" % seed)
        js_main = os.path.join(sdir, "final_scdc_extract_H400_300.json")
        js_inf = os.path.join(sdir, "final_infeasible_best_scdc_extract_H400_300.json")
        out.append(dict(stage=s, sim=sim, sub=sub, sdir=sdir, rdir=rdir, job=job, seed=seed, final=fin, hist=hist,
                        json_main=js_main, json_inf=(js_inf if os.path.isfile(js_inf) else None),
                        filt=float(job["filt_radius_nm"]), md5_8=r["md5_8"], row=r, fin_names=fin_names))
    return out


def set_stage_module(filt_nm):
    """same call order as Main_rcwa_scd_c.py:544 (set_period) then :1465 (set_min_feature(filt_radius_nm))."""
    M.set_period(PERIOD_NM)
    px = M.set_min_feature(filt_nm)
    assert M.NX == NX and M.GRID_2D is None
    return px


def map_xy(xy, beta):
    """raw design variables (1024) -> projected density of L1, L2 (driver density(), nest = 0)."""
    r1 = np.asarray(M.density(xy[:NX], beta), float)
    r2 = np.asarray(M.density(xy[NX:], beta), float)
    return r1, r2


def start_xy(stage, prev_final=None):
    """stage 1: analytic multi_q seed of seed_bank(harmonics (1, 2)); stages 2-9: 0.15 + 0.70 clip(previous final) (Main_rcwa_scd_c.py:1584)."""
    if stage == 1:
        bank = M.seed_bank(0, rng=None, harmonics=(1, 2))
        nm, x1, x2 = [b for b in bank if b[0] == "multi_q"][0]
        return np.clip(np.concatenate([np.asarray(x1, float), np.asarray(x2, float)]), 0.0, 1.0)
    w = 0.15 + 0.70 * np.clip(np.asarray(prev_final, float), 0.0, 1.0)
    return np.clip(np.concatenate([w[0], w[1]]), 0.0, 1.0)


def load_ckpt(sdir, t):
    z = np.load(os.path.join(sdir, "ckpt_iter%03d.npz" % t), allow_pickle=False)
    assert int(z["t"]) == t, (sdir, t, int(z["t"]))
    return z
