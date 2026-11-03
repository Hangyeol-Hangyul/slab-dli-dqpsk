# -*- coding: utf-8 -*-
"""Branch B (overnight 261015): input metagrating sim752 at the six missing 100 GHz channels.

Standalone, read-only on <DATA_ROOT> (it only READS rho1.npy, rho2.npy, input2d.json and the
job file). Imports the FROZEN snapshot of the driver in .\\snapshot (never runs its main).
Calls measure_air(rho1, rho2, ph, 901, lam=lam, want_jones=True) exactly as _band() does.

Stages (written incrementally to b_results.json and appended to progress.md):
  repro   the five stored channels (literal lams of _band()), residual <= 1e-6 required
  needle  the same five with lam + 0.1 nm: the SAME check must FAIL on every one
  new     the six missing channels (+4 +2 +1 -1 -2 -4) at the exact grid wavelengths of C2
  bonus   the four off-grid stored channels re-solved at their exact grid wavelength
"""
import os
import sys
import time
import json
import threading

HERE = r"<WORK_ROOT>\overnight_261015\B_input_channels"
SNAP = os.path.join(HERE, "snapshot")
RUN = r"<DATA_ROOT>\Result\261008_scdc_in2dli\261008_sim752_in2dLi_warm_xsec_min_rf050_W1"
JOB = os.path.join(RUN, "job_21_261008_sim752_in2dLi_warm_xsec_min_rf050_W1.txt")
OUTJ = os.path.join(HERE, "b_results.json")
PROG = os.path.join(HERE, "progress.md")
CORES = [10, 11, 12]
NTHR = "3"
TOL = 1.0e-6
KEYS7 = ("eta_x", "eta_y", "leak_x", "leak_y", "refl_x", "refl_y", "refl_worst_sop")
NG_BAND = 901

# thread caps BEFORE numpy / scipy are imported (3 threads = size of the core set)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = NTHR

import numpy as np            # noqa: E402
import scipy.linalg           # noqa: E402,F401  (its own BLAS copy reads the env now, not at the driver import)
import psutil                 # noqa: E402

# the driver reads sys.argv[1] as the job file at import (CFG -> rcwa_rule liy, rcwa_ff 1, blas_threads)
sys.argv = [os.path.join(HERE, "b_band_channels.py"), JOB]
sys.dont_write_bytecode = True
sys.path.insert(0, SNAP)
import Main_rcwa_input2d as D   # noqa: E402  (snapshot copy; its main() is never called)


# ---------------------------------------------------------------------------
_state = {"rss_max": 0.0, "stop": False}


def _watch():
    p = psutil.Process()
    while not _state["stop"]:
        try:
            r = p.memory_info().rss / 1e9
            if r > _state["rss_max"]:
                _state["rss_max"] = r
            if r > 4.8:                     # hard safety net (task limit 4 GB; stop before the box suffers)
                with open(PROG, "a", encoding="utf-8") as fh:
                    fh.write("- ABORT: process RSS %.2f GB > 4.8 GB, killed by the watcher\n" % r)
                os._exit(7)
        except Exception:
            pass
        time.sleep(0.25)


threading.Thread(target=_watch, daemon=True).start()


def note(msg):
    line = "- [%s] %s\n" % (time.strftime("%H:%M:%S"), msg)
    print(line, end="", flush=True)
    with open(PROG, "a", encoding="utf-8") as fh:
        fh.write(line)


RES = {"stages": {}, "meta": {}}


def save():
    tmp = OUTJ + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=1)
    os.replace(tmp, OUTJ)


# ---------------------------------------------------------------------------
# setup exactly as main(): params from the stored job file, substrate / period, Phys
P = D.build_params(D.CFG)
D.G.set_substrate(float(P["n_sub"]))
D.G.set_period(float(P["period_nm"]))
ph = D.Phys(P)
assert D._RCWA_RULE == "liy" and D._RCWA_FF == 1 and D._LI_ON, (D._RCWA_RULE, D._RCWA_FF)
assert int(P["nG_band"]) == NG_BAND and int(P["report_band"]) == 1
assert abs(float(P["lam_nm"]) - 1550.0) < 1e-12

with open(os.path.join(RUN, "input2d.json"), "r", encoding="utf-8") as fh:
    ST = json.load(fh)
rho1 = np.load(os.path.join(RUN, "rho1.npy")).astype(float)
rho2 = np.load(os.path.join(RUN, "rho2.npy")).astype(float)
assert rho1.shape == (ph.N, ph.N) and rho2.shape == (ph.N, ph.N)
assert set(np.unique(rho1)) <= {0.0, 1.0} and set(np.unique(rho2)) <= {0.0, 1.0}
m1 = D._md5_arr(rho1.astype(np.uint8))
m2 = D._md5_arr(rho2.astype(np.uint8))
assert m1 == ST["density_md5_L1"] and m2 == ST["density_md5_L2"], (m1, m2)

D.set_bk("numpy")      # full_report() does this first; the optimiser leaves the autograd backend on

RES["meta"] = {
    "driver_snapshot": os.path.join(SNAP, "Main_rcwa_input2d.py"),
    "driver_md5_snapshot": D.DRIVER_MD5,
    "driver_md5_stored_run": ST["driver_md5"],
    "grcwa_rcwa_md5_snapshot": D._md5_file(os.path.join(D._GRCWA_DIR, "rcwa.py")),
    "grcwa_rcwa_md5_stored_run": ST["grcwa_rcwa_md5"],
    "grcwa_fft_funs_md5_snapshot": D._md5_file(os.path.join(D._GRCWA_DIR, "fft_funs.py")),
    "grcwa_fft_funs_md5_stored_run": ST["grcwa_fft_funs_md5"],
    "grcwa_li_fmm_md5_snapshot": D._md5_file(os.path.join(D._GRCWA_DIR, "li_fmm.py")),
    "grcwa_li_fmm_md5_stored_run": ST["grcwa_li_fmm_md5"],
    "scd_c_grating_md5_snapshot": D._md5_file(D.G.__file__.replace(".pyc", ".py")),
    "scd_c_grating_md5_stored_run": ST["scd_c_grating_md5"],
    "grcwa_dir": D._GRCWA_DIR, "scd_module": D.G.__file__,
    "density_md5_L1": m1, "density_md5_L2": m2,
    "rcwa_rule": D._RCWA_RULE, "rcwa_ff": D._RCWA_FF, "nG_band": NG_BAND,
    "period_nm": float(P["period_nm"]), "n_sub": float(P["n_sub"]), "H1_nm": float(P["H1_nm"]),
    "H2_nm": float(P["H2_nm"]), "material1": P["material1"], "material2": P["material2"],
    "fill1": P["fill1"], "fill2": P["fill2"], "eps": [ph.e1s, ph.e1v, ph.e2s, ph.e2v, ph.eps_sub],
    "cores": CORES, "threads": NTHR, "numpy": np.__version__, "python": sys.version.split()[0],
    "host": D.socket.gethostname(),
}
save()
note("setup OK: P from the stored job file (rule %s, ff %d, nG_band %d), density md5 L1/L2 equal the stored, "
     "affinity %s, threads %s" % (D._RCWA_RULE, D._RCWA_FF, NG_BAND, sorted(psutil.Process().cpu_affinity()), NTHR))


# ---------------------------------------------------------------------------
def solve(lam_um, label):
    est = 4.6 * (NG_BAND / 889.0) ** 1.95 * 0.55
    for _try in range(30):
        vm = psutil.virtual_memory()
        if D.ram_guard(est, label):
            break
        note("RAM wait (%s): available %.1f GB, retry %d" % (label, vm.available / 1e9, _try))
        time.sleep(60)
    else:
        note("FAILURE: RAM never became available for %s" % label)
        raise SystemExit(5)
    av = psutil.virtual_memory().available / 1e9
    t0 = time.time()
    a = D.measure_air(rho1, rho2, ph, NG_BAND, lam=lam_um, want_jones=True)
    dt = time.time() - t0
    rec = {k: float(a[k]) for k in KEYS7}
    rec.update(label=label, lam_um=lam_um, lam_nm=lam_um * 1e3, t_s=dt, nG_realised=int(a["nG_realised"]),
               K=int(a["K"]), energy_Ex=float(a["air_Ex_energy"]), energy_Ey=float(a["air_Ey_energy"]),
               halves_vs_RT=float(max(a["air_Ex_halves_vs_RT"], a["air_Ey_halves_vs_RT"])),
               avail_before_gb=av, rss_max_gb=_state["rss_max"], peak_wset_gb=D.peak_gb(),
               affinity=sorted(psutil.Process().cpu_affinity()))
    del a
    return rec


def resid(rec, stored):
    r = {k: abs(rec[k] - stored[k]) for k in KEYS7}
    rel = {k: abs(rec[k] - stored[k]) / abs(stored[k]) for k in KEYS7}
    return r, rel


def stored_of(lam_um):
    tg = "lam%.2f" % (lam_um * 1e3)
    return {k: float(ST["%s_%s" % (tg, k)]) for k in KEYS7}, tg


# the five literal wavelengths of _band() (um)
STORED_LAMS = (1.546, 1.5476, 1.550, 1.55241, 1.55402)
STORED_NAMES = {1.546: "+5", 1.5476: "+3", 1.550: "0", 1.55241: "-3", 1.55402: "-5"}

# the exact 100 GHz grid: f0 = c / 1550.000 nm, channel +k at f0 + k * 100 GHz
C0 = 299792458.0
F0 = C0 / 1550.0e-9
GRID = {k: C0 / (F0 + k * 100.0e9) * 1e9 for k in range(5, -6, -1)}          # nm
C2_GRID = {5: 1546.0034, 4: 1546.8011, 3: 1547.5996, 2: 1548.3989, 1: 1549.1990, 0: 1550.0000,
           -1: 1550.8018, -2: 1551.6044, -3: 1552.4079, -4: 1553.2122, -5: 1554.0173}
for k in C2_GRID:
    assert abs(GRID[k] - C2_GRID[k]) < 6e-5, (k, GRID[k], C2_GRID[k])
RES["meta"]["grid_nm_computed"] = {str(k): GRID[k] for k in GRID}
RES["meta"]["grid_nm_C2"] = {str(k): C2_GRID[k] for k in C2_GRID}
save()
note("grid check: computed f0 + k*100 GHz wavelengths equal the C2 list to < 6e-5 nm for all 11 channels")

# ---------------------------------------------------------------------------
# stage 1: reproduce the five stored channels
rep = []
for lam in STORED_LAMS:
    st, tg = stored_of(lam)
    rec = solve(lam, "repro " + STORED_NAMES[lam] + " " + tg)
    r, rel = resid(rec, st)
    rec.update(stage="repro", ch=STORED_NAMES[lam], tag=tg, stored=st, resid_abs=r, resid_rel=rel,
               max_resid=max(r.values()), passed=bool(max(r.values()) <= TOL))
    rep.append(rec)
    RES["stages"]["repro"] = rep
    save()
    note("repro %s (%s): max |resid| %.3e over 7 keys, %s, %.0f s, rss max %.2f GB"
         % (rec["ch"], tg, rec["max_resid"], "PASS" if rec["passed"] else "FAIL", rec["t_s"], _state["rss_max"]))
if not all(r["passed"] for r in rep):
    RES["verdict"] = "REPRODUCTION FAILED -- new channels NOT computed"
    save()
    note("REPRODUCTION FAILED at nG %d; stopping before the six new channels" % NG_BAND)
    with open(PROG, "a", encoding="utf-8") as fh:
        fh.write("STATUS: COMPLETE (reproduction failed)\n")
    raise SystemExit(3)

# stage 2: needle -- lam + 0.1 nm must make the SAME check fail
ndl = []
for lam in STORED_LAMS:
    st, tg = stored_of(lam)
    rec = solve(lam + 1.0e-4, "needle " + STORED_NAMES[lam] + " +0.1nm")
    r, rel = resid(rec, st)
    rec.update(stage="needle", ch=STORED_NAMES[lam], tag=tg, stored=st, resid_abs=r, resid_rel=rel,
               max_resid=max(r.values()), check_failed=bool(max(r.values()) > TOL))
    ndl.append(rec)
    RES["stages"]["needle"] = ndl
    save()
    note("needle %s lam+0.1 nm: max |resid| %.3e -> check %s (needed FAIL)"
         % (rec["ch"], rec["max_resid"], "FAILED as required" if rec["check_failed"] else "STILL PASSES (needle not caught)"))
RES["needle_all_caught"] = bool(all(n["check_failed"] for n in ndl))
save()
if not RES["needle_all_caught"]:
    note("NEEDLE NOT CAUGHT on every channel -- see b_results.json; the six new channels are still computed but flagged")

# stage 3: the six new channels at the exact grid wavelengths of C2
NEW = [(4, 1546.8011), (2, 1548.3989), (1, 1549.1990), (-1, 1550.8018), (-2, 1551.6044), (-4, 1553.2122)]
new = []
for k, lam_nm in NEW:
    rec = solve(lam_nm * 1e-3, "new %+d %.4f" % (k, lam_nm))
    rec.update(stage="new", ch="%+d" % k, k=k)
    new.append(rec)
    RES["stages"]["new"] = new
    save()
    note("new %+d (%.4f nm): eta_x %.6f eta_y %.6f leak %.3e/%.3e refl_worst %.3e, %.0f s, rss max %.2f GB"
         % (k, lam_nm, rec["eta_x"], rec["eta_y"], rec["leak_x"], rec["leak_y"], rec["refl_worst_sop"], rec["t_s"],
            _state["rss_max"]))

# stage 4: bonus -- the four off-grid stored channels at their exact grid wavelength (same basis as the six)
bon = []
for k, lam_nm in ((5, 1546.0034), (3, 1547.5996), (-3, 1552.4079), (-5, 1554.0173)):
    rec = solve(lam_nm * 1e-3, "bonus %+d %.4f" % (k, lam_nm))
    rec.update(stage="bonus", ch="%+d" % k, k=k)
    bon.append(rec)
    RES["stages"]["bonus"] = bon
    save()
    note("bonus %+d (%.4f nm): eta_x %.6f eta_y %.6f leak %.3e/%.3e refl_worst %.3e, %.0f s"
         % (k, lam_nm, rec["eta_x"], rec["eta_y"], rec["leak_x"], rec["leak_y"], rec["refl_worst_sop"], rec["t_s"]))

RES["rss_max_gb"] = _state["rss_max"]
RES["affinity_end"] = sorted(psutil.Process().cpu_affinity())
RES["verdict"] = "COMPLETE"
save()
_state["stop"] = True
note("all stages done; max RSS %.2f GB; affinity at end %s" % (_state["rss_max"], RES["affinity_end"]))
with open(PROG, "a", encoding="utf-8") as fh:
    fh.write("STATUS: COMPLETE\n")
