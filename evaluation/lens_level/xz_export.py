# -*- coding: utf-8 -*-
r"""
xz_export.py -- regenerate and SAVE the F1 (x-z cut) arrays of tm_focus_p769li.py (figure tm_focus_xz.png).

Method (per task): read the verified source as TEXT, rewrite ONLY the OUTDIR and SCRATCH assignments to point at
xz_export\run\, exec() the text up to (not including) the line "# SECTION 6 --" (md5 gate, CONTROL-V, CONTROL-F and the
construction of S0_M2A run exactly as in the verified run), then run the F1 loop + normalisation lines copied VERBATIM
(sliced out of the source text, not retyped). The source file is never written.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
import time
import hashlib
import json
import psutil

T0 = time.time()
SRC = r"<WORK_ROOT>\tm_focus_261011\src\tm_focus_p769li.py"
EXPORT_DIR = r"<WORK_ROOT>\tm_focus_261011\xz_export"
RUN_DIR = EXPORT_DIR + r"\run"
NPZ_PATH = EXPORT_DIR + r"\tm_focus_xz_arrays.npz"
SRC_MD5_EXPECTED = "9995a87ac8afce81b0cca56b767a912d"
CORES = [10, 11, 12, 13]


def log(m):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


_vm = psutil.virtual_memory()
log("RAM used %.1f %%" % _vm.percent)
_lim = float(os.environ.get("SIM_MAX_RAM_PCT", "0"))   # opt-in RAM ceiling
if _lim > 0 and _vm.percent >= _lim:
    log("REFUSING: RAM used >= %.0f %%" % _lim)
    sys.exit(1)

if not os.path.exists(SRC):
    print("NOT FOUND: %s" % SRC)
    sys.exit(1)
src_bytes = open(SRC, "rb").read()
src_md5 = hashlib.md5(src_bytes).hexdigest()
log("source md5 %s" % src_md5)
assert src_md5 == SRC_MD5_EXPECTED, "source changed since it was verified: %s" % src_md5
text = src_bytes.decode("utf-8")
os.makedirs(RUN_DIR, exist_ok=True)

# ---- rewrite ONLY OUTDIR and SCRATCH (each exactly once) ----
OLD_SCRATCH = 'SCRATCH = r"<WORK_ROOT>\\tm_focus_261011\\scratch"\n'
OLD_OUTDIR = 'OUTDIR = r"<WORK_ROOT>\\tm_focus_261011"\n'
assert text.count(OLD_SCRATCH) == 1, text.count(OLD_SCRATCH)
assert text.count(OLD_OUTDIR) == 1, text.count(OLD_OUTDIR)
text = text.replace(OLD_SCRATCH, 'SCRATCH = r"%s"\n' % RUN_DIR)
text = text.replace(OLD_OUTDIR, 'OUTDIR = r"%s"\n' % RUN_DIR)
assert text.count('SCRATCH = r"%s"\n' % RUN_DIR) == 1 and text.count('OUTDIR = r"%s"\n' % RUN_DIR) == 1

# ---- cut at the SECTION 6 header ("# SECTION 6 --" matches once; "# SECTION 6b/6c" are later sections) ----
CUT = "# SECTION 6 --"
lines = text.splitlines(keepends=True)
cut_idx = [i for i, ln in enumerate(lines) if ln.startswith(CUT)]
assert len(cut_idx) == 1, cut_idx
log("cut at source line %d: %s" % (cut_idx[0] + 1, lines[cut_idx[0]].strip()))
head_src = "".join(lines[:cut_idx[0]])

sys.argv = [SRC]
ns = {"__name__": "__main__", "__file__": SRC}
log("exec head of source ...")
exec(compile(head_src, SRC, "exec"), ns)
log("exec head done")


assert ns["CONTROL_V"]["pass"] is True, "CONTROL-V did not pass"
assert ns["CONTROL_F"]["pass"] is True, "CONTROL-F did not pass"
log("CONTROL-V pass=%s CONTROL-F pass=%s" % (ns["CONTROL_V"]["pass"], ns["CONTROL_F"]["pass"]))
assert "S0_M2A" in ns and "GRID1" in ns and "make_H" in ns and "eval_with_H" in ns

# ---- Z_GRID: defined in SECTION 6 (not in the head) -> take the source's own line ----
ZLINE = "Z_GRID = np.arange(0.0, 260.0 + 1e-9, 0.5)\n"
assert text.count(ZLINE) == 1
exec(compile(ZLINE, "Z_GRID_line_of_source", "exec"), ns)
assert len(ns["Z_GRID"]) == 521

# ---- F1 loop + normalisation: slice VERBATIM out of the source text ----
START = 'log("computing F1 (xz slice) ...")\n'
END = "I1n_row = np.clip(I1n_row, 0.0, 1.0)\n"
assert text.count(START) == 1 and text.count(END) == 1
i0 = text.index(START)
i1 = text.index(END) + len(END)
f1_src = text[i0:i1]
assert "eval_with_H(S0_M2A, H1z, GRID1, X_F1, [XC])[:, 0]" in f1_src
# keep line numbers meaningful for tracebacks: pad with newlines up to the real start line
start_line = text[:i0].count("\n") + 1
log("F1 block = source lines %d..%d" % (start_line, start_line + f1_src.count("\n") - 1))
t_f1 = time.time()
exec(compile("\n" * (start_line - 1) + f1_src, SRC, "exec"), ns)
f1_wall = time.time() - t_f1
log("F1 done in %.1f s" % f1_wall)

import numpy as np
X_F1 = ns["X_F1"]; Z_GRID = ns["Z_GRID"]; I1 = ns["I1"]
I1_peak = ns["I1_peak"]; I1n_global = ns["I1n_global"]; I1_db = ns["I1_db"]; I1n_row = ns["I1n_row"]
XC = float(ns["XC"])
assert I1.shape == (521, 601) and np.all(np.isfinite(I1))
ip = np.unravel_index(int(np.argmax(I1)), I1.shape)
log("I1_peak = %.9e at z=%.1f um, x=%.1f um (iz=%d ix=%d); XC=%.6f" % (I1_peak, Z_GRID[ip[0]], X_F1[ip[1]], ip[0], ip[1], XC))

KEY_TEXT = ("x-z cut of the TM M2 arm A near-field propagation (P769 Li palette, Li lens ladder, rB0_complex basis): "
            "|E|^2 vs (z, x) at y = +XC (XC = %.6f um, i.e. through the two upper PDs at x = +-XC; NOTE eval_with_H applies its first "
            "coordinate array to the row (y) axis, so the code literally sweeps y at x = +XC; the design is exactly x<->y symmetric "
            "so it equals the x-z slice at y = +XC). z = 0 at the metasurface plane, z increases downward, z = 200 um = PD plane "
            "(focal plane). Rows = z (Z_GRID, 521 points 0..260 step 0.5 um), columns = x (X_F1, 601 points -150..150 step 0.5 um). "
            "I1 = raw |E|^2 in script units (not power-basis); I1_peak = max(I1) = global focal peak; I1n_global = I1/I1_peak; "
            "I1_db = 10log10(clip(I1n_global, 1e-6, None)) (floor -60 dB, figure displays -40..0 dB); I1n_row = per-row I1 / "
            "(99.5th percentile over x of I1 at that z), clipped to [0,1]." % XC)

np.savez(NPZ_PATH, X_F1=X_F1, Z_GRID=Z_GRID, I1=I1, I1_peak=np.float64(I1_peak), I1n_global=I1n_global, I1_db=I1_db,
         I1n_row=I1n_row, XC_um=np.float64(XC), slice_description=np.array(KEY_TEXT),
         source_md5=np.array(src_md5), f1_wall_s=np.float64(f1_wall))
log("saved %s (%d bytes)" % (NPZ_PATH, os.path.getsize(NPZ_PATH)))
rss = psutil.Process().memory_info().rss / 1e6
meta = dict(wall_s=time.time() - T0, f1_wall_s=f1_wall, rss_mb_final=rss, source_md5=src_md5, I1_peak=float(I1_peak),
            peak_iz=int(ip[0]), peak_ix=int(ip[1]), peak_z_um=float(Z_GRID[ip[0]]), peak_x_um=float(X_F1[ip[1]]),
            affinity=psutil.Process().cpu_affinity(), run_dir=RUN_DIR)
json.dump(meta, open(EXPORT_DIR + r"\xz_export_meta.json", "w", encoding="utf-8"), indent=2)
log("ALL DONE wall=%.1fs RSS=%.1f MB" % (time.time() - T0, rss))
