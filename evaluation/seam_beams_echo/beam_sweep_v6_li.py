r"""Angular-spectrum sweep, Li basis -- Fig. 6(b) regeneration (branch C, 261015).

COPY of orig_scripts\beam_sweep_v6.py (md5 2326a945...).  CHANGED, and ONLY these, in the physics:
    1. M.set_rcwa("liy", 1)        rcwa_rule = liy, rcwa_ff = 1  (basis of the printed numbers, carli json n1708_*)
    2. NG_SWEEP = 257              nG_eval of the printed numbers (was 161, Laurent)
Everything else is verbatim: density (md5 525f74f3 asserted), stack (substrate / L1 400 / L2 300 / 465 nm SiO2 spacer /
L3B / L3 2-layer termination slab, token 0.16:172 = n1708, held FIXED across the 25 angles), w0 = 100 um, NANG = 25,
DKX_MAX = 5/w0, orders -4..4, lambda 1.55 um.
Execution-only changes: only the TM passes are run (the deck panel is TM; TE is not needed), ONE pass per process
(argv[1] = TM_pass1 | TM_pass2, argv[2] = core number, 14 or 15), the affinity pin is set AFTER the imports, output goes
to this folder.  Read-only on Result\ and code\Version_unified.
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
import json
import time

import numpy as np
import psutil

sys.path.insert(0, r"<CODE_ROOT>\Version_unified\RCWA_custom")
import scd_c_grating as M  # noqa: E402

PASS_NAME = sys.argv[1]
CORE = int(sys.argv[2])
assert PASS_NAME in ("TM_pass1", "TM_pass2") and CORE in (14, 15)
_vm = psutil.virtual_memory()
print("RAM used: %.1f%% (available %.2f GB); affinity %s" % (_vm.percent, _vm.available / 1e9,
                                                                 psutil.Process().cpu_affinity()), flush=True)
_lim = float(os.environ.get("SIM_MAX_RAM_PCT", "0"))   # opt-in RAM ceiling
assert _lim <= 0 or _vm.percent < _lim

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_li")
os.makedirs(OUT_DIR, exist_ok=True)

LAM_UM = 1.55
NG_SWEEP = 257                                   # was 161 (Laurent)
W0_UM = 100.0
NANG = 25
DKX_MAX = 5.0 / W0_UM
MS = tuple(range(-4, 5))

LI_JSON = (r"<DATA_ROOT>\Result\261009_scdc_carli\261009_sim768_carli_525f74f3_liy"
           r"\scdc_extract_H400_300.json")
TM_JSON = (r"<DATA_ROOT>\Result\260924_scdc_fin5"
           r"\260924_sim359_c2f90E_f100\scdc_extract_H400_300.json")
TM_DENSITY = (r"<DATA_ROOT>\Result\260924_scdc_fin5"
              r"\260924_sim359_c2f90E_f100\density_scdc_extract_H400_300.npy")
for p in (LI_JSON, TM_JSON, TM_DENSITY):
    if not os.path.isfile(p):
        print("NOT FOUND: %s" % p)
        sys.exit(1)
md5 = hashlib.md5(open(TM_DENSITY, "rb").read()).hexdigest()
assert md5 == "525f74f339190e5935b484fc583b0498", md5

with open(TM_JSON, "r", encoding="utf-8") as f:
    tm_meta = json.load(f)
with open(LI_JSON, "r", encoding="utf-8") as f:
    li_meta = json.load(f)
assert li_meta["rcwa_rule"] == "liy" and li_meta["rcwa_ff"] == 1 and li_meta["nG_eval"] == NG_SWEEP
assert tm_meta["out_l3_nm"] == li_meta["out_l3_nm"]
assert tm_meta["out_l3_r_list_eval"].replace(" ", "") == li_meta["out_l3_r_list_eval"].replace(" ", ""), (
    tm_meta["out_l3_r_list_eval"], li_meta["out_l3_r_list_eval"])

M.set_rcwa("liy", 1)                             # <-- change 1
print("RCWA_RULE", M.RCWA_RULE, "RCWA_FF", M.RCWA_FF, "NG_SWEEP", NG_SWEEP, flush=True)
M.set_period(1315.39)
M.set_substrate(1.444)
M.set_out_spacer(465.0, 1.444)
M.LAMS = (LAM_UM,)

eps_b = (M.eps_of("a-Si"), M.eps_of("a-Si"))
eps_f1 = M.eps_of("SiO2")
eps_f2 = M.eps_of("SiO2")
n_sub = M.N_SUB
k0 = 2.0 * np.pi / LAM_UM
G = 2.0 * np.pi / M.PERIOD
THETA_SUB_DEG = float(np.degrees(M.THETA_SUB))

tm_dens = np.load(TM_DENSITY)

out_l3_nm = tm_meta["out_l3_nm"]
r_list_eval_raw = tm_meta["out_l3_r_list_eval"]


def solve_terms(raw, h3_nm, lam_um, n_min):
    ns, pairs = [], []
    for tok in raw.replace(",", ";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        r_s, arg_s = tok.split(":")
        rt = complex(float(r_s) * np.exp(1j * np.radians(float(arg_s))))
        nb, tb, n3, res = M.solve_2layer_for_r(rt, h3_nm, lam_um, n_min=n_min)
        ns.append(n3)
        pairs.append((nb, tb))
        print("  token %-14s -> n_b=%.6f t_b=%.3f n3=%.6f residual=%.3e"
              % (tok, nb, tb, n3, res), flush=True)
    return ns, pairs


print("solving 5-token termination realisation ...", flush=True)
ns, pairs = solve_terms(r_list_eval_raw, out_l3_nm, LAM_UM, n_min=1.178358)
M.set_out_l3_list(out_l3_nm, ns)
M.set_out_l3b_list(pairs)
M._set_termination(0, ns[0])
DESIGN_N_TAG = M._n_tag(ns[0])
assert DESIGN_N_TAG == "n1708", DESIGN_N_TAG
OUT_L3B_NM = M.OUT_L3B_UM * 1e3
OUT_L3B_N = float(np.sqrt(M.OUT_L3B_EPS).real)
OUT_L3_SLAB_NM = M.OUT_L3_UM * 1e3
OUT_L3_SLAB_N = float(np.sqrt(M.OUT_L3_EPS).real)
print("FIXED design termination = %s  OUT_L3B(nm,n)=(%.3f,%.6f)  "
      "OUT_L3(nm,n)=(%.3f,%.6f) -- held fixed for the entire sweep"
      % (DESIGN_N_TAG, OUT_L3B_NM, OUT_L3B_N, OUT_L3_SLAB_NM, OUT_L3_SLAB_N),
      flush=True)

PASSES = {
    "TM_pass1": (tm_dens[0], tm_dens[1], "TM", THETA_SUB_DEG),
    "TM_pass2": (tm_dens[0], tm_dens[1], "TM", -THETA_SUB_DEG),
}
dkx = np.linspace(-DKX_MAX, DKX_MAX, NANG)

name = PASS_NAME
r1, r2, pol, theta_design_deg = PASSES[name]
print("=== %s  pol=%s theta_design=%.6f deg ===" % (name, pol, theta_design_deg), flush=True)
t_start = time.time()
kx0 = n_sub * k0 * np.sin(np.radians(theta_design_deg))
samples = []
for j, dk in enumerate(dkx):
    kx_i = kx0 + dk
    sin_i = kx_i / (n_sub * k0)
    if abs(sin_i) >= 1.0:
        print("  sample %d dkx=%.5f -> |sin|>=1, skipped" % (j, dk), flush=True)
        continue
    theta_i_deg = float(np.degrees(np.arcsin(sin_i)))
    ob, R, T = M._solve(r1, r2, 0.400, 0.300, eps_b, eps_f1, eps_f2,
                        "sub", pol, np.radians(theta_i_deg), LAM_UM, NG_SWEEP)
    Rp = M._pick(ob, np.real(R), MS)
    Tp = M._pick(ob, np.real(T), MS)
    aR, aT, bR, bT = M._args(ob, pol, MS, with_abs=True)
    samples.append(dict(
        j=j, dkx=float(dk), theta_i_deg=theta_i_deg, kx_i=float(kx_i),
        R={str(m): float(Rp[m]) for m in MS},
        T={str(m): float(Tp[m]) for m in MS},
        argR={str(m): aR[m] for m in MS},
        argT={str(m): aT[m] for m in MS},
    ))
    print("  sample %2d/%2d theta_i=%.5f deg done (%.1f s elapsed, RSS %.2f GB)"
          % (j, NANG, theta_i_deg, time.time() - t_start, psutil.Process().memory_info().rss / 1e9), flush=True)
print("  %s: %d/%d angular samples solved" % (name, len(samples), NANG), flush=True)

M.set_out_l3_list(0.0, ())
M.set_out_l3b_list(())

out = dict(
    meta=dict(lam_um=LAM_UM, w0_um=W0_UM, nang=NANG, dkx_max=DKX_MAX,
              ng_sweep=NG_SWEEP, n_sub=n_sub, period_um=M.PERIOD,
              G_per_um=G, theta_sub_deg=THETA_SUB_DEG,
              out_spacer_nm=465.0, n_out_spacer=1.444,
              H1_nm=400.0, H2_nm=300.0,
              out_l3_nm=out_l3_nm, design_n_tag=DESIGN_N_TAG,
              design_token="0.16:172",
              out_l3b_nm=OUT_L3B_NM, out_l3b_n=OUT_L3B_N,
              out_l3_slab_nm=OUT_L3_SLAB_NM, out_l3_slab_n=OUT_L3_SLAB_N,
              rcwa_rule=M.RCWA_RULE, rcwa_ff=M.RCWA_FF,
              tm_density_source=TM_DENSITY, tm_density_md5=md5),
    passes={name: dict(theta_design_deg=theta_design_deg, pol=pol, kx0=float(kx0), samples=samples)},
)
out_path = os.path.join(OUT_DIR, "beam_sweep_raw_li_%s.json" % name)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f)
print("\nWrote %s   total elapsed %.1f s" % (out_path, time.time() - t_start), flush=True)
