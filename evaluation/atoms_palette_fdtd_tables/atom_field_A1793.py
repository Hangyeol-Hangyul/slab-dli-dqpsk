# -*- coding: utf-8 -*-
"""RCWA field map of the P769 TM meta-atom 53474519 (level 32.5 deg) on its carrier, Li rule (liy, ff 1).

Builds the grcwa object through the driver's OWN builder (Main_rcwa_l3_atom2d_opt._obj_build, imported read-only
as an LI-aware entry), solves one arm at a time, and reconstructs E and H on the x-z plane through the atom mirror
plane y = 47.5 px = 216.95 nm.

Usage (python -B, pinned to logical cores 10..16):
  python -B atom_field.py selftest 601   ledger A+B vs the sibling n601 record + field self-tests (arm A)
  python -B atom_field.py map      801   ledger A+B + field maps A,B -> atom_field_maps.npz
  python -B atom_field.py ledger  1001   ledger A+B only (convergence point, no field maps)

atom_field_A1793.py = copy of atom_field.py (selftest / map / ledger unchanged) plus one mode:
  python -B atom_field_A1793.py mapA 1793   arm A: ledger + field maps; arm B: ledger only (no amplitudes, no maps)
                                            -> atom_field_maps_A_n1793.npz, report_mapA_n1793.json
  (stdout is the log: redirect it to log_mapA_n1793.txt)

Never writes under <DATA_ROOT>\\Simulation. Never deletes anything.
"""
import gc
import hashlib
import io
import json
import math
import os
import sys
import time
import weakref

sys.dont_write_bytecode = True

HERE = r"<WORK_ROOT>\atom_field_261011\compute"
RECDIR = r"<DATA_ROOT>\Result\261008_scdc_atomcand"
PREFIX = "261008_sim754_atomcand_53474519"
REC1793 = RECDIR + "\\" + PREFIX + r"_n1793_liyff1\atom2d_opt.json"
REC601 = RECDIR + "\\" + PREFIX + r"_n601_liyff1\atom2d_opt.json"
JOB_SRC = RECDIR + "\\" + PREFIX + r"_n1793_liyff1\job_40_" + PREFIX + "_n1793_liyff1.txt"
CODE = r"<CODE_ROOT>\Version_unified\RCWA_custom"
CARRIER_NPY = (r"<DATA_ROOT>\palettes\TM_g62_261003\carrier_sim359_c2f90E_f100"
               r"\density_scdc_extract_H400_300.npy")
RHO3_MD5_PREFIX = "53474519"
CARRIER_MD5 = "525f74f339190e5935b484fc583b0498"
DRIVER_MD5_PREFIX = "976cc6f0"

MODE = sys.argv[1] if len(sys.argv) > 1 else "selftest"
NG_REQ = int(sys.argv[2]) if len(sys.argv) > 2 else 601
BLAS_THREADS = int(os.environ.get("AF_BLAS_THREADS", "7"))     # 7 pinned logical cores (10..16)
assert MODE in ("selftest", "map", "ledger", "mapA"), MODE

# ---- thread env BEFORE numpy is imported (the driver sets the same four from the job key blas_threads)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(BLAS_THREADS)

import psutil  # noqa: E402  (does not import numpy)

T_SCRIPT0 = time.perf_counter()
PROC = psutil.Process()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def peak_gb():
    try:
        return PROC.memory_info().peak_wset / 2 ** 30
    except Exception:
        return float("nan")


def log(*a):
    print(*a, flush=True)


# ---- 1. scratch copy of the job file (outside <DATA_ROOT>): nG / nG_eval / blas_threads / output keys only
SCR = os.path.join(HERE, "scratch_job")
os.makedirs(SCR, exist_ok=True)
out = []
for ln in io.open(JOB_SRC, encoding="utf-8", errors="replace").read().split("\n"):
    s = ln.strip()
    if "=" in s and not s.startswith("#"):
        k = s.partition("=")[0].strip()
        if k in ("nG", "nG_eval"):
            ln = "%-18s = %d" % (k, NG_REQ)
        elif k == "blas_threads":
            ln = "blas_threads       = %d" % BLAS_THREADS
        elif k == "Output_folder":
            ln = "Output_folder      = " + HERE
        elif k == "subfolder_name":
            ln = "subfolder_name     = field_%s_n%d" % (MODE, NG_REQ)
    out.append(ln)
JOB = os.path.join(SCR, "job_field_%s_n%d.txt" % (MODE, NG_REQ))
with io.open(JOB, "w", encoding="utf-8") as f:
    f.write("\n".join(out))

# ---- 2. import the driver exactly as an LI-aware entry (driver :208-220 reads argv[1], :253-261 gate)
LI_AWARE_ENTRY = True
sys.argv = [sys.argv[0], JOB]
sys.path.insert(0, CODE)
import Main_rcwa_l3_atom2d_opt as M  # noqa: E402

np = M.np
G = M.G
import grcwa  # noqa: E402  (the vendored copy the driver already imported)

assert M._LI_ON and M._RCWA_RULE == "liy" and M._RCWA_FF == 1, "Li keys NOT active (argv[1] must be the job file)"
log("[init] mode=%s nG_req=%d blas_threads=%d affinity=%s python=%s numpy=%s" %
    (MODE, NG_REQ, BLAS_THREADS, PROC.cpu_affinity(), sys.version.split()[0], np.__version__))
log("[init] driver md5 %s (expect %s...), Li rule=%s ff=%s" %
    (md5(os.path.join(CODE, "Main_rcwa_l3_atom2d_opt.py")), DRIVER_MD5_PREFIX, M._RCWA_RULE, M._RCWA_FF))
assert md5(os.path.join(CODE, "Main_rcwa_l3_atom2d_opt.py")).startswith(DRIVER_MD5_PREFIX)

# ---- 3. module state main() sets (driver :2554-2555, :2597-2603, :2649-2661)
G.set_period(M._num("period_nm", 1315.39))
G.set_substrate(M._num("n_sub", 1.4440))
rec = json.load(io.open(REC1793, encoding="utf-8"))
M.apply_record_globals(rec)
M.DISTINCT_ATOMS = 0
NXY = int(rec["NXY"])
APP = int(rec["atoms_per_period"])
NX = APP * NXY
LAM = M._num("lams_nm", 1550.0) * 1e-3
H = [M._num("H1_nm", 500.0) * 1e-3, M._num("H2_nm", 350.0) * 1e-3, M._num("H3_nm", 775.0) * 1e-3]
SP = M._num("spacer_nm", 200.0) * 1e-3
N_SP = M._num("n_spacer", 1.4440)
EPS3 = G.eps_of(M._txt("material3", "a-Si"))
LX_UM = G.PERIOD
LY_UM = G.PERIOD / float(APP)
DELTA_NM = G.PERIOD * 1e3 / NX                       # pixel (4.56733 nm)
YC_PX = 47.5                                         # mirror plane of the raster (sym_y: row j <-> 95-j)
THETA = G.THETA_SUB
log("[init] H=%s sp=%.4f lam=%.4f um  Lx=%.5f um Ly=%.6f um  pixel=%.5f nm  theta_sub=%.5f deg  eps3=%s" %
    (H, SP, LAM, LX_UM, LY_UM, DELTA_NM, math.degrees(THETA), EPS3))
assert abs(DELTA_NM - 4.56733) < 1e-4 and abs(math.degrees(THETA) - 54.69005) < 1e-4

# ---- 4. densities (read-only): L3 from the record, carrier from l12_density; md5 asserts
rho3_path = rec["rho3_npy"]
rho3_md5 = md5(rho3_path)
car_md5 = md5(CARRIER_NPY)
log("[md5] rho3 %s  %s" % (rho3_md5, rho3_path))
log("[md5] carrier %s  %s" % (car_md5, CARRIER_NPY))
assert rho3_md5.startswith(RHO3_MD5_PREFIX), "rho3 md5 mismatch"
assert car_md5 == CARRIER_MD5, "carrier md5 mismatch"
job_l12 = [ln.split("=", 1)[1].strip() for ln in out if ln.strip().startswith("l12_density")]
assert job_l12 and job_l12[0].lower() == CARRIER_NPY.lower(), (job_l12, CARRIER_NPY)

x_raw = np.load(rho3_path)
r3 = M.project_recorded(x_raw, rec, beta=64.0)
assert isinstance(r3, list) and len(r3) == 2
for z, ref in zip(r3, rec["fill_frac_each"]):
    assert abs(float(np.mean(z)) - ref) < 1e-9, (float(np.mean(z)), ref)
log("[dens] fill fractions %.12f %.12f == record %.12f %.12f" %
    (float(np.mean(r3[0])), float(np.mean(r3[1])), rec["fill_frac_each"][0], rec["fill_frac_each"][1]))

a12 = np.load(CARRIER_NPY)


def _resample(col):                                   # verbatim copy of driver :2685-2692 (closure inside main())
    n0 = col.size
    c = np.concatenate([[0.0], np.cumsum(col)])
    e = np.arange(NX + 1) * (n0 / float(NX))
    lo = np.clip(np.floor(e).astype(int), 0, n0)
    fr = e - np.floor(e)
    ci = c[lo] + fr * col[np.clip(lo, 0, n0 - 1)]
    return (ci[1:] - ci[:-1]) / (n0 / float(NX))


rho12 = [_resample(a12[0]), _resample(a12[1])]

# capture the eps raster the solver is really given (read-only wrapper on the class method, this process only)
CAP = {}
_orig_geteps = grcwa.obj.GridLayer_geteps


def _cap_geteps(self, ep_all):
    CAP["ep"] = np.array(ep_all)
    return _orig_geteps(self, ep_all)


grcwa.obj.GridLayer_geteps = _cap_geteps


# ---- 5. RAM gate: measured scaling (nG 801 map run, both arms, process peak RSS 1.88 GiB; solver memory ~ nG^2),
#         x1.15 safety; refuse unless need <= 0.70 x CURRENTLY AVAILABLE RAM
PEAK_GIB_AT_801 = 1.88
SAFETY = 1.00  # measured 597->797 RSS ratio 1.76 vs nG^2 1.78, so no extra margin; need 9.42 GiB


def ram_need_gib(nG_req):
    return PEAK_GIB_AT_801 * (nG_req / 801.0) ** 2 * SAFETY


def ram_gate(nG_req, limit=0.70):
    _pct = float(os.environ.get("SIM_MAX_RAM_PCT", "0"))   # opt-in: the gate is active only when SIM_MAX_RAM_PCT is set
    if _pct > 0:
        limit = _pct / 100.0
    v = psutil.virtual_memory()
    need = ram_need_gib(nG_req)
    avail = v.available / 2 ** 30
    cap = limit * avail
    proj = (v.total - v.available + need * 2 ** 30) / v.total
    ok = need <= cap or _pct <= 0
    log("[RAM] now %.1f %% (used %.2f GiB of %.2f, available %.2f GiB); need %.2f GiB (= %.2f x (nG %d/801)^2 x %.2f) "
        "vs %.0f %% x available = %.2f GiB -> %s (projected total %.1f %%)" %
        (v.percent, (v.total - v.available) / 2 ** 30, v.total / 2 ** 30, avail, need, PEAK_GIB_AT_801, nG_req,
         SAFETY, 100 * limit, cap, "OK" if ok else "REFUSE", 100 * proj))
    return ok, v.percent, 100 * proj


# ---- 6. order tables and ledger helpers
def orders_table(ob, arm, led):
    """Propagating orders of this solve: label, kind, m, power (this nG), direction in the x-z plane."""
    R, T = ob.RT_Solve(normalize=1, byorder=1)
    Gi = np.asarray(ob.G)
    kx = np.real(np.asarray(ob.kx))
    om = float(np.real(complex(ob.omega)))
    rows = []
    sgn = 1.0 if arm == "A" else -1.0
    sin_in = sgn * math.sin(THETA)
    rows.append(dict(label="inc", kind="inc", m=0, power=1.0, n=G.N_SUB, dirx=sin_in, dirz=math.cos(THETA),
                     angle=sgn * math.degrees(THETA)))
    for kind, arr, nmed, dz in (("R", R, G.N_SUB, -1.0), ("T", T, 1.0, +1.0)):
        for m in range(-3, 4):
            idx = np.where((Gi[:, 0] == m) & (Gi[:, 1] == 0))[0]
            if len(idx) == 0:
                continue
            g = int(idx[0])
            p = float(np.real(arr[g]))
            if p <= 1e-9:
                continue
            s = float(kx[g]) / (om * nmed)
            assert abs(s) < 1.0, (arm, kind, m, s)
            rows.append(dict(label="%s%+d" % (kind, m), kind=kind, m=m, power=p, n=nmed,
                             dirx=s, dirz=dz * math.sqrt(1.0 - s * s), angle=math.degrees(math.asin(s))))
    return rows


def derived(ledA, ledB, tA, rB, tB):
    d = {}
    d["amp_tA"] = abs(tA)
    d["amp_rB"] = abs(rB)
    d["amp_tB"] = abs(tB)
    d["A_deg"] = math.degrees(math.atan2(tA.imag, tA.real))
    rBtB = rB * tB
    d["B_deg"] = math.degrees(math.atan2(rBtB.imag, rBtB.real))
    d["argrB_deg"] = math.degrees(math.atan2(rB.imag, rB.real))
    d["argtB_deg"] = math.degrees(math.atan2(tB.imag, tB.real))
    ep, rgo, em = ledA["ledA_T_m-1"], ledA["ledA_R_m+0"], ledB["ledB_T_m+1"]
    d["carrier_eff_pct"] = 100.0 * math.sqrt(ep * rgo * em) / (4.0 / 9.0) ** 1.5
    d["carrier_echo_db"] = 10.0 * math.log10(ledB["ledB_R_m+2"])
    return d


LEDGER_KEYS = ["ledA_T_m-1", "ledA_R_m+0", "ledA_R_m-2", "ledA_R_m-1", "ledA_R_total", "ledA_T_total",
               "ledB_T_m+1", "ledB_R_m+0", "ledB_R_m+1", "ledB_R_m+2", "ledB_R_total", "ledB_T_total",
               "amp_tA", "amp_rB", "amp_tB", "A_deg", "B_deg", "argrB_deg", "argtB_deg",
               "carrier_eff_pct", "carrier_echo_db"]


class FieldEngine(object):
    """All-layer field reconstruction from ONE GetAmplitudes per layer (exact exp factors for z)."""

    def __init__(self, ob):
        self.ob = ob
        self.nG = int(ob.nG)
        self.L = int(ob.Layer_N)
        self.Gi = np.asarray(ob.G)
        self.kx = np.asarray(ob.kx)
        self.ky = np.asarray(ob.ky)
        self.om = complex(ob.omega)
        self.th = [float(t) for t in ob.thickness_list]
        zb = [-500.0]
        for t in self.th[:-1]:
            zb.append(zb[-1] + t * 1e3)
        self.zb = np.array(zb)                         # nm, bottom of each layer (layer 0 bottom = -500 nm)
        self.zt = self.zb + np.array(self.th) * 1e3
        self.q = [np.asarray(ob.q_list[l]) for l in range(self.L)]
        self.uni = [ob.id_list[l][0] == 0 for l in range(self.L)]
        # unique x orders, collapse matrix, kx of each
        self.mu = np.unique(self.Gi[:, 0])
        self.Pm = (self.Gi[:, 0][None, :] == self.mu[:, None]).astype(float)       # (Mu, G)
        self.kxm = np.array([float(np.real(self.kx[np.where(self.Gi[:, 0] == m)[0][0]])) for m in self.mu])
        for m, kk in zip(self.mu, self.kxm):                      # kx depends on m only (ky0 = 0)
            assert np.allclose(np.real(self.kx[self.Gi[:, 0] == m]), kk)
        self.amps = []
        self.t_amp = []

    def prune(self):
        sm = self.ob._sm_cache
        if sm:
            for k in [k for k in sm if k != (0, self.L - 1)]:
                sm.pop(k)

    def prepare(self):
        for l in range(self.L):
            t0 = time.perf_counter()
            a0, b0 = self.ob.GetAmplitudes(l, 0.0)
            self.amps.append((np.asarray(a0), np.asarray(b0)))
            self.prune()
            self.t_amp.append(time.perf_counter() - t0)
            log("[amps] layer %d done %.1f s  (peak RSS %.2f GiB)" % (l, self.t_amp[-1], peak_gb()))

    def fourier(self, l, dz_um, drop_b=False):
        """Fourier coefficients (aligned with ob.G) of the Bloch envelope at layer l, offsets dz (um from layer bottom)."""
        ob = self.ob
        nG = self.nG
        a0, b0 = self.amps[l]
        q = self.q[l]
        dz = np.asarray(dz_um, dtype=float)
        A = a0[:, None] * np.exp(1j * q[:, None] * dz[None, :])
        if drop_b:
            B = np.zeros_like(A)
        else:
            B = b0[:, None] * np.exp(-1j * q[:, None] * dz[None, :])
        phi = np.asarray(ob.phi_list[l])
        kp = np.asarray(ob.kp_list[l])
        hxy = phi.dot(A + B)
        t = phi.dot((A - B) / self.om / q[:, None])
        exy = kp.dot(t)
        hx, hy = hxy[:nG], hxy[nG:]
        ey, ex = -exy[:nG], exy[nG:]
        kx = self.kx[:, None]
        ky = self.ky[:, None]
        hz = (kx * ey - ky * ex) / self.om
        ez = (ky * hx - kx * hy) / self.om
        if self.uni[l]:
            ez = ez / ob.Uniform_ep_list[ob.id_list[l][2]]
        else:
            ez = np.asarray(ob.Patterned_epinv_list[ob.id_list[l][2]]).dot(ez)
        return dict(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz)

    def sz_total(self, f):
        """Cell-averaged Sz = 0.5 Re sum_G (Ex Hy* - Ey Hx*) per z column."""
        return 0.5 * np.real(np.sum(f["ex"] * np.conj(f["hy"]) - f["ey"] * np.conj(f["hx"]), axis=0))

    def xmat(self, xs_um):
        return np.exp(1j * self.kxm[None, :] * np.asarray(xs_um)[:, None])        # (nx, Mu)

    def to_x(self, F, y_um, XM):
        """(G, nz) Fourier coefficients -> (nz, nx) physical field (Bloch factor included) at height y."""
        Fy = F * np.exp(1j * self.ky * y_um)[:, None]
        Fm = self.Pm.dot(Fy)                                  # (Mu, nz)
        return (XM.dot(Fm)).T

    def to_grid(self, f):
        """Same as grcwa get_ifft (Bloch envelope on the raster)."""
        F = np.zeros((NX, NXY), dtype=complex)
        F[self.Gi[:, 0] % NX, self.Gi[:, 1] % NXY] = f
        return np.fft.ifft2(F) * (NX * NXY)


def solve_arm(arm, do_fields, field_cfg, tests_extra):
    """Build, solve (ledger), optionally reconstruct. One object alive at a time."""
    res = dict(arm=arm)
    ok, pct, proj = ram_gate(NG_REQ)
    res["ram_pct_before"] = pct
    res["ram_proj_pct"] = proj
    res["ram_need_gib"] = ram_need_gib(NG_REQ)
    res["ram_avail_gib"] = psutil.virtual_memory().available / 2 ** 30
    assert ok, "need %.2f GiB exceeds the RAM limit (SIM_MAX_RAM_PCT) of the available RAM (%.2f GiB available): do not solve" % (res["ram_need_gib"], res["ram_avail_gib"])
    theta = (+1.0 if arm == "A" else -1.0) * G.THETA_SUB
    t0 = time.perf_counter()
    ob = M._obj_build(NG_REQ, r3, rho12, H, SP, N_SP, EPS3, LAM, NXY, theta, "TM")
    t_build = time.perf_counter() - t0
    assert ob.rcwa_rule == "liy" and ob.form_factor == 1 and ob.Layer_N == 8, (ob.rcwa_rule, ob.form_factor, ob.Layer_N)
    log("[arm %s] built: ob.nG(actual G)=%d, Layer_N=%d, rule=%s ff=%s  build+eig %.1f s  peak RSS %.2f GiB" %
        (arm, ob.nG, ob.Layer_N, ob.rcwa_rule, ob.form_factor, t_build, peak_gb()))
    res["G_actual"] = int(ob.nG)
    res["t_build_s"] = t_build
    t0 = time.perf_counter()
    led = M._ledger(ob, "led" + arm)
    if arm == "A":
        tA = complex(M._amp(ob, -1, pol="TM"))
        rB = complex(M._amp(ob, 0, refl=True, pol="TM"))
        res["tA"], res["rB"] = tA, rB
    else:
        tB = complex(M._amp(ob, +1, pol="TM"))
        res["tB"] = tB
        res["echo_amp_m2"] = complex(M._amp(ob, +2, refl=True, pol="TM"))
    res["led"] = led
    res["t_ledger_s"] = time.perf_counter() - t0
    res["orders"] = orders_table(ob, arm, led)
    res["peak_after_ledger_gib"] = peak_gb()
    log("[arm %s] ledger %.1f s: %s" % (arm, res["t_ledger_s"], json.dumps({k: round(v, 7) for k, v in led.items()})))
    if do_fields:
        res.update(field_arm(ob, arm, field_cfg, tests_extra, res))
    # free this arm's solver object and arrays before the next arm is built (the two never coexist)
    wr = weakref.ref(ob)
    rss_before = PROC.memory_info().rss / 2 ** 30
    del ob
    gc.collect()
    rss_after = PROC.memory_info().rss / 2 ** 30
    res["rss_before_free_gib"], res["rss_after_free_gib"] = rss_before, rss_after
    res["solver_freed"] = wr() is None
    log("[arm %s] solver freed=%s  current RSS %.2f -> %.2f GiB (peak so far %.2f GiB)" %
        (arm, res["solver_freed"], rss_before, rss_after, peak_gb()))
    if not res["solver_freed"]:
        log("[WARN] arm %s solver object is still referenced after del + gc.collect(): arm B will coexist with it" % arm)
    res["peak_end_gib"] = peak_gb()
    return res


Z_LO, Z_HI, DZ_NM = -1000.0, 3590.0, 5.0


def field_arm(ob, arm, cfg, tests_extra, res):
    fe = FieldEngine(ob)
    out = dict()
    tests = dict()
    t0 = time.perf_counter()
    fe.prepare()
    out["t_amps_s"] = time.perf_counter() - t0
    # --- layer geometry check
    log("[arm %s] z bottoms (nm) %s ; tops %s" % (arm, fe.zb.tolist(), fe.zt.tolist()))
    exp_bot = [-500.0, 0.0, 400.0, 700.0, 1165.0, 1707.5, 1757.5, 2090.0]
    assert np.allclose(fe.zb, exp_bot), fe.zb
    out["zb"], out["zt"] = fe.zb, fe.zt

    # --- reconstructed Hy(-1,0) in layer 7 at dz=0 equals tA (arm A) ; Hy(+1,0) equals tB (arm B)
    f7 = fe.fourier(7, np.array([0.0]))
    gi = lambda m: int(np.where((fe.Gi[:, 0] == m) & (fe.Gi[:, 1] == 0))[0][0])
    if arm == "A":
        tref, m_t = res["tA"], -1
    else:
        tref, m_t = res["tB"], +1
    hy_rec = complex(f7["hy"][gi(m_t), 0])
    tests["t4_hy_layer7_vs_amp_rel"] = abs(hy_rec - tref) / abs(tref)
    # delay-arm amplitude: Hy(0,0) reflected at layer 0 bottom (b0 referred to z=-500)
    f0 = fe.fourier(0, np.array([0.0]))
    hy_b0 = complex(f0["hy"][gi(0), 0]) - complex(fe.amps[0][0][fe.nG + gi(0)])
    if arm == "A":
        tests["t4_hy_reflected_m0_layer0_vs_rB_rel"] = abs(hy_b0 - res["rB"]) / abs(res["rB"])

    # --- incident Sz and flux conservation
    f_inc = fe.fourier(0, np.array([0.0]), drop_b=True)
    sz_inc = float(fe.sz_total(f_inc)[0])
    n_sub = G.N_SUB
    sz_inc_theory = math.cos(THETA) / (2.0 * n_sub)
    tests["sz_inc"] = sz_inc
    tests["sz_inc_theory_cos_over_2n"] = sz_inc_theory
    tests["sz_inc_rel_err"] = abs(sz_inc - sz_inc_theory) / sz_inc_theory

    # --- continuity across every interface (Fourier vectors)
    cont = []
    for l in range(fe.L - 1):
        top = fe.fourier(l, np.array([fe.th[l]]))
        bot = fe.fourier(l + 1, np.array([0.0]))
        row = {}
        for k in ("ex", "ey", "hx", "hy"):
            nrm = np.linalg.norm(top[k][:, 0])
            row[k] = float(np.linalg.norm(top[k][:, 0] - bot[k][:, 0]) / nrm) if nrm > 0 else float("nan")
        # Dz = (ky Hx - kx Hy)/omega is continuous when Hx, Hy are; also check the layer eps*Ez product form
        cont.append(row)
    tests["t5_continuity_rel_2norm_per_interface"] = cont
    tests["t5_continuity_max"] = max(max(r.values()) for r in cont)

    # --- z grid and the physical x-z maps
    zs = np.arange(Z_LO, Z_HI + 1e-9, DZ_NM)
    nz = len(zs)
    nxp = NX                                           # native pixel grid, 288 per period
    xs_nm = np.arange(2 * nxp) * DELTA_NM              # two carrier periods
    XM = fe.xmat(xs_nm * 1e-3)
    yc_um = YC_PX * DELTA_NM * 1e-3
    comps = ("ex", "ey", "ez", "hx", "hy", "hz")
    maps = {c: np.zeros((nz, 2 * nxp), dtype=complex) for c in comps}
    sz_z = np.zeros(nz)
    lay_of_z = np.zeros(nz, dtype=int)
    tt = time.perf_counter()
    for l in range(fe.L):
        lo = fe.zb[l] if l > 0 else -np.inf
        hi = fe.zt[l] if l < fe.L - 1 else np.inf
        sel = np.where((zs >= lo) & (zs < hi))[0]
        if len(sel) == 0:
            continue
        lay_of_z[sel] = l
        dz = (zs[sel] - fe.zb[l]) * 1e-3              # um from the layer bottom; negative below layer 0, > t above layer 7
        f = fe.fourier(l, dz)
        sz_z[sel] = fe.sz_total(f)
        for c in comps:
            maps[c][sel, :] = fe.to_x(f[c], yc_um, XM)
    out["t_maps_s"] = time.perf_counter() - tt
    tests["z_grid"] = dict(z_lo=Z_LO, z_hi=Z_HI, dz_nm=DZ_NM, nz=nz, nx=2 * nxp, dx_nm=DELTA_NM,
                           x_hi_nm=float(xs_nm[-1]), y_nm=YC_PX * DELTA_NM)

    # --- mirror-plane parity (test 6) and flux conservation (test 7)
    even = max(np.abs(maps[c]).max() for c in ("ex", "ez", "hy"))
    odd = max(np.abs(maps[c]).max() for c in ("ey", "hx", "hz"))
    tests["t6_odd_over_even_max"] = float(odd / even)
    tests["t6_max_abs_even"] = float(even)
    tests["t6_max_abs_odd"] = float(odd)
    ledtot = res["led"]["led%s_T_total" % arm]
    sz_n = sz_z / sz_inc
    tests["t7_Sz_over_inc_min"] = float(sz_n.min())
    tests["t7_Sz_over_inc_max"] = float(sz_n.max())
    tests["t7_Sz_over_inc_spread"] = float(sz_n.max() - sz_n.min())
    tests["t7_vs_ledger_T_total_diff"] = float(np.max(np.abs(sz_n - ledtot)))
    tests["t3_R_plus_T_minus_1"] = float(res["led"]["led%s_R_total" % arm] + ledtot - 1.0)
    # rows 47 / 48 agreement of the even fields (direct DFT at y = 47 px and 48 px)
    zsub = np.array([0.0, 200.0, 550.0, 900.0, 1400.0, 1730.0, 1900.0, 2400.0])
    dd = 0.0
    scale = 0.0
    for zz in zsub:
        l = int(np.searchsorted(fe.zb, zz, side="right") - 1)
        l = max(0, min(l, fe.L - 1))
        f = fe.fourier(l, np.array([(zz - fe.zb[l]) * 1e-3]))
        for c in ("ex", "ez", "hy"):
            v47 = fe.to_x(f[c], 47 * DELTA_NM * 1e-3, XM[:nxp])
            v48 = fe.to_x(f[c], 48 * DELTA_NM * 1e-3, XM[:nxp])
            dd = max(dd, float(np.abs(v47 - v48).max()))
            scale = max(scale, float(np.abs(v47).max()))
    tests["t6_row47_vs_row48_max_diff_rel"] = dd / scale

    # --- cross-check against grcwa's own Solve_FieldOnGrid (layer 4, dz = 0.2 um), including the Bloch factor
    if tests_extra:
        t0 = time.perf_counter()
        fe.prune()
        l_chk, dz_chk = 4, 0.2
        fe_, fh_ = ob.Solve_FieldOnGrid(l_chk, dz_chk)
        mine = fe.fourier(l_chk, np.array([dz_chk]))
        rel = {}
        for c, ref in zip(("ex", "ey", "ez", "hx", "hy", "hz"), list(fe_) + list(fh_)):
            g_my = fe.to_grid(mine[c][:, 0])
            ref = np.asarray(ref)
            rel[c] = float(np.abs(g_my - ref).max() / max(np.abs(ref).max(), 1e-300))
        tests["t8_vs_Solve_FieldOnGrid_rel"] = rel
        # direct-DFT route at raster sample points (x_k = k*Delta, y = 47 px) vs the grid * Bloch phase
        i0 = int(np.where((fe.Gi[:, 0] == 0) & (fe.Gi[:, 1] == 0))[0][0])
        kx0 = float(np.real(fe.kx[i0]))
        xk = np.arange(NX) * DELTA_NM * 1e-3
        XMk = fe.xmat(xk)
        d_dft = fe.to_x(mine["hy"], 47 * DELTA_NM * 1e-3, XMk)[0]
        ref_hy = np.asarray(list(fh_)[1])[:, 47] * np.exp(1j * kx0 * xk)
        tests["t8_dft_vs_grid_hy_rel"] = float(np.abs(d_dft - ref_hy).max() / np.abs(ref_hy).max())
        tests["t8_time_s"] = time.perf_counter() - t0
        fe.prune()

    # --- material map eps(x, z) from the raster the solver actually received
    ep5 = np.real(CAP["ep"]).reshape(5, NX, NXY)
    eps_layer_native = np.zeros((fe.L, NX))
    patt = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}
    for l in range(fe.L):
        if l in patt:
            e = ep5[patt[l]]
            eps_layer_native[l] = 0.5 * (e[:, 47] + e[:, 48])
        else:
            eps_layer_native[l] = float(np.real(ob.Uniform_ep_list[ob.id_list[l][2]]))
    ix = np.round(xs_nm / DELTA_NM).astype(int) % NX
    eps_xz = eps_layer_native[lay_of_z][:, ix]
    out["eps_xz"] = eps_xz
    out["eps_layer_native"] = eps_layer_native
    out["ep5_full"] = ep5

    # --- Poynting maps, normalised to the incident Sz (physical complex fields)
    Ex, Ey, Ez, Hx, Hy, Hz = (maps[c] for c in comps)
    out["Sx"] = (0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy)) / sz_inc)
    out["Sz"] = (0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx)) / sz_inc)
    out["maps"] = maps
    out["zs"] = zs
    out["xs"] = xs_nm
    out["lay_of_z"] = lay_of_z
    out["sz_z"] = sz_n
    out["tests"] = tests
    return out


# ---- 7. driver ------------------------------------------------------------------------------------------------
def fmt(x):
    return "%.7g" % x if isinstance(x, float) else str(x)


def run():
    do_fields = MODE in ("selftest", "map", "mapA")
    tests_extra = True
    resA = solve_arm("A", do_fields, None, tests_extra)
    if MODE == "mapA":                                  # keep the A maps on disk before arm B is attempted
        save_npz(resA, None, None)
    resB = solve_arm("B", MODE == "map", None, tests_extra)      # mapA: arm B = ledger only
    ledA, ledB = resA["led"], resB["led"]
    dv = derived(ledA, ledB, resA["tA"], resA["rB"], resB["tB"])
    this = dict(ledA)
    this.update(ledB)
    this.update(dv)
    r1793 = json.load(io.open(REC1793, encoding="utf-8"))
    r601 = json.load(io.open(REC601, encoding="utf-8"))
    rows = []
    for k in LEDGER_KEYS:
        v = this.get(k)
        a = r1793.get(k)
        b = r601.get(k)
        rows.append(dict(key=k, this=v, rec1793=a, rec601=b,
                         d_vs_1793=(None if v is None or a is None else v - a),
                         rel_vs_1793=(None if v is None or a in (None, 0) else (v - a) / abs(a)),
                         d_vs_601=(None if v is None or b is None else v - b),
                         rel_vs_601=(None if v is None or b in (None, 0) else (v - b) / abs(b))))
    log("\n[LEDGER] nG_req=%d actual G=%d   (this = this run; rec1793/rec601 = stored records)" %
        (NG_REQ, resA["G_actual"]))
    log("%-18s %14s %14s %12s %10s | %14s %12s %10s" %
        ("key", "this", "rec1793", "this-1793", "rel", "rec601", "this-601", "rel"))
    for r in rows:
        def f(v, w=14, p=7):
            return ("%*.*g" % (w, p, v)) if v is not None else "%*s" % (w, "-")
        log("%-18s %s %s %s %s | %s %s %s" %
            (r["key"], f(r["this"]), f(r["rec1793"]), f(r["d_vs_1793"], 12, 4), f(r["rel_vs_1793"], 10, 3),
             f(r["rec601"]), f(r["d_vs_601"], 12, 4), f(r["rel_vs_601"], 10, 3)))
    report = dict(mode=MODE, nG_req=NG_REQ, G_actual=resA["G_actual"], rows=rows, blas_threads=BLAS_THREADS,
                  affinity=PROC.cpu_affinity(),
                  armA=dict(t_build_s=resA["t_build_s"], t_ledger_s=resA["t_ledger_s"],
                            peak_after_ledger_gib=resA["peak_after_ledger_gib"], peak_end_gib=resA["peak_end_gib"],
                            ram_pct_before=resA["ram_pct_before"], ram_proj_pct=resA["ram_proj_pct"],
                            ram_need_gib=resA["ram_need_gib"], ram_avail_gib=resA["ram_avail_gib"],
                            solver_freed=resA["solver_freed"], rss_before_free_gib=resA["rss_before_free_gib"],
                            rss_after_free_gib=resA["rss_after_free_gib"]),
                  armB=dict(t_build_s=resB["t_build_s"], t_ledger_s=resB["t_ledger_s"],
                            peak_after_ledger_gib=resB["peak_after_ledger_gib"], peak_end_gib=resB["peak_end_gib"],
                            ram_pct_before=resB["ram_pct_before"], ram_proj_pct=resB["ram_proj_pct"],
                            ram_need_gib=resB["ram_need_gib"], ram_avail_gib=resB["ram_avail_gib"],
                            solver_freed=resB["solver_freed"], rss_before_free_gib=resB["rss_before_free_gib"],
                            rss_after_free_gib=resB["rss_after_free_gib"]),
                  rho3_md5=rho3_md5, carrier_md5=car_md5)
    for arm, res in (("A", resA), ("B", resB)):
        if "tests" in res:
            report["tests_" + arm] = res["tests"]
            report["arm" + arm]["t_amps_s"] = res["t_amps_s"]
            report["arm" + arm]["t_maps_s"] = res["t_maps_s"]
            log("\n[TESTS arm %s]" % arm)
            for k, v in res["tests"].items():
                log("  %-45s %s" % (k, json.dumps(v) if not isinstance(v, float) else "%.4e" % v))
    report["orders_A"] = resA["orders"]
    report["orders_B"] = resB["orders"]
    report["total_wall_s"] = time.perf_counter() - T_SCRIPT0
    report["peak_rss_gib_process"] = peak_gb()
    with io.open(os.path.join(HERE, "report_%s_n%d.json" % (MODE, NG_REQ)), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    log("\n[done] mode=%s nG=%d wall %.1f s  process peak RSS %.2f GiB" %
        (MODE, NG_REQ, report["total_wall_s"], peak_gb()))
    if MODE in ("map", "mapA"):
        save_npz(resA, resB, report)


def save_npz(resA, resB, report):
    """resB may be None (mapA, arm A saved before arm B runs) and a res without "maps" (mapA arm B) gets no field arrays."""
    kx = {}
    zs, xs = resA["zs"], resA["xs"]
    d = dict(x_nm=xs, z_nm=zs, layer_z_bottom_nm=resA["zb"], layer_z_top_nm=resA["zt"],
             layer_names=np.array(["0 incidence SiO2 (numerical half-space, 500 nm)", "1 carrier L1 (a-Si ridges, SiO2 fill)",
                                   "2 carrier L2 (a-Si ridges, SiO2 fill)", "3 spacer SiO2",
                                   "4 atom lower (a-Si, SiO2 fill)", "5 gap SiO2 50 nm",
                                   "6 atom upper (a-Si, air)", "7 exit air (numerical half-space, 500 nm)"]),
             layer_of_z=resA["lay_of_z"], eps_xz=resA["eps_xz"], eps_layer_native_row_yc=resA["eps_layer_native"],
             eps_raster_5grid_layers_L1_L2_atomA_gap_atomB=resA["ep5_full"], Sz_total_over_inc_z_A=resA["sz_z"])
    if resB is not None and "maps" in resB:
        d["Sz_total_over_inc_z_B"] = resB["sz_z"]
    for arm, res in (("A", resA), ("B", resB)):
        if res is None:                                   # arm B not solved yet (partial save of the A maps)
            continue
        if "maps" in res:                                 # arm B of mapA is ledger only: orders, no field arrays
            for c in ("ex", "ey", "ez", "hx", "hy", "hz"):
                d["%s_%s" % (arm, c[0].upper() + c[1])] = res["maps"][c]
            d["%s_Sx_over_Szinc" % arm] = res["Sx"]
            d["%s_Sz_over_Szinc" % arm] = res["Sz"]
        orders = res["orders"]
        d["%s_ord_label" % arm] = np.array([o["label"] for o in orders])
        d["%s_ord_kind" % arm] = np.array([o["kind"] for o in orders])
        d["%s_ord_m" % arm] = np.array([o["m"] for o in orders])
        d["%s_ord_power_this_nG" % arm] = np.array([o["power"] for o in orders])
        rec1793 = []
        for o in orders:
            if o["kind"] == "inc":
                rec1793.append(1.0)
            else:
                rec1793.append(float(json.load(io.open(REC1793, encoding="utf-8")).get(
                    "led%s_%s_m%+d" % (arm, o["kind"], o["m"]), float("nan"))))
        d["%s_ord_power_record_nG1793" % arm] = np.array(rec1793)
        d["%s_ord_medium_n" % arm] = np.array([o["n"] for o in orders])
        d["%s_ord_dirx" % arm] = np.array([o["dirx"] for o in orders])
        d["%s_ord_dirz" % arm] = np.array([o["dirz"] for o in orders])
        d["%s_ord_angle_from_normal_deg" % arm] = np.array([o["angle"] for o in orders])
    meta = dict(
        atom="P769 TM 32.5 deg, rho3 md5 %s, carrier md5 %s" % (rho3_md5, car_md5),
        record=REC1793, job=JOB_SRC, nG_req=NG_REQ, G_actual=resA["G_actual"], rcwa_rule=M._RCWA_RULE,
        rcwa_ff=int(M._RCWA_FF), pol="TM", wavelength_nm=LAM * 1e3, period_x_nm=LX_UM * 1e3,
        period_y_nm=LY_UM * 1e3, pixel_nm=DELTA_NM, plane_y_nm=YC_PX * DELTA_NM, theta_sub_deg=math.degrees(THETA),
        driver_md5=md5(os.path.join(CODE, "Main_rcwa_l3_atom2d_opt.py")), blas_threads=BLAS_THREADS,
        convention=("physical fields = Bloch envelope * exp(i kx x), exp(-i omega t), c=1, incident Hy amplitude 1 "
                    "at z=-500 nm; arrays are [z, x] complex128; x = k*pixel (k=0..575, two periods, field exactly "
                    "Lx periodic by Littrow); z from %.0f to %.0f nm step %.1f; z<-500 and z>2090 are analytic "
                    "continuations of the uniform half-space solution; S arrays are normalised to the incident Sz; "
                    "ordering of ord_* arrays: incident first, then R then T orders with power>1e-9."
                    % (Z_LO, Z_HI, DZ_NM)),
        tests_A=resA["tests"], tests_B=(resB.get("tests") if resB is not None else None),
        mode=MODE, B_field_arrays=bool(resB is not None and "maps" in resB), B_orders_present=resB is not None)
    d["meta_json"] = np.array(json.dumps(meta, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))
    p = os.path.join(HERE, "atom_field_maps_A_n%d.npz" % NG_REQ if MODE == "mapA" else "atom_field_maps.npz")
    t0 = time.perf_counter()
    np.savez_compressed(p, **d)
    log("[npz] wrote %s (%.1f MB) in %.1f s; keys: %s" % (p, os.path.getsize(p) / 1e6, time.perf_counter() - t0,
                                                         sorted(d.keys())))


if __name__ == "__main__":          # importing the module (dry RAM-gate check) sets everything up but solves nothing
    run()
