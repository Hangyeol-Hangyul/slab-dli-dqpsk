# -*- coding: ascii -*-
r"""Main_fdtd_carrier2d_verify.py -- 2-D FullWAVE FDTD check of the bottom metagrating carrier (L1 + L2 + spacer + a
2-layer slab termination) at the Littrow angle +-theta_sub, TM, 1550 nm.   261016, review1 PASS + FIX1 261016 (empty-run stability gate, load_result).

PLAN: <WORK_ROOT>\fdtd_carrier_261016\PLAN.md (items 1-7, controls C0 / C1 / C2). Fork of the Tier B path of
Main_fdtd_atom_verify.py with `dimension = 2` (x lateral and periodic, z the propagation axis), 1-D index profiles
(UF_DATAFILE, np.repeat of the binary density: EXACT for every k), line monitors, and the post module
fdtd_carrier2d_post.py (1-D order projection, two-plane forward / backward solve in the spacer, energy, reciprocity).
Reuses by import: fdtd_atom_post (helpers, log parser, tmm_h, quality gates) and Main_fdtd_input2d_verify.run_fullwave.

Stack (bottom to top, z = 0 at the L1 bottom face): SiO2 substrate n_sub (semi-infinite, incidence side; the launch
plane is inside it) | L1 h1 | L2 h2 | spacer (n_sub) | term_layers (uniform segments, "n:t_nm,n:t_nm,...") | air.
Layer faces and the lower domain face are checked to lie on z nodes (dz = dz_nm / k).

Job keys (ASCII values; unknown keys are refused, exit 2)
  job-file keys   version module script sweep_keys Output_folder subfolder_name blas_threads mpi_np mpi_np_final result_folder GPU_on
  mode          carrier | empty_only        carrier  density | zeros | ones (zeros: L1, L2 = SiO2 ; ones: L1, L2 = a-Si slab)
  launch        pth | mth                   k  integer refinement (Nx = nx0 * k, dz = dz_nm / k, x and z refined together)
  l12_density   npy of shape (2, nx0)       (carrier = density only)         nx0 512
  term_layers   "n:t_nm,n:t_nm"             term_name label
  n_sub 1.4440  n_asi 3.48  lam_nm 1550  period_nm 1315.39  h1_nm 400  h2_nm 300  spacer_nm 465  dz_nm 2.5
  pad_air_nm 2000 (MEASURED 261016: 1000 nm lets the PML absorb the evanescent tail of a TIR wave, 1.7e-2 of the power at kx/k0 1.178; 2000 nm: 1e-4)  pad_sub_nm 1000  mon_off1_nm 250  mon_off2_nm 450  s1_nm 800  s2_nm 1000
  pml_cells 12 (at k = 1; the effective count is pml_cells * k, so the PML keeps its physical thickness)
  stop_periods 300  emp_periods (default = stop_periods)  emp_check_periods (0 = off; else a 2nd, shorter empty run whose launched-order
                phasor is compared with the first: the stability number that gates the empty-run convergence waiver, stored in the result
                and in emp_stability.json beside the empty run; a job that reuses emp_dir reads that file)  ramp_periods 3  fdtd_stop_auto 2  fdtd_stop_tol 1e-5  fdtd_dispersion 0
  polarization 1 = TM in the 2-D tool (MEASURED 261016: polarization 0 writes ey / hx / hz = TE, 1 writes ex / ez / hy = TM; the 3-D tool is the opposite).
                Control C0 confirms TM from the Ex/|E| = cos(theta) signature; field_mask 14 writes ex + ez for TM
  field_mask 14  prof_shift_x 0.5  lx_snap 1  emp_dir (reuse the empty run of an earlier job folder run_root/..)
  run_root (local scratch, required)  fdtd_gpu 0|1  fdtd_mpi_np 1  rsoft_timeout_s 7200  max_run_s  max_ram_pct (default: env SIM_MAX_RAM_PCT, else off)  resume 1
  dry_run 0|1  pin_cores "10-19" (direct runs only)  bytes_per_cell 60  qc_energy 2e-3  qc_spread 5e-3  qc_phase_adv 0.5

Outputs in Output_folder\subfolder_name (or result_folder): fdtd2d_manifest.json, fdtd2d_orders.json, fdtd2d_result.json.
Raw monitor files stay in run_root. A flagged result is NOT ok (exit 1) and is re-run, not resumed.
ASCII-only source and console output (cp949 consoles); every open() has encoding="utf-8".
"""
import hashlib
import io
import json
import math
import os
import socket
import sys
import time


def _early_blas_env():
    n = "1"
    try:
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            for ln in io.open(sys.argv[1], encoding="utf-8", errors="replace"):
                if ln.lstrip().startswith("#") or "=" not in ln:
                    continue
                k, _, v = ln.partition("=")
                if k.strip() == "blas_threads" and v.split("#")[0].strip():
                    n = v.split("#")[0].strip()
    except OSError:
        pass
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = n


_early_blas_env()
_HERE = os.path.dirname(os.path.abspath(__file__))
_GT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "solver", "Version_unified", "RCWA_custom"))
for _p in (_HERE, os.path.join(_HERE, "_vendor"), os.path.dirname(_HERE), _GT, os.path.join(_GT, "_vendor"), os.path.dirname(_GT)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                            # noqa: E402
import fdtd_atom_post as P                                    # noqa: E402
import fdtd_carrier2d_post as Q                               # noqa: E402

TAG = "[FDTD-C2D]"
CFG = {}

JOB_META_KEYS = ("version", "module", "script", "sweep_keys", "Output_folder", "subfolder_name", "blas_threads", "mpi_np",
               "mpi_np_final", "result_folder", "GPU_on")
DRIVER_KEYS = ("mode", "launch", "k", "carrier", "l12_density", "nx0", "term_layers", "term_name", "n_sub", "n_asi", "lam_nm",
               "period_nm", "h1_nm", "h2_nm", "spacer_nm", "dz_nm", "pad_air_nm", "pad_sub_nm", "mon_off1_nm", "mon_off2_nm",
               "s1_nm", "s2_nm", "pml_cells", "stop_periods", "emp_periods", "emp_check_periods", "ramp_periods", "fdtd_stop_auto", "fdtd_stop_tol",
               "fdtd_dispersion", "polarization", "field_mask", "prof_shift_x", "lx_snap", "emp_dir", "run_root", "fdtd_gpu",
               "fdtd_mpi_np", "rsoft_timeout_s", "max_run_s", "max_ram_pct", "resume", "dry_run", "pin_cores", "bytes_per_cell",
               "qc_energy", "qc_spread", "qc_phase_adv")
KNOWN_KEYS = frozenset(JOB_META_KEYS + DRIVER_KEYS)
REQUIRED_KEYS = ("mode", "launch", "k", "carrier", "term_layers", "run_root")


def log(msg):
    print("%s %s" % (TAG, msg), flush=True)


def load_cfg(path):
    if not path or not os.path.isfile(path):
        print("%s job file not found: %s" % (TAG, path))
        raise SystemExit(2)
    d = {}
    for ln in io.open(path, encoding="utf-8", errors="replace").read().split("\n"):
        if "=" not in ln or ln.lstrip().startswith("#"):
            continue
        k, _, v = ln.partition("=")
        d[k.strip()] = v.split("#")[0].strip()
    return d


def _txt(k, d=""):
    v = CFG.get(k, "")
    return str(v).strip() if str(v).strip() != "" else d


def _num(k, d):
    try:
        return float(_txt(k, ""))
    except ValueError:
        return float(d)


def _int(k, d):
    return int(round(_num(k, d)))


def out_root():
    r = _txt("result_folder", "")
    if not r:
        o, s = _txt("Output_folder", ""), _txt("subfolder_name", "")
        if not o or not s:
            raise ValueError("result_folder (or Output_folder + subfolder_name) is required")
        r = os.path.join(o, s)
    os.makedirs(r, exist_ok=True)
    return r


def pin_cores():
    pc = _txt("pin_cores", "")
    if pc:
        import psutil
        lo = int(pc.split("-")[0])
        hi = int(pc.split("-")[1]) if "-" in pc else psutil.cpu_count() - 1
        cs = list(range(lo, hi + 1))
        psutil.Process().cpu_affinity(cs)
        log("affinity -> logical %s" % cs)
        return cs
    return None


def md5_file(path):
    h = hashlib.md5()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ RSoft .ind (2-D; head from the validated 3-D Tier B template)
HEAD = r"""
Lx = {lx:.9f}
GridS = {grid:.9f}
alpha = 0
background_index = {nbg:.6f}
boundary_max = Lx/2
boundary_min = -Lx/2
cad_aspectratio = 1
color_outline = -1
delta = 0
dimension = 2
domain_max = {zmax:.9f}
domain_min = {zmin:.9f}
fdtd_bc_x = FDTD_BC_PERIODIC
fdtd_bc_ka_auto = 0
fdtd_bc_ka_n0 = {nbg:.6f}
fdtd_complex = 0
fdtd_dispersion = {disp:d}
fdtd_display_res_auto = DISPLAY_RES_AUTO
fdtd_excitation_type = TIME_CW
fdtd_field_output = 1
fdtd_launch_direction = 3
fdtd_launch_pml = 1
fdtd_launch_pml_factor = 1.0000
fdtd_monitor_time = lambda/4
fdtd_monitor_time_auto = MONITOR_TIME_AUTO
fdtd_monmemcheck_warning = 0
fdtd_nondft_spatial_warning = 0
fdtd_nonuniform_mem_warning = 0
fdtd_pml_cells_enable = 1
fdtd_pml_cells = {pml:d}
fdtd_ramp_time = lambda*{ramp:.3f}
fdtd_simmemcheck_warning = 0
fdtd_stop_auto = {sauto:d}
fdtd_stop_tol = {stol:.3e}
fdtd_stop_time = {stop:.4f}
fdtd_stop_time_auto = 0
fdtd_time_step_auto = 1
fdtd_time_step_small_warning = 0
fdtd_update_time = 9*lambda/4
fdtd_update_time_auto = DISPLAY_TIME_AUTO
free_space_wavelength = {lam:.9f}
grid_align_x = 2
grid_align_z = 2
grid_nonuniform = 0
grid_size = GridS
grid_size_z = {gz:.9f}
index = 1
index_profile = 1
k0 = (2*pi)/free_space_wavelength
lambda = free_space_wavelength
launch_align_file = 1
launch_angle = {phi:.9f}
launch_phase_ref = 1
launch_position = 0
launch_position_z = {zmin:.9f}
launch_tilt = 1
launch_type = LAUNCH_PLANEWAVE
launch_width = inf
launch_neff = {nbg:.6f}
polarization = {pol:d}
prefix = {prefix}
sim_tool = ST_FULLWAVE
slice_position_z = 0
slice_step_size = GridS
step_size = {gz:.9f}
structure = STRUCT_CHANNEL
width = Lx
"""

SEG = r"""
segment {n}
    color = {c}
    extended = 1
    begin.x = 0
    begin.z = {z0:.9f}
    begin.width = Lx
    end.x = 0
    end.z = {z1:.9f}
    end.width = Lx
    {extra}
end segment
"""

MON = r"""
time_monitor {n}
    profile_type = PROF_INACTIVE
    color = COLOR_NONE
    type = TIMEMON_EXTENDED
    timeaverage = 2
    complexmonitor = 1
    monitoroutputmask = 0
    monitoroutputformat = OUTPUT_AMP_PHASE
    fieldoutputmask = {mask:d}
    fieldoutputformat = OUTPUT_AMP_PHASE
    frequencyanalysis = TIMEMON_FA_DFT
    begin.x = 0
    begin.z = {z:.9f}
    begin.width = Lx
end time_monitor
"""


def write_ind(path, *, lx, grid, gz, zmin, zmax, phi, prefix, layers, mons, n_bg, lam, pml, stop_periods, ramp, sauto, stol, disp,
              gpu, mask, pol):
    """layers: (z0, z1, kind, val): 'n' -> uniform index val (skipped if == n_bg), 'prof' -> (datafile, n_ref).
    mons: (name, z). Returns {name: monitor number}. time_monitor / segment share ONE counter from 1."""
    body = HEAD.format(lx=lx, grid=grid, nbg=n_bg, zmin=zmin, zmax=zmax, pml=int(pml), stop=stop_periods * lam, lam=lam, phi=phi,
                       prefix=prefix, gz=gz, ramp=ramp, sauto=int(sauto), stol=stol, disp=int(disp), pol=int(pol))
    if int(gpu) == 1:
        body += "fdtd_gpu = 1\n"
    seg, blocks, kp = 1, "", 0
    for z0, z1, kind, val in layers:
        if kind == "n":
            if abs(val - n_bg) < 1e-12:
                continue
            extra = "delta = %.9f" % (val - n_bg)
        else:
            kp += 1
            extra = "profile_type = PROF_USER_%d\n    delta = %.9f" % (kp, val[1] - n_bg)
            blocks += "\nuser_profile %d\n    type = UF_DATAFILE\n    filename = %s\nend user_profile\n" % (kp, val[0])
        body += SEG.format(n=seg, c=seg % 15 + 1, z0=z0, z1=z1, extra=extra)
        seg += 1
    body += blocks
    ids = {}
    for name, z in mons:
        body += MON.format(n=seg, z=z, mask=int(mask))
        ids[name] = seg
        seg += 1
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    return ids


def write_profile1d(path, n_vec, n_bg, n_ref, shift=0.5, pad=2):
    """1-D analogue of Main_fdtd_atom_verify.write_profile (validated 3-D recipe: nodes at the cell centres + shift, padded and wrapped):
    value f = (n - n_bg) / (n_ref - n_bg) at nx + 2 pad + 1 nodes of spacing 2 / nx on [-1, 1]."""
    nx = n_vec.size
    f = (np.asarray(n_vec, float) - n_bg) / (n_ref - n_bg)
    ix = np.arange(-pad, nx + pad + 1) % nx
    ext = f[ix]
    x0 = -1.0 + 2.0 * (-pad + shift) / nx
    x1 = -1.0 + 2.0 * (nx + pad + shift) / nx
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("/rn,a,b/nx0/ls1\n")
        fh.write("%d %.9f %.9f 0.1 OUTPUT_REAL\n" % (ext.size, x0, x1))
        for v in ext:
            fh.write("  %.9E\n" % v)


def read_profile1d(path):
    with io.open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    h = lines[1].split()
    n = int(h[0])
    f = np.array([float(v) for v in lines[2:2 + n]])
    return f, n, float(h[1]), float(h[2])


def on_node(z, dz, tol=1e-6):
    q = z / dz
    return abs(q - round(q)) < tol


def parse_term(s):
    out = []
    for tok in s.replace(";", ",").split(","):
        if tok.strip():
            n, _, t = tok.partition(":")
            out.append((float(n), float(t) * 1e-3))
    if not out:
        raise SystemExit("%s term_layers is empty" % TAG)
    return out


def estimate(nx, nz, dx, dz, periods, lam, gpu_rate=4.3e9, cpu_rate=5.7e7, b_per_cell=60.0):
    cdt = 0.995 / math.sqrt(1 / dx ** 2 + 1 / dz ** 2)
    steps = periods * lam / cdt
    cells = (nx + 3) * (nz + 24)
    return dict(cells=cells, cdt_nm=cdt * 1e3, steps=steps, cell_updates=cells * steps, gpu_min=cells * steps / gpu_rate / 60.0,
                cpu_core_h=cells * steps / cpu_rate / 3600.0, rss_gb=cells * b_per_cell / 1e9)


# ------------------------------------------------------------------ build every file of one job
def build(root):
    launch = _txt("launch").lower()
    if launch not in ("pth", "mth"):
        raise SystemExit("%s launch must be pth or mth" % TAG)
    carrier = _txt("carrier").lower()
    if carrier not in ("density", "zeros", "ones"):
        raise SystemExit("%s carrier must be density | zeros | ones" % TAG)
    kref = _int("k", 1)
    if kref < 1:
        raise SystemExit("%s k must be >= 1" % TAG)
    lam = _num("lam_nm", 1550.0) * 1e-3
    n_sub, n_asi = _num("n_sub", 1.444), _num("n_asi", 3.48)
    period = _num("period_nm", 1315.39) * 1e-3
    h1, h2, sp = _num("h1_nm", 400.0) * 1e-3, _num("h2_nm", 300.0) * 1e-3, _num("spacer_nm", 465.0) * 1e-3
    dz = _num("dz_nm", 2.5) * 1e-3 / kref
    nx0 = _int("nx0", 512)
    nx = nx0 * kref
    snap = _int("lx_snap", 1)
    grid = round(period / nx, 9)
    lx = round(nx * grid, 9) if snap else round(period, 9)
    pml = _int("pml_cells", 12) * kref
    gpu, mask, pol = _int("fdtd_gpu", 0), _int("field_mask", 14), _int("polarization", 1)
    sauto, stol = _int("fdtd_stop_auto", 2), _num("fdtd_stop_tol", 1e-5)
    disp, ramp = _int("fdtd_dispersion", 0), _num("ramp_periods", 3.0)
    stop_p = _num("stop_periods", 300.0)
    emp_p = _num("emp_periods", stop_p)
    emp_c = _num("emp_check_periods", 0.0)
    term = parse_term(_txt("term_layers"))
    z_h1, z_h2 = h1, h1 + h2
    z_sp = z_h2 + sp
    z_edges = [z_sp]
    for _n, t in term:
        z_edges.append(z_edges[-1] + t)
    top = z_edges[-1]
    zmin, zmax = -_num("pad_sub_nm", 1000.0) * 1e-3, top + _num("pad_air_nm", 2000.0) * 1e-3
    zR = (-_num("mon_off1_nm", 250.0) * 1e-3, -_num("mon_off2_nm", 450.0) * 1e-3)
    zR = (zR[1], zR[0])                                       # R1 deeper (-0.45), R2 nearer (-0.25), as PLAN item 5
    zT = (top + _num("mon_off1_nm", 250.0) * 1e-3, top + _num("mon_off2_nm", 450.0) * 1e-3)
    zS = (_num("s1_nm", 800.0) * 1e-3, _num("s2_nm", 1000.0) * 1e-3)
    if not (z_sp - sp < zS[0] < zS[1] < z_sp):
        raise SystemExit("%s spacer monitors s1 / s2 must lie inside the spacer (%.3f .. %.3f um)" % (TAG, z_sp - sp, z_sp))
    wd_e, wd_s = os.path.join(root, "emp"), os.path.join(root, "st")
    emp_dir = _txt("emp_dir", "")
    for d in ((wd_s,) if emp_dir else (wd_e, wd_s)):
        os.makedirs(d, exist_ok=True)
    if emp_dir:
        wd_e = os.path.join(emp_dir, "emp")
    wd_c = os.path.join(root, "emp_chk")
    if emp_c > 0 and not emp_dir:
        os.makedirs(wd_c, exist_ok=True)
    checks = {}
    # ---- profiles / segments
    lay = []
    if carrier == "density":
        path = _txt("l12_density", "")
        if not path or not os.path.isfile(path):
            raise SystemExit("NOT FOUND: l12_density %s" % path)
        a = np.load(path)
        if a.shape != (2, nx0):
            raise SystemExit("%s l12_density shape %s != (2, %d)" % (TAG, a.shape, nx0))
        n_lo, n_hi = n_sub, n_asi
        dens = [np.repeat(a[i].astype(float), kref) for i in (0, 1)]
        grey = [int(np.sum((r > 1e-9) & (r < 1 - 1e-9))) for r in (a[0], a[1])]
        checks["grey_pixels"] = grey
        checks["fill"] = [float(a[0].mean()), float(a[1].mean())]
        nmaps = [np.sqrt(n_sub ** 2 + r * (n_asi ** 2 - n_sub ** 2)) for r in dens]
        for fn, nm in (("L1.dat", nmaps[0]), ("L2.dat", nmaps[1])):
            write_profile1d(os.path.join(wd_s, fn), nm, n_sub, n_asi, shift=_num("prof_shift_x", 0.5))
        checks["repeat_exact"] = float(max(np.abs(dens[i].reshape(nx0, kref).mean(axis=1) - a[i]).max() for i in (0, 1)))
        rb = {}
        for fn, nm in (("L1.dat", nmaps[0]), ("L2.dat", nmaps[1])):
            f, n, x0, x1 = read_profile1d(os.path.join(wd_s, fn))
            pad = 2
            core = f[pad:pad + nx]
            rb[fn] = dict(max_abs_n_diff=float(np.abs(n_sub + (n_asi - n_sub) * core - nm).max()), nodes=n,
                          node_dx_over_cell=float(((x1 - x0) / (n - 1)) / (2.0 / nx)),
                          wrap_ok=bool(np.allclose(f[:pad], f[nx:nx + pad], atol=1e-8)))
        checks["profile_readback"] = rb
        lay += [(0.0, z_h1, "prof", ("L1.dat", n_asi)), (z_h1, z_h2, "prof", ("L2.dat", n_asi))]
    elif carrier == "ones":
        lay += [(0.0, z_h2, "n", n_asi)]
    lay += [(z_h2, z_sp, "n", n_sub)]
    for i, (n, t) in enumerate(term):
        lay.append((z_edges[i], z_edges[i + 1], "n", n))
    lay.append((top, top + _num("pad_air_nm", 2000.0) * 1e-3 + 0.3, "n", 1.0))
    faces = [0.0, z_h1, z_h2, z_sp] + z_edges[1:]
    checks["faces_on_nodes"] = {("%.6f" % z): bool(on_node(z, dz)) for z in faces}
    checks["zmin_on_node"] = bool(on_node(zmin, dz))
    nz = int(round((zmax - zmin) / dz))
    theta = math.degrees(math.asin(lam / (lx * n_sub)))
    phi = theta if launch == "pth" else -theta
    m_inc = +1 if launch == "pth" else -1
    mons = [("R1", zR[0]), ("R2", zR[1]), ("T1", zT[0]), ("T2", zT[1]), ("S1", zS[0]), ("S2", zS[1])]
    kw = dict(lx=lx, grid=grid, gz=dz, zmin=zmin, zmax=zmax, phi=phi, mons=mons, n_bg=n_sub, lam=lam, pml=pml, ramp=ramp,
              sauto=sauto, stol=stol, disp=disp, gpu=gpu, mask=mask, pol=pol)
    ids_e = write_ind(os.path.join(wd_e, "emp.ind"), prefix="emp", layers=[], stop_periods=emp_p, **kw) if not emp_dir else None
    if emp_c > 0 and not emp_dir:
        write_ind(os.path.join(wd_c, "emp_chk.ind"), prefix="emp", layers=[], stop_periods=emp_c, **kw)
    ids_s = write_ind(os.path.join(wd_s, "st.ind"), prefix="st", layers=lay, stop_periods=stop_p, **kw)
    if emp_dir:
        ids_e = {n: i + 1 for i, (n, _z) in enumerate(mons)}                   # same numbering as write_ind (no segments in the empty run)
        old = io.open(os.path.join(wd_e, "emp.ind"), encoding="utf-8").read()
        probe = os.path.join(root, "_emp_probe.ind")
        write_ind(probe, prefix="emp", layers=[], stop_periods=emp_p, **kw)
        if io.open(probe, encoding="utf-8").read() != old:
            raise SystemExit("%s emp_dir %s holds a different empty run (emp.ind differs) -- refusing to reuse it" % (TAG, emp_dir))
    ce = dict(nx=nx, nx_empty=nx, nz=nz)
    man = dict(schema="fdtd_carrier2d_manifest_v1", mode=_txt("mode"), launch=launch, carrier=carrier, term_name=_txt("term_name", ""),
               term_layers=[[n, t] for n, t in term], k=kref, lam_um=lam, n_sub=n_sub, n_out=1.0, n_asi=n_asi, period_um=period,
               lx_um=lx, grid_um=grid, dz_um=dz, nx_cells=nx, nz_cells=nz, pml_cells=pml, stop_periods=stop_p, emp_periods=emp_p, emp_check_periods=emp_c,
               fdtd_gpu=gpu, polarization=pol, field_mask=mask, theta_deg=theta, phi_deg=phi, m_inc=m_inc, top_um=top,
               domain_um=[zmin, zmax], z=dict(top=top, z_in_ref=0.0, R1=zR[0], R2=zR[1], T1=zT[0], T2=zT[1], S1=zS[0], S2=zS[1],
                                              h1=z_h1, h2=z_h2, sp_top=z_sp, sp_bot=z_h2, edges=z_edges),
               faces_um=faces, orders_R=[-1, 0, 1], orders_T=[0], m_t=0, tier="B",
               runs=dict(emp=dict(prefix="emp", dir=wd_e, lx_um=lx, mon=ids_e), st=dict(prefix="st", dir=wd_s, lx_um=lx, mon=ids_s)),
               checks=dict(checks, cells_expected=ce, lx_over_grid=lx / grid), emp_reused=bool(emp_dir),
               inputs=dict(l12_density=_txt("l12_density", ""), l12_md5=(md5_file(_txt("l12_density")) if carrier == "density" else "")),
               qc={})
    for key, name in (("qc_energy", "energy"), ("qc_spread", "spread_rel"), ("qc_phase_adv", "phase_adv_deg")):
        man["qc"][name] = _num(key, Q.QC_C2D[name])
    for name in Q.QC_C2D:
        man["qc"].setdefault(name, Q.QC_C2D[name])
    if emp_c > 0 and not emp_dir:
        man["runs"]["emp_chk"] = dict(prefix="emp", dir=wd_c, lx_um=lx, mon=ids_e)
    man["estimate"] = estimate(nx, nz, grid, dz, stop_p, lam, b_per_cell=_num("bytes_per_cell", 60.0))
    return man, dict(emp=wd_e, st=wd_s, emp_chk=wd_c)


def _stability(man, orders, dirs):
    """Empty-run incident-phasor stability (the number that gates the empty-run convergence waiver). Computed here when this job ran both
    empty lengths (and written to emp_stability.json beside the empty run); read from that file when the empty run is reused."""
    f = os.path.join(dirs["emp"], "emp_stability.json")
    if "emp_chk" in man["runs"]:
        st = Q.emp_stability(orders, man["m_inc"])
        P.save_json(f, st)
    elif os.path.isfile(f):
        st = json.load(io.open(f, encoding="utf-8"))
    else:
        st = None
    orders["emp_stability"] = st
    log("empty-run stability: %s" % ("none" if st is None else "max rel %.3e (tol %.1e)" % (st["max_rel"], st["tol"])))


def _ram_ok(est_gb, max_pct):
    try:
        import psutil
        if max_pct <= 0:
            return True, None
        vm = psutil.virtual_memory()
        after = (vm.total - vm.available + est_gb * 1e9) / vm.total
        return after <= max_pct / 100.0, after
    except Exception:
        return True, None


def main():
    global CFG
    if len(sys.argv) < 2:
        print("usage: Main_fdtd_carrier2d_verify.py <job_file>")
        return 2
    CFG = load_cfg(sys.argv[1])
    unknown = sorted(k for k in CFG if k not in KNOWN_KEYS)
    if unknown:
        print("%s unknown job keys: %s" % (TAG, unknown))
        return 2
    miss = [k for k in REQUIRED_KEYS if _txt(k, "") == ""]
    if miss:
        print("%s missing job keys: %s" % (TAG, miss))
        return 2
    if _txt("mode") not in ("carrier", "empty_only"):
        print("%s mode must be carrier | empty_only" % TAG)
        return 2
    log("job %s host %s" % (sys.argv[1], socket.gethostname()))
    rr = out_root()
    run_root = _txt("run_root")
    os.makedirs(run_root, exist_ok=True)
    pin_cores()
    man, dirs = build(run_root)
    man.update(driver=os.path.abspath(__file__), driver_md5=md5_file(os.path.abspath(__file__)),
               post_md5=md5_file(os.path.abspath(Q.__file__)), host=os.environ.get("COMPUTERNAME", "?"),
               started=time.strftime("%F %T"), job_file=sys.argv[1], dry_run=_int("dry_run", 0))
    core_keys = ("mode", "launch", "carrier", "term_layers", "k", "grid_um", "lx_um", "dz_um", "z", "inputs", "stop_periods",
                 "emp_periods", "emp_check_periods", "polarization", "pml_cells")
    man["manifest_core_md5"] = hashlib.md5(json.dumps({k: man[k] for k in core_keys}, sort_keys=True).encode("ascii")).hexdigest()
    mpath = os.path.join(rr, "fdtd2d_manifest.json")
    P.save_json(mpath, man)
    est, chk = man["estimate"], man["checks"]
    log("%s %s k %d  cell %d x %d  grid %.4f nm dz %.4f nm  theta %.5f  est %.2f M cells, ~%d steps, CPU %.2f core-h, RSS %.3f GB"
        % (man["launch"], man["term_name"], man["k"], man["nx_cells"], man["nz_cells"], man["grid_um"] * 1e3, man["dz_um"] * 1e3,
           man["theta_deg"], est["cells"] / 1e6, est["steps"], est["cpu_core_h"], est["rss_gb"]))
    bad = (not all(chk["faces_on_nodes"].values()) or not chk["zmin_on_node"]
           or abs(chk["lx_over_grid"] - round(chk["lx_over_grid"])) > 1e-6
           or chk.get("repeat_exact", 0.0) > 1e-12
           or any(v["max_abs_n_diff"] > 1e-8 or not v["wrap_ok"] for v in chk.get("profile_readback", {}).values()))
    log("raster checks: faces on nodes %s, lx/grid %.9f, repeat_exact %s" % (all(chk["faces_on_nodes"].values()),
                                                                               chk["lx_over_grid"], chk.get("repeat_exact")))
    if bad:
        log("RASTER CHECK FAILED -- refusing to run")
        return 2
    if _int("dry_run", 0):
        log("dry_run: files written under %s, manifest %s" % (run_root, mpath))
        return 0
    rpath = os.path.join(rr, "fdtd2d_result.json")
    if _int("resume", 1) and os.path.isfile(rpath):
        try:
            old = json.load(io.open(rpath, encoding="utf-8"))
            if old.get("ok") and old.get("manifest_core_md5") == man["manifest_core_md5"]:
                log("resume: result already in %s" % rpath)
                return 0
        except Exception:
            pass
    ok_r, after = _ram_ok(est["rss_gb"], _num("max_ram_pct", float(os.environ.get("SIM_MAX_RAM_PCT", "0"))))
    if not ok_r:
        log("SKIPPED: RAM would reach %.0f %% > %.0f %%" % (100 * after, _num("max_ram_pct", float(os.environ.get("SIM_MAX_RAM_PCT", "0")))))
        return 2
    import Main_fdtd_input2d_verify as F                      # noqa: E402  (run_fullwave only)
    F.CFG = CFG
    tmo, npr = _num("rsoft_timeout_s", 7200.0), _int("fdtd_mpi_np", 1)
    runs = {}
    empty_only = _txt("mode") == "empty_only"
    todo = ["emp"] if empty_only else (["st"] if man["emp_reused"] else ["emp", "st"])
    if "emp_chk" in man["runs"]:
        todo.insert(todo.index("emp") + 1, "emp_chk")
    for run in todo:
        res = F.run_fullwave(dirs[run], run + ".ind", timeout=tmo, mpi_np=npr)
        facts = {k: v for k, v in res.items() if k != "log_tail"}
        runs[run] = facts
        lf = Q.parse_log2d(os.path.join(dirs[run], "log.txt"), index=-1)
        facts["logfacts"] = lf
        log("%s run ok=%s rc=%s %.1f s peak %.3f GB  NX,NY,NZ %s,%s,%s  NT %s  converged %s" % (
            run, res.get("ok"), res.get("rc"), res.get("elapsed", 0.0), res.get("peak_rss_gb") or 0.0, lf.get("nx"), lf.get("ny"),
            lf.get("nz"), lf.get("nt_run"), lf.get("converged")))
        if not res.get("ok"):
            P.save_json(rpath, dict(ok=False, error="%s run failed: %s" % (run, res.get("why")), runs=runs,
                                    manifest_core_md5=man["manifest_core_md5"], log_tail=res.get("log_tail")))
            return 1
    if man["emp_reused"] and not empty_only:
        runs["emp"] = dict(logfacts=Q.parse_log2d(os.path.join(dirs["emp"], "log.txt"), index=-1), reused=True)
    if empty_only:                                            # control C0: the empty domain only
        try:
            orders = Q.extract_orders(man, dirs, runs=("emp",) + (("emp_chk",) if "emp_chk" in man["runs"] else ()))
            _stability(man, orders, dirs)
            c0 = Q.c0_launch(man, orders)
        except Exception as e:
            log("C0 EXTRACTION FAILED: %s: %s" % (type(e).__name__, str(e)[:300]))
            P.save_json(rpath, dict(ok=False, error="c0 extraction failed: %s: %s" % (type(e).__name__, str(e)[:300]), runs=runs))
            return 1
        P.save_json(os.path.join(rr, "fdtd2d_orders.json"), orders)
        P.save_json(rpath, dict(ok=True, c0=c0, runs=runs, manifest_core_md5=man["manifest_core_md5"]))
        log("C0: %s" % json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in c0.items()}, sort_keys=True))
        return 0
    try:
        orders = Q.extract_orders(man, dirs, runs=("emp", "st") + (("emp_chk",) if "emp_chk" in man["runs"] else ()))
        _stability(man, orders, dirs)
    except Exception as e:
        log("EXTRACTION FAILED: %s: %s" % (type(e).__name__, str(e)[:300]))
        P.save_json(rpath, dict(ok=False, error="extraction failed: %s: %s" % (type(e).__name__, str(e)[:300]), runs=runs,
                                manifest_core_md5=man["manifest_core_md5"]))
        return 1
    P.save_json(os.path.join(rr, "fdtd2d_orders.json"), orders)
    try:
        result = Q.analyse(man, orders, logfacts={r: runs[r].get("logfacts") for r in runs})
    except Exception as e:
        log("ANALYSIS FAILED (orders saved): %s: %s" % (type(e).__name__, str(e)[:300]))
        P.save_json(rpath, dict(ok=False, error="analysis failed: %s: %s" % (type(e).__name__, str(e)[:300]), runs=runs,
                                manifest_core_md5=man["manifest_core_md5"]))
        return 1
    result.update(ok=bool(not result["flags"]), runs=runs, manifest_core_md5=man["manifest_core_md5"])
    if not result["ok"]:
        log("RESULT NOT OK -- quality flags: %s" % result["flags"])
    P.save_json(rpath, result)
    log("result: %s" % json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in result["record"].items()
                                   if not isinstance(v, (str, dict, list))}, sort_keys=True))
    man["finished"] = time.strftime("%F %T")
    P.save_json(mpath, man)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
