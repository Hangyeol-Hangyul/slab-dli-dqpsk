# -*- coding: ascii -*-
r"""Main_fdtd_atom_verify.py -- external FullWAVE FDTD check of the Li-vs-Laurent result of an SCDC L3 atom (sim758).

DRAFT of 261009 (fdtdbuild role), FIXED 261009 by the fix758 role (quality flags fill `flags`, `ok` requires them) and by the fix2 758 role (non-finite
monitor data = flag QC_NONFINITE, ok false, exit 1; a failed extraction leaves a result), NOT INSTALLED: the build
role may not write code\. Install it TOGETHER with
fdtd_atom_post.py into <CODE_ROOT>\Version_unified\RCWA_custom
before any sim758 job is released. The run loop has NOT been exercised (no solver run by the build role): the first
release must be the acceptance job (sim758_ACC_*, proxy k1, ~2 min on the CPU) which must reproduce the stored S4
prototype numbers (d arg r -53.5656, d arg t -105.3374 deg) -- see the hold README.

What it does (PLAN.md of fdtdplan 261008, sections 1-5; no design change)
  Tier A  atom ALONE, normal incidence from the SiO2 spacer side, Ex launch (TM lane), single-atom cell ly x ly
          (ly = PERIOD / 3): SiO2 | L3 lower (rhoA a-Si in SiO2) | SiO2 gap | L3 upper (rhoB a-Si in air) | air.
          Measures r, t of the atom -> absR00, argR00_deg and the atom part of A_deg / argtB_deg (arg t_n).
  Tier B  FULL stack, 3-atom cell Lambda x ly, TM, REAL periodic FDTD at the Littrow angle +-theta_sub
          (k_x = +-G exactly, so plain periodic BC are exact): SiO2 | L1 | L2 | spacer | L3 lower | gap | L3 upper | air.
          Measures A_deg, argtB_deg and the ECHO (order m_inc+2 of the -th launch); rB and carrier_eff reported, not judged.
  Proxy   Tier A on the 24 x 24 block-averaged raster (`proxy_nc`), FDTD-order calibration (known limit Li).

Exact raster (plan section 2): the design pixel is PERIOD/288 = 4.5673 nm in x AND y. The FDTD cell = the design pixel
(k = 1) or an exact np.kron refinement (k = 2, 3, ...): never a resample. The record's projected density
(Main_rcwa_l3_atom2d_opt.project_recorded, beta_eval) is used verbatim, grey edge pixels included. The z grid is set with
step_size (= dz, 2.5 nm; grid_size_z is IGNORED by fullwave) so every layer face is a grid node. The printed GridS is
rounded to 9 decimals and Lx, Ly are printed as EXACT multiples of it (`lx_snap`, default 1): ratio = N to 1e-15, and the
Littrow angle is taken from the snapped Lx so k_x Lx = 2 pi exactly. Stored runs show fullwave rounds Lx / GridS to the
nearest integer (nx = N + 3 for ratios N +- 2e-5); the run is nevertheless checked against the log (`cells_ok`).

Job keys (all but the first block optional; ASCII values; `KNOWN_KEYS` below is what the validator accepts)
  job-file keys   version module script sweep_keys Output_folder subfolder_name blas_threads mpi_np mpi_np_final result_folder GPU_on
  tier          A | B                         launch  normal (A) | pth | mth (B)
  atom_name     label (032p5 ...)             atom_json  atom2d_opt.json of the record     atom_job  the record's job_*.txt
  rho3_npy      raw design variable           l12_density  carrier density (Tier B; read from atom_job if absent)
  k             exact kron refinement (1)     proxy_nc  block-average the raster to nc x nc first (0 = off)
  dz_nm         z step (2.5)                  pad_air_nm / pad_sub_nm  (A: 600 / 650; B: 1000 / 1000)
  pml_cells 12  stop_periods (A 160, B 320)   est_periods (estimate only)  ramp_periods 3  fdtd_stop_auto 2  fdtd_stop_tol 1e-5
  fdtd_dispersion 0   thin_cells 4   empty_full_cell 0   lx_snap 1   prof_shift_x 0.5   prof_shift_y 0.0   field_mask 6
  run_root (local scratch)   fdtd_gpu 0|1   fdtd_mpi_np 1   rsoft_timeout_s   max_run_s   max_ram_pct (default: env SIM_MAX_RAM_PCT, else off)   resume 1
  dry_run 0|1 (write every file + the manifest, run nothing)   pin_cores (direct runs only)
  rcwa_ref 0|1   rcwa_ng "198,402,594"  (Tier A: same-raster grcwa Laurent / Li numbers, S1 code; ~1 min per nG-594 solve)

Outputs in Output_folder\subfolder_name (or result_folder): fdtd_atom_manifest.json, fdtd_atom_orders.json,
fdtd_atom_result.json (fdtd_atom_post.analyse + run facts), rcwa_ref.json (rcwa_ref = 1). Raw monitor files stay in run_root.
`ok` in the result = cells_ok AND no quality flag (fdtd_atom_post.quality: both runs converged, energy |R + T - 1|, monitor-pair spread,
empty-run |R2 / R1| and phase advance, cross-pol, cell count). A flagged result is NOT ok: it exits 1 and is re-run, not resumed. Tolerances are
code constants of fdtd_atom_post (QC_DEFAULT), not job keys; re-judge saved orders with `fdtd_atom_post.py analyse --qc key=val` (recorded).
Result / post-processing: fdtd_atom_post.py (extract / analyse / combine / ladder).

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
    """BLAS thread env from the job's blas_threads (default 1), set BEFORE numpy is imported -- the atom2d driver does the
    same; without it numpy would start one BLAS thread per host core inside a 2-core allocation."""
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

TAG = "[FDTD-ATOM]"
CFG = {}

JOB_META_KEYS = ("version", "module", "script", "sweep_keys", "Output_folder", "subfolder_name", "blas_threads", "mpi_np",
               "mpi_np_final", "result_folder", "GPU_on")
DRIVER_KEYS = ("tier", "launch", "atom_name", "atom_json", "atom_job", "rho3_npy", "l12_density", "k", "proxy_nc", "dz_nm",
               "pad_air_nm", "pad_sub_nm", "pml_cells", "stop_periods", "est_periods", "ramp_periods", "fdtd_stop_auto",
               "fdtd_stop_tol", "fdtd_dispersion", "thin_cells", "empty_full_cell", "lx_snap", "prof_shift_x", "prof_shift_y",
               "field_mask", "run_root", "fdtd_gpu", "fdtd_mpi_np", "rsoft_timeout_s", "max_run_s", "max_ram_pct", "resume",
               "dry_run", "pin_cores", "rcwa_ref", "rcwa_ng", "bytes_per_cell")
KNOWN_KEYS = frozenset(JOB_META_KEYS + DRIVER_KEYS)
REQUIRED_KEYS = ("tier", "launch", "atom_name", "atom_json", "atom_job", "rho3_npy", "k", "run_root")

N_SIO2_CARRIER, N_ASI_CARRIER = 1.444, 3.48                   # atom2d_opt._obj_build constants for the carrier layers
RCWA_NG_MAP = {"601": 198, "1201": 402, "1793": 594}          # 3-atom nG -> single-atom nG with the identical G set (S1 log)


def log(msg):
    print("%s %s" % (TAG, msg), flush=True)


# ------------------------------------------------------------------ job file
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


def read_job_txt(path):
    d = {}
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        ln = ln.split("#")[0]
        if "=" in ln:
            k, _, v = ln.partition("=")
            d[k.strip()] = v.strip()
    return d


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


# ------------------------------------------------------------------ atom / carrier -> permittivity maps
def _resample_cols(col, nx):
    """atom2d_opt.main()._resample verbatim: area-conserving resample of a 1-D density to nx columns."""
    n0 = col.size
    c = np.concatenate([[0.0], np.cumsum(col)])
    e = np.arange(nx + 1) * (n0 / float(nx))
    lo = np.clip(np.floor(e).astype(int), 0, n0)
    fr = e - np.floor(e)
    ci = c[lo] + fr * col[np.clip(lo, 0, n0 - 1)]
    return (ci[1:] - ci[:-1]) / (n0 / float(nx))


def load_atom():
    """Rebuild what the optimiser simulated, from the record: projected L3 density of both sub-layers (beta_eval) and
    the stack numbers. Imports Main_rcwa_l3_atom2d_opt with the DEFAULT rcwa keys only (D2 guard: this job file must
    carry no rcwa_rule / rcwa_ff). Read-only on every input."""
    for bad in ("rcwa_rule", "rcwa_ff", "rcwa_pol_epsinv", "eval_only"):
        if bad in CFG:
            raise SystemExit("%s job key %r would be read by the imported atom2d module (D2 guard) -- refusing" % (TAG, bad))
    rec = json.load(io.open(_txt("atom_json"), encoding="utf-8"))
    job = read_job_txt(_txt("atom_job"))
    x = np.load(_txt("rho3_npy"))
    import Main_rcwa_l3_atom2d_opt as M                       # noqa: E402
    G = M.G
    G.set_period(float(job.get("period_nm", "1315.39")))
    G.set_substrate(float(job.get("n_sub", "1.4440")))
    r3 = M.project_recorded(x, rec, float(rec.get("beta_eval", 64.0)), job=job)
    if not isinstance(r3, list) or len(r3) != 2:
        raise SystemExit("%s the record is not a 2-layer atom (l3_layers %s)" % (TAG, rec.get("l3_layers")))
    rA, rB = np.asarray(r3[0], float), np.asarray(r3[1], float)
    fill_n = float(rec.get("l3_fill_n", 1.444))
    n_gap = float(rec.get("n_l3_gap", fill_n))
    if abs(n_gap - fill_n) > 1e-12:
        raise SystemExit("%s n_l3_gap %.6f != l3_fill_n %.6f: the gap layer would be rhoA-patterned (unsupported)" % (TAG, n_gap, fill_n))
    if int(rec.get("atoms_per_period", 3)) != 3 or int(rec.get("l3_etch_stack", 0)) != 0:
        raise SystemExit("%s record has atoms_per_period / l3_etch_stack outside the plan" % TAG)
    eps3 = float(np.real(complex(G.eps_of(job.get("material3", "a-Si")))))
    H = [float(v) * 1e-3 for v in rec["H_nm"]]
    split = float(rec["l3_split"])
    n_sub = float(job.get("n_sub", "1.4440"))
    n_sp = float(job.get("n_spacer", "1.4440"))
    if abs(n_sp - n_sub) > 1e-12:
        raise SystemExit("%s n_spacer %.6f != n_sub %.6f (the spacer would need its own segment)" % (TAG, n_sp, n_sub))
    d = dict(rec=rec, job=job, rA=rA, rB=rB, fill_n=fill_n, eps3=eps3, n_asi=math.sqrt(eps3),
             eps_lo=fill_n ** 2 + rA * (eps3 - fill_n ** 2), eps_up=1.0 + rB * (eps3 - 1.0),
             H1=H[0], H2=H[1], H3=H[2], h_lo=H[2] * split, h_up=H[2] * (1.0 - split),
             gap=float(rec.get("l3_gap_nm", 0.0)) * 1e-3, spacer=float(rec["spacer_nm"]) * 1e-3,
             period=float(G.PERIOD), n_sub=n_sub, lam=float(job.get("lams_nm", "1550.00")) * 1e-3,
             nxy=int(rec["NXY"]),
             grey_lo=int(np.sum((rA > 1e-3) & (rA < 1 - 1e-3))), grey_up=int(np.sum((rB > 1e-3) & (rB < 1 - 1e-3))),
             fill_lo=float(rA.mean()), fill_up=float(rB.mean()))
    return d


def load_carrier(atom, nx):
    """Tier B: the record's 1-D carrier, area-resampled to nx = 3 * NXY columns (atom2d_opt._resample), extruded in y."""
    path = _txt("l12_density", "") or atom["job"].get("l12_density", "")
    if not path or not os.path.isfile(path):
        raise SystemExit("NOT FOUND: l12_density %s" % path)
    a = np.load(path)
    r1, r2 = _resample_cols(a[0], nx), _resample_cols(a[1], nx)
    e = [N_SIO2_CARRIER ** 2 + r * (N_ASI_CARRIER ** 2 - N_SIO2_CARRIER ** 2) for r in (r1, r2)]
    return dict(path=path, rho1=r1, rho2=r2, eps1=e[0][:, None] * np.ones((1, atom["nxy"])), eps2=e[1][:, None] * np.ones((1, atom["nxy"])))


def block_avg(a, nc):
    n = a.shape[0]
    b = n // nc
    if b * nc != n:
        raise ValueError("proxy_nc %d does not divide %d" % (nc, n))
    return a.reshape(nc, b, nc, b).mean(axis=(1, 3))


# ------------------------------------------------------------------ RSoft .ind (templates = the validated prototypes S2 / S4)
HEAD = r"""
Lx = {lx:.9f}
Ly = {ly:.9f}
GridS = {grid:.9f}
alpha = 0
background_index = {nbg:.6f}
boundary_max = Lx/2
boundary_min = -Lx/2
boundary_max_y = Ly/2
boundary_min_y = -Ly/2
cad_aspectratio = 1
color_outline = -1
delta = 0
dimension = 3
domain_max = {zmax:.9f}
domain_min = {zmin:.9f}
fdtd_bc_x = FDTD_BC_PERIODIC
fdtd_bc_y = FDTD_BC_PERIODIC
fdtd_bc_ka_auto = {ka:d}
fdtd_bc_ka_n0 = {nbg:.6f}
fdtd_complex = {cplx:d}
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
grid_align_y = 2
grid_align_z = 2
grid_nonuniform = 0
grid_size = GridS
grid_size_y = GridS
grid_size_z = {gz:.9f}
height = Ly
index = 1
index_profile = 1
k0 = (2*pi)/free_space_wavelength
lambda = free_space_wavelength
launch_align_file = 1
launch_angle = {phi:.9f}
launch_phase_ref = 1
launch_position = 0
launch_position_y = 0
launch_position_z = {zmin:.9f}
launch_tilt = 1
launch_type = LAUNCH_PLANEWAVE
launch_width = inf
launch_height = inf
launch_neff = {nbg:.6f}
polarization = 0
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
    begin.y = 0
    begin.z = {z0:.9f}
    begin.width = Lx
    begin.height = Ly
    end.x = 0
    end.y = 0
    end.z = {z1:.9f}
    end.width = Lx
    end.height = Ly
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
    begin.y = 0
    begin.z = {z:.9f}
    begin.width = Lx
    begin.height = Ly
end time_monitor
"""


def write_ind(path, *, lx, ly, grid, gz, zmin, zmax, phi, prefix, layers, mons, n_bg, lam, pml=12, stop_periods=160.0, ramp=3.0,
              sauto=2, stol=1e-5, disp=0, gpu=0, mask=6):
    """layers: list of (z0, z1, kind, val): 'n' -> uniform index val (skipped if == n_bg), 'prof' -> (datafile, n_ref).
    mons: list of (name, z). Returns {name: monitor number}. Same text as the validated S2 / S4 prototypes."""
    body = HEAD.format(lx=lx, ly=ly, grid=grid, nbg=n_bg, zmin=zmin, zmax=zmax, ka=0, cplx=0, pml=int(pml), stop=stop_periods * lam,
                       lam=lam, phi=phi, prefix=prefix, gz=gz, ramp=ramp, sauto=int(sauto), stol=stol, disp=int(disp))
    if int(gpu) == 1:
        body += "fdtd_gpu = 1\n"
    seg = 1
    blocks = ""
    k = 0
    for z0, z1, kind, val in layers:
        if kind == "n":
            if abs(val - n_bg) < 1e-12:
                continue
            extra = "delta = %.9f" % (val - n_bg)
        else:
            k += 1
            extra = "profile_type = PROF_USER_%d\n    delta = %.9f" % (k, val[1] - n_bg)
            blocks += "\nuser_profile %d\n    type = UF_DATAFILE\n    filename = %s\nend user_profile\n" % (k, val[0])
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


def write_profile(path, n_map, n_bg, n_ref, shift=(0.0, 0.0), pad=2):
    """Main_fdtd_input2d_verify.write_profile verbatim (validated 261006, rows = X, columns = Y)."""
    nx, ny = n_map.shape
    f = (np.asarray(n_map, float) - n_bg) / (n_ref - n_bg)
    ix = np.arange(-pad, nx + pad + 1) % nx
    iy = np.arange(-pad, ny + pad + 1) % ny
    ext = f[np.ix_(ix, iy)]
    x0 = -1.0 + 2.0 * (-pad + shift[0]) / nx
    x1 = -1.0 + 2.0 * (nx + pad + shift[0]) / nx
    y0 = -1.0 + 2.0 * (-pad + shift[1]) / ny
    y1 = -1.0 + 2.0 * (ny + pad + shift[1]) / ny
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("/rn,a,b/nx0\n/r,qa,qb\n")
        fh.write("%d %.9f %.9f 0.1 OUTPUT_REAL_3D\n" % (ext.shape[0], x0, x1))
        fh.write("%d %.9f %.9f\n" % (ext.shape[1], y0, y1))
        for row in ext:
            fh.write("   ".join("%.9E" % v for v in row) + "\n")


def read_profile(path):
    """Parse a profile datafile back: (f[nrow, ncol], x0, x1, y0, y1)."""
    with io.open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    h3, h4 = lines[2].split(), lines[3].split()
    nr, x0, x1 = int(h3[0]), float(h3[1]), float(h3[2])
    nc, y0, y1 = int(h4[0]), float(h4[1]), float(h4[2])
    data = np.array([[float(v) for v in ln.split()] for ln in lines[4:4 + nr]])
    return data, nr, nc, x0, x1, y0, y1


# ------------------------------------------------------------------ geometry
def snap_cell(L_phys, n_cells, snap):
    """grid printed with 9 decimals; L printed as the exact multiple n_cells * grid_print (snap) or as given."""
    grid = round(L_phys / n_cells, 9)
    L = round(n_cells * grid, 9) if snap else round(L_phys, 9)
    return grid, L


def estimate(nx, ny, nz, dx, dy, dz, periods, lam, gpu_rate=4.3e9, cpu_rate=4.5e7, b_per_cell=59.0):
    cdt = 0.995 / math.sqrt(1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2)
    steps = periods * lam / cdt
    cells = (nx + 3) * (ny + 3) * (nz + 24)
    upd = cells * steps
    return dict(cells=cells, cdt_nm=cdt * 1e3, steps=steps, cell_updates=upd, gpu_min=upd / gpu_rate / 60.0,
                cpu_h=upd / cpu_rate / 3600.0, rss_gb=cells * b_per_cell / 1e9)


def on_node(z, dz, tol=1e-6):
    q = z / dz
    return abs(q - round(q)) < tol


# ------------------------------------------------------------------ build every file of one job
def build(root):
    """Write profiles, .ind files and the manifest. Returns (manifest, dirs, checks). No solver is run."""
    tier, launch = _txt("tier").upper(), _txt("launch").lower()
    if tier not in ("A", "B"):
        raise SystemExit("%s tier must be A or B" % TAG)
    if (tier == "A") != (launch == "normal") or (tier == "B" and launch not in ("pth", "mth")):
        raise SystemExit("%s launch %r does not fit tier %s" % (TAG, launch, tier))
    kref = _int("k", 1)
    if kref < 1:
        raise SystemExit("%s k must be >= 1" % TAG)
    atom = load_atom()
    nxy = atom["nxy"]
    lam, n_sub, period = atom["lam"], atom["n_sub"], atom["period"]
    dz = _num("dz_nm", 2.5) * 1e-3
    pml = _int("pml_cells", 12)
    snap = _int("lx_snap", 1)
    pnc = _int("proxy_nc", 0)
    eps_lo, eps_up = atom["eps_lo"], atom["eps_up"]
    if pnc:
        eps_lo, eps_up = block_avg(eps_lo, pnc), block_avg(eps_up, pnc)
    n_src = eps_lo.shape[0]                              # pixels per atom side in the structure map (96 or proxy_nc)
    kron = np.ones((kref, kref))
    ncell_atom = n_src * kref                            # FDTD cells per atom side
    ly_phys = period / 3.0
    grid, ly = snap_cell(ly_phys, ncell_atom, snap)
    nbg = n_sub
    shift = (_num("prof_shift_x", 0.5), _num("prof_shift_y", 0.0))
    thin = _int("thin_cells", 4)
    mask = _int("field_mask", 6)
    gpu = _int("fdtd_gpu", 0)
    sauto, stol = _int("fdtd_stop_auto", 2), _num("fdtd_stop_tol", 1e-5)
    disp = _int("fdtd_dispersion", 0)
    ramp = _num("ramp_periods", 3.0)
    stop_p = _num("stop_periods", 160.0 if tier == "A" else 320.0)
    est_p = _num("est_periods", 80.0 if tier == "A" else 260.0)
    n_asi = atom["n_asi"]
    checks = {}
    dirs = {}
    for run in ("emp", "st"):
        dirs[run] = os.path.join(root, run)
        os.makedirs(dirs[run], exist_ok=True)
    wd_e, wd_s = dirs["emp"], dirs["st"]
    man = dict(schema=P.SCHEMA, tier=tier, launch=launch, atom=_txt("atom_name"), k=kref, proxy_nc=pnc, lam_um=lam, n_sub=n_sub,
               period_um=period, dz_um=dz, grid_um=grid, pml_cells=pml, stop_periods=stop_p, est_periods=est_p, fdtd_gpu=gpu,
               field_mask=mask, ramp_periods=ramp, fdtd_stop_auto=sauto, fdtd_stop_tol=stol, lx_snap=snap,
               prof_shift=list(shift), h_lo_um=atom["h_lo"], gap_um=atom["gap"], h_up_um=atom["h_up"], H1_um=atom["H1"],
               H2_um=atom["H2"], spacer_um=atom["spacer"], n_asi=n_asi, fill_lo=atom["fill_lo"], fill_up=atom["fill_up"],
               grey_lo=atom["grey_lo"], grey_up=atom["grey_up"], pixel_um=period / 288.0,
               inputs=dict(atom_json=_txt("atom_json"), atom_job=_txt("atom_job"), rho3_npy=_txt("rho3_npy"),
                           atom_json_md5=md5_file(_txt("atom_json")), rho3_md5=md5_file(_txt("rho3_npy"))))
    if tier == "A":
        ly_cell = ly
        lx_cell = ly
        top = atom["h_lo"] + atom["gap"] + atom["h_up"]
        zmin, zmax = -_num("pad_sub_nm", 650.0) * 1e-3, top + _num("pad_air_nm", 600.0) * 1e-3
        z0_l3 = 0.0
        zR, zT = (-0.45, -0.25), (top + 0.25, top + 0.45)
        layers_extra = []
        # profile files: structure map (atom), n = sqrt(eps)
        nmap_lo, nmap_up = np.sqrt(np.kron(eps_lo, kron)), np.sqrt(np.kron(eps_up, kron))
        nx_cells = ny_cells = ncell_atom
        files = {"LO.dat": nmap_lo, "UP.dat": nmap_up}
        for fn, nm in files.items():
            write_profile(os.path.join(wd_s, fn), nm, nbg, n_asi, shift=shift)
        lay = [(z0_l3, z0_l3 + atom["h_lo"], "prof", ("LO.dat", n_asi)),
               (z0_l3 + atom["h_lo"], z0_l3 + atom["h_lo"] + atom["gap"], "n", nbg),
               (z0_l3 + atom["h_lo"] + atom["gap"], top, "prof", ("UP.dat", n_asi)),
               (top, top + _num("pad_air_nm", 600.0) * 1e-3 + 0.3, "n", 1.0)]
        faces = [z0_l3, z0_l3 + atom["h_lo"], z0_l3 + atom["h_lo"] + atom["gap"], top]
        # kron exactness: block means of the refined permittivity reproduce the source map
        e_back_lo = (np.kron(eps_lo, kron)).reshape(n_src, kref, n_src, kref).mean(axis=(1, 3))
        checks["kron_exact_max_abs"] = float(max(np.abs(e_back_lo - eps_lo).max(),
                                                 np.abs(np.kron(eps_up, kron).reshape(n_src, kref, n_src, kref).mean(axis=(1, 3)) - eps_up).max()))
        z_in_ref, m_inc, orders_R, orders_T, m_t, phi = 0.0, 0, [0], [0], 0, 0.0
        emp_lx = ly_cell if _int("empty_full_cell", 0) else round(thin * grid, 9)
        emp_ly = emp_lx
    else:
        carrier = load_carrier(atom, 3 * nxy)
        ncell_x = 3 * nxy * kref
        grid, lx_cell = snap_cell(period, ncell_x, snap)
        gridy, ly_cell = snap_cell(ly_phys, ncell_atom, snap)
        if abs(grid - gridy) > 1e-12:
            raise SystemExit("%s x and y cells differ: %.9f vs %.9f" % (TAG, grid, gridy))
        ly = ly_cell
        man["grid_um"] = grid
        lx = lx_cell
        z_h1, z_h2 = atom["H1"], atom["H1"] + atom["H2"]
        z_sp = z_h2 + atom["spacer"]
        z_lo0 = z_sp
        top = z_lo0 + atom["h_lo"] + atom["gap"] + atom["h_up"]
        zmin, zmax = -_num("pad_sub_nm", 1000.0) * 1e-3, top + _num("pad_air_nm", 1000.0) * 1e-3
        zR, zT = (-0.45, -0.25), (top + 0.25, top + 0.45)
        # three identical atoms along x (K = 1, distinct_atoms 0), carrier extruded in y
        eps_lo3, eps_up3 = np.concatenate([eps_lo] * 3, axis=0), np.concatenate([eps_up] * 3, axis=0)
        nm = {"L1.dat": np.sqrt(np.kron(carrier["eps1"], kron)), "L2.dat": np.sqrt(np.kron(carrier["eps2"], kron)),
              "LO.dat": np.sqrt(np.kron(eps_lo3, kron)), "UP.dat": np.sqrt(np.kron(eps_up3, kron))}
        if pnc:
            raise SystemExit("%s proxy_nc is Tier A only" % TAG)
        nx_cells, ny_cells = ncell_x, ncell_atom
        for fn, m_ in nm.items():
            write_profile(os.path.join(wd_s, fn), m_, nbg, n_asi, shift=shift)
        lay = [(0.0, z_h1, "prof", ("L1.dat", n_asi)), (z_h1, z_h2, "prof", ("L2.dat", n_asi)),
               (z_h2, z_sp, "n", nbg),
               (z_lo0, z_lo0 + atom["h_lo"], "prof", ("LO.dat", n_asi)),
               (z_lo0 + atom["h_lo"], z_lo0 + atom["h_lo"] + atom["gap"], "n", nbg),
               (z_lo0 + atom["h_lo"] + atom["gap"], top, "prof", ("UP.dat", n_asi)),
               (top, top + _num("pad_air_nm", 1000.0) * 1e-3 + 0.3, "n", 1.0)]
        faces = [0.0, z_h1, z_h2, z_sp, z_lo0 + atom["h_lo"], z_lo0 + atom["h_lo"] + atom["gap"], top]
        checks["kron_exact_max_abs"] = float(max(
            np.abs(np.kron(eps_lo3, kron).reshape(ncell_x // kref, kref, n_src, kref).mean(axis=(1, 3)) - eps_lo3).max(),
            np.abs(np.kron(carrier["eps1"], kron).reshape(ncell_x // kref, kref, n_src, kref).mean(axis=(1, 3)) - carrier["eps1"]).max()))
        man["inputs"]["l12_density"] = carrier["path"]
        man["inputs"]["l12_density_md5"] = md5_file(carrier["path"])
        # Littrow angle from the SNAPPED period so that kx * Lx = 2 pi exactly
        theta = math.degrees(math.asin(lam / (lx_cell * n_sub)))
        phi = theta if launch == "pth" else -theta
        z_in_ref, m_inc = -0.5, (+1 if launch == "pth" else -1)
        orders_R, orders_T, m_t = [-1, 0, 1], [0], 0
        man["theta_deg"] = theta
        man["theta_phys_deg"] = math.degrees(math.asin(lam / (period * n_sub)))
        emp_lx, emp_ly = lx_cell, round(thin * grid, 9)
    # ---- z-node check, domain, monitors
    checks["faces_on_nodes"] = {("%.4f" % z): bool(on_node(z, dz)) for z in faces}
    checks["zmin_on_node"] = bool(on_node(zmin, dz))
    mons = [("R1", zR[0]), ("R2", zR[1]), ("T1", zT[0]), ("T2", zT[1])]
    nz = int(round((zmax - zmin) / dz))
    # ---- the two .ind files
    ids_e = write_ind(os.path.join(wd_e, "emp.ind"), lx=emp_lx, ly=emp_ly, grid=grid, gz=dz, zmin=zmin,
                      zmax=zmax, phi=phi, prefix="emp", layers=[], mons=mons, n_bg=nbg, lam=lam, pml=pml, stop_periods=stop_p,
                      ramp=ramp, sauto=sauto, stol=stol, disp=disp, gpu=gpu, mask=mask)
    ids_s = write_ind(os.path.join(wd_s, "st.ind"), lx=(lx_cell if tier == "B" else ly_cell), ly=ly_cell, grid=grid, gz=dz, zmin=zmin,
                      zmax=zmax, phi=phi, prefix="st", layers=lay, mons=mons, n_bg=nbg, lam=lam, pml=pml, stop_periods=stop_p,
                      ramp=ramp, sauto=sauto, stol=stol, disp=disp, gpu=gpu, mask=mask)
    lx_s = lx_cell if tier == "B" else ly_cell
    # ---- profile read-back (exact-raster check of what the .dat files really carry)
    prof_ok = {}
    for fn in sorted(f for f in os.listdir(wd_s) if f.endswith(".dat") and f[:2] in ("LO", "UP", "L1", "L2")):
        f, nr, nc, x0, x1, y0, y1 = read_profile(os.path.join(wd_s, fn))
        prof_ok[fn] = dict(rows=nr, cols=nc, node_dx=(x1 - x0) / (nr - 1), node_dy=(y1 - y0) / (nc - 1),
                           cells_x=nx_cells, cells_y=ny_cells)
    checks["profiles"] = prof_ok
    # n-map read-back: rebuild the in-file index from f and compare with the n-map written
    rb = {}
    maps = {"LO.dat": np.sqrt(np.kron(eps_lo if tier == "A" else eps_lo3, kron)),
            "UP.dat": np.sqrt(np.kron(eps_up if tier == "A" else eps_up3, kron))}
    if tier == "B":
        maps["L1.dat"] = np.sqrt(np.kron(carrier["eps1"], kron))
        maps["L2.dat"] = np.sqrt(np.kron(carrier["eps2"], kron))
    for fn, nm_ in maps.items():
        f, nr, nc, x0, x1, y0, y1 = read_profile(os.path.join(wd_s, fn))
        pad = 2
        core = f[pad:pad + nm_.shape[0], pad:pad + nm_.shape[1]]
        n_read = nbg + (n_asi - nbg) * core
        wrap_ok = bool(np.allclose(f[:pad, pad:pad + nm_.shape[1]], f[nm_.shape[0]:nm_.shape[0] + pad, pad:pad + nm_.shape[1]], atol=1e-8))
        rb[fn] = dict(max_abs_n_diff=float(np.abs(n_read - nm_).max()), wrap_ok=wrap_ok, shape=list(nm_.shape),
                      node_dx_over_cell=float(((x1 - x0) / (nr - 1)) / (2.0 / nm_.shape[0])))
    checks["profile_readback"] = rb
    checks["lx_over_grid"] = lx_s / grid
    checks["ly_over_grid"] = ly_cell / grid
    checks["cells_expected"] = dict(nx=nx_cells, ny=ny_cells, nx_empty=int(round(emp_lx / grid)), ny_empty=int(round(emp_ly / grid)), nz=nz)
    man.update(lx_um=lx_s, ly_um=ly_cell, nx_cells=nx_cells, ny_cells=ny_cells, nz_cells=nz, m_inc=m_inc, m_t=m_t, orders_R=orders_R,
               orders_T=orders_T, phi_deg=phi, top_um=top, domain_um=[zmin, zmax],
               z=dict(top=top, z_in_ref=z_in_ref, R1=zR[0], R2=zR[1], T1=zT[0], T2=zT[1]),
               n_out=1.0,
               runs=dict(emp=dict(prefix="emp", dir=wd_e, lx_um=emp_lx, ly_um=emp_ly, mon=ids_e),
                         st=dict(prefix="st", dir=wd_s, lx_um=lx_s, ly_um=ly_cell, mon=ids_s)),
               faces_um=faces, checks=checks)
    man["estimate"] = estimate(nx_cells, ny_cells, nz, grid, grid, dz, est_p, lam, b_per_cell=_num("bytes_per_cell", 59.0))
    man["estimate_empty"] = estimate(int(round(emp_lx / grid)), int(round(emp_ly / grid)), nz, grid, grid, dz, est_p / 2.0, lam)
    return man, dirs, checks


# ------------------------------------------------------------------ same-raster grcwa (S1 / S4 code), Tier A only
def rcwa_atom_alone(eps_lo, eps_up, nG, rule, ff, h_lo, gap, h_up, ly, lam, n_sub):
    import grcwa
    grcwa.set_backend("numpy")
    ob = grcwa.obj(int(nG), [ly, 0.0], [0.0, ly], 1.0 / lam, 0.0, 0.0, verbose=0)
    if rule != "laurent" or ff:
        ob.rcwa_rule = rule
        ob.form_factor = int(ff)
    n = eps_lo.shape[0]
    ob.Add_LayerUniform(0.5, n_sub ** 2)
    ob.Add_LayerGrid(h_lo, n, n)
    ob.Add_LayerUniform(gap, n_sub ** 2)
    ob.Add_LayerGrid(h_up, n, n)
    ob.Add_LayerUniform(0.5, 1.0)
    ob.Init_Setup()
    ob.GridLayer_geteps(np.concatenate([eps_lo.reshape(-1), eps_up.reshape(-1)]))
    ob.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    i0 = [i for i, g in enumerate(ob.G) if int(g[0]) == 0 and int(g[1]) == 0][0]
    off = ob.nG
    a, b = ob.GetAmplitudes(0, 0.5)
    r = complex(b[i0 + off]) / complex(a[i0 + off])
    aN, _bN = ob.GetAmplitudes(ob.Layer_N - 1, 0.0)
    t = complex(aN[i0 + off]) / complex(a[i0 + off])
    R, T = ob.RT_Solve(normalize=1)
    return dict(nG_real=int(ob.nG), abs_r=abs(r), arg_r=math.degrees(math.atan2(r.imag, r.real)), abs_t=abs(t),
                arg_t=math.degrees(math.atan2(t.imag, t.real)), R=float(np.real(R)), T=float(np.real(T)))


def run_rcwa_ref(man, atom, out_dir):
    pnc = man["proxy_nc"]
    eps_lo, eps_up = atom["eps_lo"], atom["eps_up"]
    if pnc:
        eps_lo, eps_up = block_avg(eps_lo, pnc), block_avg(eps_up, pnc)
    ly = man["period_um"] / 3.0
    ngs = [int(v) for v in _txt("rcwa_ng", "198,402,594").replace(";", ",").split(",") if v.strip()]
    rows = []
    for ng in ngs:
        for rule, ff in (("laurent", 0), ("liy", 1)):
            t0 = time.time()
            row = rcwa_atom_alone(eps_lo, eps_up, ng, rule, ff, atom["h_lo"], atom["gap"], atom["h_up"], ly, atom["lam"], atom["n_sub"])
            blank = rcwa_atom_alone(np.full_like(eps_lo, atom["n_sub"] ** 2), np.full_like(eps_up, 1.0), ng, rule, ff,
                                    atom["h_lo"], atom["gap"], atom["h_up"], ly, atom["lam"], atom["n_sub"])
            row.update(nG1=ng, rule=rule, ff=ff, d_arg_r=P.wrap180(row["arg_r"] - blank["arg_r"]),
                       d_arg_t=P.wrap180(row["arg_t"] - blank["arg_t"]), blank_arg_r=blank["arg_r"], blank_arg_t=blank["arg_t"],
                       s=time.time() - t0)
            rows.append(row)
            log("rcwa_ref nG1 %d %-7s ff%d |r| %.5f arg r %+8.3f d arg r %+8.3f d arg t %+8.3f (%.0f s)"
                % (ng, rule, ff, row["abs_r"], row["arg_r"], row["d_arg_r"], row["d_arg_t"], row["s"]))
    P.save_json(os.path.join(out_dir, "rcwa_ref.json"), dict(atom=man["atom"], proxy_nc=pnc, rows=rows))
    return rows


# ------------------------------------------------------------------ run
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
        print("usage: Main_fdtd_atom_verify.py <job_file>")
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
    log("job %s host %s" % (sys.argv[1], socket.gethostname()))
    rr = out_root()
    run_root = _txt("run_root")
    os.makedirs(run_root, exist_ok=True)
    pin_cores()
    man, dirs, checks = build(run_root)
    man.update(driver=os.path.abspath(__file__), driver_md5=md5_file(os.path.abspath(__file__)),
               post_md5=md5_file(os.path.abspath(P.__file__)), host=os.environ.get("COMPUTERNAME", "?"),
               started=time.strftime("%F %T"), job_file=sys.argv[1], dry_run=_int("dry_run", 0))
    man["manifest_core_md5"] = hashlib.md5(json.dumps({k: man[k] for k in ("tier", "launch", "atom", "k", "proxy_nc", "grid_um", "lx_um",
                                                                          "ly_um", "dz_um", "z", "inputs", "stop_periods")},
                                                      sort_keys=True).encode("ascii")).hexdigest()
    mpath = os.path.join(rr, "fdtd_atom_manifest.json")
    P.save_json(mpath, man)
    est = man["estimate"]
    log("%s tier %s launch %s k %d  cell %d x %d x %d  grid %.4f nm dz %.3f nm  est %.1f M cells, ~%d steps, GPU %.1f min, CPU %.1f h, RSS %.2f GB"
        % (man["atom"], man["tier"], man["launch"], man["k"], man["nx_cells"], man["ny_cells"], man["nz_cells"], man["grid_um"] * 1e3,
           man["dz_um"] * 1e3, est["cells"] / 1e6, est["steps"], est["gpu_min"], est["cpu_h"], est["rss_gb"]))
    log("raster checks: kron_exact %.1e, lx/grid %.12f, ly/grid %.12f, faces on nodes %s, profile read-back max |dn| %s"
        % (checks["kron_exact_max_abs"], checks["lx_over_grid"], checks["ly_over_grid"], all(checks["faces_on_nodes"].values()),
           {k: "%.1e" % v["max_abs_n_diff"] for k, v in checks["profile_readback"].items()}))
    bad = (checks["kron_exact_max_abs"] > 1e-12
           or (man["lx_snap"] and abs(checks["lx_over_grid"] - round(checks["lx_over_grid"])) > 1e-6)
           or not all(checks["faces_on_nodes"].values()) or not checks["zmin_on_node"]
           or any(v["max_abs_n_diff"] > 1e-8 or not v["wrap_ok"] for v in checks["profile_readback"].values()))
    if bad:
        log("RASTER CHECK FAILED -- refusing to run")
        return 2
    if _int("dry_run", 0):
        log("dry_run: files written under %s, manifest %s" % (run_root, mpath))
        return 0
    # ---- resume
    rpath = os.path.join(rr, "fdtd_atom_result.json")
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
    if est["cpu_h"] * 3600.0 > _num("max_run_s", 1e9) and not man["fdtd_gpu"]:
        log("SKIPPED: estimated CPU time %.1f h > max_run_s" % est["cpu_h"])
        return 2
    import Main_fdtd_input2d_verify as F                      # noqa: E402  (run_fullwave only; its main() is not run)
    F.CFG = CFG
    tmo, npr = _num("rsoft_timeout_s", 7200.0), _int("fdtd_mpi_np", 1)
    runs = {}
    for run in ("emp", "st"):
        res = F.run_fullwave(dirs[run], run + ".ind", timeout=tmo, mpi_np=npr)
        facts = {k: v for k, v in res.items() if k != "log_tail"}
        runs[run] = facts
        lf = res.get("logfacts") or {}
        log("%s run ok=%s rc=%s %.1f s peak %.2f GB  NX,NY,NZ %s,%s,%s  NT %s  converged %s" % (
            run, res.get("ok"), res.get("rc"), res.get("elapsed", 0.0), res.get("peak_rss_gb") or 0.0, lf.get("nx"), lf.get("ny"),
            lf.get("nz"), lf.get("nt_run"), lf.get("converged")))
        if not res.get("ok"):
            P.save_json(rpath, dict(ok=False, error="%s run failed: %s" % (run, res.get("why")), runs=runs,
                                    manifest_core_md5=man["manifest_core_md5"], log_tail=res.get("log_tail")))
            return 1
        exp = checks["cells_expected"]
        want = (exp["nx"], exp["ny"]) if run == "st" else (exp["nx_empty"], exp["ny_empty"])
        facts["cells_ok"] = bool(lf.get("nx") == want[0] + 3 and lf.get("ny") == want[1] + 3)
        if not facts["cells_ok"]:
            log("**WARNING cells_ok False: log NX,NY %s,%s expected %d,%d (+3 ghost) -- the raster is NOT exact, result VOID**"
                % (lf.get("nx"), lf.get("ny"), want[0] + 3, want[1] + 3))
    try:                                                         # a missing / ragged monitor file must leave a result (ok false), not a bare traceback
        orders = P.extract_orders(man, {"emp": dirs["emp"], "st": dirs["st"]})
    except Exception as e:
        log("EXTRACTION FAILED: %s: %s" % (type(e).__name__, str(e)[:300]))
        P.save_json(rpath, dict(ok=False, error="extraction failed: %s: %s" % (type(e).__name__, str(e)[:300]), runs=runs,
                                manifest_core_md5=man["manifest_core_md5"]))
        return 1
    P.save_json(os.path.join(rr, "fdtd_atom_orders.json"), orders)
    try:
        result = P.analyse(man, orders, logfacts={r: runs[r].get("logfacts") for r in runs})
    except Exception as e:                                       # the orders are saved: the analysis can be redone with fdtd_atom_post.py
        log("ANALYSIS FAILED (orders saved in fdtd_atom_orders.json): %s: %s" % (type(e).__name__, str(e)[:300]))
        P.save_json(rpath, dict(ok=False, error="analysis failed: %s: %s" % (type(e).__name__, str(e)[:300]), runs=runs,
                                manifest_core_md5=man["manifest_core_md5"]))
        return 1
    cells_ok = all(r.get("cells_ok", False) for r in runs.values())
    converged = all(bool((runs[r].get("logfacts") or {}).get("converged")) for r in runs)      # BOTH runs (the empty run too)
    result.update(ok=bool(cells_ok and not result["flags"]), runs=runs, manifest_core_md5=man["manifest_core_md5"],
                  cells_ok=cells_ok, converged=converged)
    if not result["ok"]:
        log("RESULT NOT OK -- cells_ok %s, quality flags: %s" % (cells_ok, result["flags"] if result["flags"] else "none"))
    if man["tier"] == "A":
        tmm = P.tmm_blank_tierA(man["h_lo_um"], man["gap_um"], man["h_up_um"], man["n_sub"], man["lam_um"])
        r0, t0 = result["refl"][0]["r_E"], result["trans"][0]["t_E"]
        result["vs_tmm_blank"] = dict(d_arg_r=P.wrap180(P.deg(r0 / tmm["r_E"])), d_arg_t=P.wrap180(P.deg(t0 / tmm["t_E"])),
                                      d_abs_r=abs(r0) - abs(tmm["r_E"]), blank_argR00_deg=(P.deg(tmm["r_E"]) + 180.0) % 360.0)
    P.save_json(rpath, result)
    log("result: %s" % json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in result["record"].items()
                                   if not isinstance(v, str)}, sort_keys=True))
    if result.get("vs_tmm_blank"):
        log("vs analytic blank: %s" % json.dumps(result["vs_tmm_blank"], sort_keys=True))
    if _int("rcwa_ref", 0) and man["tier"] == "A":
        try:
            run_rcwa_ref(man, load_atom(), rr)
        except Exception as e:                               # the FDTD result is already saved: never fail the job over the reference
            log("rcwa_ref FAILED (the FDTD result above is saved): %s: %s" % (type(e).__name__, str(e)[:200]))
    man["finished"] = time.strftime("%F %T")
    P.save_json(mpath, man)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
