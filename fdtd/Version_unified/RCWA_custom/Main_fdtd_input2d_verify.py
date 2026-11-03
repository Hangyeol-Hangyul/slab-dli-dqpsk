# -*- coding: utf-8 -*-
r"""Main_fdtd_input2d_verify.py -- independent-solver check (FullWAVE 3-D FDTD, forward only) of the
2-D crossed-lane INPUT grating of Main_rcwa_input2d.py (SCDC, 261006).

Why
---
The 2-D input grating is optimised with grcwa (Laurent factorisation only). For p-polarised crossed
gratings Laurent converges slowly and the optimiser can find truncation-artefact optima. FDTD errors
(grid staircasing, numerical dispersion) are uncorrelated with RCWA truncation, so AGREEMENT between
the two is the information. Forward only (an FDTD adjoint would not verify the grcwa VJP).

Model
-----
Square cell P x P (job `period_nm`), periodic in x and y. +z is the propagation direction. From the
incidence side: air | L2 (material2 / fill2, raster rho2) | L1 (material1 / fill1, raster rho1) |
substrate (n_sub, semi-infinite). Normal incidence from air at `lam_nm`, two launches: E_x and E_y.
Every thickness, index and fill is read from the stack keys of the design (input2d.json `key_*`
entries, else the job_*.txt in the folder, else the default of the input2d driver); a key of THIS
job file overrides both. Nothing about the stack is hard coded here.

Raster -> FDTD map: rho[i, j] is the permittivity at x = x0 + i*dx, y = y0 + j*dy, i.e. row = x,
column = y, no flip, no transpose (RSoft 2-D data file rows are X, columns are Y; validated 261006 with
the CAD index profile, bsimw32 index_profile=1, no license). The area-conserving resample to the FDTD
grid is centred on the nodes and permittivity-averaged; the datafile carries n(x, y).
PROFILE X-SHIFT (261006 measurement, `prof_shift_x` default 0.5 cell): FullWAVE samples a user-profile
datafile half a cell off in x relative to y (CAD index profile does not show it). With shift 0 a 1-D
x-grating under E_x and the same grating rotated to y under E_y differ by 2-3 pp in the diffracted
orders (R 0.0641 vs 0.0555 at 64 cells); shifting the file by +-0.5 cell in x makes them agree
(R 0.05546 both, (0,0) 0.03248 both) and the 2-D smoke design Ex / Ey runs then agree to 1e-3. The
shift is a calibration of the profile reader, NOT a physical translation; C-c (`mode = compare`) is the
standing check and must be run on every grid before a number is quoted.

Output per launch
-----------------
Power into every propagating substrate order ((0,0), (+-1,0), (0,+-1)) split p / s with respect to
each order's plane of incidence (from the tangential E of the order only, so no H field is needed),
the reflected (0,0) field, the 2 x 2 Jones matrix of the reflection and its worst-SOP power (largest
singular value squared, G3), G1 (p power Ex -> (+1,0), Ey -> (0,+1)), G2 (total leak into the other
lane's order), energy balance, and the S1 / S2 monitor spread (a standing wave from the exit PML).
Echo / relaunch (G4a / G4b, substrate-side launch at 54.69 deg) are NOT implemented here. The project
notes (260908 C1b) record a tilted-incidence empty-domain loss of 1.85e-02 in power in
this FullWAVE build, 9x the 0.0020 gate, so a tilted FDTD echo number could not resolve the gate.

Cost and run length (measured 261006, single process, about 1.1 cores, memory bound, ~5.7e7
cell-updates/s; fwmpirun was 1.4x faster at 4 ranks and its ranks escape CPU affinity, so it is NOT used):
  * The smoke design rings down with tau ~ 16 periods (Q ~ 50): at 64 cells eta_x = 0.5315 (40 periods),
    0.4957 (80), 0.4983 (160), 0.4985 (240). Use >= 160 periods (error ~5e-4); 40 periods is WRONG by 3e-2.
    FullWAVE's own convergence stop at the default tolerance stopped at 63 periods with 1e-2 error; at
    fdtd_stop_tol = 1e-5 it stopped at 226 periods (error 1e-4). `fdtd_stop_auto = 0` (fixed time) is the
    default here; the held jobs use 2 (convergence or the limit) with tol 1e-5.
  * wall per launch at 160 periods: 64 cells 4-6 min, 80 cells 12-14 min, 96 cells ~25 min (est),
    128 cells ~45-60 min (est), 192 cells ~5 h (est). peak RAM 0.08 / 0.14 / 0.2 / 0.5 / ~1.5 GB.
  * sigma_d symmetry: the Ey launch only checks the Ex launch. A finalist needs the Ex launch alone
    (`launches = Ex`): gates_from() then mirrors the Jones matrix, J = [[a, b], [b, a]] (flag
    assumed_sigma_d_mirror); the lane-y numbers equal the lane-x numbers up to the ~1e-3 residual that
    C-c measures.

Modes (job key `mode`)
----------------------
selftest   no RSoft: TMM two ways, synthetic-field order extraction (needle test), resample
tmm        analytic table of the uniform stacks (no RSoft)
uniform    C-a: uniform rasters (all 0 / all 1 per layer) in a thin cell vs the analytic transfer matrix
cell       a design: grids x launches; writes fdtd_input2d.json (resume = 1 skips finished runs)
compare    C-b / C-c / C-d tables from fdtd_input2d.json files (no RSoft)

Job keys (all optional unless stated): cell_folder (or rho1_npy + rho2_npy), grid_cells ("128, 192"),
launches ("Ex, Ey"), pad_air_nm, pad_sub_nm, mon_dz_nm, pml_cells, stop_periods, ramp_periods,
result_folder (or Output_folder + subfolder_name), run_root, pin_cores (first logical core of the
pinned range, direct runs only), max_ram_pct (default: env SIM_MAX_RAM_PCT, else off), max_run_s (a run estimated to
take longer is skipped and reported), dry_run, resume, fdtd_dispersion, thin_cells.

ASCII-only source and console output (cp949 consoles). Every open() has encoding="utf-8".
"""
import glob
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_GT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "solver", "Version_unified", "RCWA_custom"))
for _p in (_HERE, os.path.join(_HERE, "_vendor"), os.path.dirname(_HERE), _GT,
           os.path.join(_GT, "_vendor"), os.path.dirname(_GT)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                            # noqa: E402

TAG = "[FDTD-IN2D]"
FULLWAVE = "fullwave.exe"            # by PATH, never an absolute vendor path
CFG = {}


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


def _ints(k, d):
    return [int(float(v)) for v in _txt(k, d).replace(";", ",").split(",") if v.strip()]


def stop_kw():
    """fdtd_stop_auto: 0 = fixed stop time (default), 1 = convergence / no limit, 2 = convergence or the
    stop time, whichever first. fdtd_stop_tol only when given."""
    kw = dict(stop_auto=_int("fdtd_stop_auto", 0))
    if _txt("fdtd_stop_tol", "") != "":
        kw["stop_tol"] = _num("fdtd_stop_tol", 1e-4)
    return kw


# ------------------------------------------------------------------ materials / stack
_G = None


def mats():
    """scd_c_grating is the single source of the material table (same one the optimiser used)."""
    global _G
    if _G is None:
        import scd_c_grating as G                              # noqa: E402
        _G = G
    return _G


def n_of(name):
    G = mats()
    e = complex(G.eps_of(name))
    if abs(e.imag) > 1e-12:
        raise ValueError("material %r is absorbing (eps %s); this verifier is lossless only" % (name, e))
    return float(math.sqrt(e.real))


STACK_DEFAULTS = dict(period_nm=1315.39, n_sub=1.4440, lam_nm=1550.0, H1_nm=525.0, H2_nm=450.0,
                      material1="a-Si", material2="Si3N4", fill1="SiO2", fill2="air", nxy=256)
_STACK_NUM = ("period_nm", "n_sub", "lam_nm", "H1_nm", "H2_nm", "nxy")


def read_job_txt(path):
    d = {}
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        ln = ln.split("#")[0]
        if "=" in ln:
            k, _, v = ln.partition("=")
            d[k.strip()] = v.strip()
    return d


def load_stack(folder):
    """Stack keys: defaults < job_*.txt < input2d.json key_* < this job's own keys."""
    src = {}
    prov = {k: "default" for k in STACK_DEFAULTS}
    vals = dict(STACK_DEFAULTS)
    if folder:
        jt = sorted(glob.glob(os.path.join(folder, "job_*.txt")))
        if jt:
            d = read_job_txt(jt[0])
            for k in STACK_DEFAULTS:
                if k in d and d[k] != "":
                    vals[k] = d[k]
                    prov[k] = "job_txt"
            src["job_txt"] = jt[0]
        jp = os.path.join(folder, "input2d.json")
        if os.path.isfile(jp):
            j = json.load(io.open(jp, encoding="utf-8"))
            for k in STACK_DEFAULTS:
                if ("key_" + k) in j and j["key_" + k] is not None:
                    vals[k] = j["key_" + k]
                    prov[k] = "input2d.json"
            src["input2d_json"] = jp
    for k in STACK_DEFAULTS:
        if _txt(k, "") != "":
            vals[k] = _txt(k, "")
            prov[k] = "verify_job"
    st = {}
    for k in _STACK_NUM:
        st[k] = float(vals[k])
    for k in ("material1", "material2", "fill1", "fill2"):
        st[k] = str(vals[k])
    st["nxy"] = int(round(st["nxy"]))
    st["P"] = st["period_nm"] * 1e-3
    st["lam"] = st["lam_nm"] * 1e-3
    st["H1"] = st["H1_nm"] * 1e-3
    st["H2"] = st["H2_nm"] * 1e-3
    st["n1s"], st["n1f"] = n_of(st["material1"]), n_of(st["fill1"])
    st["n2s"], st["n2f"] = n_of(st["material2"]), n_of(st["fill2"])
    st["n_sub"] = float(st["n_sub"])
    st["provenance"] = prov
    st["sources"] = src
    st["theta_sub_deg"] = math.degrees(math.asin(st["lam"] / (st["P"] * st["n_sub"])))
    return st


# ------------------------------------------------------------------ analytic reference (normal incidence)
def tmm_recursive(layers, n_in, n_out, lam):
    """Recursive Fresnel, exp(-i w t): returns complex (r, t) referenced to the first / last interface.
    layers = [(n, d_um), ...] from the incidence side."""
    k0 = 2.0 * math.pi / lam
    ns = [n_in] + [n for n, _ in layers] + [n_out]
    ds = [d for _, d in layers]
    nif = len(ns) - 1
    R, T = 0.0 + 0j, 1.0 + 0j
    for j in range(nif - 1, -1, -1):
        rj = (ns[j] - ns[j + 1]) / (ns[j] + ns[j + 1])
        tj = 2.0 * ns[j] / (ns[j] + ns[j + 1])
        if j == nif - 1:
            R, T = rj, tj
        else:
            ph = k0 * ns[j + 1] * ds[j]
            e1, e2 = np.exp(1j * ph), np.exp(2j * ph)
            den = 1.0 + rj * R * e2
            T = tj * T * e1 / den
            R = (rj + R * e2) / den
    return complex(R), complex(T)


def tmm_charmatrix(layers, n_in, n_out, lam):
    """Characteristic-matrix form (Macleod, exp(-i w t)): same quantities, independent algebra."""
    k0 = 2.0 * math.pi / lam
    M = np.eye(2, dtype=complex)
    for n, d in layers:
        dl = k0 * n * d
        M = M @ np.array([[np.cos(dl), -1j * np.sin(dl) / n], [-1j * n * np.sin(dl), np.cos(dl)]])
    B, C = M @ np.array([1.0, n_out], dtype=complex)
    r = (n_in * B - C) / (n_in * B + C)
    t = 2.0 * n_in / (n_in * B + C)
    return complex(r), complex(t)


def uniform_layers(st, a1, a2):
    """Layer list (n, d) for uniform rasters a1 (L1), a2 (L2) in {0, 1} (or any area fraction)."""
    n2 = n_of_mix(st["n2f"], st["n2s"], a2)
    n1 = n_of_mix(st["n1f"], st["n1s"], a1)
    return [(n2, st["H2"]), (n1, st["H1"])]


def n_of_mix(nf, ns, a):
    return math.sqrt(nf * nf + a * (ns * ns - nf * nf))


# ------------------------------------------------------------------ raster -> FDTD grid
def resample_matrix(n_src, n_dst):
    """Periodic area-overlap matrix W (n_dst x n_src). Source pixel i is the box [i-1/2, i+1/2) (the
    pixel value sits at x_i = i*dx, grcwa's sample convention), destination cell g the box
    [(g-1/2)*r, (g+1/2)*r) with r = n_src/n_dst, both in source-pixel units. Rows sum to 1."""
    r = n_src / float(n_dst)
    W = np.zeros((n_dst, n_src))
    for g in range(n_dst):
        lo, hi = (g - 0.5) * r, (g + 0.5) * r
        i0, i1 = int(math.floor(lo + 0.5)), int(math.ceil(hi - 0.5))
        for i in range(i0, i1 + 1):
            a, b = max(lo, i - 0.5), min(hi, i + 0.5)
            if b > a:
                W[g, i % n_src] += (b - a) / r
    return W


def resample_area(rho, n_dst):
    rho = np.asarray(rho, float)
    if rho.shape[0] == n_dst and rho.shape[1] == n_dst:
        return rho.copy()
    W = resample_matrix(rho.shape[0], n_dst)
    return W @ rho @ W.T


def index_map(frac, nf, ns):
    """Cell index from the area fraction of solid: permittivity average (first-order subpixel rule)."""
    return np.sqrt(nf * nf + frac * (ns * ns - nf * nf))


def write_profile(path, n_map, n_bg, n_ref, shift=(0.0, 0.0), pad=2):
    """RSoft user-profile datafile for a segment with delta = n_ref - n_bg: n = n_bg + delta * f,
    f = (n_map - n_bg) / delta. Node k of an axis sits at x' = -1 + 2*(k + shift)/N on [-1, 1] (x' = x /
    (P/2)); the periodic extension (`pad` cells each side) keeps the data defined across the whole
    domain when shifted (outside the data domain RSoft sets the profile to ZERO). shift is in cells.
    Rows = X, columns = Y."""
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


# ------------------------------------------------------------------ .ind writer
_IND_HEAD = r"""
Lx = {lx:.9f}
Ly = {ly:.9f}
GridS = {grid:.9f}
alpha = 0
background_index = 1.000000
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
fdtd_bc_ka_auto = 0
fdtd_bc_ka_n0 = 1.000000
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
{stol}fdtd_stop_time = {stop:.4f}
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
grid_size_z = GridS
height = Ly
index = 1
index_profile = 1
k0 = (2*pi)/free_space_wavelength
lambda = free_space_wavelength
launch_align_file = 1
launch_angle = 0
launch_phase_ref = 1
launch_position = 0
launch_position_y = 0
launch_position_z = {zmin:.9f}
launch_tilt = 1
launch_type = LAUNCH_PLANEWAVE
launch_width = inf
launch_height = inf
launch_neff = 1.000000
polarization = {pol:d}
prefix = {prefix}
sim_tool = ST_FULLWAVE
slice_position_z = 0
slice_step_size = GridS
step_size = GridS
structure = STRUCT_CHANNEL
width = Lx
"""

_SEG = r"""
segment {n}
    color = {color}
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

_MON = r"""
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

MASK_EXEY = 2 + 4                     # Ex = 2, Ey = 4 (5.D.1 order: Default, Ex, Ey, Ez, Hx, Hy, Hz)


def geometry(st, pad_air, pad_sub, mon_dz, off_a=0.70, off_s=0.55):
    """z layout (um): air/L2 interface at z = 0, light travels +z."""
    ztop = st["H2"] + st["H1"]
    g = dict(zmin=-pad_air, zmax=ztop + pad_sub, ztop=ztop, z_l2=(0.0, st["H2"]), z_l1=(st["H2"], ztop))
    g["zA1"] = -off_a
    g["zA2"] = -off_a + mon_dz
    g["zS1"] = ztop + off_s
    g["zS2"] = ztop + off_s + mon_dz
    if g["zA2"] > -0.05 or g["zS2"] > g["zmax"] - 0.3:
        raise ValueError("monitor layout does not fit: zA2 %.3f, zS2 %.3f, zmax %.3f" % (g["zA2"], g["zS2"], g["zmax"]))
    return g


def build_ind(path, st, geo, *, lx, ly, grid, pol, prefix, empty, files, pml=12, stop_periods=40.0,
              ramp=3.0, disp=0, mask=MASK_EXEY, stop_auto=0, stop_tol=None):
    """One .ind. `files` = {'L2': (datafile name, n_ref), 'L1': (...)} or None for a flat layer given by
    ('const', n). empty=True: no structure at all (all air)."""
    lam = st["lam"]
    body = _IND_HEAD.format(lx=lx, ly=ly, grid=grid, zmax=geo["zmax"], zmin=geo["zmin"], disp=int(disp),
                            pml=int(pml), ramp=ramp, stop=stop_periods * lam, lam=lam, pol=int(pol),
                            prefix=prefix, sauto=int(stop_auto),
                            stol=("" if stop_tol is None else "fdtd_stop_tol = %.3e\n" % stop_tol))
    if _int("fdtd_gpu", 0) == 1:     # [261007] RSoft GPU FDTD (via job key GPU_on = 1); default 0 = .ind unchanged
        body += "fdtd_gpu = 1\n"
    seg = 1
    if not empty:
        # substrate first (semi-infinite: runs to the domain end; the PML continues it)
        body += _SEG.format(n=seg, color=1, z0=geo["ztop"], z1=geo["zmax"] + 0.2,
                            extra="delta = %.9f" % (st["n_sub"] - 1.0))
        seg += 1
        prof_no = 0
        for key, (za, zb) in (("L2", geo["z_l2"]), ("L1", geo["z_l1"])):
            kind, val = files[key]
            if kind == "const":
                extra = "delta = %.9f" % (val - 1.0)
            else:
                prof_no += 1
                extra = "profile_type = PROF_USER_%d\n    delta = %.9f" % (prof_no, val[1] - 1.0)
            body += _SEG.format(n=seg, color=seg % 15 + 1, z0=za, z1=zb, extra=extra)
            seg += 1
        # user_profile blocks (numbered in the order the layers use them)
        blocks = ""
        k = 0
        for key in ("L2", "L1"):
            kind, val = files[key]
            if kind != "const":
                k += 1
                blocks += "\nuser_profile %d\n    type = UF_DATAFILE\n    filename = %s\nend user_profile\n" % (k, val[0])
        body += blocks
    mons = []
    for name, z in (("A1", geo["zA1"]), ("A2", geo["zA2"]), ("S1", geo["zS1"]), ("S2", geo["zS2"])):
        body += _MON.format(n=seg, z=z, mask=mask)
        mons.append((name, seg))
        seg += 1
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    return dict(mon=dict(mons))


# ------------------------------------------------------------------ running FullWAVE
def _rss_tree(pid):
    try:
        import psutil
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
        return sum(q.memory_info().rss for q in procs if q.is_running()), len(procs)
    except Exception:
        return 0, 0


def _my_ranks(workdir):
    """fullwave.exe processes whose working directory is `workdir` (the MPI ranks are spawned by the
    Hydra proxy, NOT by this process, so they are not in our process tree and do not inherit affinity)."""
    import psutil
    out = []
    want = os.path.normcase(os.path.abspath(workdir))
    for q in psutil.process_iter(["pid", "name"]):
        if (q.info["name"] or "").lower() == "fullwave.exe":
            try:
                if os.path.normcase(os.path.abspath(q.cwd())) == want:
                    out.append(q)
            except Exception:
                pass
    return out


def run_fullwave(workdir, ind_name, timeout=7200, max_ram_gb=None, mpi_np=1):
    """Launch FullWAVE and sample its memory / CPU while it runs.

    mpi_np <= 1: one fullwave.exe, a child of this process (inherits this process's CPU affinity).
    mpi_np  > 1: fwmpirun -np N. The ranks are started by the Hydra proxy (parent outside this tree,
    affinity = all cores), so each rank is pinned to THIS process's affinity set within ~0.05 s of its
    appearance (polling every 20 ms); the pin time is reported (`pin_delay_s`). The cluster-wide mpiexec
    init lock (core.rsoft_utils) is held until all ranks are up and pinned."""
    from core.rsoft_utils import (_rsoft_stagger, _check_rsoft_success_fullwave,                 # noqa: E402
                                  _mpiexec_lock_acquire, _mpiexec_lock_release)
    import re
    import psutil
    logf = os.path.join(workdir, "log.txt")
    prev = os.path.getsize(logf) if os.path.exists(logf) else 0
    mine = sorted(psutil.Process().cpu_affinity())
    use_mpi = int(mpi_np) > 1
    if use_mpi:
        _mpiexec_lock_acquire()
    else:
        _rsoft_stagger()
    out_f = io.open(os.path.join(workdir, "stdout.txt"), "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    cmd = (["fwmpirun.exe", "-np%d" % int(mpi_np), "-hide", ind_name] if use_mpi else [FULLWAVE, "-hide", ind_name])
    try:
        proc = subprocess.Popen(cmd, cwd=workdir, stdout=out_f, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        out_f.close()
        if use_mpi:
            _mpiexec_lock_release()
        return dict(ok=False, why="%s not on PATH" % cmd[0], elapsed=0.0)
    st = dict(peak=0, nproc=0, aff=None, nthr=0, pinned={}, cpu0=None, cpu1=0.0, killed=False, released=False)
    stop = threading.Event()

    def watch():
        try:
            pr = psutil.Process(proc.pid)
        except Exception:
            pr = None
        seen_cpu = {}
        while not stop.is_set():
            try:
                if use_mpi:
                    ranks = _my_ranks(workdir)
                    for q in ranks:
                        if q.pid not in st["pinned"]:
                            try:
                                q.cpu_affinity(mine)
                                st["pinned"][q.pid] = max(0.0, time.time() - q.create_time())
                            except Exception:
                                pass
                    if (not st["released"]) and len(st["pinned"]) >= int(mpi_np):
                        _mpiexec_lock_release()
                        st["released"] = True
                    procs = ranks
                    rss = sum(q.memory_info().rss for q in procs)
                    for q in procs:
                        try:
                            ct = q.cpu_times()
                            seen_cpu[q.pid] = ct.user + ct.system
                        except Exception:
                            pass
                    st["cpu1"] = sum(seen_cpu.values())
                    if st["cpu0"] is None and procs:
                        st["cpu0"] = (time.time(), st["cpu1"])
                    st["nproc"] = max(st["nproc"], len(procs))
                    if st["aff"] is None and procs:
                        st["aff"] = sorted(procs[0].cpu_affinity())
                else:
                    rss, npr = _rss_tree(proc.pid)
                    st["nproc"] = max(st["nproc"], npr)
                    if pr is not None:
                        ct = pr.cpu_times()
                        st["cpu1"] = ct.user + ct.system
                        if st["cpu0"] is None:
                            st["cpu0"] = (time.time(), st["cpu1"])
                        if st["aff"] is None:
                            st["aff"] = sorted(pr.cpu_affinity())
                        st["nthr"] = max(st["nthr"], pr.num_threads())
                st["peak"] = max(st["peak"], rss)
                if max_ram_gb is not None and st["peak"] / 1e9 > max_ram_gb:
                    for q in (_my_ranks(workdir) if use_mpi else []):
                        try:
                            q.kill()
                        except Exception:
                            pass
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    st["killed"] = True
                    break
            except Exception:
                pass
            stop.wait(0.02 if (use_mpi and len(st["pinned"]) < int(mpi_np)) else 1.0)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        proc.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        for q in (_my_ranks(workdir) if use_mpi else []):
            try:
                q.kill()
            except Exception:
                pass
        timed_out = True
    stop.set()
    th.join(5.0)
    if use_mpi and not st["released"]:
        _mpiexec_lock_release()
    out_f.close()
    el = time.time() - t0
    ok = (not timed_out) and (not st["killed"]) and _check_rsoft_success_fullwave(workdir, prev, n_expected=1)
    cr = None
    if st["cpu0"] is not None and el > 2.0:
        cr = (st["cpu1"] - st["cpu0"][1]) / max(el - (st["cpu0"][0] - t0), 1e-6)
    newlog = ""
    tail = ""
    try:
        with open(logf, "rb") as fh:
            fh.seek(prev)
            newlog = fh.read().decode("utf-8", errors="ignore")
        tail = newlog[-1500:]
    except Exception:
        pass
    info = {}
    m = re.search(r"NX=(\d+) NY=(\d+) NZ=(\d+), NT=(\d+)", newlog)
    if m:
        info.update(nx=int(m.group(1)), ny=int(m.group(2)), nz=int(m.group(3)), nt=int(m.group(4)))
    m = re.search(r"FDTD Auto Time Step = ([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", newlog)
    if m:
        info["dt_um"] = float(m.group(1))
    m = re.search(r"Converged at L=(\d+)\.\s+New NT=(\d+), Stop Time = ([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", newlog)
    if m:
        info.update(converged=True, conv_L=int(m.group(1)), nt_run=int(m.group(2)), stop_time_um=float(m.group(3)))
    elif "nt" in info and "dt_um" in info:
        info.update(converged=False, nt_run=info["nt"], stop_time_um=info["nt"] * info["dt_um"])
    m = re.search(r"Actual Computation Time = ([0-9.]+) min", newlog)
    if m:
        info["compute_min"] = float(m.group(1))
    return dict(ok=ok, rc=proc.returncode, elapsed=el, peak_rss_gb=st["peak"] / 1e9, nproc=st["nproc"], logfacts=info,
                cpu_ratio=cr, affinity=st["aff"], nthreads=st["nthr"], timed_out=timed_out, killed=st["killed"],
                mpi_np=int(mpi_np), pin_delay_s=(max(st["pinned"].values()) if st["pinned"] else None),
                pinned_n=len(st["pinned"]), log_tail=tail,
                why="" if ok else "no 'Actual Computation Time' in log.txt / timeout / killed")


# ------------------------------------------------------------------ monitor files / orders
def load_field(path):
    """RSoft 2-D complex amp/phase file -> (cx[nx, ny], xg, yg). Rows = X, columns = Y."""
    with io.open(path, encoding="utf-8") as f:
        hdr = [f.readline() for _ in range(4)]
    p3, p4 = hdr[2].split(), hdr[3].split()
    nx, xmin, xmax = int(p3[0]), float(p3[1]), float(p3[2])
    ny, ymin, ymax = int(p4[0]), float(p4[1]), float(p4[2])
    raw = np.loadtxt(path, skiprows=4)
    raw = np.atleast_2d(raw)
    cx = raw[:, 0::2] * np.exp(1j * np.radians(raw[:, 1::2]))
    return cx, np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)


def find_dft(workdir, prefix, mon, comp):
    hits = sorted(glob.glob(os.path.join(workdir, "%s_m%d_f*_%s.dat" % (prefix, mon, comp))))
    if not hits:
        hits = sorted(glob.glob(os.path.join(workdir, "%s_m%d_*_%s.dat" % (prefix, mon, comp))))
    return hits[0] if hits else None


def periodic_crop(cx, xg, yg, lx, ly):
    """Drop a duplicated closing row / column when the file includes both edges of the period."""
    nx, ny = cx.shape
    dxg = (xg[-1] - xg[0]) / max(nx - 1, 1)
    dyg = (yg[-1] - yg[0]) / max(ny - 1, 1)
    if abs((xg[-1] - xg[0]) - lx) < 0.25 * dxg:
        cx, xg = cx[:-1, :], xg[:-1]
    if abs((yg[-1] - yg[0]) - ly) < 0.25 * dyg:
        cx, yg = cx[:, :-1], yg[:-1]
    return cx, xg, yg


def order_coeff(cx, xg, yg, m, n, lx, ly, sigma):
    """c_mn = <E exp(-i sigma (m Gx x + n Gy y))> over the (cropped) period."""
    kx = 2.0 * math.pi * m / lx
    ky = 2.0 * math.pi * n / ly
    ph = np.exp(-1j * sigma * (kx * xg[:, None] + ky * yg[None, :]))
    return complex(np.mean(cx * ph))


def prop_orders(P, lam, n_med, kmax=3):
    out = []
    for m in range(-kmax, kmax + 1):
        for n in range(-kmax, kmax + 1):
            q2 = (lam / P) ** 2 * (m * m + n * n)
            if q2 < n_med ** 2 - 1e-12:
                out.append((m, n))
    return out


def order_power(cxo, cyo, m, n, P, lam, n_med, E0abs, n_in=1.0):
    """Power into order (m, n) relative to the incident power flux, p / s split, from the tangential E of
    the order (c_x, c_y). Incident flux = n_in |E0|^2 / (2 eta0); order flux = n_med (cos|Es|^2 +
    |Ept|^2/cos) / (2 eta0). Returns dict(T, p, s, ex, ey)."""
    q = (lam / P) * math.hypot(m, n)
    cos_t = math.sqrt(max(1.0 - (q / n_med) ** 2, 0.0))
    res = dict(ex=abs(cxo) ** 2 * n_med * cos_t / (n_in * E0abs ** 2),
               ey=abs(cyo) ** 2 * n_med * cos_t / (n_in * E0abs ** 2))
    if m == 0 and n == 0:
        res.update(p=None, s=None, T=res["ex"] * 0 + (abs(cxo) ** 2 + abs(cyo) ** 2) * n_med / (n_in * E0abs ** 2))
        return res
    ux, uy = m / math.hypot(m, n), n / math.hypot(m, n)
    ept = cxo * ux + cyo * uy
    es = -cxo * uy + cyo * ux
    p = n_med * abs(ept) ** 2 / (cos_t * n_in * E0abs ** 2)
    s = n_med * cos_t * abs(es) ** 2 / (n_in * E0abs ** 2)
    res.update(p=p, s=s, T=p + s)
    return res


# ------------------------------------------------------------------ one launch, one grid
def _spec(st, ncell):
    return dict(lx=st["P"], grid=st["P"] / ncell)


def estimate(st, geo, ncell, pml, stop_periods, throughput=None):
    grid = st["P"] / ncell
    nz = int(round((geo["zmax"] - geo["zmin"]) / grid)) + 2 * pml
    cells = float(ncell) * ncell * nz
    dt = 0.98 * grid / math.sqrt(3.0)
    nt = stop_periods * st["lam"] / dt
    return dict(grid_nm=grid * 1e3, nz=nz, cells=cells, nt=nt, cell_updates=cells * nt)


def ram_ok(est_gb, max_pct):
    try:
        import psutil
        if max_pct <= 0:
            return True, None
        vm = psutil.virtual_memory()
        after = (vm.total - vm.available + est_gb * 1e9) / vm.total
        return after <= max_pct / 100.0, after
    except Exception:
        return True, None


def read_launch_fields(workdir, prefix, mons, lx, ly, tag_mon):
    """Return {mon_name: {'ex': (cx, xg, yg), 'ey': ...}} for the monitors in `tag_mon`."""
    out = {}
    for name in tag_mon:
        d = {}
        for comp in ("ex", "ey"):
            f = find_dft(workdir, prefix, mons[name], comp)
            if f is None:
                raise FileNotFoundError("NOT FOUND: %s_m%d_f*_%s.dat in %s" % (prefix, mons[name], comp, workdir))
            cx, xg, yg = load_field(f)
            d[comp] = periodic_crop(cx, xg, yg, lx, ly)
        out[name] = d
    return out


def run_empty(st, geo, wd, pol, grid, thin, pml, stop_periods, ramp, disp, sk=None):
    """Empty (all-air) thin-cell run: incident phasor at A1 / A2, the launched component, and sigma."""
    os.makedirs(wd, exist_ok=True)
    pref = "e%d" % pol
    lx = ly = thin * grid
    ind = os.path.join(wd, pref + ".ind")
    info = build_ind(ind, st, geo, lx=lx, ly=ly, grid=grid, pol=pol, prefix=pref, empty=True, files=None,
                     pml=pml, stop_periods=stop_periods, ramp=ramp, disp=disp, **(sk or {}))
    res = run_fullwave(wd, pref + ".ind")
    if not res["ok"]:
        return dict(ok=False, run=res)
    F = read_launch_fields(wd, pref, info["mon"], lx, ly, ("A1", "A2"))
    c = {}
    for mn in ("A1", "A2"):
        for comp in ("ex", "ey"):
            cx, xg, yg = F[mn][comp]
            c[(mn, comp)] = complex(np.mean(cx))
    comp = "ex" if abs(c[("A1", "ex")]) >= abs(c[("A1", "ey")]) else "ey"
    other = "ey" if comp == "ex" else "ex"
    k0 = 2.0 * math.pi / st["lam"]
    dz = geo["zA2"] - geo["zA1"]
    ratio = c[("A2", comp)] / c[("A1", comp)]
    sigma = 1 if np.angle(ratio) > 0 else -1
    return dict(ok=True, comp=comp, pol=pol, inc_A1=c[("A1", comp)], inc_A2=c[("A2", comp)],
                cross_over_main=abs(c[("A1", other)]) / abs(c[("A1", comp)]), sigma=sigma,
                phase_adv_deg=math.degrees(float(np.angle(ratio))),
                phase_adv_expect_deg=math.degrees(sigma * k0 * dz), abs_ratio_A2_A1=abs(ratio),
                run=res)


def analyse_launch(st, geo, F, emp, lx, ly, mons_order=("A1", "A2", "S1", "S2")):
    """Orders, powers, reflection field and energy from one structure run."""
    sigma = emp["sigma"]
    P, lam, nsub = st["P"], st["lam"], st["n_sub"]
    comp = emp["comp"]
    other = "ey" if comp == "ex" else "ex"
    res = dict(sigma=sigma, launch_comp=comp)
    # incident phasor at each air monitor
    inc = {"A1": emp["inc_A1"], "A2": emp["inc_A2"]}
    E0 = {k: abs(v) for k, v in inc.items()}
    res["E0"] = E0
    # reflected (0,0): total - incident, per air monitor
    refl = {}
    for mn in ("A1", "A2"):
        cxx = order_coeff(*F[mn]["ex"], 0, 0, lx, ly, sigma)
        cyy = order_coeff(*F[mn]["ey"], 0, 0, lx, ly, sigma)
        tot = {"ex": cxx, "ey": cyy}
        r_main = (tot[comp] - inc[mn]) / inc[mn]
        r_cross = tot[other] / inc[mn]
        refl[mn] = dict(main=r_main, cross=r_cross, R=abs(r_main) ** 2 + abs(r_cross) ** 2)
    res["refl"] = {k: dict(main=[v["main"].real, v["main"].imag], cross=[v["cross"].real, v["cross"].imag],
                           R=v["R"]) for k, v in refl.items()}
    # substrate orders
    subord = prop_orders(P, lam, nsub)
    res["orders"] = {}
    for mn in ("S1", "S2"):
        ords = {}
        for (m, n) in subord:
            cxo = order_coeff(*F[mn]["ex"], m, n, lx, ly, sigma)
            cyo = order_coeff(*F[mn]["ey"], m, n, lx, ly, sigma)
            ords["%+d%+d" % (m, n)] = order_power(cxo, cyo, m, n, P, lam, nsub, E0["A1"])
        res["orders"][mn] = ords
    # evanescent / non-propagating content is not summed (it carries no flux)
    for mn in ("S1", "S2"):
        res.setdefault("T_total", {})[mn] = float(sum(v["T"] for v in res["orders"][mn].values()))
    res["R_total"] = {k: v["R"] for k, v in refl.items()}
    res["energy"] = {mn: res["T_total"][mn] + res["R_total"]["A1"] for mn in ("S1", "S2")}
    keys = sorted(res["orders"]["S1"].keys())
    spread = 0.0
    for k in keys:
        spread = max(spread, abs(res["orders"]["S1"][k]["T"] - res["orders"]["S2"][k]["T"]))
    res["S1_S2_order_spread"] = spread
    res["A1_A2_R_spread"] = abs(res["R_total"]["A1"] - res["R_total"]["A2"])
    # quoted value = mean of S1 and S2 (cancels part of a standing wave)
    mean = {}
    for k in keys:
        a, b = res["orders"]["S1"][k], res["orders"]["S2"][k]
        mean[k] = {f: (None if a[f] is None else 0.5 * (a[f] + b[f])) for f in ("T", "p", "s", "ex", "ey")}
    res["orders_mean"] = mean
    res["R_mean"] = 0.5 * (res["R_total"]["A1"] + res["R_total"]["A2"])
    res["energy_mean"] = res["R_mean"] + sum(v["T"] for v in mean.values())
    res["refl_jones_col"] = {k: [v["main"], v["cross"]] for k, v in refl.items()}
    return res


def gates_from(launch_x, launch_y, st):
    """G1 / G2 / G3 from the Ex and Ey launch analyses (quoted = S1/S2 mean, A1/A2 mean for R)."""
    out = {}
    if launch_x is not None:
        mx = launch_x["orders_mean"]
        out["eta_x"] = mx["+1+0"]["p"]
        out["leak_x"] = mx["+0+1"]["T"]
        out["ownlane_s_x"] = mx["+1+0"]["s"]
        out["leak_away_x"] = mx["+0-1"]["T"]
        out["refl_x"] = launch_x["R_mean"]
        out["vertical_x"] = mx["+0+0"]["T"]
        out["energy_x"] = launch_x["energy_mean"]
    if launch_y is not None:
        my = launch_y["orders_mean"]
        out["eta_y"] = my["+0+1"]["p"]
        out["leak_y"] = my["+1+0"]["T"]
        out["ownlane_s_y"] = my["+0+1"]["s"]
        out["leak_away_y"] = my["-1+0"]["T"]
        out["refl_y"] = launch_y["R_mean"]
        out["vertical_y"] = my["+0+0"]["T"]
        out["energy_y"] = launch_y["energy_mean"]
    if launch_x is not None and launch_y is not None:
        # Jones of the (0,0) reflection, columns = launches (Ex, Ey), rows = (x out, y out); mean of A1, A2
        # entries is NOT taken (phases differ by the path); singular values are taken per monitor, then mean.
        sv2 = []
        for mn in ("A1", "A2"):
            cx_, cy_ = launch_x["refl_jones_col"][mn], launch_y["refl_jones_col"][mn]
            # launch comp may be ex or ey: order rows as (x out, y out)
            col_x = _col_xy(launch_x, cx_)
            col_y = _col_xy(launch_y, cy_)
            J = np.array([[col_x[0], col_y[0]], [col_x[1], col_y[1]]], complex)
            sv2.append(float(np.linalg.svd(J, compute_uv=False)[0] ** 2))
        out["refl_worst_sop"] = 0.5 * (sv2[0] + sv2[1])
        out["refl_worst_sop_A1_A2"] = sv2
        out["G1_eta_min"] = min(out["eta_x"], out["eta_y"])
        out["G2_leak_max"] = max(out["leak_x"], out["leak_y"])
        out["G3_refl_worst_sop"] = out["refl_worst_sop"]
    elif launch_x is not None:
        # Ex launch only (a finalist check): the design is sigma_d symmetric (x <-> y), so the Ey column of
        # the Jones matrix is the mirror of the Ex column, J = [[a, b], [b, a]], and every y quantity equals
        # its x counterpart. ASSUMED, not measured: C-c (both launches) must have passed on the same grid
        # family first. The flag below marks these numbers.
        sv2 = []
        for mn in ("A1", "A2"):
            col_x = _col_xy(launch_x, launch_x["refl_jones_col"][mn])
            J = np.array([[col_x[0], col_x[1]], [col_x[1], col_x[0]]], complex)
            sv2.append(float(np.linalg.svd(J, compute_uv=False)[0] ** 2))
        out["refl_worst_sop"] = 0.5 * (sv2[0] + sv2[1])
        out["refl_worst_sop_A1_A2"] = sv2
        for k in ("eta", "leak", "ownlane_s", "leak_away", "refl", "vertical", "energy"):
            out[k + "_y"] = out[k + "_x"]
        out["G1_eta_min"] = out["eta_x"]
        out["G2_leak_max"] = out["leak_x"]
        out["G3_refl_worst_sop"] = out["refl_worst_sop"]
        out["assumed_sigma_d_mirror"] = True
    return out


def _col_xy(launch, col):
    """col = [main, cross] relative to the launched component; return (x out, y out)."""
    main, cross = (_cplx(col[0]), _cplx(col[1]))
    return (main, cross) if launch["launch_comp"] == "ex" else (cross, main)


def _cplx(v):
    """complex from a complex or from the [re, im] list a stored json holds."""
    if isinstance(v, (list, tuple)):
        return complex(v[0], v[1])
    return complex(v)


# ------------------------------------------------------------------ modes
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


def save_json(path, obj):
    tmp = path + ".tmp"

    def san(o):
        if isinstance(o, dict):
            return {str(k): san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [san(v) for v in o]
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, (np.floating, float)):
            v = float(o)
            return v if math.isfinite(v) else None
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    with io.open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(san(obj), ensure_ascii=True, indent=1))
    os.replace(tmp, path)


def mode_selftest():
    ok_all = True

    def chk(name, val, tol):
        nonlocal ok_all
        good = bool(val <= tol)
        ok_all = ok_all and good
        log("selftest %-46s %.3e (tol %.1e) %s" % (name, val, tol, "ok" if good else "**FAIL**"))

    st = load_stack(_txt("cell_folder", ""))
    # (1) the two TMM algebras against each other and a single-interface closed form
    r, t = tmm_recursive([], 1.0, 2.0, 1.55)
    chk("single interface r", abs(r - (-1.0 / 3.0)), 1e-14)
    chk("single interface t", abs(t - 2.0 / 3.0), 1e-14)
    worst = 0.0
    for a1 in (0.0, 1.0, 0.37):
        for a2 in (0.0, 1.0, 0.62):
            L = uniform_layers(st, a1, a2)
            r1, t1 = tmm_recursive(L, 1.0, st["n_sub"], st["lam"])
            r2, t2 = tmm_charmatrix(L, 1.0, st["n_sub"], st["lam"])
            worst = max(worst, abs(r1 - r2), abs(t1 - t2))
            # energy
            Rr = abs(r1) ** 2
            Tt = st["n_sub"] * abs(t1) ** 2
            worst = max(worst, abs(Rr + Tt - 1.0))
    chk("TMM recursive vs characteristic matrix, R+T", worst, 1e-13)
    # (2) resample
    N = st["nxy"]
    rng = np.random.default_rng(1)
    a = rng.random((N, N))
    a = 0.5 * (a + a.T)
    for nd in (128, 192, 256, 100):
        b = resample_area(a, nd)
        chk("resample %d: mean conserved" % nd, abs(a.mean() - b.mean()), 1e-12)
        chk("resample %d: sigma_d symmetry kept" % nd, float(np.abs(b - b.T).max()), 1e-13)
    W = resample_matrix(N, 128)
    chk("resample 128: rows sum to 1", float(np.abs(W.sum(axis=1) - 1.0).max()), 1e-13)
    # an isolated pixel keeps its weight (needle: nothing is dropped)
    z = np.zeros((N, N))
    z[40, 77] = 1.0
    chk("resample 128: isolated pixel not dropped", abs(resample_area(z, 128).sum() * (N / 128.0) ** 2 - 1.0), 1e-12)
    # (3) order extraction needle: a synthetic periodic field with known orders, both signs of sigma
    P, lam, nsub = st["P"], st["lam"], st["n_sub"]
    nc = 96
    xg = (np.arange(nc) + 0.5) * P / nc - P / 2
    yg = xg.copy()
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    E0 = 1.7
    # orders: (+1,0) p-pol amplitude (ex) 0.5, s (ey) 0.2 ; (0,+1) ey 0.4 (p) ; (0,0) ex 0.3 ey 0.1 ; (-1,0) ex 0.05
    for sigma in (+1, -1):
        Gx = 2.0 * math.pi / P
        ph = lambda m, n: np.exp(1j * sigma * (m * Gx * X + n * Gx * Y))   # noqa: E731
        ex = 0.5 * ph(1, 0) + 0.3 * ph(0, 0) + 0.05 * ph(-1, 0) + 0.1 * ph(0, 1) * 0.0
        ey = 0.2 * ph(1, 0) + 0.4 * ph(0, 1) + 0.1 * ph(0, 0)
        # add a high-order content that must not leak in
        ex = ex + 0.3 * ph(3, 2)
        c10x = order_coeff(ex, xg, yg, 1, 0, P, P, sigma)
        c10y = order_coeff(ey, xg, yg, 1, 0, P, P, sigma)
        c01y = order_coeff(ey, xg, yg, 0, 1, P, P, sigma)
        c00x = order_coeff(ex, xg, yg, 0, 0, P, P, sigma)
        c_m10x = order_coeff(ex, xg, yg, -1, 0, P, P, sigma)
        chk("needle sigma %+d: c(+1,0)x" % sigma, abs(c10x - 0.5), 1e-12)
        chk("needle sigma %+d: c(+1,0)y" % sigma, abs(c10y - 0.2), 1e-12)
        chk("needle sigma %+d: c(0,+1)y" % sigma, abs(c01y - 0.4), 1e-12)
        chk("needle sigma %+d: c(0,0)x" % sigma, abs(c00x - 0.3), 1e-12)
        chk("needle sigma %+d: c(-1,0)x (sign of the order)" % sigma, abs(c_m10x - 0.05), 1e-12)
        # wrong sigma must NOT recover (needle on the sign convention)
        wrong = order_coeff(ex, xg, yg, 1, 0, P, P, -sigma)
        ok_all = ok_all and (abs(wrong - 0.5) > 0.4)
        log("selftest needle sigma %+d: wrong-sign extraction gives %.3f (must differ from 0.5) %s"
            % (sigma, abs(wrong), "ok" if abs(wrong - 0.5) > 0.4 else "**FAIL**"))
        pw = order_power(c10x, c10y, 1, 0, P, lam, nsub, E0)
        cos_t = math.sqrt(1.0 - (lam / P / nsub) ** 2)
        chk("needle sigma %+d: p power (+1,0)" % sigma, abs(pw["p"] - nsub * 0.25 / (cos_t * E0 ** 2)), 1e-12)
        chk("needle sigma %+d: s power (+1,0)" % sigma, abs(pw["s"] - nsub * cos_t * 0.04 / E0 ** 2), 1e-12)
        pw2 = order_power(0.0, c01y, 0, 1, P, lam, nsub, E0)
        chk("needle sigma %+d: p power (0,+1) is the y component" % sigma, abs(pw2["p"] - nsub * 0.16 / (cos_t * E0 ** 2)), 1e-12)
    log("selftest %s" % ("ALL PASS" if ok_all else "**FAILED**"))
    return 0 if ok_all else 1


def mode_tmm():
    st = load_stack(_txt("cell_folder", ""))
    log("stack: P %.5f um lam %.4f H2 %.4f H1 %.4f n_sub %.4f  L2 %s/%s (%.4f/%.4f)  L1 %s/%s (%.4f/%.4f)"
        % (st["P"], st["lam"], st["H2"], st["H1"], st["n_sub"], st["material2"], st["fill2"], st["n2s"], st["n2f"],
           st["material1"], st["fill1"], st["n1s"], st["n1f"]))
    for a1, a2 in ((0, 0), (1, 1), (1, 0), (0, 1)):
        r, t = tmm_recursive(uniform_layers(st, a1, a2), 1.0, st["n_sub"], st["lam"])
        log("uniform L1=%d L2=%d: R %.6f  T %.6f  arg r %+.3f deg  arg t %+.3f deg"
            % (a1, a2, abs(r) ** 2, st["n_sub"] * abs(t) ** 2, math.degrees(np.angle(r)), math.degrees(np.angle(t))))
    return 0


def _dirs(rr):
    base = _txt("run_root", "") or os.path.join(rr, "runs")
    os.makedirs(base, exist_ok=True)
    return base


def mode_uniform():
    cell = _txt("cell_folder", "")
    st = load_stack(cell)
    rr = out_root()
    base = _dirs(rr)
    pml = _int("pml_cells", 12)
    thin = _int("thin_cells", 4)
    stop_p, ramp = _num("stop_periods", 40.0), _num("ramp_periods", 3.0)
    disp = _int("fdtd_dispersion", 0)
    pad_air, pad_sub, dz = _num("pad_air_nm", 1000.0) * 1e-3, _num("pad_sub_nm", 1450.0) * 1e-3, _num("mon_dz_nm", 350.0) * 1e-3
    geo = geometry(st, pad_air, pad_sub, dz)
    pin_cores()
    cases = [(0, 0), (1, 1), (1, 0), (0, 1)] if _txt("uniform_cases", "all") == "all" else [(0, 0)]
    rows = []
    out = dict(mode="uniform", stack={k: v for k, v in st.items() if k not in ("sources",)}, rows=rows,
               started=time.strftime("%F %T"))
    for ncell in _ints("grid_cells", "128, 192"):
        grid = st["P"] / ncell
        # the incident field is the same for every case: one empty run per polarisation per grid
        emp = {}
        for pol in (0, 1):
            wd = os.path.join(base, "uni_g%03d_empty_p%d" % (ncell, pol))
            emp[pol] = run_empty(st, geo, wd, pol, grid, thin, pml, stop_p, ramp, disp, sk=stop_kw())
            if not emp[pol]["ok"]:
                log("empty run failed: %s" % emp[pol]["run"].get("why"))
                out["verdict"] = "FAIL_EMPTY"
                save_json(os.path.join(rr, "fdtd_input2d_uniform.json"), out)
                return 1
            log("grid %d cells (%.3f nm) empty pol %d: comp %s sigma %+d  phase adv %.3f deg (expect %.3f) |A2/A1| %.5f  cross/main %.2e  %.1f s  peak %.2f GB"
                % (ncell, grid * 1e3, pol, emp[pol]["comp"], emp[pol]["sigma"], emp[pol]["phase_adv_deg"],
                   emp[pol]["phase_adv_expect_deg"], emp[pol]["abs_ratio_A2_A1"], emp[pol]["cross_over_main"],
                   emp[pol]["run"]["elapsed"], emp[pol]["run"]["peak_rss_gb"]))
        for (a1, a2) in cases:
            for pol in (0, 1):
                e = emp[pol]
                wd = os.path.join(base, "uni_g%03d_L1%d_L2%d_p%d" % (ncell, a1, a2, pol))
                os.makedirs(wd, exist_ok=True)
                lx = ly = thin * grid
                n_map1 = np.full((thin, thin), n_of_mix(st["n1f"], st["n1s"], a1))
                n_map2 = np.full((thin, thin), n_of_mix(st["n2f"], st["n2s"], a2))
                write_profile(os.path.join(wd, "L1.dat"), n_map1, 1.0, max(st["n1s"], st["n1f"]))
                write_profile(os.path.join(wd, "L2.dat"), n_map2, 1.0, max(st["n2s"], st["n2f"]))
                files = dict(L1=("prof", ("L1.dat", max(st["n1s"], st["n1f"]))),
                             L2=("prof", ("L2.dat", max(st["n2s"], st["n2f"]))))
                pref = "u%d%d%d" % (a1, a2, pol)
                info = build_ind(os.path.join(wd, pref + ".ind"), st, geo, lx=lx, ly=ly, grid=grid, pol=pol,
                                 prefix=pref, empty=False, files=files, pml=pml, stop_periods=stop_p, ramp=ramp,
                                 disp=disp, **stop_kw())
                res = run_fullwave(wd, pref + ".ind")
                if not res["ok"]:
                    log("run failed in %s: %s" % (wd, res.get("why")))
                    rows.append(dict(ncell=ncell, a1=a1, a2=a2, pol=pol, ok=False, why=res.get("why")))
                    continue
                F = read_launch_fields(wd, pref, info["mon"], lx, ly, ("A1", "A2", "S1", "S2"))
                sg = e["sigma"]
                comp = e["comp"]
                k0 = 2.0 * math.pi / st["lam"]
                ksub = k0 * st["n_sub"]
                row = dict(ncell=ncell, grid_nm=grid * 1e3, a1=a1, a2=a2, pol=pol, comp=comp, ok=True,
                           elapsed_s=res["elapsed"], peak_gb=res["peak_rss_gb"])
                # reflection at A1 / A2 -> r referenced to z = 0 ; transmission at S1 / S2 -> t at z = ztop
                rs, ts = [], []
                for mn, zr in (("A1", geo["zA1"]), ("A2", geo["zA2"])):
                    c_tot = order_coeff(*F[mn][comp], 0, 0, lx, ly, sg)
                    c_inc = e["inc_" + mn]
                    # incident phasor exp(+i sigma k0 z); reflected exp(-i sigma k0 z)
                    rs.append((c_tot - c_inc) / c_inc * np.exp(2j * sg * k0 * zr))
                for mn, zt in (("S1", geo["zS1"]), ("S2", geo["zS2"])):
                    c_t = order_coeff(*F[mn][comp], 0, 0, lx, ly, sg)
                    c_inc_A1 = e["inc_A1"]
                    # incident phasor at z = 0 : c_inc_A1 * exp(-i sigma k0 zA1)
                    ts.append(c_t / (c_inc_A1 * np.exp(-1j * sg * k0 * geo["zA1"])) * np.exp(-1j * sg * ksub * (zt - geo["ztop"])))
                # sigma = -1 means the phasor is the complex conjugate of the exp(-i w t) one
                if sg < 0:
                    rs = [np.conj(v) for v in rs]
                    ts = [np.conj(v) for v in ts]
                r_a, t_a = tmm_recursive(uniform_layers(st, a1, a2), 1.0, st["n_sub"], st["lam"])
                for tag, rv, tv in (("A1S1", rs[0], ts[0]), ("A2S2", rs[1], ts[1])):
                    R, T = abs(rv) ** 2, st["n_sub"] * abs(tv) ** 2
                    row[tag] = dict(R=R, T=T, dR=R - abs(r_a) ** 2, dT=T - st["n_sub"] * abs(t_a) ** 2,
                                    d_arg_r_deg=math.degrees(float(np.angle(rv / r_a))),
                                    d_arg_t_deg=math.degrees(float(np.angle(tv / t_a))), RT=R + T)
                row["R_ana"], row["T_ana"] = abs(r_a) ** 2, st["n_sub"] * abs(t_a) ** 2
                rows.append(row)
                log("grid %3d  L1=%d L2=%d pol %d  R %.6f (ana %.6f, d %+.2e)  T %.6f (ana %.6f, d %+.2e)  R+T %.6f  d arg r %+.3f t %+.3f deg  %.1f s"
                    % (ncell, a1, a2, pol, row["A1S1"]["R"], row["R_ana"], row["A1S1"]["dR"], row["A1S1"]["T"],
                       row["T_ana"], row["A1S1"]["dT"], row["A1S1"]["RT"], row["A1S1"]["d_arg_r_deg"],
                       row["A1S1"]["d_arg_t_deg"], res["elapsed"]))
                save_json(os.path.join(rr, "fdtd_input2d_uniform.json"), out)
    out["verdict"] = "DONE"
    save_json(os.path.join(rr, "fdtd_input2d_uniform.json"), out)
    return 0


def load_design(cell):
    r1p = _txt("rho1_npy", "") or os.path.join(cell, "rho1.npy")
    r2p = _txt("rho2_npy", "") or os.path.join(cell, "rho2.npy")
    for p in (r1p, r2p):
        if not os.path.isfile(p):
            print("NOT FOUND: %s" % p)
            raise SystemExit(2)
    r1, r2 = np.load(r1p).astype(float), np.load(r2p).astype(float)
    return r1, r2, r1p, r2p


def mode_cell():
    cell = _txt("cell_folder", "")
    st = load_stack(cell)
    rho1, rho2, r1p, r2p = load_design(cell)
    if rho1.shape != (st["nxy"], st["nxy"]) or rho2.shape != rho1.shape:
        raise ValueError("raster shape %s != nxy %d" % (rho1.shape, st["nxy"]))
    sym = float(max(np.abs(rho1 - rho1.T).max(), np.abs(rho2 - rho2.T).max()))
    rr = out_root()
    base = _dirs(rr)
    pml = _int("pml_cells", 12)
    thin = _int("thin_cells", 4)
    stop_p, ramp = _num("stop_periods", 40.0), _num("ramp_periods", 3.0)
    disp = _int("fdtd_dispersion", 0)
    pad_air, pad_sub, dzm = _num("pad_air_nm", 1000.0) * 1e-3, _num("pad_sub_nm", 1450.0) * 1e-3, _num("mon_dz_nm", 350.0) * 1e-3
    geo = geometry(st, pad_air, pad_sub, dzm)
    launches = [s.strip().lower() for s in _txt("launches", "Ex, Ey").split(",") if s.strip()]
    max_pct = _num("max_ram_pct", float(os.environ.get("SIM_MAX_RAM_PCT", "0")))
    max_run_s = _num("max_run_s", 900.0)
    bytes_per_cell = _num("bytes_per_cell", 150.0)
    thr = _num("cell_updates_per_s", 6.0e7)
    dry = _int("dry_run", 0)
    resume = _int("resume", 1)
    md5 = hashlib.md5(rho1.tobytes() + rho2.tobytes()).hexdigest()
    jpath = os.path.join(rr, "fdtd_input2d.json")
    out = dict(mode="cell", cell_folder=cell, rho1_npy=r1p, rho2_npy=r2p, raster_md5=md5, sym_err=sym,
               stack={k: v for k, v in st.items()}, started=time.strftime("%F %T"),
               host=os.environ.get("COMPUTERNAME", "?"), driver=os.path.abspath(__file__), grids={})
    if os.path.isfile(jpath) and resume:
        try:
            old = json.load(io.open(jpath, encoding="utf-8"))
            if old.get("raster_md5") == md5:
                out["grids"] = old.get("grids", {})
                log("resume: %d grids already in %s" % (len(out["grids"]), jpath))
        except Exception:
            pass
    pin_cores()
    log("design %s md5 %s sym_err %.1e  stack P %.5f H2 %.4f H1 %.4f n_sub %.4f" % (cell, md5[:10], sym, st["P"], st["H2"], st["H1"], st["n_sub"]))
    for ncell in _ints("grid_cells", "128"):
        grid = st["P"] / ncell
        est = estimate(st, geo, ncell, pml, stop_p)
        gk = "c%d_s%g_d%d_x%g" % (ncell, stop_p, disp, _num("prof_shift_x", 0.5))
        G_ = out["grids"].setdefault(gk, dict(ncell=ncell, grid_nm=grid * 1e3, stop_periods=stop_p, estimate=est,
                                              pads_um=[pad_air, pad_sub], mon_dz_um=dzm, pml_cells=pml, launches={}))
        est_gb = est["cells"] * bytes_per_cell / 1e9
        est_s = est["cell_updates"] / thr
        log("grid %d cells (%.4f nm): NZ %d, %.2f M cells, ~%d steps, est %.1f GB, est %.0f s per run" % (ncell, grid * 1e3, est["nz"], est["cells"] / 1e6, est["nt"], est_gb, est_s))
        if dry:
            continue
        if est_s > max_run_s:
            log("SKIPPED grid %d: estimated %.0f s > max_run_s %.0f -- write a held job instead" % (ncell, est_s, max_run_s))
            G_["skipped"] = "estimated %.0f s > %.0f s" % (est_s, max_run_s)
            save_json(jpath, out)
            continue
        ok_r, after = ram_ok(est_gb, max_pct)
        if not ok_r:
            log("SKIPPED grid %d: RAM would reach %.0f %% > %.0f %%" % (ncell, 100 * after, max_pct))
            G_["skipped"] = "RAM %.0f %%" % (100 * after)
            save_json(jpath, out)
            continue
        emp = {}
        for pol in (0, 1):
            wd = os.path.join(base, "cell_%s_%s_empty_p%d" % (md5[:8], gk, pol))
            emp[pol] = run_empty(st, geo, wd, pol, grid, thin, pml, stop_p, ramp, disp, sk=stop_kw())
            if not emp[pol]["ok"]:
                log("empty run failed: %s" % emp[pol]["run"].get("why"))
                G_["error"] = "empty run failed"
                save_json(jpath, out)
                break
        if "error" in G_:
            continue
        by_comp = {emp[p]["comp"]: emp[p] for p in (0, 1)}
        if set(by_comp) != {"ex", "ey"}:
            log("polarisation map failed: components %s" % sorted(by_comp))
            G_["error"] = "pol map"
            save_json(jpath, out)
            continue
        G_["incident"] = {c: dict(pol=v["pol"], sigma=v["sigma"], inc_A1=v["inc_A1"], inc_A2=v["inc_A2"],
                                  phase_adv_deg=v["phase_adv_deg"], phase_adv_expect_deg=v["phase_adv_expect_deg"],
                                  cross_over_main=v["cross_over_main"]) for c, v in by_comp.items()}
        # the raster -> index maps (identical for both launches)
        f1 = resample_area(rho1, ncell)
        f2 = resample_area(rho2, ncell)
        n1 = index_map(f1, st["n1f"], st["n1s"])
        n2 = index_map(f2, st["n2f"], st["n2s"])
        res_l = {}
        for lname in launches:
            comp = "ex" if lname == "ex" else "ey"
            e = by_comp[comp]
            wd = os.path.join(base, "cell_%s_%s_%s" % (md5[:8], gk, lname))
            os.makedirs(wd, exist_ok=True)
            lx = ly = st["P"]
            if resume and lname in G_["launches"] and G_["launches"][lname].get("ok"):
                log("resume: launch %s of %s already done" % (lname, gk))
                res_l[lname] = G_["launches"][lname]["analysis"]
                continue
            write_profile(os.path.join(wd, "L1.dat"), n1, 1.0, max(st["n1s"], st["n1f"]), shift=(_num("prof_shift_x", 0.5), _num("prof_shift_y", 0.0)))
            write_profile(os.path.join(wd, "L2.dat"), n2, 1.0, max(st["n2s"], st["n2f"]), shift=(_num("prof_shift_x", 0.5), _num("prof_shift_y", 0.0)))
            files = dict(L1=("prof", ("L1.dat", max(st["n1s"], st["n1f"]))), L2=("prof", ("L2.dat", max(st["n2s"], st["n2f"]))))
            pref = "d_%s" % lname
            info = build_ind(os.path.join(wd, pref + ".ind"), st, geo, lx=lx, ly=ly, grid=grid, pol=e["pol"], prefix=pref,
                             empty=False, files=files, pml=pml, stop_periods=stop_p, ramp=ramp, disp=disp, **stop_kw())
            log("launch %s (pol %d): running %s" % (lname, e["pol"], wd))
            res = run_fullwave(wd, pref + ".ind", timeout=_num("rsoft_timeout_s", 7200.0), mpi_np=_int("fdtd_mpi_np", 1))
            if not res["ok"]:
                log("RUN FAILED: %s (rc %s, %.0f s)" % (res.get("why"), res.get("rc"), res.get("elapsed", 0)))
                G_["launches"][lname] = dict(ok=False, run={k: v for k, v in res.items() if k != "log_tail"}, log_tail=res.get("log_tail"))
                save_json(jpath, out)
                continue
            F = read_launch_fields(wd, pref, info["mon"], lx, ly, ("A1", "A2", "S1", "S2"))
            an = analyse_launch(st, geo, F, e, lx, ly)
            res_l[lname] = an
            G_["launches"][lname] = dict(ok=True, comp=comp, pol=e["pol"], analysis=an,
                                         run={k: v for k, v in res.items() if k != "log_tail"})
            lf = res.get("logfacts", {})
            log("launch %s done %.1f s, peak %.2f GB, cpu ratio %s, threads %s | NX,NY,NZ %s,%s,%s NT %s ran %.1f periods%s | energy %.5f  S1/S2 spread %.2e"
                % (lname, res["elapsed"], res["peak_rss_gb"], ("%.2f" % res["cpu_ratio"]) if res["cpu_ratio"] else "?",
                   res["nthreads"], lf.get("nx"), lf.get("ny"), lf.get("nz"), lf.get("nt_run"),
                   lf.get("stop_time_um", 0.0) / st["lam"], " (converged)" if lf.get("converged") else "",
                   an["energy_mean"], an["S1_S2_order_spread"]))
            save_json(jpath, out)
        lx_, ly_ = res_l.get("ex"), res_l.get("ey")
        if lx_ is not None or ly_ is not None:
            G_["gates"] = gates_from(lx_, ly_, st)
            log("grid %d gates: %s" % (ncell, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in G_["gates"].items() if not isinstance(v, list)})))
        save_json(jpath, out)
    out["finished"] = time.strftime("%F %T")
    save_json(jpath, out)
    return 0


def mode_compare():
    """C-b / C-c / C-d from stored json files. Keys: fdtd_json (comma list), rcwa_json, tol_cb, tol_cc, tol_cd."""
    paths = [p.strip() for p in _txt("fdtd_json", "").split(",") if p.strip()]
    docs = [json.load(io.open(p, encoding="utf-8")) for p in paths]
    tol_cb, tol_cc, tol_cd = _num("tol_cb", 0.02), _num("tol_cc", 2e-3), _num("tol_cd", 0.03)
    log("declared tolerances: C-b grid convergence %.3g, C-c symmetry %.3g, C-d vs grcwa %.3g (absolute, power units)" % (tol_cb, tol_cc, tol_cd))
    # merge the launches of the same grid key found in several json files (one job per launch), then
    # recompute the gates from the merged launches
    merged = {}
    for d in docs:
        for k, g in d.get("grids", {}).items():
            m = merged.setdefault(k, dict(ncell=g["ncell"], grid_nm=g.get("grid_nm"), launches={}))
            for ln, v in g.get("launches", {}).items():
                if v.get("ok"):
                    m["launches"][ln] = v
    grids = []
    for k, m in merged.items():
        lx_ = m["launches"].get("ex", {}).get("analysis")
        ly_ = m["launches"].get("ey", {}).get("analysis")
        if lx_ is None and ly_ is None:
            continue
        m["gates"] = gates_from(lx_, ly_, None)
        if "eta_x" in m["gates"] and "eta_y" in m["gates"]:
            grids.append((m["ncell"], m, k))
    grids.sort(key=lambda t: t[0])
    keys = ("eta_x", "eta_y", "leak_x", "leak_y", "refl_x", "refl_y", "refl_worst_sop", "vertical_x", "ownlane_s_x", "leak_away_x")
    for ncell, g, k in grids:
        gt = g["gates"]
        log("grid %d (%s): " % (ncell, k) + "  ".join("%s %.5f" % (q, gt[q]) for q in keys if q in gt))
        # C-c (needs both launches on this grid)
        if not all(q in g["launches"] and g["launches"][q].get("ok") for q in ("ex", "ey")):
            log("C-c grid %d: only one launch stored -- not testable" % ncell)
            continue
        a = g["launches"]["ex"]["analysis"]["orders_mean"]
        b = g["launches"]["ey"]["analysis"]["orders_mean"]
        pairs = {"+1+0": "+0+1", "-1+0": "+0-1", "+0+1": "+1+0", "+0-1": "-1+0", "+0+0": "+0+0"}
        worst = 0.0
        for ox, oy in pairs.items():
            for f in ("T", "p", "s", "ex", "ey"):
                va, vb = a[ox][f], b[oy][f]
                if va is None or vb is None:
                    continue
                # an x <-> y swap also swaps the component labels ex <-> ey
                if f in ("ex", "ey"):
                    vb = b[oy]["ey" if f == "ex" else "ex"]
                worst = max(worst, abs(va - vb))
        ra, rb = g["launches"]["ex"]["analysis"]["R_mean"], g["launches"]["ey"]["analysis"]["R_mean"]
        worst = max(worst, abs(ra - rb))
        log("C-c grid %d: worst |Ex-run - mirrored Ey-run| over all orders/pol/R = %.3e  (tol %.1e)  %s" % (ncell, worst, tol_cc, "PASS" if worst <= tol_cc else "FAIL"))
    if len(grids) >= 2:
        (n0, g0, _), (n1, g1, _) = grids[0], grids[-1]
        worst = 0.0
        for q in keys:
            if q in g0["gates"] and q in g1["gates"]:
                dq = g1["gates"][q] - g0["gates"][q]
                worst = max(worst, abs(dq))
                log("C-b %-16s %d cells %.5f -> %d cells %.5f  delta %+.5f" % (q, n0, g0["gates"][q], n1, g1["gates"][q], dq))
        log("C-b worst delta %.5f (tol %.3g)  %s" % (worst, tol_cb, "PASS" if worst <= tol_cb else "FAIL"))
    rj = _txt("rcwa_json", "")
    if rj:
        j = json.load(io.open(rj, encoding="utf-8"))
        rng = _ints("rcwa_nG", "601")
        for ng in rng:
            pre = "ng%d_" % ng
            ref = {q: j.get(pre + q) for q in ("eta_x", "eta_y", "leak_x", "leak_y", "refl_x", "refl_y", "refl_worst_sop", "vertical_x", "ownlane_s_x", "leak_away_x")}
            if ref["eta_x"] is None:
                log("C-d: rung nG %d not in %s" % (ng, rj))
                continue
            for ncell, g, k in grids:
                worst = 0.0
                for q, rv in ref.items():
                    if rv is None or q not in g["gates"]:
                        continue
                    dq = g["gates"][q] - rv
                    worst = max(worst, abs(dq))
                    log("C-d nG %d  %-16s grcwa %.5f  FDTD(%d cells) %.5f  delta %+.5f" % (ng, q, rv, ncell, g["gates"][q], dq))
                log("C-d nG %d vs FDTD %d cells: worst |delta| %.5f (tol %.3g)  %s" % (ng, ncell, worst, tol_cd, "PASS" if worst <= tol_cd else "FAIL"))
    return 0


def main():
    global CFG
    if len(sys.argv) < 2:
        print("usage: Main_fdtd_input2d_verify.py <job_file>")
        raise SystemExit(2)
    CFG = load_cfg(sys.argv[1])
    mode = _txt("mode", "cell").lower()
    log("mode %s  job %s  host %s" % (mode, sys.argv[1], socket.gethostname()))
    if mode == "selftest":
        return mode_selftest()
    if mode == "tmm":
        return mode_tmm()
    if mode == "uniform":
        return mode_uniform()
    if mode == "cell":
        return mode_cell()
    if mode == "compare":
        return mode_compare()
    print("%s unknown mode %r" % (TAG, mode))
    return 2


if __name__ == "__main__":
    sys.exit(main())
