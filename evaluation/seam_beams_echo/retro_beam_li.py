# -*- coding: utf-8 -*-
"""retro_beam_li.py -- finite-beam retro-reflection of the retro grating on the Li basis (comment C12, 2026-10-12).

STANDALONE SCRIPT (~60-120 RCWA solves of a single 1-D period, minutes on 2 cores, < 1 GB).

What it is: a port of the 260825 figure script <WORK_ROOT>\\RCWA\\upper_grating_beams.py (function
scan() and the retro half of fig_retro()) that produced Figure\\260825_upper_grating_beams\\data\\profile_bottom_retro_TM.dat
(the source of the "Delta x_c = +0.63 um" of Figure 4, facts C9/X3). The recipe is kept line for line:
  1. Gaussian beam at the retro-grating plane z = 0: footprint radius wx = W0 / cos(theta_sub), amplitude spectrum
     A(u) = exp(-(u wx / 2)^2) on NANG samples u in [-8/wx, +8/wx], kx_i = kx0 + u / k0.
  2. one RCWA solve per kx_i (substrate incidence, TM, side "sub"), complex reflection coefficients of the orders
     m = 0, -1, -2 with the kz power correction  a = sqrt(P kz_in / kz_out)  and the phase of GetAmplitudes(0, 0).
  3. the incident beam is moved to the grating by exp(-i (kx_i - kx_c) k0 s), s = t tan(theta_sub), t = 592.39 um.
  4. incident and outgoing intensity |E|^2 are reconstructed on the bottom face z = -t (profile_bottom_*.dat columns
     x_um, I_incident, I_outgoing, both divided by the incident maximum).
What is NEW: (a) the factorization rule is a flag (--rule liy --ff 1 = the sim778 Li basis, nG 257); (b) any beam
radius (the paper beam is W0 = 100 um, the old figure used 50 um); (c) the C9 numbers are produced by the script
itself instead of a separate post-processing step.

C9 DEFINITIONS (reverse-engineered, see calc_dxc.py):
  window "old"  = x grid np.linspace(-200, 1800, 1000)   (the stored window, 2.002 um step)
  full-window centroid shift  = centroid(I_outgoing) - centroid(I_incident)  over the window   [the +0.6302 um of the figure]
  main-lobe shift             = the same restricted to |x| < 300 um                            [-0.0588 um for the old cell]
  I = |E|^2 (intensity), NOT a power flux. Extra, not in C9: the same with the per-order kz weight (flux), the lobe
  centroids / power fractions per order, and a wide window x in [-600, 2800] (the m = 0 mirror lobe sits at 2 s = 1672.7 um
  and is cut by the old window at 1800).

CONTROL (needle): --cell old --rule laurent --ff 0 --nG 161 --w0 50 --nang 61 must give +0.6302 um (full window) and
-0.0588 um (main lobe) and reproduce profile_bottom_retro_TM_OLD.dat (the script prints max |diff| of both columns).

usage (cores pinned AFTER the imports; logical 12-15 only):
  python retro_beam_li.py --cell new --rule liy --ff 1 --nG 257 --w0 100 --nang 61 --affinity 12-13 --tag new_liy_w100
  python retro_beam_li.py --cell old --rule laurent --ff 0 --nG 161 --w0 50 --nang 61 --affinity 14-15 --tag ctrl_old
Outputs go to --out (default <DATA_ROOT>\\Result\\261012_scdc_liseam\\beam\\<tag>): beam_<tag>.json,
profile_<tag>.dat, angular_<tag>.dat, run_log.txt. Nothing is overwritten (refuses an existing out dir unless --force-new-name).
"""
import argparse
import io
import json
import os
import sys
import time

sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument("--cell", choices=("new", "old"), required=True)
ap.add_argument("--rule", choices=("laurent", "liy", "lix"), default="liy")
ap.add_argument("--ff", type=int, default=1)
ap.add_argument("--nG", type=int, default=257)
ap.add_argument("--w0", type=float, nargs="+", default=[100.0])
ap.add_argument("--nang", type=int, default=61)
ap.add_argument("--t_um", type=float, default=592.39)
ap.add_argument("--affinity", default="")
ap.add_argument("--blas", default="2")
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--lam_nm", type=float, default=0.0, help="0 = the cell record's design wavelength (1550.0 / M.LAM)")
args = ap.parse_args()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = args.blas

import numpy as np   # noqa: E402

RC = r"<CODE_ROOT>\Version_unified\RCWA_custom"
sys.path.insert(0, RC)
import scd_c_grating as M   # noqa: E402
import psutil               # noqa: E402

if args.affinity:
    lo, hi = (int(v) for v in args.affinity.split("-"))
    psutil.Process().cpu_affinity(list(range(lo, hi + 1)))   # after the imports (some modules reset it at import)

INP = r"<DATA_ROOT>\Result\261012_scdc_liseam\_inputs"
CELLS = {
    "new": (INP + r"\retro_A001_79631f61\density_scdc_retro_H525_450.npy",
            INP + r"\retro_A001_79631f61\scdc_retro_H525_450_liy_n257.json"),
    "old": (INP + r"\retro_old2312_c15are\density_scdc_retro_H525_450.npy",
            INP + r"\retro_old2312_c15are\scdc_retro_H525_450.json"),
}
OLD_PROFILE = INP + r"\retro_old2312_c15are\profile_bottom_retro_TM_OLD.dat"
OUT = args.out or (r"<DATA_ROOT>\Result\261012_scdc_liseam\beam\%s" % args.tag)
if os.path.exists(OUT):
    raise SystemExit("exists, not overwriting: " + OUT)
os.makedirs(OUT)
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


if args.rule != "laurent" or args.ff:
    M.set_rcwa(args.rule, args.ff)
log("[beam] cell %s  rule %s ff %d  nG %d  w0 %s  nang %d  t %.2f um  (scd_c_grating RULE %s FF %s)"
    % (args.cell, args.rule, args.ff, args.nG, args.w0, args.nang, args.t_um, M.RCWA_RULE, M.RCWA_FF))

dens_path, js_path = CELLS[args.cell]
dens = np.load(dens_path)
J = json.load(io.open(js_path, encoding="utf-8"))
assert dens.shape == (2, 512)
# setup(): same order as the old script (the project note: set_period -> set_substrate; both end consistent)
M.LAMS = tuple(float(v) * 1e-3 for v in J["lams_nm"])
M.set_period(float(J["period_nm"]))
M.set_substrate(float(J["n_sub"]))
H1, H2 = float(J["H1_nm"]) / 1000.0, float(J["H2_nm"]) / 1000.0
eps_b = (M.eps_of(J["material1"]), M.eps_of(J["material2"]))
ef1, ef2 = M.eps_of(J["fill1"]), M.eps_of(J["fill2"])
lam, k0 = M.LAM, 2 * np.pi / M.LAM
if args.lam_nm > 0:
    lam = args.lam_nm * 1e-3
    k0 = 2 * np.pi / lam
G = lam / M.PERIOD
n = float(J["n_sub"])
th0 = M.THETA_SUB
s_um = args.t_um * np.tan(th0)
log("[beam] materials %s/%s fills %s/%s  H %.3f/%.3f um  period %.5f  n_sub %.4f  theta_sub %.6f deg  s %.4f um (2s %.4f)  "
    "lam %.4f um" % (J["material1"], J["material2"], J["fill1"], J["fill2"], H1, H2, M.PERIOD, n, np.degrees(th0),
                     s_um, 2 * s_um, lam))


def kz_of(v, nn):
    v = np.asarray(v, float)
    return np.sqrt(np.maximum(nn * nn - v * v, 0.0))


MS_R = (0, -1, -2)


def scan(w0, nang):
    """old scan(): (A, kxs, r) with r = {m: complex array}."""
    kx0 = n * np.sin(th0)
    wx = w0 / np.cos(th0)
    umax = 4.0 * 2.0 / wx
    u = np.linspace(-umax, umax, nang)
    kxs = kx0 + u / k0
    A = np.exp(-(u * wx / 2.0) ** 2)
    r = {m: np.zeros(nang, complex) for m in MS_R}
    t0 = time.time()
    for i, kxi in enumerate(kxs):
        thi = float(np.arcsin(np.clip(kxi / n, -1.0, 1.0)))
        ob, R, T = M._solve(dens[0], dens[1], H1, H2, eps_b, ef1, ef2, "sub", "TM", thi, lam, args.nG)
        Rm = M._pick(ob, R, MS_R)
        AmR, AmT = M._args(ob, "TM", MS_R)
        kzi = float(kz_of(kxi, n))
        for m in MS_R:
            kzm = float(kz_of(kxi + m * G, n))
            if kzm <= 1e-9:
                continue
            a = np.sqrt(max(float(Rm[m]), 0.0) * kzi / kzm)
            p = np.radians(float(AmR[m])) if AmR.get(m) is not None else 0.0
            r[m][i] = a * np.exp(1j * p)
        if i == 0 or (i + 1) % 10 == 0:
            log("   angle %3d/%d  theta %.4f deg  R(-2) %.8f  (%.1f s)" % (i + 1, nang, np.degrees(thi), float(Rm[-2]),
                                                                        time.time() - t0))
    log("[beam] RCWA %d angles in %.1f s (%.2f s/solve), angular half-width %.4f deg, peak %.2f GB"
        % (nang, time.time() - t0, (time.time() - t0) / nang, np.degrees(umax / k0 / (n * np.cos(th0))),
           psutil.Process().memory_info().peak_wset / 1e9))
    return A, kxs, r, wx


def profiles(A, kxs, r, xs, nang):
    """bottom-face intensity: incident, outgoing total, outgoing per order (z = -t)."""
    A = A * np.exp(-1j * (kxs - kxs[nang // 2]) * k0 * s_um)        # beam centre -> grating (old script)
    Z = -args.t_um
    ein = np.zeros(len(xs), complex)
    eo = {m: np.zeros(len(xs), complex) for m in MS_R}
    for i in range(nang):
        kxi = kxs[i]
        ein += A[i] * np.exp(1j * k0 * (kxi * xs + float(kz_of(kxi, n)) * Z))
        for m in MS_R:
            if r[m][i] == 0:
                continue
            ko = kxi + m * G
            eo[m] += A[i] * r[m][i] * np.exp(1j * k0 * (ko * xs - float(kz_of(ko, n)) * Z))
    eout = eo[0] + eo[-1] + eo[-2]
    # reference = incident maximum over the whole map; the waist (z = 0) is where it is largest
    e0 = np.zeros(len(xs), complex)
    for i in range(nang):
        e0 += A[i] * np.exp(1j * k0 * kxs[i] * xs)
    return np.abs(ein) ** 2, np.abs(eout) ** 2, {m: np.abs(eo[m]) ** 2 for m in MS_R}, float(np.max(np.abs(e0) ** 2))


def cent(x, I, mask=None):
    if mask is not None:
        x, I = x[mask], I[mask]
    return float(np.sum(x * I) / np.sum(I))


res_all = {}
for w0 in args.w0:
    log("\n=== W0 = %.1f um ===" % w0)
    A, kxs, r, wx = scan(w0, args.nang)
    period_alias = 2 * np.pi / ((2 * 4.0 * 2.0 / wx) / (args.nang - 1))
    log("[beam] discrete-angle spatial period 2 pi / du = %.0f um (replicas of a lobe at x0 appear at x0 +- this)" % period_alias)
    out = {"w0_um": w0, "wx_um": float(wx), "alias_period_um": float(period_alias)}
    # angular facts: R(-2) at the centre angle and its phase slope
    ic = args.nang // 2
    out["R_m2_centre"] = float(np.abs(r[-2][ic]) ** 2 * float(kz_of(kxs[ic] - 2 * G, n)) / float(kz_of(kxs[ic], n)))
    out["R_m2_min_in_beam"] = float(min((np.abs(r[-2][i]) ** 2 * float(kz_of(kxs[i] - 2 * G, n)) / float(kz_of(kxs[i], n)))
                                        for i in range(args.nang) if abs(A[i]) > 1e-2))
    ph = np.unwrap(np.angle(r[-2]))
    wsel = np.abs(A) > 1e-2
    # lateral shift of the main lobe predicted by the phase slope: dx = -(1/k0) d arg r / d kx  (reference plane independent)
    pp = np.polyfit(kxs[wsel] - kxs[ic], ph[wsel], 1)
    out["phase_slope_dx_um"] = float(-pp[0] / k0)
    for wname, xs in (("old", np.linspace(-200, 1800, 1000)), ("wide", np.arange(-600.0, 2800.0 + 1e-9, 2.0))):
        Ii, Io, Im, ref = profiles(A, kxs, r, xs, args.nang)
        ref = max(ref, float(Ii.max()))
        w = {}
        w["x_step_um"] = float(xs[1] - xs[0])
        w["cent_inc"] = cent(xs, Ii)
        w["cent_out"] = cent(xs, Io)
        w["shift_full_um"] = w["cent_out"] - w["cent_inc"]
        ml = np.abs(xs) < 300
        w["shift_mainlobe_um"] = cent(xs, Io, ml) - cent(xs, Ii, ml)
        # [not in C9] main-lobe radius scaled with the beam: 3.5 footprint radii (= 300 um for the old W0 = 50 um)
        mls = np.abs(xs) < 3.5 * wx
        w["mainlobe_radius_scaled_um"] = float(3.5 * wx)
        w["shift_mainlobe_scaled_um"] = cent(xs, Io, mls) - cent(xs, Ii, mls)
        w["out_frac_beyond_300"] = float(np.sum(Io[~ml]) / np.sum(Io))
        w["out_over_inc_total"] = float(np.sum(Io) / np.sum(Ii))
        # flux weighting per order: weight kz_m / kz_in at the centre angle (|E|^2 -> power flux through the plane)
        kzc = float(kz_of(kxs[ic], n))
        wts = {m: float(kz_of(kxs[ic] + m * G, n)) / kzc for m in MS_R}
        Fo = sum(wts[m] * Im[m] for m in MS_R)
        w["shift_full_flux_um"] = cent(xs, Fo) - w["cent_inc"]
        w["shift_mainlobe_flux_um"] = cent(xs, Fo, ml) - cent(xs, Ii, ml)
        w["orders"] = {}
        for m in MS_R:
            tot = float(np.sum(Im[m]))
            w["orders"][str(m)] = dict(sum_I=tot, frac_of_out=tot / float(np.sum(Io)),
                                       centroid_um=(cent(xs, Im[m]) if tot > 0 else None),
                                       flux_weight=wts[m])
        out["win_" + wname] = w
        log("[beam W0 %.0f window %-4s %.0f..%.0f]  full-window shift %+.4f um   main-lobe (|x|<300) shift %+.4f um   "
            "flux-weighted full %+.4f / main %+.4f   main-lobe(|x|<%.0f) %+.4f   power beyond |x|>=300: %.3e   (incident centroid %+.4f, outgoing %+.4f)"
            % (w0, wname, xs[0], xs[-1], w["shift_full_um"], w["shift_mainlobe_um"], w["shift_full_flux_um"],
               w["shift_mainlobe_flux_um"], w["mainlobe_radius_scaled_um"], w["shift_mainlobe_scaled_um"],
               w["out_frac_beyond_300"], w["cent_inc"], w["cent_out"]))
        for m in MS_R:
            o = w["orders"][str(m)]
            log("      order %+d  share of |E|^2 %.4e  lobe centroid %s um" % (m, o["frac_of_out"],
                                                                         "%.3f" % o["centroid_um"] if o["centroid_um"] is not None else "-"))
        if wname == "old":
            prof = np.column_stack([xs, Ii / ref, Io / ref])
            with io.open(os.path.join(OUT, "profile_%s_w%g.dat" % (args.tag, w0)), "w", encoding="utf-8", newline="\n") as f:
                f.write("# retro_beam_li.py cell %s rule %s ff %d nG %d W0 %g nang %d t %.2f um\n"
                        % (args.cell, args.rule, args.ff, args.nG, w0, args.nang, args.t_um))
                f.write("# x_um\tI_incident\tI_outgoing   (divided by the incident maximum)\n")
                for row in prof:
                    f.write("%.6e\t%.6e\t%.6e\n" % tuple(row))
            if args.cell == "old" and abs(w0 - 50.0) < 1e-9 and args.nang == 61 and os.path.isfile(OLD_PROFILE):
                a = np.loadtxt(OLD_PROFILE, comments="#")
                d_inc = float(np.max(np.abs(a[:, 1] - prof[:, 1])))
                d_out = float(np.max(np.abs(a[:, 2] - prof[:, 2])))
                d_x = float(np.max(np.abs(a[:, 0] - prof[:, 0])))
                rel = float(np.max(np.abs(a[:, 2] - prof[:, 2])) / max(np.max(a[:, 2]), 1e-300))
                out["control_vs_stored"] = dict(max_dx=d_x, max_d_inc=d_inc, max_d_out=d_out, max_d_out_rel_to_peak=rel,
                                                stored_shift_full=float(cent(a[:, 0], a[:, 2]) - cent(a[:, 0], a[:, 1])))
                log("[CONTROL vs stored profile_bottom_retro_TM.dat] max|dx| %.2e  max|dI_inc| %.2e  max|dI_out| %.2e (%.2e of the "
                    "outgoing peak)  stored-profile shift %+.4f um" % (d_x, d_inc, d_out, rel,
                                                                       out["control_vs_stored"]["stored_shift_full"]))
    # angular table
    with io.open(os.path.join(OUT, "angular_%s_w%g.dat" % (args.tag, w0)), "w", encoding="utf-8", newline="\n") as f:
        f.write("# kx_over_k0\ttheta_deg\tA_weight\tabs_m-2\targ_m-2_deg\tabs_m-1\targ_m-1_deg\tabs_m+0\targ_m+0_deg\n")
        for i in range(args.nang):
            f.write("%.8f\t%.6f\t%.6e" % (kxs[i], np.degrees(np.arcsin(kxs[i] / n)), abs(A[i])))
            for m in (-2, -1, 0):
                f.write("\t%.6e\t%.4f" % (abs(r[m][i]), np.degrees(np.angle(r[m][i]))))
            f.write("\n")
    res_all["w0_%g" % w0] = out
res_all["config"] = vars(args)
res_all["cell_files"] = dict(density=dens_path, record=js_path)
with open(os.path.join(OUT, "beam_%s.json" % args.tag), "w", encoding="utf-8") as f:
    json.dump(res_all, f, indent=1)
with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
log("[beam] done -> %s  (peak %.2f GB)" % (OUT, psutil.Process().memory_info().peak_wset / 1e9))
