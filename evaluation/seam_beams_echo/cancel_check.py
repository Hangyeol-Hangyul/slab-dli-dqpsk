# -*- coding: utf-8 -*-
"""[261015] Balanced-pair cancellation of the echo in the ACTUAL quadrant layout (SI si_v2_r13.md line 104, Note S5, Eq. S9).

Claim: translating the whole bottom pattern by d_q gives arm A = T(-1)|+th exp(+iGd), specular R(0)|+th exp(0), arm B extraction
T(+1)|-th exp(-iGd), echo order R(+2)|-th exp(-2iGd)  (verified element by element on 2026-10-13, echo_translation_check.py).
Per quadrant (plane waves, equal beam power):   A = a,   B = r0*rho*t,   E = r0*rho*e2*rho*t   (echo = B*rho*e2, one round trip later)
with a = T(-1)|+th, r0 = R(0)|+th, t = T(+1)|-th, e2 = R(+2)|-th, rho = 0.99995**0.5 (phase 0, a constant common to all quadrants).
Beat products (phases): S = A B*  ->  +2Gd;  X_A = A E*  ->  +4Gd;  X_B = B E*  ->  +2Gd.
Balanced pair (q, q'), biases 180 deg apart, shifts differ by Lambda/4 (128 px): Gd differs by 90 deg
  -> S, X_B flip sign (anti-symmetric, survive doubled),   X_A repeats after 4*90 = 360 deg (common mode, cancels).
This script solves both passes for k in {0, +32, +96, -96, -32} px (np.roll along x, the driver's own call) and tests this on the
complex amplitudes, with two needles (wrong pairing, wrong echo phase) and an end-to-end power-level check.

Stack = exactly the 10-13 stack: sim768 job file parsed (not hand-copied), carrier density 525f74f3 (eval_only), a-Si 400/300 nm,
SiO2 fill, 465 nm SiO2 out-spacer, 2-layer complex-reflection terminations of out_l3_r_list_eval (5), TM, 1550 nm, theta = +-THETA_SUB,
passes 'sub'.  RCWA code imported READ-ONLY.  Direct run: 1 BLAS thread, pinned to logical cores 10-13 AFTER all imports.
Writes only next to this script: cancel_out.txt, cancel_result.json, progress.md (append).
[261015 verify2 corrections] limits text (retro returns arm B and the echo onto the arm-A exit spot), hypothesis labels (echo/arm-B phase = R(+2)
element phase; SI exp(-3iGd) is the absolute echo field of the TRUE case), kappa(pow) uses nu_pow, RAM values printed as measured.  Original: pre_fix/.
"""
import io
import itertools
import json
import math
import os
import re
import sys
import time

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import psutil  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = r"<CODE_ROOT>\Version_unified\RCWA_custom"
JOB = r"<DATA_ROOT>\Result\261009_scdc_carli\261009_sim768_carli_525f74f3_liy\job_30_261009_sim768_carli_525f74f3_liy.txt"
STORED = r"<DATA_ROOT>\Result\261009_scdc_carli\261009_sim768_carli_525f74f3_liy\scdc_extract_H400_300.json"
REF_JSON = r"<WORK_ROOT>\atomsnap_261013\echo_translation_result.json"          # 10-13 result, READ-ONLY
OUT_TXT = os.path.join(HERE, "cancel_out.txt")
OUT_JSON = os.path.join(HERE, "cancel_result.json")
PROG = os.path.join(HERE, "progress.md")
NX = 512
KS = (0, 32, 96, -96, -32)
QUAD = {"NW": 32, "NE": 96, "SW": -96, "SE": -32}                    # np.roll shifts in px
BIAS = {"NW": -45, "NE": -135, "SW": 135, "SE": 45}                   # deg, SI Note S5 / facts_v3 G3
PAIRS_TRUE = [("NW", "SW"), ("NE", "SE")]
PAIRS_WRONG = [("NW", "NE"), ("SW", "SE"), ("NW", "SE"), ("NE", "SW")]
# e2 *= exp(i*c*Gd_q).  Labels name the echo/arm-B phase = the phase of the R(+2) element (true value -2Gd); HYP_ECHO_ABS is the matching
# echo-FIELD absolute phase = r0 (0) + e2 + t (-Gd).  The SI wording exp(-3iGd) is the ABSOLUTE echo field of the true case (e2 -2Gd), not a hypothesis.
H_A = "echo/arm-B phase (R(+2) element) -Gd"
H_B = "echo/arm-B phase (R(+2) element) 0"
H_C = "echo/arm-B phase (R(+2) element) -3Gd"
HYPS = {H_A: 1.0, H_B: 2.0, H_C: -1.0}
HYP_ECHO_ABS = {H_A: "-2Gd", H_B: "-Gd", H_C: "-4Gd"}
NEEDLE_HYP = H_A                                                      # needle (ii)
PLAN = [("liy", 1, 81, (0, 1, 2, 3, 4)), ("laurent", 0, 81, (0, 1, 2, 3, 4)), ("liy", 1, 257, (0, 2))]
RHO = 0.99995 ** 0.5
TOL_CANCEL = 1e-7           # relative residual accepted as "cancels"
TOL_NEEDLE = 0.5            # a needle must leave a relative residual above this
TOL_REPRO = 1e-10           # relative deviation from the 10-13 numbers

_log = io.open(OUT_TXT, "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log.write(s + "\n")
    _log.flush()


def PR(msg):
    with io.open(PROG, "a", encoding="utf-8") as f:
        f.write("- " + time.strftime("%H:%M:%S") + " " + msg + "\n")


def read_job(p):
    d = {}
    for ln in io.open(p, encoding="utf-8", errors="replace").read().split("\n"):
        if "=" not in ln or ln.lstrip().startswith("#"):
            continue
        k, _, v = ln.partition("=")
        d[k.strip()] = v.strip()
    return d


def wrap(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def GDrad(k):
    return 2.0 * math.pi * k / NX


def cx(z):
    return [float(np.real(z)), float(np.imag(z))]


def arms(el, k, hyp=0.0):
    """Complex fields of the three paths at one quadrant. hyp: e2 *= exp(i*hyp*Gd) (hyp = 0 is the measurement itself)."""
    e2 = el["e2"] * np.exp(1j * hyp * GDrad(k))
    A = el["a"]
    B = el["r0"] * RHO * el["t"]
    E = el["r0"] * RHO * e2 * RHO * el["t"]
    return dict(A=A, B=B, E=E, S=A * np.conj(B), XA=A * np.conj(E), XB=B * np.conj(E))


def pair_metrics(q, p):
    return dict(rA=float(abs(q["XA"] - p["XA"]) / abs(q["XA"])),
                rB=float(abs(q["XB"] + p["XB"]) / abs(q["XB"])),
                rS=float(abs(q["S"] + p["S"]) / abs(q["S"])))


def end_to_end(q, p):
    """Power-level balanced output D = P_q - P_q' for all 64 triples of DQPSK symbol phases (phi0 on arm A, phi1 on arm B = one symbol
    earlier, phi2 on the echo = two symbols earlier), computed from the field sums (no use of S, X_A, X_B in the 'measurement').
    leak = echo-induced change of D minus the pure arm-B beat 4 Re[X_B,q exp(i(phi1-phi2))], over the signal amplitude 4|S_q|."""
    leak = sig_dev = unbal = 0.0
    for p0, p1, p2 in itertools.product(range(4), repeat=3):
        f0, f1, f2 = [np.exp(1j * 0.5 * math.pi * x) for x in (p0, p1, p2)]

        def Pw(d, echo):
            return abs(d["A"] * f0 + d["B"] * f1 + (d["E"] * f2 if echo else 0.0)) ** 2
        D_full = Pw(q, True) - Pw(p, True)
        D_0 = Pw(q, False) - Pw(p, False)
        dD = D_full - D_0
        pred_B = 4.0 * float(np.real(q["XB"] * f1 * np.conj(f2)))
        norm = 4.0 * abs(q["S"])
        leak = max(leak, abs(dD - pred_B) / norm)
        sig_dev = max(sig_dev, abs(D_0 - 4.0 * float(np.real(q["S"] * f0 * np.conj(f1)))) / norm)
        # one quadrant alone (no balancing): echo-induced change of its own power, relative to its own signal beat 2|S|
        unbal = max(unbal, abs(Pw(q, True) - Pw(q, False)) / (2.0 * abs(q["S"])))
    return dict(leak=float(leak), sig_dev=float(sig_dev), unbalanced=float(unbal))


def main():
    t_start = time.time()
    P("== machine state before the run ==")
    vm = psutil.virtual_memory()
    P("   RAM measured at start: psutil avail %.2f GiB = %.2f GB decimal (total %.2f GiB, %.0f %% used)" % (
        vm.available / 2**30, vm.available / 1e9, vm.total / 2**30, vm.percent))
    P("   this script is 1-D RCWA, expected RSS < 1 GB, hard stop at 3.5 GB")
    _lim = float(os.environ.get("SIM_MAX_RAM_PCT", "0"))   # opt-in RAM ceiling
    assert _lim <= 0 or vm.percent < _lim, "RAM use above SIM_MAX_RAM_PCT -- refusing to run"
    sys.path.insert(0, CODE)
    import scd_c_grating as G                                      # noqa: E402  (read-only import)
    P("   affinity after imports:", psutil.Process().cpu_affinity(), "| BLAS threads env:", os.environ["OPENBLAS_NUM_THREADS"])
    P("scd_c_grating imported from", G.__file__)
    job = read_job(JOB)
    P("job file", JOB)
    for k in ("kind", "pol", "H1_nm", "H2_nm", "material1", "material2", "fill1", "fill2", "out_spacer_nm", "n_out_spacer", "out_l3_nm",
              "out_l3_2layer", "out_l3_r_list_eval", "lams_nm", "nG", "nG_eval", "rcwa_rule", "rcwa_ff", "period_nm", "n_sub",
              "seed_npy", "eval_only"):
        P("   %-20s = %s" % (k, job.get(k)))
    assert job["kind"] == "extract" and job["pol"] == "TM" and job["eval_only"] == "1" and job["nest"] == "0"
    dens = np.load(job["seed_npy"])
    assert dens.shape == (2, G.NX) and G.NX == NX, dens.shape
    P("density", job["seed_npy"], dens.shape, "fill L1 %.4f L2 %.4f" % (dens[0].mean(), dens[1].mean()), "binary:",
      bool(np.all((dens == 0) | (dens == 1))))

    # ---------------- stack, as Main_rcwa_scd_c.main() builds it (copied from the 10-13 script) ----------------
    G.set_substrate(float(job["n_sub"]))
    _p, th_deg = G.set_period(float(job["period_nm"]))
    G.LAMS = tuple(float(v) * 1e-3 for v in job["lams_nm"].replace(",", " ").split())
    H1, H2 = float(job["H1_nm"]) * 1e-3, float(job["H2_nm"]) * 1e-3
    eps_b = (G.eps_of(job["material1"]), G.eps_of(job["material2"]))
    e_f1, e_f2 = G.eps_of(job["fill1"]), G.eps_of(job["fill2"])
    G.set_gap(float(job.get("gap_nm", "0") or "0"))
    G.set_gap_list(())
    G.set_out_spacer(float(job["out_spacer_nm"]), float(job["n_out_spacer"]))
    G.set_out_l3b_list(())
    G.set_out_l3b(0.0)
    t3 = float(job["out_l3_nm"])
    nmin = 1.178358
    lu = sorted(G.LAMS)[len(G.LAMS) // 2]
    toks = [t.strip() for t in job["out_l3_r_list_eval"].replace(",", ";").split(";") if t.strip()]
    ns, pairs = [], []
    P("terminations (out_l3_r_list_eval), total thickness %.1f nm, design lambda %.4f um, THETA_SUB %.4f deg, Lambda %.2f nm, px %.4f nm:" % (
        t3, lu, th_deg, G.PERIOD * 1e3, G.PX_NM))
    for tok in toks:
        a, b = tok.split(":")[:2]
        rt = float(a) * np.exp(1j * np.radians(float(b)))
        nb, tb, n3, res = G.solve_2layer_for_r(rt, t3, lu, n_min=nmin, lams_um=None)
        assert res <= 0.005, (tok, res)
        ns.append(n3)
        pairs.append((nb, tb))
        P("   %-9s n_b %.4f t_b %.1f nm n3 %.4f residual %.2e" % (tok, nb, tb, n3, res))
    assert len(set("%.6f" % v for v in ns)) == len(ns)
    G.set_out_l3_list(t3, ns)
    G.set_out_l3b_list(pairs)
    th = float(G.THETA_SUB)
    ms = (-1, 0, 1, 2)
    ref = json.load(io.open(REF_JSON, encoding="utf-8"))
    px_nm = float(G.PX_NM)

    def solve_pass(rho, ng, side_sign, pol="TM"):
        ob, R, T = G._solve(rho[0], rho[1], H1, H2, eps_b, e_f1, e_f2, "sub", pol, side_sign * th, lu, ng)
        n = ob.nG
        _a0, b0 = ob.GetAmplitudes(0, 0.0)
        aN, _bN = ob.GetAmplitudes(ob.Layer_N - 1, 0.0)
        b0, aN = np.asarray(b0), np.asarray(aN)
        off = n if pol == "TM" else 0
        out = {"nG_actual": int(n)}
        Rr, Tr = np.real(np.asarray(R)), np.real(np.asarray(T))
        for m in ms:
            i = G.T1._order_index(ob, m)
            out["R%+d" % m] = complex(b0[i + off])
            out["T%+d" % m] = complex(aN[i + off])
            out["PR%+d" % m] = float(Rr[i])
            out["PT%+d" % m] = float(Tr[i])
        return out

    P("== quadrant layout used (checked against SI Eq. S5 / Note S5 and facts_v3 G3) ==")
    P("   quadrant  bias(deg)  shift d (nm)   k (px of 512)   Gd (deg)   2Gd (deg)")
    for q in ("NW", "NE", "SW", "SE"):
        k = QUAD[q]
        P("   %-8s %8d  %12.2f  %14d  %9.3f  %9.3f" % (q, BIAS[q], k * px_nm, k, 360.0 * k / NX, 720.0 * k / NX))
    P("   SI Eq. S5: I = P(-45) - P(+135) -> pair (NW, SW);  Q = P(+45) - P(-135) -> pair (SE, NE).  Biases of a pair differ by 180 deg.")
    P("   Pair shift difference NW-SW = %.2f nm = Lambda/4 (128 px); NE-SE likewise. Both pairs straddle y = 0 (seam Lambda/4 of facts_v3 G3)." % (
        (QUAD['NW'] - QUAD['SW']) * px_nm))
    P("   rho (retroreflection amplitude) = 0.99995**0.5 = %.10f, phase 0 (constant common to all quadrants; no influence on the phases)." % RHO)

    results = {}
    rows = []
    repro_lines = []
    worst_repro = 0.0
    worst_phase_resid = 0.0
    worst_damp = 0.0
    worst_ref_phase = 0.0
    worst_echo_abs = 0.0
    max_rss = 0.0
    for rule, ff, ng, term_idx in PLAN:
        G.set_rcwa(rule, ff)
        for it in term_idx:
            n3 = ns[it]
            G._set_termination(it, n3)
            key = "%s_ff%d_nG%d_term%d_%s" % (rule, ff, ng, it, toks[it].replace(":", "@"))
            t0 = time.time()
            els = {}
            raw = {}
            for k in KS:
                rho_d = np.vstack([np.roll(dens[0], k), np.roll(dens[1], k)])
                pp = solve_pass(rho_d, ng, +1)
                pm = solve_pass(rho_d, ng, -1)
                els[k] = dict(a=pp["T-1"], r0=pp["R+0"], t=pm["T+1"], e2=pm["R+2"])
                raw[k] = dict(PTa=pp["PT-1"], PR0=pp["PR+0"], PTt=pm["PT+1"], PR2=pm["PR+2"], nG_actual=pp["nG_actual"])
            rss = psutil.Process().memory_info().rss / 2**30
            max_rss = max(max_rss, rss)
            assert rss < 3.5, "RSS %.2f GB above the 3.5 GB stop" % rss
            # ---- per-element translation identity (re-verification of 10-13 on the new shifts) ----
            pred = {"a": +1.0, "r0": 0.0, "t": -1.0, "e2": -2.0}
            resid = {}
            for el in ("a", "r0", "t", "e2"):
                r = []
                for k in KS:
                    meas = np.degrees(np.angle(els[k][el] / els[0][el]))
                    r.append(abs(float(wrap(meas - pred[el] * 360.0 * k / NX))))
                resid[el] = max(r)
                worst_phase_resid = max(worst_phase_resid, resid[el])
            damp = max(abs(abs(els[k][el]) / abs(els[0][el]) - 1.0) for k in KS for el in ("a", "r0", "t", "e2"))
            worst_damp = max(worst_damp, damp)
            # ---- reproduction of the 10-13 numbers (d = 0 amplitudes/powers; phase change at k = 32) ----
            rk = ref["results"].get(key)
            if rk is not None:
                m0 = {"T-1|+th": ("a", "PTa"), "R0|+th": ("r0", "PR0"), "T+1|-th": ("t", "PTt"), "R+2|-th": ("e2", "PR2")}
                devs = []
                for nm, (el, pk) in m0.items():
                    d_abs = abs(abs(els[0][el]) - rk["abs_d0"][nm]) / rk["abs_d0"][nm]
                    d_pow = abs(raw[0][pk] - rk["pow_d0"][nm]) / rk["pow_d0"][nm]
                    devs += [d_abs, d_pow]
                row32 = [x for x in rk["rows"] if x["k"] == 32][0]
                dph = []
                for nm, (el, pk) in m0.items():
                    mine = float(np.degrees(np.angle(els[32][el] / els[0][el])))
                    dph.append(abs(float(wrap(mine - row32["meas"][nm]))))
                worst_repro = max(worst_repro, max(devs))
                worst_ref_phase = max(worst_ref_phase, max(dph))
                repro_lines.append("   %-34s |R(+2)|-th|_k0 %.16f (10-13: %.16f)  power %.16f (10-13: %.16f)  max rel dev (|amp|,power; 4 elements) %.1e  max dphase@k32 %.1e deg" % (
                    key, abs(els[0]["e2"]), rk["abs_d0"]["R+2|-th"], raw[0]["PR2"], rk["pow_d0"]["R+2|-th"], max(devs), max(dph)))
            else:
                repro_lines.append("   %-34s (no 10-13 record for this configuration)" % key)
            # ---- quadrant algebra ----
            quad = {q: arms(els[QUAD[q]], QUAD[q]) for q in QUAD}
            E0 = arms(els[0], 0)["E"]                              # echo field of the unshifted pattern
            for q in QUAD:                                         # absolute echo-field phase law: r0 (0) + e2 (-2Gd) + t (-Gd) = -3Gd
                worst_echo_abs = max(worst_echo_abs, abs(float(wrap(np.degrees(np.angle(quad[q]["E"] / E0)) + 3.0 * 360.0 * QUAD[q] / NX))))
            cfg = dict(key=key, rule=rule, ff=ff, nG=ng, term=it, term_tok=toks[it], n3=n3)
            for pr in PAIRS_TRUE:
                m = pair_metrics(quad[pr[0]], quad[pr[1]])
                e = end_to_end(quad[pr[0]], quad[pr[1]])
                rows.append(dict(cfg, kind="true", pair="%s-%s" % pr, hyp=None, **m, **e))
            for pr in PAIRS_WRONG:
                m = pair_metrics(quad[pr[0]], quad[pr[1]])
                e = end_to_end(quad[pr[0]], quad[pr[1]])
                rows.append(dict(cfg, kind="wrong_pair", pair="%s-%s" % pr, hyp=None, **m, **e))
            for hn, hc in HYPS.items():
                hq = {q: arms(els[QUAD[q]], QUAD[q], hyp=hc) for q in QUAD}
                for pr in PAIRS_TRUE:
                    m = pair_metrics(hq[pr[0]], hq[pr[1]])
                    e = end_to_end(hq[pr[0]], hq[pr[1]])
                    rows.append(dict(cfg, kind="hyp", pair="%s-%s" % pr, hyp=hn, **m, **e))
            # ---- |X_B|/|S| versus kappa ----
            kap = {}
            for q in QUAD:
                el = els[QUAD[q]]
                rw = raw[QUAD[q]]
                c_plus = rw["PTa"] / abs(el["a"]) ** 2            # power/|amp|^2 of T(-1) at +theta
                c_minus = rw["PTt"] / abs(el["t"]) ** 2           # power/|amp|^2 of T(+1) at -theta
                nu = abs(el["e2"]) * RHO
                nu_pow = math.sqrt(rw["PR2"]) * RHO               # nu from the solver power PR2 (not from the amplitude)
                ba_amp = abs(quad[q]["B"]) / abs(quad[q]["A"])
                ba_pow = math.sqrt(rw["PR0"] * RHO ** 2 * rw["PTt"] / rw["PTa"])
                kap[q] = dict(nu=nu, nu_pow=nu_pow, nu_rel=abs(nu / nu_pow - 1.0), ba_amp=ba_amp, ba_pow=ba_pow, kappa_amp=ba_amp * nu, kappa_pow=ba_pow * nu_pow,
                              XB_over_S=abs(quad[q]["XB"]) / abs(quad[q]["S"]), XA_over_S=abs(quad[q]["XA"]) / abs(quad[q]["S"]),
                              c_plus=c_plus, c_minus=c_minus, e2_dB=10 * math.log10(rw["PR2"]))
            results[key] = dict(rule=rule, ff=ff, nG=ng, nG_actual=raw[0]["nG_actual"], term=toks[it], n3=n3,
                                amps={str(k): {n: cx(v) for n, v in els[k].items()} for k in KS},
                                powers={str(k): raw[k] for k in KS},
                                arms={q: {n: cx(v) for n, v in quad[q].items()} for q in QUAD},
                                phase_identity_resid_deg=resid, max_rel_amp_change=damp, kappa=kap)
            k0 = kap["NW"]
            P("[%s] nG actual %d, %.1f s, RSS %.2f GB | translation-identity max |resid| deg %s | max rel |amp| change %.1e | nu %.4f (%.2f dB) |B|/|A| %.4f kappa %.5f |XB|/|S| %.5f" % (
                key, raw[0]["nG_actual"], time.time() - t0, rss, {n: "%.1e" % v for n, v in resid.items()}, damp, k0["nu"], k0["e2_dB"],
                k0["ba_amp"], k0["kappa_amp"], k0["XB_over_S"]))
            PR("done %s (%.1f s)" % (key, time.time() - t0))

    # =====================================================================================================================
    P("== check 1: reproduction of the 10-13 numbers (same stack, same code path; d = 0 for amplitudes/powers, k = 32 for phases) ==")
    for ln in repro_lines:
        P(ln)
    ok_repro = worst_repro < TOL_REPRO and worst_ref_phase < 1e-6
    P("   worst relative deviation of |amp| and power from the 10-13 result over all configurations: %.2e (limit %.0e)" % (worst_repro, TOL_REPRO))
    P("   worst phase difference at k = 32 against the 10-13 rows: %.2e deg (limit 1e-6)" % worst_ref_phase)
    stored = json.load(io.open(STORED, encoding="utf-8"))
    sdev = 0.0
    for it in (0, 2):
        key = "liy_ff1_nG257_term%d_%s" % (it, toks[it].replace(":", "@"))
        tag = G._n_tag(ns[it])
        lamtag = "%.1f" % (lu * 1e3)
        mp = {"PTa": "%s_%s_sub_p_TM_T_m1" % (tag, lamtag), "PR0": "%s_%s_sub_p_TM_R_0" % (tag, lamtag),
              "PTt": "%s_%s_sub_m_TM_T_1" % (tag, lamtag), "PR2": "%s_%s_sub_m_TM_R_2" % (tag, lamtag)}
        for pk, sk in mp.items():
            sdev = max(sdev, abs(results[key]["powers"]["0"][pk] - stored[sk]) / abs(stored[sk]))
    P("   d = 0 powers (Li, nG 257, terminations 0 and 2) against the STORED sim768 record %s: worst relative deviation %.2e" % (os.path.basename(STORED), sdev))
    ok_repro = ok_repro and sdev < 1e-9

    P("== check 2: per-element translation identity on the new shifts k in %s (all configurations) ==" % (KS,))
    P("   worst |phase residual| over all rules, nG, terminations, shifts, elements: %.2e deg; worst relative |amp| change under shifts: %.2e" % (
        worst_phase_resid, worst_damp))
    ok_ident = worst_phase_resid < 1e-6 and worst_damp < 1e-9

    def sel(kind, **kw):
        return [r for r in rows if r["kind"] == kind and all(r[a] == b for a, b in kw.items())]

    groups = [(rule, ng) for rule, ff, ng, _t in PLAN]
    P("== check 3: balanced-pair cancellation, TRUE pairs (NW,SW) and (NE,SE); max over the terminations of each rule/nG ==")
    P("   rA = |X_A(q) - X_A(q')| / |X_A(q)|  (echo-arm A beat, must vanish)   rB = |X_B(q) + X_B(q')| / |X_B(q)|   rS = |S(q) + S(q')| / |S(q)|")
    P("   leak = end-to-end: echo-induced change of the balanced power D = P_q - P_q' minus 4 Re[X_B,q e^{i(phi1-phi2)}], max over the 64 symbol-phase triples, / 4|S_q|")
    P("   %-10s %-5s %-9s %5s | %-10s %-10s %-10s | %-10s %-10s" % ("rule", "nG", "pair", "terms", "rA", "rB", "rS", "leak", "sig dev"))
    worst_true = 0.0
    for rule, ng in groups:
        for pr in PAIRS_TRUE:
            rr = sel("true", rule=rule, nG=ng, pair="%s-%s" % pr)
            mx = {n: max(r[n] for r in rr) for n in ("rA", "rB", "rS", "leak", "sig_dev")}
            worst_true = max(worst_true, mx["rA"], mx["rB"], mx["rS"], mx["leak"], mx["sig_dev"])
            P("   %-10s %-5d %-9s %5d | %-10.2e %-10.2e %-10.2e | %-10.2e %-10.2e" % (rule, ng, "%s-%s" % pr, len(rr), mx["rA"], mx["rB"], mx["rS"], mx["leak"], mx["sig_dev"]))
    ok_cancel = worst_true < TOL_CANCEL
    P("   worst over everything: %.2e (limit %.0e) -> cancellation holds: %s" % (worst_true, TOL_CANCEL, ok_cancel))
    ub = [r["unbalanced"] for r in rows if r["kind"] == "true"]
    P("   context: echo-induced change of ONE quadrant's own power (no balancing), relative to its signal beat 2|S|: %.4f..%.4f (nu-sized, from echo-A + echo-B beats)" % (min(ub), max(ub)))

    P("== needle (i): WRONG pairing (biases NOT 180 deg apart); true e2 phase -2Gd; min / max over all configurations ==")
    ok_n1 = True
    for pr in PAIRS_WRONG:
        rr = [r for r in rows if r["kind"] == "wrong_pair" and r["pair"] == "%s-%s" % pr]
        mnA, mxA = min(r["rA"] for r in rr), max(r["rA"] for r in rr)
        mnB, mxB = min(r["rB"] for r in rr), max(r["rB"] for r in rr)
        mnS, mxS = min(r["rS"] for r in rr), max(r["rS"] for r in rr)
        mnL = min(r["leak"] for r in rr)
        dk = abs(QUAD[pr[0]] - QUAD[pr[1]])
        P("   pair %s-%s (|dk| %3d px, dbias %4d deg): rA %.4f..%.4f   rB %.4f..%.4f   rS %.4f..%.4f   end-to-end leak min %.4f" % (
            pr[0], pr[1], dk, abs(BIAS[pr[0]] - BIAS[pr[1]]), mnA, mxA, mnB, mxB, mnS, mxS, mnL))
        ok_n1 &= mnA > TOL_NEEDLE
    P("   needle (i) discriminates (X_A difference > %.1f for every wrong pairing, expected 2 or sqrt 2): %s" % (TOL_NEEDLE, ok_n1))

    P("== needle (ii): WRONG echo/arm-B phase on the TRUE pairs (e2 = the R(+2) element multiplied by exp(i c Gd_q); measured value -2Gd); min / max over all configurations ==")
    ok_n2 = True
    for hn in HYPS:
        for pr in PAIRS_TRUE:
            rr = [r for r in rows if r["kind"] == "hyp" and r["hyp"] == hn and r["pair"] == "%s-%s" % pr]
            mnA, mxA = min(r["rA"] for r in rr), max(r["rA"] for r in rr)
            mnB, mxB = min(r["rB"] for r in rr), max(r["rB"] for r in rr)
            mxS = max(r["rS"] for r in rr)
            mnL, mxL = min(r["leak"] for r in rr), max(r["leak"] for r in rr)
            P("   %-38s [echo-field absolute phase %s] pair %s-%s: rA %.4f..%.4f   rB %.4f..%.4f   rS (unchanged) <= %.1e   end-to-end leak %.4f..%.4f (nu-sized)" % (
                hn, HYP_ECHO_ABS[hn], pr[0], pr[1], mnA, mxA, mnB, mxB, mxS, mnL, mxL))
            if hn == NEEDLE_HYP:
                ok_n2 &= (mnA > TOL_NEEDLE)
    P("   needle (ii) discriminates (-Gd hypothesis: X_A difference > %.1f; expected sqrt 2): %s" % (TOL_NEEDLE, ok_n2))
    P("   SI exp(-3iGd) refers to the echo FIELD (r0 0 + e2 -2Gd + t -Gd); it equals the measured case (e2 -2Gd, echo/arm B -2Gd) and passes."
      " [measured: max |arg(E_q/E_0) + 3Gd_q| = %.2e deg over all configurations and quadrants; pair cancellation check above: %s]" % (worst_echo_abs, ok_cancel))

    P("== check 4: |X_B|/|S| versus kappa = (|B|/|A|) nu,  nu = |e2| rho (SI: nu = (0.99995 eps)^(1/2)); per termination, NW quadrant (all quadrants listed in the JSON) ==")
    P("   Identity: |X_B|/|S| = |B||E|/(|A||B|) = |E|/|A| = (|B|/|A|) nu = kappa, and |X_A|/|S| = |E|/|B| = nu. The test is on the bookkeeping, incl. power vs amplitude normalisation of T.")
    P("   kappa(amp) = (|B|/|A|) |e2| rho from the complex amplitudes;  kappa(pow) = kappa from solver powers (PT, PR0, PR2) = (PR0 rho^2 PTt/PTa)^(1/2) * (PR2)^(1/2) rho, no amplitude entering.")
    P("   %-34s %9s %9s %9s %10s %10s %11s %11s %9s" % ("config", "R2 [dB]", "nu", "|B|/|A|", "kappa(amp)", "kappa(pow)", "|XB|/|S|", "|XA|/|S|", "XB/S/kap-1"))
    worst_kap = 0.0
    worst_kap_pow = 0.0
    worst_nu = 0.0
    worst_spread = 0.0
    for key, r in results.items():
        kq = r["kappa"]
        k0 = kq["NW"]
        dev = max(abs(v["XB_over_S"] / v["kappa_amp"] - 1.0) for v in kq.values())
        dev_p = max(abs(v["XB_over_S"] / v["kappa_pow"] - 1.0) for v in kq.values())
        spread = max(abs(v["kappa_amp"] / k0["kappa_amp"] - 1.0) for v in kq.values())
        worst_kap, worst_kap_pow, worst_spread = max(worst_kap, dev), max(worst_kap_pow, dev_p), max(worst_spread, spread)
        worst_nu = max(worst_nu, max(v["nu_rel"] for v in kq.values()))
        P("   %-34s %9.3f %9.5f %9.5f %10.6f %10.6f %11.6f %11.6f %9.1e" % (key, k0["e2_dB"], k0["nu"], k0["ba_amp"], k0["kappa_amp"], k0["kappa_pow"],
                                                                         k0["XB_over_S"], k0["XA_over_S"], dev))
    P("   worst |XB/S / kappa(amp) - 1| over 4 quadrants x all configurations: %.2e;  against kappa from solver powers (PT, PR0, PR2): %.2e" % (worst_kap, worst_kap_pow))
    P("   worst |nu/nu_pow - 1| (nu = |e2| rho from the amplitude, nu_pow = PR2^(1/2) rho from the solver power) over 4 quadrants x all configurations: %.2e (limit 1e-9)" % worst_nu)
    P("   quadrant-to-quadrant spread of kappa (translation invariance of the amplitudes): %.2e" % worst_spread)
    cpm = max(abs(v["c_plus"] / v["c_minus"] - 1.0) for r in results.values() for v in r["kappa"].values())
    P("   power/|amp|^2 factor of T(-1)|+th versus T(+1)|-th (must be equal for |B|/|A| in amplitude = in power): worst relative difference %.2e" % cpm)
    ok_kap = worst_kap < 1e-9 and worst_kap_pow < 1e-6 and worst_nu < 1e-9

    # ---------------- sensitivity and limits (algebra on the verified identity, not extra solves) ----------------
    G_per_nm = 360.0 / (G.PERIOD * 1e3)
    P("== limits (plane-wave per-quadrant amplitudes, equal beam power per quadrant, no seams / beam overlap / PD registration) ==")
    P("   Power imbalance of a pair: quadrant powers p_q, p_q' (all three arms of a quadrant scale with its power), delta = |p_q - p_q'| / p_q.")
    P("     X_A(q) - X_A(q') = delta * X_A(q)  ->  residual / X_A = delta  (task formula); in the balanced output D the leftover echo-A term relative to the")
    P("     signal amplitude is  nu*delta/(2 - delta) ~ nu*delta/2, against the surviving echo-B term kappa; leftover/surviving = delta/(2 - delta) * |A|/|B|.")
    nus = [v["NW"]["nu"] for v in (r["kappa"] for r in results.values())]
    kas = [v["NW"]["kappa_amp"] for v in (r["kappa"] for r in results.values())]
    P("     measured here: nu %.4f..%.4f, kappa %.4f..%.4f over the terminations; for delta = 10 %% (0.46 dB): leftover/signal = nu*0.0526 = %.2e..%.2e;" % (
        min(nus), max(nus), min(kas), max(kas), min(nus) * 0.1 / 1.9, max(nus) * 0.1 / 1.9))
    P("     for delta = 20 %% (0.97 dB): %.2e..%.2e (SI criterion nu = 9.44 %%: %.2e / %.2e)." % (min(nus) * 0.2 / 1.8, max(nus) * 0.2 / 1.8, 0.0944 * 0.1 / 1.9, 0.0944 * 0.2 / 1.8))
    P("   Pair-shift error: the X_A cancellation needs the shift difference of a pair to be a multiple of Lambda/4 (4*Gd repeats after 360 deg), and the S / X_B sign flip")
    P("     needs Lambda/4 modulo Lambda/2 (2*Gd differs by 180 deg); together exactly Lambda/4 + n*Lambda/2 (here 328.85 nm). An error eps (nm) in")
    P("     the pair separation leaves residual / X_A = 2|sin(2 G eps)| ~ 4 G eps = %.5f per nm (%.4f deg per nm); one pixel (%.4f nm) gives %.4f" % (
        4 * math.radians(G_per_nm), 4 * G_per_nm, px_nm, 2 * math.sin(math.radians(2 * G_per_nm * px_nm))))
    P("     (algebra from the verified identity: X_A phase = 4 G d_q, X_B and S phase = 2 G d_q; the same eps makes the S / X_B sign-flip error 2G eps = %.4f deg/nm.)" % (2 * G_per_nm))
    P("   Not covered: finite-beam overlap with the quadrant seams. The retroreflective grating returns arm B and the echo onto the exit spot of arm A (net lateral displacement 0 for a plane wave;")
    P("     main-lobe centroid -0.0145 um per return for w0 = 100 um, SI Note S2 line 76 of", r"<WORK_ROOT>\paper_scdc_261011_v3\draft\si_v2_r13.md", "and facts_v3.md line 70);")
    P("     the overlap of the finite beam with the seams is not modelled. Also not covered: PD registration, finite-aperture quadrants, wavelength/temperature (a common bias offset leaves the pair relation intact, since it rotates all four biases together).")

    ok_all = ok_repro and ok_ident and ok_cancel and ok_n1 and ok_n2 and ok_kap
    P("== VERDICT ==")
    P("   reproduction of the 10-13 numbers: %s | translation identity on new shifts: %s | pair cancellation (X_A common mode, S and X_B anti-symmetric): %s" % (
        ok_repro, ok_ident, ok_cancel))
    P("   needle (i) wrong pairing fails as it must: %s | needle (ii) wrong echo phase fails as it must: %s | |X_B|/|S| = kappa: %s" % (ok_n1, ok_n2, ok_kap))
    P("   AFFINITY at the end:", psutil.Process().cpu_affinity())
    pk = psutil.Process().memory_info()
    P("peak working set %.2f GB, wall %.1f s" % (getattr(pk, "peak_wset", pk.rss) / 2**30, time.time() - t_start))
    json.dump(dict(job=JOB, rho=RHO, quad_k=QUAD, bias_deg=BIAS, pairs_true=PAIRS_TRUE, pairs_wrong=PAIRS_WRONG, hyps=HYPS, ks=list(KS),
                   px_nm=px_nm, period_nm=G.PERIOD * 1e3, theta_sub_deg=float(np.degrees(G.THETA_SUB)),
                   verdict=dict(repro=bool(ok_repro), identity=bool(ok_ident), cancel=bool(ok_cancel), needle_pair=bool(ok_n1),
                                needle_phase=bool(ok_n2), kappa=bool(ok_kap), worst_true=worst_true, worst_repro=worst_repro,
                                worst_phase_resid_deg=worst_phase_resid, worst_amp_change=worst_damp, worst_kappa_dev=worst_kap,
                                worst_kappa_pow_dev=worst_kap_pow, worst_nu_rel=worst_nu, worst_echo_abs_phase_deg=worst_echo_abs),
                   results=results, rows=rows),
              io.open(OUT_JSON, "w", encoding="utf-8"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    return 0 if ok_all else 1


if __name__ == "__main__":
    rc = main()
    _log.close()
    sys.exit(rc)
