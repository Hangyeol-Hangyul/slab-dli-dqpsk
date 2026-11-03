r"""SELF-TEST for the Fig. 6(b) Li regeneration (branch C, 261015).

Derived from orig_scripts\carrier_only_v6.py.  ONLY the RCWA rule / truncation changed:
    M.set_rcwa("liy", 1)      (rcwa_rule = liy, rcwa_ff = 1  -- the basis of the printed numbers)
    NG = 257                  (nG_eval of the printed numbers; 257 = NX//2 + 1 alias ceiling)
Also: TM lane only (the deck panel is TM), density from the sim359 folder (md5 525f74f3 asserted),
termination solved exactly as in the carrier Li job (out_l3_r_list_eval of the stored Li json).

What it does: plane-wave order table of the TM carrier at +theta (pass 1) and -theta (pass 2) for ALL FIVE
evaluation terminations, compared key by key with the stored Li json
    <DATA_ROOT>\Result\261009_scdc_carli\261009_sim768_carli_525f74f3_liy\scdc_extract_H400_300.json
(keys n{tag}_1550.0_sub_{p,m}_TM_{R,T}_{m3..3}); then the eight printed 6(b) percentages are formed from the
n1708 termination and compared to their printed 2-decimal values.

Read-only on Result\ and on code\Version_unified.  Pinned to core(s) given by argv[1] (default 14) AFTER imports.
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

CORES = [int(c) for c in (sys.argv[1] if len(sys.argv) > 1 else "14").split(",")]
_vm = psutil.virtual_memory()
print("RAM used: %.1f%% (available %.2f GB); affinity %s" % (_vm.percent, _vm.available / 1e9,
                                                                 psutil.Process().cpu_affinity()), flush=True)
_lim = float(os.environ.get("SIM_MAX_RAM_PCT", "0"))   # opt-in RAM ceiling
assert _lim <= 0 or _vm.percent < _lim

HERE = os.path.dirname(os.path.abspath(__file__))
LAM_UM = 1.55
NG = 257
MS = tuple(range(-3, 4))

LI_JSON = (r"<DATA_ROOT>\Result\261009_scdc_carli\261009_sim768_carli_525f74f3_liy"
           r"\scdc_extract_H400_300.json")
TM_DENSITY = (r"<DATA_ROOT>\Result\260924_scdc_fin5"
              r"\260924_sim359_c2f90E_f100\density_scdc_extract_H400_300.npy")
for p in (LI_JSON, TM_DENSITY):
    if not os.path.isfile(p):
        print("NOT FOUND: %s" % p)
        sys.exit(1)
md5 = hashlib.md5(open(TM_DENSITY, "rb").read()).hexdigest()
print("TM density md5", md5)
assert md5 == "525f74f339190e5935b484fc583b0498", md5

with open(LI_JSON, "r", encoding="utf-8") as f:
    li = json.load(f)
assert li["rcwa_rule"] == "liy" and li["rcwa_ff"] == 1 and li["nG_eval"] == 257
r_list_eval_raw = li["out_l3_r_list_eval"]
out_l3_nm = li["out_l3_nm"]

M.set_rcwa("liy", 1)                                        # <-- the ONLY physics change vs carrier_only_v6.py
print("RCWA_RULE", M.RCWA_RULE, "RCWA_FF", M.RCWA_FF, flush=True)
M.set_period(li["period_nm"])
M.set_substrate(li["n_sub"])
M.set_out_spacer(li["out_spacer_nm"], li["n_out_spacer"])
M.LAMS = (LAM_UM,)

eps_b = (M.eps_of("a-Si"), M.eps_of("a-Si"))
eps_f1 = M.eps_of("SiO2")
eps_f2 = M.eps_of("SiO2")
dens = np.load(TM_DENSITY)
r1, r2 = dens[0], dens[1]
TH = M.THETA_SUB
print("theta_sub_deg %.12f (json %.12f)" % (np.degrees(TH), li["theta_sub_deg"]), flush=True)

ns, pairs = [], []
for tok in r_list_eval_raw.replace(",", ";").split(";"):
    tok = tok.strip()
    if not tok:
        continue
    r_s, arg_s = tok.split(":")
    rt = complex(float(r_s) * np.exp(1j * np.radians(float(arg_s))))
    nb, tb, n3, res = M.solve_2layer_for_r(rt, out_l3_nm, LAM_UM, n_min=1.178358)
    ns.append(n3)
    pairs.append((nb, tb))
    print("  token %-10s -> n_b=%.6f t_b=%.3f n3=%.6f residual=%.3e" % (tok, nb, tb, n3, res), flush=True)
ns_used = li["out_l3_ns_used"]
ns_diff = max(abs(a - b) for a, b in zip(ns, ns_used))
print("max |n3 - stored out_l3_ns_used| = %.3e" % ns_diff, flush=True)
assert ns_diff < 1e-6
M.set_out_l3_list(out_l3_nm, ns)
M.set_out_l3b_list(pairs)
tags = [M._n_tag(n) for n in ns]
print("n_tags", tags, flush=True)


def key(m):
    return "m%d" % -m if m < 0 else "%d" % m


results = {}
t0 = time.time()
for i, n3 in enumerate(ns):
    M._set_termination(i, n3)
    tag = tags[i]
    for nm, th in (("sub_p_TM", TH), ("sub_m_TM", -TH)):
        ts = time.time()
        ob, R, T = M._solve(r1, r2, 0.400, 0.300, eps_b, eps_f1, eps_f2, "sub", "TM", th, LAM_UM, NG)
        Rd, Td = M._pick(ob, np.real(R), MS), M._pick(ob, np.real(T), MS)
        results[(tag, nm)] = (Rd, Td)
        print("  solve %s %s  %.1f s   energy %.12f   RSS %.2f GB" % (
            tag, nm, time.time() - ts, float(np.real(R).sum() + np.real(T).sum()),
            psutil.Process().memory_info().rss / 1e9), flush=True)
print("all solves %.1f s" % (time.time() - t0), flush=True)

# --- compare against the stored Li json, every order, every termination ---
maxdiff = 0.0
rows = []
n_cmp = 0
for (tag, nm), (Rd, Td) in results.items():
    for m in MS:
        for kind, d in (("R", Rd), ("T", Td)):
            k = "%s_1550.0_%s_%s_%s" % (tag, nm, kind, key(m))
            stored = li[k]
            mine = float(d[m])
            diff = abs(mine - stored)
            maxdiff = max(maxdiff, diff)
            n_cmp += 1
            if max(abs(stored), abs(mine)) > 1e-6:
                rows.append((tag, nm, kind, m, stored, mine, diff))
print("compared %d stored order powers (all 5 terminations x pass1/pass2 x R/T x m=-3..3): max |diff| = %.3e" % (n_cmp, maxdiff), flush=True)
for r in rows:
    if r[0] == "n1708":
        print("  %s %-9s %s m=%+d stored %.9f  mine %.9f  diff %.2e" % r, flush=True)

# --- the eight printed percentages (n1708) ---
Rp, Tp = results[("n1708", "sub_p_TM")]
Rm, Tm = results[("n1708", "sub_m_TM")]
mine8 = [
    ("pass1 R(-2) back-flow", Rp[-2], li["n1708_back_p1_Rm2"], 27.16),
    ("pass1 T(-1) eta_plus", Tp[-1], li["n1708_eta_plus"], 28.08),
    ("pass1 R(0) R_go", Rp[0], li["n1708_R_go"], 44.68),
    ("pass1 R(-1) dump (stray_R - back)", Rp[-1], li["n1708_stray_R"] - li["n1708_back_p1_Rm2"], 0.08),
    ("pass2 R(0) back_R0_rev", Rm[0], li["n1708_back_R0_rev"], 44.68),
    ("pass2 T(+1) eta_minus", Tm[1], li["n1708_eta_minus"], 53.94),
    ("pass2 R(+2) echo", Rm[2], li["n1708_echo_worst_p2"], 0.46),
    ("pass2 R(+1) dump (1 - sum)", 1.0 - (Rm[0] + Tm[1] + Rm[2]),
     1.0 - (li["n1708_back_R0_rev"] + li["n1708_eta_minus"] + li["n1708_echo_worst_p2"]), 0.92),
]
out_rows = []
allok = True
print("\n%-36s %-14s %-14s %-9s %-9s %s" % ("quantity", "json n1708", "this run", "json %", "run %", "printed (2 dp) / match"), flush=True)
for name, mv, sv, printed in mine8:
    ok = (round(mv * 100, 2) == printed) and (round(sv * 100, 2) == printed)
    allok = allok and ok
    out_rows.append(dict(quantity=name, json_value=float(sv), run_value=float(mv), json_pct=float(sv * 100),
                         run_pct=float(mv * 100), printed_pct=printed, match=bool(ok), abs_diff_run_vs_json=float(abs(mv - sv))))
    print("%-36s %-14.9f %-14.9f %-9.4f %-9.4f %.2f %s" % (name, sv, mv, sv * 100, mv * 100, printed, "OK" if ok else "MISMATCH"), flush=True)
print("ALL EIGHT MATCH PRINTED PRECISION:", allok, flush=True)

summary = dict(rcwa_rule=M.RCWA_RULE, rcwa_ff=M.RCWA_FF, nG=NG, density_md5=md5, n_tags=tags, ns=ns, pairs=pairs,
               n_compared=n_cmp, max_abs_diff_all_stored_orders=maxdiff, eight=out_rows, all_eight_match=allok,
               theta_sub_deg=float(np.degrees(TH)), n1708_slab=dict(n_b=pairs[0][0], t_b_nm=pairs[0][1], n3=ns[0],
                                                                   t3_nm=out_l3_nm - pairs[0][1]),
               full_n1708_orders={nm: dict(R={str(m): float(Rd[m]) for m in MS}, T={str(m): float(Td[m]) for m in MS})
                                  for (tg, nm), (Rd, Td) in results.items() if tg == "n1708"})
with open(os.path.join(HERE, "selftest_li_planewave.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print("wrote", os.path.join(HERE, "selftest_li_planewave.json"), flush=True)
M.set_out_l3_list(0.0, ())
M.set_out_l3b_list(())
