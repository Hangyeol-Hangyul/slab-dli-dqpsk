# -*- coding: utf-8 -*-
r"""lens_total_li.py -- [TM] adopted lens total at f200 (power_centre) and the zone-weighted lens echo aperture, from a configurable superramp cell set:
    --set laurent  the adopted Laurent ladder (sim741 for x06 x09 x15 x18 x24 x36, sim740 for x12; ladder_lib.NEWPP)         [default; this is the CONTROL set]
    --set li       the sim757 Li re-run (Result\261008_scdc_tmzone_li\261008_sim757_lensli_x<MM>_sh<k>)
    --root DIR     replace the folder that holds the cells of the chosen set (same cell names), e.g. a copy
Method = total_rb0.py + lens_variants.py 'ladder (adopted, control)', copied verbatim (node / conv / betaL from ladder_lib.py, the ladder646 evaluator exec'd unmodified, pmean /
aperture / ZONE_W of echo_lens_261008.py):  class node = registration-rotated mean of own beta (amp_t{A,B}_target / lpa a_pred, rotation exp(sg 2 pi i sh / 3));  rB0 node = plain mean of
amp_rB_zero / lpa rB0 a_pred;  total = beta-L (decay, held) + deficit_p' + purity_p' f200;  lens echo aperture = zone-weighted coherent sum of the class power-mean echo_p2_db.
Printed: WITHOUT rB0, WITH rB0 (complex), WITH |rB0| (phase ignored), and the echo aperture.
LIMIT (read this): ONLY the beta-L term and the echo aperture come from the chosen cells.  The deficit term (-0.068834 dB) and the purity term (-0.017550 dB) are the stored LAURENT
atom-level values of deficit_p302_result.json / purity_p302_result.json (nG-1793 atom records); a Li re-evaluation of those needs the Li atom records and is NOT done here.
Guards:  --set li refuses (exit 3, lists the missing cells) unless all 21 cells carry superramp_result.json;  --partial instead substitutes the Laurent cells for every class that
does not have all 3 Li cells and says so;  every Li cell must carry rcwa_rule liy / rcwa_ff 1 in its JSON and match its Laurent twin (atoms_requested, nG_req, nG_ref, nG_eval_ref,
ramp_s, sc_shift, pol), a Laurent cell must carry neither Li key;  sc_shift must equal the registration index.
--control  (with --set laurent): recompute and compare with the stored total_rb0_result.json['adopted'] and lens_variants_result.json['ladder (adopted, control)'] (exact, tol 0)
and with the four dB figures / the echo figure quoted by the task (-0.4613 / -0.4707 / -0.5736 / -0.5902, echo -21.946 dB).
numpy-free (json / math / cmath), NO solver; the ladder646 evaluator head needs numpy + psutil.  Pinned to logical cores 10-17.
python -I -B lens_total_li.py [--set laurent|li] [--root DIR] [--partial] [--control] [--tag NAME]"""
import argparse
import cmath
import io
import json
import math
import os
import sys

sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import psutil                                                                  # noqa: E402

R = r"<DATA_ROOT>\Result"
WD = r"<WORK_ROOT>\loop_261008\lens_li"
LT = r"<WORK_ROOT>\loop_261008\lens_total"
SCR = r"<WORK_ROOT>\readout_261005_evening\scratch_tm646"
CLASSES = (6, 9, 12, 15, 18, 24, 36)
CL = ["x%02d" % M for M in CLASSES]
GATE_TOTAL = -0.65
GATE_ECHO = -20.5


def _lau_name(M, sh):
    if M == 12:
        return "261008_sim740_tmx12_P302_L044_P257_fx1000_sh%d_e401" % sh
    return "261008_sim741_tmP302P257_x%02d_sh%d_e401" % (M, sh)


SETS = {
    "laurent": dict(root=R + r"\261007_scdc_tmzone46", name=_lau_name),
    "li": dict(root=R + r"\261008_scdc_tmzone_li", name=lambda M, sh: "261008_sim757_lensli_x%02d_sh%d" % (M, sh)),
}


def cell_path(setname, M, sh, root=None):
    s = SETS[setname]
    return os.path.join(root or s["root"], s["name"](M, sh))


def jload(p):
    return json.load(io.open(os.path.join(p, "superramp_result.json"), encoding="utf-8"))


# ------------------------------------------------------------------ copied verbatim from ladder_lib.py (conv / betaL / db / evaluator) ----------------------------------------
def db(z):
    return 20 * math.log10(abs(z))


_EV = {}


def evaluator():
    if not _EV:
        src = io.open(SCR + r"\ladder646.py", encoding="utf-8").read()
        ns = {"__name__": "ladder646_head", "__file__": SCR + r"\ladder646.py"}
        exec(compile(src[:src.index("OUT = {}")], "ladder646_head", "exec"), ns)
        _EV.update(evaluate=ns["evaluate"], LAMG=ns["LAMG"], LAM=ns["LAM"])
    return _EV["evaluate"], _EV["LAMG"], _EV["LAM"]


def conv(pts):
    evaluate, LAMG, LAM = evaluator()
    out = []
    for (P, bA, bB) in pts:
        Mat = int(round(P * 3))
        c = math.sqrt(1.0 - (LAM / (P * LAMG)) ** 2)
        ph = cmath.exp(-1j * math.pi * (96 - 1) / (Mat * 96))
        out.append((P, bA * math.sqrt(c) * ph, bB * math.sqrt(c) * ph, 3))
    return out


def betaL(nd, f=200.0):
    evaluate = evaluator()[0]
    q = conv([(M / 3.0, nd[M]["A"], nd[M]["B"]) for M in sorted(nd)])
    return evaluate(q, f, "decay")[2], evaluate(q, f, "held")[2]


# ------------------------------------------------------------------ copied from echo_lens_261008.py (pmean / aperture) --------------------------------------------------------
ZONE_W = {"2-3L": 2.666, "3-4L": 7.462, "4-6L": 25.6, "6-12L": 47.9}
ZONE_W[">12L"] = 100.0 - sum(ZONE_W.values())
ZMAP = {"2-3L": ["x06"], "3-4L": ["x09"], "4-6L": ["x12", "x15"], "6-12L": ["x18", "x24", "x36"]}


def pmean(v):
    return 10 * math.log10(sum(10 ** (x / 10.0) for x in v) / len(v))


def aperture(md):
    amp = 0.0
    for z, w in ZONE_W.items():
        if z == ">12L":
            e = GATE_ECHO
        else:
            vals = [md[c] for c in ZMAP[z] if md.get(c) is not None]
            if not vals:
                return None
            e = pmean(vals)
        amp += (w / 100.0) * math.sqrt(10 ** (e / 10.0))
    return 10 * math.log10(amp * amp)


# ------------------------------------------------------------------ cell reading ----------------------------------------------------------------------------------------------
def read_class(paths):
    """paths: {sh: folder}.  Returns (node dict {A,B}, rB0 node, [echo_p2_db per registration]).  Same arithmetic as ladder_lib.node + total_rb0.node_r + lens_variants.node."""
    acc = {"A": 0j, "B": 0j}
    rs, ech = [], []
    lp = None
    for sh in (0, 1, 2):
        j = jload(paths[sh])
        r = j["needle_results"]["production"]
        assert int(r["sc_shift"]) == sh, (paths[sh], r["sc_shift"], sh)
        for arm, sg in (("A", -1), ("B", +1)):
            L = r["lpa"]["arm%s_s%+d" % (arm, int(r["ramp_s"]))]
            acc[arm] += complex(*r["amp_t%s_target" % arm]) / complex(L["a_pred_re"], L["a_pred_im"]) * cmath.exp(sg * 2j * math.pi * sh / 3.0) / 3
        L0 = r["lpa"]["rB0"]
        p = complex(L0["a_pred_re"], L0["a_pred_im"])
        assert lp is None or abs(p - lp) < 1e-12, (paths[sh], p, lp)            # LPA must be registration-independent (total_rb0.node_r)
        lp = p
        rs.append(complex(*r["amp_rB_zero"]) / p)
        ech.append(r["echo_p2_db"])
    return acc, sum(rs) / 3, ech


def check_li_flags(setname, j, where):
    """Li cell must carry rcwa_rule liy / rcwa_ff 1; a Laurent cell must carry neither non-default key."""
    rule, ff = j.get("rcwa_rule"), j.get("rcwa_ff")
    if setname == "li":
        if (rule, ff) != ("liy", 1):
            raise SystemExit("REFUSE: %s is not a Li cell (rcwa_rule %r, rcwa_ff %r in its superramp_result.json)" % (where, rule, ff))
    else:
        if rule not in (None, "laurent") or ff not in (None, 0):
            raise SystemExit("REFUSE: %s carries Li keys (rcwa_rule %r, rcwa_ff %r) but belongs to the Laurent set" % (where, rule, ff))


PAIR_KEYS = ("atoms_requested", "nG_req", "nG_ref", "nG_eval_ref", "ramp_s", "sc_shift", "pol", "ramp_axis_requested", "ref_mode")


def pair_check(M, sh, li_path, lau_path):
    a, b = jload(li_path), jload(lau_path)
    bad = [k for k in PAIR_KEYS if a.get(k) != b.get(k)]
    pa = a["needle_results"]["production"].get("atom_paths")
    pb = b["needle_results"]["production"].get("atom_paths")
    if pa != pb:
        bad.append("production.atom_paths")
    return bad


# ------------------------------------------------------------------ evaluation ------------------------------------------------------------------------------------------------
def load_terms():
    DEF = json.load(io.open(LT + r"\deficit_p302_result.json", encoding="utf-8"))
    PUR = json.load(io.open(LT + r"\purity_p302_result.json", encoding="utf-8"))["R"]
    return DEF["p_prime"]["deficit"], PUR["p_prime"]["f200"]["verdict_delta"]


def evaluate_cells(src_by_class):
    """src_by_class: {M: {sh: folder}} -> result dict"""
    DFC, PRT = load_terms()
    nd, nr, echo = {}, {}, {}
    for M in CLASSES:
        a, r, e = read_class(src_by_class[M])
        nd[M], nr[M], echo["x%02d" % M] = a, r, pmean(e)
    res = {}
    for var, mk in (("no_rB0", lambda M: nd[M]["B"]), ("rB0_complex", lambda M: nd[M]["B"] * nr[M]), ("rB0_abs", lambda M: nd[M]["B"] * abs(nr[M]))):
        b = betaL({M: {"A": nd[M]["A"], "B": mk(M)} for M in CLASSES})
        tot = (b[0] + DFC + PRT, b[1] + DFC + PRT)
        res[var] = dict(betaL=b, total=tot, margin=(tot[0] - GATE_TOTAL, tot[1] - GATE_TOTAL))
    ap = aperture({c: echo[c] for c in CL})
    return dict(res=res, echo_class=echo, echo_aperture=ap, deficit=DFC, purity=PRT,
                node_dB={"x%02d" % M: dict(A=db(nd[M]["A"]), B=db(nd[M]["B"]), rB0=db(nr[M]), rB0_deg=math.degrees(cmath.phase(nr[M]))) for M in CLASSES})


def describe(title, ev):
    print("== %s" % title)
    print("   class node A / B / rB0 (dB), rB0 phase (deg), class echo (dB):")
    for M in CLASSES:
        c = "x%02d" % M
        n = ev["node_dB"][c]
        print("     %s  A %+8.4f  B %+8.4f  rB0 %+8.4f (%+6.2f deg)  echo %8.3f" % (c, n["A"], n["B"], n["rB0"], n["rB0_deg"], ev["echo_class"][c]))
    for var, lab in (("no_rB0", "WITHOUT rB0"), ("rB0_complex", "WITH rB0 (complex)"), ("rB0_abs", "WITH |rB0|")):
        x = ev["res"][var]
        print("   %-20s beta-L decay %+.4f held %+.4f | TOTAL decay %+.4f held %+.4f | margin to %.2f: %+.4f / %+.4f" % (
            lab, x["betaL"][0], x["betaL"][1], x["total"][0], x["total"][1], GATE_TOTAL, x["margin"][0], x["margin"][1]))
    print("   lens echo aperture %.3f dB (gate %.1f dB, margin %+.3f dB) ; deficit_p' %+.6f , purity_p' f200 %+.6f (LAURENT atom-level values, unchanged)" % (
        ev["echo_aperture"], GATE_ECHO, GATE_ECHO - ev["echo_aperture"], ev["deficit"], ev["purity"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["laurent", "li"], default="laurent")
    ap.add_argument("--root", default=None)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    lau = {M: {sh: cell_path("laurent", M, sh) for sh in (0, 1, 2)} for M in CLASSES}
    # Laurent set always has to be complete and Li-free (it is the control and the comparison base)
    miss_l = [p for M in CLASSES for sh, p in lau[M].items() if not os.path.isfile(os.path.join(p, "superramp_result.json"))]
    if miss_l and a.set == "laurent":
        print("REFUSE: Laurent cells missing:\n  " + "\n  ".join(miss_l))
        return 3
    if a.set == "laurent":
        src = {M: {sh: cell_path("laurent", M, sh, a.root) for sh in (0, 1, 2)} for M in CLASSES}
        missing = [p for M in CLASSES for p in src[M].values() if not os.path.isfile(os.path.join(p, "superramp_result.json"))]
        if missing:
            print("REFUSE: cells missing:\n  " + "\n  ".join(missing))
            return 3
        for M in CLASSES:
            for sh in (0, 1, 2):
                check_li_flags("laurent", jload(src[M][sh]), src[M][sh])
        ev = evaluate_cells(src)
        describe("Laurent adopted ladder (sim741 + sim740 x12), root %s" % (a.root or SETS["laurent"]["root"]), ev)
        out = dict(set="laurent", root=a.root or SETS["laurent"]["root"], **ev)
    else:
        li = {M: {sh: cell_path("li", M, sh, a.root) for sh in (0, 1, 2)} for M in CLASSES}
        have = {M: [p for p in li[M].values() if os.path.isfile(os.path.join(p, "superramp_result.json"))] for M in CLASSES}
        missing = [p for M in CLASSES for p in li[M].values() if not os.path.isfile(os.path.join(p, "superramp_result.json"))]
        if missing and not a.partial:
            print("REFUSE: the Li set is PARTIAL -- %d of 21 cells missing (use --partial to substitute Laurent cells for incomplete classes):" % len(missing))
            for p in missing:
                print("  " + p)
            return 3
        full = [M for M in CLASSES if len(have[M]) == 3]
        if not full:
            print("REFUSE: no class has all 3 Li cells -- nothing to evaluate (missing %d of 21)" % len(missing))
            return 3
        src = {}
        for M in CLASSES:
            if M in full:
                src[M] = li[M]
                for sh in (0, 1, 2):
                    check_li_flags("li", jload(li[M][sh]), li[M][sh])
                    bad = pair_check(M, sh, li[M][sh], lau[M][sh])
                    if bad:
                        print("REFUSE: Li cell %s differs from its Laurent twin in %s" % (li[M][sh], bad))
                        return 4
            else:
                src[M] = lau[M]
        ev_lau = evaluate_cells(lau)
        ev = evaluate_cells(src)
        if len(full) < len(CLASSES):
            print("!! PARTIAL / MIXED lens: classes from the Li cells: %s ; classes from the LAURENT cells: %s ; missing Li cells: %d of 21" % (
                " ".join("x%02d" % M for M in full), " ".join("x%02d" % M for M in CLASSES if M not in full), len(missing)))
        describe("Laurent adopted ladder (reference)", ev_lau)
        describe("Li set%s, root %s" % (" (PARTIAL / MIXED)" if len(full) < len(CLASSES) else "", a.root or SETS["li"]["root"]), ev)
        print("== Li minus Laurent")
        for var, lab in (("no_rB0", "WITHOUT rB0"), ("rB0_complex", "WITH rB0 (complex)")):
            x, y = ev["res"][var]["total"], ev_lau["res"][var]["total"]
            print("   %-20s TOTAL decay %+.4f dB held %+.4f dB" % (lab, x[0] - y[0], x[1] - y[1]))
        print("   lens echo aperture %+.3f dB" % (ev["echo_aperture"] - ev_lau["echo_aperture"]))
        out = dict(set="li", root=a.root or SETS["li"]["root"], classes_from_li=["x%02d" % M for M in full], missing_li_cells=missing, laurent_reference=ev_lau, **ev)
    rc = 0
    if a.control:
        if a.set != "laurent":
            print("--control needs --set laurent")
            return 2
        st = json.load(io.open(LT + r"\total_rb0_result.json", encoding="utf-8"))["adopted"]
        lv = json.load(io.open(LT + r"\lens_variants_result.json", encoding="utf-8"))["ladder (adopted, control)"]
        diffs = []
        for var in ("no_rB0", "rB0_complex", "rB0_abs"):
            for i in (0, 1):
                diffs.append(("total_rb0 %s total[%d]" % (var, i), ev["res"][var]["total"][i], st[var]["total"][i]))
                diffs.append(("total_rb0 %s betaL[%d]" % (var, i), ev["res"][var]["betaL"][i], st[var]["betaL"][i]))
        for i in (0, 1):
            diffs.append(("lens_variants total_no_rB0[%d]" % i, ev["res"]["no_rB0"]["total"][i], lv["total_no_rB0"][i]))
            diffs.append(("lens_variants total_rB0[%d]" % i, ev["res"]["rB0_complex"]["total"][i], lv["total_rB0"][i]))
        diffs.append(("lens_variants echo_aperture", ev["echo_aperture"], lv["echo_aperture"]))
        for c in CL:
            diffs.append(("lens_variants class echo %s" % c, ev["echo_class"][c], lv["class_echo"][c]))
        worst = 0.0
        for nm, x, y in diffs:
            worst = max(worst, abs(x - y))
        quoted = [(ev["res"]["no_rB0"]["total"][0], -0.4613), (ev["res"]["no_rB0"]["total"][1], -0.4707), (ev["res"]["rB0_complex"]["total"][0], -0.5736),
                  (ev["res"]["rB0_complex"]["total"][1], -0.5902), (ev["echo_aperture"], -21.946)]
        ok_q = all(round(x, 4 if abs(y) < 1 else 3) == y for x, y in quoted)
        print("== CONTROL: %d stored numbers compared (total_rb0_result.json adopted, lens_variants_result.json ladder control): max |diff| = %.3e" % (len(diffs), worst))
        print("   quoted figures reproduced after rounding (-0.4613 / -0.4707 / -0.5736 / -0.5902, echo -21.946): %s   (recomputed %s)" % (
            ok_q, " / ".join("%.6f" % x for x, _ in quoted)))
        rc = 0 if (worst == 0.0 and ok_q) else 1
        print("   CONTROL %s" % ("PASS (bit-identical)" if rc == 0 else "FAIL"))
        out["control"] = dict(max_abs_diff=worst, n_compared=len(diffs), quoted_ok=ok_q)
    name = "lens_total_li_result_%s%s.json" % (a.set, ("_" + a.tag) if a.tag else "")
    json.dump(out, io.open(os.path.join(WD, name), "w", encoding="utf-8"), indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else str(o))
    print("wrote %s" % os.path.join(WD, name))
    return rc


if __name__ == "__main__":
    sys.exit(main())
