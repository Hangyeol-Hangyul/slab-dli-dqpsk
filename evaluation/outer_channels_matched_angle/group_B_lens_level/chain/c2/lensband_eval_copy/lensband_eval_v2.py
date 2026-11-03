# -*- coding: utf-8 -*-
r"""lensband_eval_v2.py -- lensband_eval.py (v1, imported UNMODIFIED, keep both files together) + the ATOM-LEVEL TERMS deficit_p' and purity_p' f200 at the channel.

v1 builds the x-only lens total = betaL(cells) + deficit_p' + purity_p' with the two atom-level terms frozen at their 1550 values (-0.06152057785449305 / -0.018900632905384462).
v2 recomputes those two terms at each channel from the sim783 records of the eight P769 atoms with the SAME code chain as a1_def_pur.py (its head is exec'd verbatim, only the
scratch folder WD redirected into this folder; deficit646 -> palette_lens_options.deficit_tool, purity646 -> row_from / cases / T200), so the terms are bit-comparable.

INPUT MAP per channel (every input is read by name, printed, validated; a missing / inconsistent input makes the channel's terms REFUSED, never filled):
  level      atom record (A, rec, eff)                                                               delay-arm job (tok, pm)
  blank_122p5 .. level_347p5  Result\261012_scdc_wlband\261012_sim783_wl_a<lv>_<md5>_l<chan>\atom2d_opt.json     Result\261012_scdc_wlband\261012_sim783_wl_tok_525f74f3_<lv>_b11\scdc_extract_H400_300.json
  A    = atom2d_opt.json A_deg mod 360          (a1_def_pur.load_level: float(a["A_deg"]) % 360)
  rec  = atom2d_opt.json purity                 (reference-arm purity of the record, nG 1793)
  tok  = tok json  n<ns>_<CH>_purity            (reference-arm purity at the channel; a1: purity_worst of the one-channel pm job)
  pm   = tok json  n<ns>_<CH>_puritym           (delay-arm (-theta) purity at the channel; a1: purity_m_worst)
  CH = "%.1f" % channel nm (driver label, scd_c_grating.py:2589), ns = out_l3_ns_used[0] -> prefix n%d_ % round(ns*1000)  (a1_def_pur.load_level)
  row_from: uA = max(u(rec), u(tok)), uB = u(pm), u = 1.368 sqrt(p/(1-p));  purity term = cases(T200, rows)['verdict_delta'];  deficit = deficit_tool(A)['deficit'].
Caveats carried from the sim783 job log (2026-10-12 14:14): the delay-arm token is held at its 1550 nm value (fixed 2-layer termination), fixed 1550 nm angle, nG_eval 161, one termination.

RUN (about 1.5 min, no solver):  python -I -B <WORK_ROOT>\wlband_261012\lensband_eval\lensband_eval_v2.py
  --atom-terms channel|1550   (default channel; 1550 = v1 behaviour, terms frozen at 1550, atom machinery loaded only for the controls)
  --channels 1546.003,1550,1554.017   --models fixed|all|list (default fixed)   --no-control   --terms-all-channels (print the two terms on all 11 sim783 channels)
Writes lensband_eval_v2_stdout.txt and lensband_eval_v2_result.json next to this file; exit 0 = all controls passed."""
import os
import sys

sys.dont_write_bytecode = True
WD = r"<WORK_ROOT>\paper_scdc_261011_v3\review\tbd_analysis_261014\lensband_eval_copy"
sys.path.insert(0, WD)
import argparse                                                                  # noqa: E402
import contextlib                                                                # noqa: E402
import io                                                                        # noqa: E402
import json                                                                      # noqa: E402
import math                                                                      # noqa: E402
import re                                                                        # noqa: E402
import time                                                                      # noqa: E402
import lensband_eval as V1                                                       # noqa: E402  (v1, unmodified: pins cores 10-17, threads 1)
import psutil                                                                    # noqa: E402

R = V1.R
WLB = R + r"\261012_scdc_wlband"
A1SRC = r"<WORK_ROOT>\loop_261008\improve_261009\tm_lens_total_a1\a1_def_pur.py"
A1WD_LINE = 'WD =r"<WORK_ROOT>\\loop_261008\\improve_261009\\tm_lens_total_a1"'
CH11 = ["l1546p003", "l1546p801", "l1547p600", "l1548p399", "l1549p199", "l1550p000", "l1550p802", "l1551p604", "l1552p408", "l1553p212", "l1554p017"]
CHMAP = {"1546.003": "l1546p003", "1550": "l1550p000", "1554.017": "l1554p017"}      # v1 channel names -> sim783 folder tags
# (name in the a1 NAMES list, atom tag, atom md5, tok tag)
LEVELS = [("blank_122p5", "122p5", "b59ed0ba", "122p5"), ("level_032p5", "032p5", "53474519", "032p5"), ("level_077p5", "077p5", "f8176952", "077p5"),
          ("level_167p5", "167p5", "ed750b02", "167p5"), ("level_212p5", "212p5", "78259586", "212p5"), ("level_257p5", "257p5", "accded83", "257acc"),
          ("level_302p5", "302p5", "ee3609ef", "302p5"), ("level_347p5", "347p5", "accf4825", "347p5")]
LOG = []
V1_SAY = V1.say


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


V1.say = say                                                                     # v1's functions log through this module's log
T0 = time.time()


def tag_nm(tag):
    return float(tag[1:].replace("p", "."))


# ======================================================================= the a1_def_pur.py chain, exec'd verbatim ============================================================
_AT = {}


def atom_machinery():
    """exec the head of a1_def_pur.py (deficit tool + purity machinery + load_level / palette_eval) with WD redirected here; ~60 s, once"""
    if _AT:
        return _AT
    src = io.open(A1SRC, encoding="utf-8").read()
    assert src.count(A1WD_LINE) == 1, "WD line of a1_def_pur.py not found"
    src = src.replace(A1WD_LINE, "WD = " + repr(WD))
    head = src[:src.index('OUT = {"controls"')]
    ns = {"__name__": "a1_head", "__file__": A1SRC}
    buf = io.StringIO()
    t = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(head, "a1_def_pur_head", "exec"), ns)
    _AT.update(ns=ns, deficit_tool=ns["deficit_tool"], row_from=ns["row_from"], cases=ns["cases"], T200=ns["T200"], NAMES=ns["NAMES"], palette_eval=ns["palette_eval"],
               P763=ns["P763"], AC=ns["AC"], PM=ns["PM"], Q8=ns["Q8"], head_log=buf.getvalue())
    say("atom machinery: a1_def_pur.py head exec'd in %.1f s (stdout of the head captured; its own control lines follow)" % (time.time() - t))
    for ln in buf.getvalue().splitlines():
        if "CONTROL" in ln:
            say("   [a1 head] " + ln.strip())
    assert [n for n, *_ in LEVELS] == list(_AT["NAMES"]), (LEVELS, _AT["NAMES"])
    return _AT


def terms_from_inputs(inp):
    """inp {name: dict(A, rec, tok, pm, eff)} -> same two numbers as a1_def_pur.palette_eval (deficit_tool(A), cases(T200, rows)['verdict_delta'])"""
    AT = atom_machinery()
    rows, A = [], []
    for n in AT["NAMES"]:
        i = inp[n]
        d = {"purity_ref_record": {"purity": i["rec"]}, "delay_arm": {"purity_ref_token_nGeval161": i["tok"], "purity_m_nGeval161": i["pm"]}, "eff_pct": i["eff"]}
        rows.append(AT["row_from"](n, d))
        A.append(i["A"])
    dt = AT["deficit_tool"](A)
    c2 = AT["cases"](AT["T200"], rows)
    return dict(deficit=dt["deficit"], purity=c2["verdict_delta"], quant=dt["quant"], rows=rows, loi={k: v["phase_dB"] for k, v in c2["loi"].items()}, A=A)


# ======================================================================= sim783 inputs ==============================================================================================
def job_kv(p):
    out = {}
    try:
        for fn in os.listdir(p):
            if fn.startswith("job_") and fn.endswith(".txt"):
                for ln in io.open(os.path.join(p, fn), encoding="utf-8"):
                    m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", ln)
                    if m:
                        out[m.group(1)] = m.group(2)
                out["_file"] = fn
    except OSError:
        pass
    return out


_TOKJ = {}


def tok_json(ttag):
    if ttag not in _TOKJ:
        p = WLB + r"\261012_sim783_wl_tok_525f74f3_%s_b11" % ttag
        f = os.path.join(p, "scdc_extract_H400_300.json")
        _TOKJ[ttag] = (p, json.load(io.open(f, encoding="utf-8")) if os.path.isfile(f) else None)
    return _TOKJ[ttag]


def sim783_inputs(chtag, a1info=None):
    """read + validate the 8 atom records and 8 delay-arm records at one channel.  -> (inp or None, table rows, problems).  Nothing is ever filled in."""
    chnm = tag_nm(chtag)
    label = "%.1f" % chnm
    inp, table, problems = {}, [], []
    for name, atag, amd, ttag in LEVELS:
        row = dict(level=name, atom_dir="261012_sim783_wl_a%s_%s_%s" % (atag, amd, chtag), tok_dir="261012_sim783_wl_tok_525f74f3_%s_b11" % ttag, label=label)
        ap = WLB + "\\" + row["atom_dir"]
        af = os.path.join(ap, "atom2d_opt.json")
        tp, tj = tok_json(ttag)
        miss = []
        if not os.path.isfile(af):
            miss.append("atom record " + row["atom_dir"])
        if tj is None:
            miss.append("delay-arm record " + row["tok_dir"])
        if miss:
            problems.append("%s @%s: MISSING %s" % (name, chtag, "; ".join(miss)))
            row["status"] = "MISSING"
            table.append(row)
            continue
        a = json.load(io.open(af, encoding="utf-8"))
        why = []
        if not (int(a["nG"]) == 1793 and int(a["eval_only"]) == 1 and a.get("rcwa_rule") == "liy" and int(a.get("rcwa_ff")) == 1):
            why.append("atom record is not nG 1793 / eval_only / Li")
        kv = job_kv(ap)
        if "lams_nm" not in kv or abs(float(kv["lams_nm"]) - chnm) > 1e-9:
            why.append("atom job lams_nm %r != %s" % (kv.get("lams_nm"), chnm))
        rho = kv.get("rho3_npy", "")
        try:
            mm = V1.m8(rho) if rho else None
        except Exception:                                                        # noqa: BLE001
            mm = None
        if mm != amd:
            why.append("atom job rho3_npy md5 prefix %r != %s" % (mm, amd))
        nsu = tj.get("out_l3_ns_used")
        if not (isinstance(nsu, list) and len(nsu) == 1):
            why.append("tok out_l3_ns_used %r not a single termination" % (nsu,))
        if not (int(tj.get("nG_eval", 0)) == 161 and int(tj.get("eval_only", 0)) == 1 and tj.get("rcwa_rule") == "liy" and int(tj.get("rcwa_ff", 0)) == 1):
            why.append("tok record is not nG_eval 161 / eval_only / Li")
        lev = tj.get("lams_eval_nm") or []
        if not any(abs(float(x) - chnm) < 1e-9 for x in lev):
            why.append("channel %s not in tok lams_eval_nm %s" % (chnm, lev))
        tkv = job_kv(tp)
        if abs(float(tkv.get("out_l3_lam_nm", "nan")) - 1550.0) > 1e-9:
            why.append("tok job out_l3_lam_nm %r != 1550" % tkv.get("out_l3_lam_nm"))
        tokstr = tkv.get("out_l3_r_list", "")
        try:                                                                     # the token of the delay-arm job = absR00 : argR00_token : arg_rho_TM of the 1550.000 atom record, 3 decimals
            got = [float(x) for x in tokstr.split(":")]
            ref = [a_ref(atag, amd, "absR00"), a_ref(atag, amd, "argR00_token_deg"), a_ref(atag, amd, "arg_rho_TM_deg")]
            dd = [abs(got[0] - ref[0])] + [abs(((g - r + 180.0) % 360.0) - 180.0) for g, r in zip(got[1:], ref[1:])]
            if len(got) != 3 or any(x > 5.1e-4 for x in dd):
                why.append("tok job token %s != the 1550.000 atom record %s" % (tokstr, ["%.3f" % x for x in ref]))
        except (ValueError, KeyError, OSError):
            why.append("tok job token %r unreadable / atom 1550.000 record missing" % tokstr)
        row["out_l3_nm"] = tj.get("out_l3_nm")
        if a1info is not None and a1info[name].get("out_l3_nm") is not None and float(a1info[name]["out_l3_nm"]) != float(tj.get("out_l3_nm")):
            why.append("tok out_l3_nm %r != the a1 pm job %r" % (tj.get("out_l3_nm"), a1info[name]["out_l3_nm"]))
        pref = "n%d_" % int(round(float(nsu[0]) * 1000)) if isinstance(nsu, list) and len(nsu) == 1 else None
        k_tok, k_pm = (pref + label + "_purity", pref + label + "_puritym") if pref else (None, None)
        if pref and (k_tok not in tj or k_pm not in tj or not isinstance(tj[k_tok], float) or not isinstance(tj[k_pm], float)):
            why.append("tok keys %s / %s missing or not numeric" % (k_tok, k_pm))
        elif pref:
            for kk in (k_tok, k_pm):
                ku = kk[len(pref):]
                if ku in tj and tj[ku] != tj[kk]:
                    why.append("unprefixed %s != prefixed %s" % (ku, kk))
        if why:
            problems.append("%s @%s: %s" % (name, chtag, "; ".join(why)))
            row["status"] = "INVALID"
            row["why"] = "; ".join(why)
            table.append(row)
            continue
        inp[name] = dict(A=float(a["A_deg"]) % 360.0, rec=float(a["purity"]), tok=float(tj[k_tok]), pm=float(tj[k_pm]), eff=float(a["carrier_eff_pct"]))
        row.update(status="OK", k_tok=k_tok, k_pm=k_pm, **inp[name])
        table.append(row)
    return (inp if len(inp) == len(LEVELS) else None), table, problems


_A_REF = {}


def a_ref(atag, amd, key):
    """the 1550.000 atom record (token reference)"""
    k = (atag, amd)
    if k not in _A_REF:
        _A_REF[k] = json.load(io.open(WLB + r"\261012_sim783_wl_a%s_%s_l1550p000\atom2d_opt.json" % (atag, amd), encoding="utf-8"))
    return _A_REF[k][key]


def print_mapping(chtag, table, terms=None):
    say("   INPUT MAP @ %s (channel label %s): level | atom record folder | A deg, rec purity, eff %% | delay-arm folder, keys | tok pur, delay pur | uA (source), uB" % (chtag, table[0]["label"]))
    rows = {r["level"]: r for r in terms["rows"]} if terms else {}
    for r in table:
        if r["status"] != "OK":
            say("     %-12s %s %s" % (r["level"], r["status"], r.get("why", "")))
            continue
        rr = rows.get(r["level"])
        say("     %-12s %s | A %8.3f rec %.9f eff %.2f | %s %s %s | tok %.9f pm %.9f | %s" % (
            r["level"], r["atom_dir"], r["A"], r["rec"], r["eff"], r["tok_dir"], r["k_tok"], r["k_pm"], r["tok"], r["pm"],
            ("uA %.4f (%s) uB %.4f" % (rr["uA"], rr["srcA"], rr["uB"])) if rr else ""))


# ======================================================================= controls ===================================================================================================
def run_atom_controls():
    AT = atom_machinery()
    ok = True
    say("=" * 150)
    say("ATOM CONTROL 1: the a1_def_pur.py chain on its own 1550 inputs (P769 = P763 + 212.5 sim764 t1150 + 257.5 sim764 t1150, as a1_def_pur_p769.py) -> stored terms, compared with ==")
    P769 = dict(AT["P763"])
    P769["level_212p5"] = (AT["AC"] % "78259586", AT["PM"] + "261009_sim764_pmli_TM_c212p5_78259586_n1793_t1150")
    P769["level_257p5"] = (AT["AC"] % "accded83", AT["PM"] + "261009_sim764_pmli_TM_c257p5_accded83_n1793_t1150")
    E = AT["palette_eval"]("P769", P769, show=False)
    st = V1.STORED
    for lab, got, want in (("deficit", E["deficit"], V1.DFC769), ("purity f200", E["pur_f200"], V1.PRT769), ("deficit vs final json", E["deficit"], st["deficit"]), ("purity vs final json", E["pur_f200"], st["purity"])):
        good = (got == want)
        ok = ok and good
        say("   %-22s chain %r  stored %r  -> %s" % (lab, got, want, "EXACT" if good else "DIFF %.3e" % abs(got - want)))
    inp_a1 = {n: dict(A=i["A"], rec=i["rec"], tok=i["tok"], pm=i["pm"], eff=i["eff"]) for n, i in E["infos"].items()}
    say("ATOM CONTROL 2: terms_from_inputs() of this script fed with the same a1 inputs (A, rec, tok, pm of the 8 stored 1550 records / pm jobs) -> palette_eval, compared with ==")
    T = terms_from_inputs(inp_a1)
    for lab, got, want in (("deficit", T["deficit"], E["deficit"]), ("purity f200", T["purity"], E["pur_f200"])):
        good = (got == want)
        ok = ok and good
        say("   %-22s this script %r  a1 palette_eval %r -> %s" % (lab, got, want, "EXACT" if good else "DIFF %.3e" % abs(got - want)))
    same_rows = all(r1["uA"] == r2["uA"] and r1["uB"] == r2["uB"] for r1, r2 in zip(T["rows"], E["rows"]))
    ok = ok and same_rows
    say("   per-level uA / uB of the 8 rows identical: %s" % same_rows)
    return ok, E, inp_a1


def run_input_refusal_needles(E):
    say("ATOM CONTROL 4: input-map refusal needles (a missing / inconsistent input must make the channel's terms REFUSED, never filled)")
    inp, table, problems = sim783_inputs("l1555p000")
    ok_a = inp is None and len(problems) == 8 and all("MISSING" in x for x in problems)
    say("   (a) channel folder tag l1555p000 (no such campaign channel): inputs %s, %d problems, e.g. %s -> %s" % ("None" if inp is None else "FILLED", len(problems), problems[0][:90] if problems else "-", "needle detected" if ok_a else "NEEDLE NOT DETECTED"))
    bad = {n: dict(i) for n, i in E["infos"].items()}
    bad["level_167p5"]["out_l3_nm"] = 1150.0
    inp, table, problems = sim783_inputs("l1546p003", a1info=bad)
    ok_b = inp is None and any("out_l3_nm" in x for x in problems)
    say("   (b) a1 reference with a wrong out_l3_nm for 167.5 vs the tok job: inputs %s, problems %s -> %s" % ("None" if inp is None else "FILLED", problems, "needle detected" if ok_b else "NEEDLE NOT DETECTED"))
    inp, table, problems = sim783_inputs("l1546p003", a1info=E["infos"])
    ok_c = inp is not None and not problems
    say("   (c) positive control, channel 1546.003 with the true a1 reference: inputs %s, problems %s -> %s" % ("complete" if inp is not None else "None", problems, "OK as expected" if ok_c else "UNEXPECTED"))
    return ok_a and ok_b and ok_c


def run_control_1550_sim783(E, inp_a1):
    say("ATOM CONTROL 3: the sim783 records at channel 1550.000 (atom records + tok b11 key 1550.0) -> same terms; inputs and terms against the stored 1550 chain")
    inp, table, problems = sim783_inputs("l1550p000", a1info=E["infos"])
    if inp is None:
        say("   sim783 1550.000 inputs incomplete: %s" % problems)
        return False, None, table, None
    say("   per level, sim783 1550.000 minus a1 stored input (A deg, rec, tok, pm):")
    worst = 0.0
    for n, *_ in LEVELS:
        d = [inp[n][k] - inp_a1[n][k] for k in ("A", "rec", "tok", "pm")]
        worst = max(worst, max(abs(x) for x in d))
        say("     %-12s dA %+.3e deg  drec %+.3e  dtok %+.3e  dpm %+.3e" % (n, d[0], d[1], d[2], d[3]))
    T = terms_from_inputs(inp)
    for lab, got, want in (("deficit", T["deficit"], V1.DFC769), ("purity f200", T["purity"], V1.PRT769)):
        say("   %-12s sim783 1550.000 %r  stored %r  diff %+.3e -> %s" % (lab, got, want, got - want, "EXACT" if got == want else "not bit-identical (inputs differ at the level shown above)"))
    ok = (abs(T["deficit"] - V1.DFC769) < 1e-9 and abs(T["purity"] - V1.PRT769) < 1e-9)
    say("   control 3 verdict: |diff| < 1e-9 for both terms: %s (max input difference %.3e)" % (ok, worst))
    print_mapping("l1550p000", table, T)
    return ok, T, table, inp


def u_of(pur):
    """row_from's u = 1.368 sqrt(p/(1-p)), p = 1 - purity (purity646.u_of), re-derived here for an independent prediction"""
    p = 1.0 - pur
    return 1.368 * math.sqrt(p / (1.0 - p))


def run_needle(inp, label):
    say("NEEDLE (%s): one atom-record purity lowered by 1e-4 (all other inputs unchanged). uA = max(u(rec), u(tok)), so the purity term moves iff u(rec - 1e-4) > max(u(rec), u(tok)) (predicted independently below)" % label)
    base = terms_from_inputs(inp)
    srcs = {r["level"]: r["srcA"] for r in base["rows"]}
    moved, ok_all = [], True
    for n, *_ in LEVELS:
        i2 = {k: dict(v) for k, v in inp.items()}
        i2[n]["rec"] = i2[n]["rec"] - 1e-4
        T2 = terms_from_inputs(i2)
        dp, dd = T2["purity"] - base["purity"], T2["deficit"] - base["deficit"]
        pred = u_of(i2[n]["rec"]) > max(u_of(inp[n]["rec"]), u_of(inp[n]["tok"]))
        good = ((dp != 0.0) == pred) and dd == 0.0
        ok_all = ok_all and good
        if dp != 0.0:
            moved.append(n)
        say("   %-12s uA source %-6s : purity term %r -> %r (delta %+.3e dB) ; deficit delta %+.3e ; predicted %s -> %s" % (
            n, srcs[n], base["purity"], T2["purity"], dp, dd, "moves" if pred else "unchanged", "OK" if good else "MISMATCH"))
    say("   -> terms moved for %d of 8 levels %s ; observed == independent prediction for all 8 levels, deficit untouched: %s ; at least one moved: %s" % (len(moved), moved, ok_all, bool(moved)))
    return ok_all and bool(moved)


# ======================================================================= channel evaluation ========================================================================================
def run_channel_v2(ch, g1550, model_names, twins, mode, terms_ch, term_problems):
    CH = V1.CHANNELS[ch]
    g = g1550 if ch == "1550" else V1.gather(ch, twins)
    miss, inval, lens_missing = V1.inventory(g)
    say("=" * 150)
    say("CHANNEL %s nm: cells found %d of %d ; MISSING (%d): %s%s" % (ch, sum(1 for c in g["cells"].values() if c["status"] == "OK"), len(g["cells"]), len(miss), " ".join(miss) or "none",
                                                                  ("   INVALID: " + " | ".join(inval)) if inval else ""))
    say("   classes complete: %s ; classes UNAVAILABLE for the lens: %s" % (" ".join("x%02d" % M for M in V1.CLASSES if M in g["nodes"]) or "none", " ".join("x%02d" % M for M in lens_missing) or "none"))
    out = dict(channel=ch, missing=miss, invalid=inval, classes_unavailable=["x%02d" % M for M in lens_missing], models={})
    d0, p0 = V1.DFC769, V1.PRT769
    if mode == "channel":
        if terms_ch is None:
            say("   ATOM TERMS @ channel: REFUSED -- %s" % "; ".join(term_problems))
            out["atom_terms"] = "REFUSED: " + "; ".join(term_problems)
            dch = pch = None
        else:
            dch, pch = terms_ch["deficit"], terms_ch["purity"]
            say("   ATOM TERMS: 1550 (stored) deficit %+.6f purity %+.6f sum %+.6f | channel (sim783) deficit %+.6f purity %+.6f sum %+.6f | shift deficit %+.6f purity %+.6f sum %+.6f dB (additive into the total)" % (
                d0, p0, d0 + p0, dch, pch, dch + pch, dch - d0, pch - p0, (dch + pch) - (d0 + p0)))
            out["atom_terms"] = dict(deficit=dch, purity=pch, deficit_1550=d0, purity_1550=p0, shift_sum=(dch + pch) - (d0 + p0))
    else:
        dch = pch = None
    for mn in model_names:
        if lens_missing:
            out["models"][mn] = "REFUSED: classes unavailable " + " ".join("x%02d" % M for M in lens_missing)
            say("   [%s] LENS TOTAL / APERTURE REFUSED: class(es) %s unavailable -- nothing is filled in%s" % (
                mn, " ".join("x%02d" % M for M in lens_missing), " (the atom-term shift above is independent of the cells and is added to the cells-only total once they land)" if dch is not None else ""))
            continue
        geo = V1.Geo(mn, CH["lam_um"])
        ev_c = V1.lens_from_nodes(g["nodes"], geo, d0, p0)                        # cells at the channel, atom terms at 1550
        ev_f = V1.lens_from_nodes(g["nodes"], geo, dch, pch) if dch is not None else None
        ap48 = V1.aperture({c: ev_c["echo_class"][c] for c in V1.CL}, e12=g["nodes"][48]["echo"]) if 48 in g["nodes"] else None
        def fm(ev):
            r_ = ev["res"]
            return "rB0_complex %+.4f / %+.4f (margin %+.4f / %+.4f) | no_rB0 %+.4f / %+.4f (margin %+.4f / %+.4f)" % (
                r_["rB0_complex"]["total"][0], r_["rB0_complex"]["total"][1], r_["rB0_complex"]["margin"][0], r_["rB0_complex"]["margin"][1],
                r_["no_rB0"]["total"][0], r_["no_rB0"]["total"][1], r_["no_rB0"]["margin"][0], r_["no_rB0"]["margin"][1])
        say("   [%s] CELLS-ONLY (atom terms at 1550): %s" % (mn, fm(ev_c)))
        if ev_f is not None:
            say("   [%s] FULL (atom terms at channel):   %s" % (mn, fm(ev_f)))
        if ev_f is not None:
            dfull = ev_f["res"]["rB0_complex"]["total"][0] - ev_c["res"]["rB0_complex"]["total"][0]
            dsh = (dch + pch) - (d0 + p0)
            say("   [%s] additivity check: full - cells-only (rB0_complex decay) %+.12f dB vs atom-term shift %+.12f dB -> %s" % (mn, dfull, dsh, "OK" if abs(dfull - dsh) < 1e-12 else "MISMATCH"))
        say("   [%s] aperture echo %.3f dB (gate %.1f, margin %+.3f)%s" % (mn, ev_c["echo_aperture"], V1.GATE_ECHO, V1.GATE_ECHO - ev_c["echo_aperture"],
                                                                          ("   [>12L = measured x48: %.3f dB]" % ap48) if ap48 is not None else ""))
        out["models"][mn] = dict(cells_only=ev_c, full=ev_f, aperture_x48=ap48)
    return out, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atom-terms", choices=["1550", "channel"], default="channel")
    ap.add_argument("--channels", default="1546.003,1550,1554.017")
    ap.add_argument("--models", default="fixed")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--terms-all-channels", action="store_true")
    ap.add_argument("--band-root", default=None)
    a = ap.parse_args()
    if a.band_root:
        V1.BAND_ROOT = a.band_root
    chs = [c.strip() for c in a.channels.split(",")]
    for c in chs:
        assert c in V1.CHANNELS, c
    mnames = list(V1.MODELS) if a.models == "all" else [m.strip() for m in a.models.split(",")]
    for m in mnames:
        assert m in V1.MODELS, m
    say("lensband_eval_v2.py  %s  cores %s  atom-terms %s  models %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), psutil.Process().cpu_affinity(), a.atom_terms, ",".join(mnames)))
    say("band root: %s" % V1.BAND_ROOT)
    g1550 = V1.gather("1550")
    miss, inval, lens_missing = V1.inventory(g1550)
    if miss or inval or lens_missing:
        say("ABORT: stored 1550 cells not all usable: %s %s" % (miss, inval))
        return 3
    twins = dict(g1550["cells"])
    result = dict(stamp=time.strftime("%Y-%m-%d %H:%M:%S"), atom_terms_mode=a.atom_terms, channels={}, controls={})
    rc = 0
    E = inp_a1 = None
    if not a.no_control:
        ok1, ev1550 = V1.run_control(g1550)
        ok_c = V1.run_control_ctl()
        ok_g = V1.run_control_gating(g1550)
        say("CELL CONTROL SUMMARY (v1 controls re-run): stored-values %s ; ctl cells %s ; refusal needles %s" % ("PASS" if ok1 else "FAIL", "PASS" if ok_c else "FAIL", "PASS" if ok_g else "FAIL"))
        okA, E, inp_a1 = run_atom_controls()
        okB, T1550, tab1550, inp1550 = run_control_1550_sim783(E, inp_a1)
        okR = run_input_refusal_needles(E)
        okN = run_needle(inp_a1, "stored 1550 inputs")
        okN2 = run_needle(inp1550, "sim783 1550.000 inputs") if okB else False
        result["controls"] = dict(cells=ok1 and ok_c and ok_g, atom_chain_stored=okA, sim783_1550=okB, input_refusal=okR, needle_stored=okN, needle_sim783=okN2)
        say("ATOM CONTROL SUMMARY: a1 chain == stored (bit for bit) and this script == a1 chain: %s ; sim783 1550.000 vs stored (< 1e-9): %s ; input refusal needles: %s ; purity needles (stored / sim783 1550): %s / %s" % (
            "PASS" if okA else "FAIL", "PASS" if okB else "FAIL", "PASS" if okR else "FAIL", "PASS" if okN else "FAIL", "PASS" if okN2 else "FAIL"))
        okNch = all(v for k, v in result["controls"].items() if k.startswith("needle_") and k not in ("needle_stored", "needle_sim783"))
        rc = 0 if (ok1 and ok_c and ok_g and okA and okB and okR and okN and okN2) else 1
    # ---- channel terms
    terms, tprob, ttab, tinp = {}, {}, {}, {}
    if a.atom_terms == "channel":
        for ch in chs:
            if ch == "1550":
                continue
            inp, table, problems = sim783_inputs(CHMAP[ch], a1info=E["infos"] if E else None)
            ttab[ch] = table
            tinp[ch] = inp
            tprob[ch] = problems
            terms[ch] = terms_from_inputs(inp) if inp is not None else None
            say("=" * 150)
            say("ATOM TERMS at channel %s: %s" % (ch, ("deficit %r purity f200 %r" % (terms[ch]["deficit"], terms[ch]["purity"])) if terms[ch] else "REFUSED (inputs missing/invalid): " + "; ".join(problems)))
            print_mapping(CHMAP[ch], table, terms[ch])
            if terms[ch]:
                say("   levels A deg (mod 360, NAMES order): %s ; quant %.6f dB" % (" ".join("%.3f" % x for x in terms[ch]["A"]), terms[ch]["quant"]))
                say("   leave-one-in purity penalty per level (dB): %s" % " ".join("%s %+.5f" % (k.replace("level_", "").replace("blank_", "b"), v) for k, v in terms[ch]["loi"].items()))
        if "1550" in chs:
            terms["1550"] = dict(deficit=V1.DFC769, purity=V1.PRT769)
        if not a.no_control:
            for ch in chs:
                if ch != "1550" and tinp.get(ch) is not None:
                    result["controls"]["needle_" + ch] = run_needle(tinp[ch], "sim783 channel %s inputs" % ch)
    if any(not v for k, v in result["controls"].items() if k.startswith("needle_")):
        rc = 1
    for ch in chs:
        out, g = run_channel_v2(ch, g1550, mnames, twins, a.atom_terms, terms.get(ch), tprob.get(ch, []))
        out["input_map"] = ttab.get(ch)
        result["channels"][ch] = out
    if a.terms_all_channels:
        say("=" * 150)
        say("ATOM TERMS on all 11 sim783 channels (deficit, purity f200, sum; shift vs the stored 1550 sum %+.6f):" % (V1.DFC769 + V1.PRT769))
        result["terms_all_channels"] = {}
        for tag in CH11:
            inp, table, problems = sim783_inputs(tag, a1info=E["infos"] if E else None)
            if inp is None:
                say("   %s REFUSED: %s" % (tag, "; ".join(problems)))
                continue
            T = terms_from_inputs(inp)
            result["terms_all_channels"][tag] = dict(deficit=T["deficit"], purity=T["purity"], uA_src={r["level"]: r["srcA"] for r in T["rows"]}, loi=T["loi"])
            say("   %-10s deficit %+.6f purity %+.6f sum %+.6f shift %+.6f | active uA source: %s" % (tag, T["deficit"], T["purity"], T["deficit"] + T["purity"],
                                                                                                  T["deficit"] + T["purity"] - (V1.DFC769 + V1.PRT769), "".join(r["srcA"][0] for r in T["rows"])))
    say("=" * 150)
    say("SUMMARY (model %s; TOTAL rB0_complex decay / held, dB; margin to %.2f; atom terms 'channel' = sim783 recomputed, '1550' = stored)" % (",".join(mnames), V1.GATE_TOTAL))
    for ch in chs:
        for mn in mnames:
            m = result["channels"][ch]["models"].get(mn)
            if isinstance(m, dict):
                c = m["cells_only"]["res"]["rB0_complex"]
                s = "%-9s %-12s cells-only %+.4f / %+.4f" % (ch, mn, c["total"][0], c["total"][1])
                if m["full"] is not None:
                    f = m["full"]["res"]["rB0_complex"]
                    s += "   full %+.4f / %+.4f (margin %+.4f / %+.4f)" % (f["total"][0], f["total"][1], f["margin"][0], f["margin"][1])
                say("   " + s + "   echo aperture %.3f dB" % m["cells_only"]["echo_aperture"])
            else:
                at = result["channels"][ch].get("atom_terms")
                s = "%-9s %-12s %s" % (ch, mn, m)
                if isinstance(at, dict):
                    s += "   | atom-term shift vs 1550 %+.4f dB (deficit %+.4f, purity %+.4f)" % (at["shift_sum"], at["deficit"] - at["deficit_1550"], at["purity"] - at["purity_1550"])
                say("   " + s)
    result["elapsed_s"] = time.time() - T0
    result["band_root"] = V1.BAND_ROOT
    jn = "lensband_eval_v2_result.json" if V1.BAND_ROOT == V1.DEFAULT_BAND_ROOT else "lensband_eval_v2_result_altroot.json"
    json.dump(result, io.open(os.path.join(WD, jn), "w", encoding="utf-8"), indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else str(o))
    say("wrote %s (%.1f s)" % (os.path.join(WD, jn), time.time() - T0))
    return rc


if __name__ == "__main__":
    _rc = main()
    _name = "lensband_eval_v2_stdout.txt" if V1.BAND_ROOT == V1.DEFAULT_BAND_ROOT else "lensband_eval_v2_stdout_altroot.txt"
    io.open(os.path.join(WD, _name), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    sys.exit(_rc)
