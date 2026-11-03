# -*- coding: utf-8 -*-
r"""edge_readout.py -- band-edge (1546.003 / 1554.017 nm) + 1550 lens read-out with ALL 45-degree cells MEASURED (d12 d15 d18 d24 d36 at both psi, no placeholder, no transfer).
Task lensband_edge_261017.  Method = the 1550 nm v6 read-out (readout45_v6_xe201): d ratio = beta_pc(d) / x e201 reference interpolated in 1/P over x06 x09 x12 x15 x18 x24 x36
(measured e201 references; band edges x06 / x09 = sim802), global 1/P map, FULL enumeration of the per-class orientation choices (x / y / d225 / d315; 6912 combinations),
minimum per tail (decayed / held), ties keep x, echo = share-weighted aperture (ip_db).  Code = copy of the 261016 Task A library (lib_edge.py = lensband_orient_v4_dmeas.py with two paths
changed; derived from lensband_orient_v3.py) + the xref patch of xref69_readout.py (copied below).  Originals are only read.
1550 is run first with the same code as a REGRESSION against lensdir45_verdict_1overP_v6.json (TOTAL -0.6073 / -0.6282, echo -20.2177): if the printed digits differ the run STOPS.
python -X utf8 -I -B edge_readout.py [--workers 6] [--tag NAME]"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys

sys.dont_write_bytecode = True
HERE = r"<WORK_ROOT>\fill_B_261103\work\ro_new"
sys.path.insert(0, HERE)
import argparse                                                                 # noqa: E402
import datetime                                                                 # noqa: E402
import hashlib                                                                  # noqa: E402
import io                                                                       # noqa: E402
import json                                                                     # noqa: E402
import math                                                                     # noqa: E402
import time                                                                     # noqa: E402
import lib_edge as W                                                            # noqa: E402


V6DIR = r"<WORK_ROOT>\loop_261008\lens45_li_261011\readout45_v6_xe201"
V6_PRIMARY = V6DIR + r"\lensdir45_readout_v6_primary.json"
V6_VERDICT = V6DIR + r"\lensdir45_verdict_1overP_v6.json"
XREF69_R2 = r"<WORK_ROOT>\finished_sims_261016\A_results\bandedge_dmeas\xref69_readout_result_261016_1045_r2.json"
V3_ORIENT = r"<WORK_ROOT>\wlband_261012\lensband_orient_v3\lensband_orient_v3_result_261014_0550_dev1.json"
XE = W.R + r"\261016_scdc_xe201_ref"
XREF_NEW = (6, 9, 12, 15, 18, 24, 36)
CRIT_TOTAL, CRIT_ECHO = -0.65, -20.0                                             # design criteria as given in the task text (numbers only)
CHORDER = ("1546.003", "1550", "1554.017")


# ------------------------------------------------------------------ x06 / x09 e201 references (copied from xref69_readout.py, arithmetic unchanged)
def xp_new(ch, M):
    if ch == "1550":
        if M == 6:
            return [W.R + r"\261011_scdc_tmzone_licolp3\261011_sim780_g_x06_sh%d_li_e201" % sh for sh in (0, 1, 2)]
        return [XE + r"\261016_sim802_x09e201_sh%d_l1550p000" % sh for sh in (0, 1, 2)]
    tag = W.V1.CHANNELS[ch]["tag"]
    return [W.R + r"\261030_scdc_kxheld\261030_sim900_kx_x%02de201_sh%d_%s" % (M, sh, tag) for sh in (0, 1, 2)]   # [fill_B] sim802 -> sim900


def validate_ref(ch, M, paths, G):
    why = []
    nom = W.V1.CHANNELS[ch]["lam_nm"]
    for sh, p in enumerate(paths):
        if not os.path.isfile(os.path.join(p, "superramp_result.json")):
            why.append("NOT FOUND: " + p)
            continue
        j = W.jl(p)
        r = j["needle_results"]["production"]
        if (j.get("rcwa_rule"), j.get("rcwa_ff")) != ("liy", 1) or j.get("pol") != "TM":
            why.append("%s: not Li / TM" % p)
        if j.get("ramp_axis_requested") != "x" or r.get("ramp_axis") != "x" or j.get("ramp_s") != -1 or int(r["ramp_s"]) != -1:
            why.append("%s: axis / ramp_s" % p)
        if int(j.get("sc_shift", -9)) != sh or int(r["sc_shift"]) != sh:
            why.append("%s: sc_shift" % p)
        if int(r["M"]) != M:
            why.append("%s: M %r != %d" % (p, r["M"], M))
        if j.get("nG_ref") != 201 or j.get("nG_ref_matched") is not True:
            why.append("%s: nG_ref %r matched %r" % (p, j.get("nG_ref"), j.get("nG_ref_matched")))
        jlam = W.V1.job_lams(p)
        if ch == "1550":
            if j.get("lam_eval_override") is not None or j.get("lam_eval_nm") is not None:
                why.append("%s: 1550 cell carries lam_eval keys" % p)
            if jlam is not None and abs(jlam - 1550.0) > 1e-9:
                why.append("%s: job lams_nm %r" % (p, jlam))
        else:
            if j.get("lam_eval_override") != 1 or j.get("lam_eval_nm") is None or abs(float(j["lam_eval_nm"]) - nom) > 1e-9:
                why.append("%s: lam_eval keys %r %r" % (p, j.get("lam_eval_override"), j.get("lam_eval_nm")))
            if j.get("atom_lams_nm") is None or abs(float(j["atom_lams_nm"]) - 1550.0) > 1e-9:
                why.append("%s: atom_lams_nm %r" % (p, j.get("atom_lams_nm")))
            if jlam is None or abs(jlam - nom) > 1e-9:
                why.append("%s: job lams_nm %r" % (p, jlam))
        md5 = [W.V1.m8(a) for a in r["atom_paths"]]
        twin = G["1550"]["cells"][(M, sh)]["md5"]
        if md5 != twin:
            why.append("%s: atom md5 list != adopted twin" % p)
    return why


def make_xref(G):
    def xref_ch69(ch, EXch):
        XR = {}
        for M in W.XREF_M:
            if M in (6, 9):
                xp = xp_new(ch, M)
                why = validate_ref(ch, M, xp, G)
                if why:
                    raise RuntimeError("REFUSED x%02d e201 reference at %s: %s" % (M, ch, " | ".join(why)))
            else:
                xp = W.x36_paths(ch) if M == 36 else W.pair_paths(ch, M)[1]
            for p in xp:
                jx = W.jl(p)
                assert (jx.get("rcwa_rule"), jx.get("rcwa_ff"), jx.get("ramp_s"), jx.get("ramp_axis_requested"), jx.get("nG_ref")) == ("liy", 1, -1, "x", 201), p
            e201 = W.pmean_e(xp)
            XR[M] = dict(P=M / 3.0, node=W.node_pc(xp), r0=sum(W.r0_of(W.jl(p)) for p in xp) / len(xp), e201=e201, off=EXch[M] - e201, paths=list(xp))
        return XR
    return xref_ch69


def worst_orient_echo(EX, anchors, dan):
    """v6 echoes(): ec_w = max over x / y / d225 / d315 class echoes (ip_db), then aperture"""
    ec_w = {}
    for M in W.CLASSES:
        ex = EX[M]
        ey = anchors[M]["ey"] if (M in anchors and anchors[M].get("ey") is not None) else None
        ed = [W.ip_d_echo(dan, psi, M) for psi in W.PSIS]
        ec_w["x%02d" % M] = max([ex] + ([ey] if ey is not None else []) + [v for v in ed if v is not None])
    return W.aperture_of(ec_w), ec_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="r1")
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M") + "_" + a.tag
    W.LG = W.Log(os.path.join(HERE, "edge_readout_stdout_%s.txt" % stamp))
    say = W.say
    T0 = time.time()
    mem = W.psutil.virtual_memory()
    say("edge_readout.py %s affinity %s workers %d ; RAM at start %.1f GB avail of %.1f GB (%.0f %% used)" % (time.strftime("%Y-%m-%d %H:%M:%S"), W.psutil.Process().cpu_affinity(), a.workers,
                                                                                                           mem.available / 1e9, mem.total / 1e9, mem.percent))
    controls = []

    def ctl(name, ok, text):
        controls.append((name, bool(ok), text))
        say("CONTROL %-52s %s  %s" % (name, "PASS" if ok else "FAIL", text))

    g1550 = W.V1.gather("1550")
    G0 = {"1550": g1550}
    W.XREF_M = XREF_NEW
    W.xref_ch = make_xref(G0)
    try:
        P = W.prepare()
    except Exception as ex_:
        say("PREPARE FAILED: %s: %s" % (type(ex_).__name__, ex_))
        W.LG.close("FAILED")
        return 3
    CTX, ANCH, XR, DAN, MISS = P["CTX"], P["ANCH"], P["XR"], P["DAN"], P["MISS"]
    for ch in W.CHS:
        if MISS[ch]:
            say("NOT FOUND: d cells missing at %s: %s" % (ch, MISS[ch]))
            W.LG.close("FAILED")
            return 3
        say("%-8s d cells MEASURED: %s" % (ch, ", ".join("d%02d/psi%d" % (x["cls"], psi) for psi in W.PSIS for x in DAN[ch][psi])))
    n_meas = {ch: sum(1 for psi in W.PSIS for x in DAN[ch][psi] if not x["carried"]) for ch in W.CHS}
    ctl("(0) all d anchors measured, none carried", all(v == 10 for v in n_meas.values()), "measured anchors per channel: %s (d12 d15 d18 d24 d36 x psi 225 315)" % n_meas)
    say("d36 sources: " + " | ".join("%s psi%d %s" % (ch, psi, [x for x in DAN[ch][psi] if x["cls"] == 36][0]["src"]) for ch in CHORDER for psi in W.PSIS))

    # (1) 1550 d anchors == v6 primary json (bit)
    v6 = json.load(io.open(V6_PRIMARY, encoding="utf-8"))
    v6v = json.load(io.open(V6_VERDICT, encoding="utf-8"))
    bad, n_cmp = [], 0
    for psi in W.PSIS:
        s6 = {x["cls"]: x for x in v6["anchors_d"][str(psi)]}
        for a_ in DAN["1550"][psi]:
            s_ = s6[a_["cls"]]
            n_cmp += 1
            for lab, x, y in (("rA", complex(*s_["r"][0]), a_["r"][0]), ("rB", complex(*s_["r"][1]), a_["r"][1]), ("rR", complex(*s_["rR"]), a_["rR"])):
                if not (W.bits_equal(x.real, y.real) and W.bits_equal(x.imag, y.imag)):
                    bad.append("d%02d psi%d %s %r vs %r" % (a_["cls"], psi, lab, x, y))
            for lab in ("e_raw", "off", "e", "P"):
                if not W.bits_equal(s_[lab], a_[lab]):
                    bad.append("d%02d psi%d %s %r vs %r" % (a_["cls"], psi, lab, s_[lab], a_[lab]))
    ctl("(1) 1550 d anchors == v6 primary json (bit)", not bad and n_cmp == 10, "%d anchors (r A, r B, rB0, e_raw, offset, e, P): %s %s" % (n_cmp, "bit-identical" if not bad else "DIFFER", "; ".join(bad[:4])))
    bad = []
    for M in W.YM:
        s_, m_ = v6["anchors_y"][str(M)], ANCH["1550"][M]
        for lab, x, y in (("rA", complex(*s_["r"][0]), m_["r"][0]), ("rB", complex(*s_["r"][1]), m_["r"][1]), ("rR", complex(*s_["rR"]), m_["rR"])):
            if not (W.bits_equal(x.real, y.real) and W.bits_equal(x.imag, y.imag)):
                bad.append("y%02d %s" % (M, lab))
        if not W.bits_equal(s_["ey"], m_["ey"]):
            bad.append("y%02d ey" % M)
    ctl("(1b) 1550 y anchors == v6 primary json (bit)", not bad, "4 y anchors (r A, r B, rB0, ey): %s" % ("bit-identical" if not bad else bad))
    # (1c) twin check of every edge d cell against its 1550 twin
    tw = []
    for ch in ("1546.003", "1554.017"):
        for psi in W.PSIS:
            for x in DAN[ch][psi]:
                x0 = [y for y in DAN["1550"][psi] if y["cls"] == x["cls"]][0]
                d_ = W.twin_diff(x["src"], x0["src"])
                if d_:
                    tw.append((ch, x["cls"], psi, d_))
    ctl("(1c) edge d cells vs 1550 twins (job keys, atom md5)", not tw, "20 edge cells: %s" % ("no differing key" if not tw else tw))

    EV = W.Eval2(CTX, a.workers)
    # (2) V-x == lensband_eval_v2 stored
    for ch in W.CHS:
        m = P["v2"]["channels"][ch]["models"]["fixed"]
        st_t, st_e = m["full"]["res"]["rB0_complex"]["total"], m["cells_only"]["echo_aperture"]
        tt = EV.totals(ch, [W.build_nodes(CTX[ch], {}, {})])[0]
        ap_x = W.aperture_of({"x%02d" % M: CTX[ch]["EX"][M] for M in W.CLASSES})
        d = [abs(tt[0] - st_t[0]), abs(tt[1] - st_t[1]), abs(ap_x - st_e)]
        ctl("(2) V-x == lensband_eval_v2 @%s" % ch, max(d) <= W.TOL, "V-x %+.10f / %+.10f echo %.6f ; max |diff| %.2e" % (tt[0], tt[1], ap_x, max(d)))

    cache = W.Cache(os.path.join(HERE, "cache_edge.pkl"))
    RES, SS = {}, {}

    def run_channel(ch):
        t0 = time.time()
        ctx = CTX[ch]
        S = W.search_c(EV, ctx, ANCH[ch], DAN[ch], cache)
        SS[ch] = S
        pk = W.pick(S)
        echo = W.echo_summary(ctx["EX"], ANCH[ch], DAN[ch])
        ew, ecw = worst_orient_echo(ctx["EX"], ANCH[ch], DAN[ch])
        ec_ne = W.echoes(ctx["EX"], ANCH[ch], DAN[ch])
        say("[%s] full enumeration %d combinations (%d new) %.0f s : TOTAL %.10f / %.10f ; echo NE %.10f swapped %.10f agnostic %.10f ; worst-orientation echo %.6f" % (
            ch, len(S["combos"]), S["n_new"], time.time() - t0, pk["total"][0], pk["total"][1], echo["NE"], echo["swapped"], echo["agnostic"], ew))
        return pk, echo, ew, ecw, ec_ne

    out_ch = {}

    def summarize(ch, pk, echo, ew, ecw, ec_ne):
        S = SS[ch]
        idx = {tuple(sorted(c.items())): i for i, c in enumerate(S["combos"])}
        tot = pk["total"]
        contrib = {}
        for tail, tn in ((0, "decayed"), (1, "held")):
            ch_ = pk["choice"][tail]
            rows = {}
            for M in sorted(ch_):
                sub = {k: v for k, v in ch_.items() if k != M}
                t_sub = S["totals"][idx[tuple(sorted(sub.items()))]][tail]
                rows["x%02d" % M] = dict(orientation=ch_[M], total_with=tot[tail], total_class_reverted_to_x=t_sub, contribution_dB=tot[tail] - t_sub)
            contrib[tn] = rows
        so = W.single_options(S)
        x_only = S["totals"][0]
        ec_x = {"x%02d" % M: CTX[ch]["EX"][M] for M in W.CLASSES}
        cls_echo = {}
        for c in ec_x:
            ec_alt = dict(ec_ne)
            ec_alt[c] = ec_x[c]
            cls_echo[c] = dict(x_only=ec_x[c], share_weighted=ec_ne[c], worst_orientation=ecw[c], aperture_if_reverted_to_x=W.aperture_of(ec_alt), aperture_delta_dB=echo["NE"] - W.aperture_of(ec_alt))
        o = dict(lam_nm=W.V1.CHANNELS[ch]["lam_nm"], n_combinations=len(S["combos"]),
                 total_decayed_dB=tot[0], total_held_dB=tot[1], x_only_total=list(x_only),
                 echo_aperture_NE_dB=echo["NE"], echo_aperture_swapped_dB=echo["swapped"], echo_aperture_agnostic_dB=echo["agnostic"], echo_worst_orientation_dB=ew,
                 worst_orientation_decayed={"x%02d" % M: pk["choice"][0].get(M, "x") for M in W.CLASSES}, worst_orientation_held={"x%02d" % M: pk["choice"][1].get(M, "x") for M in W.CLASSES},
                 class_contribution_total=contrib, single_option_delta_vs_x_only=[dict(cls=M, option=nm, d_decayed=d0, d_held=d1) for M, nm, d0, d1 in so], class_echo=cls_echo,
                 criteria=dict(total_criterion_dB=CRIT_TOTAL, total_decayed_minus_criterion=tot[0] - CRIT_TOTAL, total_held_minus_criterion=tot[1] - CRIT_TOTAL,
                               echo_criterion_dB=CRIT_ECHO, echo_NE_minus_criterion=echo["NE"] - CRIT_ECHO, echo_agnostic_minus_criterion=echo["agnostic"] - CRIT_ECHO,
                               module_gate_echo_dB=W.GATE_ECHO, echo_NE_minus_module_gate=echo["NE"] - W.GATE_ECHO),
                 d_anchors=[dict(cls=x["cls"], psi=psi, P=x["P"], A_dB_deg=W.dbdeg(x["r"][0]), B_dB_deg=W.dbdeg(x["r"][1]), rB0_dB_deg=W.dbdeg(x["rR"]), echo_used_dB=x["e"], src=x["src"])
                            for psi in W.PSIS for x in DAN[ch][psi]],
                 y_anchors={str(M): dict(A_dB_deg=W.dbdeg(ANCH[ch][M]["r"][0]), B_dB_deg=W.dbdeg(ANCH[ch][M]["r"][1]), rB0_dB_deg=W.dbdeg(ANCH[ch][M]["rR"]), echo_dB=ANCH[ch][M]["ey"]) for M in W.YM},
                 x_class_echo_dB=ec_x, atom_terms=dict(deficit=CTX[ch]["DFC"], purity=CTX[ch]["PRT"]))
        out_ch[ch] = o
        say("   %s chosen decayed: %s | held: %s" % (ch, W.cstr(dict(choice=pk["choice"])).split(" | ")[0], W.cstr(dict(choice=pk["choice"])).split(" | ")[1]))
        for tn in ("decayed", "held"):
            say("   %s contributions (%s, dB, total minus total with that class at x): %s" % (ch, tn, "; ".join("%s %s %+.4f" % (k, v["orientation"], v["contribution_dB"]) for k, v in contrib[tn].items())))

    # ------------------------------------------------------------------ 1550 regression FIRST
    pk, echo, ew, ecw, ec_ne = run_channel("1550")
    summarize("1550", pk, echo, ew, ecw, ec_ne)
    v6t = (v6v["total"]["decay"], v6v["total"]["held"])
    dtot = max(abs(pk["total"][0] - v6t[0]), abs(pk["total"][1] - v6t[1]))
    decho = abs(echo["NE"] - v6v["echo"]["share_NE"])
    dsw = abs(echo["swapped"] - v6v["echo"]["share_swapped"])
    dew = abs(ew - v6v["echo"]["worst_orientation"])
    printed_ok = ("%.4f %.4f %.3f" % (pk["total"][0], pk["total"][1], echo["NE"])) == ("%.4f %.4f %.3f" % (v6t[0], v6t[1], v6v["echo"]["share_NE"]))
    bits = W.bits_equal(pk["total"][0], v6t[0]) and W.bits_equal(pk["total"][1], v6t[1]) and W.bits_equal(echo["NE"], v6v["echo"]["share_NE"])
    ctl("(R) 1550 REGRESSION vs v6 (TOTAL, echo, printed digits)", printed_ok and max(dtot, decho, dsw, dew) <= 1e-9,
        "this %.10f / %.10f echo NE %.10f swapped %.10f worst %.10f ; v6 %.10f / %.10f echo %.10f %.10f %.10f ; max |diff| total %.2e echo %.2e swapped %.2e worst %.2e ; bit-identical %s ; printed digits equal %s" % (
            pk["total"][0], pk["total"][1], echo["NE"], echo["swapped"], ew, v6t[0], v6t[1], v6v["echo"]["share_NE"], v6v["echo"]["share_swapped"], v6v["echo"]["worst_orientation"], dtot, decho, dsw, dew, bits, printed_ok))
    cs6 = v6["configs"]["x_y_d"]["used"]
    ch6 = [{int(M): u["chosen"][i] for M, u in cs6.items() if u["chosen"][i] != "x"} for i in (0, 1)]
    ctl("(R2) 1550 chosen orientations == v6", pk["choice"][0] == ch6[0] and pk["choice"][1] == ch6[1], "this %s | v6 %s" % ((pk["choice"][0], pk["choice"][1]), (ch6[0], ch6[1])))
    if not (printed_ok and max(dtot, decho, dsw, dew) <= 1e-9):
        say("STOP: the 1550 regression does not reproduce v6 -- band-edge numbers are NOT computed. Differences: total %.3e, echo NE %.3e, swapped %.3e, worst %.3e" % (dtot, decho, dsw, dew))
        EV.close()
        W.LG.close("FAILED_REGRESSION")
        return 2
    say("1550 regression PASSED; proceeding to the band edges (elapsed %.0f s)" % (time.time() - T0))
    for ch in ("1546.003", "1554.017"):
        pk, echo, ew, ecw, ec_ne = run_channel(ch)
        summarize(ch, pk, echo, ew, ecw, ec_ne)
    # needle: d36 psi225 A and B ratio -0.1 dB must move the (d36 psi225)-only total
    g = 10 ** (-0.1 / 20.0)
    for ch in ("1546.003", "1554.017"):
        dan2 = {psi: [dict(x, r=[x["r"][0] * g, x["r"][1] * g]) if (x["cls"] == 36 and psi == 225) else x for x in DAN[ch][psi]] for psi in W.PSIS}
        opts2 = W.make_opts(ANCH[ch], dan2)
        t2 = EV.totals(ch, [W.build_nodes(CTX[ch], {36: "d225"}, opts2)])[0]
        S = SS[ch]
        idx = {tuple(sorted(c.items())): i for i, c in enumerate(S["combos"])}
        t1 = S["totals"][idx[((36, "d225"),)]]
        ctl("(N) needle d36 psi225 -0.1 dB moves total @%s" % ch, abs(t2[0] - t1[0]) > 1e-6 and abs(t2[1] - t1[1]) > 1e-6, "d36-only total %+.6f / %+.6f -> %+.6f / %+.6f (shift %+.6f / %+.6f dB)" % (t1[0], t1[1], t2[0], t2[1], t2[0] - t1[0], t2[1] - t1[1]))
    allok = all(ok for _, ok, _ in controls)
    say("CONTROL SUMMARY: %d controls, %d PASS, %d FAIL" % (len(controls), sum(1 for _, ok, _ in controls if ok), sum(1 for _, ok, _ in controls if not ok)))

    # ------------------------------------------------------------------ comparison with the provisional values (read from the stored jsons)
    x69 = json.load(io.open(XREF69_R2, encoding="utf-8"))["results"]
    v3o = json.load(io.open(V3_ORIENT, encoding="utf-8"))["results"]
    comp = {}
    for ch in CHORDER:
        n = out_ch[ch]
        prov = {"orient_v3_xyd (d36 and all d transferred from 1550)": dict(total=v3o[ch]["xyd"]["total"], echo=v3o[ch]["xyd"]["echo"]["NE"])}
        if ch == "1550":
            prov["xref69 1550 full (= v6)"] = dict(total=x69[ch]["full"]["total"], echo=x69[ch]["measured"]["echo"]["NE"])
        else:
            for mode in ("pending", "transfer"):
                prov["xref69 d36 %s" % mode] = dict(total=x69[ch][mode]["total"], echo=x69[ch][mode]["echo"]["NE"])
        comp[ch] = {k: dict(v, d_total_decayed=n["total_decayed_dB"] - v["total"][0], d_total_held=n["total_held_dB"] - v["total"][1], d_echo=n["echo_aperture_NE_dB"] - v["echo"]) for k, v in prov.items()}
    lines = ["# Band-edge lens read-out with all 45-degree cells measured (lensband_edge_261017)", "",
             "Numbers: MEASURED cell data read by edge_readout.py (full 6912-combination enumeration per wavelength). Stdout: edge_readout_stdout_%s.txt" % stamp, "",
             "| wavelength nm | TOTAL decayed dB | TOTAL held dB | echo NE dB | echo swapped | echo agnostic | TOTAL decayed - (-0.65) | TOTAL held - (-0.65) | echo NE - (-20) |", "|---|---|---|---|---|---|---|---|---|"]
    for ch in CHORDER:
        n = out_ch[ch]
        c = n["criteria"]
        lines.append("| %s | %.4f | %.4f | %.3f | %.3f | %.3f | %+.4f | %+.4f | %+.3f |" % (ch, n["total_decayed_dB"], n["total_held_dB"], n["echo_aperture_NE_dB"], n["echo_aperture_swapped_dB"], n["echo_aperture_agnostic_dB"],
                                                                                     c["total_decayed_minus_criterion"], c["total_held_minus_criterion"], c["echo_NE_minus_criterion"]))
    lines += ["", "## Change against the provisional values (stored json values; new minus provisional)", "", "| wavelength nm | provisional reading | prov. TOTAL decayed / held | prov. echo NE | d TOTAL decayed | d TOTAL held | d echo |", "|---|---|---|---|---|---|---|"]
    for ch in CHORDER:
        for k, v in comp[ch].items():
            lines.append("| %s | %s | %.4f / %.4f | %.3f | %+.4f | %+.4f | %+.3f |" % (ch, k, v["total"][0], v["total"][1], v["echo"], v["d_total_decayed"], v["d_total_held"], v["d_echo"]))
    lines += ["", "## Worst orientation per class (decayed / held) and class contribution to TOTAL (dB, total minus total with that class at x)", ""]
    for ch in CHORDER:
        n = out_ch[ch]
        lines.append("- %s decayed: %s" % (ch, "; ".join("%s %s %+.4f" % (k, v["orientation"], v["contribution_dB"]) for k, v in n["class_contribution_total"]["decayed"].items())))
        lines.append("- %s held: %s" % (ch, "; ".join("%s %s %+.4f" % (k, v["orientation"], v["contribution_dB"]) for k, v in n["class_contribution_total"]["held"].items())))
    with io.open(os.path.join(HERE, "comparison_%s.md" % stamp), "x", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    rname = os.path.join(HERE, "lensband_edge_results_%s.json" % stamp)
    res = dict(stamp=stamp, wavelengths_nm=list(CHORDER), results=out_ch, comparison_to_provisional=comp, criteria=dict(total_dB=CRIT_TOTAL, echo_dB=CRIT_ECHO, module_gate_echo_dB=W.GATE_ECHO),
               regression_1550_vs_v6=dict(v6_verdict_json=V6_VERDICT, v6_total=list(v6t), v6_echo_NE=v6v["echo"]["share_NE"], max_abs_diff_total=dtot, max_abs_diff_echo=decho, bit_identical=bits),
               controls=[dict(name=nm, passed=ok, text=t) for nm, ok, t in controls], all_controls_pass=allok,
               definitions="TOTAL = minimum over per-class orientation choices (x / y / d225 / d315, full enumeration) of the lens total with atom deficit and purity terms; echo = share-weighted aperture (ip_db, NE quadrant), v6 definition",
               provenance=dict(script=os.path.join(HERE, "edge_readout.py"), script_md5=hashlib.md5(io.open(os.path.join(HERE, "edge_readout.py"), "rb").read()).hexdigest(),
                               lib_md5=hashlib.md5(io.open(os.path.join(HERE, "lib_edge.py"), "rb").read()).hexdigest(), elapsed_s=time.time() - T0, betaL_evaluations_dispatched=EV.n))
    with io.open(rname, "x", encoding="utf-8") as fh:
        json.dump(W.j2(res), fh, indent=1)
    say("wrote %s" % rname)
    say("elapsed %.0f s" % (time.time() - T0))
    EV.close()
    W.LG.close("COMPLETE" if allok else "COMPLETE_WITH_FAILED_CONTROLS")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
