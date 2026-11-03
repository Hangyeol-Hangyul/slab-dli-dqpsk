# -*- coding: utf-8 -*-
"""[261014] Task 2 data collector: nine-stage seed_npy chain of the bottom metagrating (kind extract, TM, H1/H2 400/300 nm).
Reads ONLY stored files (no solver): per stage the output density (2 x 512, a-Si fraction of L1 / L2), the job file (keys, seed_npy parent), the result JSON /
runs_log row (signal_worst, echo, purity, fid, minimum feature), the FoM history. Verifies the chain by md5 (each stage's seed_npy resolves to a file whose md5 equals
the previous stage's output md5; the first stage starts from the multi_q cosine seed of scd_c_grating.seed_bank, reproduced here line for line).
Writes botchain_data.json, botchain_table.tsv, botchain_keydiff.tsv and a stdout log. Read-only on the result folders."""
import glob
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
ARCH = r"<DATA_ROOT>\Result"
SIMR = r"<DATA_ROOT>\Result"
CEIL = (4.0 / 9.0) ** 1.5
NX = 512
STAGES = [
    (1, "sim261", ARCH + r"\260922_scdc_csio2\260922_sim261_csio2_400_300_f030", "6c5c9510"),
    (2, "sim300", ARCH + r"\260923_scdc_gap\260923_sim300_w00s_warm", "412caade"),
    (3, "sim302", ARCH + r"\260923_scdc_cont\260923_sim302_c2_fid10", "6a8ee627"),
    (4, "sim303", ARCH + r"\260923_scdc_cont2\260923_sim303_c2_f10fr80", "77eaafe0"),
    (5, "sim304", ARCH + r"\260923_scdc_cont3\260923_sim304_c2_e125", "b276a14d"),
    (6, "sim321", ARCH + r"\260924_scdc_fidm\260924_sim321_c2e125_f100", "8cda353d"),
    (7, "sim342", ARCH + r"\260924_scdc_fidm3\260924_sim342_c2e125f100_f90w10", "6aac4bae"),
    (8, "sim356", ARCH + r"\260924_scdc_near5\260924_sim356_c2f90_E", "e37e284a"),
    (9, "sim359", SIMR + r"\260924_scdc_fin5\260924_sim359_c2f90E_f100", "525f74f3"),
]
KEYS = ("signal_worst", "signal", "eta_plus", "R_go", "eta_minus", "purity_worst", "fid_worst", "echo_worst_p2_db", "signal_worst_spread")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def parse(p):
    d = {}
    for ln in io.open(p, encoding="utf-8", errors="replace").read().split("\n"):
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, _, v = ln.partition("=")
            d[k.strip()] = v.strip()
    return d


def resolve(p):
    if os.path.isfile(p):
        return p
    if p.lower().startswith(SIMR.lower()):
        alt = ARCH + p[len(SIMR):]
        if os.path.isfile(alt):
            return alt
    return None


def seed_multi_q():
    """scd_c_grating.seed_bank(harmonics=(1, 2)) -> 'multi_q', line for line (scd_c_grating.py:3313-3336)."""
    u = (np.arange(NX) + 0.5) / NX
    a = 0.45 / 2

    def multi(ph):
        return np.clip(0.5 + a * sum(np.cos(2 * np.pi * h * u + ph) for h in (1, 2)), 0.0, 1.0)
    return np.vstack([multi(np.pi / 2), multi(-np.pi / 2)])


def runs(a):
    """pixel runs of 1 in a 0/1 vector (no wrap): list of (start, length)."""
    b = np.concatenate([[0], (a > 0.5).astype(int), [0]])
    d = np.diff(b)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(s.tolist(), (e - s).tolist()))


def main():
    data = {"ceiling": CEIL, "NX": NX, "period_nm": 1315.39, "stages": []}
    prev_md5 = None
    log = []
    for n, sim, d, want in STAGES:
        dens_name = "density_scdc_extract_H400_300.npy" if os.path.isfile(os.path.join(d, "density_scdc_extract_H400_300.npy")) else "density_scdc_extract_H400_300_warm.npy"
        if n == 1:
            dens_name = "density_scdc_extract_H400_300.npy"
        dp = os.path.join(d, dens_name)
        assert md5(dp)[:8] == want, (dp, md5(dp))
        # the warm-named copy (the seed the next stage reads) must be byte-identical to the stage output
        for alt in glob.glob(os.path.join(d, "density_scdc_extract_H400_300*.npy")):
            if "_infeasible" in alt or alt.endswith("_warm.npy"):
                assert md5(alt)[:8] == want, ("stage output copies differ", alt)
        dens = np.load(dp)
        assert dens.shape == (2, NX) and set(np.unique(dens).tolist()) <= {0.0, 1.0}
        jf = glob.glob(os.path.join(d, "job_*.txt"))
        job = parse(jf[0]) if jf else {}
        row = {"stage": n, "sim": sim, "folder": d, "density_file": dp, "md5": md5(dp), "job_file": jf[0] if jf else None,
               "filt_radius_nm": float(job.get("filt_radius_nm", "nan")) if job else None, "dens": dens.astype(int).tolist()}
        if n > 1:
            sp = resolve(job["seed_npy"])
            assert sp is not None, job["seed_npy"]
            assert md5(sp) == prev_md5, ("chain broken at stage %d: seed %s md5 %s != previous output %s" % (n, sp, md5(sp), prev_md5))
            row["parent_seed_npy"] = job["seed_npy"]
            row["parent_md5"] = md5(sp)
            row["changed_px"] = [int((np.load(sp)[i] != dens[i]).sum()) for i in (0, 1)]
        # metrics: result JSON (feasible) else the infeasible_best JSON; the runs_log row is the cross-check
        jj = os.path.join(d, "scdc_extract_H400_300.json")
        src = jj
        jd = json.load(io.open(jj, encoding="utf-8"))
        if "signal_worst" not in jd:
            src = os.path.join(d, "infeasible_best_scdc_extract_H400_300.json")
            jd = json.load(io.open(src, encoding="utf-8"))
        row["metrics_source"] = src
        row["metrics"] = {k: jd.get(k) for k in KEYS}
        row["eff_pct_of_ceiling"] = 100.0 * jd["signal_worst"] / CEIL
        # runs_log winner row
        rl = io.open(glob.glob(os.path.join(d, "runs_log_*.tsv"))[0], encoding="utf-8", errors="replace").read().strip().split("\n")
        hdr = rl[0].split("\t")
        tab = [r.split("\t") for r in rl[1:]]
        pick = None
        want_tag = "multi_q" if n == 1 else "warm"
        for r in tab:
            if r[0].endswith("_" + want_tag):
                pick = r
        assert pick is not None, (d, [r[0] for r in tab])
        rr = dict(zip(hdr, pick))
        row["runs_log"] = {k: rr[k] for k in ("tag", "signal_worst", "purity_worst", "echo_worst_p2_db", "minf1_nm", "minf2_nm", "sec")}
        row["runs_log"]["verdict_ascii"] = "ok" if rr["verdict"].strip().startswith("ok") else "INFEASIBLE"
        assert abs(float(rr["signal_worst"]) - jd["signal_worst"]) < 1e-5, (n, rr["signal_worst"], jd["signal_worst"])
        assert abs(float(rr["echo_worst_p2_db"]) - jd["echo_worst_p2_db"]) < 1e-4, (n, rr["echo_worst_p2_db"], jd["echo_worst_p2_db"])
        row["min_feature_nm"] = min(float(rr["minf1_nm"]), float(rr["minf2_nm"]))
        row["min_feature_gate_nm"] = float(job.get("min_feature_nm", "nan")) if job else None
        hp = glob.glob(os.path.join(d, "hist_scdc_extract_H400_300_*.npy"))
        if hp:
            h = np.load(hp[0])
            row["fom_hist_last"] = [float(h[-1, 0]), float(h[-1, 1])]
            row["fom_hist_first"] = [float(h[0, 0]), float(h[0, 1])]
        row["runs"] = {"L1": runs(dens[0]), "L2": runs(dens[1])}
        data["stages"].append(row)
        prev_md5 = md5(dp)
        log.append("stage %d %s md5 %s OK; eff %.2f %% echo %.3f dB purity %.6f minfeat %.1f nm (gate %s) verdict %s; filt %.0f nm; changed px %s" % (
            n, sim, prev_md5[:8], row["eff_pct_of_ceiling"], jd["echo_worst_p2_db"], jd["purity_worst"], row["min_feature_nm"], row["min_feature_gate_nm"],
            row["runs_log"]["verdict_ascii"], row["filt_radius_nm"], row.get("changed_px")))
    # start (stage 1 cold seed). The recorded winner of stage 1 is multi_q; its seed is analytic, so check it against the stage-1 checkpoint-free record:
    # the multi_q row's density file md5 equals the stage output md5 (asserted above via the *_multi_q copy in stage 1).
    mq = os.path.join(STAGES[0][2], "density_scdc_extract_H400_300_multi_q.npy")
    assert md5(mq)[:8] == "6c5c9510"
    start = seed_multi_q()
    data["start"] = {"name": "multi_q (stage-1 winner seed)", "x0": start.tolist(), "range": [float(start.min()), float(start.max())],
                     "mean": [float(start[0].mean()), float(start[1].mean())]}
    log.append("start multi_q: layer means %.4f / %.4f, range %.3f..%.3f (analytic, scd_c_grating.seed_bank)" % (start[0].mean(), start[1].mean(), start.min(), start.max()))
    # stage-1 JSON must carry harmonics [1, 2] and seed_rng 7, seeds 8
    j1 = json.load(io.open(os.path.join(STAGES[0][2], "scdc_extract_H400_300.json"), encoding="utf-8"))
    log.append("stage-1 json: seed %s, seeds %s, seed_rng %s, harmonics %s, iters %s" % (j1.get("seed"), j1.get("seeds"), j1.get("seed_rng"), j1.get("harmonics"), j1.get("iters")))
    assert j1.get("seed") == "multi_q" and j1.get("harmonics") in ([1, 2], (1, 2)), j1.get("harmonics")
    # key differences between the nine job files
    jobs = [parse(s["job_file"]) for s in data["stages"]]
    allk = sorted(set().union(*[set(j) for j in jobs]) - {"Output_folder", "subfolder_name", "seed_npy"})
    with io.open(os.path.join(OUT, "botchain_keydiff.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("key\t" + "\t".join("S%d_%s" % (s["stage"], s["sim"]) for s in data["stages"]) + "\n")
        for k in allk:
            vals = [j.get(k, "-") for j in jobs]
            if len(set(vals)) > 1:
                fh.write(k + "\t" + "\t".join(vals) + "\n")
    with io.open(os.path.join(OUT, "botchain_table.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("stage\tsim\tmd5_8\tfilt_radius_nm\tsignal_worst\teff_pct_of_ceiling\techo_worst_p2_db\tpurity_worst\tfid_worst\tmin_feature_nm\tgate_nm\tverdict\tchanged_px_L1\tchanged_px_L2\tfom_hist_last_it400\tsec\tmetrics_source\n")
        for s in data["stages"]:
            m = s["metrics"]
            cp = s.get("changed_px", ["", ""])
            fh.write("%d\t%s\t%s\t%.0f\t%.6f\t%.2f\t%.3f\t%.6f\t%s\t%.1f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                s["stage"], s["sim"], s["md5"][:8], s["filt_radius_nm"], m["signal_worst"], s["eff_pct_of_ceiling"], m["echo_worst_p2_db"], m["purity_worst"],
                ("%.6f" % m["fid_worst"]) if m.get("fid_worst") is not None else "", s["min_feature_nm"], s["min_feature_gate_nm"], s["runs_log"]["verdict_ascii"],
                cp[0], cp[1], ("%.6f" % s["fom_hist_last"][1]) if "fom_hist_last" in s else "", s["runs_log"]["sec"], os.path.basename(s["metrics_source"])))
    io.open(os.path.join(OUT, "botchain_data.json"), "w", encoding="utf-8", newline="\n").write(json.dumps(data))
    print("\n".join(log))
    print("chain md5 verified for 9 stages; wrote botchain_data.json, botchain_table.tsv, botchain_keydiff.tsv in", OUT)


if __name__ == "__main__":
    sys.exit(main())
