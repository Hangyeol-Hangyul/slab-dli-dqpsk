r"""[261010] CURRENT TM palette P769 under Li (261010 /XY_cut "1. atom"): copy of palette_xycut_261008\xycut_tm_261008.py with EVERY level
projected through its own Li nG-1793 record (rcwa_rule liy, rcwa_ff 1, eval_only, nG_eval 1793) and the delay arm read from the Li
carrier-at-token runs (sim761 / sim762 / sim764, Result\261009_scdc_tmpmli). Palette (per-slot md5, = sim769 lens cells):
32.5 53474519, 77.5 f8176952, 122.5 b59ed0ba (blank), 167.5 ed750b02, 212.5 78259586, 257.5 accded83, 302.5 S1 ee3609ef, 347.5 accf4825.
The loader part of run_tm.py is exec'd only for drv / kv_file / Member / PXNM / NXY / BETA.  No solver call; nothing is written under
the simulation data folder."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
sys.dont_write_bytecode = True
import glob
import hashlib
import io
import json
import textwrap

LS = r"<WORK_ROOT>\scdc_scratch_261001\lens_seam_261005"
OUT = r"<WORK_ROOT>\palette_xycut_261010"
RES = r"<DATA_ROOT>\Result"
PM = RES + r"\261009_scdc_tmpmli"
# level key, slot, Li nG-1793 record folder, expected density path, md5 prefix, delay-arm run folder (Li carrier-at-token), title suffix
LEVELS = [
    ("032p5", 32.5, RES + r"\261008_scdc_atomcand\261008_sim754_atomcand_53474519_n1793_liyff1",
     RES + r"\261005_scdc_tm325\_seeds\E7_53474519\rho3_opt.npy", "53474519", PM + r"\261009_sim761_pmli_TM_08_c032p5_53474519_n1793", " (E7, 261009)"),
    ("077p5", 77.5, RES + r"\261008_scdc_atomli\261008_sim751_atomli_077p5_n1793_liyff1",
     RES + r"\260926_scdc_g50fill\260926_sim460_L078_SG_g50_f110\rho3_opt.npy", "f8176952", PM + r"\261009_sim761_pmli_TM_01_ad077p5_f8176952_n1793", ""),
    ("122p5_blank", 122.5, RES + r"\261008_scdc_atomli\261008_sim751_atomli_122p5_n1793_liyff1",
     RES + r"\260925_scdc_blank\rho3_zero.npy", "b59ed0ba", PM + r"\261009_sim761_pmli_TM_02_ad122p5_b59ed0ba_n1793", ""),
    ("167p5", 167.5, RES + r"\261008_scdc_atomli\261008_sim751_atomli_167p5_n1793_liyff1",
     RES + r"\260927_scdc_thk2b\260927_sim502_s2b_L542_U332_RS_T167\rho3_opt.npy", "ed750b02", PM + r"\261009_sim761_pmli_TM_03_ad167p5_ed750b02_n1793", ""),
    ("212p5", 212.5, RES + r"\261008_scdc_atomcand\261008_sim754_atomcand_78259586_n1793_liyff1",
     RES + r"\260925_scdc_p2R0\260925_sim403_L232_SA_g50_m2000\rho3_opt.npy", "78259586", PM + r"\261009_sim764_pmli_TM_c212p5_78259586_n1793_t1200", " (261009)"),
    ("257p5", 257.5, RES + r"\261008_scdc_atomcand\261008_sim754_atomcand_accded83_n1793_liyff1",
     RES + r"\260926_scdc_g50fill\260926_sim460_L258_SF_g50_f110\rho3_opt.npy", "accded83", PM + r"\261009_sim764_pmli_TM_c257p5_accded83_n1793_t1200", " (261009)"),
    ("302p5", 302.5, RES + r"\261009_scdc_atomreopt\261009_sim759_reopt_302p5_S1_488fc44a_ev1793_liyff1",
     RES + r"\261009_scdc_atomreopt\261009_sim759_reopt_302p5_S1_488fc44a\rho3_opt.npy", "ee3609ef",
     PM + r"\261009_sim762_pmli_TM_00_c302p5_S1_488fc44a_ee3609ef_n1793_t1150", " (S1, 261009)"),
    ("347p5", 347.5, RES + r"\261008_scdc_atomli\261008_sim751_atomli_347p5_n1793_liyff1",
     RES + r"\261002_scdc_g62axis\261002_sim608_g62_TM_L347p5_F\rho3_opt.npy", "accf4825", PM + r"\261009_sim761_pmli_TM_07_ad347p5_accf4825_n1793", ""),
]
DELAY_NOTE = {"212p5": "out_l3_nm 1200 twin (t1150 / t1500 agree to 7e-7)", "257p5": "out_l3_nm 1200 twin (t1150 / t1500 agree to 2e-6)",
              "302p5": "out_l3_nm 1150 (the 875 / 1200 jobs stopped on the termination-synthesis guard)"}
TAGS = {"077p5": "G62 exception (x 6 px); its 261005 seam waiver to 32.5 not re-audited (32.5 changed)"}

os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, LS)
os.chdir(LS)
ns_tm = {"__name__": "tm_loader"}
src_tm = io.open(LS + r"\run_tm.py", encoding="utf-8").read()
exec(compile(src_tm[:src_tm.index("DOC = {}")], "run_tm_loader", "exec"), ns_tm)
os.chdir(OUT)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

PX = ns_tm["PXNM"]
NXY = ns_tm["NXY"]
BETA = ns_tm["BETA"]
drv = ns_tm["drv"]
kv_file = ns_tm["kv_file"]
Member = ns_tm["Member"]

plt.rcParams.update({"font.family": "Arial", "font.size": 5, "axes.linewidth": 0.4, "pdf.fonttype": 42, "ps.fonttype": 42})
CMAP = ListedColormap(["#FFFFFF", "#0072B2"])
VERM = "#D55E00"
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def md5_of(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def void_runs(b):
    """min self-wrap void run along x (axis 0) and y (axis 1) over lines that hold both solid and void, px (same definition as xycut_palettes.py)."""
    out = []
    for ax in (0, 1):
        z = b if ax == 0 else b.T
        best = None
        for line in z.T:
            if line.all() or not line.any():
                continue
            s = np.roll(line, -int(np.argmax(line)))   # start on a solid pixel: every void run lies inside one period
            runs, run = [], 0
            for v in s:
                if not v:
                    run += 1
                else:
                    if run:
                        runs.append(run)
                    run = 0
            if run:
                runs.append(run)
            if runs:
                best = min(runs) if best is None else min(best, min(runs))
        out.append(best)
    return out


def project_candidate(rec_dir, expect_md5_prefix, expect_rho):
    """record json + the job_*.txt beside it, density = the record's own rho3_opt.npy or rho3_npy, drv.project_recorded with its own settings."""
    rec_path = os.path.join(rec_dir, "atom2d_opt.json")
    rec = json.load(io.open(rec_path, encoding="utf-8"))
    jobs = glob.glob(os.path.join(rec_dir, "job_*.txt"))
    assert len(jobs) == 1, (rec_dir, jobs)
    kv = kv_file(jobs[0])
    assert (kv.get("rcwa_rule"), str(kv.get("rcwa_ff"))) == ("liy", "1"), ("not a Li record", rec_dir, kv.get("rcwa_rule"), kv.get("rcwa_ff"))
    assert str(kv.get("nG_eval")) == "1793" and str(kv.get("eval_only")) == "1", (rec_dir, kv.get("nG_eval"), kv.get("eval_only"))
    own = os.path.join(rec_dir, "rho3_opt.npy")
    vp = own if os.path.isfile(own) else (rec.get("rho3_npy") or kv["rho3_npy"])
    assert os.path.normcase(vp) == os.path.normcase(expect_rho), (vp, expect_rho)
    m5 = md5_of(vp)
    assert m5.startswith(expect_md5_prefix), (vp, m5)
    say("  density %s  md5 %s  (record job %s, BETA %s, record beta_eval %s)" % (vp, m5, os.path.basename(jobs[0]), BETA, rec.get("beta_eval")))
    per = drv.project_recorded(np.asarray(np.load(vp), float).reshape(-1), rec, beta=BETA, job=kv)
    P = [np.asarray(z, float) for z in per]
    assert len(P) == 2 and all(z.shape == (NXY, NXY) for z in P), [z.shape for z in P]
    return rec, P, m5, vp, rec_path


def _cdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def delay_arm(run_dir, rec):
    """-theta delay-arm purity (purity_m_worst) of one level from its Li carrier-at-token run; the run's single token must match the record's
    (absR00, argR00_token_deg, arg_rho_TM_deg) to the 3-decimal precision the token string carries."""
    p = os.path.join(run_dir, "scdc_extract_H400_300.json")
    d = json.load(io.open(p, encoding="utf-8"))
    assert (d.get("rcwa_rule"), d.get("rcwa_ff")) == ("liy", 1), (p, d.get("rcwa_rule"), d.get("rcwa_ff"))
    tk = d["out_l3_r_list_eval"].strip()
    assert ";" not in tk, tk
    f = [float(x) for x in tk.split(":")]
    rt = (rec["absR00"], rec["argR00_token_deg"], rec["arg_rho_TM_deg"])
    assert abs(f[0] - rt[0]) <= 6e-4 and _cdiff(f[1], rt[1]) <= 6e-4 and _cdiff(f[2], rt[2]) <= 6e-4, ("token does not match the record", p, tk, rt)
    kk = "n%04d_purity_m_worst" % round(d["out_l3_ns_used"][0] * 1000)
    assert d["purity_m_worst"] == d[kk], (p, kk)
    return dict(value=d["purity_m_worst"], src=p, key="purity_m_worst (token %s, out_l3_nm %s)" % (tk, d["out_l3_nm"]))


def draw_compact(rows, fname):
    """2 x 4 blocks (phase order), each block = two equal squares (L0 lower | L1 upper), text below.  All sizes in mm."""
    FIGW = 183.0
    LM, RM = 8.0, 1.0
    PITCH = (FIGW - LM - RM) / 4.0
    S, GAP = 19.0, 2.0
    TITLE_H, SUB_H, TICK_H, XLAB_H, TEXT_H, ROWGAP, NOTE_H = 4.2, 3.0, 3.0, 2.8, 14.5, 1.5, 7.5
    LPITCH = 2.35
    row_h = [TITLE_H + SUB_H + S + TICK_H + TEXT_H + ROWGAP, TITLE_H + SUB_H + S + TICK_H + XLAB_H + TEXT_H + ROWGAP]
    H = sum(row_h) + NOTE_H
    fig = plt.figure(figsize=(FIGW / 25.4, H / 25.4))
    W = NXY * PX
    texts = []

    def T(x_mm, ytop_mm, s, **kw):
        t = fig.text(x_mm / FIGW, 1.0 - ytop_mm / H, s, va="top", ha="left", **kw)
        texts.append(t)
        return t

    for idx, r in enumerate(rows):
        ri, ci = divmod(idx, 4)
        y0 = 0.0 if ri == 0 else row_h[0]
        x0 = LM + ci * PITCH
        T(x0, y0 + 0.2, r["title"], fontsize=6.5, fontweight="bold", color="#000000")
        ysq = y0 + TITLE_H + SUB_H
        for k in range(2):
            xs = x0 + k * (S + GAP)
            T(xs, y0 + TITLE_H, "L0 lower" if k == 0 else "L1 upper", fontsize=5.5, color="#000000")
            a = fig.add_axes([xs / FIGW, 1.0 - (ysq + S) / H, S / FIGW, S / H])
            a.imshow(r["m"].bin[k].T, origin="lower", extent=[0, W, 0, W], cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
            a.set_xlim(0, W)
            a.set_ylim(0, W)
            a.set_aspect("equal")
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_color("#9A9A9A")
                sp.set_linewidth(0.4)
            outer_x, outer_y = (ri == 1), (ci == 0 and k == 0)
            a.set_xticks([0, 200, 400] if outer_x else [])
            a.set_yticks([0, 200, 400] if outer_y else [])
            a.tick_params(length=1.5, width=0.4, labelsize=5, pad=1, colors="#222222")
            if outer_x:
                a.set_xlabel("x (nm)", fontsize=5, labelpad=1)
            if outer_y:
                a.set_ylabel("y (nm)", fontsize=5, labelpad=1)
        yt = ysq + S + TICK_H + (XLAB_H if ri == 1 else 0.0) + 0.3
        for j, ln in enumerate(r["lines"]):
            T(x0, yt + j * LPITCH, ln, fontsize=5, color="#000000")
        if r["tag"]:
            for j, ln in enumerate(textwrap.wrap(r["tag"], 34)):
                T(x0, yt + (len(r["lines"]) + j) * LPITCH, ln, fontsize=5, fontweight="bold", color=VERM)
    T(LM, H - NOTE_H + 1.0,
      "Axes: x = plane-of-incidence axis (TM sensitive axis), y = in-plane perpendicular axis; ticks in nm. Each square is one 438.46 nm cell (96 x 96 px, 4.567 nm/px); "
      "blue = solid, white = void.", fontsize=5, color="#333333")
    T(LM, H - NOTE_H + 1.0 + LPITCH,
      "Palette P769 (approved, Li). A / eff / echo / purity ref / lw = each atom's own Li nG-1793 record; delay = -theta delay-arm purity, Li carrier-at-token run.",
      fontsize=5, color="#333333")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    over = []
    for t in texts:
        wmm = t.get_window_extent(rend).width / fig.dpi * 25.4
        xmm = t.get_position()[0] * FIGW
        if t.get_text().startswith("Axes:") or t.get_text().startswith("Palette"):
            if xmm + wmm > FIGW:
                over.append((round(xmm + wmm - FIGW, 2), t.get_text()[:40]))
        elif wmm > PITCH - 1.0:
            over.append((round(wmm, 2), t.get_text()))
    say("text-width check (limit %.2f mm per block; notes limit figure width): %s" % (PITCH - 1.0, over if over else "OK, none exceed"))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, fname + "." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fnum(x, p, na="n/a"):
    return na if x is None else p % x


# =========================================================================================== P769 members, each through its own Li record
say("=== TM palette P769 (Li): every level projected through its own Li nG-1793 record ===")
rows = []
for key, slot, rdir, rho, m8, pmdir, deco in LEVELS:
    say("level %s (%.1f):" % (key, slot))
    rec, P, m5, vp, rp = project_candidate(rdir, m8, rho)
    ff = rec.get("fill_frac_each")
    soft = [float(z.mean()) for z in P]
    say("  record fill_frac_each %s | projected soft mean %s | binary %s | record A_deg %.4f mod 360 %.4f" % (
        [round(x, 6) for x in ff], [round(x, 6) for x in soft], [round(float((z > 0.5).mean()), 6) for z in P], rec["A_deg"], rec["A_deg"] % 360))
    assert ff is not None and all(abs(a - b) < 1e-3 for a, b in zip(ff, soft)), ("projection does not reproduce the record", key, ff, soft)
    m = Member("%.1f(%s)" % (slot, m8), "atom", P, slot, dict(level_key=key, vec=vp, md5=m5, rec=rp))
    A, eff, echo = rec["A_deg"] % 360, rec["carrier_eff_pct"], rec.get("carrier_echo_db", rec.get("echo_p2_db"))
    lw = rec.get("min_feature_meas_nm")
    lw = None if (lw is None or lw == float("inf")) else lw
    voids = [void_runs(b) for b in m.bin]
    vx = [v[0] for v in voids if v[0] is not None]
    vy = [v[1] for v in voids if v[1] is not None]
    vxm, vym = (min(vx) if vx else None), (min(vy) if vy else None)
    dl = delay_arm(pmdir, rec)
    blank = key == "122p5_blank"
    title = "%.1f  %s" % (slot, ("blank (%s)" % m8) if blank else (m8 + deco))
    l1 = "A %.1f deg | eff %.2f %% | echo %.2f dB" % (A, eff, echo)
    l2 = "purity ref %.5f / delay %.5f | lw %s" % (rec["purity"], dl["value"], ("n/a (blank)" if lw is None else "%.1f nm" % lw))
    l3 = "no structure (blank atom)" if blank else "min self void x / y = %s / %s px" % (fnum(vxm, "%d", "none"), fnum(vym, "%d", "none"))
    rows.append(dict(m=m, key=key, slot=slot, md5=m5, md8=m8, title=title, lines=[l1, l2, l3], tag=TAGS.get(key), rec=rec, rec_path=rp,
                     A=A, eff=eff, echo=echo, pur=rec["purity"], lw=lw, voids=voids, vxm=vxm, vym=vym, dl=dl, fill=[float(b.mean()) for b in m.bin],
                     dnote=DELAY_NOTE.get(key, "out_l3_nm 875")))
assert [r["slot"] for r in rows] == [32.5, 77.5, 122.5, 167.5, 212.5, 257.5, 302.5, 347.5]
say("figure rows:")
for r in rows:
    say("  " + r["title"] + "   //   " + "   //   ".join(r["lines"]) + (("   //   TAG: " + r["tag"]) if r["tag"] else ""))

draw_compact(rows, "tm_palette_xycut_p769")

# =========================================================================================== table
L = []
L.append("# TM palette P769 XY cuts (261010): numbers behind `tm_palette_xycut_p769.png` / `.pdf`")
L.append("")
L.append("Generated by `<WORK_ROOT>\\palette_xycut_261010\\xycut_tm_p769.py` (copy of `<WORK_ROOT>\\palette_xycut_261008\\xycut_tm_261008.py`). "
         "No solver was run. Cell = %d x %d px at %.4f nm/px = %.2f nm. Every level is projected through its own Li nG-1793 record (rcwa_rule liy, rcwa_ff 1, eval_only, nG_eval 1793)." % (NXY, NXY, PX, NXY * PX))
L.append("")
L.append("## 1. Numbers shown in the figure")
L.append("")
L.append("| level (deg) | atom (md5, 8) | A (deg) | eff (%) | echo (dB) | purity ref | purity delay | lw (nm) | min self void x (px) | min self void y (px) | status tag |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    bl = r["key"] == "122p5_blank"
    L.append("| %.1f | %s | %.2f | %.2f | %.2f | %.6f | %.6f | %s | %s | %s | %s |" % (
        r["slot"], r["md8"], r["A"], r["eff"], r["echo"], r["pur"], r["dl"]["value"], fnum(r["lw"], "%.2f", "n/a (blank)"),
        "blank" if bl else fnum(r["vxm"], "%d", "none"), "blank" if bl else fnum(r["vym"], "%d", "none"), r["tag"] or "-"))
L.append("")
L.append("## 2. Void runs and fill per layer (px, from `void_runs()` on the projected binary)")
L.append("")
L.append("| level (deg) | L0 x | L0 y | L1 x | L1 y | fill L0 | fill L1 |")
L.append("|---|---|---|---|---|---|---|")
for r in rows:
    v = r["voids"]
    L.append("| %.1f | %s | %s | %s | %s | %.3f | %.3f |" % (r["slot"], fnum(v[0][0], "%d", "-"), fnum(v[0][1], "%d", "-"), fnum(v[1][0], "%d", "-"),
                                                          fnum(v[1][1], "%d", "-"), r["fill"][0], r["fill"][1]))
L.append("")
L.append("## 3. Sources, one row per level")
L.append("")
L.append("| level (deg) | density md5 (full) | Li nG-1793 record | delay-arm run | delay key | delay note |")
L.append("|---|---|---|---|---|---|")
for r in rows:
    L.append("| %.1f | %s | `%s` | `%s` | %s | %s |" % (r["slot"], r["md5"], r["rec_path"], r["dl"]["src"], r["dl"]["key"], r["dnote"]))
L.append("")
L.append("## 4. Notes")
L.append("")
L.append("- All eight numbers sets are on ONE basis (Li nG 1793 records; delay arm Li carrier-at-token, nG_eval as in each run). The 261008 figure mixed Laurent records.")
L.append("- Gates: atom eff >= 85 %, echo <= -20.5 dB, purity (reference and delay arm) >= 0.996722. 257.5 accded83 clears eff by 0.17 pp; 32.5 clears echo by 0.45 dB.")
L.append("- Seams: 53474519, 78259586 and S1 ee3609ef were seam-audited CLEAN when they were selected (261009); the 77.5 tag from 261005 refers to the old 32.5 neighbour and was not re-audited for this figure.")
L.append("- 122.5 is the blank atom (no structure); lw is `inf` in its record and is shown as n/a.")
L.append("")
io.open(os.path.join(OUT, "tm_palette_table_p769.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
io.open(os.path.join(OUT, "xycut_tm_p769_log.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
for fn in ("tm_palette_xycut_p769.png", "tm_palette_xycut_p769.pdf", "tm_palette_table_p769.md", "xycut_tm_p769_log.txt"):
    say("wrote " + os.path.join(OUT, fn))
