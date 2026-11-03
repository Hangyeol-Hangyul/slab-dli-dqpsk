r"""[261006] FINAL TM palette: copy of palette_xycut_261005c\xycut_tm_compact.py with level 32.5 replaced by O3 (sim696 O3, db4998a5; 261006 14:0x),
projected through its own sim704 nG-1793 record; delay arm from sim708. Original docstring follows.
ONE compact figure of the XY cuts of the CURRENT TM L3 palette (copied from palette_xycut_261005b\xycut_current.py; TE part dropped; only the drawing replaced).  No solver call.
  TM current = TM-2 (palettes\TM_g62_261003, run_tm.tm_members(None, 'B0')) with the 77.5 level replaced by f8176952 (261005), projected through
               ITS OWN record exactly as tm_atom_xycut_261005\xycut_candidates.py does (drv.project_recorded with its own atom2d_opt.json + job_*.txt).
The loader part of run_tm.py is exec'd (its audit body is not run).  Nothing is written under the simulation data folder."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
sys.dont_write_bytecode = True
import csv
import glob
import hashlib
import io
import json
import textwrap

LS = r"<WORK_ROOT>\scdc_scratch_261001\lens_seam_261005"
OUT = r"<WORK_ROOT>\palette_xycut_261006"
RES = r"<DATA_ROOT>\Result"
V6CSV = r"<WORK_ROOT>\scdc_scratch_261001\readout_261002_2035\verify604\v6_table.csv"
PMTM = r"<WORK_ROOT>\scdc_scratch_261001\purity_m_261005\purity_m_tm.json"
J597 = r"<DATA_ROOT>\Result" + r"\261002_scdc_puremscore\261002_sim597_pms_tmpal_n161\scdc_extract_H400_300.json"
J643 = RES + r"\261005_scdc_tm077f817\261005_sim643_pm_TM_f8176952\scdc_extract_H400_300.json"
J82 = RES + r"\261005_scdc_puritym\261005_sim636_pm_TM_L302p5_t1300\scdc_extract_H400_300.json"
J81 = RES + r"\261005_scdc_puritym\261005_sim636_pm_TM_L347p5\scdc_extract_H400_300.json"
PALDIR = r"<DATA_ROOT>\palettes\TM_g62_261003"
REC77 = r"<DATA_ROOT>\Result\261005_scdc_tm077eval\261005_sim641_tm077_f8176952_n1793\atom2d_opt.json"
RHO77 = r"<DATA_ROOT>\Result\260926_scdc_g50fill\260926_sim460_L078_SG_g50_f110\rho3_opt.npy"
REC32 = r"<DATA_ROOT>\Result\261006_scdc_tm325n1793\261006_sim704_tm325o_O3_db4998a5_n1793\atom2d_opt.json"
RHO32 = r"<DATA_ROOT>\Result\261006_scdc_tm325n1793\_inputs\O3\rho3_opt.npy"
J708 = RES + r"\261006_scdc_tm325pm\261006_sim708_pm_TM_O3_db4998a5\scdc_extract_H400_300.json"
EFF_PER_AMP = 843.167          # driver _eff: eff_pct = 843.167 * prod_amp (palettes\TM_g62_261003\README.md section 2)

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


def rec_numbers(rec):
    """A (deg, as recorded), eff (%), echo (dB), eff_note.  eff = carrier_eff_pct if present, else 843.167 * prod_amp (flagged 'derived')."""
    eff, note = rec.get("carrier_eff_pct"), "record"
    if eff is None and rec.get("prod_amp") is not None:
        eff, note = EFF_PER_AMP * rec["prod_amp"], "derived 843.167*prod_amp"
    echo = rec.get("carrier_echo_db", rec.get("echo_p2_db"))
    return rec.get("A_deg"), eff, echo, note


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


def project_candidate(rec_path, expect_md5_prefix, expect_rho):
    """exactly xycut_candidates.load(): record json + the job_*.txt beside it, density = the record's rho3_npy, drv.project_recorded with its own settings."""
    rec = json.load(io.open(rec_path, encoding="utf-8"))
    jobs = glob.glob(os.path.join(os.path.dirname(rec_path), "job_*.txt"))
    assert len(jobs) == 1, (rec_path, jobs)
    kv = kv_file(jobs[0])
    own = os.path.join(os.path.dirname(rec_path), "rho3_opt.npy")
    vp = own if os.path.isfile(own) else (rec.get("rho3_npy") or kv["rho3_npy"])
    assert os.path.normcase(vp) == os.path.normcase(expect_rho), (vp, expect_rho)
    m5 = md5_of(vp)
    assert m5.startswith(expect_md5_prefix), (vp, m5)
    say("  density %s  md5 %s  (record job %s, BETA %s, record beta_eval %s)" % (vp, m5, os.path.basename(jobs[0]), BETA, rec.get("beta_eval")))
    per = drv.project_recorded(np.asarray(np.load(vp), float).reshape(-1), rec, beta=BETA, job=kv)
    P = [np.asarray(z, float) for z in per]
    assert len(P) == 2 and all(z.shape == (NXY, NXY) for z in P), [z.shape for z in P]
    return rec, P, m5, vp


def _load(p):
    return json.load(io.open(p, encoding="utf-8"))


def _cdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _tok(s):
    f = [float(x) for x in s.split(":")]
    return f if len(f) == 3 else None


def _tok_match(t, rt, ta=6e-4, targ=0.51):
    """t = 'abs:arg:rho' string, rt = (absR00, argR00_token_deg, arg_rho_TM_deg) of a record."""
    f = _tok(t)
    return f is not None and abs(f[0] - rt[0]) <= ta and _cdiff(f[1], rt[1]) <= targ and _cdiff(f[2], rt[2]) <= targ


def delay_arm(key, md5_8, rec):
    """-theta delay-arm purity (purity_m_worst) of one level, read from disk -> dict(value, method, src, key); value None when not found."""
    rt = (rec["absR00"], rec["argR00_token_deg"], rec["arg_rho_TM_deg"])
    na = dict(value=None, method="not found", src="-", key="-")
    if key == "032p5" and md5_8 == "db4998a5":
        d = _load(J708)
        tk = d["out_l3_r_list_eval"].strip()
        assert ";" not in tk and _tok_match(tk, rt, 1.1e-3, 0.06), (key, tk, rt)
        return dict(value=d["purity_m_worst"], method="sim708 carrier-at-token, eval_only (O3 delay arm, 261006)", src=J708,
                    key="purity_m_worst (token %s, out_l3_nm %s)" % (tk, d["out_l3_nm"]))
    if key == "032p5":
        assert md5_8 == "7e7648f7", md5_8    # p62c_Y0 (palettes\TM_g62_261003\README.md section 3)
        for r in csv.DictReader(io.open(V6CSV, encoding="utf-8")):
            if r["job"] == "p62c_Y0_n1793":
                return dict(value=float(r["pur_delay_1d"]), method="sim604 verification (v6_table.md): 1-D full stack, y-invariant ridge, nG 1793 (M41)",
                            src=V6CSV, key="pur_delay_1d, row p62c_Y0_n1793")
        return na
    if key in ("077p5", "347p5", "302p5"):
        p = {"077p5": J643, "347p5": J81, "302p5": J82}[key]
        if not os.path.isfile(p):
            return na
        d = _load(p)
        tk = d["out_l3_r_list_eval"].strip()
        assert ";" not in tk and _tok_match(tk, rt, 1.1e-3, 0.06), (key, tk, rt)
        assert d["purity_m_worst"] == d["n%04d_purity_m_worst" % round(d["out_l3_ns_used"][0] * 1000)]
        meth = {"077p5": "sim643 carrier-at-token, eval_only, nG_eval 161 (control reproduces stored to 5e-14)",
                "347p5": "sim636 job_81 carrier-at-token, eval_only, nG_eval 161",
                "302p5": "sim636 job_82 carrier-at-token, eval_only, nG_eval 161, out_l3_nm 1300 (875 token unreachable)"}[key]
        return dict(value=d["purity_m_worst"], method=meth, src=p, key="purity_m_worst (token %s, out_l3_nm %s)" % (tk, d["out_l3_nm"]))
    # 122.5 / 167.5 / 212.5 / 257.5: sim597 (frozen = current density; same_density checked in purity_m_tm.json)
    lv = [L for L in _load(PMTM)["levels"] if L["level"] == "level_" + key]
    assert len(lv) == 1 and lv[0]["sim597"]["same_density"] is True, (key, lv)
    d = _load(J597)
    toks = [t.strip() for t in d["out_l3_r_list_eval"].split(";")]
    ns = d["out_l3_ns_used"]
    assert len(toks) == len(ns), (len(toks), len(ns))
    hit = [i for i, t in enumerate(toks) if _tok_match(t, rt)]
    assert len(hit) == 1, (key, hit)
    kk = "n%04d_purity_m_worst" % round(ns[hit[0]] * 1000)
    return dict(value=d[kk], method="sim597 carrier-at-token, eval_only, nG_eval 161 (n161 job)", src=J597, key="%s (token %s)" % (kk, toks[hit[0]]))


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
      "Order: phase (A) levels 32.5 ... 347.5 deg. Vermilion = exception / waiver. Delay = -theta delay-arm purity (purity_m_worst); method differs per level (see tm_palette_table.md).",
      fontsize=5, color="#333333")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    over = []
    for t in texts:
        wmm = t.get_window_extent(rend).width / fig.dpi * 25.4
        xmm = t.get_position()[0] * FIGW
        if t.get_text().startswith("Axes:") or t.get_text().startswith("Order:"):
            if xmm + wmm > FIGW:
                over.append((round(xmm + wmm - FIGW, 2), t.get_text()[:40]))
        elif wmm > PITCH - 1.0:
            over.append((round(wmm, 2), t.get_text()))
    say("text-width check (limit %.2f mm per block; notes limit figure width): %s" % (PITCH - 1.0, over if over else "OK, none exceed"))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, fname + "." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================================================== TM current palette
say("=== TM current palette = TM-2 (loader run_tm.tm_members(None,'B0')) with 77.5 -> f8176952 (own record projection) ===")
tm = ns_tm["tm_members"](None, "B0")
i77 = [i for i, m in enumerate(tm) if m.extra["level_key"] == "077p5"]
assert len(i77) == 1
b0 = tm[i77[0]]
say("  replaced B0 77.5 member: %s md5 %s (loader name %s)" % (b0.extra["vec"], b0.extra["md5"][:8], b0.name))
rec77, P77, m5_77, _ = project_candidate(REC77, "f8176952", RHO77)
say("  f8176952 record fill_frac_each %s vs projected-binary fill L0 %.4f L1 %.4f ; record A_deg %.4f mod 360 %.4f" % (
    rec77.get("fill_frac_each"), (P77[0] > 0.5).mean(), (P77[1] > 0.5).mean(), rec77["A_deg"], rec77["A_deg"] % 360))
new77 = Member("77.5(f8176952)", "atom", P77, 77.5, dict(level_key="077p5", vec=RHO77, md5=m5_77, rec=REC77))
tm[i77[0]] = new77
i32 = [i for i, m in enumerate(tm) if m.extra["level_key"] == "032p5"]
assert len(i32) == 1
say("  replaced 32.5 member: %s md5 %s (loader name %s)" % (tm[i32[0]].extra["vec"], tm[i32[0]].extra["md5"][:8], tm[i32[0]].name))
rec32, P32, m5_32, _ = project_candidate(REC32, "db4998a5", RHO32)
say("  O3 record fill_frac_each %s vs projected-binary fill L0 %.4f L1 %.4f ; record A_deg %.4f" % (
    rec32.get("fill_frac_each"), (P32[0] > 0.5).mean(), (P32[1] > 0.5).mean(), rec32["A_deg"]))
new32 = Member("32.5(O3)", "atom", P32, 32.5, dict(level_key="032p5", vec=RHO32, md5=m5_32, rec=REC32))
tm[i32[0]] = new32


def fnum(x, p, na="n/a"):
    return na if x is None else p % x



# =========================================================================================== rows for the figure and the table
TAGS = {"077p5": "G62 exception (x 6 px), 3 px seam to 32.5 waived",
        "257p5": "eff 81.89 accepted + G62 exception",
        "302p5": "G62 pass by 1 px"}
DECO = {"032p5": " (O3, new 261006)", "077p5": " (new 261005)", "347p5": " (new 261003)"}
assert [round(m.slot, 1) for m in tm] == [32.5, 77.5, 122.5, 167.5, 212.5, 257.5, 302.5, 347.5], [m.slot for m in tm]
rows = []
for m in tm:
    key = m.extra["level_key"]
    rp = REC32 if m is new32 else REC77 if m is new77 else os.path.join(PALDIR, "level_" + key, "nG1793_record_atom2d_opt.json")
    rec = _load(rp)
    A, eff, echo, eff_note = rec_numbers(rec)
    md8 = m.extra["md5"][:8]
    lw = rec.get("min_feature_meas_nm")
    lw = None if (lw is None or lw == float("inf")) else lw
    voids = [void_runs(b) for b in m.bin]
    vx = [v[0] for v in voids if v[0] is not None]
    vy = [v[1] for v in voids if v[1] is not None]
    vxm, vym = (min(vx) if vx else None), (min(vy) if vy else None)
    dl = delay_arm(key, md8, rec)
    blank = key == "122p5_blank"
    title = "%.1f  %s" % (m.slot, ("blank (%s)" % md8) if blank else (md8 + DECO.get(key, "")))
    l1 = "A %.1f deg | eff %.2f %% | echo %.2f dB" % (A % 360, eff, echo)
    l2 = "purity ref %.5f / delay %s | lw %s" % (rec["purity"], fnum(dl["value"], "%.5f"), ("n/a (blank)" if lw is None else "%.1f nm" % lw))
    l3 = "no structure (blank atom)" if blank else "min self void x / y = %s / %s px" % (fnum(vxm, "%d", "none"), fnum(vym, "%d", "none"))
    rows.append(dict(m=m, key=key, slot=m.slot, md5=m.extra["md5"], md8=md8, title=title, lines=[l1, l2, l3], tag=TAGS.get(key), rec=rec, rec_path=rp,
                     A=A % 360, eff=eff, eff_note=eff_note, echo=echo, pur=rec["purity"], lw=lw, voids=voids, vxm=vxm, vym=vym, dl=dl,
                     fill=[float(b.mean()) for b in m.bin]))
say("figure rows:")
for r in rows:
    say("  " + r["title"] + "   //   " + "   //   ".join(r["lines"]) + (("   //   TAG: " + r["tag"]) if r["tag"] else ""))

draw_compact(rows, "tm_palette_xycut_final")


# =========================================================================================== tm_palette_table.md
def f(x, p, na="n/a"):
    return na if x is None else p % x


L = []
L.append("# TM palette XY cuts: numbers behind `tm_palette_xycut.png` / `.pdf`")
L.append("")
L.append("Generated by `<WORK_ROOT>\\palette_xycut_261005c\\xycut_tm_compact.py` (copy of `<WORK_ROOT>\\palette_xycut_261005b\\xycut_current.py`; TE part dropped, only the drawing replaced). No solver was run. Cell = %d x %d px at %.4f nm/px = %.2f nm." % (NXY, NXY, PX, NXY * PX))
L.append("")
L.append("Palette = TM-2 (`<DATA_ROOT>\\palettes\\TM_g62_261003`, loader `run_tm.tm_members(None, 'B0')`) with level 77.5 replaced by f8176952, projected through its own record `%s`." % REC77)
L.append("")
L.append("## 1. Numbers shown in the figure")
L.append("")
L.append("A is `A_deg` mod 360. eff, echo and reference purity are the nG 1793 record values (eff from `carrier_eff_pct` where the record has it, otherwise 843.167 x `prod_amp`, see section 3). Delay = -theta delay-arm purity (`purity_m_worst`). Min self void = smaller of the L0 and L1 values from `void_runs()` (px; none = no line holds both solid and void on that axis).")
L.append("")
L.append("| level (deg) | atom (md5, 8) | A (deg) | eff (%) | echo (dB) | purity ref | purity delay | lw (nm) | min self void x (px) | min self void y (px) | status tag |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    L.append("| %.1f | %s | %.2f | %.2f | %.2f | %.6f | %s | %s | %s | %s | %s |" % (
        r["slot"], r["md8"], r["A"], r["eff"], r["echo"], r["pur"], f(r["dl"]["value"], "%.6f"), f(r["lw"], "%.2f", "n/a (blank)"),
        "blank" if r["key"] == "122p5_blank" else f(r["vxm"], "%d", "none"), "blank" if r["key"] == "122p5_blank" else f(r["vym"], "%d", "none"), r["tag"] or "-"))
L.append("")
L.append("## 2. Void runs and fill per layer (px, from `void_runs()` on the projected binary)")
L.append("")
L.append("| level (deg) | L0 x | L0 y | L1 x | L1 y | fill L0 | fill L1 |")
L.append("|---|---|---|---|---|---|---|")
for r in rows:
    v = r["voids"]
    L.append("| %.1f | %s | %s | %s | %s | %.3f | %.3f |" % (r["slot"], f(v[0][0], "%d", "-"), f(v[0][1], "%d", "-"), f(v[1][0], "%d", "-"), f(v[1][1], "%d", "-"), r["fill"][0], r["fill"][1]))
L.append("")
L.append("## 3. Sources, one row per level")
L.append("")
L.append("Record = file the A / eff / echo / reference purity / lw values were read from (keys `A_deg`, `carrier_eff_pct` or `prod_amp`, `carrier_echo_db` or `echo_p2_db`, `purity`, `min_feature_meas_nm`).")
L.append("")
L.append("| level (deg) | record file | eff basis | reference purity (full) | lw (full) |")
L.append("|---|---|---|---|---|")
for r in rows:
    L.append("| %.1f | `%s` | %s (%.4f) | %.16f | %s |" % (r["slot"], r["rec_path"], r["eff_note"], r["eff"], r["pur"], "n/a (blank, record = inf)" if r["lw"] is None else "%.14f" % r["lw"]))
L.append("")
L.append("Delay-arm purity (`purity_m_worst`), value read from disk by the script:")
L.append("")
L.append("| level (deg) | delay purity (full) | method | file | key |")
L.append("|---|---|---|---|---|")
for r in rows:
    d = r["dl"]
    L.append("| %.1f | %s | %s | `%s` | %s |" % (r["slot"], "n/a" if d["value"] is None else "%.16f" % d["value"], d["method"], d["src"], d["key"]))
L.append("")
L.append("## 4. Notes")
L.append("")
L.append("- **Delay-arm methods are not identical across levels.** 32.5 is the sim604 verification (v6_table, 1-D full stack at nG 1793, column pur_delay_1d). The other seven are carrier-at-token estimates (nG_eval 161). For 32.5 the sim604 carrier-token value is 0.9991790207339124 and the 2-D nG 601 value is 0.9992383627815191 (`<WORK_ROOT>\\scdc_scratch_261001\\purity_m_261005\\purity_m_tm.json`, level_032p5); the earlier quoted value 0.99918 is the carrier-token number. Per the task, 0.99924 (v6_table.md) is the one printed.")
L.append("- 77.5 = f8176952: delay 0.999388 and reference 0.999794 (record) are both above the floor 0.996722; sim643 control reproduces the stored sim636 value to 4.985e-14 (`<WORK_ROOT>\\scdc_scratch_261001\\gen261005_tm077f817\\readout_sim643.md`).")
L.append("- 302.5 delay was measured at `out_l3_nm` 1300 (the 875 nm token is unreachable); 0.998735 and 0.999434 (347.5) confirmed in the result jsons listed in section 3.")
L.append("- 257.5 min self void x: `void_runs()` returns the smallest run over all lines, 2 px in L1 (not inspected further here); the G62 self gap of record for this atom is 4 px = 18.27 nm (`palettes\\TM_g62_261003\\README.md`, `manifest.csv`), and the 261005 slit-rules note records 2-3 px one-row notches on 257.5 as exempt staircase (D4). The figure prints the function value unchanged.")
L.append("- Min self void for 32.5 has no y value because the atom is y-invariant (1-D ridge): no line along y holds both solid and void.")
L.append("- 122.5 is the blank atom (no structure); lw is `inf` in its record and is shown as n/a.")
L.append("")
io.open(os.path.join(OUT, "tm_palette_table_final.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
io.open(os.path.join(OUT, "xycut_tm_compact_log.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
for fn in ("tm_palette_xycut_final.png", "tm_palette_xycut_final.pdf", "tm_palette_table_final.md", "xycut_tm_compact_log.txt"):
    say("wrote " + os.path.join(OUT, fn))
