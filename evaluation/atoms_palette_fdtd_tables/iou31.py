r"""iou31.py - r26 comment #31: pixel IoU (solid pixels, both layers together) between the shape bank, the chosen bank shape 53474519 and the optimized atoms.
Read only on all data; writes only into this folder (iou31_out.txt, iou31_result.json). No simulation.
Binary: bank arrays are stored 0/1 (checked). Optimized atoms: raw density -> drv.project_recorded with the atom's own record (as xycut_tm_final.py), solid = projected > 0.5.
IoU = (sum over the two layers of intersections) / (sum over the two layers of unions) on the stored 96 x 96 pixel grid, at the stored alignment and maximised over cyclic
translations of the cell (both axes, one common shift for both layers), by FFT cross-correlation.
usage: python -I -B iou31.py"""
import os, sys, io, glob, json, hashlib
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
HERE = r"<WORK_ROOT>\caption_trim_261020\draft_r25\scratch_iou31"
LS = r"<WORK_ROOT>\scdc_scratch_261001\lens_seam_261005"
RES = r"<DATA_ROOT>\Result"
D2 = r"<DATA_ROOT>\Result\261002_scdc_shapebank2\_shapes"
D1 = r"<DATA_ROOT>\Result\261002_scdc_shapebank\_shapes"
sys.path.insert(0, LS)
os.chdir(LS)
ns = {"__name__": "tm_loader"}
src = io.open(LS + r"\run_tm.py", encoding="utf-8").read()
exec(compile(src[:src.index("DOC = {}")], "run_tm_loader", "exec"), ns)
os.chdir(HERE)
import numpy as np
drv = ns["drv"]; kv_file = ns["kv_file"]; BETA = ns["BETA"]; NXY = ns["NXY"]
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ---- records of the atoms: (label, expected md5 prefix, record dir)
RECS = [
    ("O3 (seeded from 53474519)", "db4998a5", RES + r"\261006_scdc_tm325n1793\261006_sim704_tm325o_O3_db4998a5_n1793", RES + r"\261006_scdc_tm325n1793\_inputs\O3\rho3_opt.npy"),
    ("77.5", "f8176952", RES + r"\261008_scdc_atomli\261008_sim751_atomli_077p5_n1793_liyff1", None),
    ("167.5", "ed750b02", RES + r"\261008_scdc_atomli\261008_sim751_atomli_167p5_n1793_liyff1", None),
    ("212.5", "78259586", RES + r"\261008_scdc_atomcand\261008_sim754_atomcand_78259586_n1793_liyff1", None),
    ("257.5", "accded83", RES + r"\261008_scdc_atomcand\261008_sim754_atomcand_accded83_n1793_liyff1", None),
    ("302.5", "ee3609ef", RES + r"\261009_scdc_atomreopt\261009_sim759_reopt_302p5_S1_488fc44a_ev1793_liyff1", None),
    ("347.5", "accf4825", RES + r"\261008_scdc_atomli\261008_sim751_atomli_347p5_n1793_liyff1", None),
]


def project(rec_dir, exp_md5, vp_over):
    rec = json.load(io.open(rec_dir + r"\atom2d_opt.json", encoding="utf-8"))
    jobs = glob.glob(rec_dir + r"\job_*.txt")
    assert len(jobs) == 1, jobs
    kv = kv_file(jobs[0])
    vp = vp_over or rec.get("rho3_npy") or kv["rho3_npy"]
    m5 = md5f(vp)
    assert m5.startswith(exp_md5), (vp, m5, exp_md5)
    raw = np.asarray(np.load(vp), float).reshape(-1)
    per = drv.project_recorded(raw, rec, beta=BETA, job=kv)
    P = [np.asarray(z, float) for z in per]
    assert len(P) == 2 and all(z.shape == (NXY, NXY) for z in P)
    soft_fill = [float(z.mean()) for z in P]
    bin_fill = [float((z > 0.5).mean()) for z in P]
    rf = rec.get("fill_frac_each")
    dev_soft = max(abs(a - b) for a, b in zip(soft_fill, rf))
    dev_bin = max(abs(a - b) for a, b in zip(bin_fill, rf))
    grey = int(sum(((z > 1e-4) & (z < 1 - 1e-4)).sum() for z in P))
    B = np.stack([z > 0.5 for z in P]).astype(np.float64)
    return dict(vp=vp, md5=m5, rec_fill=rf, soft_fill=soft_fill, bin_fill=bin_fill, dev_soft=dev_soft, dev_bin=dev_bin, beta_used=float(BETA), beta_eval=rec.get("beta_eval"),
                grey_px=grey, B=B, raw=raw, rec=rec)


def cyc_iou(A, B):
    """A, B: (2,96,96) 0/1 float. returns (iou at shift 0, max iou over cyclic shifts, shift (dx,dy) of the max (B rolled by shift))"""
    FA = np.fft.rfft2(A, axes=(1, 2)); FB = np.fft.rfft2(B, axes=(1, 2))
    # correlation c[s] = sum_x A[x] B[x - s]  (B rolled by +s)
    c = np.fft.irfft2(np.sum(FA * np.conj(FB), axis=0), s=A.shape[1:])
    c = np.rint(c)
    sa, sb = A.sum(), B.sum()
    U = sa + sb - c
    iou = c / U
    i0 = c[0, 0] / U[0, 0]
    k = np.unravel_index(int(np.argmax(iou)), iou.shape)
    return float(i0), float(iou[k]), (int(k[0]), int(k[1]))


# ---- bank
files = sorted(glob.glob(D2 + r"\*.npy")) + sorted(glob.glob(D1 + r"\*.npy"))
say("bank files: shapebank2 %d + shapebank %d = %d" % (len(glob.glob(D2 + r"\*.npy")), len(glob.glob(D1 + r"\*.npy")), len(files)))
bank = {}
sizes = {}
nonbin = 0
NB = []
for f in files:
    a = np.load(f)
    sizes[a.size] = sizes.get(a.size, 0) + 1
    if not np.all((a == 0) | (a == 1)):
        nonbin += 1
        NB.append((os.path.basename(f), float(a.min()), float(a.max()), int(((a > 1e-6) & (a < 1 - 1e-6)).sum())))
        a = (a > 0.5).astype(a.dtype)      # non-binary bank arrays: threshold at 0.5 (the paper's binary convention)
    bank[f] = (md5f(f), a)
say("array sizes (count): %s ; non-binary bank arrays (thresholded at 0.5 here): %d" % (sizes, nonbin))
for x in NB:
    say("   non-binary: %s min %.3f max %.3f grey px %d" % x)
assert list(sizes) == [2 * NXY * NXY], sizes
bm = {f: bank[f][0] for f in bank}
say("distinct bank md5: %d of %d files" % (len(set(bm.values())), len(files)))
f0367 = D2 + r"\shp_s602_0367.npy"
assert bank[f0367][0].startswith("53474519"), bank[f0367][0]
B53 = bank[f0367][1].reshape(2, NXY, NXY).astype(np.float64)
say("chosen bank shape: %s md5 %s fill %.4f / %.4f" % (f0367, bank[f0367][0], B53[0].mean(), B53[1].mean()))
bank_arr = {f: bank[f][1].reshape(2, NXY, NXY).astype(np.float64) for f in files}

res = {"nonbinary_list": NB, "bank_files": len(files), "bank_n_shapebank2": len(glob.glob(D2 + r"\*.npy")), "bank_n_shapebank": len(glob.glob(D1 + r"\*.npy")), "bank_nonbinary": nonbin,
       "chosen": {"file": f0367, "md5": bank[f0367][0]}, "atoms": []}
say("")
say("BINARIZATION: bank arrays are stored 0/1 except the non-binary ones listed above (thresholded at 0.5). Atoms: drv.project_recorded(raw, own record, beta=%s) then solid = projected > 0.5 (as xycut_tm_final.py / lens-seam audit)." % BETA)
for lab, m8, rdir, vpo in RECS:
    p = project(rdir, m8, vpo)
    say("")
    say("== %s  md5 %s  density %s" % (lab, p["md5"], p["vp"]))
    say("   record fill_frac_each %s | projected mean %s (dev %.2e) | (>0.5) fraction %s (dev %.2e) | beta used %s record beta_eval %s | grey px (1e-4..1-1e-4): %d"
        % (["%.6f" % x for x in p["rec_fill"]], ["%.6f" % x for x in p["soft_fill"]], p["dev_soft"], ["%.6f" % x for x in p["bin_fill"]], p["dev_bin"], p["beta_used"], p["beta_eval"], p["grey_px"]))
    # raw layout check: raw>0.5 vs projected>0.5 agreement
    rawb = (p["raw"].reshape(2, NXY, NXY) > 0.5).astype(float)
    say("   raw>0.5 vs projected agreement (pixel fraction): %.4f" % float((rawb == p["B"]).mean()))
    ent = {"label": lab, "md5": p["md5"], "density": p["vp"], "rec_fill": p["rec_fill"], "soft_fill": p["soft_fill"], "bin_fill": p["bin_fill"], "dev_soft": p["dev_soft"], "dev_bin": p["dev_bin"], "grey_px": p["grey_px"]}
    if lab.startswith("O3"):
        i0, im, sh = cyc_iou(B53, p["B"])
        say("   IoU(53474519, O3): stored alignment %.4f ; cyclic max %.4f at shift %s" % (i0, im, sh))
        ent.update({"iou_vs_53474519_stored": i0, "iou_vs_53474519_cyc": im, "shift": sh})
        # for information: IoU of O3 against all bank shapes
    best = (-1, None, None, None)
    rows = []
    for f in files:
        i0, im, sh = cyc_iou(bank_arr[f], p["B"])
        rows.append((im, i0, f))
        if im > best[0]:
            best = (im, i0, f, sh)
    rows.sort(reverse=True)
    say("   max over the %d bank shapes (cyclic max): %.4f, file %s (md5 %s), IoU at stored alignment %.4f, shift %s" % (len(files), best[0], os.path.basename(best[2]), bank[best[2]][0][:8], best[1], best[3]))
    say("   next best: " + "; ".join("%s %.4f" % (os.path.basename(r[2]), r[0]) for r in rows[1:4]))
    # stored-alignment max over the bank too
    rows0 = max(rows, key=lambda r: r[1])
    say("   max over bank at the STORED alignment: %.4f (%s)" % (rows0[1], os.path.basename(rows0[2])))
    ent.update({"best_bank_iou_cyc": best[0], "best_bank_file": best[2], "best_bank_md5": bank[best[2]][0], "best_bank_iou_stored_at_that_file": best[1], "best_shift": best[3],
                "max_bank_iou_stored": rows0[1], "max_bank_stored_file": rows0[2]})
    # also IoU to 53474519 for every optimized level
    i0, im, sh = cyc_iou(B53, p["B"])
    ent.update({"iou_vs_53474519_stored": i0, "iou_vs_53474519_cyc": im})
    say("   IoU vs 53474519 itself: stored %.4f, cyclic max %.4f" % (i0, im))
    res["atoms"].append(ent)

io.open(HERE + r"\iou31_out.txt", "w", encoding="utf-8", newline="\n").write("\n".join(OUT) + "\n")
io.open(HERE + r"\iou31_result.json", "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1, default=str))
