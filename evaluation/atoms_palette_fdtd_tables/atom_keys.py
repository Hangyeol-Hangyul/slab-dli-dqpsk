import re
files = {
 "77.5 sim460": r"<DATA_ROOT>\Result\260926_scdc_g50fill\260926_sim460_L078_SG_g50_f110\job_10_260926_sim460_L078_SG_g50_f110.txt",
 "167.5 s1 sim490": r"<DATA_ROOT>\Result\260926_scdc_randinit\260926_sim490_randinit_RS_T167\job_07_260926_sim490_randinit_RS_T167.txt",
 "167.5 s2 sim502": r"<DATA_ROOT>\Result\260927_scdc_thk2b\260927_sim502_s2b_L542_U332_RS_T167\job_05_260927_sim502_s2b_L542_U332_RS_T167.txt",
 "212.5 sim403": r"<DATA_ROOT>\Result\260925_scdc_p2R0\260925_sim403_L232_SA_g50_m2000\job_10_260925_sim403_L232_SA_g50_m2000.txt",
 "257.5 sim460": r"<DATA_ROOT>\Result\260926_scdc_g50fill\260926_sim460_L258_SF_g50_f110\job_10_260926_sim460_L258_SF_g50_f110.txt",
 "302.5 sim759": r"<DATA_ROOT>\Result\261009_scdc_atomreopt\261009_sim759_reopt_302p5_S1_488fc44a\job_37_261009_sim759_reopt_302p5_S1_488fc44a.txt",
 "347.5 sim608": r"<DATA_ROOT>\Result\261002_scdc_g62axis\261002_sim608_g62_TM_L347p5_F\job_00_261002_sim608_g62_TM_L347p5_F.txt",
}
keys = None
rows = {}
for nm, p in files.items():
    d = {}
    for ln in open(p, encoding="utf-8", errors="replace"):
        if ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        d[k.strip()] = v.strip()
    rows[nm] = d
allk = sorted({k for d in rows.values() for k in d if k.startswith(("w_", "echo", "phi", "dpsi", "prod", "abs", "arg", "T_", "min_f", "lw", "pure", "fid", "r_", "rmag", "rarg", "gate", "eta", "tmin", "e_"))})
for k in allk:
    vals = [rows[n].get(k, "-") for n in files]
    print(k, "|", " | ".join(vals))
print("names:", list(files))
for nm, d in rows.items():
    print(nm, "script", d.get("script"), "pol", d.get("pol"), "rcwa_rule", d.get("rcwa_rule"), "iters", d.get("iters"))
