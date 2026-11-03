import glob, os, re
names = ["sim261_csio2_400_300_f030", "sim300_w00s_warm", "sim302_c2_fid10", "sim303_c2_f10fr80",
         "sim304_c2_e125", "sim321_c2e125_f100", "sim342_c2e125f100_f90w10", "sim356_c2f90_E", "sim359_c2f90E_f100"]
roots = [r"<DATA_ROOT>\Result"]
want = ["w_pure", "w_echo2", "w_echo_cap", "echo_cap_db", "w_fid", "fid_minus", "w_slope", "w_bal", "w_ratio", "w_r2", "w_stray",
        "lams_nm", "out_l3_r_list", "filt_radius_nm", "pure_minus", "out_spacer_nm", "w_pure_m"]
for n in names:
    hits = []
    for r in roots:
        hits += glob.glob(os.path.join(r, "*", "*" + n.split("_", 1)[0] + "_" + n.split("_", 1)[1], "job_*.txt"))
    hits = [h for h in hits if "_n1793" not in h]
    print("==", n, len(hits))
    if not hits:
        continue
    h = hits[0]
    print("  ", h)
    with open(h, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    out = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("#") and "[Auto" not in s and "Design_" not in s and not re.match(r"#\s+\w+\s*=", s):
            out.append("   C: " + s[:300])
        m = re.match(r"(\w+)\s*=\s*(.*)", s)
        if m and m.group(1) in want:
            out.append("   " + m.group(1) + " = " + m.group(2)[:80])
    print("\n".join(out[:40]))
