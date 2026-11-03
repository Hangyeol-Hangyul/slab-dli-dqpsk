import json, glob, os
base = r"<DATA_ROOT>\Result\261011_scdc_retroli"
for name in ["A001_lau_n81e161","A001_lau_n257","A001_liy_n41","A001_liy_n81","A001_liy_n129","A001_liy_n161","A001_liy_n201","A001_liy_n257","A001_lix_n257","A001_liyff0_n257","A001_lauff1_n257","A001_liy_b11_n257","B010_lau_n81e161","B010_liy_n257"]:
    d = os.path.join(base, "261011_sim778_retroli_"+name)
    js = glob.glob(os.path.join(d, "scdc_retro_*.json"))
    if not js: print(name, 'NOT FOUND'); continue
    j = json.load(open(js[0]))
    keys = ['retro_R_m2','retro_R_m2_band_min','retro_R_m2_worst','retro_leak_T_m1','retro_arg_R_m2','retro_arg_R_m2_deg','nG','rcwa_rule','rcwa_ff','H1_nm','H2_nm','period_nm','retro_slope_deg_per_nm']
    out = {k:j[k] for k in keys if k in j}
    print(name, out)
j = json.load(open(glob.glob(os.path.join(base, "261011_sim778_retroli_A001_liy_n257", "scdc_retro_*.json"))[0]))
print([k for k in j.keys()][:200])
