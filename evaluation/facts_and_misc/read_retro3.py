import json, glob, os
base = r"<DATA_ROOT>\Result\261011_scdc_retroli"
for nm in ("A001_liy_b11_n257", "A001_liy_n257", "A001_lau_b11_n257"):
    d = os.path.join(base, "261011_sim778_retroli_"+nm)
    j = json.load(open(glob.glob(os.path.join(d,'scdc_retro_*.json'))[0]))
    print(nm, 'lams', j.get('lams_nm'), 'spread', j.get('retro_R_m2_spread'), 'R_m2', j.get('retro_R_m2'), 'slope TM', j.get('slope_retro_TM'), 'arg', j.get('arg_retro_R_m2'))
    # per-wavelength R_m2
    ws = sorted(set(k.split('_')[0] for k in j if k.endswith('_sub_p_TM_R_m2')), key=float)
    vals = [(w, j[w+'_sub_p_TM_R_m2'], j.get(w+'_sub_p_TM_argR_m2')) for w in ws]
    print('   per-wavelength R(-2):', [(w, round(v,7), round(a,3) if a is not None else None) for w,v,a in vals])
    print('   min over wavelengths %.7f' % min(v for _,v,_ in vals))
    for line in open(glob.glob(os.path.join(d,'job_*.txt'))[0], encoding='utf-8', errors='replace'):
        if line.startswith('lams_nm') or line.startswith('# sim778'): print('   job:', line.strip()[:300])
