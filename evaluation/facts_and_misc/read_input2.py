import json, glob, os
base = r"<DATA_ROOT>\Result"
p = os.path.join(base, "261010_scdc_in2dli_judge", "261010_sim773_in2dLijudge_warm_xsec_min_rf050_W1_n2601", "input2d.json")
d = json.load(open(p))
print("sim773 judge:", p)
for k in ['status','period_nm','final_rung','G1_eta_min','G1_pass','G1_conv_delta','G1_conv_flag','G2_leak_max','G3_refl_worst_sop','G4a_echo_max','G4b_relaunch_max','G5_min_feature_nm','density_md5_L1','density_md5_L2','density_md5','eval_only','key_nG_eval','key_rcwa_rule','key_rcwa_ff','key_eval_only']:
    print(k, '=', d.get(k))
for ng in ('2601','2593','1793'):
    for q in ('nG_realised','eta_x','eta_y','leak_x','leak_y','refl_worst_sop','echo_x','echo_y','relaunch_x','relaunch_y'):
        v = d.get('ng'+ng+'_'+q)
        if v is not None:
            print('ng'+ng+'_'+q, '=', v)
# any ng key
ngs = sorted(set(k.split('_')[0] for k in d if k.startswith('ng') and k.split('_')[0][2:].isdigit()))
print('ng rungs present', ngs)
# FDTD
pf = glob.glob(os.path.join(base, "261010_fdtd_in2d_judge", "261010_sim774_fdtd_in2d_warm_xsec_min_rf050_W1*", "*.json"))
print('fdtd json', pf)
for f in pf:
    fd = json.load(open(f))
    print(os.path.basename(f), len(fd))
    for k, v in fd.items():
        if isinstance(v, (int, float, str, bool)) and any(s in k.lower() for s in ('eta','leak','refl','energy','g1','g2','g3','cells','nxy','periods','status','qc','ok','grid','exact','lane','order')):
            print('  ', k, '=', v)
