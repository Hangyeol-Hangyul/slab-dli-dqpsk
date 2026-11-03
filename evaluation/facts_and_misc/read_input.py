import json, math
p = r"<DATA_ROOT>\Result\261008_scdc_in2dli\261008_sim752_in2dLi_warm_xsec_min_rf050_W1\input2d.json"
d = json.load(open(p))
keys = ['status','host','N','px_nm','lam_nm','period_nm','theta_sub_deg','theta_crit_deg','density_md5','density_md5_L1','density_md5_L2',
 'mf_L1_solid_nm','mf_L1_void_nm','mf_L2_solid_nm','mf_L2_void_nm','min_feature_nm','sym_err','fill_L1','fill_L2',
 'G1_eta_min','G1_pass','G1_conv_delta','G1_conv_flag','G2_leak_max','G2_pass','G3_refl_worst_sop','G3_pass','G4a_echo_max','G4a_pass','G4b_relaunch_max','G4b_pass',
 'G5_min_feature_nm','G5_pass','gates_passed','final_rung','band_spread','band_flag','pdl_db','key_H1_nm','key_H2_nm','key_material1','key_material2','key_fill1','key_fill2','key_n_sub','key_rcwa_rule','key_rcwa_ff']
for k in keys:
    print(k, '=', d.get(k))
for ng in ('601','901','1793'):
    for q in ('nG_realised','eta_x','eta_y','leak_x','leak_y','refl_x','refl_y','refl_worst_sop','echo_x','echo_y','relaunch_x','relaunch_y','vertical_x','vertical_y','leak_away_x','leak_away_y'):
        print('ng'+ng+'_'+q, '=', d.get('ng'+ng+'_'+q))
# orders at 1793 for Ex excitation
for k in d:
    if k.startswith('ng1793_air_Ex_') or k.startswith('ng1793_air_Ey_'):
        print(k, d[k])
