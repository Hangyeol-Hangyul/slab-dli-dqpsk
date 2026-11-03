import json
p = r"<DATA_ROOT>\Result\261009_scdc_carli\261009_sim768_carli_525f74f3_liy\scdc_extract_H400_300.json"
d = json.load(open(p))
for k in ['signal_worst','signal','signal_ceiling','echo_worst_p2','echo_worst_p2_db','purity_worst','purity_m_worst','fid_worst','fidm_worst','min_feature_1_nm','min_feature_2_nm','min_feature_nm','min_feature_nm_job','out_l3_ns_used','out_l3_nm','out_spacer_nm','H1_nm','H2_nm','material1','material2','fill1','fill2','termination_fold','folded_min_keys','nG','rcwa_rule','rcwa_ff','ceiling','back_total','eta_plus','eta_minus','R_go','arm_ref','arm_del']:
    print(k, '=', d.get(k))
for pre in ['n1708','n2329','n3445','n1616','n1768']:
    print(pre, 'signal', d.get(pre+'_signal'), 'echo_db', d.get(pre+'_echo_worst_p2_db'), 'purity', d.get(pre+'_purity_worst'), 'purity_m', d.get(pre+'_purity_m_worst'), 'nG', d.get(pre+'_nG'))
