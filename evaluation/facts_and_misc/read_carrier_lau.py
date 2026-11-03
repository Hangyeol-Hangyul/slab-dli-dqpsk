import json, glob, os
CEIL=(4/9)**1.5
for tag in ("lau","liy"):
    d = r"<DATA_ROOT>\Result\261009_scdc_carli\261009_sim768_carli_525f74f3_" + tag
    j = json.load(open(glob.glob(os.path.join(d,"scdc_extract_*.json"))[0]))
    print(tag, 'eff worst %.3f %%' % (100*j['signal_worst']/CEIL), 'echo %.3f' % j['echo_worst_p2_db'], 'purity %.6f' % j['purity_worst'], 'pur_m %.6f' % j['purity_m_worst'], 'rcwa_rule', j.get('rcwa_rule'), 'nG', j.get('nG'), 'min_feat', j.get('min_feature_nm'))
    print('   center n1708 signal %.5f -> %.3f %%; echo %.3f; eta+ %.4f R_go %.4f eta- %.4f' % (j['n1708_signal'], 100*j['n1708_signal']/CEIL, j['n1708_echo_worst_p2_db'], j['n1708_eta_plus'], j['n1708_R_go'], j['n1708_eta_minus']))
j = json.load(open(glob.glob(r"<DATA_ROOT>\Result\261009_scdc_carli\261009_sim768_carli_525f74f3_liy\scdc_extract_*.json")[0]))
c = 'n1708_'
print('center pass1: eta+ %.5f R0 %.5f R(-2) %.5f dump %.5f (stray - R-2)' % (j[c+'eta_plus'], j[c+'R_go'], j[c+'back_p1_Rm2'], j[c+'stray_R']-j[c+'back_p1_Rm2']))
print('center pass2: eta- %.5f R0m %.5f echo R(+2) %.5f dump(1-sum) %.5f' % (j[c+'eta_minus'], j[c+'back_R0_rev'], j[c+'echo_worst_p2'], 1-j[c+'eta_minus']-j[c+'back_R0_rev']-j[c+'echo_worst_p2']))
print('center: P_A %.4f P_B %.4f extracted %.4f return %.4f' % (j[c+'arm_ref'], j[c+'arm_del'], j[c+'arm_ref']+j[c+'arm_del'], j[c+'back_total']))
print('worst signal termination: by n_tag')
for pre in ['n1708','n2329','n3445','n1616','n1768']:
    print(pre, 'ns=', j['out_l3_ns_used'][['n1708','n2329','n3445','n1616','n1768'].index(pre)], '%.3f %%' % (100*j[pre+'_signal']/CEIL), 'echo', round(j[pre+'_echo_worst_p2_db'],3))
