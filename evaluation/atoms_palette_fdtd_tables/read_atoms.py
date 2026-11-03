import json, os, hashlib
import numpy as np
R = r"<DATA_ROOT>\Result"
recs = [
 ("32.5",  r"261008_scdc_atomcand\261008_sim754_atomcand_53474519_n1793_liyff1"),
 ("77.5",  r"261008_scdc_atomli\261008_sim751_atomli_077p5_n1793_liyff1"),
 ("122.5", r"261008_scdc_atomli\261008_sim751_atomli_122p5_n1793_liyff1"),
 ("167.5", r"261008_scdc_atomli\261008_sim751_atomli_167p5_n1793_liyff1"),
 ("212.5", r"261008_scdc_atomcand\261008_sim754_atomcand_78259586_n1793_liyff1"),
 ("257.5", r"261008_scdc_atomcand\261008_sim754_atomcand_accded83_n1793_liyff1"),
 ("302.5", r"261009_scdc_atomreopt\261009_sim759_reopt_302p5_S1_488fc44a_ev1793_liyff1"),
 ("347.5", r"261008_scdc_atomli\261008_sim751_atomli_347p5_n1793_liyff1"),
]
keys = ['A_deg','carrier_eff_pct','echo_p2_db','purity','purity_coh','min_feature_meas_nm','min_feature_gate_nm','gate_ok','eff_ok','echo_ok','purity_ok','fill_frac_each','nG','nG_eval','rcwa_rule','rcwa_ff','amp_tA','absR00','argR00_deg','H_nm','l3_split','l3_gap_nm','spacer_nm','atoms_per_period','NXY','eval_only','rho3_npy','phi_c_deg','dpsi_deg','B_deg','fid_l3_m','T_atom']
for lv, rel in recs:
    p = os.path.join(R, rel, "atom2d_opt.json")
    d = json.load(open(p))
    print("=== level", lv, rel)
    for k in keys:
        if k in d: print("  ", k, "=", d[k])
    rp = d.get('rho3_npy')
    if rp and os.path.isfile(rp):
        md5 = hashlib.md5(open(rp,'rb').read()).hexdigest()
        print("   rho3 md5", md5[:8], os.path.getsize(rp))
