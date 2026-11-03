import json, glob, os
CEIL = (4/9)**1.5
OUTF = r"<DATA_ROOT>\Result\261009_scdc_carli"
TOK = ["032p5","077p5","122p5","167p5","212p5","257acc","302p5","347p5"]
rows = {}
for t in TOK:
    d = os.path.join(OUTF, "261009_sim768b_tok_525f74f3_%s" % t)
    js = glob.glob(os.path.join(d, "scdc_extract_*.json"))
    if not js:
        print(t, "NOT FOUND", d); continue
    j = json.load(open(js[0]))
    # job file for token
    jf = glob.glob(os.path.join(d, "job_*.txt"))
    tok = ""
    out_l3 = ""
    nGe = ""
    if jf:
        for line in open(jf[0], encoding='utf-8', errors='replace'):
            ls = line.strip()
            if ls.startswith('out_l3_r_list') or ls.startswith('out_l3_nm') or ls.startswith('nG_eval') or ls.startswith('rcwa_rule') or ls.startswith('rcwa_ff') or ls.startswith('out_l3_2layer') or ls.startswith('seed_npy'):
                tok += ls + " | "
    eff = 100*j['signal_worst']/CEIL
    rows[t] = (eff, j['echo_worst_p2_db'], j['purity_worst'], j['purity_m_worst'], j.get('fid_worst'))
    print(t, "eff %.3f echo %.3f pur %.6f pur_m %.6f nG %s" % (eff, j['echo_worst_p2_db'], j['purity_worst'], j['purity_m_worst'], j.get('nG')), '|', tok[:300])
print("min eff", min((v[0],k) for k,v in rows.items()))
print("max echo", max((v[1],k) for k,v in rows.items()))
print("min purity", min((v[2],k) for k,v in rows.items()))
print("min purity_m", min((v[3],k) for k,v in rows.items()))
