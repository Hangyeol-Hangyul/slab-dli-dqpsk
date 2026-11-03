import json, os, glob
R = r"<DATA_ROOT>\Result"
def cells_p769():
    out = {}
    for M in (6,9,12,15,18,24,36):
        for sh in (0,1,2):
            if (M,sh) == (12,0):
                p = R + r"\261009_scdc_tmzone_iso\261009_sim766_iso_x12_sh0_212_257acc"
            else:
                p = R + r"\261009_scdc_tmzone_li5\261009_sim769_lensli5_x%02d_sh%d" % (M, sh)
            out[(M,sh)] = p
    return out
def cells_p757():
    return {(M,sh): R + r"\261008_scdc_tmzone_li\261008_sim757_lensli_x%02d_sh%d" % (M, sh) for M in (6,9,12,15,18,24,36) for sh in (0,1,2)}
def read(paths):
    rows = {}
    for k, p in paths.items():
        js = glob.glob(os.path.join(p, "superramp_result.json"))
        if not js: print('MISSING', k, p); continue
        j = json.load(open(js[0]))
        pr = j['needle_results']['production']
        rows[k] = (pr['power_tA_target'], pr['power_tB_target'], pr['echo_p2_db'], j.get('rcwa_rule'), j.get('nG_ref'))
    return rows
for lab, paths in (("P769 approved", cells_p769()), ("sim757 previous palette", cells_p757())):
    rows = read(paths)
    A = [v[0] for v in rows.values()]; B = [v[1] for v in rows.values()]
    print(lab, 'n cells', len(rows), 'A range %.4f-%.4f mean %.4f' % (min(A), max(A), sum(A)/len(A)), '| B range %.4f-%.4f mean %.4f' % (min(B), max(B), sum(B)/len(B)))
    for M in (6,9,12,15,18,24,36):
        a = [rows[(M,s)][0] for s in (0,1,2) if (M,s) in rows]; b = [rows[(M,s)][1] for s in (0,1,2) if (M,s) in rows]
        print('   x%02d A mean %.4f B mean %.4f' % (M, sum(a)/len(a), sum(b)/len(b)), 'rules', set(v[3] for k,v in rows.items() if k[0]==M))
