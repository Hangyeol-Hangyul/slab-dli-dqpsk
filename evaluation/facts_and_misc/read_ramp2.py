import json, os, glob
R = r"<DATA_ROOT>\Result"
def p769(M, sh):
    if (M,sh) == (12,0): return R + r"\261009_scdc_tmzone_iso\261009_sim766_iso_x12_sh0_212_257acc"
    return R + r"\261009_scdc_tmzone_li5\261009_sim769_lensli5_x%02d_sh%d" % (M, sh)
def p757(M, sh): return R + r"\261008_scdc_tmzone_li\261008_sim757_lensli_x%02d_sh%d" % (M, sh)
for lab, fn in (("P769 approved", p769), ("sim757 previous", p757)):
    A=[];B=[]; cls={}
    for M in (6,9,12,15,18,24,36):
        a=[];b=[]
        for sh in (0,1,2):
            j = json.load(open(os.path.join(fn(M,sh), "superramp_result.json")))
            pr = j['needle_results']['production']
            a.append(pr['T_total_A']); b.append(pr['T_total_B'])
        cls[M]=(sum(a)/3, sum(b)/3); A+=a; B+=b
    print(lab, 'A %.4f-%.4f mean %.4f | B %.4f-%.4f mean %.4f' % (min(A),max(A),sum(A)/21,min(B),max(B),sum(B)/21))
    print('   ', ' '.join('x%02d %.4f/%.4f' % (M, v[0], v[1]) for M, v in cls.items()))
