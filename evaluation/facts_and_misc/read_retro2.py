import json, glob, os, re
base = r"<DATA_ROOT>\Result\261011_scdc_retroli"
d = os.path.join(base, "261011_sim778_retroli_A001_liy_n257")
j = json.load(open(glob.glob(os.path.join(d, "scdc_retro_*.json"))[0]))
for k,v in j.items():
    if re.match(r'^\d', k): continue
    if isinstance(v,(list,dict)) and len(str(v))>200: print(k, '= <long>', str(v)[:150]); continue
    print(k,'=',v)
# 1550 keys
for k in j:
    if k.startswith('1550.0_sub_p_TM_') and ('R_m2' in k or 'R_m1' in k or 'R_0' in k or 'T_m1' in k or 'argR_m2' in k or 'absR_m2' in k):
        print(k, j[k])
print(open(os.path.join(d,'Variable_result_information.txt')).read())
for line in open(glob.glob(os.path.join(d,'job_*.txt'))[0], encoding='utf-8', errors='replace'):
    if line.startswith('#'): print(line.strip()[:300])
    elif '=' in line and not line.startswith('#'): print(line.strip()[:200])
