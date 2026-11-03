import json, re
p = r"<DATA_ROOT>\Result\261009_scdc_carli\261009_sim768_carli_525f74f3_liy\scdc_extract_H400_300.json"
d = json.load(open(p))
print('nG', d.get('nG'), 'period', d.get('period_nm'), 'rcwa_rule', d.get('rcwa_rule'), d.get('key_rcwa_rule'))
tops = [k for k in d if not k.startswith('1550.0_') and not k.startswith('n')]
print(len(tops), tops[:120])
# n-tag prefixes
pref = sorted(set(k.split('_')[0] for k in d if re.match(r'^n\d+_', k)))
print('prefixes', pref)
for k in d:
    if re.match(r'^n1708_', k) and not any(s in k for s in ('_sub_', '_air_')):
        print(k, d[k])
