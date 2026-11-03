import json, math, glob, os
R = r"<DATA_ROOT>\Result"
def pm(v): return 10 * math.log10(sum(10 ** (x / 10) for x in v) / len(v))
e201 = []; e401 = []
for sh in (0, 1, 2):
    p201 = R + r"\261009_scdc_tmzone_li5\261009_sim769_lensli5_x36_sh%d\superramp_result.json" % sh
    c401 = glob.glob(R + r"\261009_scdc_tmzone_li6\*x36e401_sh%d*\superramp_result.json" % sh)
    j = json.load(open(p201)); pr = j["needle_results"]["production"]
    e201.append(pr["echo_p2_db"])
    print("x36 201-eq sh%d echo %.3f nG_req %s" % (sh, pr["echo_p2_db"], j.get("nG_req")))
    if c401:
        j4 = json.load(open(c401[0])); p4 = j4["needle_results"]["production"]
        e401.append(p4["echo_p2_db"])
        print("x36 401-eq sh%d echo %.3f nG_req %s (%s)" % (sh, p4["echo_p2_db"], j4.get("nG_req"), os.path.basename(os.path.dirname(c401[0]))))
print("class echo 201-eq %.3f dB ; 401-eq %.3f dB" % (pm(e201), pm(e401) if e401 else float("nan")))
