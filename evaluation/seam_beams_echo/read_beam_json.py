# -*- coding: utf-8 -*-
"""Read the finite-beam JSONs of 261012_scdc_liseam\\beam and print the numbers quoted in facts row C9b (read-only)."""
import json
import os

ROOT = r"<DATA_ROOT>\Result\261012_scdc_liseam\beam"
for d in ("new_liy_n121", "new_liy_n61_w100", "new_liy_nG161_w100", "new_lau_n121_w100", "ctrl_old_prod"):
    p = os.path.join(ROOT, d, "beam_%s.json" % d)
    print("=====", d, os.path.isfile(p))
    j = json.load(open(p, encoding="utf-8"))

    def walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, pre + "/" + str(k))
        elif isinstance(o, list):
            if len(o) <= 8 and all(isinstance(x, (int, float)) for x in o):
                print(pre, o)
            else:
                print(pre, "list(%d)" % len(o))
        else:
            print(pre, o)
    walk(j)
