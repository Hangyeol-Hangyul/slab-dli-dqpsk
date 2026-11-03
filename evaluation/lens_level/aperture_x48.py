# -*- coding: utf-8 -*-
"""Echo aperture of the APPROVED palette with the >12 Lambda zone taken from the measured x48 class (power mean of its 3 registrations)
instead of the -20.5 dB gate value. Same zone weights / class map / power mean as echo_lens_261008.py (lens_total_li.py copies). Read-only."""
import io
import json
import math
import os
import sys

sys.path.insert(0, r"<WORK_ROOT>\loop_261008\lens_li")
import lens_total_li as L                                                     # noqa: E402

R = r"<DATA_ROOT>\Result"
fin = json.load(io.open(r"<WORK_ROOT>\loop_261008\lens_li5\lens_total_li5_final.json", encoding="utf-8"))["results"]["P769 approved (Li terms)"]
ec = dict(fin["echo_class"])
x48 = [L.jload(R + r"\261009_scdc_tmzone_li6\261009_sim771_lensli6_x48_sh%d" % sh)["needle_results"]["production"]["echo_p2_db"] for sh in (0, 1, 2)]
ZW = {"2-3L": 2.666, "3-4L": 7.462, "4-6L": 25.6, "6-12L": 47.9}
ZW[">12L"] = 100.0 - sum(ZW.values())
ZM = {"2-3L": ["x06"], "3-4L": ["x09"], "4-6L": ["x12", "x15"], "6-12L": ["x18", "x24", "x36"]}


def ap(e12):
    amp = 0.0
    for z, w in ZW.items():
        e = e12 if z == ">12L" else L.pmean([ec[c] for c in ZM[z]])
        amp += (w / 100.0) * math.sqrt(10 ** (e / 10.0))
    return 10 * math.log10(amp * amp)


print("x48 cells echo %s -> power mean %.3f dB" % (" ".join("%.3f" % v for v in x48), L.pmean(x48)))
print("aperture with >12L at the gate (-20.5 dB, as published): %.3f dB  (stored final %.3f)" % (ap(-20.5), fin["echo_aperture"]))
print("aperture with >12L = measured x48:                       %.3f dB  (gate -20.5, margin %+.3f)" % (ap(L.pmean(x48)), -20.5 - ap(L.pmean(x48))))
