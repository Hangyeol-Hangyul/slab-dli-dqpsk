# -*- coding: utf-8 -*-
"""Read-only: per-atom numbers of palette P769 straight from each Li nG-1793 record and its delay-arm run; plus the fill colours used in the
Fig 6 slide (261011 deck) for the (a) carrier cross-section. Output atom_numbers_out.txt."""
import collections
import io
import json
import os
import re

from pptx import Presentation
from pptx.util import Emu

HERE = os.path.dirname(os.path.abspath(__file__))
R = r"<DATA_ROOT>\Result"
PM = R + r"\261009_scdc_tmpmli"
LV = [(32.5, "53474519", R + r"\261008_scdc_atomcand\261008_sim754_atomcand_53474519_n1793_liyff1", PM + r"\261009_sim761_pmli_TM_08_c032p5_53474519_n1793"),
      (77.5, "f8176952", R + r"\261008_scdc_atomli\261008_sim751_atomli_077p5_n1793_liyff1", PM + r"\261009_sim761_pmli_TM_01_ad077p5_f8176952_n1793"),
      (122.5, "b59ed0ba", R + r"\261008_scdc_atomli\261008_sim751_atomli_122p5_n1793_liyff1", PM + r"\261009_sim761_pmli_TM_02_ad122p5_b59ed0ba_n1793"),
      (167.5, "ed750b02", R + r"\261008_scdc_atomli\261008_sim751_atomli_167p5_n1793_liyff1", PM + r"\261009_sim761_pmli_TM_03_ad167p5_ed750b02_n1793"),
      (212.5, "78259586", R + r"\261008_scdc_atomcand\261008_sim754_atomcand_78259586_n1793_liyff1", PM + r"\261009_sim764_pmli_TM_c212p5_78259586_n1793_t1200"),
      (257.5, "accded83", R + r"\261008_scdc_atomcand\261008_sim754_atomcand_accded83_n1793_liyff1", PM + r"\261009_sim764_pmli_TM_c257p5_accded83_n1793_t1200"),
      (302.5, "ee3609ef", R + r"\261009_scdc_atomreopt\261009_sim759_reopt_302p5_S1_488fc44a_ev1793_liyff1", PM + r"\261009_sim762_pmli_TM_00_c302p5_S1_488fc44a_ee3609ef_n1793_t1150"),
      (347.5, "accf4825", R + r"\261008_scdc_atomli\261008_sim751_atomli_347p5_n1793_liyff1", PM + r"\261009_sim761_pmli_TM_07_ad347p5_accf4825_n1793")]
out = ["level | md5 | A_deg mod 360 | eff % | echo dB | purity ref | purity delay | lw nm | |t_A| (amp_tA) | key names used"]
for lvl, m8, rd, pm in LV:
    r = json.load(io.open(os.path.join(rd, "atom2d_opt.json"), encoding="utf-8"))
    echo = r.get("carrier_echo_db", r.get("echo_p2_db"))
    lw = r.get("min_feature_meas_nm")
    d = json.load(io.open(os.path.join(pm, "scdc_extract_H400_300.json"), encoding="utf-8"))
    amp = r.get("amp_tA", r.get("amp_t", None))
    out.append("%5.1f %s A %.2f  eff %.2f  echo %.2f  pur %.6f  delay %.6f  lw %s  amp %s  [echo key %s]" % (
        lvl, m8, r["A_deg"] % 360, r["carrier_eff_pct"], echo, r["purity"], d["purity_m_worst"],
        "inf" if lw in (None, float("inf")) else "%.2f" % lw, amp, "carrier_echo_db" if "carrier_echo_db" in r else "echo_p2_db"))
# phase steps between consecutive levels (cyclic)
A = [json.load(io.open(os.path.join(rd, "atom2d_opt.json"), encoding="utf-8"))["A_deg"] % 360 for _, _, rd, _ in LV]
steps = [((A[(i + 1) % 8] - A[i]) % 360) for i in range(8)]
out.append("cyclic phase steps (deg): " + ", ".join("%.2f" % s for s in steps) + " | deviation from level: " +
           ", ".join("%+.2f" % (((a - l + 180) % 360) - 180) for a, (l, _, _, _) in zip(A, LV)))
prs = Presentation(r"<PAPER_ROOT>\261011_Figure.pptx")
sl = prs.slides[5]
c = collections.Counter()


def walk(shapes):
    for s in shapes:
        try:
            if s.fill.type is not None and s.fill.fore_color.type is not None:
                x, y = Emu(s.left).mm, Emu(s.top).mm
                if x < 60 and 10 < y < 55:
                    c[str(s.fill.fore_color.rgb)] += 1
        except Exception:  # noqa: BLE001
            pass
        if s.shape_type == 6:
            walk(s.shapes)


walk(sl.shapes)
out.append("Fig 6 (a) region fill colours: " + repr(c.most_common(10)))
io.open(os.path.join(HERE, "atom_numbers_out.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("ok")
