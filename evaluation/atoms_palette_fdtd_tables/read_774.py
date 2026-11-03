# -*- coding: utf-8 -*-
r"""read_774.py -- table of the landed sim774 FDTD c256 exact judges, read from the driver's own `gates` block
(G1 eta = +1+0 p component, G2 leak, G3 refl worst SOP); eta/energy = G1 divided by energy_x. Li values: read_752_773.py.
python -I -B read_774.py"""
import glob
import io
import json
import os

F = r"<DATA_ROOT>\Result\261010_fdtd_in2d_judge"
rows = []
for d in sorted(glob.glob(os.path.join(F, "261010_sim774_fdtd_in2d_*"))):
    f = os.path.join(d, "fdtd_input2d.json")
    tag = os.path.basename(d).replace("261010_sim774_fdtd_in2d_", "").replace("_c256_fix320_ex_gpu", "")
    if not os.path.isfile(f):
        rows.append((tag, None))
        continue
    a = json.load(io.open(f, encoding="utf-8"))
    g = list(a["grids"].values())[0]
    gt = g["gates"]                                                         # driver's own gate block (eta = +1+0 p component)
    rows.append((tag, dict(eta=gt["G1_eta_min"], en=gt["energy_x"], eta_n=gt["G1_eta_min"] / gt["energy_x"], leak=gt["G2_leak_max"],
                           R=gt["G3_refl_worst_sop"], fin=a.get("finished"))))
for tag, r in rows:
    if r is None:
        print("%-28s  (not landed)" % tag)
        continue
    print("%-28s G1 eta %.4f  energy %.4f  eta/energy %.4f  G2 leak %.1e  G3 refl worst SOP %.1e  %s" %
          (tag, r["eta"], r["en"], r["eta_n"], r["leak"], r["R"], r["fin"]))
