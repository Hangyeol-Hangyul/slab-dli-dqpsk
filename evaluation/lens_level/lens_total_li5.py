# -*- coding: utf-8 -*-
r"""lens_total_li5.py -- [TM] Li lens total of the APPROVED palette P769 (32.5 53474519, 77.5 f8176952, 122.5 b59ed0ba, 167.5 ed750b02,
212.5 78259586, 257.5 accded83, 302.5 S1 ee3609ef, 347.5 accf4825) from the sim769 cells (Result\261009_scdc_tmzone_li5) + x12_sh0 = sim766b.
Uses lens_total_li.py's own functions (read_class / betaL / aperture / pmean, imported, unmodified); only the cell paths and the palette terms change:
  terms = Li atom-level deficit_p' + purity_p' f200 of P769 from a1_def_pur_p769_result.json (-0.061521 / -0.018901), and for the reference
  sim757 both the stored Laurent terms (as published) and its Li terms (a1_def_pur_result.json P757).
Checks: every cell carries rcwa_rule liy / rcwa_ff 1, sc_shift == registration; every cell's production.atom_paths md5 list equals the expected
P769 per-slot list (build_sim769.py sim769_cells.json 'after'; x12_sh0 from the sim766b job).  Missing classes: --partial substitutes the sim757
Li cells of that class (labelled SUBSTITUTED) -- an interim estimate only.  NO solver.
python -I -B lens_total_li5.py [--partial]"""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, r"<WORK_ROOT>\loop_261008\lens_li")
import lens_total_li as L                                               # noqa: E402  (pins 10-17 at import)

R = r"<DATA_ROOT>\Result"
WD = r"<WORK_ROOT>\loop_261008\lens_li5"
PARTIAL = "--partial" in sys.argv
T769 = json.load(io.open(r"<WORK_ROOT>\loop_261008\improve_261009\tm_lens_total_a1\a1_def_pur_p769_result.json", encoding="utf-8"))
T757 = json.load(io.open(r"<WORK_ROOT>\loop_261008\improve_261009\tm_lens_total_a1\a1_def_pur_result.json", encoding="utf-8"))["P757"]
CELLS = json.load(io.open(WD + r"\sim769_cells.json", encoding="utf-8"))
EXP = {(c["M"], c["sh"]): c["after"] for c in CELLS["cells"]}
EXP[(12, 0)] = CELLS["reuse"][0]["slots"]
_md5 = {}


def m8(p):
    if p not in _md5:
        _md5[p] = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    return _md5[p]


def path769(M, sh):
    if (M, sh) == (12, 0):
        return R + r"\261009_scdc_tmzone_iso\261009_sim766_iso_x12_sh0_212_257acc"
    return R + r"\261009_scdc_tmzone_li5\261009_sim769_lensli5_x%02d_sh%d" % (M, sh)


have = {M: [sh for sh in (0, 1, 2) if os.path.isfile(os.path.join(path769(M, sh), "superramp_result.json"))] for M in L.CLASSES}
full = [M for M in L.CLASSES if len(have[M]) == 3]
missing = [(M, sh) for M in L.CLASSES for sh in (0, 1, 2) if sh not in have[M]]
if missing and not PARTIAL:
    print("REFUSE: %d of 21 cells missing: %s (use --partial for an interim estimate)" % (len(missing), " ".join("x%02d_sh%d" % c for c in missing)))
    sys.exit(3)
src = {}
for M in L.CLASSES:
    if M in full:
        src[M] = {sh: path769(M, sh) for sh in (0, 1, 2)}
        for sh in (0, 1, 2):
            j = L.jload(src[M][sh])
            L.check_li_flags("li", j, src[M][sh])
            got = [m8(p) for p in j["needle_results"]["production"]["atom_paths"]]
            assert got == EXP[(M, sh)], ("per-slot md5 mismatch", M, sh, got, EXP[(M, sh)])
    else:
        src[M] = {sh: L.cell_path("li", M, sh) for sh in (0, 1, 2)}       # sim757 Li cells (old palette) -- substitution
print("classes from sim769: %s ; SUBSTITUTED from sim757 (old palette): %s" % (" ".join("x%02d" % M for M in full) or "-",
                                                                              " ".join("x%02d" % M for M in L.CLASSES if M not in full) or "none"))
out = {}
ORIG = L.load_terms
for lab, terms, cells in (("P769 approved (Li terms)", (T769["deficit"], T769["pur_f200"]), src),
                          ("sim757 old palette (Li terms)", (T757["deficit"], T757["pur_f200"]), {M: {sh: L.cell_path("li", M, sh) for sh in (0, 1, 2)} for M in L.CLASSES}),
                          ("sim757 old palette (stored Laurent terms, as published)", None, {M: {sh: L.cell_path("li", M, sh) for sh in (0, 1, 2)} for M in L.CLASSES})):
    L.load_terms = ORIG if terms is None else (lambda t: (lambda: t))(terms)
    ev = L.evaluate_cells(cells)
    L.describe(lab, ev)
    out[lab] = ev
L.load_terms = ORIG
a, b = out["P769 approved (Li terms)"], out["sim757 old palette (Li terms)"]
print("== P769 minus sim757 (both with Li terms): TOTAL with rB0 decay %+.4f held %+.4f ; without rB0 %+.4f / %+.4f ; echo aperture %+.3f dB" % (
    a["res"]["rB0_complex"]["total"][0] - b["res"]["rB0_complex"]["total"][0], a["res"]["rB0_complex"]["total"][1] - b["res"]["rB0_complex"]["total"][1],
    a["res"]["no_rB0"]["total"][0] - b["res"]["no_rB0"]["total"][0], a["res"]["no_rB0"]["total"][1] - b["res"]["no_rB0"]["total"][1],
    a["echo_aperture"] - b["echo_aperture"]))
tag = "partial" if missing else "final"
json.dump(dict(full=full, missing=missing, results=out), io.open(WD + r"\lens_total_li5_%s.json" % tag, "w", encoding="utf-8"), indent=1,
          default=lambda o: [o.real, o.imag] if isinstance(o, complex) else str(o))
print("wrote lens_total_li5_%s.json" % tag)
