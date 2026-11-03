# -*- coding: utf-8 -*-
"""Matched supercell nG_req for x48 (201-eq) and x36 (401-eq), with the stored controls, using the definitions of
<WORK_ROOT>\\te_beta_orientation_261005\\nG_scaling.py (exec of its head, unmodified; no RCWA solve, no file written)."""
import io

SRC = r"<WORK_ROOT>\te_beta_orientation_261005\nG_scaling.py"
s = io.open(SRC, encoding="utf-8").read()
ns = {"__name__": "ngs_head", "__file__": SRC}
exec(compile(s[:s.index("# stored values (controls)")], "ngs_head", "exec"), ns)
matched, rss = ns["matched"], ns["rss_gb"]
for axis, M, ref, want in (("x", 36, 201, (2304, 2303)), ("x", 36, 401, (4698, 4697)), ("x", 24, 401, (3136, 3135)), ("x", 48, 201, None), ("x", 48, 401, None)):
    r = matched(ref, axis, M)
    print("%s M=%d nG_ref=%d -> nG_req %s nG_actual %s (single %s) rss_model %.1f GB %s" % (
        axis, M, ref, r[0], r[1], r[2], rss(r[0]), ("CONTROL " + ("OK" if (r[0], r[1]) == want else "FAIL")) if want else ""))
