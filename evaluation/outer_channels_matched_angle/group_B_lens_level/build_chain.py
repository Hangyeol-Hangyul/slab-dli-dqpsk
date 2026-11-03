import io, os, sys, shutil, glob
NEWJ = sys.argv[1]            # new lensband_edge_results json
V2J = sys.argv[2]             # new lensband_eval_v2_result.json
A = r"<WORK_ROOT>\fill_A_261102\kx_chain"
B = r"<WORK_ROOT>\fill_B_261103\work\chain"
log = []
for sub in ("c2", "fig7"):
    os.makedirs(os.path.join(B, sub), exist_ok=True)
# scripts + static inputs (outputs are regenerated)
for sub, names in (("c2", ["c2_extract.py", "c2_assemble.py", "c2_s23.py", "C2_tables_section_raw.md"]),
                   ("fig7", ["g2_device_chain.py", "g10_arm_ratio.py", "g3_eye.py", "g7_eye_traces.py", "eye_model_g.py", "g1_c0_reference.json"])):
    for n in names:
        shutil.copy2(os.path.join(A, sub, n), os.path.join(B, sub, n))
os.makedirs(os.path.join(B, "c2", "lensband_eval_copy"), exist_ok=True)
for f in glob.glob(os.path.join(A, "c2", "lensband_eval_copy", "*")):
    if os.path.isfile(f):
        shutil.copy2(f, os.path.join(B, "c2", "lensband_eval_copy", os.path.basename(f)))
# NEW lens read-out v2 json (edge cells sim900, atoms/tokens sim899) replaces the copied fixed-angle one
shutil.copy2(V2J, os.path.join(B, "c2", "lensband_eval_copy", "lensband_eval_v2_result.json"))
log.append("c2/lensband_eval_copy/lensband_eval_v2_result.json <- %s (read by c2_s23.py:10 LENS and g2 V2 terms_all_channels)" % V2J)
# g5 + noPD copies
shutil.copy2(r"<WORK_ROOT>\fig7_data_261017\g5_fig7c_data.py", os.path.join(B, "fig7", "g5_fig7c_data.py"))
shutil.copy2(r"<WORK_ROOT>\fig7_layout_261018\c78\make_c7_noPD.py", os.path.join(B, "fig7", "make_c7_noPD.py"))


def sub(path, old, new, cnt=1):
    s = io.open(path, encoding="utf-8").read()
    n = s.count(old)
    assert n == cnt, (path, old, n)
    io.open(path, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    log.append("%s: %s -> %s (x%d)" % (os.path.relpath(path, B), old[:100], new[:100], n))


# generic fill_A -> fill_B path swap
for f in glob.glob(os.path.join(B, "*", "*.py")):
    s = io.open(f, encoding="utf-8").read()
    n = s.count(A)
    if n:
        sub(f, A, B, n)
G2 = os.path.join(B, "fig7", "g2_device_chain.py")
sub(G2, r"<WORK_ROOT>\lensband_edge_261017\lensband_edge_results_261017_1226_r1.json", NEWJ)
G5 = os.path.join(B, "fig7", "g5_fig7c_data.py")
sub(G5, 'HERE = r"<WORK_ROOT>\\fig7_data_261017"', 'HERE = r"%s\\fig7"' % B)
sub(G5, 'C2T = r"<WORK_ROOT>\\paper_scdc_261011_v3\\review\\tbd_analysis_261014\\c2_tables.json"', 'C2T = r"%s\\c2\\c2_tables.json"' % B)
C7 = os.path.join(B, "fig7", "make_c7_noPD.py")
sub(C7, 'SRC = r"<WORK_ROOT>\\fig7_data_261017\\device_chain_11ch.csv"', 'SRC = r"%s\\fig7\\device_chain_11ch.csv"' % B)
sub(C7, 'OUT = r"<WORK_ROOT>\\fig7_layout_261018\\c78\\c7_total_efficiency_noPD.csv"', 'OUT = r"%s\\fig7\\c7_total_efficiency_noPD.csv"' % B)
io.open(os.path.join(os.path.dirname(B), "CHANGES_chain.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
print("\n".join(log))
