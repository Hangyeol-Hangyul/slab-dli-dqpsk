# -*- coding: utf-8 -*-
"""sim788 readout (carrier + atom; input2d pending its selftest): per k, the gate quantities and their change from the k0 control.
Also checks k0 against the stored lossless records (sim768 carrier, sim754 atom)."""
import glob
import io
import json
import math
import os

R = r"<DATA_ROOT>\Result"
B = R + r"\261013_scdc_asik"
KS = ["k0", "k1e-5", "k1e-4", "k1e-3", "k1e-2", "k1e-1"]
CEIL = (4.0 / 9.0) ** 1.5


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def carrier(path):
    d = load(path)
    return {"eff_pct": 100 * d["signal_worst"] / CEIL, "echo_db": d["echo_worst_p2_db"], "purity_worst": d.get("purity_worst"),
            "purity_m_worst": d.get("purity_m_worst", d.get("fid_m_worst"))}


def atom(path):
    d = load(path)
    keys = ("carrier_eff_pct", "echo_db", "echo_p2_db", "amp_tA", "A_deg", "purity", "purity_m", "absR00", "argR00_deg")
    return {k: d[k] for k in keys if k in d and isinstance(d[k], (int, float))}


rows = {}
for k in KS:
    c = glob.glob(B + r"\261013_sim788_asik_carrier_%s\scdc_extract_*.json" % k)
    a = glob.glob(B + r"\261013_sim788_asik_atom325_%s\atom2d_opt.json" % k)
    rows[k] = (carrier(c[0]) if c else None, atom(a[0]) if a else None)
ref_c = carrier(glob.glob(R + r"\261009_scdc_carli\261009_sim768_carli_525f74f3_liy\scdc_extract_*.json")[0])
ref_a = atom(R + r"\261008_scdc_atomcand\261008_sim754_atomcand_53474519_n601_liyff1\atom2d_opt.json")
print("stored lossless carrier sim768:", ref_c)
print("stored lossless atom    sim754:", ref_a)
c0, a0 = rows["k0"]
print("k0 carrier == sim768:", c0 == ref_c, "| k0 atom == sim754:", a0 == ref_a)
print("\nCARRIER (bottom metagrating 525f74f3, both a-Si layers lossy, five terminations, nG_eval 257)")
print("%-6s %10s %8s %10s %10s %9s" % ("k", "eff %", "echo dB", "purity", "purity_m", "d eff dB"))
for k in KS:
    c = rows[k][0]
    if c:
        print("%-6s %10.3f %8.3f %10s %10s %+9.4f" % (k, c["eff_pct"], c["echo_db"], "%.6f" % c["purity_worst"] if c["purity_worst"] is not None else "-",
                                                    "%.6f" % c["purity_m_worst"] if c["purity_m_worst"] is not None else "-", 10 * math.log10(c["eff_pct"] / c0["eff_pct"])))
print("\nATOM 53474519 (32.5 deg; L3 a-Si lossy, carrier under it lossless; nG 601)")
print("keys:", sorted(a0))
for k in KS:
    a = rows[k][1]
    if a:
        e = a.get("carrier_eff_pct")
        print("%-6s %s | d eff %+.4f dB" % (k, "  ".join("%s=%.5g" % kv for kv in sorted(a.items())), 10 * math.log10(e / a0["carrier_eff_pct"]) if e else float("nan")))
