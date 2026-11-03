# -*- coding: utf-8 -*-
"""x48 atom order for the APPROVED palette, by the gen_screen.py rule (nearest palette phase A to off + 360 j / M, ascending j = forward,
reversed for ramp_s = -1), with the approved palette's own Li nG-1793 A values and the grid offset that maximizes the coherent sampling
efficiency |mean exp(i (A_assigned - target))|^2.  Also applied to M = 36 / 24 / 18 to compare with the stored sim769 orders (which were
inherited slot-by-slot from older palettes).  No solver, nothing written."""
import cmath
import io
import json
import math

A = [35.13, 67.77, 138.01, 180.64, 219.98, 251.94, 306.37, 363.89]          # level index 0..7 = 32.5 + 45 k (Li nG 1793, approved palette)
CELLS = json.load(io.open(r"<WORK_ROOT>\loop_261008\lens_li5\sim769_cells.json", encoding="utf-8"))
LV = {"53474519": 0, "f8176952": 1, "b59ed0ba": 2, "ed750b02": 3, "78259586": 4, "accded83": 5, "ee3609ef": 6, "accf4825": 7}
stored = {c["M"]: "".join(str(LV[h]) for h in c["after"]) for c in CELLS["cells"] if c["sh"] == 0}


def circ(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def order(M, off):
    fwd = [min(range(8), key=lambda j: circ(A[j] % 360.0, off + 360.0 * k / M)) for k in range(M)]
    return list(reversed(fwd))                                              # ramp_s = -1


def eff(M, off):
    fwd = list(reversed(order(M, off)))
    z = sum(cmath.exp(1j * math.radians(A[l] - (off + 360.0 * k / M))) for k, l in enumerate(fwd)) / M
    return abs(z) ** 2


def best(M):
    cands = [i / 20.0 for i in range(0, 7200)]
    e = [(eff(M, o), o) for o in cands]
    return max(e)


for M in (18, 24, 36, 48):
    e, o = best(M)
    seq = "".join(str(x) for x in order(M, o))
    print("M=%d best offset %.2f deg  sampling eff %.5f (%.4f dB)  order %s" % (M, o, e, 10 * math.log10(e), seq))
    if M in stored:
        print("      stored sim769 order               %s  (same: %s)" % (stored[M], seq == stored[M]))
        # efficiency of the stored order with its best offset (A of the approved palette)
        st = [int(c) for c in reversed(stored[M])]
        bz = max(abs(sum(cmath.exp(1j * math.radians(A[l] - (o2 + 360.0 * k / M))) for k, l in enumerate(st)) / M) ** 2 for o2 in [i / 10.0 for i in range(3600)])
        print("      stored order sampling eff %.5f (%.4f dB)" % (bz, 10 * math.log10(bz)))
