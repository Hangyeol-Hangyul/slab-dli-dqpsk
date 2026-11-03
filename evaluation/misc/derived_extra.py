import numpy as np
LAM, LAMBDA, F, C = 1.55, 1.31539, 200.0, 39.894
h = 0.05
xs = np.arange(h / 2, 150, h)
X, Y = np.meshgrid(xs, xs, indexing="ij")
I = np.exp(-2.0 * (X ** 2 + Y ** 2) / 100.0 ** 2)
rho = np.hypot(X - C, Y - C)
P = LAM * np.sqrt(rho ** 2 + F ** 2) / np.maximum(rho, 1e-9)
tot = I.sum()
lines = []
for d in (0.689, 2.338):
    s = 100.0 * I[P < 8 * d].sum() / tot
    lines.append("delta %.3f um -> P < %.3f um = %.2f L : %.2f %% of lens power" % (d, 8 * d, 8 * d / LAMBDA, s))
print("\n".join(lines))
open(r"<WORK_ROOT>\paper_scdc_261011_v3\review\r7_memo_261013\designflow\derived_extra_out.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
