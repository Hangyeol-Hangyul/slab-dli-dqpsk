import numpy as np
p = r"<DATA_ROOT>\Result\261011_scdc_retroli\261011_sim778_retroli_A001_liy_n257\density_scdc_retro_H525_450.npy"
a = np.load(p)
r = (a[1] > 0.5).astype(int)
px = 1315.39/512
# runs (periodic)
n = len(r)
edges = [i for i in range(n) if r[i] != r[(i-1) % n]]
print('n edges', len(edges))
runs = []
for k, e in enumerate(edges):
    e2 = edges[(k+1) % len(edges)]
    L = (e2 - e) % n or n
    runs.append((r[e], e*px, L*px))
for v, s, L in runs:
    print('Si3N4 rib' if v else 'air gap ', 'start %.1f nm width %.1f nm' % (s, L))
