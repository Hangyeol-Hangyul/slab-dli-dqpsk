# Independent recomputation for the advantage-claim review (261019).
# Inputs are record values only (main r21 lines 55-61, 79-83, 121, 139; SI lines 53, 78, 80, 183, 207, 578;
# figure_revision_tm_mirror_diagonal.md table 3.2; memory line 253). Output: verify_calc_out.txt (bulk), short stdout.
import os, sys, math
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_calc_out.txt")
lines = []
def p(*a):
    lines.append(" ".join(str(x) for x in a))

# ---------- record inputs ----------
lam = 1.55          # um
Lam = 1.31539       # um, period (main line 55)
n_s = 1.444         # main line 55
n_g = 1.4626        # main line 61
t = 592.39          # um
c = 299.792458      # um/ps
a = 0.30            # mm, element width (layout assumption, SI line 80)

theta = math.asin(lam / (n_s * Lam))
s_um = t * math.tan(theta)
path = 2 * t / math.cos(theta)
tau = 2 * t * n_g / (c * math.cos(theta))
p("== A1 geometry")
p("theta_deg", round(math.degrees(theta), 3), "s_um", round(s_um, 2), "2s_um", round(2 * s_um, 2),
  "path_um", round(path, 1), "tau_ps", round(tau, 4))
p("in-plane projection of path = path*sin(theta) um", round(path * math.sin(theta), 1),
  "fraction sin(theta)", round(math.sin(theta), 4))
thc = math.asin(1 / n_s)
s_min = (c * 10.0 / (2 * n_g)) * math.sin(thc)
p("critical angle deg", round(math.degrees(thc), 2), "min walk per pass at TIR limit for 10 ps (um)", round(s_min, 1),
  "2s_min", round(2 * s_min, 1))
for name, ng in (("Si", 4.2), ("SiN", 2.0), ("silica", 1.47)):
    p("waveguide length for 10 ps", name, round(c * 10.0 / ng / 1000, 3), "mm")
s = s_um / 1000.0
L = 2 * s + a
strip = 2 * L * a - a * a
box = L * L
p("L(mm)=2s+a", round(L, 4), "lane box (2s+a)*a", round(L * a, 4), "L strip", round(strip, 4), "L box", round(box, 4),
  "walk strip 2s*a per branch", round(2 * s * a, 4))

# ---------- A4 tiling ----------
def rects_band(N):
    R = []
    for k in range(N):
        x0 = y0 = k * a
        R.append((x0, y0, x0 + L, y0 + a))   # lane X
        R.append((x0, y0, x0 + a, y0 + L))   # lane Y
    return R

def hull_area(pts):
    pts = sorted(set(pts))
    def cross(o, a_, b):
        return (a_[0] - o[0]) * (b[1] - o[1]) - (a_[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0: lo.pop()
        lo.append(q)
    for q in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0: up.pop()
        up.append(q)
    h = lo[:-1] + up[:-1]
    return 0.5 * abs(sum(h[i][0] * h[(i + 1) % len(h)][1] - h[(i + 1) % len(h)][0] * h[i][1] for i in range(len(h))))

def union_area_raster(R, step=0.005):
    xs = [r[0] for r in R] + [r[2] for r in R]; ys = [r[1] for r in R] + [r[3] for r in R]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    nx = int(round((x1 - x0) / step)); ny = int(round((y1 - y0) / step))
    xc = x0 + (np.arange(nx) + 0.5) * step; yc = y0 + (np.arange(ny) + 0.5) * step
    X, Y = np.meshgrid(xc, yc)
    cnt = np.zeros_like(X, dtype=np.int16)
    for r in R:
        cnt += ((X > r[0]) & (X < r[2]) & (Y > r[1]) & (Y < r[3])).astype(np.int16)
    # lane X and lane Y of the same channel overlap in the a x a input square by construction: subtract it
    return (cnt > 0).sum() * step * step, int((cnt > 1).sum())

p("== A4 single diagonal band (record layout: channel k input at (ka, ka))")
p("N, A_authors(box+(N-1)strip), A_union, A_axis_rect, A_hull, A_rot45_rect | D for each (Gb/s/mm2, 200 Gb/s per channel)")
res = {}
for N in (1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128):
    R = rects_band(N)
    A_auth = box + (N - 1) * strip
    A_un = N * strip
    xs = [r[0] for r in R] + [r[2] for r in R]; ys = [r[1] for r in R] + [r[3] for r in R]
    A_rect = (max(xs) - min(xs)) * (max(ys) - min(ys))
    pts = [(r[0], r[1]) for r in R] + [(r[2], r[1]) for r in R] + [(r[0], r[3]) for r in R] + [(r[2], r[3]) for r in R]
    A_h = hull_area(pts)
    u = [(x + y) / math.sqrt(2) for x, y in pts]; v = [(x - y) / math.sqrt(2) for x, y in pts]
    A_rot = (max(u) - min(u)) * (max(v) - min(v))
    D = [200 * N / A for A in (A_auth, A_un, A_rect, A_h, A_rot)]
    res[N] = (A_auth, A_un, A_rect, A_h, A_rot, D)
    p(N, *[round(x, 3) for x in (A_auth, A_un, A_rect, A_h, A_rot)], "|", *[round(x, 1) for x in D])
for N in (1, 2, 4):
    ar, ov = union_area_raster(rects_band(N))
    p("raster union check N", N, "area", round(ar, 3), "expected N*strip", round(N * strip, 3), "overlap px (input squares)", ov)
# axis-rect optimum
Nopt = 2 * s / a
p("axis-aligned rectangle: D(N)=200N/(Na+2s)^2 peaks at N=2s/a=", round(Nopt, 2), "D_peak", round(200 * Nopt / (Nopt * a + 2 * s) ** 2, 1),
  "; D(N=16)", round(200 * 16 / (16 * a + 2 * s) ** 2, 1), "D(N=64)", round(200 * 64 / (64 * a + 2 * s) ** 2, 1))
p("rot45 rectangle asymptote per channel 2aL", round(2 * a * L, 4), "D_inf", round(200 / (2 * a * L), 1))

# 2-D interlocked lattice: bands translated by (L-a, -L)
e1 = np.array([a, a]); e2 = np.array([L - a, -L])
cell = abs(e1[0] * e2[1] - e1[1] * e2[0])
p("== A4 2-D interlocked lattice e1=(a,a), e2=(L-a,-L): cell area", round(cell, 4), "(= strip", round(strip, 4), ")")
# raster overlap check of a lattice patch
R = []
for i in range(-6, 7):
    for j in range(-3, 4):
        o = i * e1 + j * e2
        R.append((o[0], o[1], o[0] + L, o[1] + a)); R.append((o[0], o[1], o[0] + a, o[1] + L))
step = 0.01
xc = np.arange(-1.0, 1.0, step) + step / 2; X, Y = np.meshgrid(xc, xc)
cnt = np.zeros_like(X, dtype=np.int16)
for r in R:
    cnt += ((X > r[0]) & (X < r[2]) & (Y > r[1]) & (Y < r[3])).astype(np.int16)
# each channel's own input square counted twice (lane X and lane Y share it): correct by counting distinct-channel overlap
p("lattice raster window 2x2 mm: uncovered fraction", round(float((cnt == 0).mean()), 4),
  "pixels with count>2", int((cnt > 2).sum()), "(count 2 = own input square)")
def count_in_square(X_mm, nscan=12):
    best = 0
    for fx in np.linspace(0, 1, nscan, endpoint=False):
        for fy in np.linspace(0, 1, nscan, endpoint=False):
            off = fx * e1 + fy * e2
            M = np.array([[e1[0], e2[0]], [e1[1], e2[1]]]); Minv = np.linalg.inv(M)
            corners = np.array([[0, 0], [X_mm, 0], [0, X_mm], [X_mm, X_mm]]) - off
            ij = corners @ Minv.T
            imin, imax = int(np.floor(ij[:, 0].min())) - 2, int(np.ceil(ij[:, 0].max())) + 2
            jmin, jmax = int(np.floor(ij[:, 1].min())) - 2, int(np.ceil(ij[:, 1].max())) + 2
            ii, jj = np.meshgrid(np.arange(imin, imax + 1), np.arange(jmin, jmax + 1))
            o = off + ii[..., None] * e1 + jj[..., None] * e2
            ok = (o[..., 0] >= 0) & (o[..., 1] >= 0) & (o[..., 0] + L <= X_mm) & (o[..., 1] + L <= X_mm)
            best = max(best, int(ok.sum()))
    return best
for Xd in (2.5, 5.0, 10.0, 20.0, 40.0):
    Nn = count_in_square(Xd)
    p("square die", Xd, "mm: channels", Nn, "D", round(200 * Nn / Xd ** 2, 1) if Nn else 0)

p("== A4 symbol-rate scaling of our strip (a fixed 0.30 mm; t and s scale with 1/Rs), D_inf = 2*Rs/strip")
for Rs in (100, 50, 25, 20, 15, 10):
    sR = s * 100.0 / Rs
    LR = 2 * sR + a
    stR = 2 * LR * a - a * a
    p("Rs", Rs, "GBd: s_mm", round(sR, 3), "strip_mm2", round(stR, 3), "box_mm2", round(LR * LR, 2),
      "D_inf", round(2 * Rs / stR, 2), "D(N=1 box)", round(2 * Rs / (LR * LR), 2))
p("literature (metrics.csv): Klamkin 100 Gb/s PDM @25 GBd 3.0 mm2 -> 33.3; Faralli 50 @25 GBd 6.3 -> 7.94; Velha 30 @15 GBd 0.7 -> 42.9; Aamer 10 Gb/s DPSK @10 GBd 0.11 -> 90.9")
p("area per receiver ratio (ours L strip / theirs): Klamkin", round(strip / 3.0, 3), "Velha", round(strip / 0.7, 3), "Aamer", round(strip / 0.11, 2))

# ---------- A2 ----------
p("== A2 PDFS / PDL")
G2 = 2 * 360.0 / (Lam * 1000)  # deg per nm
FSR = 100.0
p("2G deg/nm", round(G2, 5), "GHz/nm", round(G2 / 360 * FSR, 5))
for nm_, val in (("Lize 1%", 1.0), ("Nasu 0.78%", 0.78), ("Li 0.61%", 0.61), ("Doerr 4.6%", 4.6), ("Optoplex 3deg", 3 / 360 * 100), ("Kawashima 0.291GHz", 0.291)):
    p("tolerance dx-dy nm for", nm_, round(val / (G2 / 360 * FSR), 2))
p("6 deg bias tolerance -> nm", round(6 / G2, 2), "; 10 deg ->", round(10 / G2, 2), "; 20 deg ->", round(20 / G2, 2))
p1, p2 = 1.51e-4, 1.83e-4
T = 0.9991
ph1, ph2 = math.degrees(math.asin(math.sqrt(p1))), math.degrees(math.asin(math.sqrt(p2)))
p("report method asin(sqrt p) deg", round(ph1, 3), round(ph2, 3), "sum", round(ph1 + ph2, 3), "-> %FSR", round((ph1 + ph2) / 3.6, 3))
tt = math.sqrt(T)
dev = tt * (math.sqrt(p1) + math.sqrt(p2))
base = T + 0.5 * (p1 + p2)
pdl = 10 * math.log10((base + dev) / (base - dev))
p("co-polarized coherent leak: SOP-dependent summed gain +-", round(dev, 5), "-> PDL bound dB", round(pdl, 4))
for lab, q1, q2 in (("2601 harm", 1.40e-4, 1.58e-4), ("FDTD", 5.65e-5, 5.65e-5)):
    d2 = tt * (math.sqrt(q1) + math.sqrt(q2)); b2 = T + 0.5 * (q1 + q2)
    p("  same bound with", lab, round(10 * math.log10((b2 + d2) / (b2 - d2)), 4), "dB")
eta = 0.8713
pdfs_x = p2 / eta  # rad, cross-polarized incoherent leak, interference-amplitude ratio <= 1/eta
p("cross-polarized leak: bias error <= p*S_max/S_worst rad =", round(pdfs_x, 6), "=", round(math.degrees(pdfs_x), 4), "deg;",
  "two branches opposite:", round(2 * math.degrees(pdfs_x), 4), "deg =", round(2 * math.degrees(pdfs_x) / 3.6, 4), "%FSR")
p("PDL proxy Tx/Ty 0.00189 dB (metrics.csv) vs coherent-leak bound", round(pdl, 3), "dB; chip reports Wang <0.3, Optoplex <0.2, Lize 0.05")

# ---------- A5 ----------
p("== A5 quadrant bias")
p("Lambda/8 nm", round(Lam * 1000 / 8, 2), "; 90 deg / (2G) nm", round(90 / G2, 2))
k0 = 2 * math.pi / lam
dphidt = math.degrees(2 * k0 * n_s * math.cos(theta)) / 1000  # deg per nm
p("thickness coefficient 2 k0 n cos(theta) deg/nm", round(dphidt, 4), "(record 0.3877); 1 um ->", round(dphidt * 1000, 1), "deg")
p("common-control span needed mod 90 deg (+-45): K at 5.6 deg/K", round(45 / 5.6, 2), "; GHz at 3.6 deg/GHz", round(45 / 3.6, 2))
qsep_mm = 2 * 39.894 / 1000
p("quadrant-to-quadrant bias error per 1 nm/mm thickness wedge (centroids 79.8 um apart) deg", round(dphidt * qsep_mm, 4))
coef = t * k0 / (n_s * math.cos(theta) ** 3)
p("input tilt along grating vector: first-order DLI phase change 0 (Littrow return reverses k_x);",
  "second order", round(math.degrees(coef * 1e-6), 3), "deg at 1 mrad,", round(math.degrees(coef * 1e-8), 5), "deg at 0.1 mrad")
p("intra-layer placement: 1 nm quadrant-shift error ->", round(G2, 4), "deg quadrature error")

# ---------- A7 ----------
p("== A7 loss")
Smax = (4 / 9) ** 1.5
arch1 = -10 * math.log10(Smax / 0.5); arch2 = -10 * math.log10(0.9188)
dli = -10 * math.log10(0.8713 / 0.9188)
p("Smax", round(Smax, 5), "arch reciprocal", round(arch1, 3), "echo", round(arch2, 3), "sum", round(arch1 + arch2, 3),
  "DLI section", round(dli, 3), "impl 0.004+0.230+0.829 =", round(0.004 + 0.230 + 0.829, 3), "total", round(arch1 + arch2 + 1.063, 3))
p("Doerr 2006 per-port fiber-to-fiber 4.5-5.8 dB -> above 3.0103:", round(4.5 - 3.0103, 3), round(5.8 - 3.0103, 3),
  "; Doerr chip star coupler excess 0.1-1.1 dB (verdict.md, Doerr p.172)")
p("ours total - best reported:", round(3.703 - (4.5 - 3.0103), 3), "dB; ours impl - Doerr chip 0.1/1.1:", round(1.063 - 0.1, 3), round(1.063 - 1.1, 3))
p("Si DLI thermal drift for 10 ps (dn/dT 1.8e-4, L 0.714 mm) deg/K", round(360 * 713.8 * 1.8e-4 / lam, 1),
  "; silica waveguide (1e-5, 2.039 mm)", round(360 * 2039.4 * 1e-5 / lam, 2))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", OUT, len(lines), "lines")
