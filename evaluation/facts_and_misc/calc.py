import math, sys
sys.stdout.reconfigure(encoding="utf-8")
out = open(r"<WORK_ROOT>\review_r25_261022\scratch_rf_261022\calc_out.txt", "w", encoding="utf-8")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
# item 3
N, X = 8, 0.20
half = 180.0 / N
phi = X * half
s = math.sin(math.radians(phi))
P("half-step deg", half, "allowed phase err deg", phi, "sin =", f"{s:.6f}")
for k, lab in [(1.0, "incoherent (k=1)"), (10 ** (2.720 / 20), "coherent median +2.720 dB"), (1.368, "1.368"), (math.sqrt(2), "sqrt2 ceiling")]:
    a = s / k
    p = a * a / (1 + a * a)
    P(f"{lab}: k={k:.5f} a={a:.5f} p={p:.6f} floor={1-p:.6f}")
P("k from 2.849 dB (split median):", f"{10**(2.849/20):.5f}", "floor", f"{1 - (s/10**(2.849/20))**2/(1+(s/10**(2.849/20))**2):.6f}")
# check inverse: floor 0.996722 -> amplitude ratio and phase
p = 1 - 0.996722
a = math.sqrt(p / (1 - p))
P(f"p={p:.6f} a_rms(power-sum)={a:.5f} coherent peak 1.36813*a={1.36813*a:.5f} asin={math.degrees(math.asin(1.36813*a)):.4f} deg")
# shipped 0.95
p = 0.05; a = math.sqrt(p/(1-p))
P(f"0.95: a={a:.4f} asin(a)={math.degrees(math.asin(a)):.2f} deg frac half={math.degrees(math.asin(a))/half:.3f}; coherent asin(1.368a)={math.degrees(math.asin(min(1,1.368*a))):.2f}")
# item 4
lam, Lam, n = 1550.0, 1315.39, 1.444
g = lam / Lam
P(f"lam/Lam={g:.6f} theta={math.degrees(math.asin(g/n)):.4f} deg")
for m in range(-3, 4):
    kx = g * m  # in spacer after bottom metagrating, design order at kx=0 (index shifted)
    kz2 = n * n - kx * kx
    if kz2 >= 0:
        P(f"order offset {m:+d} (kx/k0={kx:+.5f}) propagating, angle {math.degrees(math.asin(abs(kx)/n)):.3f} deg")
    else:
        kap = math.sqrt(-kz2)
        d = lam / (2 * math.pi * kap)
        P(f"order offset {m:+d} (kx/k0={kx:+.5f}) evanescent |kz|/k0={kap:.4f} 1/e amp depth={d:.2f} nm; 465/d={465/d:.3f}; exp(-465/d)={math.exp(-465/d):.4f}; 200/d={200/d:.3f} exp={math.exp(-200/d):.4f}; power exp(-2*465/d)={math.exp(-2*465/d):.5f}")
# item 5 check
e0 = 8.8541878128e-12
for A in [(8e-6)**2, (150e-6)**2]:
    C = e0 * 13.9 * A / 0.3e-6
    P(f"A={A:.3e} C={C*1e15:.2f} fF f_RC={1/(2*math.pi*50*C)/1e9:.3f} GHz")
out.close()
