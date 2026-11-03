import math
c = 299792458.0
def n_sell(l):
    l2 = l*l
    return math.sqrt(1 + 0.6961663*l2/(l2-0.0684043**2) + 0.4079426*l2/(l2-0.1162414**2) + 0.8974794*l2/(l2-9.896161**2))
Lam = 1.31539  # um
G = 2*math.pi/Lam   # rad/um
t = 592.39          # um
def kz(nu):
    lam = c/nu*1e6   # um
    k0 = 2*math.pi/lam
    return math.sqrt((n_sell(lam)*k0)**2 - G**2)
nu0 = c/1.55e-6
def Phi(nu): return 2*t*kz(nu)      # rad
# group delay at nu0 by numerical derivative
h = 1e8
tau = (Phi(nu0+h)-Phi(nu0-h))/(2*h)/(2*math.pi)
print('tau_g from numerical dPhi/domega = %.6f ps (n_sell, theta from kz)' % (tau*1e12))
# 11 channels on a 100 GHz grid centred at 1550 nm; residual after removing the linear term 2 pi tau m dnu fitted at centre
dnu = 100e9
res = []
for m in range(-5, 6):
    nu = nu0 + m*dnu
    ph = Phi(nu) - Phi(nu0) - 2*math.pi*tau*m*dnu     # remove the first-order term
    res.append(math.degrees(ph))
print('11-channel residual geometric bias (deg), linear term at centre removed:', [round(x, 3) for x in res])
print('max abs %.3f deg' % max(abs(x) for x in res))
# GDD
dd = 1e9
gdd = (Phi(nu0+dd)-2*Phi(nu0)+Phi(nu0-dd))/(dd**2)/(2*math.pi)**2    # s^2 (phase second derivative w.r.t. omega)
print('GDD = %.1f fs^2' % (gdd*1e30))
# channel wavelengths
print('channel wavelengths nm:', [round(c/(nu0+m*dnu)*1e9, 3) for m in range(-5, 6)])
# quadrature alignment sensitivity: 0.1386 dB/um and 0.6933 at 5 um for w=100 um Gaussian
from math import erf, sqrt, log10
for d in (1.0, 5.0):
    r = (1+erf(sqrt(2)*d/100.0))/(1-erf(sqrt(2)*d/100.0))
    print('boundary offset %.1f um -> pair imbalance %.4f dB' % (d, 10*log10(r)))
