import math
c = 299792458.0
lam = 1.55  # um
def n_sell(l):
    l2 = l*l
    return math.sqrt(1 + 0.6961663*l2/(l2-0.0684043**2) + 0.4079426*l2/(l2-0.1162414**2) + 0.8974794*l2/(l2-9.896161**2))
n = n_sell(lam)
h = 1e-4
dn = (n_sell(lam+h)-n_sell(lam-h))/(2*h)
ng = n - lam*dn
print('Sellmeier n %.6f dn/dlam %.6f /um n_g %.6f' % (n, dn, ng))
for Lam_nm, nsub in [(1315.39, 1.444), (1315.36, 1.444), (1315.39, n), (1315.36, n)]:
    s = lam*1000/(Lam_nm*nsub)
    th = math.degrees(math.asin(s))
    print('Lambda %.2f n %.6f sin %.6f theta %.5f deg  lam/Lam %.6f' % (Lam_nm, nsub, s, th, lam*1000/Lam_nm))
# period that gives 54.69 with n=1.444: 
for th in (54.69, 54.690050):
    print('Lambda for theta', th, 'n=1.444 ->', 1550/(1.444*math.sin(math.radians(th))), ' n sell ->', 1550/(n*math.sin(math.radians(th))))
t = 592.39
for (Lam_nm, nsub) in [(1315.39, 1.444), (1315.39, n)]:
    th = math.asin(lam*1000/(Lam_nm*nsub))
    s = t*math.tan(th)
    path = 2*t/math.cos(th)
    tau = path*1e-6*ng/c
    tau_phase = path*1e-6*nsub/c
    print('Lambda %.2f nsub %.6f: theta %.5f s %.4f um 2s %.4f um  path 2t/cos %.4f um  tau_g %.5f ps  FSR %.4f GHz  tau_phaseindex %.4f ps' % (Lam_nm, nsub, math.degrees(th), s, 2*s, path, tau*1e12, 1/tau/1e9, tau_phase*1e12))
th_c = math.degrees(math.asin(1/1.444)); print('critical angle', th_c)
print('delay-lambda spacing for 100 GHz: nm', (1.55e-6)**2/c*100e9*1e9)
print('0.5474 deg/nm = 2G*1nm: ', 360/(1315.39/2), 'deg/nm  (360/(Lam/2))')
print('dPhi/dt deg/nm', )
th = math.asin(lam*1000/(1315.39*1.444))
kz = 2*math.pi/(lam)*1.444*math.cos(th)
print('kz rad/um', kz, 'dPhi/dt (2 t kz) deg per nm', math.degrees(2*kz*1e-3))
