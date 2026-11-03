import math
from math import erf, sqrt, log10
# quadrant tail beyond 150 um for intensity exp(-2 r^2 / w^2), w = 100 um, quarter [0, inf)^2
w = 100.0; a = 150.0
inside = (erf(sqrt(2)*a/w))**2
print('quadrant: inside [0,150]^2 fraction of quarter %.5f ; tail %.4f %%' % (inside, 100*(1-inside)))
c8 = 0.824264
for lab, t in (("x decay", -0.5470182455060397), ("x held", -0.5581150950176934), ("gate", -0.65), ("y decay", -0.5616955172328308), ("y held", -0.5731416023662669)):
    print(lab, 'abs capture %.3f %%' % (100*c8*10**(t/10)))
print('continuous vs ideal-8: %.4f dB' % (10*log10(0.824264/0.867544)))
# palette phase gaps
ph = {32.5:35.12860778544088, 77.5:67.76822906852532, 122.5:138.00727141630796, 167.5:-179.361926211024+360, 212.5:-140.02472500971888+360, 257.5:-108.06392624071611+360, 302.5:-53.63290130744263+360, 347.5:3.892604592832921}
vals = sorted(ph.values())
gaps = [(vals[(i+1)%8]-vals[i])%360 for i in range(8)]
print('sorted', [round(v,2) for v in vals]); print('gaps', [round(g,2) for g in gaps], 'max', max(gaps))
tgt = {32.5:32.5,77.5:77.5,122.5:122.5,167.5:167.5,212.5:212.5,257.5:257.5,302.5:302.5,347.5:347.5}
print('meas - target', {k: round((ph[k]-k+180)%360-180,2) for k in ph})
dpsi = [-161.9147358884901,-158.28478902153006,-161.9895789074804,-160.0219474981199,-161.55408681312895,-161.0215570257834,-163.41266527332505,-160.51616461455444]
print('dpsi range', max(dpsi)-min(dpsi))
print('lane tolerance: 20 deg ->', 20/0.547366180372361, 'nm; 10 deg ->', 10/0.547366180372361, 'nm; 90 deg ->', 90/0.547366180372361, 'nm; Lambda/8', 1315.39/8)
print('Lambda/4', 1315.39/4, 'Lambda/16', 1315.39/16, '3L/16', 3*1315.39/16, 'Lambda/2', 1315.39/2)
print('footprint s,2s', 836.3544, 2*836.3544)
