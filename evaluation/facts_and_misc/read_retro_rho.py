import numpy as np, hashlib
p = r"<DATA_ROOT>\Result\261011_scdc_retroli\261011_sim778_retroli_A001_liy_n257\density_scdc_retro_H525_450.npy"
a = np.load(p)
print('shape', a.shape, 'md5', hashlib.md5(open(p,'rb').read()).hexdigest()[:8])
for i in range(a.shape[0]):
    r = a[i]
    print('layer', i+1, 'min %.3f max %.3f mean(fill) %.4f ; binary-fraction(>0.5) %.4f' % (r.min(), r.max(), r.mean(), (r>0.5).mean()))
