import json
base = r"<DATA_ROOT>\Result"
def load(p): return json.load(open(p))
d17 = load(base + r"\261008_scdc_in2dli\261008_sim752_in2dLi_warm_xsec_min_rf050_W1\input2d.json")
d26 = load(base + r"\261010_scdc_in2dli_judge\261010_sim773_in2dLijudge_warm_xsec_min_rf050_W1_n2601\input2d.json")
def lanes(d, ng):
    g = lambda k: d['ng%s_%s' % (ng, k)]
    ex = 'air_Ex_'; ey = 'air_Ey_'
    out = {}
    out['eta_x'] = g('eta_x'); out['eta_y'] = g('eta_y'); out['eta_avg'] = 0.5*(g('eta_x')+g('eta_y'))
    out['1-eta_x'] = 1-g('eta_x'); out['1-eta_y'] = 1-g('eta_y')
    out['refl_x'] = g('refl_x'); out['refl_y'] = g('refl_y'); out['refl_avg'] = 0.5*(g('refl_x')+g('refl_y'))
    out['refl_worst_sop'] = g('refl_worst_sop')
    out['leak_x'] = g('leak_x'); out['leak_y'] = g('leak_y'); out['leak_avg'] = 0.5*(g('leak_x')+g('leak_y'))
    out['echo_x'] = g('echo_x'); out['echo_y'] = g('echo_y')
    out['relaunch_x'] = g('relaunch_x'); out['relaunch_y'] = g('relaunch_y')
    if ('ng%s_air_Ex_T_o-1+0_p' % ng) in d:
        m1x = d['ng%s_air_Ex_T_o-1+0_p' % ng] + d['ng%s_air_Ex_T_o-1+0_s' % ng]
        m1y = d['ng%s_air_Ey_T_o+0-1_p' % ng] + d['ng%s_air_Ey_T_o+0-1_s' % ng]
        z0x = d['ng%s_air_Ex_T_o+0+0_Ex' % ng] + d['ng%s_air_Ex_T_o+0+0_Ey' % ng]
        z0y = d['ng%s_air_Ey_T_o+0+0_Ex' % ng] + d['ng%s_air_Ey_T_o+0+0_Ey' % ng]
        out['m1_x'] = m1x; out['m1_y'] = m1y; out['m1_avg'] = 0.5*(m1x+m1y)
        out['T00_x'] = z0x; out['T00_y'] = z0y; out['T00_avg'] = 0.5*(z0x+z0y)
    return out
for name, d, ng in (("Li nG1793 (realised %s)" % d17['ng1793_nG_realised'], d17, '1793'), ("Li nG2601 (realised %s)" % d26['ng2601_nG_realised'], d26, '2601')):
    print("==", name)
    for k, v in lanes(d, ng).items():
        print("  %-16s %.6e" % (k, v))
f = load(base + r"\261010_fdtd_in2d_judge\261010_sim774_fdtd_in2d_warm_xsec_min_rf050_W1_c256_fix320_ex_gpu\fdtd_input2d.json")
g = list(f['grids'].values())[0]['gates']
print("FDTD eta %.6f energy %.6f eta/energy %.6f leak %.3e refl_worst %.3e" % (g['eta_x'], g['energy_x'], g['eta_x']/g['energy_x'], g['leak_x'], g['refl_worst_sop']))
print("lane avg Li 1793 eta", 0.5*(d17['ng1793_eta_x']+d17['ng1793_eta_y']))
# band data
for lam in ('1546.00','1547.60','1550.00','1552.41','1554.02'):
    print(lam, 'eta_x %.5f eta_y %.5f' % (d17['lam%s_eta_x' % lam], d17['lam%s_eta_y' % lam]), 'band nG', d17.get('band_nG'))
print('band_spread', d17['band_spread'], d17['band_spread_x'], d17['band_spread_y'], d17['band_flag'], d17.get('band_flag_text'))
