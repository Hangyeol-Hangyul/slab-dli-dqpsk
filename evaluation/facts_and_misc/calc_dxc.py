import numpy as np
base = r"<PAPER_ROOT>\Figure\260825_upper_grating_beams\data"
for pol in ("TM","TE"):
    a = np.loadtxt(base + "\profile_bottom_retro_%s.dat" % pol, comments="#")
    x, ii, io = a[:,0], a[:,1], a[:,2]
    c_i = np.sum(x*ii)/np.sum(ii); c_o = np.sum(x*io)/np.sum(io)
    print(pol, 'N', len(x), 'x range', x.min(), x.max(), 'centroid inc %.4f out %.4f  shift %.4f um' % (c_i, c_o, c_o-c_i))
    m = np.abs(x) < 300
    ci2 = np.sum(x[m]*ii[m])/np.sum(ii[m]); co2 = np.sum(x[m]*io[m])/np.sum(io[m])
    print('   main-lobe only |x|<300: shift %.4f um ; fraction of outgoing power beyond |x|>=300: %.3e' % (co2-ci2, np.sum(io[~m])/np.sum(io)))
