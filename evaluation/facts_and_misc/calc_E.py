import numpy as np
from scipy.optimize import minimize
# Bottom-arm bound: maximise signal sqrt(ep R_go em) of a lossless reciprocal 4-port with echo <= -20.5 dB,
# pass budgets ep+R_go+dp+bf = 1, em+R_go+dm+echo = 1 (R0m = R_go by reciprocity), and the column-orthogonality
# (polygon) inequalities on the four overlap terms sqrt(ep em), sqrt(dp dm), sqrt(R_go echo), sqrt(bf R_go).
echo_max = 10 ** (-20.5 / 10)
ceil = (4 / 9) ** 1.5
print('echo limit linear %.6f ; ceiling %.6f ; 0.85 gate in signal units %.6f' % (echo_max, ceil, 0.85 * ceil))

def obj(x):
    return -np.sqrt(max(x[0] * x[1] * x[4], 0))

def terms(x):
    ep, R, dp, bf, em, dm, ec = x
    return np.array([np.sqrt(max(ep * em, 0)), np.sqrt(max(dp * dm, 0)), np.sqrt(max(R * ec, 0)), np.sqrt(max(bf * R, 0))])

cons = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1},
        {'type': 'eq', 'fun': lambda x: x[4] + x[1] + x[5] + x[6] - 1},
        {'type': 'ineq', 'fun': lambda x: echo_max - x[6]}]
for i in range(4):
    cons.append({'type': 'ineq', 'fun': (lambda x, i=i: (terms(x).sum() - terms(x)[i]) - terms(x)[i])})
best = None
rng = np.random.default_rng(3)
for t in range(80):
    x0 = np.array([rng.uniform(.2, .4), rng.uniform(.3, .5), 0, rng.uniform(.1, .3), rng.uniform(.3, .6), 0, rng.uniform(0, echo_max)])
    r = minimize(obj, x0, constraints=cons, bounds=[(0, 1)] * 7, method='SLSQP', options={'maxiter': 800, 'ftol': 1e-15})
    if r.success and (best is None or r.fun < best.fun):
        best = r
x = best.x
print('ep %.6f Rgo %.6f dp %.6f bf %.6f em %.6f dm %.6f echo %.6f' % tuple(x))
print('signal %.5f = %.3f %% of the ceiling' % (-best.fun, 100 * -best.fun / ceil))
print('percent of 100 %% input: pass 1 reference arm ep %.2f ; pass 2 arm R_go x em %.2f ; extracted %.2f ; back to fibre bf + R_go^2 %.2f ; echo path R_go x echo %.3f' % (100 * x[0], 100 * x[1] * x[4], 100 * (x[0] + x[1] * x[4]), 100 * (x[3] + x[1] ** 2), 100 * x[1] * x[6]))
print('Figure 5a style percentages: reference input bf %.2f / R_go %.2f / ep %.2f ; delayed input echo %.2f / R0m %.2f / em %.2f' % (100 * x[3], 100 * x[1], 100 * x[0], 100 * x[6], 100 * x[1], 100 * x[4]))
