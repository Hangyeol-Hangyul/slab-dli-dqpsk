"""
rcwa_multi.py -- 1D RCWA with an ARBITRARY binary profile inside the period.

Extension of  <DATA_ROOT>\\single_work\\260708_ML_test\\rcwa_1d.py
(single centred ridge, cross-validated against dfmod: worst order-resolved DE
diff 5.6e-4).  Everything about the solver core is unchanged; only the
permittivity Fourier expansion is generalised, plus two corrections that the
single-ridge case did not expose.

Why this extension is needed
----------------------------
The self-referenced PSK receiver needs two gratings that are NOT zero-order
(out/in-coupling requires a propagating higher order, so a per-atom phase LUT
is structurally invalid).  The valid library element is therefore a whole
SUPERCELL characterised at the real incidence angle.  Blazing -- i.e. breaking
the +1 / -1 degeneracy -- requires an ASYMMETRIC arrangement of several ridges
inside one period, which the single-centred-ridge version cannot express.
A single etch depth is retained on purpose: one binary etch is what the process
delivers, and an asymmetric multi-ridge binary cell is already enough to blaze.

Two corrections vs the single-ridge version
-------------------------------------------
1. Toeplitz construction.  For a centred ridge the Fourier coefficients are
   real, so toeplitz(col, col) happened to be right.  A general profile has
   COMPLEX coefficients and eps(-m) = conj(eps(m)), so the matrix must be
   built as toeplitz(c, conj(c)).
2. Lateral origin is now explicit.  Ridges are given in u = x/period measured
   from the LEFT edge of the cell.  This matters because a lateral shift du
   multiplies order m by exp(-2j*pi*m*du): the phase of every m != 0 order is
   origin-dependent (m = 0 is not).
   -> arg(t_TM) - arg(t_TE) IS origin-independent (both pick up the same
      factor), so the phase-matching objective is well posed either way.
   -> the absolute arg(t_+1) used when assembling a chirped lens is NOT, so the
      assembly step must carry each cell's physical position.

Conventions (inherited, unchanged)
----------------------------------
    exp(+j*w*t); forward wave exp(-j*kz*z)
    kx_m = k0*n_inc*sin(theta) + m*2*pi/period
    TE: E along the grooves.  TM: H along the grooves (Li inverse rule).
    t, r referenced at z = H and z = 0 respectively.
"""

import numpy as np
from scipy.linalg import toeplitz, eig, solve

__all__ = ["eps_fourier", "solve_profile", "centred_ridge", "propagating_orders"]


# --------------------------------------------------------------------------- #
#  permittivity Fourier expansion for an arbitrary binary profile
# --------------------------------------------------------------------------- #
def eps_fourier(ridges, e_hi, e_lo, N):
    """Fourier coefficients eps_m, m = 0 .. 2N, of a binary eps(x).

    ridges : iterable of (u0, u1) with 0 <= u0 < u1 <= 1, in period units,
             measured from the LEFT edge of the cell.  Segments must not
             overlap (not checked -- overlapping input is a caller error).
    Returns complex array of length 2N+1.
    """
    ridges = [(float(u0), float(u1)) for u0, u1 in ridges]
    fill = sum(u1 - u0 for u0, u1 in ridges)
    if not (-1e-12 <= fill <= 1 + 1e-12):
        raise ValueError(f"total fill {fill:.6f} outside [0, 1]")

    c = np.empty(2 * N + 1, dtype=complex)
    c[0] = e_lo + (e_hi - e_lo) * fill
    m = np.arange(1, 2 * N + 1)
    acc = np.zeros(2 * N, dtype=complex)
    for u0, u1 in ridges:
        acc += (np.exp(-2j * np.pi * m * u1) - np.exp(-2j * np.pi * m * u0)) / (-2j * np.pi * m)
    c[1:] = (e_hi - e_lo) * acc
    return c


def _toeplitz_eps(c):
    """Matrix A[i,j] = eps_{i-j} from c = [eps_0 .. eps_2N], using eps_-m = conj(eps_m)."""
    return toeplitz(c, np.conj(c))


def centred_ridge(fill):
    """Ridge list for one ridge centred in the cell -- the classic single-ridge case."""
    return [(0.5 * (1 - fill), 0.5 * (1 + fill))]


def propagating_orders(wavelength, period, theta_inc, n_inc, n_side, N):
    """Orders m whose kx_m propagates in a medium of index n_side."""
    k0 = 2 * np.pi / wavelength
    m = np.arange(-N, N + 1)
    kx = k0 * n_inc * np.sin(theta_inc) + m * 2 * np.pi / period
    return m[np.abs(kx) < k0 * n_side]


# --------------------------------------------------------------------------- #
#  solver
# --------------------------------------------------------------------------- #
def solve_profile(wavelength, period, H, ridges, theta_inc, pol,
                  n_inc=3.4777, n_out=1.0, n_ridge=None, n_groove=1.0, N=25):
    """Solve one supercell.  theta_inc [rad] is measured in the incidence medium.

    ridges : see eps_fourier.  Pass centred_ridge(fill) to reproduce the
             single-ridge behaviour of rcwa_1d.solve_grating.

    Returns dict: complex r/t per order (m = -N..N), DEr/DEt, energy balance.
    """
    if n_ridge is None:
        n_ridge = n_inc
    k0 = 2 * np.pi / wavelength
    M = 2 * N + 1
    i0 = N
    m = np.arange(-N, N + 1)

    kx = k0 * n_inc * np.sin(theta_inc) + m * 2 * np.pi / period
    Kx = np.diag(kx / k0)
    kzI = _kz(k0, n_inc, kx)
    kzIII = _kz(k0, n_out, kx)

    E = _toeplitz_eps(eps_fourier(ridges, n_ridge ** 2, n_groove ** 2, N))
    Ea = _toeplitz_eps(eps_fourier(ridges, 1.0 / n_ridge ** 2, 1.0 / n_groove ** 2, N))
    I = np.eye(M, dtype=complex)

    if pol.upper() == "TE":
        A = Kx @ Kx - E
        YI = np.diag(kzI / k0)
        YIII = np.diag(kzIII / k0)
        lam, W = eig(A)
        Q = _branch(np.sqrt(lam))
        V = W @ np.diag(Q)
    elif pol.upper() == "TM":
        A = solve(Ea, Kx @ solve(E, Kx) - I)      # Li inverse rule
        YI = np.diag(kzI / (k0 * n_inc ** 2))
        YIII = np.diag(kzIII / (k0 * n_out ** 2))
        lam, W = eig(A)
        Q = _branch(np.sqrt(lam))
        V = Ea @ W @ np.diag(Q)
    else:
        raise ValueError("pol must be 'TE' or 'TM'")

    X = np.diag(np.exp(-k0 * Q * H))
    delta = np.zeros(M, dtype=complex)
    delta[i0] = 1.0

    Z = np.zeros((M, M), dtype=complex)
    sys_A = np.block([
        [-I,        W,      W @ X,  Z],
        [1j * YI,   V,     -V @ X,  Z],
        [Z,         W @ X,  W,     -I],
        [Z,         V @ X, -V,     -1j * YIII],
    ])
    rhs = np.concatenate([delta, 1j * (YI @ delta), np.zeros(M), np.zeros(M)])
    sol = solve(sys_A, rhs)
    r, t = sol[:M], sol[3 * M:]

    if pol.upper() == "TE":
        norm = kzI[i0].real
        DEr = np.abs(r) ** 2 * kzI.real / norm
        DEt = np.abs(t) ** 2 * kzIII.real / norm
    else:
        norm = (kzI[i0] / n_inc ** 2).real
        DEr = np.abs(r) ** 2 * (kzI / n_inc ** 2).real / norm
        DEt = np.abs(t) ** 2 * (kzIII / n_out ** 2).real / norm

    return {
        "m": m, "i0": i0, "r": r, "t": t,
        "DEr": DEr, "DEt": DEt,
        "R_total": DEr.sum(), "T_total": DEt.sum(),
        "energy": DEr.sum() + DEt.sum(),
    }


def _kz(k0, n, kx):
    """kz with engineering-convention branch: propagating Re>0, evanescent -j|kz|."""
    return np.conj(np.sqrt((k0 * n) ** 2 - kx ** 2 + 0j))


def _branch(Q):
    """Consistent square-root branch for the layer eigenvalues."""
    Q = np.where(Q.real < 0, -Q, Q)
    return np.where((np.abs(Q.real) < 1e-12) & (Q.imag < 0), -Q, Q)
