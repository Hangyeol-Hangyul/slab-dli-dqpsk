"""[261008 li_integrate] OPT-IN Fourier-factorization rules for the vendored grcwa.

NOTHING in this module is reached unless an `obj` is configured away from its defaults
(`rcwa_rule != 'laurent'`, `form_factor != 0`, or `pol_epsinv != 'conv1'`); `rcwa.py`
imports it lazily inside that branch only, so the default code path of a half-synced
tree never touches it.

Conventions (read from the vendored `rcwa.py`):
  * the in-plane block `ep2` acts on (-Ey, Ex) and returns (-Dy, Dx), so
        ep2 = blkdiag(eps_yy, eps_xx)         # UPPER-LEFT = eps_yy, LOWER-RIGHT = eps_xx
    (`Solve_FieldFourier`: `fey = -fexy[:nG]; fex = fexy[nG:]`).
  * eps_grid has shape (Nx, Ny); axis 0 <-> G[:,0] (x), axis 1 <-> G[:,1] (y).
  * pixel k is CENTRED on the sample k*P/N (same phase reference as `get_conv`), so the exact
    Fourier coefficient of a pixel-constant raster is  DFT[m mod N] * sinc(m/N)  per axis,
    valid for every integer m (it removes the raster dependence and the index wrap exactly).

Rules (Li, JOSA A 14, 2758 (1997), slice-consistent forms for a Manhattan raster):
  'laurent' : eps_xx = eps_yy = [[eps]]                         (stock rule; with form_factor=1 plus sinc)
  'liy'     : eps_xx = ( [[ [[eps]]_y^-1 ]]_x )^-1 ,  eps_yy = [[ [[1/eps]]_y^-1 ]]_x     (y-lines inner)
  'lix'     : eps_xx = [[ [[1/eps]]_x^-1 ]]_y ,       eps_yy = ( [[ [[eps]]_x^-1 ]]_y )^-1  (x-lines inner)
  E_z is tangential to every lateral wall: epsinv = inv([[eps]]).
`lix` is the exact x<->y mirror of `liy`; |liy - lix| is a factorization-error estimate.

Autograd: written with `autograd.numpy` when the input is an autograd box (so the tape flows),
plain numpy otherwise; the inversions go through the backend's `inv` (grcwa's batched VJP).
"""
import numpy as np

from . import backend as bd

try:                                    # same optional import as backend.py
    import autograd.numpy as _npa
except ImportError:                     # pragma: no cover
    _npa = None

RULES = ('laurent', 'liy', 'lix')
POL_EPSINV = ('conv1', 'inv')
# memory fail-safe of nested(): the (Nb, K, K) complex blocks and their autograd temporaries are about 4x their size
_NESTED_MAX_BYTES = 4.0e9


def _boxed(*arrs):
    for a in arrs:
        if _npa is not None and type(a).__module__.startswith('autograd'):
            return True
    return False


def _xp(*arrs):
    return _npa if _boxed(*arrs) else np


def _inv(a, boxed):
    return bd.inv(a) if boxed else np.linalg.inv(a)


def _sinc(d, N):
    """Pixel form factor sin(pi d/N)/(pi d/N) (numpy.sinc); a constant array."""
    return np.sinc(np.asarray(d, dtype=float) / float(N))


def conv_laurent(f, G, ff):
    """Toeplitz (Laurent) convolution matrix [[f]] over the retained orders G, (nG, nG)."""
    xp = _xp(f)
    Nx, Ny = f.shape
    F = xp.fft.fft2(f) / (Nx * Ny)
    dm = G[:, 0][:, None] - G[:, 0][None, :]
    dn = G[:, 1][:, None] - G[:, 1][None, :]
    C = F[dm % Nx, dn % Ny]
    if ff:
        C = C * (_sinc(dm, Nx) * _sinc(dn, Ny))
    return C


def nested(f, G, inner, ff):
    """[[ [[f]]_a^-1 ]]_b : per line, the 1-D Toeplitz matrix of f along axis a is inverted,
    then the outer axis b is Fourier-expanded (block Toeplitz) and the retained orders selected.
    inner = 0 -> a = x, b = y ;  inner = 1 -> a = y, b = x.   Returns (nG, nG)."""
    boxed = _boxed(f)
    xp = _npa if boxed else np
    if inner == 1:
        f = xp.transpose(f)
        Gs = G[:, ::-1]
    else:
        Gs = G
    Na, Nb = f.shape
    Ma = int(np.abs(Gs[:, 0]).max())
    K = 2 * Ma + 1
    if not ff and K > Na:
        # without the form factor the K x K Toeplitz matrix is built from only Na distinct (wrapped) coefficients:
        # its rank is <= Na, so the inverse does not exist. With the form factor the exact coefficients decay and
        # the matrix is invertible for any K.
        raise ValueError("li_fmm: inner order range K = %d exceeds the raster size %d along the inverted axis: the "
                         "inner Toeplitz matrix is singular without the form factor (alias). Use rcwa_ff = 1, a "
                         "finer raster or a smaller nG." % (K, Na))
    if 4.0 * K * K * Nb * 16.0 > _NESTED_MAX_BYTES:
        raise ValueError("li_fmm: the inner-axis blocks (K = %d, %d lines) need about %.1f GB; refusing (limit %.1f GB). "
                         "Use the other inner axis (liy instead of lix on a long x raster) or a smaller nG."
                         % (K, Nb, 4.0 * K * K * Nb * 16.0 / 1e9, _NESTED_MAX_BYTES / 1e9))
    mo = np.arange(-Ma, Ma + 1)
    d = mo[:, None] - mo[None, :]                       # (K, K), K = 2*Ma + 1
    Fa = xp.fft.fft(f, axis=0) / Na                     # 1-D coefficients along a, one column per line
    T = Fa[d % Na, :]                                   # (K, K, Nb)
    if ff:
        T = T * _sinc(d, Na)[:, :, None]
    T = xp.transpose(T, (2, 0, 1))                      # (Nb, K, K)
    A = _inv(T, boxed)                                  # batched: one K x K inverse per line
    Ah = xp.fft.fft(A, axis=0) / Nb                     # Fourier along b
    dn = Gs[:, 1][:, None] - Gs[:, 1][None, :]
    ia = (Gs[:, 0] + Ma)[:, None]
    ib = (Gs[:, 0] + Ma)[None, :]
    E = Ah[dn % Nb, ia, ib]
    if ff:
        E = E * _sinc(dn, Nb)
    return E


def epsilon_li(dN, eps_grid, G, rule, ff, anisotropic=False):
    """Drop-in for `Epsilon_fft(dN, eps_grid, G)` -> (epsinv, eps2).  `dN` is unused (kept for the
    signature)."""
    if anisotropic:
        raise ValueError("rcwa_rule / form_factor are not supported for anisotropic (3-component) grids")
    if rule not in RULES:
        raise ValueError("unknown rcwa_rule %r (allowed: %s)" % (rule, ", ".join(RULES)))
    ff = bool(ff)
    boxed = _boxed(eps_grid)
    G = np.asarray(G)
    C = conv_laurent(eps_grid, G, ff)
    epsinv = _inv(C, boxed)
    if rule == 'laurent':
        exx = eyy = C
    elif rule == 'liy':
        exx = _inv(nested(eps_grid, G, 1, ff), boxed)
        eyy = nested(1.0 / eps_grid, G, 1, ff)
    else:                                               # 'lix'
        exx = nested(1.0 / eps_grid, G, 0, ff)
        eyy = _inv(nested(eps_grid, G, 0, ff), boxed)
    # grcwa basis (-Ey, Ex): UPPER-LEFT = eps_yy, LOWER-RIGHT = eps_xx
    Z = bd.zeros_like(eyy)
    eps2 = bd.vstack((bd.hstack((eyy, Z)), bd.hstack((Z, exx))))
    return epsinv, eps2


def epsilon_pol(dN, eps_grid, G, pol_sigma, pol_niter, pol_epsinv, ff, rule):
    """Pol path with the opt-in E_z rule.  pol_epsinv 'inv' replaces ONLY the returned `epsinv`
    (used in kp) by inv([[eps]]); ep2 is the stock Pol block, untouched."""
    from .fft_funs import Epsilon_fft_pol, get_conv
    if rule != 'laurent':
        raise ValueError("rcwa_rule %r cannot be combined with fmm_method='pol'" % (rule,))
    if ff:
        raise ValueError("form_factor is not implemented for fmm_method='pol'")
    if pol_epsinv not in POL_EPSINV:
        raise ValueError("unknown pol_epsinv %r (allowed: %s)" % (pol_epsinv, ", ".join(POL_EPSINV)))
    epinv, ep2 = Epsilon_fft_pol(dN, eps_grid, G, pol_sigma=pol_sigma, pol_niter=pol_niter)
    if pol_epsinv == 'inv':
        epinv = bd.inv(get_conv(dN, eps_grid, G))
    return epinv, ep2
