"""[261011 li_skew] OPT-IN Li factorization for a SKEWED lattice whose raster is made of SQUARE pixels.

Reached ONLY when a driver sets `obj.li_raster` (an integer 2 x 2 matrix) AND a non-default
`rcwa_rule` / `form_factor`; `rcwa.py` imports this module lazily inside that branch, so a default
job (and a half-synced code tree) never touches it.  `li_fmm.py` is NOT edited: it is hashed into
the atom2d checkpoint fingerprint, so this rule lives in its own file and borrows only the helpers.

Geometry (DESIGN_LI_SKEW.md 2.7, 3.2)
  The raster axes are the lattice vectors L1, L2 (axis 0 along L1, axis 1 along L2; fractional
  coordinates), but the PIXELS are delta_x x delta_y Cartesian rectangles with x-hat / y-hat edges.
  A raster step along axis c is p_c = U[c, 0] e_x + U[c, 1] e_y in pixel units, U unimodular
  (v5 d cell: U = ((0, 1), (1, -ds)), pixel (i, j) = (x, y) = (j, i - ds j)).  With W = U^-1 the
  Cartesian components of an order difference (dm_0, dm_1), in cycles per pixel, are

        X = G_x delta_x / 2 pi = W[0, 0] dm_0 / N_0 + W[0, 1] dm_1 / N_1
        Y = G_y delta_y / 2 pi = W[1, 0] dm_0 / N_0 + W[1, 1] dm_1 / N_1

  The exact Fourier coefficient of a square pixel is DFT[dm_0 mod N_0, dm_1 mod N_1] sinc(X) sinc(Y).

Rule `licol` (the Cartesian-edge column rule), key semantics PHYSICAL:
  'liy' : inner factorization along physical y.  The inner raster axis a is the one whose step has no
          x component (U[a, 0] == 0).  Inside a line (normal y-hat edges) E_y gets the inverse rule,
          E_x the Laurent rule; across lines (normal x-hat edges) the inputs (E_y, D_x) are continuous:
              eps_yy = [[ [[1/eps]]_a^-1 ]]_b ,    eps_xx = ( [[ [[eps]]_a^-1 ]]_b )^-1
  'lix' : the mirror: inner axis with U[a, 1] == 0 (physical x),
              eps_xx = [[ [[1/eps]]_a^-1 ]]_b ,    eps_yy = ( [[ [[eps]]_a^-1 ]]_b )^-1
  'laurent' (only with form_factor = 1): eps_xx = eps_yy = [[eps]] with the square-pixel sinc.
  eps_xy = eps_yx = 0 (ep2 stays block-diagonal); E_z: epsinv = inv([[eps]]) as in li_fmm.
  An inner axis that is parallel to neither x-hat nor y-hat is refused.
  The OUTER form factor is the SHEARED sinc of the Cartesian component normal to the inner lines
  (sinc(X) for a y-hat inner axis, sinc(Y) for an x-hat one); the INNER sinc is sinc(dm_a / N_a).

On a rectangular lattice (U = I) every expression below reduces to `li_fmm` bit for bit: the order
differences are combined as sums of integer-times-difference terms over N, and a term whose integer
coefficient is zero is dropped, so the IEEE arithmetic is the li_fmm one.

Conventions inherited from `li_fmm` (and `rcwa.py`): the in-plane block acts on (-Ey, Ex), so
ep2 = blkdiag(eps_yy, eps_xx); autograd boxes are handled with the same `_xp` / `_inv` / `_boxed`.
"""
import numpy as np

from . import backend as bd
from .li_fmm import RULES, _NESTED_MAX_BYTES, _boxed, _inv, _npa, _sinc, _xp


def _as_int_matrix(U):
    """2 x 2 integer matrix from a nested sequence / array; ValueError otherwise."""
    A = np.asarray(U)
    if A.shape != (2, 2):
        raise ValueError("li_raster must be a 2 x 2 integer matrix (got shape %s)" % (A.shape,))
    R = np.rint(A.astype(float))
    if not np.all(R == A.astype(float)):
        raise ValueError("li_raster must be integer-valued (got %r)" % (A.tolist(),))
    return R.astype(int)


def _uinv(U):
    """Integer inverse of the 2 x 2 raster matrix U; ValueError unless abs(det U) == 1."""
    U = _as_int_matrix(U)
    det = int(U[0, 0]) * int(U[1, 1]) - int(U[0, 1]) * int(U[1, 0])
    if abs(det) != 1:
        raise ValueError("li_raster must be unimodular (abs(det) = 1); got det = %d for %r"
                         % (det, U.tolist()))
    # for det = +-1 the integer inverse adj(U) / det equals adj(U) * det
    return np.array([[U[1, 1], -U[0, 1]], [-U[1, 0], U[0, 0]]], dtype=int) * det


def check_raster(U, L1, L2, N1, N2, tol=1e-9):
    """Verify that the raster (U, N1, N2) is an exact relabelling of Cartesian pixels on the lattice
    (L1, L2).  The raster step along axis c is L_c / N_c = (U[c, 0] delta_x, U[c, 1] delta_y); solve
    delta_x, delta_y from the four component equations and require (i) every equation with a zero
    integer coefficient to have a zero left-hand side, (ii) all estimates of one pixel size to agree
    to a relative `tol`, (iii) delta_x, delta_y > 0.  Returns (delta_x, delta_y); ValueError otherwise."""
    U = _as_int_matrix(U)
    _uinv(U)
    L = ((float(L1[0]), float(L1[1])), (float(L2[0]), float(L2[1])))
    N = (int(N1), int(N2))
    if N[0] < 1 or N[1] < 1:
        raise ValueError("raster sizes must be positive (got %r)" % (N,))
    steps = [(L[c][0] / N[c], L[c][1] / N[c]) for c in (0, 1)]
    scale = max(abs(v) for s in steps for v in s)
    if scale == 0.0:
        raise ValueError("degenerate lattice (all lattice vector components are zero)")
    est = ([], [])
    for c in (0, 1):
        for k in (0, 1):
            u = int(U[c, k])
            if u == 0:
                if abs(steps[c][k]) > tol * scale:
                    raise ValueError("li_raster inconsistent with the lattice: raster axis %d has a zero "
                                     "integer step along %s but the lattice step is %.6g (U=%r, L1=%r, L2=%r, "
                                     "N=%r)" % (c, "xy"[k], steps[c][k], U.tolist(), L[0], L[1], N))
            else:
                est[k].append(steps[c][k] / u)
    delta = []
    for k in (0, 1):
        if not est[k]:
            raise ValueError("li_raster has no integer step along %s" % "xy"[k])
        d0 = est[k][0]
        for e in est[k][1:]:
            if abs(e - d0) > tol * abs(d0):
                raise ValueError("li_raster inconsistent with the lattice: pixel size along %s is %.12g from "
                                 "one raster axis and %.12g from the other (U=%r, L1=%r, L2=%r, N=%r)"
                                 % ("xy"[k], d0, e, U.tolist(), L[0], L[1], N))
        if not d0 > 0.0:
            raise ValueError("li_raster gives a non-positive pixel size %.6g along %s (U=%r, L1=%r, L2=%r, "
                             "N=%r)" % (d0, "xy"[k], U.tolist(), L[0], L[1], N))
        delta.append(d0)
    return delta[0], delta[1]


def _cart_arg(Ui, comp, a, dm, dn, Na, Nb):
    """Cartesian component `comp` (0 = X, 1 = Y) of an order difference in cycles per pixel, for an
    inner axis `a` with inner / outer differences dm / dn and raster sizes Na / Nb.  Terms with a zero
    integer coefficient are dropped (so U = I gives exactly the li_fmm argument)."""
    b = 1 - a
    arg = None
    for coef, d, N in ((int(Ui[comp, a]), dm, Na), (int(Ui[comp, b]), dn, Nb)):
        if coef == 0:
            continue
        term = (coef * d) / float(N)
        arg = term if arg is None else arg + term
    if arg is None:                                       # pragma: no cover (Ui is invertible)
        raise ValueError("singular raster matrix")
    return arg


def conv_sq(f, G, ff, U):
    """Laurent matrix [[f]] over the retained orders G for SQUARE pixels on a raster whose axis steps are
    p_c = U[c, 0] e_x + U[c, 1] e_y.  With ff the pixel form factor is sinc(X) sinc(Y) (see the module
    docstring); without it this is exactly li_fmm.conv_laurent.  Returns (nG, nG)."""
    xp = _xp(f)
    N0, N1 = f.shape
    F = xp.fft.fft2(f) / (N0 * N1)
    d0 = G[:, 0][:, None] - G[:, 0][None, :]
    d1 = G[:, 1][:, None] - G[:, 1][None, :]
    C = F[d0 % N0, d1 % N1]
    if ff:
        Ui = _uinv(U)
        X = _cart_arg(Ui, 0, 0, d0, d1, N0, N1)
        Y = _cart_arg(Ui, 1, 0, d0, d1, N0, N1)
        C = C * (np.sinc(X) * np.sinc(Y))
    return C


def _inner_component(U, a):
    """0 if raster axis a steps along y-hat (inner lines along y, column boundaries normal to x-hat),
    1 if it steps along x-hat; ValueError if it is parallel to neither."""
    if int(U[a, 0]) == 0:
        return 0
    if int(U[a, 1]) == 0:
        return 1
    raise ValueError("inner raster axis %d steps along neither x-hat nor y-hat (U row %r): the column "
                     "rule needs a Cartesian inner axis" % (a, U[a].tolist()))


def nested_sq(f, G, a, ff, U):
    """[[ [[f]]_a^-1 ]]_b for square pixels: per line along raster axis `a` the 1-D Toeplitz matrix of f is
    inverted, then the other axis b is Fourier-expanded (block Toeplitz) and the retained orders gathered.
    With ff the inner sinc is sinc(dm_a / N_a) and the OUTER one is the sheared sinc of the Cartesian
    component normal to the lines (module docstring).  `a` is a RASTER axis (0 or 1).
    Returns (matrix (nG, nG), comp) with comp 0 (inner lines along y-hat) or 1 (along x-hat)."""
    U = _as_int_matrix(U)
    Ui = _uinv(U)
    a = int(a)
    if a not in (0, 1):
        raise ValueError("inner raster axis must be 0 or 1 (got %r)" % (a,))
    b = 1 - a
    comp = _inner_component(U, a)
    boxed = _boxed(f)
    xp = _npa if boxed else np
    if a == 1:
        f = xp.transpose(f)
        Gs = G[:, ::-1]
    else:
        Gs = G
    Na, Nb = f.shape
    Ma = int(np.abs(Gs[:, 0]).max())
    K = 2 * Ma + 1
    if not ff and K > Na:
        # same alias argument as li_fmm.nested: without the form factor the K x K Toeplitz matrix is built from
        # only Na distinct (wrapped) coefficients, its rank is <= Na and the inverse does not exist.
        raise ValueError("li_skew: inner order range K = %d exceeds the raster size %d along the inverted axis: "
                         "the inner Toeplitz matrix is singular without the form factor (alias). Use "
                         "rcwa_ff = 1, a finer raster or a smaller nG." % (K, Na))
    if 4.0 * K * K * Nb * 16.0 > _NESTED_MAX_BYTES:
        raise ValueError("li_skew: the inner-axis blocks (K = %d, %d lines) need about %.1f GB; refusing "
                         "(limit %.1f GB). Use a smaller nG." % (K, Nb, 4.0 * K * K * Nb * 16.0 / 1e9,
                                                                  _NESTED_MAX_BYTES / 1e9))
    mo = np.arange(-Ma, Ma + 1)
    d = mo[:, None] - mo[None, :]                       # (K, K)
    Fa = xp.fft.fft(f, axis=0) / Na                     # 1-D coefficients along a, one column per line
    T = Fa[d % Na, :]                                   # (K, K, Nb)
    if ff:
        T = T * _sinc(d, Na)[:, :, None]
    T = xp.transpose(T, (2, 0, 1))                      # (Nb, K, K)
    A = _inv(T, boxed)                                  # batched: one K x K inverse per line
    Ah = xp.fft.fft(A, axis=0) / Nb                     # Fourier along the outer axis
    dn = Gs[:, 1][:, None] - Gs[:, 1][None, :]
    ia = (Gs[:, 0] + Ma)[:, None]
    ib = (Gs[:, 0] + Ma)[None, :]
    E = Ah[dn % Nb, ia, ib]
    if ff:
        E = E * np.sinc(_cart_arg(Ui, comp, a, ia - ib, dn, Na, Nb))
    return E, comp


def epsilon_li_skew(dN, eps_grid, G, rule, ff, U, geom, anisotropic=False):
    """Drop-in for `li_fmm.epsilon_li` on a skewed lattice with a square-pixel raster.
    `dN` is unused (signature compatibility); `U` is the raster matrix (module docstring); `geom` is
    (L1, L2, Nx, Ny) of the layer (used only to validate U).  Returns (epsinv, eps2) with the same
    meaning and basis as `li_fmm.epsilon_li`."""
    if anisotropic:
        raise ValueError("rcwa_rule / form_factor are not supported for anisotropic (3-component) grids")
    if rule not in RULES:
        raise ValueError("unknown rcwa_rule %r (allowed: %s)" % (rule, ", ".join(RULES)))
    U = _as_int_matrix(U)
    L1, L2, Nx, Ny = geom
    if tuple(int(s) for s in eps_grid.shape) != (int(Nx), int(Ny)):
        raise ValueError("li_skew: grid shape %s does not match the layer raster (%d, %d)"
                         % (tuple(eps_grid.shape), int(Nx), int(Ny)))
    check_raster(U, L1, L2, Nx, Ny)
    ff = bool(ff)
    boxed = _boxed(eps_grid)
    G = np.asarray(G)
    C = conv_sq(eps_grid, G, ff, U)
    epsinv = _inv(C, boxed)
    if rule == 'laurent':
        exx = eyy = C
    else:
        k = 0 if rule == 'liy' else 1                   # liy: inner axis has no x step; lix: no y step
        cand = [ax for ax in (0, 1) if int(U[ax, k]) == 0]
        if len(cand) != 1:
            raise ValueError("li_skew: no raster axis steps along %s-hat (U=%r); rule %r is not available on "
                             "this raster" % ("yx"[k], U.tolist(), rule))
        a = cand[0]
        Ninv, comp = nested_sq(1.0 / eps_grid, G, a, ff, U)
        Neps, _ = nested_sq(eps_grid, G, a, ff, U)
        if comp == 0:                                   # inner along y-hat: E_y is normal inside a line
            eyy = Ninv
            exx = _inv(Neps, boxed)
        else:                                           # inner along x-hat: E_x is normal inside a line
            exx = Ninv
            eyy = _inv(Neps, boxed)
    # grcwa basis (-Ey, Ex): UPPER-LEFT = eps_yy, LOWER-RIGHT = eps_xx
    Z = bd.zeros_like(eyy)
    eps2 = bd.vstack((bd.hstack((eyy, Z)), bd.hstack((Z, exx))))
    return epsinv, eps2
