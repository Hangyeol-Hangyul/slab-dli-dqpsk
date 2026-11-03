# -*- coding: utf-8 -*-
r"""[260914] C2v symmetry reduction of grcwa's layer eigenproblem.

ISOLATED. Nothing here touches the simulation data folder. Import it beside a COPY
of grcwa and monkeypatch; the live vendored tree is not modified.

THE OPERATOR, read verbatim from grcwa/rcwa.py:

    Jk = vstack(diag(-ky), diag(kx))          2nG x nG
    kp = w^2 I - Jk @ epinv @ Jk^T            2nG x 2nG   (MakeKPMatrix)
    k  = vstack(diag(kx), diag(ky))
    M  = ep2 @ kp - k @ k^T                   2nG x 2nG   (SolveLayerEigensystem)
    q, phi = eig(M);  q = sqrt(q) with Im q >= 0

with ep2 = blkdiag(T, T), epinv = inv(T), T = Toeplitz of eps-hat.

WHY IT BLOCK-DIAGONALISES.  Let R_x permute G: (R_x v)(m,n) = v(-m,n).  Then
diag(kx) R_x = -R_x diag(kx), diag(ky) R_x = +R_x diag(ky), and for an x-even
structure T commutes with R_x.  Put Sx = blkdiag(sA R_x, sB R_x).  Conjugating
M by Sx leaves every block invariant IFF sA*sB = -1: the (1,2) block of k k^T
is kx*ky, which R_x negates, so the -1 is needed to undo it; the same works out
for kp's off-diagonal blocks.  Measured parity (co-pol = SECOND half, EVEN in m
and n; cross-pol = FIRST half, ODD in both) fixes the branch:

    Sx = blkdiag(-R_x, +R_x)      Sy = blkdiag(-R_y, +R_y)

and the excited sector is the (+1,+1) joint eigenspace.  Sector dimensions come
out as (M+1)(N+1) + MN = 2MN+M+N+1, exactly Zhou & Li 2004 section 4.

THE HALF-PIXEL TRAP.  The raster grid is u = (arange(N)+0.5)/N - 0.5, so the
atom is centred at sample index (N-1)/2, not 0.  In the DFT's own coordinates
the structure is DISPLACED by delta = (N-1)/(2N) of a period, which multiplies
eps-hat(G) by exp(-2i pi G.delta).  Hence

    T_measured = D T_centred D^-1 ,   D = diag(exp(-2i pi G.delta))

and T_measured does NOT commute with R.  Fix: use the phase-weighted projector
P' = D P.  D is unitary, so P'^H M P' = P^H T_centred-based M P and the
restriction is exact.  Eigenvalues are untouched (similarity), and eigenvectors
come back already in the measured frame, so nothing downstream changes.

TWO STAGES, and this file implements the first.

  1a  Solve ALL FOUR sectors and reassemble a complete 2nG mode set.
      Drop-in for SolveLayerEigensystem; the S-matrix chain is untouched.
      Cost 4*(nG/2)^3 against (2nG)^3 = 16x.
  1b  Carry only the (+1,+1) sector through GetSMatrix/SolveExterior.
      64x, but the chain needs sector-restricted variants.

Starting at 1a because its acceptance test is unambiguous -- the reassembled
mode set must reproduce the stock answer to machine precision -- and because
16x already lifts the reachable nG by 2.5x at fixed cost.
"""
import numpy as np


def _index(G):
    """map (m,n) -> position in the retained G list."""
    return {(int(g[0]), int(g[1])): i for i, g in enumerate(np.asarray(G))}


def orbits(G):
    """Group the retained G into mirror orbits.

    Returns a list of (rep, members) where members is the dict
    {(sx, sy): index} for sx, sy in {+1, -1}, sx/sy meaning the sign applied to
    m / n.  Closure under both mirrors is asserted -- the circular truncation
    has it, but a future truncation scheme might not, and silently dropping a
    partner would corrupt the projector rather than fail.
    """
    Gv = np.asarray(G)
    idx = _index(Gv)
    seen, out = set(), []
    for i, g in enumerate(Gv):
        m, n = int(g[0]), int(g[1])
        if (m, n) in seen:
            continue
        mem = {}
        for sm in (1, -1):
            for sn in (1, -1):
                key = (sm * m, sn * n)
                if key not in idx:
                    raise ValueError(
                        "G set is not closed under the mirrors: %r missing "
                        "for %r -- symmetry reduction refused" % (key, (m, n)))
                mem[(sm, sn)] = idx[key]
                seen.add(key)
        out.append(((abs(m), abs(n)), mem))
    return out


def projectors(G, nxy=None, half_pixel=True):
    """The four sector projectors, orthonormal, 2nG x d each.

    Sector labels are (ex, ey) = the eigenvalues of (Sx, Sy).  Column blocks:
    the SECOND half (co-pol) carries parity (ex, ey) under (m -> -m, n -> -n);
    the FIRST half carries (-ex, -ey), because Sx = blkdiag(-R, +R).
    """
    Gv = np.asarray(G)
    nG = len(Gv)
    ors = orbits(Gv)
    if half_pixel:
        if nxy is None:
            raise ValueError("half_pixel=True needs nxy")
        d = (nxy - 1) / (2.0 * nxy)
        ph = np.exp(-2j * np.pi * d * (Gv[:, 0] + Gv[:, 1]))
    else:
        ph = np.ones(nG, dtype=complex)

    out = {}
    for ex in (1, -1):
        for ey in (1, -1):
            cols = []
            for half, (px, py) in ((nG, (ex, ey)), (0, (-ex, -ey))):
                for _rep, mem in ors:
                    v = np.zeros(2 * nG, dtype=complex)
                    for (sm, sn), i in mem.items():
                        s = (px if sm < 0 else 1) * (py if sn < 0 else 1)
                        v[half + i] += s * ph[i]
                    nrm = np.linalg.norm(v)
                    if nrm > 1e-9:          # orbit killed by the parity
                        cols.append(v / nrm)
            out[(ex, ey)] = np.array(cols).T if cols else np.zeros((2 * nG, 0))
    return out


def commutation_residual(M, P):
    """||(I - Pi) M Pi|| / ||M||, Pi = P P^H.  Exactly 0 for a valid sector."""
    MP = M @ P
    return float(np.linalg.norm(MP - P @ (P.conj().T @ MP))
                 / max(np.linalg.norm(M), 1e-300))


def eig_blocked(M, Ps, tol=1e-9, strict=True):
    """Stage 1a: eig each sector, reassemble a COMPLETE 2nG mode set.

    Returns (q2, phi) with the same meaning as grcwa's bd.eig(M) output, i.e.
    BEFORE the sqrt and branch-cut choice that SolveLayerEigensystem applies.
    """
    n2 = M.shape[0]
    dims = sum(P.shape[1] for P in Ps.values())
    if dims != n2:
        raise ValueError("sector dims %d != %d -- projector set incomplete"
                         % (dims, n2))
    qs, vs = [], []
    for key in sorted(Ps):
        P = Ps[key]
        if P.shape[1] == 0:
            continue
        if strict:
            r = commutation_residual(M, P)
            if r > tol:
                raise ValueError(
                    "sector %r does not commute with M (residual %.3e > %.1e) "
                    "-- the structure is not C2v-symmetric about the cell "
                    "centre; symmetry reduction REFUSED" % (key, r, tol))
        Mr = P.conj().T @ M @ P
        w, V = np.linalg.eig(Mr)
        qs.append(w)
        vs.append(P @ V)
    return np.concatenate(qs), np.hstack(vs)


# ---------------------------------------------------------------------------
# Sparse fold -- Zhou & Li 2004 eqs (19)(20)(27)(28)(29) in operational form.
#
# Each projector column has at most FOUR nonzeros (the mirror orbit), so
# P^H M P must never be formed as a dense triple product: that costs O(n^2 d)
# and, measured, it DOMINATED the eigensolve it was meant to save (4.0x
# instead of the theoretical 16x at nG 385).  Folding by index arithmetic is
# O(n d) and is exactly what the paper prescribes -- their eq (27) is the
# four-term combination below, written out.
# ---------------------------------------------------------------------------


def to_sparse(P, tol=1e-12):
    """(rows, vals) of shape (d, 4); zero-padded. Refuses >4 nonzeros."""
    n2, d = P.shape
    R = np.zeros((d, 4), dtype=np.intp)
    V = np.zeros((d, 4), dtype=complex)
    for j in range(d):
        nz = np.nonzero(np.abs(P[:, j]) > tol)[0]
        if len(nz) > 4:
            raise ValueError("column %d has %d nonzeros, expected <= 4" % (j, len(nz)))
        R[j, :len(nz)] = nz
        V[j, :len(nz)] = P[nz, j]
    return R, V


def fold(M, P, sp=None):
    """P^H M P using the 4-nonzero structure. Identical to the dense form."""
    R, V = to_sparse(P) if sp is None else sp
    MP = np.einsum("iab,ab->ia", M[:, R], V, optimize=True)   # (2nG, d)
    return np.einsum("ja,jak->jk", V.conj(), MP[R], optimize=True)


def unfold(sp, Vv, n2):
    """P @ Vv using the same 4-nonzero structure -- a scatter, not a matmul.

    Measured: the dense back-projection cost as much as the eigensolve it was
    saving ((2nG) x d x d is the same order as (2nG)^3 / 4), which is why the
    first fast version topped out near 8x instead of 16x.
    """
    R, V = sp
    out = np.zeros((n2, Vv.shape[1]), dtype=complex)
    for a in range(R.shape[1]):
        np.add.at(out, R[:, a], V[:, a, None] * Vv)
    return out


def eig_blocked_fast(M, Ps, tol=1e-9, strict=True, sparse=None):
    n2 = M.shape[0]
    if sum(P.shape[1] for P in Ps.values()) != n2:
        raise ValueError("projector set incomplete")
    qs, vs = [], []
    for key in sorted(Ps):
        P = Ps[key]
        if P.shape[1] == 0:
            continue
        if strict:
            r = commutation_residual(M, P)
            if r > tol:
                raise ValueError(
                    "sector %r does not commute with M (residual %.3e > %.1e) "
                    "-- the structure is not C2v-symmetric about the cell "
                    "centre; symmetry reduction REFUSED" % (key, r, tol))
        sp = sparse[key] if sparse is not None else to_sparse(P)
        Mr = fold(M, P, sp)
        w, Vv = np.linalg.eig(Mr)
        qs.append(w)
        vs.append(unfold(sp, Vv, M.shape[0]))
    return np.concatenate(qs), np.hstack(vs)


# ---------------------------------------------------------------------------
# [260915] STAGE 1b -- carry ONE sector through the S-matrix chain.
#
# `GetSMatrix` reads its dimension from `len(q_list[0])` and builds `eye` from
# it, so the chain is generic in the dimension: fill the per-layer lists at
# sector dimension d ~ 2nG/4 and the EXISTING chain runs unmodified at
# (1/4)^3 = 1/64.  Measured on the isolated prototype: d = 289 against
# 2nG = 1154, i.e. 63.7x, reproducing stock t and r to 5e-14 .. 5e-11.
#
# The excitation needs no projection argument -- it is EXACTLY one sector.
# The (0,0) order is a mirror FIXED POINT, so its orbit is a singleton and the
# projector accumulates all four sign terms onto the same index: the co-pol
# half gets (1+ex)(1+ey), which is 4 for (+1,+1) and 0 for the other three.
# Verified numerically: ||P^H a0||/||a0|| = 1.000000000000 in (+1,+1) and
# 0.000000000000 in the other three.
# ---------------------------------------------------------------------------

SECTOR = (1, 1)


def fold_mat(A, sp):
    """P^H A P, four orbit slots at a time.

    `fold` above materialises `A[:, R]` of shape (2nG, d, 4) -- 366 MB at
    nG 2393 -- which on a 2-core allocation is a real risk.  Looping keeps the
    temporary at (2nG, d), 91 MB, for the same arithmetic.
    """
    R, V = sp
    d = R.shape[0]
    AP = np.zeros((A.shape[0], d), dtype=complex)
    for a in range(R.shape[1]):
        AP += A[:, R[:, a]] * V[:, a]
    out = np.zeros((d, d), dtype=complex)
    for a in range(R.shape[1]):
        out += V[:, a].conj()[:, None] * AP[R[:, a], :]
    return out


def fold_kkT(sp, kx, ky, nG):
    """P^H (k k^T) P without ever forming the 2nG x 2nG matrix.

    `k = vstack(diag(kx), diag(ky))` gives `kkT[p,q] = u[p] u[q] [g(p)==g(q)]`
    with `g(p) = p % nG` and `u[p] = kx[g] or ky[g]` by half -- four DIAGONAL
    blocks.  So `kkT = Z Z^T` with `Z[p,g] = u[p] [g(p)==g]`, and
    `P^H kkT P = W^H W` for `W = Z^T P` of shape (nG, d).  Forming W is
    O(4d); the product is nG*d^2, about 12x cheaper than the sector eig.
    """
    R, V = sp
    d = R.shape[0]
    u = np.concatenate((np.asarray(kx), np.asarray(ky)))
    W = np.zeros((nG, d), dtype=complex)
    g = R % nG
    for a in range(R.shape[1]):
        np.add.at(W, (g[:, a], np.arange(d)), u[R[:, a]] * V[:, a])
    return W.conj().T @ W


def fold_vec(sp, v):
    """P^H v."""
    R, V = sp
    return np.einsum("ja,ja->j", V.conj(), np.asarray(v)[R], optimize=True)


def unfold_vec(sp, w, n2):
    """P w -- a scatter, not a matmul."""
    R, V = sp
    out = np.zeros(n2, dtype=complex)
    for a in range(R.shape[1]):
        np.add.at(out, R[:, a], V[:, a] * w)
    return out


def grid_is_c2v(ep_grid, tol=0.0):
    """The actual precondition, tested in REAL SPACE where it is free.

    The raster is `u = (arange(N)+0.5)/N - 0.5`, so the mirror about the cell
    centre is the index reversal `i -> N-1-i`.  O(N^2) against the O(n^2 d) of
    a commutation residual, which at nG 2393 would cost 16x the eigensolve it
    is guarding -- the guard would eat the entire saving.

    This is load-bearing, not decoration: measured on the prototype, an
    asymmetric atom pushed through the sector chain returns a silently WRONG
    answer with 23 % relative error.
    """
    A = np.asarray(ep_grid)
    if A.ndim != 2:
        return False
    dx = float(np.max(np.abs(A - A[::-1, :])))
    dy = float(np.max(np.abs(A - A[:, ::-1])))
    scale = max(float(np.max(np.abs(A))), 1e-300)
    return (dx <= tol * scale) and (dy <= tol * scale), dx / scale, dy / scale


def sector_eig(ep2, kp, kx, ky, sp, nG):
    """Sector q and phi for a patterned layer, never forming M at full size.

    `ep2`, `kp` and `k k^T` each commute with the sector operators, and
    restriction is multiplicative on a jointly invariant subspace, so
    `M_r = (ep2)_r (kp)_r - (k k^T)_r` with every product at d x d.
    """
    Mr = fold_mat(ep2, sp) @ fold_mat(kp, sp) - fold_kkT(sp, kx, ky, nG)
    w, V = np.linalg.eig(Mr)
    q = np.sqrt(w)
    q = np.where(np.imag(q) < 0., -q, q)
    return q, V


def probe_residual(A, sp, n2, nprobe=3, seed=0):
    """Randomised ||(I-Pi) A Pi v|| / ||A Pi v||, O(n^2) per probe.

    A cheap second opinion on `grid_is_c2v`: the grid test proves the
    STRUCTURE is symmetric, this one proves the assembled OPERATOR is.  Both
    are needed -- a Fourier-factorisation bug could break the second while the
    first still passes.
    """
    rng = np.random.RandomState(seed)
    worst = 0.0
    for _ in range(nprobe):
        v = rng.randn(sp[0].shape[0]) + 1j * rng.randn(sp[0].shape[0])
        x = unfold_vec(sp, v, n2)
        Ax = A @ x
        Px = unfold_vec(sp, fold_vec(sp, Ax), n2)
        den = max(float(np.linalg.norm(Ax)), 1e-300)
        worst = max(worst, float(np.linalg.norm(Ax - Px)) / den)
    return worst


# ---------------------------------------------------------------------------
# [260915] GENERALISED TO THE SURVIVING MIRRORS -- 4 / 2 / 1 sectors.
#
# The reduction is NOT all-or-nothing. Count the mirrors the CELL keeps about
# its centre and branch:
#
#     both (C2v)   4 sectors   d ~ 2nG/4   ->  64x
#     one  (C1v)   2 sectors   d ~ 2nG/2   ->   8x
#     none         1 sector    d = 2nG     ->   1x  (fall back to stock)
#
# Measured census: every library atom and the same-atom LPA supercell keep
# both; the LPA PAIR supercell and the bottom L1/L2 keep only y (x residual
# 1.000), because a mirror swaps the two atoms in the first case and because a
# blazed grating must be x-asymmetric to favour one order in the second. That
# asymmetry is the design, not an accident, so 8x is the ceiling there -- but
# 8x is not nothing, and refusing those cases outright (as the first version
# did) throws it away.
#
# The excited sector is always the all-(+1) one: the (0,0) order is a fixed
# point of every mirror, so its orbit is a singleton and the co-pol half
# accumulates prod(1+e_i), nonzero only when every e_i = +1.
# ---------------------------------------------------------------------------


def mirrors_of(ep_grid, tol=1e-12):
    """Which mirrors does this real-space layer keep about the cell centre?

    Returns (use_x, use_y, res_x, res_y).  The raster is
    `u = (arange(N)+0.5)/N - 0.5`, so the mirror is the index reversal
    `i -> N-1-i`.  O(N^2), against the O(n^2 d) of a commutation residual --
    which at nG 2393 costs 64 % of the eigensolve it would be guarding.
    """
    A = np.asarray(ep_grid)
    if A.ndim != 2:
        raise ValueError("mirrors_of expects a 2-D grid, got shape %s"
                         % (A.shape,))
    s = max(float(np.max(np.abs(A))), 1e-300)
    rx = float(np.max(np.abs(A - A[::-1, :]))) / s
    ry = float(np.max(np.abs(A - A[:, ::-1]))) / s
    return (rx <= tol), (ry <= tol), rx, ry


def orbits_g(G, use_x, use_y):
    """Mirror orbits under the ACTIVE mirrors only.

    Closure is asserted, as in `orbits`: a G set that is not closed would make
    the projector silently wrong rather than fail.
    """
    Gv = np.asarray(G)
    idx = _index(Gv)
    sx_set = (1, -1) if use_x else (1,)
    sy_set = (1, -1) if use_y else (1,)
    seen, out = set(), []
    for g in Gv:
        m, n = int(g[0]), int(g[1])
        if (m, n) in seen:
            continue
        mem = {}
        for sm in sx_set:
            for sn in sy_set:
                key = (sm * m, sn * n)
                if key not in idx:
                    raise ValueError(
                        "G set is not closed under the active mirrors: %r "
                        "missing for %r -- reduction refused" % (key, (m, n)))
                mem[(sm, sn)] = idx[key]
                seen.add(key)
        out.append(mem)
    return out


def projectors_g(G, nxy, use_x, use_y, half_pixel=True):
    """Sector projectors for the active mirrors. Keys are (ex, ey) tuples.

    An inactive mirror contributes a fixed +1 label, so the key shape is
    uniform and (1, 1) is always the excited sector.
    """
    Gv = np.asarray(G)
    nG = len(Gv)
    if not (use_x or use_y):
        return {(1, 1): np.eye(2 * nG, dtype=complex)}
    ors = orbits_g(Gv, use_x, use_y)
    if half_pixel:
        if nxy is None:
            raise ValueError("half_pixel=True needs nxy")
        d = (nxy - 1) / (2.0 * nxy)
        ph = np.exp(-2j * np.pi * d * (Gv[:, 0] + Gv[:, 1]))
    else:
        ph = np.ones(nG, dtype=complex)

    ex_set = (1, -1) if use_x else (1,)
    ey_set = (1, -1) if use_y else (1,)
    out = {}
    for ex in ex_set:
        for ey in ey_set:
            cols = []
            for half, (px, py) in ((nG, (ex, ey)), (0, (-ex, -ey))):
                for mem in ors:
                    v = np.zeros(2 * nG, dtype=complex)
                    for (sm, sn), i in mem.items():
                        s = (px if sm < 0 else 1) * (py if sn < 0 else 1)
                        v[half + i] += s * ph[i]
                    nrm = np.linalg.norm(v)
                    if nrm > 1e-9:
                        cols.append(v / nrm)
            out[(ex, ey)] = (np.array(cols).T if cols
                             else np.zeros((2 * nG, 0)))
    return out


def to_sparse_g(P, tol=1e-12):
    """`to_sparse` for orbits of any size up to 4 (1, 2 or 4 nonzeros)."""
    n2, d = P.shape
    w = 0
    nzs = []
    for j in range(d):
        nz = np.nonzero(np.abs(P[:, j]) > tol)[0]
        if len(nz) > 4:
            raise ValueError("column %d has %d nonzeros, expected <= 4"
                             % (j, len(nz)))
        nzs.append(nz)
        w = max(w, len(nz))
    w = max(w, 1)
    R = np.zeros((d, w), dtype=np.intp)
    V = np.zeros((d, w), dtype=complex)
    for j, nz in enumerate(nzs):
        R[j, :len(nz)] = nz
        V[j, :len(nz)] = P[nz, j]
    return R, V
