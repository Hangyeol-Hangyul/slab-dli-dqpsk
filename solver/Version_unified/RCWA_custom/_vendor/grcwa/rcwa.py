import warnings
from . import backend as bd
from .fft_funs import Epsilon_fft,Epsilon_fft_pol,get_ifft
from .kbloch import Lattice_Reciprocate,Lattice_getG,Lattice_SetKs

def _lattice_is_rectangular(L1,L2):
    '''[261011 li_skew] True when L1 is along x and L2 along y (the assumption of li_fmm.py and of the Pol
    path: raster axis 0 <-> x, axis 1 <-> y). Exact up to a 1e-12 relative tolerance.'''
    return (abs(L1[1]) <= 1e-12*abs(L1[0])) and (abs(L2[0]) <= 1e-12*abs(L2[1]))

class obj:
    # [261008 li_integrate] marker: this vendored copy understands rcwa_rule / form_factor /
    # pol_epsinv. A driver that asks for a non-default value checks it BEFORE it relies on the
    # setting, so an old tree (attribute assignment would silently do nothing) fails loudly.
    LI_SUPPORT = 1
    # [261011 li_skew] marker: this vendored copy understands `obj.li_raster` (Li factorization on a skewed
    # lattice whose raster is made of square pixels, li_skew.py). A driver that sets `li_raster` checks it
    # BEFORE relying on it, because an old tree would ignore the attribute and silently run another rule.
    LI_SKEW_SUPPORT = 1

    def __init__(self,nG,L1,L2,freq,theta,phi,verbose=1,fmm_method=None,pol_sigma=3.0,pol_niter=20,sym=False,sym_nxy=None,sym_stage=1):
        '''The time harmonic convention is exp(-i omega t), speed of light = 1

        Two kinds of layers are currently supported: uniform layer,
        patterned layer from grids. Interface for patterned layer by
        direct analytic expression of Fourier series is included, but
        no examples inclded so far.

        nG: truncation order, but the actual truncation order might not be nG
        L1,L2: lattice vectors, in the list format, (x,y)
        fmm_method: None (Laurent's rule, default) or 'pol' (Pol method, S4 Eq. 51)
        pol_sigma: Gaussian blur sigma (in pixels) for the Pol tangent field.
            Only used when fmm_method='pol'.
        pol_niter: number of blur+reset iterations for the Pol tangent field.
            At each iteration the tangent field is blurred then reset to the
            exact gradient at interface pixels, approximating a Laplace solve.
            0 means a single Gaussian blur (no reset). Only used when
            fmm_method='pol'.

        '''
        self.fmm_method = fmm_method
        self.pol_sigma = pol_sigma
        self.pol_niter = pol_niter
        # [261008 li_integrate] OPT-IN Fourier factorization switches (set these attributes on the
        # object after construction; all defaults reproduce the previous behaviour exactly).
        #   rcwa_rule   : 'laurent' (default, stock) | 'liy' | 'lix'   (Li 1997, li_fmm.py)
        #   form_factor : 0 (default) | 1   exact pixel form factor sinc(m/N) per axis
        #   pol_epsinv  : 'conv1' (default, stock Pol: Toeplitz(1/eps)) | 'inv' (inv(Toeplitz(eps)))
        #                 -- only read when fmm_method == 'pol'
        self.rcwa_rule = 'laurent'
        self.form_factor = 0
        self.pol_epsinv = 'conv1'
        # [261011 li_skew] None (default): the li_fmm rules, which assume L1 // x and L2 // y. For a skewed
        # lattice whose raster is made of Cartesian (square) pixels a driver sets an integer 2 x 2 matrix:
        # row c = the raster step of grid axis c in pixel units, columns = (x, y) (v5 d cell:
        # ((0, 1), (1, -ds))). Only read when rcwa_rule != 'laurent' or form_factor != 0.
        self.li_raster = None
        # [260914] C2v parity block-diagonalisation of the layer
        # eigenproblem (Zhou & Li, J. Opt. A 6, 43 (2004)). OFF by
        # default; when off this class is byte-identical to upstream.
        # Valid ONLY for a structure that is mirror-symmetric about
        # both cell axes at normal incidence -- enforced, not assumed.
        self.sym = bool(sym)
        self.sym_nxy = sym_nxy
        self._sym_P = None
        self._sym_sp = None
        # [260915] STAGE 1b. sym_stage 1 = reassemble all sectors and hand a
        # complete mode set to the stock chain (the eig alone, 16x). stage 2 =
        # carry ONLY the excited sector through the S-matrix chain.
        #
        # The reduction is graded by how many mirrors the CELL keeps about its
        # centre -- both 4 sectors (64x), one 2 sectors (8x), none 1 (no
        # reduction). Measured: every library atom and the same-atom LPA
        # supercell keep both; the LPA PAIR supercell and the bottom L1/L2 keep
        # only y, because a mirror swaps the two atoms in the first case and a
        # blazed grating must be x-asymmetric to favour one order in the
        # second. That asymmetry IS the design, so 8x is the ceiling there.
        #
        # The excited sector is always all-(+1): the (0,0) order is a fixed
        # point of every mirror, so its orbit is a singleton and the co-pol
        # half accumulates prod(1+e_i), nonzero only when every e_i = +1.
        # Verified ||P^H a0||/||a0|| = 1.000000000000 for all four mirror sets.
        self.sym_stage = int(sym_stage)
        self._sec_q = None
        self._sec_phi = None
        self._sec_kp = None
        self._sec_sp = None
        self._sec_dim = None
        self._sec_mirrors = None
        # [260915] S-matrix cache, keyed by (indi, indj).
        # `RT_Solve` and the two exterior `GetAmplitudes` calls each ran a
        # FULL `GetSMatrix(0, Layer_N-1)` -- three identical chains per solve.
        # Measured in this codebase at nG 401 (`Main_rcwa_l3_fullstack2d.py`
        # docstring): `RT_Solve` 8.31 s + `GetAmplitudes` 8.05/8.07 s is 87 %
        # of the cost while the eigendecomposition is 3.51 s. The S-matrix
        # depends only on the LAYERS, never on the excitation, so changing
        # `a0`/`bN` reuses it -- and any change to the layers invalidates it.
        self._sm_cache = None
        self.freq = freq
        self.omega = 2*bd.pi*freq+0.j
        self.L1 = L1
        self.L2 = L2
        self.phi = phi
        self.theta = theta
        self.nG = nG
        self.verbose = verbose
        self.Layer_N = 0  # total number of layers
      
        # the length of the following variables = number of total layers
        self.thickness_list = []
        self.id_list = []  #[type, No., No. in patterned/uniform, No. in its family] starting from 0
        # type:0 for uniform, 1 for Grids, 2 for Fourier

        self.kp_list = []                
        self.q_list = []  # eigenvalues
        self.phi_list = [] #eigenvectors

        # Uniform layer
        self.Uniform_ep_list = []
        self.Uniform_N = 0
        
        # Patterned layer
        self.Patterned_N = 0  # total number of patterned layers        
        self.Patterned_epinv_list = []
        self.Patterned_ep2_list = []
        
        # patterned layer from Grids
        self.GridLayer_N = 0
        self.GridLayer_Nxy_list = []

        # layers of analytic Fourier series (e.g. circles)
        self.FourierLayer_N = 0
        self.FourierLayer_params = []        
        
    def _invalidate_smatrix(self):
        '''[260915] Anything that changes q/phi/kp/thickness must call this.

        Conservative by design: the cache is dropped whole rather than
        selectively, because a stale S-matrix is a SILENT wrong answer, and
        under autograd a stale ArrayBox would be a silently wrong gradient.
        '''
        self._sm_cache = None
        self._sec_q = None
        self._sec_phi = None
        self._sec_kp = None
        self._sec_sp = None
        self._sec_dim = None
        self._sec_mirrors = None
        self._sec_Ps = None
        self._sec_ready = False
        self._sec_key = None

    def Add_LayerUniform(self,thickness,epsilon):
        #assert type(thickness) == float, 'thickness should be a float'
        self._invalidate_smatrix()

        self.id_list.append([0,self.Layer_N,self.Uniform_N])
        self.Uniform_ep_list.append(epsilon)
        self.thickness_list.append(thickness)
        
        self.Layer_N += 1
        self.Uniform_N += 1

    def Add_LayerGrid(self,thickness,Nx,Ny):
        self._invalidate_smatrix()
        self.thickness_list.append(thickness)
        self.GridLayer_Nxy_list.append([Nx,Ny])
        self.id_list.append([1,self.Layer_N,self.Patterned_N,self.GridLayer_N])

        self.Layer_N += 1
        self.GridLayer_N += 1
        self.Patterned_N += 1

    def Add_LayerFourier(self,thickness,params):
        self._invalidate_smatrix()
        self.thickness_list.append(thickness)
        self.FourierLayer_params.append(params)
        self.id_list.append([2,self.Layer_N,self.Patterned_N,self.FourierLayer_N])

        self.Layer_N += 1
        self.Patterned_N += 1
        self.FourierLayer_N += 1

    def Init_Setup(self,Pscale=1.,Gmethod=0):
        '''
        Set up reciprocal lattice (Gmethod:truncation scheme, 0 for circular, 1 for rectangular)
        Pscale: scale the period
        Compute eigenvalues for uniform layers
        Initialize vectors for patterned layers
        '''
        self._invalidate_smatrix()
        kx0 = self.omega*bd.sin(self.theta)*bd.cos(self.phi)*bd.sqrt(self.Uniform_ep_list[0])
        ky0 = self.omega*bd.sin(self.theta)*bd.sin(self.phi)*bd.sqrt(self.Uniform_ep_list[0])

        # set up reciprocal lattice
        self.Lk1, self.Lk2 = Lattice_Reciprocate(self.L1,self.L2)
        self.G,self.nG = Lattice_getG(self.nG,self.Lk1,self.Lk2,method=Gmethod)
        
        self.Lk1 = self.Lk1/Pscale
        self.Lk2 = self.Lk2/Pscale
        # self.kx = kx0 + 2*bd.pi*(self.Lk1[0]*self.G[:,0]+self.Lk2[0]*self.G[:,1])
        # self.ky = ky0 + 2*bd.pi*(self.Lk1[1]*self.G[:,0]+self.Lk2[1]*self.G[:,1])
        self.kx,self.ky = Lattice_SetKs(self.G, kx0, ky0, self.Lk1, self.Lk2)
        
        #normalization factor for energies off normal incidence
        self.normalization = bd.sqrt(self.Uniform_ep_list[0])/bd.cos(self.theta)
        
        #if comm.rank == 0 and verbose>0:
        if self.verbose>0:
            print('Total nG = ',self.nG)

        self.Patterned_ep2_list = [None]*self.Patterned_N
        self.Patterned_epinv_list = [None]*self.Patterned_N            
        for i in range(self.Layer_N):
            if self.id_list[i][0] == 0:
                ep = self.Uniform_ep_list[self.id_list[i][2]]
                kp = MakeKPMatrix(self.omega,0,1./ep,self.kx,self.ky)
                self.kp_list.append(kp)
                
                q,phi = SolveLayerEigensystem_uniform(self.omega,self.kx,self.ky,ep)
                self.q_list.append(q)
                self.phi_list.append(phi)
            else:
                self.kp_list.append(None)
                self.q_list.append(None)
                self.phi_list.append(None)
                
    def MakeExcitationPlanewave(self,p_amp,p_phase,s_amp,s_phase,order = 0, direction = 'forward'):
        '''
        Front incidence
        '''
        self.direction = direction
        theta = self.theta
        phi = self.phi
        a0 = bd.zeros(2*self.nG,dtype=complex)
        bN = bd.zeros(2*self.nG,dtype=complex)
        if direction == 'forward':
            tmp1 = bd.zeros(2*self.nG,dtype=complex)
            tmp1[order] = 1.0
            a0 = a0 + tmp1*(-s_amp*bd.cos(theta)*bd.cos(phi)*bd.exp(1j*s_phase) \
                        -p_amp*bd.sin(phi)*bd.exp(1j*p_phase))

            tmp2 = bd.zeros(2*self.nG,dtype=complex)
            tmp2[order+self.nG] = 1.0            
            a0 = a0 + tmp2*(-s_amp*bd.cos(theta)*bd.sin(phi)*bd.exp(1j*s_phase) \
                            +p_amp*bd.cos(phi)*bd.exp(1j*p_phase))
        elif direction == 'backward':
            tmp1 = bd.zeros(2*self.nG,dtype=complex)
            tmp1[order] = 1.0
            bN = bN + tmp1*(-s_amp*bd.cos(theta)*bd.cos(phi)*bd.exp(1j*s_phase) \
                            -p_amp*bd.sin(phi)*bd.exp(1j*p_phase))

            tmp2 = bd.zeros(2*self.nG,dtype=complex)
            tmp2[order+self.nG] = 1.0
            bN = bN + tmp2*(-s_amp*bd.cos(theta)*bd.sin(phi)*bd.exp(1j*s_phase) \
                            +p_amp*bd.cos(phi)*bd.exp(1j*p_phase))
        
        self.a0 = a0
        self.bN = bN
        
    def GridLayer_geteps(self,ep_all):
        '''
        Fourier transform + eigenvalue for grid layer
        '''
        self._invalidate_smatrix()
        ptri = 0
        ptr = 0
        for i in range(self.Layer_N):
            if self.id_list[i][0] != 1:
                continue

            Nx = self.GridLayer_Nxy_list[ptri][0]
            Ny = self.GridLayer_Nxy_list[ptri][1]
            dN = 1./Nx/Ny

            if len(ep_all) == 3 and ep_all[0].ndim>0:
                ep_grid = [bd.reshape(ep_all[0][ptr:ptr+Nx*Ny],[Nx,Ny]),bd.reshape(ep_all[1][ptr:ptr+Nx*Ny],[Nx,Ny]),bd.reshape(ep_all[2][ptr:ptr+Nx*Ny],[Nx,Ny])]
            else:
                ep_grid = bd.reshape(ep_all[ptr:ptr+Nx*Ny],[Nx,Ny])

            # Pol method: better Fourier convergence for discontinuous eps.
            # Falls back to FFT for anisotropic grids (3-component eps_grid)
            # since Pol currently only supports isotropic permittivity.
            is_anisotropic = len(ep_all) == 3 and ep_all[0].ndim > 0
            if self.fmm_method == 'pol' and not is_anisotropic:
                if self.li_raster is not None or not _lattice_is_rectangular(self.L1,self.L2):
                    # [261011 li_skew] the Pol tangent field takes finite differences along the raster axes
                    # as d/dx and d/dy: wrong on a lattice with L1 not along x or L2 not along y
                    raise ValueError("fmm_method='pol' assumes a rectangular lattice (L1 along x, L2 along y); "
                                     "it is refused on a skewed lattice or with obj.li_raster set")
                if self.pol_epsinv != 'conv1' or self.form_factor or self.rcwa_rule != 'laurent':
                    # [261008 li_integrate] opt-in only: lazy import, never reached by default
                    from .li_fmm import epsilon_pol
                    epinv, ep2 = epsilon_pol(dN,ep_grid,self.G,self.pol_sigma,self.pol_niter,
                                             self.pol_epsinv,self.form_factor,self.rcwa_rule)
                else:
                    epinv, ep2 = Epsilon_fft_pol(dN,ep_grid,self.G,pol_sigma=self.pol_sigma,pol_niter=self.pol_niter)
            else:
                if self.fmm_method == 'pol' and is_anisotropic:
                    warnings.warn(
                        "fmm_method='pol' is not supported for anisotropic "
                        "(3-component) grids; falling back to standard "
                        "Laurent factorization for this layer.",
                        stacklevel=2,
                    )
                if self.rcwa_rule != 'laurent' or self.form_factor:
                    # [261008 li_integrate] opt-in only: lazy import, never reached by default
                    if self.li_raster is not None:
                        # [261011 li_skew] skewed lattice, square-pixel raster: Cartesian-edge column rule
                        from .li_skew import epsilon_li_skew
                        epinv, ep2 = epsilon_li_skew(dN,ep_grid,self.G,self.rcwa_rule,self.form_factor,
                                                     self.li_raster,(self.L1,self.L2,Nx,Ny),is_anisotropic)
                    else:
                        if not _lattice_is_rectangular(self.L1,self.L2):
                            # [261011 li_skew] any non-default rule / form factor without li_raster: liy / lix name
                            # the raster axes x / y, and the laurent + form-factor sinc is a parallelogram sinc, which
                            # is not the physics of any raster in this codebase; on a skewed lattice li_fmm would
                            # silently apply the wrong rule
                            raise ValueError("rcwa_rule %r / rcwa_ff %r assumes L1 along x and L2 along y "
                                             "(li_fmm.py); this lattice is skewed: set obj.li_raster (li_skew.py)"
                                             % (self.rcwa_rule, self.form_factor))
                        from .li_fmm import epsilon_li
                        epinv, ep2 = epsilon_li(dN,ep_grid,self.G,self.rcwa_rule,self.form_factor,is_anisotropic)
                else:
                    epinv, ep2 = Epsilon_fft(dN,ep_grid,self.G)

            self.Patterned_epinv_list[self.id_list[i][2]] = epinv
            self.Patterned_ep2_list[self.id_list[i][2]] = ep2

            kp = MakeKPMatrix(self.omega,1,epinv,self.kx,self.ky)
            self.kp_list[self.id_list[i][1]] = kp

            if self.sym and self.sym_stage >= 2:
                # [260915] STAGE 1b. Do NOT solve the full eigenproblem --
                # only the excited sector is ever used, and it is built after
                # this loop, once every patterned layer has reported which
                # mirrors it keeps (the group is the INTERSECTION over layers:
                # one asymmetric layer demotes the whole stack).
                from .symred import mirrors_of
                if not type(ep2).__module__.startswith('numpy'):
                    raise RuntimeError(
                        "sym_stage=2 needs plain numpy arrays; got %s. It "
                        "cannot carry an autograd tape, so it refuses rather "
                        "than return a silently wrong gradient."
                        % type(ep2).__name__)
                _ux,_uy,_rx,_ry = mirrors_of(ep_grid)
                if self._sec_mirrors is None:
                    self._sec_mirrors = (_ux,_uy)
                else:
                    self._sec_mirrors = (self._sec_mirrors[0] and _ux,
                                         self._sec_mirrors[1] and _uy)
                if self.verbose > 0:
                    print('layer %d mirrors: x %s (%.2e)  y %s (%.2e)'
                          % (i,_ux,_rx,_uy,_ry))
                self.q_list[self.id_list[i][1]] = None
                self.phi_list[self.id_list[i][1]] = None
                ptr += Nx*Ny
                ptri += 1
                continue
            if self.sym:
                from .symred import projectors, to_sparse, eig_blocked_fast
                if self._sym_P is None:
                    _n = self.sym_nxy if self.sym_nxy else Nx
                    self._sym_P = projectors(self.G, _n, half_pixel=True)
                    self._sym_sp = {k: to_sparse(v)
                                    for k, v in self._sym_P.items()}
                _k = bd.vstack((bd.diag(self.kx),bd.diag(self.ky)))
                _M = bd.dot(ep2,kp) - bd.dot(_k,bd.transpose(_k))
                # [260914] Test the OBJECT, not the backend's name.
                # `set_backend` swaps the backend instance's __class__, so a
                # name check is order-dependent and misfired in the pool processes
                # (6 jobs lost, 260914 22:41).  An autograd ArrayBox lives in
                # autograd.numpy.*, so this is a direct test of the real
                # precondition: symred is raw numpy and cannot carry a tape.
                if not type(_M).__module__.startswith('numpy'):
                    raise RuntimeError(
                        "sym=True needs plain numpy arrays; got %s from the "
                        "%s backend. It cannot carry an autograd tape, so it "
                        "refuses rather than return a silently wrong gradient."
                        % (type(_M).__name__, bd.__class__.__name__))
                q,phi = eig_blocked_fast(_M,self._sym_P,sparse=self._sym_sp)
                q = bd.sqrt(q)
                q = bd.where(bd.imag(q)<0.,-q,q)
            else:
                q,phi = SolveLayerEigensystem(self.omega,self.kx,self.ky,kp,ep2)
            self.q_list[self.id_list[i][1]] = q
            self.phi_list[self.id_list[i][1]] = phi

            ptr += Nx*Ny
            ptri += 1            

        if self.sym and self.sym_stage >= 2:
            self._build_sector()

    def _build_sector(self):
        '''[260915] Build the excited sector's (q, phi, kp) for every layer.

        `M = ep2 @ kp - k k^T` is never formed at full size. Each factor
        commutes with the sector operators -- `ep2` because the structure is
        mirror-symmetric, and the kinematic ones directly, since
        `Sx k = +k R` and `Sx Jk = -Jk R` -- and restriction is multiplicative
        on a jointly invariant subspace, so every product is d x d.
        '''
        import numpy as _np
        from .symred import (projectors_g, to_sparse_g, fold_mat, fold_kkT)
        ux,uy = self._sec_mirrors if self._sec_mirrors else (False,False)
        # [260915] **THE INCIDENCE BREAKS MIRRORS TOO, and the real-space
        # grid cannot see it.** `kx = kx0 + Gx`, so under `Gx -> -Gx` we get
        # `kx0 - Gx`, which equals `-kx` only when `kx0 = 0`. At oblique
        # incidence in the x-plane the x mirror is gone no matter how
        # symmetric the structure is. Test the RETAINED k-sets directly rather
        # than the angle -- it is the quantity the operator actually uses.
        _kx = _np.asarray(self.kx); _ky = _np.asarray(self.ky)
        _tol = 1e-9*max(float(_np.max(_np.abs(_kx))),
                        float(_np.max(_np.abs(_ky))), 1e-300)
        _kxok = bool(_np.allclose(_np.sort_complex(_kx),
                                  _np.sort_complex(-_kx), atol=_tol))
        _kyok = bool(_np.allclose(_np.sort_complex(_ky),
                                  _np.sort_complex(-_ky), atol=_tol))
        if self.verbose > 0 and (ux and not _kxok or uy and not _kyok):
            print('incidence removes a mirror: kx-symmetric %s, ky-symmetric %s'
                  % (_kxok,_kyok))
        ux = ux and _kxok
        uy = uy and _kyok
        self._sec_mirrors = (ux,uy)
        if not (ux or uy):
            raise RuntimeError(
                "sym_stage=2: no mirror survives about the cell centre, so "
                "there is nothing to reduce. Re-run with sym=0.")
        self._sec_Ps = projectors_g(self.G, self.sym_nxy, ux, uy,
                                    half_pixel=True)
        self._sec_ready = False
        if self.verbose > 0:
            print('sector: mirrors x=%s y=%s, dims %s of %d'
                  % (ux,uy,{k:v.shape[1] for k,v in self._sec_Ps.items()},
                     2*self.nG))

    def _ensure_sector_lists(self):
        '''[260915] Pick the EXCITED sector from `a0`, then build its lists.

        The excited sector is NOT always (+1,+1). `MakeExcitationPlanewave`
        puts a p-polarised wave into the co-pol half weighted by `cos(phi)` and
        into the cross-pol half by `-sin(phi)`, so at `phi = 90 deg` the
        excitation sits in the CROSS-pol half and therefore in a sector with a
        negative label. Hardcoding (+1,+1) returned a silently WRONG answer
        there (measured: relative error 1.00, i.e. the projection was zero).

        So the sector is read off `a0`, and an excitation that straddles more
        than one sector is REFUSED rather than truncated.
        '''
        import numpy as _np
        from .symred import to_sparse_g, fold_mat, fold_kkT
        if self._sec_ready:
            return
        a0 = _np.asarray(self.a0)
        bN = _np.asarray(self.bN)
        nrm = max(float(_np.linalg.norm(a0)), float(_np.linalg.norm(bN)), 1e-300)
        comp = {}
        for k,P in self._sec_Ps.items():
            if P.shape[1] == 0:
                continue
            c = max(float(_np.linalg.norm(P.conj().T.dot(a0))),
                    float(_np.linalg.norm(P.conj().T.dot(bN))))
            comp[k] = c/nrm
        key = max(comp, key=lambda k: comp[k])
        leak = max([v for k,v in comp.items() if k != key] or [0.0])
        if leak > 1e-9:
            raise RuntimeError(
                "sym_stage=2: the excitation is spread over more than one "
                "symmetry sector (largest %.3f in %r, %.3e elsewhere). Only a "
                "single-sector excitation can be reduced; this one cannot."
                % (comp[key], key, leak))
        P = self._sec_Ps[key]
        sp = to_sparse_g(P)
        d = P.shape[1]
        Rw,Vv = sp
        kx = _np.asarray(self.kx); ky = _np.asarray(self.ky)
        kkT_r = fold_kkT(sp,kx,ky,self.nG)
        self._sec_q = [None]*self.Layer_N
        self._sec_phi = [None]*self.Layer_N
        self._sec_kp = [None]*self.Layer_N
        for i in range(self.Layer_N):
            li = self.id_list[i][1]
            kp = _np.asarray(self.kp_list[li])
            kp_r = fold_mat(kp,sp)
            self._sec_kp[li] = kp_r
            if self.id_list[i][0] == 0:
                q = _np.asarray(self.q_list[li])
                qcol = q[Rw[:,0]]
                for a in range(Rw.shape[1]):
                    m = _np.abs(Vv[:,a]) > 0
                    if m.any() and not _np.allclose(q[Rw[m,a]],qcol[m]):
                        raise RuntimeError(
                            "layer %d: q is not constant on a mirror orbit -- "
                            "the uniform reduction is invalid" % i)
                self._sec_q[li] = qcol
                self._sec_phi[li] = _np.eye(d,dtype=complex)
            else:
                ep2 = _np.asarray(self.Patterned_ep2_list[self.id_list[i][2]])
                Mr = fold_mat(ep2,sp).dot(kp_r) - kkT_r
                w,V = _np.linalg.eig(Mr)
                qq = _np.sqrt(w)
                qq = _np.where(_np.imag(qq)<0.,-qq,qq)
                self._sec_q[li] = qq
                self._sec_phi[li] = V
        self._sec_sp = sp
        self._sec_dim = d
        self._sec_key = key
        self._sec_ready = True
        if self.verbose > 0:
            print('sector %r: dim %d of %d, chain gain %.1fx'
                  % (key,d,2*self.nG,(2.0*self.nG/d)**3))

    def Return_eps(self,which_layer,Nx,Ny,component='xx'):
        '''
        For patterned layer component = 'xx','xy','yx','yy','zz'
        For uniform layer, currently it's assumed to be isotropic
        '''
        i = which_layer
        # uniform layer
        if self.id_list[i][0] == 0:
            ep = self.Uniform_ep_list[self.id_list[i][2]]
            return bd.ones((Nx,Ny))*ep

        # patterned layer
        elif self.id_list[i][0] == 1:
            if component == 'zz':
                epk = bd.inv(self.Patterned_epinv_list[self.id_list[i][2]])
            elif component == 'xx':
                epk = self.Patterned_ep2_list[self.id_list[i][2]][:self.nG,:self.nG]
            elif component == 'xy':
                epk = self.Patterned_ep2_list[self.id_list[i][2]][:self.nG,self.nG:]
            elif component == 'yx':
                epk = self.Patterned_ep2_list[self.id_list[i][2]][self.nG:,:self.nG]
            elif component == 'yy':
                epk = self.Patterned_ep2_list[self.id_list[i][2]][self.nG:,self.nG:]
                
            return get_ifft(Nx,Ny,epk[0,:],self.G)

            
    def _get_smatrix(self,indi,indj):
        '''[260915] Cached `GetSMatrix`, keyed by the layer RANGE.

        Serves the exterior chain (0, Layer_N-1) and the two half-chains
        `SolveInterior` needs, so a `spacer_purity` call after an `RT_Solve`
        recomputes only what it must.
        '''
        key = (int(indi),int(indj))
        if self._sm_cache is None:
            self._sm_cache = {}
        if key not in self._sm_cache:
            if self._sector_live():
                self._sm_cache[key] = GetSMatrix(indi,indj,self._sec_q,self._sec_phi,self._sec_kp,self.thickness_list)
            else:
                self._sm_cache[key] = GetSMatrix(indi,indj,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
        return self._sm_cache[key]

    def _sector_live(self):
        '''True when the chain should run in sector coordinates.'''
        if not (bool(self.sym) and self.sym_stage >= 2
                and getattr(self,'_sec_Ps',None) is not None):
            return False
        self._ensure_sector_lists()
        return self._sec_sp is not None

    def _solve_exterior(self):
        '''`SolveExterior` against the cached S-matrix -- 4 matvecs, no chain.'''
        if self._sector_live():
            from .symred import fold_vec, unfold_vec
            import numpy as _np
            n2 = 2*self.nG
            a0r = fold_vec(self._sec_sp, _np.asarray(self.a0))
            bNr = fold_vec(self._sec_sp, _np.asarray(self.bN))
            S11, S12, S21, S22 = self._get_smatrix(0,self.Layer_N-1)
            aNr = S11.dot(a0r) + S12.dot(bNr)
            b0r = S21.dot(a0r) + S22.dot(bNr)
            # Unfold before anything reads a polarisation half: GetZPoyntingFlux
            # splits its argument as v[:n]+v[n:], which is meaningless in sector
            # coordinates (the two parities keep different numbers of orbits,
            # and an odd d would silently drop an element).
            return (unfold_vec(self._sec_sp,aNr,n2),
                    unfold_vec(self._sec_sp,b0r,n2))
        S11, S12, S21, S22 = self._get_smatrix(0,self.Layer_N-1)
        aN = bd.dot(S11,self.a0) + bd.dot(S12,self.bN)
        b0 = bd.dot(S21,self.a0) + bd.dot(S22,self.bN)
        return aN,b0

    def _solve_interior(self,which_layer):
        '''`SolveInterior` against the cached S-matrices.'''
        if self._sector_live():
            from .symred import fold_vec, unfold_vec
            import numpy as _np
            n2 = 2*self.nG
            a0r = fold_vec(self._sec_sp, _np.asarray(self.a0))
            bNr = fold_vec(self._sec_sp, _np.asarray(self.bN))
            S11, S12, S21, S22 = self._get_smatrix(0,which_layer)
            pS11, pS12, pS21, pS22 = self._get_smatrix(which_layer,self.Layer_N-1)
            tmp = _np.linalg.inv(_np.eye(self._sec_dim)-S12.dot(pS21))
            air = tmp.dot(S11.dot(a0r)+S12.dot(pS22.dot(bNr)))
            bir = pS21.dot(air) + pS22.dot(bNr)
            return (unfold_vec(self._sec_sp,air,n2),
                    unfold_vec(self._sec_sp,bir,n2))
        nG2 = len(self.q_list[0])
        S11, S12, S21, S22 = self._get_smatrix(0,which_layer)
        pS11, pS12, pS21, pS22 = self._get_smatrix(which_layer,self.Layer_N-1)
        tmp = bd.inv(bd.eye(nG2)-bd.dot(S12,pS21))
        ai = bd.dot(tmp,  bd.dot(S11,self.a0)+bd.dot(S12,bd.dot(pS22,self.bN)))
        bi = bd.dot(pS21,ai) + bd.dot(pS22,self.bN)
        return ai,bi

    def RT_Solve(self,normalize = 0, byorder = 0):
        '''
        Reflection and transmission power computation
        Returns 2R and 2T, following Victor's notation
        Maybe because 2* makes S_z = 1 for H=1 in vacuum

        if normalize = 1, it will be divided by n[0]*cos(theta)
        '''
        aN, b0 = self._solve_exterior()
        fi,bi = GetZPoyntingFlux(self.a0,b0,self.omega,self.kp_list[0],self.phi_list[0],self.q_list[0],byorder=byorder)
        fe,be = GetZPoyntingFlux(aN,self.bN,self.omega,self.kp_list[-1],self.phi_list[-1],self.q_list[-1],byorder=byorder)

        if self.direction == 'forward':
            R = bd.real(-bi)
            T = bd.real(fe)
        elif self.direction == 'backward':
            R = bd.real(fe)
            T = bd.real(-bi)

        if normalize == 1:
            R = R*self.normalization
            T = T*self.normalization
        return R,T

    def GetAmplitudes(self,which_layer,z_offset):
        '''
        returns fourier amplitude
        '''
        if which_layer == 0 :
            aN, b0 = self._solve_exterior()
            ai = self.a0
            bi = b0

        elif which_layer == self.Layer_N-1:
            aN, b0 = self._solve_exterior()
            ai = aN
            bi = self.bN

        else:
            if self._sector_live() and self.q_list[which_layer] is None:
                raise RuntimeError(
                    "sym_stage=2: the field inside PATTERNED layer %d needs "
                    "that layer's full eigenvectors, which the sector build "
                    "deliberately does not compute. Uniform interior layers "
                    "(a spacer, which is what spacer_purity reads) are fine."
                    % which_layer)
            ai, bi = self._solve_interior(which_layer)

        ai, bi = TranslateAmplitudes(self.q_list[which_layer],self.thickness_list[which_layer],z_offset,ai,bi)

        return ai,bi
    
    def Solve_FieldFourier(self,which_layer,z_offset):
        '''
        returns field amplitude in fourier space: [ex,ey,ez], [hx,hy,hz]
        '''
        ai, bi = self.GetAmplitudes(which_layer,z_offset)

        # hx, hy in Fourier space
        fhxy = bd.dot(self.phi_list[which_layer],ai+bi)
        fhx = fhxy[:self.nG]
        fhy = fhxy[self.nG:]

        # ex,ey in Fourier space
        tmp1 = (ai-bi)/self.omega/self.q_list[which_layer]
        tmp2 = bd.dot(self.phi_list[which_layer],tmp1)
        fexy = bd.dot(self.kp_list[which_layer],tmp2)
        fey = - fexy[:self.nG]
        fex = fexy[self.nG:]
        
        #hz in Fourier space
        fhz = (self.kx*fey - self.ky*fex)/self.omega

        #ez in Fourier space
        fez = (self.ky*fhx - self.kx*fhy)/self.omega
        if self.id_list[which_layer][0] == 0:
            fez = fez / self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            fez = bd.dot(self.Patterned_epinv_list[self.id_list[which_layer][2]],fez)

        return [fex,fey,fez],[fhx,fhy,fhz]

    def Solve_FieldOnGrid(self,which_layer,z_offset):
        #assert self.id_list[which_layer][0] == 1, 'Needs to be grids layer'

        Nxy = self.GridLayer_Nxy_list[self.id_list[which_layer][3]]
        Nx = Nxy[0]
        Ny = Nxy[1]

        # e,h in Fourier space
        fe,fh = self.Solve_FieldFourier(which_layer,z_offset)

        ex = get_ifft(Nx,Ny,fe[0],self.G)
        ey = get_ifft(Nx,Ny,fe[1],self.G)
        ez = get_ifft(Nx,Ny,fe[2],self.G)

        hx = get_ifft(Nx,Ny,fh[0],self.G)
        hy = get_ifft(Nx,Ny,fh[1],self.G)
        hz = get_ifft(Nx,Ny,fh[2],self.G)

        return [ex,ey,ez],[hx,hy,hz]

    def Volume_integral(self,which_layer,Mx,My,Mz,normalize=0):
        '''Mxyz is convolution matrix.
        This function computes 1/A\int_V Mx|Ex|^2+My|Ey|^2+Mz|Ez|^2
        To be consistent with Poynting vector defintion here, the absorbed power will be just omega*output
        '''
        kp = self.kp_list[which_layer]
        q = self.q_list[which_layer]
        phi = self.phi_list[which_layer]

        if self.id_list[which_layer][0] == 0:
            epinv = 1. / self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            epinv = self.Patterned_epinv_list[self.id_list[which_layer][2]]

        # un-translated amplitdue
        ai, bi = self._solve_interior(which_layer)
        ab = bd.hstack((ai,bi))
        abMatrix = bd.outer(ab,bd.conj(ab))
        
        Mt = Matrix_zintegral(q,self.thickness_list[which_layer])
        # overall
        abM = abMatrix * Mt

        # F matrix
        Faxy = bd.dot(bd.dot(kp,phi), bd.diag(1./self.omega/q))
        Faz1 = 1./self.omega*bd.dot(epinv,bd.diag(self.ky))
        Faz2 = -1./self.omega*bd.dot(epinv,bd.diag(self.kx))
        Faz = bd.dot(bd.hstack((Faz1,Faz2)),phi)

        tmp1 = bd.vstack((Faxy,Faz))
        tmp2 = bd.vstack((-Faxy,Faz))
        F = bd.hstack((tmp1,tmp2))

        # consider Mtotal
        Mzeros = bd.zeros_like(Mx)
        Mtotal = bd.vstack((bd.hstack((Mx,Mzeros,Mzeros)),\
                            bd.hstack((Mzeros,My,Mzeros)),\
                            bd.hstack((Mzeros,Mzeros,Mz))))

        # integral = Tr[ abMatrix * F^\dagger *  Matconv *F ] 
        tmp = bd.dot(bd.dot(bd.conj(bd.transpose(F)),Mtotal),F)
        val = bd.trace(bd.dot(abM,tmp))

        if normalize == 1:
            val = val*self.normalization
        return val
        
    def Solve_ZStressTensorIntegral(self,which_layer):
        '''
        returns 2F_x,2F_y,2F_z, integrated over z-plane
        '''
        z_offset = 0.
        e,h = self.Solve_FieldFourier(which_layer,z_offset)
        ex = e[0]
        ey = e[1]
        ez = e[2]

        hx = h[0]
        hy = h[1]
        hz = h[2]

        # compute D = epsilon E
        ## Dz = epsilon_z E_z = (ky*hx - kx*hy)/self.omega
        dz = (self.ky*hx - self.kx*hy)/self.omega

        ## Dxy = epsilon2 * Exy
        if self.id_list[which_layer][0] == 0:
            dx = ex * self.Uniform_ep_list[self.id_list[which_layer][2]]
            dy = ey * self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            exy = bd.hstack((-ey,ex))
            dxy = bd.dot(self.Patterned_ep2_list[self.id_list[which_layer][2]],exy)
            dx = dxy[self.nG:]
            dy = -dxy[:self.nG]

        Tx = bd.sum(ex*bd.conj(dz)+hx*bd.conj(hz))
        Ty = bd.sum(ey*bd.conj(dz)+hy*bd.conj(hz))
        Tz = 0.5*bd.sum(ez*bd.conj(dz)+hz*bd.conj(hz)-ey*bd.conj(dy)-ex*bd.conj(dx)-bd.abs(hx)**2-bd.abs(hy)**2)

        Tx = bd.real(Tx)
        Ty = bd.real(Ty)
        Tz = bd.real(Tz)

        return Tx,Ty,Tz

def MakeKPMatrix(omega,layer_type,epinv,kx,ky):
    """kp = omega^2 I - Jk epinv Jk^T.

    [260915] **Neither product with `Jk` is a real matrix multiply.**
    `Jk = vstack(diag(-ky), diag(kx))` has exactly ONE nonzero per column, so
    `Jk @ A` is a ROW scaling of a stacked copy and `A @ Jk^T` is a COLUMN
    scaling -- both O(n^2), against the O(n^3) the dense `bd.dot` was paying.
    Verified identical to the old form, max|diff| 0.00e+00, both branches.

    The old cost was not a rounding error in the budget. Per layer at
    nG 3197: uniform 4 n^3 = 1.3e11 flops, patterned 6 n^3 = 2.0e11 -- each
    MORE than the Stage 1b sector eigensolve it sits beside (1.0e11). This is
    the bottleneck Stage 1b exposed, and it needs no symmetry to fix, so
    `sym = 0` runs get it too.

    The commented-out `np.block` on the uniform branch was the closed form all
    along; it was written and then not used.
    """
    nG = len(kx)

    # uniform layer, epinv has length 1
    if layer_type == 0:
        # Jk Jk^T is four DIAGONAL blocks.
        JkkJT = bd.vstack((bd.hstack((bd.diag(ky*ky),  bd.diag(-ky*kx))),
                           bd.hstack((bd.diag(-kx*ky), bd.diag(kx*kx)))))
        kp = omega**2*bd.eye(2*nG) - epinv*JkkJT
    # patterned layer
    else:
        _ky = bd.reshape(ky,(-1,1))
        _kx = bd.reshape(kx,(-1,1))
        tmp = bd.vstack((-_ky*epinv, _kx*epinv))          # Jk @ epinv
        kp = omega**2*bd.eye(2*nG) - bd.hstack(            # ... @ Jk^T
            (tmp*bd.reshape(-ky,(1,-1)), tmp*bd.reshape(kx,(1,-1))))

    return kp

def SolveLayerEigensystem_uniform(omega,kx,ky,epsilon):
    nG = len(kx)
    q = bd.sqrt(epsilon*omega**2 - kx**2 - ky**2)
    # branch cut choice
    q = bd.where(bd.imag(q)<0.,-q,q)

    q = bd.concatenate((q,q))
    phi = bd.eye(2*nG)
    return q,phi

def SolveLayerEigensystem(omega,kx,ky,kp,ep2):
    nG = len(kx)
    
    k = bd.vstack((bd.diag(kx),bd.diag(ky)))
    kkT = bd.dot(k,bd.transpose(k))
    M = bd.dot(ep2,kp) - kkT
    
    q,phi = bd.eig(M)

    q = bd.sqrt(q)
    # branch cut choice
    q = bd.where(bd.imag(q)<0.,-q,q)
    return q,phi

def GetSMatrix(indi,indj,q_list,phi_list,kp_list,thickness_list):
    ''' S_ij: size 4n*4n
    '''
    #assert type(indi) == int, 'layer index i must be integar'
    #assert type(indj) == int, 'layer index j must be integar'
    
    nG2 = len(q_list[0])
    S11 = bd.eye(nG2,dtype=complex)
    S12 = bd.zeros_like(S11)
    S21 = bd.zeros_like(S11)
    S22 = bd.eye(nG2,dtype=complex)
    if indi == indj:
        return S11,S12,S21,S22
    elif indi>indj:
        raise Exception('indi must be < indj')
   
    for l in range(indi,indj):
        ## next layer
        lp1 = l+1

        ## Q = inv(phi_l) * phi_lp1
        Q = bd.dot(bd.inv(phi_list[l]),  phi_list[lp1])
        ## P = ql*inv(kp_l*phi_l) * kp_lp1*phi_lp1*q_lp1^-1
        P1 = bd.dot(bd.diag(q_list[l]),   bd.inv(bd.dot(kp_list[l],phi_list[l])))
        P2 = bd.dot(bd.dot(kp_list[lp1],phi_list[lp1]),   bd.diag(1./q_list[lp1]))
        P = bd.dot(P1,P2)
        # P1 = bd.dot(kp_list[l],phi_list[l])
        # P2 = bd.dot(bd.dot(kp_list[lp1],phi_list[lp1]),   bd.diag(1./q_list[lp1]))
        # P = np.linalg.solve(P1,P2)
        # P = np.dot(np.diag(q_list[l]),P)

        #T11=T22, T12=T21
        T11 = 0.5*(Q+P)
        T12 = 0.5*(Q-P)

        # phase
        d1 = bd.diag(bd.exp(1j*q_list[l]*thickness_list[l]))
        d2 = bd.diag(bd.exp(1j*q_list[lp1]*thickness_list[lp1]))

        # S11 = inv(T11-d1*S12o*T12)*d1*S11o
        P1 = T11 - bd.dot(bd.dot(d1,S12),T12)
        P1 = bd.inv(P1)  # hold for further use
        S11 = bd.dot(bd.dot(P1,d1),S11)

        # S12 = P1*(d1*S12o*T11-T12)*d2
        P2 = bd.dot(d1,bd.dot(S12,T11))-T12
        S12 = bd.dot(bd.dot(P1,P2),d2)

        # S21 = S22o*T12*S11+S21o
        S21 = S21 + bd.dot(S22,bd.dot(T12,S11))

        # S22 = S22o*T12*S12+S22o*T11*d2
        P2 = bd.dot(S22,bd.dot(T12,S12))
        P1 = bd.dot(S22,bd.dot(T11,d2))
        S22 = P1 + P2
        
    return S11,S12,S21,S22

def SolveExterior(a0,bN,q_list,phi_list,kp_list,thickness_list):
    '''
    Given a0, bN, solve for b0, aN
    '''

    Nlayer = len(thickness_list) # total number of layers
    S11, S12, S21, S22 = GetSMatrix(0,Nlayer-1,q_list,phi_list,kp_list,thickness_list)

    aN = bd.dot(S11,a0) + bd.dot(S12,bN)
    b0 = bd.dot(S21,a0) + bd.dot(S22,bN)

    return aN,b0

def SolveInterior(which_layer,a0,bN,q_list,phi_list,kp_list,thickness_list):
    '''
    Given a0, bN, solve for ai, bi
    Layer numbering starts from 0
    '''
    Nlayer = len(thickness_list) # total number of layers
    nG2 = len(q_list[0])
    
    S11, S12, S21, S22 = GetSMatrix(0,which_layer,q_list,phi_list,kp_list,thickness_list)
    pS11, pS12, pS21, pS22 = GetSMatrix(which_layer,Nlayer-1,q_list,phi_list,kp_list,thickness_list)

    # tmp = inv(1-S12*pS21)
    tmp = bd.inv(bd.eye(nG2)-bd.dot(S12,pS21))
    # ai = tmp * (S11 a0 + S12 pS22 bN)
    ai = bd.dot(tmp,  bd.dot(S11,a0)+bd.dot(S12,bd.dot(pS22,bN)))
    # bi = pS21 ai + pS22 bN
    bi = bd.dot(pS21,ai) + bd.dot(pS22,bN)
    
    return ai,bi

def TranslateAmplitudes(q,thickness,dz,ai,bi):
    ai = ai*bd.exp(1j*q*dz)
    bi = bi*bd.exp(1j*q*(thickness-dz))
    return ai,bi
        

def GetZPoyntingFlux(ai,bi,omega,kp,phi,q,byorder=0):
    '''
     Returns 2S_z/A, following Victor's notation
     Maybe because 2* makes S_z = 1 for H=1 in vacuum
    '''
    n2 = len(ai)
    n = int(n2/2)
    # A = kp phi inv(omega*q)
    A = bd.dot(bd.dot(kp,phi),  bd.diag(1./omega/q))

    pa = bd.dot(phi,ai)
    pb = bd.dot(phi,bi)
    Aa = bd.dot(A,ai)
    Ab = bd.dot(A,bi)

    # diff = 0.5*(pb* Aa - Ab* pa)
    diff = 0.5*(bd.conj(pb)*Aa-bd.conj(Ab)*pa)
    #forward = real(Aa* pa) + diff
    forward_xy = bd.real(bd.conj(Aa)*pa) + diff
    backward_xy = -bd.real(bd.conj(Ab)*pb) + bd.conj(diff)

    forward = forward_xy[:n] + forward_xy[n:]
    backward = backward_xy[:n] + backward_xy[n:]
    if byorder == 0:
        forward = bd.sum(forward)
        backward = bd.sum(backward)

    return forward, backward

def Matrix_zintegral(q,thickness,shift=1e-12):
    ''' Generate matrix for z-integral
    '''
    nG2 = len(q)
    qi,qj = Gmeshgrid(q)

    # # Maa = \int exp(i q_i z)^* exp(i q_j z)
    # #     = [exp(i(q_j-q_i^*)t)-1]/i(q_j-q_i^*)
    # # Mbb = \int exp(i q_i (t-z))^* exp(i q_j (t-z))
    # #     = exp(i(q_j-q_i^*)t)  [exp(i(q_i^*-q_j)t)-1]/i(q_i^*-q_j)
    # #     = Maa
    # # Mab = \int exp(i q_i z)^* exp(i q_j (t-z))
    # #     = exp(iq_j t) [1-exp(-i(q_i^*+q_j)t)]/i(q_i^*+q_j)
    # #     = [exp(iq_j t)-exp(-i q_i^* t)]/i(q_i^*+q_j)
    # # Mba = \int exp(i q_i (t-z))^* exp(i q_j z)
    # #     = exp(-iq_i^* t) [exp(i(q_j+q_i^*)t)-1]/i(q_j+q_i^*)
    # #     = Mab
    # Maa = (np.exp(1j*(qj-np.conj(qi))*thickness)-1)/1j/(qj-np.conj(qi))
    # Mab = (np.exp(1j*qj*thickness)-np.exp(-1j*np.conj(qi)*thickness))/1j/(qj+np.conj(qi))

    # M = t exp(0.5it (qj-qi^*)) sinc(0.5d (sjqj-siqi^*), note in python sinc = sin(pi x)/pi x
    qij = qj-bd.conj(qi)+bd.eye(nG2)*shift
    Maa = (bd.exp(1j*qij*thickness)-1)/1j/qij # this is more robust
    #Maa = thickness * np.exp(0.5j*thickness*qij) * np.sinc(0.5*thickness*qij/np.pi)

    qij2 = qj+bd.conj(qi)
    #Mab = thickness * np.exp(0.5j*thickness*qij) * np.sinc(0.5*thickness*qij2/np.pi)
    Mab = (bd.exp(1j*qj*thickness)-bd.exp(-1j*bd.conj(qi)*thickness))/1j/qij2

    tmp1 = bd.vstack((Maa,Mab))
    tmp2 = bd.vstack((Mab,Maa))
    Mt = bd.hstack((tmp1,tmp2))
    return Mt

def Gmeshgrid(x):
    N = len(x)
    qj = []
    for i in range(N):
        qj.append(x)
    qj = bd.array(qj)
    qi = bd.transpose(qj)
    return qi,qj
