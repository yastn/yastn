from yamps.mps import Env3
from yamps.tensor import eigs

#################################
#           dmrg                #
#################################
def dmrg_sweep_0site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, eigs_tol=1e-14, opts_svd=None):
    """ 
    Perform sweep with 0site-DMRG where update is made on the central site.
    Assume input psi is right canonical. 
    Sweep consists of iterative updates from last site to first and back to the first one. 
    
    
    Parameters
    ----------
    psi: Mps, nr_phys=1, nr_aux=0 or 1
        Initial state.
        Can be gives as an MPS (nr_aux=0) or a purification (nr_aux=1).

    H: Mps, nr_phys=2
        Operator given in MPO decomposition. 
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]

    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
        
    dtype: str
        default='complex128'
        Type of Tensor.

    hermitian: bool
        default=True
        Is MPO hermitian
            
    k: int 
        default=4
        Dimension of Krylov subspace for eigs(.)
            
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
            
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    
    psi: Mps
        Is self updated.
    """

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last'): # sweep from fist to last 
        psi.orthogonalize_site(n, towards=psi.g.last)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)
        if n!=psi.g.sweep(to='last')[-1]:
            init = psi.A[psi.pC]
            # update site n using eigs
            if not hermitian:
                val, vec, _ = eigs(Av=lambda v: env.Heff0(v, psi.pC), Bv=lambda v: env.Heff0(v, psi.pC, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
            else:
                val, vec, _ = eigs(Av=lambda v: env.Heff0(v, psi.pC), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
            init = vec[list(val).index(min(list(val)))]
            # canonize and save
            if opts_svd != None:
                U, S, V = init.split_svd(axes=((0), (1)), sU=-1, **opts_svd)
                psi.A[psi.pC] = U.dot(S.dot(V, axes=((1),(0))) , axes=((1),(0)))
            else:
                psi.A[psi.pC] = init
        psi.absorb_central(towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)
        if n!=psi.g.sweep(to='first')[-1]:
            init = psi.A[psi.pC]
            # update site n using eigs
            if not hermitian:
                val, vec, _ = eigs(Av=lambda v: env.Heff0(v, psi.pC), Bv=lambda v: env.Heff0(v, psi.pC, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
            else:
                val, vec, _ = eigs(Av=lambda v: env.Heff0(v, psi.pC), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
            init = vec[list(val).index(min(list(val)))]
            # canonize and save
            if opts_svd != None:
                U, S, V = init.split_svd(axes=((0), (1)), sU=-1, **opts_svd)
                psi.A[psi.pC] = U.dot(S.dot(V, axes=((1),(0))) , axes=((1),(0)))
            else:
                psi.A[psi.pC] = init
        psi.absorb_central(towards=psi.g.first)

    return env


def dmrg_sweep_1site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, eigs_tol=1e-14, opts_svd=None):
    """ 
    Perform sweep of single-site DMRG. 
    Assume input psi is right canonical. 
    Sweep consists of iterative updates from last site to first and back to the first one. 
    
    Parameters
    ----------
    psi: Mps, nr_phys=1, nr_aux=0 or 1
        Initial state.
        Can be gives as an MPS (nr_aux=0) or a purification (nr_aux=1).

    H: Mps, nr_phys=2
        Operator given in MPO decomposition. 
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]

    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
        
    dtype: str
        default='complex128'
        Type of Tensor.

    hermitian: bool
        default=True
        Is MPO hermitian
            
    k: int 
        default=4
        Dimension of Krylov subspace for eigs(.)
            
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
            
    opts_svd: dict
        default=None
        options for truncation
    
    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    
    psi: Mps
        Is self updated.
    """

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last'): # sweep from fist to last 
        psi.absorb_central(towards=psi.g.last)
        init = psi.A[n]
        # update site n using eigs
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff1(v, n), Bv=lambda v: env.Heff1(v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff1(v, n), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        # canonize 
        if opts_svd != None:
            U, S, V = init.split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), sU=-1, **opts_svd)
            psi.A[n] = V
            psi.pC = (n - 1,n)
            psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
        else:
            psi.A[n] = init
            psi.orthogonalize_site(n, towards=psi.g.last)
        # update environment
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        psi.absorb_central(towards=psi.g.first)
        init = psi.A[n]
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff1(v, n), Bv=lambda v: env.Heff1(v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff1(v, n), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        # canonize and save
        if opts_svd != None:
            U, S, V = init.split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), sU=-1, **opts_svd)
            psi.A[n] = U
            psi.pC = (n,n+1)
            psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
        else:
            psi.A[n] = init
            psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)

    return env
    

def dmrg_sweep_2site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, eigs_tol=1e-14, opts_svd={}):
    """ 
    Perform sweep of two-site DMRG. 
    Assume input psi is right canonical. 
    Sweep consists of iterative updates from last site to first and back to the first one. 
    
    Parameters
    ----------
    psi: Mps, nr_phys=1, nr_aux=0 or 1
        Initial state.
        Can be gives as an MPS (nr_aux=0) or a purification (nr_aux=1).

    H: Mps, nr_phys=2
        Operator given in MPO decomposition. 
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]

    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
        
    dtype: str
        default='complex128'
        Type of Tensor.

    hermitian: bool
        default=True
        Is MPO hermitian
            
    k: int 
        default=4
        Dimension of Krylov subspace for eigs(.)
            
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
            
    opts_svd: dict
        default=None
        options for truncation
    
    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    
    psi: Mps
        Is self updated.
    """

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        # update site n using eigs
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff2(v, n), Bv=lambda v: env.Heff2(v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff2(v, n), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        #split and save
        A1, S, A2 = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = A1
        psi.A[n1] = A2.dot_diag(S, axis=0)
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        # update site n using eigs
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff2(v, n), Bv=lambda v: env.Heff2(v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff2(v, n), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        #split and save
        A1, S, A2 = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = A1.dot_diag(S, axis=2)
        psi.A[n1] = A2
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n1, towards=psi.g.first)
    env.update(n, towards=psi.g.first)

    return env  # can be used in the next sweep


def dmrg_sweep_2site_group(psi, H, env=None, dtype='complex128', hermitian=True, k=4, eigs_tol=1e-14, opts_svd={}):
    """ 
    Perform sweep of two-site DMRG with groupping neigbouring sites. 
    Assume input psi is right canonical. 
    Sweep consists of iterative updates from last site to first and back to the first one. 
    
    Parameters
    ----------
    psi: Mps, nr_phys=1, nr_aux=0 or 1
        Initial state.
        Can be gives as an MPS (nr_aux=0) or a purification (nr_aux=1).

    H: Mps, nr_phys=2
        Operator given in MPO decomposition. 
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]

    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
        
    dtype: str
        default='complex128'
        Type of Tensor.

    hermitian: bool
        default=True
        Is MPO hermitian
            
    k: int 
        default=4
        Dimension of Krylov subspace for eigs(.)
            
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
            
    opts_svd: dict
        default=None
        options for truncation
    
    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    
    psi: Mps
        Is self updated.
    """

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
        # update site n using eigs
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff2_group(v, n), Bv=lambda v: env.Heff2_group(v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff2_group(v, n), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        init = init.ungroup_leg(axis=1, leg_order=leg_order)
        #split and save
        A1, S, A2 = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = A1
        psi.A[n1] = A2.dot_diag(S, axis=0)
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
        # update site n using eigs
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff2_group(v, n), Bv=lambda v: env.Heff2_group(v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff2_group(v, n), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        init = init.ungroup_leg(axis=1, leg_order=leg_order)
        #split and save
        A1, S, A2 = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = A1.dot_diag(S, axis=2)
        psi.A[n1] = A2
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n1, towards=psi.g.first)
    env.update(n, towards=psi.g.first)

    return env  # can be used in the next sweep
