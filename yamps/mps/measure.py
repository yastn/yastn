from yamps.tensor import ncon

#from tensor.eigs import eigs
# from tensor.eigs import expmw

# List of changes:
# -- change 'direction' to 'towards' (a given site); this would allow easier extension to TTN
# -- similarly, use only indices of existing sites [or None in defining key for bond outside leaf]
# -- introduce class Geometry which deals with reltive positions of tensor sites
# -- remove settings/backend from here (class is agnostic to specific tensor and settings; it should only implement "structure" of mps)
# -- to the extend possible, do the above also with tensor [leaving tensor would make sense only if it is unique,
#     this might happen later---when there is a tensor with any number of symmetries (including 0)]
# -- divide envs into envs2 for <mps|mps> and  envs3 for <mps|mpo|mps>
#

#################################
#           dmrg                #
#################################


def dmrg_sweep_1site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, tol=1e-14, opts=None):
    """ Assume psi is in left cannonical form (=Q-R-)
    """
    if env is None:
        env = Envs3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep_to_last():
        psi.absorb_central(towards=psi.g.last)
        val, vec, happy = eigs(Av=lambda v: env.Heff1(v, n), init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        psi.A[n] = vec[val.index(min(val))]
        psi.orthogonalize_site(n, towards=psi.g.last)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep_to_first():
        psi.absorb_central(towards=psi.g.first§)
        val, vec, happy = eigs(Av=lambda v: env.Heff1(v, n), init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        psi.A[n] = vec[val.index(min(val))]
        psi.orthogonalize_site(n, towards=psi.g.first)
        env.update(n, towards=psi.g.first)

    return env  # can be used in the next sweep


def measure_entropy(psi):
    psi0 = psi.copy()
    entropy = np.zeros(psi0.N)
    Smin = np.zeros(psi0.N)
    no = np.zeros(psi0.N)
    for n in range(psi0.N - 1, -1, -1):
        entropy[n], Smin[n], no[n] = psi0.A[n].entropy(axes=(psi0.left + psi0.phys + psi0.aux, psi0.right), alpha=1)
        psi0.orth_right(n)
        if n != 0:
            psi0.absorb_central(direction=-1)
    return entropy, Smin, no


def measure_1site(psi, O, n=None, nor=None):
    r"""
    expectation value of 1-site operators
    """
    #  n - a list of sites to measure expectation value of the operator O
    #  default: n=None - all sites
    if nor is None:
        nor = Envs(layers=[psi, psi])
        nor.setup(direction=-1)
    norma = nor.F[(0, -1)].to_number().real
    if n is not None:
        N = len(n)
        ran = n
    else:
        N = psi.N
        ran = range(N)
    out = [None] * N
    
    if psi.nr_aux == 0:
        inds = [[3,1], [3,4,6], [4,2], [1,2,5], [5,6]]
    elif psi.nr_aux == 1:
        inds = [[3,1], [3,4,7,6], [4,2], [1,2,7,5], [5,6]]            
    n_old = 0
    for im in ran:
        n = ran[im]
        for it in range(n_old,n):
            nor.update(it, direction = +1)
        out[im] = ncon([nor.F[(n - 1,n)],psi.A[n],O,psi.A[n],nor.F[(n + 1,n)]],inds,[0,1,0,0,0])
        n_old = n
    return (out / norma)


def measure_2site(psi, OPs, n=None, nor=None):
    r"""
    expectation value of 1-site operators
    """

    #  n - a list of sites to measure expectation value of the operator O
    #  default: n=None - all sites
    #  two site formed by operator OL acting on site n and OR acting on n+1
    if nor == None:
        nor = Envs(layers=[psi, psi])
        nor.setup(direction=-1)
    norma = nor.F[(0, -1)].to_number().real
    
    if n != None:
        N = len(n)
        ran = n
    else:
        N = psi.N - 1
        ran = range(N)
    out = [None] * N
    if psi.nr_aux == 0:
        inds = [[3,1], [3,4,7], [4,11,2], [1,2,5], [7,8,10], [8,11,6], [5,6,9], [9,10]]
    elif psi.nr_aux == 1:
        inds = [[3,1], [3,4,12,7], [4,11,2], [1,2,12,5], [7,8,13,10], [8,11,6], [5,6,13,9], [9,10]]
    n_old = 0
    iran = 0
    for n in range(max(ran) + 1):
        for it in range(n_old,n):
            nor.update(it, direction = +1)
        if n in ran:
            out[iran] = ncon([nor.F[(n - 1,n)],psi.A[n],OPs[0],psi.A[n],psi.A[n + 1],OPs[1],psi.A[n + 1],nor.F[(n + 2,n + 1)]],inds,[0,1,0,0,1,0,0,0])
            iran+=1
        n_old = n
    return out #(out / norma)

