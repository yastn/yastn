from tens.utils import ncon
from eigs import eigs
from eigs import expmw
import settings
from settings import backend
from settings import Tensor
import numpy
import scipy as sp

# change heff to use internal psi.pC etc..
def mult(x):
    out = 1
    if  isinstance(x, list) or isinstance(x, tuple):
        for it in range(len(x)):
            out*=x[it]
    else:
        out = x
    return out
    
def rand(setup, nr_aux): # think of a good way to initialise various mps-s for symmetric tensors
    N = len(setup)
    psi = Mps(N, nr_aux=nr_aux)
    psi = Mps(N, nr_aux=nr_aux)
    for ii in range(N):
        if nr_aux == 0:
            psi.A[ii] = Tensor.rand(settings=settings, s=[1, 1, -1], n=0, **setup[ii])
        elif nr_aux == 1:
            psi.A[ii] = Tensor.rand(settings=settings, s=[1, 1, -1, -1], n=0, **setup[ii])
    return psi

def ones(setup, nr_aux): # think of a good way to initialise various mps-s for symmetric tensors
    N = len(setup)
    psi = Mps(N, nr_aux=nr_aux)
    for ii in range(N):
        if nr_aux == 0:
            psi.A[ii] = Tensor.ones(settings=settings, s=[1, 1, -1], n=0, **setup[ii])
        elif nr_aux == 1:
            psi.A[ii] = Tensor.ones(settings=settings, s=[1, 1, -1, -1], n=0, **setup[ii])
    return psi

"""
            ####################     DMRG SECTION     ####################
"""

def idmrg(psi, H,env=None,  nor=None, measure_O=None, cutoff_sweep=20, cutoff_dE=1e-9, cutoff_dS=1e-9, hermitian=True, k=8, eigs_tol=1e-14, dtype='complex128', version='central', opts=None):
    """
    Perform infinite-DMRG

    Output is [out, E, dE,dS, N, spec_L0, env, nor] 
    with:
    out - list given by measure_O,
    E-energy, N-norm, 
    dE, dS - changes of energy end spectrum, 
    spec_L0-spectrum, 
    env, nor - outputenvironments

    Parameters
    ----------
    psi: Mps, nr_phys=1
        initial state
        it can be MPS (nr_aux=0) or purification (nr_aux=1)
        initial MPS have to be central canonical

    H: list
        list of Hamiltonians of type Mps , nr_phys=2
        H=[H1, H0], where H0 - MPO with corner sites as for finit DMRG and H1 - bulk MPO

    env: Envs
        default = None
        initial Env( layers=[psi,H,psi] )
        initial environments have to be setup with respect to central
       
    nor: Envs
        default = None
        initial Env( layers=[psi,psi] )
        initial environments have to be setup with respect to central
       
    measure_O: list 
        default=None
        list of things to measure
        1-site and 2-site operators only
        For 1-site: measute_O[.] =[1,Op,list_n]
        For 2-site: measute_O[.] =[2,Op,list_n], Op = [Op1, Op2] 
        Op - operator, list_n - sites to measure
        
    cutoff_sweep: int
        default=20
        number of idmrg sweeps
        
    cutoff_dE: float
        default=1e-9
        cutoof on energy variance
            
    cutoff_dS: flaot
        default=1e-9
        cutoof on central site Schmidt values changes

    hermitian: bool
        default=True
        is MPO hermitian
            
    k: int 
        default=8
        dimension of Krylov subspace for eigs(.)
            
    eigs_tol: float
        default=1e-14
        cutoff for krylov subspace for eigs(.)
            
    dtype: str
        default='complex128'
        type of Tensors
            
    version: str
        default='central'
        version of dmrg sweep
        '2site', '1site', 'central' (update only central matrix before regauging)
    opts: dict
        default=None
        options for truncation
    """
    current_size = psi.N
    H, Hx = H[0], H[1]
    
    # initialize psi, env, nor
    central = (int(psi.N / 2 - 1), int(psi.N / 2)) # pointer to central site , for N=2, central = (2/2-1,2/2) = (0,1)
    if not env and not nor:
        psi.canonize(pC=central, normalize=True)
        L0 = psi.A[psi.pC]
        psi.absorb_central(direction=+1)
    
    spec_Lm = numpy.zeros(1)
    # setup env-s , energy and norm
    if not env:
        env = Envs(layers=[psi, Hx, psi])
        env.setup(pC=central)
    if not nor:
        nor = Envs(layers=[psi, psi])
        nor.setup(pC=central)
    
    dmrg = False # it starts from regauging

    iter, E0 = 0, 0
    dE, dS = cutoff_dE + 1, cutoff_dS + 1
    while ((iter < cutoff_sweep or iter % 2 == 1) and (dE > cutoff_dE or dS > cutoff_dS)):
        # --------------- DMRG update ---------------
        # update central > right > left > central
        if dmrg:
            if version == '2site':

                for n in range(min(central),psi.N - 1): # it is 2site update # sweep in direction = -1
                    psi.pC = (n, n + 1)
                    # update site n using eigs
                    init = psi.merge_mps(psi.pC)
                    Hv = lambda v: env.Heff2(v, psi.pC)
                    if not hermitian:
                        Hv_dag = lambda v : env.Heff2(v, psi.pC, conj=True)
                        val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=eigs_tol, k=k, hermitian=False, dtype=dtype)
                    else:
                        val, vec, happy = eigs(Av=Hv, init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
                    val = list(val)
             
                    init = vec[val.index(min(val))]
                    #split mps using SVD and use truncation
                    U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), opts=opts)
                    psi.A[n] = U
                    psi.A[psi.pC] = S
                    psi.A[n + 1] = V
                    psi.absorb_central(direction=+1)
                    # update environment
                    env.update(n, direction = +1)
                    nor.update(n, direction = +1)
        
                for n in range(psi.N - 1, 0, -1): # it is a 2-site update # sweep in direction = -1
                    psi.pC = (n - 1, n)
                    # update site n using eigs
                    init = psi.merge_mps(psi.pC)        
                    Hv = lambda v: env.Heff2(v, psi.pC)
        
                    if not hermitian:
                        Hv_dag = lambda v : env.Heff2(v, psi.pC, conj=True)
                        val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=eigs_tol, k=k, hermitian=False, dtype=dtype)
                    else:
                        val, vec, happy = eigs(Av=Hv, init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
                    val = list(val)
            
                    init = vec[val.index(min(val))]
                    #split mps using SVD and use truncation
                    U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), opts=opts)
                    psi.A[n - 1] = U
                    psi.A[psi.pC] = S
                    psi.A[n] = V
                    psi.absorb_central(direction=-1)
                    # update environment
                    env.update(n, direction = -1)
                    nor.update(n, direction = -1)
        
        
                for n in range(min(central) + 1): # it is 2site update # sweep in direction = -1
                    psi.pC = (n, n + 1)
                    # update site n using eigs
                    init = psi.merge_mps(psi.pC)
                    Hv = lambda v: env.Heff2(v, psi.pC)
                    if not hermitian:
                        Hv_dag = lambda v : env.Heff2(v, psi.pC, conj=True)
                        val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=eigs_tol, k=k, hermitian=False, dtype=dtype)
                    else:
                        val, vec, happy = eigs(Av=Hv, init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
                    val = list(val)
             
                    init = vec[val.index(min(val))]
                    #split mps using SVD and use truncation
                    U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), opts=opts)
                    psi.A[n] = U
                    psi.A[psi.pC] = S
                    psi.A[n + 1] = V
                    psi.absorb_central(direction=+1)
                    # update environment
                    env.update(n, direction = +1)
                    nor.update(n, direction = +1)
            
            elif version == '1site':
            
                for n in range(min(central),psi.N - 1): # it is 2site update # sweep in direction = -1
                    # update site n using eigs
                    init = psi.A[n]
                    Hv = lambda v : env.Heff1(v, n)
                    if not hermitian:
                        Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                        val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=eigs_tol, k=k, hermitian=False, dtype=dtype)
                    else:
                        val, vec, happy = eigs(Av=Hv, init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
                    val = list(val)
                    init = vec[val.index(min(val))]

                    # truncate if truncation options defined
                    if opts != None:
                        U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, psi.right), opts=opts)
                        psi.A[n] = U
                        psi.pC = (n,n + 1)
                        psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
                    else:
                        psi.A[n] = init
                        # left cannonise site n
                        psi.orth_left(n)

                    psi.absorb_central(direction=+1)
                    env.update(n, direction = +1)
                    nor.update(n, direction = +1)

                for n in range(psi.N - 1, 0, -1): # it is a 2-site update # sweep in direction = -1
                    # update site n using eigs
                    init = psi.A[n]
                    if n == psi.N - 1:
                        env.update(n, direction = -1)
                    Hv = lambda v : env.Heff1(v, n)
        
                    if not hermitian:
                        Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                        val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=eigs_tol, k=k, hermitian=False, dtype=dtype)
                    else:
                        val, vec, happy = eigs(Av=Hv, init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        
                    val = list(val)
                    init = vec[val.index(min(val))]
                    # truncate if truncation options defined
                    if opts != None:
                        U, S, V = init.split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), opts=opts)
                        psi.A[n] = V
                        psi.pC = (n - 1,n)
                        psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
                    else:
                        psi.A[n] = init
                        # right cannonise site n
                        psi.orth_right(n)
                        # absorb_central
                    psi.absorb_central(direction=-1)
                    env.update(n, direction = -1)
                    nor.update(n, direction = -1)
        
        
                for n in range(min(central) + 1): # it is 2site update # sweep in direction = -1
                    # update site n using eigs
                    init = psi.A[n]
                    Hv = lambda v : env.Heff1(v, n)
                    if not hermitian:
                        Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                        val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=eigs_tol, k=k, hermitian=False, dtype=dtype)
                    else:
                        val, vec, happy = eigs(Av=Hv, init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
                    val = list(val)
                    init = vec[val.index(min(val))]

                    # truncate if truncation options defined
                    if opts != None:
                        U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, psi.right), opts=opts)
                        psi.A[n] = U
                        psi.pC = (n,n + 1)
                        psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
                    else:
                        psi.A[n] = init
                        psi.orth_left(n)

                    psi.absorb_central(direction=+1)
                    env.update(n, direction = +1)
                    nor.update(n, direction = +1)
            
        elif version == 'central':
        
                n = max(central)
                psi.orth_right(n)
                env.update(n, direction = -1)
        
                pC = psi.pC
                init = psi.A[pC]
                Hv = lambda v : env.Heff0(v, pC)
        
                if not hermitian:
                    Hv_dag = lambda v : env.Heff0(v, pC, conj=True)
                    val, vec, _ = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=eigs_tol, k=k, hermitian=False, dtype=dtype)
                else:
                    val, vec, _ = eigs(Av=Hv, init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        
                val = list(val)
                init = vec[val.index(min(val))]
                # truncate if truncation options defined
                if opts != None:
                    U, S, V = init.split_svd(axes=((0),(1)), opts=opts)
                    init = U.dot(S.dot(V, axes=((1),(0))) , axes=((1),(0)))
                psi.A[pC] = init
                psi.absorb_central(direction=+1)
                env.update(n, direction = -1)
                nor.update(n, direction = -1)
                
        # --------------- regauging ---------------
        if iter < 1:
            regauge_iter = 1 # regauge_iter - for version='2site'
                             #should be odd to make dmrg update on both sides
                                                         #of MPS cell
        else:
            regauge_iter = 1

        dmrg = True
        for ir in range(regauge_iter):
            psi.orth_right(max(central))
            L1 = psi.A[psi.pC].copy()
            L1 = L1 * (1. / L1.norm())

            if iter == 0 and ir == 0:
                env.layers[1] = H
            
            # ROTATE LEFT
            for n in range(min(central),psi.N): 
                psi.absorb_central(direction=+1)
                psi.orth_left(n)
                env.update(n, direction = +1)
                nor.update(n, direction = +1)
            
            # output of the sweep
            L_L = psi.A[psi.pC].copy()
            new_L = env.F[central].copy()
            new_NL = nor.F[central].copy()
            new_A = [psi.A[it].copy() for it in range(max(central),psi.N)]
            new_HA = [env.layers[1].A[it].copy() for it in range(max(central),psi.N)]
            
            if iter == 0 and ir == 0:
                env.layers[1] = Hx # I replace Hx-H twice to avoid making copy of env.Could be done better?
            
            # ROTATE RIGHT
            for n in range(psi.N - 1, -1, -1):
                psi.absorb_central(direction=-1)
                psi.orth_right(n)
                if iter == 0 and ir == 0:
                    if n < min(central):
                        env.layers[1] = H
                env.update(n, direction = -1)
                nor.update(n, direction = -1)
            
            # output of the sweep
            L_R = psi.A[psi.pC].copy()
            new_R = env.F[central[::-1]].copy() 
            new_NR = nor.F[central[::-1]].copy()
            new_B = [ psi.A[it].copy() for it in range(min(central) + 1) ] 
            new_HB = [ env.layers[1].A[it].copy() for it in range(min(central) + 1) ] 
            
            # UPDATE : expand the lattice and regauge the working area
            env.F[(-1,0)] = new_L
            nor.F[(-1,0)] = new_NL
            env.F[(psi.N,psi.N - 1)] = new_R
            nor.F[(psi.N,psi.N - 1)] = new_NR
            
            for it in range(min(central) + 1):
                psi.A[it] = new_A[it]
                env.layers[1].A[it] = new_HA[it]
            
                env.update(it, direction=+1)
                nor.update(it, direction=+1)
                
                psi.A[psi.N - 1 - it] = new_B[- 1 - it]
                env.layers[1].A[psi.N - 1 - it] = new_HB[- 1 - it]
                
                env.update(psi.N - 1 - it, direction=-1)
                nor.update(psi.N - 1 - it, direction=-1)
            
            # create initial guess for central matrix
            if iter == 0 and ir == 0:
                L0 = L1
                
            # pinv(L0)
            u, s, v = L0.split_svd(axes=(0,1), opts=opts)
            Lm1 = v.transpose(axes=(1,0)).conj().dot(s.inv(), axes=((1),(0)))
            Lm1 = Lm1.dot(u.transpose(axes=(1,0)).conj(), axes=((1),(0)))
            
            L = L_L.dot(Lm1, axes=((1),(0))).dot(L_R, axes=((1),(0)))
            L*=(1. / L.norm())

            s*=(1. / s.norm())
            spec_L0 = numpy.sort(s.to_numpy().diagonal())[::-1]
           
            psi.pC = central
            psi.A[psi.pC] = L
            psi.absorb_central(direction=+1)
            current_size+=psi.N
            L0 = L1
            #print('Central spectrum: ',spec_L0[range(min([len(spec_L0),2]))])

        mini = min([len(spec_L0), len(spec_Lm)])
        dS = sum(abs(spec_Lm[range(mini)] - spec_L0[range(mini)])) / mini
        spec_Lm = spec_L0

        env.update(max(central), direction=-1)
        nor.update(max(central), direction=-1)
        E = env.F[central].dot(env.F[central[::-1]], axes=((0,1,2),(2,1,0))).to_number().real
        N = nor.F[central].dot(nor.F[central[::-1]], axes=((0,1),(1,0))).to_number().real
        E*=1. / current_size * psi.N /N
        dE = abs(E - E0)

        for it in range(max(central), psi.N):
            nor.update(psi.N - 1 - it, direction=-1)
        if measure_O and iter % 2 == 0:
            out = measure(psi=psi, nor=nor, measure_O=measure_O)
        print('Iteration:',iter, ' Energy per unit cell:' ,round(E , 6) , ' dE:' ,round(dE , 6) , ' dS:' ,dS , ' Norma:',round(N.real,4),' Site occupations:',[round(it.real/N, 5) for it in out[0]])
        E0 = E
        iter+=1
    
    n = max(central)
    psi.entropy, psi.Smin, psi.no = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.aux, psi.right), alpha=1)
    if psi.nr_aux == 1:
        psi.entropy_aux, psi.Smin_aux, psi.no_aux = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.right, psi.aux), alpha=1)
    
    return [out, E, dE,dS, N, spec_L0, env, nor]

def dmrg_OBC(psi, H, env=None, nor=None, measure_O=None, cutoff_sweep=20, cutoff_dE=1e-9, hermitian=True, k=8, tol=1e-14, dtype='complex128', version='1site', opts=None):
    # meaure_O - list of things to measure e.g.  [2,[OL, OR], [1,2,3]] -
    # measure exp.  val of 2-site operator OL, OR on sites (1,2), (2,3), (3,4)
    # opts - optional info for MPS truncation
    #  nor - dummy - is not used in DMRG
    E0 = 0
    dE = cutoff_dE + 1
    sweep = 0
    while sweep < cutoff_sweep and dE > cutoff_dE:
        if version == '0site':
            env = dmrg_sweep_0site(psi=psi, H=H, env=env, dtype=dtype, k=k, hermitian=hermitian, tol=tol, opts=opts)
        elif version == '2site':
            env = dmrg_sweep_2site(psi=psi, H=H, env=env, dtype=dtype, k=k, hermitian=hermitian, tol=tol, opts=opts)
        else:
            env = dmrg_sweep_1site(psi=psi, H=H, env=env, dtype=dtype, k=k, hermitian=hermitian, tol=tol, opts=opts)
        
        #psi.Dmax is maximal virtual dimansion of physical block
        psi.get_Dmax()
            
        norma = psi.Norma
        E = env.F[(psi.N - 1,psi.N)].to_number() / norma
        dE = abs(E - E0)
        if measure_O != None:
            measured = [None] * len(measure_O)
            for it in range(len(measure_O)):
                tmp = measure_O[it]
                if len(tmp) > 2:
                    n = tmp[2]
                else:
                    n = None
                if tmp[0] == 1:
                    measured[it] = measure_1site(psi, tmp[1], n=n, nor=nor) / norma
                elif tmp[0] == 2:
                    tmp = measure_2site(psi, tmp[1], n=n, nor=nor)
                    measured[it] = tmp # norma assumed to be 1
        E0 = E 
        print('Iteration: ', sweep,' energy: ',E, ' dE: ',dE, ' norma:', norma,' D: ',psi.Dmax)
        sweep+=1
        #create results' list
        out = ()
        if measure_O != None:
            out+=(measured,)
    #psi updated in place
    return env, E, dE, out


def dmrg_sweep_0site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, tol=1e-14, opts=None):
    """ Assume psi is in left cannonical form (=Q-R-)
    """
    if env == None:  # setup environments
        env = Envs(layers=[psi, H, psi])
        env.setup(direction=+1)

    for n in range(psi.N - 1, 0, -1): # sweep in direction = -1
        # update site n using eigs
        psi.orth_right(n)
        env.update(n, direction = -1)
        
        pC = psi.pC
        init = psi.A[pC]
        Hv = lambda v : env.Heff0(v, pC)
        
        if not hermitian:
            Hv_dag = lambda v : env.Heff0(v, pC, conj=True)
            val, vec, _ = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=tol, k=k, hermitian=False, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=Hv, init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        
        val = list(val)
        init = vec[val.index(min(val))]
        # truncate if truncation options defined
        if opts != None:
            U, S, V = init.split_svd(axes=((0),(1)), opts=opts)
            init = U.dot(S.dot(V, axes=((1),(0))) , axes=((1),(0)))
        psi.A[pC] = init
        # absorb_central
        if n != 0:
            psi.absorb_central(direction=-1)
        else:
            psi.pC = None
            #normalize
            normA0 = psi.A[n].norm()
            psi.A[n] = (1 / normA0) * psi.A[n]

    for n in range(psi.N):  # sweep in direction = +1
        # update site n using eigs
        psi.orth_left(n)
        env.update(n, direction = +1)
        
        pC = psi.pC
        init = psi.A[pC]
        Hv = lambda v : env.Heff0(v, pC)
        
        if not hermitian:
            Hv_dag = lambda v : env.Heff0(v, pC, conj=True)
            val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=tol, k=k, hermitian=False, dtype=dtype)
        else:
            val, vec, happy = eigs(Av=Hv, init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        
        val = list(val)
        init = vec[val.index(min(val))]
        # truncate if truncation options defined
        if opts != None:
            U, S, V = init.split_svd(axes=((0),(1)), opts=opts)
            init = U.dot(S.dot(V, axes=((1),(0))) , axes=((1),(0)))
        psi.A[pC] = init
        
        # absorb_central
        if n != psi.N - 1:
            psi.absorb_central(direction=+1)
        else:
            psi.pC = None
            #normalize
            normA0 = psi.A[n].norm()
            psi.A[n] = (1 / normA0) * psi.A[n]
        if n == round(psi.N / 2):
            psi.entropy, psi.Smin, psi.no = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.aux, psi.right), alpha=1)
            if psi.nr_aux == 1:
                psi.entropy_aux, psi.Smin_aux, psi.no_aux = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.right, psi.aux), alpha=1)



    # psi is updated in place
    return env # can be used in the next sweep
def dmrg_sweep_1site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, tol=1e-14, opts=None):
    """ Assume psi is in left cannonical form (=Q-R-)
    """
    if env == None:  # setup environments
        env = Envs(layers=[psi, H, psi])
        env.setup(direction=+1)

    for n in range(psi.N - 1, 0, -1): # sweep in direction = -1
        # update site n using eigs
        init = psi.A[n]
        if n == psi.N - 1:
            env.update(n, direction = -1)
        Hv = lambda v : env.Heff1(v, n)
        
        if not hermitian:
            Hv_dag = lambda v : env.Heff1(v, n, conj=True)
            val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=tol, k=k, hermitian=False, dtype=dtype)
        else:
            val, vec, happy = eigs(Av=Hv, init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        
        val = list(val)
        init = vec[val.index(min(val))]
        # truncate if truncation options defined
        if opts != None:
            U, S, V = init.split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), opts=opts)
            psi.A[n] = V
            psi.pC = (n - 1,n)
            psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
        else:
            psi.A[n] = init
            # right cannonise site n
            psi.orth_right(n)
            # absorb_central
        if n != 0:
            psi.absorb_central(direction=-1)
        else:
            psi.pC = None
            #normalize
            normA0 = psi.A[n].norm()
            psi.A[n] = (1 / normA0) * psi.A[n]
        # update environment
        env.update(n, direction = -1)
            
    for n in range(psi.N):  # sweep in direction = +1
        # update site n using eigs
        init = psi.A[n]
        if n == 0:
            env.update(n, direction = +1)
        Hv = lambda v : env.Heff1(v, n)
        if not hermitian:
            Hv_dag = lambda v : env.Heff1(v, n, conj=True)
            val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=tol, k=k, hermitian=False, dtype=dtype)
        else:
            val, vec, happy = eigs(Av=Hv, init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        init = vec[val.index(min(val))]

        # truncate if truncation options defined
        if opts != None:
            U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, psi.right), opts=opts)
            psi.A[n] = U
            psi.pC = (n,n + 1)
            psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
        else:
            psi.A[n] = init
            # left cannonise site n
            psi.orth_left(n)

        if n == round(psi.N / 2):
            psi.entropy, psi.Smin, psi.no = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.aux, psi.right), alpha=1)
            if psi.nr_aux == 1:
                psi.entropy_aux, psi.Smin_aux, psi.no_aux = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.right, psi.aux), alpha=1)

        # absorb_central
        if n != psi.N - 1:
            psi.absorb_central(direction=+1)
        else:
            psi.pC = None
            #normalize
            normA0 = psi.A[n].norm()
            psi.A[n] = (1 / normA0) * psi.A[n]
        # update environment
        env.update(n, direction = +1)

    # psi is updated in place
    return env # can be used in the next sweep
def dmrg_sweep_2site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, tol=1e-14, opts=None, central=False):
    """ Assume psi is in left cannonical form (=Q-R-)
    """
    if env == None:  # setup environments
        env = Envs(layers=[psi, H, psi])
        env.setup(direction=+1)

    for n in range(psi.N - 1, 0, -1): # sweep in direction = -1
        psi.pC = (n - 1, n)
        # update site n using eigs
        if n == psi.N - 1:
            env.update(n, direction = -1)
        init = psi.merge_mps(psi.pC)        
        Hv = lambda v: env.Heff2(v, psi.pC)
        
        if not hermitian:
            Hv_dag = lambda v : env.Heff2(v, psi.pC, conj=True)
            val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=tol, k=k, hermitian=False, dtype=dtype)
        else:
            val, vec, happy = eigs(Av=Hv, init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        
        init = vec[val.index(min(val))]
        #split mps using SVD and use truncation
        U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), opts=opts)
        psi.A[n - 1] = U
        psi.A[psi.pC] = S
        psi.A[n] = V
        psi.absorb_central(direction=-1)
        # update environment
        env.update(n, direction = -1)
        
    for n in range(psi.N - 1): # sweep in direction = -1
        psi.pC = (n, n + 1)
        # update site n using eigs
        if n == 0:
            env.update(n, direction = +1)
        
        init = psi.merge_mps(psi.pC)
        Hv = lambda v: env.Heff2(v, psi.pC)
        
        if not hermitian:
            Hv_dag = lambda v : env.Heff2(v, psi.pC, conj=True)
            val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=tol, k=k, hermitian=False, dtype=dtype)
        else:
            val, vec, happy = eigs(Av=Hv, init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val, vec, happy = eigs(Av=Hv, init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        
        init = vec[val.index(min(val))]
        #split mps using SVD and use truncation
        U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), opts=opts)
        psi.A[n] = U
        psi.A[psi.pC] = S
        psi.A[n + 1] = V
        psi.absorb_central(direction=+1)

        # update environment
        env.update(n, direction = +1)

        if n == round(psi.N / 2):
            psi.entropy, psi.Smin, psi.no = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.aux, psi.right), alpha=1)
            if psi.nr_aux == 1:
                psi.entropy_aux, psi.Smin_aux, psi.no_aux = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.right, psi.aux), alpha=1)



    for n in [psi.N - 1]: #optimalize last site with 1site dmrg
        init = psi.A[n]
        Hv = lambda v : env.Heff1(v, n)
        
        if not hermitian:
            Hv_dag = lambda v : env.Heff1(v, n, conj=True)
            val, vec, happy = eigs(Av=Hv, Bv=Hv_dag, init=[init], tol=tol, k=k, hermitian=False, dtype=dtype)
        else:
            val, vec, happy = eigs(Av=Hv, init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)

        val = list(val)
        init = vec[val.index(min(val))]
        psi.A[n] = init

        # left cannonise site n
        psi.orth_left(n)
        # absorb_central
        psi.pC = None
        #normalize
        normA0 = psi.A[n].norm()
        psi.A[n] = (1 / normA0) * psi.A[n]
        # update environment
        env.update(n, direction = +1)
    
    # psi is updated in place
    return env # can be used in the next sweep
"""
####################     TDVP SECTION ####################
"""


def tdvp_OBC(psi, tmax, dt=1, H=False, M=False, env=None, measure_O=None, cutoff_sweep=20, cutoff_dE=1e-9, hermitian=True, fermionic=False, k=8, tol=1e-14, dtype='complex128', biorth=True, NA=None, version='1site', opts=None, optsK=None, disentangle=False):
    #evolve with TDVP method, up to tmax and initial guess of the time step dt
    # meaure_O - list of things to measure e.g.  [2,[OL, OR], [1,2,3]] -
    # measure exp.  val of 2-site operator OL, OR on sites (1,2), (2,3), (3,4)
    # opts - optional info for MPS truncation
    sweep = 0
    curr_t = 0
    if env == None and H:
        env = Envs(layers=[psi, H, psi])
        env.setup(direction=+1)
        
    if H:
        E0 = env.F[(psi.N - 1,psi.N)].to_number()
        dE = cutoff_dE + 1
    else:
        E0, dE = 0, 0
    
    sgn = 1j * (dt.real + 1j * (dt.imag)) / abs(dt)
    while abs(curr_t) < abs(tmax):
        dt = min([abs(tmax - curr_t), abs(dt)])

        if not H and not M:
            print('yamps.tdvp: Neither Hamiltonian nor Kraus operators defined.')
        else:
            if version == '0site':
                raise YampsError('yamps.tdvp: Only 1-site and 2-site versions defined.')
            elif version == '2site':
                env = tdvp_sweep_2site(psi=psi, H=H, M=M, dt= sgn * dt, env=env, dtype=dtype, k=k, hermitian=hermitian, fermionic=fermionic, tol=tol, biorth=biorth, NA=NA, opts=opts, optsK=optsK, disentangle=disentangle)        
            else:
                env = tdvp_sweep_1site(psi=psi, H=H, M=M, dt= sgn * dt, env=env, dtype=dtype, k=k, hermitian=hermitian, fermionic=fermionic, tol=tol, biorth=biorth, NA=NA, opts=opts, optsK=optsK)                
            
        norma = psi.Norma

        #psi.Dmax is maximal virtual dimansion of physical block
        psi.get_Dmax()
        
        if H:
            E = env.F[(psi.N - 1,psi.N)].to_number() / norma
        else:
            E = 0
        dE = abs(E - E0)
        if measure_O != None:
            measured = [None] * len(measure_O)
            for it in range(len(measure_O)):
                tmp = measure_O[it]
                if len(tmp) > 2:
                    n = tmp[2]
                else:
                    n = None
                if tmp[0] == 1:
                    measured[it] = measure_1site(psi, tmp[1], n=n, nor=nor)
                elif tmp[0] == 2:
                    measured[it] = measure_2site(psi, tmp[1], n=n, nor=nor)   
        E0 = E 
        print('Iteration: ', sweep,' energy: ',E, ' dE: ',dE,' time: ', ' norma:', norma, curr_t,' D: ',psi.Dmax,' Smin: ',psi.Smin,' S: ',psi.entropy,' S_aux: ',psi.entropy_aux)
        sweep+=1
        curr_t+=dt
        #create results' list
        out = (env, E,dE,)
        if measure_O != None:
            out+=(measured,)
    #psi updated in place
    return out 

def tdvp_sweep_1site(psi, H=False, M=False, dt=1, env=None, dtype='complex128', k=4, hermitian=True, fermionic=False, tol=1e-14, biorth=True, NA=None, opts=None, optsK=None):
    # Assume psi is in left cannonical form (=Q-R-)
    #    default bioth=True - biothogonalize Krylov vectors if is non-Hermitian
    #    version
    #    NA - cost of single Av(init) contraction < < < err?
    #    evolve the system with the time dt
    
    if H and not env:  # setup environments
        env = Envs(layers=[psi, H, psi])
        env.setup(direction=+1)

    for n in range(psi.N - 1, -1, -1): # sweep in direction = -1
    # forward in time evolution of a single site: T(+dt*.5)
        if M: # Apply the Kraus operator
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK) # discard V
            init = u.dot(s, axes=((3,),(0,)))
            psi.A[n] = init.transpose(axes=(0,1,3,2))

            if not H:
                U, S, V = psi.A[n].split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), opts=opts)
                psi.A[n] = V
                psi.pC = (n - 1,n)
                psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
                if n != 0:
                    psi.absorb_central(direction=-1)
                else:
                    pC = None
        if H:
            init = psi.A[n]
            Hv = lambda v : env.Heff1(v, n)
            if not hermitian:
                Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=+dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=+dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
            
            if opts != None:
                U, S, V = init.split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), opts=opts)
                psi.A[n] = V
                psi.pC = (n - 1,n)
                psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
            else:
                psi.A[n] = init
                psi.orth_right(n)
            
            env.update(n, direction = -1)
                
        if H and n != 0:
            # backward in time evolution of a central site: T(-dt*.5)
            init = psi.A[psi.pC]
            Hv = lambda v : env.Heff0(v, psi.pC)
            if not hermitian:
                Hv_dag = lambda v : env.Heff0(v, psi.pC, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=-dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=-dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[psi.pC] = init[0]
            psi.absorb_central(direction=-1)
            
    for n in range(psi.N):  # sweep in direction = +1
        init = psi.A[n]
        if M: # Apply the Kraus operator
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK) # discard V
            init = u.dot(s, axes=((3,),(0,)))
            psi.A[n] = init.transpose(axes=(0,1,3,2))

            if not H:
                U, S, V = psi.A[n].split_svd(axes=(psi.left + psi.phys + psi.aux , psi.right), opts=opts)
                psi.A[n] = U
                psi.pC = (n,n + 1)
                psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
                if n != psi.N - 1:
                    psi.absorb_central(direction=-1)
                else:
                    pC = None
        if H:
            init = psi.A[n]
            # forward in time evolution of a single site: T(+dt*.5)^*
            Hv = lambda v : env.Heff1(v, n)
            if not hermitian:
                Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=+dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=+dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]

            if opts != None:
                U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, psi.right), opts=opts)
                psi.A[n] = U
                psi.pC = (n,n + 1)
                psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
            else:
                psi.A[n] = init
                psi.orth_left(n)            
            env.update(n, direction = +1)

        if n == round(psi.N / 2):
            psi.entropy, psi.Smin, psi.no = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.aux, psi.right), alpha=1)
            if psi.nr_aux == 1:
                psi.entropy_aux, psi.Smin_aux, psi.no_aux = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.right, psi.aux), alpha=1)
            
           
        if H:
            if n != psi.N - 1:
                # backward in time evolution of a central site: T(-dt*.5)^*
                init = psi.A[psi.pC]
                Hv = lambda v : env.Heff0(v, psi.pC)
                if not hermitian:
                    Hv_dag = lambda v : env.Heff0(v, psi.pC, conj=True)
                    init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=-dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
                else:
                    init = expmw(Av=Hv, init=[init], dt=-dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
                psi.A[psi.pC] = init[0]
                psi.absorb_central(direction=+1)
            else:
                psi.pC = None

    return env # can be used in the next sweep
def tdvp_sweep_2site(psi, H=False, M=False, dt=1, env=None, dtype='complex128', k=4, hermitian=True, fermionic=False, tol=1e-14, biorth=True, NA=None, opts=None, optsK=None, disentangle=True):
    # Assume psi is in left cannonical form (=Q-R-)
    #    default bioth=True - biothogonalize Krylov vectors if is non-Hermitian
    #    version
    #    NA - cost of single Av(init) contraction < < < err?
    #    evolve the system with the time dt
    if not env and H:  # setup environments
        env = Envs(layers=[psi, H, psi])
        env.setup(direction=+1)

    for n in range(psi.N - 1, 0, -1): # sweep in direction = -1
        
        if M: # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK) # discard V
            init = u.dot(s, axes=((3,),(0,)))
            psi.A[n] = init.transpose(axes=(0,1,3,2))
            if not H:
                U, S, V = psi.A[n].split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), opts=opts)
                psi.A[n] = V
                psi.pC = (n - 1,n)
                psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
                psi.absorb_central(direction = -1)
        
        # forward evolution on n
        if H:
            psi.pC = (n - 1, n)
            init = psi.merge_mps(psi.pC)
            Hv = lambda v: env.Heff2(v,psi.pC)
            if not hermitian:
                Hv_dag = lambda v : env.Heff2(v, psi.pC, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=+dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=+dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
                
            #split mps using SVD and use truncation
            U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), opts=opts)
            psi.A[n - 1] = U
            psi.A[psi.pC] = S
            psi.A[n] = V
            psi.absorb_central(direction=-1)
            env.update(n, direction = -1)
        # backward evolution on n-1 if not at the end (n!=1)
        n+= -1
        init = psi.A[n]
        if n > 0 and H:
            Hv = lambda v : env.Heff1(v, n)
            if not hermitian:
                Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=-dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=-dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
            psi.A[n] = init
        else:
            if n == 0 and M: # Apply the Kraus operator on n
                tmp = M.A[n].dot(init, axes=((2,),(1,)))
                tmp.swap_gate(axes=(0,2), fermionic=fermionic)
                u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK) # discard V as it is not nessesery to keep
                init = u.dot(s, axes=((3,),(0,)))
                psi.A[n] = init.transpose(axes=(0,1,3,2))
    
    for n in range(psi.N - 1): # sweep in direction = +1
        init = psi.A[n]
        if n > 0 and H:
            Hv = lambda v : env.Heff1(v, n)
            if not hermitian:
                Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=-dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=-dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n] = init[0]
        else:
            if n == 0 and M: # Apply the Kraus operator on n
                tmp = M.A[n].dot(init, axes=((2,),(1,)))
                tmp.swap_gate(axes=(0,2), fermionic=fermionic)
                u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK) # discard V as it is not nessesery to keep
                init = u.dot(s, axes=((3,),(0,)))
                psi.A[n] = init.transpose(axes=(0,1,3,2))
                if not H:
                    U, S, V = psi.A[n].split_svd(axes=(psi.left + psi.phys + psi.aux , psi.right), opts=opts)
                    psi.A[n] = U
                    psi.pC = (n,n + 1)
                    psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
                    psi.absorb_central(direction = +1)

        if H: # forward time evolution of two-site: n is updated
            psi.pC = (n, n + 1)
            init = psi.merge_mps(psi.pC)        
            Hv = lambda v: env.Heff2(v, psi.pC)
            if not hermitian:
                Hv_dag = lambda v : env.Heff2(v, psi.pC, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=+dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=+dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
        
            #split mps using SVD and use truncation
            U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), opts=opts)
            psi.A[n] = U
            psi.A[psi.pC] = S
            psi.A[n + 1] = V
            psi.absorb_central(direction=+1)
            env.update(n, direction = +1)

        n += +1
        if M: # Apply the Kraus operator on n
            init = psi.A[n]
            tmp = M.A[n].dot(init, axes=((2,),(1,)))
            tmp.swap_gate(axes=(0,2), fermionic=fermionic)
            u,s,_ = tmp.split_svd(axes=((2,1,4),(0,3)), opts=optsK) # discard V as it is not nessesery to keep
            init = u.dot(s, axes=((3,),(0,)))
            psi.A[n] = init.transpose(axes=(0,1,3,2))   
            if not H and n < psi.N - 1:
                U, S, V = psi.A[n].split_svd(axes=(psi.left + psi.phys + psi.aux , psi.right), opts=opts)
                psi.A[n] = U
                psi.pC = (n,n + 1)
                psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
                psi.absorb_central(direction = +1)

        if n == round(psi.N / 2):
            psi.entropy, psi.Smin, psi.no = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.aux, psi.right), alpha=1)
            if psi.nr_aux == 1:
                psi.entropy_aux, psi.Smin_aux, psi.no_aux = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.right, psi.aux), alpha=1)
           
    psi.orth_left(n)
    psi.pC = None
    if H:
        env.update(n, direction = +1)
    #psi updates in place
    return env # can be used in the next sweep
def tdvp_OBC_aux(psi, tmax, dt=1, H=False, M=False, env=None, measure_O=None, cutoff_sweep=20, cutoff_dE=1e-9, hermitian=True, fermionic=False, k=8, tol=1e-14, dtype='complex128', biorth=True, NA=None, version='1site', opts=None, optsK=None, disentangle=False):
    #evolve with TDVP method, up to tmax and initial guess of the time step dt
    # meaure_O - list of things to measure e.g.  [2,[OL, OR], [1,2,3]] -
    # measure exp.  val of 2-site operator OL, OR on sites (1,2), (2,3), (3,4)
    # opts - optional info for MPS truncation
    sweep = 0
    curr_t = 0
    if env == None and H:
        env = Envs(layers=[psi, H, psi])
        env.on_aux = True
        env.setup(direction=+1)
        
    if H:
        E0 = env.F[(psi.N - 1,psi.N)].to_number()
        dE = cutoff_dE + 1
    else:
        E0, dE = 0, 0
    
    sgn = 1j * (dt.real + 1j * (dt.imag)) / abs(dt)
    while abs(curr_t) < abs(tmax):
        dt = min([abs(tmax - curr_t), abs(dt)])

        if not H and not M:
            print('yamps.tdvp: Neither Hamiltonian nor Kraus operators defined.')
        else:
            if version == '0site':
                raise YampsError('yamps.tdvp: Only 1, 2-site versions defined.')
            elif version == '1site':
                env = tdvp_sweep_1site_aux(psi=psi, H=H, M=M, dt= sgn * dt, env=env, dtype=dtype, k=k, hermitian=hermitian, fermionic=fermionic, tol=tol, biorth=biorth, NA=NA, opts=opts, optsK=optsK, disentangle=disentangle)        
            elif version == '2site':
                env = tdvp_sweep_2site_aux(psi=psi, H=H, M=M, dt= sgn * dt, env=env, dtype=dtype, k=k, hermitian=hermitian, fermionic=fermionic, tol=tol, biorth=biorth, NA=NA, opts=opts, optsK=optsK, disentangle=disentangle)        

        nor = Envs(layers=[psi, psi])
        nor.on_aux = True
        nor.setup(direction=-1)
        norma = nor.F[(0,-1)].to_number()

        #psi.Dmax is maximal virtual dimansion of physical block
        psi.get_Dmax()
        
        if H:
            E = env.F[(psi.N - 1,psi.N)].to_number() / norma
        else:
            E = 0
        dE = abs(E - E0)
        if measure_O != None:
            measured = [None] * len(measure_O)
            for it in range(len(measure_O)):
                tmp = measure_O[it]
                if len(tmp) > 2:
                    n = tmp[2]
                else:
                    n = None
                if tmp[0] == 1:
                    measured[it] = measure_1site(psi, tmp[1], n=n, nor=nor)
                elif tmp[0] == 2:
                    measured[it] = measure_2site(psi, tmp[1], n=n, nor=nor)   
        E0 = E 
        #print('Iteration: ', sweep,' energy: ',E, ' dE: ',dE,' time:
        #',curr_t,' D: ',psi.Dmax,' Smin: ',psi.Smin,' S: ',psi.entropy,'
        #S_aux: ',psi.entropy_aux)
        sweep+=1
        curr_t+=dt
        #create results' list
        out = (env, E,dE,)
        if measure_O != None:
            out+=(measured,)
    #psi updated in place
    return out 

def tdvp_sweep_1site_aux(psi, H=False, M=False, dt=1, env=None, dtype='complex128', k=4, hermitian=True, fermionic=False, tol=1e-14, biorth=True, NA=None, opts=None, optsK=None, disentangle=False):
    # Assume psi is in left cannonical form (=Q-R-)
    #    default bioth=True - biothogonalize Krylov vectors if is non-Hermitian
    #    version
    #    NA - cost of single Av(init) contraction < < < err?
    #    evolve the system with the time dt
    
    if H and not env:  # setup environments
        env = Envs(layers=[psi, H, psi])
        env.on_aux = True
        env.setup(direction=+1)
    
    for n in range(psi.N - 1, -1, -1): # sweep in direction = -1
    # forward in time evolution of a single site: T(+dt*.5)
        
        if H:
            init = psi.A[n]
            Hv = lambda v : env.Heff1(v, n)
            if not hermitian:
                Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=+dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=+dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
            if opts != None:
                U, S, V = init.split_svd(axes=(psi.left, psi.phys + psi.aux + psi.right), opts=opts)
                psi.A[n] = V
                psi.pC = (n - 1,n)
                psi.A[psi.pC] = U.dot(S, axes=((1),(0)))
            else:
                psi.A[n] = init
                psi.orth_right(n)
            
            env.update(n, direction = -1)

        if H and n != 0:
            # backward in time evolution of a central site: T(-dt*.5)
            init = psi.A[psi.pC]
            Hv = lambda v : env.Heff0(v, psi.pC)
            if not hermitian:
                Hv_dag = lambda v : env.Heff0(v, psi.pC, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=-dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=-dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[psi.pC] = init[0]
            psi.absorb_central(direction=-1)
            
    for n in range(psi.N):  # sweep in direction = +1
        init = psi.A[n]

        if H:
            init = psi.A[n]
            # forward in time evolution of a single site: T(+dt*.5)^*
            Hv = lambda v : env.Heff1(v, n)
            if not hermitian:
                Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=+dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=+dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]

            if opts != None:
                U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, psi.right), opts=opts)
                psi.A[n] = U
                psi.pC = (n,n + 1)
                psi.A[psi.pC] = S.dot(V, axes=((1),(0)))
            else:
                psi.A[n] = init
                psi.orth_left(n)            
            env.update(n, direction = +1)

        if n == round(psi.N / 2):
            psi.entropy, psi.Smin, psi.no = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.aux, psi.right), alpha=1)
            if psi.nr_aux == 1:
                psi.entropy_aux, psi.Smin_aux, psi.no_aux = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.right, psi.aux), alpha=1)

        if H:
            if n != psi.N - 1:
                # backward in time evolution of a central site: T(-dt*.5)^*
                init = psi.A[psi.pC]
                Hv = lambda v : env.Heff0(v, psi.pC)
                if not hermitian:
                    Hv_dag = lambda v : env.Heff0(v, psi.pC, conj=True)
                    init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=-dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
                else:
                    init = expmw(Av=Hv, init=[init], dt=-dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
                psi.A[psi.pC] = init[0]
                psi.absorb_central(direction=+1)
            else:
                psi.pC = None
    return env # can be used in the next sweep
def tdvp_sweep_2site_aux(psi, H=False, M=False, dt=1, env=None, dtype='complex128', k=4, hermitian=True, fermionic=False, tol=1e-14, biorth=True, NA=None, opts=None, optsK=None, disentangle=True):
    # Assume psi is in left cannonical form (=Q-R-)
    #    default bioth=True - biothogonalize Krylov vectors if is non-Hermitian
    #    version
    #    NA - cost of single Av(init) contraction < < < err?
    #    evolve the system with the time dt
    if not env and H:  # setup environments
        env = Envs(layers=[psi, H, psi])
        env.on_aux = True
        env.setup(direction=+1)

    for n in range(psi.N - 1, 0, -1): # sweep in direction = -1

        # forward evolution on n
        if H:
            psi.pC = (n - 1, n)
            init = psi.merge_mps(psi.pC)
            Hv = lambda v: env.Heff2(v,psi.pC)
            if not hermitian:
                Hv_dag = lambda v : env.Heff2(v, psi.pC, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=+dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=+dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
                
            #split mps using SVD and use truncation
            U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), opts=opts)
            psi.A[n - 1] = U
            psi.A[psi.pC] = S
            psi.A[n] = V
            psi.absorb_central(direction=-1)
            env.update(n, direction = -1)
        
        # backward evolution on n-1 if not at the end (n!=1)
        n+= -1
        init = psi.A[n]
        if n > 0 and H:
            Hv = lambda v : env.Heff1(v, n)
            if not hermitian:
                Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=-dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=-dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
            psi.A[n] = init

    
    for n in range(psi.N - 1): # sweep in direction = +1
        init = psi.A[n]
        if n > 0 and H:
            Hv = lambda v : env.Heff1(v, n)
            if not hermitian:
                Hv_dag = lambda v : env.Heff1(v, n, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=-dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=-dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            psi.A[n] = init[0]

        if H: # forward time evolution of two-site: n is updated
            psi.pC = (n, n + 1)
            init = psi.merge_mps(psi.pC) 
            Hv = lambda v: env.Heff2(v, psi.pC)
            if not hermitian:
                Hv_dag = lambda v : env.Heff2(v, psi.pC, conj=True)
                init = expmw(Av=Hv, init=[init], Bv=Hv_dag, dt=+dt * .5, tol=tol, k=k, hermitian=False, biorth=biorth,  dtype=dtype, NA=NA)
            else:
                init = expmw(Av=Hv, init=[init], dt=+dt * .5, tol=tol, k=k, hermitian=True, dtype=dtype, NA=NA)
            init = init[0]
        
            #split mps using SVD and use truncation
            U, S, V = init.split_svd(axes=(psi.left + psi.phys + psi.aux, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.aux + psi.right)), opts=opts)
            psi.A[n] = U
            psi.A[psi.pC] = S
            psi.A[n + 1] = V
            psi.absorb_central(direction=+1)
            env.update(n, direction = +1)

        n += +1
        if n == round(psi.N / 2):
            psi.entropy, psi.Smin, psi.no = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.aux, psi.right), alpha=1)
            if psi.nr_aux == 1:
                psi.entropy_aux, psi.Smin_aux, psi.no_aux = psi.A[n].entropy(axes=(psi.left + psi.phys + psi.right, psi.aux), alpha=1)
           
    psi.orth_left(n)
    psi.pC = None
    if H:
        env.update(n, direction = +1)
    #psi updates in place
    return env # can be used in the next sweep


            #######    MEASURE-DEVICES SECTION #######
def measure(psi, nor, measure_O):
    measured = None
    if measure_O != None:
        measured = [None] * len(measure_O)
        for it in range(len(measure_O)):
            tmp = measure_O[it]
            if len(tmp) > 2:
                n = tmp[2]
            else:
                n = None
            if tmp[0] == 1:
                measured[it] = measure_1site(psi, tmp[1], n=n, nor=nor)
            elif tmp[0] == 2:
                measured[it] = measure_2site(psi, tmp[1], n=n, nor=nor)   
    return measured

def measure_1site(psi, O, n=None, nor=None):
    #  n - a list of sites to measure expectation value of the operator O
    #  default: n=None - all sites
    if nor == None:
        nor = Envs(layers=[psi, psi])
        nor.setup(direction=-1)
    if n != None:
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
    return (out)
    
    
def measure_2site(psi, OPs, n=None, nor=None):
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
    return (out)
class YampsError(Exception):
    pass

class Mps:
    """
    ===============================
        Yet Another Matrix Product States
    ===============================
    """

    def __init__(self, N, nr_phys=1, nr_aux=0):
        r"""
        :param N: number of sites
        :param nr_phys: number of physical legs

        Initialize Matrix Product State
        """
        self.entropy = None
        self.Smin = None
        self.no = None
        
        self.entropy_aux = None
        self.Smin_aux = None
        self.no_aux = None
        
        self.N = N  # length
        self.A = {}  # dict of MPS tensors
        self.U = {}  # dict of disentangling tensors
        self.pC = None  # position of the center site
        # legs order in MPS tensor: left (0), physical(s), auxilliary(s), right
                              # (-1 i.e.  last)
        self.Dmax = None
        self.nr_phys = nr_phys
        self.nr_aux = nr_aux
        self.left = (0,)
        self.right = (nr_phys + nr_aux + 1,)
        self.phys = tuple(ii for ii in range(1 , nr_phys + 1))
        self.aux = tuple(ii for ii in range(1 + nr_phys, nr_aux + nr_phys + 1))
        
    def copy(self):
        r""" Makes a copy of mps. """
        phi = Mps(N=self.N, nr_phys=self.nr_phys, nr_aux=self.nr_aux)
        for ind in self.A:
            phi.A[ind] = self.A[ind].copy()
        phi.pC = self.pC
        return phi
        
    def orth_left(self, n):
        r"""Left orthogonalization of n-th site; generates central site"""
        if self.pC is not None:
            raise YampsError('Only one central site possible')
        self.pC = (n, n + 1)
        Q, R = self.A[n].split_qr(axes=(self.left + self.phys + self.aux, self.right))
        self.A[n] = Q
        normC = R.norm(ord='inf', round2=True)
        self.A[self.pC] = (1 / normC) * R

    def orth_right(self, n):
        r"""Right orthogonalization of n-th site; generates ceratal site"""
        if self.pC is not None:
            raise YampsError('Only one central site possible')
        self.pC = (n - 1, n)
        R, Q = self.A[n].split_rq(axes=(self.left, self.phys + self.aux + self.right))
        self.A[n] = Q
        normC = R.norm(ord='inf', round2=True)
        self.A[self.pC] = (1 / normC) * R

    def absorb_central(self, direction):
        r"""Abosrb central site;
            direction=-1 to the left; direction=1 to the right"""
        if self.pC is None:
            raise YampsError('No central site')
        pC = self.pC
        C = self.A.pop(pC)
        if direction == -1:
            self.A[pC[0]] = self.A[pC[0]].dot(C, axes=(self.right, 0))
        elif direction == 1:
            self.A[pC[1]] = C.dot(self.A[pC[1]], axes=(1, self.left))
        self.pC = None

    def canonize_left(self, normalize=True):
        for n in range(self.N - 1):
            self.orth_left(n)
            self.absorb_central(direction=1)
        normA0 = self.A[self.N - 1].norm()
        if normalize:
            self.A[self.N - 1] = (1 / normA0) * self.A[self.N - 1]
        else:
            self.Norma = normA0
            
    def canonize_right(self, normalize=True):
        for n in range(self.N - 1, 0, -1):
            self.orth_right(n)
            self.absorb_central(direction=-1)
        normA0 = self.A[0].norm()
        if normalize:
            self.A[0] = (1 / normA0) * self.A[0]
        else:
            self.Norma = normA0

    def canonize(self, pC=(-1,0), normalize=True, diagonal_C=False):
        # make mps left canonical from 0 to bond pC and right canonical from pC
        # to the last site
        # save remaining central site in bond pC
        for n in range(min(pC) + 1):
            self.orth_left(n)
            self.absorb_central(direction=1)
            
        for n in range(self.N - 1, max(pC), -1):
            self.orth_right(n)
            self.absorb_central(direction=-1)
        self.orth_right(max(pC))

        if diagonal_C:
            u, s, v = self.A[self.pC].split_svd(axes=((0,),(1,)))
            self.A[self.pC] = u
            self.absorb_central(direction=-1)
            self.pC = pC
            self.A[self.pC] = v
            self.absorb_central(direction=+1)
            self.pC = pC
            self.A[self.pC] = s
            self.pC = pC
        if normalize:
            self.A[self.pC] = self.A[self.pC].__mul__(1. / self.A[self.pC].norm())
        else:
            self.Norma =  self.A[self.pC].norm()

    def merge_mps(self, pC=False):
        #merge mps[n] and mps[n+direction] into two-site mps
        if not pC:
            pC = psi.pC
        nL = pC[0]
        nR = pC[1]
        return self.A[nL].dot(self.A[nR], axes=(self.right,self.left))

    def get_Dmax(self):
        self.Dmax = 0
        for n in range(self.N):
            #self.Dmax =max([self.Dmax,sum((self.A[n].get_tD_list()[1])[0]),sum((self.A[n].get_tD_list()[1])[-1])])
            self.Dmax = max([self.Dmax , max(self.A[n].get_shape())])

    def save_to_file(self, file_name):
        d = {}
        for it in range(self.N):
            d[it] = self.A[it].to_dict()
        numpy.save(file_name, d)

class Envs:
    """
    =============================
        Environments of Mps
    =============================
    """

    def __init__(self, layers=[], conj_list=[]):
        """Initialize environments"""
        # labeling convention: F[(k,l)] = environment on the link (k->l) [order
        # matters]
        # for purigication: additionally tracing out the auxillary degrees of
        # freedom
        self.layers = layers  # order [<phi|, H, H, |psi>, <mps| mpo (some or none) |mps>]
        self.nr_aux = self.layers[-1].nr_aux # use nr_auzx of psi tensor set
        self.nl = len(layers)  # number of layers
        self.N = layers[0].N
        self.on_aux = False
        if not conj_list: 
            self.conj_list = [0] * self.nl
            self.conj_list[0] = 1  
            # by default first layer is conjugated -- it should correspond to
            # <bra|
        self.F = {}  # dict for environments
        self.reset_boundary()  # set environments at boundaries
        self.set_contraction_ind()
        
    def reset_boundary(self):
        """ Set environments at boundaries 
            Matches the symmetry sectors of the bondary tensors.
        """
        # left boundary
        leg_list = [obj.left[0] for obj in self.layers]
        tensor_list = [obj.A[0] for obj in self.layers]
        Ds = Tensor.match_legs(tensor_list=tensor_list, leg_list=leg_list, conj_list=self.conj_list)
        tmp = Tensor.ones(settings=settings, **Ds)
        #tmp = Tensor.zeros(settings=settings, **Ds)
        #tmp.A[0][-1,-1] = 1
        self.F[(-1, 0)] = tmp
        # right boundary
        leg_list = [obj.right[0] for obj in self.layers[::-1]]
        tensor_list = [obj.A[self.N - 1] for obj in self.layers[::-1]]
        Ds = Tensor.match_legs(tensor_list=tensor_list, leg_list=leg_list, conj_list=self.conj_list[::-1])
        tmp = Tensor.ones(settings=settings, **Ds)
        #tmp = Tensor.zeros(settings=settings, **Ds)
        #tmp.A[0][0,0] = 1
        self.F[(self.N, self.N - 1)] = tmp
        
    def set_contraction_ind(self):
        if self.nr_aux == 0:
            if self.nl == 2:
                self._inds_dp1 = [[3, 1], [3, 2, -1], [1, 2, -2]]
                self._inds_dm1 = [[-2, 2, 3], [-1, 2, 1], [1, 3]] 
            elif self.nl == 3: 
                self._inds_dp1 = [[5, 3, 1], [5, 4, -1], [3, 4, 2, -2], [1, 2, -3]]
                self._inds_dm1 = [[-3, 4, 5], [-2, 4, 2, 3], [-1, 2, 1], [1, 3, 5]]
        elif self.nr_aux == 1:
            if self.on_aux == True:
                if self.nl == 2:
                    self._inds_dp1 = [[3,4], [3,2,1,-1], [4,2,1,-2]]
                    self._inds_dm1 = [[-2,2,1,4], [-1,2,1,5], [5,4]] 
                elif self.nl == 3: 
                    self._inds_dp1 = [[6,3,1], [6,5,4,-1], [3,4,2,-2], [1,5,2,-3]]
                    self._inds_dm1 = [[-3,5,4,6], [-2,4,2,3], [-1,5,2,1], [1,3,6]]
            else:
                if self.nl == 2:
                    self._inds_dp1 = [[3,4], [3,2,1,-1], [4,2,1,-2]]
                    self._inds_dm1 = [[-2,2,1,4], [-1,2,1,5], [5,4]] 
                elif self.nl == 3: 
                    self._inds_dp1 = [[6,3,1], [6,4,5,-1], [3,4,2,-2], [1,2,5,-3]]
                    self._inds_dm1 = [[-3,4,5,6], [-2,4,2,3], [-1,2,5,1], [1,3,6]]
                
    def update(self, n, direction, in_place=True):
        r"""
        :param n : site
        :param direction : +/- 1

        Update environment including n-th site
        direction = 1 -> moving right (i.e. left environment)
        direction = -1 -> moving left (i.e. right environment)
        """
        if direction == +1:
            tensor_list = [self.F[(n - 1, n)]] + [obj.A[n] for obj in self.layers]
            conjs = [0] + self.conj_list
            inds = self._inds_dp1
        elif direction == -1:
            tensor_list = [obj.A[n] for obj in self.layers] + [self.F[(n + 1, n)]]
            conjs = self.conj_list + [0]
            inds = self._inds_dm1
        if in_place != True:
            return ncon(tensor_list, inds, conjs)
        else:
            self.F[(n, n + direction)] = ncon(tensor_list, inds, conjs)

    def setup(self, direction=None, pC=None):
        if direction == None and pC != None:
            for n in range(min(pC) + 1):
                self.update(n, direction=+1)
            for n in range(self.N - 1, max(pC) - 1, -1):
                self.update(n, direction=-1)
        else:
            it = range(self.N) if direction == 1 else range(self.N - 1, -1, -1) 
            for n in it:
                self.update(n, direction)
       
    def Heff0(self, C=False, pC=False, conj=False):
        ## calculate Heff0 acting with C on pC bond
        ## pC - bond given as tuple e.g.  (1,2)-bond between 1 and 2
        ## if conj=True - calculate Heff1^\dagger acting on A on site n
        if not pC:
            # use existing pC, C if not defined
            pC = self.layers[-1].pC
        if not C:
            C = self.layers[-1].A[pC]
        
        inds_heffD = [[3,2,-1], [3,1], [-2,2,1]]
        inds_heff = [[-1,3,2], [2,1], [1,3,-2]]
       
        if conj:
            out = ncon([self.F[pC], C, self.F[pC[::-1]]], inds_heffD, [1,0,1])
        else:
           out = ncon([self.F[pC], C, self.F[pC[::-1]]], inds_heff, [0,0,0])
        return out
    
    
    def Heff1(self, A, n, conj=False):
        ## calculate Heff1 acting on A on site n
        ## if conj=True - calculate Heff1^\dagger acting on A on site n

        if self.nr_aux == 0:
            inds_heffD = [[4,5,-1], [5,3,-2,2], [4,3,1], [-3,2,1]]
            inds_heff = [[-1,5,4], [5,-2,2,3], [4,2,1], [1,3,-3]]
        elif self.nr_aux == 1:
            if self.on_aux:
                inds_heffD = [[4,5,-1], [5,3,-3,2], [4,-2,3,1], [-4,2,1]]
                inds_heff = [[-1,5,4], [5,-3,2,3], [4,-2,2,1], [1,3,-4]]
                #A.s[1], A.s[2] = -1, 1
            else:
                inds_heffD = [[4,5,-1], [5,3,-2,2], [4,3,-3,1], [-4,2,1]]
                inds_heff = [[-1,5,4], [5,-2,2,3], [4,2,-3,1], [1,3,-4]]
            
        if conj:
            out = ncon([self.F[(n - 1,n)], self.layers[1].A[n], A, self.F[(n + 1, n)]], inds_heffD, [1,1,0,1])
        else:
            out = ncon([self.F[(n - 1,n)], self.layers[1].A[n], A, self.F[(n + 1, n)]], inds_heff, [0,0,0,0])
        return out
        
        
    def Heff2(self, A=False, pC=False, conj=False):
        ## calculate Heff2 acting on 2site mps
        ## if conj=True - calculate Heff1^\dagger acting on A on site n
        if not pC:
            # use existing pC, A if not defined
            pC = self.layers[-1].pC
        if not A:
            A = self.layers[-1].merge_mps(pC=pC)

        if self.nr_aux == 0:
            inds_heffD = [[6,7,-1], [7,5,-2,4], [6,5,2,1], [4,2,-3,3], [-4,3,1]]
            #inds_heff = [[-1,7,6], [7,-2,5,4], [6,5,3,2], [4,-3,3,1],
            #[2,1,-4]]
            inds_heff = [[-1, 6, 7], [6, -2, 3, 1], [7, 3, 4, 2], [1, -3, 4, 5], [2, 5, -4]]
        elif self.nr_aux == 1:
            if self.on_aux:
                inds_heffD = [[6,7,-1], [7,5,-3,4], [6,-2,5,-4,2,1], [4,2,-5,3], [-6,3,1]]
                inds_heff = [[-1, 6, 7], [6,-3,3,1], [7,-2,3,-4,4,2], [1, -5, 4, 5], [2, 5, -6]]
                #A.s[1], A.s[2], A.s[3], A.s[4] = -1, 1, -1, 1
            else:
                inds_heffD = [[6,7,-1], [7,5,-2,4], [6,5,-3,2,-5,1], [4,2,-4,3], [-6,3,1]]
                #inds_heff = [[-1,7,6], [7,-2,5,4], [6,5,-3,3,-5,2],
                #[4,-4,3,1],
                #[2,1,-6]]
                inds_heff = [[-1, 6, 7], [6, -2, 3, 1], [7, 3, -3, 4, -5, 2], [1, -4, 4, 5], [2, 5, -6]]
            
        nL, nR = min(pC), max(pC)
        if conj:
            out = ncon([self.F[(nL - 1,nL)], self.layers[1].A[nL], A, self.layers[1].A[nR], self.F[(nR + 1, nR)]], inds_heffD, [1,1,0,1,1])
        else:
            out = ncon([self.F[(nL - 1,nL)], self.layers[1].A[nL], A, self.layers[1].A[nR], self.F[(nR + 1, nR)]], inds_heff, [0,0,0,0,0])
        return out
        
    
    def edge(self, L, R, A, pC, conj=False):
        if self.nr_aux == 0:
            inds_heffD = [[6,7,-1], [7,5,-2,4], [6,5,2,1], [4,2,-3,3], [-4,3,1]]
            inds_heff = [[-1, 6, 7], [6, -2, 3, 1], [7, 3, 4, 2], [1, -3, 4, 5], [2, 5, -4]]
        elif self.nr_aux == 1:
            if self.on_aux:
                inds_heffD = [[6,7,-1], [7,5,-3,4], [6,-2,5,-4,2,1], [4,2,-5,3], [-6,3,1]]
                inds_heff = [[-1, 6, 7], [6,-3,3,1], [7,-2,3,-4,4,2], [1, -5, 4, 5], [2, 5, -6]]
            else:
                inds_heffD = [[6,7,-1], [7,5,-2,4], [6,5,-3,2,-5,1], [4,2,-4,3], [-6,3,1]]
                inds_heff = [[-1, 6, 7], [6, -2, 3, 1], [7, 3, -3, 4, -5, 2], [1, -4, 4, 5], [2, 5, -6]]
        nL, nR = min(pC), max(pC)
        if conj:
            out = ncon([L, self.layers[1].A[nL], A, self.layers[1].A[nR], R], inds_heffD, [1,1,0,1,1])
        else:
            out = ncon([L, self.layers[1].A[nL], A, self.layers[1].A[nR], R], inds_heff, [0,0,0,0,0])
        return out
    