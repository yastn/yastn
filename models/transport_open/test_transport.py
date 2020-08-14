import logging
import h5py
import numpy as np
import yamps.mps as mps
import transport_vectorization as main_fun
import transport_vectorization_full as main_full
import transport_vectorization_Z2 as main_Z2
import transport_vectorization_U1 as main_U1
import transport_vectorization_general as general


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("models/transport_open/transport.log"),
        logging.StreamHandler()
    ])


def transport(main, basis, tensor):
    # main - module to produce hamiltonian 
    # choice - basis, Majorana or Dirac
    # tensor - tensor type: full, Z2 or U1
    #
    # MODEL
    NL = 3
    NS = 1
    w0 = 1
    v = .5
    wS = 0
    mu = .5
    dV = 0
    gamma = 1.  # dissipation rate
    temp = 0.  # temperature
    distribution = 1
    # how to arrange modes. 0-spatial 1d to energy modes via sine-transformation.
    # 1-uniform spacing through W=4*w0. Are not places on the very corners of W.
    ordered = False  # order basis for mps/mpo

    # MPS
    tensor_type = main.get_tensor_type(basis)
    tol_svd = 1e-6
    D_total = 32
    io = [0.5]  # initial occupation on the impurity

    # algorithm
    sgn = 1j  # imaginary time evolution for sgn == 1j
    dt = .125 * sgn  # time step - time step for single tdvp
    tmax = 200. * dt * sgn  # total time
    opts_svd = {'tol': tol_svd, 'D_total': D_total}
    eigs_tol = 1e-14

    # STORAGE
    directory = 'models/transport_open/'
    name = directory+'test_v2'
    name_txt = directory+'test_v2'+'_output.txt'
    big_file = h5py.File(name + '_output.h5', 'w')

    # SAVE information about simulation
    names = ['NL', 'w0', 'v', 'wS', 'mu', 'dV', 'gamma', 'temp', 'distribution',
             'ordered', 'basis', 'tol_svd', 'D_total', 'io', 'sym', 'dtype', 'dt']
    values = [NL, w0, v, wS, mu, dV, gamma, temp, distribution,
              ordered, basis, tol_svd, D_total, io, tensor_type[0], tensor_type[1], dt]
    g_param = big_file.create_group('parameters')
    for inm, ival in zip(names, values):
        g_param.create_dataset(inm, data=ival)

    # EXECUTE
    LSR, wk, temp, vk, dV, gamma = general.generate_discretization(NL=NL, w0=w0, wS=wS, mu=mu, v=v, dV=dV, tempL=temp, tempR=temp, method=distribution, ordered=ordered, gamma=gamma)
    psi = main.thermal_state(tensor_type=tensor_type, LSR=LSR, io=io, ww=wk, temp=temp, basis=basis)
    LL, LdagL = main.Lindbladian_1AIM_mixed(tensor_type=tensor_type, NL=NL, LSR=LSR, wk=wk, temp=temp, vk=vk, dV=dV, gamma=gamma, basis=basis, AdagA=True)
    H, hermitian, dmrg, version, HH = LL, False, False, None, LdagL
    #H, hermitian, dmrg, version, HH = LdagL, True, True, '2site', LdagL

    # canonize MPS
    psi.canonize_sweep(to='last')
    psi.canonize_sweep(to='first')

    # compress MPO
    H.canonize_sweep(to='last', normalize=False)
    H.sweep_truncate(to='first', opts={'tol': 1e-12}, normalize=False)

    # trace rho
    trace_rho = main.identity(tensor_type=tensor_type, N=psi.N, basis=basis)

    # current
    JLS = main.current(tensor_type=tensor_type, LSR=LSR, vk=-2.*np.pi*vk, cut='LS', basis=basis)
    JSR = main.current(tensor_type=tensor_type, LSR=LSR, vk=-2.*np.pi*vk, cut='SR', basis=basis)

    # Occupation
    NL = main.measure_sumOp(tensor_type=tensor_type, choice=-1, LSR=LSR, Op='nn', basis=basis)
    NS = main.measure_sumOp(tensor_type=tensor_type, choice=2, LSR=LSR, Op='nn', basis=basis)
    NR = main.measure_sumOp(tensor_type=tensor_type, choice=1, LSR=LSR, Op='nn', basis=basis)

    OP_Nocc = [0.]*psi.N
    for n in range(psi.N):
        OP_Nocc[n] = main.measure_Op(tensor_type=tensor_type, N=psi.N, id=n, Op='nn', basis=basis)
    
    # EVOLUTION
    qt = 0
    Dmax = max(psi.get_D())
    E = general.measure_MPOs(psi, [HH])
    out = general.measure_overlaps(psi, [NL, NS, NR, JLS, JSR], norm=trace_rho)
    #
    g_msr = big_file.create_group('measurements')
    max_steps = int((tmax/dt).real)+10
    Es = np.zeros(max_steps)
    occ_L = np.zeros(max_steps)
    occ_S = np.zeros(max_steps)
    occ_R = np.zeros(max_steps)
    curr_LS = np.zeros(max_steps)
    curr_SR = np.zeros(max_steps)
    Ds = np.zeros((max_steps, psi.N+1))
    occ = np.zeros((max_steps, psi.N))
    #
    env = None
    init_steps = 5
    qt = 0
    it_step = 0

    E, nl, ns, nr, jls, jsr = E[0].real, out[0].real, out[1].real, out[2].real, out[3].real, out[4].real
    Nocc = general.measure_overlaps(psi, OP_Nocc, norm=trace_rho)
    
    print('Time: ', round(abs(qt), 4), ' Dmax: ', Dmax, ' E = ', E, ' Tot_Occ= ', round(nl+ns+nr, 4), ' JLS=', jls, ' JSR=', jsr, ' NL=', round(nl, 5), ' NS=', round(ns, 5), ' NR=', round(nr, 5))
    with open(name_txt, 'w') as f:
        print(qt, E, jls, jsr, nl, ns, nr, file=f)
    #
    Es[it_step] = E
    occ_L[it_step] = nl
    occ_S[it_step] = ns
    occ_R[it_step] = nr
    curr_LS[it_step] = jls
    curr_SR[it_step] = jsr
    Ds[it_step,:] = psi.get_D()
    occ[it_step,:] = Nocc
    #
    export = {'Es': Es, 'occ_L': occ_L, 'occ_S': occ_S, 'occ_R': occ_R, 'curr_LS': curr_LS, 'curr_SR': curr_SR, 'Ds': Ds, 'occ': occ}
    for inm, ival in export.items():
        g_msr.create_dataset(inm, data=ival)
    while abs(qt) < abs(tmax):
        exp_tol = tol_svd*.01
        if dmrg:
            ddt = dt
            env, _, _ = mps.dmrg.dmrg_OBC(psi=psi, H=H, env=env, version=version, cutoff_sweep=1, eigs_tol=eigs_tol, hermitian=True,  opts_svd=opts_svd)
        else:
            for it in range(init_steps):
                exp_tol = 1e-1
                ddt = dt*2**(it-init_steps)
                if D_total > 1:
                    env = mps.tdvp.tdvp_sweep_2site(
                        psi=psi, H=H, env=env, dt=ddt, eigs_tol=eigs_tol, exp_tol=exp_tol,  dtype=tensor_type[1], hermitian=hermitian,  opts_svd=opts_svd)
                else:
                    env = mps.tdvp.tdvp_sweep_1site(
                        psi=psi, H=H, env=env, dt=ddt, eigs_tol=eigs_tol, exp_tol=exp_tol,  dtype=tensor_type[1], hermitian=hermitian,  opts_svd=opts_svd)
            else:
                ddt = dt
                env = mps.tdvp.tdvp_sweep_1site(psi=psi, H=H, env=env, dt=ddt, eigs_tol=eigs_tol, exp_tol=exp_tol,  dtype=tensor_type[1], hermitian=hermitian,  opts_svd=opts_svd)
        qt += abs(ddt)
        #
        Dmax = max(psi.get_D())
        E = general.measure_MPOs(psi, [HH])
        out = general.measure_overlaps(psi, [NL, NS, NR, JLS, JSR], norm=trace_rho)
        E, nl, ns, nr, jls, jsr = E[0].real, out[0].real, out[1].real, out[2].real, out[3].real, out[4].real
        Nocc = general.measure_overlaps(psi, OP_Nocc, norm=trace_rho)

        print('Time: ', round(abs(qt), 4), ' Dmax: ', Dmax, ' E = ', E, ' Tot_Occ= ', round(nl+ns+nr, 4), ' JLS=', jls, ' JSR=', jsr, ' NL=', round(nl, 5), ' NS=', round(ns, 5), ' NR=', round(nr, 5))
        with open(name_txt, 'a') as f:
            print(qt, E, jls, jsr, nl, ns, nr, file=f)
        # SAVE EXP. VALUES
        export = {'Es': Es, 'occ_L': occ_L, 'occ_S': occ_S, 'occ_R': occ_R, 'curr_LS': curr_LS, 'curr_SR': curr_SR, 'Ds': Ds, 'occ': occ}
        for inm, ival in export.items():
            del g_msr[inm]
            g_msr.create_dataset(inm, data=ival)
        it_step += 1
    # SAVE state tensors
    main_fun.save_psi_to_h5py(big_file, psi)
    big_file.close()
    

def transport_full(choice):
    transport(main_full, choice, 'full')


def transport_Z2(choice):
    if choice == 'Majorana':
        transport(main_Z2, choice, 'Z2')
    elif choice == 'Dirac':
        print('Z2/Dirac - Option is not implemented.')
        pass


def transport_U1(choice):
    if choice == 'Dirac':
        transport(main_U1, choice, 'U1')
    elif choice == 'Majorana':
        print('U1/Majorana - Option is not implemented.')
        pass


if __name__ == "__main__":
    # pass
    transport_full('Majorana'); print()
    #transport_full('Dirac'); print()
    #transport_Z2('Majorana'); print()
    #transport_Z2('Dirac'); print()
    #transport_U1('Majorana'); print()
    #transport_U1('Dirac'); print()
