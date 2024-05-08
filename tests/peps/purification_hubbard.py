import yastn
import yastn.tn.fpeps as fpeps
import logging
import argparse
import time
import os
import numpy as np

try:
    from .configs import config as cfg
except ImportError:
    from configs import config as cfg

FIX_dimensions = {
    10: {(0,0,0):2, (-1,0,1):1, (1,0,1):1, (0,-1,1):1, (0,1,1):1, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1},
    14: {(0,0,0):2, (-1,0,1):2, (1,0,1):2, (0,-1,1):2, (0,1,1):2, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1},
    16: {(0,0,0):4, (-1,0,1):2, (1,0,1):2, (0,-1,1):2, (0,1,1):2, (-1,-1,0):1, (-1,1,0):1, (1,-1,0):1, (1,1,0):1},
    20: {(0,0,0):4, (-1,0,1):2, (1,0,1):2, (0,-1,1):2, (0,1,1):2, (-1,-1,0):2, (-1,1,0):2, (1,-1,0):2, (1,1,0):2},
    25: {(0,0,0):5, (-1,0,1):3, (1,0,1):3, (0,-1,1):3, (0,1,1):3, (-1,-1,0):2, (-1,1,0):2, (1,-1,0):2, (1,1,0):2},
    29: {(0,0,0):5, (-1,0,1):4, (1,0,1):4, (0,-1,1):4, (0,1,1):4, (-1,-1,0):2, (-1,1,0):2, (1,-1,0):2, (1,1,0):2}
}

def ensure_directory(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_directory(sym, D, mu, U, beta, dbeta, step, ntu_environment):
    """Generate the directory structure based on parameters."""
    formatted_beta = f"{2*beta:.2f}"  # Ensure beta is formatted to two decimal places
    dir_path = f"data/{sym}/NTU_{ntu_environment}/mu_{mu:.4f}/U_{U:.1f}/beta_{formatted_beta}/D_{D}/dbeta_{dbeta:.3f}/{step}/"
    return ensure_directory(dir_path)

def setup_logging(args):
    """Setup logging configuration."""
    log_directory = 'logs'
    ensure_directory(log_directory)
    log_file_path = os.path.join(log_directory, f'simulation_E={args.NTUEnvironment}_D={args.D}.log')
    logging.basicConfig(level=logging.INFO, filename=log_file_path, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')

def NTU_Hubbard_Purification(sym, D, mu, t, U, beta_target, dbeta, step, ntu_environment):
    geometry = fpeps.CheckerboardLattice()
    ops = yastn.operators.SpinfulFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
    I = ops.I()
    c_up, c_dn = ops.c(spin='u'), ops.c(spin='d')
    cdag_up, cdag_dn = ops.cp(spin='u'), ops.cp(spin='d')
    n_up, n_dn = ops.n(spin='u'), ops.n(spin='d')

    g_hop_u = fpeps.gates.gate_nn_hopping(t, dbeta / 2, I, c_up, cdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t, dbeta / 2, I, c_dn, cdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu, mu, U, dbeta / 2, I, n_up, n_dn)
    gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

    psi = fpeps.product_peps(geometry, I)
    env_evolution = fpeps.EnvNTU(psi, which=ntu_environment)
    num_steps = round((beta_target / 2) / dbeta)
    dbeta = (beta_target / 2) / num_steps

    opts_svd = {"D_total": D, 'tol_block': 1e-15, "D_block": FIX_dimensions.get(D, {})}

    for num in range(num_steps):
        beta = (num + 1) * dbeta
        fpeps.evolution_step_(env_evolution, gates, opts_svd=opts_svd, initialization="EAT")

        print(f"{beta:0.3f}")

        if abs(2 * beta % 0.5) < 1e-10:  # Save only at specific intervals of beta
            directory = get_directory(sym, D, mu, U, beta, dbeta, step, ntu_environment)
            file_name = f"PEPS_tensor.npy"  # Use formatted beta for filename
            info = fpeps.evolution_step_(env_evolution, gates, opts_svd=opts_svd, initialization="EAT")
            ntu_error_max, ntu_error_mean = np.max(info.truncation_error), np.mean(info.truncation_error)
            logging.info(f"Step {num + 1}, beta = {beta:.2f}, NTU Error Max: {ntu_error_max:.2e}, Mean: {ntu_error_mean:.2e}")

            mdata = psi.save_to_dict()

            np.save(os.path.join(directory, file_name), mdata)
            logging.info(f"Data saved: {os.path.join(directory, file_name)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", type=int, default=25)
    parser.add_argument("-S", default='U1xU1xZ2')
    parser.add_argument("-MU", type=float, default=-2.2)
    parser.add_argument("-t", type=float, default=1)
    parser.add_argument("-U", type=float, default=8)
    parser.add_argument("-BT", type=float, default=10)
    parser.add_argument("-DBETA", type=float, default=0.01)
    parser.add_argument("-STEP", default='FIX')
    parser.add_argument("-NTUEnvironment", default='NN+')
    args = parser.parse_args()
    setup_logging(args)

    start_time = time.time()
    NTU_Hubbard_Purification(sym=args.S, D=args.D, mu=args.MU, t=args.t, U=args.U,
                             beta_target=args.BT, dbeta=args.DBETA, step=args.STEP, ntu_environment=args.NTUEnvironment)
    logging.info(f'Elapsed time: {time.time() - start_time:.2f} s.')
