import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
import argparse
import time
import os
import logging

# Try to import the configuration settings
try:
    from configs import config as cfg
except ImportError:
    from .configs import config as cfg

def ensure_directory(directory):
    """Ensure directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def setup_logging():
    """Setup logging configuration."""
    log_directory = 'logs'
    ensure_directory(log_directory)
    log_file_path = os.path.join(log_directory, 'ctmrg_analysis.log')
    logging.basicConfig(level=logging.INFO, filename=log_file_path, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')


def perform_ctmrg(psi, chi, num_ctmrg_sweeps=10, tol_exp=1e-6):
    """Perform CTMRG and calculate Hubbard model properties."""
    env = fpeps.EnvCTM(psi, chi=chi, cfg=cfg, init='ones')
    energy_old = float('inf')
    
    for _ in range(num_ctmrg_sweeps):
        env.update_(opts_svd={'D_total': chi, 'tol': 1e-12}, method='2site')
        cdagc_up = env.measure_nn(psi.operators.cdag_up, psi.operators.c_up)
        cdagc_dn = env.measure_nn(psi.operators.cdag_dn, psi.operators.c_dn)
        energy = -2 * np.mean([*cdagc_up.values(), *cdagc_dn.values()])
        
        logging.info(f"CTMRG Step: Energy = {energy}")
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    ctm_dir = f"data/{psi.symmetry}/NTU_{psi.ntu_environment}/mu_{psi.mu:.4f}/U_{psi.U:.1f}/beta_{psi.beta:.2f}/D_{psi.D}/dbeta_{psi.dbeta}/{psi.step}/ctm/chi_{chi}/"
    ensure_directory(ctm_dir)
    ctm_file_name = "CTM_tensors.npy"
    ctm_data = {site: {
        'tl': env[site].tl.save_to_dict(),
        'tr': env[site].tr.save_to_dict(),
        'bl': env[site].bl.save_to_dict(),
        'br': env[site].br.save_to_dict(),
        't': env[site].t.save_to_dict(),
        'b': env[site].b.save_to_dict(),
        'l': env[site].l.save_to_dict(),
        'r': env[site].r.save_to_dict()
    } for site in psi.sites()}
    np.save(os.path.join(ctm_dir, ctm_file_name), ctm_data)
    logging.info(f"CTMRG tensors saved at: {ctm_dir + ctm_file_name}")

    return energy

def main(args):
    """Main function to perform analysis based on loaded PEPS tensor data."""
    start_time = time.time()
    file_path = f"data/{args.S}/NTU_{args.NTUEnvironment}/mu_{args.MU:.4f}/U_{args.U:.1f}/beta_{args.BT:.2f}/D_{args.D}/dbeta_{args.DBETA:.3f}/{args.STEP}/PEPS_tensor.npy"
    tensor_data = np.load(file_path, allow_pickle=True)

    psi = yastn.tn.fpeps.load_from_dict(cfg, tensor_data)
    
    energy = perform_ctmrg(psi, args.X)
    logging.info(f"Computed Energy: {energy}")
    logging.info(f"Total Elapsed Time: {time.time() - start_time:.2f} s.")


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description="Analyze PEPS Tensor Data with CTMRG")
    parser.add_argument("-D", type=int, default=10, help="PEPS bond dimension")
    parser.add_argument("-S", default='U1xU1xZ2', help="Symmetry")
    parser.add_argument("-MU", type=float, default=0, help="Chemical potential")
    parser.add_argument("-t", type=float, default=1, help="Hopping parameter")
    parser.add_argument("-U", type=float, default=0, help="Coulomb interaction strength")
    parser.add_argument("-BT", type=float, default=2, help="Beta value at which to analyze the tensor")
    parser.add_argument("-DBETA", type=float, default=0.01)
    parser.add_argument("-STEP", default='FIX', help="Step method used during NTU simulation")
    parser.add_argument("-NTUEnvironment", default='NN', help="NTU environment configuration")
    parser.add_argument("-X", type=int, default=40, help="CTMRG bond dimension")
    args = parser.parse_args()
    
    main(args)
