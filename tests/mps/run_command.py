import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './../..'))  # folder with yast
sys.path.append(os.path.join(os.path.dirname(__file__), './../configs'))  # folder with configs for tests
from test_dmrg import test_full_dmrg, test_Z2_dmrg, test_U1_dmrg, test_OBC_dmrg
from test_tdvp import test_full_tdvp, test_Z2_tdvp, test_U1_tdvp, test_OBC_tdvp
from test_checks import test_cache


if __name__ == '__main__':
    test_full_dmrg()
    test_Z2_dmrg()
    test_U1_dmrg()
    test_OBC_dmrg()
    test_full_tdvp()
    test_Z2_tdvp()
    test_U1_tdvp()
    test_OBC_tdvp()
    test_cache()