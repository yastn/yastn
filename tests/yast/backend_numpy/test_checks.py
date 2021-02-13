import yamps.yast as yast
import config_Z2_R
from math import isclose
import numpy as np
import logging

tol = 1e-12

logging.basicConfig(level='INFO')

def test_check_signs():


    a = yast.rand(config=config_Z2_R, s=(-1, 1, 1, -1),
                    t=((0, 1), (0, 1), (0, 1), (0, 1)),
                    D=((1, 2), (3, 4), (5, 6), (7, 8)))

    b = yast.rand(config=config_Z2_R, s=(-1, 1, 1),
                    t=((0, 1), (0, 1), (0, 1)),
                    D=((1, 2), (3, 4), (7, 8)))
    yast.check_signatures_match(value=False)
    r1 = a.dot(b, axes=((0, 1), (0, 1)), conj=(0, 0))
    r2 = a.dot(b, axes=((0, 1), (0, 1)), conj=(1, 0))
    assert isclose(r1.norm_diff(r2), 0, rel_tol=tol, abs_tol=tol)
    yast.check_signatures_match(value=True)


if __name__ == '__main__':
    test_check_signs()
