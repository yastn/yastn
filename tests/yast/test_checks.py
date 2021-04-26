from context import yast
from context import config_Z2

tol = 1e-12


def test_check_signs():
    a = yast.rand(config=config_Z2, s=(-1, 1, 1, -1),
                  t=((0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8)))

    b = yast.rand(config=config_Z2, s=(-1, 1, 1),
                  t=((0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (7, 8)))
    yast.check_signatures_match(value=False)
    r1 = yast.tensordot(a, b, axes=((0, 1), (0, 1)), conj=(0, 0))
    r2 = yast.tensordot(a, b, axes=((0, 1), (0, 1)), conj=(1, 0))
    assert r1.norm_diff(r2) < tol
    yast.check_signatures_match(value=True)


def test_cache():
    a = yast.rand(config=config_Z2, s=(-1, 1, 1, -1),
                  t=((0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8)))
    for _ in range(100):
        a.svd(axes=((0, 1), (2, 3)))
        a.svd(axes=((0, 2), (1, 3)))
        a.svd(axes=((1, 3), (2, 0)))

    yast.set_cache_maxsize(maxsize=10)
    cache_info = yast.get_cache_info()
    assert cache_info == (0, 0, 10, 0)

    for _ in range(100):
        a.svd(axes=((0, 1), (2, 3)))
        a.svd(axes=((0, 2), (1, 3)))
        a.svd(axes=((1, 3), (2, 0)))

    cache_info = yast.get_cache_info()
    assert cache_info == (297, 3, 10, 3)


if __name__ == '__main__':
    test_check_signs()
    test_cache()
