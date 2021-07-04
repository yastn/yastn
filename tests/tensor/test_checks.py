from .context import yast
from .context import config_Z2
from .test_fuse_hard import test_dot_1_super_sparse

tol = 1e-12


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
    assert cache_info["meta_merge_to_matrix"] == (0, 0, 10, 0)

    for _ in range(100):
        a.svd(axes=((0, 1), (2, 3)))
        a.svd(axes=((0, 2), (1, 3)))
        a.svd(axes=((1, 3), (2, 0)))

    cache_info = yast.get_cache_info()
    assert cache_info["meta_merge_to_matrix"] == (297, 3, 10, 3)


def test_cache2():
    yast.set_cache_maxsize(maxsize=100)
    print(yast.get_cache_info())
    for _ in range(100):
        test_dot_1_super_sparse()
    print(yast.get_cache_info())



if __name__ == '__main__':
    test_cache()
    test_cache2()
