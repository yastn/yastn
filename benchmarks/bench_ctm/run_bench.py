import argparse
import timeit

from model_yastn_no_fuse import CtmBenchYastnNoFuse
from model_yastn_fpeps import CtmBenchYastnfPeps

def run_bench(model, args):
    bench = model(args)

    bench.print_info()

    res=timeit.repeat(stmt='bench.enlarged_corner()', setup='pass', repeat=5, number=1, globals=locals())
    print("enlarged_corner times [s] \n", *(f"{x:.3f}" for x in res))

    res=timeit.repeat(stmt='bench.fuse_enlarged_corner()', setup='pass', repeat=5, number=1, globals=locals())
    print("fuse_enlarged_corner times [s] \n", *(f"{x:.3f}" for x in res))

    res=timeit.repeat(stmt='bench.svd_enlarged_corner()', setup='pass', repeat=5, number=1, globals=locals())
    print("svd time [s] \n", *(f"{x:.3f}" for x in res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fname", type=str, default='U1xU1_d=27_D=13_chi=293.json')
    parser.add_argument("-backend", type=str, default='np', choices=['np', 'torch'])
    parser.add_argument("-device", type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    run_bench(CtmBenchYastnfPeps, args)
    run_bench(CtmBenchYastnNoFuse, args)
