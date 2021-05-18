import argparse
import time

## controls
ta = ((-1, -1, -1, -1), (-1, -1, 1, 1), (-1, -1, 2, 2), (-1, 1, -1, 1), (-1, 2, -1, 2), (1, -1, 1, -1), (1, 1, -1, -1), (1, 1, 1, 1), (1, 1, 2, 2), (1, 2, 1, 2), (2, -1, 2, -1), (2, 1, 2, 1), (2, 2, -1, -1), (2, 2, 1, 1), (2, 2, 2, 2))
Da = ((1, 4, 7, 10), (1, 4, 8, 11), (1, 4, 9, 12), (1, 5, 7, 11), (1, 6, 7, 12), (2, 4, 8, 10), (2, 5, 7, 10), (2, 5, 8, 11), (2, 5, 9, 12), (2, 6, 8, 12), (3, 4, 9, 10), (3, 5, 9, 11), (3, 6, 7, 10), (3, 6, 8, 11), (3, 6, 9, 12))

tb = ((-1, -1, 0), (1, 1, 0), (1, 2, 1), (2, 1, -1), (2, 2, 0))
Db = ((1, 4, 7), (2, 5, 7), (2, 6, 11), (3, 5, 10), (3, 6, 7))

meta_new_a = (((-1, -1), (96, 12)),
              ((0, 0), (266, 32)),
              ((1, 1), (99, 15)))
meta_mrg_a = (((0, 0), (-1, -1, -1, -1), (0, 70), 70, (0, 4), 4),
              ((0, 0), (-1, -1, 1, 1), (70, 158), 88, (0, 4), 4),
              ((0, 0), (-1, -1, 2, 2), (158, 266), 108, (0, 4), 4),
              ((0, 0), (1, 1, -1, -1), (0, 70), 70, (4, 14), 10),
              ((0, 0), (1, 1, 1, 1), (70, 158), 88, (4, 14), 10),
              ((0, 0), (1, 1, 2, 2), (158, 266), 108, (4, 14), 10),
              ((-1, -1), (1, 2, 1, 2), (0, 96), 96, (0, 12), 12),
              ((1, 1), (2, 1, 2, 1), (0, 99), 99, (0, 15), 15),
              ((0, 0), (2, 2, -1, -1), (0, 70), 70, (14, 32), 18),
              ((0, 0), (2, 2, 1, 1), (70, 158), 88, (14, 32), 18),
              ((0, 0), (2, 2, 2, 2), (158, 266), 108, (14, 32), 18))
order_a = (2, 3, 0, 1)

meta_new_b = (((-1, -1), (12, 11)),
              ((0, 0), (32, 7)),
              ((1, 1), (15, 10)))
meta_mrg_b = (((0, 0), (-1, -1, 0), (0, 4), 4, (0, 7), 7),
              ((0, 0), (1, 1, 0), (4, 14), 10, (0, 7), 7),
              ((-1, -1), (1, 2, 1), (0, 12), 12, (0, 11), 11),
              ((1, 1), (2, 1, -1), (0, 15), 15, (0, 10), 10),
              ((0, 0), (2, 2, 0), (14, 32), 18, (0, 7), 7))
order_b = (0, 1, 2)

meta_dot = (((-1, -1), (-1, -1), (-1, -1)), ((0, 0), (0, 0), (0, 0)), ((1, 1), (1, 1), (1, 1)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-backend", type=str, choices=['np', 'torch'], default='np')
    parser.add_argument("-device", type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument("-niter", type=int, default=1000)
    args = parser.parse_args()

    if args.backend == 'np':
        import backend_np as backend
    elif args.backend == 'torch':
        import backend_torch as backend

    Aa = {t: backend.randR(D, args.device) for t, D in zip(ta, Da)}
    Ab = {t: backend.randR(D, args.device) for t, D in zip(tb, Db)}

    keep_time = time.time()
    for _ in range(args.niter):
        mAa = backend.merge_to_matrix(Aa, order_a, meta_new_a, meta_mrg_a, args.device)
        mAb = backend.merge_to_matrix(Ab, order_b, meta_new_b, meta_mrg_b, args.device)
        Ac = backend.dot(mAa, mAb, meta_dot)

    print('Total time : %.2f seconds' % (time.time() - keep_time))
