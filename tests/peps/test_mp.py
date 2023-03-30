import numpy as np
import multiprocessing as mp
import time

# Function to perform SVD on a matrix
def add_mat(matrix1, matrix2):
    print("AA")
    time.sleep(3)
    print("BB")
    x = np.random.rand(1)
    """U, S, V = np.linalg.svd(matrix)"""
    return x # (matrix1+matrix2)

if __name__ == '__main__':
    # Generate 13 random matrices of size 1000 x 1000
    matrices = [np.random.rand(7000,7000) for _ in range(5)]

    print("Launch without multiprocessing")

    # Single processor execution
    start_time = time.time()
    results = list(add_mat(matrices[s], matrices[s+1]) for s in range(len(matrices)-1))
    
    end_time = time.time()
    print(results)
    print(f"Single processor execution time: {end_time - start_time:.2f} seconds")

    print("Launch with multiprocessing")

    # Multiple processor execution
    num_processors = mp.cpu_count()
    print(f"Number of processors: {num_processors}")
    
    pool = mp.Pool(processes=num_processors)
    start_time = time.time()
    results = list(pool.apply_async(add_mat, args=(matrices[s], matrices[s+1])) for s in range(len(matrices)-1))
    print(results)
    results = [r.get() for r in results]
    end_time = time.time()
    print(results)


    print(f"Multiple processor execution time: {end_time - start_time:.2f} seconds")
