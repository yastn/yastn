import numpy as np
import time

# Define the number of matrices and their dimensions
num_matrices = 7
m, n = 1000, 1000

# Generate the list of random matrices
matrices = [np.random.rand(m, n) for i in range(num_matrices)]

# Measure the execution time
t_start = time.time()

# Compute the singular values of the matrices
singular_values = []
for i in range(num_matrices):
    singular_values.append(np.linalg.svd(matrices[i])[1])

# Measure the execution time
t_end = time.time()
print("Elapsed time:", t_end - t_start)