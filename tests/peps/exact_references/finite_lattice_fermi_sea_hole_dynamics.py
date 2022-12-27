import numpy as np
import scipy.linalg as LA

r"""
This code calculates the magnitude of fermionic hopping correlator between any two sites of a finite lattice
"""
#### insert a hole in a 3x3 hamiltonian spinless fermi sea with a lattice sites filled with a particle
#### construct the 3x3 Hamiltonian

nx = 3
ny = 3
nn = nx*ny

siden = np.array([[1, 0], [0, 1]])
sc = np.array([[0, 1], [0, 0]])
scdag = np.array([[0, 1], [0, 0]])


zero = np.array([1, 0])
one = np.array([0, 1])

def c(posn):
    sarr = [None] * 9
    for x in range(len(sarr)):
        sarr[x] = siden
    sarr[posn] = sc
    sarray = op(sarr)

    return sarray

def cdag(posn):
    sarr = [None] * 9
    for x in range(len(sarr)):
        sarr[x] = siden
    sarr[posn] = scdag
    sarray = op(sarr)
    
    return sarray

def op(sarray):
    s = np.kron(sarray[0], sarray[1])
    s = np.kron(s, sarray[2])
    s = np.kron(s, sarray[3])
    s = np.kron(s, sarray[4])
    s = np.kron(s, sarray[5])
    s = np.kron(s, sarray[6])
    s = np.kron(s, sarray[7])
    s = np.kron(s, sarray[8])

    return s

def initialize_hole(posn):
    sarr = [None] * 9
    for x in range(len(sarr)):
        sarr[x] = one
   # sarr[posn] = one
    sarray = op(sarr)
    
    return sarray

t=1

"""H = -t*(cdag(0)@c(1) + cdag(1)@c(0) +cdag(1)@c(2) + cdag(2)@c(1) +cdag(3)@c(4) + cdag(4)@c(3) +cdag(4)@c(5) +cdag(5)@c(4) +
 cdag(6)@c(7) + cdag(7)@c(6) +cdag(7)@c(8) + cdag(8)@c(7) +cdag(0)@c(3) + cdag(3)@c(0) +cdag(1)@c(4) + cdag(4)@c(1) +cdag(2)@c(5)
+ cdag(5)@c(2) +cdag(3)@c(6) + cdag(6)@c(3) + cdag(4)@c(7) + cdag(7)@c(4) +cdag(5)@c(8) + cdag(8)@c(5))"""

eig_init = initialize_hole(4)
n4 = cdag(0)@c(0)
hh= n4@eig_init
print(hh)





    
