
# def test_NtuEnv():
#     """ nearest environmental sites around a bond;  creates indices of the NTU environment """

#     bd00_01_h = fpeps.Bond(site0 = (0, 0), site1=(0, 1), dirn='h')
#     bd11_21_v = fpeps.Bond(site0 = (1, 1), site1=(2, 1), dirn='v')

#     net_1 = fpeps.SquareLattice(lattice='square', dims=(3, 4), boundary='obc')  # dims = (rows, columns) # finite lattice
#     assert net_1.tensors_NtuEnv(bd00_01_h) == {'tl': None, 'l': None, 'bl': (1, 0), 'tr': None, 'r': (0, 2), 'br': (1, 1)}
#     assert net_1.tensors_NtuEnv(bd11_21_v) == {'tl': (1, 0), 't': (0, 1), 'tr': (1, 2), 'bl': (2, 0), 'b': None, 'br': (2, 2)}

#     net_2 = fpeps.SquareLattice(lattice='square', dims=(3, 4), boundary='infinite')  # dims = (rows, columns) # infinite lattice
#     assert net_2.tensors_NtuEnv(bd00_01_h) == {'tl': (2, 0), 'l': (0, 3), 'bl': (1, 0), 'tr': (2, 1), 'r': (0, 2), 'br': (1, 1)}
#     assert net_2.tensors_NtuEnv(bd11_21_v) == {'tl': (1, 0), 't': (0, 1), 'tr': (1, 2), 'bl': (2, 0), 'b': (0, 1), 'br': (2, 2)}
