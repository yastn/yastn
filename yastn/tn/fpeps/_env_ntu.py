
# class EnvCluster(depth = 1):
#     def __init__(self, psi):
#         self.psi = psi
#         self.g = psi.g

#     # does not have data container
#     def bond_metric(self, bond):
#         pass



# class EnvCtm:
#     def __init__(self, psi):
#         self.psi = psi
#         self.g = psi.g



def tensors_NtuEnv(self, bds):
    r""" Returns a dictionary containing the neighboring sites of the bond `bds`.
                The keys of the dictionary are the direction of the neighboring site with respect to
                the bond: 'tl' (top left), 't' (top), 'tr' (top right), 'l' (left), 'r' (right),
                'bl' (bottom left), and 'b' (bottom)"""

    neighbors = {}
    site_1, site_2 = bds.site_0, bds.site_1
    if self.lattice == 'checkerboard':
        if bds.dirn == 'h':
            neighbors['tl'], neighbors['l'], neighbors['bl'] = site_2, site_2, site_2
            neighbors['tr'], neighbors['r'], neighbors['br'] = site_1, site_1, site_1
        elif bds.dirn == 'v':
            neighbors['tl'], neighbors['t'], neighbors['tr'] = site_2, site_2, site_2
            neighbors['bl'], neighbors['b'], neighbors['br'] = site_1, site_1, site_1
    else:
        if bds.dirn == 'h':
            neighbors['tl'], neighbors['l'], neighbors['bl'] = self.nn_site(site_1, d='t'), self.nn_site(site_1, d='l'), self.nn_site(site_1, d='b')
            neighbors['tr'], neighbors['r'], neighbors['br'] = self.nn_site(site_2, d='t'), self.nn_site(site_2, d='r'), self.nn_site(site_2, d='b')
        elif bds.dirn == 'v':
            neighbors['tl'], neighbors['t'], neighbors['tr'] = self.nn_site(site_1, d='l'), self.nn_site(site_1, d='t'), self.nn_site(site_1, d='r')
            neighbors['bl'], neighbors['b'], neighbors['br'] = self.nn_site(site_2, d='l'), self.nn_site(site_2, d='b'), self.nn_site(site_2, d='r')

    return neighbors
