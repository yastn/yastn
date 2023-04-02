from ..initialize import make_config

class meta_operators():
    # Predefine common elements of all operator classes.
    def __init__(self, **kwargs):
        r""" Common elements for all operator class. """
        self.config = make_config(**kwargs)
        self.s = (1, -1)

    def random_seed(self, seed):
        self.config.backend.random_seed(seed)
