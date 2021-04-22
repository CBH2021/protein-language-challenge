import torch

class PlaceHolder(object):
    """ A placeholder augmentation class """
    def __init__(self):
        pass

    def __call__(self, x: torch.tensor):
        # do batch transformation
        return x