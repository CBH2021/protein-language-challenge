import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """ Base class for dataset """

    def __init__(self, path: str):
        """ Constructor
        Args:
            path: file path for the dataset
        """
        data = torch.from_numpy(np.load(path)['data'])

        self.X = data[:, :, :1300].clone().detach().float()
        self.y = data[:, :, 1300:].clone().detach().float()

        del data

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, torch.tensor):
        """ Returns input, label and mask
        Args:
            index: Index of the array
        """

        return self.X[index], self.y[index], self.y[index][0]

    def __len__(self):
        """ Returns the length of the data """

        return len(self.X)