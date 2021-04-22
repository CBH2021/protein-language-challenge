import numpy as np
import challenge.data_loader.dataset_loaders as module_dataset

from challenge.base import DataLoaderBase


class ChallengeDataLoader(DataLoaderBase):
    def __init__(self, *args, **kwargs):
        self.dataset_loader = getattr(module_dataset, kwargs['dataset_loader'])

        super(ChallengeDataLoader, self).__init__(*args, **kwargs)
