import torch
import numpy as np
from challenge.base import DatasetBase


class ChallengeData(DatasetBase):
    def __init__(self, *args, **kwargs):
        super(ChallengeData, self).__init__(*args, **kwargs)


class ChallengeDataOnlyEncoding(DatasetBase):
    def __init__(self, *args, **kwargs):
        super(ChallengeDataOnlyEncoding, self).__init__(*args, **kwargs)

        self.X = self.X[:, :, :20]


class ChallengeDataOnlyEmbedding(DatasetBase):
    def __init__(self, *args, **kwargs):
        super(ChallengeDataOnlyEmbedding, self).__init__(*args, **kwargs)

        self.X = self.X[:, :, 20:1300]
