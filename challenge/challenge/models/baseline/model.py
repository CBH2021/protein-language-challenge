import torch
import torch.nn as nn

from challenge.base import ModelBase
from challenge.utils import setup_logger


log = setup_logger(__name__)


class Baseline(ModelBase):
    def __init__(self, in_features: int):
        """ Simple baseline model for prediction secondary structure
        Args:
            in_features: size in features
        """
        super(Baseline, self).__init__()

        # Task block
        self.ss8 = nn.Linear(in_features=in_features, out_features=8)
        self.ss3 = nn.Linear(in_features=in_features, out_features=3)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        ss8 = self.ss8(x)
        ss3 = self.ss3(x)

        return [ss8, ss3]

class NetSurfModel(ModelBase):
    def __init__(self, in_features: int, dropout):
        """ Simple baseline model for prediction secondary structure
        Args:
            in_features: size in features
        """
        super(NetSurfModel, self).__init__()

        self.in_features = in_features

        self.cnn1 = nn.Conv1d(in_channels=in_features, out_channels=(in_features/32), kernel_size=32, stride=1, padding=0) # 1280/32 = 40 out features
        self.cnn2 = nn.Conv1d(in_channels=in_features, out_channels=(in_features/32), kernel_size=32, stride=1, padding=0)
        self.bilstm = nn.LSTM(input_size=n_features/16, hidden_size=1024, num_layers=2, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.ss8 = nn.Linear(in_features=in_features, out_features=8)
        self.ss3 = nn.Linear(in_features=in_features, out_features=3)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        # Pass embeddings to two parallel CNNs
        x1 = self.cnn1(self.in_features)
        x2 = self.cnn2(self.in_features)

        # Concatenate outputs of parallel CNNs to form identity layer
        x = torch.cat(x1, x2, dim=0) 

        # Pass identity layer output to two-layer biLSTM
        x = self.bilstm(x)

        ss8 = self.ss8(x)
        ss3 = self.ss3(x)

        return [ss8, ss3]
