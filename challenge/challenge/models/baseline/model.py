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

        ss8 = self.ss8(x) # shape is [aa, feats, 8]
        ss3 = self.ss3(x)
        return [ss8, ss3]

class NetSurfModel(ModelBase):
    def __init__(self, in_features: int, dropout):
        """ Simple baseline model for prediction secondary structure
        Args:
            in_features: size in features
        """
        super(NetSurfModel, self).__init__()
        in_features = 1280
        self.in_features = in_features

        self.cnn1 = nn.Conv1d(in_channels=in_features, out_channels=int(in_features/32), kernel_size=32, stride=1, padding=0) # in = 1280, out = 40
        self.cnn2 = nn.Conv1d(in_channels=in_features, out_channels=int(in_features/32), kernel_size=32, stride=1, padding=0) # in = 1632, out = 51
        self.bilstm = nn.LSTM(input_size=int(in_features/16), hidden_size=1024, num_layers=2, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.ss8 = nn.Linear(in_features=in_features, out_features=8)
        self.ss3 = nn.Linear(in_features=in_features, out_features=3)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        print(f"0. x shape is: {x.shape}")
        x = x.permute(0,2,1)
         # Pass embeddings to two parallel CNNs
        print(f"1. x shape is: {x.shape}")
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        print(f"x1 shape is {x1.shape} and x2 is {x2.shape}")

        # Concatenate outputs of parallel CNNs to form identity layer
        x = torch.cat((x1, x2), dim=1) 
        print(f"2. x shape is: {x.shape}")

        # Pass identity layer output to two-layer biLSTM
        x = self.bilstm(x)
        print(f"3. x shape is: {x.shape}")

        # Convert to 8 outputs for each class
        x = nn.Linear(x, out_features=8)

        ss8 = x
        ss3 = self.ss3(x)

        return [ss8, ss3]
