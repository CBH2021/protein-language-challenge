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
    def __init__(self, in_features: int, hidden_size, lstm_layers, dropout):
        """ Simple baseline model for prediction secondary structure
        Args:
            in_features: size in features
        """
        super(NetSurfModel, self).__init__()
        in_features = 1280
        self.in_features = in_features

        self.cnn1 = nn.Conv1d(in_channels=in_features, out_channels=int(in_features/32), kernel_size=32, stride=1, padding=0) # in = 1280, out = 40
        self.cnn2 = nn.Conv1d(in_channels=in_features, out_channels=int(in_features/32), kernel_size=32, stride=1, padding=0) # in = 1632, out = 51
        self.bilstm = nn.LSTM(input_size=int(in_features/16), hidden_size=hidden_size, num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(in_features=int(hidden_size*2), out_features=8)
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
        x = x.permute(0, 2, 1)
        print(f"3. x shape is: {x.shape}")
        x, (h, c) = self.bilstm(x) # Output of LSTM is output, (hidden, cells)
        print(f"4. x shape is: {x.shape}")

        # Convert to 8 outputs for each class
        x = self.fc1(x)
        print(f"5. x shape is: {x.shape}")

        ss8 = x
        # print(ss8)
        ss3 = self.ss3(x)

        return [ss8, ss3]
