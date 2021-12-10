import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import ACCIDENTMODEL


@ACCIDENTMODEL.register_module()
class AccidentLSTM(nn.Module):
    def __init__(self,
                 temporal_feature: int,
                 hidden_feature: int,
                 num_layers: int):
        super(AccidentLSTM, self).__init__()

        self.acc_lstm = torch.nn.LSTM(temporal_feature, hidden_feature, num_layers=num_layers)
        self.hidden_feature = hidden_feature
        self.pred = nn.Sequential(
                nn.Linear(hidden_feature, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1))

    def forward(self,
                inputs,
                h0,
                c0):

        output, (hn, cn) = self.acc_lstm(inputs, (h0, c0))

        return self.pred(output)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

