from typing import Optional
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..builder import LOSSES


@LOSSES.register_module()
class LogLoss(nn.Module):

    def __init__(self):
        super(LogLoss, self).__init__()
        self.accident = True

    def forward(self, frame, labels):
        if self.accident:
            loss = (- math.exp(-max(0, 76 - frame)) * torch.log(labels))
        else:
            loss = - torch.log(1 - labels)

        return loss
    
