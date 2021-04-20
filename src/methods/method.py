import torch.nn as nn
import torch.tensor as tensor
import argparse
from typing import Tuple


class FSmethod(nn.Module):
    '''
    Abstract class for few-shot methods
    '''
    def __init__(self, args: argparse.Namespace):
        super(FSmethod, self).__init__()

    def forward(self,
                x_q: tensor,
                y_s: tensor,
                y_q: tensor,
                model: nn.Module) -> Tuple[tensor, tensor]:

        raise NotImplementedError
