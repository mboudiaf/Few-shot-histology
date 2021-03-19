import torch
import torch.nn as nn
from .loss import _Loss


class _FocalLoss(_Loss):

    def __init__(self,
                 **kwargs) -> None:
        super(_FocalLoss, self).__init__(**kwargs)
        self.gamma = kwargs['args'].focal_gamma

    def loss_fn(self,
                logits: torch.tensor,
                one_hot_targets: torch.tensor):
        softmax = logits.softmax(1)
        logsoftmax = torch.log(softmax + 1e-10)
        loss = - (one_hot_targets * (1 - softmax)**self.gamma * logsoftmax).sum(1)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss

    def forward(self,
                input_: torch.tensor,
                targets: torch.tensor,
                model: nn.Module):
        one_hot_targets = self.smooth_one_hot(targets)
        if self.augmentation:
            return self.augmentation(input_, one_hot_targets, model)
        else:
            logits = model(input_)
            return self.loss_fn(logits, one_hot_targets)