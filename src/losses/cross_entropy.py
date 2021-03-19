import torch
import torch.nn as nn
from .loss import _Loss


class _CrossEntropy(_Loss):

    def loss_fn(self,
                logits: torch.tensor,
                one_hot_targets: torch.tensor):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        loss = - (one_hot_targets * logsoftmax).sum(1)
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