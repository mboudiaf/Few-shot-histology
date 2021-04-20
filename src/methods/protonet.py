import torch
import torch.nn as nn
import argparse
from .utils import get_one_hot, compute_centroids, extract_features
from .method import FSmethod


class ProtoNet(FSmethod):
    '''
    Simple Shot method
    '''

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def forward(self,
                x_s: torch.tensor,
                x_q: torch.tensor,
                y_s: torch.tensor,
                y_q: torch.tensor,
                model: nn.Module):
        """
        inputs:
            x_s : torch.Tensor of size [s_shot, c, h, w]
            x_q : torch.Tensor of size [q_shot, c, h, w]
            y_s : torch.Tensor of size [s_shot]
            y_q : torch.Tensor of size [q_shot]
        """
        num_classes = y_s.unique().size(0)
        if not self.training:
            with torch.no_grad():
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
        else:
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

        centroids = compute_centroids(z_s, y_s)  # [batch, num_class, d]

        l2_distance = (- 2 * z_q.matmul(centroids.transpose(1, 2)) \
                        + (centroids**2).sum(2).unsqueeze(1)  # noqa: E127
                        + (z_q**2).sum(2).unsqueeze(-1))  # [batch, q_shot, num_class]

        log_probas = (-l2_distance).log_softmax(-1)  # [batch, q_shot, num_class]
        one_hot_q = get_one_hot(y_q, num_classes)  # [batch, q_shot, num_class]
        ce = - (one_hot_q * log_probas).sum(-1)  # [batch, q_shot, num_class]

        preds_q = log_probas.detach().exp()

        return ce, preds_q
