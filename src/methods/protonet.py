import torch
import torch.nn as nn
import argparse
from .utils import get_one_hot
from .method import FSmethod


class Protonet(FSmethod):
    '''
    Simple Shot method
    '''

    def __init__(self, args: argparse.Namespace):

        self.episodic_training = True
        super().__init__(args)

    def compute_centroids(self,
                          z_s: torch.tensor,
                          y_s: torch.tensor):
        """
        inputs:
            z_s : torch.Tensor of shape [s_shot, d]
            y_s : torch.Tensor of shape [s_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, d]
        """
        one_hot = get_one_hot(y_s)  # [s_shot, K]
        counts = one_hot.sum(0)  # [K]
        weights = one_hot.transpose(0, 1).matmul(z_s)  # [K, d]
        centroids = weights / counts.unsqueeze(1)
        return centroids

    def forward(self,
                x_s: torch.tensor,
                x_q: torch.tensor,
                y_s: torch.tensor,
                y_q: torch.tensor,
                model: nn.Module):
        """
        inputs:
            x_s : torch.Tensor of shape [s_shot, c, h, w]
            x_q : torch.Tensor of shape [q_shot, c, h, w]
            y_s : torch.Tensor of shape [s_shot]
            y_q : torch.Tensor of shape [q_shot]
        """
        z_s = model.extract_features(x_s)  # [s_shot, d]
        z_q = model.extract_features(x_q)  # [q_shot, d]

        centroids = self.compute_centroids(z_s, y_s)  # [s_shot, K]

        l2_distance = (- 2 * z_q.matmul(centroids.transpose(0, 1)) \
                        + (centroids**2).sum(1).view(1, -1)  #  # noqa: E127
                        + (z_q**2).sum(1).view(-1, 1))  # [q_shot, K]

        log_probas = (-l2_distance).log_softmax(-1)
        one_hot_q = get_one_hot(y_q)  # [s_shot, K]
        ce = - (one_hot_q * log_probas).sum(-1)

        preds_q = l2_distance.detach().argmin(-1)

        return ce, preds_q
