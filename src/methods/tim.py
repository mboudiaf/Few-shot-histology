import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from .utils import get_one_hot, compute_centroids, extract_features
from .method import FSmethod


class TIM(FSmethod):
    '''
    Simple Shot method
    '''

    def __init__(self, args: argparse.Namespace):

        self.episodic_training = False
        self.temp = args.temp
        self.normalize = args.normalize
        self.iter = args.iter
        self.lr = args.tim_lr
        self.loss_weights = args.loss_weights
        super().__init__(args)

    def get_logits(self, samples: torch.tensor, centroids: torch.tensor):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = self.temp * (samples.matmul(centroids.transpose(1, 2)) \
                              - 1 / 2 * (centroids**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

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
        if self.normalize:
            z_s = F.normalize(z_s, dim=2)
            z_q = F.normalize(z_q, dim=2)

        centroids = compute_centroids(z_s, y_s)  # [batch, num_class, d]
        centroids.requires_grad_()
        optimizer = torch.optim.Adam([centroids], lr=self.lr)

        one_hot_s = get_one_hot(y_s, num_classes)  # [batch, q_shot, num_class]

        for i in range(self.iter):
            logits_s = self.get_logits(z_s, centroids)
            logits_q = self.get_logits(z_q, centroids)

            ce = - (one_hot_s * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)
            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds_q = self.get_logits(z_q, centroids).detach()
        return loss, preds_q