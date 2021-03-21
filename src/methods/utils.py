import torch
import torch.tensor as tensor
import torch.nn as nn


def get_one_hot(y_s: torch.tensor, num_classes: int):
    """
        args:
            y_s : torch.Tensor of shape [n_task, shot]
        returns
            y_s : torch.Tensor of shape [n_task, shot, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    one_hot = torch.zeros(one_hot_size, device=y_s.device)
    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)
    return one_hot


def compute_centroids(z_s: torch.tensor,
                      y_s: torch.tensor):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0)).transpose(1, 2)  # [batch, K, s_shot]
    centroids = one_hot.bmm(z_s) / one_hot.sum(-1, keepdim=True)  # [batch, K, d]
    return centroids


def extract_features(x_s: tensor, x_q: tensor, model: nn.Module):
    """
    Extract features from support and query set using the provided model
        args:
            x_s : torch.Tensor of size [batch, s_shot, c, h, w]
            x_q : torch.Tensor of size [batch, q_shot, c, h, w]
        returns
            z_s : torch.Tensor of shape [batch, s_shot, d]
            z_s : torch.Tensor of shape [batch, q_shot, d]
    """
    batch, s_shot = x_s.size()[:2]
    q_shot = x_q.size(1)
    feat_dim = x_s.size()[-3:]
    z_s = model.extract_features(x_s.view(batch * s_shot, *feat_dim))
    z_q = model.extract_features(x_q.view(batch * q_shot, *feat_dim))
    z_s = z_s.view(batch, s_shot, -1)  # [batch, s_shot, d]
    z_q = z_q.view(batch, q_shot, -1)  # [batch, q_shot, d]

    return z_s, z_q
