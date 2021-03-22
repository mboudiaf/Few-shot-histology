import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .method import FSmethod
from .utils import extract_features


class MatchingNet(FSmethod):
    '''
    Simple Shot method
    '''

    def __init__(self, args: argparse.Namespace):

        self.episodic_training = True
        self.eps = args.eps
        super().__init__(args)

    def forward(self,
                x_s: torch.tensor,
                x_q: torch.tensor,
                y_s: torch.tensor,
                y_q: torch.tensor,
                model: nn.Module):
        """
        inputs:
            x_s : torch.Tensor of shape [batch, s_shot, c, h, w]
            x_q : torch.Tensor of shape [batch, q_shot, c, h, w]
            y_s : torch.Tensor of shape [batch, s_shot]
            y_q : torch.Tensor of shape [batch, q_shot]
        """
        if not self.training:
            with torch.no_grad():
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
        else:
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

        logits = self.matching_log_probas(z_s,
                                          y_s,
                                          z_q)  # [batch, num_classes, q_shot]
        loss = F.nll_loss(logits, y_q)
        preds = logits.detach().permute(0, 2, 1).softmax(-1)
        return loss, preds

    def pairwise_cosine_similarity(self, z_s1, z_s2):
        r"""Computes the pairwise cosine similarity between two tensors of z_s.
        Parameters
        ----------
        z_s1 : `torch.Tensor` instance
            A tensor containing z_s with shape
            `(batch_size, N, d)`.
        z_s2 : `torch.Tensor` instance
            A tensor containing z_s with shape
            `(batch_size, M, d)`.
        Returns
        -------
        similarities : `torch.Tensor` instance
            A tensor containing the pairwise cosine similarities between the vectors
            in `z_s1` and `z_s2`. This tensor has shape
            `(batch_size, N, M)`.
        Notes
        -----
        The cosine similarity is computed as
            .. math ::
                \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
        """
        sq_norm1 = torch.sum(z_s1 ** 2, dim=2, keepdim=True)
        sq_norm2 = torch.sum(z_s2 ** 2, dim=2).unsqueeze(1)
        dot_product = torch.bmm(z_s1, z_s2.transpose(1, 2))
        inverse_norm = torch.rsqrt(torch.clamp(sq_norm1 * sq_norm2, min=self.eps ** 2))
        return dot_product * inverse_norm

    def matching_log_probas(self, z_s, y_s, z_q):
        """Computes the log-probability of test samples given the training dataset
        for the matching network [1].
        Parameters
        ----------
        z_s : `torch.Tensor` instance
            A tensor containing the z_s of the train/support inputs. This
            tensor has shape `(batch_size, num_train_samples, d)`.
        y_s : `torch.LongTensor` instance
            A tensor containing the targets of the train/support dataset. This tensor
            has shape `(batch_size, num_train_samples)`.
        z_q : `torch.Tensor` instance
            A tensor containing the z_s of the test/query inputs. This tensor
            has shape `(batch_size, num_test_samples, d)`.
        num_classes : int
            Number of classes (i.e. `N` in "N-way classification") in the
            classification task.
        eps : float (default: 1e-8)
            Small value to avoid division by zero.
        Returns
        -------
        log_probas : `torch.Tensor` instance
            A tensor containing the log-probabilities of the test samples given the
            training dataset for the matching network. This tensor has shape
            `(batch_size, num_classes, num_test_samples)`.
        References
        ----------
        .. [1] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016).
               Matching Networks for One Shot Learning. In Advances in Neural
               Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)
        """
        num_classes = y_s.unique().size(0)
        batch_size, num_samples, _ = z_q.shape
        similarities = self.pairwise_cosine_similarity(z_s, z_q)
        logsumexp = torch.logsumexp(similarities, dim=1, keepdim=True)

        max_similarities, _ = torch.max(similarities, dim=1, keepdim=True)
        exp_similarities = torch.exp(similarities - max_similarities)

        sum_exp = exp_similarities.new_zeros((batch_size, num_classes, num_samples))
        indices = y_s.unsqueeze(-1).expand_as(exp_similarities)
        sum_exp.scatter_add_(1, indices, exp_similarities)

        return torch.log(sum_exp) + max_similarities - logsumexp