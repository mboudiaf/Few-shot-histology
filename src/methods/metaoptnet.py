import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .method import FSmethod
from .utils import extract_features, get_one_hot
from torch.autograd import Variable
from qpth.qp import QPFunction
from .classification_heads import ClassificationHead


class MetaOptNet(FSmethod):
    '''
    Metaoptnet method
    @inproceedings{lee2019meta,
    title={Meta-Learning with Differentiable Convex Optimization},
    author={Kwonjoon Lee and Subhransu Maji and Avinash Ravichandran and Stefano Soatto},
    booktitle={CVPR},
    year={2019}
    }
    '''

    def __init__(self, args: argparse.Namespace):

        self.eps = args.eps
        self.head = args.head
        self.batch_size = args.batch_size
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

        num_classes = y_s.unique().size(0)
        n_shots = int(x_s.shape[1]/num_classes)

        if not self.training:
            with torch.no_grad():
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
        else:
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

        if self.head == 'ProtoNet':
            cls_head = ClassificationHead(base_learner='ProtoNet').to(x_s.device)
        elif self.head == 'Ridge':
            cls_head = ClassificationHead(base_learner='Ridge').to(x_s.device)
        elif self.head == 'R2D2':
            cls_head = ClassificationHead(base_learner='R2D2').to(x_s.device)
        elif self.head == 'SVM':
            cls_head = ClassificationHead(base_learner='SVM-CS').to(x_s.device)
        else:
            print ("Cannot recognize the dataset type")
            assert(False)
            
        if not self.training and "SVM" in self.head:
            logits_q = cls_head(z_q, z_s, y_s, num_classes, n_shots, maxIter=3)
        else:
            logits_q = cls_head(z_q, z_s, y_s, num_classes, n_shots)

        one_hot_q = get_one_hot(y_q.reshape(-1), num_classes)
        smoothed_one_hot_q = one_hot_q * (1 - self.eps) + (1 - one_hot_q) * self.eps / (num_classes - 1)
        log_probas = logits_q.reshape(-1, num_classes).log_softmax(-1)
        ce = -(smoothed_one_hot_q * log_probas).sum(-1)

        preds_q = logits_q.argmax(2)

        return ce, preds_q



