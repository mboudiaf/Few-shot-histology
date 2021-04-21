import argparse
import torch
from typing import Dict, Tuple
import torch.distributed as dist
import torch.nn.functional as F

from .utils import get_one_hot, extract_features
from .method import FSmethod


class Finetune(FSmethod):

    def __init__(self,
                 args: argparse.Namespace):
        self.iter = args.iter
        self.lr = args.finetune_lr
        self.finetune_all_layers = args.finetune_all_layers
        self.normalize = args.normalize
        super().__init__(args)

    # def record_info(self,
    #                 metrics: dict,
    #                 task_ids: tuple,
    #                 iteration: int,
    #                 preds_q: torch.tensor,
    #                 probs_s: torch.tensor,
    #                 y_q: torch.tensor,
    #                 y_s: torch.tensor):
    #     """
    #     inputs:
    #         x_s : torch.Tensor of shape [n_task, s_shot, feature_dim]
    #         x_q : torch.Tensor of shape [n_task, q_shot, feature_dim]
    #         y_s : torch.Tensor of shape [n_task, s_shot]
    #         y_q : torch.Tensor of shape [n_task, q_shot] :
    #     """
    #     if metrics:
    #         kwargs = {'preds': preds_q, 'gt': y_q, 'probs_s': probs_s,
    #                   'gt_s': y_s}

    #         for metric_name in metrics:
    #             metrics[metric_name].update(task_ids[0],
    #                                         task_ids[1],
    #                                         iteration,
    #                                         **kwargs)

    def forward(self,
                model: torch.nn.Module,
                x_s: torch.tensor,
                x_q: torch.tensor,
                y_s: torch.tensor,
                y_q: torch.tensor,
                task_ids: Tuple[int, int] = None):
        """
        Corresponds to the TIM-GD inference
        inputs:
            x_s : torch.Tensor of shape [n_task, s_shot, feature_dim]
            x_q : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        device = x_s.device
        model.eval()
        n_tasks = x_s.size(0)
        if n_tasks > 1:
            raise ValueError('Finetune method can only deal with 1 task at a time. \
                             Currently {} tasks.'.format(n_tasks))
        y_s = y_s[0]
        y_q = y_q[0]
        num_classes = y_s.unique().size(0)
        y_s_one_hot = get_one_hot(y_s, num_classes)

        # Initialize classifier
        with torch.no_grad():
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)
            if self.normalize:
                z_s = F.normalize(z_s, dim=-1)
                z_q = F.normalize(z_q, dim=-1)
            classifier = torch.nn.Linear(z_s.size(-1), num_classes, bias=True).to(device)
            preds_q = classifier(z_q[0]).argmax(-1)
            probs_s = classifier(z_s[0]).softmax(-1)
            # self.record_info(iteration=0,
            #                  task_ids=task_ids,
            #                  preds_q=preds_q,
            #                  probs_s=probs_s,
            #                  y_q=y_q,
            #                  y_s=y_s)

        # Define optimizer
        if self.finetune_all_layers:
            params = list(model.parameters()) + list(classifier.parameters())
        else:
            params = classifier.parameters()  # noqa: E127
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Run adaptation
        with torch.set_grad_enabled(self.finetune_all_layers):
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)
            if self.normalize:
                z_s = F.normalize(z_s, dim=-1)
                z_q = F.normalize(z_q, dim=-1)

        for i in range(1, self.iter):
            probs_s = classifier(z_s[0]).softmax(-1)
            loss = - (y_s_one_hot * probs_s.log()).sum(-1).mean(-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # with torch.no_grad():
                # preds_q = classifier(z_q[0]).argmax(-1)
                # self.record_info(iteration=i,
                #                  task_ids=task_ids,
                #                  preds_q=preds_q,
                #                  probs_s=probs_s,
                #                  y_q=y_q,
                #                  y_s=y_s)

        probs_q = classifier(z_q[0]).softmax(-1).unsqueeze(0)
        return loss.detach(), probs_q.detach()
