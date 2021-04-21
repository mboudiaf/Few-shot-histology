import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from collections import OrderedDict
from .method import FSmethod
from ..models.meta.metamodules import MetaModule
from ..models.meta.metamodules.batchnorm import _MetaBatchNorm
from .utils import get_one_hot


class MAML(FSmethod):
    '''
    Simple Shot method
    '''

    def __init__(self, args: argparse.Namespace):

        self.step_size = args.step_size
        self.first_order = args.first_order
        self.num_steps = args.num_steps
        super().__init__(args)

    def freeze_bn(self, model):
        for module in model.modules():
            if isinstance(module, _MetaBatchNorm):
                module.eval()

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
        model.train()
        device = x_s.device
        outer_loss = torch.tensor(0., device=device)
        soft_preds = torch.zeros_like(get_one_hot(y_q, y_s.unique().size(0)))
        for task_idx, (support, y_support, query, y_query) in \
                enumerate(zip(x_s, y_s, x_q, y_q)):
            params = None
            for i in range(self.num_steps):
                train_logit = model(support, params=params)
                inner_loss = F.cross_entropy(train_logit, y_support)

                model.zero_grad()
                params = self.gradient_update_parameters(model,
                                                         inner_loss,
                                                         step_size=self.step_size,
                                                         first_order=self.first_order,
                                                         params=params)

            with torch.set_grad_enabled(self.training):
                query_logit = model(query, params=params)
                outer_loss += F.cross_entropy(query_logit, y_query)
                soft_preds[task_idx] = query_logit.detach().softmax(-1)
        model.eval()
        return outer_loss, soft_preds

    def gradient_update_parameters(self,
                                   model,
                                   loss,
                                   params=None,
                                   step_size=0.5,
                                   first_order=False):
        """Update of the meta-parameters with one step of gradient descent on the
        loss function.
        Parameters
        ----------
        model : `torchmeta.modules.MetaModule` instance
            The model.
        loss : `torch.Tensor` instance
            The value of the inner-loss. This is the result of the training dataset
            through the loss function.
        params : `collections.OrderedDict` instance, optional
            Dictionary containing the meta-parameters of the model. If `None`, then
            the values stored in `model.meta_named_parameters()` are used. This is
            useful for running multiple steps of gradient descent as the inner-loop.
        step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
            The step size in the gradient update. If an `OrderedDict`, then the
            keys must match the keys in `params`.
        first_order : bool (default: `False`)
            If `True`, then the first order approximation of MAML is used.
        Returns
        -------
        updated_params : `collections.OrderedDict` instance
            Dictionary containing the updated meta-parameters of the model, with one
            gradient update wrt. the inner-loss.
        """
        if not isinstance(model, MetaModule):
            raise ValueError('The model must be an instance of `torchmeta.modules.'
                             'MetaModule`, got `{0}`'.format(type(model)))

        if params is None:
            params = OrderedDict(model.meta_named_parameters())

        grads = torch.autograd.grad(loss,
                                    params.values(),
                                    create_graph=not first_order)

        updated_params = OrderedDict()

        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size * grad

        return updated_params