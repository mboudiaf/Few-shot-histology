import os
import yaml
import copy
from ast import literal_eval
from typing import List
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
import sys
cmaps = ['winter', 'hsv', 'Wistia', 'BuGn']


def make_episode_visualization(args: argparse.Namespace,
                               img_s: np.ndarray,
                               img_q: np.ndarray,
                               gt_s: np.ndarray,
                               gt_q: np.ndarray,
                               preds: np.ndarray,
                               save_path: str,
                               mean: List[float] = [0.485, 0.456, 0.406],
                               std: List[float] = [0.229, 0.224, 0.225]):

    max_support = args.max_s_visu
    max_query = args.max_q_visu
    # 0) Preliminary checks
    assert len(img_s.shape) == 4, f"Support shape expected : Ks x 3 x H x W or Ks x H x W x 3. Currently: {img_s.shape}"
    assert len(img_q.shape) == 4, f"Query shape expected : Kq x 3 x H x W or Kq x H x W x 3. Currently: {img_q.shape}"
    assert len(preds.shape) == 2, f"Predictions shape expected : Kq x num_classes. Currently: {preds.shape}"
    assert len(gt_s.shape) == 1,  f"Support GT shape expected : Ks. Currently: {gt_s.shape}"
    assert len(gt_q.shape) == 1,  f"Query GT shape expected : Kq. Currently: {gt_q.shape}"
    # assert img_s.shape[-1] == img_q.shape[-1] == 3, "Images need to be in the format H x W x 3"
    if img_s.shape[1] == 3:
        img_s = np.transpose(img_s, (0, 2, 3, 1))
    if img_q.shape[1] == 3:
        img_q = np.transpose(img_q, (0, 2, 3, 1))

    assert img_s.shape[-3:-1] == img_q.shape[-3:-1], f"Support's resolution is {img_s.shape[-3:-1]} \
                                                      Query's resolution is {img_q.shape[-3:-1]}"

    print(f"Support images between {img_s.min()} and {img_s.max()} -> Renormalizing")
    img_s *= std
    img_s += mean
    print(f"Post normalization : {img_s.min()} and {img_s.max()}")

    # if img_q.min() < 0:
    print(f"Query images between {img_q.min()} and {img_q.max()} -> Renormalizing")
    img_q *= std
    img_q += mean
    print(f"Post normalization : {img_q.min()} and {img_q.max()}")

    Kq, num_classes = preds.shape
    Ks = img_s.shape[0]

    # Group samples by class
    samples_s = {}
    samples_q = {}
    preds_q = {}
    for class_ in np.unique(gt_s):
        samples_s[class_] = img_s[gt_s == class_]
        samples_q[class_] = img_q[gt_q == class_]
        preds_q[class_] = preds[gt_q == class_]
    # Create Grid
    max_s = min(max_support, np.max([v.shape[0] for v in samples_s.values()]))
    max_q = min(max_query, np.max([v.shape[0] for v in samples_q.values()]))
    n_rows = max_s + max_q
    n_columns = num_classes
    fig = plt.figure(figsize=(18, 18), dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_rows, n_columns),
                     axes_pad=(0.4, 0.4),
                     direction='row',
                     )

    # 1) visualize the support set
    for i in range(max_s):
        for j in range(n_columns):
            ax = grid[n_columns*i + j]
            if i < len(samples_s[j]):
                img = samples_s[j][i]
                # print(img.min(), img.max())
                # assert img.min() >= 0. and img.max() <= 1.0, (img.min(), img.max())
                make_plot(ax, img)
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Class {j+1}', size=20)

    # 1) visualize the query set
    for i in range(max_s, max_s + max_q):
        for j in range(n_columns):
            ax = grid[n_columns*i + j]
            if i - max_s < len(samples_q[j]):
                img = samples_q[j][i - max_s]
                # print(img.min(), img.max())
                # assert img.min() >= 0. and img.max() <= 1.0, (img.min(), img.max())
                make_plot(ax, img, preds_q[j][i - max_s])
            ax.axis('off')
    fig.suptitle(args.method, size=32, weight='bold', y=0.97)
    fig.text(-0.1, 0.72, 'Support', rotation=90, size=45)
    fig.text(-0.1, 0.35, 'Query', rotation=90, size=45)
    # fig.text(0.06, 0.32, '{', size=500, weight=0.)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    fig.clf()
    print(f"Figure saved at {save_path}")


def make_plot(ax: matplotlib.axes.Axes,
              img: np.ndarray,
              preds: np.ndarray = None):

    ax.imshow(img)
    if preds is not None:
        title = ['{:.2f}'.format(p) for p in preds]
        title[np.argmax(preds)] = r'$\mathbf{{{}}}$'.format(title[np.argmax(preds)])
        title = '/'.join(title)
        # print(title)
        ax.set_title(title, size=14)


def load_checkpoint(model, model_path, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path))
        print(f'Loaded model from {model_path}/model_best.pth.tar')
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(model_path))
        print(f'Loaded model from {model_path}/checkpoint.pth.tar')
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    state_dict = checkpoint['state_dict']
    names = []
    for k, v in state_dict.items():
        names.append(k)
    model.load_state_dict(state_dict)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, init, alpha=0.2):
        self.val = val
        if init:
            self.avg = val
        else:
            self.avg = alpha * val + (1 - alpha) * self.avg


def get_model_dir(args: argparse.Namespace, seed: int):
    model_type = args.method if args.episodic_training else 'standard'
    train = "train={}".format('_'.join(args.train_sources))
    valid = "valid={}".format('_'.join(args.val_sources))
    return os.path.join(args.ckpt_path,
                        train,
                        valid,
                        f'arch={args.arch}',
                        f'method={model_type}',
                        f'seed={seed}')


def save_checkpoint(state, folder, filename='model_best.pth.tar'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))


def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    return m, pm


def enablePrint():
    sys.stdout = sys.__stdout__


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], subkey, full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg
