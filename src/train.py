import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import time
import torch.backends.cudnn as cudnn
import random
from functools import partial

from .models.standard import __dict__ as standard_dict
from .models.meta import __dict__ as meta_dict
from .dataset.utils import Split
from .dataset import config as config_lib
from .dataset import dataset_spec as dataset_spec_lib
from .dataset import pipeline
from .dataset.pipeline import worker_init_fn_
from .methods import __dict__ as all_methods
from .losses import __dict__ as all_losses
from .utils import load_cfg_from_cfg_file, merge_cfg_from_list, AverageMeter, \
                   save_checkpoint, get_model_dir, make_episode_visualization


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--method_config', type=str, default=True, help='Base config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_dataloader(args: argparse.Namespace,
                   sources: List[str],
                   episodic: bool,
                   batch_size: int,
                   split: Split):
    data_config = config_lib.DataConfig(args)
    episod_config = config_lib.EpisodeDescriptionConfig(args)
    use_bilevel_ontology_list = [False]*len(sources)
    if episod_config.num_ways and len(sources) > 1:
        raise ValueError('For fixed episodes, not tested yet on > 1 dataset')
    else:
        # Enable ontology aware sampling for breakhis
        if 'breakhis' in sources:
            use_bilevel_ontology_list[sources.index('breakhis')] = True
    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list

    all_dataset_specs = []
    for dataset_name in sources:
        dataset_records_path = os.path.join(data_config.data_path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    if episodic:
        pipeline_fn = pipeline.make_episode_pipeline
    else:
        pipeline_fn = pipeline.make_batch_pipeline

    dataset = pipeline_fn(dataset_spec_list=all_dataset_specs,
                          split=split,
                          data_config=data_config,
                          episode_descr_config=episod_config)
    worker_init_fn = partial(worker_init_fn_, seed=args.seed)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=args.num_workers,
                        worker_init_fn=worker_init_fn)
    num_classes = sum([len(d_spec.get_classes(split=Split["TRAIN"])) for d_spec in all_dataset_specs])
    return loader, num_classes


def main(args):

    if args.seeds:
        args.seed = args.seeds[0]
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

    # ============ Device ================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = get_model_dir(args, args.seed)

    # ============ Training method ================
    method = all_methods[args.method](args=args)
    print(f"=> Using {args.method} method")

    # ============ Data loaders =========
    train_loader, num_classes = get_dataloader(args=args,
                                               sources=args.train_sources,
                                               episodic=args.episodic_training,
                                               batch_size=args.batch_size,
                                               split=Split["TRAIN"])

    val_loader, num_classes_val = get_dataloader(args=args,
                                                 sources=args.val_sources,
                                                 episodic=True,
                                                 batch_size=args.val_batch_size,
                                                 split=Split["VALID"])

    test_loader, num_classes_test = get_dataloader(args=args,
                                                   sources=args.test_sources,
                                                   episodic=True,
                                                   batch_size=args.val_batch_size,
                                                   split=Split["TEST"])

    #  If you want to get the total number of classes (i.e from combined datasets)
    if not args.episodic_training:
        loss_fn = all_losses[args.loss](args=args, num_classes=num_classes, reduction='none')

    print(f"=> There are {num_classes} classes in the train datasets")
    print(f"=> There are {num_classes_val} classes in the validation datasets")
    print(f"=> There are {num_classes_test} classes in the test datasets")

    # ============ Model and optim ================
    if 'MAML' in args.method:
        print(f"Meta {args.arch} loaded")
        model = meta_dict[args.arch](pretrained=args.pretrained, num_classes=args.num_ways)
    else:
        print(f"Standard {args.arch} loaded")
        model = standard_dict[args.arch](pretrained=args.pretrained, num_classes=num_classes)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.train_iter, eta_min=1e-9)

    # ============ Prepare metrics ================
    metrics: Dict[str, torch.tensor] = {"train_loss": torch.zeros(int(args.train_iter / args.train_freq)).type(torch.float32),
                                        "train_acc": torch.zeros(int(args.train_iter / args.train_freq)).type(torch.float32),
                                        "val_acc": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        "val_loss": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        "test_acc": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        "test_loss": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        }
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    best_val_acc = 0.

    # ============ Training loop ============
    model.train()
    tqdm_bar = tqdm(train_loader, total=args.train_iter, ascii=True)
    i = 0
    for data in tqdm_bar:

        if i >= args.train_iter:
            break

        # ============ Make a training iteration ============
        t0 = time.time()
        if args.episodic_training:
            support, query, support_labels, target = data
            support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
            query, target = query.to(device), target.to(device, non_blocking=True)

            loss, preds_q = method(x_s=support,
                                   x_q=query,
                                   y_s=support_labels,
                                   y_q=target,
                                   model=model)  # [batch, q_shot]
        else:
            (input_, target) = data
            input_, target = input_.to(device), target.to(device, non_blocking=True).long()
            loss = loss_fn(input_, target, model)

            model.eval()
            with torch.no_grad():
                preds_q = model(input_).softmax(-1).argmax(-1)
            model.train()

        # Perform optim
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Log metrics
        train_loss.update(loss.mean().detach(), i == 0)
        train_acc.update((preds_q == target).float().mean(), i == 0)
        batch_time.update(time.time() - t0, i == 0)

        if i % args.train_freq == 0:
            tqdm_bar.set_description(
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                        'Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                                batch_time=batch_time,
                                loss=train_loss,
                                acc=train_acc))
            for k in metrics:
                if 'train' in k:
                    metrics[k][int(i / args.train_freq)] = eval(k).avg

        # ============ Evaluation ============
        if i % args.val_freq == 0:
            evaluate(i, val_loader, model, method, model_dir, metrics, best_val_acc, device)

        # ============ Testing ============
        if i % args.val_freq == 0:
            test(i, test_loader, model, method, model_dir, metrics, best_val_acc, device)

            for k, e in metrics.items():
                metrics_path = os.path.join(model_dir, f"{k}.npy")
                np.save(metrics_path, e.detach().cpu().numpy())

        i += 1


def test(i, loader, model, method, model_dir, metrics, best_val_acc, device):
    print('Starting testing ...')
    model.eval()
    method.eval()
    tqdm_test_bar = tqdm(loader, total=args.val_iter, ascii=True)
    test_acc = 0.
    test_loss = 0.
    for j, data in enumerate(tqdm_test_bar):
        support, query, support_labels, query_labels = data
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
        loss, preds_q = method(x_s=support,
                                    x_q=query,
                                    y_s=support_labels,
                                    y_q=query_labels,
                                    model=model)
        if args.visu and j == 0:
            task_id = 0
            root = os.path.join(model_dir, 'visu', 'test')
            os.makedirs(root, exist_ok=True)
            save_path = os.path.join(root, f'{i}.png')
            make_episode_visualization(
                       args,
                       support[task_id].cpu().numpy(),
                       query[task_id].cpu().numpy(),
                       support_labels[task_id].cpu().numpy(),
                       query_labels[task_id].cpu().numpy(),
                       preds_q[task_id].cpu().numpy(),
                       save_path)
        test_acc += (preds_q == query_labels).float().mean()
        tqdm_test_bar.set_description(
            f'Test Prec@1 {test_acc/(j+1):.3f})')
        if loss is not None:
            test_loss += loss.detach().mean()
        if j >= args.val_iter:
            break
    test_acc /= args.val_iter
    test_loss /= args.val_iter

    print(f'Iteration: [{i}/{args.train_iter}] \t Test Prec@1 {test_acc:.3f} ')

    for k in metrics:
        if 'test' in k:
            metrics[k][int(i / args.val_freq)] = eval(k)

    model.train()
    method.train()


def evaluate(i, loader, model, method, model_dir, metrics, best_val_acc, device):
    print('Starting validation ...')
    model.eval()
    method.eval()

    tqdm_eval_bar = tqdm(loader, total=args.val_iter, ascii=True)
    val_acc = 0.
    val_loss = 0.
    for j, data in enumerate(tqdm_eval_bar):
        support, query, support_labels, query_labels = data
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
        loss, preds_q = method(x_s=support,
                                    x_q=query,
                                    y_s=support_labels,
                                    y_q=query_labels,
                                    model=model)

        if args.visu and j == 0:
            task_id = 0
            root = os.path.join(model_dir, 'visu', 'valid')
            os.makedirs(root, exist_ok=True)
            save_path = os.path.join(root, f'{i}.png')
            make_episode_visualization(
                       args,
                       support[task_id].cpu().numpy(),
                       query[task_id].cpu().numpy(),
                       support_labels[task_id].cpu().numpy(),
                       query_labels[task_id].cpu().numpy(),
                       preds_q[task_id].cpu().numpy(),
                       save_path)
        val_acc += (preds_q == query_labels).float().mean()
        tqdm_eval_bar.set_description(
            f'Val Prec@1 {val_acc/(j+1):.3f})')
        if loss is not None:
            val_loss += loss.detach().mean()
        if j >= args.val_iter:
            break
    val_acc /= args.val_iter
    val_loss /= args.val_iter

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(state={'iter': i,
                               'arch': args.arch,
                               'state_dict': model.state_dict(),
                               'best_acc': best_val_acc},
                        folder=model_dir)
    print(f'Iteration: [{i}/{args.train_iter}] \t Val Prec@1 {val_acc:.3f} ({best_val_acc:.3f})\t')

    for k in metrics:
        if 'val' in k:
            metrics[k][int(i / args.val_freq)] = eval(k)

    for k, e in metrics.items():
        metrics_path = os.path.join(model_dir, f"{k}.npy")
        np.save(metrics_path, e.detach().cpu().numpy())

    model.train()
    method.train()
    return best_val_acc


if __name__ == '__main__':
    args = parse_args()
    main(args)
