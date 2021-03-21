import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import SGD, Adam
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import time
from dataset.utils import Split
import dataset.config as config_lib
import dataset.dataset_spec as dataset_spec_lib
import dataset.pipeline as pipeline
from models import __dict__ as all_models
from methods import __dict__ as all_methods
from losses import __dict__ as all_losses
from utils import load_cfg_from_cfg_file, merge_cfg_from_list, AverageMeter, \
                   save_checkpoint, get_model_dir


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

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=args.num_workers)
    num_classes = sum([len(d_spec.get_classes(split=Split["TRAIN"])) for d_spec in all_dataset_specs])
    return loader, num_classes


def main(args):

    # ============ Device ================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = get_model_dir(args)

    # ============ Data spec ================

    # ============ Training method ================
    method = all_methods[args.method](args=args)

    # ============ Data loaders =========
    train_loader, num_classes = get_dataloader(args=args,
                                               sources=args.train_sources,
                                               episodic=method.episodic_training,
                                               batch_size=args.batch_size,
                                               split=Split["TRAIN"])

    val_loader, num_classes_val = get_dataloader(args=args,
                                                 sources=args.val_sources,
                                                 episodic=True,
                                                 batch_size=args.val_batch_size,
                                                 split=Split["TRAIN"])

    #  If you want to get the total number of classes (i.e from combined datasets)
    if not method.episodic_training:
        loss_fn = all_losses[args.loss](args=args, num_classes=num_classes, reduction='none')

    print(f"=> There are {num_classes} classes in the train datasets")
    print(f"=> There are {num_classes_val} classes in the validation datasets")

    # ============ Model and optim ================
    model = all_models[args.arch](num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None

    # ============ Prepare metrics ================
    metrics: Dict[str, torch.tensor] = {"train_loss": torch.zeros(int(args.train_iter / args.train_freq)).type(torch.float32),
                                        "val_acc": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        "val_loss": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        }
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    best_val_acc = 0.

    # ============ Training loop ============
    model.train()
    tqdm_bar = tqdm(train_loader, total=args.train_iter, ascii=True)
    for i, data in enumerate(tqdm_bar):
        t0 = time.time()
        if method.episodic_training:
            support, query, support_labels, query_labels = data
            support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
            query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)

            loss, _ = method(x_s=support,
                             x_q=query,
                             y_s=support_labels,
                             y_q=query_labels,
                             model=model)  # [batch, q_shot]
        else:
            (input_, target) = data
            input_, target = input_.to(device), target.to(device, non_blocking=True).long()
            loss = loss_fn(input_, target, model)

        # Perform optim
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Log metrics
        train_loss.update(loss.mean().detach(), i == 0)
        batch_time.update(time.time() - t0, i == 0)

        if i % args.train_freq == 0:
            tqdm_bar.set_description(
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                                batch_time=batch_time,
                                loss=train_loss))
            for k in metrics:
                if 'train' in k:
                    metrics[k][int(i / args.train_freq)] = eval(k).avg

        # ============ Evaluation ============
        if i % args.val_freq == 0:
            print('Starting evaluation ...')
            model.eval()
            tqdm_eval_bar = tqdm(val_loader, total=args.val_iter)
            with torch.no_grad():
                val_acc = 0.
                val_loss = 0.
                for j, data in enumerate(tqdm_eval_bar):
                    support, query, support_labels, query_labels = data
                    support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
                    query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
                    loss, pred_q = method(x_s=support,
                                          x_q=query,
                                          y_s=support_labels,
                                          y_q=query_labels,
                                          model=model)
                    val_acc += (pred_q == query_labels).float().mean()
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
                path = os.path.join(model_dir, f"{k}.npy")
                np.save(path, e.detach().cpu().numpy())

            model.train()

        if i >= args.train_iter:
            break


if __name__ == '__main__':
    args = parse_args()
    main(args)