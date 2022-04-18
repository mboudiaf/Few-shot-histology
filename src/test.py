import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from .dataset.utils import Split
from .models.standard import __dict__ as standard_dict
from .models.meta import __dict__ as meta_dict
from .methods import __dict__ as all_methods
from .utils import load_cfg_from_cfg_file, merge_cfg_from_list, \
                   get_model_dir, make_episode_visualization, \
                   load_checkpoint, blockPrint, enablePrint, \
                   compute_confidence_interval
from .train import get_dataloader


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--method_config', type=str, default=True, help='Base config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main(args):

    # ============ Device ================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_path = os.path.join(args.res_path, args.method)
    os.makedirs(res_path, exist_ok=True)

    # ============ Testing method ================
    method = all_methods[args.method](args=args)
    average_acc = []
    average_std = []

    for seed in args.seeds:
        args.seed = seed
        # ============ Data loaders =========
        _, num_classes_tr = get_dataloader(args=args,
                                           sources=args.train_sources,
                                           episodic=args.episodic_training,
                                           batch_size=args.batch_size,
                                           split=Split["TRAIN"])

        test_loader, num_classes = get_dataloader(args=args,
                                                  sources=args.test_sources,
                                                  episodic=True,
                                                  batch_size=args.test_batch_size,
                                                  split=Split["TEST"])
        print(f"=> There are {num_classes} classes in the test datasets")
        # ============ Model ================
        model_dir = get_model_dir(args, seed)
        if 'MAML' in args.method:
            print(f"Meta {args.arch} loaded")
            model = meta_dict[args.arch](pretrained=args.pretrained, num_classes=args.num_ways)
        else:
            print(f"Standard {args.arch} loaded")
            model = standard_dict[args.arch](pretrained=args.pretrained, num_classes=num_classes_tr)
        load_checkpoint(model, model_dir, type='best')
        model = model.to(device)

        # ============ Training loop ============
        model.eval()
        method.eval()
        print(f'Starting testing for seed {seed}')
        test_acc = 0.
        test_loss = 0.
        predictions = []
        tqdm_bar = tqdm(test_loader, total=args.test_iter)
        i = 0
        for data in tqdm_bar:
            support, query, support_labels, query_labels = data
            support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
            query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)

            if args.method == 'MAML':  # MAML uses transductive batchnorm, whic corrupts the model
                blockPrint()
                load_checkpoint(model, model_dir, type='best')
                enablePrint()

            # ============ Evaluation ============
            loss, preds_q = method(x_s=support,
                                        x_q=query,
                                        y_s=support_labels,
                                        y_q=query_labels,
                                        model=model)
            if args.visu and i % 100 == 0:
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
            predictions.append((preds_q == query_labels).float().mean().item())
            test_acc, test_std = compute_confidence_interval(predictions)
            if loss is not None:
                test_loss += loss.detach().mean().item()
            if i % 10 == 0:
                tqdm_bar.set_description(f'Test Prec@1 {test_acc :.3f}  \
                                           Test 95CI@1 {test_std :.4f}  \
                                           Test loss {test_loss / (i+1):.3f}',
                                         )
                update_csv(args=args,
                           task_id=i,
                           acc=test_acc,
                           std=test_std,
                           path=os.path.join(res_path, 'test.csv'))
            if i >= args.test_iter:
                break
            i += 1
        average_acc.append(test_acc)
        average_std.append(test_std)

    print(f'--- Average accuracy over {len(args.seeds)} seeds = {np.mean(average_acc):.3f}')
    print(f'--- Average 95\% Confidence Interval over {len(args.seeds)} seeds = {np.mean(average_std):.4f}')


def update_csv(args: argparse.Namespace,
               task_id: int,
               acc: float,
               std: float,
               path: str):
    # res = OrderedDict()
    try:
        res = pd.read_csv(path)
    except:
        res = pd.DataFrame()
    records = res.to_dict('records')
    # Check whether the entry exist already, if yes, simply update the accuracy
    match = False
    for entry in records:
        match = [str(value) == str(args[param]) for param, value in list(entry.items()) if param not in ['acc', 'task', 'std']]
        match = (sum(match) == len(match))
        if match:
            entry['task'] = task_id
            entry['acc'] = round(acc, 4)
            entry['std'] = round(std, 4)
            break

    # If entry did not exist, just create it
    if not match:
        new_entry = {param: args[param] for param in args.simu_params}
        new_entry['task'] = task_id
        new_entry['acc'] = round(acc, 4)
        new_entry['std'] = round(std, 4)
        records.append(new_entry)
    # Save back to dataframe
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)