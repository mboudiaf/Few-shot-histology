from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import argparse
from collections import defaultdict
from .utils import compute_confidence_interval
plt.style.use('ggplot')

colors = ["g", "b", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
               'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato']

styles = ['--', '-.', ':', '-']


def infinite_defaultdict():
    return defaultdict(infinite_defaultdict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--folder', type=str,
                        help='Folder to search')
    parser.add_argument('--fontsize', type=int, default=12)
    parser.add_argument('--fontfamily', type=str, default='sans-serif')
    parser.add_argument('--fontweight', type=str, default='normal')
    parser.add_argument('--figsize', type=list, default=[10, 10])
    parser.add_argument('--dpi', type=list, default=200,
                        help='Dots per inch when saving the fig')
    parser.add_argument('--max_col', type=int, default=2,
                        help='Maximum number of columns for legend')

    args = parser.parse_args()
    return args


abrv2name = {'val_loss': 'Validation loss',
             'val_acc': 'Validation accuracy',
             'test_loss': 'Test loss',
             'test_acc': 'Test accuracy',
             'train_loss': 'Training loss',
             'train_acc': 'Training accuracy'}


def main(args: argparse.Namespace) -> None:
    plt.rc('font',
           size=args.fontsize,
           family=args.fontfamily,
           weight=args.fontweight)

    # Recover all files that match .npy pattern in folder/
    p = Path(args.folder)
    all_files = p.glob('**/*.npy')

    # Group files by metric name
    filenames_dic = infinite_defaultdict()
    for path in all_files:
        parts = path.parts
        metric = '_'.join(path.stem.split('_')[:2])
        method = [part.split('=')[1] for part in parts if 'method' in part][0]
        seed = [part.split('=')[1] for part in parts if 'seed' in part][0]
        arch = [part.split('=')[1] for part in parts if 'arch' in part][0]
        filenames_dic[metric][method][arch][seed] = path

    # Do one plot per metric
    for metric in filenames_dic:
        fig = plt.Figure(args.figsize)
        ax = fig.gca()
        for style, color, method in zip(cycle(styles), cycle(colors), filenames_dic[metric]):
            for arch in filenames_dic[metric][method]:
                all_y = np.concatenate(
                            [np.expand_dims(np.load(path), 0) \
                             for path in filenames_dic[metric][method][arch].values()], 0)  # [num_seeds, num_epochs]
                mean, conf_interv = compute_confidence_interval(all_y, axis=0)

                valid = mean > 0
                mean = mean[valid]
                conf_interv = conf_interv[valid]

                n_epochs = mean.shape[0]
                x = np.linspace(0, n_epochs - 1, (n_epochs))

                label = f'{method} ({arch})'
                ax.set_title(abrv2name[metric], size=32, weight='bold', y=1.1)
                ax.plot(x, mean, label=label, color=color, linestyle=style)
                ax.fill_between(x, mean-conf_interv, mean+conf_interv, color=color, alpha=0.3)

        n_cols = min(args.max_col, len(filenames_dic[metric]))
        ax.legend(bbox_to_anchor=(0.5, 1.05), loc='center', ncol=n_cols, shadow=True)
        ax.set_xlabel("Epochs")
        ax.grid(True)
        fig.tight_layout()
        save_path = p / f'{metric}.png'
        fig.savefig(save_path, dpi=args.dpi)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
