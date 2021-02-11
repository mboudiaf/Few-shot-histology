import torch
from src.dataset.utils import Split
import src.dataset.config as config_lib
import src.dataset.dataset_spec as dataset_spec_lib
from torch.utils.data import DataLoader
import os
import src.dataset.pipeline as pipeline
import argparse
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Records conversion')

    # General data info
    parser.add_argument('--image_size', type=int,
                        default=126, help='Images will be resized to this value')
    parser.add_argument('--sources', nargs="+",
                        default=['breakhis'], help='Which dataset to use')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Root to data')
    parser.add_argument('--train_transforms', nargs="+", help='Transforms applied to training data',
                        default=['random_resized_crop', 'random_flip']
                        )
    parser.add_argument('--test_transforms', nargs="+", help='Transforms applied to training data',
                        default=['resize', 'center_crop'])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    # Info on episodes
    parser.add_argument('--num_ways', type=int,
                        default=None, help='Set it if you want a fixed # of ways per task')
    parser.add_argument('--num_support', type=int,
                        default=None, help='Set it if you want a fixed # of support samples per class')
    parser.add_argument('--num_query', type=int,
                        default=None, help='Set it if you want a fixed # of query samples per class')
    parser.add_argument('--min_ways', type=int,
                        default=2, help='Minimum # of ways per task')
    parser.add_argument('--max_ways_upper_bound', type=int,
                        default=10, help='Maximum # of ways per task')
    parser.add_argument('--max_num_query', type=int,
                        default=10, help='Maximum # of query samples')
    parser.add_argument('--max_support_set_size', type=int,
                        default=100, help='Maximum # of support samples')
    parser.add_argument('--min_examples_in_class', type=int,
                        default=0, help='Classes that have less samples will be skipped')
    parser.add_argument('--max_support_size_contrib_per_class', type=int,
                        default=10, help='Maximum # of support samples per class')
    parser.add_argument('--min_log_weight', type=float,
                        default=-0.69314718055994529, help='Do not touch, used to randomly sample support set')
    parser.add_argument('--max_log_weight', type=float,
                        default=0.69314718055994529, help='Do not touch, used to randomly sample support set')
    parser.add_argument('--ignore_bilevel_ontology', type=bool,
                        default=False, help='Whether or not to use superclass for BiLevel datasets (e.g breakhist)')
    args = parser.parse_args()
    return args


def main(args):

    # Define your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recovering configurations
    data_config = config_lib.DataConfig(args)
    episod_config = config_lib.EpisodeDescriptionConfig(args)

    # Get the data specifications
    datasets = data_config.sources
    if episod_config.num_ways:
        if len(datasets) > 1:
            raise ValueError('For fixed episodes, not tested yet on > 1 dataset')
    else:
        use_bilevel_ontology_list = [False]*len(datasets)
        # Enable ontology aware sampling for breakhis    and ImageNet.
        if 'breakhis' in datasets:
            use_bilevel_ontology_list[datasets.index('breakhis')] = True

        use_bilevel_ontology_list = use_bilevel_ontology_list
    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list

    all_dataset_specs = []
    for dataset_name in datasets:
        dataset_records_path = os.path.join(data_config.data_path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    split = Split["TRAIN"]
    # ============ Form an episodic dataset from the training data
    episodic_dataset = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                      split=split,
                                                      data_config=data_config,
                                                      episode_descr_config=episod_config)

    #  If you want to get the total number of classes (i.e from combined datasets)
    num_classes = sum([len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs])
    print(f"=> There are {num_classes} classes in the combined datasets")

    # Use a standard dataloader
    episodic_loader = DataLoader(dataset=episodic_dataset,
                                 batch_size=1,
                                 num_workers=data_config.num_workers)

    # Training or validation loop
    for i, (support, query, support_labels, query_labels) in enumerate(episodic_loader):
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
        # Do some operations
        print("=> Example of episode")
        print("Number of ways: {}   Support size: {}   Query Size: {} \n".format(
                            support_labels.unique().size(0),
                            list(support.size()),
                            list(query.size())))
        break

    # ============ Form a standard (batch) dataset from the training data
    batch_dataset = pipeline.make_batch_pipeline(dataset_spec_list=all_dataset_specs,
                                                 split=split,
                                                 data_config=data_config)

    # Use a standard dataloader
    batch_loader = DataLoader(dataset=batch_dataset,
                              batch_size=data_config.batch_size,
                              num_workers=data_config.num_workers)
    # Training or validation loop
    for i, (input, target) in enumerate(batch_loader):
        example_image = input[0]
        print(input.min(), input.max(), target)
        plt.imshow(example_image.permute(1, 2, 0).numpy())
        plt.show()
        input, target = input.to(device), target.long().to(device, non_blocking=True)
        # Do some operations
        print("=> Example of a batch")
        print(f"Shape of batch: {list(input.size())}")
        break


if __name__ == '__main__':
    args = parse_args()
    main(args)