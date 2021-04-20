import argparse


class DataConfig(object):
    """Common configuration options for creating data processing pipelines."""

    def __init__(
            self,
            args: argparse.Namespace
    ):
        """Initialize a DataConfig.
        """

        # General info
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.shuffle = args.shuffle

        # Transforms and augmentations
        self.image_size = args.image_size
        self.test_transforms = args.test_transforms
        self.train_transforms = args.train_transforms


class EpisodeDescriptionConfig(object):
    """Configuration options for episode characteristics."""

    def __init__(self, args: argparse.Namespace):

        arg_groups = {
                'num_ways': (args.num_ways, ('min_ways', 'max_ways_upper_bound'), (args.min_ways, args.max_ways_upper_bound)),
                'num_query': (args.num_query, ('max_num_query',), (args.max_num_query,)),
                'num_support':
                        (args.num_support,  # noqa: E131
                        ('max_support_set_size', 'max_support_size_contrib_per_class',  # noqa: E128
                         'min_log_weight', 'max_log_weight'),
                        (args.max_support_set_size, args.max_support_size_contrib_per_class,  # noqa: E128
                         args.min_log_weight, args.max_log_weight)),
        }

        for first_arg_name, values in arg_groups.items():
            first_arg, required_arg_names, required_args = values
            if ((first_arg is None) and any(arg is None for arg in required_args)):
                # Get name of the nones
                none_arg_names = [
                        name for var, name in zip(required_args, required_arg_names)
                        if var is None
                ]
                raise RuntimeError(
                        'The following arguments: %s can not be None, since %s is None. '
                        'Please ensure the following arguments of EpisodeDescriptionConfig are set: '
                        '%s' % (none_arg_names, first_arg_name, none_arg_names))

        self.num_ways = args.num_ways
        self.num_support = args.num_support
        self.num_query = args.num_query
        self.min_ways = args.min_ways
        self.max_ways_upper_bound = args.max_ways_upper_bound
        self.max_num_query = args.max_num_query
        self.max_support_set_size = args.max_support_set_size
        self.max_support_size_contrib_per_class = args.max_support_size_contrib_per_class
        self.min_log_weight = args.min_log_weight
        self.max_log_weight = args.max_log_weight
        self.min_examples_in_class = args.min_examples_in_class
        self.ignore_bilevel_ontology = args.ignore_bilevel_ontology

    def max_ways(self):
        """Returns the way (maximum way if variable) of the episode."""
        return self.num_ways or self.max_ways_upper_bound
