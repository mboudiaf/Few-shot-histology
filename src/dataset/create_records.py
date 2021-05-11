from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import os
import collections
import json
import io
from PIL import Image
from PIL import ImageOps
import traceback
from collections import defaultdict
from pathlib import Path
from typing import List
import argparse
from .utils import Split
from . import dataset_spec as ds_spec

DEFAULT_FILE_PATTERN = '{}.tfrecords'
TRAIN_TEST_FILE_PATTERN = '{}_{}.tfrecords'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Records conversion')
    parser.add_argument('--splits_root', type=str, help='Path to pre-defined splits')
    parser.add_argument('--records_root', type=str, required=True, help='Where to store records files')
    parser.add_argument('--name', type=str, required=True, help='Which dataset')
    parser.add_argument('--data_root', type=str, required=True, help='Root to data')
    args = parser.parse_args()
    return args


def write_tfrecord_from_directory(class_files: List[Path],
                                  class_label,
                                  output_path,
                                  invert_img=False,
                                  skip_on_error=False,
                                  shuffle_with_seed=None):
    """Create and write a tf.record file for the images corresponding to a class.
    Args:
        class_directory: the home of the images of class class_label.
        class_label: the label of the class that a record is being made for.
        output_path: the location to write the record.
        invert_img: change black pixels to white ones and vice versa. Used for
            Omniglot for example to change the black-background-white-digit images
            into more conventional-looking white-background-black-digit ones.
        files_to_skip: a set containing names of files that should be skipped if
            present in class_directory.
        skip_on_error: whether to skip an image if there is an issue in reading it.
            The default it to crash and report the original exception.
        shuffle_with_seed: An integer, optional. If provided, the images will be
            shuffled using that seed.
    Returns:
        The number of images written into the records file.
    """
    # class_files = []
    # filenames = sorted(os.listdir(class_directory))
    # for filename in filenames:
    #     if filename in files_to_skip:
    #         logging.info('skipping file %s', filename)
    #         continue
    #     filepath = os.path.join(class_directory, filename)
    #     if tf.io.gfile.isdir(filepath):
    #         continue
    #     class_files.append(filepath)

    if shuffle_with_seed is not None:
        rng = np.random.RandomState(shuffle_with_seed)
        rng.shuffle(class_files)

    written_images_count = write_tfrecord_from_image_files(
            class_files,
            class_label,
            output_path,
            invert_img,
            skip_on_error=skip_on_error)

    if not skip_on_error:
        assert len(class_files) == written_images_count
    return written_images_count


def write_tfrecord_from_image_files(class_files,
                                    class_label,
                                    output_path,
                                    invert_img=False,
                                    bboxes=None,
                                    output_format='JPEG',
                                    skip_on_error=False):
    """Create and write a tf.record file for the images corresponding to a class.
    Args:
        class_files: the list of paths to images of class class_label.
        class_label: the label of the class that a record is being made for.
        output_path: the location to write the record.
        invert_img: change black pixels to white ones and vice versa. Used for
            Omniglot for example to change the black-background-white-digit images
            into more conventional-looking white-background-black-digit ones.
        bboxes: list of bounding boxes, one for each filename passed as input. If
            provided, images are cropped to those bounding box values.
        output_format: a string representing a PIL.Image encoding type: how the
            image data is encoded inside the tf.record. This needs to be consistent
            with the record_decoder of the DataProvider that will read the file.
        skip_on_error: whether to skip an image if there is an issue in reading it.
            The default it to crash and report the original exception.
    Returns:
        The number of images written into the records file.
    """

    def load_and_process_image(path, bbox=None):
        """Process the image living at path if necessary.
        If the image does not need any processing (inverting, converting to RGB
        for instance), and is in the desired output_format, then the original
        byte representation is returned.
        If that is not the case, the resulting image is encoded to output_format.
        Args:
            path: the path to an image file (e.g. a .png file).
            bbox: bounding box to crop the image to.
        Returns:
            A bytes representation of the encoded image.
        """
        with open(path, 'rb') as f:
            image_bytes = f.read()
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except:
            logging.warn('Failed to open image: %s', path)
            raise

        img_needs_encoding = False

        if img.format != output_format:
            img_needs_encoding = True
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img_needs_encoding = True
        if bbox is not None:
            img = img.crop(bbox)
            img_needs_encoding = True
        if invert_img:
            img = ImageOps.invert(img)
            img_needs_encoding = True

        if img_needs_encoding:
            # Convert the image into output_format
            buf = io.BytesIO()
            img.save(buf, format=output_format)
            buf.seek(0)
            image_bytes = buf.getvalue()
        return image_bytes

    writer = tf.python_io.TFRecordWriter(output_path)
    written_images_count = 0
    for i, path in enumerate(class_files):
        bbox = bboxes[i] if bboxes is not None else None
        try:
            img = load_and_process_image(path, bbox)
        except (IOError, tf.errors.PermissionDeniedError) as e:
            if skip_on_error:
                logging.warn('While trying to load file %s, got error: %s', path, e)
            else:
                raise
        else:
            # This gets executed only if no Exception was raised
            write_example(img, class_label, writer)
            written_images_count += 1

    writer.close()
    return written_images_count


def write_example(data_bytes,
                  class_label,
                  writer,
                  input_key='image',
                  label_key='label'):
    """Create and write an Example protocol buffer for the given image.
    Create a protocol buffer with an integer feature for the class label, and a
    bytes feature for the image.
    Args:
        data_bytes: bytes, an encoded image representation or serialized feature.
            For images, the usual encoding is JPEG, but could be different
            as long as the DataProvider's record_decoder accepts it.
        class_label: the integer class label of the image.
        writer: a TFRecordWriter
        input_key: String used as key for the input (image of feature).
        label_key: String used as key for the label.
    """
    example = make_example([(input_key, 'bytes', [data_bytes]), (label_key, 'int64', [class_label])])
    writer.write(example)


def make_example(features):
    """Creates an Example protocol buffer.
    Create a protocol buffer with an integer feature for the class label, and a
    bytes feature for the input (image or feature)
    Args:
        features: sequence of (key, feature_type, value) tuples. Features to encode
            in the Example. `key` corresponds to the feature name, `feature_type` can
            either be 'int64', 'float32', or 'bytes', and `value` corresponds to the
            feature itself.
    Returns:
        example_serial: A string corresponding to the serialized example.
    """

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float32_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    feature_fns = {
            'int64': _int64_feature,
            'float32': _float32_feature,
            'bytes': _bytes_feature
    }

    feature_dict = dict((key, feature_fns[feature_type](value)) for key, feature_type, value in features)

    # Create an example protocol buffer.
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    example_serial = example.SerializeToString()
    return example_serial


def gen_rand_split_inds(num_train_classes, num_valid_classes, num_test_classes):
    """Generates a random set of indices corresponding to dataset splits.
    It assumes the indices go from [0, num_classes), where the num_classes =
    num_train_classes + num_val_classes + num_test_classes. The returned indices
    are non-overlapping and cover the entire range.
    Note that in the current implementation, valid_inds and test_inds are sorted,
    but train_inds is in random order.
    Args:
        num_train_classes : int, number of (meta)-training classes.
        num_valid_classes : int, number of (meta)-valid classes.
        num_test_classes : int, number of (meta)-test classes.
    Returns:
        train_inds : np array of training inds.
        valid_inds : np array of valid inds.
        test_inds  : np array of test inds.
    """
    num_trainval_classes = num_train_classes + num_valid_classes
    num_classes = num_trainval_classes + num_test_classes

    # First split into trainval and test splits.
    trainval_inds = np.random.choice(
            num_classes, num_trainval_classes, replace=False)
    test_inds = np.setdiff1d(np.arange(num_classes), trainval_inds)
    # Now further split trainval into train and val.
    train_inds = np.random.choice(trainval_inds, num_train_classes, replace=False)
    valid_inds = np.setdiff1d(trainval_inds, train_inds)

    logging.info(
            'Created splits with %d train, %d validation and %d test classes.',
            len(train_inds), len(valid_inds), len(test_inds))
    return train_inds.tolist(), valid_inds.tolist(), test_inds.tolist()


class DatasetConverter(object):
    """Converts a dataset to the format required to integrate it in the benchmark.
    In particular, this involves:
    1) Creating a tf.record file for each class of the dataset.
    2) Creating an instance of DatasetSpecification or BiLevelDatasetSpecification
        (as appropriate) for the dataset. This includes information about the
        splits, classes, super-classes if applicable, etc that is required for
        creating episodes from the dataset.
    1) and 2) are accomplished by calling the convert_dataset() method.
    This will create and write the dataset specification and records in
    self.records_path.
    """

    def __init__(self,
                 name,
                 data_root,
                 records_root,
                 has_superclasses=False,
                 split_file=None,
                 records_path=None,
                 random_seed=22,
                 **kwargs):
        """Initialize a DatasetConverter.
        Args:
            name: the name of the dataset
            data_root: the root of the dataset
            has_superclasses: Whether the dataset's classes are organized in a two
                level hierarchy of coarse and fine classes. In that case, a
                BiLevelDatasetSpecification will be created.
            records_path: optional path to store the created records. If it's not
                provided, the default path for the dataset will be used.
            split_file: optional path to a file storing the training, validation and
                testing splits of the dataset's classes. If provided, it's a JSON file
                that stores a dictionary whose keys are 'train', 'valid', and 'test' and
                whose values indicate the corresponding classes assigned to these
                splits. Note that not all datasets require a split file. For example it
                may be the case that a dataset indicates the intended assignment of
                classes to splits via their structure (e.g. all train classes live in a
                'train' folder etc).
            random_seed: a random seed used for creating splits (when applicable) in a
                reproducible way.
        """
        self.name = name
        self.data_root = os.path.join(data_root, name)
        self.has_superclasses = has_superclasses
        self.seed = random_seed
        if records_path is None:
            records_path = os.path.join(records_root, name)
        os.makedirs(records_path, exist_ok=True)
        self.records_path = records_path

        # Where to write the DatasetSpecification instance.
        self.dataset_spec_path = os.path.join(self.records_path, 'dataset_spec.json')

        self.split_file = split_file

        # Sets self.dataset_spec to an initial DatasetSpecification or
        # BiLevelDatasetSpecification.
        self._init_specification()

    def _init_data_specification(self):
        """Sets self.dataset_spec to an initial DatasetSpecification."""
        # Maps each Split to the number of classes assigned to it.
        self.classes_per_split = {
                Split.TRAIN: 0,
                Split.VALID: 0,
                Split.TEST: 0
        }

        self._create_data_spec()

    def _init_bilevel_data_specification(self):
        """Sets self.dataset_spec to an initial BiLevelDatasetSpecification."""
        # Maps each Split to the number of superclasses assigned to it.
        self.superclasses_per_split = {
                Split.TRAIN: 6,
                Split.VALID: 0,
                Split.TEST: 0
        }

        # Maps each superclass id to the number of classes it contains.
        self.classes_per_superclass = collections.defaultdict(int)

        # Maps each superclass id to the name of its class.
        self.superclass_names = {}

        self._create_data_spec()

    def _init_specification(self):
        """Returns an initial DatasetSpecification or BiLevelDatasetSpecification.
        Creates this instance using initial values that need to be overwritten in
        every sub-class implementing the converter for a different dataset. In
        particular, in the case of a DatasetSpecification, each sub-class must
        overwrite the 3 following fields accordingly: classes_per_split,
        images_per_class, and class_names. In the case of its bi-level counterpart,
        each sub-class must overwrite: superclasses_per_split,
        classes_per_superclass, images_per_class, superclass_names, and class_names.
        In both cases, this happens in create_dataset_specification_and_records().
        Note that if other, non-mutable fields are updated, or if these objects are
        replaced with other ones, see self._create_data_spec() to create a new spec.
        """
        # First initialize the fields that are common to both types of data specs.
        # Maps each class id to its number of images.
        self.images_per_class = collections.defaultdict(int)

        # Maps each class id to the name of its class.
        self.class_names = {}

        # Pattern that each class' filenames should adhere to.
        self.file_pattern = DEFAULT_FILE_PATTERN

        if self.has_superclasses:
            self._init_bilevel_data_specification()
        else:
            self._init_data_specification()

    def _create_data_spec(self):
        """Create a new [BiLevel]DatasetSpecification given the fields in self.
        Set self.dataset_spec to that new object. After the initial creation,
        this is needed in the case of datasets with example-level splits, since
        file_pattern and images_per_class have to be replaced by new objects.
        """
        if self.has_superclasses:
            self.dataset_spec = ds_spec.BiLevelDatasetSpecification(
                    self.name, self.superclasses_per_split, self.classes_per_superclass,
                    self.images_per_class, self.superclass_names, self.class_names,
                    self.records_path, self.file_pattern)
        else:
            self.dataset_spec = ds_spec.DatasetSpecification(
                    self.name, self.classes_per_split, self.images_per_class,
                    self.class_names, self.records_path, self.file_pattern)

    def convert_dataset(self):
        """Converts dataset as required to integrate it in the benchmark.
        Wrapper for self.create_dataset_specification_and_records() which does most
        of the work. This method additionally handles writing the finalized
        DatasetSpecification to the designated location.
        """
        self.create_dataset_specification_and_records()

        # Write the DatasetSpecification to the designated location.
        self.write_data_spec()

    def create_dataset_specification_and_records(self):
        """Creates a DatasetSpecification and records for the dataset.
        Specifically, the work that needs to be done here is twofold:
        Firstly, the initial values of the following attributes need to be updated:
        1) self.classes_per_split: a dict mapping each split to the number of
            classes assigned to it
        2) self.images_per_class: a dict mapping each class to its number of images
        3) self.class_names: a dict mapping each class (e.g. 0) to its (string) name
            if available.
        This automatically results to updating self.dataset_spec as required.
        Important note: Must assign class ids in a certain order:
        lowest ones for training classes, then for validation classes and highest
        ones for testing classes.
        The reader data sources operate under this assumption.
        Secondly, a tf.record needs to be created and written for each class. There
        are some general functions at the top of this file that may be useful for
        this (e.g. write_tfrecord_from_npy_single_channel,
        write_tfrecord_from_image_files).
        """
        raise NotImplementedError('Must be implemented in each sub-class.')

    def read_splits(self):
        """Reads the splits for the dataset from self.split_file.
        This will not always be used (as we noted earlier there are datasets that
        define the splits in other ways, e.g. via structure of their directories).
        Returns:
            A splits dictionary mapping each split to a list of class names belonging
            to it, or False upon failure (e.g. the splits do not exist).
        """
        logging.info('Attempting to read splits from %s...', self.split_file)
        if self.split_file and os.exists(self.split_file):
            with open(self.split_file, 'r') as f:
                try:
                    splits = json.load(f)
                except json.decoder.JSONDecodeError:
                    logging.info('Unsuccessful: file exists, but loading failed. %s', traceback.format_exc())
                    return False
                logging.info('Successful.')
                return splits
        else:
            logging.info('Unsuccessful.')
            return False

    def write_data_spec(self):
        """Write the dataset's specification to a JSON file."""
        with open(self.dataset_spec_path, 'w') as f:
            # Use 2-space indentation (which also add newlines) for legibility.
            json.dump(self.dataset_spec.to_dict(), f, indent=2)

    def get_splits(self, force_create=False):
        """Returns the class splits.
        If the splits already exist in the designated location, they are simply
        read. Otherwise, they are created. For this, first reset the random seed to
        self.seed for reproducibility, then create the splits and finally writes
        them to the designated location.
        The actual split creation takes place in self.create_splits() which each
        sub-class must override.
        Args:
            force_create: bool. if True, the splits will be created even if they
                already exist.
        Returns:
            splits: a dictionary whose keys are 'train', 'valid', and 'test', and
            whose values are lists of the corresponding class names.
        """
        # Check if the splits already exist.
        if not force_create:
            splits = self.read_splits()
            if splits:
                return splits

        # First, re-set numpy's random seed, for reproducibility.
        np.random.seed(self.seed)

        # Create the dataset-specific splits.
        splits = self.create_splits()

        # Finally, write the splits in the designated location.
        logging.info('Saving new splits for dataset %s at %s...', self.name, self.split_file)
        with open(self.split_file, 'w') as f:
            json.dump(splits, f, indent=2)
        logging.info('Done.')

        return splits

    def create_splits(self):
        """Create class splits.
        Specifically, create a dictionary whose keys are 'train', 'valid', and
        'test', and whose values are lists of the corresponding classes.
        """
        raise NotImplementedError('Must be implemented in each sub-class.')


class SimpleConverter(DatasetConverter):

    def create_splits(self):
        """Create splits for Quickdraw and store them in the default path."""
        # Quickdraw is stored in a number of .npy files, one for every class
        # with each .npy file storing an array containing the images of that class.
        data_path = Path(self.data_root)
        class_files = defaultdict(list)
        images_per_class = defaultdict(int)
        images = []
        toberemoved = []
        for ext in ['png', 'tif', 'jpg', 'JPG', 'jpeg']:
            images += data_path.glob(f'**/*.{ext}')
            toberemoved += data_path.glob(f'**/ignore/**/*.{ext}')
        images = set(images)
        toberemoved = set(toberemoved)
        images -= toberemoved

        for image in images:
            class_name = image.parts[-2]
            class_files[class_name].append(image)
            images_per_class[class_name] += 1

        # Sort the class names, for reproducibility.
        class_names = list(images_per_class.keys())
        class_names.sort()
        num_classes = len(class_names)

        # Split into train, validation and test splits that have 70% / 15% / 15%
        # of the data, respectively.

        train_inds = range(num_classes)
        valid_inds = []
        test_inds = []
        # num_trainval_classes = int(0. * num_classes)
        # num_train_classes = int(1.0 * num_classes)
        # num_valid_classes = num_trainval_classes - num_train_classes
        # num_test_classes = num_classes - num_trainval_classes

        # train_inds, valid_inds, test_inds = gen_rand_split_inds(num_train_classes,
        #                                                         num_valid_classes,
        #                                                         num_test_classes)
        splits = {Split.TRAIN: [class_names[i] for i in train_inds],
                  Split.VALID: [class_names[i] for i in valid_inds],
                  Split.TEST: [class_names[i] for i in test_inds]}
        return splits, images_per_class, class_files

    def create_dataset_specification_and_records(self):
        splits, images_per_class, class_files = self.create_splits()
        class_label = 0
        for split in splits:
            self.classes_per_split[split] = len(splits[split])
            for class_name in splits[split]:
                self.images_per_class[class_label] = images_per_class[class_name]
                self.class_names[class_label] = class_name
                class_records_path = os.path.join(
                            self.records_path,
                            self.dataset_spec.file_pattern.format(class_label))
                write_tfrecord_from_directory(class_files[class_name],
                                              class_label,
                                              class_records_path)
                class_label += 1


class BilevelConverter(DatasetConverter):

    def __init__(self, *args, **kwargs):
        """Initialize an OmniglotConverter."""
        # Make has_superclasses default to True for the Omniglot dataset.
        if 'has_superclasses' not in kwargs:
            kwargs['has_superclasses'] = True
        super(BilevelConverter, self).__init__(*args, **kwargs)

    # def get_image_dic(path: pathlib.PurePath):
    #     rec_dd = lambda: defaultdict(rec_dd)
    #     image_dic = defaultdict(rec_dd)
    #     for dataset in path.iterdir():
    #         splits_dir = [dir for dir in dataset.iterdir() if dir.stem != 'ignore' ]
    #         for split in splits_dir:
    #             for class_ in split.iterdir():
    #                 images = []
    #                 for ext in ['png', 'tif', 'jpg', 'JPG', 'jpeg']:
    #                     images += list(class_.glob(f'**/*.{ext}'))
    #                 for image in images:
    #                     image_dic[dataset.stem][split.stem][class_.stem][image]
    #     return image_dic

    def parse_split_data(self, split, superclasses):
        data_path = Path(self.data_root)
        # class_files = defaultdict(list)
        images_per_class = defaultdict(int)
        class_files = defaultdict(list)
        self.superclasses_per_split[split] = len(superclasses)
        # splits_dir = [dir for dir in dataset.iterdir() if dir.stem != 'ignore']

        images = []
        for ext in ['png', 'tif', 'jpg', 'JPG', 'jpeg']:
            images += list(data_path.glob(f'**/*.{ext}'))

        for image in images:
            class_ = image.parts[-4]
            superclass_ = image.parts[-2]
            if superclass_ in superclasses:
                class_name = f'{superclass_}-{class_}'
                class_files[class_name].append(image)
                images_per_class[class_name] += 1

        # For compatibility, we fill out everything
        superclass_offset = len(self.superclass_names)
        for label, superclass_name in enumerate(superclasses):
            label += superclass_offset
            self.superclass_names[label] = superclass_name

        class_offset = len(self.class_names)
        for label, class_name in enumerate(images_per_class):
            label += class_offset
            self.class_names[label] = class_name
            self.images_per_class[label] = images_per_class[class_name]
            superclass_name = class_name.split('-')[0]
            self.classes_per_superclass[superclass_offset + superclasses.index(superclass_name)] += 1
            # Create and write the tf.Record of the examples of this class.
            class_records_path = os.path.join(
                self.records_path,
                self.dataset_spec.file_pattern.format(label))
            write_tfrecord_from_directory(
                class_files[class_name], label, class_records_path)

    def create_dataset_specification_and_records(self):
        """Implements DatasetConverter.create_dataset_specification_and_records().
        We use Lake's original train/test splits as we believe this is a more
        challenging setup and because we like that it's hierarchically structured.
        We also held out a subset of that train split to act as our validation set.
        Specifically, the 5 magnifs from that set with the least number of
        canceracters were chosen for this purpose.
        """

        # We chose the 5 smallest magnifs (i.e. those with the least canceracters)
        # out of the 'background' set of magnifs that are intended for train/val
        # We keep the 'evaluation' set of magnifs for testing exclusively
        # The chosen magnifs have 14, 14, 16, 17, and 20 canceracters, respectively.
        training_magnifs = ['40X', '100X', '200X', '400X']
        validation_magnifs = []
        test_magnifs = []

        self.parse_split_data(Split.TRAIN, training_magnifs)
        self.parse_split_data(Split.VALID, validation_magnifs)
        self.parse_split_data(Split.TEST, test_magnifs)


if __name__ == '__main__':
    args = parse_args()
    if args.name.lower() in ['breakhis']:
        converter_build = BilevelConverter
    else:
        converter_build = SimpleConverter
    converter = converter_build(**vars(args))
    print('Creating %s specification and records in directory %s...',
          args.name, args.records_root)
    converter.convert_dataset()
