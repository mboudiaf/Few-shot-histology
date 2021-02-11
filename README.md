# Histology Meta-dataset (pytorch)


## Overall structure

Here is a rough overview of the different modules of the data pipeline and how they relate with one another:


<img src="figures/overview.png" width="800"/>


The general idea is that each `*.tfrecords` represents a class of a dataset. Therefore, we can create one TFRecordDataset per class. So for each dataset (e.g breakhis), we will have a list of all TFRecordDataset (one per class), which are randomly sampled from.

## Data

#### Downloading data (recommended)

I have put all the converted data at [converted_data](https://drive.google.com/file/d/1W2xxzag9oetbXlR5lcxeRjlmq1ibNFjm/view?usp=sharing).

#### Direct download does not work ? (not tested from scratch)

In this case, you will need to download the data by yourself, and put all folders in a single folder. Then you can run the conversion script `scripts/convert_datasets.sh` (please before doing this, change the data_root and record_root path):

```python
    bash scripts/convert_datasets.sh
```

This scripts will put all the datasets in the same `*.tfrecords` format.


## Usage

I provide an example of how to create your dataloaders in `example.py`. To run it, execute:
```python
python3 example.py with --batch_size 124 --sources 'breakhis' 'bach' --data_path 'your_path_to_data_folder'
```



#### Batch dataset (standard form)

For a standard batch dataset:

```python
    # Training or validation loop
    for i, (support, query, support_labels, query_labels) in enumerate(episodic_loader):
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
        # Do some operations

    # Form a batch dataset
    batch_dataset = pipeline.make_batch_pipeline(dataset_spec_list=all_dataset_specs,
                                                 split=split,
                                                 data_config=data_config)

    # Use a standard dataloader
    batch_loader = DataLoader(dataset=batch_dataset,

                              batch_size=data_config.batch_size,
                              num_workers=data_config.num_workers)
```

#### Episodic dataset

Once the dataset_spec of each dataset has been recovered, we are ready to get the datasets. For an episodic dataset:
```python
    # Form an episodic dataset
    split = Split["TRAIN"]
    episodic_dataset = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                      spl
                                                      it=split,
                                                      data_config=data_config,
                                                      episode_descr_config=episod_config)

    # Use a standard dataloader
    episodic_loader = DataLoader(dataset=episodic_dataset,
                                 batch_size=1,
                                 num_workers=data_config.num_workers)

    # Training or validation loop
    for i, (support, query, support_labels, query_labels) in enumerate(episodic_loader):
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
        # Do some operations
```


#### One source vs multi-sources

For both the episodic and batch datasets, you can use multiples source datasets simultaneously.

**Batch dataset**: In the case where multiples sources are provided (through `--sources`), the dataset will yield samples randomly chosen from possible sources. Hence, samples from different sources can co-exist in the same batch.

**Episodic dataset**: In the case where multiples sources are provided (through `--sources`), the dataset will first randomly chose a source, then provide an episode from this source only. Hence, samples from different sources are never mixed in the same task.

#### Options

There exist several options to build the dataset. If you want the full list, please use:
```python
    python3 example.py --help
```
 IF you want to know all possible transforms, go to `transforms.py`.


## Contributions

Please, start a new branch for every new feature you want to add :) The idea is that this repo will be the one we may use for the challenge, so let us try to keep it as clean as possible.

## Acknowledgments

I thank the authors of the [open-source TFRecord reader](https://github.com/vahidk/tfrecord) for open sourcing an awesome Pytorch-compatible TFRecordReader !


