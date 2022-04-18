# FHIST: A benchmark for Few-shot Classification of Histological Images

This Repo contains the code and few-shot tasks of our paper [FHIST](). FHIST introduces a highly diversified public benchmark, gathered from various public datasets, for few-shot histology data classification. 
We build few-shot tasks in three different scenarios with various tissue types, different levels of domain shifts stemming from various cancer sites, and different class-granularity levels. We evaluate the performances of state-of-the-art few-shot learning methods, initially designed for natural images, on our benchmark. 

## Getting Started

### Installation

This code is tested with python 3.8. To install required packages:

```python
    pip install -r requirements
```

### Data

You need to download the datasets ([crc-tp](https://warwick.ac.uk/fac/cross_fac/tia/data/crc-tp), [nct](https://zenodo.org/record/1214456#.Ylxt0XVKiUk), [lc25000](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af), [breakhis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-BreakHis/)), and put all folders in a single folder. Then you can run the conversion script `scripts/convert_datasets.sh` (change the data_root and record_root path):

```python
    bash scripts/convert_datasets.sh
```

This scripts will put all the datasets in the same `*.tfrecords` format.

#### Make Index files

Run the following to create index files for each tfrecord:(Specify the data_path in the bash script file)

```python
    bash scripts/make_index_files.sh
```

## Usage

### Training

You can find implementations of some SOTA few-shot learning methods under src/methods. In order to train a model, run the following command specifying the few-shot method, train, validation and test sources:

```python
    bash scripts/train.sh <method> <train> <valid> <test>
```

Here's an example:

```python
    bash scripts/train.sh tim crc-tp nct lc25000
```

The above script will use standard supervised cross-entropy for training, and TIM method for evaluation on valid and test sets. Replace TIM with a meta-learning method(e.g. protonet), to start episodic training.

In order to change the backbone network, simply change the "arch" value in the base.yaml file in the config path or add --arch in the opts list. base.yaml file includes all the parameters used in the code. You can also change hyperparameters of each method from their corresponding yaml file in the config path.  

### Inference

To run few-shot inference on the trained models:

```python
    bash scripts/test.sh <method> <shot> <train> <test> 
```

Results will be saved as csv files under the specified res_path for each method.


#### One source vs multi-sources

For both the episodic and batch datasets, you can use multiples source datasets simultaneously.

**Batch dataset**: In the case where multiples sources are provided (through `--sources`), the dataset will yield samples randomly chosen from possible sources. Hence, samples from different sources can co-exist in the same batch.

**Episodic dataset**: In the case where multiples sources are provided (through `--sources`), the dataset will first randomly chose a source, then provide an episode from this source only. Hence, samples from different sources are never mixed in the same task.



## Acknowledgments

I thank the authors of the [open-source TFRecord reader](https://github.com/vahidk/tfrecord) for open sourcing an awesome Pytorch-compatible TFRecordReader !


