#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6    # There are 40 CPU cores on Beluga GPU nodes
#SBATCH --mem=20000M
#SBATCH --time=10:00:00
#SBATCH --account=def-ibenayed
#SBATCH --array=0-8

#SBATCH --mail-user=malik.boudiaf.1@etsmtl.net
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

source ~/.bash_profile
module load python/3.8.2
source ~/ENV/bin/activate

# Prepare data
DATA_DIR=$SLURM_TMPDIR/data
mkdir $DATA_DIR
tar xf ~/scratch/histo_fsl/data/converted.tar.gz -C $DATA_DIR
DATA_DIR=$SLURM_TMPDIR/data/converted
METHODS="simpleshot protonet maml"
SEEDS="2021 2022 2023"

METHODS=($METHODS)
SEEDS=($SEEDS)

base_config_path="config/base.yaml"

method_config_path="config/${METHODS[$((SLURM_ARRAY_TASK_ID % 3))]}.yaml"

python3 -m src.train --base_config ${base_config_path} \
                     --method_config ${method_config_path} \
                     --opts data_path ${DATA_DIR} \
                            manual_seed ${SEEDS[$((SLURM_ARRAY_TASK_ID / 3))]} \
                            train_sources "['breakhis']" \
                            val_sources "['nct']" \
                            test_sources "['crc-tp']" \
                            visu False
