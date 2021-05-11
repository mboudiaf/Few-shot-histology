method=$1
shot=$2
train=$3
test=$4

visu="False"

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

seeds="[2021,2022,2023]"
train="['$train']"
valid="['nct']"
test="['$test']"
DATA_DIR=$SLURM_TMPDIR/data/converted
# dirname="results/train=${train}/valid=${valid}/test=${test}/shot=${shot}/"
# mkdir -p -- "$dirname"
python3 -m src.test --base_config ${base_config_path} \
                    --method_config ${method_config_path} \
                    --opts num_support ${shot} \
                           train_sources ${train} \
                           data_path ${DATA_DIR} \
                           seeds ${seeds} \
                           visu ${visu} \
                           val_sources ${valid} \
                           test_sources ${test} \
                            # | tee ${dirname}/log_${method}.txt