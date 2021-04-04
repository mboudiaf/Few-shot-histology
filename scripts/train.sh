method=$1
train=$2
test=$3

seeds="[2021]"
train="['$train']"
valid="['nct']"
test="['$test']"

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

python3 -m src.train --base_config ${base_config_path} \
                     --method_config ${method_config_path} \
                     --opts train_sources ${train} \
                            seeds ${seeds} \
                            val_sources ${valid} \
                            test_sources ${test} \
