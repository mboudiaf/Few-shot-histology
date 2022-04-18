method=$1
train=$2
valid=$3
test=$4

seeds="[2021]"
train="['$train']"
valid="['$valid']"
test="['$test']"

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

python3 -m src.train --base_config ${base_config_path} \
                     --method_config ${method_config_path} \
                     --opts train_sources ${train} \
                            seeds ${seeds} \
                            val_sources ${valid} \
                            test_sources ${test} \
