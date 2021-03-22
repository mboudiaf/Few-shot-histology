method=$1

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

python3 -m src.train --base_config ${base_config_path} \
                     --method_config ${method_config_path}