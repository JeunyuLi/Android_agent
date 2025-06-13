import os
import yaml

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(__file__).replace(".py", ".yaml")
    configs = dict()
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    configs.update(yaml_data)
    return configs
