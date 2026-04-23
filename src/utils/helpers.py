import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
