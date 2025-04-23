import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()