import yaml
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
from box import Box


SEQUENCING_CONFIG_PATH = Path(__file__).parent / "sequencing_config.yaml"
MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.yaml"
DATASET_CONFIG_PATH = Path(__file__).parent / "dataset_config.yaml"



def load_config(path):
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        return Box(yaml.safe_load(f), default_box=True, frozen_box=False)

class Config:
    @property
    @lru_cache()
    def sequencing_config(self):
        return load_config(SEQUENCING_CONFIG_PATH)

    @property
    @lru_cache()
    def model_config(self):
        return load_config(MODEL_CONFIG_PATH)

    @property
    @lru_cache()
    def dataset_config(self):
        return load_config(DATASET_CONFIG_PATH)

config = Config()
