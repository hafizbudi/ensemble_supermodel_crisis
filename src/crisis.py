# This modules sets the environment for running experiments
# You can use set_environment to adapt it to your directories.

from pathlib import Path
import os

CRISIS_ROOT_DIR=None
CRISIS_DATA_DIR=None
CRISIS_MODEL_DIR=None

def set_environment(root_dir=None, data_dir=None, model_dir=None):
    global CRISIS_ROOT_DIR, CRISIS_DATA_DIR, CRISIS_MODEL_DIR
    CRISIS_ROOT_DIR= Path(os.path.dirname(os.path.abspath(__file__))).parent
    if data_dir is None:
        CRISIS_DATA_DIR = CRISIS_ROOT_DIR / "data"
    else:
        CRISIS_DATA_DIR = Path(data_dir)
    if model_dir is None:
        CRISIS_MODEL_DIR = CRISIS_ROOT_DIR / "ckpt"
    else:
        CRISIS_MODEL_DIR = Path(model_dir)

set_environment()
