"""
Startup script to run in the beggining of every Jupyter Notebook for the competition
- Import common libraries
- Jupyter Notebook Setup: autoreload, display all, add to sys.path
- Import common functions, classes & constants
- Import competition specific functions / constants
"""

# Commonly Used Libraries
from functools import partial
from termcolor import colored
from tqdm.auto import tqdm
from pathlib import Path
from time import time
import pandas as pd
import numpy as np
import random
import pickle
import sys
import os
import re

# Uncommonly Used Libraries
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, asdict
from distutils.dir_util import copy_tree
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython import get_ipython
from PIL import Image
import subprocess
import warnings
import shutil
import math
import glob
import yaml
import json
import cv2
import gc
import wandb

from IPython.core.magic import register_line_cell_magic
from IPython.display import display, Markdown
from IPython.display import clear_output
from IPython import get_ipython

# Package Imports
from .core import (
    ENV, HARDWARE, IS_ONLINE, KAGGLE_INPUT_DIR, WORKING_DIR, TMP_DIR,
    red, blue, green, yellow,
)

# Install omegaconf if not already available
try:
    from omegaconf import OmegaConf
except:
    print('Installing omeaconf')
    os.system('pip install -q omegaconf')
    from omegaconf import OmegaConf


def _setup_jupyter_notebook():
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = 'all'
    ipython = get_ipython()
    try:
        ipython.magic('matplotlib inline')
        ipython.magic('load_ext autoreload')
        ipython.magic('autoreload 2')
    except:
        print('could not load ipython magic extensions')
_setup_jupyter_notebook()


def _ignore_deprecation_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
_ignore_deprecation_warnings()


# Startup Notebook Functions
REPO_NAME = 'toxic'
REPO_PATH = f'https://github.com/sarthak-314/{REPO_NAME}'
def sync():
    'Sync Notebook with VS Code'
    os.chdir(WORKING_DIR/REPO_NAME)
    subprocess.run(['git', 'pull'])
    sys.path.append(str(WORKING_DIR/REPO_NAME))
    os.chdir(WORKING_DIR)


# Mount Drive in Colab
def _colab_mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')
if ENV == 'Colab':
    _colab_mount_drive()

# Hyperparameters Magic Command
@register_line_cell_magic
def hyperparameters(_, cell):
    'Magic command to write hyperparameters into a yaml file and load it with omegaconf'
    # Save hyperparameters in experiment.yaml
    with open('experiment.yaml', 'w') as f:
        f.write(cell)

    # Load the YAML file into the variable HP
    HP = OmegaConf.load('experiment.yaml')
    get_ipython().user_ns['HP'] = HP

# Competition Specific Constants & Functions
COMP_NAME = 'jigsaw-toxic-severity-rating'
DRIVE_DIR = Path('/content/drive/MyDrive/Chai')
DF_DIR = {
    'Kaggle': KAGGLE_INPUT_DIR/'toxic-dataframes',
    'Colab': DRIVE_DIR/'Dataframes',
    'Surface Pro': WORKING_DIR/'data',
}[ENV]
