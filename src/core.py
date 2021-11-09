"""
Commonly used functions and constants used in both Jupyter Notebooks and other Python Scripts
- ENV, HARDWARE, IS_ONLINE
- termcolor color wrappers
"""
from IPython.display import display, Markdown
from termcolor import colored
from pathlib import Path
import torch
import os

# Solve Environment, Hardware & Internet Status
def _solve_env():
    if 'KAGGLE_CONTAINER_NAME' in os.environ:
        return 'Kaggle'
    elif Path('/content/').exists():
        return 'Colab'
    elif Path('C:\\Users\\sarth\\Desktop').exists():
        return 'Surface Pro'
    else:
        return 'Gaming Laptop'

def _solve_hardware():
    if torch.cuda.is_available():
        print('GPU Device:', colored(torch.cuda.get_device_name(0), 'green'))
        return 'GPU'
    elif 'TPU_NAME' in os.environ:
        return 'TPU'
    else:
        return 'CPU'

def _solve_internet_status():
    # TODO: Implement a better way
    try:
        os.system('pip install -q wandb')
        return True
    except:
        return False

def _solve_working_dir(env):
    if env == 'Colab':
        return Path('/content')
    elif env == 'Kaggle':
        return Path('/kaggle/working')
    elif env == 'Surface Pro':
        return Path('C:\\Users\\sarth\\Desktop\\chai-v2')
    elif env == 'Gaming Laptop':
        return Path('C:\\Users\\sarthak\\Desktop\\chai-v2')

def _solve_tmp_dir(working_dir):
    'Make a temp dir to store downloaded data, models, cache etc'
    tmp_dir = working_dir/'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


# Hardware and Environment Config
ENV = _solve_env()
HARDWARE = _solve_hardware()
IS_ONLINE = _solve_internet_status()
print('Notebook running on', colored(ENV, 'blue'), 'on', colored(HARDWARE, 'blue'))

# Useful Paths for each Environment
KAGGLE_INPUT_DIR = Path('/kaggle/input')
WORKING_DIR = _solve_working_dir(ENV)
TMP_DIR = _solve_tmp_dir(WORKING_DIR)

# Termcolor Colors
red = lambda str: colored(str, 'red')
blue = lambda str: colored(str, 'blue')
green = lambda str: colored(str, 'green')
yellow = lambda str: colored(str, 'yellow')

# Jupyter Pretty Display
def heading(title, level=3):
    margin = 'margin-left:5px;'
    try:
        text = title.title()
        html = f"<h{level} style='text-align:center; {margin}'> {text} </h{level}> <hr/>"
        print()
        display(Markdown(html))
    except Exception as e:
        print(e)