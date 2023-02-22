import json
import os
import sys
import yaml
import numpy as np
import torch
import random
from utils.general import load_yaml_config


try:
    with open(os.path.join(os.path.dirname(__file__), 'setpyenv.json')) as file:
        json_data = file.read()
    setup_file = json.loads(json_data)
    if 'syspath' in setup_file.keys():
        sys.path.append(setup_file["syspath"])

    # set random seeds
    np.random.seed(10)
    random.seed(10)
    torch.manual_seed(10)

except:
    class SetupFileIsMissing(Exception): pass
    raise SetupFileIsMissing('Please make a setpyenv.json file in the root directory.')


def path_to_echograms():
    # Directory path to echogram data
    return setup_file['path_to_echograms']

def path_to_korona_data():
    # Directory path to predictions from the Korona algorithm in LSSS
    return setup_file['path_to_korona_data']

def path_to_korona_transducer_depths():
    # Directory path to the transducer depths in the Korona predictions (necessary for vertical calibration)
    return setup_file['path_to_korona_transducer_depths']

def path_to_trained_model():
    # Directory path to trained models
    return setup_file['path_to_trained_model']

def path_to_baseline_model():
    # Directory path for trained baseline model
    return setup_file['path_to_baseline_model']

def path_to_zarr_files():
    # Directory path to zarr files
    return setup_file['path_to_zarr_files']

def path_for_saving_figs():
    # Directory path for saving figures relating to results evaluation
    return setup_file['path_for_saving_figs']

def path_for_saving_preds_labels():
    # Directory path for saving figures relating to results evaluation
    return setup_file['path_for_saving_preds_labels']


def pipeline_config(yaml_file):
    return load_yaml_config(os.path.join(os.path.dirname(__file__), 'pipeline_config.yaml'))