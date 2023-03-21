import json
import time
import os
from pathlib import Path

# Get the environmental variables passed on from the container
SV_FILE = '/datain/'
BOTTOM_FILE = '/datain/'
PRED_DIR = '/dataout/'
PRED_FILE = PRED_DIR + os.getenv('ZARRFILE')
CONFIG_DIR = '/configs/'
MODELWEIGHTS = '/model/'+os.getenv('MODEL')
SURVEY = os.getenv('SURVEY')
CONFIG_FILE = CONFIG_DIR + os.getenv('CONFIG')#'configs/config_baseline.yaml'

# The docker mounts the main data folder as /datain/
print('********************')
print(' ')
print('Files:')
print(f'sv file {SV_FILE} exists: {os.path.isdir(SV_FILE)}')
print(f'Bottom file dir {BOTTOM_FILE} exists: {os.path.isdir(BOTTOM_FILE)}')
print(f'Models weights file {MODELWEIGHTS} exists: {os.path.isfile(MODELWEIGHTS)}')
print(f'Prediction dir {PRED_DIR} exists: {os.path.isdir(PRED_DIR)}')
print(f'Config file {CONFIG_FILE} exists: {os.path.isfile(CONFIG_FILE)}')

# The file locations are coded in the setpyenv file
setpyenv = {
    "syspath": "/crimac_unet/",
    "path_to_zarr_files": SV_FILE}

# Write setpyenv file based on environmental variables
with open("/crimac_unet/setpyenv.json", "w") as fp:
    json.dump(setpyenv, fp, indent=4)

# Set the correct paths to the files
from paths import *
from pipeline_train_predict.pipeline import SegPipeUNet
from utils.general import (get_argparse_parser, load_yaml_config,
                           parse_config_options)
from data.data_reader import DataReaderZarr
from pipeline_train_predict.save_predict import save_survey_predictions_zarr

path_to_sv_file = os.path.join(SV_FILE, SURVEY, 'ACOUSTIC', 'GRIDDED', f'{SURVEY}_sv.zarr')

# Check paths
print(' ')
print('Check paths:')
print('Path to sv file:', path_to_sv_file)
print('Path to prediction file', PRED_FILE)

print('\nCUDA is available:', str(torch.cuda.is_available()))

# Configuration options
argparse_args = get_argparse_parser(mode="docker_predict").parse_args()
configuration = load_yaml_config(CONFIG_FILE)
config_args = parse_config_options(configuration, argparse_args)

# Frequencies in the config file is set as kHz -> change to Hz
freqs = config_args['frequencies']
config_args['frequencies'] = [f*1000 for f in freqs]

print('\nKey configuration arguments (inference):')
relevant_keys = ['frequencies', 'num_workers', 'patch_size', 'batch_size',
                 'preload_n_pings', 'resume_writing']
for rel_key in relevant_keys:
    print(f"  {rel_key}: {config_args[rel_key]}")
print()

# Load trained model
experiment_name = Path(CONFIG_FILE).resolve(strict=True).stem
segpipe = SegPipeUNet(**config_args, experiment_name=experiment_name)
segpipe.load_model_params(checkpoint_path=MODELWEIGHTS)

# Load data reader for the survey
reader = DataReaderZarr(path_to_sv_file)

# Save segmentation predictions
print("\nSave predictions")
start = time.time()
save_survey_predictions_zarr(reader, segpipe, target_dname=PRED_FILE, **config_args)

print(f"Executed time for saving all prediction (h): {np.round((time.time() - start) / 3600, 2)}")
