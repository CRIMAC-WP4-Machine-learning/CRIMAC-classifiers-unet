import json
import time
import os

# Get the environmental variables passed on from the container
SV_FILE = '/datain/'
BOTTOM_FILE = '/datain/'
PRED_DIR = '/dataout/'
PRED_FILE = PRED_DIR + os.getenv('ZARRFILE')
MODELWEIGHTS = '/model/'+os.getenv('MODEL')
SURVEY = os.getenv('SURVEY')
CONFIG_FILE = '/configs/config_baseline.yaml'

# The docker mounts the main data folder as /datain/
print('********************')
print(' ')
print('Files:')
print('sv file:'+SV_FILE+' exists:'+ str(os.path.isdir(SV_FILE)))
print('Bottom file dir:'+BOTTOM_FILE+' exists:'+str(os.path.isdir(BOTTOM_FILE)))
print('Models weights file:'+MODELWEIGHTS+' exists:'+str(os.path.isfile(MODELWEIGHTS)))
print('Prediction dir:'+PRED_DIR+' exists:'+str(os.path.isdir(PRED_DIR)))

# The file locations are coded in the setpyenv file
setpyenv = {
    "syspath": "/crimac_unet/",
    "path_to_echograms":  SV_FILE,
    "path_to_trained_model": MODELWEIGHTS,
    "path_to_zarr_files": SV_FILE}

# Write setpyenv file based on environmental variables
with open("/crimac_unet/setpyenv.json", "w") as fp:
    json.dump(setpyenv, fp, indent=4)

# Set the correct paths to the files
from paths import *
from pipeline_train_predict.pipeline import SegPipeUNet
from utils.general import (get_argparse_parser, load_yaml_config,
                           parse_config_options)
from data.partition import DataZarr
from pipeline_train_predict.save_predict import save_survey_predictions_zarr

# Check paths
print(' ')
print('Check paths:')
print('path_to_zarr_files: '+path_to_zarr_files()+' Content:')
print(os.listdir(path_to_zarr_files()))

# Configuration options
argparse_args = get_argparse_parser(mode="save_predict").parse_args()
configuration = load_yaml_config(argparse_args.yaml_path)
config_args = parse_config_options(configuration, argparse_args)

config_args['save_prediction_surveys'] = [SURVEY]
print(' ')
print('configuration:')
print(configuration)

# Set up data loader and load trained model
data_partition_object = DataZarr(**config_args)
experiment_name = config_args["yaml_path"].stem
segpipe = SegPipeUNet(**config_args, experiment_name=experiment_name)
segpipe.load_model_params(checkpoint_path=MODELWEIGHTS)

# Save segmentation predictions
print("Save predictions")
start = time.time()
save_prediction_surveys = config_args["save_prediction_surveys"]
for survey in save_prediction_surveys:
    readers = data_partition_object.get_survey_readers(survey)
    print(f'Saving predictions for survey {survey}, {len(readers)} data reader(s)')

    for reader in readers:
        target_dname = os.path.join(PRED_DIR, reader.name + '_prediction.zarr')
        save_survey_predictions_zarr(reader, segpipe, target_dname=target_dname, **config_args)

print(f"Executed time for saving all prediction (h): {np.round((time.time() - start) / 3600, 2)}")
