import json
import setpaths
import torch
import numpy as np
import random
import time
import os

# Get the environmental variables passed on from the container
SV_FILE = '/datain/'
BOTTOM_FILE = '/datain/'
PRED_DIR = '/dataout/'
PRED_FILE = PRED_DIR + os.getenv('ZARRFILE')
MODELWEIGHTS = '/model/'+os.getenv('MODEL')
SURVEY = os.getenv('SURVEY')

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
    "path_to_korona_data": SV_FILE,
    "path_to_korona_transducer_depths": SV_FILE,
    "path_to_trained_model": MODELWEIGHTS,
    "path_to_zarr_files": SV_FILE,
    "path_for_saving_figs": PRED_DIR,
    "path_for_saving_preds_labels": PRED_FILE}

# Write setpyenv file based on environmental variables
with open("/crimac_unet/setpyenv.json", "w") as fp:
 json.dump(setpyenv, fp, indent = 4)

# Set the correct paths to the files
from paths import *
from pipeline_train_predict.pipeline import Config_Options, SegPipeUNet, pipeline_config, DataZarr
from data.echogram import get_data_readers

# Check paths
print(' ')
print('Check paths:')
print('path_to_echograms: '+path_to_echograms())
print('path_to_korona_data: '+path_to_korona_data())
print('path_to_korona_transducer_depths: '+path_to_korona_transducer_depths())
print('path_to_trained_model: '+path_to_trained_model())
print('path_to_zarr_files: '+path_to_zarr_files()+' Content:')
print(os.listdir(path_to_zarr_files()))
print('path_for_saving_figs: '+path_for_saving_figs()+' Content:')
print('path_for_saving_preds_labels: '+path_for_saving_preds_labels())

print(' ')
print('CUDA is avaialable: '+ str(torch.cuda.is_available()))

# Configuration options dictionary
configuration = pipeline_config()
configuration['selected_surveys'] = [SURVEY]
print(' ')
print('configuration:')
print(configuration)

# Create options instance
opt = Config_Options(configuration)

# Set up the code
data_obj = DataZarr(opt)
segpipe = SegPipeUNet(opt)
# Save segmentation predictions
print("Save predictions")
start = time.time()
segpipe.save_segmentation_predictions_zarr(data_obj, resume=opt.resume_writing)

print(f"Executed time for saving all prediction (h): {np.round((time.time() - start) / 3600, 2)}")
