import os
os.chdir('/home/nilsolav/repos/CRIMAC-classifiers-unet/crimac_unet/')
print(os.getcwd())
import json
import setpaths
import torch
import numpy as np
import random
import time
from pipeline_train_predict.pipeline import Config_Options, SegPipeUNet, pipeline_config
from data.echogram import get_data_readers
from paths import *

# Run the predictions

'''
1. Set the following configuration options in the pipeline_config.yaml file:    
    * 'data_mode' can be either 'zarr' (if working with zarr files) or 'memm' (if working with memmap files)
    * 'frequencies' should be a list of frequencies the model is trained on ([18, 38, 120, 200] for the model from Olav's paper). 
    * 'unit_frequency' should be set to 'Hz' if 'zarr' mode is selected and to 'kHz' if 'memm' is selected    
    * 'partition_predict' can be 'selected surveys', 'single survey' or 'all surveys'
    * 'selected_surveys' should be a list of the names of the selected surveys. Should not be an empty list if the previous parameter ('partition_predict') is 'selected surveys' OR 'single survey'.
    * 'dir_save_preds_labels': should not be None
    * 'save_labels': if set to True the labels are assumed to exist (option 'labels_available'=True) and they will also be saved to disk. Otherwise the labels will not be saved.
    * 'eval_mode' can be 'all' (Consider all pixels), 'region' (Exclude all pixels not in a neighborhood of a labeled School) or 'fish' (only evaluate the discrimination on species). Note that the saved labels will look different depending on the chosen configuration parameter 'eval_mode'.
    * 'resume_writing': if set to True it is assumed that a zarr directory of predictions exists and if new raw files are detected, predictions will be appended to the zarr directory
'''

# Configuration options dictionary
configuration = pipeline_config()

# Set specific parameters for this case
# configuration['selected_surveys'] = [os.getenv('OUTPUT_NAME', 'out')]
configuration['labels_available'] = False
configuration['dev'] = 0
# Create options instance
opt = Config_Options(configuration)

segpipe = SegPipeUNet(opt)

# Save segmentation predictions
print("Save predictions")
start = time.time()
segpipe.save_segmentation_predictions_in_zarr(selected_surveys=opt.selected_surveys, resume=opt.resume_writing)
print(f"Executed time for saving all prediction (h): {np.round((time.time() - start) / 3600, 2)}")
