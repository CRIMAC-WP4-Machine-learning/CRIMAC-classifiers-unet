import json
import setpaths
import torch
import numpy as np
import random
import time
import os
from crimac_unet.pipeline_train_predict.pipeline import Config_Options, SegPipeUNet, pipeline_config
from crimac_unet.data.echogram import get_data_readers
from crimac_unet.paths import *

np.random.seed(5)
random.seed(5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_model_params = '/model/paper_v2_heave_2.pt'

# Run the predictions

# Configuration options dictionary
configuration = pipeline_config()
# Set specific parameters for this case
configuration['selected_surveys'] = [os.getenv('OUTPUT_NAME', 'out')]
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
