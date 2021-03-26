import json
import setpaths
import torch
import numpy as np
import random
import time
from pipeline_train_predict.pipeline import Config_Options, SegPipeUNet
from data.echogram import get_data_readers
from paths import *

import pdb

np.random.seed(5)
random.seed(5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_model_params = '/model/paper_v2_heave_2.pt'
ncfile = '/datawork/'

# Run the predictions

# Configuration options
opt = Config_Options(
    data_mode='zarr',
    unit_frequency='Hz',
    dev=device,
    path_model_params=path_to_trained_model(),
    eval_mode='all',
    partition_predict='selected surveys',
    selected_surveys=['S2018823'],
    dir_save_preds_labels=path_for_saving_preds_labels(),
    save_labels=False,
    labels_available=False
)


segpipe = SegPipeUNet(opt)
# Save segmentation predictions
print("Save predictions")
start = time.time()
segpipe.save_segmentation_predictions_sandeel(selected_surveys=opt.selected_surveys)
print(f"Executed time for saving all prediction (h): {np.round((time.time() - start) / 3600, 2)}")
