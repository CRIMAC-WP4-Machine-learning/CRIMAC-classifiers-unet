import json
import setpaths
import segmentation2nd
import torch
import numpy as np
import random
from data_preprocessing.generate_heave_compensation_files import generate_and_save_heave_files

import pdb

np.random.seed(5)
random.seed(5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_model_params = '/model/paper_v2_heave_2.pt'
ncfile = '/datawork/'

# generate_and_save_heave_files()

### Uncomment and run script ###
#segmentation2nd.plot_echograms_with_sandeel_prediction(
#    year=2016, device=device,
#    path_model_params=path_model_params, ignore_mode='normal')

segmentation2nd.write_predictions(
    year=2016, device=device,
    path_model_params=path_model_params,
    ignore_mode='normal', ncfile=ncfile)



# Set parameters
# Write parameters

# Run the predictions
