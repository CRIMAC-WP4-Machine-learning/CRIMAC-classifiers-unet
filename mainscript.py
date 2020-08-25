import json
import setpaths
import segmentation2nd
import torch
import numpy as np
import random

np.random.seed(5)
random.seed(5)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
path_model_params = '/model/paper_v2_heave_2.pt'

### Uncomment and run script ###
segmentation2nd.plot_echograms_with_sandeel_prediction(
    year=2016, device=device,
    path_model_params=path_model_params, ignore_mode='normal')

# Set parameters
# Write parameters

# Run the predictions
