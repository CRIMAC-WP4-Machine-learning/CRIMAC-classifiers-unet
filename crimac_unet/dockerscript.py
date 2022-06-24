import json
import setpaths
import torch
import numpy as np
import random
import time
import os
from pipeline_train_predict.pipeline import Config_Options, SegPipeUNet, pipeline_config
from data.echogram import get_data_readers
# from paths import *


# Get the file paths
SV_FILE = '/datain'+os.getenv('SV_FILE')
BOTTOM_FILE = '/datain'+os.getenv('BOTTOM_FILE')
PRED_FILE = '/dataout'+os.getenv('PRED_FILE')
MODELWEIGHTS = '/model'+os.getenv('MODELWEIGHTS')

# The docker mounts the main data folder as /datain/
print('********************')
print(' ')
print('Files:')
print('sv file:'+SV_FILE+' exists:'+ str(os.path.isdir(SV_FILE)))
print('Bottom file:'+BOTTOM_FILE+' exists:'+str(os.path.isdir(BOTTOM_FILE)))
print('Pred ouptutfile:'+PRED_FILE+' exists:'+str(os.path.isdir(PRED_FILE)))
print('Models weihts file:'+MODELWEIGHTS+' exists:'+str(os.path.isfile(MODELWEIGHTS)))





