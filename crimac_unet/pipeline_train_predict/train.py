""""
Copyright 2021 the Norwegian Computing Center

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
"""

import time
import numpy as np
from pipeline_train_predict.pipeline import Config_Options, DataMemm, DataZarr, SegPipeUNet
from paths import *


if __name__ == '__main__':

    # Configuration options
    configuration = pipeline_config()
    opt = Config_Options(configuration)

    if opt.data_mode == 'zarr':
        data_obj = DataZarr(opt)
    elif opt.data_mode == 'memm':
        data_obj = DataMemm(opt)

    print("Preparing data samplers")
    start = time.time()
    samplers_train, samplers_test, sampler_probs = data_obj.sample_data()
    print(f"Executed time for preparing samples (s): {np.round((time.time() - start), 2)}")

    print("Preparing data loaders")
    start = time.time()
    segpipe = SegPipeUNet(opt)
    dataloader_train, dataloader_test = segpipe.define_data_loaders(samplers_train, samplers_test, sampler_probs)
    print(f"Executed time for preparing data loaders (s): {np.round((time.time() - start), 2)}")

    print("Start training")
    start = time.time()
    segpipe.train_model(dataloader_train, dataloader_test)
    print(f"Executed time for training (h): {np.round((time.time() - start) / 3600, 2)}")