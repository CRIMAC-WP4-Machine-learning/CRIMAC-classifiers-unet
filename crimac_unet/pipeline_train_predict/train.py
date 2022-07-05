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
from utils.logger import TensorboardLogger


if __name__ == '__main__':

    # Configuration options
    configuration = pipeline_config()
    opt = Config_Options(configuration)

    if opt.data_mode == 'zarr':
        data_obj = DataZarr(opt)
    elif opt.data_mode == 'memm':
        data_obj = DataMemm(opt)


    segpipe = SegPipeUNet(opt)

    print("Start training")
    logger = TensorboardLogger(name=segpipe.model_name,
                               log_dir='/'.join(segpipe.path_model_params.split('/')[:-1]) + '/log/',
                               include_datetime=False)
    start = time.time()
    segpipe.train_model(data_obj, logger)
    print(f"Executed time for training (h): {np.round((time.time() - start) / 3600, 2)}")
    logger.close()