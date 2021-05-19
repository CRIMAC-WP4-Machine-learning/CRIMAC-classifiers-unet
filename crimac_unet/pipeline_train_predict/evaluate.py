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
from pipeline_train_predict.pipeline import Config_Options, SegPipeUNet
from data.echogram import get_data_readers
from paths import *

if __name__ == '__main__':

    # Configuration options
    configuration = pipeline_config()
    opt = Config_Options(configuration)


    segpipe = SegPipeUNet(opt)
    # Plot and save metrics
    print("Save evaluation results")
    start = time.time()
    segpipe.compute_and_plot_evaluation_metrics(selected_surveys=opt.selected_surveys,
                                                colors_list=opt.colors_list)
    print(f"Executed time for computing metrics (min): {np.round((time.time() - start) / 60, 2)}")