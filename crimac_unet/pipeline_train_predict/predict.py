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
import os
import numpy as np
from pipeline_train_predict.pipeline import Config_Options, SegPipeUNet
from data.echogram import get_data_readers
from paths import *

if __name__ == '__main__':

    # Configuration options
    configuration = pipeline_config()
    opt = Config_Options(configuration)

    segpipe = SegPipeUNet(opt)
    frequencies = segpipe.frequencies

    ## Zarr prediction
    if opt.data_mode == 'zarr':
        if opt.partition_predict == 'all surveys':
            surveys_list = get_data_readers(mode=opt.data_mode, frequencies=frequencies)
        if opt.partition_predict == 'single survey' or opt.partition_predict == 'selected surveys':
            assert len(opt.selected_surveys) != 0
            surveys_list = []
            for survey in get_data_readers(mode=opt.data_mode, frequencies=frequencies):
                for selected_survey in opt.selected_surveys:
                    if survey.name == selected_survey:
                        surveys_list.append(survey)

        for ii, echs in enumerate(surveys_list):
            echs = surveys_list[ii]
            print(f'Survey {echs.name}')
            for raw_file in echs.raw_file_included:
                start = time.time()
                print(f"Predicting echogram {raw_file}")
                if opt.labels_available:
                    seg, labels = segpipe.predict_echogram(echs, raw_file)
                    print(f"Executed time for prediction (s): {np.round((time.time() - start), 2)}")
                    echs.visualize(raw_file=raw_file, predictions=seg, frequencies=frequencies[:2])
                else:
                    seg = segpipe.predict_echogram(echs, raw_file)
                    print(f"Executed time for prediction (s): {np.round((time.time() - start), 2)}")
                    echs.visualize(raw_file=raw_file, predictions=seg, frequencies=frequencies[:2],
                                   show_labels=False)

    ## Memm prediction
    if opt.data_mode == 'memm':
        echs = get_data_readers(mode=opt.data_mode)
        for ech in [e for e in echs if e.year in opt.selected_surveys]:
            start = time.time()
            print(f"Predicting echogram {ech.name}")
            if opt.labels_available:
                seg, labels = segpipe.predict_echogram(ech)
                print(f"Executed time for prediction (s): {np.round((time.time() - start), 2)}")
                ech.visualize(predictions=seg, frequencies=frequencies[:2])
            else:
                seg = segpipe.predict_echogram(ech)
                print(f"Executed time for prediction (s): {np.round((time.time() - start), 2)}")
                ech.visualize(predictions=seg, frequencies=frequencies[:2],
                               show_labels=False)
