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

import json
import os
import sys
import yaml

try:
    with open('./setpyenv.json') as file:
        json_data = file.read()
    setup_file = json.loads(json_data)
    if 'syspath' in setup_file.keys():
        sys.path.append(setup_file["syspath"])

except:
    class SetupFileIsMissing(Exception): pass
    raise SetupFileIsMissing('Please make a setpyenv.json file in the root directory.')

def load_yaml_config(path_configuration):
    try:
        with open(path_configuration, "r") as stream:
            try:
                return(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    except:
        class SetupFileIsMissing(Exception):pass
        raise SetupFileIsMissing('Please make a pipeline_config.yaml file in the root directory.')

def path_to_echograms():
    # Directory path to echogram data
    return setup_file['path_to_echograms']

def path_to_korona_data():
    # Directory path to predictions from the Korona algorithm in LSSS
    return setup_file['path_to_korona_data']

def path_to_korona_transducer_depths():
    # Directory path to the transducer depths in the Korona predictions (necessary for vertical calibration)
    return setup_file['path_to_korona_transducer_depths']

def path_to_trained_model():
    # Directory path to trained models
    return setup_file['path_to_trained_model']

def path_to_zarr_files():
    # Directory path to zarr files
    return setup_file['path_to_zarr_files']

def path_for_saving_figs():
    # Directory path for saving figures relating to results evaluation
    return setup_file['path_for_saving_figs']

def path_for_saving_preds_labels():
    # Directory path for saving figures relating to results evaluation
    return setup_file['path_for_saving_preds_labels']

def pipeline_config():
    return load_yaml_config(os.path.join(os.path.dirname(__file__), 'pipeline_config.yaml'))
