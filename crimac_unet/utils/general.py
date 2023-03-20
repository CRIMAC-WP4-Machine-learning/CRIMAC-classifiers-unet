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

import argparse
import datetime
import random
from collections import OrderedDict
from pathlib import Path
import os
from shutil import copyfile
from glob import glob

import numpy as np
import torch
import yaml


def load_yaml_config(path_configuration):
    with open(path_configuration, "r") as stream:
        return yaml.safe_load(stream)


def get_argparse_parser(mode='train'):
    assert mode in ['train', 'eval', 'save_predict', 'docker_predict']

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    if mode == 'docker_predict':
        parser.add_argument("--save_model_params", action="store_true", default=False)

        return parser

    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--depth",
        dest="depth",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--data_mode",
        dest="data_mode",
        required=False,
        choices=["memm", "zarr"],
        type=str,
    )

    parser.add_argument(
        "--yaml_path",
        dest="yaml_path",
        type=lambda p: Path(p).resolve(strict=True),
        required=True,
    )

    # Inference modes, add path to trained model, and set save_model_params to False
    if mode in ['eval', 'save_predict']:
        parser.add_argument(
            "--checkpoint_path",
            dest="checkpoint_path",
            type=lambda p: Path(p).resolve(strict=True),
            required=True,
        )

        parser.add_argument("--save_model_params", action="store_true", default=False)

        if mode == 'eval':  # Add paths to save evaluation results
            parser.add_argument(
                "--save_path_metrics",
                dest="save_path_metrics",
                type=lambda p: Path(p).resolve(strict=True),
                required=True,
            )

            parser.add_argument(
                "--save_path_plot",
                dest="save_path_plot",
                type=lambda p: Path(p).resolve(strict=True),
                required=True,
            )

        else:  # Mode is save_predict, add path to save predictions
            parser.add_argument(
                "--save_predictions_path",
                dest="save_predictions_path",
                type=lambda p: Path(p).resolve(strict=True),
                required=True,
            )

    return parser


def fix_seeds(random_seed):
    # fix random seeds to reproduce results
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_config_options(configuration, argparse_args):
    """ Object to represent configuration options for the training-prediction pipeline """
    args_dict = {}
    for configs in [configuration, vars(argparse_args)]:
        # Note: if same key in two configs, argparse args take presedence
        for k, v in configs.items():
            args_dict[k] = v

    return args_dict


def config_args_to_markdown(config_args):
    config_markdown = "| Variable | Value |\n| ---- | ---------- |"
    for k, w in config_args.items():
        if k[0] != '_':
            config_markdown += f"\n|{str(k)}|{str(w)}|"
    return config_markdown


def get_experiment_name_from_args(argparse_args):
    """generate an experiment name from yaml_path and specified argparse args"""

    args_subset = OrderedDict(vars(argparse_args))

    # skip some argparse args that are not part of experiment specification
    for k in ['save_model_params', 'checkpoint_path', 'num_workers']:
        args_subset.pop(k, None)

    args_subset['yaml_path'] = args_subset['yaml_path'].stem

    return ",".join(
        [f"{k}={v}" for k, v in args_subset.items()]
    )


def get_datetime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def copy_source(log_dir):
    source_copy_dir = os.path.join(log_dir, 'code')
    os.makedirs(source_copy_dir)

    # Get folders in which to search
    paths = [os.getcwd()]
    # Todo: Should we use current working directory only, or all folders in python-path?
    # list(set([p for p in sys.path    if not any([ignore in p for ignore in ignore_patterns])]))

    folders_2_ignore = ['/log', 'lib/python', '/tensorboard_logs', '/saved_models']



    for path in paths:
        for root, _, files in os.walk(path):
            base_folder = path.split('/')[-1]

            if '__ignore__.py' in files or any([p2i in root for p2i in folders_2_ignore]):
                folders_2_ignore.append(root)
                continue

            for file in files:
                if file.endswith(".py"):
                    old = os.path.join(root, file)

                    # TODO something more robust
                    new_dir = source_copy_dir + str(root.replace(path, ''))
                    new = os.path.join(new_dir, file)

                    if not os.path.isdir(new_dir):
                        os.makedirs(new_dir)

                    copyfile(old, new)

    print(' - Copying source for backup - FINISHED')

