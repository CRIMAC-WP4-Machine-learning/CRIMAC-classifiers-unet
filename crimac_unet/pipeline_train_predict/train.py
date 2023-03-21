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
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from batch.dataset import Dataset, DatasetGriddedReader
from batch.transforms import (define_data_augmentation, define_data_transform, define_data_transform_test,
                              define_label_transform_train, define_label_transform_test, is_use_metadata)
from data.partition import DataMemm, DataZarr
from paths import *
from utils.general import (fix_seeds, get_argparse_parser, get_datetime_str,
                           load_yaml_config, config_args_to_markdown, copy_source,
                           parse_config_options, seed_worker)
from pipeline_train_predict.pipeline import SegPipeUNet


def define_data_loaders(data_obj,
                        batch_size,
                        iterations,
                        test_iter,
                        patch_size,
                        meta_channels,
                        num_workers,
                        **kwargs):
    # Are we training with metadata?
    use_metadata = is_use_metadata(meta_channels)
    frequencies = data_obj.frequencies

    # Divide data into training and test
    print("Preparing data samplers")
    start = time.time()
    readers_train, readers_test = data_obj.partition_data_train()
    samplers_train, samplers_test, sampler_probs = data_obj.get_samplers_train(readers_train, readers_test)
    print(f"Executed time for preparing samples (s): {np.round((time.time() - start), 2)}\n")

    print("Preparing data loaders")
    # Define data augmentation, and data and label transforms for training
    data_augmentation = define_data_augmentation(use_metadata)
    label_transform_train = define_label_transform_train(frequencies)
    data_transform = define_data_transform(use_metadata)

    # Prepare dataset and dataloader for training
    dataset_train = Dataset(
        samplers_train,
        patch_size,
        frequencies,
        meta_channels,
        n_samples=batch_size * iterations,
        sampler_probs=sampler_probs,
        augmentation_function=data_augmentation,
        label_transform_function=label_transform_train,
        data_transform_function=data_transform,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    # Define label and data transform for testing
    label_transform_test = define_label_transform_test(frequencies, label_masks="all", patch_overlap=0)

    # TODO consider same transform as in test -> currently included to match testing scheme from earlier code
    # Values outside data boundary set to 0
    data_transform_test = define_data_transform_test(use_metadata)

    # Create test dataloader
    dataset_test = Dataset(
        samplers_test,
        patch_size,
        frequencies,
        meta_channels,
        n_samples=batch_size * test_iter,
        sampler_probs=sampler_probs,
        augmentation_function=None,
        label_transform_function=label_transform_test,
        data_transform_function=data_transform_test,
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    print(f"Executed time for preparing dataloaders (s): {np.round((time.time() - start), 2)}\n")
    return dataloader_train, dataloader_test


if __name__ == '__main__':
    # Parse configuration options
    argparse_args = get_argparse_parser().parse_args()
    yaml_path = argparse_args.yaml_path
    configuration = load_yaml_config(yaml_path)
    config_args = parse_config_options(configuration, argparse_args)

    # Get paths to save model and tensorboard log
    experiment_name = config_args["yaml_path"].stem
    experiment_id = get_datetime_str()
    checkpoint_dir = Path("saved_models", experiment_name, experiment_id)
    log_dir = Path("tensorboard_logs", experiment_name, experiment_id)

    # Ensure model training can be reproduced by fixing seeds, setting cudnn.deterministic = True
    fix_seeds(config_args["random_seed"])

    print("Data mode:", config_args["data_mode"])
    # Create dataloader for training and validation
    if config_args["data_mode"] == 'zarr':
        data_partition_object = DataZarr(**config_args)
    elif config_args["data_mode"] == 'memm':
        data_partition_object = DataMemm(**config_args)
    dataloader_train, dataloader_test = define_data_loaders(data_partition_object, **config_args)

    # Initialize training pipeline
    segpipe = SegPipeUNet(**config_args, checkpoint_dir=checkpoint_dir, experiment_name=experiment_name)

    # Initialize tensorboard logger
    print("Start training")
    logger = SummaryWriter(
        log_dir=log_dir,
        comment=f"{get_datetime_str()}, experiment_name={experiment_name}",
    )

    # Log config file, and save a copy in the tensorboard directory
    config_markdown = config_args_to_markdown(config_args)
    logger.add_text('Config', config_markdown, global_step=0)
    os.system(f"cp {yaml_path} {log_dir}")

    # save a copy of the code in the tensorboard
    # copy_source(log_dir)

    # Train model
    start = time.time()
    segpipe.train_model(dataloader_train, dataloader_test, logger)
    print(f"Executed time for training (h): {np.round((time.time() - start) / 3600, 2)}")
