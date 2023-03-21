import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from paths import *
from batch.dataset import DatasetGriddedReader
from batch.transforms import define_data_transform, define_data_transform_test, define_label_transform_test, \
    is_use_metadata
from torch.utils.data import DataLoader, ConcatDataset
from utils.general import (fix_seeds, get_argparse_parser, load_yaml_config,
                           parse_config_options, seed_worker)
from utils.preload_data_split import get_data_split
from pipeline_train_predict.pipeline import SegPipeUNet
from data.partition import DataMemm, DataZarr


def save_metrics_dict(metrics, save_path_metrics):
    thresholds = np.array(list(metrics['thresholds']) + [np.nan])
    metrics['thresholds'] = thresholds
    df = pd.DataFrame(metrics)
    df.to_csv(save_path_metrics)


def save_plot(metrics, save_path_plot):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Recall", fontsize=8)
    ax.set_ylabel("Precision", fontsize=8)
    ax.set_xticks(ticks)
    ax.scatter(metrics['recall'], metrics['precision'], s=2)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    plt.savefig(save_path_plot)


def validate_model_survey_zarr(readers, segpipe, meta_channels, patch_size, patch_overlap, eval_mode, batch_size,
                               num_workers,
                               save_path_metrics,
                               save_path_plot,
                               preload_n_pings,
                               **kwargs):
    use_metadata = is_use_metadata(meta_channels)
    frequencies = segpipe.frequencies

    # get data and label transforms
    data_transform = define_data_transform(use_metadata)
    label_transform = define_label_transform_test(frequencies=frequencies, label_masks=eval_mode,
                                                  patch_overlap=patch_overlap)

    assert len(readers) == 1, print("Current evaluation code assumes one zarr file contains an entire survey")

    for reader in readers:
        assert preload_n_pings == 0, print("Current evaluation code for zarr only works when 'preloading_n_pings' = 0")

        dataset = DatasetGriddedReader(reader, patch_size, frequencies,
                                       meta_channels=meta_channels,
                                       grid_start=None,
                                       grid_end=None,
                                       patch_overlap=patch_overlap,
                                       data_preload=False,
                                       augmentation_function=None,
                                       label_transform_function=label_transform,
                                       data_transform_function=data_transform,
                                       grid_mode='all')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

        segpipe.validate_model_testing(dataloader,
                                       save_path_metrics=os.path.join(save_path_metrics, f"{survey}_test.csv"),
                                       save_path_plot=os.path.join(save_path_plot, f"{survey}_pr.png"))


def validate_model_survey_memm(readers, segpipe, meta_channels, patch_size, patch_overlap, eval_mode, batch_size,
                               num_workers,
                               save_path_metrics,
                               save_path_plot,
                               **kwargs):
    use_metadata = is_use_metadata(meta_channels)
    frequencies = segpipe.frequencies

    # get data and label transforms
    data_transform = define_data_transform_test(use_metadata)
    label_transform = define_label_transform_test(frequencies=frequencies, label_masks=eval_mode,
                                                  patch_overlap=patch_overlap)

    # For each relevant reader, one dataset is created
    list_of_datasets = []
    for reader in readers:
        list_of_datasets.append(DatasetGriddedReader(reader, patch_size, frequencies, meta_channels=meta_channels,
                                                     grid_start=None,
                                                     grid_end=None,
                                                     patch_overlap=patch_overlap,
                                                     augmentation_function=None,
                                                     label_transform_function=label_transform,
                                                     data_transform_function=data_transform,
                                                     grid_mode='all'))
    concatenated_dataset = ConcatDataset(list_of_datasets)
    dataloader = DataLoader(
        concatenated_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    segpipe.validate_model_testing(dataloader,
                                   save_path_metrics=os.path.join(save_path_metrics, f"{survey}_test.csv"),
                                   save_path_plot=os.path.join(save_path_plot, f"{survey}_pr.png"))


if __name__ == "__main__":
    # Configuration options
    argparse_args = get_argparse_parser(mode="eval").parse_args()
    configuration = load_yaml_config(argparse_args.yaml_path)
    config_args = parse_config_options(configuration, argparse_args)

    experiment_name = config_args["yaml_path"].stem

    # Initialize pipeline
    fix_seeds(config_args["random_seed"])
    segpipe = SegPipeUNet(**config_args, experiment_name=experiment_name)
    segpipe.load_model_params(checkpoint_path=config_args["checkpoint_path"])

    print(f'\nLoading {config_args["data_mode"]} data partition object...')
    start = time.time()
    if config_args["data_mode"] == 'zarr':
        data_partition_object = DataZarr(**config_args)
    elif config_args["data_mode"] == 'memm':
        data_partition_object = DataMemm(**config_args)
    else:
        raise ValueError('data_mode not in ["zarr", "memm"]')
    print("Executed time for loading data partition object (min):" f" {np.round((time.time() - start) / 60, 2)}")

    # get surveys to run evaluation on
    evaluation_surveys = data_partition_object.get_evaluation_surveys()

    # TODO create initialize function
    config_args["save_path_metrics"] = os.path.join(*[config_args["save_path_metrics"], experiment_name,
                                       os.path.normpath(config_args["checkpoint_path"]).split(os.path.sep)[-2]])
    config_args["save_path_plot"] = os.path.join(*[config_args["save_path_plot"], experiment_name,
                                    os.path.normpath(config_args["checkpoint_path"]).split(os.path.sep)[-2]])

    if not os.path.isdir(config_args["save_path_metrics"]):
        os.makedirs(config_args["save_path_metrics"])
    if not os.path.isdir(config_args["save_path_plot"]):
        os.makedirs(config_args["save_path_plot"])

    print(f"\nMetrics directory:", config_args["save_path_metrics"])
    print(f"Plot directory:", config_args["save_path_plot"], "\n")

    for survey in evaluation_surveys:
        readers = data_partition_object.get_survey_readers(survey)

        print('Running evaluation for', survey)
        if config_args["data_mode"] == "zarr":
            validate_model_survey_zarr(readers, segpipe, **config_args)
        else:
            validate_model_survey_memm(readers, segpipe, **config_args)
