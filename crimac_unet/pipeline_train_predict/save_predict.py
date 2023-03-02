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

import xarray as xr
from numcodecs import Blosc
import shutil
from tqdm import tqdm
import dask

from utils.preload_data_split import get_data_split
from batch.dataset import DatasetGriddedReader
from batch.transforms import define_data_transform, define_data_transform_test, define_label_transform_test, \
    is_use_metadata
from torch.utils.data import DataLoader
from utils.general import (fix_seeds, get_argparse_parser, load_yaml_config,
                           parse_config_options, seed_worker)
from pipeline_train_predict.pipeline import SegPipeUNet
from data.partition import DataMemm, DataZarr
from utils.np import patch_coord_to_data_coord
from paths import *
from constants import *

dask.config.set(scheduler="synchronous")


def fill_out_array(out_array, preds, labels, center_coordinates, ping_start):
    # TODO gather in post-processing step
    selected_label_idxs = np.argwhere((labels != LABEL_OVERLAP_VAL)  # Ignore overlap areas
                                      & (labels != LABEL_SEABED_MASK_VAL)  # Ignore areas under seabed
                                      & (labels != LABEL_BOUNDARY_VAL))  # Ignore areas outside data boundary

    if len(selected_label_idxs) == 0:
        return out_array

    # TODO better variable names
    y_label, x_label = np.transpose(selected_label_idxs)
    # patch_coord = np.transpose(selected_label_idxs)
    # x_array = x_label + center_coordinates[1] - labels.shape[1] // 2 + 1
    # y_array = y_label + center_coordinates[0] - labels.shape[0] // 2 + 1

    # Get corresponding coordinates in data
    data_coords = patch_coord_to_data_coord(np.array(selected_label_idxs),
                                            np.array(center_coordinates),
                                            np.array(labels.shape))
    y_array, x_array = np.transpose(data_coords)

    # adjust according to ping start time
    x_array -= ping_start

    out_array[:, y_array, x_array] = preds[[SANDEEL, OTHER]][:, y_label, x_label]


def fill_out_array_labels(out_array, preds, labels, center_coordinates, ping_start):
    # TODO gather in post-processing step
    selected_label_idxs = np.argwhere((labels != LABEL_OVERLAP_VAL)  # Ignore overlap areas
                                      & (labels != LABEL_SEABED_MASK_VAL)  # Ignore areas under seabed
                                      & (labels != LABEL_BOUNDARY_VAL))  # Ignore areas outside data boundary
    if len(selected_label_idxs) == 0:
        return out_array

    # TODO better variable names
    y_label, x_label = np.transpose(selected_label_idxs)
    x_array = x_label + center_coordinates[1] - labels.shape[1] // 2
    y_array = y_label + center_coordinates[0] - labels.shape[0] // 2

    # adjust according to ping start time
    x_array -= ping_start

    out_array[y_array, x_array] = preds[y_label, x_label]


def create_xarray_ds_predictions(reader, predictions, start_ping, end_ping, model_name):
    ds = xr.Dataset({"annotation": xr.DataArray(data=np.swapaxes(predictions, 1, 2),  # swap axes to match zarr
                                                dims=["category", "ping_time", "range"],
                                                coords={"category": [27, 1],
                                                        "ping_time": reader.time_vector[start_ping:end_ping],
                                                        "range": reader.range_vector})},
                    attrs={"description": f"{model_name} predictions"}).astype(np.float16)
    return ds


def create_xarray_ds_labels(reader, labels, start_ping, end_ping, model_name):
    ds = xr.Dataset({"annotation": xr.DataArray(data=np.swapaxes(labels, 0, 1),  # swap axes to match zarr
                                                dims=["ping_time", "range"],
                                                coords={"ping_time": reader.time_vector[start_ping:end_ping],
                                                        "range": reader.range_vector})},
                    # TODO do this properly
                    attrs={"description": f"{model_name} predictions"}).astype(np.float16)
    return ds


def initialize_zarr_directory(target_dname, resume):
    if not resume:
        # Delete existing zarr dir of predictions
        if os.path.isdir(target_dname):
            print(f"Overwrite {target_dname}")
            shutil.rmtree(target_dname)
        write_first_loop = True
        start_ping = 0
    else:
        assert os.path.isdir(target_dname), \
            f"Cannot resume saving predictions as no existing prediction directory was fount at {target_dname}"
        print("Attempting to resume predictions")
        start_ping = xr.open_zarr(target_dname).sizes['ping_time']
        write_first_loop = False
    return start_ping, write_first_loop


def append_to_zarr(ds, target_dname, write_first_loop):
    # Re-chunk so that we have a full range in a chunk
    ds = ds.chunk({"range": ds.range.shape[0], "ping_time": "auto"})

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {var: {"compressor": compressor} for var in ds.data_vars}

    if write_first_loop:
        ds.to_zarr(target_dname, mode="w", encoding=encoding)
    else:
        ds.to_zarr(target_dname, append_dim="ping_time")


def save_survey_predictions_zarr(reader, segpipe, meta_channels,
                                 patch_size,
                                 patch_overlap,
                                 batch_size,
                                 num_workers,
                                 preload_n_pings,
                                 target_dname,
                                 resume_writing=False, **kwargs):
    use_metadata = is_use_metadata(meta_channels)
    frequencies = segpipe.frequencies

    # get data and label transform
    data_transform = define_data_transform(use_metadata)
    label_transform = define_label_transform_test(frequencies, label_masks='all', patch_overlap=patch_overlap)

    reader_n_pings = reader.shape[0]
    reader_n_range = reader.shape[1]

    # Initialize zarr output directory from the last ping written to the directory if resume_writing=True
    start_prediction_ping, write_first_loop = initialize_zarr_directory(target_dname, resume_writing)

    # For zarr files with large chunks, loading chunks of data can speed up the data loading
    # This will use more memeory, and is not necessary if the chunk size is relatively small
    if preload_n_pings > 0:
        # split data in more managable chunks
        splits = get_data_split([[start_prediction_ping, reader_n_pings]], preload_n_pings)
        data_preload = True
    else:
        splits = get_data_split([[start_prediction_ping, reader_n_pings]], 5000)
        data_preload = False

    model_name = segpipe.model_name

    # Disable this loop if data is not preloaded
    for (start_ping, end_ping) in tqdm(splits, total=len(splits), desc='Predicting ...'):
        # Create data loader for chunk
        dataset = DatasetGriddedReader(reader, patch_size, frequencies, meta_channels=meta_channels,
                                       grid_start=start_ping,
                                       grid_end=end_ping,
                                       patch_overlap=patch_overlap,
                                       data_preload=data_preload,
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

        # Initialize output array
        out_array = np.zeros([2, reader_n_range, end_ping - start_ping])

        # Disable this loop if data is preloaded
        for batch in dataloader:
            # Get softmax predictions and convert to numpy
            predictions = segpipe.predict_batch(batch, return_softmax=True).cpu().numpy()

            # Retrieve labels and center coordinates - used for filling out the array
            labels = batch['labels'].cpu().numpy()
            center_coordinates = batch['center_coordinates'].cpu().numpy()

            # Fill output array with predictions in the batch
            for patch in range(len(batch['center_coordinates'])):
                #data_patch = batch['data'][patch].cpu().numpy()
                pred_patch = predictions[patch]
                label_patch = labels[patch]
                center_coordinates_patch = center_coordinates[patch]

                fill_out_array(out_array, pred_patch, label_patch, center_coordinates_patch, start_ping)

        # After array has been filled, create xarray dataset
        ds = create_xarray_ds_predictions(reader, out_array, start_ping, end_ping, model_name=model_name)

        # Write to zarr
        append_to_zarr(ds, target_dname, write_first_loop)

        write_first_loop = False

        # Rechunk array
        # TODO

def save_reader_predictions_memm(reader, segpipe, meta_channels,
                                 patch_size,
                                 patch_overlap,
                                 batch_size,
                                 num_workers,
                                 target_dname,
                                 resume_writing=False, **kwargs):
    use_metadata = is_use_metadata(meta_channels)
    frequencies = segpipe.frequencies

    # If predictions have already been saved
    if resume_writing and os.path.isfile(target_dname):
        print(f'{os.path.split(target_dname)[1]} already exist')
        return 0

    # get data and label transform
    data_transform = define_data_transform_test(use_metadata)
    label_transform = define_label_transform_test(frequencies, label_masks='all', patch_overlap=patch_overlap)

    # Create Dataset and data loader
    reader_n_pings = reader.shape[1]
    reader_n_range = reader.shape[0]
    dataset = DatasetGriddedReader(reader, patch_size, frequencies, meta_channels=meta_channels,
                                   grid_start=0, grid_end=reader_n_pings, patch_overlap=patch_overlap,
                                   augmentation_function=None, label_transform_function=label_transform,
                                   data_transform_function=data_transform, grid_mode='all')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            worker_init_fn=seed_worker)

    # Initialize out_array
    out_array = np.zeros([2, reader_n_range, reader_n_pings])
    for batch in dataloader:
        # Get softmax predictions and convert to numpy
        predictions = segpipe.predict_batch(batch, return_softmax=True).cpu().numpy()

        for patch in range(len(batch['center_coordinates'])):
            pred_patch = predictions[patch].astype(np.float16)
            label_patch = batch['labels'][patch].cpu().numpy()
            center_coordinates = batch['center_coordinates'][patch].cpu().numpy()

            fill_out_array(out_array, pred_patch, label_patch, center_coordinates, ping_start=0)

    # After array has been filled, save as npy-file
    np.save(target_dname, out_array)


if __name__ == "__main__":
    # Configuration options
    argparse_args = get_argparse_parser(mode="save_predict").parse_args()
    configuration = load_yaml_config(argparse_args.yaml_path)
    config_args = parse_config_options(configuration, argparse_args)

    # Get directory where predictions should be saved
    predictions_dir = config_args["save_predictions_path"]
    experiment_name = config_args["yaml_path"].stem

    # Initialize pipeline
    fix_seeds(config_args["random_seed"])
    segpipe = SegPipeUNet(**config_args, experiment_name=experiment_name)

    # Load model
    segpipe.load_model_params(checkpoint_path=config_args["checkpoint_path"])

    # Get data partition objects
    if config_args["data_mode"] == 'zarr':
        data_partition_object = DataZarr(**config_args)
        survey_tqdm = True
    elif config_args["data_mode"] == 'memm':
        data_partition_object = DataMemm(**config_args)
        survey_tqdm = False
    else:
        raise ValueError("Config variable `data_mode` must be either `zarr` or `memm`")

    # Get list of surveys to save predictions for
    save_prediction_surveys = config_args["save_prediction_surveys"]

    # For each survey (year), get relevant readers
    for survey in save_prediction_surveys:

        readers = data_partition_object.get_survey_readers(survey)
        print(f'Saving predictions for survey {survey}, {len(readers)} data reader(s)')

        for reader in tqdm(readers, total=len(readers), disable=survey_tqdm):
            if config_args["data_mode"] == 'memm':
                target_dname = os.path.join(predictions_dir, reader.name + '_pred.npy')
                save_reader_predictions_memm(reader, segpipe, target_dname=target_dname, **config_args)
            else:
                target_dname = os.path.join(predictions_dir, reader.name + '_pred.zarr')
                save_survey_predictions_zarr(reader, segpipe, target_dname=target_dname, **config_args)
