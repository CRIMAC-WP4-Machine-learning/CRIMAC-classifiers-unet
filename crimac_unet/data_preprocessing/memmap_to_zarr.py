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

import numpy as np
import os
from datetime import datetime, timedelta
import xarray as xr
from tqdm import tqdm

from data.data_reader import get_echograms
from pipeline_train_predict.save_predict import initialize_zarr_directory, append_to_zarr


def timevector_to_datetime(time_vector):
    days = time_vector % 1
    return np.datetime64(datetime.fromordinal(int(time_vector)) + timedelta(days=days) - timedelta(days=366))


def create_data_ds(ech, range_vector):
    max_range = len(range_vector)

    # Transpose data to match zarr data
    data = ech.data_numpy().T
    n_freqs, n_pings, n_range = data.shape

    # Fill with nans if range is smaller than max range
    data_full = np.full((n_freqs, n_pings, max_range), np.nan).astype(float)
    data_full[:] = np.nan
    data_full[:, :, :n_range] = data

    # Initialize xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            sv=(["frequency", "ping_time", "range"], data_full),  # Add sv data
            heave=(["ping_time"], ech.heave),  # Add heave data
        ),
        coords=dict(
            frequency=(["frequency"], ech.frequencies.astype(np.float32)),  # add coordinates
            ping_time=(
                ["ping_time"], np.array([timevector_to_datetime(t) for t in ech.time_vector]).astype('datetime64[ns]')),
            range=(["range"], range_vector),
            raw_file=(["ping_time"], [ech.name + '.raw' for _ in range(len(ech.time_vector))]),
        ),
        attrs=dict(description="Sv data extracted from memmap file"),
    )

    # This does not seem to do anything ...
    ds = ds.chunk({'frequency': 1, 'ping_time': 1000, 'range': 1000})
    return ds


def create_labels_ds(ech, categories, range_vector):
    max_range = len(range_vector)

    # Retrieve labels
    labels = ech.label_numpy()

    # Create zarr annotation array
    label_full = np.zeros([len(categories), labels.shape[1], max_range]).astype(np.float32)
    label_range = labels.shape[0]
    for j, cat in enumerate(categories):
        label_full[j, :, :label_range] = (labels == cat).T.astype(np.float32)

    # Initialize xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            annotation=(["category", "ping_time", "range"], label_full),  # Add sv data
        ),
        coords=dict(
            category=(["category"], categories),  # add coordinates
            ping_time=(
                ["ping_time"], np.array([timevector_to_datetime(t) for t in ech.time_vector]).astype('datetime64[ns]')),
            range=(["range"], range_vector),
            raw_file=(["ping_time"], [ech.name + '.raw' for _ in range(len(ech.time_vector))]),
        ),
        attrs=dict(description="Annotations extracted from memmap file"),
    )

    # This does not seem to do anything ...
    ds = ds.chunk({'category': 1, 'ping_time': 1000, 'range': 1000})
    return ds


def create_seabed_ds(ech, range_vector):
    max_range = len(range_vector)

    seabed = ech.get_seabed(0, n_pings=ech.shape[1])
    new_sb = np.full([ech.shape[1], max_range], np.nan).astype(float)
    new_sb[:] = np.nan
    for j, s in enumerate(seabed):
        new_sb[j, s:] = 1.0

    # Initialize xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            bottom_range=(["ping_time", "range"], new_sb),  # Add sv data
        ),
        coords=dict(  # add coordinates
            ping_time=(
                ["ping_time"], np.array([timevector_to_datetime(t) for t in ech.time_vector]).astype('datetime64[ns]')),
            range=(["range"], range_vector),
            raw_file=(["ping_time"], [ech.name + '.raw' for _ in range(len(ech.time_vector))]),
        ),
        attrs=dict(description="Seabed extracted from memmap file"),
    )

    # This does not seem to do anything ...
    ds = ds.chunk({'ping_time': 1000, 'range': 1000})
    return ds


def write_zarr_files(out_dir, year):
    # Output zarr directory
    echograms = get_echograms(years=[year])

    # Get zarr paths
    cruise_name = np.unique([ech.name.split('-')[0] for ech in echograms])[0]
    target_data = os.path.join(out_dir, f"{year}/{cruise_name}/ACOUSTIC/GRIDDED/{cruise_name}_sv.zarr")
    target_labels = os.path.join(out_dir, f"{year}/{cruise_name}/ACOUSTIC/GRIDDED/{cruise_name}_labels.zarr")
    target_bottom = os.path.join(out_dir, f"{year}/{cruise_name}/ACOUSTIC/GRIDDED/{cruise_name}_bottom.zarr")

    # Get maximum range for this survey -> fill with nans in range for echograms with smaller range
    max_range = max([ech.shape[0] for ech in echograms])
    max_ech = [ech for ech in echograms if ech.shape[0] == max_range][0]
    range_vector = max_ech.range_vector

    # Get all unique categories in the survey
    categories = [-1]
    for ech in echograms:
        categories += list(ech.label_types_in_echogram)
    categories = sorted(np.unique(categories))

    # Initialize zarr output
    _ = initialize_zarr_directory(target_data, resume=False)
    _ = initialize_zarr_directory(target_labels, resume=False)
    _ = initialize_zarr_directory(target_bottom, resume=False)

    # Add data for each echogram
    for i, ech in tqdm(enumerate(echograms), total=len(echograms)):
        # Create xr datasets for data, labels and seabed
        data_ds = create_data_ds(ech, range_vector)
        labels_ds = create_labels_ds(ech, categories=categories, range_vector=range_vector)
        seabed_ds = create_seabed_ds(ech, range_vector)

        # Append to zarr
        if i == 0:
            write_first_loop = True
        else:
            write_first_loop = False

        append_to_zarr(data_ds, target_data, write_first_loop)
        append_to_zarr(labels_ds, target_labels, write_first_loop)
        append_to_zarr(seabed_ds, target_bottom, write_first_loop)

        del data_ds
        del labels_ds
        del seabed_ds


if __name__ == '__main__':
    out_dir = ''

    for year in [2007, 2008, 2009]:
        print(year)
        write_zarr_files(out_dir, year)
