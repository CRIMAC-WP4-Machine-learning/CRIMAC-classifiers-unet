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

from utils.np import getGrid, new_get_crop_2d, new_get_crop_3d, patch_coord_to_data_coord

from batch.samplers.gridded import Gridded, get_data_grid
from constants import *

# TODO figure out metadata stuff



class Dataset():

    def __init__(self, samplers, window_size, frequencies,
                 meta_channels=[],
                 n_samples=1000,
                 sampler_probs=None,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size [height, width]
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """

        self.samplers = samplers
        self.window_size = window_size
        self.frequencies = frequencies
        self.meta_channels = meta_channels
        self.sampler_probs = sampler_probs
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function

        if len(self.meta_channels) > 0:
            # Check valid meta_channels input
            assert all([isinstance(cond, bool) for cond in self.meta_channels.values()])
            assert set(self.meta_channels.keys()) == \
                   {'portion_year', 'portion_day', 'depth_rel', 'depth_abs_surface', 'depth_abs_seabed', 'time_diff'}

        # Normalize sampling probabilities
        if self.sampler_probs is None:
            self.sampler_probs = np.ones(len(samplers))

        self.sampler_probs = np.array(self.sampler_probs)
        self.sampler_probs = np.cumsum(self.sampler_probs).astype(float)
        self.sampler_probs /= np.max(self.sampler_probs)

        self.n_samples = n_samples

    def __getitem__(self, index):
        # Randomly select which sampler to use, and draw a random sample that fits the sampler criteria
        i = np.random.rand()
        sampler = self.samplers[np.where(i < self.sampler_probs)[0][0]]
        center_location, data_reader = sampler.get_sample()

        # Get data/labels-patches
        if len(self.meta_channels) > 0:
            data, meta, labels = get_crop(data_reader, center_location, self.window_size, self.frequencies,
                                          self.meta_channels)
        else:
            data, labels = get_crop(data_reader, center_location, self.window_size, self.frequencies, [])

        # Apply augmentation
        if self.augmentation_function is not None:
            if len(self.meta_channels) == 0:
                data, labels, data_reader = self.augmentation_function(data, labels, data_reader)
            else:
                data, meta, labels, data_reader = self.augmentation_function(data, meta, labels, data_reader)

        # Apply label-transform-function
        if self.label_transform_function is not None:
            data, labels, _, data_reader = self.label_transform_function(data, labels, center_location, data_reader)

        # Apply data-transform-function
        if self.data_transform_function is not None:
            data, labels, data_reader, frequencies = self.data_transform_function(data, labels, data_reader,
                                                                                  self.frequencies)

        labels = labels.astype('int16')
        if len(self.meta_channels) == 0:
            return {'data': data, 'labels': labels, 'center_coordinates': np.array(center_location)}
        else:
            return {'data': np.concatenate((data, meta), axis=0), 'labels': labels,
                    'center_coordinates': np.array(center_location)}

    def __len__(self):
        return self.n_samples


class DatasetGriddedReader:
    """
    Grid a data reader, return regular gridded data patches
    """
    def __init__(self, data_reader, window_size, frequencies,
                 meta_channels=[],
                 grid_start=None,
                 grid_end=None,
                 data_preload=False,
                 patch_overlap=20,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None,
                 grid_mode='all'):

        self.data_reader = data_reader
        self.window_size = window_size
        self.frequencies = frequencies
        self.meta_channels = meta_channels
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function
        self.grid_mode = grid_mode
        self.patch_overlap = patch_overlap

        if data_reader.data_format == 'memmap':
            (H, W) = data_reader.shape
        elif data_reader.data_format == 'zarr':
            (W, H) = data_reader.shape
        else:
            print('Unknown reader data format')

        self.grid_ping_start = grid_start if grid_start is not None else 0
        self.grid_ping_end = grid_end if grid_end is not None else W
        self.ping_boundary = [self.grid_ping_start, self.grid_ping_end]

        self.data_shape = (self.grid_ping_end - self.grid_ping_start, H)
        self.data_numpy = None
        self.labels_numpy = None
        self.data_preload = False

        # Initialize sampler
        self.data_grid = get_data_grid(self.data_reader, patch_size=self.window_size,
                                       patch_overlap=patch_overlap, start_ping=grid_start, end_ping=grid_end,
                                       mode=self.grid_mode)

        # For "smaller" chunks of the data, preload

        if np.prod(self.data_shape) < 6e6 and data_reader.data_format == 'zarr' and data_preload:
            self.data_preload = True
            # Load all annotations in the grid range
            # NB swap axes on labels and data to get height as first dimension, not width

            self.label = self.data_reader.get_label_slice(idx_ping=self.grid_ping_start,
                                                n_pings=self.data_shape[0],
                                                return_numpy=True).T
            self.label_preload_start = self.grid_ping_start


            # Load all data covered in grid
            self.data_preload_start = max(0, self.data_grid[0, 1] - self.window_size[1] // 2)
            self.data_preload_end = min(W, self.data_grid[-1, 1] + self.window_size[1] // 2)
            self.n_preload_pings = self.data_preload_end - self.data_preload_start

            self.data = self.data_reader.get_data_slice(idx_ping=self.data_preload_start,
                                                        n_pings=self.n_preload_pings,
                                                        frequencies=self.frequencies,
                                                        return_numpy=True).swapaxes(1, 2)
            self.data_shape = (self.n_preload_pings, H)
        else:
            self.data_preload = False

    def __len__(self):
        return len(self.data_grid)

    # TODO explain this function better
    def get_preload_data_labels(self, center_location):
        # Initialize output arrays
        boundary_val_data = 0

        # TODO check speed and results against prev function
        center_location_labels = center_location - np.array([0, self.grid_ping_start, ])
        grid_labels = getGrid(self.window_size) + np.expand_dims(np.expand_dims(center_location_labels, 1), 1)
        labels = new_get_crop_2d(self.label, grid_labels, boundary_val=LABEL_BOUNDARY_VAL)

        center_location_data = center_location - np.array([0, self.data_preload_start])
        grid_data = getGrid(self.window_size) + np.expand_dims(np.expand_dims(center_location_data, 1), 1)
        data = new_get_crop_3d(self.data, grid_data, boundary_val=boundary_val_data)

        return data, labels

    def __getitem__(self, index):
        center_location = self.data_grid[index]

        # Get data/labels-patches
        if self.data_preload:
            data, labels = self.get_preload_data_labels(center_location)
        else:
            if len(self.meta_channels) > 0:
                data, meta, labels = get_crop(self.data_reader, center_location, self.window_size, self.frequencies,
                                              self.meta_channels, ping_boundary=self.ping_boundary)
            else:
                data, labels = get_crop(self.data_reader, center_location, self.window_size, self.frequencies, [],
                                        ping_boundary=self.ping_boundary)

        # Apply augmentation
        if self.augmentation_function is not None:
            if len(self.meta_channels) == 0:
                data, labels, echogram = self.augmentation_function(data, labels, self.data_reader)
            else:
                data, meta, labels, echogram = self.augmentation_function(data, meta, labels, self.data_reader)

        # Apply label-transform-function
        if self.label_transform_function is not None:
            data, labels, _, echogram = self.label_transform_function(data, labels, center_location, self.data_reader)

        # Apply data-transform-function
        if self.data_transform_function is not None:
            data, labels, echogram, frequencies = self.data_transform_function(data, labels, self.data_reader,
                                                                               self.frequencies)

        labels = labels.astype('int16')
        if len(self.meta_channels) == 0:
            return {'data': data, 'labels': labels, 'center_coordinates': np.array(center_location)}
        else:
            return {'data': np.concatenate((data, meta), axis=0), 'labels': labels,
                    'center_coordinates': np.array(center_location)}


def get_crop(data_reader, center_location, window_size, freqs, meta_channels, ping_boundary=None, range_boundary=None):
    if data_reader.data_format == 'memmap':
        return get_crop_memmap(data_reader, center_location, window_size, freqs, meta_channels)
    elif data_reader.data_format == 'zarr':
        return get_crop_zarr(data_reader, center_location, window_size, freqs, ping_boundary, range_boundary)
    else:
        raise TypeError(f"Reader {type(data_reader)} unknown")


def get_crop_memmap(echogram, center_location, window_size, freqs, meta_channels=[]):
    """
    Returns a crop of data around the pixels specified in the center_location.
    """

    # If window covers the entire water column, adjust y to center
    if echogram.shape[0] <= window_size[0]:
        center_location[0] = echogram.shape[0] // 2

    # Get grid sampled around center_location
    grid = getGrid(window_size) + np.expand_dims(np.expand_dims(center_location, 1), 1)

    channels = []
    for f in freqs:

        # Interpolate data onto grid
        memmap = echogram.data_memmaps(f)[0]
        # data = linear_interpolation(memmap, grid, boundary_val=100, out_shape=window_size)
        # New crop method as the old one never included right + bottom border values
        data = new_get_crop_2d(memmap, grid, boundary_val=DATA_BOUNDARY_VAL)

        del memmap

        # Set non-finite values (nan, positive inf, negative inf) to zero
        if np.any(np.invert(np.isfinite(data))):
            data[np.invert(np.isfinite(data))] = 0

        channels.append(np.expand_dims(data, 0))
    channels = np.concatenate(channels, 0)

    # labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=LABEL_BOUNDARY_VAL, out_shape=window_size)
    labels = new_get_crop_2d(echogram.label_memmap(), grid, boundary_val=LABEL_BOUNDARY_VAL)

    # Case with metadata
    if len(meta_channels) > 0:
        meta = []
        ### Add channels with metadata ###
        # Portion of the year, resp. the day, represented with scalar value, approx. constant within one crop.

        # Portion of the year (scalar)
        if meta_channels['portion_year']:
            portion_year = echogram.portion_of_year_scalar
            meta.append(np.full(window_size, portion_year)[None, ...])

        # Portion of the day (scalar): represented as [sin(t), cos(t)] to make time continuous at midnight (i.e. modulo 24 hours)
        if meta_channels['portion_day']:
            portion_day_idx = center_location[1]
            if portion_day_idx < 0:
                portion_day_idx = 0
            if portion_day_idx >= echogram.portion_of_day_vector.size:
                portion_day_idx = -1
            portion_day = echogram.portion_of_day_vector[portion_day_idx]
            meta.append(np.full(window_size, np.sin(2 * np.pi * portion_day))[None, ...])
            meta.append(np.full(window_size, np.cos(2 * np.pi * portion_day))[None, ...])

        # Relative time vector
        if meta_channels['time_diff']:
            crop_idx = np.arange(center_location[1] - window_size[1] // 2, center_location[1] + window_size[1] // 2)
            crop_idx[crop_idx < 0] = 0
            crop_idx[crop_idx >= echogram.time_vector_diff.size] = -1

            time_vector_diff_for_current_crop = echogram.time_vector_diff[crop_idx]
            out_array = time_vector_diff_for_current_crop.reshape(1, -1) * np.ones((window_size[0], 1))
            meta.append(out_array[None, ...])

        # Depth channels: Relative, absolute distance to surface, absolute distance to seabed
        if any([meta_channels[kw] for kw in ['depth_rel', 'depth_abs_surface', 'depth_abs_seabed']]):
            seabed = echogram._seabed
            crop_idx = [
                np.arange(center_location[0] - window_size[0] // 2, center_location[0] + window_size[0] // 2),
                np.arange(center_location[1] - window_size[1] // 2, center_location[1] + window_size[1] // 2)
            ]
            crop_idx[1][crop_idx[1] < 0] = 0
            crop_idx[1][crop_idx[1] >= seabed.size] = -1

            if meta_channels['depth_rel']:
                depth_rel = crop_idx[0].reshape(-1, 1) / seabed[crop_idx[1]].reshape(1, -1)
                assert depth_rel.shape == tuple(window_size)
                meta.append(depth_rel[None, ...])

            if meta_channels['depth_abs_surface']:
                depth_abs_surface = crop_idx[0].reshape(-1, 1) * np.ones((1, window_size[1])) / window_size[
                    0]  # Div by w_size to get range in the order of [0, 1]
                assert depth_abs_surface.shape == tuple(window_size)
                meta.append(depth_abs_surface[None, ...])

            if meta_channels['depth_abs_seabed']:
                depth_abs_seabed = (seabed[crop_idx[1]].reshape(1, -1) - crop_idx[0].reshape(-1, 1)) / window_size[
                    0]  # Div by w_size to get range in the order of [0, 1]
                assert depth_abs_seabed.shape == tuple(window_size)
                meta.append(depth_abs_seabed[None, ...])

        ###

        if meta != []:
            meta = np.concatenate(meta, 0)

        return channels, meta, labels

    # Case without metadata
    else:
        return channels, labels


def get_crop_zarr(data_reader, center_location, window_size, freqs, ping_boundary=None, range_boundary=None):
    # Initialize output arrays
    boundary_val_data = 0
    out_data = np.ones(shape=(len(freqs), window_size[0], window_size[1])) * boundary_val_data
    out_labels = np.ones(shape=(window_size[0], window_size[1])) * LABEL_BOUNDARY_VAL

    # Get upper left and lower right corners of patch in data
    patch_corners = np.array([[0, 0], [window_size[0], window_size[1]]])
    data_corners = patch_coord_to_data_coord(patch_corners, np.array(center_location), np.array(window_size))

    y0, x0 = data_corners[0]
    y1, x1 = data_corners[1]

    # get data
    if ping_boundary is None:
        ping_boundary = [0, len(data_reader.time_vector)]
    if range_boundary is None:
        range_boundary = [0, len(data_reader.range_vector)]

    zarr_crop_x = (max(x0, ping_boundary[0]), min(ping_boundary[1], x1))
    zarr_crop_y = (max(y0, range_boundary[0]), min(range_boundary[1], y1))

    n_pings = int(zarr_crop_x[1] - zarr_crop_x[0])
    n_range = int(zarr_crop_y[1] - zarr_crop_y[0])

    # channels = np.array(zarr_file.get_data_ping_range(zarr_crop_x, zarr_crop_y, frequencies=freqs, drop_na=False))
    channels = data_reader.get_data_slice(idx_ping=int(zarr_crop_x[0]), n_pings=n_pings,
                                          idx_range=int(zarr_crop_y[0]), n_range=n_range,
                                          frequencies=freqs,
                                          drop_na=False,
                                          return_numpy=True)

    labels = data_reader.get_label_slice(idx_ping=int(zarr_crop_x[0]), n_pings=n_pings,
                                         idx_range=int(zarr_crop_y[0]), n_range=n_range,
                                         drop_na=False,
                                         return_numpy=True)
    #
    # if np.all(np.isnan(channels)):
    #     return out_data, out_labels

    # add to crop
    crop = [zarr_crop_x[0] - x0, window_size[0] - (x1 - zarr_crop_x[1]),
            zarr_crop_y[0] - y0, window_size[1] - (y1 - zarr_crop_y[1])]


    # outputshape freqs, y, x
    out_data[:, crop[2]:crop[3], crop[0]:crop[1]] = np.nan_to_num(channels.swapaxes(1, 2), nan=boundary_val_data)
    out_labels[crop[2]:crop[3], crop[0]:crop[1]] = np.nan_to_num(labels.T, nan=LABEL_BOUNDARY_VAL)

    return out_data, out_labels
