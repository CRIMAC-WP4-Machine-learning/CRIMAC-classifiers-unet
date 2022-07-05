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

from utils.np import getGrid, linear_interpolation, nearest_interpolation


class Dataset():

    def __init__(self, samplers, window_size, frequencies,
                 n_samples = 1000,
                 sampler_probs=None,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """

        self.samplers = samplers
        self.window_size = window_size
        self.n_samples = n_samples
        self.frequencies = frequencies
        self.sampler_probs = sampler_probs
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function

        # Normalize sampling probabilities
        if self.sampler_probs is None:
            self.sampler_probs = np.ones(len(samplers))
        self.sampler_probs = np.array(self.sampler_probs)
        self.sampler_probs = np.cumsum(self.sampler_probs).astype(float)
        self.sampler_probs /= np.max(self.sampler_probs)

    def __getitem__(self, index):
        #Select which sampler to use
        i = np.random.rand()
        sampler = self.samplers[np.where(i < self.sampler_probs)[0][0]]

        #Draw coordinate and echogram with sampler
        center_location, echogram = sampler.get_sample()

        #Get data/labels-patches
        data, labels = get_crop(echogram, center_location, self.window_size, self.frequencies)

        # Apply augmentation
        if self.augmentation_function is not None:
            data, labels, echogram = self.augmentation_function(data, labels, echogram)

        # Apply label-transform-function
        if self.label_transform_function is not None:
            data, labels, echogram = self.label_transform_function(data, labels, echogram)

        # Apply data-transform-function
        if self.data_transform_function is not None:
            data, labels, echogram, frequencies = self.data_transform_function(data, labels, echogram, self.frequencies)

        labels = labels.astype('int16')
        return data, labels

    def __len__(self):
        return self.n_samples

def get_crop(reader, center_location, window_size, freqs):
    if reader.data_format == 'memmap':
        return get_crop_memmap(reader, center_location, window_size, freqs)
    elif reader.data_format == 'zarr':
        return get_crop_zarr(reader, center_location, window_size, freqs)
    else:
        raise TypeError(f"Reader {type(reader)} unknown")

def get_crop_memmap(echogram, center_location, window_size, freqs):
    """
    Returns a crop of data around the pixels specified in the center_location.
    """
    # Get grid sampled around center_location
    grid = getGrid(window_size) + np.expand_dims(np.expand_dims(center_location, 1), 1)

    channels = []
    for f in freqs:

        # Interpolate data onto grid
        memmap = echogram.data_memmaps(f)[0]
        data = linear_interpolation(memmap, grid, boundary_val=0, out_shape=window_size)
        del memmap

        # Set non-finite values (nan, positive inf, negative inf) to zero
        if np.any(np.invert(np.isfinite(data))):
            data[np.invert(np.isfinite(data))] = 0

        channels.append(np.expand_dims(data, 0))
    channels = np.concatenate(channels, 0)

    labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=window_size)

    return channels, labels


def get_crop_zarr(zarr_file, center_loc, window_size, freqs):
    # Initialize output arrays
    boundary_val_data = 0
    boundary_val_label = -100
    out_data = np.ones(shape=(len(freqs), window_size[0], window_size[1]))*boundary_val_data
    out_labels = np.ones(shape=(window_size[0], window_size[1]))*boundary_val_label

    x0, x1 = (int(center_loc[0]) - window_size[0] // 2, int(center_loc[0]) + window_size[0] // 2)
    y0, y1 = (int(center_loc[1]) - window_size[1] // 2, int(center_loc[1]) + window_size[1] // 2)

    # get data
    zarr_crop_x = (max(x0, 0), min(zarr_file.shape[0], x1))
    zarr_crop_y = (max(y0, 0), min(zarr_file.shape[1], y1))

    #channels = np.array(zarr_file.get_data_ping_range(zarr_crop_x, zarr_crop_y, frequencies=freqs, drop_na=False))
    channels = zarr_file.get_data_slice(idx_ping=zarr_crop_x[0], n_pings=zarr_crop_x[1] - zarr_crop_x[0],
                                        idx_range=zarr_crop_y[0], n_range=zarr_crop_y[1] - zarr_crop_y[0],
                                        frequencies=freqs,
                                        drop_na=False,
                                        return_numpy=True)

    labels = zarr_file.get_label_slice(idx_ping=zarr_crop_x[0], n_pings=zarr_crop_x[1] - zarr_crop_x[0],
                                       idx_range=zarr_crop_y[0], n_range=zarr_crop_y[1] - zarr_crop_y[0],
                                       drop_na=False,
                                       return_numpy=True)

    # add to crop
    crop = [zarr_crop_x[0]-x0, window_size[0]-(x1-zarr_crop_x[1]),
            zarr_crop_y[0]-y0, window_size[1]-(y1-zarr_crop_y[1])]

    # outputshape freqs, y, x
    out_data[:, crop[2]:crop[3], crop[0]:crop[1]] = np.nan_to_num(channels.swapaxes(1, 2), nan=boundary_val_data)
    out_labels[crop[2]:crop[3], crop[0]:crop[1]] = np.nan_to_num(labels.T, nan=boundary_val_label)

    return out_data, out_labels
