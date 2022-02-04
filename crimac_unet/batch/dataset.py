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
from data.echogram import Echogram, DataReaderZarr
from torchvision import transforms

class Dataset():

    def __init__(self, samplers, window_size, frequencies,
                 n_samples=1000,
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
        # Select which sampler to use
        i = np.random.rand()
        sampler = self.samplers[np.where(i < self.sampler_probs)[0][0]]

        # Draw coordinate and echogram with sampler
        center_location, echogram = sampler.get_sample()

        # Adjust coordinate by random shift in y and x direction
        center_location[0] += np.random.randint(-self.window_size[0] // 2, self.window_size[0] // 2 + 1)
        center_location[1] += np.random.randint(-self.window_size[1] // 2, self.window_size[1] // 2 + 1)

        # Get data/labels-patches
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
    if isinstance(reader, Echogram):
        return get_crop_memmap(reader, center_location, window_size, freqs)
    elif isinstance(reader, DataReaderZarr):
        return get_crop_zarr(reader, center_location, window_size, freqs)

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

    # Get corner indexes
    x0, x1 = (center_loc[0] - window_size[0] // 2, center_loc[0] + window_size[0] // 2)
    y0, y1 = (center_loc[1] - window_size[1] // 2, center_loc[1] + window_size[1] // 2)

    # Get data selection in ping_range
    # Handle cases where y0 < 0 or x0 <0:
    if y0 < 0:
        y0 = 0
    if x0 < 0:
        x0 = 0

    # TODO replace get_data_ping_range with get_data_slice
    channels = zarr_file.get_data_ping_range((x0, x1), (y0, y1), frequencies=freqs, drop_na=True)
    labels = zarr_file.get_label_ping_range((x0, x1), (y0, y1), drop_na=True)

    # Get grid sampled around center_location
    grid = getGrid(window_size) + np.expand_dims(np.expand_dims([window_size[0] // 2, center_loc[1]], 1), 1)

    channels_out = []
    for ii in range(len(freqs)):

        # Interpolate data onto grid
        xr_data = channels.values[ii, :, :]
        data = linear_interpolation(xr_data, grid, boundary_val=0, out_shape=window_size)
        del xr_data

        # Set non-finite values (nan, positive inf, negative inf) to zero
        if np.any(np.invert(np.isfinite(data))):
            data[np.invert(np.isfinite(data))] = 0

        channels_out.append(np.expand_dims(data, 0))
    channels_out = np.concatenate(channels_out, 0)

    labels_out = nearest_interpolation(labels.values, grid, boundary_val=-100, out_shape=window_size)

    # Transpose to match memmap
    return channels_out.swapaxes(1, 2), labels_out.T
