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
                 use_metadata=False,
                 meta_channels=[],
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
        self.use_metadata = use_metadata
        self.meta_channels = meta_channels
        self.sampler_probs = sampler_probs
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function

        if self.use_metadata:
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

    def __getitem__(self, index):
        # Select which sampler to use
        i = np.random.rand()
        sampler = self.samplers[np.where(i < self.sampler_probs)[0][0]]

        # Draw coordinate and echogram with sampler
        center_location, echogram = sampler.get_sample()

        # Get data/labels-patches
        if self.use_metadata:
            data, meta, labels = get_crop(echogram, center_location, self.window_size, self.frequencies,
                                          self.use_metadata, self.meta_channels)
        else:
            data, labels = get_crop(echogram, center_location, self.window_size, self.frequencies, False, [])

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
        if self.meta_channels == []:
            return data, labels
        else:
            return np.concatenate((data, meta), axis=0), labels

    def __len__(self):
        return self.n_samples

def get_crop(reader, center_location, window_size, freqs, use_metadata, meta_channels):
    if isinstance(reader, Echogram):
        return get_crop_memmap(reader, center_location, window_size, freqs, use_metadata, meta_channels)
    elif isinstance(reader, DataReaderZarr):
        return get_crop_zarr(reader, center_location, window_size, freqs)

def get_crop_memmap(echogram, center_location, window_size, freqs, use_metadata=False, meta_channels=[]):
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

    # Case with metadata
    if use_metadata:
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
                depth_abs_surface = crop_idx[0].reshape(-1, 1) * np.ones((1, window_size[1])) / window_size[0] # Div by w_size to get range in the order of [0, 1]
                assert depth_abs_surface.shape == tuple(window_size)
                meta.append(depth_abs_surface[None, ...])

            if meta_channels['depth_abs_seabed']:
                depth_abs_seabed =  (seabed[crop_idx[1]].reshape(1, -1) - crop_idx[0].reshape(-1, 1)) / window_size[0] # Div by w_size to get range in the order of [0, 1]
                assert depth_abs_seabed.shape == tuple(window_size)
                meta.append(depth_abs_seabed[None, ...])

        ###

        if meta != []:
            meta = np.concatenate(meta, 0)

        return channels, meta, labels

    # Case without metadata
    else:
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
