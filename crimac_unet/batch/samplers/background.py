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
from utils.np import getGrid, nearest_interpolation


class Background():
    def __init__(self, echograms, window_size):
        """
        :param echograms: A list of all echograms in set
        :param window_size: (tuple), [height, width]
        """
        self.echograms = echograms
        self.window_size = window_size


    def get_sample(self):
        """
        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        # Random echogram
        ech_index = np.random.randint(len(self.echograms))
        ech = self.echograms[ech_index]

        # Random x, y-loc above seabed
        # If window width is greater than echogram width, set x = echogram senter
        if ech.shape[1] <= self.window_size[1]:
            x = ech.shape[1]//2
        else:
            half_patch_width = self.window_size[1] // 2 - 20
            x = np.random.randint(half_patch_width, ech.shape[1]-half_patch_width)

        # Select random location above seabed, or in the middle of the echogram if window height > echogram height
        seabed = int(ech.get_seabed(x))
        if seabed <= self.window_size[0]:
            y = ech.shape[0]//2
        else:
            y = np.random.randint(self.window_size[0]//2, seabed-self.window_size[0]//2)

        # Check if there is any fish-labels in crop
        grid = getGrid(self.window_size) + np.expand_dims(np.expand_dims([y, x], 1), 1)
        labels = nearest_interpolation(ech.label_memmap(), grid, boundary_val=0, out_shape=self.window_size)

        if np.any(labels != 0):
            return self.get_sample() #Draw new sample

        return [y, x], ech


class BackgroundZarr():
    def __init__(self, zarr_files, window_size=(256, 256)):
        """
        Sample from zarr-files
        :param zarr_files: (list)
        :param window_size: (tuple), height, width
        """
        self.zarr_files = zarr_files
        self.window_size = window_size

    def get_sample(self):
        # Select random zarr file in list
        zarr_rand = np.random.choice(self.zarr_files)

        # select random valid ping range in zarr file
        valid_pings_ranges = zarr_rand.get_valid_pings()
        start_ping, end_ping = valid_pings_ranges[np.random.randint(len(valid_pings_ranges))]

        x = np.random.randint(start_ping, end_ping)

        # Get y-loc above seabed
        seabed = int(zarr_rand.get_seabed(x))

        if seabed - self.window_size[0]//2 <= 0:
            return self.get_sample()
        y = np.random.randint(0, seabed-self.window_size[0]//2)

        # Check if any fish_labels in the crop
        labels = zarr_rand.get_label_slice(idx_ping=x-self.window_size[1]//2,
                                           n_pings=self.window_size[1],
                                           idx_range=max(0, y-self.window_size[0]//2),
                                           n_range=self.window_size[0],
                                           drop_na=False,
                                           return_numpy=False)

        # Check if any fish-labels in crop
        if (labels > 0).any(): # Possible bottleneck?
            return self.get_sample()

        return [y, x], zarr_rand