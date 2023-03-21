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
from utils.np import random_point_containing


class Seabed:
    def __init__(self, echograms, window_size):
        """
        :param echograms: A list of all echograms in set
        :param window_size: (tuple) [height, width]
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

        # Random x-loc with y-location of seabed
        # If window width is greater than echogram width, set x = echogram senter
        if ech.shape[1] <= self.window_size[1]:
            x = ech.shape[1]//2
        else:
            half_patch_width = self.window_size[1] // 2 - 20
            x = np.random.randint(half_patch_width, ech.shape[1]-half_patch_width)

        # Get seabed
        seabed = int(ech.get_seabed(x))

        # If window height is greater than seabed depth, set y = echogram senter
        y = random_point_containing(ech.shape[0], self.window_size[0], seabed)

        return [y, x], ech


class SeabedZarr:
    def __init__(self, zarr_files, window_size=(256, 256)):
        self.zarr_files = zarr_files
        self.window_size = window_size

    def get_sample(self):
        # Get random zarr file
        zarr_rand = np.random.choice(self.zarr_files)

        # get random ping in zarr file
        x = np.random.randint(self.window_size[1] // 2, zarr_rand.shape[0] - self.window_size[1] // 2)

        # Get y-loc at seabed
        y = int(zarr_rand.get_seabed(x))

        if y <= 0:
            return self.get_sample()

        # "adjust" y so that seabed is not always in the middle of the crop
        y += np.random.randint(-self.window_size[0] // 2, self.window_size[0] // 2 + 1)

        return [y, x], zarr_rand
