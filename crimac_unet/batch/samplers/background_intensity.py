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

from data.normalization import db
from utils.np import getGrid, nearest_interpolation
import numpy as np


class HighIntensityBackground():
    def __init__(self, echograms, window_size, partition=0.3):
        """
        :param echograms: A list of all echograms in set
        """
        self.echograms = echograms
        self.window_size = window_size
        self.partition = partition  # Sample from the upper X% of the echogram (more plankton)

    def get_sample(self):
        # Random echogram
        ei = np.random.randint(len(self.echograms))

        # Random x,y-loc above partiion
        (H, W) = self.echograms[ei].shape
        x = np.random.randint(self.window_size[1] // 2, W - self.window_size[1] // 2)

        y_max = int(self.echograms[ei].get_seabed()[x] * self.partition)
        if y_max <= self.window_size[0] // 2:
            y_max = self.window_size[0] // 2 + 1
        y = np.random.randint(self.window_size[1] // 2, y_max)

        # Check if there is any fish-labels in crop
        grid = getGrid(self.window_size) + np.expand_dims(np.expand_dims([y, x], 1), 1)
        labels = nearest_interpolation(self.echograms[ei].label_memmap(), grid, boundary_val=0,
                                       out_shape=self.window_size)

        if np.any(labels != 0):
            return self.get_sample()  # Draw new sample

        # check intensity in crop
        data = nearest_interpolation(self.echograms[ei].data_memmaps(frequencies=[200])[0], grid)
        mean_200 = np.nanmean(db(data))
        if mean_200 < -75:
            return self.get_sample()

        return [y, x], self.echograms[ei]


if __name__ == '__main__':
    from data.echogram import get_echograms

    echograms = get_echograms(years=[2014, 2016])
    window_size = (256, 256)
    high_int = HighIntensityBackground(echograms, window_size)
