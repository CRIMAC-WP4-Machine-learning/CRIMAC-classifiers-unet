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
from constants import *

# TODO add padding above seabed
# TODO no need for class, can be regular function
class mask_label_seabed():
    def __init__(self, ignore_val=-50):
        """
        Set labels below seabed to LABEL_SEABED_MASK_VAL
        :param ignore_val:
        """
        self.ignore_val = ignore_val

    def __call__(self, data, labels, center_coord, echogram):

        # Mask areas under seabed
        # TODO function to go from grid to label coord?
        y_upper, x_left = np.array(center_coord) - np.array(labels.shape) // 2 + 1
        y_lower, x_right = np.array(center_coord) + np.array(labels.shape) // 2 + 1

        reader_shape = echogram.shape
        if echogram.data_format == 'zarr':
            reader_shape = (reader_shape[1], reader_shape[0])

        # Get seabed
        seabed_x_left = max(x_left, 0)
        seabed_y_upper = max(y_upper, 0)
        seabed_x_right = min(x_right, reader_shape[1])
        seabed_y_lower = min(y_lower, reader_shape[0])


        seabed_mask = echogram.get_seabed_mask(seabed_x_left, seabed_x_right-seabed_x_left,
                                               seabed_y_upper, seabed_y_lower-seabed_y_upper,
                                               seabed_pad=10).astype(np.int8)   # Move seabed 10 pixels lower to be conservative
        if echogram.data_format == 'zarr':
            seabed_mask = seabed_mask.T

        seabed_mask_padded = np.zeros_like(labels)

        x_diff = seabed_x_left-x_left
        y_diff = seabed_y_upper-y_upper
        seabed_mask_padded[y_diff:seabed_mask.shape[0]+y_diff,
                            x_diff:seabed_mask.shape[1]+x_diff] = seabed_mask

        # Label boundary ignore val takes precedence over seabed mask
        # Sometimes fish annotations appear below seabed, they also take precedence over seabed mask
        mask = (seabed_mask_padded.astype(bool)) & (labels == BACKGROUND)
        labels[mask] = LABEL_SEABED_MASK_VAL

        return data, labels, center_coord, echogram



