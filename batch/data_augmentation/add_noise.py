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

def add_noise(data, labels, echogram):

    # Apply random noise to crop with probability p = 0.5
    if np.random.randint(2):

        # Change pixel value in 5% of the pixels
        change_pixel_value = np.random.binomial(1, 0.05, data.shape)

        # Pixels that are changed:
        # 50% are increased (multiplied by random number in [1, 10]
        # 50% are reduced (multiplied by random number in [0, 1]
        increase_pixel_value = np.random.binomial(1, 0.5, data.shape)

        data *= (1 - change_pixel_value) + \
                change_pixel_value * \
                (
                        increase_pixel_value * np.random.uniform(1, 10, data.shape) +
                        (1 - increase_pixel_value) * np.random.uniform(0, 1, data.shape)
                )

    return data, labels, echogram