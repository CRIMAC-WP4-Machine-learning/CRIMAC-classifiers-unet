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


def get_data_split(valid_pings_ranges, max_n_pings=1000):
    """ Split data into smaller portions which can be preloaded """
    splits = []
    for start, end in valid_pings_ranges:
        n_splits = np.ceil((end - start) / max_n_pings)
        split_range = np.linspace(start, end, int(n_splits + 1)).astype(int)

        splits.extend([[split_range[i], split_range[i + 1]] for i in range(len(split_range) - 1)])
    return np.array(splits)
