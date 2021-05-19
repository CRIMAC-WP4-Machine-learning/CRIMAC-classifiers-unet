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

def remove_nan_inf(data, labels, echogram, frequencies, new_value=0.0):
    '''
    Reassigns all non-finite data values (nan, positive inf, negative inf) to new_value.
    :param data:
    :param labels:
    :param echogram:
    :param new_value:
    :return:
    '''
    data[np.invert(np.isfinite(data))] = new_value
    return data, labels, echogram, frequencies