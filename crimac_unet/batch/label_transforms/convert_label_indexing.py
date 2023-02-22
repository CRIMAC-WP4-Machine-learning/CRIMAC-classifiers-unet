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
import xarray as xr
from constants import *


def convert_label_indexing(data, labels, center_coord, echogram, ignore_val=-100):
    '''
    Re-assign labels to: Background==0, Sandeel==1, Other==2 - all remaining are set to ignore_value.
    '''

    new_labels = np.zeros(labels.shape)
    new_labels.fill(ignore_val)
    new_labels[labels == 0] = BACKGROUND
    new_labels[labels == 27] = SANDEEL
    new_labels[labels == 1] = OTHER

    return data, new_labels, center_coord, echogram

def convert_label_indexing_unused_species(data, labels, center_coord, echogram, ignore_val=-100):
    '''
    Re-assign labels to: Background==0, Sandeel==1, Other==2 - all remaining are set to ignore_value.
    '''

    _, new_labels, _, _ = convert_label_indexing(data, labels, center_coord, echogram)

    # Mark unused species
    new_labels[(labels > 0) & (labels != 1) & (labels != 27)] = LABEL_UNUSED_SPECIES

    return data, new_labels, center_coord, echogram

def convert_label_indexing_xr(data, labels, center_coord, echogram, ignore_val=-100):
    '''
    Re-assign labels to: Background==0, Sandeel==1, Other==2 - all remaining are set to ignore_value.
    '''

    new_labels = xr.ones_like(labels)*ignore_val
    new_labels = xr.where(new_labels == 0, BACKGROUND, new_labels)
    new_labels = xr.where(new_labels == 27, SANDEEL, new_labels)
    new_labels = xr.where(new_labels == 1, OTHER, new_labels)


    return data, new_labels, center_coord, echogram