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
class mask_label_overlap():
    """
    Set label regions overlapping with neighboring patches to LABEL_BOUNDARY_VAL
    """
    def __init__(self, overlap=20):
        # The overlap
        self.overlap = overlap

    def __call__(self, data, labels, center_coord, echogram):
        """
        Ignore all predictions below seabed
        :return:
        """

        # If no overlap is specified, no transform is necessary
        if self.overlap == 0:
            return data, labels, center_coord, echogram

        out_labels = np.ones_like(labels) * LABEL_OVERLAP_VAL
        out_labels[self.overlap:-self.overlap, self.overlap:-self.overlap] = labels[self.overlap:-self.overlap,
                                                                             self.overlap:-self.overlap]

        # Label boundary value should take precedence over label overlap val
        out_labels[labels == LABEL_BOUNDARY_VAL] = LABEL_BOUNDARY_VAL

        return data, out_labels, center_coord, echogram
