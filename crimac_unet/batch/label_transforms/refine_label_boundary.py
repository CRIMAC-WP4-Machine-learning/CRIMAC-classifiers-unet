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
from scipy.ndimage.morphology import binary_closing
from constants import *


class refine_label_boundary():
    def __init__(self,
                 frequencies=[18, 38, 120, 200],
                 threshold_freq=200,
                 threshold_val=[1e-7, 1e-4],
                 ignore_zero_inside_bbox=True
                 ):
        self.frequencies = frequencies
        self.threshold_freq = threshold_freq
        self.threshold_val = threshold_val
        self.ignore_zero_inside_bbox=ignore_zero_inside_bbox

    def __call__(self, data, labels, center_coord, echogram):
        '''
        Refine existing labels based on thresholding with respect to pixel values in image.
        Low intensity areas near schools are set to LABEL_REFINE_BOUNDARY_VAL
        :param data: (numpy.array) Image (C, H, W)
        :param labels: (numpy.array) Labels corresponding to image (H, W)
        :param echogram: (Echogram object) Echogram
        :param threshold_freq: (int) Image frequency channel that is used for thresholding
        :param threshold_val: (float) Threshold value that is applied to image for assigning new labels
        :param ignore_val: (int) Ignore value (specific label value) instructs loss function not to compute gradients for these pixels
        :param ignore_zero_inside_bbox: (bool) labels==1 that is relabeled to 0 are set to ignore_value if True, 0 if False
        :return: data, new_labels, echogram
        '''

        closing = np.array([
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0]
        ])

        # if self.ignore_val == None:
        #     self.ignore_val = 0

        # Set new label for all pixels inside bounding box that are below threshold value
        if self.ignore_zero_inside_bbox:
            label_below_threshold = LABEL_REFINE_BOUNDARY_VAL
        else:
            label_below_threshold = 0

        # Get refined label masks
        freq_idx = self.frequencies.index(self.threshold_freq)

        # Relabel
        new_labels = labels.copy()

        # Relevant labels -> this is to mimic previous version
        # NB this matches previous version but may not be the best idea. Causes strange gaps

        # TODO There is a rare bug where all labels are set = LABEL_BOUNDARY_VAL, which caused the
        # training to crash. Was unable to recreate it by drawing 160k random samples. The check
        # prevents the crash
        idxs = np.argwhere(new_labels != LABEL_BOUNDARY_VAL)
        if len(idxs) == 0:
            print(f"WARNING - patch with center coordinate {center_coord} for {echogram.name} is outside data boundary")
            return data, new_labels, center_coord, echogram
        else:
            crop_y0 = np.min(idxs[:, 0])
            crop_y1 = np.max(idxs[:, 0]) + 1
            crop_x0 = np.min(idxs[:, 1])
            crop_x1 = np.max(idxs[:, 1]) + 1
            relevant_labels = new_labels[crop_y0:crop_y1, crop_x0:crop_x1]

            # TODO will labels > 0 do the same trick?
            mask_threshold = (labels > 0) & (data[freq_idx, :, :] > self.threshold_val[0]) & (
                        data[freq_idx, :, :] < self.threshold_val[1])

            mask_threshold_closed = binary_closing(mask_threshold[crop_y0:crop_y1, crop_x0:crop_x1], structure=closing)

            mask = np.zeros_like(new_labels).astype(bool)
            mask[crop_y0:crop_y1, crop_x0:crop_x1] = (mask_threshold_closed == 0) & (relevant_labels > 0)

            new_labels[mask] = label_below_threshold
            new_labels[labels == LABEL_IGNORE_VAL] = LABEL_IGNORE_VAL

            return data, new_labels, center_coord, echogram
