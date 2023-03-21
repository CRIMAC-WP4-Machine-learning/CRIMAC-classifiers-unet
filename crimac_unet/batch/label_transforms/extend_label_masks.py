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


def overlap(a, b):
    # returns True if rectangles intersect, false if they don't
    dy = min(a[1], b[1]) - max(a[0], b[0])
    dx = min(a[3], b[3]) - max(a[2], b[2])

    if (dx >= 0) and (dy >= 0):
        return True
    else:
        return False


# TODO figure out border vals
# TODO ignore val
class get_extended_label_mask_for_crop():
    """
    Mark background areas as 'ignore' if they are not close to a fish region, as these areas may
    not be annotated
    """

    def __init__(self, extend_size=20, ignore_val=-1, mask_type='region'):
        """
        :param extend_size: Nr of pixels to include around fish regions
        :param ignore_val: Fill ignore regions with this variable
        :param mask_type: Either 'region' or 'trace'. 'region' returns a mask where all pixels except a bounding box
        around fish schools are set to ignore. 'trace' returns a mask where all pings except those within
        'extend_size' distance of a ping containing a fish school are set to ignore.
        """
        self.mask_type = mask_type
        self.ignore_val = ignore_val
        self.extend_size = extend_size

        assert self.mask_type in ['region', 'trace', 'all'], print(f"Uknown mask_type {self.mask_type}! Must be 'all', 'region' or"
                                                            f"'trace'")

    def __call__(self, data, labels, center_coord, echogram):
        if self.mask_type == 'all':
            return data, labels, center_coord, echogram


        # TODO check for zarr, write comments
        fish_types = [1, 27]
        # TODO filter on fish types!
        y_upper_left, x_upper_left = np.array(center_coord) - np.array(labels.shape) // 2

        # Get all bounding boxes
        bboxes = echogram.get_object_bounding_boxes()

        # For region mode, bbox is extended similarly in all directions
        if self.mask_type == 'region':
            bboxes[:, 0] += -self.extend_size
            bboxes[:, 1] += self.extend_size
        # For trace mode, every ping containing fish is relevant -> Bbox extended through entire water column
        else:
            bboxes[:, 0] = 0
            bboxes[:, 1] = echogram.shape[0]

        bboxes[:, 2] += -self.extend_size
        bboxes[:, 3] += self.extend_size

        # Initialize output label array with only ignore values
        label_crop_bbox = [y_upper_left, y_upper_left + labels.shape[0],
                           x_upper_left, x_upper_left + labels.shape[1]]
        out_labels = np.ones_like(labels) * self.ignore_val

        # Fill relevant areas in label based on fish school bounding boxes
        for bbox in bboxes:
            if overlap(bbox, label_crop_bbox):
                y_upper_left_mask = max(bbox[0] - y_upper_left, 0)
                x_upper_left_mask = max(bbox[2] - x_upper_left, 0)
                y_lower_right_mask = min(bbox[1] - y_upper_left, labels.shape[0])
                x_lower_right_mask = min(bbox[3] - x_upper_left, labels.shape[1])

                out_labels[y_upper_left_mask:y_lower_right_mask, x_upper_left_mask:x_lower_right_mask] = \
                    labels[y_upper_left_mask:y_lower_right_mask, x_upper_left_mask:x_lower_right_mask]

        # Return all function inputs as output may be input to other label transform functions
        return data, out_labels, center_coord, echogram
