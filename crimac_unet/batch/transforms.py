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

from batch.data_augmentation.flip_x_axis import flip_x_axis, flip_x_axis_metadata
from batch.data_augmentation.add_noise import add_noise, add_noise_metadata
from batch.data_transforms.remove_nan_inf import remove_nan_inf
from batch.data_transforms.db_with_limits import db_with_limits, db_with_limits_scaled
from batch.data_transforms.set_data_border_value import set_data_border_value
from batch.label_transforms.convert_label_indexing import convert_label_indexing, convert_label_indexing_unused_species
from batch.label_transforms.refine_label_boundary import refine_label_boundary
from batch.label_transforms.extend_label_masks import get_extended_label_mask_for_crop
from batch.label_transforms.mask_label_overlap import mask_label_overlap
from batch.label_transforms.mask_label_seabed import mask_label_seabed
from utils.combine_functions import CombineFunctions


def is_use_metadata(meta_channels):
    use_metadata = False
    if len(meta_channels) > 0:
        use_metadata = True
    return use_metadata


def define_data_augmentation(use_metadata=False):
    """Returns data augmentation functions to be applied when training"""
    if use_metadata:
        data_augmentation = CombineFunctions([add_noise_metadata, flip_x_axis_metadata])
    else:
        data_augmentation = CombineFunctions([add_noise, flip_x_axis])
    return data_augmentation


def define_data_transform(use_metadata=False):
    """Returns data transform functions to be applied when training"""
    if use_metadata:
        data_transform = CombineFunctions([remove_nan_inf, db_with_limits_scaled])
    else:
        data_transform = CombineFunctions([remove_nan_inf, db_with_limits])
    return data_transform

# TODO consider other border values
def define_data_transform_test(use_metadata=False):
    """Returns data transform functions to be applied when testing
    This corresponds to earlier versions of the codebase, where border value is set to 0.0"""
    if use_metadata:
        data_transform = CombineFunctions([remove_nan_inf, db_with_limits_scaled, set_data_border_value])
    else:
        data_transform = CombineFunctions([remove_nan_inf, db_with_limits, set_data_border_value])
    return data_transform



# TODO explain label masks, extend size
def define_label_transform_train(frequencies):
    """Returns label transform functions to be applied when training or testing """
    label_transform = CombineFunctions([
        refine_label_boundary(frequencies=frequencies, threshold_freq=frequencies[-1]),
        convert_label_indexing])

    return label_transform


def define_label_transform_test(frequencies, label_masks='all', extend_size=20, patch_overlap=0):
    label_transform_functions = [
        convert_label_indexing_unused_species, # Mark areas with unused species as well, ignored in training
        refine_label_boundary(frequencies=frequencies, threshold_freq=frequencies[-1]),
        mask_label_seabed(),    # Mark areas below seabed as ignore
        mask_label_overlap(overlap=patch_overlap)]  # When gridding, overlapping areas are marked ignore

    # We may choose to only evaluate on areas near annotations (eval mode = 'region' or 'trace')
    # The extend size indicates how big the crop is around the labels
    if label_masks in ['region', 'trace']:
        label_transform_functions.append(get_extended_label_mask_for_crop(mask_type=label_masks,
                                                                          extend_size=extend_size))
    label_transform = CombineFunctions(label_transform_functions)

    return label_transform