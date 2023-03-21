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


def get_data_grid(reader, start_ping=None, end_ping=None, start_range=None, end_range=None,
                  patch_size=[256, 256], patch_overlap=20, mode='all', filter_below_seabed=True):
    """
    Get the center coordinates for a grid of data patches
    :param reader: Data reader
    :param patch_size: Size of the data patches
    :param patch_overlap: Nr of pixels of overlap between neighboring patches
    :param mode: Optionally ignore patches far away from fish schools
    :param max_depth: Maximum depth to consider (as the height of zarr files are much larger than the distance from
    surface to seabed)
    :return: list of center coordinates for the grid
    """
    # Get coordinates for data area
    start_ping, end_ping, start_range, end_range = get_start_end_coordinates(reader, start_ping, end_ping, start_range,
                                                                             end_range, filter_below_seabed)
    (patch_width, patch_height) = patch_size

    # Get grid with center point of all patches
    ys_upper_left = np.arange(start_range - (patch_overlap + 1), end_range - (patch_overlap + 1),
                              step=patch_height - 2 * patch_overlap)
    xs_upper_left = np.arange(start_ping - (patch_overlap + 1), end_ping - (patch_overlap + 1),
                              step=patch_width - 2 * patch_overlap)

    # Get center coordinates of all grid points
    ys_center = ys_upper_left + patch_height // 2
    xs_center = xs_upper_left + patch_width // 2

    # Return center coordinates of all patches in the grid
    if mode == 'all':
        # Get all combinations of the coordinates
        mesh = np.array(np.meshgrid(ys_center, xs_center)).T.reshape(-1, 2)

        return mesh

    # Otherwise, we're only interested in a subset of the patches containing fish
    elif mode == 'region' or mode == 'trace':
        xs_relevant = []
        ys_relevant = []
        if mode == 'trace':
            ys_relevant = ys_center  # We're interested in the entire water column

        for obj in reader.objects:
            y0, y1, x0, x1 = obj['bounding_box']

            # Get the closest patch(es) to the object in horizontal dimension
            closest_x0_index = np.abs(x0 - xs_center).argmin()
            closest_x1_index = np.abs(x1 - xs_center).argmin()
            x_coords = [xs_center[closest_x0_index]]
            if closest_x0_index != closest_x1_index:
                x_coords.append(xs_center[closest_x1_index])

            # For region mode, also find closest patches in vertical dimension
            if mode == 'region':
                closest_y0_index = np.abs(y0 - ys_center).argmin()
                closest_y1_index = np.abs(y1 - ys_center).argmin()
                y_coords = [ys_center[closest_y0_index]]
                if closest_y0_index != closest_y1_index:
                    y_coords.append(ys_center[closest_y1_index])

                mesh = np.meshgrid(y_coords, x_coords)
                coords = np.array(mesh).T.reshape(-1, 2)
                ys_relevant.extend(list(coords[:, 0]))
                xs_relevant.extend(list(coords[:, 1]))

            # For trace mode, the entire water column is pre-selected
            else:
                xs_relevant.extend(x_coords)
        if mode == 'trace':
            mesh = np.meshgrid(ys_relevant, np.unique(xs_relevant))
            coords = np.array(mesh).T.reshape(-1, 2)
        else:
            # TODO check for uniqueness
            coords = np.array([ys_relevant, xs_relevant]).T
        return coords


class Gridded:
    def __init__(self, echograms, window_size, patch_overlap=20, mode='all'):
        """
        :param echograms: A list of all echograms in set
        """
        self.echograms = echograms
        self.window_size = window_size

        all_coords = []
        for i, ech in enumerate(self.echograms):
            coords = get_data_grid(ech, window_size, patch_overlap, mode=mode)
            all_coords.append(np.concatenate((np.ones((len(coords), 1)) * i, coords), axis=1))  # add reader index

        self.coords_list = np.concatenate(all_coords, axis=0).astype(int)

    def __len__(self):
        return len(self.coords_list)

    def get_sample(self, i):
        ei, y, x = self.coords_list[i]
        return [y, x], self.echograms[ei]


def get_start_end_coordinates(reader, start_ping=None, end_ping=None, start_range=None, end_range=None,
                              filter_below_seabed=True):
    if reader.data_format == 'memmap':
        (H, W) = reader.shape
    elif reader.data_format == 'zarr':
        (W, H) = reader.shape
    else:
        print('Unknown reader data format')

    # set start and end ping
    if start_ping is None:
        start_ping = 0
    if end_ping is None:
        end_ping = W

    assert end_ping > start_ping

    # Set start range
    if start_range is None:
        start_range = 0

    # if no max depth is specified, set to data height
    if end_range is None:
        end_range = H

    # If data height is smaller than set max depth, set max depth to data height
    if H < end_range:
        end_range = H

    # Optionally find maximum seabed in the area and set max depth to this value
    if filter_below_seabed:
        # Get maximum seabed in the relevant area
        if reader.data_format == 'zarr':
            max_seabed = reader.get_seabed(start_ping, end_ping - start_ping, return_numpy=False).max().values + 50
        else:
            max_seabed = np.max(reader.get_seabed(start_ping, end_ping - start_ping)) + 50

        # Set range to maximum seabed
        end_range = max_seabed if max_seabed < end_range else end_range

    assert end_range > start_range

    return start_ping, end_ping, start_range, end_range
