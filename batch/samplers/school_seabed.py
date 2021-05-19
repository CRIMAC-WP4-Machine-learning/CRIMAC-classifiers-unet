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

from batch.samplers.school import School


class SchoolSeabed():
    def __init__(self, echograms, max_dist_to_seabed, fish_type='all'):
        """

        :param echograms: A list of all echograms in set
        """
        self.echograms = echograms

        #Get Schools:
        self.Schools = School(echograms, fish_type).Schools

        #Remove Schools that are not close to seabed
        self.Schools = \
            [(e, o) for e, o in self.Schools if
             np.abs(e.get_seabed()[int((o['bounding_box'][2] + o['bounding_box'][3]) / 2)] - o['bounding_box'][1]) <
             max_dist_to_seabed]

    def get_sample(self):
        """

        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        #Random object

        oi = np.random.randint(len(self.Schools))
        e,o  = self.Schools[oi]

        #Random pixel in object
        pi = np.random.randint(o['n_pixels'])
        y,x = o['indexes'][pi,:]

        #Todo: Call get_sample again if window does not contain seabed

        return [y,x], e


class SchoolSeabedZarr():
    def __init__(self, zarr_files, max_dist_to_seabed=10, window_size=(256, 256), fish_type='all'):
        self.zarr_files = zarr_files
        self.window_size = window_size

        self.school_coords = np.empty((0, 3), np.int32)
        for idx, zarr_file in enumerate(self.zarr_files):
            objects = zarr_file.objects

            if fish_type == 'all':
                school_coords_year = np.argwhere(~np.isnan(objects.fish_type_index.values))
            else:
                if type(fish_type) == int:
                    fish_type = [fish_type]
                school_coords_year = np.argwhere(
                    ~np.isnan(objects.fish_type_index.where(objects.fish_type_index.isin(fish_type)).values))

            # ignore schools not close to seabed
            # TODO cleanup lines here
            _objects = objects.bounding_box[:, xr.DataArray(school_coords_year[:, 0]),
                       xr.DataArray(school_coords_year[:, 1])]
            close = (np.abs(zarr_file.get_seabed()[
                                ((_objects[2, :] + _objects[3, :]) / 2).astype(np.uint32)] - _objects[1, :])
                     < max_dist_to_seabed)
            school_coords_year = school_coords_year[close]
            school_coords_year = np.hstack((school_coords_year, np.ones((len(school_coords_year), 1)) * idx)).astype(
                np.int32)
            self.school_coords = np.append(self.school_coords, school_coords_year, 0)

        self.nr_schools = np.shape(self.school_coords)[0]

    def get_sample(self):
        # Get random school
        rand_idx = np.random.randint(self.nr_schools)

        # Get school location in objects file
        obj_len_idx, raw_file_idx, zarr_idx = self.school_coords[rand_idx, :]

        # Get bounding box of random school
        zarr_file = self.zarr_files[zarr_idx]
        obj_box = zarr_file.objects.bounding_box[:, obj_len_idx, raw_file_idx]

        # Get random x, y in bounding box
        x = np.random.randint(obj_box[2], obj_box[3]) if obj_box[2] < obj_box[3] else int(obj_box[2].values)
        y = np.random.randint(obj_box[0], obj_box[1]) if obj_box[0] < obj_box[1] else int(obj_box[0].values)

        # Add start idx of raw_file to get x relative to entire zarr-file (not just raw file)
        start_idx = zarr_file.get_rawfile_start_idx()[
            np.argwhere(zarr_file.raw_file_included == obj_box.raw_file.values).squeeze()]
        if start_idx.size == 0:
            return self.get_sample()

        x += start_idx

        # Adjust coordinate by random shift in y and x direction
        x += np.random.randint(-self.window_size[0] // 2, self.window_size[0] // 2 + 1)
        y += np.random.randint(-self.window_size[1] // 2, self.window_size[1] // 2 + 1)

        return [x, y], zarr_file
