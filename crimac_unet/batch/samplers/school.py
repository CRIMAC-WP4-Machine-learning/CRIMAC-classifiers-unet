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
from data.echogram import get_zarr_files, get_echograms
import time

class School():
    def __init__(self, echograms, fish_type='all'):
        """

        :param echograms: A list of all echograms in set
        """

        self.echograms = echograms
        self.fish_type = fish_type

        self.Schools = []
        #Remove echograms without fish
        if self.fish_type == 'all':
            self.echograms = [e for e in self.echograms if len(e.objects)>0]
            for e in self.echograms:
                for o in e.objects:
                    self.Schools.append((e,o))

        elif type(self.fish_type) == int:
            self.echograms = [e for e in self.echograms if any([o['fish_type_index'] == self.fish_type for o in e.objects])]
            for e in self.echograms:
                for o in e.objects:
                    if o['fish_type_index'] == self.fish_type:
                        self.Schools.append((e,o))

        elif type(self.fish_type) == list:
            self.echograms = [e for e in self.echograms if any([o['fish_type_index'] in self.fish_type for o in e.objects])]
            for e in self.echograms:
                for o in e.objects:
                    if o['fish_type_index'] in self.fish_type:
                        self.Schools.append((e,o))

        else:
            class UnknownFishType(Exception):pass
            raise UnknownFishType('Should be int, list of ints or "all"')

        if len(self.echograms) == 0:
            class EmptyListOfEchograms(Exception):pass
            raise EmptyListOfEchograms('fish_type not found in any echograms')

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

        return [y,x], e


class SchoolZarr():
    def __init__(self, zarr_files, fish_type='all'):
        self.zarr_files = zarr_files

        # Initialize array with schools. Array shape is (N x 3), where N is the total number of schools in the zarr files,
        # For each school, object length index and raw file index of the school in the object file is saved, along with
        # the index of the corresponding zarr file in self.zarr_files
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

            school_coords_year = np.hstack((school_coords_year, np.ones((len(school_coords_year), 1)) * idx)).astype(
                np.int32)
            self.school_coords = np.append(self.school_coords, school_coords_year, 0)

        self.nr_schools = np.shape(self.school_coords)[0]

    def get_sample(self):
        # Get random school
        rand_idx = np.random.randint(self.nr_schools)

        # Get "coordinates" of the school in the objects file
        obj_len_idx, raw_file_idx, zarr_idx = self.school_coords[rand_idx, :]

        # Retrieve school bounding box
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

        return [x, y], zarr_file

if __name__ == '__main__':
    zarr_files = get_zarr_files()
    t0 = time.time()
    sz = SchoolZarr(zarr_files)
    print(time.time() - t0)
    print()

    #echograms = get_echograms(years=2017)
    #t0 = time.time()
    #s = School(echograms)
    #print(time.time() - t0)
#

