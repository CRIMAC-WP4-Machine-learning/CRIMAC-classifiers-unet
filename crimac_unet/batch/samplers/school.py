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
from utils.np import random_point_containing

class School():
    def __init__(self, echograms, window_size, fish_type='all'):
        """
        :param echograms: A list of all echograms in set
        :window_size: (tuple), [height, width]
        """

        self.echograms = echograms
        self.window_size = window_size
        self.fish_type = fish_type

        self.Schools = []

        # Remove echograms without fish
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
        # Random object
        i = np.random.randint(len(self.Schools))
        ech, obj = self.Schools[i]

        # Random pixel in object
        random_pixel_idx = np.random.randint(obj['n_pixels'])
        y, x = obj['indexes'][random_pixel_idx, :]

        # If window width is greater than echogram width, set x = echogram senter
        x = random_point_containing(ech.shape[1], self.window_size[1], x)
        y = random_point_containing(ech.shape[0], self.window_size[0], y)

        return [y, x], ech


class SchoolZarr():
    def __init__(self, zarr_files, window_size, fish_type='all'):
        self.zarr_files = zarr_files
        self.window_size = window_size

        self.schools = []
        self.n_schools = 0

        # For each survey
        for idx, zarr_file in enumerate(self.zarr_files):
            # Get dataframe with bboxes of fish schools
            df = zarr_file.get_fish_schools(category=fish_type)

            # Extract bounding box indexes and append to array
            bboxes = df[['startpingindex', 'endpingindex', 'upperdepthindex', 'lowerdepthindex']].values

            self.schools.append((zarr_file, bboxes))
            self.n_schools += bboxes.shape[0]

    def get_sample(self):
        # get random zarr file
        zarr_file_idx = np.random.randint(len(self.schools))
        zarr_file, bboxes = self.schools[zarr_file_idx]

        # get random bbox
        bbox = bboxes[np.random.randint(bboxes.shape[0])]

        # get random x, y value from bounding box. If bbox has width/height == 1, add 1 to avoid index error
        if bbox[0] == bbox[1]:
            bbox[1] += 1
        if bbox[2] == bbox[3]:
            bbox[3] += 1

        x = np.random.randint(bbox[0], bbox[1])  # ping dimension
        y = np.random.randint(bbox[2], bbox[3])  # range dimension

        # Adjust coordinate by random shift in y and x direction, ensures school is not always in the middle of the crop
        x += np.random.randint(-self.window_size[1]//2, self.window_size[1]//2 + 1)
        y += np.random.randint(-self.window_size[0]//2, self.window_size[0]//2 + 1)

        return [y, x], zarr_file

