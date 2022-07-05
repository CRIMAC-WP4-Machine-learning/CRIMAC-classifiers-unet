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
from batch.samplers.school import School


class SchoolSeabed():
    def __init__(self, echograms, window_size, max_dist_to_seabed, fish_type='all'):
        """

        :param echograms: A list of all echograms in set
        """
        self.echograms = echograms
        self.window_size = window_size

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
        e,o = self.Schools[oi]

        #Random pixel in object
        pi = np.random.randint(o['n_pixels'])
        y,x = o['indexes'][pi,:]

        #Todo: Call get_sample again if window does not contain seabed

        # Adjust coordinate by random shift in y and x direction so that school is not always in the middle of the crop
        x += np.random.randint(-self.window_size[0]//2, self.window_size[0]//2 + 1)
        y += np.random.randint(-self.window_size[1]//2, self.window_size[1]//2 + 1)

        return [y,x], e


class SchoolSeabedZarr():
    def __init__(self, zarr_files, window_size, max_dist_to_seabed=20, fish_type='all'):
        self.zarr_files = zarr_files
        self.window_size = window_size

        self.schools = []
        for idx, zarr_file in enumerate(self.zarr_files):
            df = zarr_file.get_fish_schools(category=fish_type)

            # Filter on distance to seabed
            df = df.loc[df.distance_to_seabed < max_dist_to_seabed]
            bboxes = df[['startpingindex', 'endpingindex', 'upperdepthindex', 'lowerdepthindex']].values

            self.schools.append((zarr_file, bboxes))  # object id is not needed


    def get_sample(self):
        # get random zarr file
        zarr_file_idx = np.random.randint(len(self.schools))
        zarr_file, bboxes = self.schools[zarr_file_idx]

        # get random bbox
        bbox = bboxes[np.random.randint(bboxes.shape[0])]

        # get random x, y value from bounding box
        if bbox[0] == bbox[1]:
            bbox[1] += 1
        if bbox[2] == bbox[3]:
            bbox[2] += 1
        x = np.random.randint(bbox[0], bbox[1])  # ping dimension
        y = np.random.randint(bbox[2], bbox[3])  # range dimension

        # Adjust coordinate by random shift in y and x direction so that school is not always in the middle of the crop
        x += np.random.randint(-self.window_size[0]//2, self.window_size[0]//2 + 1)
        y += np.random.randint(-self.window_size[1]//2, self.window_size[1]//2 + 1)

        return [x, y], zarr_file
