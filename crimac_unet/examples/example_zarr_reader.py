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

import os
import sys
sys.path.append("../data")

from data.data_reader import DataReaderZarr

if __name__ == '__main__':
    data_path = ""

    assert os.path.isdir(data_path)

    # Load zarr data using the DataReader class
    reader = DataReaderZarr(data_path)

    # Print some information
    print(f"Included fish categories: {reader.fish_categories}")

    objects_df = reader.objects_df
    for cat in reader.fish_categories:
        # Get nr of schools in each fish category
        n_schools = len(objects_df.loc[objects_df.category == cat])
        print(f" Nr of schools in category {cat} = {n_schools}")

    print(f"\nNumber of pings: {reader.shape[0]}")
    print(f"Survey start time: {reader.time_vector[0].values}")
    print(f"Survey end time: {reader.time_vector[-1].values}")

    # Visualize 2000 random pings in the data
    reader.visualize(range_idx=50, draw_seabed=False)

    # Visualize data with fish school
    row = objects_df.query('(category == 27)').sample(n=1) # Sample a random sandeel fish school

    # extranct the bounding box
    start_ping_idx = int(row.startpingindex)
    end_ping_idx = int(row.endpingindex)
    start_range_idx = int(row.upperdeptindex)
    end_range_idx = int(row.lowerdeptindex)


    reader.visualize(ping_idx=start_ping_idx - 50,
                     n_pings=end_ping_idx - start_ping_idx + 1000,
                     range_idx=max(start_range_idx - 100, 0),
                     n_range=end_range_idx - start_range_idx + 100,
                     draw_seabed=False)



