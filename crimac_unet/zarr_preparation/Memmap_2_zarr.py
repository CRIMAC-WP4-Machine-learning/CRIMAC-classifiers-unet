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

import sys
import os
import pickle

import numpy as np
import xarray as xr
import os.path
import glob
from datetime import datetime, timedelta
import time
from numcodecs import Blosc
import paths
import shutil

def load_meta(folder, name):
    with open(os.path.join(folder, name) + '.pkl', 'rb') as f:
        f.seek(0)
        return pickle.load(f, encoding='latin1')


def timevector_to_datetime(time_vector):
    days = time_vector % 1
    return datetime.fromordinal(int(time_vector)) + timedelta(days=days) - timedelta(days=366)


def create_ds(path_to_echogram):
    shape = load_meta(path_to_echogram, 'shape')
    ping_time = load_meta(path_to_echogram, 'time_vector')[0, :]
    ping_time = np.array(
        [np.datetime64(timevector_to_datetime(ping_time[i])) for i in range(len(ping_time))])  # Convert to datetime64
    #ping_time_start = np.datetime64(timevector_to_datetime(load_meta(path_to_echogram, 'time_vector')[0, 0])).reshape(
    #    -1, )
    #ping_time_end = np.datetime64(timevector_to_datetime(load_meta(path_to_echogram, 'time_vector')[0, -1])).reshape(
    #    -1, )
    da_range = load_meta(path_to_echogram, 'range_vector')[:, 0]
    heave = load_meta(path_to_echogram, 'heave')[0, :]
    if not os.path.isfile(path_to_echogram + 'labels_heave.dat'):
        prefix = path_to_echogram.split('/')[-2].split('-')[0]
        # Open the file with list containing echograms with missing labels_heave.data (access mode 'a')
        with open(paths.path_to_zarr_files() + prefix + '_echos_missing_labels_heave.txt', 'a') as f:
            f.write(path_to_echogram.split('/')[-2] + '\n')
        print('WARNING-> labels_heave.dat does not exist, ignoring the echogram')
        return None
    else:
        labels_heave = np.memmap(path_to_echogram + 'labels_heave.dat', dtype='uint16', mode='r', shape=shape)
        labels = np.memmap(path_to_echogram + 'labels.dat', dtype='uint16', mode='r', shape=shape)

        seabed = np.load(path_to_echogram + 'seabed.npy')
        depths = load_meta(path_to_echogram, 'depths')
        frequencies = load_meta(path_to_echogram, 'frequencies')[0, :]
        da_sv = [np.memmap(path_to_echogram + 'data_for_freq_' + str(int(f)) + '.dat', dtype='float32', mode='r',
                           shape=shape) for f in frequencies]
        da_sv = [np.array(d[:]) for d in da_sv]
        da_sv = [np.expand_dims(d, 0) for d in da_sv]  # Add channel dimension
        da_sv = np.concatenate(da_sv, 0)

        # Create xarray dataset
        ds = xr.Dataset({
            'heave': xr.DataArray(
                data=heave,  # enter data here
                dims=['ping_time'],
                coords={'ping_time': ping_time, },
            ),

            'seabed': xr.DataArray(
                data=seabed,  # enter data here
                dims=['ping_time'],
                coords={'ping_time': ping_time, },
            ),

            'sv': xr.DataArray(
                data=da_sv.transpose(0, 2, 1),  # enter data here
                dims=['frequency', 'ping_time', 'range'],
                coords={'frequency': frequencies,
                        'ping_time': ping_time,
                        'range': da_range
                        },
            ),

            'labels': xr.DataArray(
                data=labels.transpose(1, 0),  # enter data here
                dims=['ping_time', 'range'],
                coords={
                    'ping_time': ping_time,
                    'range': da_range
                },
            ),

            'labels_heave': xr.DataArray(
                data=labels_heave.transpose(1, 0),  # enter data here
                dims=['ping_time', 'range'],
                coords={
                    'ping_time': ping_time,
                    'range': da_range
                },
            ),

            'depths': xr.DataArray(
                data=depths.transpose(1, 0),  # enter data here
                dims=['frequency', 'ping_time'],
                coords={'frequency': frequencies,
                        'ping_time': ping_time,
                        },
            )
        },
            attrs={'description': 'memmap converted to zarr'}
        )

        ds.coords["raw_file"] = ("ping_time", [path_to_echogram.split('/')[-2]] * len(ds.ping_time))

        return ds


def create_ds_objects(path_to_echogram):
    objects = load_meta(path_to_echogram, 'objects')
    echo_name = path_to_echogram.split('/')[-2]

    if len(objects) == 0:
        prefix = path_to_echogram.split('/')[-2].split('-')[0]
        # Open the file with list containing echograms with empty objects (access mode 'a')
        with open(paths.path_to_zarr_files() + prefix + '_echos_empty_objects.txt', 'a') as f:
            f.write(echo_name + '\n')
        print('WARNING-> objects.pickle is empty, ignoring the echogram')
        return None
    else:
        fish_type_list = [xr.DataArray(
            data=np.array(objects[i]['fish_type_index']).reshape(-1),  # enter data here
            dims=['raw_file'],
            coords={'raw_file': [echo_name]},
        ) for i in range(len(objects))]

        n_pixels_list = [xr.DataArray(
            data=np.array(objects[i]['n_pixels']).reshape(-1),  # enter data here
            dims=['raw_file'],
            coords={'raw_file': [echo_name]},
        ) for i in range(len(objects))]

        labeled_as_segmentation_list = [xr.DataArray(
            data=np.array(objects[i]['labeled_as_segmentation']).reshape(-1),  # enter data here
            dims=['raw_file'],
            coords={'raw_file': [echo_name]},
        ) for i in range(len(objects))]

        bounding_box_list = [xr.DataArray(
            data=np.array(objects[i]['bounding_box']).reshape(-1, 1),  # enter data here
            dims=['bounding_box_els', 'raw_file'],
            coords={'bounding_box_els': np.arange(0, 4),
                    'raw_file': [echo_name]
                    },
        ) for i in range(len(objects))]

        da_fish_type = xr.concat(fish_type_list, dim='object_length')
        da_n_pixels = xr.concat(n_pixels_list, dim='object_length')
        da_labeled_as_segmentation = xr.concat(labeled_as_segmentation_list, dim='object_length')
        da_bounding_box = xr.DataArray(data=xr.concat(bounding_box_list, dim='object_length').values.transpose(1, 0, 2),
                                       # enter data here
                                       dims=['bounding_box_els', 'object_length', 'raw_file'],
                                       coords={'bounding_box_els': np.arange(0, 4),
                                               'object_length': np.arange(0, len(objects)),
                                               'raw_file': [echo_name]
                                               }
                                       )
        ds_obj = xr.Dataset({'fish_type_index': da_fish_type,
                             'n_pixels': da_n_pixels,
                             'bounding_box': da_bounding_box,
                             'labeled_as_segmentation': da_labeled_as_segmentation
                             })

        return ds_obj


if __name__ == '__main__':
    years = ['2007', '2008', '2009', '2010','2011', '2013', '2014',
             '2015', '2016', '2017','2018']
    year = '2018'
    selected_echos = np.sort(glob.glob('/lokal_uten_backup/pro/COGMAR/acoustic_new_5/memmap/' + year + '*'))
    prefix = np.unique([selected_echos[i].split('/')[-1].split('-')[0] for i in range(len(selected_echos))])

    assert len(prefix) == 1

    path_to_zarr = paths.path_to_zarr_files()
    create_ds_bool = True
    create_ds_object_bool = True

    # Save ds containing anything else which is not object in zarr
    if create_ds_bool:
        print('###############################################')
        print('Save zarr containing everything except objects')
        print('###############################################')

        name_zarr = prefix[0]
        range_vector_max = load_meta(
            selected_echos[np.argmax([load_meta(path_to_echogram, 'shape')[0] for path_to_echogram in selected_echos])],
            'range_vector')[:, 0]
        start = time.time()
        write_first_loop = True

        # Delete file with list containing echograms with missing labels_heave.data if it exists
        file_missing_labels_heave = paths.path_to_zarr_files() + prefix[0] + '_echos_missing_labels_heave.txt'
        if os.path.isfile(file_missing_labels_heave):
            os.remove(file_missing_labels_heave)

        # Delete existing zarr dir
        target_fname = path_to_zarr + name_zarr + '.zarr'
        if os.path.isdir(target_fname):
            shutil.rmtree(target_fname)

        for (ii, path_to_echogram) in enumerate(selected_echos[:]):
            echo = path_to_echogram.split('/')[-1]
            path_to_echogram = path_to_echogram + '/'
            print("[Step %d/%d - echo %s]" % (ii, len(selected_echos), echo))
            ds = create_ds(path_to_echogram)


            if ds is not None:
                compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
                encoding = {var: {"compressor": compressor} for var in ds.data_vars}
                if write_first_loop == False:
                    ds.reindex({"range": range_vector_max}).to_zarr(target_fname, append_dim="ping_time")
                else:
                    ds.reindex({"range": range_vector_max}).to_zarr(target_fname, mode="w", encoding=encoding)

                write_first_loop = False

        print(
            f'Executed time to create and save {ii + 1} echograms for one survey to zarr (s): {np.round(time.time() - start, 3)}')

    # Save ds object in zarr
    if create_ds_object_bool:

        print('###############################################')
        print('Save zarr containing only objects')
        print('###############################################')

        name_zarr = prefix[0] + '_obj'
        obj_len_list = [len(load_meta(path_to_echogram, 'objects')) for path_to_echogram in selected_echos]

        start = time.time()
        write_first_loop = True

        # Delete file with list containing echograms with missing empty objects if it exists
        file_empty_objects = paths.path_to_zarr_files() + prefix[0] + '_echos_empty_objects.txt'
        if os.path.isfile(file_empty_objects):
            os.remove(file_empty_objects)

        # Delete existing zarr dir
        target_fname = path_to_zarr + name_zarr + '.zarr'
        if os.path.isdir(target_fname):
            shutil.rmtree(target_fname)

        for (ii, path_to_echogram) in enumerate(selected_echos[:]):
            echo = path_to_echogram.split('/')[-1]
            path_to_echogram = path_to_echogram + '/'
            print("[Step %d/%d - echo %s]" % (ii, len(selected_echos), echo))
            ds_obj = create_ds_objects(path_to_echogram)

            if ds_obj is not None:
                compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
                encoding = {var: {"compressor": compressor} for var in ds_obj.data_vars}

                if write_first_loop == False:
                    ds_obj.reindex({"object_length": np.arange(0, np.max(obj_len_list))}).to_zarr(target_fname,
                                                                                                  append_dim="raw_file")
                else:
                    ds_obj.reindex({"object_length": np.arange(0, np.max(obj_len_list))}).to_zarr(target_fname,
                                                                                                  mode="w",
                                                                                                  encoding=encoding)
                    # Propagate range to the rest of the files

                write_first_loop = False

        print(
            f'Executed time to create and save {ii + 1} objects of echograms for one survey to zarr (s): {np.round(time.time() - start, 3)}')

