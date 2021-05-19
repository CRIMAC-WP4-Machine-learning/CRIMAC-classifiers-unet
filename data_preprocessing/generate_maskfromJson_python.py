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
import numpy as np
import json
import h5py
from datetime import datetime, timedelta

from paths import path_to_korona_data
from paths import path_to_korona_transducer_depths


def timevector_to_datetime(time_vector):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = time_vector % 1
    return datetime.fromordinal(int(time_vector)) + timedelta(days=days) - timedelta(days=366)


def lsss_time_to_datetime(time_vector):
    try:
        return datetime.strptime(time_vector, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        return datetime.strptime(time_vector, '%Y-%m-%dT%H:%M:%SZ')


def get_korona_list_from_json(path_json_korona):

    with open(path_json_korona, "r") as file:
        korona = json.load(file)
    for ping in korona:
        ping['time'] = lsss_time_to_datetime(ping['time'])

    # Check that korona list is ascending in 'time' and 'pingNumber'
    for i in range(len(korona) - 1):
        assert korona[i]['time'] <= korona[i + 1]['time'],\
            'file: {}, i: {}, time (i, i+1): {}, {}, ping number (i, i+1): {}, {}'.format(path_json_korona, i, korona[i]['time'], korona[i+1]['time'], korona[i]['pingNumber'], korona[i+1]['pingNumber'])
        assert korona[i]['pingNumber'] <= korona[i + 1]['pingNumber'],\
            'file: {}, i: {}, time (i, i+1): {}, {}, ping number (i, i+1): {}, {}'.format(path_json_korona, i, korona[i]['time'], korona[i+1]['time'], korona[i]['pingNumber'], korona[i+1]['pingNumber'])

    return korona


def get_transducer_depths(echogram):

    root_depths = path_to_korona_transducer_depths()
    file_path = root_depths + echogram.name + '.h5'
    assert os.path.isfile(file_path), file_path + ' does not exist.'

    with h5py.File(file_path, 'r') as f:
        key = list(f['transducer'])[0]
        data = np.array(f['transducer'][key])

    return data


def compensate_heave(echogram, labels_korona_initial):

    # Get vertical pixel resolution
    r = echogram.range_vector
    r_diff = np.median(r[1:] - r[:-1])

    # Convert heave value from meters to number of pixels
    heave = np.round(echogram.heave / r_diff).astype(np.int)
    assert heave.size == echogram.shape[1]

    # Create new labels: Move each labels column up/down corresponding to heave
    labels_out = np.zeros_like(labels_korona_initial)
    for x, h in enumerate(list(heave)):
        if h == 0:
            labels_out[:, x] = labels_korona_initial[:, x]
        elif h > 0:
            labels_out[:-h, x] = labels_korona_initial[h:, x]
        else:
            labels_out[-h:, x] = labels_korona_initial[:h, x]

    return labels_out


def get_korona_labels(echogram, korona_list):

    range_vector = echogram.range_vector
    time_vector = echogram.time_vector

    assert echogram.year not in [2006, 2019]

    assert (time_vector.ndim == 1) and (range_vector.ndim == 1)
    time_vector = np.array([timevector_to_datetime(t) for t in time_vector])

    shape_ech = echogram.shape
    if range_vector.shape[0] != shape_ech[0]:
        print('WARNING: range_vector length (', range_vector.shape[0],
              ') not equal to echogram shape at dim 0 (', shape_ech[0],
              '), off by', shape_ech[0] - range_vector.shape[0], ', in ', echogram.name)
    if time_vector.shape[0] != shape_ech[1]:
        print('WARNING: time_vector length (', time_vector.shape[0],
              ') not equal to echogram shape at dim 1 (', shape_ech[1],
              '), off by', shape_ech[1] - time_vector.shape[0], '.')

    time_start = time_vector[0]
    time_stop = time_vector[-1]
    ping_list_ech = [ping for ping in korona_list if time_start <= ping['time'] <= time_stop]

    labels_korona = np.zeros(shape=shape_ech, dtype=np.int16)

    transducer_depths = get_transducer_depths(echogram)

    for ping in ping_list_ech:
        n_x = np.argmax(ping['time'] <= time_vector)
        assert 0 <= n_x < shape_ech[1], 'n_x: ' + str(n_x) + ' ech.shape[1]: ' + str(shape_ech[1]) + ' ech.name: ' + str(echogram.name)

        for minmax in ping['depthRanges']:

            assert minmax['min'] >= transducer_depths[n_x, 0], 'ping_min: ' + str(minmax['min']) + ' transducer depth: ' + str(transducer_depths[n_x, 0])
            n_y_min = np.argmax(minmax['min'] - transducer_depths[n_x, 0] < range_vector)
            n_y_max = np.argmax(minmax['max'] - transducer_depths[n_x, 0] < range_vector)

            assert n_y_min >= 0, str(minmax) + '\n' + str(range_vector)
            assert n_y_min >= 0, 'n_y_min: ' + str(n_y_min) + ' ech.name: ' + str(echogram.name)
            assert n_y_max <= shape_ech[0], 'n_y_max: ' + str(n_y_max) + ' ech.shape[0]: ' + str(shape_ech[0]) + ' ech.name: ' + str(echogram.name)
            labels_korona[n_y_min:n_y_max, n_x] = 1

    # Compensate labels for heave
    labels_korona_out = compensate_heave(echogram=echogram, labels_korona_initial=labels_korona)

    return labels_korona_out


if __name__ == '__main__':

    ### Example of use ###

    from random import shuffle
    from data.echogram import get_echograms
    from batch.label_transforms.convert_label_indexing import convert_label_indexing

    year = 2010
    root_json_korona = path_to_korona_data()
    path_json_korona = root_json_korona + "korona_" + str(year) + ".json"
    korona_list = get_korona_list_from_json(path_json_korona=path_json_korona)

    freqs = [18, 38, 120, 200]
    echograms_year = get_echograms(years=year, frequencies=freqs, minimum_shape=256)
    shuffle(echograms_year)

    for i, ech in enumerate(echograms_year):

        print(ech.name)
        labels_original = convert_label_indexing(None, ech.label_numpy(), None)[1]
        labels_korona = get_korona_labels(echogram=ech, korona_list=korona_list)
        ech.visualize(frequencies=freqs, labels_original=labels_original, labels_refined=labels_korona, show_labels_str=False)
