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


def _save_memmap(data, path, dtype, force_write=False):
    path = (path + '.dat').replace('.dat.dat','.dat')
    if not os.path.isfile(path) or force_write:
        fp = np.memmap(path, dtype=dtype, mode='w+', shape=data.shape)
        fp[:] = data.astype(dtype)
        del fp


def write_label_file_without_heave_correction_one_echogram(echogram, force_write=False):
    '''
    For one echogram: Create a new label file 'labels_heave.dat' without heave corrections
     based on original labels file 'labels.dat'.
    NOTE: This function can also be used as a stand-alone function called from a data.Echogram object.
    :param echogram: (data.Echogram object) Echogram object
    :param force_write: (bool) If True, the generated file will be written to file even if the file already exists.
    :return: None
    '''

    # Get vertical pixel resolution
    r = echogram.range_vector
    r_diff = np.median(r[1:] - r[:-1])

    # Convert heave value from meters to number of pixels
    heave = np.round(echogram.heave / r_diff).astype(np.int)
    assert heave.size == echogram.shape[1]

    labels_old = echogram.label_numpy(heave=False)
    labels_new = np.zeros_like(labels_old)

    # Create new labels: Move each labels column (ping) up/down corresponding to heave
    for x, h in enumerate(list(heave)):
        if h == 0:
            labels_new[:, x] = labels_old[:, x]
        elif h > 0:
            labels_new[:-h, x] = labels_old[h:, x]
        else:
            labels_new[-h:, x] = labels_old[:h, x]

    # Save new labels as new memmap file
    write_path = os.path.join(echogram.path, 'labels_heave')
    _save_memmap(labels_new, write_path, dtype=labels_new.dtype, force_write=force_write)


def write_label_files_without_heave_correction_all_echograms(force_write=False):
    '''
    For all echograms: Create a new label file 'labels_heave.dat' without heave corrections
     based on original labels file 'labels.dat'.
    :param force_write: (bool) If True, the generated file will be written to file even if the file already exists.
    :return: None
    '''

    echs = get_echograms()
    for i, ech in enumerate(echs):

        if i % 100 == 0:
            print(len(echs), i)

        if not os.path.isfile(os.path.join(ech.path, 'labels_heave.dat')) or force_write:
            write_label_file_without_heave_correction_one_echogram(echogram=ech, force_write=force_write)


if __name__ == '__main__':

    ### The original labels received from IMR have already been corrected for heave.
    ### This is not the case for the echogram (backscatter) data, not corrected for heave.
    ### Thus the original labels and echogram data do not match, being a ping-wise vertical displacement between them.
    ### To match labels and echogram data, we revert the heave-correction for the labels,
    ### obtaining labels and echogram data that are both NOT corrected for heave.

    ### Uncomment line below and run this script:
    ### Will generate and write numpy.memmap label files without heave compensation for all echograms. ###
    ### The 'force_write' option set to True will overwrite any existing files.

    from data.echogram import get_echograms

    force_write = False
    #write_label_files_without_heave_correction_all_echograms(force_write)



