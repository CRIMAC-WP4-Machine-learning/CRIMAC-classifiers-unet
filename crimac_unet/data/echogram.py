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

import collections
import os
import pickle
import numpy as np
import matplotlib.colors as mcolors
from scipy.signal import convolve2d as conv2d
import xarray as xr
import pandas as pd

import paths
from data.normalization import db
from data.missing_korona_depth_measurements import depth_excluded_echograms
from data_preprocessing.generate_heave_compensation_files import write_label_file_without_heave_correction_one_echogram

from tqdm import tqdm

#from utils.plotting import setup_matplotlib
#plt = setup_matplotlib()
# import matplotlib
# matplotlib.use('tkagg')
#
import matplotlib.pyplot as plt

class Echogram():
    """ Object to represent a echogram """

    def __init__(self, path):
        self.object_ids_with_label = {}  # Id (index) to echogram with the given label

        # Load meta data
        def load_meta(folder, name):
            with open(os.path.join(folder, name) + '.pkl', 'rb') as f:
                f.seek(0)
                return pickle.load(f, encoding='latin1')

        self.path = path
        self.name = os.path.split(path)[-1]
        self.frequencies  = load_meta(path, 'frequencies').squeeze().astype(int)
        self.range_vector = load_meta(path, 'range_vector').squeeze()
        self.time_vector  = load_meta(path, 'time_vector').squeeze()
        self.heave = load_meta(path, 'heave').squeeze()
        self.data_dtype = load_meta(path, 'data_dtype')
        self.label_dtype = load_meta(path, 'label_dtype')
        self.shape = load_meta(path, 'shape')
        self.objects = load_meta(path, 'objects')
        self.n_objects = len(self.objects)
        self.year = int(self.name[9:13])
        self._seabed = None

        self.date = np.datetime64(self.name[9:13] + '-' + self.name[13:15] + '-' + self.name[15:17] + 'T' + self.name[19:21] + ':' + self.name[21:23]) #'yyyy-mm-ddThh:mm'

        #Check which labels that are included
        self.label_types_in_echogram = np.unique([o['fish_type_index'] for o in self.objects])

        #Make dictonary that points to objects with a given label
        for object_id, object in enumerate(self.objects):
            label = object['fish_type_index']
            if label not in self.object_ids_with_label.keys():
                self.object_ids_with_label[label] = []
            self.object_ids_with_label[label].append(object_id)


    def visualize(self,
                  predictions=None,
                  prediction_strings=None,
                  labels_original=None,
                  labels_refined=None,
                  labels_korona=None,
                  pred_contrast=1.0,
                  frequencies=None,
                  draw_seabed=True,
                  show_labels=True,
                  show_object_labels=True,
                  show_grid=False,
                  show_name=True,
                  show_freqs=True,
                  show_labels_str=True,
                  show_predictions_str=True,
                  return_fig=False,
                  figure=None,
                  data_transform=db):
        '''
        Visualize echogram, labels and predictions
        :param predictions: (numpy.array, list(numpy.array)) One or more predictions
        :param prediction_strings: (str, list(str)) Description for each prediction, typically model name
        :param labels_original: (np.array) Original label mask
        :param labels_refined: (np.array) Refined label mask
        :param labels_korona: (np.array) KorOna prediction mask from the LSSS software
        :param pred_contrast: (float) Gamma-correction for predictions (initial values are raised to this exponent)
        :param frequencies: (list) Data frequencies to plot (in kHz)
        :param draw_seabed: (bool) Plot seabed line
        :param show_labels: (bool) Plot labels
        :param show_object_labels: (bool) Plot species code as text for each school
        :param show_grid: (bool) Plot grid lines
        :param show_name: (bool) Plot echogram name
        :param show_freqs: (bool) Plot frequencies as text for each data subplot
        :param show_labels_str: (bool) Plot label type as text ('original annotations' etc.) for each label subplot
        :param show_predictions_str: Plot prediction_str as text for each prediction subplot
        :param return_fig: (bool) Return the matplotlib.figure object instead of showing the plot
        :param figure: (matplotlib.figure object)
        :param data_transform: (function) Function for data transformation (e.g. 10*log_10(x + eps))
        :return: None or matplotlib.figure object
        '''

        #Todo: Refactor this method to using "subplots" (instead of "subplot") - which is much easier to configure.

        # Get data
        data = self.data_numpy(frequencies)
        if labels_original is not None:
            labels = labels_original
        else:
            labels = self.label_numpy()
        if frequencies is None:
            frequencies = self.frequencies

        # Transform data
        if data_transform != None:
            data = data_transform(data)

        # Initialize plot
        #plt = setup_matplotlib()
        if figure is not None:
            plt.clf()

        plt.tight_layout()

        # Tick labels
        tick_labels_y = self.range_vector
        tick_labels_y = tick_labels_y - np.min(tick_labels_y)
        tick_idx_y = np.arange(start=0, stop=len(tick_labels_y), step=int(len(tick_labels_y) / 4))
        tick_labels_x = self.time_vector * 24 * 60  # convert from days to minutes
        tick_labels_x = tick_labels_x - np.min(tick_labels_x)
        tick_idx_x = np.arange(start=0, stop=len(tick_labels_x), step=int(len(tick_labels_x) / 6))
        tick_labels_x_empty = [''] * len(tick_labels_x)

        # Format settings
        color_seabed = {'seabed': 'white'}
        lw = {'seabed': 0.4}
        cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
        boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
        norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

        # Number of subplots
        n_plts = data.shape[2]
        if show_labels:
            n_plts += 1
        if labels_refined is not None:
            n_plts += 1
        if labels_korona is not None:
            n_plts += 1
        if predictions is not None:
            if type(predictions) is np.ndarray:
                n_plts += 1
            elif type(predictions) is list:
                n_plts += len(predictions)

        # Channels
        for i in range(data.shape[2]):
            if i == 0:
                main_ax = plt.subplot(n_plts, 1, i + 1)
                str_title = ''
                if show_name:
                    str_title += self.name + ' '
                if show_freqs:
                    str_title += '\n' + str(frequencies[i]) + ' kHz'
                if show_name or show_freqs:
                    plt.title(str_title, fontsize=8)
            else:
                plt.subplot(n_plts, 1, i + 1, sharex=main_ax, sharey=main_ax)
                if show_freqs:
                    plt.title(str(frequencies[i]) + ' kHz', fontsize=8)
            plt.imshow(data[:, :, i], cmap='jet', aspect='auto')

            # Hide grid
            if not show_grid:
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=6)
                plt.xticks(tick_idx_x, tick_labels_x_empty, fontsize=6)
                plt.ylabel("Depth\n[meters]", fontsize=8)
            if draw_seabed:
                plt.plot(np.arange(data.shape[1]), self.get_seabed(), c=color_seabed['seabed'], lw=lw['seabed'])

        # Labels
        if show_labels:
            i += 1
            plt.subplot(n_plts, 1, i + 1, sharex = main_ax, sharey = main_ax)
            plt.imshow(labels, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            if show_labels_str:
                plt.title("Annotations (original)", fontsize=8)
            if draw_seabed:
                plt.plot(np.arange(data.shape[1]), self.get_seabed(), c=color_seabed['seabed'], lw=lw['seabed'])

            # Hide grid
            if not show_grid:
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=6)
                plt.xticks(tick_idx_x, tick_labels_x_empty, fontsize=6)
                plt.ylabel("Depth\n[meters]", fontsize=8)

            # Object labels
            if show_object_labels:
                for object in self.objects:
                    y = object['bounding_box'][0]
                    x = object['bounding_box'][2]
                    s = object['fish_type_index']
                    plt.text(x, y, s, {'FontSize': 8, 'color': 'white', 'backgroundcolor': [0, 0, 0, .2]})

        # Refined labels
        if labels_refined is not None:
            i += 1
            plt.subplot(n_plts, 1, i + 1, sharex=main_ax, sharey=main_ax)
            plt.imshow(labels_refined, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            if show_labels_str:
                plt.title("Annotations (modified)", fontsize=8)
            if draw_seabed:
                plt.plot(np.arange(data.shape[1]), self.get_seabed(), c=color_seabed['seabed'], lw=lw['seabed'])
            # Hide grid
            if not show_grid:
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=6)
                plt.xticks(tick_idx_x, tick_labels_x_empty, fontsize=6)
                plt.ylabel("Depth\n[meters]", fontsize=8)

        # Korona labels
        if labels_korona is not None:
            i += 1
            plt.subplot(n_plts, 1, i + 1, sharex=main_ax, sharey=main_ax)
            plt.imshow(labels_korona, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            if show_labels_str:
                plt.title("Korneliussen et al. method", fontsize=8)
            if draw_seabed:
                plt.plot(np.arange(data.shape[1]), self.get_seabed(), c=color_seabed['seabed'], lw=lw['seabed'])
            # Hide grid
            if not show_grid:
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=6)
                plt.xticks(tick_idx_x, tick_labels_x_empty, fontsize=6)
                plt.ylabel("Depth\n[meters]", fontsize=8)

        # Predictions
        if predictions is not None:
            if type(predictions) is np.ndarray:
                plt.subplot(n_plts, 1, i + 2, sharex=main_ax, sharey=main_ax)
                plt.imshow(np.power(predictions, pred_contrast), cmap='viridis', aspect='auto', vmin=0, vmax=1)
                if show_predictions_str:
                    plt.title("Predictions", fontsize=8)
                if draw_seabed:
                    plt.plot(np.arange(data.shape[1]), self.get_seabed(), c=color_seabed['seabed'], lw=lw['seabed'])
                # Hide grid
                if not show_grid:
                    plt.axis('off')
                else:
                    plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=6)
                    #plt.xticks(tick_idx_x, tick_labels_x_empty, fontsize=6)
                    plt.xticks(tick_idx_x, [int(tick_labels_x[j]) for j in tick_idx_x], fontsize=6)
                    plt.ylabel("Depth\n[meters]", fontsize=8)
            elif type(predictions) is list:
                if prediction_strings is not None:
                    assert len(prediction_strings) == len(predictions)
                for p in range(len(predictions)):
                    plt.subplot(n_plts, 1, i + 2 + p, sharex=main_ax, sharey=main_ax)
                    plt.imshow(np.power(predictions[p], pred_contrast), cmap='viridis', aspect='auto', vmin=0, vmax=1)
                    if prediction_strings is not None:
                        plt.title(prediction_strings[p], fontsize=8)
                    if draw_seabed:
                        plt.plot(np.arange(data.shape[1]), self.get_seabed(), c=color_seabed['seabed'], lw=lw['seabed'])
                    # Hide grid
                    if not show_grid:
                        plt.axis('off')
                    else:
                        plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=6)
                        plt.xticks(tick_idx_x, tick_labels_x_empty, fontsize=6)
                        plt.ylabel("Depth\n[meters]", fontsize=8)

        plt.xlabel("Time [minutes]", fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, top=0.95, bottom=0.05, hspace=0.20)

        if return_fig:
            pass
        else:
            plt.show()


    def data_memmaps(self, frequencies=None):
        """ Returns list of memory map arrays, one for each frequency in frequencies """

        #If no frequency is provided, show all
        if frequencies is None:
            frequencies = self.frequencies[:]

        #Make iterable
        if not isinstance(frequencies, collections.Iterable):
            frequencies = [frequencies]

        return [np.memmap(self.path + '/data_for_freq_' + str(int(f)) + '.dat', dtype=self.data_dtype, mode='r', shape=tuple(self.shape)) for f in frequencies]


    def data_numpy(self, frequencies=None):
        """ Returns numpy array with data (H x W x C)"""
        data = self.data_memmaps(frequencies=frequencies) #Get memory maps
        data = [np.array(d[:]) for d in data] #Read memory map into memory
        [d.setflags(write=1) for d in data] #Set write permissions to array
        data = [np.expand_dims(d,-1) for d in data] #Add channel dimension
        data = np.concatenate(data,-1)
        return data.astype('float32')


    def label_memmap(self, heave=True):
        '''
        Returns memory map array with labels.
        'heave' == True: returns labels without heave-corrections, i.e. labels that match the echogram data.
        'heave' == False: returns original heave-corrected labels, which *does not* match the echogram data.
        'labels_heave.dat' is generated from 'labels.dat', i.e. with 'heave' set to False, running the script:
        data_preprocessing/generate_label_files_without_heave_compensation.py
        :param heave: (bool)
        :return: (numpy.memmap) Memory map to label array
        '''
        if heave:
            # If label file without heave compensation does not exist, generate the file and write to disk
            if not os.path.isfile(self.path + '/labels_heave.dat'):
                write_label_file_without_heave_correction_one_echogram(self, force_write=False)
            return np.memmap(self.path + '/labels_heave.dat', dtype=self.label_dtype, mode='r', shape=tuple(self.shape))
        else:
            return np.memmap(self.path + '/labels.dat', dtype=self.label_dtype, mode='r', shape=tuple(self.shape))


    def label_numpy(self, heave=True):
        '''
        Returns numpy array with labels (H x W)
        :param heave: (bool) See self.label_memmap
        :return: (numpy.array) Label array
        '''
        label = self.label_memmap(heave)
        label = np.array(label[:])
        label.setflags(write=1)
        return label


    def get_seabed(self, save_to_file=True, ignore_saved=False):
        """
        Returns seabed approximation line as maximum vertical second order gradient
        :param save_to_file: (bool)
        :param ignore_saved: (bool) If True, this function will re-estimate the seabed even if there exist a saved seabed
        :return:
        """

        if self._seabed is not None and not ignore_saved:
            return self._seabed

        # elif os.path.isfile(os.path.join(self.path, 'seabed.npy')) and not ignore_saved:
        #     self._seabed = np.load(os.path.join(self.path, 'seabed.npy'))
        #     return self._seabed

        else:
            print("Estimate seabed")
            def set_non_finite_values_to_zero(input):
                input[np.invert(np.isfinite(input))] = 0
                return input

            def seabed_gradient(data):
                gradient_filter_1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                gradient_filter_2 = np.array([[1, 5, 1], [-2, -10, -2], [1, 5, 1]])
                grad_1 = conv2d(data, gradient_filter_1, mode='same')
                grad_2 = conv2d(data, gradient_filter_2, mode='same')
                return np.multiply(np.heaviside(grad_1, 0), grad_2)

            # Number of pixel rows at top of image (noise) not included when computing the maximal gradient
            n = 10 + int(0.05 * self.shape[0])
            # Vertical shift of seabed approximation line (to give a conservative line)
            a = int(0.004 * self.shape[0])

            data = set_non_finite_values_to_zero(self.data_numpy())
            seabed = np.zeros((data.shape[1:]))
            for i in range(data.shape[2]):
                seabed[:, i] = -a + n + np.argmax(seabed_gradient(data[:, :, i])[n:, :], axis=0)

            # Repair large jumps in seabed altitude
            repair_threshold = -8

            # Set start/stop for repair interval [i_edge:-i_edge] to avoid repair at edge of echogram
            i_edge = 2

            sb_max = np.max(data[n:, :, :], axis=0)
            sb_max = np.log(1e-10 + sb_max)
            sb_max -= np.mean(sb_max, axis=0)
            sb_max *= 1 / np.std(sb_max, axis=0)

            for f in range(sb_max.shape[1]):

                i = i_edge
                while i < sb_max.shape[0] - i_edge:

                    # Get interval [idx_0, idx_1] where seabed will be repaired for frequency f
                    if sb_max[i, f] < repair_threshold:
                        idx_0 = i
                        while i < sb_max.shape[0]:
                            if sb_max[i, f] < repair_threshold:
                                i += 1
                            else:
                                break
                        idx_1 = i - 1
                        # Replace initial seabed values with mean value before/after repair interval
                        if idx_0 <= i_edge:
                            seabed[idx_0:idx_1 + 1, f] = seabed[idx_1 + 1, f]
                        elif idx_1 >= sb_max.shape[0] - i_edge:
                            seabed[idx_0:idx_1 + 1, f] = seabed[idx_0 - 1, f]
                        else:
                            seabed[idx_0:idx_1 + 1, f] = np.mean(seabed[[idx_0 - 1, idx_1 + 1], f])
                    i += 1

            self._seabed = np.rint(np.median(seabed, axis=1)).astype(int)
            if save_to_file:
                np.save(os.path.join(self.path, 'seabed.npy'), self._seabed)
            return self._seabed


class DataReaderZarr():
    """
    Data reader for zarr files. Expectation is that the zarr file contains data from one year only
    """

    def __init__(self, path):
        data = xr.open_zarr(path, chunks={'frequency': 'auto'})
        self.ds = xr.Dataset(data)

        self.path = path
        self.name = os.path.split(path)[-1].split('.')[0]

        # seabed
        self.seabed_path = self.path.split('.zarr')[0] + '_seabed.zarr'
        self.seabed_dataset = None

        # Coordinates
        self.frequencies = self.ds.frequency
        self.channel_id = self.ds.get('channelID')
        self.latitude = self.ds.get('latitude')
        self.longitude = self.ds.get('longitude')
        self.range_vector = self.ds.range
        self.time_vector = self.ds.ping_time
        self.shape = (self.ds.sizes['ping_time'], self.ds.sizes['range'])

        self.raw_file = self.ds.raw_file  # List of raw files, length = nr of pings
        self.raw_file_included = np.unique(self.ds.raw_file.values)  # list of unique raw files contained in zarr file
        self.raw_file_excluded = []
        self.raw_file_start = None

        # data variables
        self.heave = self.ds.heave

        self.year = int(self.ds.ping_time[0].dt.year)
        self.date_range = (self.ds.ping_time[0], self.ds.ping_time[-1])

        # Objects (schools of fish defined by bounding boxes) are saved in a separate zarr-file
        self.objects = None
        if os.path.isdir(path.split('.')[0] + '_obj.zarr'):
            objs = xr.open_zarr(path.split('.')[0] + '_obj.zarr', chunks='auto')
            self.objects = xr.Dataset(objs)

        # Get seabed if saved in zarr_file, returns None otherwise
        self._seabed = self.ds.get('seabed')

    def get_rawfile_start_idx(self):
        """
        Get the start index of each raw file. Returns vector where length = nr of rawfiles in zarr file
        """
        if self.raw_file_start is None:
            name_changes = np.argwhere(self.raw_file[:-1].values != self.raw_file[1:].values) + 1
            if len(name_changes) == 0:
                self.raw_file_start = np.array([0])
                return self.raw_file_start

            self.raw_file_start = np.insert(name_changes, 0, 0, axis=0).squeeze()

        return self.raw_file_start

    # TODO test get last rawfile
    def get_data_rawfile(self, raw_file, frequencies=None, drop_na=False):
        """
        Get sv data for specified raw_file
        :param raw_file: (str)
        :param frequencies: (list) if None, all frequencies are returned
        :param drop_na: (bool) if True, nans at the bottom of data is dropped, if any
        """
        if frequencies is None:
            frequencies = self.frequencies

        # Get start and end index of rawfile
        idx = np.argwhere(self.raw_file_included == raw_file).squeeze()
        if (idx + 1 == len(self.raw_file_included)) or (len(self.raw_file_included) == 1):
            (x0, x1) = (self.get_rawfile_start_idx()[idx], self.shape[0])
        else:
            (x0, x1) = self.get_rawfile_start_idx()[idx:idx + 2]

        # get data
        data = self.ds.sv.loc[frequencies][:, x0:x1, :]

        # drop nans at the bottom of rawfile
        if drop_na:
            data = data.dropna(dim='range')

        return data

    def get_labels_rawfile(self, raw_file, drop_na=False, heave=False):

        """
        Get annotation mask for specified rawfile
        :param raw_file: (str)
        :param drop_na: (bool) if True, nans at the bottom of data is dropped, if any
        :param heave:
        'heave' == True: returns labels without heave-corrections, i.e. labels that match the echogram data.
        'heave' == False: returns original heave-corrected labels, which *does not* match the echogram data.
        """
        idx = np.argwhere(self.raw_file_included == raw_file).squeeze()
        if idx + 1 == len(self.raw_file_included) or len(self.raw_file_included) == 1:
            (x0, x1) = (self.get_rawfile_start_idx()[idx], self.shape[0])
        else:
            (x0, x1) = self.get_rawfile_start_idx()[idx:idx + 2]

        if heave:
            if self.ds.get('labels_heave') is None: # create label mask from parquet file if not included in zarr
                self._create_label_mask(heave)
            labels = self.ds.labels_heave[x0:x1, :]
        else:
            if self.ds.get('labels') is None: # create label mask from parquet file if not included in zarr
                self._create_label_mask(heave)
            labels = self.ds.labels[x0:x1, :]

        # drop nans
        if drop_na:
            labels = labels.dropna(dim='ping_time')
            labels = labels.dropna(dim='range')
        return labels

    def get_data_ping(self, ping_idx, frequencies=None, drop_na=True):
        """
        Get data for specified ping or ping interval
        :param ping_idx: (tuple/list/int) ping index or ping interval
        :param frequencies: (list)
        :param drop_na: (bool)
        """
        if frequencies is None:
            frequencies = self.frequencies

        if type(ping_idx) == tuple or type(ping_idx) == list:
            data = self.ds.sv.loc[frequencies][:, ping_idx[0]:ping_idx[1], :]
        else:
            data = self.ds.sv.loc[frequencies][:, ping_idx, :]

        if drop_na:
            data = data.dropna(dim='range')
        return data

    def get_data_ping_range(self, ping_idx, range_idx, frequencies=None, drop_na=True):
        """
        Get data for specified ping or ping interval and range or range interval
        :param ping_idx: (tuple/list/int) ping index or ping interval
        :param range_idx: (tuple/list/int) range index or range interval
        :param frequencies: (list)
        :param drop_na: (bool)
        """
        if frequencies is None:
            frequencies = self.frequencies

        if (type(ping_idx) == tuple or type(ping_idx) == list) and (
                type(range_idx) == tuple or type(range_idx) == list):
            data = self.ds.sv.loc[frequencies][:, ping_idx[0]:ping_idx[1], range_idx[0]: range_idx[1]]
        elif (type(ping_idx) == tuple or type(ping_idx) == list) and (
                type(range_idx) != tuple or type(range_idx) != list):
            data = self.ds.sv.loc[frequencies][:, ping_idx[0]:ping_idx[1], range_idx]
        elif (type(ping_idx) != tuple or type(ping_idx) != list) and (
                type(range_idx) == tuple or type(range_idx) == list):
            data = self.ds.sv.loc[frequencies][:, ping_idx, range_idx[0]: range_idx[1]]
        else:
            data = self.ds.sv.loc[frequencies][:, ping_idx, range_idx]

        if drop_na:
            data = data.dropna(dim='range')
        return data

    def get_label_ping(self, ping_idx, drop_na=True, heave=True):
        """
        Get annotation mask for specified ping or ping interval
        :param ping_idx: (tuple/list/int) ping index or ping interval
        :param drop_na: (bool)
        :param heave:
        'heave' == True: returns labels without heave-corrections, i.e. labels that match the echogram data.
        'heave' == False: returns original heave-corrected labels, which *does not* match the echogram data.
        """

        if heave:
            if self.ds.get('labels_heave') is None:  # create label mask from parquet file if not included in zarr
                self._create_label_mask(heave)
            ds_labels = self.ds.labels_heave
        else:
            if self.ds.get('labels') is None:  # create label mask from parquet file if not included in zarr
                self._create_label_mask(heave)
            ds_labels = self.ds.labels

        if type(ping_idx) == tuple or type(ping_idx) == list:
            labels = ds_labels[ping_idx[0]:ping_idx[1], :]
        else:
            labels = ds_labels[ping_idx, :]

        if drop_na:
            return labels.dropna(dim='range')
        else:
            return labels

    def get_label_ping_range(self, ping_idx, range_idx, drop_na=True, heave=True):
        """
        Get annotation mask for specified ping or ping interval and range or range interval
        :param ping_idx: (tuple/list/int) ping index or ping interval
        :param range_idx: (tuple/list/int) range index or range interval
        :param drop_na: (bool)
        :param heave:
        'heave' == True: returns labels without heave-corrections, i.e. labels that match the echogram data.
        'heave' == False: returns original heave-corrected labels, which *does not* match the echogram data.
        """

        if heave:
            if self.ds.get('labels_heave') is None:  # create label mask from parquet file if not included in zarr
                self._create_label_mask(heave)
            ds_labels = self.ds.labels_heave
        else:
            if self.ds.get('labels') is None:  # create label mask from parquet file if not included in zarr
                self._create_label_mask(heave)
            ds_labels = self.ds.labels

        if (type(ping_idx) == tuple or type(ping_idx) == list) and (
                type(range_idx) == tuple or type(range_idx) == list):
            labels = ds_labels[ping_idx[0]:ping_idx[1], range_idx[0]: range_idx[1]]
        elif (type(ping_idx) == tuple or type(ping_idx) == list) and (
                type(range_idx) != tuple or type(range_idx) != list):
            labels = ds_labels[ping_idx[0]:ping_idx[1], range_idx]
        elif (type(ping_idx) != tuple or type(ping_idx) != list) and (
                type(range_idx) == tuple or type(range_idx) == list):
            labels = ds_labels[ping_idx, range_idx[0]: range_idx[1]]
        else:
            labels = ds_labels[ping_idx, range_idx]

        if drop_na:
            return labels.dropna(dim='range')
        else:
            return labels

    def data_numpy(self, raw_file, frequencies=None):
        """
        Get data for specified raw file in numpy format
        :param raw_file: (str)
        :param frequencies: (list)
        :return (numpy.array)
        """
        data = self.get_data_rawfile(raw_file, frequencies)
        return np.array(data)  # read into memory

    def label_numpy(self, raw_file):
        """
        Get annotation mask for specified raw file
        :param raw_file:
        :return: (numpy.array)
        """
        label = self.get_labels_rawfile(raw_file)
        return np.array(label)  # read into memory

    def get_bounding_boxes(self, raw_file):
        """
        Retrieve object (fish school) bounding boxes for specified raw file
        :param raw_file: (str)
        """
        raw_obj = self.objects.where(self.objects.raw_file == raw_file, drop=True).dropna(dim='object_length')
        if raw_obj.sizes['raw_file'] == 0:
            return raw_obj.bounding_box.values, raw_obj.fish_type_index.values

        fish_labels = raw_obj.fish_type_index.values  # .squeeze(1)
        bounding_boxes = raw_obj.bounding_box.values  # .squeeze(2)  # (bbox, length)
        return bounding_boxes, fish_labels

    def data_numpy(self, raw_file, frequencies=None):
        """
        Get data for specified raw file in numpy format
        :param raw_file: (str)
        :param frequencies: (list)
        :return (numpy.array)
        """
        data = self.get_data_rawfile(raw_file, frequencies)
        return np.array(data)  # read into memory

    def label_numpy(self, raw_file):
        """
        Get annotation mask for specified raw file
        :param raw_file:
        :return: (numpy.array)
        """
        label = self.get_labels_rawfile(raw_file)
        return np.array(label)  # read into memory

    def filter(self, min_shape=256):
        raw_file_excluded = []
        for raw_file in tqdm(self.raw_file_included):
            # Shape
            shape = self.get_data_rawfile(raw_file, drop_na=True).shape
            # print(shape, shape[2])
            if shape[2] < min_shape:
                raw_file_excluded.append(raw_file)

        self.raw_file_excluded = raw_file_excluded

    # TODO change from subplot to subplots -> more flexible?
    def visualize(self,
                  raw_file=None,
                  predictions=None,
                  prediction_strings=None,
                  labels_original=None,
                  labels_refined=None,
                  labels_korona=None,
                  pred_contrast=1.0,
                  frequencies=None,
                  draw_seabed=False,
                  show_labels=True,
                  show_object_labels=False,
                  show_grid=False,
                  show_name=True,
                  show_freqs=True,
                  show_labels_str=True,
                  show_predictions_str=True,
                  return_fig=False,
                  figure=None,
                  data_transform=db,
                  drop_na=False,
                  frequency_unit='Hz'):
        """
        Visualize echogram from zarr format, labels and predictions
        """
        # retrieve data
        data = self.get_data_rawfile(raw_file, frequencies, drop_na=drop_na)
        if data_transform != None:
            data = data_transform(data)

        # Initialize plot
        # plt = setup_matplotlib()
        fig = plt.figure(dpi=200)
        plt.tight_layout()

        # Tick labels
        tick_labels_y = data.range
        tick_labels_y = tick_labels_y - np.min(tick_labels_y)
        tick_idx_y = np.arange(start=0, stop=len(tick_labels_y), step=int(len(tick_labels_y) / 4))
        tick_labels_x = data.ping_time
        tick_idx_x = np.arange(start=0, stop=len(tick_labels_x), step=int(len(tick_labels_x) / 6))
        tick_labels_x = pd.DatetimeIndex(tick_labels_x[tick_idx_x].values)
        tick_labels_x = (tick_labels_x - tick_labels_x.min()).total_seconds() / 60

        # Format settings
        color_seabed = {'seabed': 'white'}
        lw = {'seabed': 0.4}
        cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
        boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
        norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

        cmap_seg = mcolors.ListedColormap(['black', 'red', 'firebrick'])
        boundaries_seg = [0, 0.6, 0.8, 1]
        norm_seg = mcolors.BoundaryNorm(boundaries_seg, cmap_seg.N, clip=True)

        # get nr of subplots
        n_plts = data.shape[0] + int(show_labels)
        if predictions is not None:
            if type(predictions) is np.ndarray:
                n_plts += 1
            elif type(predictions) is list:
                n_plts += len(predictions)

        for i in range(data.shape[0]):
            if i == 0:
                main_ax = plt.subplot(n_plts, 1, i + 1)
            else:
                plt.subplot(n_plts, 1, i + 1, sharex=main_ax, sharey=main_ax)

            if show_freqs:
                str_title = str(data[i].frequency.values) + frequency_unit
                plt.title(str_title, fontsize=8)

            plt.imshow(data[i, :, :].T, cmap='jet', aspect='auto')

            # Grid
            if not show_grid:
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=6)
                plt.xticks(tick_idx_x, np.round(tick_labels_x, 2), fontsize=6)
                plt.ylabel("Depth\n[meters]", fontsize=8)
            if draw_seabed:
                plt.plot(np.arange(data.shape[1]), self.get_seabed(raw_file), c=color_seabed['seabed'], lw=lw['seabed'])

        # Labels
        if show_labels:
            i += 1
            labels = self.get_labels_rawfile(raw_file)
            if drop_na:
                labels = labels.where(~labels.isnull(), drop=True)
            plt.subplot(n_plts, 1, i + 1, sharex=main_ax, sharey=main_ax)
            if show_labels_str:
                plt.title("Annotations (original)", fontsize=8)
            plt.imshow(labels.T, cmap=cmap_labels, norm=norm_labels, aspect='auto')

            if not show_grid:
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=6)
                plt.xticks(tick_idx_x, np.round(tick_labels_x, 2), fontsize=6)
                plt.ylabel("Depth\n[meters]", fontsize=8)
            if draw_seabed:
                plt.plot(np.arange(data.shape[1]), self.get_seabed(raw_file), c=color_seabed['seabed'], lw=lw['seabed'])

            # object labels
            if show_object_labels and raw_file is not None:
                objects, labels = self.get_bounding_boxes(raw_file)
                if objects.shape[-1] != 0:
                    for obj_idx in range(objects.shape[1]):
                        y = objects[0, obj_idx]
                        x = objects[2, obj_idx]
                        s = int(labels[obj_idx])
                        plt.text(x, y, s, {'FontSize': 8, 'color': 'white', 'backgroundcolor': [0, 0, 0, .2]})

        # Show predictions
        # TODO test this
        if predictions is not None:
            if type(predictions) is np.ndarray:
                predictions = [predictions]
                prediction_strings = ['Predictions']
            for p in range(len(predictions)):
                i += p
                plt.subplot(n_plts, 1, i + 2, sharex=main_ax, sharey=main_ax)
                plt.imshow(np.power(predictions[p], pred_contrast),
                           cmap=cmap_seg, norm=norm_seg, aspect='auto', vmin=0, vmax=1)
                           #cmap='viridis', aspect='auto', vmin=0, vmax=1)
                if prediction_strings is not None:
                    plt.title(prediction_strings[p], fontsize=8)
                if draw_seabed:
                    plt.plot(np.arange(data.shape[1]), self.get_seabed(raw_file), c=color_seabed['seabed'],
                             lw=lw['seabed'])
                if not show_grid:
                    plt.axis('off')
                else:
                    plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=6)
                    plt.xticks(tick_idx_x, np.round(tick_labels_x, 2), fontsize=6)
                    plt.ylabel("Depth\n[meters]", fontsize=8)

        if show_name and raw_file is not None:
            fig.suptitle(raw_file, fontsize=10)

        plt.xlabel('Time [minutes]', fontsize=8)
        plt.tight_layout()
        plt.show()

    def get_seabed(self, save_to_file=True):
        """ Return, load or calculate seabed for entire reader"""
        if self.seabed_dataset is not None:
            return self.seabed_dataset.seabed
        elif os.path.isdir(self.seabed_path):
            self.seabed_dataset = xr.open_zarr(self.seabed_path)
            return self.seabed_dataset.seabed
        else:
            print("Estimate seabed")
            def seabed_gradient(data):
                gradient_filter_1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                gradient_filter_2 = np.array([[1, 5, 1], [-2, -10, -2], [1, 5, 1]])
                grad_1 = conv2d(data, gradient_filter_1, mode='same')
                grad_2 = conv2d(data, gradient_filter_2, mode='same')
                return np.multiply(np.heaviside(grad_1, 0), grad_2)

            data = self.ds.sv.fillna(0)  # fill nans with 0

            # Number of pixel rows at top of image (noise) not included when computing the maximal gradient
            n = 150  # 10*int(0.05*500)

            # Vertical shift of seabed approximation line (to give a conservative line)
            a = int(0.004 * 500)

            seabed = xr.DataArray(data=np.zeros((data.shape[:2])),
                                  dims=['frequency', 'ping_time'],
                                  coords=[data.frequency,
                                          data.ping_time])  # seabed = np.zeros((data.shape[:2]))  # (freq, ping_time)
            for i in range(data.shape[0]):
                seabed_grad = xr.apply_ufunc(seabed_gradient, data[i, :, :], dask='allowed')
                seabed[i, :] = -a + n + seabed_grad[:, n:].argmax(axis=1)

            # Repair large jumps in seabed altitude
            # Set start/stop for repair interval [i_edge:-i_edge] to avoid repair at edge of echogram
            i_edge = 2

            # Use rolling mean and rolling std with window of 500 to find jumps in the seabed altitude
            repair_threshold = 0.75
            window_size = 500
            sb_max = seabed - seabed.rolling(ping_time=window_size, min_periods=1, center=True).mean()
            sb_max *= 1 / seabed.rolling(ping_time=window_size, min_periods=1, center=True).std()

            for f in range(sb_max.shape[0]):
                i = i_edge

                # Get indices of
                to_fix = np.argwhere(abs(sb_max[f, i:]).values > repair_threshold).ravel() + i
                k = 0
                while k < len(to_fix):
                    idx_0 = to_fix[k]

                    # Check if there is multiple subsequent indexes that needs repair
                    c = 0
                    while to_fix[k + c] == idx_0 + c:
                        c += 1
                        if k + c == len(to_fix):
                            break
                    idx_1 = idx_0 + c - 1

                    if idx_0 <= i_edge:
                        seabed[f, idx_0:idx_1 + 1] = seabed[f, idx_1 + 1]
                    elif idx_1 >= sb_max.shape[1] - i_edge:
                        seabed[f, idx_0:idx_1 + 1] = seabed[f, idx_0 - 1]
                    else:
                        seabed[f, idx_0:idx_1 + 1] = (seabed[f, [idx_0 - 1, idx_1 + 1]]).mean()

                    k += c

            s = xr.ufuncs.rint(seabed.median(dim='frequency'))
            self.seabed_dataset = xr.Dataset(data_vars={'seabed': s}, coords={'ping_time': s.ping_time})

            # save to zarr file
            if save_to_file:
                self.seabed_dataset.to_zarr(self.seabed_path)
            return self.seabed_dataset

    # TODO Save to file, not in zarr?
    def _create_label_mask(self, heave=True):
        parquet_path = os.path.join(self.path.split('.')[0] + '_work.parquet')
        transducer_offset = self.ds.transducer_draft.mean(dim='frequency')

        if os.path.isfile(parquet_path):
            # read parquet data
            parquet_data = pd.read_parquet(os.path.join(parquet_path), engine='pyarrow')
            labels = np.zeros(shape=(self.ds.dims['ping_time'], self.ds.dims['range']))

            # add labels as variable to zarr
            self.ds["labels"] = (['ping_time', 'range'], labels)

            for _, row in parquet_data.iterrows():
                x0 = row['mask_depth_upper']-transducer_offset.loc[row['pingTime']]
                x1 = row['mask_depth_lower']-transducer_offset.loc[row['pingTime']]
                fish_id = int(row['ID'].split('-')[-1])

                if heave:
                    h = self.heave.loc[row['pingTime']]
                    if h == 0:
                        self.ds["labels_heave"].loc[row['pingTime'], x0:x1] = fish_id
                    else:
                        self.ds["labels_heave"].loc[row['pingTime'], x0-h:x1-h] = fish_id
                else:
                    # add fish observation to label mask
                    self.ds["labels"].loc[row['pingTime'], x0:x1] = fish_id


def get_zarr_files(frequencies=[18, 38, 120, 200], minimum_shape=256):
    path_to_zarr_files = paths.path_to_zarr_files()
    zarr_files = sorted([z_file for z_file in os.listdir((path_to_zarr_files)) \
                         if '_obj' not in z_file and 'seabed' not in z_file and '.zarr' in z_file])

    zarr_readers = [DataReaderZarr(os.path.join(path_to_zarr_files, zarr_file)) for zarr_file in zarr_files]

    # Filter on frequencies
    zarr_readers = [z for z in zarr_readers if all([f in z.frequencies for f in frequencies])]

    # Filter on shape: minimum size
    # for zarr_reader in zarr_readers:
    #     zarr_reader.filter()
    #     print(zarr_reader.raw_file_excluded)
    # Filter

    return zarr_readers

def get_echograms(years='all', frequencies=[18, 38, 120, 200], minimum_shape=256):
    """ Returns all the echograms for a given year that contain the given frequencies"""

    path_to_echograms = paths.path_to_echograms()
    eg_names = os.listdir(path_to_echograms)
    eg_names = sorted(eg_names) # To visualize echogram predictions in the same order with two different models
    eg_names = [name for name in eg_names if '.' not in name] # Include folders only: exclude all root files (e.g. '.tar')

    echograms = [Echogram(os.path.join(path_to_echograms, e)) for e in eg_names]

    #Filter on frequencies
    echograms = [e for e in echograms if all([f in e.frequencies for f in frequencies])]

    # Filter on shape: minimum size
    echograms = [e for e in echograms if (e.shape[0] > minimum_shape) & (e.shape[1] > minimum_shape)]

    # Filter on shape of time_vector vs. image data: discard echograms with shape deviation
    echograms = [e for e in echograms if e.shape[1] == e.time_vector.shape[0]]

    # Filter on Korona depth measurements: discard echograms with missing depth files or deviating shape
    echograms = [e for e in echograms if e.name not in depth_excluded_echograms]

    # Filter on shape of heave vs. image data: discard echograms with shape deviation
    echograms = [e for e in echograms if e.shape[1] == e.heave.shape[0]]

    if years == 'all':
        return echograms
    else:
        #Make sure years is an itterable
        if type(years) not in [list, tuple, np.array]:
            years = [years]

        #Filter on years
        echograms = [e for e in echograms if e.year in years]

        return echograms

def get_data_readers(years='all', frequencies=[18, 38, 120, 200], minimum_shape=256, mode='zarr'):
    if mode == 'memm':
        return get_echograms(years, frequencies, minimum_shape)
    elif mode == 'zarr':
        return get_zarr_files(frequencies, minimum_shape)

if __name__ == '__main__':
    # Memmap reader
    readers = get_data_readers(years=[2017], mode='memm')
    #reader = [reader for reader in readers if reader.name == '2017843-D20170426-T063457']
    reader = [reader for reader in readers if reader.name == '2017843-D20170513-T081028']
    reader = reader[0]
    print(reader.shape)
    seabed = reader.visualize(frequencies=[200], show_grid=False)

   # Zarr reader
    readers = get_data_readers(mode='zarr', frequencies=[18000, 38000, 120000, 200000])
    reader = readers[0]
    print(reader.shape)
    seabed = reader.get_seabed(raw_file=reader.raw_file_included[0])
    reader.visualize(raw_file=reader.raw_file_included[0], frequencies=[200000])
