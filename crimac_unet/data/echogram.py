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
from glob import glob
import time
import dask
import csv

import paths
from data.normalization import db
from data.missing_korona_depth_measurements import depth_excluded_echograms
from data_preprocessing.generate_heave_compensation_files import write_label_file_without_heave_correction_one_echogram

# from utils.plotting import setup_matplotlib
# plt = setup_matplotlib()
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
        self.frequencies  = np.array(load_meta(path, 'frequencies')).squeeze().astype(int)
        self.range_vector = np.array(load_meta(path, 'range_vector')).squeeze()
        self.time_vector  = np.array(load_meta(path, 'time_vector')).squeeze()
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

        self.data_format = 'memmap'


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
        elif os.path.isfile(os.path.join(self.path, 'seabed.npy')) and not ignore_saved:
            self._seabed = np.load(os.path.join(self.path, 'seabed.npy'))
            return self._seabed

        else:
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
    :param path: path to survey directory (i.e. /cruise_data/2017/S2017843)
    """

    def __init__(self, path):
        # Get all paths
        self.sv_path = os.path.abspath(path)
        self.name = os.path.split(self.sv_path)[-1].replace('_sv.zarr', '')
        self.path = os.path.split(self.sv_path)[0]
        self.annotation_path = os.path.join(*[self.path, f'{self.name}_labels.zarr'])
        self.seabed_path = os.path.join(*[self.path, f'{self.name}_bottom.zarr'])
        self.work_path = os.path.join(*[self.path, f'{self.name}_labels.parquet'])
        self.objects_df_path = os.path.join(*[self.path, f'{self.name}_labels.parquet.csv'])
        self.data_format = 'zarr'
        assert os.path.isdir(self.sv_path), f"No Sv data found at {self.sv_path}"

        # Load data
        self.ds = xr.open_zarr(self.sv_path, chunks={'frequency': 'auto'})

        # Read coordinates
        self.frequencies = self.ds.frequency.astype(np.int)
        self.heave = self.ds.heave
        self.channel_id = self.ds.get('channelID')
        self.latitude = self.ds.get('latitude')
        self.longitude = self.ds.get('longitude')
        self.range_vector = self.ds.range
        self.time_vector = self.ds.ping_time
        self.year = int(self.ds.ping_time[0].dt.year)
        self.date_range = (self.ds.ping_time[0], self.ds.ping_time[-1])
        self.shape = (self.ds.sizes['ping_time'], self.ds.sizes['range'])
        self.raw_file = self.ds.raw_file  # List of raw files, length = nr of pings
        self.raw_file_included = np.unique(self.ds.raw_file.values)  # list of unique raw files contained in zarr file
        self.raw_file_excluded = []
        self.raw_file_start = None

        # Used for seabed estimation
        transducer_offset = self.ds.transducer_draft.mean()
        self.transducer_offset_pixels = int(transducer_offset/(self.range_vector.diff(dim='range').mean()).values)

        # Load annotations files
        self.annotation = None
        if os.path.isdir(self.annotation_path):
            self.annotation = xr.open_zarr(self.annotation_path)
            self.labels = self.annotation.annotation
            self.objects = self.annotation.object

            # Fish categories used in survey
            self.fish_categories = [cat for cat in self.annotation.category.values if cat != -1]
        else:
            print(f' No annotation file found at {self.annotation_path}')

        # Load seabed file
        self.seabed = None
        if os.path.isdir(self.seabed_path):
            self.seabed = xr.open_zarr(self.seabed_path)

        # Objects dataframe
        self.objects_df = None

    def get_ping_index(self, ping_time):
        """
        Due to rounding errors, the ping_time variable for labels and data are not exactly equal.
        This function returns the closest index to the input ping time
        :param ping_time: (np.datetime64)
        :return: (int) index of closest index in data time_vector
        """
        return int(np.abs((self.time_vector - ping_time)).argmin().values)

    def get_range_index(self, range):
        """
        Get closest index in range_vector
        """
        return int(np.abs((self.range_vector - range)).argmin().values)

    def get_fish_schools(self, category='all'):
        """
        Get all bounding boxes for the input categories
        :param category: Categories to include ('all', or list)
        :return: dataframe with bounding boxes
        """
        df = self.get_objects_file()
        if category == 'all':
            category = self.fish_categories

        if not isinstance(category, (list, np.ndarray)):
            category = [category]

        return df.loc[(df.category.isin(category)) & (df.valid_object)]

    def get_objects_file(self):
        """
        Get or compute dataframe with bounding box indexes for all fish schools
        :return: Pandas dataframe with object info and bounding boxes
        """
        if self.objects_df is not None:
            return self.objects_df

        parsed_objects_file_path = os.path.join(os.path.split(self.objects_df_path)[0],
                                                self.name + '_objects_parsed.csv')

        if os.path.isfile(parsed_objects_file_path):
            return pd.read_csv(parsed_objects_file_path, index_col=0)
        elif os.path.isfile(self.objects_df_path) and os.path.isfile(self.work_path):
            print('Generating objects file with seabed distances ... ')

            # Create parsed objects file from object file and work file
            df = pd.read_csv(self.objects_df_path, header=0)
            df = df.rename(columns={"upperdept": "upperdepth",
                                    "lowerdept": "lowerdepth",
                                    "upperdeptindex": "upperdepthindex",
                                    "lowerdeptindex": "lowerdepthindex"})

            categories = df.category.values
            upperdeptindices = df.upperdepthindex.values
            lowerdeptindices = df.lowerdepthindex.values
            startpingindices = df.startpingindex.values
            endpingindices = df.endpingindex.values

            distance_to_seabed = np.zeros_like(upperdeptindices, dtype=np.float32)
            distance_to_seabed[:] = np.nan
            valid_object = np.zeros_like(upperdeptindices, dtype='bool')

            assert len(df['object']) == len(df), print('Object IDs not unique!')
            #
            for idx, (category, upperdeptindex, lowerdeptindex, startpingindex, endpingindex) in \
                enumerate(zip(categories, upperdeptindices, lowerdeptindices, startpingindices, endpingindices)):

                # Skip objects with ping errors og of category -1
                # TODO better solution for this? Fix ping errors?
                if startpingindex > endpingindex or category == -1:
                    valid_object[idx] = False
                    continue

                # Add distance to seabed
                if os.path.isdir(self.seabed_path):
                    center_ping_idx = startpingindex + int((endpingindex - startpingindex)/2)
                    distance_to_seabed[idx] = self.get_seabed(center_ping_idx) - lowerdeptindex

                valid_object[idx] = True

            # # Save parsed objecs file
            df['distance_to_seabed'] = distance_to_seabed
            df['valid_object'] = valid_object
            df.to_csv(parsed_objects_file_path)
            self.objects_df = df
            return df
        else:
            # Cannot return object file
            raise FileNotFoundError(f'Cannot compute objects dataframe from {self.objects_df_path} and {self.work_path}')

    def get_data_slice(self, idx_ping: (int, None) = None, n_pings: (int, None) = None, idx_range: (int, None) = None, n_range: (int, None) = None,
                  frequencies: (int, list, None) = None, drop_na=False, return_numpy=True):
        '''
        Get slice of xarray.Dataset based on indices in terms of (frequency, ping_time, range).
        Arguments for 'ping_time' and 'range' indices are given as the start index and the number of subsequent indices.
        'range' and 'frequency' arguments are optional.

        :param idx_ping: (int) First ping_time index of the slice
        :param n_pings: (int) Number of subsequent ping_time indices of the slice
        :param idx_range: (int | None) First range index of the slice (None slices from first range index)
        :param n_range: (int | None) Number of subsequent range indices of the slice (None slices to last range index)
        :param frequencies: (int | list[int] | None) Frequencies in slice (None returns all frequencies)
        :return: Sliced xarray.Dataset

        Example:
        ds_slice = ds.get_slice(idx_ping=20000, n_pings=256) # xarray.Dataset sliced in 'ping_time' dimension [20000:20256]
        sv_data = ds_slice.sv # xarray.DataArray of underlying sv data
        sv_data_numpy = sv_data.values # numpy.ndarray of underlying sv data
        '''

        assert isinstance(idx_ping, (int, np.integer, type(None)))
        assert isinstance(n_pings, (int, np.integer, type(None)))
        assert isinstance(idx_range, (int, type(None)))
        assert isinstance(n_range, (int, np.integer, type(None)))
        assert isinstance(frequencies, (int, np.integer, list, np.ndarray, type(None)))
        if isinstance(frequencies, list):
            assert all([isinstance(f, (int, np.integer)) for f in frequencies])

        slice_ping_time = slice(idx_ping, idx_ping + n_pings)

        if idx_range is None:
            slice_range = slice(None, n_range)  # Valid for n_range int, None
        elif n_range is None:
            slice_range = slice(idx_range, None)
        else:
            slice_range = slice(idx_range, idx_range + n_range)

        if frequencies is None:
            frequencies = self.frequencies
        # Make sure frequencies is array-like to preserve dims when slicing
        if isinstance(frequencies, (int, np.integer)):
            frequencies = [frequencies]

        data = self.ds.sv.sel(frequency=frequencies).isel(ping_time=slice_ping_time, range=slice_range)

        if drop_na:
            data = data.dropna(dim='range')

        if return_numpy:
            return data.values
        else:
            return data

    def get_label_slice(self, idx_ping: int, n_pings: int, idx_range: (int, None) = None, n_range: (int, None) = None,
                        drop_na=False, categories=None, return_numpy=True, correct_transducer_offset=False,
                        mask=True):
        """
        Get slice of labels
        :param idx_ping: (int) Index of start ping
        :param n_pings: (int) Width of slice
        :param idx_range: (int) Index of start range
        :param n_range: (int) Height of slice
        :param drop_na: (bool) Drop nans at the bottom of data (data is padded with nans since echograms have different heights)
        :return: np.array with labels
        """
        assert isinstance(idx_ping, (int, np.integer))
        assert isinstance(n_pings, (int, np.integer))
        assert isinstance(idx_range, (int, np.integer, type(None)))
        assert isinstance(n_range, (int, np.integer, type(None)))

        slice_ping_time = slice(idx_ping, idx_ping + n_pings)

        if idx_range is None:
            slice_range = slice(None, n_range)  # Valid for n_range int, None
        elif n_range is None:
            slice_range = slice(idx_range, None)
        else:
            slice_range = slice(idx_range, idx_range + n_range)

        # Convert labels from set of binary masks to 2D segmentation mask
        if categories is None:
            categories = np.array(self.fish_categories)

        # Initialize label mask and fill
        label_slice = self.labels.isel(ping_time=slice_ping_time, range=slice_range)

        if mask:
            labels = label_slice.sel(category=-1)

            #labels = self.labels.sel(category=categories[0]).isel(ping_time=slice_ping_time, range=slice_range) * categories[0]
            for cat in categories:
                labels = labels.where(label_slice.sel(category=cat) <= 0, cat) # Where condition is False, fill with cat

            # Drop nans in range dimension
            if drop_na:
                labels = labels.dropna(dim='range')
        else:
            # TODO: mask away -1?
            labels = label_slice.sel(category=categories)

        # Convert to np array
        if return_numpy:
            return labels.values
        else:
            return labels


    def get_seabed_mask(self, idx_ping: int, n_pings: int, idx_range: (int, None) = None, n_range: (int, None) = None,
                        return_numpy=True, correct_transducer_offset=True):
        """
        Get seabed mask from slice
        :param idx_ping: Start ping index (int)
        :param n_pings: End ping index (int)
        :param idx_range: Number of pings (int)
        :param n_range: Number of vertical samples to include (int)
        :param return_numpy: Return mask as numpy array
        :return: Mask where everything below seafloor is marked with 1, everything above is marked with 0
        """

        assert isinstance(idx_ping, (int, np.integer))
        assert isinstance(n_pings, (int, np.integer))
        assert isinstance(idx_range, (int, np.integer, type(None)))
        assert isinstance(n_range, (int, np.integer, type(None)))

        slice_ping_time = slice(idx_ping, idx_ping + n_pings)

        if idx_range is None:
            idx_range = 0

        if n_range is None:
            slice_range = slice(idx_range, None)
        else:
            slice_range = slice(idx_range,
                                idx_range + n_range)

        # Everything below seafloor has value 1, everything above has value 0
        seabed_slice = self.seabed.bottom_range.isel(ping_time=slice_ping_time, range=slice_range).fillna(0)

        if return_numpy:
            return seabed_slice.values
        else:
            return seabed_slice

    def get_seabed(self, idx_ping: int, n_pings: (int) = 1, idx_range: (int, None) = None, n_range: (int, None) = None):
        """
        Get vector of range indices for the seabed
        WARNING slow for large stretches of data

        :param idx_ping: index of start ping (int)
        :param n_pings: number of pings to include (int)
        :return: vector with seabed range indices (np.array)
        """

        # Get seabed mask for the specified pings
        seabed_mask = self.get_seabed_mask(idx_ping, n_pings, idx_range, n_range, return_numpy=True)

        # Find indexes with non-zero values
        seabed_idx = np.argwhere(seabed_mask>0)
        ping_idxs, range_idxs = (seabed_idx[:, 0], seabed_idx[:, 1])

        # Fill a vector with the smallest non-zero value to get the indices of the seabed
        seabed = np.ones(n_pings)*-1
        for i in range(n_pings):
            if len(range_idxs[ping_idxs == i]) == 0:
                seabed[i] = seabed_mask.shape[1]
            else:
                seabed[i] = np.min(range_idxs[ping_idxs == i])

        return seabed.astype(int)

    def get_rawfile_index(self, rawfile):
        relevant_pings = np.argwhere(self.raw_file.values == rawfile).ravel()
        start_ping = relevant_pings[0]
        n_pings = len(relevant_pings)
        return start_ping, n_pings

    # These two functions are (currently) necessary to predict on zarr-data
    def get_data_rawfile(self, rawfile, frequencies, drop_na):
        start_ping, n_pings = self.get_rawfile_index(rawfile)

        return self.get_data_slice(idx_ping=start_ping, n_pings=n_pings, frequencies=frequencies, drop_na=drop_na, return_numpy=True)

    def get_labels_rawfile(self, rawfile):
        start_ping, n_pings = self.get_rawfile_index(rawfile)

        return self.get_label_slice(idx_ping=start_ping, n_pings=n_pings, return_numpy=True)

    def get_seabed_rawfile(self, rawfile):
        start_ping, n_pings = self.get_rawfile_index(rawfile)

        return self.get_seabed(idx_ping=start_ping, n_pings=n_pings)

    def visualize(self,
                  ping_idx=None,
                  n_pings=2000,
                  range_idx=None,
                  n_range=None,
                  raw_file=None,
                  frequencies=None,
                  draw_seabed=True,
                  show_labels=True,
                  predictions=None,
                  data_transform=db):
        """
        Visualize data from xarray
        :param ping_idx: Index of start ping (int)
        :param n_pings: Nr of pings to visualize (int)
        :param range_idx: Index of start range (int)
        :param n_range: Nr of range samples to visualize (int)
        :param raw_file: Visualize data from a single raw file (overrides ping index arguments!) (str)
        :param frequencies: Frequencies to visualize (list)
        :param draw_seabed: Draw seabed on plots (bool)
        :param show_labels: Show annotation (bool)
        :param predictions: Predictions data variables should follow annotation format or be presented as a numpy array (xarray.Dataset, numpy.ndarray)
        :param data_transform: Data transform before visualization (db transform recommended) (function)
        """

        # Visualize data from a single raw file
        if raw_file is not None:
            idxs = np.argwhere(self.raw_file.values == raw_file).ravel()
            ping_idx = idxs[0]
            n_pings = len(idxs)

        # retrieve data
        if ping_idx is None:
            ping_idx = np.random.randint(0, len(self.time_vector) - n_pings)
        if frequencies is None:
            frequencies = list(self.frequencies.values)
        if range_idx is None:
            range_idx = 0
        if n_range is None:
            n_range = self.shape[1]

        data = self.get_data_slice(ping_idx, n_pings, range_idx, n_range, frequencies, drop_na=True)

        # Optionally transform data
        if data_transform != None:
            data = data_transform(data)

        # Initialize plot
        nrows = len(frequencies) + int(show_labels)
        if predictions is not None:
            nrows += 1
        fig, axs = plt.subplots(ncols=1, nrows=nrows, figsize=(16, 16), sharex=True)
        axs = axs.ravel()
        plt.tight_layout()

        # Get tick labels
        tick_idx_y = np.arange(start=0, stop=data.shape[-1], step=int(data.shape[-1] / 4))
        tick_labels_y = self.range_vector[range_idx:range_idx+n_range].values
        tick_labels_y = np.round(tick_labels_y[tick_idx_y], decimals=0).astype(np.int32)

        tick_idx_x = np.arange(start=0, stop=n_pings, step=int(n_pings / 6))
        tick_labels_x = self.time_vector[ping_idx:ping_idx + n_pings]
        tick_labels_x = tick_labels_x[tick_idx_x].values
        tick_labels_x = [np.datetime_as_string(t, unit='s').replace('T', '\n') for t in tick_labels_x]
        #
        plt.setp(axs, xticks=tick_idx_x, xticklabels=tick_labels_x,
                 yticks=tick_idx_y, yticklabels=tick_labels_y)


        # Format settings
        cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'green', 'red']) # green = other, red = sandeel
        boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
        norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

        # Get seabed
        if draw_seabed:
            seabed = self.get_seabed(idx_ping=ping_idx, n_pings=n_pings, idx_range=range_idx, n_range=n_range).astype(np.float)
            seabed[seabed >= data.shape[-1]] = None

        # Plot data
        for i in range(data.shape[0]):
            axs[i].imshow(data[i, :, :].T, cmap='jet', aspect='auto')
            axs[i].set_title(f"{str(frequencies[i])} Hz", fontsize=8)
            axs[i].set_ylabel('Range (m)')

        # Optionally plot labels
        if show_labels:
            labels = self.get_label_slice(ping_idx, n_pings, range_idx, n_range, drop_na=True)

            # crop labels
            labels = labels[:, :data.shape[-1]]
            axs[i+1].imshow(labels.T, cmap=cmap_labels, norm=norm_labels, aspect='auto')
            axs[i+1].set_ylabel('Range (m)')
            axs[i+1].set_title('Annotations')

        # Optionally draw seabed
        if draw_seabed:
            for ax in axs:
                ax.plot(np.arange(data.shape[1]), seabed, c='white', lw=1)

        if predictions is not None:
            if type(predictions) != np.ndarray:
                predictions = predictions.annotation.sel(category=27)[range_idx:range_idx + n_range,
                             ping_idx:ping_idx + n_pings].values.astype(np.float32)

            # crop predictions (since we cut nans from the data)
            predictions = predictions[:data.shape[-1], :]

            assert predictions.shape == data[0, :, :].T.shape, print(f"Prediction shape {predictions.shape} does not match data shape {data.T.shape}")
            axs[i+2].imshow(predictions, cmap='twilight_shifted', vmin=0, vmax=1, aspect='auto')
            axs[i+2].set_title('Prediction (sandeel)')

        plt.xlabel('Ping time')
        plt.show()

    def estimate_seabed(self, raw_file=None, save_to_file=True):
        """ Return, load or calculate seabed for entire reader"""
        if self.seabed_dataset is not None:
            if raw_file is None:
                return self.seabed_dataset.seabed
            else:
                return self.seabed_dataset.seabed.where(self.seabed_dataset.raw_file == raw_file, drop=True).astype(int).values
        elif os.path.isdir(self.seabed_path):
            self.seabed_dataset = xr.open_zarr(self.seabed_path)
            return self.get_seabed(raw_file)
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
                                  coords={'frequency': data.frequency,
                                          'ping_time': data.ping_time,
                                          'raw_file': ("ping_time", data.raw_file)})
            
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
            self.seabed_dataset = xr.Dataset(data_vars={'seabed': s.astype(int)}, coords={'ping_time': s.ping_time})

            # save to zarr file
            if save_to_file:
                self.seabed_dataset.to_zarr(self.seabed_path)
            return self.get_seabed(raw_file=raw_file)

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


def get_zarr_files(years='all', frequencies=[18, 38, 120, 200], minimum_shape=256, path_to_zarr_files=None):
    if path_to_zarr_files is None:
        path_to_zarr_files = paths.path_to_zarr_files()

    zarr_files = sorted([z_file for z_file in glob(path_to_zarr_files + '/**/*sv.zarr', recursive=True)])
    assert len(zarr_files) > 0, f"No survey data found at {path_to_zarr_files}"
    zarr_readers = [DataReaderZarr(zarr_file) for zarr_file in zarr_files]

    # Filter on frequencies
    zarr_readers = [z for z in zarr_readers if all([f in z.frequencies for f in frequencies])]

    # Filter on years
    if years == 'all':
        return zarr_readers
    else:
        assert type(years) is list, f"Uknown years variable format: {type(years)}"
        zarr_readers = [reader for reader in zarr_readers if reader.year in years]

    return zarr_readers

def get_echograms(years='all', path_to_echograms=None, frequencies=[18, 38, 120, 200], minimum_shape=256):
    """ Returns all the echograms for a given year that contain the given frequencies"""

    if path_to_echograms is None:
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

def get_data_readers(years='all', frequencies=[18, 38, 120, 200], minimum_shape=50, mode='zarr'):
    if mode == 'memm':
        return get_echograms(years=years, frequencies=frequencies, minimum_shape=minimum_shape)
    elif mode == 'zarr':
        return get_zarr_files(years, frequencies, minimum_shape)

if __name__ == '__main__':
    pass


