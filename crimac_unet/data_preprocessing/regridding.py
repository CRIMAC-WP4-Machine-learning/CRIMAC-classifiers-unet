import iris
from iris.coords import DimCoord
import numpy as np
import os
import ntpath
import pickle
import scipy
import time

from data.echogram import get_echograms, Echogram

_data_dtype = 'float32'
_label_dtype = 'int16'  # Max ac category is less than 10000. Int16 covers \pm 32767
_overwrite = True


def regrid_data(data, old_dims, new_dims, regridder=None):
    """
    :param data: data to be regridded, 2D or 3D
    :param old_dims: old data dimensions (list of Iris DimCoord)
    :param new_dims: new data dimensions (list of Iris DimCoord)
    :param regridder: iris regrid algorithm
    :return:
    """
    orig_cube = iris.cube.Cube(data, dim_coords_and_dims=old_dims)
    grid_cube = iris.cube.Cube(np.zeros([coord[0].shape[0] for coord in new_dims]), dim_coords_and_dims=new_dims)

    try:
        orig_cube.coord('projection_y_coordinate').guess_bounds()
        orig_cube.coord('projection_x_coordinate').guess_bounds()
        grid_cube.coord('projection_y_coordinate').guess_bounds()
        grid_cube.coord('projection_x_coordinate').guess_bounds()
    except ValueError:
        pass

    if regridder is None:
        regridder = iris.analysis.AreaWeighted(mdtol=1)
    regrid = orig_cube.regrid(grid_cube, regridder)
    return regrid.data

# TODO: rewrite to regrid 3D predictions
def regrid_prediction(pred, ech, pred_range_diff=None, pred_ping_rate=None):
    """
    Regrid prediction to the original echogram resolution
    :param pred: predictions
    :param ech: original echogram
    :return: regridded predictions
    """
    # Original
    time_vector = ech.time_vector
    range_vector = ech.range_vector

    # Negative time difference will mess up algorithm -> remove pings where this is the case
    idx = np.argwhere(time_vector[1:] - time_vector[:-1] < 0)
    time_vector = np.delete(time_vector, idx+1)

    # Original grid
    original_dims = [(DimCoord(ech.range_vector, standard_name='projection_y_coordinate', units='meter'), 0),
                     (DimCoord(time_vector, standard_name='projection_x_coordinate', units='s'), 1)]

    # Predictions grid
    if pred_ping_rate is not None:
        time_diff = ping_rate_to_time_difference(pred_ping_rate)
        time_vector = np.arange(time_vector[0], time_vector[-1], step=time_diff)

    if pred_range_diff is not None:
        range_vector = np.arange(range_vector[0], range_vector[-1], step=pred_range_diff)

    prediction_dimensions = [(DimCoord(range_vector, standard_name='projection_y_coordinate', units='meter'), 0),
                             (DimCoord(time_vector, standard_name='projection_x_coordinate', units='s'), 1)]

    # regrid predictions to original grid
    data_regridder = iris.analysis.Nearest() # consider another regridding algo?
    regridded_predictions = regrid_data(pred, prediction_dimensions, original_dims, regridder=data_regridder)

    return regridded_predictions

def regrid_predictions(pred_path, save_path, model_name, regridded_range_diff=0.5, regridded_ping_diff=None):
    # Find all predictions for model name
    rel_echograms = []
    for ech in os.listdir(pred_path):
        if os.path.isdir(os.path.join(*[pred_path, ech, model_name])):
            rel_echograms.append(ech)

    for ech in rel_echograms:
        # Load prediction
        pred = np.load(os.path.join(*[pred_path, ech, 'cons_regrid_rd_50cm_midy_nr1', 'pred.npy']))

        # Load original echogram
        echogram = Echogram(os.path.join("/lokal_uten_backup/pro/COGMAR/acoustic_new_5/memmap/", ech))

        # Regrid to original echogram dimensions
        regridded_pred = np.array(regrid_prediction(pred, echogram, pred_range_diff=regridded_range_diff,
                                            pred_ping_rate=regridded_ping_diff))


        # save
        if not os.path.isdir(os.path.join(*[save_path, ech, model_name])):
            os.makedirs(os.path.join(*[save_path, ech, model_name]))
        np.save(os.path.join(*[save_path, ech, model_name, 'pred.npy']), regridded_pred)


def ping_rate_to_time_difference(ping_rate):
    """ Ping rate (pings/s) to time_difference (in days) """
    return np.power(float(ping_rate), -1) / (24 * 60 * 60)

def fix_pings_and_regrid(ech, ping_rate=1, range_diff=None, save_dir=None, data_regridder=None,
                             label_regridder=None):

    time_vector = ech.time_vector
    range_vector = ech.range_vector
    frequencies = ech.frequencies

    idx = np.argwhere(time_vector[1:] - time_vector[:-1] < 0)
    time_vector = np.delete(time_vector, idx+1)

    original_dims = [(DimCoord(ech.range_vector, standard_name='projection_y_coordinate', units='meter'), 0),
                     (DimCoord(time_vector, standard_name='projection_x_coordinate', units='s'), 1),
                     (DimCoord(ech.frequencies, standard_name='sound_frequency', units='kHz'), 2)]

    # New time and range vectors
    if ping_rate is not None:
        time_diff = ping_rate_to_time_difference(ping_rate)
        time_vector = np.arange(time_vector[0], time_vector[-1], step=time_diff)

    if range_diff is not None:
        range_vector = np.arange(range_vector[0], range_vector[-1], step=range_diff)

    # Iris dimensions setup
    new_dims = [(DimCoord(range_vector, standard_name='projection_y_coordinate', units='meter'), 0),
                (DimCoord(time_vector, standard_name='projection_x_coordinate', units='s'), 1),
                (DimCoord(ech.frequencies, standard_name='sound_frequency', units='kHz'), 2)]

    # Regrid data and labels
    if data_regridder is None:
        data_regridder = iris.analysis.AreaWeighted(mdtol=1)
    if label_regridder is None:
        label_regridder = iris.analysis.Nearest()

    # t0 = time.time()
    data = np.delete(ech.data_numpy(), idx + 1, 1)
    labels = np.delete(ech.label_numpy(), idx + 1, 1)

    regridded_data = regrid_data(data, original_dims, new_dims, regridder=data_regridder)
    regridded_labels = regrid_data(labels, original_dims[:-1], new_dims[:-1], regridder=label_regridder)
    # print('Regridded date in ', time.time()-t0)

    save_path = os.path.join(save_dir, ech.name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # save data
    for i, freq in enumerate(frequencies):
        save_memmap(regridded_data[:, :, i], os.path.join(save_path, 'data_for_freq_' + str(int(freq))), _data_dtype,
                    _overwrite)

    # save labels
    save_memmap(regridded_labels, os.path.join(save_path, 'labels_heave'), _label_dtype, _overwrite)

    # save metadata
    save_pickle(frequencies, 'frequencies', save_path)
    save_pickle(time_vector, 'time_vector', save_path)
    save_pickle(range_vector, 'range_vector', save_path)
    save_pickle(_data_dtype, 'data_dtype', save_path)
    save_pickle(_label_dtype, 'label_dtype', save_path)
    save_pickle(np.zeros_like(time_vector), 'heave', save_path)  # TODO heave file that is not a dummy
    save_pickle(regridded_labels.shape, 'shape', save_path)
    # print('Saved data in ', time.time()-t1)

    # Create and save objects pickle
    objects = []
    indexes = np.indices(regridded_labels.shape).transpose([1, 2, 0])
    for fish_type_ind in np.unique(regridded_labels):
        if fish_type_ind != 0 and fish_type_ind != -100:

            # Do connected components analysis
            labeled_img, n_components = scipy.ndimage.label(regridded_labels == fish_type_ind)

            # Loop through components
            for i in range(1, n_components + 1):
                object = {}

                # Collect indexes for component
                indexes_for_components = indexes[labeled_img == i]

                # Collect data + metadata
                object['fish_type_index'] = fish_type_ind
                object['indexes'] = indexes_for_components
                object['n_pixels'] = indexes_for_components.shape[0]
                object['bounding_box'] = [np.min(indexes_for_components[:, 0]), np.max(indexes_for_components[:, 0]),
                                          np.min(indexes_for_components[:, 1]), np.max(indexes_for_components[:, 1])]
                area_of_bounding_box = (object['bounding_box'][1] - object['bounding_box'][0] + 1) * (
                        object['bounding_box'][3] - object['bounding_box'][2] + 1)
                object['labeled_as_segmentation'] = area_of_bounding_box != object['n_pixels']

                objects.append(object)

    save_pickle(objects, 'objects', save_path)


def regrid_and_save_echogram(ech, ping_rate=1, range_diff=None, save_dir=None, data_regridder=None,
                             label_regridder=None):
    """
    Regrid an echogram and save files
    NB! heave vector is a dummy file with only zeros
    """

    time_vector = ech.time_vector
    range_vector = ech.range_vector
    frequencies = ech.frequencies

    # This throws an error in DimCoord, it must be strictly monotonic
    if any((time_vector[1:] - time_vector[:-1]) <= 0):
        print('Failed to interpolate', ech.name)
        print('Check time vector')
        return 0
    if any((range_vector[1:] - range_vector[:-1]) <= 0):
        print('Failed to interpolate', ech.name)
        print('Check range vector')
        return 0

    # New time and range vectors
    if ping_rate is not None:
        time_diff = ping_rate_to_time_difference(ping_rate)
        time_vector = np.arange(time_vector[0], time_vector[-1], step=time_diff)

    if range_diff is not None:
        range_vector = np.arange(range_vector[0], range_vector[-1], step=range_diff)

    # Iris dimensions setup
    original_dims = [(DimCoord(ech.range_vector, standard_name='projection_y_coordinate', units='meter'), 0),
                     (DimCoord(ech.time_vector, standard_name='projection_x_coordinate', units='s'), 1),
                     (DimCoord(ech.frequencies, standard_name='sound_frequency', units='kHz'), 2)]
    new_dims = [(DimCoord(range_vector, standard_name='projection_y_coordinate', units='meter'), 0),
                (DimCoord(time_vector, standard_name='projection_x_coordinate', units='s'), 1),
                (DimCoord(ech.frequencies, standard_name='sound_frequency', units='kHz'), 2)]

    # Regrid data and labels
    if data_regridder is None:
        data_regridder = iris.analysis.AreaWeighted(mdtol=1)
    if label_regridder is None:
        label_regridder = iris.analysis.Nearest()

    #t0 = time.time()
    regridded_data = regrid_data(ech.data_numpy(), original_dims, new_dims, regridder=data_regridder)
    regridded_labels = regrid_data(ech.label_numpy(), original_dims[:-1], new_dims[:-1], regridder=label_regridder)
    #print('Regridded date in ', time.time()-t0)

    # save data
    t1 = time.time()

    save_path = os.path.join(save_dir, ech.name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for i, freq in enumerate(frequencies):
        save_memmap(regridded_data[:, :, i], os.path.join(save_path, 'data_for_freq_' + str(int(freq))), _data_dtype,
                    _overwrite)

    # save labels
    save_memmap(regridded_labels, os.path.join(save_path, 'labels_heave'), _label_dtype, _overwrite)

    # save metadata
    save_pickle(frequencies, 'frequencies', save_path)
    save_pickle(time_vector, 'time_vector', save_path)
    save_pickle(range_vector, 'range_vector', save_path)
    save_pickle(_data_dtype, 'data_dtype', save_path)
    save_pickle(_label_dtype, 'label_dtype', save_path)
    save_pickle(np.zeros_like(time_vector), 'heave', save_path)  # TODO heave file that is not a dummy
    save_pickle(regridded_labels.shape, 'shape', save_path)
   # print('Saved data in ', time.time()-t1)

    # Create and save objects pickle
    objects = []
    indexes = np.indices(regridded_labels.shape).transpose([1, 2, 0])
    for fish_type_ind in np.unique(regridded_labels):
        if fish_type_ind != 0 and fish_type_ind != -100:

            # Do connected components analysis
            labeled_img, n_components = scipy.ndimage.label(regridded_labels == fish_type_ind)

            # Loop through components
            for i in range(1, n_components + 1):
                object = {}

                # Collect indexes for component
                indexes_for_components = indexes[labeled_img == i]

                # Collect data + metadata
                object['fish_type_index'] = fish_type_ind
                object['indexes'] = indexes_for_components
                object['n_pixels'] = indexes_for_components.shape[0]
                object['bounding_box'] = [np.min(indexes_for_components[:, 0]), np.max(indexes_for_components[:, 0]),
                                          np.min(indexes_for_components[:, 1]), np.max(indexes_for_components[:, 1])]
                area_of_bounding_box = (object['bounding_box'][1] - object['bounding_box'][0] + 1) * (
                        object['bounding_box'][3] - object['bounding_box'][2] + 1)
                object['labeled_as_segmentation'] = area_of_bounding_box != object['n_pixels']

                objects.append(object)

    save_pickle(objects, 'objects', save_path)


def save_memmap(data, path, dtype, overwrite=True):
    path = (path + '.dat').replace('.dat.dat', '.dat')
    head, tail = ntpath.split(path)
    if os.path.isfile(path) and not overwrite:
        print(' - File already exist', path)
    else:
        # print(' - Saving', path.strip(path))
        fp = np.memmap(path, dtype=dtype, mode='w+', shape=data.shape)
        fp[:] = data.astype(dtype)
        del fp


def save_pickle(data, name, out_folder):
    with open(os.path.join(out_folder, name + '.pkl'), 'wb') as f:
        pickle.dump(data, f)


def ping_rate_to_time_difference(ping_rate):
    """ Ping rate (pings/s) to time_difference (in days) """
    return np.power(float(ping_rate), -1) / (24 * 60 * 60)


def sanity_check(ech, expected_ping_rate=1):
    good = 1

    # check ping rate
    time_diff = ech.time_vector[1:] - ech.time_vector[:-1]
    ping_rates = np.power(np.array(time_diff * 24 * 60 * 60), -1)

    if any(abs(ping_rates - expected_ping_rate) > 0.05):
        good = 0

    # check energy
    regrid_data = ech.data_numpy()
    ech_orig = Echogram('/lokal_uten_backup/pro/COGMAR/acoustic_new_5/memmap/' + ech.name)
    orig_data = ech_orig.data_numpy()

    max_diff = np.nansum(orig_data, axis=(0, 1)) / 5000
    if any(abs(np.nansum(regrid_data, axis=(0, 1)) - np.nansum(orig_data, axis=(0, 1))) > max_diff):
        good = 0
    return good





if __name__ == '__main__':
    from utils.plotting import setup_matplotlib

    pred_path = "/nr/project/bild/Cogmar/usr/utseth/predictions/predictions"
    model_name = 'cons_regrid_rd_50cm_midy_nr1'
    save_path = "/nr/project/bild/Cogmar/usr/utseth/predictions/regridded_predictions/"
    regrid_predictions(pred_path, save_path, model_name, regridded_range_diff=0.5, regridded_ping_diff=None)


    # plt = setup_matplotlib()
    #
    # # Regridding and save
    # save_dir = '/lokal_uten_backup/pro/COGMAR/acoustic_new_5_pr1/memmap/'
    # ping_rate = 1
    # range_diff = None
    # years = [2017]
    #
    #
    # # Get original echograms
    # echs = get_echograms(years=years, path_to_echograms='/lokal_uten_backup/pro/COGMAR/acoustic_new_5/memmap/',
    #                      minimum_shape=256)
    # #
    # # echs_regrid = get_echograms(years=years,
    # #                      path_to_echograms=save_dir, minimum_shape=100)
    #
    # regridder_conservative = iris.analysis.AreaWeighted(mdtol=1)
    # regridder_nearest = iris.analysis.Nearest()
    # #
    # for ech in tqdm(echs):
    #     fix_pings_and_regrid(ech, ping_rate=ping_rate, range_diff=range_diff, save_dir=save_dir,
    #                               data_regridder=regridder_conservative, label_regridder=regridder_nearest)
