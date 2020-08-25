import numpy as np
import torch
import random

from models.unet_bn_sequential_db import UNet
from data.echogram import get_echograms
from batch.label_transform_functions.index_0_1_27 import index_0_1_27
from batch.label_transform_functions.relabel_with_threshold_morph_close import relabel_with_threshold_morph_close
from batch.data_transform_functions.db_with_limits import db_with_limits

import utils.plotting
plt = utils.plotting.setup_matplotlib()  # Returns import matplotlib.pyplot as plt
import pdb
# Create segmentation from trained segmentation model (e.g. U-Net)
from predict._frameworks_Olav import get_prediction_function


def segmentation(model, data, patch_size, patch_overlap, device):
    """
    Due to memory restrictions on device, echogram is divided into patches.
    Each patch is segmented, and then stacked to create segmentation of the full echogram
    :param model:(torch.nn.Model object): segmentation model
    :param echogram:(Echogram object): echogram to predict
    :param window_dim_init: (positive int): initial window dimension of each patch (cropped by trim_edge parameter after prediction)
    :param trim_edge: (positive int): each predicted patch is cropped by a frame of trim_edge number of pixels
    :return:
    """

    pred_func = get_prediction_function(model)

    # Functions to convert between B x C x H x W format and W x H x C format
    def hwc_to_bchw(x):
        return np.expand_dims(np.moveaxis(x, -1, 0), 0)

    def bcwh_to_hwc(x):
        return np.moveaxis(x.squeeze(0), 0, -1)

    if type(patch_size) == int:
        patch_size = [patch_size, patch_size]

    if type(patch_overlap) == int:
        patch_overlap = [patch_overlap, patch_overlap]

    if len(data.shape) == 2:
        data = np.expand_dims(data, -1)

    # Add padding to avoid trouble when removing the overlap later
    data = np.pad(data, [[patch_overlap[0], patch_overlap[0]], [patch_overlap[1], patch_overlap[1]], [0, 0]],
                  'constant')

    # Loop through patches identified by upper-left pixel
    upper_left_x0 = np.arange(0, data.shape[0] - patch_overlap[0], patch_size[0] - patch_overlap[0] * 2)
    upper_left_x1 = np.arange(0, data.shape[1] - patch_overlap[1], patch_size[1] - patch_overlap[1] * 2)

    predictions = []
    for x0 in upper_left_x0:
        for x1 in upper_left_x1:
            # Cut out a small patch of the data
            data_patch = data[x0:x0 + patch_size[0], x1:x1 + patch_size[1], :]

            # Pad with zeros if we are at the edges
            pad_val_0 = patch_size[0] - data_patch.shape[0]
            pad_val_1 = patch_size[1] - data_patch.shape[1]

            if pad_val_0 > 0:
                data_patch = np.pad(data_patch, [[0, pad_val_0], [0, 0], [0, 0]], 'constant')

            if pad_val_1 > 0:
                data_patch = np.pad(data_patch, [[0, 0], [0, pad_val_1], [0, 0]], 'constant')

            # Run it through model
            out_patch = pred_func(model, hwc_to_bchw(data_patch), device)
            out_patch = bcwh_to_hwc(out_patch)

            # Make output array (We do this here since it will then be agnostic to the number of output channels)
            if len(predictions) == 0:
                predictions = np.concatenate(
                    [data[:-(patch_overlap[0] * 2), :-(patch_overlap[1] * 2), 0:1] * 0] * out_patch.shape[2], -1)

            # Remove potential padding related to edges
            out_patch = out_patch[0:patch_size[0] - pad_val_0, 0:patch_size[1] - pad_val_1, :]

            # Remove potential padding related to overlap between data_patches
            out_patch = out_patch[patch_overlap[0]:-patch_overlap[0], patch_overlap[1]:-patch_overlap[1], :]

            # Insert output_patch in out array
            predictions[x0:x0 + out_patch.shape[0], x1:x1 + out_patch.shape[1], :] = out_patch

    return predictions


def post_processing(seg, ech):

    """ Set all predictions below seabed to zero. """
    seabed = ech.get_seabed().copy()
    seabed += 10
    assert seabed.shape[0] == seg.shape[1]
    for x, y in enumerate(seabed):
        seg[y:, x] = 0
    return seg


def get_segmentation_sandeel(model, ech, freqs, device):

    patch_size = 256
    patch_overlap = 20

    data = ech.data_numpy(frequencies=freqs)
    data[np.invert(np.isfinite(data))] = 0

    # Get modified labels
    labels = ech.label_numpy()
    relabel = index_0_1_27(data, labels, ech)[1]
    relabel_morph_close = relabel_with_threshold_morph_close(np.moveaxis(data, -1, 0), relabel, ech)[1]
    relabel_morph_close[relabel_morph_close == -100] = -1

    data = db_with_limits(np.moveaxis(data, -1, 0), None, None, None)[0]
    data = np.moveaxis(data, 0, -1)

    # Get segmentation
    seg = segmentation(model, data, patch_size, patch_overlap, device)[:, :, 1]

    # Remove sandeel predictions 10 pixels below seabed and down
    seg = post_processing(seg, ech)

    return seg, relabel_morph_close


def get_extended_label_mask_for_echogram(ech, extend_size):

    fish_types = [1, 27]
    extension = np.array([-extend_size, extend_size, -extend_size, extend_size])
    eval_mask = np.zeros(shape=ech.shape, dtype=np.bool)

    for obj in ech.objects:

        obj_type = obj["fish_type_index"]
        if obj_type not in fish_types:
            continue
        bbox = np.array(obj["bounding_box"])

        # Extend bounding box
        bbox += extension

        # Crop extended bounding box if outside of echogram boundaries
        bbox[bbox < 0] = 0
        bbox[1] = np.minimum(bbox[1], ech.shape[0])
        bbox[3] = np.minimum(bbox[3], ech.shape[1])

        # Add extended bounding box to evaluation mask
        eval_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]] = True

    return eval_mask


def get_sandeel_probs_object_pathces(model, echs, freqs, n_echs, extend_size):
    '''Get sandeel predictions for all labeled schools (sandeel, other), and surrounding region'''

    _sandeel_probs = {0: [], 1: []}
    pixel_counts_year = np.zeros(3)

    for i, ech in enumerate(echs):

        if i >= n_echs:
            break

        # Get binary segmentation (probability of sandeel) and labels (-1=ignore, 0=background, 1=sandeel, 2=other)
        seg, labels = get_segmentation_sandeel(model, ech, freqs, device)

        # Get evaluation mask, i.e. the pixels to be evaluated
        eval_mask = get_extended_label_mask_for_echogram(ech, extend_size)

        # Set labels to -1 if not included in evaluation mask
        labels[eval_mask != True] = -1

        # Store sandeel predictions for negative labels ("other" and "background", background excluded outside of evaluation mask)
        _sandeel_probs[0].extend(list(seg[(labels == 0) | (labels == 2)]))

        # Store sandeel predictions for positive labels ("sandeel")
        _sandeel_probs[1].extend(list(seg[labels == 1]))

        pixel_counts_year += np.array([np.sum(labels == 0), np.sum(labels == 1), np.sum(labels == 2)])

    # From {list, list} to {ndarray, ndarray}
    _sandeel_probs[0] = np.array(_sandeel_probs[0])
    _sandeel_probs[1] = np.array(_sandeel_probs[1])

    return _sandeel_probs, pixel_counts_year


def get_sandeel_probs(model, echs, freqs, mode, n_echs):
    '''

    :param model:
    :param echs:
    :param freqs:
    :param mode: (str) "all" compares sandeel (pos) to other+background (neg). "fish" compares sandeel (pos) to other (neg).
    :param n_echs: (int) number of echograms (upper limit) per year
    :return:
    '''

    assert mode in ["all", "fish"]

    #random.shuffle(echs)
    _sandeel_probs = {0: [], 1: []}

    for i, ech in enumerate(echs):

        if i >= n_echs:
            break

        # Get binary segmentation (probability of sandeel) and labels (-1=ignore, 0=background, 1=sandeel, 2=other)
        seg, labels = get_segmentation_sandeel(model, ech, freqs, device)

        # Store sandeel predictions per pixel in list [0] (negatives) or [1] (positives) based on label.
        if mode == "all":
            # Negatives: Labels == "background" + "other" ("ignore" is excluded)
            _sandeel_probs[0].extend(list(seg[(labels == 0) | (labels == 2)]))
        elif mode == "fish":
            # Negatives: Labels == "other" ("background" + "ignore" is excluded)
            _sandeel_probs[0].extend(list(seg[labels == 2]))
        # Positives: Labels "sandeel"
        _sandeel_probs[1].extend(list(seg[labels == 1]))

    # From {list, list} to {ndarray, ndarray}
    _sandeel_probs[0] = np.array(_sandeel_probs[0])
    _sandeel_probs[1] = np.array(_sandeel_probs[1])

    return _sandeel_probs


def get_pr_curve(sandeel_probs, n_thresholds=200):

    # Get list of threshold values to compute p/r, adjusted to give evenly-ish distributed points on the p/r curve
    val_range = np.linspace(-20, 20, n_thresholds, endpoint=False)
    val_range = 1 / (1 + np.exp(-0.4 * (val_range + 3)))
    assert (np.min(val_range) >= 0) and (np.max(val_range) <= 1)

    pr_curve = []
    for value in val_range:

        tp = np.sum(sandeel_probs[1] >= value)
        fp = np.sum(sandeel_probs[0] >= value)
        fn = np.sum(sandeel_probs[1] < value)
        #tn = np.sum(sandeel_probs[0] < value)

        precision = tp / (tp + fp) if tp + fp != 0 else 1.0
        recall = tp / (tp + fn) if tp + fn != 0 else 1.0
        pr_curve.append([recall, precision])

    return np.array(pr_curve)


def plot_echograms_with_sandeel_prediction(year, device, path_model_params, ignore_mode='normal'):

    # ignore_mode == 'normal': difference between original and modified labels are changed to 'ignore'
    # ignore_mode == 'region': in addition to 'normal' mode, label 'background' is changed to 'ignore' outside of region around labeled schools

    assert ignore_mode in ['normal', 'region']

    freqs = [18, 38, 120, 200]
    echograms_all = get_echograms(frequencies=freqs)
    years_all = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018]
    echograms_year = {y: [ech for ech in echograms_all if ech.year == y] for y in years_all}
    echs = echograms_year[year]

    with torch.no_grad():

        model = UNet(n_classes=3, in_channels=4)
        model.to(device)
        model.load_state_dict(torch.load(path_model_params, map_location=device))
        model.eval()

        for i, ech in enumerate(echs):
            print(i, ech.name)

            # Get binary segmentation (probability of sandeel) and labels (-1=ignore, 0=background, 1=sandeel, 2=other)
            seg, labels = get_segmentation_sandeel(model, ech, freqs, device)

            if ignore_mode == 'region':
                # Get evaluation mask, i.e. the pixels to be evaluated
                eval_mask = get_extended_label_mask_for_echogram(ech, extend_size=20)
                # Set labels to -1 if not included in evaluation mask
                labels[eval_mask != True] = -1

            # Add two zero-channels to plot img as (R, G, B) = (p_sandeel, 0, 0)
            seg = np.expand_dims(seg, 2)
            seg = np.concatenate((seg, np.zeros((seg.shape[0], seg.shape[1], 2))), axis=2)

            # Visualize echogram with predictions
            ech.visualize(
                frequencies=[200],
                # frequencies=freqs,
                pred_contrast=5.0,
                # labels_original=relabel,
                labels_refined=labels,
                predictions=seg,
                draw_seabed=True,
                show_labels=False,
                show_object_labels=False,
                show_grid=False,
                show_name=False,
                show_freqs=True
            )


def plot_pr_curves(device, path_model_params):

    freqs = [18, 38, 120, 200]
    n_ech_per_year = 10000 # Upper limit for number of echograms per year
    echograms_all = get_echograms(frequencies=freqs, minimum_shape=256)
    years_all = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018]
    echograms_year = {y: [ech for ech in echograms_all if ech.year == y] for y in years_all}

    color_year = dict(zip(
        years_all,
        ["blue", "blue", "blue", "blue", "red", "red", "red", "red", "red", "blue", "blue"])
    )

    with torch.no_grad():

        model = UNet(n_classes=3, in_channels=4)
        model.to(device)
        model.load_state_dict(torch.load(path_model_params))
        model.eval()

        pixel_counts = np.zeros((len(years_all), 3))

        for j, year in enumerate(years_all):
            print(year)
            echs = echograms_year[year]
            assert np.all([e.year == year for e in echs])
            random.shuffle(echs)

            # Get sandeel probabilities for all echograms
            # sandeel_probs = get_sandeel_probs(model, echs, freqs, mode="all", n_echs=n_ech_per_year)
            sandeel_probs, pixel_counts_year = \
                get_sandeel_probs_object_pathces(model, echs, freqs, n_echs=n_ech_per_year, extend_size=20)

            pixel_counts[j, :] = pixel_counts_year

            # Compute precision/recall values
            pr_curve = get_pr_curve(sandeel_probs, n_thresholds=200)

            # Plot
            plt.subplot(3, 4, 1 + j)
            plt.scatter(pr_curve[:, 0], pr_curve[:, 1], s=5, c=color_year[year])
            plt.xlim(0, 1.01)
            plt.ylim(0, 1.01)
            plt.title(year)
            plt.xlabel("Recall", labelpad=-30)
            plt.ylabel("Precision", labelpad=-40)

        # Print pixel count statistics
        print(pixel_counts)
        print(pixel_counts / np.sum(pixel_counts, axis=1, keepdims=True))
        print(np.sum(pixel_counts, axis=0))
        print(np.sum(pixel_counts, axis=0) / np.sum(pixel_counts))

        plt.show()


if __name__ == "__main__":

    np.random.seed(5)
    random.seed(5)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    path_model_params = '/acosutic_deep/weights/paper_v2_heave_2.pt'

    ### PLOT ECHOGRAM WITH PREDICTIONS ###
    # This generates plots (sequentially) with echogram, labels
    # and predictions
    ### Uncomment and run script ###
    plot_echograms_with_sandeel_prediction(
        year=2018, device=device,
        path_model_params=path_model_params, ignore_mode='normal')

    ### PLOT PR CURVES ###
    # This generates a plot with one p/r curve per year,
    # evaluated on labeled schools (sandeel/other) and a surrounding region (+20 pixels) of background
    ### Uncomment and run script ###
    #plot_pr_curves(device=device, path_model_params=path_model_params)
