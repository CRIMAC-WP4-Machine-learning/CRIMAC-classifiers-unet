""""
The U-Net is trained using (somewhat) randomly selected data patches.
The patches are selected using various samplers, which each sample data with a particular characteristic.
"""


import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from batch.samplers.background import BackgroundZarr
from batch.samplers.school import SchoolZarr
from batch.samplers.seabed import SeabedZarr
from batch.dataset import Dataset
from data.echogram import DataReaderZarr

from batch.data_transforms.remove_nan_inf import remove_nan_inf
from batch.data_transforms.db_with_limits import db_with_limits
from batch.label_transforms.convert_label_indexing import convert_label_indexing
from batch.label_transforms.refine_label_boundary import refine_label_boundary
from batch.combine_functions import CombineFunctions
from data.normalization import db



def plot_patch(data, labels, transform=db):
    cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
    boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
    norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)
    frequencies = [18, 38, 120, 200]

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))

    axs[0].imshow(labels, cmap=cmap_labels, norm=norm_labels)
    axs[0].set_title('Labels')


    if transform is not None:
        data = transform(data)

    # Plot data
    for i in range(data.shape[0]):
        # plot 38 kHz frequency
        axs[i+1].imshow(data[i, :, :], cmap='jet', vmax=0, vmin=-75)
        axs[i+1].set_title(f'{frequencies[i]} kHz')
    plt.show()


if __name__ == '__main__':
    data_path = ""

    # Load zarr data using the DataReader class
    zarr_readers = [DataReaderZarr(data_path)]

    # Define data samplers
    window_dim = 256
    window_size = [256, 256]
    frequencies = np.array([18, 38, 120, 200]) * 1000

    samplers = [
        BackgroundZarr(zarr_readers, window_size),
        SeabedZarr(zarr_readers, window_size),
        SchoolZarr(zarr_readers, window_size, 27),
        SchoolZarr(zarr_readers, window_size, 1),
    ]

    # Define data augmentation
    data_augmentation = None
    label_transform = None
    data_transform = None

    # Define dataset
    dataset = Dataset(
        samplers=samplers,
        window_size=window_size,
        frequencies=frequencies,
        n_samples=1000,
        sampler_probs=[1],
        augmentation_function=data_augmentation,
        label_transform_function=label_transform,
        data_transform_function=data_transform)

    # Get random data patch
    data, labels = dataset[np.random.randint(1000)]
    plot_patch(data, labels, transform=db)



    ### Add label transform
    label_transform = CombineFunctions([convert_label_indexing, refine_label_boundary()])
    data_transform = CombineFunctions([remove_nan_inf, db_with_limits])

    # Define dataset
    dataset = Dataset(
        samplers=samplers,
        window_size=window_size,
        frequencies=frequencies,
        n_samples=1000,
        sampler_probs=[1],
        augmentation_function=data_augmentation,
        label_transform_function=label_transform,
        data_transform_function=data_transform)

    # Get random data patch
    data, labels = dataset[np.random.randint(1000)]
    plot_patch(data, labels, transform=None) # Data is already decibel transformed





