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

import time
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import auc, roc_curve, precision_recall_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data.echogram import get_data_readers
from batch.dataset import Dataset
from batch.samplers.background import Background, BackgroundZarr
from batch.samplers.seabed import Seabed, SeabedZarr
from batch.samplers.school import School, SchoolZarr
from batch.samplers.school_seabed import SchoolSeabed, SchoolSeabedZarr
from batch.data_augmentation.flip_x_axis import flip_x_axis
from batch.data_augmentation.add_noise import add_noise
from batch.data_transforms.remove_nan_inf import remove_nan_inf
from batch.data_transforms.db_with_limits import db_with_limits
from batch.label_transforms.convert_label_indexing import convert_label_indexing
from batch.label_transforms.refine_label_boundary import refine_label_boundary
from batch.combine_functions import CombineFunctions

import models.unet as models

import matplotlib.pyplot as plt
from paths import *
import dask
import xarray as xr
from numcodecs import Blosc
import shutil
dask.config.set(scheduler='synchronous')


class SegPipe():
    """ Object to represent segmentation training-prediction pipeline """
    def __init__(self, opt):
        self.opt = opt
        self.model = None
        self.model_is_loaded = False

        # Data
        self.data_mode = opt.data_mode  # Zarr or memmap
        self.frequencies = opt.frequencies
        if self.frequencies == 'all':
            self.frequencies = [18, 38, 120, 200]
        if self.data_mode == 'zarr': # zarr data uses Hz rather than kHz
            self.frequencies = sorted([freq*1000 for freq in self.frequencies])
        self.window_dim = opt.window_dim
        self.window_size = [self.window_dim, self.window_dim]

        # Training
        self.device = torch.device(opt.dev if torch.cuda.is_available() else "cpu")
        self.lr = opt.lr
        self.lr_reduction = opt.lr_reduction
        self.momentum = opt.momentum
        self.test_iter = opt.test_iter
        self.log_step = opt.log_step
        self.lr_step = opt.lr_step
        self.save_model_params = opt.save_model_params

        # Inference/evaluation
        self.path_model_params = path_to_trained_model()
        self.model_name = self.path_model_params.split('/')[-1].split('.')[0]
        self.eval_mode = opt.eval_mode
        self.partition_predict = opt.partition_predict
        self.dir_savefig = path_for_saving_figs()
        self.dir_save_preds_labels = path_for_saving_preds_labels()
        self.save_labels = opt.save_labels
        self.labels_available = opt.labels_available

    def define_data_augmentation(self):
        """ Returns data augmentation functions to be applied when training """
        data_augmentation = CombineFunctions([add_noise, flip_x_axis])
        return data_augmentation

    def define_data_transform(self):
        """ Returns data transform functions to be applied when training """
        data_transform = CombineFunctions([remove_nan_inf, db_with_limits])
        return data_transform

    def define_label_transform(self):
        """ Returns label transform functions to be applied when training """
        label_transform = CombineFunctions([convert_label_indexing,
                                            refine_label_boundary(frequencies=self.frequencies,
                                                                  threshold_freq=self.frequencies[-1])])
        return label_transform

    def define_data_loaders(self, samplers_train, samplers_test, sampler_probs):
        """
        Defines data loaders for training and validation in the training process
        :param samplers_train: List of the samplers used to draw samples for training
        :param samplers_test: List of the samplers used to draw samples for validation
        :param sampler_probs: List of the sampling probabilities awarded to each of the samplers
        """
        data_augmentation = self.define_data_augmentation()
        label_transform = self.define_label_transform()
        data_transform = self.define_data_transform()

        dataset_train = Dataset(
            samplers_train,
            self.window_size,
            self.frequencies,
            self.opt.batch_size * self.opt.iterations,
            sampler_probs,
            augmentation_function=data_augmentation,
            label_transform_function=label_transform,
            data_transform_function=data_transform)

        dataset_test = Dataset(
            samplers_test,
            self.window_size,
            self.frequencies,
            self.opt.batch_size * self.opt.test_iter,
            sampler_probs,
            augmentation_function=None,
            label_transform_function=label_transform,
            data_transform_function=data_transform)

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=self.opt.batch_size,
                                      shuffle=False,
                                      num_workers=self.opt.num_workers,
                                      worker_init_fn=np.random.seed)

        dataloader_test = DataLoader(dataset_test,
                                     batch_size=self.opt.batch_size,
                                     shuffle=False,
                                     num_workers=self.opt.num_workers,
                                     worker_init_fn=np.random.seed)

        return dataloader_train, dataloader_test

    def load_model_params(self):
        """
        Loads the model with pre-trained parameters (if the params are not already loaded)
        :return: None
        """

        if self.model_is_loaded:
            pass
        else:
            assert self.model is not None
            with torch.no_grad():
                self.model.to(self.device)
                self.model.load_state_dict(torch.load(self.path_model_params, map_location=self.device))
                self.model.eval()
            self.model_is_loaded = True

    def train_model(self, data_obj, logger=None):
        """
        Model training and saving of the model at the last iteration
        :param data_obj: DataZarr or DataMemm which handles data reading and partition
        """

        # TODO add tensorboard logger
        assert not os.path.exists(self.path_model_params), \
            'Attempting to train a model that already exists: ' + self.model_name + '\n' \
                                                                                    'Use a different model name or delete the model params file: ' + self.path_model_params

        # Prepare data loaders
        print("Preparing data samplers")
        start = time.time()
        echograms_train, echograms_test = data_obj.partition_data()
        samplers_train, samplers_test, sampler_probs = data_obj.sample_data(echograms_train, echograms_test)
        print(f"Executed time for preparing samples (s): {np.round((time.time() - start), 2)}")

        print("Preparing data loaders")
        start = time.time()
        dataloader_train, dataloader_test = self.define_data_loaders(samplers_train, samplers_test, sampler_probs)
        print(f"Executed time for preparing data loaders (s): {np.round((time.time() - start), 2)}")

        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=torch.Tensor([10, 300, 250]).to(self.device))
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_reduction)

        running_loss_train = 0.0

        # Train model
        for i, (inputs_train, labels_train) in enumerate(dataloader_train):
            print("[Step %d/%d]" % (i+1, len(dataloader_train)))

            # Load train data and transfer from numpy to pytorch
            inputs_train = inputs_train.float().to(self.device)
            labels_train = labels_train.long().to(self.device)

            # Forward + backward + optimize
            self.model.train()
            optimizer.zero_grad()
            outputs_train = self.model(inputs_train)
            loss_train = criterion(outputs_train, labels_train)
            loss_train.backward()
            optimizer.step()

            # Update loss count for train set
            running_loss_train += loss_train.item()
            logger.log_scalar(tag='loss_train', value=loss_train.item(), step=i + 1)

            # Validate and log
            if (i + 1) % self.log_step == 0:
                if len(echograms_test) > 0:
                    self.validate_model(logger, i, dataloader_test, echograms_test, criterion)

            # Update learning rate every 'lr_step' number of batches
            if (i + 1) % self.lr_step == 0:
                print(i + 1)
                scheduler.step()
                print('lr:', [group['lr'] for group in optimizer.param_groups])

        print('Training complete')
        self.model_is_loaded = True

        # Save model parameters to file after training
        if self.save_model_params and self.path_model_params != None:
            torch.save(self.model.state_dict(), self.path_model_params)
            print('Trained model parameters saved to file: ' + self.path_model_params)

    def validate_model(self, logger, i, dataloader_test, readers_test, criterion):
        print('##################')
        print('Start validation')
        print('##################')

        if self.data_mode == 'zarr':
            echograms_test = []
            for reader in readers_test:
                echograms_test.extend([[reader, raw_file.replace('.raw', '')] for raw_file in reader.raw_file_included])
        else:
            echograms_test = readers_test


        preds = []
        labels = []

        running_loss_test = 0.0
        with torch.no_grad():
            self.model.eval()

            for ii, ech_input in tqdm(enumerate(echograms_test), total=len(echograms_test), desc='Validating ...'):
                #print("[Prediction step %d/%d]" % (ii + 1, len(echograms_test)))

                if self.data_mode == 'zarr':
                    reader, ech = ech_input
                    seg, label = self.predict_echogram(reader, ech, sandeel_only=True, model_train=True)
                else:
                    seg, label = self.predict_echogram(ech_input, sandeel_only=True, model_train=True)
                preds += [seg.ravel()]

                if self.eval_mode == 'region':
                    # TODO This currently works only if memm data is selected, extend to zarr
                    eval_mask = self.get_extended_label_mask_for_echogram(ech, extend_size=20)
                    # Set labels to -1 if not included in evaluation mask
                    label[eval_mask != True] = -1

                if self.eval_mode == 'trace':
                    # TODO This currently works only if memm data is selected, extend to zarr
                    eval_mask = self.get_extended_label_mask_for_echogram(ech, extend_size=20, trace_mode='trace')
                    # Set labels to -1 if not included in evaluation mask
                    label[eval_mask != True] = -1

                # 'Fish' evaluation mode: Set all background pixels to 'ignore', i.e. only evaluate the discrimination on species
                if self.eval_mode == 'fish':
                    label[label == 0] = -1

                labels += [label.ravel()]

            preds = np.hstack(preds)
            labels = np.hstack(labels)

            # Precision, recall, F1
            print("Started computing metrics")
            start = time.time()
            precision, recall, thresholds = precision_recall_curve(labels[np.where(labels != -1)],
                                                                   preds[np.where(labels != -1)],
                                                                   pos_label=1)

            F1 = 2 * (precision * recall) / (precision + recall)
            F1[np.invert(np.isfinite(F1))] = 0

            # Log metrics
            logger.log_scalar(tag='F1_score_test', value=F1[np.argmax(F1)], step=i + 1)
            logger.log_scalar(tag='precision_test', value=precision[np.argmax(F1)], step=i + 1)
            logger.log_scalar(tag='recall_test', value=recall[np.argmax(F1)], step=i + 1)
            print(f"Finished computing metrics in (min): {np.round((time.time() - start) / 60, 2)}")

            # Log pr curve
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            ax.tick_params(labelsize=6)
            ax.set_xlabel("Recall", fontsize=8)
            ax.set_ylabel("Precision", fontsize=8)
            ax.set_xticks(ticks)
            ax.scatter(recall, precision, s=2)
            ax.set_xlim(-0.06, 1.06)
            ax.set_ylim(-0.06, 1.06)
            plt.close(fig)
            name_plot = 'val_pr_curve'
            logger.log_plot(name_plot, fig, i + 1)

            # Save best model in terms of best validation F1 score
            logger.log_save_model(model=self.model, filename="model_best_F1",
                                  step=i + 1, save_on_max=F1[np.argmax(F1)])

            # Log running test loss
            # Log running_loss for train and test
            print("Started computing test loss")
            start = time.time()
            for inputs_test, labels_test in dataloader_test:
                # Load test data and transfer from numpy to pytorch
                inputs_test = inputs_test.float().to(self.device)
                labels_test = labels_test.long().to(self.device)
                outputs_test = self.model(inputs_test)

                loss_test = criterion(outputs_test, labels_test)

                # Update loss count for test set
                running_loss_test += loss_test.item()

            logger.log_scalar(tag='running_loss_test', value=running_loss_test / self.test_iter, step= i + 1)
            print(f"Finished computing test loss in (s): {np.round((time.time() - start), 2)}")

        print('#####################')
        print('Validation completed')
        print('#####################')


    def predict_echogram(self, ech, raw_file=None, sandeel_only=True, model_train=False):
        """
        Predict single echogram based on the sandeel survey
        :param ech: echogram object
        :raw_file: raw_file name in the case we are using zarr data
        :sandeel_only: if set to True only focus on the sandeel predictions.
        :return segmentation predictions and modified labels

        Currently the sandeel_only option should be set to True as there are some potential issues with the post porecssing

        """

        def _segmentation(data, patch_size, patch_overlap):
            """
            Due to memory restrictions on device, echogram is divided into patches.
            Each patch is segmented, and then stacked to create segmentation of the full echogram
            """

            def _get_patch_prediction(patch):
                '''
                Converts numpy data patch to torch tensor, predicts with model, and converts back to numpy.
                :param patch: (np.array) Input data patch
                :return: (np.array) Prediction
                '''
                patch = np.expand_dims(np.moveaxis(patch, -1, 0), 0)
                self.model.eval()
                with torch.no_grad():
                    patch = torch.Tensor(patch).float()
                    patch = patch.to(self.device)
                    patch = F.softmax(self.model(patch), dim=1).cpu().numpy()
                patch = np.moveaxis(patch.squeeze(0), 0, -1)
                return patch

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
                    pred_patch = _get_patch_prediction(data_patch)

                    # Make output array (We do this here since it will then be agnostic to the number of output channels)
                    if len(predictions) == 0:
                        predictions = np.concatenate(
                            [data[:-(patch_overlap[0] * 2), :-(patch_overlap[1] * 2), 0:1] * 0] * pred_patch.shape[2],
                            -1)

                    # Remove potential padding related to edges
                    pred_patch = pred_patch[0:patch_size[0] - pad_val_0, 0:patch_size[1] - pad_val_1, :]

                    # Remove potential padding related to overlap between data_patches
                    pred_patch = pred_patch[patch_overlap[0]:-patch_overlap[0], patch_overlap[1]:-patch_overlap[1], :]

                    # Insert output_patch in out array
                    predictions[x0:x0 + pred_patch.shape[0], x1:x1 + pred_patch.shape[1], :] = pred_patch

            return predictions

        def _post_processing(seg, ech, idxs=None):
            """ Set all predictions below seabed to zero. """
            if idxs is None:
                seabed = ech.get_seabed().copy()
            else:
                seabed = ech.get_seabed(idxs[0], n_pings=len(idxs)).copy()

            seabed += 10
            assert seabed.shape[0] == seg.shape[1]
            for x, y in enumerate(seabed):
            
                if sandeel_only == True:
                    seg[y:, x] = 0
                else:
                    seg[y:, x, 0] = 1 # Set the probability of having background to 1 below sea floor
                    seg[y:, x, 1] = 0
                    seg[y:, x, 2] = 0
            return seg

        patch_size = self.window_dim
        patch_overlap = 20

        idxs = None
        if self.data_mode == 'zarr':
            idxs = np.argwhere(ech.raw_file.values == raw_file).ravel()
            data = ech.get_data_slice(idxs[0], n_pings=len(idxs), frequencies=self.frequencies, drop_na=False)

            # Swap axis to match memm echogram
            data = np.array(data).swapaxes(1, 2)

            if self.labels_available:
                # Get modified labels
                labels = ech.get_label_slice(idxs[0], n_pings=len(idxs), drop_na=False)
                labels = np.array(labels).T  # Transpose to match memm labels

                # Set infinite values of data to 0 and associated indices of labels to ignore value -1
                labels[np.invert(np.isfinite(data[0, :, :]))] = -1
                data[np.invert(np.isfinite(data))] = 0

                # Todo: Label processing should be performed with existing class method instead (verify that it does the same thing).
                labels = convert_label_indexing(data, labels, ech)[1]
                labels = refine_label_boundary(frequencies=self.frequencies,
                                               threshold_freq=self.frequencies[-1])(data, labels, ech)[1]
                labels[labels == -100] = -1
            else:
                # Set infinite values of data to 0
                data[np.invert(np.isfinite(data))] = 0

            data = db_with_limits(data, None, None, None)[0]
        else:
            data = ech.data_numpy(frequencies=self.frequencies)
            data[np.invert(np.isfinite(data))] = 0

            if self.labels_available:
                # Get modified labels
                labels = ech.label_numpy()
                # Todo: Label processing should be performed with existing class method instead (verify that it does the same thing).
                labels = convert_label_indexing(data, labels, ech)[1]
                labels = refine_label_boundary(frequencies=self.frequencies,
                                               threshold_freq=self.frequencies[-1])(np.moveaxis(data, -1, 0), labels,
                                                                                    ech)[1]
                labels[labels == -100] = -1
            data = db_with_limits(np.moveaxis(data, -1, 0), None, None, None)[0]

        data = np.moveaxis(data, 0, -1)

        if not model_train:
            self.load_model_params()

        # Get segmentation
        if sandeel_only:
            seg = _segmentation(data, patch_size, patch_overlap)[:, :, 1]
        else:
            seg = _segmentation(data, patch_size, patch_overlap)

        # Remove sandeel predictions 10 pixels below seabed and down
        # Todo: Think about current implementation of self._post_processing
        #   Olav's additional comment on this respect: " The _post_processing applied on a 3d array - not sure what happens then.
        #   I believe that everything is set to zero below the seabed. That means that the probability of
        #   sandeel+other+background no longer sums to 1, but to 0 (below the seabed).
        #   Ideally, this should be [0, 0, 1] instead of [0, 0, 0]."
        seg = _post_processing(seg, ech, idxs)

        if self.labels_available:
            return seg, labels
        else:
            return seg

    # TODO update for zarr
    def get_extended_label_mask_for_echogram(self, ech, extend_size, raw_file=None):
        """
        Computes an evaluation mask useful when the evaluation mode is set to 'region' or 'fish'
        :param ech: echogram object
        :param extend_size: extension added around the bounding box object
        :param raw_file: raw_file name in the case we are using zarr data
        :return evaluation mask
        """
        fish_types = [1, 27]
        extension = np.array([-extend_size, extend_size, -extend_size, extend_size])

        if self.data_mode == 'zarr':
            ech_shape = ech.get_data_rawfile(raw_file=raw_file, frequencies=[self.frequencies[0]], drop_na=False).squeeze().shape[
                        ::-1]  # Reversed shape to match echogram
            eval_mask = np.zeros(shape=ech_shape,
                                 dtype=np.bool)
            raw_obj = ech.objects.where(ech.objects.raw_file == raw_file, drop=True).dropna(dim='object_length')
            ech_objects = raw_obj.object_length.values
        else:
            ech_shape = ech.shape
            eval_mask = np.zeros(shape=ech_shape, dtype=np.bool)
            ech_objects = ech.objects

        for obj in ech_objects:

            if self.data_mode == 'zarr':
                obj_type = np.array(raw_obj["fish_type_index"][obj])
            else:
                obj_type = obj["fish_type_index"]

            if obj_type.size == 0 or obj_type not in fish_types:
                continue

            if self.data_mode == 'zarr':
                bbox = np.array(raw_obj["bounding_box"])[:, obj, :].squeeze().astype(int)
            else:
                bbox = np.array(obj["bounding_box"])

            # Extend bounding box
            bbox += extension

            # Crop extended bounding box if outside of echogram boundaries
            bbox[bbox < 0] = 0
            bbox[1] = np.minimum(bbox[1], ech_shape[0])
            bbox[3] = np.minimum(bbox[3], ech_shape[1])

            # Add extended bounding box to evaluation mask
            eval_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]] = True

        return eval_mask

    def save_segmentation_predictions_memm(self, data_obj, extend_size=20):
        """
        Save segmentation predictions related to sandeel
        :param extend_size: extension added around the bounding box object useful when the evaluation mode is set to 'region' or 'fish'
        :param selected_surveys: [Optional] List of names of the selected surveys.
        :return None

        This function loops over surveys and different surveys may be considered depending on the chosen partition for the prediction.
        """
        # Get survey readers and load model
        surveys_list = data_obj.get_prediction_surveys()
        surveys_names = data_obj.surveys_predict
        self.load_model_params()

        print('Evaluation mode: ', self.eval_mode)
        assert self.eval_mode in ['all', 'fish', 'region']

        assert self.dir_save_preds_labels is not None,\
            'Provide a path to the directory for saving predictions and labels in setpyenv.json'

        with torch.no_grad():
            for j, survey_name in enumerate(surveys_names):
                print(f'Saving predictions for: {survey_name}')

                if self.data_mode == 'zarr':
                    survey = [reader for reader in surveys_list if reader.name == survey_name][0]
                    selected_echograms = [ech_name.replace('.raw', '') for ech_name in list(survey.raw_file_included)]
                else:
                    selected_echograms = [e for e in surveys_list if e.year == survey_name]

                for ii, ech in enumerate(selected_echograms):
                    print("[Step %d/%d]" % (ii+1, len(selected_echograms)))
                    print(ech, survey.name)

                    if self.data_mode == 'zarr':
                        if self.labels_available:
                            preds, labels = self.predict_echogram(survey, ech, sandeel_only=True)
                        else:
                            preds = self.predict_echogram(survey, ech, sandeel_only=True)
                        ech_name = ech

                    else:
                        if self.labels_available:
                            preds, labels = self.predict_echogram(ech, sandeel_only=True)
                        else:
                            preds = self.predict_echogram(ech, sandeel_only=True)
                        ech_name = ech.name

                    file_path_pred = self.dir_save_preds_labels + 'predictions/' + '{0}/{1}/'.format(ech_name,
                                                                                                     self.model_name)

                    if not os.path.exists(file_path_pred):
                        os.makedirs(file_path_pred)

                    if self.save_labels:
                        assert self.labels_available == True
                        file_path_label = self.dir_save_preds_labels + 'relabel_morph_close/' + '{0}/'.format(
                            ech_name)
                        if not os.path.exists(file_path_label):
                            os.makedirs(file_path_label)
                        # Todo: Verify that the labels we save here are used later with the correct label transforms.
                        # Todo: Inlcude new feature "trace" as valid "self.eval_mode".
                        #  "trace" is a hybrid of "all" and "region" mode:
                        #   All regions in "region" mode are extended vertically,
                        #   i.e. we include all the pixels in a ping if the ping would contain at least one pixel in "region" mode.
                        #   Thus we either include all or none of the pixels in each ping.
                        # Todo: Careful with saving labels in np.uint8, may be too limiting later on

                        # 'Region' evaluation mode: Exclude all pixels not in a neighborhood of a labeled School ('sandeel' or 'other')
                        if self.eval_mode == 'region':
                            # Get evaluation mask, i.e. the pixels to be evaluated
                            if self.data_mode == 'zarr':
                                eval_mask = self.get_extended_label_mask_for_echogram(survey, extend_size,
                                                                                      raw_file=ech_name)
                            else:
                                eval_mask = self.get_extended_label_mask_for_echogram(ech, extend_size)
                            # Set labels to -1 if not included in evaluation mask
                            labels[eval_mask != True] = -1
                            np.save(file_path_label + 'label_region', labels.astype(np.int8))

                        # 'Fish' evaluation mode: Set all background pixels to 'ignore', i.e. only evaluate the discrimination on species
                        if self.eval_mode == 'fish':
                            labels[labels == 0] = -1
                            np.save(file_path_label + 'label_fish', labels.astype(np.int8))

                        if self.eval_mode == 'all':
                            np.save(file_path_label + 'label', labels.astype(np.int8))

                    # Numpy save
                    np.save(file_path_pred + 'pred', preds.astype(np.float16))

    def save_segmentation_predictions_zarr(self, data_obj, resume=False):
        """
        Save segmentation predictions in zarr format
        :param extend_size: extension added around the bounding box object useful when the evaluation mode is set to 'region' or 'fish'
        :param selected_surveys: [Optional] List of names of the selected surveys.
        :return None

        This function loops over surveys and different surveys may be considered depending on the chosen partition for the prediction.
        """

        def _create_ds_predictions(survey, preds, ech_name):
            t0 = np.where(survey.time_vector.raw_file.values == ech_name)[0][0]
            t1 = np.where(survey.time_vector.raw_file.values == ech_name)[0][-1]

            # Change axis order -> category from last to first axis
            preds = np.moveaxis(preds, -1, 0).astype(np.float16)

            # Create xarray dataset
            ds = xr.Dataset({'annotation': xr.DataArray(data=preds[1:, :, :], # Save sandeel and other predictions
                                                        dims=['category', 'range', 'ping_time'],
                                                        coords={'category': [27, 1], # sandeel, other
                                                                'range': survey.range_vector,
                                                                'ping_time': survey.time_vector[t0:t1 + 1],
                                                                })

                             },
                            attrs={'description': f'{self.model_name} predictions'}
                            )
            ds.coords["raw_file"] = ("ping_time", [ech_name] * len(ds.ping_time))

            return ds

        assert self.data_mode == 'zarr',\
        "--data_mode should be 'zarr' to save predictions in zarr"

        # Get survey readers and load model
        surveys_list = data_obj.get_prediction_surveys()
        self.load_model_params()

        assert self.dir_save_preds_labels is not None,\
        'Provide a path to the directory for saving predictions and labels in setpyenv.json'

        with torch.no_grad():
            for j, survey in enumerate(surveys_list):
                print(f'Survey {survey.name}')
                selected_echs = survey.raw_file_included

                # If only one survey is selected, a target file name can be given
                if self.dir_save_preds_labels[-4:] == 'zarr' and len(surveys_list) == 1:
                    target_dname = self.dir_save_preds_labels

                # Otherwise, a target directory name is given, all predictions are saved there using original survey name as base
                else:
                    target_dname = os.path.join(self.dir_save_preds_labels, surveys_list[j].name + '_pred' + '.zarr')
                print('Saving predictions to', target_dname)

                if resume != True:
                    # Delete existing zarr dir of predictions
                    if os.path.isdir(target_dname):
                        shutil.rmtree(target_dname)
                    write_first_loop = True
                    print(f'Overwrite predictions (if they exist)')
                else:
                    assert os.path.isdir(target_dname)==True,\
                    "No predictions were performed for this survey, please set the option --resume to False"
                    write_first_loop = False
                    print(f'Trying to resume predictions')
                    predicted_raw_files = np.unique(xr.open_zarr(target_dname).raw_file)
                    selected_echs = list(list(set(predicted_raw_files) - set(selected_echs)) + list(set(predicted_raw_files) - set(selected_echs)))
                    if len(selected_echs) == 0:
                        print("Cannot resume predictions as no new raw files were detected")
                        continue

                for ii, ech in enumerate(selected_echs):
                    print("[Step %d/%d]" % (ii+1, len(selected_echs)))

                    if self.labels_available:
                        preds, labels = self.predict_echogram(survey, ech, sandeel_only=False)
                    else:
                        preds = self.predict_echogram(survey, ech, sandeel_only=False)
                    ech_name = ech

                    ds = _create_ds_predictions(survey, preds, ech_name)

                    if ds is not None:
                        # Re-chunk so that we have a full range in a chunk (zarr only)
                        ds = ds.chunk({"range": ds.range.shape[0], "ping_time": 'auto'})

                        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
                        encoding = {var: {"compressor": compressor} for var in ds.data_vars}
                        if write_first_loop == False:
                            ds.to_zarr(target_dname, append_dim="ping_time")
                        else:
                            ds.to_zarr(target_dname, mode="w", encoding=encoding)

                        write_first_loop = False


    # TODO ajust to zarr files
    def compute_and_plot_evaluation_metrics(self, data_obj, selected_surveys=[], colors_list=[], model_names=None):
        """
        Compute and plot evaluation metrics for assessing the quality of the predictions obtained with a trained model.
        This function assumes that the predictions and labels are saved before hand to disk (i.e. --dir_save_preds_labels option)
        The metrics which are computed are: F1 score, precision, recall, thresholds, roc curve and auc.
        The precision-recall and the roc curves are plotted and saved to disk when the option --dir_savefig is not None.
        :param selected_surveys: [Optional] List of names of the selected surveys.
        :param colors_list: [Optional] List of color strings for the precision-recall plots

        Similarly to the function 'save_segmentation_predictions_sandeel' there is a loop over surveys
        and different surveys may be considered depending on the chosen partition.
        """

        # Get colors
        surveys_names = data_obj.surveys_predict
        surveys_list = data_obj.get_prediction_surveys()

        if len(colors_list) == 0 or len(colors_list) != len(surveys_names):
            color_survey = dict(zip(
                surveys_names,
                ["blue"]*len(surveys_names))
            )
        else:
            color_survey = dict(zip(
                surveys_names,
                colors_list)
            )

        # If not ensemble prediction
        if model_names == None:
            model_names = [self.model_name]


        print('Get metrics')
        print('Evaluation mode: ', self.eval_mode)
        print(f'Evaluation on {surveys_names}')
        assert self.eval_mode in ['all', 'fish', 'region']

        # Initialize plot
        dpi = 400
        mm_to_inch = 1 / 25.4
        figsize_x = 170.0
        figsize_y = 0.8 * figsize_x
        # fig_pr = plt.figure(figsize=(mm_to_inch * figsize_x, mm_to_inch * figsize_y), dpi=dpi)
        # fig_roc = plt.figure(figsize=(mm_to_inch * figsize_x, mm_to_inch * figsize_y), dpi=dpi)

        # Initialize dataframe
        appended_df = []

        for j, survey in enumerate(surveys_names):
            print(f'Evaluate metrics on survey: {survey}')

            if self.data_mode == 'zarr':
                print(f'Evaluate metrics on survey: {survey}')
                selected_readers = [reader for reader in surveys_list if reader.name == survey]
                selected_echograms = []
                for reader in selected_readers:
                    selected_echograms += list(reader.raw_file_included)
            else:
                print(f'Evaluate metrics on survey: {survey}')
                selected_readers = [e for e in surveys_list if e.year == survey]
                selected_echograms = [e.name for e in selected_surveys]
            preds = []
            labels = []

            if self.dir_save_preds_labels is None:
                print(
                    'Config option --dir_save_preds_label should not be None, i.e. provide a path to the directory for saving predictions and labels')
                continue
            else:
                for ii, ech in enumerate(selected_echograms):
                    print("[Step %d/%d]" % (ii+1, len(selected_echograms)))
                    if self.data_mode == 'zarr':
                        ech_name = ech.split('.raw')[0]
                    else:
                        ech_name = ech.name

                    if self.eval_mode == 'all':
                        file_path_label = self.dir_save_preds_labels + 'relabel_morph_close/' + '{0}/'.format(
                            ech_name) + 'label.npy'

                    if self.eval_mode == 'region':
                        file_path_label = self.dir_save_preds_labels + 'relabel_morph_close/' + '{0}/'.format(
                            ech_name) + 'label_region.npy'

                    if self.eval_mode == 'fish':
                        file_path_label = self.dir_save_preds_labels + 'relabel_morph_close/' + '{0}/'.format(
                            ech_name) + 'label_fish.npy'

                    # Load label
                    label = np.load(file_path_label)

                    pred_avg = [None] * len(model_names)
                    for ii, model_name in enumerate(model_names):
                        file_path_pred = self.dir_save_preds_labels + 'regridded_predictions/' + '{0}/{1}/'.format(
                            ech_name, model_name) + "pred.npy"

                        pred = np.load(file_path_pred)

                        # If there is a dimension mismatch, most likely due to errors in time vector
                        # Time vector errors are removed in regridding. Remove time vector errors and check dimensions again
                        if label.shape != pred.shape:
                            print(label.shape, pred.shape, ech.name)
                            idx = np.argwhere(ech.time_vector[1:] - ech.time_vector[:-1] < 0)
                            label = np.delete(ech.label_numpy(), idx + 1, 1)
                            assert label.shape == pred.shape

                        pred_avg[ii] = pred.ravel()

                    pred_avg = np.array(pred_avg)

                    preds += [np.mean(pred_avg, axis=0)]
                    labels += [label.ravel()]

                print('Started preparing arrays of combined predictions and labels for evaluation')
                preds = np.hstack(preds)
                labels = np.hstack(labels)
                print('Finished preparing arrays, start computing metrics')

                # Precision, recall, F1
                start = time.time()
                precision, recall, thresholds = precision_recall_curve(labels[np.where(labels != -1)],
                                                                       preds[np.where(labels != -1)],
                                                                       pos_label=1)

                # F1 = 2 * (precision * recall) / (precision + recall)
                # F1[np.invert(np.isfinite(F1))] = 0
                # print(f"Computed pr, F1 in (min): {np.round((time.time() - start) / 60, 2)}")
                #
                # # ROC, AUC
                # start = time.time()
                # fpr, tpr, _ = roc_curve(labels[np.where(labels != -1)],
                #                         preds[np.where(labels != -1)],
                #                         pos_label=1)
                # AUC = auc(x=fpr, y=tpr)
                # print(f"Computed roc, AUC in (min): {np.round((time.time() - start) / 60, 2)}")
                #
                # print(survey, 'F1: {:.3f}, AUC:{:.3f}, Precision: {:.3f}, Recall: {:.3f}, Threshold: {:.3f}'.format(
                #     F1[np.argmax(F1)],
                #     AUC,
                #     precision[np.argmax(F1)],
                #     recall[np.argmax(F1)],
                #     thresholds[np.argmax(F1)]
                # )
                #       )

                # Write metrics to pandas df
                model_name_out = model_names[0] if len(model_names) == 0 \
                    else f"{model_names[0].split('.')[0][:-1]}_{len(model_names)}ensemble"

                df = pd.DataFrame({"Survey": [survey] * len(precision),
                                        "precision": precision,
                                        "recall": recall,
                                        })
                df.to_csv(self.dir_savefig + 'metrics_' + self.eval_mode + '_' + \
                                              model_name_out + \
                                              f'_{survey}.csv', index=False
                                              )
                # df = pd.DataFrame({'Survey': [survey],
                #                    'Model': [model_name_out],
                #                    'AUC': [AUC],
                #                    'F1': [F1[np.argmax(F1)]],
                #                    'Precision': [precision[np.argmax(F1)]],
                #                    'Recall': [recall[np.argmax(F1)]],
                #                    'Threshold': [thresholds[np.argmax(F1)]],
                #                    }
                #                   )
                #
                # appended_df.append(df)

                # Plot
                # ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                # ticks_empty = [''] * len(ticks)
                # ax_pr = fig_pr.add_subplot(3, 4, 1 + j, xlim=(0, 1), ylim=(0, 1))
                # ax_pr.set_title(survey, fontsize=8)  # , pad=5)
                # ax_roc = fig_roc.add_subplot(3, 4, 1 + j, xlim=(0, 1), ylim=(0, 1))
                # ax_roc.set_title(survey, fontsize=8)  # , pad=5)
                # if j % 4 == 0:
                #     ax_pr.set_ylabel("Precision", fontsize=8)  # , labelpad=-25)
                #     ax_pr.set_yticks(ticks)
                #     ax_roc.set_ylabel("True positive rate", fontsize=8)  # , labelpad=-25)
                #     ax_roc.set_yticks(ticks)
                # else:
                #     ax_pr.set_yticklabels(ticks_empty)
                #     ax_roc.set_yticklabels(ticks_empty)
                # if 1 + j > 8:
                #     ax_pr.set_xlabel("Recall", fontsize=8)  # , labelpad=-20)
                #     ax_pr.set_xticks(ticks)
                #     ax_roc.set_xlabel("False positive rate", fontsize=8)  # , labelpad=-20)
                #     ax_roc.set_xticks(ticks)
                # else:
                #     ax_pr.set_xticklabels(ticks_empty)
                #     ax_roc.set_xticklabels(ticks_empty)
                # ax_pr.tick_params(labelsize=6)
                # ax_pr.scatter(recall, precision, s=2, c=color_survey[survey])
                # ax_pr.set_xlim(-0.06, 1.06)
                # ax_pr.set_ylim(-0.06, 1.06)
                # ax_roc.set_xlim(-0.06, 1.06)
                # ax_roc.set_ylim(-0.06, 1.06)
                # ax_roc.tick_params(labelsize=6)
                # ax_roc.scatter(fpr, tpr, s=2, c=color_survey[survey])
                #
                # # Save and show figures
                # fig_pr.tight_layout()
                # fig_roc.tight_layout()
                if self.dir_savefig is not None:
                    if not os.path.isdir(self.dir_savefig):
                        os.makedirs(self.dir_savefig)
                #
                #     name_savefig = self.dir_savefig + 'pr_' + self.eval_mode + '_' + \
                #                    self.path_model_params.split('/')[-1].split('.pt')[0]
                #     name_savefig_roc = self.dir_savefig + 'roc_' + self.eval_mode + '_' + \
                #                        self.path_model_params.split('/')[-1].split('.pt')[0]
                #     fig_pr.savefig(fname=name_savefig + '.png', dpi=dpi)
                #     fig_roc.savefig(fname=name_savefig_roc + '.png', dpi=dpi)
                #     with open(name_savefig + '.pkl', "wb") as file:
                #         pickle.dump(fig_pr, file)
                #     with open(name_savefig_roc + '.pkl', "wb") as file:
                #         pickle.dump(fig_roc, file)
                #
                #     plt.show(block=False)

                    # Save dataframe
                    # pd.concat(appended_df).to_csv(self.dir_savefig + 'metrics_' + self.eval_mode + '_' + \
                    #                               model_name_out + \
                    #                               '.csv', index=False
                    #                               )
                # else:
                #     print(
                #         'Config option --dir_savefig should not be None if you want to see the results, i.e. provide a path to the directory for saving the evaluation results')
                #     continue



class SegPipeUNet(SegPipe):
    """ Object to represent segmentation training-prediction pipeline using the UNet model

        If we wish to test other models or include metadata in the training, it is recommended to
        create another class that also inherits the methods from the SegPipe class.
    """
    def __init__(self, opt):
        super().__init__(opt=opt)
        self.model = models.UNet(n_classes=3, in_channels=len(self.frequencies), depth=5, start_filts=64, up_mode='transpose',
                                 merge_mode='concat')


class DataMemm():
    """ Object to represent memmap data features for the training-prediction pipeline """
    def __init__(self, opt):
        self.frequencies = opt.frequencies
        self.window_dim = opt.window_dim
        self.window_size = [self.window_dim, self.window_dim]
        self.partition = opt.partition
        self.echograms = get_data_readers(frequencies=self.frequencies, minimum_shape=self.window_dim,
                                          mode=opt.data_mode)

        # Evaluation
        self.partition_predict = opt.partition_predict
        self.surveys_predict = opt.selected_surveys
        self.colors_list = opt.colors_list

    # Partition data into train, test, val
    def partition_data(self, partition='random', portion_train=0.85):
        """
        Choose partitioning of data
        :param echograms: list of echogram objects
        :partition: The different options are: 'random' OR 'year' OR 'single year' OR 'all years'
        :param portion_train: portion of training in the train-test split
        :return echograms used in the training and validation sets during training.

        Regarding the partition options:
        - 'random': random train-test split
        - 'selected surveys': uses specific training years (see Olav's paper) and specific validation year
        - 'single survey': uses only a specific year for training and validation
        - 'all surveys': uses all available data for training and specific validation year
        The hard-coding in these options (excluding 'random') may be reviewed.
        """

        if partition == 'random':
            # Random partition of all echograms

            # Set random seed to get the same partition every time
            np.random.seed(seed=10)
            np.random.shuffle(self.echograms)
            train = self.echograms[:int(portion_train * len(self.echograms))]
            test = self.echograms[int(portion_train * len(self.echograms)):]

            # Reset random seed to generate random crops during training
            np.random.seed(seed=None)

        elif partition == 'selected surveys':
            # Partition by year of echogram
            train = list(filter(lambda x: any(
                [year in x.name for year in
                 ['D2011', 'D2012', 'D2013', 'D2014', 'D2015', 'D2016']]), self.echograms))
            test = list(filter(lambda x: any([year in x.name for year in ['D2017']]), self.echograms))

        elif partition == 'all surveys':
            # Partition by year of echogram
            train = list(filter(lambda x: any(
                [year in x.name for year in
                 ['D2007', 'D2008', 'D2009', 'D2010',
                  'D2011', 'D2012', 'D2013', 'D2014', 'D2015', 'D2016',
                  'D2017', 'D2018']]), self.echograms))
            test = list(filter(lambda x: any([year in x.name for year in ['D2017']]), self.echograms))

        elif partition == 'single survey':
            # Partition by year of echogram
            train = list(filter(lambda x: any(
                [year in x.name for year in
                 ['D2017']]), self.echograms))
            test = list(filter(lambda x: any([year in x.name for year in ['D2017']]), self.echograms))

        else:
            print("Parameter 'partition' must equal 'random' or 'selected surveys' or 'single survey' or 'all surveys'")

        print('Train:', len(train), ' Test:', len(test))

        return train, test


    def sample_data(self, echograms_train=None, echograms_test=None):
        """
        Provides a list of the samplers used to draw samples for training and validation
        :return list of the samplers used to draw samples for training,
        list of the samplers used to draw samples for validation and
        list of the sampling probabilities awarded to each of the samplers
        """
        if echograms_train is None or echograms_test is None:
            echograms_train, echograms_test = self.partition_data()

        samplers_train = [
            Background(echograms_train, self.window_size),
            Seabed(echograms_train, self.window_size),
            School(echograms_train, 27),
            School(echograms_train, 1),
            SchoolSeabed(echograms_train, self.window_dim // 2, 27),
            SchoolSeabed(echograms_train, self.window_dim // 2, 1)
        ]

        samplers_test = [
            Background(echograms_test, self.window_size),
            Seabed(echograms_test, self.window_size),
            School(echograms_test, 27),
            School(echograms_test, 1),
            SchoolSeabed(echograms_test, self.window_dim // 2, 27),
            SchoolSeabed(echograms_test, self.window_dim // 2, 1)
        ]

        sampler_probs = [1, 5, 5, 5, 5, 5]

        assert len(sampler_probs) == len(samplers_train)
        assert len(sampler_probs) == len(samplers_test)

        return samplers_train, samplers_test, sampler_probs

    def get_prediction_echograms(self):
        """ Get list of surveys to get predictions for / calculate evaluation metrics for """
        if self.partition_predict == 'all surveys':
            survey_names = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018]
        if self.partition_predict == 'single survey' or self.partition_predict == 'selected surveys':
            survey_names = self.surveys_predict
        else:
            raise ValueError(f"Partition options: Options: selected surveys, single survey, all surveys - default: 'all surveys', not {self.partition_predict}")

        surveys_list = [e for e in self.echograms if e.year in survey_names]
        return surveys_list


class DataZarr():
    """ Object to represent zarr data features for the training-prediction pipeline """
    def __init__(self, opt):
        self.frequencies = sorted([freq*1000 for freq in opt.frequencies])
        self.window_dim = opt.window_dim
        self.window_size = [self.window_dim, self.window_dim]
        self.partition = opt.partition
        self.train_surveys = opt.train_surveys
        self.val_surveys = opt.val_surveys

        # TODO minimum shape is currently not used in the selection of zarr files
        self.zarr_readers = get_data_readers(frequencies=self.frequencies, minimum_shape=self.window_dim,
                                          mode=opt.data_mode)

        print(f"{len(self.zarr_readers)} found:", [z.name for z in self.zarr_readers])

        # Evaluation
        self.partition_predict = opt.partition_predict
        self.surveys_predict = opt.selected_surveys
        self.colors_list = opt.colors_list

    # Partition data into train, test, val
    def partition_data(self, portion_train=0.85):
        """
        Choose partitioning of data
        Currently only the partition 'single survey' can be used, i.e. we train and validate on the same surveys
        This should be changed in the future when the training procedure changes according to the zarr pre-processed format
        """

        assert self.partition in ['random', 'selected surveys', 'single survey', 'all surveys'], \
                "Parameter 'partition' must equal 'random' or 'selected surveys' or 'single survey' or 'all surveys'"

        # Random partition of all surveys
        if self.partition == 'random':
            # Set random seed to get the same partition every time
            np.random.seed(seed=10)
            np.random.shuffle(self.zarr_readers)

            train = self.zarr_readers[:int(portion_train * len(self.zarr_readers))]
            test = self.zarr_readers[int(portion_train * len(self.zarr_readers)):]

            # Reset random seed to generate random crops during training
            np.random.seed(seed=None)

        elif self.partition == 'selected surveys':
            train = [survey for survey in self.zarr_readers if survey.name in self.train_surveys]
            test = [survey for survey in self.zarr_readers if survey.name in self.val_surveys]
        elif self.partition == 'all surveys':
            train = [survey for survey in self.zarr_readers if survey.year in list(range(2007, 2019))]
            test = [survey for survey in self.zarr_readers if survey.year == 2017] # use 2017 survey as test after training on all
        else:
            raise ValueError(
                "Parameter 'partition' must equal 'random' or 'selected surveys' or 'single survey' or 'all surveys'")

        len_train = 0
        n_pings_train = 0
        for ii in range(len(train)):
            len_train += len(train[ii].raw_file_included)
            n_pings_train += train[ii].shape[0]

        len_test = 0
        n_pings_test = 0
        for ii in range(len(test)):
            len_test += len(test[ii].raw_file_included)
            n_pings_test += test[ii].shape[0]

        print('Train: {} surveys, {} raw files, {} pings\nTest: {} surveys, {} raw files {} pings'.
              format(len(train), len_train, n_pings_train, len(test), len_test, n_pings_test))

        return train, test

    def sample_data(self, echograms_train=None, echograms_test=None):
        """
        Provides a list of the samplers used to draw samples for training and validation
        :return list of the samplers used to draw samples for training,
        list of the samplers used to draw samples for validation and
        list of the sampling probabilities awarded to each of the samplers
        """
        if echograms_train is None or echograms_test is None:
            echograms_train, echograms_test = self.partition_data()

        samplers_train = [
            BackgroundZarr(echograms_train, self.window_size),
            SeabedZarr(echograms_train, self.window_size),
            SchoolZarr(echograms_train, self.window_size, 27),
            SchoolZarr(echograms_train,  self.window_size, 1),
            SchoolSeabedZarr(echograms_train, self.window_size, max_dist_to_seabed=self.window_size[0]//2, fish_type=27),
            SchoolSeabedZarr(echograms_train, self.window_size, max_dist_to_seabed=self.window_size[0]//2, fish_type=1)
        ]

        samplers_test = [
            BackgroundZarr(echograms_test, self.window_size),
            SeabedZarr(echograms_test, self.window_size),
            SchoolZarr(echograms_test, self.window_size, 27),
            SchoolZarr(echograms_test, self.window_size, 1),
            SchoolSeabedZarr(echograms_test, self.window_size, max_dist_to_seabed=self.window_size[0]//2, fish_type=27),
            SchoolSeabedZarr(echograms_test, self.window_size, max_dist_to_seabed=self.window_size[0]//2, fish_type=1)
        ]

        sampler_probs = [1, 5, 5, 5, 5, 5]

        assert len(sampler_probs) == len(samplers_train)
        assert len(sampler_probs) == len(samplers_test)

        return samplers_train, samplers_test, sampler_probs

    def get_prediction_surveys(self):
        """ Get list of surveys to get predictions for / calculate evaluation metrics for"""
        if self.partition_predict == 'all surveys':
            surveys_list = self.zarr_readers
        elif self.partition_predict == 'single survey' or self.partition_predict == 'selected surveys':
            assert len(self.surveys_predict) != 0
            surveys_list = [survey for survey in self.zarr_readers if survey.name in self.surveys_predict]
        else:
            raise ValueError(f"Partition options: Options: selected surveys, single survey, all surveys - default: 'all surveys', not {self.partition_predict}")

        return surveys_list


class Config_Options(object):
    """ Object to represent configuration options for the training-prediction pipeline """
    def __init__(self, configuration):
        for k, v in configuration.items():
            setattr(self, k, v)