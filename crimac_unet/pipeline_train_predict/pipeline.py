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
import pandas as pd
from sklearn.metrics import precision_recall_curve
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from paths import *
import dask
from tqdm import tqdm
import torch.multiprocessing

import models.unet as models
from constants import *

torch.multiprocessing.set_sharing_strategy('file_system')
dask.config.set(scheduler="synchronous")


class SegPipe:
    """Object to represent segmentation training-prediction pipeline"""

    def __init__(self, checkpoint_dir,
                 data_mode,
                 frequencies,
                 patch_size,
                 loss_type,
                 lr,
                 lr_reduction,
                 lr_step,
                 momentum,
                 batch_size,
                 num_workers,
                 iterations,
                 test_iter,
                 log_step,
                 save_model_params,
                 meta_channels,
                 late_meta_inject,
                 eval_mode,
                 experiment_name,
                 **kwargs):
        assert not (save_model_params and (checkpoint_dir is None))

        self.model = None
        self.model_is_loaded = False

        # Data
        self.data_mode = data_mode  # Zarr or memmap
        self.frequencies = frequencies
        if self.frequencies == "all":
            self.frequencies = [18, 38, 120, 200]
        if self.data_mode == "zarr":  # zarr data uses Hz rather than kHz
            self.frequencies = sorted([freq for freq in self.frequencies])
        self.window_size = patch_size

        # Training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_type = loss_type
        self.lr = lr
        self.lr_reduction = lr_reduction
        self.momentum = momentum
        self.lr_step = lr_step
        self.iterations = iterations
        self.test_iter = test_iter
        self.log_step = log_step
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_model_params = save_model_params
        self.checkpoint_dir = checkpoint_dir

        # Model architecture (use metadata)
        self.meta_channels = meta_channels
        self.late_meta_inject = late_meta_inject
        if len(self.meta_channels) > 0:
            self.use_metadata = True
        else:
            self.use_metadata = False

        # Inference/evaluation
        self.model_name = experiment_name
        # TODO: This should be reviewed with the new chosen convention for the name of models!
        # if self.model_name == "model_best_F1":
        #    self.model_name = Path(self.path_model_params).parts[-4]
        self.eval_mode = eval_mode

        # TODO move from initialization to function call?
        self.best_F1_val = -np.inf

    def load_model_params(self, checkpoint_path=None):
        # Todo: If we do not need ensembles, remove this
        """
        Loads the model with pre-trained parameters (if the params are not already loaded)
        :return: None
        """

        if self.model_is_loaded:
            pass
        else:
            assert self.model is not None
            if checkpoint_path is None:
                checkpoint_path = Path(self.checkpoint_dir, 'best.pt')

            with torch.no_grad():
                self.model.to(self.device)
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.device)
                )
                self.model.eval()
            print('loaded model', checkpoint_path)
            self.model_is_loaded = True

    def get_criterion(self):
        criterion = None
        # TODO: move out
        weight = torch.tensor([10., 300, 250]).to(self.device)

        if self.loss_type == "CE":
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            raise ValueError("`loss_type` not recognized")
        return criterion


    def train_model(self, dataloader_train, dataloader_test, logger=None):
        """
        Model training and saving of the model at the last iteration
        """

        assert not self.checkpoint_dir.is_dir(), f"""
            Attempting to train a model that already exists: {str(self.checkpoint_dir)}
            Use a different model name or delete the saved model params file
        """

        self.model.to(self.device)

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_reduction)
        criterion = self.get_criterion()

        # Train model
        for i, batch in tqdm(enumerate(dataloader_train), desc='Training model', total=len(dataloader_train)):
            # Load train data and transfer from numpy to pytorch
            inputs_train = batch['data'].float().to(self.device)
            labels_train = batch['labels'].long().to(self.device)

            # Forward + backward + optimize
            self.model.train()
            optimizer.zero_grad()

            if not self.late_meta_inject:
                outputs_train = self.model(inputs_train)
            else:
                outputs_train = self.model(inputs_train[:, : len(self.frequencies), :, :],
                                           inputs_train[:, len(self.frequencies):, :, :])

            loss_train = criterion(outputs_train, labels_train)
            loss_train.backward()
            optimizer.step()

            # Update loss count for train set
            logger.add_scalar(tag='train/loss', scalar_value=loss_train.item(), global_step=i + 1)

            # Validate and log validation metrics and test loss
            if (i + 1) % self.log_step == 0:
                self.validate_model_training(dataloader_test, criterion, logger, i)

            # Update learning rate every 'lr_step' number of batches
            if (i + 1) % self.lr_step == 0:
                scheduler.step()

                # Log updated learning rate(s)
                for group_idx, group in enumerate(optimizer.param_groups):
                    logger.add_scalar(tag=f'learning_rate_{group_idx}', scalar_value=group["lr"], global_step=i + 1)

        print("Training complete")
        self.model_is_loaded = True

        # Save final model
        if self.save_model_params:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.checkpoint_dir / 'last.pt'
            torch.save(self.model.state_dict(), checkpoint_path)
            print('Trained model parameters saved to file:', str(checkpoint_path))

    def predict_batch(self, batch, return_softmax=False):
        self.model.eval()
        with torch.no_grad():
            inputs = batch['data'].float().to(self.device)

            if not self.late_meta_inject:
                outputs = self.model(inputs)
            else:
                outputs = self.model(
                    inputs[:, : len(self.frequencies), :, :],
                    inputs[:, len(self.frequencies):, :, :],
                )
        if return_softmax:
            return F.softmax(outputs, dim=1)
        return outputs

    # TODO move function outside pipeline class?
    def set_label_ignore_val(self, labels):
        # Ignore areas where the grid overlaps
        labels[labels == LABEL_OVERLAP_VAL] = LABEL_IGNORE_VAL

        # Ignore areas where labels have been refined
        labels[labels == LABEL_REFINE_BOUNDARY_VAL] = LABEL_IGNORE_VAL

        # Ignore areas outside data boundary
        labels[labels == LABEL_BOUNDARY_VAL] = LABEL_IGNORE_VAL

        # Ignore species other than "sandeel" and "other". Additional categories include "juvenile sandeel",
        # which the model is not trained on.
        labels[labels == LABEL_UNUSED_SPECIES] = LABEL_IGNORE_VAL

        # When calculating loss and computing evaluation metrics, we include areas below seabed, which have no fish
        labels[labels == LABEL_SEABED_MASK_VAL] = 0

        return labels

    # TODO rename?
    def get_predictions_dataloader(self, dataloader, criterion=None, disable_tqdm=False):
        """
        Get predictions for all patches in dataloader, return predictions as vector
        """
        preds = []
        labels = []

        sum_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for ii, batch_test in tqdm(enumerate(dataloader), desc='Evaluating model', total=len(dataloader),
                                       disable=disable_tqdm):
                # Get predictions
                outputs_test = self.predict_batch(batch_test, return_softmax=False)

                if criterion is not None:
                    # Update loss count for test set
                    labels_input = batch_test['labels'].long().to(self.device)

                    # Set all ignore areas to LABEL_IGNORE_VAL or background (beneath seabed) to calculate loss
                    labels_input = self.set_label_ignore_val(labels_input)

                    loss_test = criterion(outputs_test, labels_input)
                    sum_loss += loss_test.item()

                # Do softmax, convert predictions to numpy, select sandeel channel only
                # TODO gather in post-processing step?
                preds_softmax = F.softmax(outputs_test, dim=1)
                preds_numpy = preds_softmax[:, SANDEEL].cpu().numpy()
                labels_numpy = batch_test['labels'].numpy()

                # set probability of sandeel to 0 underneath seabed
                preds += [preds_numpy.ravel()]
                labels += [labels_numpy.ravel()]

        # Gather all predictions and labels in vector
        preds = np.hstack(preds).astype(np.float16)
        labels = np.hstack(labels).astype(np.int8)

        mean_loss = sum_loss / len(dataloader)
        return labels, preds, mean_loss

    def compute_evaluation_metrics(self, labels, preds):
        precision, recall, thresholds = precision_recall_curve(y_true=labels,
                                                               probas_pred=preds,
                                                               pos_label=SANDEEL)

        # avoid RuntimeWarning: invalid value encountered in true_divide
        # https://stackoverflow.com/a/66549018
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))

        return {'precision': precision, 'recall': recall, 'thresholds': thresholds, 'F1': f1_scores}

    def select_valid_predictions(self, labels, preds):
        """ Select predictions to run validation on """
        # Set all ignore label areas to same ignore value, except areas below seabed
        labels = self.set_label_ignore_val(labels)
        idx_labels_valid = np.where(labels != LABEL_IGNORE_VAL)

        return labels[idx_labels_valid], preds[idx_labels_valid]

    def validate_model_training(self, dataloader_test, criterion, logger, iteration_no):
        """
        Code to run validation during training
        :param dataloader_test: Dataloader for testing
        :param criterion: Loss criterion used during training to compare train/test losses
        :param logger: Log validation results
        :param iteration_no: Nr of training iterations completed so far
        """

        labels, preds, loss_test = self.get_predictions_dataloader(dataloader_test, criterion=criterion)

        # Set probability of sandeel below seabed to 0
        preds[labels == LABEL_SEABED_MASK_VAL] = 0

        # Select all valid predictions (areas where labels are not marked ignore)
        labels, preds = self.select_valid_predictions(labels=labels, preds=preds)

        # Compute evaluation metrics
        metrics = self.compute_evaluation_metrics(labels=labels, preds=preds)
        F1 = metrics["F1"]
        argmax_F1 = np.argmax(F1)

        # Log metrics
        iter_step = iteration_no + 1
        logger.add_scalar(tag='test/F1_score', scalar_value=F1[argmax_F1], global_step=iter_step)
        logger.add_scalar(tag='test/precision', scalar_value=metrics["precision"][argmax_F1], global_step=iter_step)
        logger.add_scalar(tag='test/recall', scalar_value=metrics["recall"][argmax_F1], global_step=iter_step)
        logger.add_scalar(tag='test/loss', scalar_value=loss_test, global_step=iter_step)
        logger.add_pr_curve(tag='test/pr_curve', labels=labels, predictions=preds, global_step=iter_step)

        # Save best model in terms of best validation F1 score
        if F1[argmax_F1] > self.best_F1_val:
            self.best_F1_val = F1[argmax_F1]

            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.checkpoint_dir / 'best.pt'
            torch.save(self.model.state_dict(), checkpoint_path)

    def validate_model_testing(self, dataloader, save_path_metrics, save_path_plot):
        if not self.model_is_loaded:
            self.load_model_params()

        labels, preds, _ = self.get_predictions_dataloader(dataloader, disable_tqdm=False)

        # Set probability of sandeel below seabed to 0
        preds[labels == LABEL_SEABED_MASK_VAL] = 0

        # Select all valid predictions (areas where labels are not marked ignore)
        labels, preds = self.select_valid_predictions(labels=labels, preds=preds)

        # Compute evaluation metrics
        metrics = self.compute_evaluation_metrics(labels=labels, preds=preds)

        # Save metrics
        if save_path_metrics is not None:
            thresholds = np.array(list(metrics['thresholds']) + [np.nan])
            metrics['thresholds'] = thresholds
            df = pd.DataFrame(metrics)
            df.to_csv(save_path_metrics)
        if save_path_plot is not None:
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            ax.tick_params(labelsize=6)
            ax.set_xlabel("Recall", fontsize=8)
            ax.set_ylabel("Precision", fontsize=8)
            ax.set_xticks(ticks)
            ax.scatter(metrics['recall'], metrics['precision'], s=2)
            ax.set_xlim(-0.06, 1.06)
            ax.set_ylim(-0.06, 1.06)
            plt.savefig(save_path_plot)
        F1 = metrics['F1']
        print(f'F1 score: {F1[np.argmax(F1)]}')


class SegPipeUNet(SegPipe):
    """Object to represent segmentation training-prediction pipeline using the UNet model

    If we wish to test other models or include metadata in the training, it is recommended to
    create another class that also inherits the methods from the SegPipe class.
    """

    def __init__(self, checkpoint_dir=None, **kwargs):
        super().__init__(checkpoint_dir, **kwargs)
        # self.opt = opt
        if not self.late_meta_inject:
            self.model = models.UNet_Baseline(
                n_classes=3,
                in_channels=4 + get_in_channels(self.meta_channels),
                late_meta_inject=False,
                depth=5,
                start_filts=64,
                up_mode="transpose",
                merge_mode="concat",
            )

        else:
            self.model = models.UNet_LateMetInject(
                n_classes=3,
                in_channels=4,
                meta_in_channels=get_in_channels(self.meta_channels),
                late_meta_inject=True,
                depth=5,
                start_filts=64,
                up_mode="transpose",
                merge_mode="concat",
            )


def get_in_channels(meta_channels):
    if len(meta_channels) != 0:
        weights = {
            "portion_year": 1,
            "portion_day": 2,
            "depth_rel": 1,
            "depth_abs_surface": 1,
            "depth_abs_seabed": 1,
            "time_diff": 1,
        }
        return np.sum([meta_channels[kw] * weights[kw] for kw in weights.keys()])
    else:
        return 0
