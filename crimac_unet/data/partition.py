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

import numpy as np
from data.data_reader import get_data_readers
from batch.samplers.background import Background, BackgroundZarr
from batch.samplers.seabed import Seabed, SeabedZarr
from batch.samplers.school import School, SchoolZarr
from batch.samplers.school_seabed import SchoolSeabed, SchoolSeabedZarr
from batch.samplers.gridded import Gridded


# TODO create parent data partition object
class DataMemm:
    """  Partition memmap data into training, test and validation datasets """

    def __init__(self, frequencies, patch_size, partition_train, train_surveys, validation_surveys,
                 partition_predict, evaluation_surveys, save_prediction_surveys, eval_mode,
                 patch_overlap=20,
                 **kwargs):
        self.frequencies = frequencies
        self.window_size = patch_size   # height, width

        # Get list of all memmap data readers (Echograms)
        self.readers = get_data_readers(
            frequencies=self.frequencies,
            minimum_shape=self.window_size[0],
            mode="memm",
        )
        self.partition_train = partition_train  # Random, selected or all
        self.train_surveys = train_surveys  # List of surveys used for training
        self.validation_surveys = validation_surveys  # List of surveys used for testing
        
        # Evaluation / inference
        self.partition_predict = partition_predict
        self.evaluation_surveys = evaluation_surveys
        self.save_prediction_surveys = save_prediction_surveys # List of surveys for which to save predictions
        self.eval_mode = eval_mode
        self.patch_overlap = patch_overlap

    # Partition data into train, test, val
    def partition_data_train(self):
        """
        Choose partitioning of data
        :param echograms: list of echogram objects
        :partition: The different options are: 'random' OR 'year' OR 'single year' OR 'all years'
        :param portion_train: portion of training in the train-test split
        :return echograms used in the training and validation sets during training.

        Regarding the partition options:
        - 'random': random train-test split
        - 'selected surveys': uses specific training years (see Olav's paper) and specific validation year
        - 'all surveys': uses all available data for training and specific validation year
        """

        assert self.partition_train in [
            "random",
            "selected surveys",
            "all surveys",
        ], "Parameter 'partition' must equal 'random' or 'selected surveys' or 'single survey' or 'all surveys'"

        if self.partition_train == "random":
            portion_train = 0.85  # Set aside 85% of the echograms for training

            # Random partition of all echograms
            # Set random seed to get the same partition every time
            np.random.seed(seed=10)
            np.random.shuffle(self.readers)
            train = self.readers[: int(portion_train * len(self.readers))]
            test = self.readers[int(portion_train * len(self.readers)):]

            # Reset random seed to generate random crops during training
            np.random.seed(seed=None)

        # Partition of echograms based on years, as specified in configuration
        elif self.partition_train == "selected surveys":
            train = [reader for reader in self.readers if reader.year in self.train_surveys]
            test = [reader for reader in self.readers if reader.year in self.validation_surveys]

        elif self.partition_train == "all surveys":
            # Partition by year of echogram
            train = self.readers
            test = [reader for reader in self.readers if reader.year == 2017]
        else:
            print("Parameter 'partition' must equal 'random' or 'selected surveys' or 'single survey' or 'all surveys'")

        training_surveys = list(np.unique([reader.year for reader in train]))
        test_surveys = list(np.unique([reader.year for reader in test]))
        print(f"\n   Training surveys: {training_surveys}, {len(train)} echograms")
        print(f"   Training surveys: {test_surveys}, {len(test)} echograms\n")
        return train, test

    # TODO Consider a more intuitive function name
    def get_samplers_train(self, readers_train=None, readers_test=None):
        """
        Provides a list of the samplers used to draw samples for training and validation
        :return list of the samplers used to draw samples for training,
        list of the samplers used to draw samples for validation and
        list of the sampling probabilities awarded to each of the samplers
        """
        if readers_train is None or readers_test is None:
            readers_train, readers_test = self.partition_data_train()

        # Use random samplers for training
        samplers_train = [
            Background(readers_train, self.window_size),
            Seabed(readers_train, self.window_size),
            School(readers_train, self.window_size, 27),
            School(readers_train, self.window_size, 1),
            SchoolSeabed(
                readers_train,
                self.window_size,
                max_dist_to_seabed=self.window_size[0] // 2,
                fish_type=27,
            ),
            SchoolSeabed(
                readers_train,
                self.window_size,
                max_dist_to_seabed=self.window_size[0] // 2,
                fish_type=1,
            ),
        ]

        # Also same random samplers for testing during training
        samplers_test = [
            Background(readers_test, self.window_size),
            Seabed(readers_test, self.window_size),
            School(readers_test, self.window_size, 27),
            School(readers_test, self.window_size, 1),
            SchoolSeabed(
                readers_test,
                self.window_size,
                max_dist_to_seabed=self.window_size[0] // 2,
                fish_type=27,
            ),
            SchoolSeabed(
                readers_test,
                self.window_size,
                max_dist_to_seabed=self.window_size[0] // 2,
                fish_type=1,
            ),
        ]

        sampler_probs = [1, 5, 5, 5, 5, 5]

        assert len(sampler_probs) == len(samplers_train)

        return samplers_train, samplers_test, sampler_probs

    # TODO separate evaluation_surveys and validation_surveys
    def get_evaluation_surveys(self):
        """Get list of surveys to get predictions for / calculate evaluation metrics for"""
        if self.partition_predict == "all surveys":
            evaluation_survey_years = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018]
        elif self.partition_predict == "selected surveys":
            evaluation_survey_years = self.evaluation_surveys
        else:
            raise ValueError(f"Partition options: Options: selected surveys or all surveys - "
                             f"default: 'all surveys', not {self.partition_predict}")
        return evaluation_survey_years

    def get_gridded_survey_sampler(self, year):
        """ Create a gridded sampler which covers all data in one survey """
        surveys = [reader for reader in self.readers if reader.year == year]

        samplers = [Gridded(surveys,
                       window_size=self.window_size,
                       patch_overlap=self.patch_overlap,
                       mode=self.eval_mode)]

        return samplers

    def get_survey_readers(self, survey):
        """ Get all readers from a survey """
        return [reader for reader in self.readers if reader.year == survey]


class DataZarr:
    """  Partition zarr data into training, test and validation datasets """


    def __init__(self, frequencies, patch_size, partition_train, train_surveys, validation_surveys,
                 partition_predict, evaluation_surveys, save_prediction_surveys, eval_mode,
                 patch_overlap=20,
                 **kwargs):

        self.frequencies = sorted([freq for freq in frequencies]) # multiply by 1000 if frequency in Hz
        self.window_size = patch_size  # height, width


        # Get list of all memmap data readers (Echograms)
        # self.readers = get_data_readers(
        #     frequencies=self.frequencies,
        #     minimum_shape=self.window_size[0],
        #     mode="zarr",
        # )

        self.partition_train = partition_train  # Random, selected or all
        self.train_surveys = train_surveys  # List of surveys used for training
        self.validation_surveys = validation_surveys  # List of surveys used for testing

        # Evaluation / inference
        self.partition_predict = partition_predict
        self.evaluation_surveys = evaluation_surveys
        self.save_prediction_surveys = save_prediction_surveys # List of surveys for which to save predictions
        self.eval_mode = eval_mode
        self.patch_overlap = patch_overlap

        # print(f"{len(self.readers)} found:", [z.name for z in self.readers])

    # Partition data into train, test, val
    def partition_data_train(self):
        """
        Choose partitioning of data
        Currently only the partition 'single survey' can be used, i.e. we train and validate on the same surveys
        This should be changed in the future when the training procedure changes according to the zarr pre-processed format
        """

        assert self.partition_train in [
            "random",
            "selected surveys",
            "all surveys",
        ], "Parameter 'partition' must equal 'random' or 'selected surveys' or 'single survey' or 'all surveys'"

        # Random partition of all surveys
        if self.partition_train == "random":
            readers = get_data_readers(
                years='all',
                frequencies=self.frequencies,
                minimum_shape=self.window_size[0],
                mode="zarr",
            )

            portion_train = 0.85

            # Set random seed to get the same partition every time
            np.random.seed(seed=10)
            np.random.shuffle(readers)

            train = readers[: int(portion_train * len(readers))]
            test = readers[int(portion_train * len(readers)):]

            # Reset random seed to generate random crops during training
            np.random.seed(seed=None)

        elif self.partition_train == "selected surveys":
            train = get_data_readers(self.train_surveys, frequencies=self.frequencies,
                                     minimum_shape=self.window_size[0],
                                     mode="zarr")
            test = get_data_readers(self.validation_surveys, frequencies=self.frequencies,
                                     minimum_shape=self.window_size[0],
                                     mode="zarr")

        elif self.partition_train == "all surveys":
            train_surveys = list(range(2007, 2019))
            train = get_data_readers(train_surveys, frequencies=self.frequencies,
                                     minimum_shape=self.window_size[0],
                                     mode="zarr")
            test = [survey for survey in train if survey.year == 2017] # use 2017 survey as test after training on all

        else:
            raise ValueError(
                "Parameter 'partition' must equal 'random' or 'selected surveys' or 'single survey' or 'all surveys'"
            )

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

        print("Train: {} surveys, {} raw files, {} pings\nValidation: {} surveys, {} raw files, {} pings"
              .format(len(train), len_train, n_pings_train, len(test), len_test, n_pings_test))

        return train, test

    def get_samplers_train(self, readers_train=None, readers_test=None):
        """
        Provides a list of the samplers used to draw samples for training and validation
        :return list of the samplers used to draw samples for training,
        list of the samplers used to draw samples for validation and
        list of the sampling probabilities awarded to each of the samplers
        """
        if readers_train is None or readers_test is None:
            readers_train, readers_test = self.partition_data_train()

        samplers_train = [
            BackgroundZarr(readers_train, self.window_size),
            SeabedZarr(readers_train, self.window_size),
            SchoolZarr(readers_train, self.window_size, 27),
            SchoolZarr(readers_train, self.window_size, 1),
            SchoolSeabedZarr(
                readers_train,
                self.window_size,
                max_dist_to_seabed=self.window_size[0] // 2,
                fish_type=27,
            ),
            SchoolSeabedZarr(
                readers_train,
                self.window_size,
                max_dist_to_seabed=self.window_size[0] // 2,
                fish_type=1,
            ),
        ]

        # Also same random samplers for testing during training
        samplers_test = [
            BackgroundZarr(readers_test, self.window_size),
            SeabedZarr(readers_test, self.window_size),
            SchoolZarr(readers_test, self.window_size, 27),
            SchoolZarr(readers_test, self.window_size, 1),
            SchoolSeabedZarr(
                readers_test,
                self.window_size,
                max_dist_to_seabed=self.window_size[0] // 2,
                fish_type=27,
            ),
            SchoolSeabedZarr(
                readers_test,
                self.window_size,
                max_dist_to_seabed=self.window_size[0] // 2,
                fish_type=1,
            ),
        ]

        sampler_probs = [1, 5, 5, 5, 5, 5]

        assert len(sampler_probs) == len(samplers_train)
        assert len(sampler_probs) == len(samplers_test)

        return samplers_train, samplers_test, sampler_probs

    def get_evaluation_surveys(self):
        """Get list of surveys to get predictions for / calculate evaluation metrics for"""
        if self.partition_predict == "all surveys":
            evaluation_survey_years = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018]
        elif self.partition_predict == "selected surveys":
            evaluation_survey_years = self.evaluation_surveys
        else:
            raise ValueError(f"Partition options: Options: selected surveys or all surveys - "
                             f"default: 'all surveys', not {self.partition_predict}")
        return evaluation_survey_years

    def get_gridded_survey_sampler(self, year):
        """ Create a gridded sampler which covers all data in one survey """
        surveys = get_data_readers([year], frequencies=self.frequencies,
                                   minimum_shape=self.window_size[0],
                                   mode="zarr")

        samplers = [Gridded(surveys,
                            window_size=self.window_size,
                            patch_overlap=self.patch_overlap,
                            mode=self.eval_mode)]

        return samplers

    def get_survey_readers(self, survey):
        return get_data_readers([survey], frequencies=self.frequencies,
                                minimum_shape=self.window_size[0],
                                mode="zarr")


