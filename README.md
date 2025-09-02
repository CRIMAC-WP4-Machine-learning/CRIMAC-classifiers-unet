# Important Notice

This repository has been archived and is no longer actively maintained.

The project has moved to a new platform: https://git.imr.no/crimac-wp4-machine-learning/CRIMAC-acoustic-target-classification

Thank you for your support and contributions! Feel free to check the new repository for the latest updates and discussion.

# Marine Acoustic Classification: Supervised Semantic Segmentation of Echosounder Data using CNNs

## Introduction
* The main objective of this repository is to **classify acoustic backscatter in echosounder data**.
* The current implementation is adapted to **Sandeel surveys**.
* This repository is developed by the Norwegian Computing Center and the Norwegian Institute of Marine Research as part of the research projects CRIMAC, COGMAR and Visual Intelligence.

This repository contains scripts for:
* Preprocessing acoustic backscatter data into a machine learning friendly format.
* Supervised training of a convolutional neural network for semantic segmentation on echosounder data.
* Making predictions with the trained network.


## Prerequisites
* The code on this repository was tested using Python 3.8 and the requirements which are listed in the requirements.txt document.
* Create the file `setpyenv.json` in the local root directory:

            ### setpyenv.json ###
            ### Replace each "/dir_path/" with appropriate directory path.
            
            {
              "path_to_echograms": "/dir_path/"
              "path_to_zarr_files": "/dir_path/"          
            }
    * `"path_to_echograms"`: Directory path to echogram folders stored in memmap format (optional if working with zarr files is wished)  
    * `"path_to_zarr_files"`: Directory path to echogram folders stored in zarr format (optional if working with memmap files is wished). 

## Make predictions with a trained model and save the results
1. Set the following configuration options in the pipeline_config.yaml file:    
    * 'data_mode' can be either 'zarr' (if working with zarr files) or 'memm' (if working with memmap files)
    * 'frequencies' should be a list of frequencies the model is trained on in kHz ([18, 38, 120, 200] for the model from Olav's paper). 
    * 'save_predictions' should be a list of survey (years) to save predictions for.
    * 'preload_n_pings' is an integer indicating how many pings of data should be loaded into memory from a zarr file before running inference. Adjusting this parameter can speed up the prediction process, especially for zarr files with large chunks. However, setting this parameter too high may consume too much memory. 
    * 'selected_surveys' should be a list of the names of the selected surveys. Should not be an empty list if the previous parameter ('partition_predict') is 'selected surveys' OR 'single survey'.
    * 'resume_writing': if set to True it is assumed that a zarr directory of predictions exists and predictions will be appended to the zarr directory

2. Run the following program: `/pipeline_train_predict/save_predict.py --yaml_path [path to pipeline yaml config] --checkpoint_path [path to checkpoint model] --save_predictions_path [path to directory where predictions should be saved]`
3. The program will then make predictions with the trained model and saved. 
 
## Evaluate the quality of the predictions obtained with a trained model
1. Set the following configuration options in the pipeline_config.yaml file:  
    * 'data_mode' can be either 'zarr' (if working with zarr files) or 'memm' (if working with memmap files)
    * 'partition_predict' can be 'selected surveys', 'single survey' or 'all surveys'.
    * 'evaluation_surveys' should be a list of the names of the selected surveys. Should not be an empty list if the previous parameter ('partition_predict') is 'selected surveys' OR 'single survey'.
    Should be an empty list if 'partition_predict' is set to 'all surveys'
    * 'eval_mode' can be 'all' (Consider all pixels), 'region' (Exclude all pixels not in a neighborhood of a labeled School) or 'fish' (only evaluate the discrimination on species)
    * 'preload_n_pings' is an integer indicating how many pings of data should be loaded into memory from a zarr file before running inference. Adjusting this parameter can speed up the prediction process, especially for zarr files with large chunks. However, setting this parameter too high may consume too much memory. 

2. Run the following program: `/pipeline_train_predict/evaluate.py`
3. The program will then compute and plot evaluation metrics for assessing the quality of the predictions obtained with a trained model.
The results will be saved to disk (possibility to use path_for_saving_figs indicated in `setpyenv.json`).

## Train the model
1. Set the following configuration options:
    * 'partition_train' can be random or selected_surveys or all surveys. We recommend using selected surveys. 
    * 'train_surveys' is a list of surveys (years) to train the model on. In Olav's paper, 2011, 2013, 2014, 2015 and 2016 was used. 
    * 'validation_surveys' is a list of surveys to validate the model on during training. 
    * [Optional] Change hyper-parameters (lr, lr_reduction, data partition, etc.)
    
2. Run the following program: `/pipeline_train_predict/train.py --yaml_path [path to pipeline yaml config]`
(e.g. /pipeline_train_predict/train.py --yaml_path ../configs/config_baseline.yaml). The config filname is considered the "experiment name", which along with the timestamp of the training start is used to differentiate between different models. 
3. The program will train the model and store the parameters to disk. Note that if the option `log_step` is set to e.g. to 2500, model weights will be saved every 2500 iterations in case a better validation F1 score is achieved.
 
The model with the best F1 score on the validation set (best.pt) and the final model (last.pt) are saved in the `saved_models/[experiment_name]/[timestamp]` directory. The Tensorboard logs with training loss along with optional validation metrics can be found in the `tensorboard/[experiment_name]/[timestamp]` directory. 

# Using docker for predictions

The save_predict script can be run from docker.

## Options to run

1. Four directories need to be mounted:

    1. `/datain` should be mounted to the data directory where the preprocessed data files are located.
    2. `/model` modelweights
    3. `/dataout` should be mounted to the directory where the zarr prediction masks are written.
    4. `/configs` should be mounted to the directory where the config file(s) is (are) located. 

2. Select model weights file name, config filename, survey name (to save predictions for) and output filename (zarr file where predictions are saved)

    ```bash
    --env MODEL=[modelweights_filename].pt
    --env CONFIG=[config_name].yaml
    --env SURVEY=[survey_name]
    --env ZARRFILE=[predictions_filename].zarr
    ```

## Example
### Build docker image
From CRIMAC_classifisers-unet directory, run:
```bash
docker build --tag unet .
```
### Run 
```bash
docker run -it --rm --name unet -v "/mnt/c/DATAscratch/crimac-scratch/2019/S2019847_0511/ACOUSTIC/GRIDDED/":/datain -v "/mnt/c/DATAscratch/crimac-scratch/NR_Unet":/model -v "/mnt/c/DATAscratch/crimac-scratch/2019/S2019847_0511/ACOUSTIC/PREDICTIONS/":/dataout -v "path/to/CRIMAC_classifiers-unet/configs/":/configs --security-opt label=disable --env MODEL="paper_v2_heave_2.pt" --env SURVEY=S2019847_0511 --env CONFIG=config_baseline.yaml --ZARRFILE=S2019847_0511_predictions.zarr unet:latest
```

