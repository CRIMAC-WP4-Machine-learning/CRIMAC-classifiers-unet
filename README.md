# Marine Acoustic Classification: Supervised Semantic Segmentation of Echosounder Data using CNNs

## Introduction
* The main objective of this repository is to **classify acoustic backscatter in echosounder data**.
* The current implementation is adapted to **Sandeel surveys**.
* This repository is developed by the Norwegian Computing Center and the Norwegian Institute of Marine Research as part of the research projects COGMAR and CRIMAC.

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
              "path_to_trained_model": "/dir_path/"    
              "path_for_saving_preds_labels": "/dir_path/"
              "path_for_saving_figs": "/dir_path/"
              "path_to_korona_data": "/dir_path/"
              "path_to_korona_transducer_depths": "/dir_path/"
            }
    * `"path_to_echograms"`: Directory path to echogram folders stored in memmap format (optional if working with zarr files is wished)  
    * `"path_to_zarr_files"`: Directory path to echogram folders stored in zarr format (optional if working with memmap files is wished)
    * `"path_to_trained_model"`: Directory path to the trained model    
    * `"path_for_saving_preds_labels"`: [Optional] Directory path for saving predictions and labels after training
    * `"path_for_saving_figs"`: [Optional] Directory path for saving figures related to the evaluation of the model    
    * `"path_to_korona_data"`: [Optional] Directory path to Korona predictions (only used when working with memmap files)
    * `"path_to_korona_transducer_depths"`: [Optional] Directory path to Korona transducer depths (only used when working with memmap files)

## Make predictions with a trained model and save the results
1. Set the following configuration options in the pipeline_config.yaml file:    
    * 'data_mode' can be either 'zarr' (if working with zarr files) or 'memm' (if working with memmap files)
    * 'frequencies' should be a list of frequencies the model is trained on ([18, 38, 120, 200] for the model from Olav's paper). 
    * 'unit_frequency' should be set to 'Hz' if 'zarr' mode is selected and to 'kHz' if 'memm' is selected    
    * 'partition_predict' can be 'selected surveys', 'single survey' or 'all surveys'
    * 'selected_surveys' should be a list of the names of the selected surveys. Should not be an empty list if the previous parameter ('partition_predict') is 'selected surveys' OR 'single survey'.
    * 'dir_save_preds_labels': should not be None
    * 'save_labels': if set to True the labels are assumed to exist (option 'labels_available'=True) and they will also be saved to disk. Otherwise the labels will not be saved.
    * 'eval_mode' can be 'all' (Consider all pixels), 'region' (Exclude all pixels not in a neighborhood of a labeled School) or 'fish' (only evaluate the discrimination on species). Note that the saved labels will look different depending on the chosen configuration parameter 'eval_mode'.
    * 'resume_writing': if set to True it is assumed that a zarr directory of predictions exists and if new raw files are detected, predictions will be appended to the zarr directory

2. Run the following program: `/pipeline_train_predict/save_predict.py`
3. The program will then make predictions with the trained model and save the predictions (and labels) to disk (possibility to use path_for_saving_preds_labels indicated in `setpyenv.json`)

## Make predictions with a trained model without saving the results
1. Set the following configuration options in the pipeline_config.yaml file:
    * 'data_mode' can be either 'zarr' (if working with zarr files) or 'memm' (if working with memmap files)
    * 'frequencies' should be a list of frequencies the model is trained on ([18, 38, 120, 200] for the model from Olav's paper). 
    * 'unit_frequency' should be set to 'Hz' if 'zarr' mode is selected and to 'kHz' if 'memm' is selected    
    * 'partition_predict' can be 'selected surveys', 'single survey' or 'all surveys'
    * 'selected_surveys' should be a list of the names of the selected surveys. Should not be an empty list if the previous parameter ('partition_predict') is 'selected surveys' OR 'single survey'.
    * 'labels_available' can be set to True if the labels are wished to be visualized with the predictions
  
2. Run the following program: `/pipeline_train_predict/predict.py`
3. The program will then make predictions with the trained model and visualize the output (without saving the predictions),
together with a couple of data frequency channels.
 
## Evaluate the quality of the predictions obtained with a trained model
1. Set the following configuration options in the pipeline_config.yaml file:  
    * 'data_mode' can be either 'zarr' (if working with zarr files) or 'memm' (if working with memmap files)
    * 'partition_predict' can be 'selected surveys', 'single survey' or 'all surveys'.
    * 'selected_surveys' should be a list of the names of the selected surveys. Should not be an empty list if the previous parameter ('partition_predict') is 'selected surveys' OR 'single survey'.
    Should be an empty list if 'partition_predict' is set to 'all surveys'
    * 'color_list' should be a list of color strings that will be used for the precision-recall curves. Cannot be empty if 'zarr' format is selected
    * 'eval_mode' can be 'all' (Consider all pixels), 'region' (Exclude all pixels not in a neighborhood of a labeled School) or 'fish' (only evaluate the discrimination on species)

2. Run the following program: `/pipeline_train_predict/evaluate.py`
3. The program will then compute and plot evaluation metrics for assessing the quality of the predictions obtained with a trained model.
The results will be saved to disk (possibility to use path_for_saving_figs indicated in `setpyenv.json`).

## Train the model
1. Set the following configuration options:    
    * 'data_mode' should be 'memm'
    * 'unit_frequency' should be set to 'kHz' (as 'memm' is selected)       
    * [Optional] Change hyper-parameters (lr, lr_reduction, data partition, etc.)
    
2. Run the following program: `/pipeline_train_predict/train.py`
3. The program will train the model and store the parameters to disk (possibility to use path_to_trained_model indicated in `setpyenv.json`).

NB: The training procedure is not yet adjusted to the pre-processed 'zarr' data 
since the format of the labels may continue changing and sampling the data for training depends on this.

# Using docker for predictions

The predictions can be run from docker.

## Options to run

1. Four directories need to be mounted:

    1. `/datain` should be mounted to the data directory where the preprocessed data files are located.
    2. `/model` modelweights
    3. `/dataout` should be mounted to the directory where the zarr prediction masks are written.

2. Select model weights file name

    ```bash
    --env MODELWEIGTS=regriddingPaper_v1_baseline.pt
    ```

## Example

```bash
export SURVEY='S2019847'
export DATAIN='/localscratch_hdd/crimac'
export DATAFILE='/2019/S2019847_0511/ACOUSTIC/GRIDDED/S2019847_0511_sv.zarr'
export MODELWEIGTS='regriddingPaper_v1_baseline.pt'

docker run --pgus all -it --rm --name unetpredictions:latest
-v "/localscratch_hdd/crimac/":/datain
-v "/localscratch_hdd/nilsolav/modelweights":/model
-v "/localscratch_hdd/nilsolav/"/:/dataout
--security-opt label=disable
--env DATA_INPUT_NAME="${SURVEY}_sv.zarr"
--env PRED_OUTPUT_NAME="${SURVEY}_labels_2.zarr"
unetprediction

```

```bash
docker run -rm -it --name reportgenerator -v "$SURVEYDIR/ACOUSTIC/GRIDDED":/datain -v "$SURVEYDIR/ACOUSTIC/GRIDDED":/predin -v "$TMPSURVEY/ACOUSTIC/REPORTS"/:/dataout --security-opt label=disable --env DATA_INPUT_NAME="${SURVEY}_sv.zarr" --env PRED_INPUT_NAME="${SURVEY}_labels.zarr" --env OUTPUT_NAME="${SURVEY}_report_0.zarr" --env WRITE_PNG="${SURVEY}_report_0.png" --env THRESHOLD=0.8 --env MAIN_FREQ=38000 --env MAX_RANGE_SRC=500 --env HOR_INTEGRATION_TYPE=ping --env HOR_INTEGRATION_STEP=100 --env VERT_INTEGRATION_TYPE=range --env VERT_INTEGRATION_STEP=10 reportgenerator
```
