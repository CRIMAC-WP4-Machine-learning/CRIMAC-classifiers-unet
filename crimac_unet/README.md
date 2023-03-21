# Marine Acoustic Classification: Supervised Semantic Segmentation of Echosounder Data using CNNs

## Introduction
* The main objective of this repository is to **classify acoustic backscatter in echosounder data**.
* The current implementation is adapted to **Sandeel surveys**.
* This repository is developed by the Norwegian Computing Center and the Norwegian Institute of Marine Research as part of the research projects COGMAR and CRIMAC.

This repository contains scripts for:
* Preprocessing acoustic backscatter data into a machine learning friendly format.
* Supervised training of a convolutional neural network for semantic segmentation on echosounder data.
* Making predictions with the trained network.


## Prerequisite(s)
* The code on this repository was tested using Python 3.8 and the requirements which are listed in the requirements.txt document.

* Create the file `setpyenv.json` in the local root directory. This file contains data paths:

            ### setpyenv.json ###
            ### Replace each "/dir_path/" with appropriate directory path.
            
            {
              "path_to_echograms": "/dir_path/"
              "path_to_zarr_files": "/dir_path/"   
              "path_to_korona_data": "/dir_path/"
              "path_to_korona_transducer_depths": "/dir_path/"
            }
    * `"path_to_echograms"`: Directory path to echogram folders stored in memmap format (optional if working with zarr files is wished)  
    * `"path_to_zarr_files"`: Directory path to echogram folders stored in zarr format (optional if working with memmap files is wished)         
    * `"path_to_korona_data"`: [Optional] Directory path to Korona predictions (only used when working with memmap files)
    * `"path_to_korona_transducer_depths"`: [Optional] Directory path to Korona transducer depths (only used when working with memmap files)
    
* Go to ./configs/ and create a yaml config file, e.g. (config_baseline.yaml) according to what you want to do. 
You have the following different options:
    * train.py: train the model    
    * evaluate.py: evaluate the model in terms of precision and recall
    * save_predict.py (not tested): save predictions
    * predict.py (not tested): visualize the predictions    

NB: If you want to use metadata (early or later injection in the network using memm data),
 check out ./config/config_early_meta_inject_paper_FR22.yaml or ./config/config_early_late_inject_paper_FR22.yaml

## Train.py: Train the model
1. Set the following configuration options in the yaml file from the ./config/ directory:              
    * 'partition_train' should be a list of the names of the selected surveys for training
    * 'validation_surveys' should be a list of the names of the selected surveys for validation during training. 
    * [Optional] Change hyper-parameters (lr, lr_reduction, data partition, etc.), decide whether to train with or without metadata, 
    whether to do a fusion of the metadata in the final layer, etc.
    * [Optional] Change how often validation is run during training by changing the `log_step` variable. The model is validated every "log_step" iterations.
2. Run the following program: `/pipeline_train_predict/train.py --yaml_path [path to pipeline yaml config]`
(e.g. /pipeline_train_predict/train.py --yaml_path ../configs/config_baseline.yaml). The config filname is considered the "experiment name", which along with the timestamp of the training start is used to differentiate between different models. 
3. The program will train the model and store the parameters to disk. Note that if the option `log_step` is set to e.g. to 2500, model weights will be saved every 2500 iterations in case a better validation F1 score is achieved.
 
The model with the best F1 score (best.pt) and the final model (last.pt) are saved in the `saved_models/[experiment_name]/[timestamp]` directory. The Tensorboard logs with training loss along with optional validation metrics can be found in the `tensorboard/[experiment_name]/[timestamp]` directory. 

NB: The training procedure is not yet adjusted to the pre-processed 'zarr' data 
since the format of the labels may continue changing and sampling the data for training depends on this.

## Evaluate.py: Evaluate the quality of the predictions obtained with a trained model
1. Set the following configuration options in the pipeline_config.yaml file:      
    * 'partition_predict' can be 'selected surveys' or 'all surveys'.
    * 'selected_surveys' should be a list of the names of the selected surveys. Should not be an empty list if the previous parameter ('partition_predict') is 'selected surveys'. Should be an empty list if 'partition_predict' is set to 'all surveys'
    * 'eval_mode' can be 'all' (Consider all pixels), 'region' (Exclude all pixels not in a neighborhood of a labeled School) or 'fish' (only evaluate the discrimination on species)
    
2. Run the following program: `/pipeline_train_predict/evaluate.py --yaml_path [path to pipeline yaml config] --checkpoint_path [path to checkpoint model] --save_path_metrics [path to save metrics] --save_path_plot [path where PR-curves are saved]`
    * `path_to_save_metrics` and `path_to_save_plot` are optional. If empty, the F1 score is computed for each selected survey and printed. 
3. The program will then compute and plot evaluation metrics for assessing the quality of the predictions obtained with a trained model. The metrics may be saved as a csv-file with the name [survey]_test.csv. The PR-curve may be saved as a png-image with the name [survey]_pr.png. 

## Save_predict.py: Make predictions with a trained model and save the results
1. Set the following configuration options in the pipeline_config.yaml file:              
    * 'save_prediction_surveys' should be a list of all surveys to compute and save predictions form. 
    * 'resume_writing': if set to True it is assumed that a zarr directory of predictions exists and if new raw files are detected, predictions will be appended to the zarr directory

2. Run the following program: `/pipeline_train_predict/save_predict.py --yaml_path [path to pipeline yaml config] --checkpoint_path [path to checkpoint model] --save_predictions_path [path to directory where predictions should be saved]`
3. The program will then make predictions with the trained model and saved. 

