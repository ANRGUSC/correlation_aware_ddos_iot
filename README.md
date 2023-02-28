# Correlation Aware DDoS Detection In IoT System

This repository presents the source code for the Correlation-Aware Neural Networks for DDoS Attack Detection In IoT Systems paper.
A preprint of this paper is available with this [link](https://arxiv.org/abs/2302.07982).

## Instructions for running the codes

The requirements.txt file contains the modules needed to run these scripts and can be installed by running the following command in the terminal:
* pip install -r requirements.txt

## Project config file

The project config file can be found in [/source_code](https://github.com/ANRGUSC/correlation_aware_ddos_iot/tree/main/source_code). The path to the dataset, source code, and output directory can be set in this file. Remember to unzip all compressed datasets before running the code.

## Preprocess the dataset and generate general training dataset

Before running any code, the original dataset need to be unzip in the [/dataset directory](https://github.com/ANRGUSC/correlation_aware_ddos_iot/tree/main/dataset). 

### run_all_pre_process.py

This file automate the pre-processing steps by getting the required inputs and run all pre-processing scripts.

### clean_dataset.py

This file genrates the bening_dataset which has N nodes and each node has time entries starting from the beginning to the end of original dataset with a step of time_step.

Input:
- Input Dataset
- Number of nodes
- Timestep

Output:
- Benign dataset

### generate_nodes_distance.py

This script generates the euclidean distance of all pairs of the IoT devices.

Input:
- Benign Dataset

Output:
- Nodes Distance Dataset

### generate_nodes_pearson_correlation.py

This script generates the Pearson's correlation of the behavior all pairs of the IoT devices.

Input:
- Benign Dataset

Output:
- Nodes Pearson's Correlation Dataset

### generate_attack.py

This script genrates the attacked dataset by considering the ratio of the nodes that are under attack, the attack duration, and also the attack start dates.

Input:
- Bening dataset
- Number of attack days
- Attack ratio
- Attack duration
- Attack start dates

Output:
- Attacked dataset

### generate_training_data.py

This script generates the general training dataset that includes the correlation information of the IoT ndoes and also the one-hot encoding.

Input:
- Attacked dataset

Output:
- Training dataset

## Statistical analysis of the dataset

Provide statistical analysis of the IoT nodes in the dataset. 

### active_nodes_percentage_comparison.py

This script generates the plot of the percentage of the active IoT nodes through the time of the day.

Input:
- Benign Dataset

Output:
- Plot of the percentage of the active IoT nodes throughout the day

### nodes_active_mean_time_comparison.py

This script generates the plot of the average time that IoT nodes are active/inactive throughout the day and night.

Input:
- Benign Dataset

Output:
- Plot of the average time that IoT nodes are active/inactive throughout the day and night.


## Training neural network and generate performance results

Train different neural network models/architectures and generate results.

### run_all_train_test.py

This script automate the process of training the neural network models and also generating the desired results for all combinations of architectures/NN models. 

### train_nn.py

This script create a neural network model/architecture to train on the training dataset for detecting the attackers. The scrip save the final model and also the epochs logs and weights.

Input:
- General training dataset

Output:
- Trained neural network model with epochs' logs and weights


### generate_results.py

This script provides analysis like, binary accuracy, recall, F1 score, etc. based on the trained model/architectures.

Input:
- General training/testing dataset
- Trained neural network model

Output:
- General analysis on the training like accuracy, loss, confusion matrix, etc.
- Plots of true positive, false positive, and true attacks versus time for different attack ratios and durations
- SHAP plots of the most important features
- Attack properties analysis

## Compare models/architectures performance

Compare the performance of different models/architectures.

### compare_models.py

Input:
- Results of each model/architecture

Output:
- Plots of different metrics such as binary accuracy, auc, recall, etc. against k to compare the performance of different model/architecture.


## Acknowledgement

   This material is based upon work supported in part by Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0160 for the Open, Programmable, Secure 5G (OPS-5G) program. Any views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government. 



