import sys
import math
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.utils import resample
import random
import os
from pickle import dump
import matplotlib.pyplot as plt
import glob
from datetime import datetime

sys.path.append("../")
sys.path.append("../../")
sys.path.append('./source_code/')

import project_config as CONFIG


def prepare_output_directory(output_path):
    dir_name = str(os.path.dirname(output_path))
    # os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    data = pd.read_csv(path)
    return data


def generate_distances(benign_dataset_path, output_path):
    benign_dataset = load_dataset(benign_dataset_path)
    nodes = list(benign_dataset["NODE"].unique())
    print(nodes)
    print(len(nodes))
    benign_dataset = benign_dataset.groupby(['NODE']).max()[['LAT', 'LNG']]
    dist_df = pd.DataFrame()
    for node1 in nodes:
        for node2 in nodes:
            if node1 == node2:
                continue
            data1 = benign_dataset.loc[node1]
            data2 = benign_dataset.loc[node2]
            dist = math.sqrt((data1["LAT"] - data2["LAT"]) ** 2 + (data1["LNG"] - data2["LNG"]) ** 2)
            temp_df = pd.DataFrame({'NODE_1': [node1], 'LAT_1': [data1['LAT']], 'LNG_1': [data1['LNG']],
                                    'NODE_2': [node2], 'LAT_2': [data2['LAT']], 'LNG_2': [data2['LNG']],
                                    'DISTANCE': [dist]})
            dist_df = pd.concat([dist_df, temp_df])
    print(dist_df)
    dist_df.to_csv(output_path, index=False)


def main_generate_distances():
    benign_dataset_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_600_num_ids_50.csv"
    output_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/metadata/distances.csv"
    prepare_output_directory(output_path)
    generate_distances(benign_dataset_path, output_path)


if __name__ == "__main__":
    main_generate_distances()
