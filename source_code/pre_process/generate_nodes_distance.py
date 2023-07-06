import sys
import math
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    classification_report,
)
from sklearn.utils import resample
import random
import os
from pickle import dump
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from itertools import product
from multiprocessing import Pool, Manager

sys.path.append("../")
sys.path.append("../../")
sys.path.append("./source_code/")

import project_config as CONFIG


def prepare_output_directory(output_path):
    dir_name = str(os.path.dirname(output_path))
    # os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    data = pd.read_csv(path)
    return data


def generate_distances(benign_data, node1, node2, distance_data_rows):
    if node1 >= node2:
        return
    # print(node1, '   ,   ', node2)
    benign_data = benign_data.groupby(["NODE"]).max()[["LAT", "LNG"]]
    data1 = benign_data.loc[node1]
    data2 = benign_data.loc[node2]
    dist = math.sqrt(
        (data1["LAT"] - data2["LAT"]) ** 2 + (data1["LNG"] - data2["LNG"]) ** 2
    )
    temp_df = pd.DataFrame(
        {
            "NODE_1": [node1, node2],
            "LAT_1": [data1["LAT"], data2["LAT"]],
            "LNG_1": [data1["LNG"], data2["LNG"]],
            "NODE_2": [node2, node1],
            "LAT_2": [data2["LAT"], data1["LAT"]],
            "LNG_2": [data2["LNG"], data1["LNG"]],
            "DISTANCE": [dist, dist],
        }
    )
    distance_data_rows.append(temp_df)


def main_generate_distances(time_step, num_nodes, group_number):
    benign_dataset_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_"
        + str(time_step)
        + "_num_ids_"
        + str(num_nodes)
        + ".csv"
    )

    benign_dataset = load_dataset(benign_dataset_path)
    nodes = list(benign_dataset["NODE"].unique())

    output_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/metadata/distances.csv"
    )
    prepare_output_directory(output_path)

    distance_data = pd.DataFrame(
        columns=["NODE_1", "LAT_1", "LNG_1", "NODE_2", "LAT_2", "LNG_2", "DISTANCE"]
    )

    manager = Manager()
    distance_data_rows = manager.list([distance_data])

    p = Pool(30)
    p.starmap(
        generate_distances,
        product([benign_dataset], nodes, nodes, [distance_data_rows]),
    )
    p.close()
    p.join()

    distance_data = pd.concat(distance_data_rows, ignore_index=True)
    distance_data = distance_data.sort_values(by=["NODE_1", "NODE_2"])
    distance_data.to_csv(output_path, index=False)

    print(distance_data)
    print("shape: ", distance_data.shape)


def main():
    time_step = 60 * 10
    num_nodes = CONFIG.NUM_NODES

    if len(sys.argv) > 1:
        time_step = int(sys.argv[1])

    for group_number in range(CONFIG.NUM_GROUPS):
        main_generate_distances(time_step, num_nodes, group_number)


if __name__ == "__main__":
    main()
