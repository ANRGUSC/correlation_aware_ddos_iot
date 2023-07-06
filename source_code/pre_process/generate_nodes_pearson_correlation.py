import sys
import math
import pandas as pd
import numpy as np
import os
from itertools import product
from multiprocessing import Pool, Manager
from pickle import dump
import matplotlib.pyplot as plt
import glob

sys.path.append("../")
sys.path.append("../../")
sys.path.append("./source_code/")

import project_config as CONFIG


def prepare_output_directory(output_path):
    dir_name = str(os.path.dirname(output_path))
    # os.system('rm -rf ' + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    data = pd.read_csv(path)
    return data


def generate_correlations(benign_data, node1, node2, correlation_data_rows):
    if node1 >= node2:
        return
    # print(node1, '   ,   ', node2)
    data1 = benign_data.loc[benign_data["NODE"] == node1].reset_index(drop=True)
    data2 = benign_data.loc[benign_data["NODE"] == node2].reset_index(drop=True)
    correlation = data1["PACKET"].corr(data2["PACKET"])
    temp_df = pd.DataFrame(
        {
            "NODE_1": [node1, node2],
            "LAT_1": [data1["LAT"][0], data2["LAT"][0]],
            "LNG_1": [data1["LNG"][0], data2["LNG"][0]],
            "NODE_2": [node2, node1],
            "LAT_2": [data2["LAT"][0], data1["LAT"][0]],
            "LNG_2": [data2["LNG"][0], data1["LNG"][0]],
            "CORRELATION": [correlation, correlation],
        }
    )
    correlation_data_rows.append(temp_df)


def main_generate_correlations(time_step, num_nodes, group_number):
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
        + "/metadata/correlations.csv"
    )
    prepare_output_directory(output_path)

    correlation_data = pd.DataFrame(
        columns=["NODE_1", "LAT_1", "LNG_1", "NODE_2", "LAT_2", "LNG_2", "CORRELATION"]
    )

    manager = Manager()
    correlation_data_rows = manager.list([correlation_data])

    p = Pool(30)
    p.starmap(
        generate_correlations,
        product([benign_dataset], nodes, nodes, [correlation_data_rows]),
    )
    p.close()
    p.join()

    correlation_data = pd.concat(correlation_data_rows, ignore_index=True)
    correlation_data = correlation_data.sort_values(by=["NODE_1", "NODE_2"])
    correlation_data.to_csv(output_path, index=False)

    print(correlation_data)
    print("shape: ", correlation_data.shape)


def main():
    time_step = 60 * 10
    num_nodes = CONFIG.NUM_NODES

    if len(sys.argv) > 1:
        time_step = int(sys.argv[1])

    for group_number in range(CONFIG.NUM_GROUPS):
        main_generate_correlations(time_step, num_nodes, group_number)


if __name__ == "__main__":
    main()
