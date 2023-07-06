import glob
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool, Manager
import multiprocessing
import statistics

sys.path.append("../")
import project_config as CONFIG


def prepare_output_directory(output_path):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """
    dir_name = str(os.path.dirname(output_path))
    os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    """Load the dataset and change the type of the 'TIME' column to datetime.

    Keyword arguments:
    path -- path to the dataset
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def add_other_nodes_packets(fname, num_days, combined_data_rows):
    print(fname)
    data = pd.read_csv(fname)
    data["TIME"] = pd.to_datetime(data["TIME"])
    begin_date = data["TIME"][0] + timedelta(days=2)
    end_date = data["TIME"][0] + timedelta(days=2 + num_days)
    data = data.loc[(data["TIME"] >= begin_date) & (data["TIME"] < end_date)]

    data = data.sort_values(by=["NODE", "TIME"])
    nodes = data["NODE"].unique()

    for node in nodes:
        data["PACKET_" + str(node)] = 0
        data["NODE_" + str(node)] = 0

    for node_1 in nodes:
        data.loc[data["NODE"] == node_1, "NODE_" + str(node_1)] = 1
        for node_2 in nodes:
            data_node_2 = data.loc[data["NODE"] == node_2]
            data.loc[data["NODE"] == node_1, "PACKET_" + str(node_2)] = data_node_2[
                "PACKET"
            ].values

    combined_data_rows.append(data)


def combine_data(num_days, input_path, output_path):
    """Combine the csv files in the input_path directory and output the combined one to the output_path.

    Keyword arguments:
    input_path -- The path to the input directory.
    output_path -- The path to the output_directory for storing the combined data.
    """
    all_files = [fname for fname in glob.glob(input_path)]
    data_tmp = pd.read_csv(all_files[0])
    nodes = data_tmp["NODE"].unique()
    columns = [
        "BEGIN_DATE",
        "END_DATE",
        "NUM_NODES",
        "ATTACK_RATIO",
        "ATTACK_START_TIME",
        "ATTACK_DURATION",
        "ATTACK_PARAMETER",
        "NODE",
        "LAT",
        "LNG",
        "TIME",
        "TIME_HOUR",
        "ACTIVE",
        "PACKET",
        "ATTACKED",
    ]
    for node in nodes:
        columns.append("PACKET_" + str(node))
        columns.append("NODE_" + str(node))

    combined_data = pd.DataFrame(columns=columns)

    manager = Manager()
    combined_data_rows = manager.list([combined_data])

    p = Pool(28)
    p.starmap(
        add_other_nodes_packets, product(all_files, [num_days], [combined_data_rows])
    )
    p.close()
    p.join()

    combined_data = pd.concat(combined_data_rows, ignore_index=True)
    combined_data.to_csv(output_path, index=False)


def main_combine_data(group_number, data_type, num_days):
    """The main function to be used for calling  combine_data function.

    Keyword arguments:
    data_type -- could be 'train' or 'test'. For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    """
    if data_type == "train":
        input_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "pre_process/Output/group_"
            + str(group_number)
            + "/attacked_data/train/*.csv"
        )
        output_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "pre_process/Output/group_"
            + str(group_number)
            + "/train_data/train_data.csv"
        )
    elif data_type == "validation":
        input_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "pre_process/Output/group_"
            + str(group_number)
            + "/attacked_data/validation/*.csv"
        )
        output_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "pre_process/Output/group_"
            + str(group_number)
            + "/validation_data/validation_data.csv"
        )
    elif data_type == "test":
        input_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "pre_process/Output/group_"
            + str(group_number)
            + "/attacked_data/test/*.csv"
        )
        output_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "pre_process/Output/group_"
            + str(group_number)
            + "/test_data/test_data.csv"
        )

    prepare_output_directory(output_path)
    combine_data(num_days, input_path, output_path)


def main():
    num_train_days = 1
    num_validation_days = 1
    num_test_days = 1
    if len(sys.argv) > 1:
        num_train_days = int(sys.argv[1])
        num_validation_days = int(sys.argv[2])
        num_test_days = int(sys.argv[3])

    for group_number in range(CONFIG.NUM_GROUPS):
        main_combine_data(group_number, "train", num_train_days)
        main_combine_data(group_number, "validation", num_validation_days)
        main_combine_data(group_number, "test", num_test_days)


if __name__ == "__main__":
    main()
