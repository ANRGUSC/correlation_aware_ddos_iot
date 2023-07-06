import math
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta
import random
from multiprocessing import Pool
from itertools import product

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


def plot_active_nodes_percentage_per_day(
    benign_dataset_path_list, date, output_path, colors, legends
):
    plt.clf()
    fig, ax = plt.subplots()

    for index, benign_dataset_path in enumerate(benign_dataset_path_list):
        data = load_dataset(benign_dataset_path)
        data = data.loc[
            (data["TIME"] >= date) & (data["TIME"] < (date + timedelta(hours=24)))
        ]
        temp = data.groupby(["TIME"]).mean().reset_index()
        temp = temp[["TIME", "ACTIVE"]]
        temp = temp.sort_values(by=["TIME"])
        times = list(temp["TIME"].values)
        ax.plot(times, temp["ACTIVE"], color=colors[index], label=legends[index])

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    myFmt = mdates.DateFormatter("%H")
    ax.xaxis.set_major_formatter(myFmt)
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Active Nodes Percentage")

    output_path += "active_nodes_percentage_" + str(date)[0:10] + ".png"
    fig.savefig(output_path)


def plot_active_nodes_percentage(
    benign_dataset_path_list, output_path, colors, legends
):
    plt.clf()
    fig, ax = plt.subplots()
    for index, benign_dataset_path in enumerate(benign_dataset_path_list):
        data = load_dataset(benign_dataset_path)
        data = data.sort_values(by=["TIME"])
        data["TIME_2"] = data["TIME"].dt.time
        data["TIME_2"] = pd.to_datetime(data["TIME_2"].astype(str))

        temp = data.groupby(["TIME_2"]).mean().reset_index()
        temp = temp[["TIME_2", "ACTIVE"]]
        temp = temp.sort_values(by=["TIME_2"])
        times = list(temp["TIME_2"].values)

        ax.plot(times, temp["ACTIVE"], color=colors[index], label=legends[index])

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    myFmt = mdates.DateFormatter("%H")
    ax.xaxis.set_major_formatter(myFmt)
    ax.legend()
    ax.set_xlabel("Time (Hour of The Day)")
    ax.set_ylabel("Average Active Nodes Percentage")
    fig.savefig(output_path)


def main_plot_active_nodes_percentage_per_day(
    benign_dataset_path_list, output_path, colors, legends
):
    benign_data = load_dataset(benign_dataset_path_list[0])
    begin_date = benign_data.loc[0, "TIME"]
    begin_date = datetime(
        begin_date.year, begin_date.month, begin_date.day + 1, 0, 0, 0
    )
    end_date = benign_data.loc[len(benign_data) - 1, "TIME"]

    dates = []
    date = begin_date
    while date < end_date:
        dates.append(date)
        date += timedelta(hours=24)

    for date in dates:
        plot_active_nodes_percentage_per_day(
            benign_dataset_path_list, date, output_path, colors, legends
        )


def main():
    time_step = 60 * 10
    num_nodes_list = [4060, 50]
    colors = ["b", "r"]
    legends = ["All Nodes", "Random 50 Nodes"]
    benign_dataset_path_list = []

    for num_nodes in num_nodes_list:
        benign_dataset_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "pre_process/Output/group_0/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_"
            + str(time_step)
            + "_num_ids_"
            + str(num_nodes)
            + ".csv"
        )
        benign_dataset_path_list.append(benign_dataset_path)
    output_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "stats/active_nodes_percentage/all/active_nodes_percentage_comparison.png"
    )
    prepare_output_directory(output_path)
    plot_active_nodes_percentage(benign_dataset_path_list, output_path, colors, legends)
    output_path = CONFIG.OUTPUT_DIRECTORY + "stats/active_nodes_percentage/per_day/"
    prepare_output_directory(output_path)
    main_plot_active_nodes_percentage_per_day(
        benign_dataset_path_list, output_path, colors, legends
    )


if __name__ == "__main__":
    main()
