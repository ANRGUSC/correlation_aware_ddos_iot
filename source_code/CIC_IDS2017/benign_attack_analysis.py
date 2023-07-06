import pandas as pd
from matplotlib import pyplot as plt
import os
import math
import sys

sys.path.append("../")
import project_config as CONFIG


def prepare_output_directory(output_path):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """

    dir_name = str(os.path.dirname(output_path))
    # os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def plot_features_benign_attack(input_dataset_path, output_path):
    data = pd.read_csv(input_dataset_path)
    attack_types_list = list(data[" Label"].unique())
    attack_types_list.remove("BENIGN")
    features_list = list(data.columns.values)
    features_list = features_list[7:-1]
    features_list.remove("Flow Bytes/s")
    features_list.remove(" Flow Packets/s")
    features_list = [" Flow Duration"]
    plot_data_benign = data.loc[data[" Label"] == "BENIGN"]
    for feature in features_list:
        print("feature: ", feature)
        plt.clf()
        plt.hist(plot_data_benign[feature], bins=50)
        feature_str = feature.replace(" ", "_")
        feature_str = feature_str.replace("/", "_")
        plt.title(feature_str)
        plot_output_path = output_path + "benign/benign_" + feature_str + ".png"
        prepare_output_directory(plot_output_path)
        plt.savefig(plot_output_path, bbox_inches="tight")

    for attack_type in attack_types_list:
        print("attack_type: ", attack_type)
        plot_data_attack = data.loc[data[" Label"] == attack_type]
        for feature in features_list:
            feature_str = feature.replace(" ", "_")
            feature_str = feature_str.replace("/", "_")
            attack_type_str = attack_type.replace(" ", "_")
            attack_type_str = attack_type_str.replace("/", "_")

            plt.clf()
            plt.hist(plot_data_attack[feature], bins=50)
            plt.title(feature_str)
            plot_output_path = (
                output_path
                + "attack/"
                + attack_type_str
                + "/attack_"
                + feature_str
                + ".png"
            )
            prepare_output_directory(plot_output_path)
            plt.savefig(plot_output_path, bbox_inches="tight")

            plt.clf()
            benign_mean = plot_data_benign[feature].mean()
            attack_mean = plot_data_attack[feature].mean()
            plt.hist(
                plot_data_benign[feature],
                bins=50,
                density=True,
                label="Benign - Mean = " + str(benign_mean),
            )
            plt.hist(
                plot_data_attack[feature],
                bins=50,
                density=True,
                label="Attack - Mean = " + str(attack_mean),
            )
            plt.title(feature_str)
            plt.legend()
            plot_output_path = (
                output_path + "compare/" + attack_type_str + "/" + feature_str + ".png"
            )
            prepare_output_directory(plot_output_path)
            plt.savefig(plot_output_path, bbox_inches="tight")


def plot_compare_feature_benign_attack(
    input_dataset_path, feature, attack_type, xlabel, ylabel, output_path
):
    data = pd.read_csv(input_dataset_path)
    data_benign = data.loc[data[" Label"] == "BENIGN"]
    data_attack = data.loc[data[" Label"] == attack_type]
    benign_mean = str(round(data_benign[feature].mean() / math.pow(10, 6), 2))
    attack_mean = str(round(data_attack[feature].mean() / math.pow(10, 6), 2))
    plt.clf()
    plt.hist(
        data_benign[feature] / math.pow(10, 6),
        bins=50,
        density=True,
        label="Benign - Average = " + benign_mean,
    )
    plt.hist(
        data_attack[feature] / math.pow(10, 6),
        bins=50,
        density=True,
        label="Attack - Average = " + attack_mean,
    )
    if feature == " Flow IAT Mean" or feature == " Flow IAT Min":
        plt.xlim([0, 20])
    elif feature == " Flow IAT Std":
        plt.xlim([0, 50])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    prepare_output_directory(output_path)
    plt.savefig(output_path, bbox_inches="tight")


def main():
    input_dataset_path = (
        CONFIG.DATASET_DIRECTORY + "CIC_IDS2017/Wednesday-workingHours.pcap_ISCX.csv"
    )
    output_directory = CONFIG.OUTPUT_DIRECTORY + "CIC_IDS2017/"
    prepare_output_directory(output_directory)
    plot_features_benign_attack(input_dataset_path, output_directory)

    output_path = output_directory + "flow_duration.png"
    plot_compare_feature_benign_attack(
        input_dataset_path,
        " Flow Duration",
        "DoS Slowhttptest",
        "Flow Duration (Seconds)",
        "Probability Density",
        output_path,
    )

    output_path = output_directory + "flow_iat_max.png"
    plot_compare_feature_benign_attack(
        input_dataset_path,
        " Flow IAT Max",
        "DoS Slowhttptest",
        "Maximum Flow Inter Arrival Time (Seconds)",
        "Probability Density",
        output_path,
    )

    output_path = output_directory + "flow_iat_min.png"
    plot_compare_feature_benign_attack(
        input_dataset_path,
        " Flow IAT Min",
        "DoS Slowhttptest",
        "Minimum Flow Inter Arrival Time (Seconds)",
        "Probability Density",
        output_path,
    )

    output_path = output_directory + "flow_iat_std.png"
    plot_compare_feature_benign_attack(
        input_dataset_path,
        " Flow IAT Std",
        "DoS Slowhttptest",
        "Standard Deviation Flow Inter Arrival Time (Seconds)",
        "Probability Density",
        output_path,
    )

    output_path = output_directory + "flow_iat_mean.png"
    plot_compare_feature_benign_attack(
        input_dataset_path,
        " Flow IAT Mean",
        "DoS Slowhttptest",
        "Mean Flow Inter Arrival Time (Seconds)",
        "Probability Density",
        output_path,
    )


if __name__ == "__main__":
    main()
