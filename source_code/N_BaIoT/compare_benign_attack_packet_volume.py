import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import os

sys.path.append("../")
sys.path.append("../../")
sys.path.append("./source_code/")

import project_config as CONFIG


def prepare_output_directory(output_path):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """
    dir_name = str(os.path.dirname(output_path))
    # os.system('rm -rf ' + dir_name)
    os.system("mkdir -p " + dir_name)


def plot_compare_benign_attack_packet_volume(
    benign_data_path, attack_data_path, output_path
):
    benign_data = pd.read_csv(benign_data_path)
    attack_data = pd.read_csv(attack_data_path)
    size = min(benign_data.shape[0], attack_data.shape[0])

    # benign_packets = random.choices(benign_data['PACKET'], k=size)
    # attack_packets = random.choices(attack_data['PACKET'], k=size)
    benign_packets = benign_data["PACKET"]
    attack_packets = attack_data["PACKET"]
    benign_mean = str(round(statistics.mean(benign_packets), 2))
    attack_mean = str(round(statistics.mean(attack_packets), 2))

    plt.clf()
    plt.hist(
        benign_packets, bins=50, density=True, label="Benign - Average = " + benign_mean
    )
    plt.xlabel("Packet Volume")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig(output_path + "benign_traffic.png", bbox_inches="tight")

    plt.clf()
    plt.hist(
        attack_packets, bins=50, density=True, label="Attack - Average = " + attack_mean
    )
    plt.xlabel("Packet Volume")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig(output_path + "attack_traffic.png", bbox_inches="tight")

    plt.clf()
    plt.hist(
        benign_packets, bins=50, density=True, label="Benign - Average = " + benign_mean
    )
    plt.hist(
        attack_packets, bins=50, density=True, label="Attack - Average = " + attack_mean
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Packet Volume")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig(output_path + "compare_benign_attack_traffic.png", bbox_inches="tight")


def main():
    benign_data_path = (
        CONFIG.DATASET_DIRECTORY
        + "N_BaIoT/SimpleHome_XCS7_1003_WHT_Security_Camera_benign_aggregation_Source-IP_time_window_10-sec_stat_Number.csv"
    )
    attack_data_path = (
        CONFIG.DATASET_DIRECTORY
        + "N_BaIoT/SimpleHome_XCS7_1003_WHT_Security_Camera_mirai_attack_udp_aggregation_Source-IP_time_window_10-sec_stat_Number.csv"
    )
    output_path = CONFIG.OUTPUT_DIRECTORY + "N_BaIoT/Plot/"
    prepare_output_directory(output_path)
    plot_compare_benign_attack_packet_volume(
        benign_data_path, attack_data_path, output_path
    )


if __name__ == "__main__":
    main()
