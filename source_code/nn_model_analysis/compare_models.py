import sys
import pandas as pd
import os
import matplotlib.pyplot as plt

sys.path.append("../")
sys.path.append("../../")
import project_config as CONFIG


def prepare_output_directory(output_path):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """
    dir_name = str(os.path.dirname(output_path))
    # os.system('rm -rf ' + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    """Load the dataset

    Keyword arguments:
    path -- path to the dataset csv file
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def combine_general_reports(
    input_path,
    metadata_metric_list,
    nn_model_list,
    architecture_list,
    metric,
    mode,
    output_directory,
):
    general_report_df = pd.DataFrame()
    for metadata_metric in metadata_metric_list:
        for nn_model in nn_model_list:
            for architecture in architecture_list:
                for group_number in range(CONFIG.NUM_GROUPS):
                    general_report_path = (
                        input_path
                        + "group_"
                        + str(group_number)
                        + "/metadata_"
                        + metadata_metric
                        + "/"
                        + nn_model
                        + "/"
                        + architecture
                        + "/run_0/report/metrics_evaluation/"
                        + "data/general_report_"
                        + metric
                        + "_"
                        + mode
                        + ".csv"
                    )
                    tmp_df = pd.read_csv(general_report_path)
                    general_report_df = pd.concat([general_report_df, tmp_df])
                output_path = (
                    output_directory
                    + "metadata_"
                    + metadata_metric
                    + "/"
                    + nn_model
                    + "/"
                    + architecture
                    + "/"
                )
                prepare_output_directory(output_path)
                output_path_general_report = (
                    output_path
                    + "general_report_combined_"
                    + metric
                    + "_"
                    + mode
                    + ".csv"
                )
                general_report_df.to_csv(output_path_general_report, index=False)
                mean_report_df = general_report_df.groupby(["k"]).mean().reset_index()
                output_path_mean_report = (
                    output_path + "mean_report_combined_" + metric + "_" + mode + ".csv"
                )
                mean_report_df.to_csv(output_path_mean_report, index=False)


def combine_roc_reports(
    input_path,
    metadata_metric_list,
    nn_model_list,
    architecture_list,
    metric,
    mode,
    output_directory,
):
    roc_report_df = pd.DataFrame()
    for metadata_metric in metadata_metric_list:
        for nn_model in nn_model_list:
            for architecture in architecture_list:
                for group_number in range(CONFIG.NUM_GROUPS):
                    roc_report_path = (
                        input_path
                        + "group_"
                        + str(group_number)
                        + "/metadata_"
                        + metadata_metric
                        + "/"
                        + nn_model
                        + "/"
                        + architecture
                        + "/run_0/report/roc/"
                        + "data/roc_clean_report_"
                        + metric
                        + "_"
                        + mode
                        + ".csv"
                    )
                    tmp_df = pd.read_csv(roc_report_path)
                    roc_report_df = pd.concat([roc_report_df, tmp_df])
                output_path = (
                    output_directory
                    + "metadata_"
                    + metadata_metric
                    + "/"
                    + nn_model
                    + "/"
                    + architecture
                    + "/"
                )
                prepare_output_directory(output_path)
                output_path_roc_report = (
                    output_path + "roc_report_combined_" + metric + "_" + mode + ".csv"
                )
                roc_report_df.to_csv(output_path_roc_report, index=False)
                roc_mean_report_df = (
                    roc_report_df.groupby(["k", "threshold"]).mean().reset_index()
                )
                output_path_roc_mean_report = (
                    output_path
                    + "roc_mean_report_combined_"
                    + metric
                    + "_"
                    + mode
                    + ".csv"
                )
                roc_mean_report_df.to_csv(output_path_roc_mean_report, index=False)


def main_combine_reports():
    input_path = CONFIG.OUTPUT_DIRECTORY + "nn_training/"
    output_directory = CONFIG.OUTPUT_DIRECTORY + "nn_model_analysis/data/"
    prepare_output_directory(output_directory)

    metric = "val_binary_accuracy"
    mode = "max"
    metadata_metric_list = ["NOT_USED"]
    nn_model_list = ["dense", "cnn", "lstm", "trans", "aen"]
    architecture_list = [
        "one_model_with_correlation",
        "multiple_models_with_correlation",
        "one_model_without_correlation",
        "multiple_models_without_correlation",
    ]
    combine_general_reports(
        input_path,
        metadata_metric_list,
        nn_model_list,
        architecture_list,
        metric,
        mode,
        output_directory,
    )
    combine_roc_reports(
        input_path,
        metadata_metric_list,
        nn_model_list,
        architecture_list,
        metric,
        mode,
        output_directory,
    )

    metadata_metric_list = ["NOT_USED", "DISTANCE", "CORRELATION", "SHAP", "RANDOM"]
    nn_model_list = ["lstm"]
    architecture_list = ["multiple_models_with_correlation"]
    combine_general_reports(
        input_path,
        metadata_metric_list,
        nn_model_list,
        architecture_list,
        metric,
        mode,
        output_directory,
    )
    combine_roc_reports(
        input_path,
        metadata_metric_list,
        nn_model_list,
        architecture_list,
        metric,
        mode,
        output_directory,
    )


def plot_metric_vs_attack_parameter(
    all_mean_report_path, mean_report_legends, colors, markers, line_styles, output_path
):
    """Plot the metrics like binary accuracy and recall for different k values

    Keyword arguments:
    all_mean_report_path -- a list of paths to the all the mean metric values for different NN models
    mean_report_legends -- a dictionary showing the mean_report_path
    output_path -- path to the output directory
    """

    metrics = [
        "binary_accuracy",
        "recall",
        "specificity",
        "precision",
        "auc",
        "f1score",
    ]

    for metric in metrics:
        plt.clf()
        if metric == "k" or metric == "node" or "val" in metric:
            continue
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            plt.plot(
                mean_report["k"],
                mean_report[metric],
                color=colors[i],
                linestyle=line_styles[0],
                marker=markers[i],
                label=mean_report_legends[i],
            )
            # plt.plot(mean_report['k'], mean_report['val_' + metric], color=colors[i], linestyle=line_styles[1],
            #         marker=markers[i], label='Test_' + mean_report_legends[i])
        plt.xlabel("Attack Packet Volume Distribution Parameter (k)")
        plt.ylabel(metric)
        if metric == "binary_accuracy":
            plt.ylabel("Binary Accuracy")
        elif metric == "recall":
            plt.ylabel("Recall")
        elif metric == "specificity":
            plt.ylabel("Specificity")
        elif metric == "precision":
            plt.ylabel("Precision")
        elif metric == "auc":
            plt.ylabel("AUC")
        elif metric == "f1score":
            plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(output_path + metric + "_train.png")

    for metric in metrics:
        plt.clf()
        if metric == "k" or metric == "node" or "val" in metric:
            continue
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            plt.plot(
                mean_report["k"],
                mean_report["val_" + metric],
                color=colors[i],
                linestyle=line_styles[1],
                marker=markers[i],
                label=mean_report_legends[i],
            )
        plt.xlabel("Attack Packet Volume Distribution Parameter (k)")
        plt.ylabel(metric)
        if metric == "binary_accuracy":
            plt.ylabel("Binary Accuracy")
        elif metric == "recall":
            plt.ylabel("Recall")
        elif metric == "specificity":
            plt.ylabel("Specificity")
        elif metric == "precision":
            plt.ylabel("Precision")
        elif metric == "auc":
            plt.ylabel("AUC")
        elif metric == "f1score":
            plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(output_path + metric + "_test.png")


def plot_roc(
    all_mean_report_path, mean_report_legends, colors, markers, line_styles, output_path
):
    """Plot the metrics like binary accuracy and recall for different k values

    Keyword arguments:
    all_mean_report_path -- a list of paths to the all the mean metric values for different NN models
    mean_report_legends -- a dictionary showing the mean_report_path
    output_path -- path to the output directory
    """

    mean_report = pd.read_csv(all_mean_report_path[0])
    k_list = mean_report["k"].unique()

    for k in k_list:
        plt.clf()
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            tmp_mean_report = mean_report.loc[mean_report["k"] == k]
            plt.plot(
                tmp_mean_report["fpr"],
                tmp_mean_report["tpr"],
                color=colors[i],
                linestyle=line_styles[0],
                label=mean_report_legends[i],
            )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(output_path + "roc_curve_k_" + str(k) + "_train.png")

    for k in k_list:
        plt.clf()
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            tmp_mean_report = mean_report.loc[mean_report["k"] == k]
            plt.plot(
                tmp_mean_report["val_fpr"],
                tmp_mean_report["val_tpr"],
                color=colors[i],
                linestyle=line_styles[1],
                label=mean_report_legends[i],
            )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(output_path + "roc_curve_k_" + str(k) + "_test.png")


def main_plot_roc(
    all_mean_report_path, mean_report_legends, colors, markers, line_styles, output_path
):
    """Plot the metrics like binary accuracy and recall for different k values

    Keyword arguments:
    all_mean_report_path -- a list of paths to the all the mean metric values for different NN models
    mean_report_legends -- a dictionary showing the mean_report_path
    output_path -- path to the output directory
    """

    mean_report = pd.read_csv(all_mean_report_path[0])
    k_list = mean_report["k"].unique()

    for k in k_list:
        plt.clf()
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            tmp_mean_report = mean_report.loc[mean_report["k"] == k]
            plt.plot(
                tmp_mean_report["fpr"],
                tmp_mean_report["tpr"],
                color=colors[i],
                linestyle=line_styles[0],
                label=mean_report_legends[i],
            )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(output_path + "roc_curve_k_" + str(k) + "_train.png")

    for k in k_list:
        plt.clf()
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            tmp_mean_report = mean_report.loc[mean_report["k"] == k]
            plt.plot(
                tmp_mean_report["val_fpr"],
                tmp_mean_report["val_tpr"],
                color=colors[i],
                linestyle=line_styles[1],
                label=mean_report_legends[i],
            )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(output_path + "roc_curve_k_" + str(k) + "_test.png")


def main_plot_compare_nn_models_architectures(metric, mode):
    """Main function for plotting the metrics like binary accuracy and recall for different k values

    Keyword arguments:
    metric -- the metric for comparing the NN models like binary accuracy, recall
    mode -- the mode for the selected metric
    """
    colors = {0: "black", 1: "red", 2: "blue", 3: "green", 4: "magenta", 5: "yellow"}
    markers = {0: ".", 1: "v", 2: "+", 3: "x", 4: "s", 5: "^"}
    line_styles = {0: "solid", 1: "dashed"}

    nn_model_list = ["dense", "cnn", "lstm", "trans", "aen"]
    architecture_list = [
        "multiple_models_with_correlation",
        "multiple_models_without_correlation",
        "one_model_with_correlation",
        "one_model_without_correlation",
    ]
    metadata_metric_list = ["NOT_USED", "DISTANCE", "CORRELATION", "RANDOM"]

    metadata_metric = "NOT_USED"
    mean_report_legends = {0: "MM-WC", 1: "MM-NC", 2: "OM-WC", 3: "OM-NC"}
    for nn_model in nn_model_list:
        metric_mean_report_path_list = []
        roc_mean_report_path_list = []
        for architecture in architecture_list:
            input_path = (
                CONFIG.OUTPUT_DIRECTORY
                + "nn_model_analysis/data/"
                + "metadata_"
                + metadata_metric
                + "/"
                + nn_model
                + "/"
                + architecture
                + "/"
            )

            metric_mean_report_path = (
                input_path + "mean_report_combined_" + metric + "_" + mode + ".csv"
            )
            metric_mean_report_path_list.append(metric_mean_report_path)

            roc_mean_report_path = (
                input_path + "roc_mean_report_combined_" + metric + "_" + mode + ".csv"
            )
            roc_mean_report_path_list.append(roc_mean_report_path)

        output_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "nn_model_analysis/plot/metadata_"
            + metadata_metric
            + "/compare_nn_models/"
            + nn_model
            + "/"
        )
        metric_output_path = output_path + "compare_" + nn_model + "_nn_model_"
        prepare_output_directory(metric_output_path)
        plot_metric_vs_attack_parameter(
            metric_mean_report_path_list,
            mean_report_legends,
            colors,
            markers,
            line_styles,
            metric_output_path,
        )

        roc_output_path = output_path + "roc/compare_roc_" + nn_model + "_nn_model_"
        prepare_output_directory(roc_output_path)
        plot_roc(
            roc_mean_report_path_list,
            mean_report_legends,
            colors,
            markers,
            line_styles,
            roc_output_path,
        )

    for metadata_metric in metadata_metric_list:
        if metadata_metric == "NOT_USED":
            nn_model_list = ["dense", "cnn", "lstm", "trans", "aen"]
            architecture_list = [
                "multiple_models_with_correlation",
                "multiple_models_without_correlation",
                "one_model_with_correlation",
                "one_model_without_correlation",
            ]
            mean_report_legends = {0: "MLP", 1: "CNN", 2: "LSTM", 3: "TRF", 4: "AEN"}
        else:
            architecture_list = ["multiple_models_with_correlation"]
            nn_model_list = ["lstm"]
            mean_report_legends = {0: "LSTM"}
        for architecture in architecture_list:
            metric_mean_report_path_list = []
            roc_mean_report_path_list = []
            for nn_model in nn_model_list:
                input_path = (
                    CONFIG.OUTPUT_DIRECTORY
                    + "nn_model_analysis/data/"
                    + "metadata_"
                    + metadata_metric
                    + "/"
                    + nn_model
                    + "/"
                    + architecture
                    + "/"
                )

                metric_mean_report_path = (
                    input_path + "mean_report_combined_" + metric + "_" + mode + ".csv"
                )
                metric_mean_report_path_list.append(metric_mean_report_path)

                roc_mean_report_path = (
                    input_path
                    + "roc_mean_report_combined_"
                    + metric
                    + "_"
                    + mode
                    + ".csv"
                )
                roc_mean_report_path_list.append(roc_mean_report_path)

            output_path = (
                CONFIG.OUTPUT_DIRECTORY
                + "nn_model_analysis/plot/metadata_"
                + metadata_metric
                + "/compare_nn_architecture/"
                + architecture
                + "/"
            )
            metric_output_path = (
                output_path + "compare_" + architecture + "_architecture_"
            )
            prepare_output_directory(metric_output_path)
            plot_metric_vs_attack_parameter(
                metric_mean_report_path_list,
                mean_report_legends,
                colors,
                markers,
                line_styles,
                metric_output_path,
            )

            roc_output_path = (
                output_path + "roc/compare_roc_" + architecture + "_architecture_"
            )
            prepare_output_directory(roc_output_path)
            plot_roc(
                roc_mean_report_path_list,
                mean_report_legends,
                colors,
                markers,
                line_styles,
                roc_output_path,
            )


def main_plot_compare_metadata_metrics(metric, mode):
    colors = {0: "black", 1: "red", 2: "blue", 3: "green", 4: "magenta", 5: "yellow"}
    markers = {0: ".", 1: "v", 2: "+", 3: "x", 4: "s", 5: "^"}
    line_styles = {0: "solid", 1: "dashed"}

    nn_model_list = ["lstm"]
    architecture_list = ["multiple_models_with_correlation"]
    metadata_metric_list = ["NOT_USED", "DISTANCE", "CORRELATION", "SHAP", "RANDOM"]
    mean_report_legends = {
        0: "All Nodes",
        1: "Nearest Neighbor",
        2: "Pearson",
        3: "SHAP",
        4: "Random",
    }

    for nn_model in nn_model_list:
        for architecture in architecture_list:
            metric_mean_report_path_list = []
            roc_mean_report_path_list = []
            for metadata_metric in metadata_metric_list:
                input_path = (
                    CONFIG.OUTPUT_DIRECTORY
                    + "nn_model_analysis/data/"
                    + "metadata_"
                    + metadata_metric
                    + "/"
                    + nn_model
                    + "/"
                    + architecture
                    + "/"
                )

                metric_mean_report_path = (
                    input_path + "mean_report_combined_" + metric + "_" + mode + ".csv"
                )
                metric_mean_report_path_list.append(metric_mean_report_path)

                roc_mean_report_path = (
                    input_path
                    + "roc_mean_report_combined_"
                    + metric
                    + "_"
                    + mode
                    + ".csv"
                )
                roc_mean_report_path_list.append(roc_mean_report_path)

            output_path = (
                CONFIG.OUTPUT_DIRECTORY
                + "nn_model_analysis/plot/metadata_comparison/compare_nn_architecture/"
                + architecture
                + "/"
                + nn_model
                + "/"
            )
            metric_output_path = (
                output_path
                + "compare_"
                + architecture
                + "_architecture_"
                + nn_model
                + "_nn_model_"
            )
            prepare_output_directory(metric_output_path)
            plot_metric_vs_attack_parameter(
                metric_mean_report_path_list,
                mean_report_legends,
                colors,
                markers,
                line_styles,
                metric_output_path,
            )

            roc_output_path = (
                output_path + "roc/compare_roc_" + architecture + "_architecture_"
            )
            prepare_output_directory(roc_output_path)
            plot_roc(
                roc_mean_report_path_list,
                mean_report_legends,
                colors,
                markers,
                line_styles,
                roc_output_path,
            )


def main():
    metric = "val_binary_accuracy"
    mode = "max"

    main_combine_reports()
    main_plot_compare_nn_models_architectures(metric, mode)
    main_plot_compare_metadata_metrics(metric, mode)


if __name__ == "__main__":
    main()
