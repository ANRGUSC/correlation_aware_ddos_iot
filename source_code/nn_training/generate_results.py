import pickle
import sys
import statistics
import tensorflow as tf

# Uncomment the following line for the SHAP analysis
tf.compat.v1.disable_v2_behavior()
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_recall_fscore_support,
)
import random
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pickle import load
import glob
import shap

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


def load_dataset(path):
    """Load the dataset

    Keyword arguments:
    path -- path to the dataset csv file
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def calculate_dataset_step(dataset):
    attack_ratios = dataset["ATTACK_RATIO"].unique()
    attack_start_times = dataset["ATTACK_START_TIME"].unique()
    attack_durations = dataset["ATTACK_DURATION"].unique()
    k_list = dataset["ATTACK_PARAMETER"].unique()
    nodes = dataset["NODE"].unique()
    dataset = dataset.sort_values(
        by=[
            "NODE",
            "ATTACK_RATIO",
            "ATTACK_START_TIME",
            "ATTACK_DURATION",
            "ATTACK_PARAMETER",
            "TIME",
        ]
    ).reset_index(drop=True)
    for node in nodes:
        for k in k_list:
            for attack_ratio in attack_ratios:
                for attack_start_time in attack_start_times:
                    for attack_duration in attack_durations:
                        temp = dataset.loc[
                            (dataset["NODE"] == node)
                            & (dataset["ATTACK_RATIO"] == attack_ratio)
                            & (dataset["ATTACK_START_TIME"] == attack_start_time)
                            & (dataset["ATTACK_DURATION"] == attack_duration)
                            & (dataset["ATTACK_PARAMETER"] == k)
                        ]
                        if temp.shape[0] == 0:
                            continue
                        return temp.shape[0]


def get_dataset_input_output(
    nn_model,
    architecture,
    dataset,
    selected_nodes_for_correlation,
    num_labels,
    time_window,
    scaler,
    use_time_hour,
    use_onehot,
):
    # set up the required features to be used for training/validating
    temp_columns = []
    if use_time_hour is True:
        temp_columns.append("TIME_HOUR")
    if (
        architecture == "multiple_models_with_correlation"
        or architecture == "one_model_with_correlation"
    ):
        for node in selected_nodes_for_correlation:
            temp_columns.append("PACKET_" + str(node))
    elif (
        architecture == "multiple_models_without_correlation"
        or architecture == "one_model_without_correlation"
    ):
        temp_columns.append("PACKET")
    if (
        architecture == "one_model_with_correlation"
        or architecture == "one_model_without_correlation"
    ):
        if use_onehot is True:
            for node in selected_nodes_for_correlation:
                temp_columns.append("NODE_" + str(node))
        else:
            temp_columns.append("NODE")
    temp_columns.append("ATTACKED")

    X_out = []
    y_out = []
    df_out = pd.DataFrame()
    dataset = dataset.sort_values(
        by=[
            "NODE",
            "ATTACK_RATIO",
            "ATTACK_START_TIME",
            "ATTACK_DURATION",
            "ATTACK_PARAMETER",
            "TIME",
        ]
    ).reset_index(drop=True)
    dataset_step = calculate_dataset_step(dataset)

    for index_start in range(0, dataset.shape[0], dataset_step):
        temp = dataset.iloc[index_start : index_start + dataset_step, :]
        if temp.shape[0] == 0:
            continue
        temp = temp.sort_values(by=["TIME"]).reset_index(drop=True)
        df_out = pd.concat([df_out, temp.iloc[time_window - 1 :, :]])
        temp = temp[temp_columns]
        X = temp.iloc[:, 0:-num_labels]
        y = temp.iloc[:, -num_labels:]
        X = np.asarray(X).astype(float)
        y = np.asarray(y).astype(float)
        X = scaler.transform(X)

        for i in range(X.shape[0] - time_window + 1):
            X_out.append(X[i : i + time_window])
            y_out.append(y[i + time_window - 1])

    X_out, y_out = np.array(X_out), np.array(y_out)
    if nn_model == "dense" or nn_model == "aen":
        X_out = X_out.reshape((X_out.shape[0], X_out.shape[1] * X_out.shape[2]))

    return X_out, y_out, df_out, temp_columns


def load_model_weights(model_path_input, node, metric, mode):
    """
    Load and return the neural network model weight
    Args:
        model_path_input: path to the model
        node: the node ID for loading the model
        metric: the metric to use for loading the model
        mode: the mode of the metric to use for loading the model

    Returns:
        model: The neural network model with loaded weights
        scaler: The scaler function for the input dataset
    """
    saved_model_path = model_path_input + str(node) + "/"
    scaler_path = saved_model_path + "scaler.pkl"
    selected_nodes_for_correlation_path = (
        saved_model_path + "selected_nodes_for_correlation.pkl"
    )
    model_path = saved_model_path + "final_model/"
    logs_path = saved_model_path + "logs/logs.csv"
    logs = pd.read_csv(logs_path)
    metrics = list(logs.columns)
    metrics.remove("epoch")

    if mode == "max":
        logs = logs.sort_values(by=[metric], ascending=False).reset_index(drop=True)
    elif mode == "min":
        logs = logs.sort_values(by=[metric]).reset_index(drop=True)

    epoch = str((int)(logs["epoch"][0]) + 1).zfill(4)
    checkpoint_path = saved_model_path + "checkpoints/all/weights-" + epoch

    model = tf.keras.models.load_model(model_path)
    model.load_weights(checkpoint_path)
    model.summary()
    scaler = load(open(scaler_path, "rb"))
    selected_nodes_for_correlation = load(
        open(selected_nodes_for_correlation_path, "rb")
    )

    return model, scaler, selected_nodes_for_correlation


def find_best_threshold(
    loaded_model,
    nn_model,
    architecture,
    node,
    train_dataset,
    selected_nodes_for_correlation,
    num_labels,
    time_window,
    scaler,
    use_time_hour,
    use_onehot,
):
    if (
        architecture == "multiple_models_with_correlation"
        or architecture == "multiple_models_without_correlation"
    ):
        train_dataset_tmp = train_dataset.loc[(train_dataset["NODE"] == node)]
    elif (
        architecture == "one_model_with_correlation"
        or architecture == "one_model_without_correlation"
    ):
        train_dataset_tmp = train_dataset
    X_train, y_train, _, _ = get_dataset_input_output(
        nn_model,
        architecture,
        train_dataset_tmp,
        selected_nodes_for_correlation,
        num_labels,
        time_window,
        scaler,
        use_time_hour,
        use_onehot,
    )
    train_predictions_baseline = loaded_model.predict(X_train)
    y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1])
    train_predictions_baseline = train_predictions_baseline.reshape(
        train_predictions_baseline.shape[0] * train_predictions_baseline.shape[1]
    )

    threshold = 0
    f1_score_max = 0
    for th in np.arange(0.1, 0.95, 0.1):
        y_pred_train_tmp = (train_predictions_baseline > th).astype(int)
        f1_score_tmp = f1_score(y_train, y_pred_train_tmp)
        if f1_score_max < f1_score_tmp:
            threshold = th
            f1_score_max = f1_score_tmp
    return threshold


def generate_metrics_evaluation(
    metrics_evaluation_row,
    train_predictions_baseline,
    test_predictions_baseline,
    y_train,
    y_test,
    threshold,
):
    row = metrics_evaluation_row
    y_pred_train = (train_predictions_baseline > threshold).astype(int)
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(
        y_train, y_pred_train
    ).ravel()
    row["tp"] = tp_train
    row["tn"] = tn_train
    row["fp"] = fp_train
    row["fn"] = fn_train
    row["binary_accuracy"] = (tp_train + tn_train) / (
        tn_train + fp_train + fn_train + tp_train + sys.float_info.epsilon
    )
    row["recall"] = tp_train / (tp_train + fn_train + sys.float_info.epsilon)
    row["precision"] = tp_train / (tp_train + fp_train + sys.float_info.epsilon)
    row["specificity"] = tn_train / (tn_train + fp_train + sys.float_info.epsilon)
    row["f1score"] = (
        2
        * (row["precision"] * row["recall"])
        / (row["precision"] + row["recall"] + sys.float_info.epsilon)
    )
    try:
        row["auc"] = roc_auc_score(y_train, train_predictions_baseline, labels=[0, 1])
    except ValueError:
        row["auc"] = 1

    y_pred_test = (test_predictions_baseline > threshold).astype(int)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
    row["val_tp"] = tp_test
    row["val_tn"] = tn_test
    row["val_fp"] = fp_test
    row["val_fn"] = fn_test
    row["val_binary_accuracy"] = (tp_test + tn_test) / (
        tn_test + fp_test + fn_test + tp_test + sys.float_info.epsilon
    )
    row["val_recall"] = tp_test / (tp_test + fn_test + sys.float_info.epsilon)
    row["val_precision"] = tp_test / (tp_test + fp_test + sys.float_info.epsilon)
    row["val_specificity"] = tn_test / (tn_test + fp_test + sys.float_info.epsilon)
    row["val_f1score"] = (
        2
        * (row["val_precision"] * row["val_recall"])
        / (row["val_precision"] + row["val_recall"] + sys.float_info.epsilon)
    )
    try:
        row["val_auc"] = roc_auc_score(y_test, test_predictions_baseline, labels=[0, 1])
    except ValueError:
        row["val_auc"] = 1

    return row


def generate_roc_data(
    train_predictions_baseline,
    test_predictions_baseline,
    attack_parameter,
    node,
    y_train,
    y_test,
):
    train_fpr, train_tpr, train_thresholds = sklearn.metrics.roc_curve(
        y_train, train_predictions_baseline, drop_intermediate=False
    )
    tmp_train_roc_data = pd.DataFrame()
    tmp_train_roc_data["threshold"] = train_thresholds
    tmp_train_roc_data["fpr"] = train_fpr
    tmp_train_roc_data["tpr"] = train_tpr
    tmp_train_roc_data["node"] = node
    tmp_train_roc_data["k"] = attack_parameter

    test_fpr, test_tpr, test_thresholds = sklearn.metrics.roc_curve(
        y_test, test_predictions_baseline, drop_intermediate=False
    )
    tmp_test_roc_data = pd.DataFrame()
    tmp_test_roc_data["threshold"] = test_thresholds
    tmp_test_roc_data["fpr"] = test_fpr
    tmp_test_roc_data["tpr"] = test_tpr
    tmp_test_roc_data["node"] = node
    tmp_test_roc_data["k"] = attack_parameter

    return tmp_train_roc_data, tmp_test_roc_data


def generate_roc_clean_data(train_roc_data, test_roc_data, metric, mode, output_path):
    nodes = list(train_roc_data["node"].unique())
    k_list = list(train_roc_data["k"].unique())
    k_list = np.round(k_list, 2)
    roc_data = pd.DataFrame()

    for node in nodes:
        for k in k_list:
            print(node, " ", k)
            tmp_train_roc_data = train_roc_data.loc[
                (train_roc_data["node"] == node) & (train_roc_data["k"] == k)
            ]
            tmp_train_roc_data.sort_values(by=["threshold"])
            tmp_test_roc_data = test_roc_data.loc[
                (test_roc_data["node"] == node) & (test_roc_data["k"] == k)
            ]
            tmp_test_roc_data.sort_values(by=["threshold"])
            for threshold in np.arange(0.0, 1.01, 0.01):
                nearest_record = tmp_train_roc_data.iloc[
                    (tmp_train_roc_data["threshold"] - threshold).abs().argsort()[:1]
                ].reset_index(drop=True)
                nearest_record.loc[0, "threshold"] = threshold
                nearest_record_test = tmp_test_roc_data.iloc[
                    (tmp_test_roc_data["threshold"] - threshold).abs().argsort()[:1]
                ].reset_index(drop=True)
                nearest_record["val_tpr"] = nearest_record_test["tpr"]
                nearest_record["val_fpr"] = nearest_record_test["fpr"]

                roc_data = pd.concat([roc_data, nearest_record])

    output_path_clean_roc = (
        output_path + "roc_clean_report_" + metric + "_" + mode + ".csv"
    )
    roc_data.to_csv(output_path_clean_roc, index=False)

    roc_data = roc_data.groupby(["k", "threshold"]).mean().reset_index()
    roc_data = roc_data.sort_values(by=["k", "threshold"], ascending=False)
    output_path_mean_roc = (
        output_path + "roc_mean_report_" + metric + "_" + mode + ".csv"
    )
    roc_data.to_csv(output_path_mean_roc, index=False)


def generate_shap_values(
    nn_model, loaded_model, node, k, feature_columns, time_window, X_train, X_test
):
    shap_values_df = pd.DataFrame()
    X_test_df = pd.DataFrame()
    feature_columns.remove("ATTACKED")
    if nn_model == "dense" or nn_model == "aen":
        feature_columns_original = feature_columns.copy()
        for index in range(1, time_window):
            feature_columns_tmp = [
                i + "_" + str(index) for i in feature_columns_original
            ]
            feature_columns.extend(feature_columns_tmp)
    X_train = np.array(random.choices(X_train, k=1000))
    X_test = np.array(random.choices(X_test, k=1000))
    explainer = shap.DeepExplainer(loaded_model, X_train)
    shap_values = explainer.shap_values(X_test)
    shap_values = np.array(shap_values[0])
    for index, shap_value in enumerate(shap_values):
        if nn_model == "dense" or nn_model == "aen":
            shap_value = shap_value.reshape(1, len(shap_value))
        tmp_df = pd.DataFrame(shap_value, columns=feature_columns)
        tmp_df["sample"] = index
        tmp_df["node"] = node
        tmp_df["k"] = k
        shap_values_df = pd.concat([shap_values_df, tmp_df])
    for index, x_t in enumerate(X_test):
        if nn_model == "dense" or nn_model == "aen":
            x_t = x_t.reshape(1, len(x_t))
        tmp_df = pd.DataFrame(x_t, columns=feature_columns)
        tmp_df["sample"] = index
        tmp_df["node"] = node
        tmp_df["k"] = k
        X_test_df = pd.concat([X_test_df, tmp_df])

    X_test_df = X_test_df.reset_index(drop=True)
    shap_values_df = shap_values_df.reset_index(drop=True)
    return X_test_df, shap_values_df


def generate_report_data(
    nn_model,
    architecture,
    train_dataset,
    test_dataset,
    model_path_input,
    time_window,
    use_time_hour,
    use_onehot,
    num_labels,
    metric,
    mode,
    output_path,
):
    """
    Generate a report on different metrics like accuracy, loss, etc of the trained model
    Args:
        train_dataset: Training dataset
        test_dataset: Testing dataset
        model_path_input: Path to the NN model used for training
        time_window: The time window used for generating the training/testing dataset
        metric: The metric used for loading weights to the model
        mode: The ID of the node for loading model's weight
        output_path: Output path for storing the reports

    Returns:
        --
    """
    metrics_evaluation_df = pd.DataFrame()
    roc_train_df = pd.DataFrame()
    roc_test_df = pd.DataFrame()
    attack_vs_time_train_df = pd.DataFrame()
    attack_vs_time_test_df = pd.DataFrame()
    shap_values_df = pd.DataFrame()
    shap_x_test_df = pd.DataFrame()

    nodes = list(train_dataset["NODE"].unique())
    if (
        architecture == "one_model_with_correlation"
        or architecture == "one_model_without_correlation"
    ):
        nodes = ["one_model"]

    k_list = list(train_dataset["ATTACK_PARAMETER"].unique())
    k_list = np.round(k_list, 2)
    thresholds = {}
    for k in k_list:
        for node in nodes:
            tf.keras.backend.clear_session()
            model, scaler, selected_nodes_for_correlation = load_model_weights(
                model_path_input, node, metric, mode
            )
            if node not in thresholds:
                thresholds[node] = find_best_threshold(
                    model,
                    nn_model,
                    architecture,
                    node,
                    train_dataset,
                    selected_nodes_for_correlation,
                    num_labels,
                    time_window,
                    scaler,
                    use_time_hour,
                    use_onehot,
                )
                # thresholds[node] = 0.5
            train_dataset_node = pd.DataFrame()
            test_dataset_node = pd.DataFrame()
            if (
                architecture == "multiple_models_with_correlation"
                or architecture == "multiple_models_without_correlation"
            ):
                train_dataset_node = train_dataset.loc[
                    (train_dataset["NODE"] == node)
                    & (train_dataset["ATTACK_PARAMETER"] == k)
                ]
                test_dataset_node = test_dataset.loc[
                    (test_dataset["NODE"] == node)
                    & (test_dataset["ATTACK_PARAMETER"] == k)
                ]
            elif (
                architecture == "one_model_with_correlation"
                or architecture == "one_model_without_correlation"
            ):
                train_dataset_node = train_dataset.loc[
                    (train_dataset["ATTACK_PARAMETER"] == k)
                ]
                test_dataset_node = test_dataset.loc[
                    (test_dataset["ATTACK_PARAMETER"] == k)
                ]

            X_train, y_train, df_train, feature_columns = get_dataset_input_output(
                nn_model,
                architecture,
                train_dataset_node,
                selected_nodes_for_correlation,
                num_labels,
                time_window,
                scaler,
                use_time_hour,
                use_onehot,
            )
            X_test, y_test, df_test, feature_columns = get_dataset_input_output(
                nn_model,
                architecture,
                test_dataset_node,
                selected_nodes_for_correlation,
                num_labels,
                time_window,
                scaler,
                use_time_hour,
                use_onehot,
            )

            # General metrics evaluation
            train_predictions_baseline = model.predict(X_train)
            test_predictions_baseline = model.predict(X_test)
            metrics_evaluation_row = {"k": k, "node": node}
            metrics_evaluation = generate_metrics_evaluation(
                metrics_evaluation_row,
                train_predictions_baseline,
                test_predictions_baseline,
                y_train,
                y_test,
                thresholds[node],
            )
            metrics_evaluation_df = pd.concat(
                [metrics_evaluation_df, pd.DataFrame(metrics_evaluation, index=[0])]
            )

            # Roc curve
            tmp_roc_train_data, tmp_roc_test_data = generate_roc_data(
                train_predictions_baseline,
                test_predictions_baseline,
                k,
                node,
                y_train,
                y_test,
            )
            roc_train_df = pd.concat([roc_train_df, tmp_roc_train_data])
            roc_test_df = pd.concat([roc_test_df, tmp_roc_test_data])

            # Attack prediction vs time
            tmp_a_vs_t_train, tmp_a_vs_t_test = generate_attack_prediction_vs_time_data(
                train_predictions_baseline,
                test_predictions_baseline,
                thresholds[node],
                df_train,
                y_train,
                df_test,
                y_test,
            )
            attack_vs_time_train_df = pd.concat(
                [attack_vs_time_train_df, tmp_a_vs_t_train]
            )
            attack_vs_time_test_df = pd.concat(
                [attack_vs_time_test_df, tmp_a_vs_t_test]
            )

            # Feature importance analysis
            if nn_model != "trans":
                tmp_shap_X_test_df, tmp_shap_values_df = generate_shap_values(
                    nn_model,
                    model,
                    node,
                    k,
                    feature_columns,
                    time_window,
                    X_train,
                    X_test,
                )
                shap_x_test_df = pd.concat([shap_x_test_df, tmp_shap_X_test_df])
                shap_values_df = pd.concat([shap_values_df, tmp_shap_values_df])

    # General metrics evaluation
    output_path_metrics_evaluation_report = (
        output_path
        + "/metrics_evaluation/data/general_report_"
        + metric
        + "_"
        + mode
        + ".csv"
    )
    prepare_output_directory(output_path_metrics_evaluation_report)
    metrics_evaluation_df.to_csv(output_path_metrics_evaluation_report, index=False)
    metrics_evaluation_df = metrics_evaluation_df.groupby(["k"]).mean().reset_index()
    output_path_metrics_evaluation_mean_report = (
        output_path
        + "/metrics_evaluation/data/mean_report_"
        + metric
        + "_"
        + mode
        + ".csv"
    )
    metrics_evaluation_df.to_csv(
        output_path_metrics_evaluation_mean_report, index=False
    )

    # Roc curve
    roc_train_df = roc_train_df[["k", "node", "threshold", "tpr", "fpr"]]
    roc_train_output_path = (
        output_path + "/roc/data/roc_train_report_" + metric + "_" + mode + ".csv"
    )
    prepare_output_directory(roc_train_output_path)
    roc_train_df.to_csv(roc_train_output_path, index=False)
    roc_test_df = roc_test_df[["k", "node", "threshold", "tpr", "fpr"]]
    roc_test_output_path = (
        output_path + "/roc/data/roc_test_report_" + metric + "_" + mode + ".csv"
    )
    prepare_output_directory(roc_test_output_path)
    roc_test_df.to_csv(roc_test_output_path, index=False)
    roc_clean_mean_path = output_path + "/roc/data/"
    generate_roc_clean_data(
        roc_train_df, roc_test_df, metric, mode, roc_clean_mean_path
    )

    # Attack prediction vs time
    attack_prediction_vs_time_output_path = (
        output_path + "/attack_prediction_vs_time/data/"
    )
    prepare_output_directory(attack_prediction_vs_time_output_path)
    attack_vs_time_train_df.to_csv(
        attack_prediction_vs_time_output_path
        + "attack_prediction_vs_time_train_data_"
        + metric
        + "_"
        + mode
        + ".csv",
        index=False,
    )
    attack_vs_time_test_df.to_csv(
        attack_prediction_vs_time_output_path
        + "attack_prediction_vs_time_test_data_"
        + metric
        + "_"
        + mode
        + ".csv",
        index=False,
    )
    # Attack detection stats - NOTE: depended on running attack prediction vs time
    input_dataset_path = (
        attack_prediction_vs_time_output_path
        + "attack_prediction_vs_time_test_data_"
        + metric
        + "_"
        + mode
        + ".csv"
    )
    attack_detection_stats_output_path = (
        output_path + "/attack_detection_stats/attack_detection_stats_test_data.txt"
    )
    prepare_output_directory(attack_detection_stats_output_path)
    generate_attack_detection_stats(
        input_dataset_path, attack_detection_stats_output_path
    )

    # Feature importance analysis
    if nn_model != "trans":
        shap_output_path = output_path + "/shap/data/"
        prepare_output_directory(shap_output_path)
        shap_x_test_df.to_csv(
            shap_output_path + "shap_x_test_df_" + metric + "_" + mode + ".csv",
            index=False,
        )
        shap_values_df.to_csv(
            shap_output_path + "shap_values_df_" + metric + "_" + mode + ".csv",
            index=False,
        )
        input_data_path = output_path
        main_plot_shap_values(input_data_path, metric, mode)
    # sys.exit()


def main_general_report(
    nn_model,
    architecture,
    metadata_metric,
    metric,
    mode,
    train_dataset_path,
    test_dataset_path,
    time_window,
    use_time_hour,
    use_onehot,
    num_labels,
    run_number,
    group_number,
):
    """
    The main function for calling the generate_general_report function
    Args:
        metric: The metric to be used for loading NN model's weights
        mode: The mode to be used for loading NN model's weights
        time_window: The time window used for generating the training dataset

    Returns:
        --
    """
    model_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/metadata_"
        + metadata_metric
        + "/"
        + nn_model
        + "/"
        + architecture
        + "/"
        + "run_"
        + str(run_number)
        + "/saved_model/"
    )
    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)

    output_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/metadata_"
        + metadata_metric
        + "/"
        + nn_model
        + "/"
        + architecture
        + "/"
        + "run_"
        + str(run_number)
        + "/report/"
    )
    generate_report_data(
        nn_model,
        architecture,
        train_dataset,
        test_dataset,
        model_path,
        time_window,
        use_time_hour,
        use_onehot,
        num_labels,
        metric,
        mode,
        output_path,
    )


def main_final_general_report(
    nn_model, architecture, metadata_metric, metric, mode, group_number
):
    """
    The main function for calling the generate_general_report function
    Args:
        metric: The metric to be used for loading NN model's weights
        mode: The mode to be used for loading NN model's weights
        time_window: The time window used for generating the training dataset

    Returns:
        --
    """
    mean_report_final = pd.DataFrame()
    roc_report_final = pd.DataFrame()
    all_runs_report_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/metadata_"
        + metadata_metric
        + "/"
        + nn_model
        + "/"
        + architecture
        + "/"
    )
    os.system("rm -rf " + all_runs_report_path + "final_report/")
    for run_report_path in glob.glob(all_runs_report_path + "*"):
        tmp_path = run_report_path + "/report/"

        mean_report_path = (
            tmp_path
            + "metrics_evaluation/data/mean_report_"
            + metric
            + "_"
            + mode
            + ".csv"
        )
        mean_report_run = pd.read_csv(mean_report_path)
        mean_report_final = pd.concat([mean_report_final, mean_report_run])

        roc_report_path = (
            tmp_path + "roc/data/roc_mean_report_" + metric + "_" + mode + ".csv"
        )
        roc_report_run = pd.read_csv(roc_report_path)
        roc_report_final = pd.concat([roc_report_final, roc_report_run])

    final_report_path = all_runs_report_path + "final_report/"

    mean_report_final = mean_report_final.groupby(["k"]).mean().reset_index()
    mean_report_final_path = (
        final_report_path
        + "metrics_evaluation/data/mean_report_"
        + metric
        + "_"
        + mode
        + ".csv"
    )
    prepare_output_directory(mean_report_final_path)
    mean_report_final.to_csv(mean_report_final_path, index=False)

    roc_report_final = roc_report_final.groupby(["k", "threshold"]).mean().reset_index()
    roc_report_final_path = (
        final_report_path + "roc/data/roc_mean_report_" + metric + "_" + mode + ".csv"
    )
    prepare_output_directory(roc_report_final_path)
    roc_report_final.to_csv(roc_report_final_path, index=False)


def plot_metric_vs_attack_parameter(mean_report, output_path):
    """
    Plot different metrics values vs k
    Args:
        mean_report: The mean report dataset generated by generate_general_report function
        output_path: Output path for storing the plots

    Returns:
        --
    """
    plt.clf()
    metrics = mean_report.columns.values

    for metric in metrics:
        if metric == "k" or metric == "node" or "val" in metric:
            continue
        plt.clf()
        plt.plot(mean_report["k"], mean_report[metric], label="Train")
        plt.plot(mean_report["k"], mean_report["val_" + metric], label="Test")
        plt.xlabel("Attack Parameter")
        plt.ylabel(metric)
        plt.title(metric + " vs attack parameter")
        plt.legend()
        plt.savefig(output_path + metric + ".png")


def main_plot_metric_vs_attack_parameter(
    nn_model, architecture, metadata_metric, metric, mode, run_number, group_number
):
    """
    The main function for calling the metric_vs_attack_parameter function
    Args:
        metric: The metric to be used for loading NN model's weights
        mode: The mode to be used for loading NN model's weights

    Returns:
        --
    """
    if run_number == -1:
        general_report_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "nn_training/group_"
            + str(group_number)
            + "/metadata_"
            + metadata_metric
            + "/"
            + nn_model
            + "/"
            + architecture
            + "/final_report/metrics_evaluation/"
        )
    else:
        general_report_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "nn_training/group_"
            + str(group_number)
            + "/metadata_"
            + metadata_metric
            + "/"
            + nn_model
            + "/"
            + architecture
            + "/"
            + "run_"
            + str(run_number)
            + "/report/metrics_evaluation/"
        )

    mean_report_path = (
        general_report_path + "data/mean_report_" + metric + "_" + mode + ".csv"
    )
    mean_report = pd.read_csv(mean_report_path)

    output_path = general_report_path + "plot/"
    prepare_output_directory(output_path)

    plot_metric_vs_attack_parameter(mean_report, output_path)


def generate_attack_prediction_vs_time_data(
    train_predictions_baseline,
    test_predictions_baseline,
    threshold,
    df_train,
    y_train,
    df_test,
    y_test,
):
    y_train = np.rint(y_train).astype(bool)
    y_pred_train = train_predictions_baseline > threshold
    df_train["TRUE"] = y_train
    df_train["PRED"] = y_pred_train
    df_train["TP"] = df_train["ATTACKED"] & df_train["PRED"]
    df_train["FP"] = df_train["TP"] ^ df_train["PRED"]
    df_train["TN"] = ~(df_train["ATTACKED"] | df_train["PRED"])
    df_train["FN"] = ~(df_train["TN"] | df_train["PRED"])
    df_train[["TRUE", "PRED", "TP", "TN", "FP", "FN"]] = df_train[
        ["TRUE", "PRED", "TP", "TN", "FP", "FN"]
    ].astype(int)

    y_test = np.rint(y_test).astype(bool)
    y_pred_test = test_predictions_baseline > threshold
    df_test["TRUE"] = y_test
    df_test["PRED"] = y_pred_test
    df_test["TP"] = df_test["ATTACKED"] & df_test["PRED"]
    df_test["FP"] = df_test["TP"] ^ df_test["PRED"]
    df_test["TN"] = ~(df_test["ATTACKED"] | df_test["PRED"])
    df_test["FN"] = ~(df_test["TN"] | df_test["PRED"])
    df_test[["TRUE", "PRED", "TP", "TN", "FP", "FN"]] = df_test[
        ["TRUE", "PRED", "TP", "TN", "FP", "FN"]
    ].astype(int)

    return df_train, df_test


def plot_attack_prediction_vs_time(
    train_result_path, test_result_path, train_output_path, test_output_path
):
    """
    Plot the attack prediction vs time for the training/testing dataset
    Args:
        train_result_path: Path to the attack prediction vs time results generated by generate_attack_prediction_vs_time function for the training dataset
        test_result_path: Path to the attack prediction vs time results generated by generate_attack_prediction_vs_time function for the testing dataset
        train_output_path: Path to store the attack prediction vs time plot for the training dataset
        test_output_path: Path to store the attack prediction vs time plot for the testing dataset

    Returns:
        --
    """
    train_result = load_dataset(train_result_path)
    attack_ratios = list(train_result["ATTACK_RATIO"].unique())
    attack_durations = list(train_result["ATTACK_DURATION"].unique())
    attack_start_times = list(train_result["ATTACK_START_TIME"].unique())
    k_list = list(train_result["ATTACK_PARAMETER"].unique())
    k_list = np.round(k_list, 2)
    prepare_output_directory(train_output_path)
    for k in k_list:
        for attack_ratio in attack_ratios:
            for attack_start_time in attack_start_times:
                for attack_duration in attack_durations:
                    plot_data = train_result.loc[
                        (train_result["ATTACK_RATIO"] == attack_ratio)
                        & (train_result["ATTACK_DURATION"] == attack_duration)
                        & (train_result["ATTACK_START_TIME"] == attack_start_time)
                        & (train_result["ATTACK_PARAMETER"] == k)
                    ]
                    if plot_data.shape[0] == 0:
                        continue
                    plot_data = plot_data.groupby(["TIME"]).mean().reset_index()
                    plot_data = plot_data.sort_values(by=["TIME"])
                    plt.clf()
                    fig, ax = plt.subplots()
                    ax.plot(plot_data["TIME"], plot_data["TRUE"], label="True")
                    ax.plot(plot_data["TIME"], plot_data["TP"], label="TP")
                    ax.plot(plot_data["TIME"], plot_data["FP"], label="FP")
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                    myFmt = mdates.DateFormatter("%H")
                    ax.xaxis.set_major_formatter(myFmt)
                    ax.legend()
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Attack Ratio")
                    # ax.set_title('Attack Ratio= ' + str(attack_ratio) + ' - Duration: ' + str(attack_duration) + '\n' +
                    #              'K= ' + str(k))

                    output_path = (
                        train_output_path
                        + "train_attack_prediction_vs_time_"
                        + str(attack_duration)
                        + "_attackRatio_"
                        + str(attack_ratio)
                        + "_startTime_"
                        + str(attack_start_time)
                        + "_duration_"
                        + str(attack_duration)
                        + "_k_"
                        + str(k)
                        + ".png"
                    )
                    fig.savefig(output_path)

    test_result = load_dataset(test_result_path)
    attack_ratios = list(test_result["ATTACK_RATIO"].unique())
    attack_durations = list(test_result["ATTACK_DURATION"].unique())
    attack_start_times = list(test_result["ATTACK_START_TIME"].unique())
    k_list = list(test_result["ATTACK_PARAMETER"].unique())
    k_list = np.round(k_list, 2)
    prepare_output_directory(test_output_path)

    for k in k_list:
        for attack_ratio in attack_ratios:
            for attack_start_time in attack_start_times:
                for attack_duration in attack_durations:
                    plot_data = test_result.loc[
                        (test_result["ATTACK_RATIO"] == attack_ratio)
                        & (test_result["ATTACK_DURATION"] == attack_duration)
                        & (test_result["ATTACK_START_TIME"] == attack_start_time)
                        & (test_result["ATTACK_PARAMETER"] == k)
                    ]
                    if plot_data.shape[0] == 0:
                        continue
                    plot_data = plot_data.groupby(["TIME"]).mean().reset_index()
                    plot_data = plot_data.sort_values(by=["TIME"])
                    plt.clf()
                    fig, ax = plt.subplots()
                    ax.plot(plot_data["TIME"], plot_data["TRUE"], label="True")
                    ax.plot(plot_data["TIME"], plot_data["TP"], label="TP")
                    ax.plot(plot_data["TIME"], plot_data["FP"], label="FP")
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                    myFmt = mdates.DateFormatter("%H")
                    ax.xaxis.set_major_formatter(myFmt)
                    ax.legend()
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Attack Ratio")
                    # ax.set_title('Attack_Ratio= ' + str(attack_ratio) + ' - Duration: ' + str(attack_duration) + '\n' +
                    #              'K= ' + str(k))

                    output_path = (
                        test_output_path
                        + "test_attack_prediction_vs_time_"
                        + str(attack_duration)
                        + "_attackRatio_"
                        + str(attack_ratio)
                        + "_startTime_"
                        + str(attack_start_time)
                        + "_duration_"
                        + str(attack_duration)
                        + "_k_"
                        + str(k)
                        + ".png"
                    )
                    fig.savefig(output_path)


def main_plot_attack_prediction_vs_time(
    nn_model, architecture, metadata_metric, metric, mode, run_number, group_number
):
    """
    The main function for calling the plot_attack_prediction_vs_time function
    Args:
        metric: The metric to be used for loading NN model's weights
        mode: The mode to be used for loading NN model's weights

    Returns:
        --
    """
    if run_number == -1:
        attack_prediction_vs_time_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "nn_training/group_"
            + str(group_number)
            + "/metadata_"
            + metadata_metric
            + "/"
            + nn_model
            + "/"
            + architecture
            + "/final_report/attack_prediction_vs_time/"
        )
    else:
        attack_prediction_vs_time_path = (
            CONFIG.OUTPUT_DIRECTORY
            + "nn_training/group_"
            + str(group_number)
            + "/metadata_"
            + metadata_metric
            + "/"
            + nn_model
            + "/"
            + architecture
            + "/"
            + "run_"
            + str(run_number)
            + "/report/attack_prediction_vs_time/"
        )

    train_result_path = (
        attack_prediction_vs_time_path
        + "data/attack_prediction_vs_time_train_data_"
        + metric
        + "_"
        + mode
        + ".csv"
    )
    test_result_path = (
        attack_prediction_vs_time_path
        + "data/attack_prediction_vs_time_test_data_"
        + metric
        + "_"
        + mode
        + ".csv"
    )

    train_output_path = attack_prediction_vs_time_path + "/plot/train/"
    test_output_path = attack_prediction_vs_time_path + "/plot/test/"
    prepare_output_directory(train_output_path)
    prepare_output_directory(test_output_path)

    plot_attack_prediction_vs_time(
        train_result_path, test_result_path, train_output_path, test_output_path
    )


def generate_attack_detection_stats(dataset_path, output_path):
    data = pd.read_csv(dataset_path)
    data = data.sort_values(
        by=[
            "NODE",
            "ATTACK_RATIO",
            "ATTACK_START_TIME",
            "ATTACK_DURATION",
            "ATTACK_PARAMETER",
            "TIME",
        ]
    ).reset_index(drop=True)
    TP_slots = [0]
    TN_slots = [0]
    FP_slots = [0]
    FN_slots = [0]
    for index, row in data.iterrows():
        if row["TP"] == 0 and TP_slots[-1] != 0:
            TP_slots.append(0)
        elif row["TP"] == 1:
            TP_slots[-1] += 1

        if row["TN"] == 0 and TN_slots[-1] != 0:
            TN_slots.append(0)
        elif row["TN"] == 1:
            TN_slots[-1] += 1

        if row["FP"] == 0 and FP_slots[-1] != 0:
            FP_slots.append(0)
        elif row["FP"] == 1:
            FP_slots[-1] += 1

        if row["FN"] == 0 and FN_slots[-1] != 0:
            FN_slots.append(0)
        elif row["FN"] == 1:
            FN_slots[-1] += 1

    fout = open(output_path, "w")
    TP_mean_slots = statistics.mean(TP_slots)
    fout.write("TP_mean_slots: " + str(TP_mean_slots) + "\n")
    TN_mean_slots = statistics.mean(TN_slots)
    fout.write("TN_mean_slots: " + str(TN_mean_slots) + "\n")
    FP_mean_slots = statistics.mean(FP_slots)
    fout.write("FP_mean_slots: " + str(FP_mean_slots) + "\n")
    FN_mean_slots = statistics.mean(FN_slots)
    fout.write("FN_mean_slots: " + str(FN_mean_slots) + "\n")

    data_2 = data.loc[data["TRUE"] == 1]
    data_2 = (
        data_2.groupby(
            [
                "ATTACK_RATIO",
                "ATTACK_START_TIME",
                "ATTACK_DURATION",
                "ATTACK_PARAMETER",
                "TIME",
            ]
        )
        .mean()
        .reset_index()
    )
    TP_mean_all_nodes = data_2["TP"].mean()
    fout.write("TP_mean_all_nodes: " + str(TP_mean_all_nodes) + "\n")
    FN_mean_all_nodes = data_2["FN"].mean()
    fout.write("FN_mean_all_nodes: " + str(FN_mean_all_nodes) + "\n")

    data_2 = data.loc[data["TRUE"] == 0]
    data_2 = (
        data_2.groupby(
            [
                "ATTACK_RATIO",
                "ATTACK_START_TIME",
                "ATTACK_DURATION",
                "ATTACK_PARAMETER",
                "TIME",
            ]
        )
        .mean()
        .reset_index()
    )
    TN_mean_all_nodes = data_2["TN"].mean()
    fout.write("TN_mean_all_nodes: " + str(TN_mean_all_nodes) + "\n")
    FP_mean_all_nodes = data_2["FP"].mean()
    fout.write("FP_mean_all_nodes: " + str(FP_mean_all_nodes) + "\n")
    fout.close()


def plot_shap_values(input_data_path, metric, mode):
    shap_x_test_df_path = (
        input_data_path + "/shap/data/shap_x_test_df_" + metric + "_" + mode + ".csv"
    )
    shap_values_df_path = (
        input_data_path + "/shap/data/shap_values_df_" + metric + "_" + mode + ".csv"
    )
    shap_x_test_df = pd.read_csv(shap_x_test_df_path)
    shap_values_df = pd.read_csv(shap_values_df_path)
    output_path = input_data_path + "/shap/plot/"
    prepare_output_directory(output_path)
    nodes = shap_values_df["node"].unique()
    for node in nodes:
        shap_values_tmp = shap_values_df.loc[shap_values_df["node"] == node]
        columns = list(shap_values_tmp.columns)
        columns.remove("node")
        columns.remove("sample")
        columns.remove("k")
        shap_values = shap_values_tmp[columns].to_numpy()
        shap_x_test_tmp = shap_x_test_df.loc[shap_x_test_df["node"] == node]
        shap_x_test = shap_x_test_tmp[columns].to_numpy()
        plt.clf()
        shap.summary_plot(
            shap_values, shap_x_test, feature_names=columns, max_display=5, show=False
        )
        # shap.waterfall_plot(shap_values, show=False)
        plt.savefig(output_path + "node_" + str(node) + ".png")


def save_most_important_features(input_data_path, metric, mode):
    shap_values_df_path = (
        input_data_path + "/shap/data/shap_values_df_" + metric + "_" + mode + ".csv"
    )
    shap_values_df = pd.read_csv(shap_values_df_path)

    feature_importance_df = pd.DataFrame()
    nodes = shap_values_df["node"].unique()
    for node in nodes:
        shap_values_tmp = shap_values_df.loc[shap_values_df["node"] == node]
        columns = list(shap_values_tmp.columns)
        columns.remove("node")
        columns.remove("sample")
        columns.remove("k")
        shap_values = shap_values_tmp[columns].to_numpy()

        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(columns, vals)),
            columns=["feature_name", "feature_importance_vals"],
        )
        feature_importance = feature_importance.sort_values(
            by=["feature_importance_vals"], ascending=False
        ).reset_index(drop=True)
        feature_importance["node"] = node
        feature_importance["feature_node"] = feature_importance["feature_name"].str[7:]
        feature_importance_df = pd.concat([feature_importance_df, feature_importance])

    feature_importance_df = feature_importance_df[
        ["node", "feature_node", "feature_name", "feature_importance_vals"]
    ]
    output_path = input_data_path + "/shap/feature_importance/feature_importance.csv"
    prepare_output_directory(output_path)
    feature_importance_df.to_csv(output_path, index=False)


def main_plot_shap_values(input_data_path, metric, mode):
    plot_shap_values(input_data_path, metric, mode)
    save_most_important_features(input_data_path, metric, mode)


def generate_attack_properties_analysis_data(
    attack_property_name,
    nn_model,
    architecture,
    train_dataset,
    test_dataset,
    model_path_input,
    time_window,
    use_time_hour,
    use_onehot,
    num_labels,
    metric,
    mode,
    output_path,
):
    """
    Generate a report on different metrics like accuracy, loss, etc of the trained model
    Args:
        train_dataset: Training dataset
        test_dataset: Testing dataset
        model_path_input: Path to the NN model used for training
        time_window: The time window used for generating the training/testing dataset
        metric: The metric used for loading weights to the model
        mode: The ID of the node for loading model's weight
        output_path: Output path for storing the reports

    Returns:
        --
    """

    nodes = list(train_dataset["NODE"].unique())
    if (
        architecture == "one_model_with_correlation"
        or architecture == "one_model_without_correlation"
    ):
        nodes = ["one_model"]

    metrics_evaluation_df = pd.DataFrame()
    thresholds = {}
    attack_property_value_list = list(test_dataset[attack_property_name].unique())
    if attack_property_name == "ATTACK_PARAMETER":
        attack_property_value_list = np.round(attack_property_value_list, 2)
    elif attack_property_name == "ATTACK_START_TIME":
        test_attack_start_times = sorted(
            list(test_dataset[attack_property_name].unique())
        )
        train_attack_start_times = sorted(
            list(train_dataset[attack_property_name].unique())
        )
        for index in range(0, len(train_attack_start_times)):
            train_dataset["ATTACK_START_TIME"] = train_dataset[
                "ATTACK_START_TIME"
            ].replace(train_attack_start_times[index], test_attack_start_times[index])
        attack_property_value_list = test_attack_start_times

    # attack_property_value_list = [0.0]
    for attack_property_value in attack_property_value_list:
        # nodes = [637, 1152]
        for node in nodes:
            tf.keras.backend.clear_session()
            model, scaler, selected_nodes_for_correlation = load_model_weights(
                model_path_input, node, metric, mode
            )
            if node not in thresholds:
                thresholds[node] = find_best_threshold(
                    model,
                    nn_model,
                    architecture,
                    node,
                    train_dataset,
                    selected_nodes_for_correlation,
                    num_labels,
                    time_window,
                    scaler,
                    use_time_hour,
                    use_onehot,
                )
                # thresholds[node] = 0.5

            train_dataset_node = pd.DataFrame()
            test_dataset_node = pd.DataFrame()
            if (
                architecture == "multiple_models_with_correlation"
                or architecture == "multiple_models_without_correlation"
            ):
                train_dataset_node = train_dataset.loc[
                    (train_dataset["NODE"] == node)
                    & (train_dataset[attack_property_name] == attack_property_value)
                ]
                test_dataset_node = test_dataset.loc[
                    (test_dataset["NODE"] == node)
                    & (test_dataset[attack_property_name] == attack_property_value)
                ]
            elif (
                architecture == "one_model_with_correlation"
                or architecture == "one_model_without_correlation"
            ):
                train_dataset_node = train_dataset.loc[
                    (train_dataset[attack_property_name] == attack_property_value)
                ]
                test_dataset_node = test_dataset.loc[
                    (test_dataset[attack_property_name] == attack_property_value)
                ]
            if (
                test_dataset_node.shape[0] == 0
                or test_dataset_node["ATTACK_RATIO"].values[0] == 0.0
            ):
                continue
            X_train, y_train, df_train, feature_columns = get_dataset_input_output(
                nn_model,
                architecture,
                train_dataset_node,
                selected_nodes_for_correlation,
                num_labels,
                time_window,
                scaler,
                use_time_hour,
                use_onehot,
            )
            X_test, y_test, df_test, feature_columns = get_dataset_input_output(
                nn_model,
                architecture,
                test_dataset_node,
                selected_nodes_for_correlation,
                num_labels,
                time_window,
                scaler,
                use_time_hour,
                use_onehot,
            )

            # General metrics evaluation
            train_predictions_baseline = model.predict(X_train)
            test_predictions_baseline = model.predict(X_test)
            metrics_evaluation_row = {
                "attack_property_name": attack_property_name,
                "attack_property_value": attack_property_value,
                "node": node,
            }
            metrics_evaluation = generate_metrics_evaluation(
                metrics_evaluation_row,
                train_predictions_baseline,
                test_predictions_baseline,
                y_train,
                y_test,
                thresholds[node],
            )
            metrics_evaluation_df = pd.concat(
                [metrics_evaluation_df, pd.DataFrame(metrics_evaluation, index=[0])]
            )

    output_path_metrics_evaluation_report = (
        output_path
        + "/attack_property_evaluation/data/"
        + attack_property_name
        + "_general_report_"
        + metric
        + "_"
        + mode
        + ".csv"
    )
    prepare_output_directory(output_path_metrics_evaluation_report)
    metrics_evaluation_df.to_csv(output_path_metrics_evaluation_report, index=False)
    metrics_evaluation_df = (
        metrics_evaluation_df.groupby(["attack_property_value"]).mean().reset_index()
    )
    output_path_metrics_evaluation_mean_report = (
        output_path
        + "/attack_property_evaluation/data/"
        + attack_property_name
        + "_mean_report_"
        + metric
        + "_"
        + mode
        + ".csv"
    )
    metrics_evaluation_df.to_csv(
        output_path_metrics_evaluation_mean_report, index=False
    )


def plot_metric_vs_attack_property(
    mean_report_path, output_path, attack_property_name, attack_property_legend
):
    """
    Plot different metrics values vs k
    Args:
        mean_report: The mean report dataset generated by generate_general_report function
        output_path: Output path for storing the plots

    Returns:
        --
    """
    mean_report = pd.read_csv(mean_report_path)
    metric_legends_list = {
        "binary_accuracy": "Binary Accuracy",
        "recall": "Recall",
        "precision": "Precision",
        "specificity": "Specificity",
        "f1score": "F1 Score",
        "auc": "Area Under Curve (AUC)",
    }
    for metric, metric_legend in metric_legends_list.items():
        if (
            metric == "attack_property_name"
            or metric == "attack_property_value"
            or metric == "node"
            or "val" in metric
        ):
            continue
        plt.clf()
        plt.plot(mean_report["attack_property_value"], mean_report["val_" + metric])
        plt.xlabel(attack_property_legend)
        plt.ylabel(metric_legend)
        plt.savefig(output_path + attack_property_name + "_" + metric + ".png")


def main_generate_attack_properties_analysis(
    nn_model,
    architecture,
    metadata_metric,
    metric,
    mode,
    train_dataset_path,
    test_dataset_path,
    time_window,
    use_time_hour,
    use_onehot,
    num_labels,
    run_number,
    group_number,
):
    model_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/metadata_"
        + metadata_metric
        + "/"
        + nn_model
        + "/"
        + architecture
        + "/"
        + "run_"
        + str(run_number)
        + "/saved_model/"
    )
    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)

    output_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/metadata_"
        + metadata_metric
        + "/"
        + nn_model
        + "/"
        + architecture
        + "/"
        + "run_"
        + str(run_number)
        + "/report/"
    )
    attack_property_name_list = {
        "ATTACK_RATIO": "Ratio of The Nodes Under Attack",
        "ATTACK_PARAMETER": "Attack Packet Volume Distribution (k)",
        "ATTACK_DURATION": "Attack Duration",
        "ATTACK_START_TIME": "Attack Start Time",
    }
    # attack_property_name_list = {
    #                              'ATTACK_START_TIME': 'Attack Start Time'}

    for (
        attack_property_name,
        attack_property_legend,
    ) in attack_property_name_list.items():
        generate_attack_properties_analysis_data(
            attack_property_name,
            nn_model,
            architecture,
            train_dataset,
            test_dataset,
            model_path,
            time_window,
            use_time_hour,
            use_onehot,
            num_labels,
            metric,
            mode,
            output_path,
        )
        mean_report_path = (
            output_path
            + "/attack_property_evaluation/data/"
            + attack_property_name
            + "_mean_report_"
            + metric
            + "_"
            + mode
            + ".csv"
        )
        plot_output_path = output_path + "/attack_property_evaluation/plot/"
        prepare_output_directory(plot_output_path)
        plot_metric_vs_attack_property(
            mean_report_path,
            plot_output_path,
            attack_property_name,
            attack_property_legend,
        )
        # sys.exit()


def generate_metrics_evaluation_for_all_attack_properties(
    attack_ratio,
    attack_duration,
    attack_start_time,
    k,
    node,
    train_predictions_baseline,
    test_predictions_baseline,
    y_train,
    y_test,
    threshold,
):
    row = {
        "attack_ratio": attack_ratio,
        "attack_duration": attack_duration,
        "attack_start_time": attack_start_time,
        "k": k,
        "node": node,
    }
    y_pred_train = (train_predictions_baseline > threshold).astype(int)
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(
        y_train, y_pred_train, labels=[0, 1]
    ).ravel()
    row["tp"] = tp_train
    row["tn"] = tn_train
    row["fp"] = fp_train
    row["fn"] = fn_train
    row["binary_accuracy"] = (tp_train + tn_train) / (
        tn_train + fp_train + fn_train + tp_train + sys.float_info.epsilon
    )
    row["recall"] = tp_train / (tp_train + fn_train + sys.float_info.epsilon)
    row["precision"] = tp_train / (tp_train + fp_train + sys.float_info.epsilon)
    row["specificity"] = tn_train / (tn_train + fp_train + sys.float_info.epsilon)
    row["f1score"] = (
        2
        * (row["precision"] * row["recall"])
        / (row["precision"] + row["recall"] + sys.float_info.epsilon)
    )
    try:
        row["auc"] = roc_auc_score(y_train, train_predictions_baseline, labels=[0, 1])
    except ValueError:
        row["auc"] = 1

    y_pred_test = (test_predictions_baseline > threshold).astype(int)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(
        y_test, y_pred_test, labels=[0, 1]
    ).ravel()
    row["val_tp"] = tp_test
    row["val_tn"] = tn_test
    row["val_fp"] = fp_test
    row["val_fn"] = fn_test
    row["val_binary_accuracy"] = (tp_test + tn_test) / (
        tn_test + fp_test + fn_test + tp_test + sys.float_info.epsilon
    )
    row["val_recall"] = tp_test / (tp_test + fn_test + sys.float_info.epsilon)
    row["val_precision"] = tp_test / (tp_test + fp_test + sys.float_info.epsilon)
    row["val_specificity"] = tn_test / (tn_test + fp_test + sys.float_info.epsilon)
    row["val_f1score"] = (
        2
        * (row["val_precision"] * row["val_recall"])
        / (row["val_precision"] + row["val_recall"] + sys.float_info.epsilon)
    )
    try:
        row["val_auc"] = roc_auc_score(y_test, test_predictions_baseline, labels=[0, 1])
    except ValueError:
        row["val_auc"] = 1

    return row


def generate_all_attack_properties_analysis_data(
    nn_model,
    architecture,
    train_dataset,
    test_dataset,
    model_path_input,
    time_window,
    use_time_hour,
    use_onehot,
    num_labels,
    metric,
    mode,
    output_path,
):
    """
    Generate a report on different metrics like accuracy, loss, etc of the trained model
    Args:
        train_dataset: Training dataset
        test_dataset: Testing dataset
        model_path_input: Path to the NN model used for training
        time_window: The time window used for generating the training/testing dataset
        metric: The metric used for loading weights to the model
        mode: The ID of the node for loading model's weight
        output_path: Output path for storing the reports

    Returns:
        --
    """

    nodes = list(test_dataset["NODE"].unique())
    if (
        architecture == "one_model_with_correlation"
        or architecture == "one_model_without_correlation"
    ):
        nodes = ["one_model"]
    attack_ratios_list = test_dataset["ATTACK_RATIO"].unique()
    attack_durations_list = test_dataset["ATTACK_DURATION"].unique()
    k_list = test_dataset["ATTACK_PARAMETER"].unique()

    test_attack_start_times = sorted(list(test_dataset["ATTACK_START_TIME"].unique()))
    train_attack_start_times = sorted(list(train_dataset["ATTACK_START_TIME"].unique()))
    for index in range(0, len(train_attack_start_times)):
        train_dataset["ATTACK_START_TIME"] = train_dataset["ATTACK_START_TIME"].replace(
            train_attack_start_times[index], test_attack_start_times[index]
        )
    attack_start_times_list = test_dataset["ATTACK_START_TIME"].unique()

    metrics_evaluation_df = pd.DataFrame()
    thresholds = {}

    for node in nodes:
        tf.keras.backend.clear_session()
        model, scaler, selected_nodes_for_correlation = load_model_weights(
            model_path_input, node, metric, mode
        )
        if node not in thresholds:
            thresholds[node] = find_best_threshold(
                model,
                nn_model,
                architecture,
                node,
                train_dataset,
                selected_nodes_for_correlation,
                num_labels,
                time_window,
                scaler,
                use_time_hour,
                use_onehot,
            )
            # thresholds[node] = 0.5
        train_dataset_node = pd.DataFrame()
        test_dataset_node = pd.DataFrame()
        if (
            architecture == "multiple_models_with_correlation"
            or architecture == "multiple_models_without_correlation"
        ):
            train_dataset_node = train_dataset.loc[(train_dataset["NODE"] == node)]
            test_dataset_node = test_dataset.loc[(test_dataset["NODE"] == node)]
        elif (
            architecture == "one_model_with_correlation"
            or architecture == "one_model_without_correlation"
        ):
            train_dataset_node = train_dataset
            test_dataset_node = test_dataset
        for attack_ratio in attack_ratios_list:
            train_dataset_attack_ratio = train_dataset_node.loc[
                (train_dataset_node["ATTACK_RATIO"] == attack_ratio)
            ]
            test_dataset_attack_ratio = test_dataset_node.loc[
                (test_dataset_node["ATTACK_RATIO"] == attack_ratio)
            ]
            for attack_duration in attack_durations_list:
                train_dataset_attack_duration = train_dataset_attack_ratio.loc[
                    (train_dataset_attack_ratio["ATTACK_DURATION"] == attack_duration)
                ]
                test_dataset_attack_duration = test_dataset_attack_ratio.loc[
                    (test_dataset_attack_ratio["ATTACK_DURATION"] == attack_duration)
                ]
                for attack_start_time in attack_start_times_list:
                    train_dataset_attack_start_time = train_dataset_attack_duration.loc[
                        (
                            train_dataset_attack_duration["ATTACK_START_TIME"]
                            == attack_start_time
                        )
                    ]
                    test_dataset_attack_start_time = test_dataset_attack_duration.loc[
                        (
                            test_dataset_attack_duration["ATTACK_START_TIME"]
                            == attack_start_time
                        )
                    ]
                    for k in k_list:
                        train_dataset_k = train_dataset_attack_start_time.loc[
                            (train_dataset_attack_start_time["ATTACK_PARAMETER"] == k)
                        ]
                        test_dataset_k = test_dataset_attack_start_time.loc[
                            (test_dataset_attack_start_time["ATTACK_PARAMETER"] == k)
                        ]
                        if (
                            test_dataset_k.shape[0] == 0
                            or test_dataset_k["ATTACK_RATIO"].values[0] == 0.0
                        ):
                            continue
                        (
                            X_train,
                            y_train,
                            df_train,
                            feature_columns,
                        ) = get_dataset_input_output(
                            nn_model,
                            architecture,
                            train_dataset_k,
                            selected_nodes_for_correlation,
                            num_labels,
                            time_window,
                            scaler,
                            use_time_hour,
                            use_onehot,
                        )
                        (
                            X_test,
                            y_test,
                            df_test,
                            feature_columns,
                        ) = get_dataset_input_output(
                            nn_model,
                            architecture,
                            test_dataset_k,
                            selected_nodes_for_correlation,
                            num_labels,
                            time_window,
                            scaler,
                            use_time_hour,
                            use_onehot,
                        )

                        # General metrics evaluation
                        train_predictions_baseline = model.predict(X_train)
                        test_predictions_baseline = model.predict(X_test)
                        metrics_evaluation = (
                            generate_metrics_evaluation_for_all_attack_properties(
                                attack_ratio,
                                attack_duration,
                                attack_start_time,
                                k,
                                node,
                                train_predictions_baseline,
                                test_predictions_baseline,
                                y_train,
                                y_test,
                                thresholds[node],
                            )
                        )
                        metrics_evaluation_df = pd.concat(
                            [
                                metrics_evaluation_df,
                                pd.DataFrame(metrics_evaluation, index=[0]),
                            ]
                        )

    output_path_metrics_evaluation_report = (
        output_path
        + "/attack_property_evaluation/data/"
        + "general_report_all_attack_properties_"
        + metric
        + "_"
        + mode
        + ".csv"
    )
    prepare_output_directory(output_path_metrics_evaluation_report)
    metrics_evaluation_df.to_csv(output_path_metrics_evaluation_report, index=False)


def plot_metric_vs_specific_attack_property_selection(report_path, output_path):
    """
    Plot different metrics values vs k
    Args:
        mean_report: The mean report dataset generated by generate_general_report function
        output_path: Output path for storing the plots

    Returns:
        --
    """
    colors = {0: "black", 1: "red", 2: "blue", 3: "green", 4: "magenta", 5: "yellow"}
    markers = {0: ".", 1: "v", 2: "+", 3: "x", 4: "s", 5: "^"}
    line_styles = {0: "solid", 1: "dashed"}
    data = pd.read_csv(report_path)
    metric_legends_list = {
        "binary_accuracy": "Binary Accuracy",
        "recall": "Recall",
        "precision": "Precision",
        "specificity": "Specificity",
        "f1score": "F1 Score",
        "auc": "Area Under Curve (AUC)",
    }
    data = data.loc[data["val_f1score"] > 0]
    # data['val_f1score'] = data['val_f1score'].replace(0, 1)

    # data = data.groupby(['k', 'attack_ratio', 'attack_duration', 'attack_start_time']).sum().reset_index()
    # data['val_recall'] = data['val_tp'] / (data['val_tp'] + data['val_fn'])
    # data['val_binary_accuracy'] = (data['val_tp'] + data['val_tn']) / (data['val_tn'] + data['val_fp'] + data['val_fn'] + data['val_tp'] )
    # data['val_recall'] = data['val_tp'] / (data['val_tp'] + data['val_fn'] )
    # data['val_precision'] = data['val_tp'] / (data['val_tp'] + data['val_fp'] )
    # data['val_specificity'] = data['val_tn'] / (data['val_tn'] + data['val_fp'] )
    # data['val_f1score'] = 2 * (data['val_precision'] * data['val_recall']) / (
    #             data['val_precision'] + data['val_recall'] )

    data = data.groupby(["k", "attack_ratio", "attack_duration"]).mean().reset_index()
    for metric, metric_legend in metric_legends_list.items():
        plot_data_1 = data.loc[
            (data["k"] == 0) & (data["attack_duration"] == "0 days 04:00:00")
        ]
        plot_data_2 = data.loc[
            (data["k"] == 0) & (data["attack_duration"] == "0 days 16:00:00")
        ]
        plot_data_3 = data.loc[
            (data["k"] == 1) & (data["attack_duration"] == "0 days 04:00:00")
        ]
        plot_data_4 = data.loc[
            (data["k"] == 1) & (data["attack_duration"] == "0 days 16:00:00")
        ]

        plt.clf()
        plt.plot(
            plot_data_1["attack_ratio"],
            plot_data_1["val_" + metric],
            color=colors[0],
            linestyle=line_styles[1],
            marker=markers[0],
            label=r"$k=0$, $a_d=4$ hours",
        )
        plt.plot(
            plot_data_2["attack_ratio"],
            plot_data_2["val_" + metric],
            color=colors[1],
            linestyle=line_styles[1],
            marker=markers[1],
            label=r"$k=0$, $a_d=16$ hours",
        )
        plt.plot(
            plot_data_3["attack_ratio"],
            plot_data_3["val_" + metric],
            color=colors[2],
            linestyle=line_styles[1],
            marker=markers[2],
            label=r"$k=1$, $a_d=4$ hours",
        )
        plt.plot(
            plot_data_4["attack_ratio"],
            plot_data_4["val_" + metric],
            color=colors[3],
            linestyle=line_styles[1],
            marker=markers[3],
            label=r"$k=1$, $a_d=16$ hours",
        )
        plt.xlabel(r"a_r")
        plt.ylabel(metric_legend)
        plt.legend()
        plt.savefig(output_path + metric + ".png")


def main_generate_all_attack_properties_analysis_data(
    nn_model,
    architecture,
    metadata_metric,
    metric,
    mode,
    train_dataset_path,
    test_dataset_path,
    time_window,
    use_time_hour,
    use_onehot,
    num_labels,
    run_number,
    group_number,
):
    model_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/metadata_"
        + metadata_metric
        + "/"
        + nn_model
        + "/"
        + architecture
        + "/"
        + "run_"
        + str(run_number)
        + "/saved_model/"
    )
    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)

    output_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/metadata_"
        + metadata_metric
        + "/"
        + nn_model
        + "/"
        + architecture
        + "/"
        + "run_"
        + str(run_number)
        + "/report/"
    )

    generate_all_attack_properties_analysis_data(
        nn_model,
        architecture,
        train_dataset,
        test_dataset,
        model_path,
        time_window,
        use_time_hour,
        use_onehot,
        num_labels,
        metric,
        mode,
        output_path,
    )

    report_path_metrics_evaluation = (
        output_path
        + "/attack_property_evaluation/data/"
        + "general_report_all_attack_properties_"
        + metric
        + "_"
        + mode
        + ".csv"
    )
    plot_output_path = output_path + "/attack_property_evaluation/plot/"
    prepare_output_directory(plot_output_path)
    plot_metric_vs_specific_attack_property_selection(
        report_path_metrics_evaluation, plot_output_path
    )


def main_generate_results(group_number):
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    metric = "val_binary_accuracy"
    mode = "max"
    use_time_hour = False
    use_onehot = True
    upsample_enabled = False
    num_labels = 1
    time_window = 10
    num_random_trains = 10
    train_dataset_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/train_data/train_data.csv"
    )
    test_dataset_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/test_data/test_data.csv"
    )
    print(sys.argv)

    if len(sys.argv) > 1:
        nn_model = sys.argv[1]
        architecture = sys.argv[2]
        metadata_metric = sys.argv[3]
        run_number = int(sys.argv[4])
        run_final_report = sys.argv[5] == "True"
        use_time_hour = sys.argv[6] == "True"
        use_onehot = sys.argv[7] == "True"
        upsample_enabled = sys.argv[8] == "True"
        num_labels = int(sys.argv[9])
        time_window = int(sys.argv[10])

        if run_number == -1:
            main_final_general_report(
                nn_model, architecture, metadata_metric, metric, mode, group_number
            )
            main_plot_metric_vs_attack_parameter(
                nn_model, architecture, metadata_metric, metric, mode, -1, group_number
            )
            # print('hello')
        else:
            main_general_report(
                nn_model,
                architecture,
                metadata_metric,
                metric,
                mode,
                train_dataset_path,
                test_dataset_path,
                time_window,
                use_time_hour,
                use_onehot,
                num_labels,
                run_number,
                group_number,
            )
            main_plot_metric_vs_attack_parameter(
                nn_model,
                architecture,
                metadata_metric,
                metric,
                mode,
                run_number,
                group_number,
            )
            main_plot_attack_prediction_vs_time(
                nn_model,
                architecture,
                metadata_metric,
                metric,
                mode,
                run_number,
                group_number,
            )

            if run_final_report:
                main_final_general_report(
                    nn_model, architecture, metadata_metric, metric, mode, group_number
                )
                main_plot_metric_vs_attack_parameter(
                    nn_model,
                    architecture,
                    metadata_metric,
                    metric,
                    mode,
                    -1,
                    group_number,
                )
                # print('hello')
            main_generate_attack_properties_analysis(
                nn_model,
                architecture,
                metadata_metric,
                metric,
                mode,
                train_dataset_path,
                test_dataset_path,
                time_window,
                use_time_hour,
                use_onehot,
                num_labels,
                run_number,
                group_number,
            )
            main_generate_all_attack_properties_analysis_data(
                nn_model,
                architecture,
                metadata_metric,
                metric,
                mode,
                train_dataset_path,
                test_dataset_path,
                time_window,
                use_time_hour,
                use_onehot,
                num_labels,
                run_number,
                group_number,
            )

    else:
        nn_model_list = ["dense", "cnn", "lstm", "aen", "trans"]
        architecture_list = [
            "one_model_with_correlation",
            "multiple_models_with_correlation",
            "multiple_models_without_correlation",
            "one_model_without_correlation",
        ]
        nn_model_list = ["dense"]
        architecture_list = ["one_model_with_correlation"]

        # Run 1
        metadata_metric = "NOT_USED"
        run_number = 0
        for nn_model in nn_model_list:
            for architecture in architecture_list:
                main_general_report(
                    nn_model,
                    architecture,
                    metadata_metric,
                    metric,
                    mode,
                    train_dataset_path,
                    test_dataset_path,
                    time_window,
                    use_time_hour,
                    use_onehot,
                    num_labels,
                    run_number,
                    group_number,
                )
                main_plot_metric_vs_attack_parameter(
                    nn_model,
                    architecture,
                    metadata_metric,
                    metric,
                    mode,
                    run_number,
                    group_number,
                )
                main_plot_attack_prediction_vs_time(
                    nn_model,
                    architecture,
                    metadata_metric,
                    metric,
                    mode,
                    run_number,
                    group_number,
                )
                main_final_general_report(
                    nn_model, architecture, metadata_metric, metric, mode, group_number
                )
                main_plot_metric_vs_attack_parameter(
                    nn_model,
                    architecture,
                    metadata_metric,
                    metric,
                    mode,
                    -1,
                    group_number,
                )
                main_generate_attack_properties_analysis(
                    nn_model,
                    architecture,
                    metadata_metric,
                    metric,
                    mode,
                    train_dataset_path,
                    test_dataset_path,
                    time_window,
                    use_time_hour,
                    use_onehot,
                    num_labels,
                    run_number,
                    group_number,
                )
                main_generate_all_attack_properties_analysis_data(
                    nn_model,
                    architecture,
                    metadata_metric,
                    metric,
                    mode,
                    train_dataset_path,
                    test_dataset_path,
                    time_window,
                    use_time_hour,
                    use_onehot,
                    num_labels,
                    run_number,
                    group_number,
                )

        # Run 2
        nn_model_list = ["lstm"]
        architecture_list = ["multiple_models_with_correlation"]
        metadata_metric_list = ["DISTANCE", "CORRELATION", "RANDOM"]
        metadata_metric_list = ["Random"]
        for metadata_metric in metadata_metric_list:
            for nn_model in nn_model_list:
                for architecture in architecture_list:
                    if metadata_metric != "RANDOM":
                        run_number = 0
                        main_general_report(
                            nn_model,
                            architecture,
                            metadata_metric,
                            metric,
                            mode,
                            train_dataset_path,
                            test_dataset_path,
                            time_window,
                            use_time_hour,
                            use_onehot,
                            num_labels,
                            run_number,
                            group_number,
                        )
                        main_plot_metric_vs_attack_parameter(
                            nn_model,
                            architecture,
                            metadata_metric,
                            metric,
                            mode,
                            run_number,
                            group_number,
                        )
                        main_plot_attack_prediction_vs_time(
                            nn_model,
                            architecture,
                            metadata_metric,
                            metric,
                            mode,
                            run_number,
                            group_number,
                        )
                    else:
                        for run_number in range(num_random_trains):
                            main_general_report(
                                nn_model,
                                architecture,
                                metadata_metric,
                                metric,
                                mode,
                                train_dataset_path,
                                test_dataset_path,
                                time_window,
                                use_time_hour,
                                use_onehot,
                                num_labels,
                                run_number,
                                group_number,
                            )
                            main_plot_metric_vs_attack_parameter(
                                nn_model,
                                architecture,
                                metadata_metric,
                                metric,
                                mode,
                                run_number,
                                group_number,
                            )
                            main_plot_attack_prediction_vs_time(
                                nn_model,
                                architecture,
                                metadata_metric,
                                metric,
                                mode,
                                run_number,
                                group_number,
                            )
                    main_final_general_report(
                        nn_model,
                        architecture,
                        metadata_metric,
                        metric,
                        mode,
                        group_number,
                    )
                    main_plot_metric_vs_attack_parameter(
                        nn_model,
                        architecture,
                        metadata_metric,
                        metric,
                        mode,
                        -1,
                        group_number,
                    )


def main(num_groups):
    for group_number in range(num_groups):
        main_generate_results(group_number)


if __name__ == "__main__":
    main(CONFIG.NUM_GROUPS)
