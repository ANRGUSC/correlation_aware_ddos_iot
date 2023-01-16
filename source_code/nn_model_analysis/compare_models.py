import sys
import pandas as pd
import os
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../../')
import project_config as CONFIG


def prepare_output_directory(output_path):
    '''Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    '''
    dir_name = str(os.path.dirname(output_path))
    #os.system('rm -rf ' + dir_name)
    os.system('mkdir -p ' + dir_name)


def load_dataset(path):
    '''Load the dataset

    Keyword arguments:
    path -- path to the dataset csv file
    '''
    data = pd.read_csv(path)
    data['TIME'] = pd.to_datetime(data['TIME'])
    return data


def plot_metric_vs_attack_parameter(all_mean_report_path, mean_report_legends, colors, markers, line_styles, output_path):
    '''Plot the metrics like binary accuracy and recall for different k values

    Keyword arguments:
    all_mean_report_path -- a list of paths to the all the mean metric values for different NN models
    mean_report_legends -- a dictionary showing the mean_report_path
    output_path -- path to the output directory
    '''

    mean_report = pd.read_csv(all_mean_report_path[0])
    metrics = mean_report.columns.values
    metrics=['binary_accuracy', 'recall', 'specificity', 'precision', 'auc', 'f1score']

    for metric in metrics:
        plt.clf()
        if metric == 'k' or metric == 'node' or 'val' in metric:
            continue
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            plt.plot(mean_report['k'], mean_report[metric], color= colors[i], linestyle=line_styles[0],
                     marker=markers[i], label=mean_report_legends[i])
            #plt.plot(mean_report['k'], mean_report['val_' + metric], color=colors[i], linestyle=line_styles[1],
            #         marker=markers[i], label='Test_' + mean_report_legends[i])
        plt.xlabel('Attack Packet Volume Distribution Parameter (k)')
        plt.ylabel(metric)
        if metric == 'binary_accuracy':
            plt.ylabel('Binary Accuracy')
        elif metric == 'recall':
            plt.ylabel('Recall')
        elif metric == 'specificity':
            plt.ylabel('Specificity')
        elif metric == 'precision':
            plt.ylabel('Precision')
        elif metric == 'auc':
            plt.ylabel('AUC')
        elif metric == 'f1score':
            plt.ylabel('F1 Score')
        #plt.title(metric + ' vs attack parameter')
        plt.legend()
        plt.savefig(output_path + metric + '_train.png')


    for metric in metrics:
        plt.clf()
        if metric == 'k' or metric == 'node' or 'val' in metric:
            continue
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            plt.plot(mean_report['k'], mean_report['val_'+metric], color= colors[i], linestyle=line_styles[1],
                     marker=markers[i], label=mean_report_legends[i])
        plt.xlabel('Attack Packet Volume Distribution Parameter (k)')
        plt.ylabel(metric)
        if metric == 'binary_accuracy':
            plt.ylabel('Binary Accuracy')
        elif metric == 'recall':
            plt.ylabel('Recall')
        elif metric == 'specificity':
            plt.ylabel('Specificity')
        elif metric == 'precision':
            plt.ylabel('Precision')
        elif metric == 'auc':
            plt.ylabel('AUC')
        elif metric == 'f1score':
            plt.ylabel('F1 Score')
        #plt.title(metric + ' vs attack parameter')
        plt.legend()
        plt.savefig(output_path + metric + '_test.png')


def plot_roc(all_mean_report_path, mean_report_legends, colors, markers, line_styles, output_path):
    """Plot the metrics like binary accuracy and recall for different k values

    Keyword arguments:
    all_mean_report_path -- a list of paths to the all the mean metric values for different NN models
    mean_report_legends -- a dictionary showing the mean_report_path
    output_path -- path to the output directory
    """

    mean_report = pd.read_csv(all_mean_report_path[0])
    k_list = mean_report['k'].unique()

    for k in k_list:
        plt.clf()
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            tmp_mean_report = mean_report.loc[mean_report['k'] == k]
            plt.plot(tmp_mean_report['fpr'], tmp_mean_report['tpr'], color= colors[i], linestyle=line_styles[0],
                     label=mean_report_legends[i])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(output_path + "roc_curve_k_" + str(k) + "_train.png")

    for k in k_list:
        plt.clf()
        for i, mean_report_path in enumerate(all_mean_report_path):
            mean_report = pd.read_csv(mean_report_path)
            tmp_mean_report = mean_report.loc[mean_report['k'] == k]
            plt.plot(tmp_mean_report['val_fpr'], tmp_mean_report['val_tpr'], color= colors[i], linestyle=line_styles[1],
                     label=mean_report_legends[i])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(output_path + "roc_curve_k_" + str(k) + "_test.png")


def main_plot_compare_current(metric, mode):
    '''Main function for plotting the metrics like binary accuracy and recall for different k values

    Keyword arguments:
    metric -- the metric for comparing the NN models like binary accuracy, recall
    mode -- the mode for the selected metric
    '''
    colors = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'magenta', 5: 'yellow'}
    markers = {0: '.', 1: 'v', 2: '+', 3: 'x', 4: 's', 5: '^'}
    line_styles = {0: 'solid', 1: 'dashed'}

    nn_model_list = ['dense', 'cnn', 'lstm', 'trans', 'aen']
    architecture_list = ['multiple_models_with_correlation',
                         'multiple_models_without_correlation',
                         'one_model_with_correlation',
                         'one_model_without_correlation']
    metadata_metric_list = ['NOT_USED', 'DISTANCE', 'CORRELATION', 'RANDOM']
    # nn_model_list = ['dense', 'cnn', 'lstm', 'trans']
    # metadata_metric_list = ['NOT_USED']
    # architecture_list = ['multiple_models_with_correlation']

    metadata_metric = 'NOT_USED'
    mean_report_legends = {0: 'MM-WC', 1: 'MM-NC', 2: 'OM-WC', 3: 'OM-NC'}
    for nn_model in nn_model_list:
        metric_mean_report_path_list = []
        roc_mean_report_path_list = []
        for architecture in architecture_list:
            input_path = CONFIG.OUTPUT_DIRECTORY + 'nn_training/' + 'metadata_' + metadata_metric + '/' + \
                         nn_model + '/' + architecture + '/final_report/'

            metric_mean_report_path = input_path + 'metrics_evaluation/data/mean_report_' + metric + '_' + mode + '.csv'
            metric_mean_report_path_list.append(metric_mean_report_path)

            roc_mean_report_path = input_path + 'roc/data/roc_mean_report_' + metric + '_' + mode + '.csv'
            roc_mean_report_path_list.append(roc_mean_report_path)

        output_path = CONFIG.OUTPUT_DIRECTORY + 'nn_model_analysis/metadata_' + metadata_metric + '/compare_nn_models/' + nn_model + '/'
        metric_output_path = output_path + 'compare_' + nn_model + '_nn_model_'
        prepare_output_directory(metric_output_path)
        plot_metric_vs_attack_parameter(metric_mean_report_path_list, mean_report_legends, colors, markers, line_styles, metric_output_path)

        roc_output_path = output_path + 'roc/compare_roc_' + nn_model + '_nn_model_'
        prepare_output_directory(roc_output_path)
        plot_roc(roc_mean_report_path_list, mean_report_legends, colors, markers, line_styles, roc_output_path)

    mean_report_legends = {0: 'MLP', 1: 'CNN', 2: 'LSTM', 3: 'TRF', 4: 'AEN'}
    for metadata_metric in metadata_metric_list:
        if metadata_metric == 'NOT_USED':
            nn_model_list = ['dense', 'cnn', 'lstm', 'trans', 'aen']
            # nn_model_list = ['dense', 'cnn', 'lstm', 'trans']
            architecture_list = ['multiple_models_with_correlation',
                                 'multiple_models_without_correlation',
                                 'one_model_with_correlation',
                                 'one_model_without_correlation']
            mean_report_legends = {0: 'MLP', 1: 'CNN', 2: 'LSTM', 3: 'TRF', 4: 'AEN'}
        else:
            architecture_list = ['multiple_models_with_correlation']
            nn_model_list = ['lstm', 'trans']
            mean_report_legends = {0: 'LSTM', 1: 'TRF'}
        for architecture in architecture_list:
            metric_mean_report_path_list = []
            roc_mean_report_path_list = []
            for nn_model in nn_model_list:
                input_path = CONFIG.OUTPUT_DIRECTORY + 'nn_training/' + 'metadata_' + metadata_metric + '/' + \
                             nn_model + '/' + architecture + '/final_report/'

                metric_mean_report_path = input_path + 'metrics_evaluation/data/mean_report_' + metric + '_' + mode + '.csv'
                metric_mean_report_path_list.append(metric_mean_report_path)

                roc_mean_report_path = input_path + 'roc/data/roc_mean_report_' + metric + '_' + mode + '.csv'
                roc_mean_report_path_list.append(roc_mean_report_path)

            output_path = CONFIG.OUTPUT_DIRECTORY + 'nn_model_analysis/metadata_' + metadata_metric + '/compare_nn_architecture/' + architecture + '/'
            metric_output_path = output_path + 'compare_' + architecture + '_architecture_'
            prepare_output_directory(metric_output_path)
            plot_metric_vs_attack_parameter(metric_mean_report_path_list, mean_report_legends, colors, markers, line_styles, metric_output_path)

            roc_output_path = output_path + 'roc/compare_roc_' + architecture + '_architecture_'
            prepare_output_directory(roc_output_path)
            plot_roc(roc_mean_report_path_list, mean_report_legends, colors, markers, line_styles, roc_output_path)


def main_plot_correlation_metrics(metric, mode):
    '''Main function for plotting the metrics like binary accuracy and recall for different k values in the case of
    using different methods for actively selecting nodes for the correlation-aware architectures

    Keyword arguments:
    metric -- the metric for comparing the NN models like binary accuracy, recall
    mode -- the mode for the selected metric
    '''
    colors = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'magenta', 5: 'yellow'}
    markers = {0: '.', 1: 'v', 2: '+', 3: 'x', 4: 's', 5: '^'}
    line_styles = {0: 'solid', 1: 'dashed'}

    nn_model_list = ['lstm', 'trans']
    architecture_list = ['multiple_models_with_correlation']
    metadata_metric_list = ['NOT_USED', 'DISTANCE', 'CORRELATION', 'RANDOM']
    mean_report_legends = {0: 'All Nodes', 1: 'Nearest Neighbor', 2: 'Pearson', 3: 'Random'}
    # metadata_metric_list = ['NOT_USED', 'DISTANCE', 'CORRELATION', 'SHAP', 'RANDOM']
    # mean_report_legends = {0: 'All Nodes', 1: 'Nearest Neighbor', 2: 'Pearson', 3: 'SHAP', 4: 'Random'}

    for nn_model in nn_model_list:
        for architecture in architecture_list:
            metric_mean_report_path_list = []
            roc_mean_report_path_list = []
            for metadata_metric in metadata_metric_list:
                input_path = CONFIG.OUTPUT_DIRECTORY + 'nn_training/' + 'metadata_' + metadata_metric + '/' + \
                             nn_model + '/' + architecture + '/final_report/'

                metric_mean_report_path = input_path + 'metrics_evaluation/data/mean_report_' + metric + '_' + mode + '.csv'
                metric_mean_report_path_list.append(metric_mean_report_path)

                roc_mean_report_path = input_path + 'roc/data/roc_mean_report_' + metric + '_' + mode + '.csv'
                roc_mean_report_path_list.append(roc_mean_report_path)

            output_path = CONFIG.OUTPUT_DIRECTORY + 'nn_model_analysis/metadata_comparison/compare_nn_architecture/' + architecture + '/' + nn_model + '/'
            metric_output_path = output_path + 'compare_' + architecture + '_architecture_' + nn_model + '_nn_model_'
            prepare_output_directory(metric_output_path)
            plot_metric_vs_attack_parameter(metric_mean_report_path_list, mean_report_legends, colors, markers,
                                            line_styles, metric_output_path)

            roc_output_path = output_path + 'roc/compare_roc_' + architecture + '_architecture_'
            prepare_output_directory(roc_output_path)
            plot_roc(roc_mean_report_path_list, mean_report_legends, colors, markers, line_styles, roc_output_path)


def main():

    metric = 'binary_accuracy'
    mode = 'max'

    main_plot_compare_current(metric, mode)
    main_plot_correlation_metrics(metric, mode)


if __name__ == '__main__':
    main()