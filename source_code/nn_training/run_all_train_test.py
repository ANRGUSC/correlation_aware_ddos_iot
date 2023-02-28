import os
import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('./source_code/')

import project_config as CONFIG


def main():
    use_time_hour = False
    use_onehot = True
    upsample_enabled = False
    epochs = 3
    batch_size = 32
    num_labels = 1
    time_window = 10

    train_file_path = CONFIG.SOURCE_CODE_DIRECTORY + 'nn_training/train_nn.py'
    results_file_path = CONFIG.SOURCE_CODE_DIRECTORY + 'nn_training/generate_results.py'
    model_analysis_file_path = CONFIG.SOURCE_CODE_DIRECTORY + 'nn_model_analysis/compare_models.py'

    nn_model_list = ['dense', 'cnn', 'lstm', 'trans', 'aen']
    architecture_list = [
        'one_model_with_correlation',
        'multiple_models_with_correlation',
        'one_model_without_correlation',
        'multiple_models_without_correlation'
    ]
    # nn_model_list = ['lstm']
    # architecture_list = ['multiple_models_with_correlation']

    # Run 1
    use_metadata = False
    metadata_path = CONFIG.OUTPUT_DIRECTORY
    metadata_metric = 'NOT_USED'
    num_selected_nodes_for_correlation = 0
    run_number = 0
    run_final_report = True
    for nn_model in nn_model_list:
        for architecture in architecture_list:
            train_command = 'python3 ' + train_file_path + ' ' + nn_model + ' ' + architecture + ' ' + \
                            str(use_metadata) + ' ' + metadata_path + ' ' + metadata_metric + ' ' \
                            + str(num_selected_nodes_for_correlation) + ' ' + str(run_number) + ' ' \
                            + str(use_time_hour) + ' ' + str(use_onehot) + ' ' + str(upsample_enabled) + ' ' \
                            + str(epochs) + ' ' + str(batch_size) + ' ' + str(num_labels) + ' ' \
                            + str(time_window)
            os.system(train_command)
            results_command = 'python3 ' + results_file_path + ' ' + nn_model + ' ' + architecture + ' ' + \
                              metadata_metric + ' ' + str(run_number) + ' ' + str(run_final_report) + \
                              ' ' + str(use_time_hour) + ' ' + str(use_onehot) + ' ' + str(upsample_enabled) + \
                              ' ' + str(num_labels) + ' ' + str(time_window)
            os.system(results_command)
            # os.system('python3 ' + model_analysis_file_path)
    # Run model analysis
    # os.system('python3 ' + model_analysis_file_path)
    # sys.exit()

    # Run 2
    architecture_list = ['multiple_models_with_correlation']
    nn_model_list = ['lstm']
    use_metadata = True
    metadata_path_list = [CONFIG.OUTPUT_DIRECTORY + 'pre_process/Output/metadata/distances.csv',
                          CONFIG.OUTPUT_DIRECTORY + 'pre_process/Output/metadata/correlations.csv',
                          CONFIG.OUTPUT_DIRECTORY,
                          CONFIG.OUTPUT_DIRECTORY]
    metadata_metric_list = ['DISTANCE', 'CORRELATION', 'RANDOM', 'SHAP']
    num_random_trains = 10
    num_selected_nodes_for_correlation = 5
    for metadata_index in range(len(metadata_metric_list)):
        metadata_path = metadata_path_list[metadata_index]
        metadata_metric = metadata_metric_list[metadata_index]
        for nn_model in nn_model_list:
            for architecture in architecture_list:
                if metadata_metric != 'RANDOM':
                    run_number = 0
                    run_final_report = True
                    train_command = 'python3 ' + train_file_path + ' ' + nn_model + ' ' + architecture + ' ' + \
                                    str(use_metadata) + ' ' + metadata_path + ' ' + metadata_metric + ' ' \
                                    + str(num_selected_nodes_for_correlation) + ' ' + str(run_number) + ' ' \
                                    + str(use_time_hour) + ' ' + str(use_onehot) + ' ' + str(upsample_enabled) + ' ' \
                                    + str(epochs) + ' ' + str(batch_size) + ' ' + str(num_labels) + ' ' \
                                    + str(time_window)
                    os.system(train_command)
                    results_command = 'python3 ' + results_file_path + ' ' + nn_model + ' ' + architecture + ' ' + \
                                      metadata_metric + ' ' + str(run_number) + ' ' + str(run_final_report) + \
                                      ' ' + str(use_time_hour) + ' ' + str(use_onehot) + ' ' + str(upsample_enabled) + \
                                      ' ' + str(num_labels) + ' ' + str(time_window)
                    os.system(results_command)
                else:
                    run_final_report = False
                    for run_number in range(num_random_trains):
                        train_command = 'python3 ' + train_file_path + ' ' + nn_model + ' ' + architecture + ' ' + \
                                        str(use_metadata) + ' ' + metadata_path + ' ' + metadata_metric + ' ' \
                                        + str(num_selected_nodes_for_correlation) + ' ' + str(run_number) + ' ' \
                                        + str(use_time_hour) + ' ' + str(use_onehot) + ' ' + str(upsample_enabled) + ' ' \
                                        + str(epochs) + ' ' + str(batch_size) + ' ' + str(num_labels) + ' ' \
                                        + str(time_window)
                        os.system(train_command)
                        results_command = 'python3 ' + results_file_path + ' ' + nn_model + ' ' + architecture + ' ' + \
                                          metadata_metric + ' ' + str(run_number) + ' ' + str(run_final_report) + \
                                          ' ' + str(use_time_hour) + ' ' + str(use_onehot) + ' ' + \
                                          str(upsample_enabled) + ' ' + str(num_labels) + ' ' + str(time_window)
                        os.system(results_command)
                    run_final_report = True
                    run_number = -1
                    results_command = 'python3 ' + results_file_path + ' ' + nn_model + ' ' + architecture + ' ' + \
                                      metadata_metric + ' ' + str(run_number) + ' ' + str(run_final_report) + \
                                      ' ' + str(use_time_hour) + ' ' + str(use_onehot) + ' ' + \
                                      str(upsample_enabled) + ' ' + str(num_labels) + ' ' + str(time_window)
                    os.system(results_command)

    # Run model analysis
    os.system('python3 ' + model_analysis_file_path)


if __name__ == '__main__':
    main()