import os
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('./source_code/')

import project_config as CONFIG


def main():
    clean_dataset_file_path = CONFIG.SOURCE_CODE_DIRECTORY + 'pre_process/clean_dataset.py'
    generate_nodes_distance_file_path = CONFIG.SOURCE_CODE_DIRECTORY + 'pre_process/generate_nodes_distance.py'
    generate_nodes_pearson_correlation_file_path = CONFIG.SOURCE_CODE_DIRECTORY + 'pre_process/generate_nodes_pearson_correlation.py'
    generate_attack_file_path = CONFIG.SOURCE_CODE_DIRECTORY + 'pre_process/generate_attack.py'
    generate_training_data_file_path = CONFIG.SOURCE_CODE_DIRECTORY + 'pre_process/generate_training_data.py'

    time_step = 60 * 10
    num_train_days = 4
    num_validation_days = 1
    num_test_days = 3
    k_list = '[0,0.1,0.3,0.5,0.7,1]'  # No space shall be included in this string
    attacked_ratio_nodes_list = '[0.5,1]'  # percentage - No space shall be included in this string
    attack_duration_list = '[4,8,16]'  # hours - No space shall be included in this string
    attack_start_times_list = '[2,6,12]'  # 24-hour format - No space shall be included in this string

    clean_dataset_command = 'python3 ' + clean_dataset_file_path + ' ' + str(time_step)
    generate_nodes_distance_command = 'python3 ' + generate_nodes_distance_file_path + ' ' + str(time_step)
    generate_nodes_pearson_correlation_command = 'python3 ' + generate_nodes_pearson_correlation_file_path + ' ' + str(time_step)

    generate_attack_command = 'python3 ' + generate_attack_file_path + ' ' + str(time_step) + ' ' + \
                    str(num_train_days) + ' ' + str(num_validation_days) + ' ' + str(num_test_days) + ' ' \
                    + str(k_list) + ' ' + str(attacked_ratio_nodes_list) + ' ' \
                    + str(attack_duration_list) + ' ' + str(attack_start_times_list)

    generate_training_data_command = 'python3 ' + generate_training_data_file_path + ' ' + str(time_step) + \
                    str(num_train_days) + ' ' + str(num_validation_days) + ' ' + str(num_test_days)

    os.system(clean_dataset_command)
    os.system(generate_nodes_distance_command)
    os.system(generate_nodes_pearson_correlation_command)
    os.system(generate_attack_command)
    os.system(generate_training_data_command)

    
if __name__ == '__main__':
    main()