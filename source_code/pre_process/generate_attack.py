import math
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from multiprocessing import Pool
from itertools import product
from scipy.stats import norm, cauchy
import tensorflow_probability as tfp

sys.path.append('../')
import project_config as CONFIG


def prepare_output_directory(output_path):
    '''Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    '''
    dir_name = str(os.path.dirname(output_path))
    os.system('rm -rf ' + dir_name)
    os.system('mkdir -p ' + dir_name)


def load_dataset(path):
    '''Load the dataset and change the type of the 'TIME' column to datetime.

    Keyword arguments:
    path -- path to the dataset
    '''
    data = pd.read_csv(path)
    data['TIME'] = pd.to_datetime(data['TIME'])
    return data


def generate_attack(benign_data, attack_begin_date, attack_end_date, attacked_ratio_nodes, attack_duration,
                          attack_start_time, attack_packet_cauchy, attack_parameter, num_days, output_path, data_type):
    '''Create attack in the benign dataset for the given features based on the data type.

    Keyword arguments:
    benign_data -- benign dataset to be used for attacking
    attack_begin_date -- the begin date of the attack
    attack_end_date -- the end date of the attack
    attacked_ratio_nodes -- the ratio of the nodes in the benign dataset to be attacked.
    attack_duration -- the duration of the attack
    attack_start_times -- the start times of the attacks withing the attack_begin_date and attack_end_date
    output_path -- the output path for storing the attacked dataset
    attack_packet_cauchy -- truncated cauchy distribution parameters for generating attack volume
    attack_parameter -- the k value for generating attack packet volume
    data_type -- could be 'train' or 'test'. For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    '''

    #if data_type == 'train':
    #    random.seed()
    #else:
    #    random.seed()
    random.seed()
    data_selected_nodes = benign_data.loc[(benign_data['TIME'] >= attack_begin_date) &
                                          (benign_data['TIME'] < attack_end_date)]
    nodes = list(data_selected_nodes['NODE'].unique())
    num_attacked_nodes = math.ceil(len(nodes)*attacked_ratio_nodes)

    # # ignore the attack start times which are included in the attack interval of the previous attack
    # i = 0
    # while i < len(attack_start_times)-1:
    #     if attack_start_times[i+1] <= attack_start_times[i] + attack_duration:
    #         attack_start_times.pop(i+1)
    #         continue
    #     i += 1

    attacked_nodes = list(random.sample(nodes, k=num_attacked_nodes))
    for attack_day in range(num_days):
        attack_start_time = attack_start_time + timedelta(days=attack_day)
        attack_finish_time = attack_start_time + attack_duration

        select_rows = ((benign_data['NODE'].isin(attacked_nodes)) &
                        (benign_data['TIME'] >= attack_start_time) &
                        (benign_data['TIME'] < attack_finish_time))

        benign_data.loc[select_rows, 'ACTIVE'] = 1
        benign_data.loc[select_rows, 'ATTACKED'] = 1
        packet_dist = tfp.substrates.numpy.distributions.TruncatedCauchy(loc=attack_packet_cauchy[0],
                                                                         scale=attack_packet_cauchy[1],
                                                                         low=0,
                                                                         high=attack_packet_cauchy[2])
        packet = packet_dist.sample([benign_data.loc[select_rows].shape[0]])
        packet = np.ceil(packet)
        benign_data.loc[select_rows, 'PACKET'] = packet

    benign_data['BEGIN_DATE'] = attack_begin_date
    benign_data['END_DATE'] = attack_end_date
    benign_data['NUM_NODES'] = len(nodes)
    benign_data['ATTACK_RATIO'] = attacked_ratio_nodes
    benign_data['ATTACK_START_TIME'] = attack_start_time
    benign_data['ATTACK_DURATION'] = attack_duration
    benign_data['ATTACK_PARAMETER'] = attack_parameter

    benign_data = benign_data[['BEGIN_DATE', 'END_DATE', 'NUM_NODES', 'ATTACK_RATIO', 'ATTACK_START_TIME',
                               'ATTACK_DURATION', 'ATTACK_PARAMETER', 'NODE', 'LAT', 'LNG', 'TIME', 'TIME_HOUR',
                               'ACTIVE', 'PACKET', 'ATTACKED']]

    output_path +=  'attacked_data_' + data_type + '_' + str(attack_begin_date) + '_' + str(attack_end_date) +\
                    '_ratio_' + str(attacked_ratio_nodes) + '_start_time_' + str(attack_start_time) +\
                    '_duration_' + str(attack_duration) +\
                    '_k_' + str(attack_parameter) + '.csv'
    benign_data.to_csv(output_path, index=False)


def main_generate_attack(benign_dataset_path, data_type, num_train_days, num_validation_days, num_test_days,
                         k_list, time_step):
    '''The main function to be used for calling generate_attack function

    Keyword arguments:
    benign_dataset_path -- the path to the benign dataset to be used for attacking
    data_type -- could be 'train' or 'test'. For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    num_train_days -- the number of days to be used for creating the attacked dataset for training purposes
    num_test_days -- the number of days to be used for creating the attacked dataset for testing purposes
    k_list -- different k values for generating training/testing dataset
    time_step -- the time step used for generating benign dataset
    '''

    benign_data = load_dataset(benign_dataset_path)

    # set the begin and end date of the dataset to be attacked
    if data_type == 'train':
        attack_begin_date = benign_data.loc[0, 'TIME'] + timedelta(days=2)
        attack_end_date = benign_data.loc[0, 'TIME'] + timedelta(days=2+num_train_days)
        num_days = num_train_days
    elif data_type == 'validation':
        attack_begin_date = benign_data.loc[0, 'TIME'] + timedelta(days=2+num_train_days+2)
        attack_end_date = benign_data.loc[0, 'TIME'] + timedelta(days=2+num_train_days+2+num_validation_days)
        num_days = num_validation_days
    elif data_type == 'test':
        attack_begin_date = benign_data.loc[0, 'TIME'] + timedelta(days=2+num_train_days+2+num_validation_days+2)
        attack_end_date = benign_data.loc[0, 'TIME'] + timedelta(days=2+num_train_days+2+num_validation_days+2+num_test_days)
        num_days = num_test_days

    output_path = CONFIG.OUTPUT_DIRECTORY + 'pre_process/Output/attacked_data/' + data_type + '/'
    prepare_output_directory(output_path)

    # set the begin and end date of the dataset to be stored for generating features in generate_training_data.py
    # here we choose begin_date - 2days because in the features we have an occupancy average of 48 hours
    slice_benign_data_start = attack_begin_date-timedelta(days=2)
    slice_benign_data_end = attack_end_date

    benign_data = benign_data.loc[(benign_data['TIME'] >= slice_benign_data_start) &
                                  (benign_data['TIME'] < slice_benign_data_end)]
    benign_data_save = benign_data.copy()
    nodes = list(benign_data_save['NODE'].unique())
    benign_data_save['BEGIN_DATE'] = attack_begin_date
    benign_data_save['END_DATE'] = attack_end_date
    benign_data_save['NUM_NODES'] = len(nodes)
    benign_data_save['ATTACK_RATIO'] = 0.0
    benign_data_save['ATTACK_START_TIME'] = attack_begin_date
    benign_data_save['ATTACK_DURATION'] = timedelta(hours=0)
    benign_data_save['ATTACK_PARAMETER'] = 0.0
    output_path_benign = output_path + 'attacked_data_' + data_type + '_' + str(attack_begin_date) + '_' + \
                         str(attack_end_date) + '_ratio_0_start_time_' + str(attack_begin_date) + 'duration_0_k_0' + '.csv'
    benign_data_save.to_csv(output_path_benign, index=False)

    # set the ratio, duration, and start time of the attack
    # in the case that start time of an attack be included in the interval of the previous attack, it will be ignored
    # note that the case of attack_ratio = 0 is already saved above. The name of the variable is 'benign_data_save'.
    attacked_ratio_nodes = [0.5, 1]
    # attacked_ratio_nodes = [0.1, 0.5, 1]
    attack_duration = [timedelta(hours=4), timedelta(hours=8), timedelta(hours=16)]
    # attack_duration = [timedelta(hours=2), timedelta(hours=4), timedelta(hours=8), timedelta(hours=16), timedelta(hours=24)]
    attack_start_times = []
    # attack_start_times.append(attack_begin_date + timedelta(hours=0))
    attack_start_times.append(attack_begin_date + timedelta(hours=2))
    attack_start_times.append(attack_begin_date + timedelta(hours=6))
    attack_start_times.append(attack_begin_date + timedelta(hours=12))
    # attack_start_times.append(attack_begin_date + timedelta(hours=18))


    benign_packet_path = CONFIG.OUTPUT_DIRECTORY + 'N_BaIoT/Output/Data/SimpleHome_XCS7_1003_WHT_Security_Camera_benign_aggregation_Source-IP_time_window_10-sec_stat_Number.csv'
    benign_packet = pd.read_csv(benign_packet_path)
    benign_packet_cauchy = list(cauchy.fit(benign_packet['PACKET']))
    benign_packet_cauchy.append(math.ceil(max(benign_packet['PACKET'])))

    attack_packet_path = CONFIG.OUTPUT_DIRECTORY + 'N_BaIoT/Output/Data/SimpleHome_XCS7_1003_WHT_Security_Camera_mirai_attack_udp_aggregation_Source-IP_time_window_10-sec_stat_Number.csv'
    attack_packet = pd.read_csv(attack_packet_path)
    attack_packet_cauchy = list(cauchy.fit(attack_packet['PACKET']))
    attack_packet_cauchy.append(math.ceil(max(attack_packet['PACKET'])))

    for k in k_list:
        k = round(k, 2)
        print('k : ', k)
        #packet_loc = (1-k) * benign_packet_cauchy[0] + k * attack_packet_cauchy[0]
        #packet_scale = (1-k) * benign_packet_cauchy[1] + k * attack_packet_cauchy[1]

        scale_factor = time_step / 10
        packet_loc = (1+k) * benign_packet_cauchy[0] * scale_factor
        packet_scale = (1+k) * benign_packet_cauchy[1] * scale_factor
        packet_max = (1+k) * benign_packet_cauchy[2] * scale_factor
        packet_cauchy = [packet_loc, packet_scale, packet_max]

        p = Pool()
        p.starmap(generate_attack, product([benign_data], [attack_begin_date], [attack_end_date],
                                                 attacked_ratio_nodes, attack_duration, attack_start_times,
                                                 [packet_cauchy], [k], [num_days], [output_path], [data_type]))
        p.close()
        p.join()


if __name__ == '__main__':
    num_train_days = 4
    num_validation_days = 1
    num_test_days = 3
    time_step = 60 * 10
    num_nodes = 50
    # k_list = np.array([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1])
    k_list = np.array([0, 0.1, 0.3, 0.5, 0.7, 1])
    # k_list = np.array([0, 1])
    benign_dataset_path = CONFIG.OUTPUT_DIRECTORY + \
    'pre_process/Output/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_' + \
    str(time_step) + '_num_ids_' + str(num_nodes) + '.csv'

    main_generate_attack(benign_dataset_path, 'train', num_train_days, num_validation_days, num_test_days, k_list, time_step)
    print('generate train data done.')
    main_generate_attack(benign_dataset_path, 'validation', num_train_days, num_validation_days, num_test_days, k_list, time_step)
    print('generate validation data done.')
    main_generate_attack(benign_dataset_path, 'test', num_train_days, num_validation_days, num_test_days, k_list, time_step)
    print('generate test data done.')