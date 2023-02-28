import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import sys
import os
from scipy.stats import norm, cauchy
from fitter import Fitter, get_common_distributions, get_distributions
sys.path.append('../')
import project_config as CONFIG
import tensorflow_probability as tfp


def prepare_output_directory(output_path):
    '''Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    '''
    dir_name = str(os.path.dirname(output_path))
    #os.system('rm -rf ' + dir_name)
    os.system('mkdir -p ' + dir_name)


def load_dataset(path):
    '''Load the benign/attacked dataset

    Keyword arguments:
    path -- path to the dataset
    '''
    data = pd.read_csv(path)
    return data


def main_fit(benign_data_path):
    '''Try different distributions to find the best fit for benign/attacked dataset
    '''
    benign_data = load_dataset(benign_data_path)

    attacked_data_path = CONFIG.DATASET_DIRECTORY + 'N_BaIoT/SimpleHome_XCS7_1003_WHT_Security_Camera_mirai_attack_udp_aggregation_Source-IP_time_window_10-sec_stat_Number.csv'
    attacked_data = load_dataset(attacked_data_path)

    f = Fitter(list(benign_data['PACKET'].values))
    f.fit()
    #print(f.summary(Nbest=80))


    g = Fitter(list(attacked_data['PACKET'].values))
    g.fit()
    #print(g.summary(Nbest=80))

    f_df = f.summary(Nbest=100)
    output_path = CONFIG.OUTPUT_DIRECTORY + 'N_BaIoT/Dists/benign_dists.csv'
    prepare_output_directory(output_path)
    f_df.to_csv(output_path)

    g_df = g.summary(Nbest=100)
    output_path = CONFIG.OUTPUT_DIRECTORY + 'N_BaIoT/Dists/attack_dists.csv'
    prepare_output_directory(output_path)
    g_df.to_csv(output_path)

    print('************************* BENIGN ****************************')
    print(f.summary(Nbest=20))
    print('************************* ATTACK ****************************')
    print(g.summary(Nbest=20))


def main_fit_cauchy(benign_data_path):
    '''Fit the Cauchy distribution to the benign/attacked dataset
    '''
    benign_data = load_dataset(benign_data_path)

    attacked_data_path = CONFIG.DATASET_DIRECTORY + 'N_BaIoT/SimpleHome_XCS7_1003_WHT_Security_Camera_mirai_attack_udp_aggregation_Source-IP_time_window_10-sec_stat_Number.csv'
    attacked_data = load_dataset(attacked_data_path)

    f = Fitter(list(benign_data['PACKET'].values), distributions=['cauchy'])
    f.fit()
    print(f.summary())


    g = Fitter(list(attacked_data['PACKET'].values), distributions=['cauchy'])
    g.fit()
    print(g.summary())


    benign_cauchy = cauchy.fit(benign_data['PACKET'])
    print(benign_cauchy)
    attack_cauchy = cauchy.fit(attacked_data['PACKET'])
    print(attack_cauchy)

    a = tfp.substrates.numpy.distributions.TruncatedCauchy(loc=benign_cauchy[0], scale=benign_cauchy[1], low=0, high=benign_data['PACKET'].max())
    print(a)
    print(type(a.sample([10])))


def main_plot_pdf(benign_data_path, output_path):
    '''Plot the fitted truncated Cauchy distribution pdf
    '''
    benign_data = load_dataset(benign_data_path)

    plt.clf()
    num_bins = 50
    count_empirical, bins_count_empirical = np.histogram(benign_data['PACKET'], bins=num_bins)
    pdf_empirical = count_empirical / sum(count_empirical)
    plt.bar(bins_count_empirical[1:], count_empirical/benign_data['PACKET'].shape[0], label='Emprical')
    #plt.plot(bins_count_empirical[1:], pdf_empirical, color='red', label='PDF-Empirical')

    benign_cauchy = cauchy.fit(benign_data['PACKET'])
    max_value = benign_data['PACKET'].max()

    colors = ['red', 'green', 'black']
    for i, k in enumerate([0, 2, 8]):
        packet_dist = tfp.substrates.numpy.distributions.TruncatedCauchy(loc=benign_cauchy[0]*(1+k),
                                                                         scale=benign_cauchy[1]*(1+k),
                                                                         low=0,
                                                                         high=benign_data['PACKET'].max()*(1+k))

        temp = packet_dist.sample([100000000])
        #temp = np.array(cauchy.rvs(loc=benign_cauchy[0]*(1+k), scale=benign_cauchy[1]*(1+k), size=1000000))
        #temp[temp<0] = 0
        #temp[temp>max_value] = max_value*(1+k)
        count, bins_count = np.histogram(temp, bins=num_bins)
        pdf = count / sum(count)
        plt.plot(bins_count[1:], pdf, color=colors[i], label='k = ' + str(k))

    plt.xlabel('Packets Volume')
    plt.ylabel('PDF')
    plt.semilogx()
    #plt.semilogy()
    plt.legend()
    plt.savefig(output_path)
    # plt.show()


def main_plot_ccdf(benign_data_path, output_path):
    '''Plot the fitted truncated Cauchy distribution ccdf
    '''
    benign_data = load_dataset(benign_data_path)

    plt.clf()
    num_bins = 50
    count_empirical, bins_count_empirical = np.histogram(benign_data['PACKET'], bins=num_bins)
    pdf_empirical = count_empirical / sum(count_empirical)
    ccdf_empirical = 1-np.cumsum(pdf_empirical)
    plt.plot(bins_count_empirical[1:], ccdf_empirical, color='blue', label='Empirical')

    benign_cauchy = cauchy.fit(benign_data['PACKET'])
    max_value = benign_data['PACKET'].max()

    colors = ['red', 'green', 'black']
    for i, k in enumerate([0, 2, 8]):
        packet_dist = tfp.substrates.numpy.distributions.TruncatedCauchy(loc=benign_cauchy[0]*(1+k),
                                                                         scale=benign_cauchy[1]*(1+k),
                                                                         low=0,
                                                                         high=benign_data['PACKET'].max()*(1+k))

        temp = packet_dist.sample([100000000])
        #temp = np.array(cauchy.rvs(loc=benign_cauchy[0]*(1+k), scale=benign_cauchy[1]*(1+k), size=1000000))
        #temp[temp<0] = 0
        #temp[temp>max_value] = max_value*(1+k)
        count, bins_count = np.histogram(temp, bins=num_bins)
        pdf = count / sum(count)
        ccdf = 1 - np.cumsum(pdf)
        plt.plot(bins_count[1:], ccdf, color=colors[i], label='k = ' + str(k))

    plt.xlabel('Packets Volume')
    plt.ylabel('CCDF')
    plt.semilogx()
    #plt.semilogy()
    plt.legend()
    plt.savefig(output_path)
    # plt.show()


def main():
    benign_data_path = CONFIG.DATASET_DIRECTORY + 'N_BaIoT/SimpleHome_XCS7_1003_WHT_Security_Camera_benign_aggregation_Source-IP_time_window_10-sec_stat_Number.csv'
    # main_fit(benign_data_path)
    main_fit_cauchy(benign_data_path)

    output_path = CONFIG.OUTPUT_DIRECTORY + 'N_BaIoT/Plot/pdf.png'
    prepare_output_directory(output_path)
    main_plot_pdf(benign_data_path, output_path)

    output_path = CONFIG.OUTPUT_DIRECTORY + 'N_BaIoT/Plot/ccdf.png'
    prepare_output_directory(output_path)
    main_plot_ccdf(benign_data_path, output_path)


if __name__ == '__main__':
    main()
