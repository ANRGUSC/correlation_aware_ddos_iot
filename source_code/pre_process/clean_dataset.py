import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool, Manager
import sys
import random
import os
from geopandas import GeoSeries
from shapely.geometry import Point
from haversine import haversine
from scipy.stats import norm, cauchy
import tensorflow_probability as tfp

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
    """Load the dataset and change the type of the "TIME" column to datetime.

    Keyword arguments:
    path -- path to the dataset
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def load_dataset_info(input_path, output_path):
    """Load the dataset information which has the location information

    Keyword arguments:
    input_path -- path to the input file
    output_path -- path to the output file for storing the dataset information
    """
    dataset_info = pd.read_csv(input_path)
    dataset_info = dataset_info[["SpaceID", "LatLng"]]
    dataset_info = dataset_info.rename(columns={"SpaceID": "NODE", "LatLng": "LOCATION"})
    dataset_info[["LAT", "LNG"]] = pd.DataFrame(dataset_info["LOCATION"].str.split(',').tolist())
    dataset_info["LAT"] = dataset_info["LAT"].str[1:]
    dataset_info["LNG"] = dataset_info["LNG"].str[0:-1]
    dataset_info["LAT"] = dataset_info["LAT"].astype(float)
    dataset_info["LNG"] = dataset_info["LNG"].astype(float)
    dataset_info = dataset_info.drop(["LOCATION"], axis=1)
    dataset_info.to_csv(output_path, index=False)
    #print(dataset_info)
    return dataset_info


def load_main_dataset(input_path, output_path):
    """Load the main dataset which has the occupancy record of the parking spaces

    Keyword arguments:
    input_path -- path to the input file
    output_path -- path to the output file for storing the dataset
    """

    init_data = pd.read_csv(input_path)
    init_data = init_data.drop(columns=[" EventTime_UTC"])
    init_data = init_data.rename(columns={"SpaceID": "NODE", " EventTime_Local": "TIME", "OccupancyState": "ACTIVE"})
    init_data["TIME"] = init_data["TIME"].astype(str)
    init_data["TIME"] = init_data["TIME"].str[:-4]
    init_data["TIME"] = pd.to_datetime(init_data["TIME"])
    init_data.loc[init_data["ACTIVE"] != "ACTIVE", "ACTIVE"] = 0
    init_data.loc[init_data["ACTIVE"] == "ACTIVE", "ACTIVE"] = 1
    init_data.to_csv(output_path, index=False)
    return init_data


def create_anonymization_map(init_data, dataset_info, output_path):
    """Create the dataset which maps the real information to the anonymized ones

    Keyword arguments:
    init_data -- the initial data for making it anonymize
    dataset_info -- the dataset which contains the location information
    output_path -- path to the output file for storing the anonymization map
    """

    # Anonymize the node name
    dataset_info["NODE_2"] = list(range(dataset_info.shape[0]))
    dataset_info["NODE_2"] = dataset_info["NODE_2"].astype(str)

    # Anonymize the node location
    dataset_info["LAT_2"] = None
    dataset_info["LNG_2"] = None

    WGS84 = {'init': 'epsg:4326'}
    MERC = {'init': 'epsg:3857'}

    center_lat = dataset_info.loc[0, "LAT"]
    center_lng = dataset_info.loc[0, "LNG"]

    origin = Point(0, 0)

    cnt = 0
    for index, row in dataset_info.iterrows():
        #print("scale rotate cnt: ", cnt)
        cnt += 1
        lat_2 = haversine((center_lat, 0), (row["LAT"], 0))
        lng_2 = haversine((0, center_lng), (0, row["LNG"]))

        p = GeoSeries(Point(lng_2, lat_2))
        p = p.scale(2, 2, origin=origin)
        p = p.rotate(37, origin=origin)
        dataset_info.loc[index, "LAT_2"] = p[0].y
        dataset_info.loc[index, "LNG_2"] = p[0].x

    dataset_info = dataset_info.set_index(["NODE"])
    mean_lat = dataset_info["LAT"].mean()
    mean_lng = dataset_info["LNG"].mean()
    mean_lat_2 = dataset_info["LAT_2"].mean()
    mean_lng_2 = dataset_info["LNG_2"].mean()

    nodes = list(init_data["NODE"].unique())
    cnt = 0
    for node in nodes:
        cnt += 1
        print("extra nodes cnt: ",cnt)
        if node not in dataset_info.index:
            temp = pd.DataFrame({"NODE_2": str(dataset_info.shape[0]),
                                 "LAT": mean_lat,
                                 "LNG": mean_lng,
                                 "LAT_2": mean_lat_2,
                                 "LNG_2": mean_lng_2,
                                 },
                                index=[node])
            dataset_info = dataset_info.append(temp)

    dataset_info = dataset_info.reset_index()
    dataset_info = dataset_info.rename(columns={"index": "NODE"})
    dataset_info.to_csv(output_path, index=False)

    return dataset_info


def combine_main_dataset_with_info_and_anonymize(init_data, dataset_info, init_data_output_path,
                                                 anonym_data_output_path):
    """Create the dataset which combines the the occupancy information of the parking spaces and their locations
    The new generated dataset holds both the real information of the nodes and also the anonymized ones

    Keyword arguments:
    init_data -- the initial data for making it anonymize
    dataset_info -- the dataset which contains the location information
    output_path -- path to the output file for storing the anonymization map
    """

    init_data["LAT"] = None
    init_data["LNG"] = None
    init_data["LAT_2"] = None
    init_data["LNG_2"] = None

    dataset_info = dataset_info.set_index(["NODE"])
    nodes = list(init_data["NODE"].unique())
    cnt = 0
    for node in nodes:
        cnt += 1
        print(cnt)
        selection = init_data["NODE"] == node

        init_data.loc[selection, "LAT"] = dataset_info.loc[node, "LAT"]
        init_data.loc[selection, "LNG"] = dataset_info.loc[node, "LNG"]
        init_data.loc[selection, "LAT_2"] = dataset_info.loc[node, "LAT_2"]
        init_data.loc[selection, "LNG_2"] = dataset_info.loc[node, "LNG_2"]
        init_data.loc[selection, "NODE_2"] = dataset_info.loc[node, "NODE_2"]

    init_data["TIME"] = pd.to_datetime(init_data["TIME"])
    delta_time = init_data["TIME"][0] - datetime(2021, 1, 1, init_data["TIME"][0].hour, init_data["TIME"][0].minute,
                                                 init_data["TIME"][0].second)
    init_data["TIME_2"] = init_data["TIME"] - delta_time

    init_data = init_data[["NODE", "NODE_2", "LAT", "LNG", "LAT_2", "LNG_2", "TIME", "TIME_2", "ACTIVE"]]
    init_data.to_csv(init_data_output_path, index=False)

    anonym_data = init_data[["NODE_2", "LAT_2", "LNG_2", "TIME_2", "ACTIVE"]]
    anonym_data = anonym_data.rename(columns={"NODE_2": "NODE", "LAT_2": "LAT", "LNG_2": "LNG", "TIME_2": "TIME"})
    anonym_data.to_csv(anonym_data_output_path, index=False)

    return init_data, anonym_data


def slice_dataset(init_data, begin_date, end_date):
    """Slice the dataset for given begin and end dates.

    Keyword arguments:
    init_data -- the dataset to be sliced
    begin_date -- the begin date to start the slicing
    end_date -- the end date to end the slicing
    """
    temp = init_data.loc[(init_data["TIME"] >= begin_date) & (init_data["TIME"] < end_date)]
    return temp


def create_benign_data_for_node(data, dates, begin_date, time_step, node, benign_data, benign_data_rows):
    """Create benign dataset for a given dataset and node, and dates.

    Keyword arguments:
    data -- the dataset to be used for generating benign data
    dates -- a list of dates to be used for assigning the occupancy status
    begin_date -- the begin date of assignment
    time_step -- the time steps between dates
    node -- the node to be used for assigning the occupancy status
    output_path -- the path for storing the benign dataset
    """
    #benign_data = pd.DataFrame()
    benign_data["TIME"] = dates
    benign_data["NODE"] = node
    benign_data["LAT"] = list(data.loc[data["NODE"] == node, "LAT"])[0]
    benign_data["LNG"] = list(data.loc[data["NODE"] == node, "LNG"])[0]
    benign_data["ACTIVE"] = 0
    benign_data = benign_data[["NODE", "LAT", "LNG", "TIME", "ACTIVE"]]
    benign_data = benign_data.sort_values(by=["TIME"])

    data_sid = data.loc[data["NODE"] == node]
    data_sid = data_sid.sort_values(by=["TIME"])
    start_time = begin_date
    for index, row in data_sid.iterrows():
        finish_time = row["TIME"]
        benign_data.loc[(benign_data["TIME"] >= row["TIME"]) &
                        (benign_data["TIME"] < row["TIME"]+timedelta(seconds=time_step)), "ACTIVE"] = row["ACTIVE"]

        benign_data.loc[(benign_data["TIME"] >= start_time) &
                        (benign_data["TIME"] < finish_time), "ACTIVE"] = int(not(row["ACTIVE"]))

        start_time = row["TIME"]+timedelta(seconds=time_step)

    #benign_data.to_csv(output_path, mode='a', header=False, index=False)
    benign_data_rows.append(benign_data)

    #return benign_data


def create_benign_dataset(data, begin_date, end_date, time_step, num_nodes, benign_packet_cauchy, output_path):
    """Create benign dataset for a given dataset. Benign dataset contains the occupancy status of each node starting
    from the begin_date to end_date with the step of time_step. num_nodes will be used to randomly select num_nodes
    nodes from all of the nodes in the original dataset.

    Keyword arguments:
    data -- the dataset to be used for generating benign data
    begin_date -- the begin date of assignment
    emd_date -- the end date of assignment
    time_step -- the time steps between dates
    num_nodes -- number of nodes to be selected out the whole nodes in the dataset
    output_path -- the path for storing the benign dataset
    """
    dates = []
    date = begin_date
    while date < end_date:
        dates.append(date)
        date += timedelta(seconds=time_step)

    nodes = list(data["NODE"].unique())
    #num_nodes = math.ceil(len(nodes)*nodes_ratio)
    num_nodes = min(num_nodes, len(nodes))
    nodes = list(random.sample(nodes, k=num_nodes))
    benign_data = pd.DataFrame(columns=["NODE", "LAT", "LNG", "TIME", "ACTIVE", "ATTACKED"])
    #benign_data.to_csv(output_path, index=False)

    manager = Manager()
    benign_data_rows = manager.list([benign_data])


    p = Pool()
    p.starmap(create_benign_data_for_node, product([data], [dates], [begin_date], [time_step],
                                                       nodes, [benign_data], [benign_data_rows]))
    p.close()
    p.join()


    benign_data = pd.concat(benign_data_rows, ignore_index=True)
    benign_data = benign_data.sort_values(by=["NODE", "TIME"])

    packet_dist = tfp.substrates.numpy.distributions.TruncatedCauchy(loc=benign_packet_cauchy[0],
                                                                     scale=benign_packet_cauchy[1],
                                                                     low=0,
                                                                     high=benign_packet_cauchy[2])
    packet = packet_dist.sample([benign_data.shape[0]])
    packet = np.ceil(packet)

    benign_data["PACKET"] = packet
    benign_data["ATTACKED"] = 0
    benign_data.loc[benign_data["ACTIVE"] == 0, "PACKET"] = 0
    benign_data["TIME_HOUR"] = benign_data["TIME"].dt.hour
    benign_data = benign_data[["NODE", "LAT", "LNG", "TIME", "TIME_HOUR", "ACTIVE", "PACKET", "ATTACKED"]]
    benign_data.to_csv(output_path, index=False)


def main_generate_benign_data():
    seed = 6
    random.seed(seed)
    original_data_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output_Initial/original_data/original_data_2021-01-01 23:00:00_2021-02-01 23:59:59.csv"
    original_data = load_dataset(original_data_path)

    benign_packet_path = CONFIG.OUTPUT_DIRECTORY + "N_BaIoT/Output/Data/SimpleHome_XCS7_1003_WHT_Security_Camera_benign_aggregation_Source-IP_time_window_10-sec_stat_Number.csv"
    benign_packet = pd.read_csv(benign_packet_path)
    benign_packet_cauchy = list(cauchy.fit(benign_packet["PACKET"]))
    benign_packet_cauchy.append(math.ceil(max(benign_packet["PACKET"])))
    #print(benign_packet_cauchy)


    begin_date = original_data["TIME"][0]
    begin_date = datetime(begin_date.year, begin_date.month, begin_date.day+1, 0, 0, 0)
    end_date = original_data["TIME"][original_data.shape[0]-1]

    original_data = original_data.loc[(original_data["TIME"] >= begin_date) &
                                      (original_data["TIME"] <= end_date)].reset_index(drop=True)

    nodes = list(original_data["NODE"].unique())
    print("len(nodes): ", len(nodes))

    # num_nodes = len(nodes)
    num_nodes = 50
    time_step = 60 * 10

    scale_factor = time_step/10
    #scale_factor = 3
    benign_packet_cauchy = [scale_factor * i for i in benign_packet_cauchy]


    benign_data_output_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/benign_data/benign_data_" +\
                  str(begin_date) + '_' + str(end_date) + "_time_step_" +\
                  str(time_step) + "_num_ids_" + str(num_nodes) + ".csv"
    #benign_data_output_path = CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/benign_data/benign_data_" +\
    #              str(begin_date) + '_' + str(end_date) + "_time_step_" +\
    #              str(30) + "_num_ids_" + str(20) + ".csv"
    prepare_output_directory(benign_data_output_path)
    create_benign_dataset(original_data, begin_date, end_date, time_step, num_nodes, benign_packet_cauchy,
                          benign_data_output_path)


if __name__ == "__main__":
    #main_generate_original_data()
    main_generate_benign_data()