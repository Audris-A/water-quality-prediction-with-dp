import copy
import pandas as pd


def configure_dataset_test_data_epsilon_above_1(EPSILON,
                                                config):

    print(f"configuring data with epsilon = {EPSILON}")
    dp_data_train = pd.read_csv(config["data_paths"]["dp_dataset_above_1"].format(EPSILON), sep=",")

    dp_data_train.drop(columns=[dp_data_train.keys()[0]], inplace=True)

    dp_data_train.rename(columns={"pressure": "Pressure", 
                                  "temperature": "Temp", 
                                  "toc": "TOC", 
                                  "turbidity": "Turbidity",
                                  "ph": "pH",
                                  "orp": "ORP",
                                  "conductivity": "Conductivity"}, inplace=True)

    data_train = pd.read_csv(config["data_paths"]["data_train"], sep=config["data"]["sep"])

    # add missing columns that didn't require adding noise
    dp_df_keys = dp_data_train.keys()
    for vanilla_data_column in data_train:
        if vanilla_data_column not in dp_df_keys and vanilla_data_column != "Unnamed: 0":
            dp_data_train[vanilla_data_column] = data_train[vanilla_data_column]

    data_train = copy.deepcopy(dp_data_train)
    data_test= pd.read_csv(config["data_paths"]["data_test"], sep=config["data"]["sep"])

    data_train['Normal'] = 1.
    data_train['Normal'] = data_train['Normal'] - data_train['Anomaly']

    data_test['Normal'] = 1.
    data_test['Normal'] = data_test['Normal'] - data_test['Anomaly']


    x_train = data_train[config["data_colums"]["x"]].copy()
    y_train = data_train[config["data_colums"]["y"]].copy()

    x_test = data_test[config["data_colums"]["x"]].copy()
    y_test = data_test[config["data_colums"]["y"]].copy()

    return x_train, y_train, x_test, y_test


def configure_dataset_test_data_epsilon_below_1(EPSILON,
                                                config):

    print(f"configuring data with epsilon = {EPSILON}")
    dp_data_train = pd.read_csv(config["data_paths"]["dp_dataset_below_1"].format(EPSILON), sep=",")

    dp_data_train.drop(columns=[dp_data_train.keys()[0]], inplace=True)

    dp_data_train.rename(columns={"pressure": "Pressure", 
                                  "temperature": "Temp", 
                                  "toc": "TOC", 
                                  "turbidity": "Turbidity",
                                  "ph": "pH",
                                  "orp": "ORP",
                                  "conductivity": "Conductivity"}, inplace=True)

    data_train = pd.read_csv(config["data_paths"]["data_train"], sep=config["data"]["sep"])

    # add missing columns that didn't require adding noise
    dp_df_keys = dp_data_train.keys()
    for vanilla_data_column in data_train:
        if vanilla_data_column not in dp_df_keys and vanilla_data_column != "Unnamed: 0":
            dp_data_train[vanilla_data_column] = data_train[vanilla_data_column]

    data_train = copy.deepcopy(dp_data_train)
    data_test= pd.read_csv(config["data_paths"]["data_test"], sep=config["data"]["sep"])

    data_train['Normal'] = 1.
    data_train['Normal'] = data_train['Normal'] - data_train['Anomaly']

    data_test['Normal'] = 1.
    data_test['Normal'] = data_test['Normal'] - data_test['Anomaly']


    x_train = data_train[config["data_colums"]["x"]].copy()
    y_train = data_train[config["data_colums"]["y"]].copy()

    x_test = data_test[config["data_colums"]["x"]].copy()
    y_test = data_test[config["data_colums"]["y"]].copy()

    return x_train, y_train, x_test, y_test


def configure_default_dataset(config):
    data_train = pd.read_csv(config["data_paths"]["data_train"], sep=config["data"]["sep"])
    data_test= pd.read_csv(config["data_paths"]["data_test"], sep=config["data"]["sep"])

    data_train['Normal'] = 1.
    data_train['Normal'] = data_train['Normal'] - data_train['Anomaly']

    data_test['Normal'] = 1.
    data_test['Normal'] = data_test['Normal'] - data_test['Anomaly']


    x_train = data_train[config["data_colums"]["x"]].copy()
    y_train = data_train[config["data_colums"]["y"]].copy()

    x_test = data_test[config["data_colums"]["x"]].copy()
    y_test = data_test[config["data_colums"]["y"]].copy()

    return x_train, y_train, x_test, y_test
