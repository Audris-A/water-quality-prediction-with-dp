import pandas as pd
import numpy as np
import csv
import statistics
import copy

# Default data path
data_folder = "../../data/default_data/"

# Folder to copy the DP data to
dp_folder = "../../data/default_data/dp_data_for_training/"

data_size = None

gaussian_epsilon_start = 1 # 0.1
c_offset_amount = 0.1
delta_calculation_ratio = 2

# Save the sensor data with the rest of the default dataset information - labels and so on.
def save_noisy_dataset_in_csv(filename):
    output_df = pd.DataFrame()
    for sensor in sensor_info:
        output_df[sensor["name"]] = sensor["noisy_data"]

    output_df.to_csv(dp_folder + filename)


def get_noisy_data(noise, initial_data):
    data_with_noise = initial_data
    for k in range(data_size):
        data_with_noise[k] = data_with_noise[k] + noise[k]

    return data_with_noise


def get_gaussian_noise(sensitivity, epsilon, c_offset):
    # add something to the square root of 2ln(1.25/delta) so that c is bigger and not equal
    delta=1/(data_size*delta_calculation_ratio)

    location = 0
    c = np.sqrt(2*np.log(1.25/delta)) + c_offset_amount
    std = c*sensitivity/epsilon

    s = np.random.normal(location, std, data_size)

    return s


# Calculates the difference between the max and min value of the dataset. 
#  That basically is the sensitivity of sensor data column.
def get_max_sensitivity_value(sensor_data):
    return max(sensor_data) - min(sensor_data)


pressure = []
temperature = []
toc = []
turbidity = []
ph = []
orp = []
conductivity = []

# Read and process the data
with open(data_folder + 'data_train.csv', 'r', newline='') as csvfile:
    file_reader = csv.reader(csvfile, delimiter=';')

    for row in file_reader:
        if file_reader.line_num > 1:
            row = [float(row[k]) for k in range(len(row) - 1)]

            # skip the first column because it is an index
            pressure.append(row[1])
            temperature.append(row[2])
            toc.append(row[3])
            turbidity.append(row[4])
            ph.append(row[5])
            orp.append(row[6])
            conductivity.append(row[7])

        # Comment this out if not testing
        # if file_reader.line_num > 5:
        #   break

    data_size = file_reader.line_num - 1

print("Data size =",data_size)

pressure_max_sensitivity = get_max_sensitivity_value(pressure)
temperature_max_sensitivity = get_max_sensitivity_value(temperature)
toc_max_sensitivity = get_max_sensitivity_value(toc)
turbidity_max_sensitivity = get_max_sensitivity_value(turbidity)
ph_max_sensitivity = get_max_sensitivity_value(ph)
orp_max_sensitivity = get_max_sensitivity_value(orp)
conductivity_max_sensitivity = get_max_sensitivity_value(conductivity)

sensor_info = [ 
    {
        "name": "pressure",
        "sensitivity": pressure_max_sensitivity,
        "data": pressure,
        "noisy_data": None,
        "no_noise_avg": None
    },
    {
        "name": "temperature",
        "sensitivity":temperature_max_sensitivity,
        "data":temperature,
        "noisy_data": None,
        "no_noise_avg": None
    },
    {
        "name": "toc",
        "sensitivity":toc_max_sensitivity,
        "data":toc,
        "noisy_data": None,
        "no_noise_avg": None
    },
    {
        "name": "turbidity",
        "sensitivity":turbidity_max_sensitivity,
        "data":turbidity,
        "noisy_data": None,
        "no_noise_avg": None
    },
    {
        "name": "ph",
        "sensitivity":ph_max_sensitivity,
        "data":ph,
        "noisy_data": None,
        "no_noise_avg": None
    },
    {
        "name": "orp",
        "sensitivity":orp_max_sensitivity,
        "data":orp,
        "noisy_data": None,
        "no_noise_avg": None
    },
    {
        "name": "conductivity",
        "sensitivity":conductivity_max_sensitivity,
        "data":conductivity,
        "noisy_data": None,
        "no_noise_avg": None
    }
]

print("\nSensitivity of the sensor data columns:")
max_word_length = max([len(sensor_info[k]["name"]) for k in range(len(sensor_info))]) - 1
for sensor in sensor_info:
    word_length = len(sensor["name"]) - 1
    whitespaces = ""
    for k in range(max_word_length - word_length):
        whitespaces += " "

    print(sensor["name"] + whitespaces,"=",sensor["sensitivity"])

print("\nAverage values without noise:")
for sensor in sensor_info:
    word_length = len(sensor["name"]) - 1
    whitespaces = ""
    for k in range(max_word_length - word_length):
        whitespaces += " "
    avg_value = statistics.mean(sensor["data"])
    sensor["no_noise_avg"] = avg_value
    print(sensor["name"] + whitespaces,"=",avg_value)


# Gaussian noise addition
#  Do the gaussian noise addition with 10 steps for epsilon in the range of 0.9 to 10. 
#  In each step epsilon is increased by 1.
for k in range(9,10):

    # Use this for dataset generation for epsilon below 1 with step 0.1 in the for loop
    #  ­`round(gaussian_epsilon_start+round(k/10,2), 2)`
    current_epsilon = gaussian_epsilon_start+k

    for sensor in sensor_info:
        data_to_noise = copy.deepcopy(sensor["data"])
        sensor["noisy_data"] = get_noisy_data(get_gaussian_noise(sensitivity=sensor["sensitivity"], 
                                                                 epsilon=current_epsilon, 
                                                                 c_offset=c_offset_amount
                                                                 ), 
                                              data_to_noise
                                              )
    print("\nAverage values with Gaussian noise with epsilon = {}:".format(current_epsilon))
    for sensor in sensor_info:
        word_length = len(sensor["name"]) - 1
        whitespaces = ""
        for k_iter in range(max_word_length - word_length):
            whitespaces += " "

        avg_value = statistics.mean(sensor["noisy_data"])
        sensor["gauseps" + str(k+1)].append(avg_value)
        print(sensor["name"] + whitespaces,"=",avg_value)
    
    # Save the data with Gaussian noise for each epsilon config
    save_noisy_dataset_in_csv("sensor_data_with_gaussian_noise_eps_{}.csv".format(current_epsilon))
