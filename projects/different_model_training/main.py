import argparse
import os

from src.helper import get_global_config
from src.dp_dataset_training import default_model_epsilon_below_1, default_model_epsilon_above_1
from src.dp_optimizer_training import optimizer_private_training
from src.hyperparameter_configuration import clipping_search, learning_rate_sweep, clipping_norm_avg_analysis

def process_log_folder(folder_path):
    if not os.path.isdir(folder_path):
        print("Creating the missing log folders.")
        os.makedirs(f"{folder_path}eval/")
        os.makedirs(f"{folder_path}img/")
        os.makedirs(f"{folder_path}models/")

parser = argparse.ArgumentParser(
                    prog='DP water quality NN training',
                    description='This script will train a neural network on water quality data with differential privacy',
                    epilog='Text at the bottom of help')

parser.add_argument('-t', '--type', 
                    required=True,
                    choices=["dataset_below_1", 
                             "dataset_above_1", 
                             "optimizer", 
                             "clipping_search",
                             "cn_search_avg", 
                             "lr_sweep"], 
                    help="the training type")

parser.add_argument('-d', '--default_clipping', 
                    choices=["1", "0"], 
                    help="Toggles default clipping - 1.5")

args = parser.parse_args()

config = get_global_config()

if args.default_clipping == "1":
    print("Default clipping enabled!")
else:
    print("Default clipping disabled!")

print(f"Received {args.type}")
if args.type == "optimizer":
    # check log folders

    default_clipping_off = True
    if args.default_clipping == "1":
        process_log_folder(config["log_paths"]["optimizer_default_clipping"])
        default_clipping_off = False
    else:
        process_log_folder(config["log_paths"]["optimizer_opt_clipping"])

    optimizer_private_training(config, default_clipping_off)
elif args.type == "dataset_above_1":
    # check log folders

    default_clipping_off = True
    if args.default_clipping == "1":
        process_log_folder(config["log_paths"]["dataset_default_clipping"])
        default_clipping_off = False
    else:
        process_log_folder(config["log_paths"]["dataset_opt_clipping"])

    default_model_epsilon_above_1(config, default_clipping_off)
elif args.type == "dataset_below_1":
    # check log folders

    default_clipping_off = True
    if args.default_clipping == "1":
        process_log_folder(config["log_paths"]["dataset_default_clipping"])
        default_clipping_off = False
    else:
        process_log_folder(config["log_paths"]["dataset_opt_clipping"])

    default_model_epsilon_below_1(config, default_clipping_off)
elif args.type == "clipping_search":
    process_log_folder(config["log_paths"]["clipping_search"])
    clipping_search(config)
elif args.type == "cn_search_avg":
    if not os.path.isdir(config["log_paths"]["clipping_search"]):
        print("I think you haven't ran the clipping search yet. Run that before running the avg analysis.")
        exit()

    process_log_folder(config["log_paths"]["clip_avg"])
    clipping_norm_avg_analysis(config)
elif args.type == "lr_sweep":
    process_log_folder(config["log_paths"]["lr_sweep"])
    learning_rate_sweep(config)
else:
    print("Unknown type passed.")
    parser.print_help()