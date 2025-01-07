# Different model training

Run input or optimizer level DP for four model architectures with water quality data.

Dependecies - `pandas`, `tensorflow`, `tensorflow`, `matplotlib`, `numpy`.

The following types of subprograms can be run (passed with `-t`, `--type`):
 1. dataset_below_1 - Input level DP training with epsilon < 1;
 2. dataset_above_1 - Input level DP training with epsilon > 1;
 3. optimizer - Optimizer level training;
 4.	clipping_search - Search for optimal clipping norm (required to run for the next subprogram);
 5. cn_search_avg - Analyzes the results from clipping_search;
 6. lr_sweep - Searches for the optimal learning rates.

You can also run the training with the default clipping norm (1.5) by passing `-d 1`

Before running any of the subprograms read the config file and make the necessary adjustments e.g. change the paths to the datasets.

Usage example: `python main.py -t dataset_below_1`