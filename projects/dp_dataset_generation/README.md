# DP dataset generation

Generate the differentially private datasets 
 - with 0.1 - 0.9 epsilon range with a 0.1 epsilon step or 
 - with 1 - 10 epsilon range and with a 1 epsilon step.

The generation uses Gaussian noise generation as provided by https://programming-dp.com/

The datasets generated here are then used by the model training subproject for input level DP.

Dependencies - `pandas`, `numpy`.

Usage example: `python generate_dp_datasets.py`