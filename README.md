# Differential privacy application in water quality prediction


The project holds the following subprojects:

 1) `different model training` - the implementation of different model architecture training for various privacy levels (levels of epsilon from 0 to 10. See the subproject config file for more details). There are two settings - input level and optimizer level DP. For this reason, we have made the used data available here `link ??`. Copy the data from the link to the folders `dp_data_for_training` and `default_data` for datasets with added privacy the default dataset (data without noise) accordingly.
 
 2) `FL_with_model_2' - the federated learning implementation using the 2nd model which was the best performing model from the previous sub project tests. Here the idea is to partition the dataset into 4 parts and launch a FL training with four clients with each of these parts respectively. In addition, three DP stages can be chosen (0 - for epsilon < 1, 1 - for epsilon < 5, 2 - for epsilon < 10).

 3) `??` - the inital data preprocesing and augmentation.