# Differential privacy application in water quality prediction

This is the source code for the article "Application of differential privacy to sensor data in water quality monitoring task" published in the journal "Ecological Informatics"
The article is available here: https://doi.org/10.1016/j.ecoinf.2025.103019

The project holds the following subprojects:

 1) `different_model_training` - the implementation of different model architecture training for various privacy levels (levels of epsilon from 0 to 10. See the subproject config file for more details). There are two settings available - input level (training with vanilla model and DP dataset) and optimizer level DP (training with vanilla dataset and differentially private optimizer).
 
 2) `FL_with_model_2` - the federated learning implementation using the 2nd model which was the best performing model from the previous sub project tests. Here, the idea is to partition the dataset into 4 parts and launch a FL training with four clients with each of these parts respectively. In addition, three DP stages can be chosen (0 - for epsilon < 1, 1 - for epsilon < 5, 2 - for epsilon < 10).

 3) `dp_dataset_generation` - the generation of differentially private datasets for input level training for the respective epsilon values.

 4) `data_preparation` - the inital data preprocesing and augmentation.

 We have made part of the used data available under the `data/` directory. This dataset then can be used for DP dataset generation with the 3rd project and for the training of the input (after the DP dataset generation) and optimizer level DP.
