from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import pandas as pd
import tensorflow as tf
import numpy as np 

from src.data_processing import configure_dataset_test_data_epsilon_above_1, configure_dataset_test_data_epsilon_below_1
from src.models import build_model, save_training_img


def default_model_epsilon_above_1(config,
                                  optimal_clipnorm):
    # The default model training with DP dataset for epsilon > 1 and optimal clipping norms

    IMG_PATH = None
    MODELS = None
    EVAL = None
    EPSILON = None

    clipping_norms = None
    if optimal_clipnorm:
        clipping_norms = config["training_parameters"]["clipping_norms"]

        IMG_PATH = config["log_paths"]["dataset_opt_clipping"] + 'img/'
        MODELS = config["log_paths"]["dataset_opt_clipping"] + 'models/'
        EVAL = config["log_paths"]["dataset_opt_clipping"] + 'eval/'
    else:
        clipping_norms = [1.5, 1.5, 1.5, 1.5, 1.5]

        IMG_PATH = config["log_paths"]["dataset_default_clipping"] + 'img/'
        MODELS = config["log_paths"]["dataset_default_clipping"] + 'models/'
        EVAL = config["log_paths"]["dataset_default_clipping"] + 'eval/'
    
    model_ids = config["models"]["model_ids"]

    data_test = pd.read_csv(config["data_paths"]["data_test"], sep=config["data"]["sep"])

    epsilon_value_array = [k+1 for k in range(10)]
    for eps_val in epsilon_value_array:
        # configure dataset
        EPSILON = eps_val
        
        x_train, y_train, x_test, y_test = configure_dataset_test_data_epsilon_above_1(EPSILON, config)

        for m in model_ids:
            
            CLIPPING_NORM = clipping_norms[m]
            
            epsilon = 1
            n = 0
            print('Training started:', 'model_' + str(m) + '_param_' + str(n))

            # Set classic to True because we need the non-DP optimizer
            model = build_model((x_train.shape[1],), config["training_parameters"]["num_classes"], config["training_parameters"]["learning_rate"], CLIPPING_NORM, m, n, classic=True)

            model_file = MODELS + 'model_' + str(m) + '_param_' + str(n) +'.h5'
            checkpointer = ModelCheckpoint(filepath=model_file, 
                                verbose=1, 
                                save_best_only=True, 
                                monitor='val_loss', 
                                mode="min")
            # stopper = EarlyStopping(monitor='val_loss', 
            #                 min_delta=min_delta, 
            #                 patience=config["training_parameters"]["patience_val"], 
            #                 verbose=0, 
            #                 mode='min',
            #                 baseline=None, 
            #                 restore_best_weights=False)

            history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=config["training_parameters"]["batch_size"],
                        epochs=config["training_parameters"]["epochs"],
                        callbacks=[checkpointer], # , stopper
                        shuffle=False,
                        verbose=0,
                        validation_data=(x_test, y_test))

            print('Saving history:', 'model_' + str(m) + '_param_' + str(n))
            pd.DataFrame(history.history['loss']).to_csv(EVAL + 'model_' + str(m) + '_epsilon_' + str(EPSILON) + '_hist.csv', sep=config["data"]["sep"])

            pd.DataFrame(history.history['val_loss']).to_csv(EVAL + 'model_' + str(m) + '_epsilon_' + str(EPSILON) + '_hist_val.csv', sep=config["data"]["sep"])

            # log loss
            print('Saving image:', 'model_' + str(m) + '_param_' + str(n))
            save_training_img(history, m, n,IMG_PATH, "epsilon_{}_".format(EPSILON))
            print('Training finished:', 'model_' + str(m) + '_param_' + str(n))
            print('Starting testing:', 'model_' + str(m) + '_param_' + str(n))

            print('Predictiong anomalies:', 'model_' + str(m) + '_param_' + str(n))
            model = tf.keras.models.load_model(model_file, compile=False)

            prognoze = model.predict(data_test[config["data_colums"]["x"]])
            data_test['AnomalyForecasted'] = prognoze.argmax(axis=1)
            data_test['AnomalyProbability'] = prognoze[:, 1]

            del(prognoze) 

            cm_val = pd.DataFrame(np.empty((config["training_parameters"]["num_classes"], config["training_parameters"]["num_classes"])), 
                                columns=['Normal', 'Anomaly'], index=['Normal', 'Anomaly'])
            print('Calculating confusion matrix:', 'model_' + str(m) + '_param_' + str(n))
            cm_val.loc['Normal', 'Normal'] = data_test.loc[(data_test['Anomaly'] == 0) & (data_test['AnomalyForecasted'] == 0)].count()[0]
            cm_val.loc['Anomaly', 'Anomaly'] = data_test.loc[(data_test['Anomaly'] == 1) & (data_test['AnomalyForecasted'] == 1)].count()[0]
            cm_val.loc['Normal', 'Anomaly'] = data_test.loc[(data_test['Anomaly'] == 0) & (data_test['AnomalyForecasted'] == 1)].count()[0]
            cm_val.loc['Anomaly', 'Normal'] = data_test.loc[(data_test['Anomaly'] == 1) & (data_test['AnomalyForecasted'] == 0)].count()[0]
            cm_val.to_csv(EVAL + 'model_' + str(m) + '_epsilon_' + str(EPSILON) + '_test_confusion_matrix.csv', sep=config["data"]["sep"])
            print('Finished testing:', 'model_' + str(m) + '_param_' + str(n))

            epsilon += 1


def default_model_epsilon_below_1(config,
                                  optimal_clipnorm):
    
    # The default model training with DP dataset and with optimal clipping norms
    IMG_PATH = None
    MODELS = None
    EVAL = None
    
    clipping_norms = None
    if optimal_clipnorm:
        clipping_norms = config["training_parameters"]["clipping_norms"]

        IMG_PATH = config["log_paths"]["dataset_opt_clipping"] + 'img/'
        MODELS = config["log_paths"]["dataset_opt_clipping"] + 'models/'
        EVAL = config["log_paths"]["dataset_opt_clipping"] + 'eval/'
    else:
        clipping_norms = [1.5, 1.5, 1.5, 1.5, 1.5]

        IMG_PATH = config["log_paths"]["dataset_default_clipping"] + 'img/'
        MODELS = config["log_paths"]["dataset_default_clipping"] + 'models/'
        EVAL = config["log_paths"]["dataset_default_clipping"] + 'eval/'

    model_ids = config["models"]["model_ids"]

    data_test = pd.read_csv(config["data_paths"]["data_test"], sep=config["data"]["sep"])

    EPSILON = None
    for eps_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # configure dataset
        EPSILON = eps_val
        
        x_train, y_train, x_test, y_test = configure_dataset_test_data_epsilon_below_1(EPSILON, config)

        for m in model_ids:
            CLIPPING_NORM = clipping_norms[m]
            
            epsilon = 1
            n = 0
            print('Training started:', 'model_' + str(m) + '_param_' + str(n))

            # Set classic to True because we need the non-DP optimizer
            model = build_model((x_train.shape[1],), config["training_parameters"]["num_classes"], config["training_parameters"]["learning_rate"], CLIPPING_NORM, m, n, classic=True)

            model_file = MODELS + 'model_' + str(m) + '_param_' + str(n) +'.h5'
            checkpointer = ModelCheckpoint(filepath=model_file, 
                                verbose=1, 
                                save_best_only=True, 
                                monitor='val_loss', 
                                mode="min")
            # stopper = EarlyStopping(monitor='val_loss', 
            #                 min_delta=min_delta, 
            #                 patience=config["training_parameters"]["patience_val"], 
            #                 verbose=0, 
            #                 mode='min',
            #                 baseline=None, 
            #                 restore_best_weights=False)

            history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=config["training_parameters"]["batch_size"],
                        epochs=config["training_parameters"]["epochs"],
                        callbacks=[checkpointer], # , stopper
                        shuffle=False,
                        verbose=0,
                        validation_data=(x_test, y_test))

            print('Saving history:', 'model_' + str(m) + '_param_' + str(n))
            pd.DataFrame(history.history['loss']).to_csv(EVAL + 'model_' + str(m) + '_epsilon_' + str(EPSILON) + '_hist.csv', sep=config["data"]["sep"])

            pd.DataFrame(history.history['val_loss']).to_csv(EVAL + 'model_' + str(m) + '_epsilon_' + str(EPSILON) + '_hist_val.csv', sep=config["data"]["sep"])

            print('Saving image:', 'model_' + str(m) + '_param_' + str(n))
            save_training_img(history, m, n,IMG_PATH, "epsilon_{}_".format(EPSILON))
            print('Training finished:', 'model_' + str(m) + '_param_' + str(n))
            print('Starting testing:', 'model_' + str(m) + '_param_' + str(n))

            print('Predictiong anomalies:', 'model_' + str(m) + '_param_' + str(n))
            model = tf.keras.models.load_model(model_file, compile=False)

            prognoze = model.predict(data_test[config["data_colums"]["x"]])
            data_test['AnomalyForecasted'] = prognoze.argmax(axis=1)
            data_test['AnomalyProbability'] = prognoze[:, 1]

            del(prognoze) 

            cm_val = pd.DataFrame(np.empty((config["training_parameters"]["num_classes"], config["training_parameters"]["num_classes"])), 
                                columns=['Normal', 'Anomaly'], index=['Normal', 'Anomaly'])
            print('Calculating confusion matrix:', 'model_' + str(m) + '_param_' + str(n))
            cm_val.loc['Normal', 'Normal'] = data_test.loc[(data_test['Anomaly'] == 0) & (data_test['AnomalyForecasted'] == 0)].count()[0]
            cm_val.loc['Anomaly', 'Anomaly'] = data_test.loc[(data_test['Anomaly'] == 1) & (data_test['AnomalyForecasted'] == 1)].count()[0]
            cm_val.loc['Normal', 'Anomaly'] = data_test.loc[(data_test['Anomaly'] == 0) & (data_test['AnomalyForecasted'] == 1)].count()[0]
            cm_val.loc['Anomaly', 'Normal'] = data_test.loc[(data_test['Anomaly'] == 1) & (data_test['AnomalyForecasted'] == 0)].count()[0]
            cm_val.to_csv(EVAL + 'model_' + str(m) + '_epsilon_' + str(EPSILON) + '_test_confusion_matrix.csv', sep=config["data"]["sep"])
            print('Finished testing:', 'model_' + str(m) + '_param_' + str(n))

            epsilon += 1
            
            