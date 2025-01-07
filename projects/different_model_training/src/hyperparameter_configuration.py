from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import pandas as pd
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

from src.models import build_model, save_training_img
from src.data_processing import configure_default_dataset


def clipping_search(config):
    # Clipping norm search for avg loss retrieval

    clipping_norms = config["search_parameters"]["clip_norms"]

    IMG_PATH = config["log_paths"]["clipping_search"] + 'img/'
    MODELS = config["log_paths"]["clipping_search"] + 'models/'
    EVAL = config["log_paths"]["clipping_search"] + 'eval/'

    x_train, y_train, x_test, y_test = configure_default_dataset(config)

    data_test = pd.read_csv(config["data_paths"]["data_test"], sep=config["data"]["sep"])

    model_ids = config["models"]["model_ids"]

    runs = config["search_parameters"]["clipping_runs"]
    for run_it in range(runs):
        for cn in clipping_norms:
            CLIPPING_NORM = cn
            for m in model_ids:
                # Searching for the smallest change in the non-private model
                n = 0

                print('Training started:', 'model_' + str(m) + '_param_' + str(n))
                model = build_model((x_train.shape[1],), config["training_parameters"]["num_classes"], config["training_parameters"]["learning_rate"], m, n)
                model_file = MODELS + 'model_' + str(m) + '_param_' + str(n) +'.h5'
                checkpointer = ModelCheckpoint(filepath=model_file, 
                                    verbose=1, 
                                    save_best_only=True, 
                                    monitor='val_loss', 
                                    mode="min")
                
                # stopper = EarlyStopping(monitor='val_loss', 
                #                 min_delta=min_delta, 
                #                 patience=patience_val, 
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
                pd.DataFrame(history.history['loss']).to_csv(EVAL + 'run_' + str(run_it) + '_clipping_search_' + str(CLIPPING_NORM) + '_model_' + str(m) + '_param_' + str(n) + '_hist.csv', sep=config["data"]["sep"])

                pd.DataFrame(history.history['val_loss']).to_csv(EVAL + 'run_' + str(run_it) + '_clipping_search_' + str(CLIPPING_NORM) + '_model_' + str(m) + '_param_' + str(n) + '_hist_val.csv', sep=config["data"]["sep"])

                print('Saving image:', 'clipping_search_model_' + str(m) + '_param_' + str(n))
                save_training_img(history, m, n,IMG_PATH, 'run_' + str(run_it) + '_clipping_search_' + str(CLIPPING_NORM))
                print('Training finished:', 'clipping_search_model_' + str(m) + '_param_' + str(n))
                print('Starting testing:', 'clipping_search_model_' + str(m) + '_param_' + str(n))

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
                cm_val.to_csv(EVAL + 'run_' + str(run_it) + '_clipping_search_' + str(CLIPPING_NORM) + '_model_' + str(m) + '_param_' + str(n) + '_test_confusion_matrix.csv', sep=config["data"]["sep"])
                print('Finished testing:', 'model_' + str(m) + '_param_' + str(n))

def clipping_norm_avg_analysis(config):
    # Clipping norm search for avg loss analysis

    clipping_norms = config["search_parameters"]["clip_norms"]

    RUNS = config["search_parameters"]["clipping_runs"]
    MODEL_TYPE_COUNT = len(config["models"]["model_ids"])

    IMG_PATH =  config["log_paths"]["clip_avg"] + 'img/'
    EVAL = config["log_paths"]["clipping_search"] + 'eval/'

    for clipping_norm in clipping_norms:
        for model_type in range(1, MODEL_TYPE_COUNT+1):
            fig, axs = plt.subplots(figsize=(28, 45))
            plt.subplots_adjust(bottom=0.2)
            plt.xticks(rotation='vertical')
            plt.ion()
            plt.pause(1)
            plt.title("Model {} Clipping {} analysis".format(model_type, clipping_norm))
            plt.xlabel("Epoch")
            plt.ylabel("Training loss")
            
            for run_it in range(RUNS):
                cl_df = pd.read_csv(EVAL + "run_" + str(run_it) + "_clipping_search_" + str(clipping_norm) + "_model_" + str(model_type) + "_param_0_hist.csv", delimiter=';')
                
                plt.plot([k for k in range(len(cl_df))], cl_df[cl_df.keys()[1]], label="run {}".format(run_it))

            plt.legend()
            plt.savefig(IMG_PATH + "model_{}_clipping_{}_analysis.png".format(model_type, clipping_norm))
            plt.close()

def learning_rate_sweep(config):
    # Learning rate sweep example

    learning_rate_start = config["search_parameters"]["learning_rate_start"] # 0.001

    learning_rates = [learning_rate_start*1/10**k for k in range(config["search_parameters"]["learning_rate_range"])]
    learning_rates.remove(0.001) # because we have already used this in all tests

    chosen_nm = config["training_parameters"]["noise"]

    clipping_norms = config["training_parameters"]["clipping_norms"]

    MODELS = config["log_paths"]["lr_sweep"] + 'models/'
    EVAL = config["log_paths"]["lr_sweep"] + 'eval/'
    IMG_PATH = config["log_paths"]["lr_sweep"] + 'img/'

    x_train, y_train, x_test, y_test = configure_default_dataset(config)

    data_test = pd.read_csv(config["data_paths"]["data_test"], sep=config["data"]["sep"])

    model_ids = config["models"]["model_ids"]

    for nm_val in chosen_nm:
        n = nm_val
        
        for m in model_ids:
            for lr_val in learning_rates:
                learning_rate = lr_val
                CLIPPING_NORM = clipping_norms[m]

                print('Training started:', 'model_' + str(m) + '_param_' + str(n))
                model = build_model((x_train.shape[1],), config["training_parameters"]["num_classes"], learning_rate, m, n)
                model_file = MODELS + f"lr_{learning_rate}_cn_{CLIPPING_NORM}" + 'model_' + str(m) + '_param_' + str(n) +'.h5'
                checkpointer = ModelCheckpoint(filepath=model_file, 
                                    verbose=1, 
                                    save_best_only=True, 
                                    monitor='val_loss', 
                                    mode="min")
                # stopper = EarlyStopping(monitor='val_loss', 
                #                 min_delta=min_delta, 
                #                 patience=patience_val, 
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
                pd.DataFrame(history.history['loss']).to_csv(EVAL + f"lr_{learning_rate}_cn_{CLIPPING_NORM}" + 'model_' + str(m) + '_param_' + str(n) + '_hist.csv', sep=config["data"]["sep"])

                pd.DataFrame(history.history['val_loss']).to_csv(EVAL + f"lr_{learning_rate}_cn_{CLIPPING_NORM}" + 'model_' + str(m) + '_param_' + str(n) + '_hist_val.csv', sep=config["data"]["sep"])

                print('Saving image:', 'model_' + str(m) + '_param_' + str(n))
                
                save_training_img(history, m, n, IMG_PATH, meta=f"lr_{learning_rate}_cn_{CLIPPING_NORM}")
                
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
                cm_val.to_csv(EVAL +  f"lr_{learning_rate}_cn_{CLIPPING_NORM}" + 'model_' + str(m) + '_param_' + str(n) + '_test_confusion_matrix.csv', sep=config["data"]["sep"])
                print('Finished testing:', 'model_' + str(m) + '_param_' + str(n))
