from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import pandas as pd
import tensorflow as tf
import numpy as np 

from src.models import build_model, save_training_img
from src.data_processing import configure_default_dataset

def optimizer_private_training(config,
                               optimal_clipnorm):
    # Clipping norm specific model training with DP settings
    IMG_PATH = None
    MODELS = None
    EVAL = None

    clipping_norms = None
    if optimal_clipnorm:
        clipping_norms = config["training_parameters"]["clipping_norms"]

        EVAL = config["log_paths"]["optimizer_opt_clipping"] + 'eval/'
        MODELS = config["log_paths"]["optimizer_opt_clipping"] + 'models/'
        IMG_PATH = config["log_paths"]["optimizer_opt_clipping"] + 'img/'
    else:
        clipping_norms = [1.5, 1.5, 1.5, 1.5, 1.5]

        IMG_PATH = config["log_paths"]["optimizer_default_clipping"] + 'img/'
        MODELS = config["log_paths"]["optimizer_default_clipping"] + 'models/'
        EVAL = config["log_paths"]["optimizer_default_clipping"] + 'eval/'

    # For epsilon 0.1 - 0.9 with 0.1 epsilon step and 1 - 10 with 1 epsilon step 
    noise = config["training_parameters"]["noise"]

    epsilon_values = [k/10 for k in range(1,10)]
    epsilon_values.extend([k for k in range(1,11)])

    x_train, y_train, x_test, y_test = configure_default_dataset(config)

    data_test = pd.read_csv(config["data_paths"]["data_test"], sep=config["data"]["sep"])

    with open(EVAL + "dp_log.txt", "w") as dpf:
        dpf.write('name,noise_multiplier,epsilon,epochs,batch_size\n')

    model_ids = config["models"]["model_ids"]

    for m in model_ids:
        
        CLIPPING_NORM = clipping_norms[m]
        
        for n in noise:
            epsilon = epsilon_values[noise.index(n)]
            
            print('Training started:', 'model_' + str(m) + '_param_' + str(n))
            model = build_model((x_train.shape[1],), config["training_parameters"]["num_classes"], config["training_parameters"]["learning_rate"], CLIPPING_NORM, m, n)
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
            pd.DataFrame(history.history['loss']).to_csv(EVAL + 'model_' + str(m) + '_param_' + str(n) + '_hist.csv', sep=config["data"]["sep"])

            pd.DataFrame(history.history['val_loss']).to_csv(EVAL + 'model_' + str(m) + '_param_' + str(n) + '_hist_val.csv', sep=config["data"]["sep"])
            
            print('Saving image:', 'model_' + str(m) + '_param_' + str(n))
            save_training_img(history, m, n, IMG_PATH)
            print('Training finished:', 'model_' + str(m) + '_param_' + str(n))
            print('Starting testing:', 'model_' + str(m) + '_param_' + str(n))
        
            with open(EVAL + "dp_log.txt", "a+") as dpf:
                dpf.write('model_{},{},{},{},{}\n'.format(str(m), str(n), epsilon, config["training_parameters"]["epochs"], config["training_parameters"]["batch_size"]))
                        
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
            cm_val.to_csv(EVAL + 'model_' + str(m) + '_param_' + str(n) + '_test_confusion_matrix.csv', sep=config["data"]["sep"])
            print('Finished testing:', 'model_' + str(m) + '_param_' + str(n))
           