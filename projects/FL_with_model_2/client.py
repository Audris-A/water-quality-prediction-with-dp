import argparse
import os
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation 

from flwr.client import ClientApp, NumPyClient
import tensorflow as tf
import tensorflow_privacy as tfp

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Rename the following file paths accordingly
data_test_path = '../../data/default_data/data_test.csv'
train_dataset_path = '../../data/default_data/data_train.csv'
test_dataset_path = '../../data/default_data/data_test.csv'

CLIENT_COUNT = 4

rounds = 1000
epochs_per_round = 1
batch_size = 128
num_classes = 2
learning_rate = 0.001
patience_val = 10

clipping_norm = 0.0000001
noise_multiplier = None

classic_optimizer = False


x_cols = ['Temp', 'TOC', 'Turbidity', 'pH', 'ORP', 'Conductivity', 'DoW', 'hour', 'workhour', 'workday']
y_cols = ['Normal','Anomaly']

def configure_default_dataset():
    # Data set preparation
    data_train = pd.read_csv(train_dataset_path, sep=";")
    data_test= pd.read_csv(test_dataset_path, sep=";")

    # The data is being augmented with attribute 'Normal'. It is equal to 1 when Anomaly is 0 and vice versa.
    data_train['Normal'] = 1.
    data_train['Normal'] = data_train['Normal'] - data_train['Anomaly']

    data_test['Normal'] = 1.
    data_test['Normal'] = data_test['Normal'] - data_test['Anomaly']


    x_train = data_train[x_cols].copy()
    y_train = data_train[y_cols].copy()

    x_test = data_test[x_cols].copy()
    y_test = data_test[y_cols].copy()

    return x_train, y_train, x_test, y_test

def get_optimizer(learning_rate:float,
                  l2_clip_norm:float, 
                  noise_multiplier: float,
                  num_microbatches: int = 1,
                  gradient_accumulation_steps: int = 1,
                  classic: bool = False):
    
    print(f"Optimizer init LR = {learning_rate}")
    
    if not classic:
        print("Using DP optimizer!")
        
        opt = tfp.DPKerasAdamOptimizer(l2_norm_clip=l2_clip_norm,
                                       noise_multiplier=noise_multiplier,
                                       num_microbatches=num_microbatches,
                                       gradient_accumulation_steps=gradient_accumulation_steps,
                                       learning_rate=learning_rate)
    else:
        print("Using classic optimizer!")
        print(f"With clipping norm = {l2_clip_norm}")
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=l2_clip_norm)
        
    return opt

# Defining the model
def build_model2(shape, output_n, opt):
    model = Sequential()
    model.add(Dense(units=25, input_shape=shape, name='Layer1'))
    model.add(Activation(tf.nn.sigmoid))
    model.add(Dense(units=output_n, name='Output'))
    model.add(Activation(tf.nn.softmax))

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])
    return model


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        
        history = model.fit(x=x_train,
                            y=y_train,
                            batch_size=batch_size,
                            epochs=epochs_per_round,
                            shuffle=False,
                            verbose=0,
                            validation_data=(x_test, y_test))
        
        print(f"train loss = {history.history['loss']}")
        print(f"val loss = {history.history['val_loss']}")

        with open(f"fl_client_{args.partition_id}_model_2_nm_{noise_multiplier}_training_loss.csv", "a+") as history_file:
            history_file.write(f"{history.history['loss'][0]}\n")
        
        with open(f"fl_client_{args.partition_id}_model_2_nm_{noise_multiplier}_validation_loss.csv", "a+") as history_file:
            history_file.write(f"{history.history['val_loss'][0]}\n")

        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        
        data_test = pd.read_csv(data_test_path, sep=";")

        prognosis = model.predict(data_test[x_cols])
        data_test['AnomalyForecasted'] = prognosis.argmax(axis=1)
        data_test['AnomalyProbability'] = prognosis[:, 1]

        del(prognosis) 

        cm_val = pd.DataFrame(np.empty((num_classes, num_classes)), 
                            columns=['Normal', 'Anomaly'], index=['Normal', 'Anomaly'])
        print('Calculating confusion matrix...')
        cm_val.loc['Normal', 'Normal'] = data_test.loc[(data_test['Anomaly'] == 0) & (data_test['AnomalyForecasted'] == 0)].count()[0]
        cm_val.loc['Anomaly', 'Anomaly'] = data_test.loc[(data_test['Anomaly'] == 1) & (data_test['AnomalyForecasted'] == 1)].count()[0]
        cm_val.loc['Normal', 'Anomaly'] = data_test.loc[(data_test['Anomaly'] == 0) & (data_test['AnomalyForecasted'] == 1)].count()[0]
        cm_val.loc['Anomaly', 'Normal'] = data_test.loc[(data_test['Anomaly'] == 1) & (data_test['AnomalyForecasted'] == 0)].count()[0]
        cm_val.to_csv(f'fl_client_{args.partition_id}_model_2_nm_{noise_multiplier}_test_confusion_matrix.csv', sep=";")
        print('Finished evaluating!')

        return 0.0, len(x_test), {"accuracy": 0.0}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()

# Parse arguments
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=[0, 1, 2, 3],
    default=0,
    help="Partition of the dataset (0, 1, 2, 3). "
    "The dataset is divided into 4 partitions created artificially.",
)

parser.add_argument(
    "--dp-stage",
    type=int,
    choices=[0, 1, 2],
    default=0,
    help="DP stages (0 - for eps < 1, 1 - for eps < 5, 2 - for eps < 10). ",
)

args, _ = parser.parse_known_args()
print(args.dp_stage)
if int(args.dp_stage) == 0:
    noise_multiplier = 2.668
elif int(args.dp_stage) == 1:
    noise_multiplier = 0.8625
elif int(args.dp_stage) == 2:
    noise_multiplier = 0.6555
else:
    print("Unknown dp stage!")
    exit(0)

x_train, y_train, x_test, y_test = configure_default_dataset()

x_train = np.array_split(x_train, CLIENT_COUNT)[args.partition_id]
y_train = np.array_split(y_train, CLIENT_COUNT)[args.partition_id]

x_test = np.array_split(x_test, CLIENT_COUNT)[args.partition_id]
y_test = np.array_split(y_test, CLIENT_COUNT)[args.partition_id]

# Load model 
optimizer = get_optimizer(learning_rate=learning_rate,
                          l2_clip_norm = clipping_norm, 
                          noise_multiplier = noise_multiplier,
                          classic = classic_optimizer)

model = build_model2((x_train.shape[1],), num_classes, optimizer)

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
