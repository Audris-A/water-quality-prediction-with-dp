import numpy as np 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation 
import tensorflow_privacy as tfp
import matplotlib.pyplot as plt


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


def build_model1(shape, output_n, opt):
    model = Sequential()
    model.add(Dense(units=200, input_shape=shape, name='Layer1'))
    model.add(Activation(tf.nn.sigmoid))
    model.add(Dropout(0.2))
    model.add(Dense(units=100, name='Layer2'))
    model.add(Activation(tf.nn.sigmoid))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_n, name='Output'))
    model.add(Activation(tf.nn.softmax))

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])
    return model


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


def build_model3(shape, output_n, opt):
    model = Sequential()
    model.add(Dense(units=10, input_shape=shape, name='Layer1'))
    model.add(Activation(tf.nn.sigmoid))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_n, name='Output'))
    model.add(Activation(tf.nn.softmax))

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])
    return model


def build_model4(shape, output_n, opt):
    model = Sequential()
    model.add(Dense(units=5, input_shape=shape, name='Layer1'))
    model.add(Activation(tf.nn.sigmoid))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_n, name='Output'))
    model.add(Activation(tf.nn.softmax))

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])
    return model


def save_training_img(hist, modelid:int=np.random.randint(0,100,1), param_val:float=np.random.random(1), img_path="", meta=''):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Modeļa kļūdas')
    plt.ylabel('Kļūdas')
    plt.xlabel('Epohas')
    plt.legend(['Apmācības kļūda (loss)', 'Testēšanas kļūda (val_loss)'], loc='upper right')
    plt.savefig(img_path + meta + 'model_' + str(modelid) + '_param_' + str(param_val) + '.png', format='png')
    plt.cla()
    plt.clf()
    plt.close(fig)
    

def build_model(x_shape, output_n, lr, CLIPPING_NORM, modelid:int, param_val, classic: bool = False):
    print("CLIPPING_NORM = {}".format(CLIPPING_NORM))
    if modelid in [1,2,3,4]:
        if modelid == 1:
            model = build_model1(x_shape,output_n,get_optimizer(lr, CLIPPING_NORM, param_val, classic=classic))
        elif modelid == 2:
            model = build_model2(x_shape,output_n,get_optimizer(lr, CLIPPING_NORM, param_val, classic=classic))
        elif modelid == 3:
            model = build_model3(x_shape,output_n,get_optimizer(lr, CLIPPING_NORM, param_val, classic=classic))
        elif modelid == 4:
            model = build_model4(x_shape,output_n,get_optimizer(lr, CLIPPING_NORM, param_val, classic=classic))
        else:
            print('Wrong model ID number. Must be 1, 2, 3 or 4.')
            return None
    return model