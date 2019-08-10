from __future__ import print_function

import os
from abc import abstractmethod

from task.square import efficient_state_extraction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers

import numpy as np
import builtins as __builtin__
import inspect


def print(*args, **kwargs):
    if 'v' in kwargs:
        if kwargs.pop("v") is False:
            return
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    filename_clean = filename.rpartition('/')[-1][:-3].upper()
    try:
        if type(args[0]) is np.ndarray:
            __builtin__.print(f'[{filename_clean}]\n', end='')
        else:
            __builtin__.print(f'[{filename_clean}] ', end='')
    except:
        __builtin__.print(f'[{filename_clean}] ', end='')
    return __builtin__.print(*args, **kwargs)


class PrintDot(keras.callbacks.Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs):
        print('\rTraining progress: {:2.1f}%'.format(100. * (epoch + 1) / self.epochs), end='')
        if epoch == self.epochs - 1:
            print('\r', end='')


class Net:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.tanh, input_shape=(input_size,)),
            keras.layers.Dropout(0.3),
            layers.Dense(64, activation=tf.nn.tanh),
            keras.layers.Dropout(0.3),
            layers.Dense(output_size)
        ])

        self.optimizer = tf.keras.optimizers.RMSprop(0.001)
        self.early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        self.model.compile(loss='mean_squared_error',
                           optimizer=self.optimizer,
                           metrics=['mean_absolute_error', 'mean_squared_error'])

    def fit(self, x, y, epochs=100):
        self.model.fit(x, y, epochs=epochs, shuffle=False, validation_split=0.2, verbose=0, callbacks=[self.early_stop])

    def predict(self, x):
        return self.model.predict(x)

    def controller(self):
        def fake_target_filter(targets):
            return self.f
        return fake_target_filter

    @abstractmethod
    def f(self, state, sensing):
        ...


class CentralizedNet(Net):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def f(self, state, sensing):
        return self.predict(efficient_state_extraction(state).flatten()).reshape(self.input_size // 4, 2),


class DistributedNet(Net):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def f(self, state, sensing):
        return self.predict(sensing.reshape(sensing.shape[0], sensing.shape[1] * sensing.shape[2])).reshape(-1, 2),
