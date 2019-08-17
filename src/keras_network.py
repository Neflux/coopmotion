from __future__ import print_function

import os
from abc import abstractmethod

from task.square import extract_from_state

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers


class PrintDot(keras.callbacks.Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs):
        print('\rTraining progress: {:2.1f}%'.format(100. * (epoch + 1) / self.epochs), end='')
        if epoch == self.epochs - 1:
            print('\r', end='')


class Net:
    def __init__(self, input_size, output_size, holonomic):
        self.input_size = input_size
        self.output_size = output_size
        self.holonomic = holonomic

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
                           metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])

    def train(self, epochs, train_dataset, test_dataset, batch_size=None):
        x, y = train_dataset
        train_history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2,
                                       verbose=1, callbacks=[self.early_stop])

        x, y = test_dataset
        test_results = self.model.evaluate(x, y, batch_size=batch_size, verbose=0)

        [print(f"\r{name}: {value:.2f}") for value, name in
         zip(test_results, ['mean_squared_error', 'mean_absolute_error', 'mean_squared_error', 'accuracy'])]

        return train_history

    def predict(self, x):
        return self.model.predict(x)

    def controller(self):
        def fake_target_filter(targets):
            return self.h_controller if self.holonomic else self.nh_controller
        return fake_target_filter

    @abstractmethod
    def h_controller(self, state, sensing):
        ...

    @abstractmethod
    def nh_controller(self,state, sensing):
        ...


class CentralizedNet(Net):
    def __init__(self, N, holonomic):
        self.N = N
        input_size = N * 2 if holonomic else N * 3
        super().__init__(input_size, N * 2, holonomic)

    def h_controller(self, state, sensing):
        return self.predict(state.reshape(-1, self.N * 2)).reshape(self.N, 2),

    def nh_controller(self, state, sensing):
        return self.predict(extract_from_state(state).reshape(-1, self.N * 3)).reshape(self.N, 2),



class DistributedNet(Net):
    def __init__(self, run):
        self.N = run.task.N
        input_size = run.sensor.get_input_size(self.N)
        super().__init__(input_size, 2, run.task.holonomic)

    def h_controller(self, state, sensing):
        return self.predict(sensing.reshape(self.N, -1)).reshape(self.N, 2),

    def nh_controller(self, state, sensing):
        ss = sensing[:, :, :2]
        ss = ss.reshape(ss.shape[0], ss.shape[1] * ss.shape[2])
        return self.predict(ss).reshape(-1, 2),

