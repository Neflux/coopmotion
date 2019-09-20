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
import numpy as np

class Net:
    def __init__(self, input_size, output_size, holonomic, patience=20, comm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.holonomic = holonomic
        self.comm = comm

        self.model = keras.Sequential([
            layers.Dense(128, activation=tf.nn.tanh, input_shape=(input_size,)),
            layers.Dense(128, activation=tf.nn.tanh),
            layers.Dense(output_size)
        ])

        self.optimizer = tf.keras.optimizers.RMSprop(0.001)
        # self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        self.model.compile(loss='mean_squared_error',
                           optimizer=self.optimizer,
                           metrics=['mean_absolute_error'])

    def train(self, epochs, train_dataset, valid_dataset, batch_size=None, training_loss=None, testing_loss=None, verbose=1):
        if training_loss is None:
            training_loss = []
        if testing_loss is None:
            testing_loss = []

        x, y = train_dataset
        train_history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=True,
                                       validation_data=valid_dataset, verbose=verbose, callbacks=[self.early_stop])
        training_loss.append(np.mean(train_history.history['loss']))
        testing_loss.append(np.mean(train_history.history['val_loss']))
        # x, y = test_dataset
        # test_results = self.model.evaluate(x, y, batch_size=batch_size, verbose=0)
        #
        # [print(f"\r{name}: {value:.2f}") for value, name in
        #  zip(test_results, ['mean_squared_error', 'mean_absolute_error'])]

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
    def nh_controller(self, state, sensing):
        ...


class CentralizedNet(Net):
    def __init__(self, N, holonomic, patience=20):
        self.N = N
        input_size = N * 2 if holonomic else N * 3
        super().__init__(input_size, N * 2, holonomic, patience=patience)

    def h_controller(self, state, sensing):
        return self.predict(state.reshape(-1, self.N * 2)).reshape(self.N, 2),

    def nh_controller(self, state, sensing):
        return self.predict(extract_from_state(state).reshape(-1, self.N * 3)).reshape(self.N, 2),


class DistributedNet(Net):
    def __init__(self, run, patience=20):
        self.N = run.task.N
        input_size = run.sensor.get_input_size(self.N)
        print(input_size)
        super().__init__(input_size, 2, run.task.holonomic, patience=patience)

    def h_controller(self, state, sensing):
        return self.predict(sensing.reshape(self.N, -1)).reshape(self.N, 2),

    def nh_controller(self, state, sensing):
        ss = sensing[:, :, :2]
        ss = ss.reshape(ss.shape[0], ss.shape[1] * ss.shape[2])
        return self.predict(ss).reshape(-1, 2),

#
# class ComNet():
#     def __init__(self, N: int, holonomic=True, broadcast: int = 1, batch_size: int = 1, unroll_window: int = 2):
#         super(ComNet, self).__init__()
#         self.N = N
#         self.broadcast = broadcast
#         self.batch_size = batch_size
#         self.unroll_window = unroll_window
#         self.n_closest = np.array([[j for j in range(N) if j != i] for i in range(N)])
#         self.single_net = Net((N - 1) * (3 + broadcast), 2 + broadcast, holonomic=holonomic, comm=True)
#
#     def input_global_comm(self, ss, comm, i):
#         rel_comm = comm[:, self.n_closest[i]]
#         return np.concatenate([ss[i].flatten(), rel_comm.flatten()], 0)
#
#     def step(self, xs, comm):
#         cs = []
#         for i in list(range(self.N)):
#             output = self.single_net.predict(self.input_global_comm(xs, comm, i))
#             comm[:, i] = output[2:]
#             cs.append(output[:2])
#         return np.concatenate(cs, 0)
#
#     def init_comm(self):
#         return np.zeros(shape=(self.broadcast, self.N))
#
#     def forward(self, runs):
#         rs = []
#         for run in runs:
#             comm = self.init_comm()
#             controls = []
#             for xs in run:
#                 controls.append(self.step(xs, comm))
#             rs.append(np.stack(controls))
#         return np.stack(rs).reshape(len(rs), -1, self.N, 2)
#
#     def h_controller(self):
#         comm = self.init_comm()
#
#         def c(state, sensing):
#             control = self.step(sensing, comm)
#             return control.reshape(self.N, 2), comm.copy()
#
#         return c
