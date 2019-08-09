from enum import Enum
from random import shuffle
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from task.square import skip_diag_strided


class SNet(nn.Module):
    def __init__(self, N: int, broadcast: int):
        super(SNet, self).__init__()
        # dx dy is_in_range comm * (4-1)
        self.l1 = torch.nn.Linear((N - 1) * (3 + broadcast), 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, 2 + broadcast)
        self.input_size = (N - 1) * 4

        self.drop_layer = torch.nn.Dropout(0.3)

    def forward(self, xs):
        ys = self.drop_layer(torch.tanh(self.l1(xs)))
        ys = self.drop_layer(torch.tanh(self.l2(ys)))
        return self.l3(ys)


class ComNet(nn.Module):
    def __init__(self, N: int, broadcast: int = 1, batch_size: int = 1,
                 unroll_window: int = 2, module: nn.Module = SNet):
        super(ComNet, self).__init__()
        self.unroll_window = unroll_window
        self.batch_size = batch_size
        self.single_net = module(N, broadcast)
        self.N = N
        self.broadcast = broadcast

        self.n_closest = skip_diag_strided(np.tile(np.arange(N), (N, 1)))

    def init_comm(self):
        return np.random.uniform(1, 2, (self.broadcast, self.N))
        # return np.zeros(shape=(self.broadcast, self.N))
        # return Variable(torch.Tensor([0] * N))

    def input_local_comm(self, ss, comm, i):
        # TODO: detach from GPU to do advanced indexing with np? really?
        #  pytorch sux, jerome help
        mask = ss[i, :, 2].detach().numpy().astype(bool)
        result = np.zeros((self.broadcast, self.N - 1))
        rel_comm = comm[:, self.n_closest[i]]
        result[:, mask] = rel_comm[:, mask]
        return torch.cat([ss[i].flatten(), torch.tensor(result.flatten()).float()], 0)

    def input_global_comm(self, ss, comm, i):
        rel_comm = comm[:, self.n_closest[i]]
        return torch.cat([ss[i].flatten(), torch.tensor(rel_comm.flatten()).float()], 0)

    def step(self, xs, comm):
        cs = []
        for i in list(range(self.N)):
            output = self.single_net(self.input_global_comm(xs, comm, i))
            comm[:, i] = output[2:].detach().numpy()
            cs.append(output[:2])
        return torch.cat(cs, 0)

    def forward(self, runs):
        rs = []
        for run in runs:
            comm = self.init_comm()
            controls = []
            for xs in run:
                controls.append(self.step(xs, comm))
            rs.append(torch.stack(controls))
            # sometimes the unroll window is less that x, hence -1
        return torch.stack(rs).reshape(len(rs), -1, self.N, 2)

    def controller(self):
        def fake_target_filter(targets):
            comm = self.init_comm()

            def f(state, sensing):
                with torch.no_grad():
                    sensing = torch.FloatTensor(sensing)
                    control = self.step(sensing, comm).numpy()
                    return control.reshape(self.N, 2), comm.copy()

            return f
        return fake_target_filter
