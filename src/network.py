from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm

def train_net(epochs: int,
              train_dataset: data.TensorDataset,
              test_dataset: data.TensorDataset,
              net: torch.nn.Module,
              batch_size: int = 100,
              learning_rate: float = 0.001,
              training_loss: Optional[List[float]] = None,
              testing_loss: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
    dl = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    tdl = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    criterion = torch.nn.MSELoss()
    if training_loss is None:
        training_loss = []
    if testing_loss is None:
        testing_loss = []
    for _ in tqdm(range(epochs)):
        epoch_loss = 0.0
        for n, (inputs, labels) in enumerate(dl):
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            epoch_loss += float(loss)
            optimizer.step()
            optimizer.zero_grad()
        training_loss.append(epoch_loss)
        with torch.no_grad():
            testing_loss.append(
                sum([float(criterion(net(inputs), labels)) for inputs, labels in tdl]))

    # import matplotlib.pyplot as plt
    # plt.title('Loss')
    # plt.semilogy(training_loss, label='training')
    # plt.semilogy(testing_loss, label='testing')
    # plt.xlabel('epoch')
    # plt.legend()
    # plt.show()

    return training_loss, testing_loss


class CentralizedNet(torch.nn.Module):

    def __init__(self, N: int):
        super(CentralizedNet, self).__init__()
        self.l1 = torch.nn.Linear(N * 2, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, N * 2)
        self.N = N

        self.drop_layer = torch.nn.Dropout(0.3)

    def forward(self, xys):
        xys = self.drop_layer(torch.tanh(self.l1(xys)))
        xys = self.drop_layer(torch.tanh(self.l2(xys)))
        return self.l3(xys)

    def controller(self):
        def f(state, sensing):
            with torch.no_grad():
                return self(torch.FloatTensor(state).flatten()).numpy().reshape(self.N, 2),

        return f


class DistributedNet(torch.nn.Module):

    def __init__(self, input_size: int):
        super(DistributedNet, self).__init__()
        # every robot
        self.l1 = torch.nn.Linear(input_size, 32)
        self.l2 = torch.nn.Linear(32, 32)
        self.l3 = torch.nn.Linear(32, 2)
        self.input_size = input_size

    def forward(self, xs):
        ys = torch.tanh(self.l1(xs))
        ys = torch.tanh(self.l2(ys))
        return self.l3(ys)

    def controller(self):
        def f(state, sensing):
            with torch.no_grad():
                return self(torch.FloatTensor(sensing.reshape(
                    sensing.shape[0], sensing.shape[1] * sensing.shape[2]))).numpy().reshape(-1, 2),

        return f

class EnhancedDistributedNet(torch.nn.Module):

    def __init__(self, input_size: int):
        super(EnhancedDistributedNet, self).__init__()
        # every robot
        self.l1 = torch.nn.Linear(input_size, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, 2)
        self.input_size = input_size

        self.drop_layer = torch.nn.Dropout(0.3)


    def forward(self, xs):
        ys = self.drop_layer(torch.tanh(self.l1(xs)))
        ys = self.drop_layer(torch.tanh(self.l2(ys)))
        return self.l3(ys)

    def controller(self):
        def f(state, sensing):
            with torch.no_grad():
                return self(torch.FloatTensor(sensing.reshape(
                    sensing.shape[0], sensing.shape[1] * sensing.shape[2]))).numpy().reshape(-1, 2),

        return f