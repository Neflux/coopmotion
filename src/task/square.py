from typing import Sequence, Tuple, List, Callable, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from scipy.spatial import distance

from . import Task, Run

RADIUS = lambda L: L / 2.


""" RAW TASK TARGET GENERATOR """

def circular_evenly_spread(N: int, L: float = 1.0) -> np.ndarray:
    radius = RADIUS(L)
    equidistant = np.linspace(0, 2 * np.pi * (N - 1) / N, N)
    return np.c_[radius * np.cos(equidistant), radius * np.sin(equidistant)] + L / 2


def circular_zipf(N: int, L: float = 1.0) -> np.ndarray:
    ds = 1 / np.arange(1, N + 2)
    ds *= L / np.sum(ds)
    ds = np.cumsum(ds)[:-1] * 2 * np.pi
    return np.c_[L / 2 * np.cos(ds), L / 2 * np.sin(ds)] + L / 2

""" CONTROLLERS AND ABSTRACTIONS """

def proportional(target_xys: float, max_speed: float = 1.0, tau: float = 1.0):
    def c(xys: float) -> float:
        return max_speed * np.clip((target_xys - xys) / tau, -1, 1)

    return c


def bang_bang(target_xys: float, max_speed: float = 1.0, epsilon: float = 0.01):
    def c(xys: float) -> Sequence[float]:
        result = []
        for xy, target_xy in zip(xys, target_xys):
            if abs(xy - target_xy) < epsilon:
                result.append(0)
            elif xy < target_xy:
                result.append(max_speed)
            else:
                result.append(-max_speed)
        return result

    return c


def mixed(target_xys: float, max_speed: float = 1.0, epsilon: float = 0.01):
    def c(xys: float) -> Sequence[float]:
        result = []
        for xy, target_xy in zip(xys, target_xys):
            if abs(xy - target_xy) < epsilon:
                result.append(max_speed * ((target_xy - xy + epsilon) / epsilon - 1))
            elif xy < target_xy:
                result.append(max_speed)
            else:
                result.append(-max_speed)
        return result

    return c


def position_controller(controller, max_speed: float = 1.0):
    def target_filter(target_xs: Sequence[float]):
        cs = [controller(target_x, max_speed) for target_x in target_xs]

        def mc(xs, ss):
            return [c(x) for c, x in zip(cs, xs)],

        return mc

    return target_filter


""" DIFFERENT SENSING MODES """

def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)


def sense_all_fixed():
    def sense(xys):
        N = len(xys)
        protagonist = np.repeat(xys, N - 1, axis=1).reshape(-1, N - 1, 2)
        result = protagonist - xys[skip_diag_strided(np.tile(np.arange(N), (N, 1)))]

        return result

    sense.get_input_size = lambda n: (n - 1) * 2
    sense.get_params = lambda: ''
    return sense


def sense_all_sorted():
    def sense(xys):
        N = len(xys)
        d = distance.cdist(xys, xys)
        n_closest = np.argsort(d, axis=1)[:, 1:]
        protagonist = np.repeat(xys, N - 1, axis=1).reshape(-1, N - 1, 2)
        result = protagonist - xys[n_closest]

        return result

    sense.get_input_size = lambda n: (n - 1) * 2
    sense.get_params = lambda: ''
    return sense


def sense_closest_subset(dnn):
    assert type(dnn) is int

    def sense(xys):
        N = len(xys)
        d = distance.cdist(xys, xys)
        n_closest = np.argsort(d, axis=1)[:, 1:dnn + 1]
        protagonist = np.repeat(xys, dnn, axis=1).reshape(-1, dnn, 2)
        result = protagonist - xys[n_closest]

        return result

    sense.get_input_size = lambda n: dnn * 2
    sense.get_params = lambda: dnn
    return sense


def sense_sorted_in_range(T=1):
    def sense(xys):
        N = len(xys)
        d = distance.cdist(xys, xys)
        d[d > T] = np.inf

        n_closest = np.argsort(d, axis=1)[:, 1:]
        protagonist = np.repeat(xys, N - 1, axis=1).reshape(-1, N - 1, 2)
        result = protagonist - xys[n_closest]

        in_range = np.empty((N, N - 1))
        for idx, row in enumerate(n_closest):
            for idy, neighbor in enumerate(row):
                if d[idx, neighbor] == np.inf:
                    result[idx, idy] = 0.
                    in_range[idx, idy] = 0.
                else:
                    in_range[idx, idy] = 1.

        return np.dstack([result, in_range])

    sense.get_input_size = lambda n: (n - 1) * 3
    sense.get_params = lambda: T
    return sense


def sense_fixed_in_range(T=1):
    def sense(xys):
        N = len(xys)
        d = distance.cdist(xys, xys)
        d[d > T] = np.inf

        n_closest = skip_diag_strided(np.tile(np.arange(N), (N, 1)))
        protagonist = np.repeat(xys, N - 1, axis=1).reshape(-1, N - 1, 2)
        result = protagonist - xys[n_closest]

        in_range = np.empty((N, N - 1))
        for idx, row in enumerate(n_closest):
            for idy, neighbor in enumerate(row):
                if d[idx, neighbor] == np.inf:
                    result[idx, idy] = 0.
                    in_range[idx, idy] = 0.
                else:
                    in_range[idx, idy] = 1.

        return np.dstack([result, in_range])

    sense.get_input_size = lambda n: (n - 1) * 3
    sense.get_params = lambda: T
    return sense


""" TASK CLASSES AND ABSTRACTIONS """

class StaticPositionTask(Task):
    def __init__(self, target_xys, L: float = 1):
        self.L = L
        self.target_xys = np.array(target_xys)
        self.N = len(self.target_xys)

    def initial(self):
        initial_config = np.random.uniform(0, self.L, size=(self.N, 2))
        initial_config = hungarian_assignment(self.target_xys, initial_config)
        # self.target_xys = hungarian_assignment(initial_config, self.target_xys)
        return initial_config

    def distance(self, xys: Sequence[float]) -> float:
        return np.max(np.abs(self.target_xys - np.array(xys)))

    @property
    def targets(self):
        return self.target_xys


class SmartStaticPositionTask(Task):
    def __init__(self, target_xys, L: float = 1):
        self.target_xys = np.array(target_xys)
        self.N = len(self.target_xys)
        self.L = L

    def initial(self):
        initial_config = np.random.uniform(0, self.L, size=(self.N, 2))
        self.target_xys = ICP_hungarian(initial_config, self.target_xys, 4)
        return initial_config

    def distance(self, xys) -> float:
        return np.max(np.abs(self.target_xys - np.array(xys)))

    @property
    def targets(self):
        return self.target_xys


class DynamicPositionTask(Task):
    def __init__(self, target_xys, L: float = 1):
        self.target_xys = np.array(target_xys)
        self.N = len(self.target_xys)
        self.L = L

    def initial(self):
        initial_config = np.random.uniform(0, self.L, size=(self.N, 2))
        self.target_xys = ICP_hungarian(initial_config, self.target_xys, 4)
        return initial_config

    def distance(self, xys) -> float:
        # TODO: careful
        self.target_xys = ICP_hungarian(xys, self.target_xys, 4)
        return np.max(np.abs(self.target_xys - np.array(xys)))

    @property
    def targets(self):
        return self.target_xys


def static_zipf_task(N: int):
    return StaticPositionTask(circular_zipf(N, L=1), L=1)

def smart_static_zipf_task(N: int):
    return SmartStaticPositionTask(circular_zipf(N, L=1), L=1)

def static_evenly_spread_task(N: int):
    return StaticPositionTask(circular_evenly_spread(N, L=1), L=1)

def smart_static_evenly_spread_task(N: int):
    return SmartStaticPositionTask(circular_evenly_spread(N, L=1), L=1)


def dynamic_evenly_spread_task(N: int):
    return DynamicPositionTask(circular_evenly_spread(N, L=1), L=1)

""" DIFFERENT SENSING MODES """


def ClosestPoints(positions, targets):
    P = targets.T
    X = positions.T
    muP = np.mean(P, axis=1, keepdims=True)
    muX = np.mean(X, axis=1, keepdims=True)
    Pprime = P - muP
    Xprime = X - muX
    U, _, V = np.linalg.svd(np.dot(Xprime, Pprime.T))
    R = U @ V.T
    t = muX - (R @ muP)

    assert (np.allclose(np.linalg.inv(R), R.T))  # R^-1 == R^T
    assert (np.isclose(np.dot(R[0, :], R[1, :]), 0))  # rows should be orthogonal
    assert (np.isclose(np.linalg.det(R), 1) or np.isclose(np.linalg.det(R), -1))  # determinant should be +-1

    Paligned = R @ P + t
    Paligned = Paligned.T

    return Paligned


def ICP_hungarian(positions, targets, n, show=False):
    if show:
        from task.fancy import plot_task
    for i in range(n):
        targets = ClosestPoints(positions, targets)
        if show:
            plot_task(positions, targets, title=f'ClosestPoints #{i + 1}')
    targets = hungarian_assignment(positions, targets)
    if show:
        plot_task(positions, targets, title=f'Hungarian')
    return targets


def hungarian_assignment(positions, targets):
    distances = distance.cdist(positions, targets, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(distances)
    return targets[col_ind, :]


def dynamic(dt: float = 0.1, max_speed: float = 1):
    def update(xs: Sequence[float], cs: Sequence[float]) -> Sequence[float]:
        return np.array(xs) + dt * np.clip(np.array(cs), -max_speed, max_speed)

    return update


class SegmentRun(Run):
    def __init__(self, task: Task, controller, sensor, dt: float = 0.1):
        super(SegmentRun, self).__init__(
            task=task, dt=dt, sensor=sensor, dynamic=dynamic(dt),
            controller=position_controller(controller))
