import numpy as np
from scipy.optimize import linear_sum_assignment
from math import atan2
from scipy.spatial import distance

from task.math import *
from . import Task, Run

""" RAW TASK TARGET GENERATOR """


def circular_evenly_spread(N: int, L: float= 1.0, radius=None) -> np.ndarray:
    radius = L / 2 if (radius is None) else radius
    equidistant = np.linspace(0, 2 * np.pi * (N - 1) / N, N)
    return np.c_[radius * np.cos(equidistant), radius * np.sin(equidistant)] + L / 2


def circular_zipf(N: int, L: float = 1.0) -> np.ndarray:
    ds = 1 / np.arange(1, N + 2)
    ds *= L / np.sum(ds)
    ds = np.cumsum(ds)[:-1] * 2 * np.pi
    return np.c_[L / 2 * np.cos(ds), L / 2 * np.sin(ds)] + L / 2


""" CONTROLLERS AND ABSTRACTIONS """


def bang_bang(target_xys: float, max_speed: float = 1.0, epsilon: float = 0.01):
    def c(xys: float):
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
    def c(xys: float):
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


def proportional(target_xys, max_speed: float = 1.0, tau: float = 1.0, fn=lambda x, y: y - x):
    def c(xys):
        return max_speed * np.clip(fn(xys, target_xys) / tau, -1, 1)

    return c


def nh_position_controller(controller, max_speed: float = 1.0):
    def target_filter(targets):
        lcs = [controller(t, max_speed=max_speed, tau=2,
                          fn=lambda x, y: euclidean_distance(x[:2, 2], y)) for t in targets]
        acs = [controller(t, max_speed=max_speed, tau=0.3,
                          fn=lambda x, y: angle_difference(steering_angle(x[:2, 2], y), np.arctan2(x[1, 0], x[0, 0])))
               for t in targets]

        def mc(pos, ss):
            return [(lc(p), ac(p)) for lc, ac, p in zip(lcs, acs, pos)],

        return mc

    return target_filter


def h_position_controller(controller, max_speed: float = 1.0):
    def target_filter(target_xs):
        cs = [controller(target_x, max_speed) for target_x in target_xs]

        def mc(xs, ss):
            return [c(x) for c, x in zip(cs, xs)],

        return mc

    return target_filter


""" DIFFERENT SENSING MODES """


def fixed_neighbors(xys):
    out = np.array([[j for j in range(len(xys)) if j != i] for i in range(len(xys))])
    return out


def sorted_neighbors(xys):
    out = np.argsort(distance.cdist(xys, xys), axis=1)[:, 1:]
    return out


def smart_subset(subset):
    if subset is None:
        return lambda N: N - 1
    else:
        return lambda N: subset


def range_step(range):
    if range is None:
        return lambda deltas: deltas
    else:
        def f(deltas):
            mask = np.linalg.norm(deltas, axis=2) < range
            deltas[~mask] = 0
            return np.dstack([deltas, mask.astype(int)])

        return f


def sense(sorted=False, subset=None, range=None):
    assert sorted or (not sorted and subset is None), \
        "You cannot have a subset of fixed-order neighbors, it does not make 'sense'"

    neighbor_ids = sorted_neighbors if sorted else fixed_neighbors
    get_subset = smart_subset(subset)
    eventual_range_step = range_step(range)

    def sense(holonomic):
        def h_sense(xys):
            N = len(xys)
            ss = get_subset(N)
            sensing_ids = neighbor_ids(xys)[:, :ss]
            protagonist = np.moveaxis(np.repeat(xys, ss, axis=1).reshape(N, 2, ss), 1, 2)
            deltas_xys = xys[sensing_ids] - protagonist
            final = eventual_range_step(deltas_xys)
            return final

        def nh_sense(positions):  # N x 3 x 3
            N = len(positions)
            ss = get_subset(N)
            # Extracted positions N x 2
            xys = extract_xys_from_state(positions)
            # Ids of the other robots for each robot N x subset
            sensing_ids = neighbor_ids(xys)[:, :ss]
            # Duplicated absolute positions to perform the subtraction trick N x subset x 2
            protagonist = np.moveaxis(np.repeat(xys, ss, axis=1).reshape(N, 2, ss), 1, 2)
            # Subtraction trick: actually obtaining the deltas N x subset x 2
            deltas_xys = xys[sensing_ids] - protagonist
            # Trailing ones for correct matrix multiplications N x subset x 3
            deltas_xys_one = np.dstack([deltas_xys, np.ones((N, ss))])
            # Final step: matmul N x subset x 3
            deltas_transform = np.array([[neighbor_delta @ pos for neighbor_delta in neighbors]
                                         for pos, neighbors in zip(positions, deltas_xys_one)])
            deltas_transform_clean = deltas_transform[:, :, :2]

            final = eventual_range_step(deltas_transform_clean)
            return final

        sense = h_sense if holonomic else nh_sense
        sense.sorted = sorted
        sense.subset = subset
        sense.range = range
        sense.get_input_size = lambda N: get_subset(N) * 2 if (range is None) else get_subset(N) * 3
        return sense

    return sense


""" TASK CLASSES AND ABSTRACTIONS """


def random_robot_poses(L, N):
    return np.array([mkultra(xy[0], xy[1], theta) for (xy, theta) in zip(np.random.uniform(0, L, size=(N, 2)),
                                                                         np.random.uniform(0, 2 * np.pi, size=N))])


class StaticPositionTask(Task):
    def __init__(self, target_xys, holonomic: bool = None, L: float = 1):
        if holonomic is None:
            print("Assuming holonomic robots, otherwise change the task parameter")
            holonomic = True
        self.L = L
        self.target_xys = np.array(target_xys)
        self.N = len(self.target_xys)
        self.holonomic = holonomic
        self.initial = self.h if self.holonomic else self.nh

    def h(self):
        self.initial_state = np.random.uniform(0, self.L, size=(self.N, 2))
        self.initial_state = self.initial_state[hungarian_matching(self.target_xys, self.initial_state)]
        return self.initial_state, self.target_xys

    def nh(self):
        self.initial_state = random_robot_poses(self.L, self.N)
        self.initial_state = self.initial_state[
            hungarian_matching(self.target_xys, extract_xys_from_state(self.initial_state))]
        return self.initial_state, self.target_xys

    def update(self):
        return lambda c, sc: c

    def distance(self):
        if self.holonomic:
            return lambda xys: np.max(np.abs(self.target_xys - np.array(xys)))
        else:
            return lambda pos: np.max(np.abs(self.target_xys - np.array(extract_xys_from_state(pos))))


class AdaptivePositionTask(Task):
    def __init__(self, target_xys, holonomic: bool = None, L: float = 1):
        if holonomic is None:
            print("Assuming holonomic robots, otherwise change the task parameter")
            holonomic = True
        self.L = L
        self.target_xys = np.array(target_xys)
        self.N = len(self.target_xys)
        self.holonomic = holonomic
        self.initial = self.h if self.holonomic else self.nh

    def h(self):
        self.initial_state = np.random.uniform(0, self.L, size=(self.N, 2))
        self.target_xys = hICP(self.initial_state, self.target_xys)
        return self.initial_state, self.target_xys

    def nh(self):
        self.initial_state = random_robot_poses(self.L, self.N)
        self.target_xys = nhICP(self.initial_state, self.target_xys)
        return self.initial_state, self.target_xys

    def update(self):
        if self.holonomic:
            return lambda c, sc: sc(hICP(self.initial_state, self.target_xys))
        else:
            return lambda c, sc: sc(nhICP(self.initial_state, self.target_xys))

    def distance(self):
        if self.holonomic:
            return lambda xys: np.max(np.abs(self.target_xys - np.array(xys)))
        else:
            return lambda pos: np.max(np.abs(self.target_xys - np.array(extract_xys_from_state(pos))))


def static_zipf_task(N: int, holonomic=None):
    return StaticPositionTask(circular_zipf(N, L=1), holonomic, L=1)


def adaptive_zipf_task(N: int, holonomic=None):
    return AdaptivePositionTask(circular_zipf(N, L=1), holonomic, L=1)


def static_evenly_spread_task(N: int, holonomic=None):
    return StaticPositionTask(circular_evenly_spread(N, L=1), holonomic, L=1)


def adaptive_evenly_spread_task(N: int, holonomic=None, radius=None):
    return AdaptivePositionTask(circular_evenly_spread(N, L=1, radius=radius), holonomic, L=1)


def h_dynamic(dt: float = 0.1, max_speed: float = 1):
    def update(xs, cs):
        return np.array(xs) + dt * np.clip(np.array(cs), -max_speed, max_speed)

    return update


def nh_dynamic(dt=0.1):
    def update(state, cs):
        return np.array([pose @ mkultra(x=dt * vel[0], y=0, theta=dt * vel[1])
                         for pose, vel in zip(state, cs)])

    return update


class SquareRun(Run):
    def __init__(self, task: Task, controller, sensor, dt: float = 0.1):
        if task.holonomic:
            controller = h_position_controller(controller)
        else:
            controller = nh_position_controller(controller)

        super(SquareRun, self).__init__(task=task, dt=dt, sensor=sensor(task.holonomic), controller=controller)
