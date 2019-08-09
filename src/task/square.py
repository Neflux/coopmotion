import numpy as np
from scipy.optimize import linear_sum_assignment
from math import atan2
from scipy.spatial import distance

from . import Task, Run

RADIUS = lambda L: L / 2.

""" ROTOTRANSLATION UTILS """


def extract_xy(mat):
    return mat[0, 2], mat[1, 2]


def extract_xys(mats):
    return np.array([extract_xy(r) for r in mats])


# TODO: check this is correct
def extract_theta(mat):
    return np.arctan2(mat[1, 0], mat[0, 0])


def extract_thetas(mats):
    return np.array([extract_theta(r) for r in mats])


def extract_xytheta(mat):
    return extract_xy(mat), extract_theta(mat)


def extract_xythetas(mats):
    return np.array([extract_xy(r) for r in mats]), np.array([extract_theta(r) for r in mats])

def efficient_state_extraction(mats):
    return np.stack([mats[:, 0, 2],
                      mats[:, 1, 2],
                      np.arctan2(mats[:, 1, 0], mats[:, 0, 0])],1)

def efficient_trace_extraction(matss):
    return np.stack([matss[:, :, 0, 2],
                      matss[:, :, 1, 2],
                      np.arctan2(matss[:, :, 1, 0], matss[:, :, 0, 0])],2)


def mktr(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def mkrot(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def mkultra(x, y, theta):
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def mk_from_vel(lin_vel, ang_vel):
    return np.array([[np.cos(ang_vel), -np.sin(ang_vel), lin_vel],
                     [np.sin(ang_vel), np.cos(ang_vel), 0],
                     [0, 0, 1]])


def random_robot_poses(L, N):
    return np.array([mkultra(xy[0], xy[1], theta) for (xy, theta) in zip(np.random.uniform(0, L, size=(N, 2)),
                                                                         np.random.uniform(0, 2 * np.pi, size=N))])


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


def euclidean_distance(start_pose, goal_pose):
    (x1, y1), (x2, y2) = start_pose, goal_pose
    return np.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))


def steering_angle(start_pose, goal_pose):
    (x1, y1), (x2, y2) = start_pose, goal_pose
    return atan2(y2 - y1, x2 - x1)


def angle_difference(angle1, angle2):
    return np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))



def proportional(max_speed: float = 1.0, kp: float = 1.0):
    def c(_input: float) -> float:
        return max_speed * np.clip(kp * (_input), -1, 1)

    return c


def position_controller(controller):
    def target_filter(targets):
        lcs = [controller(max_speed=0.5, kp=1.) for _ in targets]
        acs = [controller(max_speed=np.pi, kp=1.) for _ in targets]

        def mc(xs, ss):
            xys, thetas = extract_xythetas(xs)
            return np.array([(lc(euclidean_distance(xy, target)),
                              ac(angle_difference(steering_angle(xy, target), theta)))
                             for lc, ac, xy, theta, target in zip(lcs, acs, xys, thetas, targets)]),

        return mc

    return target_filter


""" DIFFERENT SENSING MODES """


def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)


#TODO: what about sensing also the neighbors orientation?
def sense_all():
    def sense(positions):
        xys = extract_xys(positions)
        N = len(xys)

        n_closest = skip_diag_strided(np.tile(np.arange(N), (N, 1)))
        result = xys[n_closest]
        # WORK IN PROGRESS
        tmp = positions[idx] @ np.hstack([result[idx, idy], 1]).T
        result[idx, idy] = tmp[:2].T

        return np.dstack([result, in_range])

    sense.get_input_size = lambda n: (n - 1) * 2
    sense.get_params = lambda: ''
    return sense


#TODO: what about sensing also the neighbors orientation?
def sense_in_range(T=1):
    def sense(positions):
        xys = extract_xys(positions)
        N = len(xys)
        d = distance.cdist(xys, xys)
        d[d > T] = np.inf

        n_closest = skip_diag_strided(np.tile(np.arange(N), (N, 1)))
        result = xys[n_closest]

        in_range = np.empty((N, N - 1))
        for idx, row in enumerate(n_closest):
            for idy, neighbor in enumerate(row):
                if d[idx, neighbor] == np.inf:
                    result[idx, idy] = 0.
                    in_range[idx, idy] = 0.
                else:
                    # The sensing is now relative to the robot frame
                    tmp = positions[idx] @ np.hstack([result[idx, idy], 1]).T
                    result[idx, idy] = tmp[:2].T
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

    def initialize(self):
        self.initial_poses = random_robot_poses(self.L, self.N)
        distances = distance.cdist(self.target_xys, extract_xys(self.initial_poses), 'euclidean')
        row_ind, col_ind = linear_sum_assignment(distances)
        self.initial_poses = self.initial_poses[col_ind, :]
        return self.initial_poses

    def update_targets(self, c, sc):
        return c

    def distance(self, positions):
        return np.max(np.abs(self.target_xys - np.array(extract_xys(positions))))

    @property
    def targets(self):
        return self.target_xys


class SmartStaticPositionTask(Task):
    def __init__(self, target_xys, L: float = 1):
        self.L = L
        self.target_xys = np.array(target_xys)
        self.N = len(self.target_xys)

    def initialize(self):
        self.initial_poses = random_robot_poses(self.L, self.N)
        self.target_xys = hungarian_ICP(self.initial_poses, self.target_xys, 4)

        return self.initial_poses

    def update_targets(self, c, sc):
        return c

    def distance(self, positions):
        return np.max(np.abs(self.target_xys - np.array(extract_xys(positions))))

    @property
    def targets(self):
        return self.target_xys


class DynamicPositionTask(Task):
    def __init__(self, target_xys, L: float = 1):
        self.L = L
        self.target_xys = np.array(target_xys)
        self.N = len(self.target_xys)

    def initialize(self):
        self.initial_poses = random_robot_poses(self.L, self.N)

        self.target_xys = hungarian_ICP(self.initial_poses, self.target_xys, 4)

        return self.initial_poses

    def update_targets(self, c, sc):
        self.target_xys = hungarian_ICP(self.initial_poses, self.target_xys, 4)
        return sc(self.target_xys)

    def distance(self, positions):
        return np.max(np.abs(self.target_xys - np.array(extract_xys(positions))))

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


def hungarian_ICP(positions, targets, n, show=False):
    xys, angles = extract_xythetas(positions)
    if show:
        from task.fancy import plot_task
        plot_task(positions, targets, title=f'Original')
    # targets = hungarian(xys,targets)
    for i in range(n):
        targets = ClosestPoints(xys, targets)
        if show:
            plot_task(positions, targets, title=f'ClosestPoints #{i + 1}')
    targets = hungarian(xys, targets)
    if show:
        plot_task(positions, targets, title=f'Hungarian')
    return targets


def hungarian(positions, targets):
    distances = distance.cdist(positions, targets, 'euclidean')
    _, col_ind = linear_sum_assignment(distances)
    return targets[col_ind, :]


def dynamic(dt=0.1):
    def update(pos_state, cs):
        return np.array([pose @ mk_from_vel(dt * vel[0], dt * vel[1]) for pose, vel in zip(pos_state, cs)])

    return update


class SegmentRun(Run):
    def __init__(self, task: Task, controller, sensor, dt: float = 0.1):
        super(SegmentRun, self).__init__(
            task=task, dt=dt, sensor=sensor, dynamic=dynamic(dt),
            controller=position_controller(controller))
