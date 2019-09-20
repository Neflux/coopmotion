import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from config import ICP_ITERATIONS


def extract_xys_from_state(mats):
    return np.vstack([mats[:, 0, 2], mats[:, 1, 2]]).T


def extract_thetas_from_state(mats):
    return np.arctan2(mats[:, 1, 0], mats[:, 0, 0])


def extract_from_state(mats):
    # N x 3 x 3 -> N x 3
    return np.stack([mats[:, 0, 2], mats[:, 1, 2], np.arctan2(mats[:, 1, 0], mats[:, 0, 0])], 1)


def extract_tuple_from_mat(mat):
    # 3 x 3 -> (2, 1) = (xy, theta)
    return [mat[0, 2], mat[1, 2]], np.arctan2(mat[1, 0], mat[0, 0])


def extract_tuple_from_state(mats):
    # N x 3 x 3 -> (N x 2, N x 1) = (xy, theta)
    return np.vstack([mats[:, 0, 2], mats[:, 1, 2]]).T, np.arctan2(mats[:, 1, 0], mats[:, 0, 0])


def extract_from_trace(matss):
    # n x N x 3 x 3 -> n x N x (2+1)
    return np.stack([matss[:, :, 0, 2],
                     matss[:, :, 1, 2],
                     np.arctan2(matss[:, :, 1, 0], matss[:, :, 0, 0])], 2)

def extract_xys_from_trace(matss):
    # n x N x 3 x 3 -> n x N x (2+1)
    return np.stack([matss[:, :, 0, 2], matss[:, :, 1, 2]], 2)


def mktr(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def mkrot(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def mkultra(x, y, theta):
    # return mktr(x,y) @ mkrot(theta)
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def euclidean_distance(start_pose, goal_pose):
    (x1, y1), (x2, y2) = start_pose, goal_pose
    return np.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))


def steering_angle(start_pose, goal_pose):
    (x1, y1), (x2, y2) = start_pose, goal_pose
    return np.arctan2(y2 - y1, x2 - x1)


def angle_difference(angle1, angle2):
    return np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))


def hungarian_matching(positions, targets):
    distances = distance.cdist(positions, targets, 'euclidean')
    _, col_ind = linear_sum_assignment(distances)
    return col_ind


def transform_estimation(positions, targets):
    P = targets.T
    X = positions.T
    muP = np.mean(P, axis=1, keepdims=True)
    muX = np.mean(X, axis=1, keepdims=True)
    Pprime = P - muP
    Xprime = X - muX
    U, _, V = np.linalg.svd(np.dot(Xprime, Pprime.T))
    R = U @ V.T
    t = muX - (R @ muP)
    Paligned = R @ P + t
    Paligned = Paligned.T

    return Paligned

def dagger_remappers(epochs, safe_limsup, steps, x_dilation = 3, y_dilation= 1):
    from scipy.interpolate import interp1d

    mx = interp1d([-x_dilation, x_dilation], [0, epochs])
    my = interp1d([-y_dilation, y_dilation], [steps, safe_limsup-steps])

    x = np.linspace(-x_dilation, x_dilation, epochs)
    y = y_dilation*np.tanh(x)

    x, y = mx(x), my(y)

    return x.astype(int), y.astype(int)


def calculate_side(n, r):
    theta = 360 / n
    theta_in_radians = theta * 3.14 / 180

    return 2 * r * np.sin(theta_in_radians / 2)


def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)


def is_homogenous(x):
    if x.shape[-2:] == (3, 3):
        return True
    else:
        return False


# Generalized for the notebook
def ICP(positions, targets, show=False):
    from task.fancy import plot_task

    if is_homogenous(positions):
        xys = extract_xys_from_state(positions)
    else:
        xys = positions

    plot_task(positions, targets, title=f'Original', show=show)
    for i in range(ICP_ITERATIONS):
        targets = targets[hungarian_matching(xys, targets)]
        plot_task(positions, targets, title=f'Hungarian Assignment Matching#{i + 1}', show=show)
        targets = transform_estimation(xys, targets)
        plot_task(positions, targets, title=f'Trasformation Estimation #{i + 1}', show=show)
    return targets


def hICP(positions, targets):
    for i in range(ICP_ITERATIONS):
        targets = targets[hungarian_matching(positions, targets)]
        targets = transform_estimation(positions, targets)
    return targets


def nhICP(positions, targets):
    xys = extract_xys_from_state(positions)
    for i in range(ICP_ITERATIONS):
        targets = targets[hungarian_matching(xys, targets)]
        targets = transform_estimation(xys, targets)
    return targets