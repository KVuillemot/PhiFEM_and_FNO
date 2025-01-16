import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.stats import qmc


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def call_G(xy, gamma):
    return np.array([0 * xy[0] * gamma, gamma + 0 * xy[1]])


def call_phi_i(xy, x_0, y_0, lx):
    return -(-(lx**2) + (xy[0] - x_0) ** 2 + (xy[1] - y_0) ** 2)


def call_phi_i_np(x, y, x_0, y_0, lx):
    return -(-(lx**2) + (x - x_0) ** 2 + (y - y_0) ** 2)


def phi_np(x, y, params_holes):
    nb_holes = params_holes.shape[0]
    phi = 1
    considered_holes = list(range(nb_holes))
    for hole in considered_holes:
        x_i, y_i, li = params_holes[hole]
        phi *= call_phi_i_np(x, y, x_i, y_i, li)

    return (-1.0) ** (len(considered_holes) + 1) * phi


def call_phi(xy, param_holes):

    nb_holes = param_holes.shape[0]
    phi = 1.0
    considered_holes = list(range(nb_holes))
    for hole in considered_holes:
        x_i, y_i, li = param_holes[hole]
        phi *= call_phi_i(xy, x_i, y_i, li)
    return (-1.0) ** (len(considered_holes) + 1) * phi


def call_phi_i(xy, x_0, y_0, lx):
    return -(-(lx**2) + (xy[0] - x_0) ** 2 + (xy[1] - y_0) ** 2)


def eval_phi(x, y, x_0, y_0, lx):
    return -(-(lx**2) + (x - x_0) ** 2 + (y - y_0) ** 2)


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def outside_ball(point):
    """
    Check if a given point is inside or outside the circle ((0.5,0.5), 0.45).
    """
    x, y = point
    return -0.48 + np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) > 0.0


def create_FG_numpy(nb_data, nb_vert):
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = XX.flatten()
    YY = YY.flatten()
    XXYY = np.stack([XX, YY])

    gamma_G = np.random.uniform(0.5, 0.9, size=[nb_data, 1])

    nb_holes = np.ones(nb_data) * 5
    # x0_phi = np.random.uniform(0.15, 0.85, size=[5, nb_data])
    # y0_phi = np.random.uniform(0.15, 0.85, size=[5, nb_data])
    # ll_phi = np.random.uniform(0.08, 0.22, size=[5, nb_data])

    Phi = []
    sampler = qmc.LatinHypercube(d=16)
    sample = sampler.random(nb_data)
    low_bounds = [
        0.3,
        0.15,
        0.15,
        0.05,
        0.7,
        0.15,
        0.05,
        0.15,
        0.7,
        0.05,
        0.7,
        0.7,
        0.05,
        0.45,
        0.45,
        0.05,
    ]
    up_bounds = [
        0.9,
        0.3,
        0.3,
        0.1,
        0.85,
        0.3,
        0.1,
        0.3,
        0.85,
        0.1,
        0.85,
        0.85,
        0.1,
        0.55,
        0.55,
        0.085,
    ]
    sample_scaled = qmc.scale(sample, low_bounds, up_bounds)
    gamma_G = sample_scaled[:, :1]

    x0_phi = sample_scaled[:, 1::3].T
    y0_phi = sample_scaled[:, 2::3].T
    ll_phi = sample_scaled[:, 3::3].T

    print(f"{gamma_G.shape=}")
    print(f"{x0_phi.shape=}")
    print(f"{y0_phi.shape=}")
    print(f"{ll_phi.shape=}")

    Phi = []
    for n in range(nb_data):
        xi, yi, li = (
            x0_phi[:, n],
            y0_phi[:, n],
            ll_phi[:, n],
        )
        params_holes = np.concatenate([[xi, yi, li]], axis=1).T
        phi = call_phi(XXYY, params_holes).reshape((nb_vert, nb_vert))
        Phi.append(phi)

    Phi = np.array(Phi).transpose(0, 2, 1)
    print(f"{Phi.shape=}")
    G = call_G(XXYY, gamma_G)
    G = np.reshape(G, [2, nb_data, nb_vert, nb_vert])
    G = G.transpose((1, 0, 3, 2))
    print(f"{G.shape=}")
    params = sample_scaled
    return Phi, G, params


if __name__ == "__main__":
    create_FG_numpy(10, 64)
