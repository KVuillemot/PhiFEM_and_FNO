import numpy as np
import matplotlib.pyplot as plt
import random
import os

import torch


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def call_F(xy, mu0, mu1, sigma_x, sigma_y, amplitude):
    return amplitude * np.exp(
        -(
            ((xy[0] - mu0) ** 2 / (2.0 * sigma_x**2))
            + ((xy[1] - mu1) ** 2 / (2.0 * sigma_y**2))
        )
    )


def call_G(xy, alpha, beta):
    return (
        alpha * ((xy[0] - 0.5) ** 2 - (xy[1] - 0.5) ** 2) * np.cos(np.pi * beta * xy[1])
    )


def call_phi(xy, x_0, y_0, lx, ly, theta):
    return (
        -1.0
        + ((xy[0] - x_0) * np.cos(theta) + (xy[1] - y_0) * np.sin(theta)) ** 2 / lx**2
        + ((xy[0] - x_0) * np.sin(theta) - (xy[1] - y_0) * np.cos(theta)) ** 2 / ly**2
    )


def eval_phi(x, y, x_0, y_0, lx, ly, theta):
    return (
        -1.0
        + ((x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta)) ** 2 / lx**2
        + ((x - x_0) * np.sin(theta) - (y - y_0) * np.cos(theta)) ** 2 / ly**2
    )


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
    return -0.42 + np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) > 0.0


def create_parameters(nb_data, nb_vert):
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = XX.flatten()
    YY = YY.flatten()
    XXYY = np.stack([XX, YY])

    mu0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma_x = np.random.uniform(0.15, 0.45, size=[nb_data, 1])
    sigma_y = np.random.uniform(0.15, 0.45, size=[nb_data, 1])
    amplitude = np.random.uniform(20, 30, size=[nb_data, 1]) * np.random.choice(
        [-1, 1], size=[nb_data, 1]
    )
    alpha = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])
    beta = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])

    x_0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    y_0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    lx = np.random.uniform(0.2, 0.45, size=[nb_data, 1])
    ly = np.random.uniform(0.2, 0.45, size=[nb_data, 1])
    theta = np.random.uniform(0.0, np.pi, size=[nb_data, 1])
    for n in range(nb_data):
        xx_0, yy_0, llx, lly = x_0[n][0], y_0[n][0], lx[n][0], ly[n][0]
        xx0_llxp = rotate([xx_0, yy_0], [xx_0 + llx, yy_0], theta[n])
        xx0_llxm = rotate([xx_0, yy_0], [xx_0 - llx, yy_0], theta[n])
        yy0_llyp = rotate([xx_0, yy_0], [xx_0, yy_0 + lly], theta[n])
        yy0_llym = rotate([xx_0, yy_0], [xx_0, yy_0 - lly], theta[n])
        while (
            (outside_ball(xx0_llxp))
            or (outside_ball(xx0_llxm))
            or (outside_ball(yy0_llyp))
            or (outside_ball(yy0_llym))
        ):
            x_0[n][0] = np.random.uniform(0.2, 0.8, size=[1])[0]
            y_0[n][0] = np.random.uniform(0.2, 0.8, size=[1])[0]
            lx[n][0] = np.random.uniform(0.2, 0.45, size=[1])[0]
            ly[n][0] = np.random.uniform(0.2, 0.45, size=[1])[0]
            xx_0, yy_0, llx, lly = x_0[n][0], y_0[n][0], lx[n][0], ly[n][0]
            xx0_llxp = rotate([xx_0, yy_0], [xx_0 + llx, yy_0], theta[n])
            xx0_llxm = rotate([xx_0, yy_0], [xx_0 - llx, yy_0], theta[n])
            yy0_llyp = rotate([xx_0, yy_0], [xx_0, yy_0 + lly], theta[n])
            yy0_llym = rotate([xx_0, yy_0], [xx_0, yy_0 - lly], theta[n])

    for n in range(nb_data):
        xx_0, yy_0, llx, lly, ttheta = (
            x_0[n][0],
            y_0[n][0],
            lx[n][0],
            ly[n][0],
            theta[n][0],
        )
        mmu0, mmu1 = mu0[n][0], mu1[n][0]
        while eval_phi(mmu0, mmu1, xx_0, yy_0, llx, lly, ttheta) > -0.15:
            mu0[n][0] = np.random.uniform(0.2, 0.8, size=[1])[0]
            mu1[n][0] = np.random.uniform(0.2, 0.8, size=[1])[0]
            mmu0, mmu1 = mu0[n][0], mu1[n][0]

    F = call_F(XXYY, mu0, mu1, sigma_x, sigma_y, amplitude)
    F = np.reshape(F, [nb_data, nb_vert, nb_vert]).transpose(0, 2, 1)

    phi = call_phi(XXYY, x_0, y_0, lx, ly, theta)
    phi = np.reshape(phi, [nb_data, nb_vert, nb_vert]).transpose(0, 2, 1)

    G = call_G(XXYY, alpha, beta)
    G = np.reshape(G, [nb_data, nb_vert, nb_vert]).transpose(0, 2, 1)

    params = np.concatenate(
        [mu0, mu1, sigma_x, sigma_y, amplitude, x_0, y_0, lx, ly, theta, alpha, beta],
        axis=1,
    )
    return F, phi, G, params


def test():
    create_parameters(nb_data=4, nb_vert=64)


if __name__ == "__main__":
    test()
