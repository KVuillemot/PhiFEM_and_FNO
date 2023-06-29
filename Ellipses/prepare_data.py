import numpy as np
import matplotlib.pyplot as plt
import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

seed = 2023
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)


def call_F(pre, xy, mu0, mu1, sigma):
    return 100.0 * np.exp(
        -((xy[0] - mu0) ** 2 + (xy[1] - mu1) ** 2) / (2.0 * sigma**2)
    )


def call_G(pre, xy, alpha, beta):
    return np.sin(xy[0] * alpha) + (xy[0] ** 2 - xy[1] ** 2) * np.cos(
        np.pi * beta * xy[1]
    )


def call_phi(pre, xy, x_0, y_0, lx, ly, theta):
    return (
        -1.0
        + ((xy[0] - x_0) * np.cos(theta) + (xy[1] - y_0) * np.sin(theta)) ** 2
        / lx**2
        + ((xy[0] - x_0) * np.sin(theta) - (xy[1] - y_0) * np.cos(theta)) ** 2
        / ly**2
    )


def eval_phi(pre, x, y, x_0, y_0, lx, ly, theta):
    return (
        -1.0
        + ((x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta)) ** 2
        / lx**2
        + ((x - x_0) * np.sin(theta) - (y - y_0) * np.cos(theta)) ** 2
        / ly**2
    )


def call_Omega(pre, xy, x_0, y_0, lx, ly, theta):
    return call_phi(pre, xy, x_0, y_0, lx, ly, theta) <= 0.0


def Omega_bool(x, y, x_0, y_0, lx, ly, theta):
    xy = (x, y)
    return call_Omega(None, xy, x_0, y_0, lx, ly, theta)


def omega_mask(nb_vert, x_0, y_0, lx, ly, theta):
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    F = call_Omega(np, XXYY, x_0, y_0, lx, ly, theta)
    F = np.reshape(F, [nb_vert, nb_vert])
    return F.astype(np.float64)


def test_omega():
    F, phi, G, params = create_FG_numpy(1, 64)
    mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = params[0]
    mask = omega_mask(64, x_0, y_0, lx, ly, theta)
    plt.imshow(mask, origin="lower")
    plt.title("test Omega")
    plt.show()


def test_F():
    F, phi, G, params = create_FG_numpy(1, 64)
    mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = params[0]
    mask = omega_mask(64, x_0, y_0, lx, ly, theta)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(F[0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask * F[0])
    plt.colorbar()
    plt.suptitle("test F")
    plt.show()


def test_G():
    F, phi, G, params = create_FG_numpy(1, 64)
    mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = params[0]
    mask = omega_mask(64, x_0, y_0, lx, ly, theta)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(G[0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask * G[0])
    plt.colorbar()
    plt.suptitle("Test G")
    plt.show()


def test_phi():
    F, phi, G, params = create_FG_numpy(1, 64)
    mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = params[0]
    mask = omega_mask(64, x_0, y_0, lx, ly, theta)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(phi[0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask * phi[0])
    plt.colorbar()
    plt.suptitle("Test Phi")
    plt.show()


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def outside_ball(point):
    x, y = point
    return -0.45 + np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) > 0.0


def create_FG_numpy(nb_data, nb_vert):
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    mu0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma = np.random.uniform(0.15, 0.45, size=[nb_data, 1])

    alpha = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])
    beta = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])

    x_0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    y_0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    lx = np.random.uniform(0.2, 0.45, size=[nb_data, 1])
    ly = np.random.uniform(0.2, 0.45, size=[nb_data, 1])
    theta = np.random.uniform(0.0, np.pi, size=[nb_data, 1])
    check_data = 0
    for n in range(nb_data):
        new_generation = 0
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
            x_0[n][0] = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            y_0[n][0] = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            lx[n][0] = np.random.uniform(0.2, 0.45, size=[1, 1])[0]
            ly[n][0] = np.random.uniform(0.2, 0.45, size=[1, 1])[0]
            xx_0, yy_0, llx, lly = x_0[n][0], y_0[n][0], lx[n][0], ly[n][0]
            xx0_llxp = rotate([xx_0, yy_0], [xx_0 + llx, yy_0], theta[n])
            xx0_llxm = rotate([xx_0, yy_0], [xx_0 - llx, yy_0], theta[n])
            yy0_llyp = rotate([xx_0, yy_0], [xx_0, yy_0 + lly], theta[n])
            yy0_llym = rotate([xx_0, yy_0], [xx_0, yy_0 - lly], theta[n])
            new_generation += 1
        check_data += 1

    for n in range(nb_data):
        new_generation = 0
        xx_0, yy_0, llx, lly, ttheta = (
            x_0[n][0],
            y_0[n][0],
            lx[n][0],
            ly[n][0],
            theta[n][0],
        )
        mmu0, mmu1 = mu0[n][0], mu1[n][0]
        sigma[n][0] = np.random.uniform(min(llx, lly) / 2.0, max(llx, lly))
        while eval_phi(np, mmu0, mmu1, xx_0, yy_0, llx, lly, ttheta) > -0.15:
            mu0[n][0] = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            mu1[n][0] = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            mmu0, mmu1 = mu0[n][0], mu1[n][0]

        check_data += 1

    F = call_F(np, XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, nb_vert, nb_vert])

    phi = call_phi(np, XXYY, x_0, y_0, lx, ly, theta)
    phi = np.reshape(phi, [nb_data, nb_vert, nb_vert])

    G = call_G(np, XXYY, alpha, beta)
    G = np.reshape(G, [nb_data, nb_vert, nb_vert])

    params = np.concatenate(
        [mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta], axis=1
    )
    return F, phi, G, params


def test():
    create_FG_numpy(nb_data=4, nb_vert=64)


if __name__ == "__main__":
    test()
    for i in range(5):
        test_omega()
    for i in range(5):
        test_F()
    for i in range(5):
        test_phi()
    for i in range(5):
        test_G()
