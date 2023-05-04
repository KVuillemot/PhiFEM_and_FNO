import numpy as np
import matplotlib.pyplot as plt
import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

seed = 27042023
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)

import domain_generator as dg


def call_F(pre, xy, mu0, mu1, sigma):
    return 100.0 * np.exp(
        -((xy[0] - mu0) ** 2 + (xy[1] - mu1) ** 2) / (2.0 * sigma**2)
    )


def call_Omega(phi):
    return phi <= 0.0


def create_phi(
    nx=127,
    ny=127,
    n_mode=3,
    batch_size=1,
    threshold=0.4,
    seed=seed,
    save=False,
):
    return dg.create_level_set(
        nx, ny, n_mode, batch_size, threshold, seed, save
    ).numpy()


def omega_mask(phi):
    F = call_Omega(phi)
    return F.astype(np.float64)


def test_omega():
    n = 9
    F, phi, params = create_FG_numpy(n, 64)
    plt.figure(figsize=(12, 12))
    for i in range(n):
        mask = omega_mask(phi[i, :, :])
        plt.subplot(3, 3, i + 1)
        plt.imshow(mask, origin="lower")
    plt.tight_layout()
    plt.show()


def test_F():
    F, phi, params = create_FG_numpy(1, 64)
    mask = omega_mask(phi)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(F[0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask * F[0])
    plt.show()


def test_phi():
    F, phi, params = create_FG_numpy(1, 64)
    mask = omega_mask(phi)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(phi[0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask * phi[0])
    plt.colorbar()
    plt.show()


def create_FG_numpy(nb_data, nb_vert):
    xy = np.linspace(0.0, 1.0, 2 * nb_vert - 1)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    mu0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma = np.random.uniform(0.1, 0.5, size=[nb_data, 1])

    F = call_F(np, XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, 2 * nb_vert - 1, 2 * nb_vert - 1])

    phi = create_phi(
        2 * nb_vert - 1,
        2 * nb_vert - 1,
        n_mode=3,
        batch_size=nb_data,
        threshold=0.4,
        seed=seed,
        save=True,
    )
    phi = np.reshape(phi, [nb_data, 2 * nb_vert - 1, 2 * nb_vert - 1])

    for i in range(np.shape(F)[0]):
        domain = omega_mask(phi[i])
        f = F[i]
        new_gen = 0
        while np.max(f * domain) < 80.0:
            __mu0 = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            __mu1 = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            __sigma = np.random.uniform(0.1, 0.5, size=[1, 1])[0]
            mu0[i][0] = __mu0
            mu1[i][0] = __mu1
            sigma[i][0] = __sigma
            f = call_F(np, XXYY, __mu0, __mu1, __sigma)
            f = np.reshape(f, [2 * nb_vert - 1, 2 * nb_vert - 1])

            F[i] = f
            new_gen += 1
            print(f"{new_gen=}")

    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    F = call_F(np, XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, nb_vert, nb_vert])

    params = np.concatenate([mu0, mu1, sigma], axis=1)

    return F, phi, params


def test():
    create_FG_numpy(nb_data=4, nb_vert=64)


if __name__ == "__main__":
    test()
    test_omega()
    # for i in range(5):
    #     test_F()
    # for i in range(5):
    #     test_phi()
