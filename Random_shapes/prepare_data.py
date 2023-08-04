import numpy as np
import matplotlib.pyplot as plt
import random
import os
import generate_domains as gd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def call_F(xy, mu0, mu1, sigma):
    return 100.0 * np.exp(
        -((xy[0] - mu0) ** 2 + (xy[1] - mu1) ** 2) / (2.0 * sigma**2)
    )


def call_G(xy, alpha, beta):
    return np.sin(xy[0] * alpha) + (xy[0] ** 2 - xy[1] ** 2) * np.cos(
        np.pi * beta * xy[1]
    )


def call_Omega(phi):
    return phi <= 0.0


def omega_mask(phi):
    F = call_Omega(phi)
    return F.astype(np.float64)


def test_omega():
    n = 9
    F, phi, G, params = create_FG_numpy(n, 64)
    plt.figure(figsize=(12, 12))
    for i in range(n):
        mask = omega_mask(phi[i, :, :])
        plt.subplot(3, 3, i + 1)
        plt.imshow(mask, origin="lower")
    plt.tight_layout()
    plt.show()


def test_F():
    F, phi, G, params = create_FG_numpy(1, 64)
    mask = omega_mask(phi)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(F[0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask * F[0])
    plt.show()


def test_phi():
    F, phi, G, params = create_FG_numpy(1, 64)
    mask = omega_mask(phi)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(phi[0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask * phi[0])
    plt.colorbar()
    plt.show()


def create_FG_numpy(nb_data, nb_vert, n_mode=4, seed=2023):
    xy = np.linspace(0.0, 1.0, 2 * nb_vert - 1)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    mu0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma = np.random.uniform(0.1, 0.5, size=[nb_data, 1])

    F = call_F(XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, 2 * nb_vert - 1, 2 * nb_vert - 1])

    if not (os.path.exists(f"./data_domains_{nb_data}_{n_mode}")):
        gd.generate_multiple_domains(
            nb_data=nb_data,
            nb_vert=2 * nb_vert - 1,
            seed=seed,
            n_mode=n_mode,
            save=True,
        )

    phi = np.load(
        f"./data_domains_{nb_data}_{n_mode}/level_sets_{nb_data}.npy"
    )

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
            f = call_F(XXYY, __mu0, __mu1, __sigma)
            f = np.reshape(f, [2 * nb_vert - 1, 2 * nb_vert - 1])

            F[i] = f
            new_gen += 1

    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    F = call_F(XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, nb_vert, nb_vert])

    alpha = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])
    beta = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])

    G = call_G(XXYY, alpha, beta)
    G = np.reshape(G, [nb_data, nb_vert, nb_vert])

    params = np.concatenate([mu0, mu1, sigma, alpha, beta], axis=1)

    return F, phi, G, params


def test():
    create_FG_numpy(nb_data=4, nb_vert=64)


if __name__ == "__main__":
    test()
    test_omega()
    # for i in range(5):
    #     test_F()
    # for i in range(5):
    #     test_phi()
