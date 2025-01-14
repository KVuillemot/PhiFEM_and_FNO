import numpy as np
import matplotlib.pyplot as plt
import random
import os

# import generate_domains as gd
import torch
from scipy.stats import qmc


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


def call_phi(xy, mu_x, mu_y, sigma_x, sigma_y):

    s = np.sum(
        np.exp(
            -(
                ((xy[0] - mu_x) ** 2 / (2.0 * sigma_x**2))
                + ((xy[1] - mu_y) ** 2 / (2.0 * sigma_y**2))
            )
        ),
        axis=1,
    )
    if len(s.shape) == 2:
        return -s + (np.max(s, axis=1) / 2)[:, None]
    else:
        return -s + (np.max(s) / 2)[None]


def create_parameters(
    nb_training_data=0,
    nb_rep_avg_shape=0,
    nb_validation_data=0,
    nb_test_data=300,
    nb_vert=64,
    n_mode=3,
    seed=2023,
    nb_rep_training_shapes=None,
):

    nb_data = nb_training_data + nb_validation_data + nb_test_data

    print(f"{nb_data=}")
    print(f"{nb_training_data=}")
    print(f"{nb_validation_data=}")
    print(f"{nb_test_data=}")
    nb_training_shapes = (nb_training_data - nb_rep_avg_shape) // nb_rep_training_shapes
    nb_shapes = (
        (nb_training_data - nb_rep_avg_shape) // nb_rep_training_shapes
        + nb_validation_data
        + nb_test_data
    )
    print(f"{nb_rep_training_shapes=}")
    print(f"{nb_training_shapes=}")
    print(f"{nb_shapes=}")

    x = np.linspace(0.0, 1.0, nb_vert)
    y = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(x, y)
    XX = XX.flatten()  # np.reshape(XX, [-1])
    YY = YY.flatten()  # np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    sampler = qmc.LatinHypercube(d=19)
    sample = sampler.random(n=nb_data)

    l_bounds = [
        0.35,
        0.35,
        0.15,
        0.15,
        15,
        -0.8,
        -0.8,
        0.29,
        0.55,
        0.35,
        0.54,
        0.54,
        0.29,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
    ]
    u_bounds = [
        0.65,
        0.65,
        0.45,
        0.45,
        35,
        0.8,
        0.8,
        0.45,
        0.7,
        0.65,
        0.7,
        0.7,
        0.46,
        0.26,
        0.26,
        0.26,
        0.26,
        0.22,
        0.22,
    ]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    mu0 = sample_scaled[:, :1].reshape([nb_data, 1])
    mu1 = sample_scaled[:, 1:2].reshape([nb_data, 1])
    sigma_x = sample_scaled[:, 2:3].reshape([nb_data, 1])
    sigma_y = sample_scaled[:, 3:4].reshape([nb_data, 1])
    amplitude = sample_scaled[:, 4:5].reshape([nb_data, 1])
    alpha = sample_scaled[:, 5:6].reshape([nb_data, 1])
    beta = sample_scaled[:, 6:7].reshape([nb_data, 1])

    mu_x = sample_scaled[:, 7:10].reshape([nb_data, 3, 1])
    mu_y = sample_scaled[:, 10:13].reshape([nb_data, 3, 1])
    sigma_x_phi = sample_scaled[:, 13:16].reshape([nb_data, 3, 1])
    sigma_y_phi = sample_scaled[:, 16:].reshape([nb_data, 3, 1])

    phi = call_phi(XXYY, mu_x, mu_y, sigma_x_phi, sigma_y_phi)
    phi = np.reshape(phi, [nb_data, nb_vert, nb_vert]).transpose(0, 2, 1)

    domains = phi <= 3e-16
    domains_tmp = domains.flatten()
    domains_nan = domains.copy().flatten().astype(float)
    domains_nan[np.where(domains_tmp == False)] = np.nan
    domains_nan = np.reshape(domains_nan, domains.shape)

    F = call_F(XXYY, mu0, mu1, sigma_x, sigma_y, amplitude)
    F = np.reshape(F, [nb_data, nb_vert, nb_vert]).transpose(0, 2, 1)

    G = call_G(XXYY, alpha, beta)
    G = np.reshape(G, [nb_data, nb_vert, nb_vert]).transpose(0, 2, 1)

    params = np.concatenate(
        [
            mu0,
            mu1,
            sigma_x,
            sigma_y,
            amplitude,
            alpha,
            beta,
            mu_x[:, :, 0],
            mu_y[:, :, 0],
            sigma_x_phi[:, :, 0],
            sigma_y_phi[:, :, 0],
        ],
        axis=1,
    )
    print(f"{params.shape=}")
    return F, phi, G, params


def test():
    F, phi, G, params = create_parameters(
        nb_training_data=0,
        nb_rep_avg_shape=0,
        nb_validation_data=0,
        nb_test_data=500,
        nb_vert=64,
        n_mode=3,
        seed=2023,
        nb_rep_training_shapes=1,
    )

    mu_x = params[:, 7:10].reshape([-1, 3, 1])
    mu_y = params[:, 10:13].reshape([-1, 3, 1])
    for i in range(30):
        plt.figure()
        x = np.linspace(0, 1, 64)
        plt.contourf(x, x, phi[i, :, :].T <= 3e-16, cmap="viridis")
        plt.plot(mu_x[i, :, 0], mu_y[i, :, 0], "r+")
        plt.show()


if __name__ == "__main__":
    test()
