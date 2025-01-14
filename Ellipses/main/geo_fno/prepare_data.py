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
