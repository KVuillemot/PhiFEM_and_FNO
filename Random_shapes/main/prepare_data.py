import numpy as np
import matplotlib.pyplot as plt
import random
import os
import generate_domains as gd
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


def call_Omega(phi):
    return phi <= 3e-16


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


def call_phi(x, y, coeffs_ls, threshold):
    modes = np.array(list(range(1, np.shape(coeffs_ls)[-1] + 1)))

    def make_basis_1d(x):
        l = np.pi
        onde = lambda x: np.sin(x)
        return onde(l * x[None, :] * modes[:, None])

    basis_x = make_basis_1d(x)  # (n_mode,nx)
    basis_y = make_basis_1d(y)  # (n_mode,ny)
    basis_2d = basis_x[None, :, None, :] * basis_y[:, None, :, None]
    # print(f"{basis_x.shape=}")
    # print(f"{basis_y.shape=}")
    # print(f"{basis_2d.shape=}")

    val = threshold - (
        np.sum(
            coeffs_ls[:, :, :, :, None, None] * basis_2d[None, :, :, :, :],
            axis=(2, 3),
        )
    )

    domain = (val < 0.0).astype(int)
    domain = np.reshape(
        domain, (coeffs_ls.shape[0], basis_2d.shape[-1], basis_2d.shape[-1])
    )
    val = np.reshape(val, (coeffs_ls.shape[0], basis_2d.shape[-1], basis_2d.shape[-1]))
    return val[:, :, :], domain[:, :, :]


def eval_phi(x, y, coeffs_ls, threshold):
    modes = np.array(list(range(1, np.shape(coeffs_ls)[-1] + 1)))

    basis_x = np.array([np.sin(l * np.pi * x) for l in modes])
    basis_y = np.array([np.sin(l * np.pi * y) for l in modes])

    basis_2d = basis_x[:, None] * basis_y[None, :]
    val = threshold - np.sum(coeffs_ls[:, :] * basis_2d[:, :])
    return val


def create_FG_numpy(nb_data, nb_vert, n_mode=4, seed=2023, compare_methods=False):
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    mu0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma_x = np.random.uniform(0.15, 0.45, size=[nb_data, 1])
    sigma_y = np.random.uniform(0.15, 0.45, size=[nb_data, 1])
    amplitude = np.reshape(np.repeat(25.0, nb_data), (nb_data, 1))
    alpha = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])
    beta = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])

    if not (
        os.path.exists(f"../data_domains_{n_mode}/level_sets_{nb_data}_{seed}.npy")
    ):
        gd.generate_multiple_domains(
            nb_data=nb_data,
            nb_vert=nb_vert,
            seed=seed,
            n_mode=n_mode,
            save=True,
        )

    coeffs_ls = np.load(f"../data_domains_{n_mode}/params_{nb_data}_{seed}.npy")
    phi, domain = call_phi(xy, xy, coeffs_ls=coeffs_ls, threshold=0.4)
    phi = np.reshape(phi, [nb_data, nb_vert, nb_vert])
    domains = phi <= 3e-16
    domains_tmp = domains.flatten()
    domains_nan = domains.copy().flatten().astype(float)
    domains_nan[np.where(domains_tmp == False)] = np.nan
    domains_nan = np.reshape(domains_nan, domains.shape)

    F = call_F(XXYY, mu0, mu1, sigma_x, sigma_y, amplitude)
    F = np.reshape(F, [nb_data, nb_vert, nb_vert])
    for n in range(nb_data):
        mmu0, mmu1 = mu0[n][0], mu1[n][0]
        F_i_on_domain = F[n] * domains_nan[n]
        F_i = F[n]
        while np.nanmax(F_i_on_domain) < np.max(F_i):
            mu0[n][0] = np.random.uniform(0.2, 0.8, size=[1])[0]
            mu1[n][0] = np.random.uniform(0.2, 0.8, size=[1])[0]
            mmu0, mmu1 = mu0[n][0], mu1[n][0]
            F_i = call_F(XXYY, mmu0, mmu1, sigma_x[n], sigma_y[n], amplitude[n])
            F_i = np.reshape(F_i, [nb_vert, nb_vert])
            F_i_on_domain = F_i * domains_nan[n]

    F = call_F(XXYY, mu0, mu1, sigma_x, sigma_y, amplitude)
    F = np.reshape(F, [nb_data, nb_vert, nb_vert])

    G = call_G(XXYY, alpha, beta)
    G = np.reshape(G, [nb_data, nb_vert, nb_vert])

    params = np.concatenate(
        [mu0, mu1, sigma_x, sigma_y, amplitude, alpha, beta], axis=1
    )
    if compare_methods:
        return F, phi, G, params, coeffs_ls
    else:
        return F, phi, G, params


def test():
    create_FG_numpy(nb_data=4, nb_vert=64)


if __name__ == "__main__":
    test()
