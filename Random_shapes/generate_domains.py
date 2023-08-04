import numpy as np
import matplotlib.pyplot as plt
import os
import time


def generate_domain(size, i, n_mode=4, seed=2023):
    """Generate a random connected domain and associated levelset.

    Parameters:
        size (tuple): Size of the domain as (nx, ny).
        i (int): Index used for random seed.
        n_mode (int, optional): Number of modes for basis functions. Default is 4.
        seed (int, optional): Random seed. Default is 2023.

    Returns:
        domain (ndarray): Random connected domain as a binary numpy array.
        levelset (ndarray): Levelset associated with the domain.
        coefs (ndarray): Coefficients of the basis functions used in generating the domain.
    """

    batch_size = 1
    nx, ny = size
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    modes = np.array(list(range(1, n_mode + 1)))

    def make_basis_1d(x):
        l = np.pi
        onde = lambda x: np.sin(x)
        return onde(l * x[None, :] * modes[:, None])

    basis_x = make_basis_1d(x)  # (n_mode,nx)
    basis_y = make_basis_1d(y)  # (n_mode,ny)
    basis_2d = (
        basis_x[None, :, None, :] * basis_y[:, None, :, None]
    )  # (n_mode_y, n_mode_x, n_y, n_x)

    if seed is not None:
        np.random.seed(seed + i)
    else:
        seed = seed + i
    coefs = np.random.uniform(-1, 1, size=[batch_size, n_mode, n_mode])

    coefs /= (modes[None, :, None] * modes[None, None, :]) ** 2
    levelset = 0.4 - (
        np.sum(
            coefs[:, :, :, None, None] * basis_2d[None, :, :, :, :],
            axis=(1, 2),
        )
    )
    domain = (levelset < 0.0).astype(int)
    return domain[0, :, :], levelset[0, :, :], coefs


def connected_components(domain):
    """
    Label connected components in a binary domain.

    Parameters:
        domain (ndarray): Binary numpy array representing the domain.

    Returns:
        label_map (ndarray): Numpy array with labels for each connected component.
        num_labels (int): Number of connected components found in the domain.
    """
    label_map = np.zeros_like(domain)
    label = 1
    for y in range(domain.shape[1]):
        for x in range(domain.shape[0]):
            if domain[x, y] == 0:
                continue
            if label_map[x, y] != 0:
                continue
            explore = [(x, y)]
            while explore:
                current_x, current_y = explore.pop()
                label_map[current_x, current_y] = label
                neighbors = get_neighbors(current_x, current_y, domain.shape)
                for neighbor_x, neighbor_y in neighbors:
                    if (
                        domain[neighbor_x, neighbor_y] == 1
                        and label_map[neighbor_x, neighbor_y] == 0
                    ):
                        explore.append((neighbor_x, neighbor_y))
            label += 1
    return label_map, label - 1


def get_neighbors(x, y, size):
    """
    Get neighboring points around a given (x, y) coordinate in a grid.

    Parameters:
        x (int): x-coordinate.
        y (int): y-coordinate.
        size (tuple): Size of the grid as (nx, ny).

    Returns:
        neighbors (list): List of (x, y) tuples representing neighboring points.
    """
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < size[0] - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < size[1] - 1:
        neighbors.append((x, y + 1))
    return neighbors


def check_connected(domain):
    """
    Check if a given domain is connected.

    Parameters:
        domain (ndarray): Binary numpy array representing the domain.

    Returns:
        connected (bool): True if the domain is connected, False otherwise.
    """
    label_map, num_labels = connected_components(domain)
    if num_labels == 1:
        connected = True
    else:
        connected = False
    return connected


def check_away_boundary(domain):
    """
    Check if a given domain is sufficiently away from the boundary.

    Parameters:
        domain (ndarray): Binary numpy array representing the domain.

    Returns:
        away_boundary (bool): True if the domain is away from the boundary, False otherwise.
    """
    limit = 4
    interested_vert_ = domain[:, -limit:]
    interested_hori_ = domain[-limit:, :].T
    interested_vert = domain[:, :limit]
    interested_hori = domain[:limit, :].T

    interested = np.stack(
        [interested_hori, interested_vert, interested_hori_, interested_vert_]
    )
    interested = np.unique(interested)
    if (
        len(interested) > 1
        or (len(interested) == 1 and interested[0] == 1)
        or len(np.unique(domain)) == 1
    ):
        return False
    else:
        return True


def check_domain_sufficiently_big(domain):
    """
    Check if a given domain is sufficiently big.

    Parameters:
        domain (ndarray): Binary numpy array representing the domain.

    Returns:
        big (bool): True if the domain is sufficiently big, False otherwise.
    """
    return np.sum(domain) > (domain.shape[0] * domain.shape[1]) / 20


def check_if_valid(domain):
    """
    Check if a given domain is valid (connected, away from boundary, and sufficiently big).

    Parameters:
        domain (ndarray): Binary numpy array representing the domain.

    Returns:
        valid (bool): True if the domain is valid, False otherwise.
    """
    connected = check_connected(domain)
    away_from_boundary = check_away_boundary(domain)
    big = check_domain_sufficiently_big(domain)
    if connected and away_from_boundary and big:
        return True
    else:
        return False


def generate_multiple_domains(
    nb_vert, nb_data, seed=2023, n_mode=4, save=True
):
    """
    Generate multiple random connected domains.

    Parameters:
        nb_vert (int): Number of vertices for the domain (nx = ny).
        nb_data (int): Number of random domains to generate.
        seed (int, optional): Random seed. Default is 2023.
        n_mode (int, optional): Number of modes for basis functions. Default is 4.
        save (bool, optional): Whether to save the generated domains. Default is True.
    """
    domain_size = (
        nb_vert,
        nb_vert,
    )
    if save:
        if not (os.path.exists(f"./data_domains_{nb_data}_{n_mode}")):
            os.makedirs(f"./data_domains_{nb_data}_{n_mode}")

    domains, level_sets = [], []

    final_params = []
    start = time.time()
    index = 0
    for i in range(nb_data):
        domain, levelset, params = generate_domain(
            domain_size, index, n_mode, seed
        )
        domain_ok = check_if_valid(domain)
        index += 1
        while not (domain_ok):
            domain, levelset, params = generate_domain(
                domain_size, index, n_mode, seed
            )
            domain_ok = check_if_valid(domain)
            index += 1
        domains.append(domain)
        level_sets.append(levelset)
        final_params.append(params)

    end = time.time()

    print(f"Time to generate domains : {end-start} s")

    if save:
        np.save(
            f"./data_domains_{nb_data}_{n_mode}/level_sets_{nb_data}.npy",
            np.array(level_sets),
        )
        np.save(
            f"./data_domains_{nb_data}_{n_mode}/domains_{nb_data}.npy",
            np.array(domains),
        )
        np.save(
            f"./data_domains_{nb_data}_{n_mode}/params_{nb_data}.npy",
            np.array(final_params),
        )


def plot_domain(domain, levelset):
    """
    Plot a random connected domain and its associated levelset.

    Parameters:
        domain (ndarray): Binary numpy array representing the domain.
        levelset (ndarray): Levelset associated with the domain.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(levelset, cmap="viridis", origin="lower")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(domain, cmap="gray", origin="lower")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Random Connected Domain")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    domain_size = (
        127,
        127,
    )

    domains, level_sets = [], []
    import time

    final_params = []
    start = time.time()
    index = 0
    nb_data = 5
    for i in range(nb_data):
        domain, levelset, params = generate_domain(domain_size, index)
        domain_ok = check_if_valid(domain)
        index += 1
        while not (domain_ok):
            domain, levelset, params = generate_domain(domain_size, index)
            domain_ok = check_if_valid(domain)
            index += 1
        plot_domain(domain, levelset)
        domains.append(domain)
        level_sets.append(levelset)
        final_params.append(params)

    end = time.time()

    print(f"Total time : {end-start} s")
