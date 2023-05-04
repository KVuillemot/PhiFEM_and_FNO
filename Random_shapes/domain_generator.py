from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import time

pp = print

def generate_fourier_2D(x, y, batch_size, n_mode, seed):
    modes = tf.cast(tf.range(1, n_mode + 1), tf.float32)

    def make_basis_1d(x):
        l = np.pi  
        onde = lambda x: tf.sin(x)
        return onde(l * x[None, :] * modes[:, None])

    basis_x = make_basis_1d(x)  # (n_mode,nx)
    basis_y = make_basis_1d(y)  # (n_mode,ny)
    basis_2d = (
        basis_x[None, :, None, :] * basis_y[:, None, :, None]
    )  # (n_mode_y, n_mode_x, n_y, n_x)

    if seed is not None:
        tf.random.set_seed(seed + 2347287)
    coefs = tf.random.uniform(
        shape=[batch_size, n_mode, n_mode], minval=-1, maxval=+1
    )

    coefs /= (modes[None, :, None] * modes[None, None, :]) ** 2
    return tf.reduce_sum(
        coefs[:, :, :, None, None] * basis_2d[None, :, :, :, :], axis=[1, 2]
    )  # *0.1


def generate_domain_fn(x, y, batch_size, n_mode, seed):
    """
    level: dans [0,1]. Proche de 0=> grand domaines
    """
    X = generate_fourier_2D(x, y, batch_size, n_mode, seed)
    maxs = tf.reduce_max(X, axis=[1, 2])
    mins_abs = tf.abs(tf.reduce_min(X, axis=[1, 2]))

    mask = maxs > mins_abs
    zero = tf.zeros_like(X)

    X_ = tf.where(mask[:, None, None], X / maxs[:, None, None], zero)
    X__ = tf.where(~mask[:, None, None], (-X) / mins_abs[:, None, None], zero)

    domain_fn = X_ + X__
    return domain_fn


def generate_domain_fn_from_X(X):

    maxs = tf.reduce_max(X, axis=[1, 2])
    mins_abs = tf.abs(tf.reduce_min(X, axis=[1, 2]))

    mask = maxs > mins_abs
    zero = tf.zeros_like(X)

    X_ = tf.where(mask[:, None, None], X / maxs[:, None, None], zero)
    X__ = tf.where(~mask[:, None, None], (-X) / mins_abs[:, None, None], zero)

    domain_fn = X_ + X__
    return domain_fn


def add_neighbours_in_stack(
    image, threshold, stack: list, i: int, j: int, inComponent
) -> None:
    inComponent[i, j] = True
    neighbours = [(i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1)]
    for k, l in neighbours:
        if 0 <= k < image.shape[0] and 0 <= l < image.shape[1]:
            if image[k, l] > threshold and not inComponent[k, l]:
                inComponent[k, l] = True
                stack.append((k, l))


def connected_component(image: np.ndarray, threshold, i0: int, j0: int):
    inComponent = np.zeros(image.shape, dtype=bool)
    stack = [(i0, j0)]
    nb = 0
    while len(stack) > 0:
        nb += 1
        (i, j) = stack.pop()
        add_neighbours_in_stack(image, threshold, stack, i, j, inComponent)

    return inComponent


def generate_domain(nx, ny, batch_size, n_mode, threshold, seed):
    x = tf.linspace(0.0, 1, nx)
    y = tf.linspace(0.0, 1, ny)

    domain_fn = generate_domain_fn(x, y, batch_size, n_mode, seed)

    res = []
    for i in range(batch_size):
        img = domain_fn[i, :, :]
        i0, j0 = np.unravel_index(
            np.argmax(domain_fn[i, :, :]), shape=(ny, nx)
        )
        res.append(connected_component(img, threshold, i0, j0))

    return tf.stack(res)


def test_generate_domain():
    batch_size = 5
    domain = generate_domain(
        nx=40, ny=40, batch_size=batch_size, n_mode=3, threshold=0.4, seed=None
    )
    fig, axs = plt.subplots(
        batch_size, figsize=(2, batch_size * 2), sharex="all", sharey="all"
    )
    for i in range(batch_size):
        axs[i].imshow(domain[i, :, :])

    plt.show()


def test_generate_fourier():
    nx = 20
    ny = 30
    x = tf.linspace(0.0, 1, nx)
    y = tf.linspace(0.0, 1, ny)
    batch_size = 1

    n_mode = 4

    def one(seed):
        res = generate_fourier_2D(x, y, batch_size, n_mode, seed=seed)
        sh = res.shape
        assert sh == (batch_size, ny, nx)
        plt.imshow(res[0, :, :])
        plt.colorbar()

        fig, (ax0, ax1) = plt.subplots(2, 1)

        for j in range(nx):
            ax0.plot(res[0, :, j])
        for i in range(ny):
            ax1.plot(res[0, i, :])
        plt.show()

    one(None)


def create_level_set(
    nx, ny, n_mode, batch_size, threshold, seed=None, save=False
):
    x = tf.linspace(0.0, 1, nx)
    y = tf.linspace(0.0, 1, ny)
    X = generate_fourier_2D(x, y, batch_size, n_mode, seed)

    domain_fn = generate_domain_fn_from_X(X)
    res = []
    for i in range(batch_size):
        img = domain_fn[i, :, :]
        i0, j0 = np.unravel_index(
            np.argmax(domain_fn[i, :, :]), shape=(ny, nx)
        )  
        
        res.append(connected_component(img, threshold, i0, j0))

    res = tf.stack(res)
    start = time.time()
    functions = []
    for i in range(batch_size):
        mask_1_0_tmp = res[i, :, :].numpy().astype(float)
        mask_1_0 = mask_1_0_tmp == 0  # 1 outside 0 inside

        mask_min = res[i, :, :].numpy().astype(float)
        mask_min[mask_min == 0] = np.nan
        mask_min[mask_min == 1] = 1.0  # nan outside 1 inside

        new_domain_fn_abs = np.absolute(domain_fn[i, :, :])

        inside_function = -(
            domain_fn[i, :, :] - np.nanmin((mask_min * domain_fn[i, :, :]))
        ) * res[i, :, :].numpy().astype(float)

        outside_function = np.absolute(
            (
                new_domain_fn_abs
                - np.nanmin(np.absolute(mask_min * new_domain_fn_abs))
            )
            * mask_1_0
        )
        final_function = inside_function + outside_function
        functions.append(final_function)
    end = time.time()
    print(f"Level set construction : {end-start} s")
    functions = tf.stack(functions)
    if save:
        import os

        if not (os.path.exists("./generated_domains")):
            os.makedirs("./generated_domains/")
        np.save(
            f"./generated_domains/level_set_{batch_size}_{nx}_{n_mode}.npy",
            functions,
        )
        np.save(
            f"./generated_domains/domains_{batch_size}_{nx}_{n_mode}.npy", res
        )

    return functions


def test_domains(seed):
    nx, ny, n_mode, batch_size, seed, threshold = (
        127,
        127,
        3,
        1,
        seed,
        0.2,
    )
    x = tf.linspace(0.0, 1, nx)
    y = tf.linspace(0.0, 1, ny)
    X = generate_fourier_2D(x, y, batch_size, n_mode, seed)

    domain_fn = generate_domain_fn_from_X(X)
    print(np.argmax(domain_fn[0, :, :]))
    res = []
    for i in range(batch_size):
        img = domain_fn[i, :, :]
        i0, j0 = np.unravel_index(
            np.argmax(domain_fn[i, :, :]), shape=(ny, nx)
        ) 
        print(f"{(i0,j0)=}")

        res.append(connected_component(img, threshold, i0, j0))

    res = tf.stack(res)
    mask_1_0_tmp = res[0, :, :].numpy().astype(float)
    mask_1_0 = mask_1_0_tmp == 0  # 1 outside 0 inside

    mask_min = res[0, :, :].numpy().astype(float)
    mask_min[mask_min == 0] = np.nan
    mask_min[mask_min == 1] = 1.0  # nan outside 1 inside

    new_domain_fn_abs = np.absolute(domain_fn[0, :, :])

    inside_function = -(
        new_domain_fn_abs
        - np.nanmin(np.absolute(mask_min * new_domain_fn_abs))
    ) * res[0, :, :].numpy().astype(float)

    outside_function = np.absolute(
        (
            new_domain_fn_abs
            - np.nanmin(np.absolute(mask_min * new_domain_fn_abs))
        )
        * mask_1_0
    )
    final_function = inside_function + outside_function

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(new_domain_fn_abs)
    plt.colorbar()
    plt.title("new_domain_fn_abs")
    plt.subplot(2, 3, 2)
    plt.imshow(inside_function)
    plt.colorbar()
    plt.title("inside_function")
    plt.subplot(2, 3, 3)
    plt.imshow(outside_function)
    plt.colorbar()
    plt.title("outside_function")
    plt.subplot(2, 3, 4)
    plt.imshow(final_function)
    plt.colorbar()
    plt.title("final_function")
    plt.subplot(2, 3, 5)
    plt.imshow(final_function <= 0.0)
    plt.colorbar()
    plt.title("final_function <= 0.0")
    plt.subplot(2, 3, 6)
    plt.imshow(res[0, :, :] == (final_function <= 0.0))
    plt.colorbar()
    plt.title(
        "original domain == new_domain \n min = "
        + str(np.min(res[0, :, :] == (final_function <= 0.0)))
        + "  max = "
        + str(np.max(res[0, :, :] == (final_function <= 0.0)))
    )
    plt.suptitle(f"seed = {seed}")
    plt.tight_layout()
    plt.savefig(f"./outputs_{seed}_threshold_{threshold}.png")
 

if __name__ == "__main__":
    # test_generate_fourier()

    start = time.time()

    functions, res = create_level_set(
        127, 127, 3, 20, 0.4, seed=2023, save=False
    )
    end = time.time()
    print(f"Time = {end-start}")
