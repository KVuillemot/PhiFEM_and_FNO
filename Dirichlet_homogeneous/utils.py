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
import dolfin as df
from prepare_data import (
    create_FG_numpy,
    call_F,
    call_Omega,
    call_phi,
    eval_phi,
)


def smooth_padding(W: tf.Tensor, pad_size):
    if isinstance(pad_size, tuple) or isinstance(pad_size, list):
        pad_left = pad_size[0]
        pad_right = pad_size[1]
    else:
        pad_left = pad_size
        pad_right = pad_size

    # attention, on ne peut pas aller jusqu'à pad_left = W.shape[-1], certaines fonctions de padding font des slicing
    assert (
        pad_left < W.shape[-1] and pad_right < W.shape[-1]
    ), f"we must have pad_left < W.shape[-1] and pad_right < W.shape[-1]. Here:pad_left={pad_left},pad_right={pad_right} and W.shape[-1]={W.shape[-1]}"

    left_value = W[..., 0:1]
    right_value = W[..., -1:]

    left = W[..., 1 : pad_left + 1] - left_value
    right = W[..., -1 - pad_right : -1] - right_value
    left = -left[..., ::-1]
    right = -right[..., ::-1]
    left += left_value
    right += right_value

    return tf.concat([left, W, right], axis=-1)


def pad_1d(W, pad_size, axis):
    dim = len(W.shape)
    if not (axis == -1 or axis == dim - 1):
        permutation = np.arange(dim)
        permutation[-1] = axis % dim
        permutation[axis] = dim - 1

        W = tf.transpose(W, permutation)
        W = smooth_padding(W, pad_size)
        W = tf.transpose(W, permutation)
    else:
        W = smooth_padding(W, pad_size)

    return W


def slice_at_given_axis(W, begin, size, axis):
    beg = [0 for _ in W.shape]
    sizes = [i for i in W.shape]
    beg[axis] = begin % W.shape[axis]
    sizes[axis] = size
    res = tf.slice(W, beg, sizes)
    return res


class Derivator_fd:
    def __init__(
        self,
        derivative_symbols,
        interval_lenghts,
        axes,
        centred=True,
        keep_size=True,
    ):
        """
        example, assume that the tensor to derivate U is of dim 3. axis 1 are x, axis 2 are Y
        So args must be
        axes=[1,2]
        If derivative_symbols=[(0,0),(0,1),(1,1)] the __call__ method will give
                              [U_xx, U_xy, U_yy ]

        If derivative_symbols={"xx":(0,0),"xy":(0,1),"yy":(1,1)} the __call__ method will give
                              {"xx":U_xx, "xy":U_xy, "yy":U_yy}
        """
        self.axes = axes
        self.centred = centred

        self.derivative_symbols_initial = None
        if isinstance(derivative_symbols, dict):
            self.derivative_symbols_initial = derivative_symbols
            self.derivative_symbols = derivative_symbols.values()
        else:
            self.derivative_symbols = derivative_symbols

        self.interval_lengths = interval_lenghts
        self.keep_size = keep_size

    def D_axis(self, U, i):
        axis = self.axes[i]
        interval_length = self.interval_lengths[i]
        axis_size = U.shape[axis]
        h = interval_length / (axis_size - 1)
        if self.centred:
            U1_ = slice_at_given_axis(U, 2, axis_size - 2, axis)
            U_1 = slice_at_given_axis(U, 0, axis_size - 2, axis)
            D = (U1_ - U_1) / (2 * h)
            if self.keep_size:
                D = pad_1d(D, 1, axis)
            return D
        else:
            U1_ = slice_at_given_axis(U, 1, axis_size - 1, axis)
            U_1 = slice_at_given_axis(U, 0, axis_size - 1, axis)
            D = (U1_ - U_1) / h
            if self.keep_size:
                D = pad_1d(D, (0, 1), axis)
                return D

    def one_symbol_rec(self, U, res, symbol):
        if len(symbol) == 0:
            return U

        previous_der = res.get(symbol[:-1])

        if previous_der is None:
            previous_der = self.one_symbol_rec(U, res, symbol[:-1])
            # res[symbol[:-1]]=previous_der

        der = self.D_axis(previous_der, symbol[-1])
        res[symbol] = der
        return der

    def __call__(self, U):
        res = {}
        for symbol in self.derivative_symbols:
            self.one_symbol_rec(U, res, symbol)

        if self.derivative_symbols_initial is not None:
            res_2 = {}
            for k_init, v_init in self.derivative_symbols_initial.items():
                res_2[k_init] = res[v_init]
        else:
            res_2 = []
            for symbol in self.derivative_symbols:
                res_2.append(res[symbol])

        return res_2


def convert_numpy_matrix_to_fenics(X, nb_vert, degree=2):
    """Function to convert a matrix to a FEniCS function, for degree = 1, 2

    Args:
        X (array): input array, of size nb_dof_x x nb_dof_y
        nb_vert (int): number of vertices in each direction

    Returns:
        X_FEniCS (function FEniCS): output function that can be used with FEniCS
    """
    boxmesh = df.UnitSquareMesh(nb_vert - 1, nb_vert - 1)
    V = df.FunctionSpace(boxmesh, "CG", degree)
    coords = V.tabulate_dof_coordinates()
    coords = coords.T
    new_matrix = np.zeros(
        (
            np.shape(V.tabulate_dof_coordinates())[0],
            np.shape(V.tabulate_dof_coordinates())[1] + 1,
        )
    )

    new_matrix[:, 0] = np.arange(0, np.shape(V.tabulate_dof_coordinates())[0])
    new_matrix[:, 1] = coords[0]
    new_matrix[:, 2] = coords[1]
    sorted_mat = np.array(sorted(new_matrix, key=lambda k: [k[2], k[1]]))
    mapping = sorted_mat[:, 0]
    X_FEniCS = df.Function(V)
    X_FEniCS.vector()[mapping] = X.flatten()

    return X_FEniCS


def compute_L2_error(y_true, y_pred, domain):
    nb_vert = np.shape(y_true)[1]

    if not (isinstance(y_true, np.ndarray)):
        y_true = y_true.numpy()
    if not (isinstance(y_pred, np.ndarray)):
        y_pred = y_pred.numpy()

    y_true = y_true * domain[:, :]
    y_pred = y_pred * domain[:, :]
    y_true = np.reshape(y_true, (nb_vert, nb_vert))
    y_pred = np.reshape(y_pred, (nb_vert, nb_vert))

    nb_vert = y_true.shape[1]
    boxmesh = df.RectangleMesh(
        df.Point(0.0, 0.0), df.Point(1.0, 1.0), nb_vert - 1, nb_vert - 1
    )
    V = df.FunctionSpace(boxmesh, "CG", 1)
    v2d = df.vertex_to_dof_map(V)
    y_true_FEniCS = df.Function(V)
    y_true_FEniCS.vector()[v2d] = y_true.flatten()
    y_pred_FEniCS = df.Function(V)
    y_pred_FEniCS.vector()[v2d] = y_pred.flatten()

    return (
        df.assemble((((y_true_FEniCS - y_pred_FEniCS)) ** 2) * df.dx) ** (0.5)
    ) / (df.assemble((((y_true_FEniCS)) ** 2) * df.dx) ** (0.5))


def compute_H1_error(y_true, y_pred, domain):
    nb_vert = np.shape(y_true)[1]
    y_true = y_true * domain[:, :, None]
    y_pred = y_pred * domain[:, :, None]
    y_true = np.reshape(y_true.numpy(), (nb_vert, nb_vert))
    y_pred = np.reshape(y_pred.numpy(), (nb_vert, nb_vert))

    nb_vert = y_true.shape[1]
    boxmesh = df.RectangleMesh(
        df.Point(0.0, 0.0), df.Point(1.0, 1.0), nb_vert - 1, nb_vert - 1
    )
    V = df.FunctionSpace(boxmesh, "CG", 1)
    v2d = df.vertex_to_dof_map(V)
    y_true_FEniCS = df.Function(V)
    y_true_FEniCS.vector()[v2d] = y_true.flatten()
    y_pred_FEniCS = df.Function(V)
    y_pred_FEniCS.vector()[v2d] = y_pred.flatten()
    plt.figure()
    plt.subplot(1, 2, 1)
    df.plot(y_pred_FEniCS)
    plt.subplot(1, 2, 2)
    df.plot(y_true_FEniCS)
    plt.show()
    return (
        df.assemble(((df.grad(y_true_FEniCS - y_pred_FEniCS)) ** 2) * df.dx)
        ** (0.5)
    ) / (df.assemble(((df.grad(y_true_FEniCS)) ** 2) * df.dx) ** (0.5))


def compute_Linf_error(y_true, y_pred, domain):
    nb_vert = np.shape(y_true)[1]

    if not (isinstance(y_true, np.ndarray)):
        y_true = y_true.numpy()
    if not (isinstance(y_pred, np.ndarray)):
        y_pred = y_pred.numpy()

    y_true = y_true * domain[:, :]
    y_pred = y_pred * domain[:, :]
    y_true = np.reshape(y_true, (nb_vert, nb_vert))
    y_pred = np.reshape(y_pred, (nb_vert, nb_vert))

    nb_vert = y_true.shape[1]
    boxmesh = df.RectangleMesh(
        df.Point(0.0, 0.0), df.Point(1.0, 1.0), nb_vert - 1, nb_vert - 1
    )
    V = df.FunctionSpace(boxmesh, "CG", 1)
    v2d = df.vertex_to_dof_map(V)
    y_true_FEniCS = df.Function(V)
    y_true_FEniCS.vector()[v2d] = y_true.flatten()
    y_pred_FEniCS = df.Function(V)
    y_pred_FEniCS.vector()[v2d] = y_pred.flatten()

    diff = np.absolute(y_pred_FEniCS - y_true_FEniCS)
    return max(diff)


def border(domain):
    nb_vert = domain.shape[0]
    res = np.zeros(domain.shape)
    for i in range(1, nb_vert - 1):
        for j in range(1, nb_vert - 1):
            if domain[i, j] == 1 and (
                domain[i + 1, j] == 0
                or domain[i - 1, j] == 0
                or domain[i, j + 1] == 0
                or domain[i, j - 1] == 0
                or domain[i - 1, j - 1] == 0
                or domain[i - 1, j + 1] == 0
                or domain[i + 1, j - 1] == 0
                or domain[i + 1, j + 1] == 0
            ):
                res[i, j] = 1
    return res


def call_Omega(phi):
    return phi <= 0.0


def omega_mask(nb_vert, phi):
    F = call_Omega(phi)
    F = np.reshape(F, [nb_vert, nb_vert])
    return F.astype(np.float64)


def domain_and_border(nb_vert, phi):
    domain = omega_mask(nb_vert, phi)
    boundary = border(domain)
    return domain, boundary


def create_circle_and_boundary(nb_vert, phi):
    domains, boundaries = np.zeros(np.shape(phi)), np.zeros(np.shape(phi))
    for i in range(np.shape(phi)[0]):
        domain, boundary = domain_and_border(nb_vert, phi[i])
        domains[i], boundaries[i] = domain, boundary

    return domains, boundaries


def generate_random_params(nb_data, nb_vert):
    F, Phi, params = create_FG_numpy(nb_data=nb_data, nb_vert=nb_vert)
    return params


def generate_phi_numpy(x_0, y_0, lx, ly, theta, nb_vert):
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])
    phi = call_phi(np, XXYY, x_0, y_0, lx, ly, theta)
    if isinstance(x_0, np.ndarray):
        phi = np.reshape(phi, [np.shape(x_0), nb_vert, nb_vert])
    else:
        phi = np.reshape(phi, [1, nb_vert, nb_vert])
    return phi  # np.flip(phi, axis=1)


def generate_F_numpy(mu0, mu1, sigma, nb_vert):
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])
    F = call_F(np, XXYY, mu0, mu1, sigma)
    if isinstance(mu0, np.ndarray):
        F = np.reshape(F, [np.shape(mu0), nb_vert, nb_vert])
    else:
        F = np.reshape(F, [1, nb_vert, nb_vert])
    return F  # np.flip(F, axis=1)


def generate_manual_new_data_numpy(phi, F, dtype=tf.float32):
    nb_vert = F.shape[1]
    nb_data = F.shape[0]
    phi = np.reshape(phi, [nb_data, nb_vert, nb_vert])
    F = np.reshape(F, [nb_data, nb_vert, nb_vert])

    domain, boundary = create_circle_and_boundary(nb_vert, phi)
    domain, boundary = tf.constant(domain, dtype=dtype), tf.constant(
        boundary, dtype=dtype
    )

    F = tf.constant(F, dtype=dtype)
    phi = tf.constant(phi, dtype=dtype)

    # enrichissement des données
    Fx = tf.cumsum(F, axis=1) / nb_vert
    Fy = tf.cumsum(F, axis=2) / nb_vert
    Fxx = tf.cumsum(Fx, axis=1) / nb_vert
    Fyy = tf.cumsum(Fy, axis=2) / nb_vert

    X = tf.stack([F, phi, Fx, Fy, Fxx, Fyy, domain], axis=3)
    return X


if __name__ == "__main__":
    X = np.reshape(np.cumsum(np.ones((20, 20))), [20, 20])
    X_fenics = convert_numpy_matrix_to_fenics(X)
