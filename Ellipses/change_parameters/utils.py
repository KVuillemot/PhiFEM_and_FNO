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
    call_G,
)


def smooth_padding(W: tf.Tensor, pad_size):
    """
    Smoothly pad the given tensor along its last axis.

    Parameters:
        W (tf.Tensor): Input tensor.
        pad_size (int or tuple): Number of elements to pad on both sides of the last axis.

    Returns:
        tf.Tensor: Padded tensor.
    """
    if isinstance(pad_size, tuple) or isinstance(pad_size, list):
        pad_left = pad_size[0]
        pad_right = pad_size[1]
    else:
        pad_left = pad_size
        pad_right = pad_size

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
    """
    Pad the given tensor along the specified axis.

    Parameters:
        W (tf.Tensor): Input tensor.
        pad_size (int or tuple): Number of elements to pad on both sides of the specified axis.
        axis (int): Axis along which to pad.

    Returns:
        tf.Tensor: Padded tensor.
    """
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
    """
    Slice the given tensor at a specific axis with the provided parameters.

    Parameters:
        W (tf.Tensor): Input tensor.
        begin (int): Starting index for the slice.
        size (int): Size of the slice.
        axis (int): Axis along which to perform the slice.

    Returns:
        tf.Tensor: Sliced tensor.
    """
    beg = [0 for _ in W.shape]
    sizes = [i for i in W.shape]
    beg[axis] = begin % W.shape[axis]
    sizes[axis] = size
    res = tf.slice(W, beg, sizes)
    return res


class Derivator_fd:
    """
    Finite differences derivator for computing derivatives of a tensor.

    Parameters:
        derivative_symbols (list or dict): List or dictionary of derivative symbols.
        interval_lenghts (list): List of interval lengths for each axis.
        axes (list): List of axes along which to compute the derivatives.
        centred (bool, optional): Whether to use centred finite differences. Default is True.
        keep_size (bool, optional): Whether to keep the tensor size during differentiation. Default is True.

    For example, assume that the tensor to derivate U is of dim 3. axis 1 correspond to x, axis 2 to Y.
    So args must be : axes=[1,2].
    If derivative_symbols=[(0,0),(0,1),(1,1)] the __call__ method will give
                            [U_xx, U_xy, U_yy ]

    If derivative_symbols={"xx":(0,0),"xy":(0,1),"yy":(1,1)} the __call__ method will give
                            {"xx":U_xx, "xy":U_xy, "yy":U_yy}
    """

    def __init__(
        self,
        derivative_symbols,
        interval_lenghts,
        axes,
        centred=True,
        keep_size=True,
    ):
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
        """
        Compute the derivative of the input tensor along the specified axis.

        Parameters:
            U (tf.Tensor): Input tensor.
            i (int): Index of the axis along which to compute the derivative.

        Returns:
            tf.Tensor: Tensor containing the computed derivative.
        """
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

        der = self.D_axis(previous_der, symbol[-1])
        res[symbol] = der
        return der

    def __call__(self, U):
        """
        Compute derivatives of the input tensor using the specified symbols and axes.

        Parameters:
            U (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor or dict: Tensor or dictionary containing the computed derivatives.
        """
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
    """
    Convert a numpy matrix to a FEniCS function.

    Parameters:
        X (array): Input array of size nb_dof_x x nb_dof_y.
        nb_vert (int): Number of vertices in each direction.
        degree (int, optional): Degree of the FEniCS function. Default is 2.

    Returns:
        function FEniCS: Output function that can be used with FEniCS.
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


def border(domain):
    """
    Determine the pixels on the boundary of a given domain.

    Parameters:
        domain (array): Array defining the domain (0 outside, 1 inside).

    Returns:
        array: Array with 1 at boundary pixels, 0 otherwise.
    """
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
    """
    Call function to determine the pixels inside the domain.

    Parameters:
        phi (array): Values of the level-set function.

    Returns:
        array: Binary array with 1 inside the domain and 0 outside.
    """
    return phi <= 0.0


def omega_mask(nb_vert, phi):
    """
    Generate a binary mask for the domain using the level-set function.

    Parameters:
        nb_vert (int): Number of pixels.
        phi (array): Values of the level-set function.

    Returns:
        array: Binary mask for the domain.
    """
    F = call_Omega(phi)
    F = np.reshape(F, [nb_vert, nb_vert])
    return F.astype(np.float64)


def domain_and_border(nb_vert, phi):
    """
    Compute the domain and boundary masks for the given level-set function.

    Parameters:
        nb_vert (int): Number of pixels.
        phi (array): Values of the level-set function.

    Returns:
        array, array: Binary masks for the domain and the boundary.
    """
    domain = omega_mask(nb_vert, phi)
    boundary = border(domain)
    return domain, boundary


def create_domain_and_boundary(nb_vert, phi):
    """
    Create domains and boundary masks for a set of level-set functions.

    Parameters:
        nb_vert (int): Number of pixels.
        phi (array): Values of the level-set functions.

    Returns:
        array, array: Arrays containing the domains and boundary masks.
    """
    domains, boundaries = np.zeros(np.shape(phi)), np.zeros(np.shape(phi))
    for i in range(np.shape(phi)[0]):
        domain, boundary = domain_and_border(nb_vert, phi[i])
        domains[i], boundaries[i] = domain, boundary

    return domains, boundaries


def generate_random_params(nb_data, nb_vert):
    """
    Generate random parameters.

    Parameters:
        nb_data (int): Number of data samples.
        nb_vert (int): Number of pixels.

    Returns:
        array: Array of random parameters for F, phi and G.
    """
    F, Phi, G, params = create_FG_numpy(nb_data=nb_data, nb_vert=nb_vert)
    return params


def generate_phi_numpy(x_0, y_0, lx, ly, theta, nb_vert):
    """
    Generate a level-set function using the given parameters.

    Parameters:
        x_0 (float): x position of the center.
        y_0 (float): y position of the center.
        lx (float): width of the domain.
        ly (float): height of the domain.
        theta (float): angle of rotation (radians, rotation of center (x_0, y_0)).
        nb_vert (int): number of pixels.

    Returns:
        array: Values of the level-set function.
    """
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])
    phi = call_phi(XXYY, x_0, y_0, lx, ly, theta)
    if isinstance(x_0, np.ndarray):
        phi = np.reshape(phi, [np.shape(x_0), nb_vert, nb_vert])
    else:
        phi = np.reshape(phi, [1, nb_vert, nb_vert])
    return phi


def generate_F_numpy(mu0, mu1, sigma, nb_vert):
    """
    Generate a force field (gaussian distribution) using the given parameters.

    Parameters:
        mu0 (float): x position of the center.
        mu1 (float): y position of the center.
        sigma (float): Standard deviation of the gaussian.
        nb_vert (int): Number of pixels.

    Returns:
        array: Values of the force field.
    """
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])
    F = call_F(XXYY, mu0, mu1, sigma)
    if isinstance(mu0, np.ndarray):
        F = np.reshape(F, [np.shape(mu0)[0], nb_vert, nb_vert])
    else:
        F = np.reshape(F, [1, nb_vert, nb_vert])
    return F


def generate_G_numpy(alpha, beta, nb_vert):
    """Generation of a boundary condition using the given parameters.

    Args:
        alpha (float)
        beta (float)
        nb_vert (int): number of pixels

    Returns:
        array: values of the function.
    """
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])
    G = call_G(XXYY, alpha, beta)
    if isinstance(alpha, np.ndarray):
        G = np.reshape(G, [np.shape(alpha)[0], nb_vert, nb_vert])
    else:
        G = np.reshape(G, [1, nb_vert, nb_vert])
    return G


def generate_manual_new_data_numpy(phi, F, G, dtype=tf.float32):
    """
    Generate input vectors for a FNO model using the provided phi, F, and G.

    Parameters:
        phi (array): Values of the level-set functions.
        F (array): Values of the force fields.
        G (array): Values of the boundary conditions.
        dtype (type, optional): Type of TensorFlow float. Default is tf.float32.

    Returns:
        tensor: Tensor of shape (nb_data, nb_vert, nb_vert, 4).
    """
    nb_vert = F.shape[1]
    nb_data = F.shape[0]
    phi = np.reshape(phi, [nb_data, nb_vert, nb_vert])
    F = np.reshape(F, [nb_data, nb_vert, nb_vert])
    G = np.reshape(G, [nb_data, nb_vert, nb_vert])

    domain, boundary = create_domain_and_boundary(nb_vert, phi)
    domain, boundary = tf.constant(domain, dtype=dtype), tf.constant(
        boundary, dtype=dtype
    )

    F = tf.constant(F, dtype=dtype)
    phi = tf.constant(phi, dtype=dtype)
    G = tf.constant(G, dtype=dtype)

    X = tf.stack([F, phi, G, domain], axis=3)
    return X
