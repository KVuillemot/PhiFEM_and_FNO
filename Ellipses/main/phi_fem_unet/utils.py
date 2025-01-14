import numpy as np
import matplotlib.pyplot as plt
import random
import os
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

seed = 2023
random.seed(seed)
np.random.seed(seed)
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
from prepare_data import call_F, call_phi, call_G
import torch


def smooth_padding(W, pad_size):
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
    left = -left[..., :]
    right = -right[..., :]
    left += left_value
    right += right_value
    return torch.cat((left, W, right), dim=-1)


def pad_1d(W, pad_size, axis):
    dim = len(W.shape)
    if not (axis == -1 or axis == dim - 1):
        permutation = list(range(dim))
        permutation[-1] = axis % dim
        permutation[axis] = dim - 1

        W = W.permute(permutation)
        W = smooth_padding(W, pad_size)
        W = W.permute(permutation)
    else:
        W = smooth_padding(W, pad_size)

    return W


def slice_at_given_axis(W, begin, size, axis):
    beg = [0 for _ in W.shape]
    sizes = [i for i in W.shape]
    beg[axis] = begin % W.shape[axis]
    sizes[axis] = size
    res = W[tuple(slice(b, b + s) for b, s in zip(beg, sizes))]
    return res


class Derivator_fd:
    def __init__(
        self,
        derivative_symbols,
        interval_lengths,
        axes,
        centred=True,
        keep_size=True,
    ):
        self.axes = axes
        self.centred = centred

        self.derivative_symbols_initial = None
        if isinstance(derivative_symbols, dict):
            self.derivative_symbols_initial = derivative_symbols
            self.derivative_symbols = list(derivative_symbols.values())
        else:
            self.derivative_symbols = derivative_symbols

        self.interval_lengths = interval_lengths
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


def convert_numpy_matrix_to_fenicsx(X, nb_vert, degree=2):
    """
    Convert a numpy matrix to a FEniCSx function.

    Parameters:
        X (array): Input array of size nb_dof_x x nb_dof_y.
        nb_vert (int): Number of vertices in each direction.
        degree (int, optional): Degree of the FEniCSx function. Default is 2.

    Returns:
        dolfinx.Function: Output function that can be used with FEniCSx.
    """
    # Create a mesh over the unit square
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nb_vert - 1, nb_vert - 1)
    # Define the function space
    V = dolfinx.fem.functionspace(mesh, ("CG", 1))

    # Get the degrees of freedom (dof) coordinates and map them
    dof_coords = V.tabulate_dof_coordinates()
    dof_coords = dof_coords.reshape((-1, 3))[:, :2]
    # Sort the coordinates by y first, then by x
    sorted_indices = np.lexsort((dof_coords[:, 0], dof_coords[:, 1]))
    sorted_indices = sorted_indices.astype(np.int32)

    # Create the Function
    X_fenicsx = dolfinx.fem.Function(V)

    # Flatten and map the numpy array to the degrees of freedom in the function
    X_fenicsx.vector.setValuesLocal(sorted_indices, X.T.flatten())

    # Ensure vector is synchronized across processors (necessary in parallel)
    X_fenicsx.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    return X_fenicsx


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
    XX = XX.flatten()
    YY = YY.flatten()
    XXYY = np.stack([XX, YY])
    phi = call_phi(XXYY, x_0, y_0, lx, ly, theta)
    if isinstance(x_0, np.ndarray):
        phi = np.reshape(phi, [np.shape(x_0)[0], nb_vert, nb_vert]).transpose(0, 2, 1)
    else:
        phi = np.reshape(phi, [1, nb_vert, nb_vert]).transpose(0, 2, 1)
    return phi


def generate_F_numpy(mu0, mu1, sigma_x, sigma_y, amplitude, nb_vert):
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
    XX = XX.flatten()
    YY = YY.flatten()
    XXYY = np.stack([XX, YY])
    F = call_F(XXYY, mu0, mu1, sigma_x, sigma_y, amplitude)
    if isinstance(mu0, np.ndarray):
        F = np.reshape(F, [np.shape(mu0)[0], nb_vert, nb_vert]).transpose(0, 2, 1)
    else:
        F = np.reshape(F, [1, nb_vert, nb_vert]).transpose(0, 2, 1)
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
    XX = XX.flatten()
    YY = YY.flatten()
    XXYY = np.stack([XX, YY])
    G = call_G(XXYY, alpha, beta)
    if isinstance(alpha, np.ndarray):
        G = np.reshape(G, [np.shape(alpha)[0], nb_vert, nb_vert]).transpose(0, 2, 1)
    else:
        G = np.reshape(G, [1, nb_vert, nb_vert]).transpose(0, 2, 1)
    return G


def generate_manual_new_data_numpy(F, phi, G, dtype=torch.float32):
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

    F = torch.tensor(F, dtype=dtype)
    phi = torch.tensor(phi, dtype=dtype)
    G = torch.tensor(G, dtype=dtype)

    X = torch.stack([F, phi, G], dim=1)
    return X
