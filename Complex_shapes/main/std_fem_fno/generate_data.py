import numpy as np
import os
import numpy as np
from mpi4py import MPI
import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh
import matplotlib.pyplot as plt
import ufl
from dolfinx.fem.petsc import LinearProblem
from ufl import inner, jump, grad, div, dot, avg
from utils_plot import *
import random
from prepare_data import call_phi, call_F, call_G, create_parameters
import meshio
from pymedit import P1Function, square, mmg2d, trunc

import time

np.pow = np.power
import matplotlib.pyplot as plt

seed = 10122024
random.seed(seed)
np.random.seed(seed)
import torch

torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# parameter of the ghost penalty
sigma_D = 1.0
# Polynome Pk
polV = 1
polPhi = polV + 1


class PhiExpr:
    def __init__(self, mu_x, mu_y, sigma_x_phi, sigma_y_phi):
        self.mu_x, self.mu_y, self.sigma_x_phi, self.sigma_y_phi = (
            mu_x,
            mu_y,
            sigma_x_phi,
            sigma_y_phi,
        )

    def call(self, x):
        return call_phi(
            x, self.mu_x, self.mu_y, self.sigma_x_phi, self.sigma_y_phi
        ).flatten()

    def eval(self, x):
        return call_phi(
            x, self.mu_x, self.mu_y, self.sigma_x_phi, self.sigma_y_phi
        ).flatten()

    def omega(self):
        return lambda x: self.eval(x.reshape((3, -1))[:2, :]) <= 3e-16

    def not_omega(self):
        return lambda x: self.eval(x.reshape((3, -1))[:2, :]) > 0.0


class FExpr:
    def __init__(self, mu0, mu1, sigma_x, sigma_y, amplitude):
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.amplitude = amplitude

    def eval(self, x):
        return call_F(x, self.mu0, self.mu1, self.sigma_x, self.sigma_y, self.amplitude)


class GExpr:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def eval(self, x):
        return call_G(x, self.alpha, self.beta)


def near(a, b, tol=3e-16):
    """
    Check if two numbers 'a' and 'b' are close to each other within a tolerance 'tol'.
    """
    return np.abs(a - b) <= tol


class StdFEMSolver:
    def __init__(self, nb_cell, params):
        self.N = N = nb_cell
        self.params = params

        self.mesh_macro = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0, 0], [1, 1]]),
            np.array([N, N]),
        )
        num_cells = (
            self.mesh_macro.topology.index_map(self.mesh_macro.topology.dim).size_local
            + self.mesh_macro.topology.index_map(
                self.mesh_macro.topology.dim
            ).num_ghosts
        )
        self.hmax = max(self.mesh_macro.h(2, np.array(list(range(num_cells)))))

        self.V_macro = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 1))
        coords = self.V_macro.tabulate_dof_coordinates()
        coords = coords.reshape((-1, 3))
        coords = coords[:, :2]
        self.sorted_indices = np.lexsort((coords[:, 0], coords[:, 1]))

        self.padding = 2e-2

    def make_matrix(self, expr, V):
        """Convert an expression of degree k to a matrix with nodal values.

        Args:
            expr (FEniCS expression): the expression to convert

        Returns:
            np array : a matrix of size N+1 * N+1
        """

        expr.x.scatter_forward()
        u2 = dolfinx.fem.Function(self.V_macro)
        u1_2_u2_nmm_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
            u2.function_space.mesh,
            u2.function_space.element,
            V.mesh,
            padding=self.padding,
        )

        u2.interpolate(expr, nmm_interpolation_data=u1_2_u2_nmm_data)
        u2.x.scatter_forward()
        res = u2.x.array[:]
        res_sorted = res[self.sorted_indices]
        expr_sorted = np.reshape(res_sorted, [self.N + 1, self.N + 1]).T

        return expr_sorted

    def create_standard_mesh(self, phi, hmax=0.0224):
        """Generation of a mesh over a domain defined by a level-set function.

        Args:
            phi (array): array of values of the level-set function
            hmax (float, optional): maximal size of cell. Defaults to 0.05.

        Returns:
            mesh: a FEniCS mesh.
        """
        n = np.shape(phi)[0]
        M = square(n - 1, n - 1)
        M.debug = 4  # For debugging and mmg3d output
        # Setting a P1 level set function
        phi = phi.flatten()
        phiP1 = P1Function(M, phi)
        # Remesh according to the level set
        newM = mmg2d(
            M,
            hmin=hmax / 1.7,
            sol=phiP1,
            ls=True,
            verb=0,
            extra_args="-hsiz " + str(hmax / 1.41),
        )
        # Trunc the negative subdomain of the level set
        Mf = trunc(newM, 3)
        Mf.save("Thf.mesh")  # Saving in binary format

        def create_mesh(mesh, cell_type, prune_z=False):
            cells = mesh.get_cells_type(cell_type)
            if prune_z:
                points = mesh.points[:, :2]
            else:
                points = mesh.points
            out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
            return out_mesh

        in_mesh = meshio.read("Thf.mesh")
        out_mesh = create_mesh(in_mesh, "triangle", True)
        meshio.write("Thf.xdmf", out_mesh)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Thf.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")

        return mesh

    def solve_one(self, i):
        self.index = i
        """Computation of phiFEM

        Args:
            i (int): index of the problem to solve

        Returns:
            np array : matrix of the phiFEM solution
        """
        params = self.params[i]
        mu0, mu1, sigma_x, sigma_y, amplitude, alpha, beta = params[:7]
        mu_x = params[7:10].reshape([1, 3, 1])
        mu_y = params[10:13].reshape([1, 3, 1])
        sigma_x_phi = params[13:16].reshape([1, 3, 1])
        sigma_y_phi = params[16:].reshape([1, 3, 1])
        xy = np.linspace(0.0, 1.0, 127)
        XX, YY = np.meshgrid(xy, xy)
        XX = XX.flatten()
        YY = YY.flatten()
        XXYY = np.stack([XX, YY])
        phi = call_phi(XXYY, mu_x, mu_y, sigma_x_phi, sigma_y_phi)
        phi = np.reshape(phi, (127, 127)).T

        mesh = self.create_standard_mesh(phi=phi, hmax=self.hmax)

        # FunctionSpace Pk
        degV = 1
        V = dolfinx.fem.functionspace(mesh, ("CG", degV))
        V2 = dolfinx.fem.functionspace(mesh, ("CG", 2))

        F_expr = FExpr(mu0, mu1, sigma_x, sigma_y, amplitude)
        f_expr = dolfinx.fem.Function(V2)
        f_expr.interpolate(F_expr.eval)

        G_expr = GExpr(alpha, beta)
        g_expr = dolfinx.fem.Function(V)
        g_expr.interpolate(G_expr.eval)
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(fdim, tdim)
        boundary_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(g_expr, boundary_dofs)
        dx = ufl.Measure("dx", domain=mesh)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        L = f_expr * v * dx
        problem = LinearProblem(
            a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        u_h = problem.solve()
        return self.make_matrix(u_h, V)

    def solve_several(self):
        W = []
        nb = len(self.params)
        for i in range(nb):
            print(f"Data : {i}/{nb}")
            w = self.solve_one(i)
            W.append(w)

        return np.stack(W)


def generate_parameters(
    nb_training_data=0,
    nb_rep_training_shapes=1,
    nb_rep_avg_shape=0,
    nb_validation_data=0,
    nb_test_data=300,
    nb_vert=64,
    save=True,
):
    F, phi, G, params = create_parameters(
        nb_training_data=nb_training_data,
        nb_rep_avg_shape=nb_rep_avg_shape,
        nb_validation_data=nb_validation_data,
        nb_test_data=nb_test_data,
        nb_vert=nb_vert,
        n_mode=3,
        seed=10122024,
        nb_rep_training_shapes=nb_rep_training_shapes,
    )
    if save:
        if not (os.path.exists(f"../../data")):
            os.makedirs(f"../../data")
        if not (os.path.exists(f"../../data_test")):
            os.makedirs(f"../../data_test")

        np.save(f"../../data/G.npy", G[:-nb_test_data])
        np.save(f"../../data/F.npy", F[:-nb_test_data])
        np.save(f"../../data/Phi.npy", phi[:-nb_test_data])
        np.save(f"../../data/params.npy", params[:-nb_test_data])
        np.save(f"../../data_test/G.npy", G[-nb_test_data:])
        np.save(f"../../data_test/F.npy", F[-nb_test_data:])
        np.save(f"../../data_test/Phi.npy", phi[-nb_test_data:])
        np.save(f"../../data_test/params.npy", params[-nb_test_data:])

    return params


def go():
    """
    Main function to generate data.
    """
    save = True
    nb_vert = 64

    ti0 = time.time()
    params_training = np.load(f"../../data/params.npy")
    solver = StdFEMSolver(nb_cell=nb_vert - 1, params=params_training)
    U = solver.solve_several()

    duration = time.time() - ti0
    print("duration to solve u:", duration)
    if save:
        np.save(f"../../data/U_std_fem.npy", U)


if __name__ == "__main__":
    go()
