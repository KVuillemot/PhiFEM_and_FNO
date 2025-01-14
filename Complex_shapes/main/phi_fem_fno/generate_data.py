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


class PhiFemSolver:
    def __init__(self, nb_cell, params):
        self.N = N = nb_cell
        self.params = params

        self.mesh_macro = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0, 0], [1, 1]]),
            np.array([N, N]),
        )
        self.V_macro = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 1))
        coords = self.V_macro.tabulate_dof_coordinates()
        coords = coords.reshape((-1, 3))
        coords = coords[:, :2]
        self.sorted_indices = np.lexsort((coords[:, 0], coords[:, 1]))

        self.padding = 1e-14

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

        self.mesh_macro.topology.create_connectivity(0, 2)
        cell_dim = self.mesh_macro.geometry.dim
        facet_dim = self.mesh_macro.geometry.dim - 1
        if cell_dim > 2:
            edges_dim = self.mesh_macro.geometry.dim - 2
        vertices_dim = 0
        Phi = PhiExpr(mu_x, mu_y, sigma_x_phi, sigma_y_phi)
        all_entities = np.arange(
            self.mesh_macro.topology.index_map(2).size_global, dtype=np.int32
        )
        cells_outside = dolfinx.mesh.locate_entities(
            self.mesh_macro, 2, Phi.not_omega()
        )
        interior_entities_macro = np.setdiff1d(all_entities, cells_outside)
        mesh = dolfinx.mesh.create_submesh(
            self.mesh_macro, self.mesh_macro.topology.dim, interior_entities_macro
        )[0]

        mesh.topology.create_connectivity(cell_dim, facet_dim)
        c_to_f = mesh.topology.connectivity(cell_dim, facet_dim)
        mesh.topology.create_connectivity(cell_dim, vertices_dim)
        c_to_v = mesh.topology.connectivity(cell_dim, vertices_dim)
        interior_entities = np.arange(
            mesh.topology.index_map(cell_dim).size_global, dtype=np.int32
        )

        # Selection of the boundary cells (to construct the restrictions and the measures)
        c_to_v_map = np.reshape(c_to_v.array, (-1, cell_dim + 1))
        assert c_to_v_map.shape[0] == len(interior_entities)
        points = mesh.geometry.x.T
        phi_values = call_phi(points, mu_x, mu_y, sigma_x_phi, sigma_y_phi).flatten()
        phi_j = phi_values[:]
        phi_cells = phi_j[c_to_v_map]
        all_cells = (
            ((phi_cells[:, 0] * phi_cells[:, 1]) <= 0.0)
            | ((phi_cells[:, 0] * phi_cells[:, 2]) <= 0.0)
            | ((phi_cells[:, 1] * phi_cells[:, 2]) <= 0.0)
            | (near(phi_cells[:, 0] * phi_cells[:, 1], 0.0))
            | (near(phi_cells[:, 0] * phi_cells[:, 2], 0.0))
            | (near(phi_cells[:, 1] * phi_cells[:, 2], 0.0))
        )
        boundary_cells = np.where(all_cells == True)[0]
        boundary_cells = np.unique(boundary_cells)
        c_to_f_map = np.reshape(c_to_f.array, (-1, 3))
        boundary_facets = np.unique(c_to_f_map[boundary_cells].flatten())
        sorted_facets = np.argsort(boundary_facets)
        sorted_cells = np.argsort(boundary_cells)

        Dirichlet = 1

        values_boundary = Dirichlet * np.ones(len(boundary_cells), dtype=np.intc)
        subdomains_cell = dolfinx.mesh.meshtags(
            mesh,
            mesh.topology.dim,
            boundary_cells[sorted_cells],
            values_boundary[sorted_cells],
        )  # cells of the boundary
        values_dirichlet = Dirichlet * np.ones(len(boundary_facets), dtype=np.intc)
        subdomains_facet = dolfinx.mesh.meshtags(
            mesh,
            mesh.topology.dim - 1,
            boundary_facets[sorted_facets],
            values_dirichlet[sorted_facets],
        )  # facets of the boundary
        mesh_bd = dolfinx.mesh.create_submesh(mesh, cell_dim, boundary_cells)[0]
        V = dolfinx.fem.functionspace(mesh, ("CG", polV))
        V_phi = dolfinx.fem.functionspace(mesh, ("CG", polPhi))
        V2 = dolfinx.fem.functionspace(mesh, ("CG", 2))
        phi = dolfinx.fem.Function(V_phi)
        phi.interpolate(PhiExpr(mu_x, mu_y, sigma_x_phi, sigma_y_phi).eval)

        F_expr = FExpr(mu0, mu1, sigma_x, sigma_y, amplitude)
        f_expr = dolfinx.fem.Function(V2)
        f_expr.interpolate(F_expr.eval)

        G_expr = GExpr(alpha, beta)
        g_expr = dolfinx.fem.Function(V)
        g_expr.interpolate(G_expr.eval)

        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains_cell)
        ds = ufl.Measure("ds", domain=mesh)
        dS = ufl.Measure("dS", domain=mesh, subdomain_data=subdomains_facet)

        a = (
            inner(grad(phi * u), grad(phi * v)) * dx
            - dot(inner(grad(phi * u), n), phi * v) * ds
            + sigma_D
            * avg(h)
            * dot(
                jump(grad(phi * u), n),
                jump(grad(phi * v), n),
            )
            * dS(1)
            + sigma_D
            * h**2
            * inner(
                div(grad(phi * u)),
                div(grad(phi * v)),
            )
            * dx(1)
        )
        L = (
            f_expr * v * phi * dx
            - sigma_D * h**2 * inner(f_expr, div(grad(phi * v))) * dx(1)
            - sigma_D * h**2 * div(grad(phi * v)) * div(grad(g_expr)) * dx(1)
            - inner(grad(g_expr), grad(phi * v)) * dx
            + inner(grad(g_expr), n) * phi * v * ds
            - sigma_D * avg(h) * jump(grad(g_expr), n) * jump(grad(phi * v), n) * dS(1)
        )
        problem = LinearProblem(
            a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        wh = problem.solve()
        return self.make_matrix(wh, V)

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


def go(params, nb_test_data):
    """
    Main function to generate data.
    """
    save = True
    nb_vert = 64

    ti0 = time.time()
    solver = PhiFemSolver(
        nb_cell=nb_vert - 1,
        params=params[:-nb_test_data],
    )
    W = solver.solve_several()

    duration = time.time() - ti0
    print("duration to solve u:", duration)
    if save:
        np.save(f"../../data/W.npy", W)
        # np.save(f"../../data_test/W.npy", W[-nb_test_data:])


if __name__ == "__main__":
    params = generate_parameters(
        nb_training_data=500,
        nb_rep_training_shapes=1,
        nb_rep_avg_shape=0,
        nb_validation_data=300,
        nb_test_data=300,
    )
    print(params.shape)
    go(params, 300)
