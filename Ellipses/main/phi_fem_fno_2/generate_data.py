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
from prepare_data import call_phi, call_F, call_G
import time

np.pow = np.power
import matplotlib.pyplot as plt

seed = 221024
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
    def __init__(self, x_0, y_0, lx, ly, theta):
        self.x_0 = x_0
        self.y_0 = y_0
        self.lx = lx
        self.ly = ly
        self.theta = theta

    def eval(self, x):
        return call_phi(x, self.x_0, self.y_0, self.lx, self.ly, self.theta)

    def omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) <= 3e-16

    def not_omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) > 0.0


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
        (
            mu0,
            mu1,
            sigma_x,
            sigma_y,
            amplitude,
            x_0,
            y_0,
            lx,
            ly,
            theta,
            alpha,
            beta,
        ) = self.params[i]

        self.mesh_macro.topology.create_connectivity(0, 2)

        Phi = PhiExpr(x_0, y_0, lx, ly, theta)
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

        mesh.topology.create_connectivity(1, 0)
        f_to_v = mesh.topology.connectivity(1, 0)
        mesh.topology.create_connectivity(2, 1)
        c_to_f = mesh.topology.connectivity(2, 1)
        mesh.topology.create_connectivity(1, 2)
        f_to_c = mesh.topology.connectivity(1, 2)
        mesh.topology.create_connectivity(1, 0)
        f_to_v = mesh.topology.connectivity(1, 0)
        mesh.topology.create_connectivity(2, 0)
        c_to_v = mesh.topology.connectivity(2, 0)
        boundary_cells = []
        boundary_facets = []
        num_facets = (
            mesh.topology.index_map(mesh.topology.dim - 1).size_local
            + mesh.topology.index_map(mesh.topology.dim - 1).num_ghosts
        )

        for facet_index in range(num_facets):
            vertices = f_to_v.links(facet_index)
            p1, p2 = mesh.geometry.x[vertices]
            if call_phi(p1, x_0, y_0, lx, ly, theta) * call_phi(
                p2, x_0, y_0, lx, ly, theta
            ) <= 0.0 or near(
                call_phi(p1, x_0, y_0, lx, ly, theta)
                * call_phi(p2, x_0, y_0, lx, ly, theta),
                0.0,
            ):
                boundary_facets.append(facet_index)

        boundary_facets = np.unique(boundary_facets)
        boundary_cells = [
            f_to_c.links(boundary_facets[i]) for i in range(len(boundary_facets))
        ]
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

        V = dolfinx.fem.functionspace(mesh, ("CG", polV))
        V_phi = dolfinx.fem.functionspace(mesh, ("CG", polPhi))
        V2 = dolfinx.fem.functionspace(mesh, ("CG", 2))
        phi = dolfinx.fem.Function(V_phi)
        phi.interpolate(PhiExpr(x_0, y_0, lx, ly, theta).eval)

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
        solution_v_phi = dolfinx.fem.Function(V_phi)

        phi = dolfinx.fem.Function(V_phi)
        phi.interpolate(PhiExpr(x_0, y_0, lx, ly, theta).eval)

        G_expr = GExpr(alpha, beta)
        g_expr = dolfinx.fem.Function(V_phi)
        g_expr.interpolate(G_expr.eval)

        w_V_phi = dolfinx.fem.Function(V_phi)
        w_V_phi.interpolate(wh)
        solution_v_phi.x.array[:] = (
            phi.x.array[:] * w_V_phi.x.array[:] + g_expr.x.array[:]
        )
        return self.make_matrix(solution_v_phi, V)

    def solve_several(self):
        U = []
        nb = len(self.params)
        for i in range(nb):
            print(f"Data : {i}/{nb}")
            u = self.solve_one(i)
            U.append(u)

        return np.stack(U)


def go():
    """
    Main function to generate data.
    """
    save = True
    nb_vert = 64

    ti0 = time.time()
    params_training = np.load(f"../../data/params.npy")
    solver = PhiFemSolver(nb_cell=nb_vert - 1, params=params_training)
    U = solver.solve_several()

    duration = time.time() - ti0
    print("duration to solve u:", duration)
    if save:
        np.save(f"../../data/U_phi_fem.npy", U)


if __name__ == "__main__":
    # create_parameters()
    go()
