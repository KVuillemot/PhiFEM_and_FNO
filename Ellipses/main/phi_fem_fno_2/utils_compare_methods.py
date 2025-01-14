import numpy as np
import os
import numpy as np
from mpi4py import MPI
import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh
import matplotlib.pyplot as plt
import ufl
from dolfinx.fem.petsc import LinearProblem
from ufl import inner, jump, grad, div, dot, avg
from utils_plot import plot_mesh_pyvista
import random
from prepare_data import call_phi, call_F, call_G
import time
import meshio
from matplotlib.patches import Ellipse
from pymedit import (
    P1Function,
    square,
    mmg2d,
    trunc,
)

sigma_D = 1.0
degV = 1
degPhi = degV + 1


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


class StandardFEMSolver:
    """Solver for a standard FEM resolution of a given problem."""

    def __init__(self, params):
        """
        Initialize the StandardFEMSolver.

        Parameters:
            params (ndarray): Array of parameters for the solver.
        """
        self.params = params

    def create_mesh(self, mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        if prune_z:
            points = mesh.points[:, :2]
        else:
            points = mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
        return out_mesh

    def create_standard_mesh(
        self, phi, hmax=0.05, plot_mesh_bool=False, return_times=False
    ):
        """Generation of a mesh over a domain defined by a level-set function.

        Args:
            phi (array): array of values of the level-set function
            hmax (float, optional): maximal size of cell. Defaults to 0.05.
            plot_mesh_bool (bool, optional): plot the resulting mesh or not. Defaults to False.

        Returns:
            mesh: a FEniCS mesh.
        """
        t0 = time.time()
        n = np.shape(phi)[0]
        M = square(n - 1, n - 1)
        t1 = time.time()
        construct_background_mesh = t1 - t0
        M.debug = 4  # For debugging and mmg3d output

        # Setting a P1 level set function
        phi = phi.flatten()
        t0 = time.time()
        phiP1 = P1Function(M, phi)
        t1 = time.time()
        interp_time = t1 - t0
        # Remesh according to the level set
        t0 = time.time()
        newM = mmg2d(
            M,
            hmin=hmax / 1.7,
            sol=phiP1,
            ls=True,
            verb=0,
            extra_args="-hsiz " + str(hmax / 1.41),
        )
        t1 = time.time()
        remesh_time = t1 - t0
        # Trunc the negative subdomain of the level set
        t0 = time.time()
        Mf = trunc(newM, 3)
        t1 = time.time()
        trunc_mesh = t1 - t0
        Mf.save("Thf.mesh")  # Saving in binary format
        t0 = time.time()
        t1 = time.time()
        conversion_time = t1 - t0
        t0 = time.time()

        in_mesh = meshio.read("Thf.mesh")
        out_mesh = self.create_mesh(in_mesh, "triangle", True)
        meshio.write("Thf.xdmf", out_mesh)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Thf.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
        t1 = time.time()
        fenics_read_time = t1 - t0
        plot_mesh_bool = False
        if plot_mesh_bool:
            plt.figure()
            plot_mesh_pyvista(mesh)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
        if return_times:
            return mesh, [
                construct_background_mesh,
                interp_time,
                remesh_time,
                trunc_mesh,
                conversion_time,
                fenics_read_time,
            ]
        else:
            return mesh

    def solve_one(self, i, size, nb_vert=64, reference_fem=False, force_deg2=False):
        """
        Solve a given problem using standard FEM.

        Parameters:
            i (int): Index of the problem to be solved.
            size (int): Number of vertices.
            reference_fem (bool, optional): If True, compute the reference solution on a fine mesh.

        Returns:
            tuple: If reference_fem is True, return (solution, finite element space, dx measure).
                   If reference_fem is False, return (solution, mesh size, construction time, resolution time).
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
        # construction of the mesh
        if reference_fem:
            if size >= 0.001:
                nb_vert = 512
            else:
                nb_vert = 800

            XX, YY = np.meshgrid(
                np.linspace(0.0, 1.0, nb_vert), np.linspace(0.0, 1.0, nb_vert)
            )
            XX = XX.flatten()
            YY = YY.flatten()
            XXYY = np.stack([XX, YY])
            phi = call_phi(XXYY, x_0, y_0, lx, ly, theta)
            phi = np.reshape(phi, (nb_vert, nb_vert)).T
        elif force_deg2:
            XX, YY = np.meshgrid(
                np.linspace(0.0, 1.0, 2 * nb_vert - 1),
                np.linspace(0.0, 1.0, 2 * nb_vert - 1),
            )
            XX = XX.flatten()
            YY = YY.flatten()
            XXYY = np.stack([XX, YY])
            phi = call_phi(XXYY, x_0, y_0, lx, ly, theta)
            phi = np.reshape(phi, (2 * nb_vert - 1, 2 * nb_vert - 1)).T
        else:
            XX, YY = np.meshgrid(
                np.linspace(0.0, 1.0, nb_vert),
                np.linspace(0.0, 1.0, nb_vert),
            )
            XX = XX.flatten()
            YY = YY.flatten()
            XXYY = np.stack([XX, YY])
            phi = call_phi(XXYY, x_0, y_0, lx, ly, theta)
            phi = np.reshape(phi, (nb_vert, nb_vert)).T

        start_construction = time.time()
        mesh, mesh_times = self.create_standard_mesh(
            phi=phi, hmax=size, plot_mesh_bool=False, return_times=True
        )
        end_construction = time.time()
        construction_time = end_construction - start_construction
        mesh_times.append(construction_time)

        # FunctionSpace Pk
        V = dolfinx.fem.functionspace(mesh, ("CG", degV))
        if reference_fem or force_deg2:
            V2 = dolfinx.fem.functionspace(mesh, ("CG", 2))
        else:
            V2 = dolfinx.fem.functionspace(mesh, ("CG", 1))
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
        start = time.time()
        u_h = problem.solve()
        end = time.time()

        resolution_time = end - start

        solution = dolfinx.fem.Function(V)
        solution.interpolate(u_h)

        if reference_fem:
            num_cells = (
                mesh.topology.index_map(mesh.topology.dim).size_local
                + mesh.topology.index_map(mesh.topology.dim).num_ghosts
            )
            return (
                solution,
                V,
                dx,
                max(mesh.h(2, np.array(list(range(num_cells))))),
            )

        else:
            num_cells = (
                mesh.topology.index_map(mesh.topology.dim).size_local
                + mesh.topology.index_map(mesh.topology.dim).num_ghosts
            )
            return (
                solution,
                max(mesh.h(2, np.array(list(range(num_cells))))),
                V,
                mesh_times,
                resolution_time,
            )


class PhiFemSolver_error:
    """
    Solver for computing PhiFEM solution.
    """

    def __init__(self, nb_cell, params):
        self.N = N = nb_cell
        self.params = params
        self.mesh_macro = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0, 0], [1, 1]]),
            np.array([N, N]),
        )
        self.V_macro = dolfinx.fem.functionspace(self.mesh_macro, ("CG", degV))

    def solve_one(self, i, force_deg2=False):
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

        # Construction of Omega_h
        start = time.time()
        self.mesh_macro.topology.create_connectivity(0, 2)

        Phi = PhiExpr(x_0, y_0, lx, ly, theta)
        all_entities = np.arange(
            self.mesh_macro.topology.index_map(2).size_global, dtype=np.int32
        )
        cells_outside = dolfinx.mesh.locate_entities(
            self.mesh_macro, 2, Phi.not_omega()
        )
        interior_entities_macro = np.setdiff1d(all_entities, cells_outside)
        end = time.time()
        cell_selection = end - start
        start = time.time()
        mesh = dolfinx.mesh.create_submesh(
            self.mesh_macro, self.mesh_macro.topology.dim, interior_entities_macro
        )[0]
        end = time.time()
        submesh_construction = end - start
        # End Construction of Omega_h

        # Selection of the boundary cells and facets
        start = time.time()
        mesh.topology.create_connectivity(1, 0)
        f_to_v = mesh.topology.connectivity(1, 0)
        mesh.topology.create_connectivity(2, 1)
        c_to_f = mesh.topology.connectivity(2, 1)
        mesh.topology.create_connectivity(1, 2)
        f_to_c = mesh.topology.connectivity(1, 2)
        mesh.topology.create_connectivity(1, 0)
        f_to_v = mesh.topology.connectivity(1, 0)

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
        end = time.time()
        ghost_cell_selection = end - start

        # FunctionSpaces construction and interpolations of the level-set function, rhs and boundary condtion expressions
        V = dolfinx.fem.functionspace(mesh, ("CG", degV))
        if force_deg2:
            V2 = dolfinx.fem.functionspace(mesh, ("CG", 2))
            V_phi = dolfinx.fem.functionspace(mesh, ("CG", 2))
        else:
            V2 = dolfinx.fem.functionspace(mesh, ("CG", degV))
            V_phi = dolfinx.fem.functionspace(mesh, ("CG", degV))
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

        # Modification of the measure to take boundary facets and cells into account
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains_cell)
        ds = ufl.Measure("ds", domain=mesh)
        dS = ufl.Measure("dS", domain=mesh, subdomain_data=subdomains_facet)

        # Definition of the variational formulation
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

        # Linear problem construction and resolution
        problem = LinearProblem(
            a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        start = time.time()
        wh = problem.solve()
        end = time.time()
        resolution_time = end - start

        # Post treatment of the solution
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

        num_cells = (
            mesh.topology.index_map(mesh.topology.dim).size_local
            + mesh.topology.index_map(mesh.topology.dim).num_ghosts
        )
        return (
            solution_v_phi,
            V,
            dx,
            max(mesh.h(2, np.array(list(range(num_cells))))),
            cell_selection,
            submesh_construction,
            ghost_cell_selection,
            resolution_time,
        )


def confidence_ellipse(x, y, ax, facecolor="none", **kwargs):
    """
    Construct a confidence ellipse.

    Parameters:
        x (np.ndarray): x-coordinate data points.
        y (np.ndarray): y-coordinate data points.
        ax (matplotlib axis): Axis to draw the ellipse on.
        facecolor (str, optional): Color of the ellipse. Defaults to "none".

    Returns:
        matplotlib.patches.Ellipse: The patch of the ellipse.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    ell_radius_x = np.std(x)
    ell_radius_y = np.std(y)

    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    return ax.add_patch(ellipse)
