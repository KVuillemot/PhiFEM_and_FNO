import numpy as np
import os
from mpi4py import MPI
import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh
import matplotlib.pyplot as plt
import ufl
from dolfinx.fem.petsc import LinearProblem
from ufl import inner, jump, grad, div, dot, avg
import random
from prepare_data import *
import typing
from utils import *
import seaborn as sns
import multiphenicsx as mphx
import multiphenicsx.fem
import multiphenicsx.fem.petsc as petsc
import petsc4py.PETSc
import pygmsh
import gmsh
import meshio
import time
import torch
import pyvista
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import dolfinx as dfx
from matplotlib.patches import Ellipse
from utils_plot import *

# pyvista.global_theme.jupyter_backend = "html"
# pyvista.global_theme.notebook = False


np.pow = np.power

seed = 2102
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
# print(f"Random seed set as {seed}")

model = "NH"  # "MR" "SV" "NH"

degV = 2
degPhi = degV + 1


def tensors(u):
    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))
    E = 0.97
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
    P = ufl.diff(psi, F)
    return P


class Phi_i:
    def __init__(self, x_0, y_0, lx) -> None:
        self.x_0 = x_0
        self.y_0 = y_0
        self.lx = lx

    def eval(self, x):
        return -(-(self.lx**2) + (x[0] - self.x_0) ** 2 + (x[1] - self.y_0) ** 2)

    def omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) <= 3e-16

    def not_omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) > 0.0


class Phi:
    def __init__(self, holes_params) -> None:
        self.holes_params = holes_params

    def eval(self, x):
        return call_phi(x, self.holes_params)

    def omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) <= 3e-16

    def not_omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) > 0.0


class GExpr:
    def __init__(self, gamma_G):
        self.gamma_G = gamma_G

    def eval(self, x):
        return call_G(x, self.gamma_G)


def near(a, b, tol=3e-16):
    """
    Check if two numbers 'a' and 'b' are close to each other within a tolerance 'tol'.
    """
    return np.abs(a - b) <= tol


class NonLinearPhiFEM:
    """Define a nonlinear problem, interfacing with SNES."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        F: list[ufl.Form],
        J: list[list[ufl.Form]],
        solutions,
        bcs: list[dolfinx.fem.DirichletBC],
        restriction,
        spaces,
        P: typing.Optional[list[list[ufl.Form]]] = None,
    ) -> None:
        self._F = dolfinx.fem.form(F)
        self._J = dolfinx.fem.form(J)
        self._restriction = restriction
        self._obj_vec = multiphenicsx.fem.petsc.create_vector_block(
            self._F, restriction
        )
        self._solutions = solutions
        self._spaces = spaces
        self._bcs = bcs
        self._P = P

    def create_snes_solution(self) -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
        """
        Create a petsc4py.PETSc.Vec to be passed to petsc4py.PETSc.SNES.solve.

        The returned vector will be initialized with the initial guesses provided in `self._solutions`,
        properly stacked together and restricted in a single block vector.
        """
        x = multiphenicsx.fem.petsc.create_vector_block(
            self._F, restriction=self._restriction
        )
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            x, [VV.dofmap for VV in self._spaces], self._restriction
        ) as x_wrapper:
            for x_wrapper_local, sub_solution in zip(x_wrapper, self._solutions):
                with sub_solution.x.petsc_vec.localForm() as sub_solution_local:
                    x_wrapper_local[:] = sub_solution_local
        return x

    def update_solutions(self, x: petsc4py.PETSc.Vec) -> None:  # type: ignore[no-any-unimported]
        """Update `self._solutions` with data in `x`."""
        x.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.INSERT,
            mode=petsc4py.PETSc.ScatterMode.FORWARD,
        )
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            x, [VV.dofmap for VV in self._spaces], self._restriction
        ) as x_wrapper:
            for x_wrapper_local, sub_solution in zip(x_wrapper, self._solutions):
                with sub_solution.x.petsc_vec.localForm() as sub_solution_local:
                    sub_solution_local[:] = x_wrapper_local

    def obj(  # type: ignore[no-any-unimported]
        self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec
    ) -> np.float64:
        """Compute the norm of the residual."""
        self.F(snes, x, self._obj_vec)
        return self._obj_vec.norm()  # type: ignore[no-any-return]

    def F(  # type: ignore[no-any-unimported]
        self,
        snes: petsc4py.PETSc.SNES,
        x: petsc4py.PETSc.Vec,
        F_vec: petsc4py.PETSc.Vec,
    ) -> None:
        """Assemble the residual."""
        self.update_solutions(x)
        with F_vec.localForm() as F_vec_local:
            F_vec_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector_block(  # type: ignore[misc]
            F_vec,
            self._F,
            self._J,
            self._bcs,
            x0=x,
            scale=-1.0,
            restriction=self._restriction,
            restriction_x0=self._restriction,
        )

    def J(  # type: ignore[no-any-unimported]
        self,
        snes: petsc4py.PETSc.SNES,
        x: petsc4py.PETSc.Vec,
        J_mat: petsc4py.PETSc.Mat,
        P_mat: petsc4py.PETSc.Mat,
    ) -> None:
        """Assemble the jacobian."""
        J_mat.zeroEntries()
        multiphenicsx.fem.petsc.assemble_matrix_block(
            J_mat,
            self._J,
            self._bcs,
            diagonal=1.0,  # type: ignore[arg-type]
            restriction=(self._restriction, self._restriction),
        )
        J_mat.assemble()
        if self._P is not None:
            P_mat.zeroEntries()
            multiphenicsx.fem.petsc.assemble_matrix_block(
                P_mat,
                self._P,
                self._bcs,
                diagonal=1.0,  # type: ignore[arg-type]
                restriction=(self._restriction, self._restriction),
            )
            P_mat.assemble()


class PhiFemSolver:
    def __init__(self, nb_cell, params):
        self.N = N = nb_cell
        self.params = params
        start = time.time()
        self.mesh_macro = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0, 0], [1, 1]]),
            np.array([N, N]),
        )
        end = time.time()
        self.init_time = end - start

    def solve_one(self, i):
        self.index = i
        """Computation of phiFEM

        Args:
            i (int): index of the problem to solve

        Returns:
            np array : matrix of the phiFEM solution
        """
        total_time = self.init_time
        param = self.params[i]
        nb_holes = int(5)
        params_holes = np.array(param[1:]).reshape((-1, 3))[:nb_holes]
        gamma_G = param[0]

        cell_dim = self.mesh_macro.geometry.dim
        facet_dim = self.mesh_macro.geometry.dim - 1
        vertices_dim = 0

        start = time.time()
        Phi_full = Phi(params_holes)
        all_entities = np.arange(
            self.mesh_macro.topology.index_map(cell_dim).size_global, dtype=np.int32
        )
        cells_outside = dfx.mesh.locate_entities(
            self.mesh_macro, cell_dim, Phi_full.not_omega()
        )
        interior_entities_macro = np.setdiff1d(all_entities, cells_outside)

        mesh = dfx.mesh.create_submesh(
            self.mesh_macro, self.mesh_macro.topology.dim, interior_entities_macro
        )[0]
        end = time.time()
        total_time += end - start

        V = dfx.fem.functionspace(mesh, ("CG", degV, (cell_dim,)))
        V_phi = dfx.fem.functionspace(mesh, ("CG", degPhi))
        Z_N = dfx.fem.functionspace(mesh, ("CG", degV, (cell_dim, cell_dim)))
        if degV == 1:
            Q_N = dfx.fem.functionspace(mesh, ("DG", degV - 1, (cell_dim,)))
        else:
            Q_N = dfx.fem.functionspace(mesh, ("CG", degV - 1, (cell_dim,)))

        dofs_V = np.arange(
            0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
        )
        spaces = [V]
        restricts = [dofs_V]
        restriction = [multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs_V)]
        neumann_cells, neumann_facets = [], []
        hole_restriction_cells = []

        mesh.topology.create_connectivity(cell_dim, facet_dim)
        c_to_f = mesh.topology.connectivity(cell_dim, facet_dim)
        mesh.topology.create_connectivity(cell_dim, vertices_dim)
        c_to_v = mesh.topology.connectivity(cell_dim, vertices_dim)
        interior_entities = np.arange(
            mesh.topology.index_map(cell_dim).size_global, dtype=np.int32
        )

        start = time.time()
        c_to_v_map = np.reshape(c_to_v.array, (-1, 3))
        assert c_to_v_map.shape[0] == len(interior_entities)
        xi, yi, li = params_holes.T
        xi, yi, li = xi.reshape((-1, 1)), yi.reshape((-1, 1)), li.reshape((-1, 1))
        points = mesh.geometry.x.T
        phi_values = call_phi_i(points, xi, yi, li)
        for j in range(nb_holes):
            phi_j = phi_values[j, :]
            phi_cells = phi_j[c_to_v_map]
            cells_boundary_j_all = (
                ((phi_cells[:, 0] * phi_cells[:, 1]) <= 0.0)
                | ((phi_cells[:, 0] * phi_cells[:, 2]) <= 0.0)
                | ((phi_cells[:, 1] * phi_cells[:, 2]) <= 0.0)
                | (near(phi_cells[:, 0] * phi_cells[:, 1], 0.0))
                | (near(phi_cells[:, 0] * phi_cells[:, 2], 0.0))
                | (near(phi_cells[:, 1] * phi_cells[:, 2], 0.0))
            )
            cells_boundary_j = np.where(cells_boundary_j_all == True)[0]
            hole_restriction_cells.append(cells_boundary_j)

        neumann_facets_measure, neumann_facets_stab_measure, neumann_cells_measure = (
            [],
            [],
            [],
        )
        neumann_values, neumann_stab_values = [], []
        c2f_map = np.reshape(c_to_f.array, (-1, 3))

        for j in range(nb_holes):
            hole_restriction_cells_j = hole_restriction_cells[j]
            if len(hole_restriction_cells_j) > 0:
                neumann_cells_j = np.unique(hole_restriction_cells_j)
                omega_1_small_cells_j = np.setdiff1d(interior_entities, neumann_cells_j)
                omega_1_small_facets_j = c2f_map[omega_1_small_cells_j].flatten()
                neumann_facets_j = np.unique(c2f_map[neumann_cells_j].flatten())
                neumann_stab_facets_j = np.unique(
                    np.intersect1d(omega_1_small_facets_j, neumann_facets_j)
                )

                neumann_facets_measure.append(neumann_facets_j)
                neumann_facets_stab_measure.append(neumann_stab_facets_j)
                neumann_cells_measure.append(neumann_cells_j)
                neumann_values.append(4 + j)
                neumann_stab_values.append(30 + j)

        neumann_cells = np.unique(np.concatenate(hole_restriction_cells))
        end = time.time()
        total_time += end - start
        mesh_neumann = dfx.mesh.create_submesh(mesh, cell_dim, neumann_cells)[0]
        # plot_meshes_pyvista([self.mesh_macro, mesh, mesh_neumann])
        restr_Neumann_Z_N = dfx.fem.locate_dofs_topological(
            Z_N, cell_dim, list(neumann_cells)
        )
        restr_Neumann_Q_N = dfx.fem.locate_dofs_topological(
            Q_N, cell_dim, list(neumann_cells)
        )
        restricts.append(restr_Neumann_Z_N)
        restricts.append(restr_Neumann_Q_N)

        restriction.append(
            multiphenicsx.fem.DofMapRestriction(Z_N.dofmap, restr_Neumann_Z_N)
        )
        restriction.append(
            multiphenicsx.fem.DofMapRestriction(Q_N.dofmap, restr_Neumann_Q_N)
        )
        spaces.append(Z_N)
        spaces.append(Q_N)

        start = time.time()
        # create meshtags for cells
        full_neumann_values = []
        for j in range(len(neumann_cells_measure)):
            values_Neumann = neumann_values[j] * np.ones(
                len(neumann_cells_measure[j]), dtype=np.intc
            )
            full_neumann_values.append(values_Neumann)

        values_cells = np.hstack(full_neumann_values)
        entities_cells = np.hstack(neumann_cells_measure)
        sorted_cells = np.argsort(entities_cells)

        subdomains_cell = dfx.mesh.meshtags(
            mesh,
            cell_dim,
            entities_cells[sorted_cells],
            values_cells[sorted_cells],
        )

        full_neumann_values_facets = []
        for j in range(len(neumann_facets_measure)):
            values_Neumann = neumann_values[j] * np.ones(
                len(neumann_facets_measure[j]), dtype=np.intc
            )
            full_neumann_values_facets.append(values_Neumann)

            values_Neumann_stab = neumann_stab_values[j] * np.ones(
                len(neumann_facets_stab_measure[j]), dtype=np.intc
            )
            full_neumann_values.append(values_Neumann_stab)

        values_facets = np.hstack(full_neumann_values_facets)
        entities_facets = np.hstack(neumann_facets_measure)
        sorted_facets = np.argsort(entities_facets)

        subdomains_facet = dfx.mesh.meshtags(
            mesh,
            facet_dim,
            entities_facets[sorted_facets],
            values_facets[sorted_facets],
        )

        end = time.time()
        total_time += end - start

        V_phi = dfx.fem.functionspace(mesh, ("CG", degPhi))

        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=subdomains_cell,
            metadata={"quadrature_degree": 4},
        )
        ds = ufl.Measure(
            "ds",
            domain=mesh,
            subdomain_data=subdomains_facet,
            metadata={"quadrature_degree": 4},
        )
        dS = ufl.Measure(
            "dS",
            domain=mesh,
            subdomain_data=subdomains_facet,
            metadata={"quadrature_degree": 4},
        )
        nb_incr = 3
        gamma_div, gamma_u, gamma_p = 0.01, 0.001, 0.01
        sigma_N = 0.01
        # gamma_div, gamma_u, gamma_p = 1.0, 0.01, 0.01
        # sigma_N = 0.01

        uyp_split = [dolfinx.fem.Function(VVV) for VVV in spaces]
        vzq_split = [ufl.TestFunction(VVV) for VVV in spaces]
        duyp_split = [ufl.TrialFunction(VVV) for VVV in spaces]
        Phis_list = []

        gg = np.linspace(0, gamma_G, num=nb_incr + 1)[1:]
        ii = 0
        for ind in range(nb_holes):
            xi, yi, li = params_holes[ind]
            phi_expr = Phi_i(xi, yi, li)
            phi = dolfinx.fem.Function(V_phi)
            phi.interpolate(phi_expr.eval)
            Phis_list.append(phi)

        start_solve = time.time()
        while ii < len(gg):
            print(f"Increment : {ii} / {nb_incr}")
            u1 = uyp_split[0]
            v1 = vzq_split[0]
            y, p_N = uyp_split[1], uyp_split[1 + 1]
            z, q_N = vzq_split[1], vzq_split[1 + 1]
            Pu1 = tensors(u1)
            Pv1 = ufl.derivative(Pu1, u1, v1)

            F = [0.0 for k in range(len(spaces))]

            dx_full_omega_1 = dx

            au1v1 = ufl.inner(Pu1, ufl.grad(v1)) * dx_full_omega_1
            F[0] += au1v1

            index_ = 1
            for ind in range(nb_holes):
                phi = Phis_list[ind]

                dx_Neumann_omega_1 = dx(neumann_values[ind])
                ds_Neumann_omega_1 = ds(neumann_values[ind])
                dS_Neumann = dS(neumann_values[ind])
                dS_Neumann_stab = dS(neumann_stab_values[ind])

                Gh1 = (
                    sigma_N
                    * ufl.avg(h)
                    * ufl.inner(ufl.jump(Pu1, n), ufl.jump(Pv1, n))
                    * dS_Neumann_stab
                )

                F[0] += gamma_u * ufl.inner(Pu1, Pv1) * dx_Neumann_omega_1 + Gh1

                dsi = ds_Neumann_omega_1
                dxi = dx_Neumann_omega_1
                F[0] += (
                    ufl.inner(ufl.dot(y, n), v1) * dsi
                    + gamma_u * ufl.inner(y, Pv1) * dxi
                )
                P_u = Pu1

                F[index_] += gamma_u * ufl.inner(P_u, z) * dxi

                F[index_] += (
                    gamma_u * ufl.inner(y, z) * dxi
                    + gamma_div * ufl.inner(ufl.div(y), ufl.div(z)) * dxi
                    + gamma_p
                    * h ** (-2)
                    * ufl.inner(ufl.dot(y, ufl.grad(phi)), ufl.dot(z, ufl.grad(phi)))
                    * dxi
                )
                F[index_] += (
                    gamma_p
                    * h ** (-3)
                    * ufl.inner(p_N * phi, ufl.dot(z, ufl.grad(phi)))
                    * dxi
                )

                F[index_ + 1] += (
                    gamma_p
                    * h ** (-3)
                    * ufl.inner(ufl.dot(y, ufl.grad(phi)), q_N * phi)
                    * dxi
                )

                F[index_ + 1] += (
                    gamma_p * h ** (-4) * ufl.inner(p_N * phi, q_N * phi) * dxi
                )

            J = [
                [
                    ufl.derivative(F[i], uyp_split[j], duyp_split[j])
                    for j in range(len(uyp_split))
                ]
                for i in range(len(F))
            ]

            g_expr = GExpr(gg[ii])
            g = dolfinx.fem.Function(V)
            g.interpolate(g_expr.eval)

            upper_facets = dolfinx.mesh.locate_entities_boundary(
                mesh, 1, lambda x: np.isclose(x[1], 1.0)
            )
            boundary_dofs_up = fem.locate_dofs_topological(V, 1, upper_facets)
            bc_up = fem.dirichletbc(g, boundary_dofs_up)

            g_expr_null = GExpr(0.0 * gamma_G)
            g_null = dolfinx.fem.Function(V)
            g_null.interpolate(g_expr_null.eval)
            lower_facets = dolfinx.mesh.locate_entities_boundary(
                mesh, 1, lambda x: np.isclose(x[1], 0.0)
            )
            boundary_dofs_low = fem.locate_dofs_topological(V, 1, lower_facets)
            bc_low = fem.dirichletbc(g_null, boundary_dofs_low)
            problem = NonLinearPhiFEM(
                F, J, tuple(uyp_split), [bc_low, bc_up], restriction, spaces
            )
            F_vec = mphx.fem.petsc.create_vector_block(
                problem._F, restriction=restriction
            )
            J_mat = mphx.fem.petsc.create_matrix_block(
                problem._J, restriction=(restriction, restriction)
            )
            snes = petsc4py.PETSc.SNES().create(mesh.comm)
            snes.setTolerances(max_it=50)
            snes.getKSP().setType("preonly")
            snes.getKSP().getPC().setType("lu")
            snes.getKSP().getPC().setFactorSolverType("mumps")
            snes.setObjective(problem.obj)
            snes.setFunction(problem.F, F_vec)
            snes.setJacobian(problem.J, J=J_mat, P=None)
            snes.setMonitor(lambda _, it, residual: print(it, residual))
            solution = problem.create_snes_solution()
            snes.solve(None, solution)
            converged_reason = snes.getConvergedReason()
            converged = converged_reason >= 0
            if converged:
                problem.update_solutions(solution)
                ii += 1
            else:
                ii = 0
                nb_incr += 1
                uyp_split = [dolfinx.fem.Function(VVV) for VVV in spaces]
                vzq_split = [ufl.TestFunction(VVV) for VVV in spaces]
                duyp_split = [ufl.TrialFunction(VVV) for VVV in spaces]
                gg = np.linspace(0, gamma_G, num=nb_incr + 1)[1:]
                start_solve = time.time()
        end_solve = time.time()
        total_time += end_solve - start_solve
        return uyp_split[0], V, total_time


class StandardMesh:
    def __init__(self, param):
        self.params_holes = np.array(param[1:]).reshape((-1, 3))

    def construct_mesh(self, size, create_new_mesh=False, mesh_name="mesh"):
        if (not os.path.exists(f"./{mesh_name}.xdmf")) or create_new_mesh:

            geo = pygmsh.occ.Geometry()
            model = geo.__enter__()
            model.characteristic_length_min = size
            model.characteristic_length_max = size
            full_plate = model.add_rectangle([0, 0, 0], 1, 1, mesh_size=size)
            to_remove = []
            for i in range(5):
                Cix, Ciy, li = self.params_holes[i]
                disque = geo.add_disk([Cix, Ciy, 0.0], li)
                to_remove.append(disque)
            plate = geo.boolean_difference(
                full_plate,
                geo.boolean_union(to_remove),
            )

            geo.generate_mesh(dim=2)

            gmsh.write(f"{mesh_name}.mesh")
            gmsh.clear()
            geo.__exit__()

            in_mesh = meshio.read(f"{mesh_name}.mesh")
            out_mesh = self._construct_dolfinx_mesh(in_mesh, "triangle", True)
            meshio.write(f"{mesh_name}.xdmf", out_mesh)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{mesh_name}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")

        return mesh

    def _construct_dolfinx_mesh(self, mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        if prune_z:
            points = mesh.points[:, :2]
        else:
            points = mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
        return out_mesh


class StandardFEMSolver:

    def __init__(self, params):
        self.params = params

    def solve_one(self, i, size, nb_vert=64, reference_fem=False, force_deg2=False):
        param = self.params[i]
        gamma_G = param[0]

        start = time.time()
        Mesh_Constructor = StandardMesh(param)
        mesh = Mesh_Constructor.construct_mesh(size, True)
        end = time.time()
        total_time = end - start
        V = dolfinx.fem.functionspace(mesh, ("CG", degV, (2,)))
        v = ufl.TestFunction(V)
        u = dolfinx.fem.Function(V)
        dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})
        nb_incr = 3
        ii = 1

        start_solve = time.time()
        while ii < nb_incr + 1:
            print(f"{ii=}")
            print(f"{nb_incr=}")
            P_u = tensors(u)

            F = ufl.inner(ufl.grad(v), P_u) * dx
            alpha = ii / ((nb_incr))
            print(f"{alpha=}")
            g_expr = GExpr(alpha * gamma_G)
            g = dolfinx.fem.Function(V)
            g.interpolate(g_expr.eval)

            upper_facets = dolfinx.mesh.locate_entities_boundary(
                mesh, 1, lambda x: np.isclose(x[1], 1.0)
            )
            boundary_dofs_up = fem.locate_dofs_topological(V, 1, upper_facets)
            bc_up = fem.dirichletbc(g, boundary_dofs_up)

            g_expr_null = GExpr(0.0 * gamma_G)
            g_null = dolfinx.fem.Function(V)
            g_null.interpolate(g_expr_null.eval)
            lower_facets = dolfinx.mesh.locate_entities_boundary(
                mesh, 1, lambda x: np.isclose(x[1], 0.0)
            )
            boundary_dofs_low = fem.locate_dofs_topological(V, 1, lower_facets)
            bc_low = fem.dirichletbc(g_null, boundary_dofs_low)

            bcs = [bc_up, bc_low]
            problem = NonlinearProblem(F, u, bcs)
            solver = NewtonSolver(mesh.comm, problem)

            # Set Newton solver options
            solver.atol = 1e-8
            solver.rtol = 1e-8
            solver.convergence_criterion = "incremental"
            solver.error_on_nonconvergence = False
            num_its, converged = solver.solve(u)
            u.x.scatter_forward()
            if converged:
                ii += 1
            else:
                ii = 1
                nb_incr += 1
                v = ufl.TestFunction(V)
                u = dolfinx.fem.Function(V)
                start_solve = time.time()
        end_solve = time.time()

        total_time += end_solve - start_solve
        if reference_fem:
            # V_p1 = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))
            # sol_P1 = dolfinx.fem.Function(V_p1)
            # sol_P1.interpolate(u)
            return u, V  # sol_P1, V_p1
        else:
            return u, V, total_time


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
