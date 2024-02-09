import numpy as np
import matplotlib.pyplot as plt
import random
import os

seed = 2023
random.seed(seed)
np.random.seed(seed)
import dolfin as df
import time
from utils import *
import mshr
from dolfin.function.expression import (
    BaseExpression,
    _select_element,
    _InterfaceExpression,
)
from prepare_data import (
    call_phi,
    call_F,
    call_G,
)
from dolfin import *
from matplotlib.patches import Ellipse
from prepare_data import eval_phi
from pymedit import (
    P1Function,
    square,
    mmg2d,
    trunc,
)

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"


degV = 1
degPhi = degV + 1


class MyUserExpression(BaseExpression):
    """JIT Expressions"""

    def __init__(self, degree, domain):
        cell = domain.ufl_cell()
        element = _select_element(family=None, cell=cell, degree=degree, value_shape=())
        self._cpp_object = _InterfaceExpression(self, ())

        BaseExpression.__init__(
            self,
            cell=cell,
            element=element,
            domain=domain,
            name=None,
            label=None,
        )

    def __floordiv__(self, other):
        pass

    def value_shape(self):
        return ()


class FExpr(MyUserExpression):
    def __init__(self, mu0, mu1, sigma_x, sigma_y, amplitude, degree, domain):
        super().__init__(degree, domain)
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.amplitude = amplitude

    def eval(self, value, x):
        value[0] = call_F(
            x, self.mu0, self.mu1, self.sigma_x, self.sigma_y, self.amplitude
        )


class GExpr(MyUserExpression):
    def __init__(self, alpha, beta, degree, domain):
        super().__init__(degree, domain)
        self.alpha = alpha
        self.beta = beta

    def eval(self, value, x):
        value[0] = call_G(x, self.alpha, self.beta)


class PhiExpr(MyUserExpression):
    def __init__(self, coeffs_ls, modes, threshold, degree, domain):
        super().__init__(degree, domain)
        self.coeffs_ls = coeffs_ls
        self.modes = modes
        self.threshold = threshold

    def eval(self, value, x):
        basis_x = np.array([np.sin(l * np.pi * x[0]) for l in self.modes])
        basis_y = np.array([np.sin(l * np.pi * x[1]) for l in self.modes])

        basis_2d = basis_x[:, None] * basis_y[None, :]
        value[0] = self.threshold - np.sum(self.coeffs_ls[:, :].T * basis_2d[:, :])


class StandardFEMSolver:
    """Solver for a standard FEM resolution of a given problem."""

    def __init__(self, params, coeffs_ls):
        """
        Initialize the StandardFEMSolver.

        Parameters:
            params (ndarray): Array of parameters for the solver.
        """
        self.params = params
        self.coeffs_ls = coeffs_ls

    def create_standard_mesh(self, phi, hmax=0.05, plot_mesh=False, return_times=False):
        """Generation of a mesh over a domain defined by a level-set function.

        Args:
            phi (array): array of values of the level-set function
            hmax (float, optional): maximal size of cell. Defaults to 0.05.
            plot_mesh (bool, optional): plot the resulting mesh or not. Defaults to False.

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
        phi = phi.flatten("F")
        t0 = time.time()
        phiP1 = P1Function(M, phi)
        t1 = time.time()
        interp_time = t1 - t0
        # Remesh according to the level set
        t0 = time.time()
        newM = mmg2d(
            M,
            hmax=hmax / 1.43,
            hmin=hmax / 2,
            hgrad=None,
            sol=phiP1,
            ls=True,
            verb=0,
        )
        t1 = time.time()
        remesh_time = t1 - t0
        # Trunc the negative subdomain of the level set
        t0 = time.time()
        Mf = trunc(newM, 3)
        t1 = time.time()
        trunc_mesh = t1 - t0
        Mf.save("Thf.mesh")  # Saving in binary format
        command = "meshio convert Thf.mesh Thf.xml"
        t0 = time.time()
        os.system(command)
        t1 = time.time()
        conversion_time = t1 - t0
        t0 = time.time()
        mesh = df.Mesh("Thf.xml")
        t1 = time.time()
        fenics_read_time = t1 - t0
        if plot_mesh:
            plt.figure()
            df.plot(mesh, color="purple")
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

    def solve_one(self, i, size, nb_vert=64, reference_fem=False):
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
        mu0, mu1, sigma_x, sigma_y, amplitude, alpha, beta = self.params[i]
        coeffs_ls = self.coeffs_ls[i][None, :, :, :]
        threshold = 0.4
        print(
            f"(mu0, mu1, sigma_x, sigma_y, amplitude, alpha, beta) = {self.params[i]}"
        )

        # construction of the mesh
        if reference_fem:
            xy = np.linspace(0.0, 1.0, 512)
            phi, __ = call_phi(xy, xy, coeffs_ls, threshold)
            phi = np.reshape(phi, (512, 512))
        else:
            xy = np.linspace(0.0, 1.0, 2 * nb_vert - 1)
            phi, __ = call_phi(xy, xy, coeffs_ls, threshold)
            phi = np.reshape(phi, (2 * nb_vert - 1, 2 * nb_vert - 1))

        start_construction = time.time()
        mesh, mesh_times = self.create_standard_mesh(
            phi=phi, hmax=size, plot_mesh=False, return_times=True
        )
        end_construction = time.time()
        construction_time = end_construction - start_construction
        mesh_times.append(construction_time)
        # FunctionSpace Pk
        V = df.FunctionSpace(mesh, "CG", degV)
        boundary = "on_boundary"
        dx = df.Measure("dx", domain=mesh)

        f_expr = FExpr(
            mu0, mu1, sigma_x, sigma_y, amplitude, degree=degV + 2, domain=mesh
        )
        u_D = GExpr(alpha, beta, degree=degV + 2, domain=mesh)

        bc = df.DirichletBC(V, u_D, boundary)

        v = df.TestFunction(V)
        u = df.TrialFunction(V)
        # Resolution of the variationnal problem
        a = df.inner(df.grad(u), df.grad(v)) * dx
        l = f_expr * v * dx

        u = df.Function(V)
        start = time.time()
        solve(
            a == l,
            u,
            bcs=bc,
        )
        end = time.time()
        resolution_time = end - start
        u_h = u
        if reference_fem:
            return (
                df.project(
                    u_h,
                    V,
                    solver_type="gmres",
                    preconditioner_type="hypre_amg",
                ),
                V,
                dx,
            )
        else:
            return (
                df.project(u_h, V),
                mesh.hmax(),
                mesh_times,
                resolution_time,
            )


# parameter of the ghost penalty
sigma_D = 1.0
# Polynome Pk
polV = 1
polPhi = polV + 1


class PhiFemSolver_error:
    """
    Solver for computing PhiFEM solution.
    """

    def __init__(self, nb_cell, params, coeffs_ls):
        self.N = N = nb_cell
        self.params = params
        self.coeffs_ls = coeffs_ls
        self.mesh_macro = df.RectangleMesh(
            df.Point(0.0, 0.0), df.Point(1.0, 1.0), self.N, self.N
        )
        self.V_macro = FunctionSpace(self.mesh_macro, "CG", polV)

    def solve_one(self, i):
        """
        Compute the PhiFEM solution for a given problem.

        Parameters:
            i (int): Index of the problem to solve.

        Returns:
            tuple: PhiFEM solution (FEniCS expression), finite element space, dx measure, mesh size, and computation times.
        """
        mu0, mu1, sigma_x, sigma_y, amplitude, alpha, beta = self.params[i]
        coeffs_ls = self.coeffs_ls[i]
        coeffs_ls = np.reshape(
            coeffs_ls, (np.shape(coeffs_ls)[1], np.shape(coeffs_ls)[2])
        )
        modes = np.array(list(range(1, np.shape(coeffs_ls)[0] + 1)))
        threshold = 0.4
        phi_origin = PhiExpr(
            coeffs_ls=coeffs_ls,
            modes=modes,
            threshold=threshold,
            domain=self.mesh_macro,
            degree=polPhi,
        )
        V_phi = FunctionSpace(self.mesh_macro, "CG", polPhi)
        phi = interpolate(phi_origin, V_phi)
        domains = MeshFunction(
            "size_t", self.mesh_macro, self.mesh_macro.topology().dim()
        )
        domains.set_all(0)

        start = time.time()
        for ind in range(self.mesh_macro.num_cells()):
            mycell = Cell(self.mesh_macro, ind)
            v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
            if (
                phi(v1x, v1y) <= 3e-16
                or phi(v2x, v2y) <= 3e-16
                or phi(v3x, v3y) <= 3e-16
            ):
                domains[ind] = 1
        end = time.time()

        cell_selection = end - start
        start = time.time()
        mesh = SubMesh(self.mesh_macro, domains, 1)
        end = time.time()
        submesh_construction = end - start
        V = FunctionSpace(mesh, "CG", polV)
        V_phi = FunctionSpace(mesh, "CG", polPhi)
        phi = interpolate(phi_origin, V_phi)

        mesh.init(1, 2)
        facet_ghost = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        cell_ghost = MeshFunction("size_t", mesh, mesh.topology().dim())
        facet_ghost.set_all(0)
        cell_ghost.set_all(0)
        count_cell_ghost = 0

        start = time.time()
        for mycell in cells(mesh):
            for myfacet in facets(mycell):
                v1, v2 = vertices(myfacet)
                if phi(v1.point().x(), v1.point().y()) * phi(
                    v2.point().x(), v2.point().y()
                ) <= 0.0 or df.near(
                    phi(v1.point().x(), v1.point().y())
                    * phi(v2.point().x(), v2.point().y()),
                    0.0,
                ):
                    cell_ghost[mycell] = 1
                    for myfacet2 in facets(mycell):
                        facet_ghost[myfacet2] = 1
        end = time.time()
        ghost_cell_selection = end - start
        for mycell in cells(mesh):
            if cell_ghost[mycell] == 1:
                count_cell_ghost += 1
        print("num of cell in the ghost penalty:", count_cell_ghost)

        # Initialize cell function for domains
        dx = Measure("dx")(domain=mesh, subdomain_data=cell_ghost)
        ds = Measure("ds")(domain=mesh)
        dS = Measure("dS")(domain=mesh, subdomain_data=facet_ghost)

        # Resolution
        n = FacetNormal(mesh)
        h = CellDiameter(mesh)
        u = TrialFunction(V)
        v = TestFunction(V)

        f_expr = FExpr(
            mu0, mu1, sigma_x, sigma_y, amplitude, degree=polV + 2, domain=mesh
        )
        g_expr = GExpr(alpha, beta, degree=polV + 2, domain=mesh)

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

        # Define solution function
        w_h = Function(V)
        start = time.time()
        solve(a == L, w_h)
        end = time.time()
        resolution_time = end - start
        return (
            df.project(phi * w_h + g_expr, V),
            V,
            dx,
            mesh.hmax(),
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
