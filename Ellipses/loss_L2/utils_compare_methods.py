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
    Omega_bool,
    call_F,
    call_G,
)
from dolfin import *
from matplotlib.patches import Ellipse
from prepare_data import rotate, outside_ball, eval_phi

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"


degV = 1
degPhi = degV + 1


class MyUserExpression(BaseExpression):
    """
    JIT Expressions
    """

    def __init__(self, degree, domain):
        """
        Initialize a JIT expression.

        Parameters:
            degree (int): Degree of the expression.
            domain (dolfin.Mesh): Mesh domain for the expression.
        """
        cell = domain.ufl_cell()
        element = _select_element(
            family=None, cell=cell, degree=degree, value_shape=()
        )

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


class PhiExpr(MyUserExpression):
    """
    Expression for Phi function.
    """

    def __init__(self, x_0, y_0, lx, ly, theta, degree, domain):
        """
        Initialize the Phi expression.

        Parameters:
            x_0 (float): x-coordinate of the center.
            y_0 (float): y-coordinate of the center.
            lx (float): Length along the x-axis.
            ly (float): Length along the y-axis.
            theta (float): Rotation angle in radians.
            degree (int): Degree of the expression.
            domain (dolfin.Mesh): Mesh domain for the expression.
        """
        super().__init__(degree, domain)
        self.x_0 = x_0
        self.y_0 = y_0
        self.lx = lx
        self.ly = ly
        self.theta = theta

    def eval(self, value, x):
        """
        Evaluate the Phi expression.

        Parameters:
            value (np.ndarray): Output array to store the evaluation result.
            x (dolfin.Point): Point where the expression is evaluated.
        """
        value[0] = call_phi(
            x, self.x_0, self.y_0, self.lx, self.ly, self.theta
        )

    def value_shape(self):
        """
        Get the value shape of the Phi expression.

        Returns:
            tuple: Tuple representing the shape of the Phi expression's value.
        """
        return (2,)


class FExpr(MyUserExpression):
    """
    Expression for F function.
    """

    def __init__(self, mu0, mu1, sigma, degree, domain):
        """
        Initialize the F expression.

        Parameters:
            mu0 (float): First parameter for F.
            mu1 (float): Second parameter for F.
            sigma (float): Third parameter for F.
            degree (int): Degree of the expression.
            domain (dolfin.Mesh): Mesh domain for the expression.
        """
        super().__init__(degree, domain)
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma

    def eval(self, value, x):
        """
        Evaluate the F expression.

        Parameters:
            value (np.ndarray): Output array to store the evaluation result.
            x (dolfin.Point): Point where the expression is evaluated.
        """
        value[0] = call_F(x, self.mu0, self.mu1, self.sigma)


class GExpr(MyUserExpression):
    """
    Expression for G function.
    """

    def __init__(self, alpha, beta, degree, domain):
        """
        Initialize the G expression.

        Parameters:
            alpha (float): First parameter for G.
            beta (float): Second parameter for G.
            degree (int): Degree of the expression.
            domain (dolfin.Mesh): Mesh domain for the expression.
        """
        super().__init__(degree, domain)
        self.alpha = alpha
        self.beta = beta

    def eval(self, value, x):
        value[0] = call_G(x, self.alpha, self.beta)


class StandardFEMSolver:
    """Solver for a standard FEM resolution of a given problem."""

    def __init__(self, params):
        """
        Initialize the StandardFEMSolver.

        Parameters:
            params (ndarray): Array of parameters for the solver.
        """
        self.params = params

    def solve_one(self, i, size, reference_fem=False):
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
        mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = self.params[i]
        print(
            f"(mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta) = {self.params[i]}"
        )
        domain = mshr.CSGRotation(
            mshr.Ellipse(df.Point(x_0, y_0), lx, ly), df.Point(x_0, y_0), theta
        )

        # construction of the mesh
        if reference_fem:
            H = 50
            mesh = mshr.generate_mesh(domain, H)
            h = mesh.hmax()
            while h > 0.002767:
                H += 20
                mesh = mshr.generate_mesh(domain, H)
                h = mesh.hmax()

        else:
            mesh_macro = df.UnitSquareMesh(size - 1, size - 1)
            h_macro = mesh_macro.hmax()
            H = 3
            start = time.time()
            mesh = mshr.generate_mesh(domain, H)
            end = time.time()
            construction_time = end - start

            h = mesh.hmax()
            while h > h_macro:
                H += 1
                start = time.time()
                mesh = mshr.generate_mesh(domain, H)
                end = time.time()
                construction_time = end - start
                h = mesh.hmax()

        # FunctionSpace Pk
        V = df.FunctionSpace(mesh, "CG", degV)
        boundary = "on_boundary"

        f = FExpr(mu0, mu1, sigma, degree=degV + 2, domain=mesh)
        u_D = GExpr(alpha, beta, degree=degV + 2, domain=mesh)

        bc = df.DirichletBC(V, u_D, boundary)

        v = df.TestFunction(V)
        u = df.TrialFunction(V)
        dx = df.Measure("dx", domain=mesh)
        # Resolution of the variationnal problem
        a = df.inner(df.grad(u), df.grad(v)) * dx
        l = f * v * dx

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
                construction_time,
                resolution_time,
            )


# parameter of the ghost penalty
sigma_D = 20.0
# Polynome Pk
polV = 1
polPhi = polV + 1


class PhiFemSolver_error:
    """
    Solver for computing PhiFEM solution.
    """

    def __init__(self, nb_cell, params):
        """
        Initialize the PhiFemSolver_error.

        Parameters:
            nb_cell (int): Number of cells.
            params (list): List of parameters.
        """
        self.N = N = nb_cell
        self.params = params

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
        mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = self.params[i]
        print(
            f"(mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta) = {self.params[i]}"
        )

        domains = MeshFunction(
            "size_t", self.mesh_macro, self.mesh_macro.topology().dim()
        )
        domains.set_all(0)

        start = time.time()
        for ind in range(self.mesh_macro.num_cells()):
            mycell = Cell(self.mesh_macro, ind)
            v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
            if (
                Omega_bool(v1x, v1y, x_0, y_0, lx, ly, theta)
                or Omega_bool(v2x, v2y, x_0, y_0, lx, ly, theta)
                or Omega_bool(v3x, v3y, x_0, y_0, lx, ly, theta)
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

        phi = PhiExpr(
            x_0,
            y_0,
            lx,
            ly,
            theta,
            degree=polPhi,
            domain=mesh,
        )
        phi = interpolate(phi, V_phi)

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

        f_expr = FExpr(mu0, mu1, sigma, degree=polV + 2, domain=mesh)
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
            - sigma_D
            * avg(h)
            * jump(grad(g_expr), n)
            * jump(grad(phi * v), n)
            * dS(1)
        )

        # Define solution function
        u_h = Function(V)
        start = time.time()
        solve(a == L, u_h)
        end = time.time()
        resolution_time = end - start
        return (
            df.project(phi * u_h + g_expr, V),
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


def new_create_FG_numpy(nb_data, nb_vert):
    """
    Create a test data-set.

    Parameters:
        nb_data (int): Number of data to generate.
        nb_vert (int): Number of vertices on each data.

    Returns:
        tuple: Arrays of the values of the new data (F, phi, G, params).
    """
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    mu0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma = np.random.uniform(0.15, 0.45, size=[nb_data, 1])

    alpha = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])
    beta = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])

    x_0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    y_0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    lx = np.random.uniform(0.2, 0.45, size=[nb_data, 1])
    ly = np.random.uniform(0.2, 0.45, size=[nb_data, 1])
    theta = np.random.uniform(0.0, 0.6, size=[nb_data, 1])
    check_data = 0
    for n in range(nb_data):
        new_generation = 0
        xx_0, yy_0, llx, lly = x_0[n][0], y_0[n][0], lx[n][0], ly[n][0]
        xx0_llxp = rotate([xx_0, yy_0], [xx_0 + llx, yy_0], theta[n])
        xx0_llxm = rotate([xx_0, yy_0], [xx_0 - llx, yy_0], theta[n])
        yy0_llyp = rotate([xx_0, yy_0], [xx_0, yy_0 + lly], theta[n])
        yy0_llym = rotate([xx_0, yy_0], [xx_0, yy_0 - lly], theta[n])
        while (
            (outside_ball(xx0_llxp))
            or (outside_ball(xx0_llxm))
            or (outside_ball(yy0_llyp))
            or (outside_ball(yy0_llym))
        ):
            x_0[n][0] = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            y_0[n][0] = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            lx[n][0] = np.random.uniform(0.2, 0.45, size=[1, 1])[0]
            ly[n][0] = np.random.uniform(0.2, 0.45, size=[1, 1])[0]
            xx_0, yy_0, llx, lly = x_0[n][0], y_0[n][0], lx[n][0], ly[n][0]
            xx0_llxp = rotate([xx_0, yy_0], [xx_0 + llx, yy_0], theta[n])
            xx0_llxm = rotate([xx_0, yy_0], [xx_0 - llx, yy_0], theta[n])
            yy0_llyp = rotate([xx_0, yy_0], [xx_0, yy_0 + lly], theta[n])
            yy0_llym = rotate([xx_0, yy_0], [xx_0, yy_0 - lly], theta[n])
            new_generation += 1
        check_data += 1

    for n in range(nb_data):
        new_generation = 0
        xx_0, yy_0, llx, lly, ttheta = (
            x_0[n][0],
            y_0[n][0],
            lx[n][0],
            ly[n][0],
            theta[n][0],
        )
        mmu0, mmu1 = mu0[n][0], mu1[n][0]
        sigma[n][0] = np.random.uniform(min(llx, lly) / 2.0, max(llx, lly))
        while eval_phi(mmu0, mmu1, xx_0, yy_0, llx, lly, ttheta) > -0.15:
            mu0[n][0] = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            mu1[n][0] = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            mmu0, mmu1 = mu0[n][0], mu1[n][0]

        check_data += 1

    F = call_F(XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, nb_vert, nb_vert])

    G = call_G(XXYY, alpha, beta)
    G = np.reshape(G, [nb_data, nb_vert, nb_vert])

    phi = call_phi(XXYY, x_0, y_0, lx, ly, theta)
    phi = np.reshape(phi, [nb_data, nb_vert, nb_vert])

    params = np.concatenate(
        [mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta], axis=1
    )
    return F, phi, G, params
