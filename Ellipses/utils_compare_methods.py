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
import matplotlib.transforms as transforms


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
        element = _select_element(
            family=None, cell=cell, degree=degree, value_shape=()
        )  # modifier value_shape si Expression non scalaire

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
    def __init__(self, x_0, y_0, lx, ly, theta, degree, domain):
        super().__init__(degree, domain)
        self.x_0 = x_0
        self.y_0 = y_0
        self.lx = lx
        self.ly = ly
        self.theta = theta

    def eval(self, value, x):
        value[0] = call_phi(
            df, x, self.x_0, self.y_0, self.lx, self.ly, self.theta
        )

    def value_shape(self):
        return (2,)


class FExpr(MyUserExpression):
    def __init__(self, mu0, mu1, sigma, degree, domain):
        super().__init__(degree, domain)
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma

    def eval(self, value, x):
        value[0] = call_F(df, x, self.mu0, self.mu1, self.sigma)


class GExpr(MyUserExpression):
    def __init__(self, alpha, beta, degree, domain):
        super().__init__(degree, domain)
        self.alpha = alpha
        self.beta = beta

    def eval(self, value, x):
        value[0] = call_G(df, x, self.alpha, self.beta)


def compute_standard_fem(nb_vert, param, exact_fem=False):
    mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = param
    print(f"(mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta) = {param}")
    domain = mshr.CSGRotation(
        mshr.Ellipse(df.Point(x_0, y_0), lx, ly), df.Point(x_0, y_0), theta
    )  # warning with theta : degree ? Radians ? Conversion ?

    # construction of the mesh
    if exact_fem:
        H = 50
        mesh = mshr.generate_mesh(domain, H)
        h = mesh.hmax()
        while h > 0.002767:
            H += 20
            mesh = mshr.generate_mesh(domain, H)
            h = mesh.hmax()

    else:
        mesh_macro = df.UnitSquareMesh(nb_vert - 1, nb_vert - 1)
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
    solve(a == l, u, bcs=bc)
    end = time.time()
    resolution_time = end - start
    u_h = u
    if exact_fem:
        return (
            df.project(
                u_h,
                V,  # , solver_type="gmres", preconditioner_type="hypre_amg"
            ),
            V,
            dx,
        )
    else:
        return df.project(u_h, V), construction_time, resolution_time


# parameter of the ghost penalty
sigma_D = 20.0
# Polynome Pk
polV = 1
polPhi = polV + 1


class PhiFemSolver_error:
    def __init__(self, nb_cell, params):
        self.N = N = nb_cell
        self.params = params

        self.mesh_macro = df.RectangleMesh(
            df.Point(0.0, 0.0), df.Point(1.0, 1.0), self.N, self.N
        )
        self.V_macro = FunctionSpace(self.mesh_macro, "CG", polV)

    def make_matrix(self, expr):
        expr = project(expr)
        expr = interpolate(expr, self.V_macro)
        expr = expr.compute_vertex_values(self.mesh_macro)
        expr = np.reshape(expr, [self.N + 1, self.N + 1])
        return expr

    def solve_one(self, i):
        mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = self.params
        print(
            f"(mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta) = {self.params}"
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
            cell_selection,
            submesh_construction,
            ghost_cell_selection,
            resolution_time,
        )

    def solve_several(self):
        U = []
        nb = len(self.params)
        for i in range(nb):
            print(f"{i}/{nb}:", end="")
            u = self.solve_one(i)
            U.append(u)

        return np.stack(U)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == "__main__":
    print(0)
