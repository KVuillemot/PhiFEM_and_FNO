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
from dolfin.function.expression import (
    BaseExpression,
    _select_element,
    _InterfaceExpression,
)
from prepare_data import call_F, call_G
from dolfin import *
from pymedit import (
    P1Function,
    square,
    mmg2d,
    trunc,
)
from matplotlib.patches import Ellipse


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


def create_standard_mesh(phi, hmax=0.05, plot_mesh=False):
    """Generation of a mesh over a domain defined by a level-set function.

    Args:
        phi (array): array of values of the level-set function
        hmax (float, optional): maximal size of cell. Defaults to 0.05.
        plot_mesh (bool, optional): plot the resulting mesh or not. Defaults to False.

    Returns:
        mesh: a FEniCS mesh.
    """
    n = np.shape(phi)[0]
    M = square(n - 1, n - 1)
    M.debug = 4  # For debugging and mmg3d output

    # Setting a P1 level set function
    phi = phi.flatten("F")
    phiP1 = P1Function(M, phi)

    # Remesh according to the level set
    newM = mmg2d(
        M,
        hmax=hmax / 1.43,
        hmin=hmax / 2,
        hgrad=None,
        sol=phiP1,
        ls=True,
        verb=0,
    )

    # Trunc the negative subdomain of the level set
    Mf = trunc(newM, 3)
    Mf.save("Thf.mesh")  # Saving in binary format
    command = "meshio convert Thf.mesh Thf.xml"
    os.system(command)
    mesh = df.Mesh("Thf.xml")
    if plot_mesh:
        plt.figure()
        df.plot(mesh, color="purple")
        plt.show()
    return mesh


class StandardFEMSolver:
    """Solver for a standard FEM resolution of a given problem."""

    def __init__(self, params, phi_vector):
        """
        Initialize the StandardFEMSolver.

        Parameters:
            params (ndarray): Array of parameters for the solver.
            phi_vector (ndarray) : Array of level-set functions.
        """
        self.params = params
        self.phi_vector = phi_vector

    def solve_one(self, i, size, plot_mesh=False, reference_fem=False):
        """Resolution of a given problem by a standard method.

        Args:
            i (int): index of the problem to be solved.
            size (int): maximal size of cell.
            plot_mesh (bool) : plot the resulting mesh or not. Defaults to False.
            reference_fem (bool, optional): If True, the method computes the reference solution, on a fine mesh. Defaults to False.

        Returns:
            If reference_fem : the solution, the finite element space and the dx measure.
            Else : the solution, the size of the mesh and the computation times.
        """
        mu0, mu1, sigma, alpha, beta = self.params[i]
        print(f"(mu0, mu1, sigma, alpha, beta) = {self.params}")
        start_mesh_creation = time.time()
        mesh = create_standard_mesh(self.phi_vector, size, plot_mesh)
        end_mesh_creation = time.time()
        mesh_creation = end_mesh_creation - start_mesh_creation
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
        total = mesh_creation + end - start
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
            return df.project(u_h, V), total, mesh.hmax()


class phi_expr(MyUserExpression):
    """
    Expression for phi function.
    """

    def __init__(self, coeffs, modes, threshold, degree, domain):
        """
        Initialize the phi expression.

        Parameters:
            coeffs (ndarray): coefficients of the Fourier sum.
            modes (ndarray): Fourier modes.
            threshold (float) : threshold value.
            degree (int): Degree of the expression.
            domain (dolfin.Mesh): Mesh domain for the expression.
        """
        super().__init__(degree, domain)
        self.coeffs = coeffs
        self.modes = modes
        self.threshold = threshold

    def eval(self, value, x):
        basis_x = np.array([np.sin(l * np.pi * x[0]) for l in self.modes])
        basis_y = np.array([np.sin(l * np.pi * x[1]) for l in self.modes])

        basis_2d = basis_x[:, None] * basis_y[None, :]
        value[0] = self.threshold - np.sum(self.coeffs[:, :] * basis_2d[:, :])


class PhiFemSolver_error:
    """
    Solver for computing PhiFEM solution.
    """

    def __init__(self, nb_cell, params, coeffs, sigma_D):
        self.N = N = nb_cell
        self.params = params
        self.coeffs = coeffs
        self.mesh_macro = df.RectangleMesh(
            df.Point(0.0, 0.0), df.Point(1.0, 1.0), self.N, self.N
        )
        self.V_macro = FunctionSpace(self.mesh_macro, "CG", degV)
        self.sigma_D = sigma_D

    def make_matrix(self, expr):
        expr = interpolate(expr, self.V_macro)
        expr = expr.compute_vertex_values(self.mesh_macro)
        expr = np.reshape(expr, [self.N + 1, self.N + 1])
        return expr

    def solve_one(self, i):
        """
        Compute the PhiFEM solution for a given problem.

        Parameters:
            i (int): Index of the problem to solve.

        Returns:
            tuple: PhiFEM solution (FEniCS expression), finite element space, dx measure, mesh size, computation times, and matrix of nodal values of phi.
        """
        mu0, mu1, sigma, alpha, beta = self.params[i]
        coeffs = self.coeffs[i].T
        coeffs = np.reshape(coeffs, (np.shape(coeffs)[0], np.shape(coeffs)[1]))
        modes = np.array(list(range(1, np.shape(coeffs)[0] + 1)))
        threshold = 0.4
        phi_origin = phi_expr(
            coeffs=coeffs,
            modes=modes,
            threshold=threshold,
            domain=self.mesh_macro,
            degree=degPhi,
        )
        domains = MeshFunction(
            "size_t", self.mesh_macro, self.mesh_macro.topology().dim()
        )
        domains.set_all(0)
        V_phi = FunctionSpace(self.mesh_macro, "CG", 2)
        phi = interpolate(phi_origin, V_phi)

        start_submesh_creation = time.time()
        for cell in cells(self.mesh_macro):
            for v in vertices(cell):
                if phi(v.point()) <= 0.0:
                    domains[cell] = 1
                    break

        mesh = SubMesh(self.mesh_macro, domains, 1)
        end_submesh_creation = time.time()
        submesh_creation = end_submesh_creation - start_submesh_creation
        V = FunctionSpace(mesh, "CG", degV)
        V_phi = FunctionSpace(mesh, "CG", degPhi)
        phi = interpolate(phi_origin, V_phi)

        mesh.init(1, 2)
        facet_ghost = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        cell_ghost = MeshFunction("size_t", mesh, mesh.topology().dim())
        facet_ghost.set_all(0)
        cell_ghost.set_all(0)
        count_cell_ghost = 0
        start_cell_selection = time.time()
        for cell in cells(mesh):
            for facet in facets(cell):
                v1, v2 = vertices(facet)
                if phi(v1.point()) * phi(v2.point()) <= 0.0 or near(
                    phi(v1.point()) * phi(v2.point()), 0.0
                ):
                    cell_ghost[cell] = 1
                    for facett in facets(cell):
                        facet_ghost[facett] = 1

        end_cell_selection = time.time()
        cell_selection = end_cell_selection - start_cell_selection

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

        f_expr = FExpr(mu0, mu1, sigma, degree=degV + 2, domain=mesh)
        g_expr = GExpr(alpha, beta, degree=degV + 2, domain=mesh)

        a = (
            inner(grad(phi * u), grad(phi * v)) * dx
            - dot(inner(grad(phi * u), n), phi * v) * ds
            + self.sigma_D
            * avg(h)
            * dot(
                jump(grad(phi * u), n),
                jump(grad(phi * v), n),
            )
            * dS(1)
            + self.sigma_D
            * h**2
            * inner(
                div(grad(phi * u)),
                div(grad(phi * v)),
            )
            * dx(1)
        )
        L = (
            f_expr * v * phi * dx
            - self.sigma_D * h**2 * inner(f_expr, div(grad(phi * v))) * dx(1)
            - self.sigma_D
            * h**2
            * div(grad(phi * v))
            * div(grad(g_expr))
            * dx(1)
            - inner(grad(g_expr), grad(phi * v)) * dx
            + inner(grad(g_expr), n) * phi * v * ds
            - self.sigma_D
            * avg(h)
            * jump(grad(g_expr), n)
            * jump(grad(phi * v), n)
            * dS(1)
        )

        # Define solution function
        w_h = Function(V)
        start = time.time()
        solve(a == L, w_h)
        end = time.time()
        total = submesh_creation + cell_selection + end - start
        return (
            df.project(phi * w_h + g_expr, V),
            V,
            total,
            self.make_matrix(phi_origin),
            mesh.hmax(),
        )


def create_params(domains):
    """
    Create a test data-set for given domains

    Parameters:
        domains (ndarray): Domains used to generate parameters.

    Returns:
        tuple: Arrays of the values of the new data (F, G, params).
    """
    nb_dofs = np.shape(domains)[-1]
    xy = np.linspace(0.0, 1.0, nb_dofs)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])
    nb_data = np.shape(domains)[0]
    mu0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma = np.random.uniform(0.1, 0.5, size=[nb_data, 1])

    F = call_F(XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, nb_dofs, nb_dofs])

    for i in range(np.shape(domains)[0]):
        domain = domains[i, :, :]
        f = F[i]
        new_gen = 0
        while np.max(f * domain) < 80.0:
            __mu0 = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            __mu1 = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            __sigma = np.random.uniform(0.1, 0.5, size=[1, 1])[0]
            mu0[i][0] = __mu0
            mu1[i][0] = __mu1
            sigma[i][0] = __sigma
            f = call_F(XXYY, __mu0, __mu1, __sigma)
            f = np.reshape(f, [nb_dofs, nb_dofs])
            F[i] = f
            new_gen += 1

    xy = np.linspace(0.0, 1.0, nb_dofs)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    F = call_F(XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, nb_dofs, nb_dofs])

    alpha = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])
    beta = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])

    G = call_G(XXYY, alpha, beta)
    G = np.reshape(G, [nb_data, nb_dofs, nb_dofs])

    params = np.concatenate([mu0, mu1, sigma, alpha, beta], axis=1)

    return F, G, params


def create_multiple_params_unique_domain(domain, nb_data):
    """Creation of a set of parameters for a given domain.

    Args:
        domain (array): Boolean definition of the domain.
        nb_data (int): Number of data to be generate.

    Returns:
        params (arrays)
    """
    nb_dofs = np.shape(domain)[-1]
    xy = np.linspace(0.0, 1.0, nb_dofs)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])
    mu0 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1 = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma = np.random.uniform(0.1, 0.5, size=[nb_data, 1])

    F = call_F(XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, nb_dofs, nb_dofs])

    for i in range(nb_data):
        f = F[i]
        new_gen = 0
        while np.max(f * domain) < 80.0:
            __mu0 = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            __mu1 = np.random.uniform(0.2, 0.8, size=[1, 1])[0]
            __sigma = np.random.uniform(0.1, 0.5, size=[1, 1])[0]
            mu0[i][0] = __mu0
            mu1[i][0] = __mu1
            sigma[i][0] = __sigma
            f = call_F(XXYY, __mu0, __mu1, __sigma)
            f = np.reshape(f, [nb_dofs, nb_dofs])
            F[i] = f
            new_gen += 1

    xy = np.linspace(0.0, 1.0, nb_dofs)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    F = call_F(XXYY, mu0, mu1, sigma)
    F = np.reshape(F, [nb_data, nb_dofs, nb_dofs])

    alpha = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])
    beta = np.random.uniform(-0.8, 0.8, size=[nb_data, 1])

    G = call_G(XXYY, alpha, beta)
    G = np.reshape(G, [nb_data, nb_dofs, nb_dofs])

    params = np.concatenate([mu0, mu1, sigma, alpha, beta], axis=1)
    return params


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
