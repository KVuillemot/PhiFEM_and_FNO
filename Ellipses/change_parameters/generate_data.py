import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import random
import tensorflow as tf
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
    create_FG_numpy,
    omega_mask,
)
import time

np.pow = np.power
from dolfin import *
import dolfin as dol

# noinspection PyUnresolvedReferences
from dolfin import (
    cells,
    facets,
    vertices,
    parameters,
    SubMesh,
)
import matplotlib.pyplot as plt

seed = 2023
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
# print(f"Random seed set as {seed}")


parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"


# parameter of the ghost penalty
sigma_D = 20.0
# Polynome Pk
polV = 1
polPhi = polV + 1


# noinspection PyAbstractClass
class MyUserExpression(BaseExpression):
    """JIT Expressions"""

    def __init__(self, degree, domain):
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
    def __init__(self, x_0, y_0, lx, ly, theta, degree, domain):
        super().__init__(degree, domain)
        self.x_0 = x_0
        self.y_0 = y_0
        self.lx = lx
        self.ly = ly
        self.theta = theta

    def eval(self, value, x):
        value[0] = call_phi(
            x, self.x_0, self.y_0, self.lx, self.ly, self.theta
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
        value[0] = call_F(x, self.mu0, self.mu1, self.sigma)


class GExpr(MyUserExpression):
    def __init__(self, alpha, beta, degree, domain):
        super().__init__(degree, domain)
        self.alpha = alpha
        self.beta = beta

    def eval(self, value, x):
        value[0] = call_G(x, self.alpha, self.beta)


class PhiFemSolver:
    def __init__(self, nb_cell, params):
        self.N = N = nb_cell
        self.params = params

        self.mesh_macro = dol.RectangleMesh(
            dol.Point(0.0, 0.0), dol.Point(1.0, 1.0), self.N, self.N
        )
        self.V_macro = FunctionSpace(self.mesh_macro, "CG", polV)

    def make_matrix(self, expr):
        """Convert an expression of degree k to a matrix with nodal values.

        Args:
            expr (FEniCS expression): the expression to convert

        Returns:
            np array : a matrix of size N+1 * N+1
        """
        expr = interpolate(expr, self.V_macro)
        expr = expr.compute_vertex_values(self.mesh_macro)
        expr = np.reshape(expr, [self.N + 1, self.N + 1])
        return expr

    def solve_one(self, i):
        """Computation of phiFEM

        Args:
            i (int): index of the problem to solve

        Returns:
            np array : matrix of the phiFEM solution
        """
        mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = self.params[i]

        domains = MeshFunction(
            "size_t", self.mesh_macro, self.mesh_macro.topology().dim()
        )
        domains.set_all(0)
        for ind in range(self.mesh_macro.num_cells()):
            mycell = Cell(self.mesh_macro, ind)
            v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
            if (
                Omega_bool(v1x, v1y, x_0, y_0, lx, ly, theta)
                or Omega_bool(v2x, v2y, x_0, y_0, lx, ly, theta)
                or Omega_bool(v3x, v3y, x_0, y_0, lx, ly, theta)
            ):
                domains[ind] = 1

        mesh = SubMesh(self.mesh_macro, domains, 1)

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

        for mycell in cells(mesh):
            for myfacet in facets(mycell):
                v1, v2 = vertices(myfacet)
                if phi(v1.point().x(), v1.point().y()) * phi(
                    v2.point().x(), v2.point().y()
                ) <= 0.0 or dol.near(
                    phi(v1.point().x(), v1.point().y())
                    * phi(v2.point().x(), v2.point().y()),
                    0.0,
                ):
                    cell_ghost[mycell] = 1
                    for myfacet2 in facets(mycell):
                        facet_ghost[myfacet2] = 1

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
        w_h = Function(V)
        solve(a == L, w_h)
        return self.make_matrix(w_h)

    def solve_several(self):
        W = []
        nb = len(self.params)
        for i in range(nb):
            print(f"{i}/{nb}:", end="")
            w = self.solve_one(i)
            W.append(w)

        return np.stack(W)


def go():
    """
    Main function to generate data.
    """
    save = True
    add_to_existing = False
    nb_vert = 64
    nb_data = 1500

    ti0 = time.time()
    F, phi, G, params = create_FG_numpy(nb_data=nb_data, nb_vert=nb_vert)

    solver = PhiFemSolver(nb_cell=nb_vert - 1, params=params)
    W = solver.solve_several()
    print("F", np.min(F), np.max(F))
    print("G", np.min(G), np.max(G))
    print("Phi", np.min(phi), np.max(phi))
    print("W", np.min(W), np.max(W))
    print(np.shape(phi * W + G))
    print("U", np.min(phi * W + G), np.max(phi * W + G))

    duration = time.time() - ti0
    print("duration to solve u:", duration)
    nb = 4
    assert nb <= F.shape[0]
    fig, axs = plt.subplots(nb, 5, figsize=(2 * nb, 10))
    for i in range(nb):
        mu0, mu1, sigma, x_0, y_0, lx, ly, theta, alpha, beta = params[i]
        domain = omega_mask(nb_vert, x_0, y_0, lx, ly, theta)
        axs[i, 0].imshow(F[i, :, :] * domain)
        axs[i, 1].imshow(phi[i, :, :] * domain)
        axs[i, 2].imshow(W[i, :, :] * domain)
        axs[i, 3].imshow(G[i, :, :] * domain)
        axs[i, 4].imshow((phi[i, :, :] * W[i, :, :] + G[i, :, :]) * domain)
        axs[i, 0].set_title("F")
        axs[i, 1].set_title("phi")
        axs[i, 2].set_title("W")
        axs[i, 3].set_title("G")
        axs[i, 4].set_title("U")
    fig.tight_layout()
    plt.show()

    if save:
        if not (os.path.exists(f"../data")):
            os.makedirs(f"../data")
        if add_to_existing and os.path.exists(f"../data/F.npy"):
            F_old = np.load(f"../data/F.npy")
            G_old = np.load(f"../data/G.npy")
            Phi_old = np.load(f"../data/Phi.npy")
            params_old = np.load(f"../data/agentParams.npy")
            W_old = np.load(f"../data/W.npy")
            F = np.concatenate([F_old, F])
            G = np.concatenate([G_old, G])
            phi = np.concatenate([Phi_old, phi])
            params = np.concatenate([params_old, params])
            W = np.concatenate([W_old, W])

        print("Save F,G,agentParams,W, nb_Data=", len(F))
        np.save(f"../data/F.npy", F)
        np.save(f"../data/G.npy", G)
        np.save(f"../data/Phi.npy", phi)
        np.save(
            f"../data/agentParams.npy",
            params,
        )
        np.save(f"../data/W.npy", W)


if __name__ == "__main__":
    go()
