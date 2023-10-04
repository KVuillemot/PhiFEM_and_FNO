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
    call_F,
    call_G,
    create_FG_numpy,
    omega_mask,
)
import time

np.pow = np.power
from dolfin import *
import dolfin as dol
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


parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"


# parameter of the ghost penalty
sigma_D = 1.0
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


class phi_expr(MyUserExpression):
    def __init__(self, coeffs, modes, threshold, degree, domain):
        super().__init__(degree, domain)
        self.coeffs = coeffs
        self.modes = modes
        self.threshold = threshold

    def eval(self, value, x):
        basis_x = np.array([np.sin(l * np.pi * x[0]) for l in self.modes])
        basis_y = np.array([np.sin(l * np.pi * x[1]) for l in self.modes])

        basis_2d = basis_x[:, None] * basis_y[None, :]
        value[0] = self.threshold - np.sum(self.coeffs[:, :] * basis_2d[:, :])


class PhiFemSolver:
    def __init__(self, nb_cell, params, coeffs):
        self.N = N = nb_cell
        self.coeffs = coeffs
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
            np array : matrix of the phiFEM solution and matrix of the nodal values of phi
        """
        mu0, mu1, sigma, alpha, beta = self.params[i]

        coeffs = self.coeffs[i]
        coeffs = np.reshape(coeffs, (np.shape(coeffs)[1], np.shape(coeffs)[2]))
        modes = np.array(list(range(1, np.shape(coeffs)[0] + 1)))
        threshold = 0.4
        phi_origin = phi_expr(
            coeffs=coeffs,
            modes=modes,
            threshold=threshold,
            domain=self.mesh_macro,
            degree=polPhi,
        )
        domains = MeshFunction(
            "size_t", self.mesh_macro, self.mesh_macro.topology().dim()
        )
        domains.set_all(0)
        V_phi = FunctionSpace(self.mesh_macro, "CG", 2)
        phi = interpolate(phi_origin, V_phi)

        for cell in cells(self.mesh_macro):
            for v in vertices(cell):
                if phi(v.point()) <= 0.0:
                    domains[cell] = 1
                    break

        mesh = SubMesh(self.mesh_macro, domains, 1)

        V = FunctionSpace(mesh, "CG", polV)
        V_phi = FunctionSpace(mesh, "CG", polPhi)
        phi = interpolate(phi_origin, V_phi)

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
        # Define solution function
        w_h = Function(V)
        solve(a == L, w_h)
        return self.make_matrix(w_h), self.make_matrix(phi_origin)

    def solve_several(self):
        W = []
        Phi_64 = []
        nb = len(self.params)
        for i in range(nb):
            print(f"{i}/{nb}:", end="")
            w, phi = self.solve_one(i)
            W.append(w)
            Phi_64.append(phi)

        return np.stack(W), np.stack(Phi_64)


def go():
    """
    Main function to generate data.
    """
    save = True
    add_to_existing = False
    nb_vert = 64
    nb_data = 500
    n_mode = 4

    ti0 = time.time()
    F, phi, G, params = create_FG_numpy(
        nb_data=nb_data, nb_vert=nb_vert, n_mode=n_mode
    )
    coeffs = np.load(f"./data_domains_{nb_data}_{n_mode}/params_{nb_data}.npy")

    solver = PhiFemSolver(nb_cell=nb_vert - 1, params=params, coeffs=coeffs)
    W, Phi_64 = solver.solve_several()
    print("F", np.min(F), np.max(F))
    print("G", np.min(G), np.max(G))
    print("Phi", np.min(Phi_64), np.max(Phi_64))
    print("W", np.min(W), np.max(W))
    print(np.shape(Phi_64 * W + G))
    print("U", np.min(Phi_64 * W + G), np.max(Phi_64 * W + G))

    duration = time.time() - ti0
    print("duration to solve u:", duration)
    nb = 4
    assert nb <= F.shape[0]
    fig, axs = plt.subplots(nb, 5, figsize=(2 * nb, 10))
    for i in range(nb):
        indices = [0, 1, 6, 8]
        mu0, mu1, sigma, alpha, beta = params[indices[i]]
        domain = omega_mask(Phi_64[indices[i]])
        axs[i, 0].imshow(F[indices[i], :, :] * domain)
        axs[i, 1].imshow(Phi_64[indices[i], :, :] * domain)
        axs[i, 2].imshow(W[indices[i], :, :] * domain)
        axs[i, 3].imshow(G[indices[i], :, :] * domain)
        axs[i, 4].imshow(
            (
                Phi_64[indices[i], :, :] * W[indices[i], :, :]
                + G[indices[i], :, :]
            )
            * domain
        )
        axs[i, 0].set_title("F")
        axs[i, 1].set_title("phi")
        axs[i, 2].set_title("W")
        axs[i, 3].set_title("G")
        axs[i, 4].set_title("U")
    fig.tight_layout()
    plt.show()

    if save:
        if not (os.path.exists(f"./data_{nb_data}")):
            os.makedirs(f"./data_{nb_data}")
        if add_to_existing and os.path.exists(f"./data_{nb_data}/F.npy"):
            F_old = np.load(f"./data_{nb_data}/F.npy")
            Phi_old = np.load(f"./data_{nb_data}/Phi.npy")
            Phi_64_old = np.load(f"./data_{nb_data}/Phi_64.npy")
            G_old = np.load(f"./data_{nb_data}/G.npy")

            params_old = np.load(f"./data_{nb_data}/agentParams.npy")
            W_old = np.load(f"./data_{nb_data}/W.npy")
            F = np.concatenate([F_old, F])
            phi = np.concatenate([Phi_old, phi])
            Phi_64 = np.concatenate([Phi_64_old, Phi_64])
            params = np.concatenate([params_old, params])
            W = np.concatenate([W_old, W])
            G = np.concatenate([G_old, G])

        print("Save F,G,agentParams,W, nb_Data=", len(F))
        np.save(f"./data_{nb_data}/F.npy", F)
        np.save(f"./data_{nb_data}/Phi.npy", phi)
        np.save(f"./data_{nb_data}/Phi_64.npy", Phi_64)
        np.save(
            f"./data_{nb_data}/agentParams.npy",
            params,
        )
        np.save(f"./data_{nb_data}/W.npy", W)
        np.save(f"./data_{nb_data}/G.npy", G)


if __name__ == "__main__":
    go()
