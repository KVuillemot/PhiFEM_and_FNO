import numpy as np
import os

import random
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
import torch

torch.manual_seed(seed)
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
# print(f"Random seed set as {seed}")


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


class PhiFemSolver:
    def __init__(self, nb_cell, params, coeffs_ls):
        self.N = N = nb_cell
        self.params = params
        self.coeffs_ls = coeffs_ls
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
        for ind in range(self.mesh_macro.num_cells()):
            mycell = Cell(self.mesh_macro, ind)
            v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
            if (
                phi(v1x, v1y) <= 3e-16
                or phi(v2x, v2y) <= 3e-16
                or phi(v3x, v3y) <= 3e-16
            ):
                domains[ind] = 1

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
    nb_data = 3000
    n_mode = 3

    ti0 = time.time()
    F, Phi, G, params = create_FG_numpy(
        nb_data=nb_data, nb_vert=nb_vert, n_mode=n_mode, seed=seed
    )

    coeffs_ls = np.load(f"../data_domains_{n_mode}/params_{nb_data}_{seed}.npy")

    solver = PhiFemSolver(nb_cell=nb_vert - 1, params=params, coeffs_ls=coeffs_ls)
    W = solver.solve_several()
    print("F", np.min(F), np.max(F))
    print("G", np.min(G), np.max(G))
    print("Phi", np.min(Phi), np.max(Phi))
    print("W", np.min(W), np.max(W))
    print(np.shape(Phi * W + G))
    print("U", np.min(Phi * W + G), np.max(Phi * W + G))

    print("F on domain", np.min(F * (Phi <= 3e-16)), np.max(F * (Phi <= 3e-16)))
    print("G on domain", np.min(G * (Phi <= 3e-16)), np.max(G * (Phi <= 3e-16)))
    print("W on domain", np.min(W * (Phi <= 3e-16)), np.max(W * (Phi <= 3e-16)))
    print(
        "U on domain",
        np.min((Phi * W + G) * (Phi <= 3e-16)),
        np.max((Phi * W + G) * (Phi <= 3e-16)),
    )

    duration = time.time() - ti0
    print("duration to solve u:", duration)
    nb = 4
    assert nb <= F.shape[0]
    fig, axs = plt.subplots(nb, 5, figsize=(2 * nb, 10))
    for i in range(nb):
        indices = [0, 1, 6, 8]
        domain = omega_mask(Phi[indices[i]])
        p = axs[i, 0].imshow(F[indices[i], :, :] * domain)
        plt.colorbar(p, shrink=0.7)
        p = axs[i, 1].imshow(Phi[indices[i], :, :] * domain)
        plt.colorbar(p, shrink=0.7)
        p = axs[i, 2].imshow(W[indices[i], :, :] * domain)
        plt.colorbar(p, shrink=0.7)
        p = axs[i, 3].imshow(G[indices[i], :, :] * domain)
        plt.colorbar(p, shrink=0.7)
        p = axs[i, 4].imshow(
            (Phi[indices[i], :, :] * W[indices[i], :, :] + G[indices[i], :, :]) * domain
        )
        plt.colorbar(p, shrink=0.7)
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
            Phi_old = np.load(f"../data/Phi.npy")
            G_old = np.load(f"../data/G.npy")

            params_old = np.load(f"../data/agentParams.npy")
            W_old = np.load(f"../data/W.npy")
            F = np.concatenate([F_old, F])
            Phi = np.concatenate([Phi_old, Phi])
            params = np.concatenate([params_old, params])
            W = np.concatenate([W_old, W])
            G = np.concatenate([G_old, G])

        print("Save F,G,agentParams,W, nb_Data=", len(F))
        np.save(f"../data/F.npy", F)
        np.save(f"../data/Phi.npy", Phi)
        np.save(
            f"../data/agentParams.npy",
            params,
        )
        np.save(f"../data/W.npy", W)
        np.save(f"../data/G.npy", G)


if __name__ == "__main__":
    go()
