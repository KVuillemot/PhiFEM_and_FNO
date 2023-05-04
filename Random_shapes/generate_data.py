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
    create_FG_numpy,
    omega_mask,
)

from utils import convert_numpy_matrix_to_fenics
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

seed = 27042023
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

# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
#         )
#         logical_gpus = tf.config.list_logical_devices("GPU")
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

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


class FExpr(MyUserExpression):
    def __init__(self, mu0, mu1, sigma, degree, domain):
        super().__init__(degree, domain)
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma

    def eval(self, value, x):
        value[0] = call_F(dol, x, self.mu0, self.mu1, self.sigma)


class PhiFemSolver:
    def __init__(self, nb_cell, params, phi_vector):
        self.N = N = nb_cell
        self.params = params
        self.phi_vector = phi_vector
        self.mesh_macro = dol.RectangleMesh(
            dol.Point(0.0, 0.0), dol.Point(1.0, 1.0), self.N, self.N
        )
        self.V_macro = FunctionSpace(self.mesh_macro, "CG", polV)

    def make_matrix(self, expr):
        expr = project(expr, self.V_macro)
        expr = interpolate(expr, self.V_macro)
        expr = expr.compute_vertex_values(self.mesh_macro)
        expr = np.reshape(expr, [self.N + 1, self.N + 1])
        return expr

    def make_matrix_of_phi(self, phi):
        phi = interpolate(phi, self.V_macro)
        phi = phi.compute_vertex_values(self.mesh_macro)
        phi = np.reshape(phi, [self.N + 1, self.N + 1])
        return phi

    def solve_one(self, i):
        mu0, mu1, sigma = self.params[i]
        phi_numpy = self.phi_vector[i, :, :]
        phi_origin = convert_numpy_matrix_to_fenics(phi_numpy, self.N + 1)
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

        for cell in cells(mesh):
            for facet in facets(cell):
                v1, v2 = vertices(facet)
                if phi(v1.point()) * phi(v2.point()) <= 0.0 or near(
                    phi(v1.point()) * phi(v2.point()), 0.0
                ):
                    cell_ghost[cell] = 1
                    for facett in facets(cell):
                        facet_ghost[facett] = 1

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
        L = f_expr * v * phi * dx - sigma_D * h**2 * inner(
            f_expr, div(grad(phi * v))
        ) * dx(1)

        # Define solution function
        w_h = Function(V)
        solve(a == L, w_h)
        return self.make_matrix(w_h), self.make_matrix_of_phi(phi_origin)

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
    save = True
    add_to_existing = False
    nb_vert = 64
    nb_data = 5000
    ti0 = time.time()

    F, phi, params = create_FG_numpy(nb_data=nb_data, nb_vert=nb_vert)

    solver = PhiFemSolver(nb_cell=nb_vert - 1, params=params, phi_vector=phi)
    W, Phi_64 = solver.solve_several()
    print("F", np.min(F), np.max(F))
    print("Phi", np.min(Phi_64), np.max(Phi_64))
    print("W", np.min(W), np.max(W))
    print(np.shape(Phi_64 * W))
    print("U", np.min(Phi_64 * W), np.max(Phi_64 * W))

    duration = time.time() - ti0
    print("duration to solve u:", duration)
    nb = 4
    assert nb <= F.shape[0]
    fig, axs = plt.subplots(nb, 4, figsize=(2 * nb, 8))
    for i in range(nb):
        indices = [0, 1, 6, 8]
        domain = omega_mask(Phi_64[indices[i]])
        axs[i, 0].imshow(F[indices[i], :, :] * domain)
        axs[i, 1].imshow(Phi_64[indices[i], :, :] * domain)
        axs[i, 2].imshow(W[indices[i], :, :] * domain)
        axs[i, 3].imshow(
            Phi_64[indices[i], :, :] * W[indices[i], :, :] * domain
        )
        axs[i, 0].set_title("F")
        axs[i, 1].set_title("phi")
        axs[i, 2].set_title("W")
        axs[i, 3].set_title("U")
    fig.tight_layout()
    plt.show()

    if save:
        if not (os.path.exists(f"./data_{nb_data}")):
            os.makedirs(f"./data_{nb_data}")
        if add_to_existing and os.path.exists(f"./data_{nb_data}/F.npy"):
            F_old = np.load(f"./data_{nb_data}/F.npy")
            Phi_old = np.load(f"./data_{nb_data}/Phi.npy")
            Phi_64_old = np.load(f"./data_{nb_data}/Phi_64.npy")
            params_old = np.load(f"./data_{nb_data}/agentParams.npy")
            W_old = np.load(f"./data_{nb_data}/W.npy")
            F = np.concatenate([F_old, F])
            phi = np.concatenate([Phi_old, phi])
            Phi_64 = np.concatenate([Phi_64_old, Phi_64])
            params = np.concatenate([params_old, params])
            W = np.concatenate([W_old, W])

        print("Save F,G,agentParams,W, nb_Data=", len(F))
        np.save(f"./data_{nb_data}/F.npy", F)
        np.save(f"./data_{nb_data}/Phi.npy", phi)
        np.save(f"./data_{nb_data}/Phi_64.npy", Phi_64)
        np.save(
            f"./data_{nb_data}/agentParams.npy",
            params,
        )
        np.save(f"./data_{nb_data}/W.npy", W)


if __name__ == "__main__":
    go()
