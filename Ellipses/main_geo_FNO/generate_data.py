import numpy as np
import os

import random
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
    eval_phi,
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
from pymedit import (
    P1Function,
    square,
    mmg2d,
    trunc,
)
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

seed = 2102
random.seed(seed)
np.random.seed(seed)
import torch
import meshio

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


class PhiExpr(MyUserExpression):
    def __init__(self, x_0, y_0, lx, ly, theta, degree, domain):
        super().__init__(degree, domain)
        self.x_0 = x_0
        self.y_0 = y_0
        self.lx = lx
        self.ly = ly
        self.theta = theta

    def eval(self, value, x):
        value[0] = call_phi(x, self.x_0, self.y_0, self.lx, self.ly, self.theta)

    def value_shape(self):
        return (2,)


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


class StdFemSolver:
    def __init__(self, nb_cell, params, compare_methods=False):

        self.params = params

        self.compare_methods = compare_methods

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

    def create_points_file(self, points):
        file = open(
            f"./points.mesh",
            "w",
        )
        file.write("\nMeshVersionFormatted\n1\n\n")
        file.write(f"Dimension\n2\n\nVertices\n{len(points)}\n\n")
        for i in range(len(points)):
            file.write(f"{str(points[i,0])} {points[i,1]} 0\n")

        file.write("\nEnd")
        file.close()

    def create_standard_mesh(
        self, x_0, y_0, lx, ly, theta, phi, hmax=0.05, plot_mesh=False
    ):
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
        newM = mmg2d(
            M,
            hmax=0.006,
            sol=phiP1,
            ls=True,
        )
        Mf = trunc(newM, 3)
        points = np.array([Mf.x, Mf.y]).T
        phi_values = eval_phi(Mf.x, Mf.y, x_0, y_0, lx, ly, theta)
        close_boundary = phi_values >= -0.00001
        indices_boundary = np.where(close_boundary == True)[0]
        not_boundary = np.setdiff1d(np.arange(len(Mf.x)), indices_boundary)
        indices = np.random.choice(
            not_boundary, 2600 - indices_boundary.shape[0], replace=False
        )
        final_indices = np.concatenate([indices_boundary, indices])
        points = points[final_indices]
        print(points.shape)
        self.create_points_file(points)
        command_1 = "mmg2d_O3 -noinsert -in points.mesh -out mesh.mesh -v 0"  # optimize the mesh, keeping the nb of vertices fixed
        os.system(command_1)
        command_2 = (
            "meshio convert mesh.mesh mesh.xml"  # convert the mesh to FEniCS format
        )
        os.system(command_2)
        mesh = dol.Mesh("mesh.xml")
        if plot_mesh:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            dol.plot(mesh, color="purple")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.subplot(1, 3, 2)
            plt.scatter(points[:, 0], points[:, 1])
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.subplot(1, 3, 3)
            plt.scatter(mesh.coordinates()[:, 0], mesh.coordinates()[:, 1])
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()

        return mesh

    def solve_one(self, i):
        """Computation of phiFEM

        Args:
            i (int): index of the problem to solve

        Returns:
            np array : matrix of the phiFEM solution
        """
        print(self.params[i])
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

        XX, YY = np.meshgrid(
            np.linspace(0.0, 1.0, 511),
            np.linspace(0.0, 1.0, 511),
        )
        XX = np.reshape(XX, [-1])
        YY = np.reshape(YY, [-1])
        XXYY = np.stack([XX, YY])
        phi = call_phi(XXYY, x_0, y_0, lx, ly, theta)
        phi = np.reshape(phi, (511, 511))
        plot_mesh = False
        mesh = self.create_standard_mesh(
            x_0,
            y_0,
            lx,
            ly,
            theta,
            phi=phi,
            plot_mesh=plot_mesh,
        )

        phi = PhiExpr(
            x_0,
            y_0,
            lx,
            ly,
            theta,
            degree=polV,
            domain=mesh,
        )

        # Resolution
        V = dol.FunctionSpace(mesh, "CG", polV)

        boundary = "on_boundary"

        f = FExpr(mu0, mu1, sigma_x, sigma_y, amplitude, degree=polV + 2, domain=mesh)
        u_D = GExpr(alpha, beta, degree=polV + 2, domain=mesh)

        bc = dol.DirichletBC(V, u_D, boundary)

        v = dol.TestFunction(V)
        u = dol.TrialFunction(V)
        dx = dol.Measure("dx", domain=mesh)
        # Resolution of the variationnal problem
        a = dol.inner(dol.grad(u), dol.grad(v)) * dx
        l = f * v * dx

        u = dol.Function(V)
        solve(
            a == l,
            u,
            bcs=bc,
        )
        return (
            dol.interpolate(u, V).vector()[:],
            V.tabulate_dof_coordinates(),
            dol.interpolate(f, V).vector()[:],
            dol.interpolate(u_D, V).vector()[:],
            dol.interpolate(phi, V).vector()[:],
            mesh.hmax(),
        )

    def solve_several(self):
        U, F, G, Phi = [], [], [], []
        XY = []
        hh = []
        nb = len(self.params)
        for i in range(nb):
            print(f"{i}/{nb}:", end="")
            u, xy, f, g, phi, h = self.solve_one(i)
            U.append(u)
            XY.append(xy)
            F.append(f)
            G.append(g)
            Phi.append(phi)
            hh.append(h)

        return (
            np.stack(U),
            np.stack(XY),
            np.stack(F),
            np.stack(G),
            np.stack(Phi),
            np.stack(hh),
        )


def go():
    """
    Main function to generate data.
    """
    save = True
    add_to_existing = False
    nb_vert = 64

    ti0 = time.time()
    # F, phi, G, params = create_FG_numpy(nb_data=nb_data, nb_vert=nb_vert)
    F = np.load(f"../data/F.npy")
    G = np.load(f"../data/G.npy")
    phi = np.load(f"../data/Phi.npy")
    params = np.load(f"../data/agentParams.npy")

    solver = StdFemSolver(nb_cell=nb_vert - 1, params=params[:])
    U, XY, F, G, Phi, HH = solver.solve_several()
    print(np.mean(HH))
    if save:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        np.save(f"./data/U_geo_fno.npy", U)
        np.save(f"./data/XY_geo_fno.npy", XY)
        np.save(f"./data/F.npy", F)
        np.save(f"./data/phi.npy", Phi)
        np.save(f"./data/G.npy", G)
        np.save(f"./data/params.npy", params)
        np.save(f"./data/hmax.npy", HH)


if __name__ == "__main__":
    go()