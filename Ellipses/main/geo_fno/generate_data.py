import numpy as np
import os
import numpy as np
from mpi4py import MPI
import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh
import matplotlib.pyplot as plt
import ufl
from dolfinx.fem.petsc import LinearProblem
from ufl import inner, jump, grad, div, dot, avg
from utils_plot import plot_mesh_pyvista, plot_meshes_pyvista
import random
from prepare_data import call_phi, call_F, call_G, eval_phi
from pymedit import P1Function, square, mmg2d, trunc
import time
import meshio
import matplotlib.pyplot as plt
import torch

np.pow = np.power

seed = 2102
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
# print(f"Random seed set as {seed}")

# parameter of the ghost penalty
sigma_D = 1.0
# Polynome Pk
polV = 1
polPhi = polV + 1


class PhiExpr:
    def __init__(self, x_0, y_0, lx, ly, theta):
        self.x_0 = x_0
        self.y_0 = y_0
        self.lx = lx
        self.ly = ly
        self.theta = theta

    def eval(self, x):
        return call_phi(x, self.x_0, self.y_0, self.lx, self.ly, self.theta)


class FExpr:
    def __init__(self, mu0, mu1, sigma_x, sigma_y, amplitude):
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.amplitude = amplitude

    def eval(self, x):
        return call_F(x, self.mu0, self.mu1, self.sigma_x, self.sigma_y, self.amplitude)


class GExpr:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def eval(self, x):
        return call_G(x, self.alpha, self.beta)


def near(a, b, tol=3e-16):
    """
    Check if two numbers 'a' and 'b' are close to each other within a tolerance 'tol'.
    """
    return np.abs(a - b) <= tol


# Polynome Pk
polV = 1
polPhi = polV + 1


class StdFemSolver:
    def __init__(self, nb_cell, params):

        self.params = params

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

    def create_standard_mesh(self, x_0, y_0, lx, ly, theta, phi, hmax=0.05):
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
        phi = phi.flatten()
        phiP1 = P1Function(M, phi)
        hmax = 0.006
        newM = mmg2d(
            M,
            hmin=hmax / 1.7,
            sol=phiP1,
            ls=True,
            verb=0,
            extra_args="-hsiz " + str(hmax / 1.5),
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
        self.create_points_file(points)
        command_1 = "mmg2d_O3 -noinsert -in points.mesh -out mesh.mesh -v 0"  # optimize the mesh, keeping the nb of vertices fixed
        os.system(command_1)

        def create_mesh(mesh, cell_type, prune_z=False):
            cells = mesh.get_cells_type(cell_type)
            if prune_z:
                points = mesh.points[:, :2]
            else:
                points = mesh.points
            out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
            return out_mesh

        in_mesh = meshio.read("mesh.mesh")
        out_mesh = create_mesh(in_mesh, "triangle", True)
        meshio.write("mesh.xdmf", out_mesh)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")

        return mesh

    def solve_one(self, i):
        """Computation of phiFEM

        Args:
            i (int): index of the problem to solve

        Returns:
            np array : matrix of the phiFEM solution
        """
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
        phi = np.reshape(phi, (511, 511)).T
        mesh = self.create_standard_mesh(
            x_0,
            y_0,
            lx,
            ly,
            theta,
            phi=phi,
        )

        V = dolfinx.fem.functionspace(mesh, ("CG", polV))
        V2 = dolfinx.fem.functionspace(mesh, ("CG", 2))
        F_expr = FExpr(mu0, mu1, sigma_x, sigma_y, amplitude)
        f_expr = dolfinx.fem.Function(V2)
        f_expr.interpolate(F_expr.eval)

        G_expr = GExpr(alpha, beta)
        g_expr = dolfinx.fem.Function(V)
        g_expr.interpolate(G_expr.eval)

        Phi_expr = PhiExpr(x_0, y_0, lx, ly, theta)
        phi_expr = dolfinx.fem.Function(V)
        phi_expr.interpolate(Phi_expr.eval)

        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(fdim, tdim)
        boundary_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(g_expr, boundary_dofs)
        dx = ufl.Measure("dx", domain=mesh)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        L = f_expr * v * dx
        problem = LinearProblem(
            a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        u_h = problem.solve()
        f_expr = dolfinx.fem.Function(V)
        f_expr.interpolate(F_expr.eval)

        num_cells = (
            mesh.topology.index_map(mesh.topology.dim).size_local
            + mesh.topology.index_map(mesh.topology.dim).num_ghosts
        )
        hmax = max(mesh.h(2, np.array(list(range(num_cells)))))

        return (
            u_h.x.array[:],
            V.tabulate_dof_coordinates()[:, :2],
            f_expr.x.array[:],
            g_expr.x.array[:],
            phi_expr.x.array[:],
            hmax,
        )

    def solve_several(self):
        U, F, G, Phi = [], [], [], []
        XY = []
        hh = []
        nb = len(self.params)
        for i in range(nb):
            print(f"Data {i}/{nb}")
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
    nb_vert = 64

    params = np.load(f"../../data/params.npy")

    solver = StdFemSolver(nb_cell=nb_vert - 1, params=params[:])
    U, XY, F, G, Phi, HH = solver.solve_several()
    if save:
        np.save(f"../../data/U_geo_fno.npy", U)
        np.save(f"../../data/XY_geo_fno.npy", XY)
        np.save(f"../../data/F_geo_fno.npy", F)
        np.save(f"../../data/phi_geo_fno.npy", Phi)
        np.save(f"../../data/G_geo_fno.npy", G)
        np.save(f"../../data/hmax_geo_fno.npy", HH)


if __name__ == "__main__":
    go()
