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
    call_F,
)
from dolfin import *

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"


degV = 1
degPhi = degV + 1


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
        value[0] = call_F(df, x, self.mu0, self.mu1, self.sigma)


def build_polygon_points(phi):
    mesh_macro = df.UnitSquareMesh(4096 - 1, 4096 - 1)
    V = df.FunctionSpace(mesh_macro, "CG", 1)
    phi = df.interpolate(phi, V)

    """
    Steps to create the standard mesh : 
        1. Use the image of the domain given by the level-set 
        2. Get the vertices which are on the boundary
        3. Order the vertices (counter clockwise) 
        4. Create a list of FEniCS points with the coordinates of the ordered vertices
        5. Create a mshr Polygon from this list of points
        6. Generate a mesh
    """

    mesh_macro.init(0, 1)
    boundary_points = []
    for v in df.vertices(mesh_macro):
        if phi(v.point()) <= 1e-6 and phi(v.point()) >= -1.0e-4:
            boundary_points.append(np.array([v.point().x(), v.point().y()]))

    boundary_points = np.array(boundary_points)
    Points = list(np.unique(boundary_points, axis=0))
    p0 = Points[0]
    ordered_points = [p0]
    temp_points = np.array(Points)
    temp_points = np.delete(temp_points, [0], 0)

    mesh_macro = df.UnitSquareMesh(10, 10)  # free memory
    # order the points
    while len(ordered_points) != len(Points):
        if len(ordered_points) == 1:
            distances_points = [
                np.linalg.norm(temp_points[i] - p0)
                for i in range(len(temp_points))
            ]
            index_min = np.argmin(distances_points)
            p0 = temp_points[index_min]
            ordered_points.append(p0)
            temp_points = np.delete(temp_points, [index_min], 0)

        else:
            distances_points = [
                np.linalg.norm(temp_points[i] - p0)
                for i in range(len(temp_points))
            ]
            for i in range(len(distances_points)):
                if distances_points[i] > np.linalg.norm(
                    ordered_points[-2] - temp_points[i], ord=2
                ):
                    distances_points[i] += 1e5

            index_min = np.argmin(distances_points)
            p0 = temp_points[index_min]
            ordered_points.append(p0)
            temp_points = np.delete(temp_points, [index_min], 0)

    ordered_points = np.array(ordered_points)

    return ordered_points


def build_mesh_from_polygon(ordered_points, size, plot_mesh=False):
    fenics_points = [
        df.Point(ordered_points[i]) for i in range(0, len(ordered_points), 1)
    ]
    fenics_points += [fenics_points[0]]

    if size == None:
        min_size = 0.0008
        H_mesh = 600

    else:
        min_size = df.UnitSquareMesh(size - 1, size - 1).hmax()
        H_mesh = 20
        alpha = int(min_size / 0.0008) + 1
        fenics_points = [
            fenics_points[i] for i in range(0, len(fenics_points), alpha)
        ]

    try:
        polygon = mshr.Polygon(fenics_points)
    except:
        polygon = mshr.Polygon(fenics_points[::-1])
    mesh = mshr.generate_mesh(polygon, H_mesh)
    while mesh.hmax() > min_size:
        if size == None:
            H_mesh += 20
        else:
            H_mesh += 2
        mesh = mshr.generate_mesh(polygon, H_mesh)
        print(f"{H_mesh =}    {mesh.hmax() =}")
    if plot_mesh:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(ordered_points[:, 0], ordered_points[:, 1], "-+")
        plt.subplot(1, 2, 2)
        df.plot(mesh)
        plt.tight_layout()
        plt.show()
    return mesh


def build_exact_mesh(phi, size=None, plot_meshes=False):
    mesh_macro = df.UnitSquareMesh(4096 - 1, 4096 - 1)
    V = df.FunctionSpace(mesh_macro, "CG", 1)
    phi = df.interpolate(phi, V)

    """
    Steps to create the standard mesh : 
        1. Use the image of the domain given by the level-set 
        2. Get the vertices which are on the boundary
        3. Order the vertices (counter clockwise) 
        4. Create a list of FEniCS points with the coordinates of the ordered vertices
        5. Create a mshr Polygon from this list of points
        6. Generate a mesh
    """

    mesh_macro.init(0, 1)
    boundary_points = []
    for v in df.vertices(mesh_macro):
        if phi(v.point()) <= 1e-6 and phi(v.point()) >= -1.0e-4:
            boundary_points.append(np.array([v.point().x(), v.point().y()]))

    boundary_points = np.array(boundary_points)
    Points = list(np.unique(boundary_points, axis=0))
    p0 = Points[0]
    ordered_points = [p0]
    temp_points = np.array(Points)
    temp_points = np.delete(temp_points, [0], 0)
    # order the points
    while len(ordered_points) != len(Points):
        if len(ordered_points) == 1:
            distances_points = [
                np.linalg.norm(temp_points[i] - p0)
                for i in range(len(temp_points))
            ]
            index_min = np.argmin(distances_points)
            p0 = temp_points[index_min]
            ordered_points.append(p0)
            temp_points = np.delete(temp_points, [index_min], 0)

        else:
            distances_points = [
                np.linalg.norm(temp_points[i] - p0)
                for i in range(len(temp_points))
            ]
            for i in range(len(distances_points)):
                if distances_points[i] > np.linalg.norm(
                    ordered_points[-2] - temp_points[i], ord=2
                ):
                    distances_points[i] += 1e5

            index_min = np.argmin(distances_points)
            p0 = temp_points[index_min]
            ordered_points.append(p0)
            temp_points = np.delete(temp_points, [index_min], 0)

    fenics_points = [
        df.Point(ordered_points[i]) for i in range(0, len(ordered_points), 1)
    ]
    fenics_points += [fenics_points[0]]

    ordered_points = np.array(ordered_points)

    try:
        polygon = mshr.Polygon(fenics_points)
    except:
        polygon = mshr.Polygon(fenics_points[::-1])

    if size == None:
        min_size = 0.0008
        H_mesh = 600

    else:
        min_size = df.UnitSquareMesh(size - 1, size - 1).hmax()
        H_mesh = 20
    mesh = mshr.generate_mesh(polygon, H_mesh)
    while mesh.hmax() > min_size:
        if size == None:
            H_mesh += 20
        else:
            H_mesh += 2
        mesh = mshr.generate_mesh(polygon, H_mesh)
        print(f"{H_mesh =}    {mesh.hmax() =}")
    if plot_meshes:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(ordered_points + [ordered_points[0]], "-+")
        plt.subplot(1, 2, 2)
        df.plot(mesh)
        plt.tight_layout()
        plt.show()
    return mesh


def compute_standard_fem(
    nb_vert, param, phi, exact_fem=False, plot_meshes=False
):
    mu0, mu1, sigma = param
    print(f"(mu0, mu1, sigma) = {param}")
    phi = convert_numpy_matrix_to_fenics(
        phi, np.shape(phi)[0], 1
    )  # P1 function for standard FEM

    # construction of the mesh
    if exact_fem:
        mesh = build_exact_mesh(phi, None, plot_meshes)

    else:
        mesh = build_exact_mesh(phi, nb_vert, plot_meshes)

    # FunctionSpace Pk
    V = df.FunctionSpace(mesh, "CG", degV)
    boundary = "on_boundary"

    f = FExpr(mu0, mu1, sigma, degree=degV + 2, domain=mesh)
    u_D = df.Constant("0.0")
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
    total = end - start
    u_h = u
    if exact_fem:
        return (
            df.project(
                u_h, V, solver_type="gmres", preconditioner_type="hypre_amg"
            ),
            V,
            dx,
        )
    else:
        return df.project(u_h, V), total, mesh.hmax()


class StandardFEMSolver:
    def __init__(self, params, phi_vector):
        self.params = params
        self.phi_vector = phi_vector
        self.phi_fenics = convert_numpy_matrix_to_fenics(
            self.phi_vector, np.shape(self.phi_vector)[0], 1
        )
        self.polygon_points = build_polygon_points(self.phi_fenics)

    def solve_one(self, i, size, plot_mesh):
        mu0, mu1, sigma = self.params[i]
        print(f"(mu0, mu1, sigma) = {self.params}")

        mesh = build_mesh_from_polygon(self.polygon_points, size, plot_mesh)

        # FunctionSpace Pk
        V = df.FunctionSpace(mesh, "CG", degV)
        boundary = "on_boundary"

        f = FExpr(mu0, mu1, sigma, degree=degV + 2, domain=mesh)
        u_D = df.Constant("0.0")
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
        total = end - start
        u_h = u

        mesh = df.UnitSquareMesh(10, 10)  # free memory
        if size == None:  # meaning exact FEM true
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


class PhiFemSolver_error:
    def __init__(self, nb_cell, params, phi_vector):
        self.N = N = nb_cell
        self.params = params
        self.phi_vector = phi_vector
        self.mesh_macro = df.RectangleMesh(
            df.Point(0.0, 0.0), df.Point(1.0, 1.0), self.N, self.N
        )
        self.V_macro = FunctionSpace(self.mesh_macro, "CG", degV)

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
        phi_numpy = self.phi_vector[:, :]
        phi_origin = convert_numpy_matrix_to_fenics(
            phi_numpy, int((np.shape(phi_numpy)[0] + 1) / 2)
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

        V = FunctionSpace(mesh, "CG", degV)
        V_phi = FunctionSpace(mesh, "CG", degPhi)
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

        f_expr = FExpr(mu0, mu1, sigma, degree=degV + 2, domain=mesh)
        sigma_D = 20.0
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
        start = time.time()
        solve(a == L, w_h)
        end = time.time()
        total = end - start
        return (
            df.project(phi * w_h, V),
            V,
            total,
            self.make_matrix_of_phi(phi_origin),
            mesh.hmax(),
        )


if __name__ == "__main__":
    print(0)
    l2_error = []
    hh = []
    hh_phi, error_phi = [], []
    u_ex, V_ex, dx_ex = compute_standard_fem(
        None,
        [0.3, 0.4, 0.2],
        np.load(
            "../shape_generation/compare_methods/test_level_set_10_1023.npy"
        )[3],
        exact_fem=True,
    )
    for nb_vert in [16, 32, 64, 128, 256]:
        u_h, times, size_mesh = compute_standard_fem(
            nb_vert,
            [0.3, 0.4, 0.2],
            np.load(
                "../shape_generation/compare_methods/test_level_set_10_1023.npy"
            )[3],
        )
        hh.append(size_mesh)
        l2_error.append(
            (
                df.assemble(
                    (
                        (
                            (
                                u_ex
                                - df.project(
                                    u_h,
                                    V_ex,
                                    solver_type="gmres",
                                    preconditioner_type="hypre_amg",
                                )
                            )
                        )
                        ** 2
                    )
                    * dx_ex
                )
                ** (0.5)
            )
            / (df.assemble((((u_ex)) ** 2) * dx_ex) ** (0.5))
        )

        solver = PhiFemSolver_error(
            nb_cell=nb_vert - 1,
            params=[[0.3, 0.4, 0.2]],
            phi_vector=np.load(
                "../shape_generation/compare_methods/test_level_set_10_1023.npy"
            )[3],
        )
        (
            u_phi_fem,
            V_phi_fem,
            temps_phi,
            phiiii,
            size_mesh_phi,
        ) = solver.solve_one(0)
        hh_phi.append(size_mesh_phi)
        error_phi.append(
            (
                df.assemble(
                    (
                        (
                            (
                                u_ex
                                - df.project(
                                    u_phi_fem,
                                    V_ex,
                                    solver_type="gmres",
                                    preconditioner_type="hypre_amg",
                                )
                            )
                        )
                        ** 2
                    )
                    * dx_ex
                )
                ** (0.5)
            )
            / (df.assemble((((u_ex)) ** 2) * dx_ex) ** (0.5))
        )
    plt.figure()
    plt.loglog(hh, l2_error, "-+", label="L2 error std")
    plt.loglog(hh_phi, error_phi, "-+", label="L2 error std")
    plt.loglog(hh, [h**2 for h in hh], label="$O(h^2)$")
    plt.legend()
    plt.show()
