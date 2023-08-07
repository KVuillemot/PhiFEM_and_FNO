import dolfin as df
import vedo
import vedo.dolfin as vdf
import mshr
import numpy as np
from vedo import Latex
from vedo import shapes
from vedo.dolfin import plot, show, shapes

Plot = "cells_dirichlet"

lx = 0.38
ly = 0.28
x_0, y_0 = 0.5, 0.5
theta = 0.28
domain = mshr.CSGRotation(
    mshr.Ellipse(df.Point(x_0, y_0), lx, ly), df.Point(x_0, y_0), theta
)

domain_bigger = mshr.CSGRotation(
    mshr.Ellipse(df.Point(x_0, y_0), lx + 0.0025, ly + 0.0025),
    df.Point(x_0, y_0),
    theta,
)
inside = mshr.CSGRotation(
    mshr.Ellipse(df.Point(x_0, y_0), lx - 0.0025, ly - 0.0025),
    df.Point(x_0, y_0),
    theta,
)


class phi_expr(df.UserExpression):
    def eval(self, value, x):
        value[0] = -1.0 + (
            ((x[0] - x_0) * np.cos(theta) + (x[1] - y_0) * np.sin(theta)) ** 2
            / lx**2
            + ((x[0] - x_0) * np.sin(theta) - (x[1] - y_0) * np.cos(theta))
            ** 2
            / ly**2
        )

    def value_shape(self):
        return (2,)


if Plot == "domain":
    mesh = mshr.generate_mesh(domain, 300)
    background_mesh = df.RectangleMesh(
        df.Point(0.0, 0.0), df.Point(1.0, 1.0), 16, 16
    )
    vdf.plot(background_mesh, c="white", lw=0.0, interactive=False, axes=0)
    vdf.plot(mesh, c="gray", axes=0, lw=0, add=True, interactive=True)
    vedo.close()

if Plot == "cells_dirichlet":
    degPhi = 2
    background_mesh = df.RectangleMesh(
        df.Point(0.0, 0.0), df.Point(1.0, 1.0), 16, 16
    )
    V_phi = df.FunctionSpace(background_mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    Cell_omega = df.MeshFunction(
        "size_t", background_mesh, background_mesh.topology().dim()
    )
    Cell_omega.set_all(0)
    for cell in df.cells(background_mesh):
        v1, v2, v3 = df.vertices(cell)
        if (
            phi(v1.point()) <= 0.0
            or phi(v2.point()) <= 0.0
            or phi(v3.point()) <= 0.0
            or df.near(phi(v1.point()), 0.0)
            or df.near(phi(v2.point()), 0.0)
            or df.near(phi(v3.point()), 0.0)
        ):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(background_mesh, Cell_omega, 1)

    # Creation of the FunctionSpace for Phi on Omega_h
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    # Selection of cells and facets on the boundary for Omega_h^Gamma
    mesh.init(1, 2)
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    Facet.set_all(0)
    Cell.set_all(0)

    for cell in df.cells(mesh):
        for facet in df.facets(cell):
            v1, v2 = df.vertices(facet)
            if phi(v1.point()) * phi(v2.point()) <= 0.0 or df.near(
                phi(v1.point()) * phi(v2.point()), 0.0
            ):
                # check if the cell is a cell for Dirichlet condition or Neumann condition and add every cells, facets, vertices to the restricition
                vc1, vc2, vc3 = df.vertices(cell)
                # Cells for dirichlet condition
                Cell[cell] = 1
                for facett in df.facets(cell):
                    Facet[facett] = 1
    boundary = df.SubMesh(mesh, Cell, 1)

    ellipse = mshr.generate_mesh(domain_bigger - inside, 100)

    plt = vdf.plot(
        background_mesh,
        c="white",
        interactive=False,
        axes=0,
    )
    plt += vdf.plot(
        mesh,
        c="blue8",
        interactive=False,
        axes=0,
    )
    plt += vdf.plot(
        boundary,
        c="gray4",
        axes=0,
        lw=2.8,
        interactive=False,
    )
    plt += vdf.plot(
        ellipse, c="red4", lw=0, axes=0, add=True, interactive=True
    )

    plt += vdf.plot(
        shapes.Arrow2D((0.49, 0.86, 0.0), (0.39, 0.78, 0)),
        add=True,
        axes=0,
    )  # dirichlet boundary
    plt += vdf.plot(
        shapes.Arrow2D((0.82, 0.84, 0.0), (0.72, 0.76, 0)),
        add=True,
        axes=0,
    )  # real boundary dirichlet

    plt += vdf.plot(
        shapes.Arrow2D((0.195, 0.14, 0.0), (0.155, 0.28, 0)),
        add=True,
        axes=0,
    )  # facets

    plt += vdf.plot(
        shapes.Arrow2D((0.75, 0.133, 0.0), (0.65, 0.213, 0)),
        add=True,
        axes=0,
    )  # discrete_boundary

    actors = plt.actors[:]

    Gamma_D = r"\Gamma"
    formula_1 = Latex(Gamma_D, c="r", s=0.25, usetex=True, res=60).pos(
        0.60, 0.68, 0
    )

    F_h_D = r"E \in \mathcal{F}_h^{\Gamma}"
    formula_2 = vedo.Latex(F_h_D, c="k", s=0.25, usetex=True, res=60).pos(
        -0.10, -0.11, 0
    )
    Omh_Gamma_D = r"\Omega_h^{\Gamma}"
    formula_3 = vedo.Latex(
        Omh_Gamma_D, c="gray3", s=0.25, usetex=True, res=60
    ).pos(0.31, 0.68, 0)

    T_h_O = r"\mathcal{T}_h^{\mathcal{O}}"
    formula_4 = vedo.Latex(T_h_O, c="k", s=0.25, usetex=True, res=60).pos(
        -0.06, 0.65, 0
    )

    T_h = r"\mathcal{T}_h \setminus \mathcal{T}_h^{\Gamma}"
    formula_5 = vedo.Latex(T_h, c="blue2", s=0.25, usetex=True, res=60).pos(
        0.25, 0.35, 0
    )

    partial_omega = r"E \in \partial \Omega_h"
    formula_6 = vedo.Latex(
        partial_omega, c="k", s=0.25, usetex=True, res=60
    ).pos(0.45, -0.11, 0)

    actors.append(
        [formula_1, formula_2, formula_3, formula_4, formula_5, formula_6]
    )
    vedo.show(
        actors, formula_1, formula_2, formula_3, formula_4, formula_5, axes=0
    )

    vedo.close()
