import numpy as np
from mpi4py import MPI
import vtk
import pyvista
import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.tri as tri  # noqa: I2023
import mpl_toolkits.axes_grid1
import typing
import ufl
import dolfinx.plot as plot
import seaborn as sns

sns.set_theme("paper")


def plot_mesh_matplotlib(
    mesh: dolfinx.mesh.Mesh, ax: typing.Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot a mesh object on the provied (or, if None, the current) axes object."""
    if ax is None:
        ax = plt.gca()
        ax.set_aspect("equal")
        points = mesh.geometry.x
        cells = mesh.geometry.dofmap.reshape((-1, mesh.topology.dim + 1))
        tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
        ax.triplot(tria, color="k")
    return ax


def plot_meshes_matplotlib(meshes, ax: typing.Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    colors = ["r", "b", "k", "c"]

    for i in range(len(meshes)):
        mesh = meshes[i]
        color = colors[i]
        points = mesh.geometry.x
        cells = mesh.geometry.dofmap.reshape((-1, mesh.topology.dim + 1))
        if len(points[:, 0] > 0):
            tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
            ax.triplot(tria, color=color)

    return ax


def plot_mesh_pyvista(mesh):

    nb_functions = 1
    height, width = 400, 400 * nb_functions
    subplotter = pyvista.Plotter(
        shape=(1, nb_functions), window_size=[width, height], notebook=False
    )
    cells, types, x = plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    subplotter.subplot(0, 0)
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
    subplotter.view_xy()

    subplotter.show()


def plot_meshes_pyvista(meshes, labels=[]):

    nb_functions = 1
    height, width = 400, 400 * nb_functions
    subplotter = pyvista.Plotter(
        shape=(1, nb_functions), window_size=[width, height], notebook=False
    )

    subplotter.subplot(0, 0)
    colors = [[0.6, 0.2, 0.1], [0.1, 0.6, 0.6]]
    for i in range(len(meshes)):
        if len(labels) == 0:
            label = ""
        else:
            label = labels[i]
        color = colors[i]
        cells, types, x = plot.vtk_mesh(meshes[i])
        grid = pyvista.UnstructuredGrid(cells, types, x)
        subplotter.add_mesh(grid, show_edges=True, color=color, label=label)
    subplotter.view_xy()
    subplotter.add_legend()
    subplotter.add_axes()
    subplotter.show()


def plot_mesh_tags(mesh, mesh_tags, ax=None) -> plt.Axes:
    """Plot a mesh tags object on the provied (or, if None, the current) axes object."""
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    # mesh = mesh_tags.mesh
    points = mesh.geometry.x
    colors = ["b", "r", "k", "c"]
    cmap = matplotlib.colors.ListedColormap(colors)
    cmap_bounds = [0, 0.5, 1]
    norm = matplotlib.colors.BoundaryNorm(cmap_bounds, cmap.N)
    assert mesh_tags.dim in (mesh.topology.dim, mesh.topology.dim - 1)
    if mesh_tags.dim == mesh.topology.dim:
        cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
        tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
        cell_colors = np.zeros((cells.shape[0],))
        cell_colors[mesh_tags.indices[mesh_tags.values == 1]] = 1
        mappable = ax.tripcolor(tria, cell_colors, edgecolor="k", cmap=cmap, norm=norm)
    elif mesh_tags.dim == mesh.topology.dim - 1:
        tdim = mesh.topology.dim
        geometry_dofmap = mesh.geometry.dofmap
        cells_map = mesh.topology.index_map(mesh.topology.dim)
        num_cells = cells_map.size_local + cells_map.num_ghosts
        connectivity_cells_to_facets = mesh.topology.connectivity(tdim, tdim - 1)
        connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
        connectivity_facets_to_vertices = mesh.topology.connectivity(tdim - 1, 0)
        vertex_map = {
            topology_index: geometry_index
            for c in range(num_cells)
            for (topology_index, geometry_index) in zip(
                connectivity_cells_to_vertices.links(c), geometry_dofmap.links(c)
            )
        }
        linestyles = [(0, (5, 10)), "solid"]
        lines = list()
        lines_colors_as_int = list()
        lines_colors_as_str = list()
        lines_linestyles = list()
        mesh_tags_1 = mesh_tags.indices[mesh_tags.values == 1]
        for c in range(num_cells):
            facets = connectivity_cells_to_facets.links(c)
            for f in facets:
                if f in mesh_tags_1:
                    value_f = 1
                else:
                    value_f = 0
                vertices = [
                    vertex_map[v] for v in connectivity_facets_to_vertices.links(f)
                ]
                lines.append(points[vertices][:, :2])
                lines_colors_as_int.append(value_f)
                lines_colors_as_str.append(colors[value_f])
                lines_linestyles.append(linestyles[value_f])
        mappable = matplotlib.collections.LineCollection(
            lines,
            cmap=cmap,
            norm=norm,
            colors=lines_colors_as_str,
            linestyles=lines_linestyles,
        )
        mappable.set_array(np.array(lines_colors_as_int))
        ax.add_collection(mappable)
        ax.autoscale()
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mappable, cax=cax, boundaries=cmap_bounds, ticks=cmap_bounds)
    return ax


def plot_scalar_functions_list(U, labels=[], screenshot=False, height=400, shape=None):

    nb_functions = len(U)
    if shape == None:
        total_width = height * nb_functions
        total_height = height
        shape = (1, nb_functions)
    else:
        total_width = height * shape[0]
        total_height = height * shape[1]
    subplotter = pyvista.Plotter(
        shape=shape,
        window_size=[total_width, total_height],
        notebook=True,
        border=False,
    )
    empty_str = ""
    i, j = 0, 0
    for index, u in enumerate(U):
        V = u.function_space
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = u.x.array
        grid.set_active_scalars("u")
        if index == shape[1]:
            j = 0
            i += 1
        subplotter.subplot(i, j)
        empty_str += " "
        if len(labels) > 0:
            title_i = labels[index]
        else:
            title_i = ""
        subplotter.add_title(title_i, font_size=10)
        subplotter.add_mesh(grid, show_edges=False, show_scalar_bar=False)
        subplotter.add_scalar_bar(
            title=empty_str,
            height=0.1,
            width=0.8,
            vertical=False,
            position_x=0.1,
            position_y=0.01,
            fmt="%1.1e",
            title_font_size=0,
            color="black",
            label_font_size=14,
        )
        j += 1
        subplotter.view_xy()

    subplotter.show(screenshot=screenshot)


def plot_scalar_functions_list_warped(U, labels=[], screenshot=False):

    nb_functions = len(U)
    height, width = 400, 400 * nb_functions
    subplotter = pyvista.Plotter(
        shape=(1, nb_functions), window_size=[width, height], notebook=True
    )
    empty_str = ""
    for index, u in enumerate(U):
        V = u.function_space
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = u.x.array
        grid.set_active_scalars("u")
        subplotter.subplot(0, index)
        warped = grid.warp_by_scalar()
        empty_str += " "
        if len(labels) > 0:
            title_i = labels[index]
        else:
            title_i = ""
        subplotter.add_title(title_i, font_size=10)
        subplotter.set_position([0.0, 0.0, -10.0])
        subplotter.set_focus([0, 0.0, 0.0])
        subplotter.set_viewup([0, 0, 0])
        subplotter.add_mesh(warped, show_edges=False, show_scalar_bar=False)
        subplotter.add_scalar_bar(
            title=empty_str,
            height=0.1,
            width=0.8,
            vertical=False,
            position_x=0.1,
            position_y=0.05,
            fmt="%1.2e",
            title_font_size=0,
            color="black",
            label_font_size=14,
        )
        subplotter.view_xy()
        subplotter.camera.elevation = -60

    subplotter.show(screenshot=screenshot)


def plot_2D_vector_function_list(U, labels=[], mode="displaced", screenshot=False):

    nb_functions = len(U)
    height, width = 400, 400 * nb_functions
    subplotter = pyvista.Plotter(
        shape=(1, nb_functions), window_size=[width, height], notebook=True
    )
    empty_str = ""
    if mode == "displaced":
        factor = 1.0
    else:
        factor = 0

    for index, u in enumerate(U):
        V = u.function_space
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        u_3D = np.zeros((x.shape[0], 3))
        u_3D[:, :2] = u.x.array.reshape(-1, 2)
        grid.point_data["Displacement"] = u_3D
        grid.set_active_vectors("Displacement")
        warped = grid.warp_by_vector("Displacement", factor=factor)
        subplotter.subplot(0, index)
        empty_str += " "
        if len(labels) > 0:
            title_i = labels[index]
        else:
            title_i = ""
        subplotter.add_title(title_i, font_size=10)
        subplotter.set_position([0.0, 0.0, 0.0])
        subplotter.set_focus([0, 0.0, 0.0])
        subplotter.set_viewup([0, 0, 0])
        subplotter.add_mesh(warped, show_edges=False, show_scalar_bar=False)
        subplotter.add_scalar_bar(
            title=empty_str,
            height=0.1,
            width=0.8,
            vertical=False,
            position_x=0.1,
            position_y=0.01,
            fmt="%1.2e",
            title_font_size=0,
            color="black",
            label_font_size=14,
        )
        subplotter.view_xy()
        subplotter.camera.elevation = 0

    subplotter.show(screenshot=screenshot)


def plot_3D_vector_function_list(U, labels=[], mode="displaced", screenshot=False):

    nb_functions = len(U)
    height, width = 400, 400 * nb_functions
    subplotter = pyvista.Plotter(
        shape=(1, nb_functions), window_size=[width, height], notebook=True
    )
    empty_str = ""
    if mode == "displaced":
        factor = 1.0
    else:
        factor = 0

    for index, u in enumerate(U):
        V = u.function_space
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        u_3D = np.zeros((x.shape[0], 3))
        u_3D[:, :] = u.x.array.reshape(-1, 3)
        grid.point_data["Displacement"] = u_3D
        grid.set_active_vectors("Displacement")
        warped = grid.warp_by_vector("Displacement", factor=factor)
        subplotter.subplot(0, index)
        empty_str += " "
        if len(labels) > 0:
            title_i = labels[index]
        else:
            title_i = ""
        subplotter.add_title(title_i, font_size=10)
        subplotter.set_position([0.0, 0.0, 0.0])
        subplotter.set_focus([0, 0.0, 0.0])
        subplotter.set_viewup([0, 0, 0])
        subplotter.add_mesh(warped, show_edges=False, show_scalar_bar=False)
        subplotter.add_scalar_bar(
            title=empty_str,
            height=0.1,
            width=0.8,
            vertical=False,
            position_x=0.1,
            position_y=0.01,
            fmt="%1.2e",
            title_font_size=0,
            color="black",
            label_font_size=14,
        )
        subplotter.view_xy()
        subplotter.camera.elevation = 0

    subplotter.show(screenshot=screenshot)
