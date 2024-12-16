import numpy as np
from matplotlib import rcParams
import matplotlib
#rcParams['text.usetex'] = True
#rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt

from lattice_visualizer.lattice_properties import scatter_lattice, color_fundamental_region, determinant, \
    fundamental_region_tiling, voronoi_tiling, outer_radius, inner_radius, color_orthogonalized_fundamental_region, \
    orthogonalized_tiling, orthogonalized_outer_radius, orthogonalized_inner_radius, show_basis

matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import io
import base64
from scipy.spatial import Voronoi, voronoi_plot_2d



BOUND = 15
l_color = "black"
fr_color = "hotpink"
orthogonalized_fr_color = "firebrick"
voronoi_tiling_color = "darkorange"
fr_tiling_color = "orangered"
inner_radius_color = "dimgrey"
outer_radius_color = "dimgrey"


def plot_lattice(matrix):
    description = f"The Structure of the Lattice L(B)"

    basis = np.array(matrix)

    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    scatter_lattice(basis)
    show_basis(basis)

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_fundamental_region(matrix):
    description = f"The Fundamental Region P(B) defined by B"

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Draw fundamental region
    show_basis(basis)
    color_fundamental_region(basis)
    # Show lattice point
    scatter_lattice(basis)
    # Calculate and display Det(L)
    determinant(basis)

    # Title and text
    plt.title("Fundamental Parallelepiped spanned by the basis vectors")

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_fundamental_region_tiling(matrix):
    description = f"The space span(L) tiled by P(B)"

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Draw fundamental region
    color_fundamental_region(basis, draw_vertices=False)
    # Show lattice point
    scatter_lattice(basis)
    # Calculate and display Det(L)
    determinant(basis)
    # Draw tiling through fundamental region
    fundamental_region_tiling(basis)

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_voronoi_tiling(matrix):
    description = f"The space span(L) tiled by Voronoi Tiling, that would solve CVP"

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Draw Voronoi Tiling
    voronoi_tiling(basis)
    # Draw lattice points
    scatter_lattice(basis)
    # Title and display
    plt.title("Lattice with Voronoi Tiling")

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_diff_fundamental_region_voronoi(matrix):
    description = f"The difference between simple tiling by the Fundamental Region compared to the voronoi tiling that solves CVP"

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Draw Voronoi Tiling
    voronoi_tiling(basis)
    # Draw tiling by fundamental region
    # fundamental_region_tiling(basis)
    color_fundamental_region(basis)
    # Draw lattice points
    scatter_lattice(basis)
    determinant(basis)
    inner_radius(basis)
    outer_radius(basis)


    # Title and display
    plt.title("Fundamental Region compared to Voronoi Tiling")
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_inner_radius(matrix):
    description = f"The inner radius of P(B)"

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Draw fundamental region
    color_fundamental_region(basis)
    # Show lattice point
    scatter_lattice(basis)
    # Calculate and display Det(L)
    determinant(basis)
    inner_radius(basis)
    # Title and text
    plt.title("Inner Radius of the Fundamental Region")

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_outer_radius(matrix):
    description = f"The outer radius of P(B)"

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Draw fundamental region
    color_fundamental_region(basis)
    # Show lattice point
    scatter_lattice(basis)
    # Calculate and display Det(L)
    determinant(basis)
    outer_radius(basis)

    # Title and text
    plt.title("Outer Radius of the Fundamental Region")
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_orth_region(matrix):
    description = f"How to get a better tiling? Orthogonalized Basis B* centered around origin as P(B*)."

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Draw fundamental region
    color_fundamental_region(basis)
    # Show lattice point
    scatter_lattice(basis)
    # Calculate and display Det(L)
    determinant(basis)
    color_orthogonalized_fundamental_region(basis)
    show_basis(basis)


    # Title and text
    plt.title("Orthoganalized Parallelepiped P(B*) spanned by the basis vectors")

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_orth_tiling(matrix):
    description = f"The orthogonalized Basis B* tiles the whole space span(B)"

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Draw fundamental region
    color_orthogonalized_fundamental_region(basis, draw_vertices=False)
    # Show lattice point
    scatter_lattice(basis)
    # Calculate and display Det(L)
    determinant(basis)
    # Draw tiling through orthogonalized basis
    orthogonalized_tiling(basis)

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_orth_vs_fundamental_region(matrix):
    description = f"The orthogonalized tiling P(B*) has same or better Radii than P(B)"

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Show lattice point
    scatter_lattice(basis)
    # Calculate and display Det(L)

    # Draw Tilings
    color_fundamental_region(basis)
    color_orthogonalized_fundamental_region(basis)

    # Draw Radi
    outer_radius(basis)
    inner_radius(basis)
    orthogonalized_outer_radius(basis)
    orthogonalized_inner_radius(basis)

    # Title and text
    plt.title("Inner and Outer Radi depending on the Tiling")
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data


def plot_diff_orth_fundamental_voronoi_tiling(matrix):
    description = f"P(B*) is nearer to the form of the Voronoi cell than P(B), thus approximates CVP better"

    basis = np.array(matrix)

    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))

    # Draw Voronoi Tiling
    voronoi_tiling(basis)

    # Draw Tilings
    # fundamental_region_tiling(basis)
    color_fundamental_region(basis)
    # orthogonalized_tiling(basis)
    color_orthogonalized_fundamental_region(basis)

    # Draw Radi
    outer_radius(basis)
    inner_radius(basis)
    orthogonalized_outer_radius(basis)
    orthogonalized_inner_radius(basis)

    # Draw lattice points
    scatter_lattice(basis)
    determinant(basis)

    # Title and display
    plt.title("Fundamental Region compared to Voronoi Tiling")

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return description, plot_data
