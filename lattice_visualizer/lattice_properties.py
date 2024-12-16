import numpy as np
from matplotlib import rcParams
import matplotlib
#rcParams.update(matplotlib.rcParamsDefault)
#rcParams['text.usetex'] = True
#rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import io
import base64
from scipy.spatial import Voronoi, voronoi_plot_2d



BOUND = 15
l_color = "black"
basis_color = "dimgrey"
fr_color = "hotpink"
orthogonalized_fr_color = "firebrick"
voronoi_tiling_color = "darkorange"
fr_tiling_color = "orangered"
inner_radius_color = "dimgrey"
outer_radius_color = "dimgrey"


def show_basis(basis):
    v1 = basis[0]
    v2 = basis[1]
    plt.arrow(0, 0, v1[0], v1[1], length_includes_head=True,
          head_width=0.5, head_length=0.5, color=basis_color)
    plt.arrow(0, 0, v2[0], v2[1], length_includes_head=True,
          head_width=0.5, head_length=0.5, color=basis_color)


def scatter_lattice(basis):
    # Create lists to hold the x and y coordinates of the points
    x_vals = []
    y_vals = []

    # Loop over all combinations of integer coefficients for the linear combination
    for a in range(-BOUND * 10, BOUND * 10 + 1):
        for b in range(-BOUND * 10, BOUND * 10 + 1):
            point = a * basis[0] + b * basis[1]
            if abs(point[0]) <= BOUND and abs(point[1]) <= BOUND:
                x_vals.append(point[0])
                y_vals.append(point[1])

    # Plot the scatter plot
    plt.axhline(0, color='darkgrey', linewidth=1)
    plt.axvline(0, color='darkgrey', linewidth=1)
    plt.scatter(0, 0, color='red', marker='o', s=50,
                label='Origin (0, 0)')  # bigger red point for originplt.ylim(-BOUND, BOUND)
    plt.scatter(x_vals, y_vals, color=l_color, marker='o', s=10)  # smaller points with s=10plt.xlim(-BOUND, BOUND)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Lattice")
    matrix_text = r'$B=\leftparen~\genfrac{{}}{{}}{{0}}{{0}}{{{}}}{{{}}}~\genfrac{{}}{{}}{{0}}{{0}}{{{}}}{{{}}}~\rightparen$'.format(basis[0][0], basis[0][1], basis[1][0], basis[1][1])
    #    matrix_text = r'$A=\genfrac{{[}}{{]}}{{0}}{{0}}{{{}~ {}}}{{{} ~ {}}}$'.format(basis[0][0], basis[1][0], basis[0][1], basis[1][1])
    #matrix_text = r"Basis-Matrix $B = \genfrac{[}{]}{0}{3}{\mathtt{\,{}\;{}}}{\mathtt{\,{}\;{}}}$" \
    #    .format(basis[0][0], basis[1][0], basis[0][1], basis[1][1])
    plt.text(BOUND + 5, BOUND - 5, matrix_text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))

    plt.xlim([-BOUND, BOUND])
    plt.ylim([-BOUND, BOUND])
    plt.grid(True, which="minor")

def determinant(basis):
    # Calculate the area of the parallelogram (absolute value of the determinant)
    area = abs(np.linalg.det(basis))
    plt.text(BOUND+5, BOUND-10, f"det(L): {round(area, 2)}", fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))


def color_fundamental_region(basis, draw_vertices=True):
    # Extract the two column vectors
    v1 = basis[0]
    v2 = basis[1]

    # Define the vertices of the parallelepiped and center around origin
    v_norm = 0.5 * (v1 + v2)
    vertices = np.array([
        [0, 0] - v_norm,
        v1 - v_norm,
        (v1 + v2) - v_norm,
        v2 - v_norm
    ])

    # Plot the points of the parallelogram
    if draw_vertices:
        plt.scatter(vertices[:, 0], vertices[:, 1], color=fr_color, zorder=5, s=10)

    # Draw the edges of the parallelogram
    for i in range(4):
        plt.plot([vertices[i][0], vertices[(i + 1) % 4][0]],
                 [vertices[i][1], vertices[(i + 1) % 4][1]], color=fr_color)
    plt.fill(vertices[:, 0], vertices[:, 1], color=fr_color, alpha=0.2)

    # Title and text
    plt.title("Fundamental Parallelepiped spanned by the basis vectors")


def color_orthogonalized_fundamental_region(basis, draw_vertices=True):
    def pi(b1, b2):
        return b2 - b1.dot(b2) / b1.dot(b1) * b1

    def gram_schmidt_orthogonalization(basis):
        b1_star = basis[0]
        b2_star = pi(basis[0], basis[1])
        return np.array([b1_star, b2_star])

    basis_star = gram_schmidt_orthogonalization(basis)
    v1 = basis_star[0]
    v2 = basis_star[1]

    # Define the vertices of the parallelepiped and center around origin
    v_norm = 0.5 * (v1 + v2)
    vertices = np.array([
        [0, 0] - v_norm,
        v1 - v_norm,
        (v1 + v2) - v_norm,
        v2 - v_norm
    ])

    # Plot the points of the parallelogram
    if draw_vertices:
        plt.scatter(vertices[:, 0], vertices[:, 1], color=orthogonalized_fr_color, zorder=5, s=10)

    # Draw the edges of the parallelogram
    for i in range(4):
        plt.plot([vertices[i][0], vertices[(i + 1) % 4][0]],
                 [vertices[i][1], vertices[(i + 1) % 4][1]], color=orthogonalized_fr_color)
    plt.fill(vertices[:, 0], vertices[:, 1], color=orthogonalized_fr_color, alpha=0.2)
    matrix_text = r'$B^*=\leftparen~\genfrac{{}}{{}}{{0}}{{0}}{{{}}}{{{}}}~\genfrac{{}}{{}}{{0}}{{0}}{{{}}}{{{}}}~\rightparen$' \
        .format(round(basis_star[0][0], 2), \
                round(basis_star[0][1], 2), \
                round(basis_star[1][0], 2), \
                round(basis_star[1][1], 2))
    #matrix_text = r'$\text{{Basis-Matrix $B^*$: }} \left[ \begin{{array}}{{cc}} {} & {} \\ {} & {} \end{{array}} \right]$' \
    #    .format(round(basis_star[0][0], 2), \
    #            round(basis_star[1][0], 2), \
    #            round(basis_star[0][1], 2), \
    #            round(basis_star[1][1], 2))
    plt.text(BOUND + 5, BOUND - 25, matrix_text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))


def voronoi_tiling(basis):
    """
    Must be used first, beacuse it creates new figure.
    """

    # Create array that stores the x and y coordinates
    x_vals = []
    y_vals = []
    for a in range(-BOUND, BOUND + 1):
        for b in range(-BOUND, BOUND + 1):
            point = a * basis[0] + b * basis[1]
            x_vals.append(point[0])
            y_vals.append(point[1])
    points = np.column_stack((x_vals, y_vals))

    # Now we "tile" the points to simulate infinite space:
    tile_size = 8  # Change this to increase the range of tiling (more tiles for larger areas)
    extended_points = []
    # Shift the points by multiples of the basis vectors to create tiling
    for i in range(-tile_size, tile_size + 1):
        for j in range(-tile_size, tile_size + 1):
            shift = i * basis[0] + j * basis[1]
            shifted_points = points + shift
            extended_points.append(shifted_points)
    # Flatten the list of extended points
    extended_points = np.vstack(extended_points)

    # Compute Voronoi diagram on the extended points
    vor = Voronoi(extended_points)

    # Plot the Voronoi cells (limit to the original bounds)
    voronoi_plot_2d(vor, show_vertices=False, line_colors=voronoi_tiling_color, line_width=1, line_alpha=0.6,
                    point_size=0)
    plt.xlim(-BOUND, BOUND)
    plt.ylim(-BOUND, BOUND)

    # Set equal scaling for x and y axes
    plt.gca().set_aspect('equal', adjustable='box')

    # Title and display
    plt.title("Voronoi Tiling of Lattice")


def fundamental_region_tiling(basis):

    def get_lattice_points(basis):
        x_vals = []
        y_vals = []

        # Loop over all combinations of integer coefficients for the linear combination
        for a in range(-BOUND * 10, BOUND * 10 + 1):
            for b in range(-BOUND * 10, BOUND * 10 + 1):
                point = a * basis[0] + b * basis[1]
                if abs(point[0]) <= BOUND and abs(point[1]) <= BOUND:
                    x_vals.append(point[0])
                    y_vals.append(point[1])
        return (zip(x_vals, y_vals))

    # Draw the lines parallel to fundamental region
    v1 = basis[0]
    v2 = basis[1]
    # Define the vertices of the parallelepiped and center around origin
    v_norm = 0.5 * (v1 + v2)

    # Define the vertices of the parallelepiped and center around origin
    vertices = np.array([
        [0, 0] - v_norm,
        v1 - v_norm,
        (v1 + v2) - v_norm,
        v2 - v_norm
    ])

    # Draw the edges of the parallelogram around all lattice points
    for x, y in (get_lattice_points(basis)):
        for i in range(4):
            plt.plot([vertices[i][0] + x, vertices[(i + 1) % 4][0] + x],
                     [vertices[i][1] + y, vertices[(i + 1) % 4][1] + y], color=orthogonalized_fr_color)

    plt.xlim(-BOUND, BOUND)
    plt.ylim(-BOUND, BOUND)
    # Title and text
    plt.title("Tiling of the lattice by the Fundamental Parallelepiped")


def orthogonalized_tiling(basis):
    def pi(b1, b2):
        return b2 - b1.dot(b2) / b1.dot(b1) * b1

    def gram_schmidt_orthogonalization(basis):
        b1_star = basis[0]
        b2_star = pi(basis[0], basis[1])
        return np.array([b1_star, b2_star])

    def get_lattice_points(basis):
        x_vals = []
        y_vals = []

        # Loop over all combinations of integer coefficients for the linear combination
        for a in range(-BOUND * 10, BOUND * 10 + 1):
            for b in range(-BOUND * 10, BOUND * 10 + 1):
                point = a * basis[0] + b * basis[1]
                if abs(point[0]) <= BOUND and abs(point[1]) <= BOUND:
                    x_vals.append(point[0])
                    y_vals.append(point[1])
        return (zip(x_vals, y_vals))

    basis_star = gram_schmidt_orthogonalization(basis)
    v1 = basis_star[0]
    v2 = basis_star[1]

    # Define the vertices of the parallelepiped and center around origin
    v_norm = 0.5 * (v1 + v2)
    vertices = np.array([
        [0, 0] - v_norm,
        v1 - v_norm,
        (v1 + v2) - v_norm,
        v2 - v_norm
    ])

    # Draw the edges of the parallelogram around all lattice points
    for x, y in (get_lattice_points(basis)):
        for i in range(4):
            plt.plot([vertices[i][0] + x, vertices[(i + 1) % 4][0] + x],
                     [vertices[i][1] + y, vertices[(i + 1) % 4][1] + y], color=orthogonalized_fr_color)

    plt.xlim(-BOUND, BOUND)
    plt.ylim(-BOUND, BOUND)
    # Title and text
    plt.title("Tiling of the lattice by the Fundamental Parallelepiped")


def outer_radius(basis):
    v1 = basis[0]
    v2 = basis[1]

    # Find longest vector of the parallelepiped
    vertices = np.array([
        v1,
        v1 + v2,
        v2
    ])
    distances = np.linalg.norm(vertices, axis=1)
    max_distance = np.max(distances)
    outer_radius = max_distance / 2

    # Add a circle around the origin with the radius equal to the furthest point
    circle = plt.Circle((0, 0), outer_radius, color=outer_radius_color, fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    # Wirte outer radius
    latex_text = r'$\mu(\mathcal{{P}}(B)): {}$' \
        .format(round(outer_radius, 2))
    plt.text(BOUND + 5, BOUND - 15, latex_text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))


def orthogonalized_outer_radius(basis):
    def pi(b1, b2):
        return b2 - b1.dot(b2) / b1.dot(b1) * b1

    def gram_schmidt_orthogonalization(basis):
        b1_star = basis[0]
        b2_star = pi(basis[0], basis[1])
        return np.array([b1_star, b2_star])

    basis_star = gram_schmidt_orthogonalization(basis)
    v1 = basis_star[0]
    v2 = basis_star[1]

    # Find longest vector of the parallelepiped
    vertices = np.array([
        v1,
        v1 + v2,
        v2
    ])
    distances = np.linalg.norm(vertices, axis=1)
    max_distance = np.max(distances)
    outer_radius = max_distance / 2

    # Add a circle around the origin with the radius equal to the furthest point
    circle = plt.Circle((0, 0), outer_radius, color=outer_radius_color, fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    # Wirte outer radius
    latex_text = r'$\mu(\mathcal{{P}}(B^*)): {}$' \
        .format(round(outer_radius, 2))
    plt.text(BOUND + 5, BOUND - 30, latex_text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))


def inner_radius(basis):
    def pi(b1, b2):
        return b2 - b1.dot(b2) / b1.dot(b1) * b1

    b2_star = 0.5 * pi(basis[0], basis[1])
    b1_star = 0.5 * pi(basis[1], basis[0])
    inner_radius = min(np.linalg.norm(b1_star), np.linalg.norm(b2_star))

    # Add a circle around the origin with the radius equal to the furthest point
    circle = plt.Circle((0, 0), inner_radius, color=inner_radius_color, fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    # Wirte outer radius
    latex_text = r'$\nu(\mathcal{{P}}(B)): {}$' \
        .format(round(inner_radius, 2))
    plt.text(BOUND + 5, BOUND - 20, latex_text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))


def orthogonalized_inner_radius(basis):
    def pi(b1, b2):
        return b2 - b1.dot(b2) / b1.dot(b1) * b1

    def gram_schmidt_orthogonalization(basis):
        b1_star = basis[0]
        b2_star = pi(basis[0], basis[1])
        return np.array([b1_star, b2_star])

    basis_star = gram_schmidt_orthogonalization(basis)
    v1 = basis_star[0]
    v2 = basis_star[1]
    inner_radius = 0.5 * min(np.linalg.norm(v1), np.linalg.norm(v2))

    # Add a circle around the origin with the radius equal to the furthest point
    circle = plt.Circle((0, 0), inner_radius, color=inner_radius_color, fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    # Wirte outer radius
    latex_text = r'$\nu(\mathcal{{P}}(B^*)): {}$' \
        .format(round(inner_radius, 2))
    plt.text(BOUND + 5, BOUND - 35, latex_text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))

