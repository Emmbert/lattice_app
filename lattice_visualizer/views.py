import numpy as np
import matplotlib

from .lattice_figures import plot_lattice, plot_fundamental_region, plot_fundamental_region_tiling, plot_voronoi_tiling, \
    plot_diff_fundamental_region_voronoi, plot_outer_radius, plot_inner_radius, plot_orth_region, plot_orth_tiling, \
    plot_orth_vs_fundamental_region, plot_diff_orth_fundamental_voronoi_tiling

matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from .forms import BasisInputForm


def generate_figures(matrix1, matrix2):

    functions = [plot_lattice, plot_fundamental_region,
                 plot_fundamental_region_tiling,
                 plot_inner_radius,
                 plot_outer_radius,
                 plot_voronoi_tiling,
                 plot_diff_fundamental_region_voronoi,
                 plot_orth_region,
                 plot_orth_tiling,
                 plot_orth_vs_fundamental_region,
                 plot_diff_orth_fundamental_voronoi_tiling]

    results = []

    for f in functions:
        description, plot1 = f(matrix1)
        _, plot2 = f(matrix2) if matrix2 is not None else None

        figures = {
            'plot1': plot1,
            'plot2': plot2,
            'description': description
        }
        results.append(figures)

    return results

def matrix_input(request):
    if request.method == 'POST':
        form = BasisInputForm(request.POST)
        if form.is_valid():
            # Get the first matrix from form
            matrix1 = np.array([
                [form.cleaned_data['matrix1_00'], form.cleaned_data['matrix1_01']],
                [form.cleaned_data['matrix1_10'], form.cleaned_data['matrix1_11']]
            ])

            # Get the second matrix from form (optional)
            matrix2 = np.array([
                [form.cleaned_data['matrix2_00'], form.cleaned_data['matrix2_01']],
                [form.cleaned_data['matrix2_10'], form.cleaned_data['matrix2_11']]
            ]) if form.cleaned_data['matrix2_00'] is not None else None

            # Generate plots
            figures = generate_figures(matrix1, matrix2)

            return render(request, 'lattice_visualizer/basis_input.html', {'form': form, 'figures': figures})
    else:
        form = BasisInputForm()

    return render(request, 'lattice_visualizer/basis_input.html', {'form': form})
