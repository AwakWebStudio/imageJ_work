#!/usr/bin/env python3
"""
Genera una figura científica para la micrografía GPS con un histograma semitransparente
superpuesto en la esquina inferior izquierda.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from skimage import color, exposure
from PIL import Image

IMAGE_PATH = "/workspaces/imageJ_work/GPS - .png"
DATA_PATH = "/workspaces/imageJ_work/results/gps_particles.csv"
OUTPUT_PATH = "/workspaces/imageJ_work/results/gps_scientific_figure.png"
SAMPLE_NAME = "GPS"
SCALE_LABEL = "100 µm"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Cargar imagen
img = Image.open(IMAGE_PATH).convert('RGB')
image = np.array(img)
height, width = image.shape[:2]

# Convertir a escala de grises para visualización científica
gray = color.rgb2gray(image)
gray = (gray * 255).astype(np.uint8)
# Realce moderado conservando la apariencia original
p0, p98 = np.percentile(gray, (1, 98))
gray_display = exposure.rescale_intensity(gray, in_range=(p0, p98), out_range=(0, 255)).astype(np.uint8)

# Cargar datos de tamaño de partículas
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de datos: {DATA_PATH}")

sizes = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    header = f.readline().strip().split(',')
    if 'Diametro_um' not in header and 'Diametro_µm' not in header:
        raise ValueError('El archivo de datos no contiene la columna Diametro_um o Diametro_µm')
    col_index = header.index('Diametro_um') if 'Diametro_um' in header else header.index('Diametro_µm')
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split(',')
        try:
            sizes.append(float(parts[col_index]))
        except ValueError:
            continue

sizes = np.array(sizes)
if sizes.size == 0:
    raise ValueError('No se encontraron valores de diámetro en el archivo de datos')

# Preparar histograma
bins = max(12, min(40, int(np.sqrt(len(sizes)))))
counts, bin_edges = np.histogram(sizes, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# KDE suave
kde = stats.gaussian_kde(sizes)
x_grid = np.linspace(sizes.min(), sizes.max(), 300)
pdf = kde(x_grid)
pdf_scaled = pdf * len(sizes) * (bin_edges[1] - bin_edges[0])

# Detectar barra de escala para marcarla si está presente
scale_line = None
info_fraction = 0.22
crop_y = int(height * info_fraction)
info_zone = gray_display[height - crop_y:, :]
threshold_value = np.mean(info_zone) + np.std(info_zone) * 0.5
bright_mask = info_zone > threshold_value

best_run = 0
best_row = None
best_start = None
best_end = None
max_search_rows = int(info_zone.shape[0] * 0.8)
for row in range(max_search_rows):
    run_start = None
    for col in range(info_zone.shape[1]):
        if bright_mask[row, col]:
            if run_start is None:
                run_start = col
        else:
            if run_start is not None:
                run_length = col - run_start
                if run_length > best_run:
                    best_run = run_length
                    best_row = row
                    best_start = run_start
                    best_end = col
                run_start = None
    if run_start is not None:
        run_length = info_zone.shape[1] - run_start
        if run_length > best_run:
            best_run = run_length
            best_row = row
            best_start = run_start
            best_end = info_zone.shape[1]

if best_run >= 40:
    scale_line = {
        'start': best_start,
        'end': best_end,
        'row': height - crop_y + best_row,
        'width': best_end - best_start,
    }

# Crear la figura
fig = plt.figure(figsize=(9, 9), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(gray_display, cmap='gray', vmin=0, vmax=255)
ax.axis('off')

# Agregar etiqueta de muestra
label_text = f"Sample: {SAMPLE_NAME}"
ax.text(0.02, 0.96, label_text, transform=ax.transAxes,
        color='white', fontsize=12, fontweight='bold', family='sans-serif',
        va='top', ha='left', bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=3))

# Agregar escala si se detectó
if scale_line is not None:
    y_line = scale_line['row']
    ax.plot([scale_line['start'], scale_line['end']], [y_line, y_line],
            color='yellow', linewidth=4, solid_capstyle='butt')
    ax.text(scale_line['end'] + 10, y_line - 3, SCALE_LABEL,
            color='white', fontsize=9, family='sans-serif', va='bottom', ha='left',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

# Crear histograma en la esquina inferior izquierda
hist_left = 0.06
hist_bottom = 0.06
hist_width = 0.38
hist_height = 0.28
ax_hist = fig.add_axes([hist_left, hist_bottom, hist_width, hist_height], facecolor='none')
bar_color = '#00bcd4'
ax_hist.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0],
            color=bar_color, alpha=0.45, edgecolor='white', linewidth=0.6)
ax_hist.plot(x_grid, pdf_scaled, color='white', linewidth=2)

ax_hist.set_xlabel('Size (µm)', fontsize=8, fontname='DejaVu Sans')
ax_hist.set_ylabel('Count', fontsize=8, fontname='DejaVu Sans')
ax_hist.tick_params(axis='both', which='major', labelsize=7, colors='white')
ax_hist.set_facecolor('none')
ax_hist.spines['bottom'].set_color('white')
ax_hist.spines['top'].set_color('white')
ax_hist.spines['left'].set_color('white')
ax_hist.spines['right'].set_color('white')
ax_hist.xaxis.label.set_color('white')
ax_hist.yaxis.label.set_color('white')
ax_hist.grid(axis='y', color='white', alpha=0.2, linestyle='--', linewidth=0.4)

# Ajuste de ticks y límites
ax_hist.set_xlim(sizes.min() * 0.95, sizes.max() * 1.05)
ax_hist.set_ylim(0, counts.max() * 1.3)

# Guardar figura
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

print(f"Figura científica generada: {OUTPUT_PATH}")
