#!/usr/bin/env python3
"""
Genera un histograma en PNG usando solo los datos de tamaño de partículas.
El gráfico tendrá el mismo alto que la imagen GPS y la mitad de su ancho,
con ejes cambiados y estilo en escala de grises.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image

IMAGE_PATH = "/workspaces/imageJ_work/GPS - .png"
DATA_PATH = "/workspaces/imageJ_work/results/gps_particles.csv"
OUTPUT_PATH = "/workspaces/imageJ_work/results/gps_histogram_data_only.png"
DPI = 300

# Leer tamaño de la imagen para mantener la altura y ancho proporcional
img = Image.open(IMAGE_PATH)
width, height = img.size
hist_width = int(height / 3)
# Queremos un panel vertical estrecho con relación 3:1 altura:ancho
hist_height = height

# Leer diámetros de los datos
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

# Histograma girado para panel lateral estrecho
bins = max(12, min(30, int(np.sqrt(len(sizes)))))
counts, bin_edges = np.histogram(sizes, bins=bins)
y_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_height = bin_edges[1] - bin_edges[0]

fig = plt.figure(figsize=(hist_width / DPI, hist_height / DPI), dpi=DPI)
fig.patch.set_alpha(0.0)
ax = fig.add_axes([0, 0, 1, 1], facecolor='none')

# Barras horizontales apiladas en la altura del panel
bar_color = '#7f7f7f'
ax.barh(y_centers, counts, height=bin_height * 0.92, color=bar_color, alpha=0.85, edgecolor='none')

# Usar todo el espacio para los bins y ocultar ejes
ax.set_xlim(0, counts.max() * 1.05)
ax.set_ylim(y_centers.min() - bin_height * 0.5, y_centers.max() + bin_height * 0.5)
ax.axis('off')

# Guardar con tamaño fijo sin márgenes adicionales
fig.savefig(OUTPUT_PATH, dpi=DPI, transparent=True)
plt.close(fig)

# Ajustar el resultado final para que tenga exactamente la altura de la imagen original
from PIL import Image as PILImage
result_img = PILImage.open(OUTPUT_PATH)
# Recortar márgenes transparentes
bbox = result_img.getbbox()
if bbox is not None:
    result_img = result_img.crop(bbox)
# Ajustar a tamaño vertical exacto
result_img = result_img.resize((hist_width, hist_height), PILImage.LANCZOS)
result_img.save(OUTPUT_PATH, format='PNG')

print(f"Histograma generado: {OUTPUT_PATH}")

