#!/usr/bin/env python3
"""
Genera histograma superpuesto sobre imagen PET original - versión para artículo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, filters, io, color
from PIL import Image
import os

# Configuración
IMAGE_PATH = "/workspaces/imageJ_work/PET Canamar Ander_4.gif"
OUTPUT_DIR = "/workspaces/imageJ_work/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Calibración
px_value = 0.162109375  # 1 px = 0.162109 um
unit = "um"

print("Procesando imagen...")

# Cargar imagen
img_pil = Image.open(IMAGE_PATH)
image_rgb = np.array(img_pil.convert('RGB'))
height, width = image_rgb.shape[:2]

# Recortar imagen (remover parte inferior con info SEM)
crop_y = int(height * 0.85)
image_cropped = image_rgb[:crop_y, :]

# Convertir a escala de grises
from skimage.color import rgb2gray
gray = rgb2gray(image_cropped)
gray = (gray * 255).astype(np.uint8)

# Mejora de contraste
from skimage.exposure import equalize_adapthist
gray_enhanced = equalize_adapthist(gray, clip_limit=0.02)
gray = (gray_enhanced * 255).astype(np.uint8)

# Binarización
from skimage.filters import threshold_local
local_thresh = threshold_local(gray, block_size=11, offset=2)
binary = (gray > local_thresh).astype(np.uint8) * 255

# Limpieza morfológica
kernel = morphology.disk(2)
binary = morphology.opening(binary > 0, kernel)
binary = morphology.closing(binary, kernel)

# Detectar partículas
labeled_array, num_features = ndimage.label(binary)
regions = measure.regionprops(labeled_array)

# Calcular tamaños
particle_sizes = []
for region in regions:
    area = region.area
    if area > 10:  # Filtrar ruido
        diameter_px = 2 * np.sqrt(area / np.pi)
        diameter_real = diameter_px * px_value
        particle_sizes.append(diameter_real)

particle_sizes_array = np.array(particle_sizes)

print(f"Partículas detectadas: {len(particle_sizes)}")
print(f"Promedio: {particle_sizes_array.mean():.4f} {unit}")

# Crear figura con imagen y histograma superpuesto
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(1, 1)
ax_main = fig.add_subplot(gs[0])

# Mostrar imagen
ax_main.imshow(image_cropped)
ax_main.axis('off')

# Crear histograma como inset (incrustado)
# Posición: esquina inferior derecha (left, bottom, width, height) en coordenadas de la figura
ax_inset = fig.add_axes([0.55, 0.15, 0.4, 0.3])

# Fondo semi-transparente blanco
rect = plt.Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes, 
                      facecolor='white', alpha=0.92, zorder=-1)
ax_inset.add_patch(rect)

# Plotear histograma
ax_inset.hist(particle_sizes, bins=40, color='steelblue', edgecolor='black', 
              alpha=0.8, linewidth=0.6)
ax_inset.set_xlabel(f'Diameter ({unit})', fontsize=11, fontweight='bold')
ax_inset.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax_inset.set_xlim(0, 4)
ax_inset.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Líneas de referencia
ax_inset.axvline(particle_sizes_array.mean(), color='red', linestyle='--', 
                 linewidth=2.2, label=f'Mean: {particle_sizes_array.mean():.3f} {unit}', 
                 alpha=0.9, zorder=5)
ax_inset.axvline(np.median(particle_sizes_array), color='green', linestyle='--', 
                 linewidth=2.2, label=f'Median: {np.median(particle_sizes_array):.3f} {unit}', 
                 alpha=0.9, zorder=5)

ax_inset.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor='black', fancybox=True)
ax_inset.set_axisbelow(True)
ax_inset.spines['top'].set_visible(True)
ax_inset.spines['right'].set_visible(True)
ax_inset.spines['left'].set_visible(True)
ax_inset.spines['bottom'].set_visible(True)
ax_inset.spines['top'].set_linewidth(1.5)
ax_inset.spines['right'].set_linewidth(1.5)
ax_inset.spines['left'].set_linewidth(1.5)
ax_inset.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'histograma_overlay.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Histograma overlay guardado: {output_path}")
plt.close()

# Crear versión alternativa: histograma arriba a la izquierda
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(1, 1)
ax_main = fig.add_subplot(gs[0])

# Mostrar imagen
ax_main.imshow(image_cropped)
ax_main.axis('off')

# Inset en esquina superior izquierda
ax_inset2 = fig.add_axes([0.08, 0.62, 0.35, 0.32])

# Fondo semi-transparente
rect2 = plt.Rectangle((0, 0), 1, 1, transform=ax_inset2.transAxes, 
                       facecolor='white', alpha=0.93, zorder=-1)
ax_inset2.add_patch(rect2)

# Plotear histograma
ax_inset2.hist(particle_sizes, bins=40, color='steelblue', edgecolor='black', 
               alpha=0.8, linewidth=0.6)
ax_inset2.set_xlabel(f'Diameter ({unit})', fontsize=10, fontweight='bold')
ax_inset2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax_inset2.set_xlim(0, 4)
ax_inset2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Líneas de referencia
ax_inset2.axvline(particle_sizes_array.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {particle_sizes_array.mean():.3f} {unit}', 
                  alpha=0.9, zorder=5)
ax_inset2.axvline(np.median(particle_sizes_array), color='green', linestyle='--', 
                  linewidth=2, label=f'Median: {np.median(particle_sizes_array):.3f} {unit}', 
                  alpha=0.9, zorder=5)

ax_inset2.legend(fontsize=9, loc='upper right', framealpha=0.95, edgecolor='black', fancybox=True)
ax_inset2.set_axisbelow(True)
ax_inset2.spines['top'].set_linewidth(1.5)
ax_inset2.spines['right'].set_linewidth(1.5)
ax_inset2.spines['left'].set_linewidth(1.5)
ax_inset2.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
output_path2 = os.path.join(OUTPUT_DIR, 'histograma_overlay_topleft.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Histograma overlay (top-left) guardado: {output_path2}")
plt.close()

print("\n✓ Completado - Dos versiones generadas para elegir")
