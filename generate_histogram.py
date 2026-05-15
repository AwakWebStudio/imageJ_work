#!/usr/bin/env python3
"""
Genera solo el histograma de distribución de tamaño de partículas - versión compacta
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
print(f"Mediana: {np.median(particle_sizes_array):.4f} {unit}")

# Crear histograma compacto
fig, ax = plt.subplots(figsize=(8, 2))

ax.hist(particle_sizes, bins=40, color='steelblue', edgecolor='black', alpha=0.75, linewidth=0.8)
ax.set_xlabel(f'Diameter ({unit})', fontsize=10, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax.set_xlim(0, 4)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Líneas de referencia
ax.axvline(particle_sizes_array.mean(), color='red', linestyle='--', 
          linewidth=2, label=f'Mean: {particle_sizes_array.mean():.3f} {unit}', alpha=0.8)
ax.axvline(np.median(particle_sizes_array), color='green', linestyle='--', 
          linewidth=2, label=f'Median: {np.median(particle_sizes_array):.3f} {unit}', alpha=0.8)

ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
ax.set_axisbelow(True)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'histograma_simple.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Histograma guardado: {output_path}")
plt.close()

# Guardar estadísticas en archivo txt
stats_path = os.path.join(OUTPUT_DIR, 'estadisticas.txt')
with open(stats_path, 'w') as f:
    f.write(f"ESTADÍSTICAS DE PARTÍCULAS PET\n")
    f.write(f"{'='*50}\n\n")
    f.write(f"Total de partículas: {len(particle_sizes)}\n")
    f.write(f"Diámetro mínimo: {particle_sizes_array.min():.6f} {unit}\n")
    f.write(f"Diámetro máximo: {particle_sizes_array.max():.6f} {unit}\n")
    f.write(f"Diámetro promedio: {particle_sizes_array.mean():.6f} {unit}\n")
    f.write(f"Diámetro mediana: {np.median(particle_sizes_array):.6f} {unit}\n")
    f.write(f"Desviación estándar: {particle_sizes_array.std():.6f} {unit}\n")

print(f"✓ Estadísticas guardadas: {stats_path}\n")
