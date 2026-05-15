#!/usr/bin/env python3
"""
Superpone histograma sin fondo sobre imagen PET
Barras en blanco/gris transparente integradas en la imagen
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, filters, io, color
from PIL import Image, ImageDraw
import os

# Configuración
IMAGE_PATH = "/workspaces/imageJ_work/PET.gif"
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
crop_height, crop_width = image_cropped.shape[:2]

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
    if area > 10:
        diameter_px = 2 * np.sqrt(area / np.pi)
        diameter_real = diameter_px * px_value
        particle_sizes.append(diameter_real)

particle_sizes_array = np.array(particle_sizes)

print(f"Partículas detectadas: {len(particle_sizes)}")
print(f"Promedio: {particle_sizes_array.mean():.4f} {unit}")

# Crear histograma con fondo transparente
fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# Histograma con barras grises/blancas
counts, bins, patches = ax.hist(particle_sizes, bins=40, color='#CCCCCC', 
                                 edgecolor='#666666', alpha=0.85, linewidth=0.4)

# Estilo limpio sin bordes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#666666')
ax.spines['bottom'].set_color('#666666')
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

ax.set_xlabel(f'Diameter ({unit})', fontsize=10, fontweight='bold', color='#333333')
ax.set_ylabel('Frequency', fontsize=10, fontweight='bold', color='#333333')
ax.set_xlim(0, 4)
ax.tick_params(colors='#333333', labelsize=9)
ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.4, color='#666666')

# Líneas de referencia
ax.axvline(particle_sizes_array.mean(), color='#00AA00', linestyle='--', 
          linewidth=2, label=f'Mean: {particle_sizes_array.mean():.3f}', alpha=0.9)
ax.axvline(np.median(particle_sizes_array), color='#FF6600', linestyle='--', 
          linewidth=2, label=f'Median: {np.median(particle_sizes_array):.3f}', alpha=0.9)

ax.legend(fontsize=8, loc='upper right', framealpha=0, edgecolor='none', 
         labelcolor='#333333', fancybox=False)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.5)

# Guardar figura como PNG con fondo transparente
hist_temp_path = os.path.join(OUTPUT_DIR, 'histogram_transparent.png')
plt.savefig(hist_temp_path, dpi=100, bbox_inches='tight', transparent=True)
plt.close()

print("✓ Histograma transparente generado")

# Cargar histograma
hist_img = Image.open(hist_temp_path).convert('RGBA')

# Redimensionar histograma a proporciones apropiadas
hist_width = int(crop_width * 0.4)
hist_height = int(crop_height * 0.25)
hist_img_resized = hist_img.resize((hist_width, hist_height), Image.Resampling.LANCZOS)

# Convertir imagen cropped a RGBA para composición
image_pil = Image.fromarray(image_cropped).convert('RGBA')

# Calcular posición: esquina inferior derecha con margen
x_pos = crop_width - hist_width - 20
y_pos = crop_height - hist_height - 20

# Crear capa con el histograma
histogram_layer = Image.new('RGBA', (crop_width, crop_height), (0, 0, 0, 0))
histogram_layer.paste(hist_img_resized, (x_pos, y_pos), hist_img_resized)

# Composición: imagen base + histograma transparente
result = Image.alpha_composite(image_pil, histogram_layer)

# Convertir de vuelta a RGB
result_rgb = result.convert('RGB')

# Guardar resultado
output_path = os.path.join(OUTPUT_DIR, 'PET_with_histogram.png')
result_rgb.save(output_path, dpi=(300, 300), quality=95)
print(f"✓ Imagen con histograma superpuesto guardada: PET_with_histogram.png")
print(f"  Dimensiones: {crop_width}x{crop_height}")

# Limpiar archivo temporal
os.remove(hist_temp_path)

print("\n✓ Completado")
