#!/usr/bin/env python3
"""
Análisis de la imagen GPS - .png.
Recorta la parte inferior de datos, obtiene calibración de la barra de escala 100\u03bcm,
cuenta partículas en la región restante y genera un histograma de distribución de tamaños.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color, exposure, filters, measure, morphology
from skimage.morphology import rectangle
from PIL import Image

IMAGE_PATH = "/workspaces/imageJ_work/GPS - .png"
OUTPUT_DIR = "/workspaces/imageJ_work/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Iniciando análisis de GPS...")

# 1. Cargar imagen GPS
img_pil = Image.open(IMAGE_PATH)
image_rgb = np.array(img_pil.convert('RGB'))
height, width = image_rgb.shape[:2]
print(f"Imagen cargada: {width}x{height} píxeles")

# 2. Recortar la parte inferior con datos para calibración y evitar que se analice
crop_fraction = 0.22
crop_y = int(height * crop_fraction)
info_zone = image_rgb[height - crop_y:, :, :]
analysis_image = image_rgb[:height - crop_y, :, :]

Image.fromarray(info_zone).save(os.path.join(OUTPUT_DIR, 'gps_info_zone.png'))
print(f"   ✓ Zona inferior guardada: gps_info_zone.png")

# 3. Detectar barra de escala 100 µm en la zona de información
info_gray = color.rgb2gray(info_zone)
info_gray = (info_gray * 255).astype(np.uint8)
threshold_value = filters.threshold_otsu(info_gray)
bright_mask = info_gray > (threshold_value + 10)

best_run = None
best_row = None
best_start = None
best_end = None
max_run = 0
max_search_rows = int(info_gray.shape[0] * 0.6)
for row in range(max_search_rows):
    run_start = None
    for col in range(info_gray.shape[1]):
        if bright_mask[row, col]:
            if run_start is None:
                run_start = col
        else:
            if run_start is not None:
                run_length = col - run_start
                if run_length > max_run:
                    max_run = run_length
                    best_run = run_length
                    best_row = row
                    best_start = run_start
                    best_end = col
                run_start = None
    if run_start is not None:
        run_length = info_gray.shape[1] - run_start
        if run_length > max_run:
            max_run = run_length
            best_run = run_length
            best_row = row
            best_start = run_start
            best_end = info_gray.shape[1]

if best_run and best_run >= 60:
    bar_pixels = best_run
    bar_row = best_row
    bar_start = best_start
    bar_end = best_end
    print(f"   ✓ Barra de escala detectada en fila {bar_row}: {bar_pixels} px de largo")
else:
    bar_pixels = 106
    bar_row = None
    bar_start = None
    bar_end = None
    print("   ⚠️ No se detectó la barra de escala automáticamente. Usando ancho de 106 px por defecto.")

px_to_um = 100.0 / bar_pixels
unit = 'um'
print(f"   ✓ Calibración: 1 px = {px_to_um:.6f} {unit} (100 µm corresponde a {bar_pixels} px)")

# Guardar imagen de detección de la barra de escala
fig, ax = plt.subplots(figsize=(10, 3))
ax.imshow(info_zone)
if bar_row is not None:
    ax.plot([bar_start, bar_end], [bar_row, bar_row], color='lime', linewidth=5, solid_capstyle='round')
    ax.plot([bar_start, bar_end], [bar_row, bar_row], color='white', linewidth=2, solid_capstyle='round')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gps_scale_bar_detection.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Barra de escala seleccionada resaltada en verde: gps_scale_bar_detection.png")

# 4. Preparar la imagen para análisis
analysis_gray = color.rgb2gray(analysis_image)
analysis_gray = (analysis_gray * 255).astype(np.uint8)
analysis_enhanced = exposure.equalize_adapthist(analysis_gray, clip_limit=0.03)
analysis_enhanced = (analysis_enhanced * 255).astype(np.uint8)

# 5. Binarizar y limpiar
local_thresh = filters.threshold_local(analysis_enhanced, block_size=35, offset=12)
binary = analysis_enhanced > local_thresh
binary = morphology.opening(binary, morphology.disk(2))
binary = morphology.closing(binary, morphology.disk(2))
binary = morphology.remove_small_objects(binary, min_size=30)

# 6. Guardar binarización para revisión
bin_img = (binary * 255).astype(np.uint8)
Image.fromarray(bin_img).save(os.path.join(OUTPUT_DIR, 'gps_binary.png'))
print(f"   ✓ Binarización guardada: gps_binary.png")

# 7. Etiquetar y medir
labeled, num_features = ndimage.label(binary)
regions = measure.regionprops(labeled)

particle_sizes = []
particle_info = []
largest_particle = None
largest_area = 0
for i, region in enumerate(regions, start=1):
    if region.area <= 20:
        continue
    diameter_px = 2.0 * np.sqrt(region.area / np.pi)
    diameter_um = diameter_px * px_to_um
    particle_sizes.append(diameter_um)
    particle_info.append({
        'id': i,
        'area_px2': region.area,
        'diameter_px': diameter_px,
        'diameter_um': diameter_um,
        'eccentricity': region.eccentricity,
        'solidity': region.solidity,
        'bbox': region.bbox,
    })

    if region.area > largest_area:
        largest_area = region.area
        largest_particle = particle_info[-1]

particle_sizes = np.array(particle_sizes)
print(f"   ✓ Partículas detectadas (filtro > 20 px): {len(particle_sizes)}")

if len(particle_sizes) == 0:
    raise RuntimeError('No se detectaron partículas válidas.')

# 8. Estadísticas
stats_path = os.path.join(OUTPUT_DIR, 'gps_statistics.txt')
with open(stats_path, 'w') as f:
    f.write('ANÁLISIS DE PARTÍCULAS GPS\n')
    f.write('===========================\n')
    f.write(f'Total partículas: {len(particle_sizes)}\n')
    f.write(f'Diámetro mínimo: {particle_sizes.min():.6f} {unit}\n')
    f.write(f'Diámetro máximo: {particle_sizes.max():.6f} {unit}\n')
    f.write(f'Diámetro promedio: {particle_sizes.mean():.6f} {unit}\n')
    f.write(f'Diámetro mediana: {np.median(particle_sizes):.6f} {unit}\n')
    f.write(f'Desviación estándar: {particle_sizes.std():.6f} {unit}\n')
    f.write(f'Calibración usada: 1 px = {px_to_um:.6f} {unit}\n')
    f.write(f'Escala 100 µm = {bar_pixels} px\n')
print(f"   ✓ Estadísticas guardadas: {stats_path}")

# 9. Exportar CSV
csv_path = os.path.join(OUTPUT_DIR, 'gps_particles.csv')
with open(csv_path, 'w') as f:
    f.write('ID,Area_px2,Diametro_px,Diametro_um,Eccentricity,Solidity\n')
    for p in particle_info:
        f.write(f"{p['id']},{p['area_px2']:.2f},{p['diameter_px']:.4f},{p['diameter_um']:.6f},{p['eccentricity']:.4f},{p['solidity']:.4f}\n")
print(f"   ✓ Datos exportados: {csv_path}")

if largest_particle is not None:
    lp = largest_particle
    lp_path = os.path.join(OUTPUT_DIR, 'gps_largest_particle.png')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(analysis_image)
    minr, minc, maxr, maxc = lp['bbox']
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                         edgecolor='lime', facecolor='none', linewidth=3)
    ax.add_patch(rect)
    ax.text(minc, minr - 10, f"Mayor partícula: {lp['diameter_um']:.2f} {unit}",
            color='lime', fontsize=12, weight='bold', backgroundcolor='black', alpha=0.7)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(lp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Mayor partícula marcada: {lp_path}")
    print(f"   ✓ Tamaño mayor partícula: {lp['diameter_um']:.2f} {unit}")
else:
    print("   ⚠️ No hay partícula más grande identificada.")

# 10. Generar histograma
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(particle_sizes, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
ax.set_title('Histograma de Distribución de Partículas - GPS', fontsize=14, fontweight='bold')
ax.set_xlabel(f'Diámetro equivalente ({unit})', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.axvline(particle_sizes.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Media: {particle_sizes.mean():.3f} {unit}')
ax.axvline(np.median(particle_sizes), color='green', linestyle='--', linewidth=2,
           label=f'Mediana: {np.median(particle_sizes):.3f} {unit}')
ax.legend(fontsize=10)
plt.tight_layout()

hist_path = os.path.join(OUTPUT_DIR, 'gps_histogram.png')
plt.savefig(hist_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Histograma guardado: {hist_path}")

print('\nAnálisis completado. Revisa los archivos generados en results/.')
