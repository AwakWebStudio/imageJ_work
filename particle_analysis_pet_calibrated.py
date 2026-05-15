#!/usr/bin/env python3
"""
Análisis de partículas PET con calibración SEM
Extrae la métrica de escala del SEM y convierte píxeles a unidades reales
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, filters, io, color
from PIL import Image
from pathlib import Path
import os

# Configuración
IMAGE_PATH = "/workspaces/imageJ_work/PET Canamar Ander_4.gif"
OUTPUT_DIR = "/workspaces/imageJ_work/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ANÁLISIS DE PARTÍCULAS PET CON CALIBRACIÓN SEM")
print("=" * 70)

# 1. Cargar imagen
print("\n1. Cargando imagen GIF...")
try:
    img_pil = Image.open(IMAGE_PATH)
    image_rgb = np.array(img_pil.convert('RGB'))
    print(f"   ✓ Imagen cargada: {img_pil.size[0]}x{img_pil.size[1]} píxeles")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

height, width = image_rgb.shape[:2]

# Visualizar imagen para identificar la barra de escala
print("\n2. Analizando estructura de la imagen...")
print(f"   Dimensiones totales: {width}x{height} píxeles")

# Crear visualización con zonas de interés
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Imagen completa
axes[0].imshow(image_rgb)
axes[0].set_title('Imagen completa con SEM info', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Píxeles')
axes[0].set_ylabel('Píxeles')
axes[0].grid(alpha=0.3)

# Zoom en la parte inferior donde está la métrica
crop_height = int(height * 0.15)  # 15% inferior
image_info_zone = image_rgb[-crop_height:, :]
axes[1].imshow(image_info_zone)
axes[1].set_title(f'Zona de información SEM (últimos {crop_height}px)', 
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Píxeles')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analisis_zona_sem.png'), dpi=150, bbox_inches='tight')
print("   ✓ Análisis de zonas guardado: analisis_zona_sem.png")
plt.close()

# 3. Proporcionar información para calibración manual
print("\n3. INFORMACIÓN PARA CALIBRACIÓN:")
print(f"   - Altura total de imagen: {height} píxeles")
print(f"   - Ancho total de imagen: {width} píxeles")
print(f"   - Zona SEM inferior aproximada: últimos {crop_height} píxeles")
print("\n   Por favor, inspecciona 'analisis_zona_sem.png' para:")
print("   - Identificar la barra de escala (scale bar)")
print("   - Leer su medida (ej: 10 μm, 100 nm, etc.)")
print("   - Contar píxeles de la barra de escala")

# Preguntar por la calibración
print("\n4. CALIBRACIÓN MANUAL REQUERIDA:")
print("   Ejemplos de entrada:")
print("   - 50 px = 1 um")
print("   - 100 px = 10 nm")

try:
    calibration_input = input("\n   Ingresa calibración (ej: '50 px = 1 um'): ").strip()
    
    # Parsear calibración
    parts = calibration_input.split('=')
    if len(parts) != 2:
        print("Error: Formato inválido. Usando 1px = 1px por defecto")
        px_value = 1.0
        unit = "px"
    else:
        # Extraer píxeles
        px_part = parts[0].strip().split()
        px_value_in_image = float(px_part[0])
        
        # Extraer unidad real
        unit_part = parts[1].strip().split()
        unit_value = float(unit_part[0])
        unit = unit_part[1] if len(unit_part) > 1 else "um"
        
        # Calcular factor de conversión
        px_value = unit_value / px_value_in_image
        print(f"   ✓ Calibración aceptada: 1 px = {px_value:.6f} {unit}")
        
except ValueError:
    print("Error: No se pudo parsear la calibración. Usando 1px = 1px")
    px_value = 1.0
    unit = "px"

# Guardar calibración
with open(os.path.join(OUTPUT_DIR, 'calibration.txt'), 'w') as f:
    f.write(f"Pixel conversion: 1 px = {px_value} {unit}\n")
    f.write(f"Image dimensions: {width}x{height}\n")

# 5. Recortar imagen (remover la parte inferior con info SEM)
print(f"\n5. Recortando imagen...")
# Remover aproximadamente 15% de la parte inferior
crop_y = int(height * 0.85)
image_cropped = image_rgb[:crop_y, :]
print(f"   ✓ Imagen recortada: {image_cropped.shape[1]}x{image_cropped.shape[0]} píxeles")

# 6. Procesamiento de imagen
print(f"\n6. Procesando imagen para detectar partículas...")

# Convertir a escala de grises
from skimage.color import rgb2gray
gray = rgb2gray(image_cropped)
gray = (gray * 255).astype(np.uint8)

# Mejora de contraste
from skimage.exposure import equalize_adapthist
gray_enhanced = equalize_adapthist(gray, clip_limit=0.02)
gray = (gray_enhanced * 255).astype(np.uint8)

# Binarización adaptativa
from skimage.filters import threshold_local
local_thresh = threshold_local(gray, block_size=11, offset=2)
binary = (gray > local_thresh).astype(np.uint8) * 255

# Limpieza morfológica
kernel = morphology.disk(2)
binary = morphology.opening(binary > 0, kernel)
binary = morphology.closing(binary, kernel)
binary = (binary * 255).astype(np.uint8)

# 7. Detección de partículas
print(f"\n7. Detectando partículas...")
labeled_array, num_features = ndimage.label(binary)
print(f"   Partículas detectadas: {num_features}")

# 8. Calcular propiedades
print(f"\n8. Calculando propiedades...")
regions = measure.regionprops(labeled_array)

particle_sizes = []
particle_info = []

for i, region in enumerate(regions):
    area = region.area
    perimeter = region.perimeter
    # Diámetro equivalente en píxeles
    diameter_px = 2 * np.sqrt(area / np.pi)
    
    # Convertir a unidades reales
    area_real = area * (px_value ** 2)
    diameter_real = diameter_px * px_value
    
    # Filtrar ruido (< 10 píxeles)
    if area > 10:
        particle_sizes.append(diameter_real)
        particle_info.append({
            'id': i,
            'area_px': area,
            'area_real': area_real,
            'perimeter_px': perimeter,
            'diameter_px': diameter_px,
            'diameter_real': diameter_real,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity
        })

print(f"   Partículas válidas (filtradas): {len(particle_sizes)}")

if len(particle_sizes) == 0:
    print("Error: No se detectaron suficientes partículas")
    exit(1)

particle_sizes_array = np.array(particle_sizes)

# 9. Estadísticas
print(f"\n9. ESTADÍSTICAS DE TAMAÑO:")
print(f"   {'─' * 60}")
print(f"   Diámetro (en {unit}):")
print(f"   ├─ Mínimo:        {particle_sizes_array.min():.4f} {unit}")
print(f"   ├─ Máximo:        {particle_sizes_array.max():.4f} {unit}")
print(f"   ├─ Promedio:      {particle_sizes_array.mean():.4f} {unit}")
print(f"   ├─ Mediana:       {np.median(particle_sizes_array):.4f} {unit}")
print(f"   └─ Desv. Est.:    {particle_sizes_array.std():.4f} {unit}")
print(f"   {'─' * 60}")
print(f"   Total de partículas: {len(particle_sizes)}")

# 10. Crear histogramas
print(f"\n10. Generando histogramas...")
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Histograma de diámetros (unidades reales)
ax1 = axes[0, 0]
counts, bins, patches = ax1.hist(particle_sizes, bins=40, color='steelblue',
                                  edgecolor='black', alpha=0.7)
ax1.set_xlabel(f'Diámetro Equivalente ({unit})', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax1.set_title('Distribución de Tamaño de Partículas PET', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axvline(particle_sizes_array.mean(), color='red', linestyle='--',
            linewidth=2.5, label=f'Media: {particle_sizes_array.mean():.4f} {unit}')
ax1.axvline(np.median(particle_sizes_array), color='green', linestyle='--',
            linewidth=2.5, label=f'Mediana: {np.median(particle_sizes_array):.4f} {unit}')
ax1.legend(fontsize=10)

# Histograma de áreas (unidades reales)
ax2 = axes[0, 1]
areas_real = [p['area_real'] for p in particle_info]
ax2.hist(areas_real, bins=40, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel(f'Área ({unit}²)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax2.set_title('Distribución de Área de Partículas', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Scatter: Área vs Diámetro
ax3 = axes[1, 0]
diameters = [p['diameter_real'] for p in particle_info]
ax3.scatter(diameters, areas_real, alpha=0.5, s=40, color='purple', edgecolors='black', linewidth=0.5)
ax3.set_xlabel(f'Diámetro ({unit})', fontsize=11, fontweight='bold')
ax3.set_ylabel(f'Área ({unit}²)', fontsize=11, fontweight='bold')
ax3.set_title('Relación Área vs Diámetro', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Imagen con partículas marcadas
ax4 = axes[1, 1]
from matplotlib.patches import Rectangle
for region in regions:
    if region.area > 10:
        minr, minc, maxr, maxc = region.bbox
        rect = Rectangle((minc, minr), maxc-minc, maxr-minr,
                        linewidth=1, edgecolor='lime', facecolor='none', alpha=0.7)
        ax4.add_patch(rect)

ax4.imshow(image_cropped)
ax4.set_title(f'Partículas Detectadas ({len(particle_sizes)} en total)', 
              fontsize=12, fontweight='bold')
ax4.axis('off')

plt.tight_layout()
histograma_path = os.path.join(OUTPUT_DIR, 'histograma_pet_calibrado.png')
plt.savefig(histograma_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Histograma calibrado guardado: histograma_pet_calibrado.png")
plt.close()

# 11. Tabla resumen
print(f"\n11. Generando tabla resumen...")
fig, ax = plt.subplots(figsize=(11, 7))
ax.axis('tight')
ax.axis('off')

# Datos para la tabla
table_data = [
    ['Parámetro', f'Valor ({unit})'],
    ['─' * 30, '─' * 20],
    ['Total de partículas', f'{len(particle_sizes)}'],
    ['Diámetro mínimo', f'{particle_sizes_array.min():.6f}'],
    ['Diámetro máximo', f'{particle_sizes_array.max():.6f}'],
    ['Diámetro promedio', f'{particle_sizes_array.mean():.6f}'],
    ['Diámetro mediana', f'{np.median(particle_sizes_array):.6f}'],
    ['Desv. Est. diámetro', f'{particle_sizes_array.std():.6f}'],
    ['─' * 30, '─' * 20],
    ['Área promedio', f'{np.mean(areas_real):.6f} {unit}²'],
    ['Área mínima', f'{np.min(areas_real):.6f} {unit}²'],
    ['Área máxima', f'{np.max(areas_real):.6f} {unit}²'],
]

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.55, 0.45])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Formatear
for i in range(2):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(1, i)].set_facecolor('#34495e')

for i in range(len(table_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')

plt.title(f'Resumen - Análisis PET Calibrado (Escala: 1 px = {px_value} {unit})',
          fontsize=13, fontweight='bold', pad=20)

tabla_path = os.path.join(OUTPUT_DIR, 'resumen_pet_calibrado.png')
plt.savefig(tabla_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Tabla resumen guardada: resumen_pet_calibrado.png")
plt.close()

# 12. Exportar datos a CSV
print(f"\n12. Exportando datos a CSV...")
csv_path = os.path.join(OUTPUT_DIR, 'datos_particulas_pet.csv')
with open(csv_path, 'w') as f:
    f.write(f'ID,Area_px2,Area_{unit}2,Perimetro_px,Diametro_px,Diametro_{unit},Eccentricity,Solidity\n')
    for p in particle_info:
        f.write(f"{p['id']},{p['area_px']:.2f},{p['area_real']:.6f},"
                f"{p['perimeter_px']:.2f},{p['diameter_px']:.4f},"
                f"{p['diameter_real']:.6f},{p['eccentricity']:.4f},{p['solidity']:.4f}\n")
print(f"   ✓ Datos exportados: datos_particulas_pet.csv")

print("\n" + "=" * 70)
print("✓ ANÁLISIS COMPLETADO")
print("=" * 70)
print(f"\nArchivos generados en {OUTPUT_DIR}/:")
print(f"  📊 histograma_pet_calibrado.png")
print(f"  📋 resumen_pet_calibrado.png")
print(f"  📄 datos_particulas_pet.csv")
print(f"  🔍 analisis_zona_sem.png")
print(f"  ⚙️  calibration.txt")
print(f"\nUnidad de medida utilizada: {unit}")
print(f"Factor de conversión: 1 px = {px_value} {unit}\n")
