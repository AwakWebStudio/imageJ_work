#!/usr/bin/env python3
"""
Análisis de partículas de Glitter usando ImageJ/scikit-image
Detecta partículas y genera un histograma de tamaños
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, filters, io
from PIL import Image
from pathlib import Path
import os

# Configuración
IMAGE_PATH = "/workspaces/imageJ_work/Ander NanoPhys Day Vol.4 - Template (1).png"
OUTPUT_DIR = "/workspaces/imageJ_work/results"

# Crear directorio de resultados
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("ANÁLISIS DE PARTÍCULAS DE GLITTER")
print("=" * 60)

# 1. Cargar imagen
print("\n1. Cargando imagen...")
try:
    image_rgb = io.imread(IMAGE_PATH)
except Exception as e:
    print(f"Error: No se pudo cargar la imagen en {IMAGE_PATH}")
    print(f"Detalle: {e}")
    exit(1)

height, width = image_rgb.shape[:2]
print(f"   Dimensiones: {width}x{height} píxeles")

# 2. Convertir a escala de grises
print("\n2. Convirtiendo a escala de grises...")
from skimage.color import rgb2gray
gray = rgb2gray(image_rgb)
gray = (gray * 255).astype(np.uint8)

# 3. Aplicar filtro para mejorar contraste
print("\n3. Mejorando contraste...")
from skimage.exposure import equalize_adapthist
gray_enhanced = equalize_adapthist(gray, clip_limit=0.02)
gray = (gray_enhanced * 255).astype(np.uint8)

# 4. Binarización adaptativa
print("\n4. Binarizando imagen (detectando partículas)...")
from skimage.filters import threshold_local
local_thresh = threshold_local(gray, block_size=11, offset=2)
binary = (gray > local_thresh).astype(np.uint8) * 255

# 5. Limpieza morfológica
print("\n5. Limpiando ruido...")
kernel = morphology.disk(2)
binary = morphology.opening(binary > 0, kernel)
binary = morphology.closing(binary, kernel)
binary = (binary * 255).astype(np.uint8)

# 6. Detectar partículas usando etiquetado conectado
print("\n6. Detectando partículas...")
labeled_array, num_features = ndimage.label(binary)
print(f"   Partículas detectadas: {num_features}")

# 7. Calcular propiedades de cada partícula
print("\n7. Calculando propiedades de partículas...")
regions = measure.regionprops(labeled_array)

# Extraer tamaños (área en píxeles)
particle_sizes = []
particle_info = []

for i, region in enumerate(regions):
    area = region.area
    perimeter = region.perimeter
    # Calcular diámetro equivalente (asumiendo forma aproximadamente circular)
    diameter = 2 * np.sqrt(area / np.pi)
    
    # Filtrar partículas muy pequeñas (ruido)
    if area > 10:  # Mínimo 10 píxeles
        particle_sizes.append(diameter)
        particle_info.append({
            'id': i,
            'area': area,
            'perimeter': perimeter,
            'diameter': diameter,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity
        })

print(f"   Partículas válidas (filtradas): {len(particle_sizes)}")

if len(particle_sizes) == 0:
    print("Error: No se detectaron partículas suficientes")
    exit(1)

# 8. Estadísticas
particle_sizes_array = np.array(particle_sizes)
print(f"\n8. ESTADÍSTICAS DE TAMAÑO (Diámetro equivalente en píxeles):")
print(f"   Mínimo: {particle_sizes_array.min():.2f} px")
print(f"   Máximo: {particle_sizes_array.max():.2f} px")
print(f"   Promedio: {particle_sizes_array.mean():.2f} px")
print(f"   Mediana: {np.median(particle_sizes_array):.2f} px")
print(f"   Desv. Est.: {particle_sizes_array.std():.2f} px")

# 9. Crear histograma
print("\n9. Generando histograma...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograma de diámetros
ax1 = axes[0, 0]
counts, bins, patches = ax1.hist(particle_sizes, bins=30, color='steelblue', 
                                  edgecolor='black', alpha=0.7)
ax1.set_xlabel('Diámetro Equivalente (píxeles)', fontsize=11)
ax1.set_ylabel('Frecuencia', fontsize=11)
ax1.set_title('Histograma de Tamaño de Partículas', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axvline(particle_sizes_array.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Promedio: {particle_sizes_array.mean():.2f} px')
ax1.axvline(np.median(particle_sizes_array), color='green', linestyle='--', 
            linewidth=2, label=f'Mediana: {np.median(particle_sizes_array):.2f} px')
ax1.legend()

# Histograma de áreas
ax2 = axes[0, 1]
areas = [p['area'] for p in particle_info]
ax2.hist(areas, bins=30, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Área (píxeles²)', fontsize=11)
ax2.set_ylabel('Frecuencia', fontsize=11)
ax2.set_title('Histograma de Área de Partículas', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Scatter: Área vs Diámetro
ax3 = axes[1, 0]
ax3.scatter(particle_sizes, areas, alpha=0.6, s=50, color='purple')
ax3.set_xlabel('Diámetro Equivalente (píxeles)', fontsize=11)
ax3.set_ylabel('Área (píxeles²)', fontsize=11)
ax3.set_title('Relación Área vs Diámetro', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Imagen con partículas marcadas
ax4 = axes[1, 1]
# Crear una máscara con las partículas marcadas
image_marked = image_rgb.copy()
for region in regions:
    if region.area > 10:
        minr, minc, maxr, maxc = region.bbox
        # Dibujar rectángulo usando matplotlib
        from matplotlib.patches import Rectangle
        rect = Rectangle((minc, minr), maxc-minc, maxr-minr, 
                        linewidth=1, edgecolor='lime', facecolor='none')
        ax4.add_patch(rect)

ax4.imshow(image_rgb)
ax4.set_title('Partículas Detectadas (rectángulos verdes)', fontsize=12, fontweight='bold')
ax4.axis('off')

plt.tight_layout()
histograma_path = os.path.join(OUTPUT_DIR, 'histograma_particulas.png')
plt.savefig(histograma_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Histograma guardado: {histograma_path}")
plt.close()

# 10. Crear tabla resumen
print("\n10. Generando tabla resumen...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Datos para la tabla
table_data = [
    ['Parámetro', 'Valor'],
    ['Total de partículas', f'{len(particle_sizes)}'],
    ['Diámetro mínimo', f'{particle_sizes_array.min():.2f} px'],
    ['Diámetro máximo', f'{particle_sizes_array.max():.2f} px'],
    ['Diámetro promedio', f'{particle_sizes_array.mean():.2f} px'],
    ['Diámetro mediana', f'{np.median(particle_sizes_array):.2f} px'],
    ['Desviación estándar', f'{particle_sizes_array.std():.2f} px'],
    ['Área promedio', f'{np.mean(areas):.2f} px²'],
    ['Dimensiones imagen', f'{width}x{height} px'],
]

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Formatear encabezado
for i in range(2):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternar colores de filas
for i in range(1, len(table_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Resumen de Análisis de Partículas', fontsize=14, fontweight='bold', pad=20)
tabla_path = os.path.join(OUTPUT_DIR, 'resumen_estadistico.png')
plt.savefig(tabla_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Tabla resumen guardada: {tabla_path}")
plt.close()

# 11. Exportar datos a CSV
print("\n11. Exportando datos a CSV...")
csv_path = os.path.join(OUTPUT_DIR, 'datos_particulas.csv')
with open(csv_path, 'w') as f:
    f.write('ID,Area(px2),Perimetro(px),DiametroEquivalente(px),Eccentricidad,Solidity\n')
    for p in particle_info:
        f.write(f"{p['id']},{p['area']:.2f},{p['perimeter']:.2f},"
                f"{p['diameter']:.2f},{p['eccentricity']:.4f},{p['solidity']:.4f}\n")
print(f"   ✓ Datos exportados a: {csv_path}")

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO")
print("=" * 60)
print(f"\nArchivos generados en: {OUTPUT_DIR}/")
print("  - histograma_particulas.png")
print("  - resumen_estadistico.png")
print("  - datos_particulas.csv")
print("\n✓ Todos los resultados están listos para consultar.\n")
