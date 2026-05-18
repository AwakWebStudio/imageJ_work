#!/usr/bin/env python3
"""
Genera figuras científicas para GPS-Glitter y GPS-PET.
Cada figura usa la micrografía en escala de grises como fondo y un histograma
semipermeable superpuesto en la esquina inferior izquierda.

El histograma se construye únicamente a partir de los datos de tamaño de partículas
contenido en los archivos CSV disponibles.
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from skimage import color, exposure
from PIL import Image

BASE_DIR = Path('/workspaces/imageJ_work')
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    {
        'name': 'GPS-Glitter',
        'image_path': BASE_DIR / 'GPS -  Glitter.png',
        'data_path': RESULTS_DIR / 'gps_particles.csv',
        'output_prefix': RESULTS_DIR / 'gps_glitter_scientific_figure',
        'hist_color': '#00bcd4',
    },
    {
        'name': 'GPS-PET',
        'image_path': BASE_DIR / 'GPS -  PET.png',
        'data_path': RESULTS_DIR / 'datos_particulas_pet.csv',
        'output_prefix': RESULTS_DIR / 'gps_pet_scientific_figure',
        'hist_color': '#ff9800',
    },
]


def load_sizes(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f'No se encontró la tabla de datos: {data_path}')
    sizes = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        if 'Diametro_um' not in header and 'Diametro_µm' not in header:
            raise ValueError(f'El archivo {data_path.name} no tiene columna Diametro_um o Diametro_µm')
        col_index = header.index('Diametro_um') if 'Diametro_um' in header else header.index('Diametro_µm')
        for row in reader:
            if not row or len(row) <= col_index:
                continue
            try:
                sizes.append(float(row[col_index]))
            except ValueError:
                continue
    sizes = np.array(sizes, dtype=float)
    if sizes.size == 0:
        raise ValueError(f'No se encontraron valores válidos de diámetro en {data_path.name}')
    return sizes


def detect_scale_bar(image_rgb, bottom_fraction=0.22):
    height = image_rgb.shape[0]
    crop = image_rgb[int(height * (1.0 - bottom_fraction)):]
    gray = color.rgb2gray(crop)
    gray = (gray * 255).astype(np.uint8)
    threshold = np.mean(gray) + np.std(gray) * 0.45
    bright = gray > threshold

    best_run = 0
    best_row = None
    best_start = None
    best_end = None
    for row in range(int(bright.shape[0] * 0.65)):
        run_start = None
        for col in range(bright.shape[1]):
            if bright[row, col]:
                if run_start is None:
                    run_start = col
            elif run_start is not None:
                run_len = col - run_start
                if run_len > best_run:
                    best_run = run_len
                    best_row = row
                    best_start = run_start
                    best_end = col
                run_start = None
        if run_start is not None:
            run_len = bright.shape[1] - run_start
            if run_len > best_run:
                best_run = run_len
                best_row = row
                best_start = run_start
                best_end = bright.shape[1]

    if best_run is None or best_run < 30:
        return None
    return {
        'start': int(best_start),
        'end': int(best_end),
        'row': int((image_rgb.shape[0] - crop.shape[0]) + best_row),
        'width': int(best_run),
    }


def save_distribution_table(output_prefix, counts, bin_edges):
    table_path = f'{output_prefix}_size_distribution.csv'
    with open(table_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Bin_start_um', 'Bin_end_um', 'Count'])
        for low, high, count in zip(bin_edges[:-1], bin_edges[1:], counts):
            writer.writerow([f'{low:.4f}', f'{high:.4f}', int(count)])
    return Path(table_path)


def build_scientific_figure(sample):
    image_path = sample['image_path']
    data_path = sample['data_path']
    output_prefix = sample['output_prefix']
    hist_color = sample['hist_color']

    print(f"Generando figura para {sample['name']}")
    sizes = load_sizes(data_path)
    print(f'  ✓ {sizes.size} partículas leídas de {data_path.name}')
    print(f'  ✓ Rango de diámetros: {sizes.min():.3f}–{sizes.max():.3f} µm')

    img = Image.open(image_path).convert('RGB')
    image_rgb = np.array(img)
    height, width = image_rgb.shape[:2]

    image_gray = color.rgb2gray(image_rgb)
    image_gray = (image_gray * 255).astype(np.uint8)
    p0, p98 = np.percentile(image_gray, (2, 98))
    display_gray = exposure.rescale_intensity(image_gray, in_range=(p0, p98), out_range=(0, 255)).astype(np.uint8)

    scale_line = detect_scale_bar(image_rgb, bottom_fraction=0.22)
    if scale_line is not None:
        px_to_um = 100.0 / scale_line['width']
        print(f'  ✓ Barra de escala detectada: {scale_line["width"]} px = 100 µm, conversión {px_to_um:.6f} µm/px')
    else:
        print('  ⚠️ No se detectó de forma confiable la barra de escala. Se conserva la marca visible existente.')

    bins = max(12, min(36, int(np.sqrt(len(sizes)) * 1.2)))
    counts, bin_edges = np.histogram(sizes, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    kde = stats.gaussian_kde(sizes)
    x_grid = np.linspace(sizes.min(), sizes.max(), 300)
    pdf = kde(x_grid)
    pdf_scaled = pdf * len(sizes) * bin_width

    fig = plt.figure(figsize=(10, 10), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(display_gray, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')

    label_text = f'{sample["name"]}'
    ax.text(0.04, 0.96, label_text, transform=ax.transAxes,
            color='white', fontsize=16, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', pad=6))

    if scale_line is not None:
        ax.plot([scale_line['start'], scale_line['end']], [scale_line['row'], scale_line['row']],
                color='yellow', linewidth=6, solid_capstyle='butt', alpha=0.9)
        ax.plot([scale_line['start'], scale_line['end']], [scale_line['row'], scale_line['row']],
                color='white', linewidth=2, solid_capstyle='butt', alpha=0.9)
        ax.text(scale_line['end'] + 12, scale_line['row'] - 4, '100 µm',
                color='white', fontsize=10, weight='bold', va='bottom', ha='left',
                bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', pad=3))

    ax_hist = fig.add_axes([0.06, 0.06, 0.38, 0.28], facecolor='none')
    ax_hist.bar(bin_centers, counts, width=bin_width * 0.92,
                color=hist_color, alpha=0.40, edgecolor='white', linewidth=0.6)
    ax_hist.plot(x_grid, pdf_scaled, color='white', linewidth=2.2)

    ax_hist.set_xlabel('Size (µm)', fontsize=9, fontweight='bold', color='white', labelpad=4)
    ax_hist.set_ylabel('Count', fontsize=9, fontweight='bold', color='white', labelpad=4)
    ax_hist.tick_params(axis='both', which='major', labelsize=7, colors='white')
    ax_hist.set_facecolor((0, 0, 0, 0))

    for spine in ax_hist.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(0.8)
    ax_hist.grid(axis='y', color='white', alpha=0.18, linestyle='--', linewidth=0.5)
    ax_hist.set_xlim(sizes.min() * 0.95, sizes.max() * 1.05)
    ax_hist.set_ylim(0, counts.max() * 1.25)

    if counts.max() <= 6:
        ax_hist.set_yticks(np.arange(0, int(counts.max() + 2), 1))

    for ext in ['png', 'pdf', 'svg']:
        output_path = f'{output_prefix}.{ext}'
        fig.savefig(output_path, dpi=600 if ext == 'png' else None,
                    bbox_inches='tight', pad_inches=0.03, facecolor='white')
        print(f'  ✓ Guardado: {Path(output_path).name}')

    save_distribution_table(output_prefix, counts, bin_edges)
    print(f'  ✓ Tabla de distribución guardada: {output_prefix}_size_distribution.csv')
    plt.close(fig)


def main():
    for sample in SAMPLES:
        try:
            build_scientific_figure(sample)
        except Exception as exc:
            print(f'Error en {sample["name"]}: {exc}')


if __name__ == '__main__':
    main()
