"""Extract subpixel smooth contours and extrude the lacey carbon STL to 5 nm.

Requires shapely, contourpy and mapbox-earcut in the model-building environment.
Original pixel-based models are retained for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import contourpy
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, zoom
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
import trimesh
from create_lacey_carbon import SAMPLES, STL_SAMPLES, copper_frame, otsu_threshold, resolve_diagonal_contacts, verify


def smooth_polygons(mask, sigma=.4, upsample=4):
    """Smooth the planar boundary, preserving flat upper/lower film surfaces."""
    height, width = mask.shape
    pitch_x, pitch_y = 54000./width, 54000./height
    # Continue the boundary mask beyond the crop, then clip the contour.
    # Zero padding would shrink the film away from its copper support.
    padded = np.pad(mask[::-1], 1, mode='edge')
    signed = distance_transform_edt(padded)-distance_transform_edt(~padded)
    field = zoom(gaussian_filter(signed, sigma, mode='nearest'), upsample, order=3)
    x = np.linspace(-27000-pitch_x/2, 27000+pitch_x/2, field.shape[1])
    y = np.linspace(-27000-pitch_y/2, 27000+pitch_y/2, field.shape[0])
    generator = contourpy.contour_generator(x=x, y=y, z=field, fill_type='OuterOffset')
    vertices, offsets = generator.filled(0., float(field.max())+1)
    polygons = []
    for points, rings in zip(vertices, offsets):
        boundaries = [points[a:b] for a, b in zip(rings[:-1], rings[1:])]
        polygon = Polygon(boundaries[0], boundaries[1:]).intersection(box(-27000, -27000, 27000, 27000))
        parts = [polygon] if polygon.geom_type == 'Polygon' else list(polygon.geoms)
        for part in parts:
            if part.geom_type != 'Polygon' or part.area <= 0:
                continue
            part = part.simplify(min(pitch_x, pitch_y)*.02, preserve_topology=True)
            if not part.is_valid:
                raise ValueError('Invalid smoothed polygon')
            polygons.append(orient(part, sign=1))
    return polygons


def rasterize(polygons, size=1968):
    image = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(image)
    def screen(ring):
        return [((x+27000)/54000*size, (27000-y)/54000*size) for x, y in ring.coords]
    for polygon in polygons:
        draw.polygon(screen(polygon.exterior), fill=45)
        for hole in polygon.interiors:
            draw.polygon(screen(hole), fill=255)
    return image


def generate(output, sigma=.4, stl_output=STL_SAMPLES):
    gray = np.asarray(Image.open(SAMPLES/'Carbon_film_mask.jpg').convert('L'))
    threshold = otsu_threshold(gray)
    mask, repaired = resolve_diagonal_contacts(gray <= threshold, gray)
    original_components = int(label(mask)[1])
    original_holes = int(label(np.pad(~mask, 1, constant_values=True))[1])-1
    polygons = smooth_polygons(mask, sigma)
    holes = sum(len(p.interiors) for p in polygons)
    if len(polygons) != original_components or holes != original_holes:
        raise ValueError(f'Smoothing changed topology: components {original_components}->{len(polygons)}, holes {original_holes}->{holes}')
    carbon = trimesh.util.concatenate([trimesh.creation.extrude_polygon(p, 5., engine='earcut') for p in polygons])
    carbon_area = sum(p.area for p in polygons)
    copper = copper_frame()
    combined = trimesh.util.concatenate((carbon, copper))
    output.mkdir(parents=True, exist_ok=True)
    stl_output.mkdir(parents=True, exist_ok=True)
    models = {'lacey_carbon_54um_5nm.stl': (carbon, carbon_area*5),
              'copper_grid_85um_10um.stl': (copper, (85000.**2-54000.**2)*10000),
              'lacey_carbon_with_copper_grid.stl': (combined, carbon_area*5+(85000.**2-54000.**2)*10000)}
    info = {'coordinate_unit': 'nm', 'carbon_size_nm': [54000, 54000, 5],
            'copper_outer_size_nm': [85000, 85000, 10000], 'copper_opening_size_nm': [54000, 54000],
            'smoothing_sigma_source_pixels': sigma, 'smoothing_scale_nm': sigma*54000/gray.shape[0],
            'contour_upsampling': 4, 'dark_threshold_inclusive': threshold,
            'diagonal_contact_pixels_filled': repaired, 'carbon_components': len(polygons), 'enclosed_holes': holes,
            'original_carbon_components': original_components, 'original_enclosed_holes': original_holes,
            'carbon_area_fraction': carbon_area/54000.**2,
            'area_change_percent': 100*(carbon_area/(mask.mean()*54000.**2)-1), 'parts': {}}
    for filename, (mesh, volume) in models.items():
        verify(mesh, volume)
        mesh.export(stl_output/filename)
        loaded = trimesh.load_mesh(stl_output/filename)
        verify(loaded, volume)
        info['parts'][filename] = {'faces': len(loaded.faces), 'watertight': bool(loaded.is_watertight),
                                  'bounds_nm': loaded.bounds.tolist(), 'volume_nm3': float(loaded.volume)}
    film = rasterize(polygons)
    film.save(output/'carbon_smooth_top_view.png')
    preview = Image.new('RGB', (1100, 1180), 'white')
    drawing = ImageDraw.Draw(preview)
    drawing.rectangle((50, 50, 1050, 1050), fill='#b77942')
    preview.paste(film.resize((635, 635), Image.Resampling.LANCZOS).convert('RGB'), (232, 232))
    drawing.text((50, 1080), 'Smooth lacey carbon: 54 x 54 um, 5 nm thick; copper frame: 85 x 85 um, 10 um thick', fill='black')
    preview.save(output/'lacey_carbon_grid_top_view.png')
    original = Image.open(SAMPLES/'lacey_carbon_grid'/'carbon_binary_mask.png').convert('L')
    old_crop = original.crop((100, 100, 220, 220)).resize((600, 600), Image.Resampling.NEAREST)
    new_crop = film.crop((400, 400, 880, 880)).resize((600, 600), Image.Resampling.LANCZOS)
    compare = Image.new('RGB', (1240, 650), 'white')
    compare.paste(old_crop, (10, 40))
    compare.paste(new_crop, (630, 40))
    ImageDraw.Draw(compare).text((10, 10), 'Original pixel contour', fill='black')
    ImageDraw.Draw(compare).text((630, 10), 'New smooth subpixel contour (same area)', fill='black')
    compare.save(output/'edge_smoothing_comparison.png')
    (output/'model_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    (output/'README.txt').write_text('STL units: nm. Carbon z=0..5 nm; copper z=0..10000 nm.\n'
        f'STL storage directory: {stl_output.resolve()}\n'
        'Smoothed signed-distance contours use subpixel coordinates, not voxel steps.\n'
        'Carbon components and enclosed hole counts are checked against the original.\n'
        'STL contains no material labels; separate files identify carbon and copper.\n', encoding='utf-8')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SAMPLES/'lacey_carbon_grid_smooth')
    parser.add_argument('--sigma', type=float, default=.4)
    parser.add_argument('--stl-output', type=Path, default=STL_SAMPLES)
    args = parser.parse_args()
    generate(args.output, args.sigma, args.stl_output)
