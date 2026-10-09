"""Generate a smooth connected 54 um support using a 10 um lacey exemplar.

Random rounded Voronoi pores reproduce the exemplar's pore density and carbon
area fraction without repeating its image. STL coordinates are nanometres.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from scipy.spatial import Voronoi
from shapely import union_all
from shapely.geometry import Polygon, box, Point
from shapely.ops import nearest_points
import trimesh

from create_lacey_carbon import SAMPLES, STL_SAMPLES, copper_frame, otsu_threshold, verify
from smooth_lacey_carbon import rasterize


def template_statistics():
    gray = np.asarray(Image.open(SAMPLES/'Carbon_film_mask.jpg').convert('L'))
    mask = gray <= otsu_threshold(gray)
    labels, count = label(~mask)
    boundary = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    areas = np.bincount(labels.ravel())[1:] * 10000.**2 / mask.size
    enclosed = np.array([areas[i-1] for i in range(1, count+1) if i not in boundary and areas[i-1] > 2500.])
    # Border pores are fractions of pores; area-weight them using the enclosed mean.
    density = float((~mask).mean()/enclosed.mean())
    return float(mask.mean()), density, enclosed


def make_carbon(seed):
    fraction, density, source_areas = template_statistics()
    rng = np.random.default_rng(seed)
    domain = box(-27000., -27000., 27000., 27000.)
    margin = 3000.
    count = int(round(density*(54000+2*margin)**2))
    sites = rng.uniform(-27000-margin, 27000+margin, (count, 2))
    # Distant guard sites close every Voronoi cell intersecting the opening.
    guards = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])*200000.
    voronoi = Voronoi(np.vstack((sites, guards)))
    cells = []
    for i in range(len(sites)):
        indices = voronoi.regions[voronoi.point_region[i]]
        if not indices or -1 in indices:
            continue
        cell = Polygon(voronoi.vertices[indices])
        if cell.intersects(domain):
            cells.append((cell, rng.uniform(.65, 1.4), rng.uniform(.16, .30), rng.random() < .012))

    def holes_for(width):
        holes = []
        for cell, factor, rounding, filled in cells:
            if filled:
                continue
            inset = cell.buffer(-width*factor/2)
            radius = np.sqrt(cell.area)*rounding
            hole = inset.buffer(-radius).buffer(radius, quad_segs=8).intersection(domain)
            if not hole.is_empty and hole.area > 2500.:
                holes.append(hole)
        return holes

    lower, upper = 1., 1000.
    for _ in range(15):
        width = (lower+upper)/2
        holes = holes_for(width)
        actual = 1-sum(h.area for h in holes)/domain.area
        if actual < fraction:
            lower = width
        else:
            upper = width
    holes = holes_for((lower+upper)/2)
    carbon = domain.difference(union_all(holes)).simplify(2., preserve_topology=True)
    # Move only the prism if its lowest vertex is over vacuum. Never fill pores
    # or add support underneath its projection; suspended regions remain vacuum.
    prism_path = STL_SAMPLES/'hexagonal_prism.stl'
    if prism_path.exists():
        prism = trimesh.load_mesh(prism_path)
    else:
        # Recover the same prism from the current assembly if only the combined
        # model was retained; do not resample its dimensions or orientation.
        metadata = json.loads((STL_SAMPLES/'single_cathode.json').read_text(encoding='utf-8'))
        bounds = np.asarray(metadata['models']['hexagonal_prism.stl']['bounds_nm'])
        assembly = trimesh.load_mesh(STL_SAMPLES/'hexagonal_prism_on_lacey_carbon_with_copper_grid.stl')
        triangles = assembly.triangles
        selected = (np.all((triangles >= bounds[0]-1e-3) & (triangles <= bounds[1]+1e-3), axis=(1, 2))
                    & (triangles[:, :, 2].max(axis=1) > 5.001))
        if np.count_nonzero(selected) != 20:
            raise ValueError('Cannot unambiguously recover the existing prism from the assembly')
        prism = assembly.submesh([np.flatnonzero(selected)], append=True)
    contact = prism.vertices[np.argmin(prism.vertices[:, 2])]
    inner = carbon.buffer(-10.)
    target = nearest_points(inner, Point(contact[:2]))[0]
    displacement = np.array(target.coords[0])-contact[:2]
    prism.apply_translation([*displacement, 0.])
    if not carbon.is_valid or carbon.geom_type != 'Polygon' or not carbon.covers(target):
        raise ValueError('Film must be connected and the lowest prism vertex must touch carbon')
    info = {'seed': seed, 'method': 'rounded random Voronoi pores; calibrated to 10 um exemplar',
            'coordinate_unit': 'nm', 'template_size_nm': [10000, 10000],
            'template_sha256': hashlib.sha256((SAMPLES/'Carbon_film_mask.jpg').read_bytes()).hexdigest(),
            'carbon_size_nm': [54000, 54000, 5], 'carbon_components': 1,
            'enclosed_holes': len(carbon.interiors), 'source_carbon_area_fraction': fraction,
            'carbon_area_fraction': carbon.area/domain.area, 'nominal_strut_width_nm': (lower+upper)/2,
            'source_median_pore_diameter_nm': float(np.median(2*np.sqrt(source_areas/np.pi))),
            'generated_median_pore_diameter_nm': float(np.median([2*np.sqrt(Polygon(r).area/np.pi) for r in carbon.interiors])),
            'local_cathode_support_added_area_nm2': 0., 'cathode_translation_xy_nm': displacement.tolist(),
            'copper_outer_size_nm': [85000, 85000, 10000], 'copper_opening_size_nm': [54000, 54000],
            'parts': {}}
    return carbon, prism, info


def generate(seed, output):
    region, prism, info = make_carbon(seed)
    carbon = trimesh.creation.extrude_polygon(region, 5., engine='earcut')
    copper = copper_frame(outer_nm=85000.)
    combined = trimesh.util.concatenate((carbon, copper))
    models = {'hexagonal_prism_on_lacey_carbon_with_copper_grid.stl': trimesh.util.concatenate((combined, prism))}
    expected = {'hexagonal_prism_on_lacey_carbon_with_copper_grid.stl': region.area*5+copper.volume+prism.volume}
    output.mkdir(parents=True, exist_ok=True)
    for name, mesh in models.items():
        verify(mesh, expected[name])
        mesh.export(STL_SAMPLES/name)
        loaded = trimesh.load_mesh(STL_SAMPLES/name)
        verify(loaded, expected[name])
        info['parts'][name] = {'faces': len(loaded.faces), 'watertight': bool(loaded.is_watertight),
                               'bounds_nm': loaded.bounds.tolist(), 'volume_nm3': float(loaded.volume)}
    model_info_path = STL_SAMPLES/'single_cathode.json'
    cathode = json.loads(model_info_path.read_text(encoding='utf-8'))
    dx, dy = info['cathode_translation_xy_nm']
    cathode['bottom_face_center_nm'][0] += dx
    cathode['bottom_face_center_nm'][1] += dy
    cathode['contact_points_nm'] = prism.vertices[np.isclose(prism.vertices[:, 2], 5., atol=1e-6)].tolist()
    cathode['placement'] = 'Lowest prism vertex touches unmodified carbon at z=5 nm; partial projection over vacuum is allowed.'
    cathode['film_source_sha256'] = hashlib.sha256(carbon.export(file_type='stl')).hexdigest()
    cathode['models']['hexagonal_prism.stl'] = {'faces': len(prism.faces), 'watertight': True,
        'bounds_nm': prism.vertices.astype(np.float32).astype(float).min(axis=0).reshape(1, 3).tolist()
        + prism.vertices.astype(np.float32).astype(float).max(axis=0).reshape(1, 3).tolist(),
        'volume_nm3': float(prism.volume)}
    cathode['stl_exports'] = list(models)
    cathode['support_generation'] = {k: info[k] for k in ('seed', 'method', 'template_size_nm', 'local_cathode_support_added_area_nm2', 'cathode_translation_xy_nm', 'copper_outer_size_nm')}
    for name in cathode['models']:
        if name in info['parts']:
            cathode['models'][name] = info['parts'][name]
    text = json.dumps(cathode, indent=2)
    model_info_path.write_text(text, encoding='utf-8')
    (SAMPLES/'hexagonal_prism'/'model_info.json').write_text(text, encoding='utf-8')
    from create_hexagonal_prism import preview
    preview(prism, region, SAMPLES/'hexagonal_prism')
    film = rasterize([region], size=2700)
    film.save(output/'carbon_54um_top_view.png')
    composite = Image.new('RGB', (1100, 1180), 'white')
    draw = ImageDraw.Draw(composite)
    draw.rectangle((50, 50, 1050, 1050), fill='#b77942')
    film_pixels = round(1000*54/85)
    border = 50+(1000-film_pixels)//2
    composite.paste(film.resize((film_pixels, film_pixels), Image.Resampling.LANCZOS).convert('RGB'), (border, border))
    draw.text((50, 1080), 'Synthetic lacey carbon: 54 x 54 um, 5 nm; copper: 85 x 85 um, 10 um', fill='black')
    composite.save(output/'lacey_carbon_grid_top_view.png')
    # Show a physical 10 um field next to the original 10 um image.
    pixels = round(2700*10000/54000)
    crop = film.crop((1350-pixels//2, 1350-pixels//2, 1350+pixels//2, 1350+pixels//2))
    comparison = Image.new('RGB', (1040, 560), 'white')
    comparison.paste(Image.open(SAMPLES/'Carbon_film_mask.jpg').resize((492, 492)), (10, 45))
    comparison.paste(crop.resize((492, 492)).convert('RGB'), (530, 45))
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 10), 'Original: approximately 10 x 10 um', fill='black')
    draw.text((530, 10), 'Generated: 10 x 10 um crop', fill='black')
    comparison.save(output/'scale_comparison.png')
    (output/'model_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--output', type=Path, default=SAMPLES/'lacey_carbon_generated')
    args = parser.parse_args()
    generate(args.seed, args.output)
