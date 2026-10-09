"""Build closed, nanometre-unit STL solids from a dark-carbon image mask."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
import trimesh


SAMPLES = Path(__file__).resolve().parent / 'samples'
STL_SAMPLES = Path(__file__).resolve().parents[1] / 'nion' / 'usim_device' / 'samples'


def otsu_threshold(gray):
    histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
    count = histogram.cumsum()
    moment = (histogram*np.arange(256)).cumsum()
    valid = (count > 0) & (count < gray.size)
    score = np.zeros(256)
    score[valid] = (moment[-1]*count[valid]-moment[valid]*gray.size)**2 / (count[valid]*(gray.size-count[valid]))
    return int(score.argmax())


def resolve_diagonal_contacts(mask, gray):
    """Join zero-width diagonal contacts with one pixel to make manifold walls."""
    mask = mask.copy()
    original = int(mask.sum())
    while True:
        a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
        first = a & d & ~b & ~c
        second = b & c & ~a & ~d
        if not (first.any() or second.any()):
            break
        # Prefer the darker of the two candidate pixels. No carbon is removed.
        for selected, corners in ((first, ((0, 1), (1, 0))), (second, ((0, 0), (1, 1)))):
            rows, columns = np.nonzero(selected)
            (r0, c0), (r1, c1) = corners
            choose = gray[rows+r0, columns+c0] <= gray[rows+r1, columns+c1]
            mask[rows[choose]+r0, columns[choose]+c0] = True
            mask[rows[~choose]+r1, columns[~choose]+c1] = True
    return mask, int(mask.sum())-original


def extrude_mask(mask, size_nm=54000., thickness_nm=5.):
    height, width = mask.shape
    xx, yy = np.meshgrid(np.linspace(-size_nm/2, size_nm/2, width+1),
                         np.linspace(size_nm/2, -size_nm/2, height+1))
    plane = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    vertices = np.vstack((plane, plane + [0, 0, thickness_nm]))
    rows, columns = np.nonzero(mask)
    ul = rows*(width+1)+columns
    ur, ll, lr = ul+1, ul+(width+1), ul+(width+2)
    top = xx.size
    faces = [np.column_stack((ll+top, lr+top, ur+top)), np.column_stack((ll+top, ur+top, ul+top)),
             np.column_stack((ll, ur, lr)), np.column_stack((ll, ul, ur))]
    padded = np.pad(mask, 1)
    neighbours = (padded[2:, 1:-1], padded[1:-1, 2:], padded[:-2, 1:-1], padded[1:-1, :-2])
    for neighbour, start, end in zip(neighbours, (ll, lr, ur, ul), (lr, ur, ul, ll)):
        boundary = ~neighbour[rows, columns]
        a, b = start[boundary], end[boundary]
        faces.extend((np.column_stack((a, b, b+top)), np.column_stack((a, b+top, a+top))))
    mesh = trimesh.Trimesh(vertices, np.vstack(faces), process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def copper_frame(outer_nm=85000., opening_nm=54000., thickness_nm=10000.):
    outer = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*outer_nm/2
    inner = outer*opening_nm/outer_nm
    xy = np.vstack((outer, inner))
    plane = np.column_stack((xy, np.zeros(8)))
    vertices = np.vstack((plane, plane+[0, 0, thickness_nm]))
    faces = []
    def quad(a, b, c, d):
        faces.extend(((a, b, c), (a, c, d)))
    for i in range(4):
        j = (i+1) % 4
        quad(i+8, j+8, j+12, i+12)  # top ring
        quad(i+4, j+4, j, i)        # bottom ring
        quad(i, j, j+8, i+8)       # outside
        quad(j+4, i+4, i+12, j+12) # inside opening
    return trimesh.Trimesh(vertices, faces, process=False)


def verify(mesh, expected_volume):
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Model must be a closed, consistently oriented positive-volume solid')
    if not np.isclose(mesh.volume, expected_volume, rtol=2e-6):
        raise ValueError(f'Incorrect volume: {mesh.volume} != {expected_volume}')


def generate(template, output, threshold=None, stl_output=STL_SAMPLES):
    gray = np.asarray(Image.open(template).convert('L'))
    threshold = otsu_threshold(gray) if threshold is None else threshold
    mask, repaired = resolve_diagonal_contacts(gray <= threshold, gray)
    output.mkdir(parents=True, exist_ok=True)
    stl_output.mkdir(parents=True, exist_ok=True)
    carbon = extrude_mask(mask)
    copper = copper_frame()
    carbon_volume = float(mask.sum())*(54000.**2/mask.size)*5
    copper_volume = (85000.**2-54000.**2)*10000
    verify(carbon, carbon_volume)
    verify(copper, copper_volume)
    combined = trimesh.util.concatenate((carbon, copper))
    meshes = {'lacey_carbon_54um_5nm_pixelated.stl': (carbon, carbon_volume),
              'copper_grid_85um_10um.stl': (copper, copper_volume),
              'lacey_carbon_with_copper_grid_pixelated.stl': (combined, carbon_volume+copper_volume)}
    info = {'coordinate_unit': 'nm', 'origin': 'XY centre; both solids start at z=0',
            'template': template.name, 'template_sha256': hashlib.sha256(template.read_bytes()).hexdigest(),
            'template_shape_px': list(gray.shape), 'dark_threshold_inclusive': threshold,
            'diagonal_contact_pixels_filled': repaired, 'carbon_area_fraction': float(mask.mean()),
            'carbon_components': int(label(mask)[1]), 'carbon_size_nm': [54000, 54000, 5],
            'copper_outer_size_nm': [85000, 85000, 10000], 'copper_opening_size_nm': [54000, 54000],
            'copper_frame_width_nm': 15500, 'parts': {}}
    for filename, (mesh, volume) in meshes.items():
        path = stl_output/filename
        mesh.export(path, file_type='stl')
        loaded = trimesh.load_mesh(path, process=True)
        verify(loaded, volume)
        info['parts'][filename] = {'faces': len(loaded.faces), 'bounds_nm': loaded.bounds.tolist(),
            'volume_nm3': float(loaded.volume), 'watertight': bool(loaded.is_watertight)}
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(output/'carbon_binary_mask.png')
    preview = Image.new('RGB', (1100, 1180), 'white')
    drawing = ImageDraw.Draw(preview)
    drawing.rectangle((50, 50, 1050, 1050), fill='#b77942')
    film = Image.fromarray(np.where(mask, 45, 255).astype(np.uint8)).convert('RGB')
    preview.paste(film.resize((635, 635), Image.Resampling.NEAREST), (232, 232))
    drawing.text((50, 1080), 'Top view: outer copper frame 85 x 85 um; central opening 54 x 54 um', fill='black')
    drawing.text((50, 1110), 'Dark carbon: 5 nm thick. Copper: 10 um thick. White: vacuum.', fill='black')
    preview.save(output/'lacey_carbon_grid_top_view.png')
    (output/'model_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    (output/'README.txt').write_text(
        'All STL coordinates are in nanometres (STL has no embedded unit or material).\n'
        f'STL storage directory: {stl_output.resolve()}\n'
        'Carbon: 54000 x 54000 nm footprint, z=0..5 nm, dark template regions only.\n'
        'Copper: 85000 x 85000 nm outer frame, 54000 x 54000 nm opening, z=0..10000 nm.\n'
        'The separate STL files identify materials. The combined STL preserves both solids but has no chemistry labels.\n'
        'Image rows map from positive Y downwards, preserving the template top view.\n'
        'Contours follow the input pixel grid; one pixel is approximately 109.76 nm.\n'
        'Zero-width diagonal contacts are minimally filled to obtain closed manifold surfaces; see model_info.json.\n', encoding='utf-8')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, default=SAMPLES/'Carbon_film_mask.jpg')
    parser.add_argument('--output', type=Path, default=SAMPLES/'lacey_carbon_grid')
    parser.add_argument('--threshold', type=int)
    parser.add_argument('--stl-output', type=Path, default=STL_SAMPLES)
    args = parser.parse_args()
    generate(args.template, args.output, args.threshold, args.stl_output)
