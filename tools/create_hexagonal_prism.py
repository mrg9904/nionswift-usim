"""Place a random tilted regular hexagonal prism on the smooth carbon STL.

Coordinates are nm. [abc] denotes a Cartesian direction in the sample frame;
no lattice parameters or crystal chemistry are inferred from an STL.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely import union_all
from shapely.geometry import Polygon, MultiPoint, box, Point
import trimesh

from create_lacey_carbon import SAMPLES, STL_SAMPLES, verify
from nion.usim_device import SampleGeometry


def make_prism(side, height, abc, spin):
    normal = np.asarray(abc, dtype=float)
    normal /= np.linalg.norm(normal)
    u = np.cross([0., 0., 1.], normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angles = spin + np.arange(6) * np.pi / 3
    bottom = side * (np.cos(angles)[:, None]*u + np.sin(angles)[:, None]*v)
    vertices = np.vstack((bottom, bottom + height*normal))
    faces = []
    for i in range(1, 5):
        faces.extend(((0, i+1, i), (6, 6+i, 6+i+1)))
    for i in range(6):
        j = (i+1) % 6
        faces.extend(((i, j, 6+j), (i, 6+j, 6+i)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), normal


def preview(prism, film_region, output):
    vertices = prism.vertices
    center = vertices[:, :2].mean(axis=0)
    radius = max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1])) * .85
    region = film_region.intersection(box(*(center-radius), *(center+radius)))
    image = Image.new('RGB', (1400, 760), 'white')
    draw = ImageDraw.Draw(image)
    def top(points):
        points = np.asarray(points)
        return [(float(350+(x-center[0])*300/radius), float(390-(y-center[1])*300/radius)) for x, y in points]
    for polygon in ([region] if region.geom_type == 'Polygon' else region.geoms):
        if polygon.geom_type != 'Polygon':
            continue
        draw.polygon(top(polygon.exterior.coords), fill='#777777')
        for ring in polygon.interiors:
            draw.polygon(top(ring.coords), fill='white')
    hull = MultiPoint(vertices[:, :2]).convex_hull
    draw.polygon(top(hull.exterior.coords), fill='#f0ab42', outline='black', width=2)
    draw.text((30, 20), 'Top view: gray carbon, white vacuum, orange prism projection', fill='black')
    draw.text((30, 710), f'View width: {2*radius:.1f} nm; film top z=5 nm', fill='black')
    local = vertices - np.r_[center, 5.]
    def iso(points):
        points = np.asarray(points)
        projected = np.column_stack((.8*points[:, 0]-.6*points[:, 1],
            .3*points[:, 0]+.4*points[:, 1]-.85*points[:, 2]))
        return [(float(1050+x*240/radius), float(430+y*240/radius)) for x, y in projected]
    plane = np.array([[-radius, -radius, 0], [radius, -radius, 0],
                      [radius, radius, 0], [-radius, radius, 0]])
    draw.polygon(iso(plane), fill='#eeeeee', outline='#aaaaaa')
    # Loaded STL vertex order is arbitrary; use actual faces rather than assuming
    # vertices 0..5 and 6..11 still enumerate the two hexagonal bases.
    facets = prism.faces.tolist()
    for facet in sorted(facets, key=lambda ids: float(local[ids, 1].mean()), reverse=True):
        draw.polygon(iso(local[facet]), fill='#e6a23b', outline='#5f421a', width=2)
    draw.text((730, 20), '3D view: tilted prism; gray plane marks carbon top z=5 nm', fill='black')
    image.save(output/'hexagonal_prism_preview.png')


def generate(seed, output, stl_output):
    rng = np.random.default_rng(seed)
    side, height = rng.uniform(50., 500., 2)
    abc = rng.integers(1, 101, 3)
    spin = rng.uniform(0., 2*np.pi)
    prism, normal = make_prism(side, height, abc, spin)
    assembly = trimesh.load_mesh(STL_SAMPLES/'hexagonal_prism_on_lacey_carbon_with_copper_grid.stl', process=True)
    metadata = json.loads((STL_SAMPLES/'single_cathode.json').read_text(encoding='utf-8'))
    support = assembly.submesh([np.flatnonzero(~SampleGeometry.cathode_face_mask(assembly, metadata))], append=True)
    carbon_faces = np.all(support.triangles[:, :, 2] <= 5.00001, axis=1) & np.all(np.abs(support.triangles[:, :, :2]) <= 27000.001, axis=(1, 2))
    film = support.submesh([np.flatnonzero(carbon_faces)], append=True)
    triangles = film.triangles
    upper = np.all(np.isclose(triangles[:, :, 2], 5., atol=1e-6), axis=1)
    film_region = union_all([Polygon(t[:, :2]) for t in triangles[upper]])
    contact = prism.vertices[np.argmin(prism.vertices[:, 2]), :2]
    # Only the lowest point must touch existing carbon; overhang is allowed.
    for _ in range(100000):
        xy = rng.uniform(-26000., 26000., 2)
        if film_region.covers(Point(contact+xy)):
            break
    else:
        raise ValueError('No carbon area can accommodate the sampled prism; try another seed')
    shift = np.r_[xy, 5.-prism.vertices[:, 2].min()]
    prism.apply_translation(shift)
    expected_volume = 3*np.sqrt(3)/2*side**2*height
    verify(prism, expected_volume)
    bottom = prism.vertices[:6]
    edges = np.roll(bottom, -1, axis=0)-bottom
    np.testing.assert_allclose(np.linalg.norm(edges, axis=1), side)
    np.testing.assert_allclose(edges[:3], -edges[3:], atol=1e-8)
    np.testing.assert_allclose(prism.vertices[6:]-bottom, np.tile(height*normal, (6, 1)))
    assert np.isclose(prism.bounds[0, 2], 5.)
    models = {'hexagonal_prism_on_lacey_carbon_with_copper_grid.stl': trimesh.util.concatenate((support, prism))}
    stl_output.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    info = {'seed': seed, 'coordinate_unit': 'nm', 'side_length_nm': float(side),
        'height_along_prism_axis_nm': float(height), 'film_normal': [0, 0, 1],
        'base_normal_abc': abc.tolist(), 'base_normal_unit_xyz': normal.tolist(),
        'tilt_from_film_normal_deg': float(np.degrees(np.arccos(normal[2]))),
        'spin_about_prism_axis_deg': float(np.degrees(spin)),
        'bottom_face_center_nm': bottom.mean(axis=0).tolist(),
        'contact_points_nm': prism.vertices[np.isclose(prism.vertices[:, 2], 5.)].tolist(),
        'film_source_sha256': hashlib.sha256(film.export(file_type='stl')).hexdigest(),
        'placement': 'Lowest point touches film top z=5 nm; partial XY projection over vacuum is allowed.',
        'direction_convention': '[abc] is Cartesian in sample XYZ; no CIF/lattice is assigned.',
        'stl_exports': list(models), 'models': {'hexagonal_prism.stl': {
            'watertight': True, 'faces': len(prism.faces),
            'bounds_nm': np.vstack((prism.vertices.astype(np.float32).min(axis=0), prism.vertices.astype(np.float32).max(axis=0))).astype(float).tolist(),
            'volume_nm3': float(prism.volume)}}}
    for filename, mesh in models.items():
        mesh.export(stl_output/filename)
        loaded = trimesh.load_mesh(stl_output/filename, process=True)
        verify(loaded, mesh.volume)
        info['models'][filename] = {'watertight': bool(loaded.is_watertight),
            'faces': len(loaded.faces), 'bounds_nm': loaded.bounds.tolist(), 'volume_nm3': float(loaded.volume)}
    (output/'model_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    (stl_output/'single_cathode.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    preview(prism, film_region, output)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output', type=Path, default=SAMPLES/'hexagonal_prism')
    parser.add_argument('--stl-output', type=Path, default=STL_SAMPLES)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy)
    generate(seed, args.output, args.stl_output)
