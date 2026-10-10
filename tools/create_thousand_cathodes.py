"""Place 1000 irregular parallel-opposite-edge hexagonal prisms on lacey carbon.

Coordinates and dimensions are nm. Tilted convex solids are stacked by exact
linear-programming vertical contact, without modifying the carbon support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from shapely import union_all
from shapely.geometry import Polygon, MultiPoint, Point
from shapely.affinity import translate
import trimesh

from create_lacey_carbon import SAMPLES, STL_SAMPLES, verify
from nion.usim_device import SampleGeometry


def truncated_lorentzian(rng, size, peak=300., hwhm=100.):
    """Inverse CDF on 50..1000 nm; truncation never clips values to endpoints."""
    low = np.arctan((50.-peak)/hwhm)
    high = np.arctan((1000.-peak)/hwhm)
    return peak+hwhm*np.tan(rng.uniform(low, high, size))


def make_prism(rng, peak=300., hwhm=100.):
    dimensions = truncated_lorentzian(rng, 4, peak, hwhm)
    lengths = dimensions[:3]
    height = float(dimensions[3])
    # Three distinct directions in a half-turn, followed by their negatives,
    # close a convex centrally symmetric hexagon with parallel opposite sides.
    gaps = rng.dirichlet([3., 3., 3.])*np.pi
    while gaps.min() < np.radians(12.):
        gaps = rng.dirichlet([3., 3., 3.])*np.pi
    angles = np.r_[0., gaps[0], gaps[:2].sum()]
    directions = np.column_stack((np.cos(angles), np.sin(angles)))*lengths[:, None]
    edges = np.vstack((directions, -directions))
    polygon = np.vstack((np.zeros(2), np.cumsum(edges, axis=0)))[:-1]
    polygon -= polygon.mean(axis=0)
    normal_abc = rng.integers(0, 101, 3)
    while not normal_abc[:2].any():
        normal_abc = rng.integers(0, 101, 3)
    normal = normal_abc/np.linalg.norm(normal_abc)
    u = np.cross([0., 0., 1.], normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    spin = float(rng.uniform(0., 2*np.pi))
    rotation = np.column_stack((np.cos(spin)*u+np.sin(spin)*v,
                               -np.sin(spin)*u+np.cos(spin)*v, normal))
    bottom = np.column_stack((polygon, np.zeros(6))) @ rotation.T
    vertices = np.vstack((bottom, bottom+height*normal))
    vertices[:, 2] -= vertices[:, 2].min()
    faces = []
    for i in range(1, 5):
        faces.extend(((0, i+1, i), (6, 6+i, 6+i+1)))
    for i in range(6):
        j = (i+1) % 6
        faces.extend(((i, j, 6+j), (i, 6+j, 6+i)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    verify(mesh, Polygon(polygon).area*height)
    np.testing.assert_allclose(np.linalg.norm(np.roll(bottom, -1, axis=0)-bottom, axis=1), np.tile(lengths, 2), atol=1e-8)
    np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
    # Record eight unique support planes, rather than triangulation duplicates.
    equations = np.unique(np.round(ConvexHull(vertices).equations, 10), axis=0)
    return mesh, equations, dict(base_edge_lengths_nm=np.tile(lengths, 2).tolist(), height_nm=height,
        base_vertices_local_nm=polygon.tolist(), base_normal_abc=normal_abc.tolist(),
        base_normal_unit_xyz=normal.tolist(), rotation_matrix_local_to_lab=rotation.tolist(),
        tilt_from_z_deg=float(np.degrees(np.arccos(normal[2]))),
        normal_azimuth_deg=float(np.degrees(np.arctan2(normal[1], normal[0]))),
        spin_about_normal_deg=float(np.degrees(spin)),
        euler_xyz_deg=Rotation.from_matrix(rotation).as_euler('xyz', degrees=True).tolist())


def vertical_contact(old_planes, new_planes):
    """Minimum upward shift that puts the new convex prism above the old one.

    Maximize old_z - new_z at a common XY, constrained to both solids. This
    examines their complete faceted surfaces, not just centers or vertices.
    """
    old_a, old_b = old_planes
    new_a, new_b = new_planes
    matrix = np.zeros((len(old_a)+len(new_a), 4))
    matrix[:len(old_a), :3] = old_a
    matrix[len(old_a):, :2] = new_a[:, :2]
    matrix[len(old_a):, 3] = new_a[:, 2]
    result = linprog([0., 0., -1., 1.], A_ub=matrix, b_ub=np.r_[old_b, new_b],
                     bounds=[(None, None)]*4, method='highs',
                     options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f'Contact solver failed: {result.message}')
    return float(-result.fun), result.x[:3]


class SpatialGrid:
    def __init__(self, pitch=3000.):
        self.pitch, self.cells = pitch, {}

    def keys(self, bounds):
        low = np.floor(np.array(bounds[:2])/self.pitch).astype(int)
        high = np.floor(np.array(bounds[2:])/self.pitch).astype(int)
        return [(x, y) for x in range(low[0], high[0]+1) for y in range(low[1], high[1]+1)]

    def query(self, bounds):
        return sorted(set().union(*(self.cells.get(key, set()) for key in self.keys(bounds))))

    def add(self, index, bounds):
        for key in self.keys(bounds):
            self.cells.setdefault(key, set()).add(index)


def save_overview(info, output):
    """Show the top and side projections at equal X/Z scale without plot deps."""
    image = Image.new('RGB', (1500, 1000), 'white')
    image.paste(Image.open(output/'top_view.png').resize((800, 835), Image.Resampling.LANCZOS), (0, 50))
    draw = ImageDraw.Draw(image)
    scale = min(6.5, 800/(info['bounds_nm'][1][2]/1000.))
    def side(points):
        return [(float(1150+x/1000.*scale), float(900-z/1000.*scale)) for x, z in points]
    for r in sorted(info['particles'], key=lambda r: r['center_nm'][1]):
        hull = MultiPoint(np.asarray(r['vertices_lab_nm'])[:, [0, 2]]).convex_hull
        level = r['stack_level']
        draw.polygon(side(hull.exterior.coords), fill=(80+level*3, 150, max(60, 210-level*3)), outline='#405050')
    for left, right in ((-42500, -27000), (27000, 42500)):
        draw.polygon(side([(left, 0), (right, 0), (right, 10000), (left, 10000)]), fill='#b77942')
    draw.line(side([(-27000, 0), (27000, 0)]), fill='gray', width=2)
    draw.text((70, 20), 'Top view: carbon opening 54 um; total grid 85 um', fill='black')
    draw.text((910, 20), 'Side projection (X-Z, equal scale)', fill='black')
    draw.text((910, 45), f'Particle maximum above carbon: {info["max_particle_height_above_carbon_nm"]/1000.:.2f} um', fill='black')
    step = 2 if info['bounds_nm'][1][2] <= 10000 else 50
    for z in range(0, int(info['bounds_nm'][1][2]/1000.)+1, step):
        y = 900-z*scale
        draw.line((880, y, 890, y), fill='black')
        draw.text((825, y-6), f'{z} um', fill='black')
    image.save(output/'overview.png')


def save_size_distribution(info, output):
    image = Image.new('RGB', (1400, 650), 'white')
    draw = ImageDraw.Draw(image)
    distribution = info['size_distribution']
    peak, gamma = distribution['peak_nm'], distribution['hwhm_nm']
    normalizer = np.arctan((1000.-peak)/gamma)-np.arctan((50.-peak)/gamma)
    groups = [('Three independent opposite-edge lengths',
               np.array([r['base_edge_lengths_nm'][:3] for r in info['particles']]).ravel()),
              ('Prism height', np.array([r['height_nm'] for r in info['particles']]))]
    for panel, (title, values) in enumerate(groups):
        left, top, width, height = 70+panel*700, 90, 580, 450
        counts, bins = np.histogram(values, np.arange(50., 1001., 25.))
        x = np.linspace(50., 1000., 400)
        theoretical = len(values)*25./(gamma*(1+((x-peak)/gamma)**2)*normalizer)
        maximum = max(counts.max(), theoretical.max())*1.1
        for count, a, b in zip(counts, bins[:-1], bins[1:]):
            xa = left+(a-50)/950*width
            xb = left+(b-50)/950*width
            draw.rectangle((xa, top+height-count/maximum*height, xb, top+height), fill='#7db9d1', outline='white')
        points = [(left+(value-50)/950*width, top+height-y/maximum*height) for value, y in zip(x, theoretical)]
        draw.line(points, fill='#bd3e31', width=3)
        draw.line((left, top, left, top+height, left+width, top+height), fill='black', width=2)
        for tick in (50, 300, 500, 750, 1000):
            draw.text((left+(tick-50)/950*width-12, top+height+12), str(tick), fill='black')
        draw.text((left, 30), title, fill='black')
        draw.text((left, 55), f'Peak={peak:g} nm; HWHM={gamma:g} nm; median={np.median(values):.1f} nm', fill='black')
        draw.text((left+200, top+height+42), 'Size (nm)', fill='black')
    draw.text((70, 610), 'Blue: generated counts (25 nm bins). Red: expected truncated Lorentzian density.', fill='black')
    image.save(output/'size_distribution.png')


def generate(count, seed, stack_probability, output, stl_output, peak=300., hwhm=100., max_stack_height_nm=5000.):
    rng = np.random.default_rng(seed)
    source = STL_SAMPLES/'hexagonal_prism_on_lacey_carbon_with_copper_grid.stl'
    assembly = trimesh.load_mesh(source)
    metadata = json.loads((STL_SAMPLES/'single_cathode.json').read_text(encoding='utf-8'))
    support = assembly.submesh([np.flatnonzero(~SampleGeometry.cathode_face_mask(assembly, metadata))], append=True)
    triangles = support.triangles
    top = np.all(np.isclose(triangles[:, :, 2], 5., atol=1e-6), axis=1)
    film = union_all([Polygon(t[:, :2]) for t in triangles[top]])
    if not np.isclose(np.ptp(support.vertices[:, :2], axis=0), 85000.).all():
        raise ValueError('Expected the current 85 um grid support')
    particles, footprints, planes, records = [], [], [], []
    grid = SpatialGrid()
    height_rejections = 0
    for index in range(count):
        mesh, equations, record = make_prism(rng, peak, hwhm)
        clustered = bool(index and rng.random() < stack_probability)
        low, high = mesh.bounds
        contact_vertex = mesh.vertices[np.argmin(mesh.vertices[:, 2])]
        local_footprint = MultiPoint(mesh.vertices[:, :2]).convex_hull
        a = equations[:, :3]
        for attempt in range(10000):
            if clustered and attempt < 1000:
                parent = int(rng.integers(index))
                origin = np.asarray(records[parent]['center_nm'][:2])+rng.normal(0., 1500., 2)
            else:
                origin = rng.uniform(-27000.-low[:2], 27000.-high[:2])
            if ((origin+low[:2]) < -27000.).any() or ((origin+high[:2]) > 27000.).any():
                continue
            if not film.covers(Point(origin+contact_vertex[:2])):
                continue
            footprint = translate(local_footprint, xoff=origin[0], yoff=origin[1])
            b = -equations[:, 3]+a[:, :2]@origin
            height_shift, contact_id = 5., 0
            contact_point = np.r_[origin+contact_vertex[:2], 5.]
            for other in grid.query(footprint.bounds):
                if not footprint.intersects(footprints[other]):
                    continue
                contact = vertical_contact(planes[other], (a, b))
                if contact is not None and contact[0] > height_shift:
                    height_shift, contact_point = contact
                    contact_id = other+1
            if height_shift+high[2] > 5.+max_stack_height_nm:
                height_rejections += 1
                continue
            break
        else:
            raise ValueError('Could not place prism on carbon within the stack height limit')
        mesh.apply_translation([*origin, height_shift])
        particles.append(mesh)
        footprints.append(footprint)
        planes.append((a, b+a[:, 2]*height_shift))
        grid.add(index, footprint.bounds)
        record.update(id=index+1, center_nm=mesh.vertices.mean(axis=0).tolist(),
            bottom_face_center_nm=mesh.vertices[:6].mean(axis=0).tolist(),
            vertices_lab_nm=mesh.vertices.tolist(), bounds_nm=mesh.bounds.tolist(),
            translation_nm=[*origin.tolist(), height_shift], contact_position_nm=np.asarray(contact_point).tolist(),
            contact_particle_id=contact_id, stacked=contact_id != 0,
            intentional_cluster=clustered, stack_level=records[contact_id-1]['stack_level']+1 if contact_id else 0)
        transform = np.eye(4)
        transform[:3, :3] = record['rotation_matrix_local_to_lab']
        transform[:3, 3] = record['bottom_face_center_nm']
        record['transform_base_local_to_lab'] = transform.tolist()
        records.append(record)
        if (index+1) % 50 == 0:
            print(f'Placed {index+1}/{count}; stacked: {sum(r["stacked"] for r in records)}', flush=True)
    combined = trimesh.util.concatenate([support, *particles])
    expected_volume = support.volume+sum(p.volume for p in particles)
    verify(combined, expected_volume)
    stl_output.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    name = f'{count}_hexagonal_prisms_on_lacey_carbon_with_copper_grid.stl'
    path = stl_output/name
    combined.export(path)
    loaded = trimesh.load_mesh(path)
    verify(loaded, expected_volume)
    np.testing.assert_allclose(np.ptp(loaded.vertices[:, :2], axis=0), [85000., 85000.])
    info = dict(seed=seed, particle_count=count, coordinate_unit='nm',
        source_stl=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        stl_file=name, stl_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        carbon_size_nm=[54000, 54000, 5], grid_size_nm=[85000, 85000, 10000],
        size_distribution=dict(type='truncated Lorentzian (Cauchy)', peak_nm=peak, hwhm_nm=hwhm,
            fwhm_nm=2*hwhm, minimum_nm=50., maximum_nm=1000., sampling='inverse CDF'),
        max_stack_height_nm=max_stack_height_nm,
        max_particle_height_above_carbon_nm=max(r['bounds_nm'][1][2] for r in records)-5.,
        height_limit_rejected_positions=height_rejections,
        orientation_convention='Cartesian [abc], integers 0..100; nonzero tilt, c=0 allowed; local +Z is base normal',
        euler_convention='active extrinsic xyz in degrees; use rotation_matrix_local_to_lab for unambiguous transforms',
        stack_probability=stack_probability, stacked_count=sum(r['stacked'] for r in records),
        stacking_method='vertical contact of convex polyhedra; no particle interpenetration; not a mechanical equilibrium simulation',
        max_stack_level=max(r['stack_level'] for r in records),
        bounds_nm=loaded.bounds.tolist(), faces=len(loaded.faces), watertight=bool(loaded.is_watertight),
        particles=records)
    (stl_output/f'{count}_cathodes.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    np.savez_compressed(stl_output/f'{count}_cathodes.npz',
        id=np.arange(1, count+1), center_nm=np.array([r['center_nm'] for r in records]),
        base_edge_lengths_nm=np.array([r['base_edge_lengths_nm'] for r in records]),
        height_nm=np.array([r['height_nm'] for r in records]),
        normal_abc=np.array([r['base_normal_abc'] for r in records]),
        rotation_matrix=np.array([r['rotation_matrix_local_to_lab'] for r in records]),
        euler_xyz_deg=np.array([r['euler_xyz_deg'] for r in records]),
        vertices_lab_nm=np.array([r['vertices_lab_nm'] for r in records]),
        transform_base_local_to_lab=np.array([r['transform_base_local_to_lab'] for r in records]),
        contact_position_nm=np.array([r['contact_position_nm'] for r in records]),
        contact_particle_id=np.array([r['contact_particle_id'] for r in records]),
        stack_level=np.array([r['stack_level'] for r in records]),
        max_stack_height_nm=np.array(max_stack_height_nm),
        size_peak_nm=np.array(peak), size_hwhm_nm=np.array(hwhm),
        seed=np.array(seed), coordinate_unit=np.array('nm'))
    image = Image.new('RGB', (1600, 1670), 'white')
    draw = ImageDraw.Draw(image)
    def screen(points):
        return [(float(50+(x+42500)/85000*1500), float(50+(42500-y)/85000*1500)) for x, y in points]
    draw.rectangle((50, 50, 1550, 1550), fill='#b77942')
    draw.polygon(screen([[-27000, -27000], [27000, -27000], [27000, 27000], [-27000, 27000]]), fill='white')
    for polygon in ([film] if film.geom_type == 'Polygon' else film.geoms):
        draw.polygon(screen(polygon.exterior.coords), fill='#555555')
        for ring in polygon.interiors:
            draw.polygon(screen(ring.coords), fill='white')
    for index in sorted(range(count), key=lambda i: records[i]['center_nm'][2]):
        level = records[index]['stack_level']
        color = (240, max(60, 180-level*5), 55)
        draw.polygon(screen(footprints[index].exterior.coords), fill=color, outline='#70401a')
    draw.text((50, 1575), f'{count} irregular tilted hexagonal prisms; carbon 54 um; copper outside 85 um', fill='black')
    draw.text((50, 1600), f'Stacked: {info["stacked_count"]}; max stack level: {info["max_stack_level"]}; seed: {seed}', fill='black')
    image.save(output/'top_view.png')
    save_overview(info, output)
    save_size_distribution(info, output)
    summary = {k: v for k, v in info.items() if k != 'particles'}
    (output/'model_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--stack-probability', type=float, default=.25)
    parser.add_argument('--size-peak-nm', type=float, default=300.)
    parser.add_argument('--size-hwhm-nm', type=float, default=100.)
    parser.add_argument('--max-stack-height-nm', type=float, default=5000.)
    parser.add_argument('--output', type=Path, default=SAMPLES/'thousand_cathodes')
    parser.add_argument('--stl-output', type=Path, default=STL_SAMPLES)
    args = parser.parse_args()
    if args.count < 1 or not 0 <= args.stack_probability <= 1:
        parser.error('count must be positive and stack probability must be in [0, 1]')
    if not (50. <= args.size_peak_nm <= 1000. and np.isfinite(args.size_hwhm_nm)
            and args.size_hwhm_nm > 0 and np.isfinite(args.max_stack_height_nm) and args.max_stack_height_nm > 0):
        parser.error('peak must be in 50..1000 nm; width and stack height must be positive and finite')
    generate(args.count, args.seed, args.stack_probability, args.output, args.stl_output,
             args.size_peak_nm, args.size_hwhm_nm, args.max_stack_height_nm)
