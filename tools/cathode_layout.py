"""Rigid initial rotation and a translated, seam-free copper grid array.

The initial [abc] labels are Cartesian generation directions, not Miller
indices. Their lab vectors rotate; crystal-local [001] remains [001].
"""
import numpy as np
from scipy.spatial.transform import Rotation
from shapely import union_all
from shapely.geometry import box, Polygon, MultiPoint
from shapely.affinity import affine_transform
import trimesh
from PIL import Image, ImageDraw


def rotation_z(degrees):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def prism_faces():
    faces = []
    for i in range(1, 5):
        faces.extend(((0, i+1, i), (6, 6+i, 6+i+1)))
    for i in range(6):
        j = (i+1) % 6
        faces.extend(((i, j, 6+j), (i, 6+j, 6+i)))
    return np.asarray(faces)


def transform_records(records, matrix, *, remember_initial=False):
    for record in records:
        if remember_initial:
            record['base_normal_abc_initial'] = np.rint(record['base_normal_abc']).astype(int).tolist()
        for field in ('center_nm', 'bottom_face_center_nm', 'vertices_lab_nm', 'translation_nm', 'contact_position_nm'):
            record[field] = (np.asarray(record[field]) @ matrix.T).tolist()
        normal = matrix @ np.asarray(record['base_normal_abc'])
        record['base_normal_abc'] = normal.tolist()
        record['base_normal_unit_xyz'] = (normal/np.linalg.norm(normal)).tolist()
        record['normal_azimuth_deg'] = float(np.degrees(np.arctan2(normal[1], normal[0])))
        orientation = matrix @ np.asarray(record['rotation_matrix_local_to_lab'])
        record['rotation_matrix_local_to_lab'] = orientation.tolist()
        record['euler_xyz_deg'] = Rotation.from_matrix(orientation).as_euler('xyz', degrees=True).tolist()
        vertices = np.asarray(record['vertices_lab_nm'])
        record['bounds_nm'] = np.stack((vertices.min(axis=0), vertices.max(axis=0))).tolist()
        transform = np.eye(4)
        transform[:3, :3] = orientation
        transform[:3, 3] = record['bottom_face_center_nm']
        record['transform_base_local_to_lab'] = transform.tolist()


def apply_layout(support, records, degrees=60., tiles=3):
    if not np.isfinite(degrees) or tiles < 1 or tiles % 2 != 1:
        raise ValueError('Rotation must be finite and grid count must be positive and odd')
    triangles = support.triangles
    copper_mask = (triangles[:, :, 2].max(axis=1) > 5.01) | (np.abs(triangles[:, :, :2]).max(axis=(1, 2)) > 27000.01)
    carbon = support.submesh([np.flatnonzero(~copper_mask)], append=True)
    centers = np.array([[x*85000., y*85000., 0.] for y in range(-(tiles//2), tiles//2+1)
                        for x in range(-(tiles//2), tiles//2+1)])
    # Union the translated copies before extrusion. Keeping duplicated
    # touching frame faces in STL would create non-manifold seams on loading.
    ring = box(-42500., -42500., 42500., 42500.).difference(box(-27000., -27000., 27000., 27000.))
    from shapely.affinity import translate
    copper_2d = union_all([translate(ring, xoff=c[0], yoff=c[1]) for c in centers])
    copper = trimesh.creation.extrude_polygon(copper_2d, height=10000., engine='earcut')
    rotation = rotation_z(degrees)
    transform_records(records, rotation, remember_initial=True)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    carbon.apply_transform(matrix)
    copper.apply_transform(matrix)
    particles = [trimesh.Trimesh(vertices=r['vertices_lab_nm'], faces=prism_faces(), process=False) for r in records]
    combined = trimesh.util.concatenate([carbon, copper, *particles])
    metadata = dict(initial_rotation_deg=float(degrees), initial_rotation_matrix=rotation.tolist(),
        initial_rotation_pivot_nm=[0., 0., 0.], grid_tiles_per_axis=tiles,
        grid_pitch_nm=85000., grid_tile_centers_local_nm=centers.tolist(),
        grid_tile_centers_lab_nm=(centers @ rotation.T).tolist(),
        overall_grid_size_local_nm=[tiles*85000., tiles*85000., 10000.],
        populated_grid_tile=[0, 0], carbon_faces=len(carbon.faces), copper_faces=len(copper.faces),
        orientation_convention='base_normal_abc_initial: original Cartesian integers 0..100; base_normal_abc: rotated lab Cartesian vector; crystal-local [001] is unchanged')
    return combined, metadata


def save_arrays(info, root):
    records = info['particles']
    np.savez_compressed(root/f'{len(records)}_cathodes.npz',
        id=np.array([r['id'] for r in records]), center_nm=np.array([r['center_nm'] for r in records]),
        base_edge_lengths_nm=np.array([r['base_edge_lengths_nm'] for r in records]),
        height_nm=np.array([r['height_nm'] for r in records]),
        normal_abc=np.array([r['base_normal_abc'] for r in records]),
        normal_abc_initial=np.array([r['base_normal_abc_initial'] for r in records]),
        rotation_matrix=np.array([r['rotation_matrix_local_to_lab'] for r in records]),
        euler_xyz_deg=np.array([r['euler_xyz_deg'] for r in records]),
        vertices_lab_nm=np.array([r['vertices_lab_nm'] for r in records]),
        transform_base_local_to_lab=np.array([r['transform_base_local_to_lab'] for r in records]),
        contact_position_nm=np.array([r['contact_position_nm'] for r in records]),
        contact_particle_id=np.array([r['contact_particle_id'] for r in records]),
        stack_level=np.array([r['stack_level'] for r in records]),
        max_stack_height_nm=info['max_stack_height_nm'], size_peak_nm=info['size_distribution']['peak_nm'],
        size_hwhm_nm=info['size_distribution']['hwhm_nm'], seed=info['seed'], coordinate_unit='nm',
        initial_rotation_deg=info['initial_rotation_deg'], initial_rotation_matrix=info['initial_rotation_matrix'],
        grid_tile_centers_lab_nm=info['grid_tile_centers_lab_nm'], grid_pitch_nm=info['grid_pitch_nm'])


def save_layout_preview(info, original_support, output, *, film=None):
    if film is None:
        triangles = original_support.triangles
        film = union_all([Polygon(t[:, :2]) for t in triangles[np.all(np.isclose(triangles[:, :, 2], 5., atol=1e-6), axis=1)]])
    r = np.asarray(info['initial_rotation_matrix'])
    film = affine_transform(film, [r[0, 0], r[0, 1], r[1, 0], r[1, 1], 0., 0.])
    def render(path, half_extent, label):
        image = Image.new('RGB', (1600, 1670), 'white')
        draw = ImageDraw.Draw(image)
        def screen(points):
            return [(float(800+x/half_extent*750), float(800-y/half_extent*750)) for x, y in points]
        def square(center, size):
            xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*size/2+np.asarray(center[:2])
            return xy @ r[:2, :2].T
        for center in info['grid_tile_centers_local_nm']:
            draw.polygon(screen(square(center, 85000)), fill='#b77942')
        # Draw all holes after copper, since neighboring tiles share boundaries.
        for center in info['grid_tile_centers_local_nm']:
            draw.polygon(screen(square(center, 54000)), fill='white')
        for polygon in ([film] if film.geom_type == 'Polygon' else film.geoms):
            draw.polygon(screen(polygon.exterior.coords), fill='#555555')
            for hole in polygon.interiors:
                draw.polygon(screen(hole.coords), fill='white')
        for record in sorted(info['particles'], key=lambda r: r['center_nm'][2]):
            hull = MultiPoint(np.asarray(record['vertices_lab_nm'])[:, :2]).convex_hull
            draw.polygon(screen(hull.exterior.coords), fill=(240, max(60, 180-record['stack_level']*5), 55), outline='#70401a')
        draw.text((50, 1575), label, fill='black')
        empty_count = info['grid_tiles_per_axis']**2-1
        draw.text((50, 1600), f'Carbon and {info["particle_count"]} particles in central opening only; other {empty_count} openings are vacuum.', fill='black')
        image.save(path)
    bounds = np.asarray(info['bounds_nm'])
    tiles = info['grid_tiles_per_axis']
    render(output/'top_view.png', np.max(np.abs(bounds[:, :2])), f'{tiles}x{tiles} copper grid; local size {tiles*85} um; initial rotation {info["initial_rotation_deg"]:g} deg')
    render(output/'center_window.png', 42500.*np.abs(r[:2, :2]).sum(axis=1).max(), 'Central tile: frame 85 um; carbon opening 54 um; edges/heights 50..1000 nm')
    overview = Image.new('RGB', (1900, 1050), 'white')
    for x, name in ((0, 'top_view.png'), (950, 'center_window.png')):
        overview.paste(Image.open(output/name).resize((950, 992), Image.Resampling.LANCZOS), (x, 30))
    overview.save(output/'overview.png')
